import os
import sys
import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser as parser
sys.path.append(os.getcwd())
from pretraining.srcs.functions import float_separator, BAR_FORMAT
from nlu_tasks.data_configs.KorNLI.data_utils import load_task_dataset as KorNLI_dataset
from nlu_tasks.data_configs.KorSTS.data_utils import load_task_dataset as KorSTS_dataset
from nlu_tasks.data_configs.NSMC.data_utils import load_task_dataset as NSMC_dataset
from nlu_tasks.data_configs.PAWS_X.data_utils import load_task_dataset as PAWS_X_dataset
from nlu_tasks.srcs.nlu_utils import get_bert_tokenizer, get_task_model, get_optimizer, get_lr_scheduler


class Trainer(nn.Module):
    def __init__(self, hparams: parser.parse_args, logger):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = self.hparams.model_name

        self.dataset = self.get_dataset()
        self.tokenizer = get_bert_tokenizer(self.hparams)
        self.config, self.model, self.criterion = get_task_model(self.hparams, self.tokenizer)
        self.model.to(self.device)
        self.optimizer, self.lr_scheduler = self.get_optimizer_and_scheduler()
        self.scaler = amp.GradScaler()

        self.tb_writer = SummaryWriter(hparams.tb_dir)

        self.vocab_size = self.tokenizer.vocab_size

        self.global_step = 0
        self.cur_ep = 0
        self.stack = 0

        self.best_ckpt = {
            "best_epoch": 0,
            "best_dev_score": 0,
            "best_test_score": 0,
        }

    def get_optimizer_and_scheduler(self):
        data_size = len(self.dataset['train']['label'])
        batch_num = data_size // self.hparams.batch_size if data_size % self.hparams.batch_size == 0 \
            else (data_size // self.hparams.batch_size) + 1
        self.hparams.total_steps = (batch_num / self.hparams.gradient_accumulation_steps) * self.hparams.max_epochs
        self.hparams.num_warmup_steps = round(self.hparams.total_steps * self.hparams.warmup_ratio)

        optimizer = get_optimizer(self.hparams, self.model)
        lr_scheduler = get_lr_scheduler(self.hparams, optimizer)

        return optimizer, lr_scheduler

    def get_dataset(self):
        if self.hparams.task_name == 'KorNLI':
            dataset = KorNLI_dataset(self.hparams.remain_lang, self.hparams.do_hangeulize, self.hparams.data_remove)
        elif self.hparams.task_name == 'KorSTS':
            dataset = KorSTS_dataset(self.hparams.remain_lang, self.hparams.do_hangeulize, self.hparams.data_remove)
        elif self.hparams.task_name == 'NSMC':
            dataset = NSMC_dataset(self.hparams.remain_lang, self.hparams.do_hangeulize, self.hparams.data_remove)
        elif self.hparams.task_name == 'PAWS_X':
            dataset = PAWS_X_dataset(self.hparams.remain_lang, self.hparams.do_hangeulize, self.hparams.data_remove)
        else:
            self.logger.info(
                "It's a Wrong Task Name. Please enter the right task name among [KorNLI, KorSTS, NSMC, PAWS_X]")
            raise ValueError
        return dataset

    def split_dataset(self):
        # Get datasets for each steps
        train_dataset = self.dataset['train']
        dev_dataset = self.dataset['dev']
        test_dataset = self.dataset['test']
        return train_dataset, dev_dataset, test_dataset

    @staticmethod
    def _shuffle_dataset(dataset: Dict[str, Union[List[str], List[int]]]):
        keys = list(dataset.keys())             # sentence1, sentence2, label, ...
        data_size = len(dataset[keys[0]])
        per = np.random.permutation(range(data_size))

        # self.logger.info("Shuffling the dataset")
        new_dataset = dict()
        for key in keys:
            new_dataset[key] = [dataset[key][idx] for idx in per]
        return new_dataset

    def _batch_dataset(self, dataset: Dict[str, Union[List[str], List[int]]]):
        keys = list(dataset.keys())
        data_size = len(dataset[keys[0]])
        batch_num = data_size // self.hparams.batch_size if data_size % self.hparams.batch_size == 0 else (data_size // self.hparams.batch_size) + 1

        # self.logger.info("Split into batch")
        new_dataset = dict()
        for key in keys:
            new_dataset[key] = [dataset[key][i*self.hparams.batch_size: (i+1)*self.hparams.batch_size] for i in range(batch_num)]
        return new_dataset

    def get_input(self, dataset, shuffle: bool):
        if shuffle:
            shuffled_dataset = self._shuffle_dataset(dataset)
        else:
            shuffled_dataset = dataset

        batched_dataset = self._batch_dataset(shuffled_dataset)

        if self.hparams.task_name in ['KorNLI', 'KorSTS', 'PAWS_X']:
            sentence1s = batched_dataset['sentence1']
            sentence2s = batched_dataset['sentence2']
            labels = batched_dataset['label']
        elif self.hparams.task_name == 'NSMC':
            sentence1s = batched_dataset['sentence']
            sentence2s = [None for _ in range(len(sentence1s))]
            labels = batched_dataset['label']
        else:
            raise NotImplementedError

        inputs = []
        for i in tqdm(range(len(labels)), desc="Getting inputs...", bar_format=BAR_FORMAT):
            encoded_input = self.tokenizer(sentence1s[i], sentence2s[i], truncation=True, padding="max_length", max_length=self.hparams.max_seq_len, return_tensors="pt")
            inputs.append(encoded_input)
        return inputs, labels

    def _forward(self, inputs, labels):
        with amp.autocast():
            outputs, logits = self.model.forward(inputs)
            seq_len = outputs.hidden_states[-2].shape[1]
            loss = self.criterion(logits, labels)           # MSELoss
        return logits, loss, seq_len

    def _train_step(self, inputs, labels):
        outputs, loss, seq_len = self._forward(inputs, labels)
        loss /= self.hparams.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        self.stack += 1
        if self.stack == self.hparams.gradient_accumulation_steps:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.hparams.max_grad_norm, norm_type=2)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.stack = 0
            self.global_step += 1

        return outputs, loss.item(), seq_len

    def _eval_step(self, inputs, labels):
        outputs, _, _ = self._forward(inputs, labels)
        return outputs

    def _evaluation(self, eval_dataset):
        self.model.eval()

        inputs, labels = self.get_input(eval_dataset, shuffle=False)
        batch_num = len(inputs)

        targets = []
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(batch_num), desc="Evaluation...", bar_format=BAR_FORMAT, total=batch_num):
                batch = inputs[i]
                for key in batch:  # keys are {token_ids, attention_mask, token_type_ids, (start_positions), (end_positions)}
                    batch[key] = batch[key].to(self.device)

                if self.hparams.task_name == 'KorSTS':
                    # label = torch.tensor(labels[i], dtype=torch.float16).to(self.device)
                    label = torch.FloatTensor(labels[i]).to(self.device)
                else:
                    label = torch.LongTensor(labels[i]).to(self.device)

                outputs = self._eval_step(batch, label)

                if self.hparams.task_name == 'KorSTS':
                    targets.extend(list(label.detach().cpu()))
                    predictions.extend(list(outputs.detach().cpu()))
                else:
                    targets.extend(list(label.detach().cpu()))
                    predictions.extend(list(outputs.detach().cpu().argmax(-1)))

        targets = torch.tensor(targets)
        predictions = torch.tensor(predictions)

        assert len(targets) == len(predictions)
        return targets, predictions

    def fine_tuning(self):
        self.logger.info("========== fine_tuning ==========")
        self.logger.info(f"task name                : {self.hparams.task_name}")
        self.logger.info(f"model                    : {self.hparams.model_name}")
        self.logger.info(f"tokenizer                : {self.hparams.tok_name}")
        self.logger.info(f"vocab size               : {self.config.vocab_size}")
        if 'kombo' in self.hparams.model_name:
            if self.config.jamo_fusion:
                self.logger.info(f"jamo_fusion              : {self.config.jamo_fusion}")
                self.logger.info(f"jamo_residual            : {bool(self.config.jamo_residual)}")
                self.logger.info(f"cho_joong_first          : {bool(self.config.cho_joong_first)}")
        self.logger.info(f"device                   : {self.device}")
        if self.hparams.save_dir:
            self.logger.info(f"save_dir                 : {self.hparams.save_dir}")
        self.logger.info(f"random seed              : {self.hparams.random_seed}")
        self.logger.info(f"train dataset size       : {float_separator(len(self.dataset['train']['label']))}")
        self.logger.info(f"dev dataset size         : {float_separator(len(self.dataset['dev']['label']))}")
        self.logger.info(f"test dataset size        : {float_separator(len(self.dataset['test']['label']))}")
        self.logger.info(f"total epochs             : {self.hparams.max_epochs}")
        self.logger.info(f"batch size               : {self.hparams.batch_size}")
        self.logger.info(f"gradient accum steps     : {self.hparams.gradient_accumulation_steps}")
        self.logger.info(f"learning rate            : {self.hparams.learning_rate}")
        self.logger.info(f"dropout prob             : {self.hparams.dropout_rate}")
        self.logger.info(f"warmup ratio             : {self.hparams.warmup_ratio}")
        self.logger.info(f"max seq len              : {self.hparams.max_seq_len}\n")

        train_dataset, dev_dataset, test_dataset = self.split_dataset()

        train_cnt = 0
        train_loss = 0
        train_seq_len = 0
        train_targets = []
        train_predictions = []
        for epoch in range(self.hparams.max_epochs):
            self.cur_ep = epoch + 1

            print('\n')
            self.logger.info(f"[{self.cur_ep} Epoch]")
            self.model.train()

            inputs, labels = self.get_input(train_dataset, shuffle=True)

            batch_num = len(inputs)

            for i in tqdm(range(batch_num), desc=f"Fine-tuning...", bar_format=BAR_FORMAT, total=batch_num):
                encoded_input = inputs[i]
                for key in encoded_input:      # keys are {token_ids, attention_mask, (token_type_ids), (start_positions), (end_positions)}
                    encoded_input[key] = encoded_input[key].to(self.device)

                if self.hparams.task_name == 'KorSTS':
                    label = torch.FloatTensor(labels[i]).to(self.device)
                else:
                    label = torch.LongTensor(labels[i]).to(self.device)

                outputs, loss, seq_len = self._train_step(encoded_input, label)    # outputs are "logits"

                train_cnt += 1
                train_loss += loss * self.hparams.gradient_accumulation_steps
                train_seq_len += seq_len

                if self.hparams.task_name == 'KorSTS':
                    train_targets.extend(list(label.detach().cpu()))
                    train_predictions.extend(list(outputs.detach().cpu()))
                else:
                    train_targets.extend(list(label.detach().cpu()))
                    train_predictions.extend(list(outputs.argmax(-1).detach().cpu()))

                if self.global_step != 0 and ((self.global_step % self.hparams.tb_interval) == 0):
                    train_targets = torch.tensor(train_targets)
                    train_predictions = torch.tensor(train_predictions)

                    _ = self.log_results('train', train_loss/train_cnt, train_seq_len/train_cnt, train_targets, train_predictions)
                    train_cnt = 0
                    train_loss = 0
                    train_seq_len = 0
                    train_targets = []
                    train_predictions = []

            # evaluate dev and test set every epoch
            dev_targets, dev_predictions = self._evaluation(dev_dataset)
            dev_acc = self.log_results('dev', None, None, dev_targets, dev_predictions)

            test_targets, test_predictions = self._evaluation(test_dataset)
            test_acc = self.log_results('test', None, None, test_targets, test_predictions)

            if ((dev_acc >= self.best_ckpt['best_dev_score']) and (test_acc >= self.best_ckpt['best_test_score'])) or \
                    ((dev_acc + test_acc) >= (self.best_ckpt['best_dev_score'] + self.best_ckpt['best_test_score'])):
                self.best_ckpt['best_epoch'] = epoch + 1
                self.best_ckpt['best_dev_score'] = dev_acc
                self.best_ckpt['best_test_score'] = test_acc

                print("Save the Best Model")
                torch.save(self.model.state_dict(), os.path.join(self.hparams.ckpt_dir, "pytorch_model.bin"))

        print("\n")
        self.logger.info("######### BEST RESULT #########")
        self.logger.info(f"- Epoch: {self.best_ckpt['best_epoch']}")
        self.logger.info(f"- DEV score: {self.best_ckpt['best_dev_score']*100:.2f} [%]")
        self.logger.info(f"- TEST score: {self.best_ckpt['best_test_score']*100:.2f} [%]")

    def log_results(self, mode: str, running_loss, running_seq_len, targets, predictions):
        if mode == 'train':
            if self.hparams.task_name == 'KorSTS':
                corr = spearmanr(targets, predictions)[0]
                self.tb_writer.add_scalar(f'{self.hparams.task_name}/{mode}_loss/step', running_loss, self.global_step)
                self.tb_writer.add_scalar(f'{self.hparams.task_name}/{mode}_seq_len/step', running_seq_len, self.global_step)
                self.tb_writer.add_scalar(f'{self.hparams.task_name}/{mode}_spearman/step', corr, self.global_step)
                self.tb_writer.add_scalar(f'{self.hparams.task_name}/{mode}_lr/step',
                                          self.optimizer.param_groups[0]['lr'], self.global_step)
                self.tb_writer.flush()
                return corr
            else:
                acc = accuracy_score(targets, predictions)
                self.tb_writer.add_scalar(f'{self.hparams.task_name}/{mode}_loss/step', running_loss, self.global_step)
                self.tb_writer.add_scalar(f'{self.hparams.task_name}/{mode}_seq_len/step', running_seq_len, self.global_step)
                self.tb_writer.add_scalar(f'{self.hparams.task_name}/{mode}_acc/step', acc, self.global_step)
                self.tb_writer.add_scalar(f'{self.hparams.task_name}/{mode}_lr/step',
                                          self.optimizer.param_groups[0]['lr'], self.global_step)
                self.tb_writer.flush()
                return acc
        elif mode in ['dev', 'test']:
            if self.hparams.task_name == 'KorSTS':
                corr = spearmanr(targets, predictions)[0]
                self.logger.info(f"######### {mode.upper()} REPORT #EP{self.cur_ep} #########")
                self.logger.info(f"Spearman Corr {corr*100:.2f} [%]")
                self.tb_writer.add_scalar(f"{self.hparams.task_name}/{mode}_spearman/step", corr, self.global_step)
                self.tb_writer.flush()
                return corr
            else:
                acc = accuracy_score(targets, predictions)
                self.logger.info(f"######### {mode.upper()} REPORT #EP{self.cur_ep} #########")
                self.logger.info(f"Acc {acc * 100:.2f} [%]")
                self.tb_writer.add_scalar(f"{self.hparams.task_name}/{mode}_acc/step", acc, self.global_step)
                self.tb_writer.flush()
                return acc
