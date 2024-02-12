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
from transformers import AutoTokenizer
from argparse import ArgumentParser as parser
sys.path.append(os.getcwd())
from pretraining.srcs.functions import float_separator, BAR_FORMAT
from toxic.srcs.utils import load_data, get_task_model
from nlu_tasks.srcs.nlu_utils import get_bert_tokenizer, get_optimizer, get_lr_scheduler
from sklearn.metrics import f1_score, precision_recall_fscore_support



class Trainer(nn.Module):
    def __init__(self, hparams: parser.parse_args, logger):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = self.hparams.model_name

        self.label_map = None
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
            "best_dev_score": {"precision": 0, "recall": 0, "f1": 0},
            "best_test_score": {"precision": 0, "recall": 0, "f1": 0}
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
        dataset = load_data()
        dataset = dataset[self.hparams.label_level]
        self.label_map = dataset.pop("label_map")

        train_dataset, dev_dataset, test_dataset = self.split_dataset(dataset)
        dataset = {
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset
        }
        return dataset

    def split_dataset(self, dataset):
        # Get datasets for each steps
        split_ratio = list(map(float, self.hparams.split_ratio.split('_')))
        _shuffle_dataset = self._shuffle_dataset(dataset)

        train_size = int(len(_shuffle_dataset['label']) * split_ratio[0])
        dev_size = int(len(_shuffle_dataset['label']) * split_ratio[1])

        train_dataset = {
            'sentence1': _shuffle_dataset['title'][: train_size],
            'sentence2': _shuffle_dataset['comment'][: train_size],
            'label': _shuffle_dataset['label'][: train_size]
        }
        dev_dataset = {
            'sentence1': _shuffle_dataset['title'][train_size: train_size+dev_size],
            'sentence2': _shuffle_dataset['comment'][train_size: train_size+dev_size],
            'label': _shuffle_dataset['label'][train_size: train_size+dev_size]
        }
        test_dataset = {
            'sentence1': _shuffle_dataset['title'][train_size+dev_size:],
            'sentence2': _shuffle_dataset['comment'][train_size+dev_size:],
            'label': _shuffle_dataset['label'][train_size+dev_size:]
        }
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

        sentence1s = batched_dataset['sentence1']
        sentence2s = batched_dataset['sentence2']
        labels = batched_dataset['label']

        inputs = []
        for i in tqdm(range(len(labels)), desc="Getting inputs...", bar_format=BAR_FORMAT):
            encoded_input = self.tokenizer(sentence1s[i], sentence2s[i], truncation=True, padding="max_length", max_length=self.hparams.max_seq_len, return_tensors="pt")
            inputs.append(encoded_input)
        return inputs, labels

    def _forward(self, inputs, labels):
        with amp.autocast():
            outputs, logits = self.model.forward(inputs)
            loss = self.criterion(logits, labels)           # MSELoss
        return logits, loss

    def _train_step(self, inputs, labels):
        outputs, loss = self._forward(inputs, labels)
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

        return outputs, loss.item()

    def _eval_step(self, inputs, labels):
        outputs, _ = self._forward(inputs, labels)
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

                if self.hparams.label_level == 'C':
                    label = torch.FloatTensor(labels[i]).to(self.device)
                else:
                    label = torch.LongTensor(labels[i]).to(self.device)

                outputs = self._eval_step(batch, label)

                targets.extend(list(label.detach().cpu()))
                if self.hparams.label_level == 'C':
                    predictions.extend(list(torch.sigmoid(outputs).detach().cpu()))
                else:
                    predictions.extend(list(outputs.argmax(-1).detach().cpu()))

        if self.hparams.label_level == 'C':
            targets = np.stack(targets)
            predictions = np.stack(predictions)
            targets = (targets == 1) * 1
            predictions = (predictions >= 0.5) * 1

        else:
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


        train_cnt = 0
        train_loss = 0
        train_targets = []
        train_predictions = []
        for epoch in range(self.hparams.max_epochs):
            self.cur_ep = epoch + 1

            print('\n')
            self.logger.info(f"[{self.cur_ep} Epoch]")
            self.model.train()

            inputs, labels = self.get_input(self.dataset['train'], shuffle=True)

            batch_num = len(inputs)

            for i in tqdm(range(batch_num), desc=f"Fine-tuning...", bar_format=BAR_FORMAT, total=batch_num):
                encoded_input = inputs[i]
                for key in encoded_input:      # keys are {token_ids, attention_mask, (token_type_ids), (start_positions), (end_positions)}
                    encoded_input[key] = encoded_input[key].to(self.device)

                if self.hparams.label_level == 'C':
                    label = torch.FloatTensor(labels[i]).to(self.device)
                else:
                    label = torch.LongTensor(labels[i]).to(self.device)

                outputs, loss = self._train_step(encoded_input, label)    # outputs are "logits"

                train_cnt += 1
                train_loss += loss * self.hparams.gradient_accumulation_steps

                train_targets.extend(label.detach().cpu())

                if self.hparams.label_level == 'C':
                    train_predictions.extend(list(torch.sigmoid(outputs).detach().cpu()))
                else:
                    train_predictions.extend(list(outputs.argmax(-1).detach().cpu()))

                if self.global_step != 0 and ((self.global_step % self.hparams.tb_interval) == 0):
                    if self.hparams.label_level == 'C':
                        train_targets = np.stack(train_targets)
                        train_predictions = np.stack(train_predictions)
                        # print("train_targets: ", type(train_targets[0][0]))
                        # print("train_predictions: ", type(train_predictions[0][0]))
                        train_targets = (train_targets == 1) * 1
                        train_predictions = (train_predictions >= 0.5) * 1
                    else:
                        train_predictions = torch.tensor(train_predictions)

                    _ = self.log_results('train', train_loss / train_cnt, train_predictions, train_targets)
                    train_cnt = 0
                    train_loss = 0
                    train_targets = []
                    train_predictions = []

            # evaluate dev and test set every epoch
            dev_predictions, dev_targets = self._evaluation(self.dataset["dev"])
            _, _, _ = self.log_results('dev', None, dev_predictions, dev_targets)

            test_predictions, test_targets = self._evaluation(self.dataset["test"])
            test_precision, test_recall, test_f1 = self.log_results('test', None, test_predictions, test_targets)

            if test_f1 >= self.best_ckpt['best_test_score']["f1"]:
                self.best_ckpt['best_epoch'] = epoch + 1
                self.best_ckpt['best_test_score'] = {"precision": test_precision,
                                                     "recall": test_recall,
                                                     "f1": test_f1}

                print("Save the Best Model")
                torch.save(self.model.state_dict(), os.path.join(self.hparams.ckpt_dir, "pytorch_model.bin"))

        print("\n")
        self.logger.info("######### BEST RESULT #########")
        self.logger.info(f"- Epoch: {self.best_ckpt['best_epoch']}")
        self.logger.info(f"- TEST score")
        self.logger.info(f"  ㄴ Precision: {self.best_ckpt['best_test_score']['precision'] * 100:.2f} [%]")
        self.logger.info(f"  ㄴ Recall   : {self.best_ckpt['best_test_score']['recall'] * 100:.2f} [%]")
        self.logger.info(f"  ㄴ F1-score : {self.best_ckpt['best_test_score']['f1'] * 100:.2f} [%]")
        self.logger.info("###############################")
        print("\n\n")
    def log_results(self, mode: str, running_loss, predictions, targets):
        metric = "binary" if self.hparams.label_level == "A" else "macro"
        labels = [i for i in range(0, self.config.level_C_label_num)] if self.hparams.label_level == "C" else [i for i in range(len(self.label_map))]
        if mode == 'train':
            # print(f"train_predictions.shape: {predictions[:10]}")
            # print(f"train_labels.shape: {targets[:10]}")
            precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=metric, labels=labels,
                                                                       zero_division=0)
            self.tb_writer.add_scalar(f'results/{mode}_loss/step', running_loss, self.global_step)
            self.tb_writer.add_scalar(f'results/{mode}_precision/step', precision, self.global_step)
            self.tb_writer.add_scalar(f'results/{mode}_recall/step', recall, self.global_step)
            self.tb_writer.add_scalar(f'results/{mode}_f1/step', f1, self.global_step)
            self.tb_writer.add_scalar(f'results/{mode}_lr/step',
                                      self.optimizer.param_groups[0]['lr'], self.global_step)
            self.tb_writer.flush()
            return precision, recall, f1
        elif mode in ['dev', 'test']:
            precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=metric, labels=labels,
                                                                       zero_division=0)
            self.logger.info(f"######### {mode.upper()} REPORT #EP{self.cur_ep} #########")
            self.logger.info(f"Precision {precision * 100:.2f} [%]")
            self.logger.info(f"Recall {recall * 100:.2f} [%]")
            self.logger.info(f"F1-score {f1 * 100:.2f} [%]")
            self.tb_writer.add_scalar(f'results/{mode}_precision/step', precision, self.global_step)
            self.tb_writer.add_scalar(f'results/{mode}_recall/step', recall, self.global_step)
            self.tb_writer.add_scalar(f'results/{mode}_f1/step', f1, self.global_step)
            self.tb_writer.flush()
            return precision, recall, f1
        else:
            raise NotImplementedError
