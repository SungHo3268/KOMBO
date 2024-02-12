import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser as parser
from typing import Union, List, Dict
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()
import torch
import torch.cuda.amp as amp
import torch.nn as nn
sys.path.append(os.getcwd())
from typo.data_configs.KorNLI.data_utils import load_task_dataset as KorNLI_dataset
from typo.data_configs.KorSTS.data_utils import load_task_dataset as KorSTS_dataset
from typo.data_configs.NSMC.data_utils import load_task_dataset as NSMC_dataset
from typo.data_configs.PAWS_X.data_utils import load_task_dataset as PAWS_X_dataset

from nlu_tasks.srcs.nlu_utils import get_bert_tokenizer, get_task_model
from pretraining.srcs.functions import BAR_FORMAT


class Trainer(nn.Module):
    def __init__(self, hparams: parser.parse_args, logger):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = self.hparams.model_name

        self.tokenizer = get_bert_tokenizer(self.hparams)
        self.config, self.model, _ = get_task_model(self.hparams, self.tokenizer)
        self.model.to(self.device)

        self.vocab_size = self.tokenizer.vocab_size

    def get_dataset(self, typo_ratio):
        if self.hparams.task_name == 'KorNLI':
            dataset = KorNLI_dataset(typo_ratio, self.hparams.remain_lang, self.hparams.do_hangeulize, self.hparams.data_remove)
        elif self.hparams.task_name == 'KorSTS':
            dataset = KorSTS_dataset(typo_ratio, self.hparams.remain_lang, self.hparams.do_hangeulize, self.hparams.data_remove)
        elif self.hparams.task_name == 'NSMC':
            dataset = NSMC_dataset(typo_ratio, self.hparams.remain_lang, self.hparams.do_hangeulize, self.hparams.data_remove)
        elif self.hparams.task_name == 'PAWS_X':
            dataset = PAWS_X_dataset(typo_ratio, self.hparams.remain_lang, self.hparams.do_hangeulize, self.hparams.data_remove)
        else:
            self.logger.info(
                "It's a Wrong Task Name. Please enter the right task name among [KorNLI, KorSTS, NSMC, PAWS_X]")
            raise ValueError
        return dataset


    def _batch_dataset(self, dataset: Dict[str, Union[List[str], List[int]]]):
        keys = list(dataset.keys())
        data_size = len(dataset[keys[0]])
        batch_num = data_size // self.hparams.batch_size if data_size % self.hparams.batch_size == 0 else (data_size // self.hparams.batch_size) + 1

        # self.logger.info("Split into batch")
        new_dataset = dict()
        for key in keys:
            new_dataset[key] = [dataset[key][i*self.hparams.batch_size: (i+1)*self.hparams.batch_size] for i in range(batch_num)]
        return new_dataset

    def get_input(self, dataset, typo_ratio):
        batched_dataset = self._batch_dataset(dataset)

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
        # for i in tqdm(range(len(labels)), desc=f"Getting inputs for Typo {typo_ratio*100}%", bar_format=BAR_FORMAT):
        for i in range(len(labels)):
            encoded_input = self.tokenizer(sentence1s[i], sentence2s[i], truncation=True, padding="max_length",
                                           max_length=self.hparams.max_seq_len, return_tensors="pt")
            inputs.append(encoded_input)
        return inputs, labels

    def _forward(self, inputs, labels):
        with amp.autocast():
            outputs, logits = self.model.forward(inputs)
        return logits

    def _eval_step(self, inputs, labels):
        outputs = self._forward(inputs, labels)
        return outputs

    def _evaluation(self, eval_dataset, typo_ratio):
        self.model.eval()

        inputs, labels = self.get_input(eval_dataset, typo_ratio)
        batch_num = len(inputs)

        targets = []
        predictions = []
        with torch.no_grad():
            # for i in tqdm(range(batch_num), desc=f"Evaluation for Typo {typo_ratio*100}%...", bar_format=BAR_FORMAT, total=batch_num):
            for i in range(batch_num):
                batch = inputs[i]
                for key in batch:       # keys are {token_ids, attention_mask, token_type_ids, (start_positions), (end_positions)}
                    batch[key] = batch[key].to(self.device)

                if self.hparams.task_name == 'KorSTS':
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

    def fine_tuning(self, typo_rates):
        scores = []
        for typo_ratio in tqdm(typo_rates, desc=f"Evaluation...", bar_format=BAR_FORMAT, total=len(typo_rates)):
            test_dataset = self.get_dataset(typo_ratio)["test"]

            test_targets, test_predictions = self._evaluation(test_dataset, typo_ratio)
            test_score = self.metric_results(test_targets, test_predictions)
            scores.append(round(test_score, 4))

        return scores


    def metric_results(self, targets, predictions):
        if self.hparams.task_name == 'KorSTS':
            corr = spearmanr(targets, predictions)[0]
            return corr
        else:
            acc = accuracy_score(targets, predictions)
            return acc
