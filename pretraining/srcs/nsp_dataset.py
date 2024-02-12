import os
import sys
import random
import numpy as np
from tqdm import tqdm
from typing import List
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
sys.path.append(os.getcwd())
from pretraining.srcs.functions import float_separator, BAR_FORMAT, init_random


class TextDatasetForNextSentencePrediction(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            nsp_path: str,
            file_path: str,
            max_seq_len: int,
            random_seed: int,
            short_seq_probability=0.1,
            nsp_probability=0.5,
            logger=None,
            sen_a_file="sentence_as.txt",
            sen_b_file="sentence_bs.txt",
            nsp_label_file="nsp_labels.txt"
    ):
        self.tokenizer = tokenizer
        self.logger = logger
        self.rng = random.Random(random_seed)

        self.max_num_tokens = max_seq_len - self.tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        self.nsp_path = nsp_path
        self.sen_a_file = sen_a_file
        self.sen_b_file = sen_b_file
        self.nsp_label_file = nsp_label_file

        self.num_doc = 0

        os.makedirs(nsp_path, exist_ok=True)
        self.logger.info(f"* Data directory:{nsp_path}\n")
        if (not os.path.isfile(os.path.join(nsp_path, self.sen_a_file))) and \
                (not os.path.isfile(os.path.join(nsp_path, self.sen_b_file))) and \
                (not os.path.isfile(os.path.join(nsp_path, self.nsp_label_file))):
            self.logger.info(f"Creating the nsp dataset")
            self.create_examples(file_path, nsp_path)

        self.logger.info(f"Loading nsp dataset... from {nsp_path}...")

        self.sentence_as = open(os.path.join(nsp_path, self.sen_a_file), 'r')
        self.sentence_bs = open(os.path.join(nsp_path, self.sen_b_file), 'r')
        self.labels = open(os.path.join(nsp_path, self.nsp_label_file), 'r')

        self.data_size = len(open(os.path.join(nsp_path, self.nsp_label_file), 'r').readlines())

    def create_examples(self, file_path: str, nsp_path: str):
        # Separate the corpus in document size.
        doc_sep = 0
        doc = []
        documents = []
        corpus = open(file_path, 'r', encoding='utf-8').readlines()
        for line in tqdm(corpus, total=len(corpus), desc="Splitting into doc-level...", bar_format=BAR_FORMAT):
            if line == '\n':
                doc_sep += 1
                if doc_sep == 2:
                    if len(doc) == 0:
                        doc = []
                        doc_sep = 0
                        continue
                    documents.append(doc)
                    doc = []
                    doc_sep = 0
            else:
                line = line.strip()
                if len(line) == 0:
                    continue
                doc.append(line)
        self.logger.info(f"Total number of documents: {float_separator(len(documents))} docs")
        self.logger.info(f"Avg num of sentences in doc : {np.average([len(doc) for doc in documents]):.2f} sentences\n")

        if not os.path.exists(nsp_path):
            os.makedirs(nsp_path, exist_ok=True)

        temp_a = 'temp_' + self.sen_a_file
        temp_b = 'temp_' + self.sen_b_file
        temp_l = 'temp_' + self.nsp_label_file

        self.num_doc = len(documents)
        with open(os.path.join(nsp_path, temp_a), 'w') as fw_sen_a:
            with open(os.path.join(nsp_path, temp_b), 'w') as fw_sen_b:
                with open(os.path.join(nsp_path, temp_l), 'w') as fw_l:
                    for doc_index in tqdm(range(self.num_doc), desc="Creating examples: ", bar_format=BAR_FORMAT):
                        sentence_a, sentence_b, label = self.create_examples_from_document(documents, doc_index)
                        for i in range(len(sentence_a)):
                            fw_sen_a.write(sentence_a[i] + '\n')
                            fw_sen_b.write(sentence_b[i] + '\n')
                            fw_l.write(str(label[i]) + '\n')
                fw_l.close()
            fw_sen_b.close()
        fw_sen_a.close()

        init_random(42)

        labels = open(os.path.join(nsp_path, temp_l), 'r').readlines()
        data_size = len(labels)
        rnd_idx = np.random.permutation(range(data_size))

        with open(os.path.join(nsp_path, self.nsp_label_file), 'w') as fw:
            for idx in tqdm(rnd_idx, total=data_size, desc="Saving shuffled labels: ", bar_format=BAR_FORMAT):
                label = labels[idx].strip()
                fw.write(label + '\n')
        fw.close()
        del labels
        os.remove(os.path.join(nsp_path, temp_l))

        sentences = open(os.path.join(nsp_path, temp_a), 'r').readlines()
        with open(os.path.join(nsp_path, self.sen_a_file), 'w') as fw:
            for idx in tqdm(rnd_idx, total=data_size, desc="Saving shuffled sentence_as: ", bar_format=BAR_FORMAT):
                sentence = sentences[idx].strip()
                fw.write(sentence + '\n')
        fw.close()
        del sentences
        os.remove(os.path.join(nsp_path, temp_a))

        sentences = open(os.path.join(nsp_path, temp_b), 'r').readlines()
        with open(os.path.join(nsp_path, self.sen_b_file), 'w') as fw:
            for idx in tqdm(rnd_idx, total=data_size, desc="Saving shuffled sentence_bs: ", bar_format=BAR_FORMAT):
                sentence = sentences[idx].strip()
                fw.write(sentence + '\n')
        fw.close()
        del sentences
        os.remove(os.path.join(nsp_path, temp_b))

        print("Complete to create the NSP dataset.. !")

    def create_examples_from_document(self, documents: List[List[str]], doc_index: int):
        """
        Create examples for a single document.
        The corpus for Next Sentence Prediction (NSP) task

        - If the random number < 0.5, then assign the next sentence (label is 0)
        - Otherwise, assign the random sentence (= is not next sentence) (label is 1)
        """
        document = documents[doc_index]
        sentence_as, sentence_bs, labels = [], [], []

        target_seq_length = self.max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, self.max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            segment = self.tokenizer.tokenize(segment)
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = self.rng.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    # Random next
                    if len(current_chunk) == 1 or self.rng.random() < self.nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        random_document_index = 0
                        for _ in range(10):
                            random_document_index = self.rng.randint(0, self.num_doc - 1)
                            if random_document_index != doc_index:
                                break
                        random_document = documents[random_document_index]
                        random_start = self.rng.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            random_line = self.tokenizer.tokenize(random_document[j])
                            tokens_b.extend(random_line)
                            if len(tokens_b) >= target_b_length:
                                break

                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    sentence_as.append(" ".join(tokens_a))
                    sentence_bs.append(" ".join(tokens_b))
                    labels.append(int(is_random_next))

                current_chunk = []
                current_length = 0
            i += 1
        return sentence_as, sentence_bs, labels

    def __tokenize__(self, sentence):
        # tokens = self.tokenizer.tokenize(sentence)
        # tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return sentence.split()

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        tokens_a = self.tokenizer.convert_tokens_to_ids(self.sentence_as.readline().strip().split())
        tokens_b = self.tokenizer.convert_tokens_to_ids(self.sentence_bs.readline().strip().split())

        if len(tokens_a) < 1 or len(tokens_b) < 1:
            print("Initialize the epoch.")
            self.sentence_as = open(os.path.join(self.nsp_path, self.sen_a_file), 'r')
            self.sentence_bs = open(os.path.join(self.nsp_path, self.sen_b_file), 'r')
            self.labels = open(os.path.join(self.nsp_path, self.nsp_label_file), 'r')

            tokens_a = self.tokenizer.convert_tokens_to_ids(self.sentence_as.readline().strip().split())
            tokens_b = self.tokenizer.convert_tokens_to_ids(self.sentence_bs.readline().strip().split())

        def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
            """Truncates a pair of sequences to a maximum sequence length."""
            if self.tokenizer.tok_config_name == 'bts_units_var_info':
                for tokens in [tokens_a, tokens_b]:
                    remain = len(tokens) % self.tokenizer.trunc_num
                    if remain != 0:
                        del tokens[-remain:]
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_num_tokens:
                    break
                trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                assert len(trunc_tokens) >= 1
                # We want to sometimes truncate from the front and sometimes from the
                # back to add more randomness and avoid biases.
                if self.rng.random() < 0.5:
                    del trunc_tokens[: self.tokenizer.trunc_num]
                else:
                    del trunc_tokens[-self.tokenizer.trunc_num:]
        truncate_seq_pair(tokens_a, tokens_b, self.max_num_tokens)
        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1
        if self.tokenizer.tok_config_name == 'bts_units_var_info':
            assert len(tokens_a) % self.tokenizer.trunc_num == 0
            assert len(tokens_b) % self.tokenizer.trunc_num == 0
        # add special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        # add token type ids, 0 for sentence a, 1 for sentence b
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        label = int(self.labels.readline().strip())
        example = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "next_sentence_label": torch.tensor(1 if label else 0, dtype=torch.long),
        }
        return example
