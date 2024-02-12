from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from typing import Union, Dict, Tuple
import collections
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class BertTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a BERT tokenizer. Based on WordPiece.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`, defaults to :obj:`None`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.색
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to tokenize Chinese characters.
            This should likely be deactivated for Japanese:
            see: https://github.com/huggingface/transformers/issues/328
    """

    def __init__(
        self,
        vocab_file,
        custom_tokenizer,
        max_length,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'."
            )
        self.vocab = self.load_vocab(vocab_file)        # tokens_to_ids
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        self.custom_tokenizer = custom_tokenizer

        self.pad_token = self.custom_tokenizer.config["pad_piece"]
        self.unk_token = self.custom_tokenizer.config["unk_piece"]
        self.bos_token = self.custom_tokenizer.config["bos_piece"]
        self.eos_token = self.custom_tokenizer.config["eos_piece"]
        self.cls_token, self.sep_token, self.mask_token = self.custom_tokenizer.config["special_symbols"].split(",")

        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            cls_token=self.cls_token,
            sep_token=self.sep_token,
            mask_token=self.mask_token,
            **kwargs,
        )

        self.special_tokens_encoder = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
            self.cls_token: 4,
            self.sep_token: 5,
            self.mask_token: 6,
        }
        self.special_tokens_decoder: Dict[str, int] = {
            v: k for k, v in self.special_tokens_encoder.items()
        }
        self._num_special_tokens = len(self.special_tokens_encoder)

        self.tok_config_name = self.custom_tokenizer.config.name
        self.max_length = max_length

        self.model_max_length = max_length
        self.max_len_single_sentence = max_length - 2
        self.max_len_sentences_pair = max_length - 3

        if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
            self.space_symbol = self.custom_tokenizer.config["space_symbol"]
            self.empty_jamo = self.custom_tokenizer.config["empty_jamo_symbol"]

            self.space_symbol_id = self.vocab[self.space_symbol]
            self.empty_jamo_id = self.vocab[self.empty_jamo]

        if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
            self.trunc_num = self.custom_tokenizer.char_len
        else:
            self.trunc_num = 1

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        """
        self.added_tokens_encoder = {}
        """
        return dict(self.vocab, **self.added_tokens_encoder)

    @staticmethod
    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.strip().split("\t")[0]
            vocab[token] = index
        return vocab

    def _tokenize(self, text):
        tokens = self.custom_tokenizer.tokenize(text)
        return tokens

    def truncate_sequences(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            num_tokens_to_remove: int = 0,
            truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
            stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (`int`, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                The strategy to follow for truncation. Can be:

                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will truncate
                  token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
                  batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
                  than the model maximum admissible input size).
            stride (`int`, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of
            overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair
            of sequences (or a batch of pairs) is provided.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
                truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None
        ):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                    if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                        remain = len(ids) % self.trunc_num
                        if remain != 0:
                            ids = ids[remain:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                    if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                        remain = len(ids) % self.trunc_num
                        if remain != 0:
                            ids = ids[:-remain]
                else:
                    raise ValueError(f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'.")

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. "
                )
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                            error_msg
                            + "Please select another truncation strategy than "
                              f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            # logger.warning(
            #     "Be aware, overflowing tokens are not returned for the setting you have chosen,"
            #     f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
            #     "truncation strategy. So the returned list will always be empty even if some "
            #     "tokens have been removed."
            # )
            for _ in range(num_tokens_to_remove // self.trunc_num + 1):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if self.truncation_side == "right":
                        ids = ids[:-self.trunc_num]
                    elif self.truncation_side == "left":
                        ids = ids[self.trunc_num:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
                else:
                    if self.truncation_side == "right":
                        pair_ids = pair_ids[:-self.trunc_num]
                    elif self.truncation_side == "left":
                        pair_ids = pair_ids[self.trunc_num:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == "right":
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                    if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                        remain = len(pair_ids) % self.trunc_num
                        if remain != 0:
                            pair_ids = pair_ids[:-remain]
                elif self.truncation_side == "left":
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                    if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                        remain = len(pair_ids) % self.trunc_num
                        if remain != 0:
                            pair_ids = pair_ids[remain:]
                else:
                    raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    "for instance 'longest_first' or 'only_first'."
                )

        return (ids, pair_ids, overflowing_tokens)

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length."""
        if tokens_b is None:
            while True:
                total_length = len(tokens_a)
                if total_length <= max_num_tokens:
                    break

                trunc_tokens = tokens_a
                assert len(trunc_tokens) >= 1

                # We want to sometimes truncate from the front and sometimes from the
                # back to add more randomness and avoid biases.
                if self.rng.random() < 0.5:
                    del trunc_tokens[: self.trunc_num]
                else:
                    del trunc_tokens[-self.trunc_num:]
        else:
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_num_tokens:
                    break

                trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                assert len(trunc_tokens) >= 1

                # We want to sometimes truncate from the front and sometimes from the
                # back to add more randomness and avoid biases.
                if self.rng.random() < 0.5:
                    del trunc_tokens[: self.trunc_num]
                else:
                    del trunc_tokens[-self.trunc_num:]

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and (index in self.all_special_ids):
                if (self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']) and (index == self.sep_token_id):
                    tokens += [self._convert_id_to_token(self.space_symbol_id)] + \
                              [self._convert_id_to_token(self.empty_jamo_id)] * (self.custom_tokenizer.char_len - 1)
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = self.custom_tokenizer.detokenize(tokens)
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        pad = [self.pad_token_id]

        if token_ids_1 is None:
            if self.pad_token_id in token_ids_0:
                first_pad_loc = token_ids_0.index(self.pad_token_id)
                token_ids_0[first_pad_loc] = self.sep_token_id
                return cls + token_ids_0 + pad
            return cls + token_ids_0 + sep

        if self.pad_token_id in token_ids_1:
            first_pad_loc = token_ids_1.index(self.pad_token_id)
            token_ids_1[first_pad_loc] = self.sep_token_id
            return cls + token_ids_0 + sep + token_ids_1 + pad
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def decode(
            self,
            token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
    ) -> str:

        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens)
        if (not skip_special_tokens) and (self.tok_config_name == 'jamo_var_info'):
            tokens_padded = []
            for tok in tokens:
                if tok in self.special_tokens_encoder:
                    tokens_padded.extend([tok] + [self.custom_tokenizer.empty_jamo] * (self.custom_tokenizer.char_len-1))
                else:
                    tokens_padded.append(tok)
            tokens = tokens_padded
        else:
            pass
        sequence = self.custom_tokenizer.detokenize(tokens)
        return sequence

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id, self.pad_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, "vocab.txt")
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


if __name__ == '__main__':
    from tokenization.srcs.functions import get_tokenizer

    tok_type = "stroke_var"
    tok_vocab_size = "200"
    lang = "ko"
    max_length = 128

    if tok_type in ['subword', 'morphemeSubword']:
        tok_name = f"{tok_type}_{lang}_wiki_{tok_vocab_size}"
    else:
        tok_name = f"{tok_type}_{lang}_{tok_vocab_size}"

    custom_tokenizer = get_tokenizer(tok_type)
    if tok_type in ['subword', 'morphemeSubword']:
        custom_tokenizer.load_model(f"tokenization/resources/{tok_name}/tok.model")
    vocab_file = f"tokenization/resources/{tok_name}/tok.vocab"

    tokenizer = BertTokenizer(vocab_file=vocab_file,
                              custom_tokenizer=custom_tokenizer,
                              max_length=max_length,
                              lowercase=True,
                              clean_text=True
                              )

    # text = "감사합니다."
    text = "한영 자판 상태에서 히라가나를 입력할 경우 ㄸ+한자 키를 누르면 않된다. 가타카나의 경우 장음 등 일부 문자를 제외하면, 꼭 ㅃ+한자 조합을 해야 한다. \t뒤에꺼는 테스트 문장입니다."

    text1 = "한영 자판 상태에서 히라가나를 입력할 경우 ㄸ+한자 키를 누르면 않된다. 가타카나의 경우 장음 등 일부 문자를 제외하면, 꼭 ㅃ+한자 조합을 해야 한다."
    text2 = "뒤에꺼는 테스트 문장입니다."

    print("[original inputs]")
    print(f"text: {text}")
    print(f"text1: {text1}")
    print(f"text2: {text2}")
    print("")

    print("[tokenize method]")
    print(f"- tokenizer.tokenize: {tokenizer.tokenize(text)}")
    print(f"- tokenizer.custom_tokenizer.tokenize{tokenizer.custom_tokenizer.tokenize(text)}")
    print("")

    print("[encode method]")
    print(f"- one input (text): {tokenizer(text).input_ids}")
    print(f"- two inputs (text1, text2): {tokenizer(text1, text2).input_ids}")
    print("")

    print("[encode method - w max_length]")
    print(f"- one input (text): {tokenizer(text, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids}")
    print(f"- two inputs (text1, text2): {tokenizer(text1, text2, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids}")
    print("")

    print("[decode method - w special tokens]")
    print(f"- one input (text):          {tokenizer.decode(tokenizer(text).input_ids)}")
    print(f"- two inputs (text1, text2): {tokenizer.decode(tokenizer(text1, text2).input_ids)}")
    print("")

    print("[decode method - w/o special tokens]")
    print(f"- one input (text):          {tokenizer.decode(tokenizer(text).input_ids, skip_special_tokens=True)}")
    print(f"- two inputs (text1, text2): {tokenizer.decode(tokenizer(text1, text2).input_ids, skip_special_tokens=True)}")
    print("")

    print("[decode method - w special tokens and max_length]")
    print(f"- one input (text):          {tokenizer.decode(tokenizer(text, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids)}")
    print(f"- two inputs (text1, text2): {tokenizer.decode(tokenizer(text1, text2, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids)}")
    print("")

    print("[decode method - w/o special tokens and max_length]")
    print(f"- one input (text):          {tokenizer.decode(tokenizer(text, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids, skip_special_tokens=True)}")
    print(f"- two inputs (text1, text2): {tokenizer.decode(tokenizer(text1, text2, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids, skip_special_tokens=True)}")
    print("")
