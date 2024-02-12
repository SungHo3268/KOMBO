import os
import sys
import time
import hgtk
import MeCab
import sentencepiece as spm
from unicodedata import normalize
from mosestokenizer import MosesTokenizer
from tqdm.auto import tqdm
sys.path.append(os.getcwd())
from tokenization.srcs.base_tokenizer import *
from tokenization.srcs.bts_rule import subchar_dict, subchar_reverse_dict, jamo_seperator

class JamoTokenizer(BaseTokenizer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['jamo_info']

    def tokenize(self, text: str) -> List[str]:
        """Normalization Form KD (NFKD)	: Compatibility Decomposition"""
        tokens = list(self.config['space_symbol'].join([normalize("NFKD", token) for token in text.strip().split(" ")]))
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Normalization Form KC (NFKC) :	Compatibility Decomposition, followed by Canonical Composition"""
        text = normalize("NFKC", "".join(tokens)).replace(self.config['space_symbol'], " ")
        return text


class CharTokenizer(BaseTokenizer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['char_info']

    def tokenize(self, text: str) -> List[str]:
        text = text.strip().replace(" ", self.config['space_symbol'])
        tokens = list(text)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace(self.config['space_symbol'], " ").strip()
        return text


class MorphemeTokenizer(BaseTokenizer):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.config = self.config_parser['morpheme_info']

        self.mecab = MeCab.Tagger(f"--dicdir utils/mecab-0.996-ko-0.9.2/lib/mecab/dic/mecab-ko-dic")

    def tokenize(self, text: str) -> List[str]:
        text = text.strip().replace("\t", "").replace("\n", "")
        text_ptr = 0
        tokens = []
        for mor in self.mecab.parse(text).split("\n"):
            if "\t" in mor:
                splitted = mor.split("\t")
                token = splitted[0]

                if text[text_ptr] == " ":
                    while text[text_ptr] == " ":  # if it's a white space, move one point to the right
                        text_ptr += 1
                    assert (
                            text[text_ptr] == token[0]
                    ), f"{repr(text)}//{text_ptr}//{text[text_ptr]}//{token}//{token[0]}\n"

                    tokens.append(self.config["space_symbol"])

                tokens.append(token)
                text_ptr += len(token)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace(self.config['space_symbol'], " ").strip()
        return text


class SubwordTokenizer(BaseTokenizer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['subword_info']

        self.sp = spm.SentencePieceProcessor()
        self.reverse = self.config['reverse']

    def load_model(self, model_path):
        self.sp.Load(model_path)

    def train_model(self, args):
        print(f"Start the training BPE ...")
        start_time = time.time()

        train_cmd = f"--input={args.input_corpus} "
        train_cmd += f"--model_prefix={os.path.join(args.output_dir, 'tok')} "  # model name to save
        train_cmd += f"--vocab_size={args.vocab_size} "
        train_cmd += f"--model_type=bpe "
        train_cmd += f"--character_coverage={self.config['character_coverage']} "
        train_cmd += f"--normalization_rule_name={self.config['ko_normalization_rule_name']} " if args.tok_type == 'ko' else f"--character_coverage={self.config['en_normalization_rule_name']} "
        train_cmd += f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        train_cmd += f"--pad_piece={self.config['pad_piece']} "
        train_cmd += f"--unk_piece={self.config['unk_piece']} "
        train_cmd += f"--bos_piece={self.config['bos_piece']} "
        train_cmd += f"--eos_piece={self.config['eos_piece']} "
        train_cmd += f"--unk_surface={self.config['unk_surface']} "
        train_cmd += f"--user_defined_symbols={self.config['special_symbols']} "

        print('\n\n\n', train_cmd, '\n\n\n')

        spm.SentencePieceTrainer.Train(train_cmd)

        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Complete tokenization for all files. (Elapsed time: {elapsed_time})")

        self.sp.Load(os.path.join(args.output_dir, 'tok.model'))

    def tokenize(self, text: str) -> List[str]:
        tokens = self.sp.EncodeAsPieces(text.strip())

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▁", " ").strip()
        return text


class MorphemeSubwordTokenizer(BaseTokenizer):
    def __init__(self, config_path, mecab: MorphemeTokenizer, sp: SubwordTokenizer):
        super().__init__(config_path)
        self.config = self.config_parser['morphemeSubword_info']

        self.mecab = mecab
        self.sp = sp

    def get_morpheme_corpus(self, args):
        morpheme_corpus_dir = os.path.join(args.input_corpus.replace(f"/{os.path.basename(args.input_corpus)}", ''),
                                           'morpheme_tokenized')
        os.makedirs(morpheme_corpus_dir, exist_ok=True)

        start_time = time.time()
        print(f"Start Mecab(morpheme-aware) tokenization ...")

        corpus = []
        data_size = len(open(args.input_corpus, "r", encoding="utf-8").readlines())
        with open(args.input_corpus, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc='tokenization...', bar_format="{l_bar}{bar:15}{r_bar}", total=data_size):
                tokenized = self.mecab.tokenize(line.strip())
                corpus.append(tokenized)
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Complete tokenization for all files. (Elapsed time: {elapsed_time})")

        # save the mecab tokenized corpus
        with open(os.path.join(morpheme_corpus_dir, os.path.basename(args.input_corpus)), "w", encoding="utf-8") as f:
            for tokens in corpus:
                f.write(" ".join(tokens) + "\n")
        print(f"Save the morpheme-aware tokenized corpus.")

    def load_model(self, model_path):
        self.sp.load_model(model_path)

    def train_model(self, args):
        morpheme_corpus = os.path.join(args.input_corpus.replace(f"/{os.path.basename(args.input_corpus)}", ''),
                                       'morpheme_tokenized', os.path.basename(args.input_corpus))
        if not os.path.exists(morpheme_corpus):
            self.get_morpheme_corpus(args)

        args.input_corpus = morpheme_corpus
        self.sp.train_model(args)

    def tokenize(self, text: str) -> List[str]:
        tokenized = self.mecab.tokenize(text)
        tokenized = self.sp.tokenize(" ".join(tokenized))

        tokens = []
        for i in range(0, len(tokenized)):
            if i + 1 < len(tokenized) and (tokenized[i] == "▁" and tokenized[i + 1] == "▃"):
                continue
            if tokenized[i] == "▁▃":
                tokenized[i] = "▃"
            tokens.append(tokenized[i])
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▁", "").replace(" ", "").replace("▃", " ").strip()
        return text


class WordTokenizer(BaseTokenizer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['word_info']
        self.tokenizer = MosesTokenizer()

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer(text.strip())
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        # if tokens[-1] in list(string.punctuation):      # control the end of the tokens
        #     tokens = tokens[:-2] + [(tokens[-2] + tokens[-1])]
        text = " ".join(tokens).strip()
        return text

    def close(self):
        self.tokenizer.close()


class StrokeTokenizer(BaseTokenizer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['bts_units_info']
        self.special_tokens = [
                                  self.config["pad_piece"],
                                  self.config["unk_piece"],
                                  self.config["bos_piece"],
                                  self.config["eos_piece"]
                              ] + self.config["special_symbols"].split(",")

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        splitted = text.split(' ')
        num_words = len(splitted)
        for i, word in enumerate(splitted):
            decomposed_word = []
            for j, char in enumerate(word):
                char = char.strip()
                if len(char) == 0 or char.isspace():
                    continue
                try:
                    cho, joong, jong = hgtk.letter.decompose(char)

                    cho = [sub for sub in subchar_dict[cho]] if len(cho) > 0 else []
                    joong = [joong] if len(joong) > 0 else []
                    jong = [sub for sub in subchar_dict[jong]] if len(jong) > 0 else []

                    char_seq = cho + joong + jong

                    decomposed_word.extend(char_seq + [self.config["end_char_symbol"]]) if j != len(word) - 1 \
                        else decomposed_word.extend(char_seq)
                except hgtk.exception.NotHangulException:
                    decomposed_word.extend(char + self.config['end_char_symbol'])
                    continue
            tokens.extend(decomposed_word + [self.config["space_symbol"]]) if i != num_words - 1 \
                else tokens.extend(decomposed_word)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        joint_text = "".join(tokens).strip()
        for spc in self.special_tokens:
            if spc in joint_text:
                joint_text = joint_text.replace(spc,
                                                self.config["end_char_symbol"] + spc + self.config["end_char_symbol"])
        space_splitted = joint_text.split(self.config["space_symbol"])
        words = []
        for word in space_splitted:
            char_splitted = word.strip().split(self.config["end_char_symbol"])
            detok_word = ""
            for char in char_splitted:
                jamos = jamo_seperator(char)
                jamo_word = ""
                for jamo in jamos:
                    try:
                        jamo = subchar_reverse_dict[jamo]
                        jamo_word += jamo
                    except KeyError:
                        jamo_word += jamo.strip()
                jamo_word += self.config["end_char_symbol"]  # it serves as the char_end symbol

                detok = hgtk.text.compose(jamo_word, compose_code=self.config["end_char_symbol"])
                detok_word += detok
            words.append(detok_word)
        text = ' '.join(words)
        return text


class CjiTokenizer(BaseTokenizer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['bts_units_info']
        self.special_tokens = [
                                  self.config["pad_piece"],
                                  self.config["unk_piece"],
                                  self.config["bos_piece"],
                                  self.config["eos_piece"]
                              ] + self.config["special_symbols"].split(",")

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        splitted = text.split(' ')
        num_words = len(splitted)
        for i, word in enumerate(splitted):
            decomposed_word = []
            for j, char in enumerate(word):
                char = char.strip()
                if len(char) == 0 or char.isspace():
                    continue
                try:
                    cho, joong, jong = hgtk.letter.decompose(char)

                    cho = [cho] if len(cho) > 0 else []
                    joong = [sub for sub in subchar_dict[joong]] if len(joong) > 0 else []
                    jong = [jong] if len(jong) > 0 else []

                    char_seq = cho + joong + jong

                    decomposed_word.extend(char_seq + [self.config["end_char_symbol"]]) if j != len(word) - 1 \
                        else decomposed_word.extend(char_seq)
                except hgtk.exception.NotHangulException:
                    decomposed_word.extend(char + self.config['end_char_symbol'])
                    continue
            tokens.extend(decomposed_word + [self.config["space_symbol"]]) if i != num_words - 1 \
                else tokens.extend(decomposed_word)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        joint_text = "".join(tokens).strip()
        for spc in self.special_tokens:
            if spc in joint_text:
                joint_text = joint_text.replace(spc,
                                                self.config["end_char_symbol"] + spc + self.config["end_char_symbol"])
        space_splitted = joint_text.split(self.config["space_symbol"])
        words = []
        for word in space_splitted:
            char_splitted = word.strip().split(self.config["end_char_symbol"])
            detok_word = ""
            for char in char_splitted:
                jamos = jamo_seperator(char)
                jamo_word = ""
                for jamo in jamos:
                    try:
                        jamo = subchar_reverse_dict[jamo]
                        jamo_word += jamo
                    except KeyError:
                        jamo_word += jamo.strip()
                jamo_word += self.config["end_char_symbol"]  # it serves as the char_end symbol

                detok = hgtk.text.compose(jamo_word, compose_code=self.config["end_char_symbol"])
                detok_word += detok
            words.append(detok_word)
        text = ' '.join(words)
        return text


class BtsTokenizer(BaseTokenizer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['bts_units_info']
        self.special_tokens = [
                                  self.config["pad_piece"],
                                  self.config["unk_piece"],
                                  self.config["bos_piece"],
                                  self.config["eos_piece"]
                              ] + self.config["special_symbols"].split(",")

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        splitted = text.split(' ')
        num_words = len(splitted)
        for i, word in enumerate(splitted):
            decomposed_word = []
            for j, char in enumerate(word):
                char = char.strip()
                if len(char) == 0 or char.isspace():
                    continue
                try:
                    cho, joong, jong = hgtk.letter.decompose(char)

                    cho = [sub for sub in subchar_dict[cho]] if len(cho) > 0 else []
                    joong = [sub for sub in subchar_dict[joong]] if len(joong) > 0 else []
                    jong = [sub for sub in subchar_dict[jong]] if len(jong) > 0 else []

                    char_seq = cho + joong + jong

                    decomposed_word.extend(char_seq + [self.config["end_char_symbol"]]) if j != len(word) - 1 \
                        else decomposed_word.extend(char_seq)
                except hgtk.exception.NotHangulException:
                    decomposed_word.extend(char + self.config['end_char_symbol'])
                    continue
            tokens.extend(decomposed_word + [self.config["space_symbol"]]) if i != num_words - 1 \
                else tokens.extend(decomposed_word)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        joint_text = "".join(tokens).strip()
        for spc in self.special_tokens:
            if spc in joint_text:
                joint_text = joint_text.replace(spc,
                                                self.config["end_char_symbol"] + spc + self.config["end_char_symbol"])
        space_splitted = joint_text.split(self.config["space_symbol"])
        words = []
        for word in space_splitted:
            char_splitted = word.strip().split(self.config["end_char_symbol"])
            detok_word = ""
            for char in char_splitted:
                jamos = jamo_seperator(char)
                jamo_word = ""
                for jamo in jamos:
                    try:
                        jamo = subchar_reverse_dict[jamo]
                        jamo_word += jamo
                    except KeyError:
                        jamo_word += jamo.strip()
                jamo_word += self.config["end_char_symbol"]  # it serves as the char_end symbol

                detok = hgtk.text.compose(jamo_word, compose_code=self.config["end_char_symbol"])
                detok_word += detok
            words.append(detok_word)
        text = ' '.join(words)
        return text


class StrokeVarTokenizer(BaseTokenizer):
    """
    Basically, this tokenizer is decomposed into BTS units. However, in the later model,
    while synthesizing embeddings in units of jamo, it is assumed that each token is distinguished
    for a total of 4 tokens, including the chosung, joongsung, jongsung, and the last or third empty token.
    """

    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['bts_units_var_info']
        self.special_tokens = [
                                  self.config["pad_piece"],
                                  self.config["unk_piece"],
                                  self.config["bos_piece"],
                                  self.config["eos_piece"]
                              ] + self.config["special_symbols"].split(",")

        self.space_symbol = self.config["space_symbol"]
        self.empty_jamo = self.config["empty_jamo_symbol"]
        self.end_char = self.config["end_char_symbol"]

        self.cons_len = 4
        self.vow_len = 1
        self.char_len = self.cons_len * 2 + self.vow_len

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        splitted = text.strip().split(' ')
        num_words = len(splitted)
        for i, word in enumerate(splitted):
            decomposed_word = []
            for j, char in enumerate(word):
                char = char.strip()
                if len(char) == 0 or char.isspace():
                    continue
                try:
                    cho, joong, jong = hgtk.letter.decompose(char)

                    cho = [sub for sub in subchar_dict[cho]] if len(cho) > 0 else [self.empty_jamo]
                    joong = [joong] if len(joong) > 0 else [self.empty_jamo]
                    jong = [sub for sub in subchar_dict[jong]] if len(jong) > 0 else [self.empty_jamo]

                    cho += [self.empty_jamo] * (self.cons_len - len(cho))
                    joong += [self.empty_jamo] * (self.vow_len - len(joong))
                    jong += [self.empty_jamo] * (self.cons_len - len(jong))

                    char_seq = cho + joong + jong
                except hgtk.exception.NotHangulException:
                    char_seq = [char] + \
                               [self.empty_jamo] * (self.char_len - 1)
                decomposed_word.extend(char_seq)
            tokens.extend(
                decomposed_word + [self.space_symbol] + [self.empty_jamo] * (self.char_len - 1)) if i != num_words - 1 \
                else tokens.extend(decomposed_word)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        joint_text = "".join(tokens).strip()
        for s_token in ["[SEP]", "[CLS]", "[PAD]"]:
            if s_token in joint_text:
                joint_text = joint_text.replace(s_token,
                                                s_token + self.space_symbol + self.empty_jamo * (self.char_len - 6))
        padded_text = [joint_text[i * self.char_len: (i + 1) * self.char_len]
                       .replace(self.empty_jamo, "")
                       for i in range(len(joint_text) // self.char_len)
                       ]
        words = []
        for word in padded_text:
            jamos = jamo_seperator(word)
            jamo_word = ""
            for jamo in jamos:
                try:
                    jamo = subchar_reverse_dict[jamo]
                    jamo_word += jamo
                except KeyError:
                    jamo_word += jamo.strip()
            jamo_word += self.end_char  # it serves as the char_end symbol

            detok = hgtk.text.compose(jamo_word, compose_code=self.end_char)
            words.append(detok)

        text = ''.join(words).replace(self.space_symbol, ' ')
        return text


class CjiVarTokenizer(BaseTokenizer):
    """
    Basically, this tokenizer is decomposed into BTS units. However, in the later model,
    while synthesizing embeddings in units of jamo, it is assumed that each token is distinguished
    for a total of 4 tokens, including the chosung, joongsung, jongsung, and the last or third empty token.
    """

    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['bts_units_var_info']
        self.special_tokens = [
                                  self.config["pad_piece"],
                                  self.config["unk_piece"],
                                  self.config["bos_piece"],
                                  self.config["eos_piece"]
                              ] + self.config["special_symbols"].split(",")

        self.space_symbol = self.config["space_symbol"]
        self.empty_jamo = self.config["empty_jamo_symbol"]
        self.end_char = self.config["end_char_symbol"]

        self.cons_len = 1
        self.vow_len = 5
        self.char_len = self.cons_len * 2 + self.vow_len

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        splitted = text.strip().split(' ')
        num_words = len(splitted)
        for i, word in enumerate(splitted):
            decomposed_word = []
            for j, char in enumerate(word):
                char = char.strip()
                if len(char) == 0 or char.isspace():
                    continue
                try:
                    cho, joong, jong = hgtk.letter.decompose(char)

                    cho = [cho] if len(cho) > 0 else [self.empty_jamo]
                    joong = [sub for sub in subchar_dict[joong]] if len(joong) > 0 else [self.empty_jamo]
                    jong = [jong] if len(jong) > 0 else [self.empty_jamo]

                    cho += [self.empty_jamo] * (self.cons_len - len(cho))
                    joong += [self.empty_jamo] * (self.vow_len - len(joong))
                    jong += [self.empty_jamo] * (self.cons_len - len(jong))

                    char_seq = cho + joong + jong
                except hgtk.exception.NotHangulException:
                    char_seq = [char] + \
                               [self.empty_jamo] * (self.char_len - 1)
                decomposed_word.extend(char_seq)
            tokens.extend(
                decomposed_word + [self.space_symbol] + [self.empty_jamo] * (self.char_len - 1)) if i != num_words - 1 \
                else tokens.extend(decomposed_word)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        joint_text = "".join(tokens).strip()
        for s_token in ["[SEP]", "[CLS]", "[PAD]"]:
            if s_token in joint_text:
                joint_text = joint_text.replace(s_token,
                                                s_token + self.space_symbol + self.empty_jamo * (self.char_len - 6))
        padded_text = [joint_text[i * self.char_len: (i + 1) * self.char_len]
                       .replace(self.empty_jamo, "")
                       for i in range(len(joint_text) // self.char_len)
                       ]
        words = []
        for word in padded_text:
            jamos = jamo_seperator(word)
            jamo_word = ""
            for jamo in jamos:
                try:
                    jamo = subchar_reverse_dict[jamo]
                    jamo_word += jamo
                except KeyError:
                    jamo_word += jamo.strip()
            jamo_word += self.end_char  # it serves as the char_end symbol

            detok = hgtk.text.compose(jamo_word, compose_code=self.end_char)
            words.append(detok)

        text = ''.join(words).replace(self.space_symbol, ' ')
        return text


class BtsVarTokenizer(BaseTokenizer):
    """
    Basically, this tokenizer is decomposed into BTS units. However, in the later model,
    while synthesizing embeddings in units of jamo, it is assumed that each token is distinguished
    for a total of 4 tokens, including the chosung, joongsung, jongsung, and the last or third empty token.
    """

    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['bts_units_var_info']
        self.special_tokens = [
                                  self.config["pad_piece"],
                                  self.config["unk_piece"],
                                  self.config["bos_piece"],
                                  self.config["eos_piece"]
                              ] + self.config["special_symbols"].split(",")

        self.space_symbol = self.config["space_symbol"]
        self.empty_jamo = self.config["empty_jamo_symbol"]
        self.end_char = self.config["end_char_symbol"]

        self.cons_len = 4
        self.vow_len = 5
        self.char_len = self.cons_len * 2 + self.vow_len

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        splitted = text.strip().split(' ')
        num_words = len(splitted)
        for i, word in enumerate(splitted):
            decomposed_word = []
            for j, char in enumerate(word):
                char = char.strip()
                if len(char) == 0 or char.isspace():
                    continue
                try:
                    cho, joong, jong = hgtk.letter.decompose(char)

                    cho = [sub for sub in subchar_dict[cho]] if len(cho) > 0 else [self.empty_jamo]
                    joong = [sub for sub in subchar_dict[joong]] if len(joong) > 0 else [self.empty_jamo]
                    jong = [sub for sub in subchar_dict[jong]] if len(jong) > 0 else [self.empty_jamo]

                    cho += [self.empty_jamo] * (self.cons_len - len(cho))
                    joong += [self.empty_jamo] * (self.vow_len - len(joong))
                    jong += [self.empty_jamo] * (self.cons_len - len(jong))

                    char_seq = cho + joong + jong
                except hgtk.exception.NotHangulException:
                    char_seq = [char] + \
                               [self.empty_jamo] * (self.char_len - 1)
                decomposed_word.extend(char_seq)
            tokens.extend(
                decomposed_word + [self.space_symbol] + [self.empty_jamo] * (self.char_len - 1)) if i != num_words - 1 \
                else tokens.extend(decomposed_word)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        joint_text = "".join(tokens).strip()
        for s_token in ["[SEP]", "[CLS]", "[PAD]"]:
            if s_token in joint_text:
                joint_text = joint_text.replace(s_token,
                                                s_token + self.space_symbol + self.empty_jamo * (self.char_len - 6))
        padded_text = [joint_text[i * self.char_len: (i + 1) * self.char_len]
                       .replace(self.empty_jamo, "")
                       for i in range(len(joint_text) // self.char_len)
                       ]
        words = []
        for word in padded_text:
            jamos = jamo_seperator(word)
            jamo_word = ""
            for jamo in jamos:
                try:
                    jamo = subchar_reverse_dict[jamo]
                    jamo_word += jamo
                except KeyError:
                    jamo_word += jamo.strip()
            jamo_word += self.end_char  # it serves as the char_end symbol

            detok = hgtk.text.compose(jamo_word, compose_code=self.end_char)
            words.append(detok)

        text = ''.join(words).replace(self.space_symbol, ' ')
        return text

class JamoDistinctTokenizer(BaseTokenizer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config = self.config_parser['jamo_var_info']
        self.special_tokens = [
                                  self.config["pad_piece"],
                                  self.config["unk_piece"],
                                  self.config["bos_piece"],
                                  self.config["eos_piece"]
                              ] + self.config["special_symbols"].split(",")

        self.space_symbol = self.config["space_symbol"]
        self.empty_jamo = self.config["empty_jamo_symbol"]

        self.cons_len = 1
        self.vow_len = 1
        self.char_len = self.cons_len * 2 + self.vow_len

        # There are valid keys which can be located in jongsung regardless of chosung
        self.cho_jong_converter = {'\u1100': '\u11A8',  # ㄱ
                                   '\u1101': '\u11A9',  # ㄲ
                                   '\u1102': '\u11AB',  # ㄴ
                                   '\u1115': '\u11C6',  # ㅦ
                                   '\u115C': '\u11AC',  # ㄵ
                                   '\u115D': '\u11AD',  # ㄶ
                                   '\u1103': '\u11AE',  # ㄷ
                                   '\u1105': '\u11AF',  # ㄹ
                                   '\uA964': '\u11B0',  # ㄺ
                                   '\uA968': '\u11B1',  # ㄻ
                                   '\uA969': '\u11B2',  # ㄼ
                                   '\uA96C': '\u11B3',  # ㄽ
                                   '\u111A': '\u11B6',  # ㅀ
                                   '\u1106': '\u11B7',  # ㅁ
                                   '\u1107': '\u11B8',  # ㅂ
                                   '\u1121': '\u11B9',  # ㅄ
                                   '\u1109': '\u11BA',  # ㅅ
                                   '\u110A': '\u11BB',  # ㅆ
                                   '\u110B': '\u11BC',  # ㅇ
                                   '\u110C': '\u11BD',  # ㅈ
                                   '\u110E': '\u11BE',  # ㅊ
                                   '\u110F': '\u11BF',  # ㅋ
                                   '\u1110': '\u11C0',  # ㅌ
                                   '\u1111': '\u11C1',  # ㅍ
                                   '\u1112': '\u11C2',  # ㅎ
                                   }

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        splitted = text.strip().split(' ')
        num_words = len(splitted)
        for i, word in enumerate(splitted):
            decomposed_word = []
            for j, char in enumerate(word):
                char = char.strip()
                if len(char) == 0 or char.isspace():
                    continue
                try:
                    cho, joong, jong = [normalize("NFKD", token) for token in hgtk.letter.decompose(char)]
                    try:
                        jong = self.cho_jong_converter[jong]
                    except:
                        jong = jong

                    cho = [cho] if len(cho) > 0 else [self.empty_jamo]
                    joong = [joong] if len(joong) > 0 else [self.empty_jamo]
                    jong = [jong] if len(jong) > 0 else [self.empty_jamo]

                    cho += [self.empty_jamo] * (self.cons_len - len(cho))
                    joong += [self.empty_jamo] * (self.vow_len - len(joong))
                    jong += [self.empty_jamo] * (self.cons_len - len(jong))

                    char_seq = cho + joong + jong
                except hgtk.exception.NotHangulException:
                    char_seq = [char] + \
                               [self.empty_jamo] * (self.char_len - 1)
                decomposed_word.extend(char_seq)
            tokens.extend(decomposed_word + [self.space_symbol] + [self.empty_jamo] * (self.char_len - 1)) if i != num_words - 1 \
                else tokens.extend(decomposed_word)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        joint_text = "".join(tokens).strip()
        joint_text = joint_text \
            .replace(self.empty_jamo, "") \
            .replace(self.space_symbol, " ")
        text = normalize("NFKC", joint_text)
        return text


if __name__ == '__main__':
    # _tokenizer = MorphemeTokenizer(config_path="tokenization/utils/tok_info.cfg")
    # _tokenizer = StrokeVarTokenizer(config_path="tokenization/utils/tok_info.cfg")
    # _tokenizer = CjiVarTokenizer(config_path="tokenization/utils/tok_info.cfg")
    # _tokenizer = BtsVarTokenizer(config_path="tokenization/utils/tok_info.cfg")
    # _tokenizer = JamoDistinctTokenizer(config_path="tokenization/utils/tok_info.cfg")
    mecab = MorphemeTokenizer(config_path="tokenization/utils/tok_info.cfg")
    sentencepiece = SubwordTokenizer(config_path="tokenization/utils/tok_info.cfg")
    _tokenizer = MorphemeSubwordTokenizer(config_path="tokenization/utils/tok_info.cfg", mecab=mecab, sp=sentencepiece)
    _tokenizer.load_model(f"tokenization/resources/morphemeSubword_ko_wiki_32k/tok.model")

    _tokenizer = SubwordTokenizer(config_path="tokenization/utils/tok_info.cfg")
    _tokenizer.load_model(f"tokenization/resources/subword_ko_wiki_32k/tok.model")

    # _text = "ㄷㄷ"
    # _text = "ㅏㅏ 그거?"
    # _text = "[CLS] 안녕하세요. [MASK] 뵙겠습니다."
    _text = "한영 자판 상태에서 히라가나를 입력할 경우 ㄸ+한자 키를 누르면 된다. 가타카나의 경우 장음 등 일부 문자를 제외하면, 꼭 ㅃ+한자 조합을 해야 한다. \t뒤에꺼는 테스트 문장입니다."
    # _text = "한국어 표기에 쓰이는 문자인 한글은 세종대왕이 원로대신들의 반대를 무릅쓰고 언문청() 또는 정음청()을 설치하여, 훈민정음이라는 명칭으로 1446년 반포하였다. " \
    #         "한글은 각 자음과 모음이 하나의 기호로 표시되고, 그 자음과 모음을 모아써서 소리를 나타내는 표음문자이다. 한글의 자음은 발음기관의 모양을 본뜨고 " \
    #         "모음은 천(하늘:),지(땅:ㅡ),인(사람:ㅣ)을 나타내는 각 부호의 조합으로 만든, 세계에서 유일하게 그 창제 원리가 밝혀진 문자이다. 한글 창제 이전에는 " \
    #         "삼국 시대 혹은 그 이전부터 써왔던 한자와 이를 당시의 한국인이 쓰던 입말에 맞는 한자의 소리만 따서 문자로 표기한 향찰, 구결, 이두 등이 한국어 기록 수단이었다."

    _tokens = _tokenizer.tokenize(_text)
    print(f"[tokenized]")
    print(f"length: {len(_tokens)}")
    print(_tokens)
    print("")

    print("[original text]")
    print(f"length: {len(_text)}")
    print(_text)
    print("")

    detokens = _tokenizer.detokenize(_tokens)
    print("[detokenized text]")
    print(f"length: {len(detokens)}")
    print(detokens)
