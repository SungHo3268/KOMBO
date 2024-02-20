from tokenization.srcs.tokenizers import (
    JamoTokenizer,
    CharTokenizer,
    MorphemeTokenizer,
    SubwordTokenizer,
    MorphemeSubwordTokenizer,
    WordTokenizer,
    StrokeTokenizer,
    CjiTokenizer,
    BtsTokenizer,
    StrokeVarTokenizer,
    CjiVarTokenizer,
    BtsVarTokenizer,
    JamoDistinctTokenizer,
)
from collections import Counter
from itertools import chain
from tqdm import tqdm
from multiprocessing import Pool
import time
import os


def get_tokenizer(tok_name, config_path="tokenization/utils/tok_info.cfg"):
    tokenizer = None
    if tok_name == 'jamo':
        tokenizer = JamoTokenizer(config_path)
    elif tok_name == 'char':
        tokenizer = CharTokenizer(config_path)
    elif tok_name == 'morpheme':
        tokenizer = MorphemeTokenizer(config_path)
    elif tok_name == 'subword':
        tokenizer = SubwordTokenizer(config_path)
    elif tok_name == 'morphemeSubword':
        mecab = MorphemeTokenizer(config_path)
        sentencepiece = SubwordTokenizer(config_path)
        tokenizer = MorphemeSubwordTokenizer(config_path=config_path, mecab=mecab, sp=sentencepiece)
    elif tok_name == 'word':
        tokenizer = WordTokenizer(config_path)
    elif tok_name == 'stroke':
        tokenizer = StrokeTokenizer(config_path)
    elif tok_name == 'cji':
        tokenizer = CjiTokenizer(config_path)
    elif tok_name == 'bts':
        tokenizer = BtsTokenizer(config_path)
    elif tok_name == 'stroke_var':
        tokenizer = StrokeVarTokenizer(config_path)
    elif tok_name == 'cji_var':
        tokenizer = CjiVarTokenizer(config_path)
    elif tok_name == 'bts_var':
        tokenizer = BtsVarTokenizer(config_path)
    elif tok_name == 'jamo_distinct':
        tokenizer = JamoDistinctTokenizer(config_path)
    return tokenizer


def get_vocab(tok_name, lang, vocab_size, bpe_corpus):
    if type(vocab_size) == int:
        if vocab_size < 1000:
            vocab_size = str(vocab_size)
        else:
            vocab_size = f'{vocab_size//1000}k'
    tok = f"{tok_name}_{lang}_{vocab_size}" if tok_name not in ['subword', 'morphemeSubword'] else f"{tok_name}_{lang}_{bpe_corpus}_{vocab_size}"
    raw_vocab = open(f"tokenization/resources/{tok}/tok.vocab", 'r', encoding='utf8').readlines()
    id2tok = []
    for line in raw_vocab:
        voc = line.strip().split('\t')[0]
        id2tok.append(voc)
    tok2id = {voc: i for i, voc in enumerate(id2tok)}
    return id2tok, tok2id


def build_vocab(args, tokenizer):
    if args.tok_name in ['subword', 'morphemeSubword']:
        # Training the BPE model
        tokenizer.train_model(args)         # save the tok.vocab and tok.model
        """
        # -------- Save the results --------
        # fairseq vocab
        with open(os.path.join(args.output_dir, "fairseq.vocab"), "w") as fout:
            with open(os.path.join(args.output_dir, "tok.vocab"), "r") as fin:
                start_idx = 4 + len(tokenizer.config['special_symbols'].split(","))  # pad[0], unk[1], bos[2], eos[3] + special_symbols(cls[4], sep[5], mask[6])
                for line in fin.readlines()[start_idx:]:
                    splitted = line.split("\t")
                    fout.write(f"{' '.join(splitted)}")
        """
    else:
        counter = Counter()
        start_time = time.time()
        print(f"Start tokenization ...")
        data_size = len(open(args.input_corpus, "r", encoding="utf-8").readlines())

        if args.tok_name in ['word', 'morpheme']:
            with open(args.input_corpus, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc='tokenization...', total=data_size, bar_format="{l_bar}{bar:15}{r_bar}"):
                    tokenized = tokenizer.tokenize(line.strip())
                    counter.update(tokenized)
        else:
            with open(args.input_corpus, "r", encoding="utf-8") as f:
                with Pool(args.n_jobs) as p:
                    tokenized = tqdm(p.imap_unordered(tokenizer.tokenize, f, chunksize=args.n_jobs), bar_format="{l_bar}{bar:15}{r_bar}", total=data_size, desc="Train set: ")
                    counter.update(chain.from_iterable(tokenized))

        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Complete tokenization for all files. (elapsed time: {elapsed_time})\n")

        # special tokens
        special_tokens = [tokenizer.config["pad_piece"], tokenizer.config["unk_piece"], tokenizer.config["bos_piece"], tokenizer.config["eos_piece"]]
        special_tokens.extend(tokenizer.config["special_symbols"].split(","))       # pad[0], unk[1], bos[2], eos[3] + special_symbols(cls[4], sep[5], mask[6])

        # slice with vocab size
        vocab = counter.most_common(args.vocab_size - len(special_tokens))

        # # print out-of-vocabulary
        # total_freq = sum(counter.values())
        # oov_freq = total_freq - sum([v[1] for v in vocab])
        # print(f"oov: {oov_freq}/{total_freq} ({oov_freq * 100.0 / total_freq:.2f}%)\n")

        # save vocab
        print("Write tok vocab file...")
        output_vocab_path = os.path.join(args.output_dir, "tok.vocab")
        with open(output_vocab_path, "w", encoding="utf-8") as f:
            for token in special_tokens:
                f.write(f"{token}\t-1\n")
            for token, freq in vocab:
                f.write(f"{token}\t{freq}\n")
    """
    # save fairseq vocab
    print("Write fairseq vocab file...")
    with open(os.path.join(args.output_dir, "fairseq.vocab"), "w") as fout:
        with open(os.path.join(args.output_dir, "tok.vocab"), "r") as fin:
            start_idx = 4 + len(tokenizer.config["special_symbols"].split(","))  # pad[0], unk[1], bos[2], eos[3] + special_symbols(cls[4], sep[5], mask[6])
            for line in fin.readlines()[start_idx:]:
                splitted = line.split("\t")
                fout.write(f"{' '.join(splitted)}")
    """
    if args.tok_name == 'word':
        tokenizer.close()
    print("--------------------- done ---------------------\n")
