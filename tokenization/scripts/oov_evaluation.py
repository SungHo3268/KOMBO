import os
import sys
import time
import argparse
from tqdm.auto import tqdm
from itertools import chain
from multiprocessing import Pool
from collections import Counter
sys.path.append(os.getcwd())
from tokenization.srcs.functions import get_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=int, default=56789)
parser.add_argument('--host', type=str, default='dummy')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--tok_type', type=str, default="ko", help="ko  |  en")
parser.add_argument('--tok_name', type=str, default="bts", help="jamo  |  char  |  morpheme  |  subword  |  morphemeSubword  |  word  |  stroke  |  cji  |  bts")
parser.add_argument('--bpe_corpus', type=str, default='wiki', help='wiki  |  aihub')
parser.add_argument('--vocab_size', type=int, default=200, help="200  |  2000  |  4000  |  8000  |  16000  |  32000  |  64000")
parser.add_argument('--config_path', type=str, default="tokenization/utils/tok_info.cfg")
parser.add_argument('--test_corpus', type=str, default="datasets/aihub/clean-aihub_news_ko_test.txt")
parser.add_argument('--output_dir', type=str, default="tokenization/resources")
parser.add_argument('--n_jobs', type=int, default=20)
args = parser.parse_args()


input_f = '_' + args.bpe_corpus if args.tok_name in ['subword', 'morphemeSubword'] else ''
save_dir = os.path.join(args.output_dir, f"{args.tok_name}_{args.tok_type}{input_f}_{args.vocab_size//1000}k") if args.vocab_size >= 1000 \
    else os.path.join(args.output_dir, f"{args.tok_name}_{args.tok_type}{input_f}_{args.vocab_size}")


# -----------------------------------
#         Get the Tokenizer
# -----------------------------------
tokenizer = get_tokenizer(args.tok_name, args.config_path)
if args.tok_name in ['subword', 'morphemeSubword']:
    model_path = os.path.join(save_dir, "tok.model")
    if not os.path.exists(model_path):
        raise IOError(f"\nThere is no model and vocab for [ {os.path.basename(save_dir)} ]'. "
                      f"Please train BPE model first by doing build_vocab.py\n\n")
    tokenizer.load_model(model_path)


# -----------------------------------
#      Get the Train Vocabulary
# -----------------------------------
vocab = []
vocab_path = os.path.join(save_dir, "tok.vocab")
with open(vocab_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        token = line.split('\t')[0]
        vocab.append(token)


# -----------------------------------
#      Get the Test Vocabulary
# -----------------------------------
counter = Counter()
start_time = time.time()
print(f"\nStart tokenization...")
with open(args.test_corpus, "r", encoding="utf-8") as f:
    if args.tok_name in ['word', 'morpheme', 'subword', 'morphemeSubword']:
        for line in tqdm(f.readlines(), bar_format="{l_bar}{bar:15}{r_bar}", desc="Test set: "):
            tokenized = tokenizer.tokenize(line.strip())
            counter.update(tokenized)
    else:
        data_size = len(open(args.test_corpus, "r", encoding="utf-8").readlines())
        with Pool(args.n_jobs) as p:
            tokenized = tqdm(p.imap_unordered(tokenizer.tokenize, f, chunksize=args.n_jobs), bar_format="{l_bar}{bar:15}{r_bar}", total=data_size, desc="Test set: ")
            counter.update(chain.from_iterable(tokenized))
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
print(f"Complete tokenization for all files. (elapsed time: {elapsed_time})\n")

# slice with vocab size
test_vocab = counter.most_common()

# print out-of-vocabulary
total_freq = sum([item[1] for item in test_vocab])
oov_freq = sum([v[1] for v in test_vocab if v[0] not in vocab])
print(f"OOV rate: {oov_freq * 100.0 / total_freq:.2f}% ({oov_freq}/{total_freq})\n")
print("==================================================================\n")
