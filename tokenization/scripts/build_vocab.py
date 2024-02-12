import os
import sys
import json
import argparse
from pprint import pprint
sys.path.append(os.getcwd())
from tokenization.srcs.functions import get_tokenizer, build_vocab


# ----------------------------------------
#                 Argparse
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--host', type=str, default='dummy')
parser.add_argument('--port', type=int, default=56789)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--tok_type', type=str, default="ko", help="ko  |  en")
parser.add_argument('--tok_name', type=str, default="jamo_position5", help="jamo  |  char  |  morpheme  |  subword  |  morphemeSubword  |  word  |  stroke  |  cji  |  bts")
parser.add_argument('--vocab_size', type=int, default=200)
parser.add_argument('--config_path', type=str, default="tokenization/utils/tok_info.cfg")
parser.add_argument('--input_corpus', type=str, default="datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt")
parser.add_argument('--output_dir', type=str, default="tokenization/resources")
parser.add_argument('--n_jobs', type=int, default=20)
args = parser.parse_args()


# ----------------------------------------
#             Initial Setting
# ----------------------------------------
print(f"\n[ Build {args.tok_name} vocabulary on '{args.input_corpus}' ]\n")
input_f = '_' + args.input_corpus.split('/')[1] if args.tok_name in ['subword', 'morphemeSubword'] else ''
args.output_dir = os.path.join(args.output_dir, f"{args.tok_name}_{args.tok_type}{input_f}_{args.vocab_size//1000}k") if args.vocab_size >= 1000 else os.path.join(args.output_dir, f"{args.tok_name}_{args.tok_type}{input_f}_{args.vocab_size}")
os.makedirs(args.output_dir, exist_ok=True)

# Save the arguments info
pprint(vars(args), indent=4, sort_dicts=False)
print('')
json.dump(vars(args), open(os.path.join(args.output_dir, 'tok_build_info.json'), 'w'), indent=4, sort_keys=False)


# -----------------------------------
#         Get the Tokenizer
# -----------------------------------
tokenizer = get_tokenizer(args.tok_name, args.config_path)


# -----------------------------------
#         Build the vocabulary
# -----------------------------------
build_vocab(args, tokenizer)
