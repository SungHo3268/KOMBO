import re
import os
from tqdm.auto import tqdm
import argparse
import string


def clean_str(text: str, lang: str, corpus_type: str) -> str:
    PUNCT = '\\'.join(string.punctuation)
    PUNCT_MAPPING = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", '”': '"', '“': '"', "£": "e", '∞': 'infinity',
                     'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi',
                     '\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': '', '·': '.'}
    HTML = re.compile('<[^>]*>')
    ONLY_REMAIN = re.compile(rf"[^A-Za-z0-9ㄱ-ㅣ가-힣{PUNCT}\s]")     # string.punctuation


    text = re.sub(HTML, "", text)
    if lang == 'ko':
        if corpus_type == 'aihub':
            for p in PUNCT_MAPPING:
                text = text.replace(p, PUNCT_MAPPING[p])
        elif corpus_type == 'wiki':
            for p in PUNCT_MAPPING:
                text = text.replace(p, PUNCT_MAPPING[p])
            text = re.sub(ONLY_REMAIN, "", text)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    parser.add_argument('--port', type=int, default=56789)
    parser.add_argument('--lang', type=str, default='ko', help='ko  |  en')
    parser.add_argument('--corpus_type', type=str, default='wiki', help="wiki  |  aihub")
    parser.add_argument('--input_corpus', type=str, default="datasets/wiki/wikiextracted/ko-wiki-20220923.txt")
    args = parser.parse_args()

    file_name = os.path.basename(args.input_corpus)
    output_path = os.path.join(args.input_corpus.replace(file_name, 'clean-'+file_name))
    with open(args.input_corpus, "r", encoding="utf-8") as fin:
        dataset = fin.readlines()
        data_size = len(dataset)
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(dataset, desc='Cleaning the data...', total=data_size, bar_format="{l_bar}{bar:15}{r_bar}"):
                line = line.strip()
                # preprocess
                if line.isspace() or len(line) == 0 or ('</doc>' in line):
                    continue

                line = clean_str(line, args.lang, args.corpus_type)

                # postprocess
                if line.isspace() or len(line.strip()) == 0:
                    continue
                line = " ".join(line.split())
                line = line.replace('( )', '()')

                fout.write(line+'\n')
    print("done.\n")
