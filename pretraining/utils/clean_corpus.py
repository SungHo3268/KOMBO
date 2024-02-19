import re
import sys
import os
from tqdm.auto import tqdm
import argparse
import string


def clean_str(text: str) -> str:
    PUNCT = '\\'.join(string.punctuation)
    PUNCT_MAPPING = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", '”': '"', '“': '"', "£": "e", '∞': 'infinity',
                     'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi',
                     '\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': '', '·': '.'}
    HTML = re.compile('<[^>]*>')
    ONLY_REMAIN = re.compile(rf"[^A-Za-z0-9ㄱ-ㅣ가-힣{PUNCT}\s]")     # string.punctuation

    text = re.sub(HTML, "", text)

    for p in PUNCT_MAPPING:
        text = text.replace(p, PUNCT_MAPPING[p])
    text = re.sub(ONLY_REMAIN, "", text)

    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    parser.add_argument('--port', type=int, default=56789)
    parser.add_argument('--input_corpus', type=str, default='kowiki', help="kowiki  |  namuwiki")
    args = parser.parse_args()

    if args.input_corpus == 'kowiki':
        file_path = "datasets/wiki/wikiextracted/ko-wiki-20220923.txt"
    elif args.input_corpus == 'namuwiki':
        file_path = "datasets/namuwiki/extracted/namuwiki_20200302.txt"
    else:
        sys.exit("[ERROR] WRONG corpus type.")


    temp_dataset = []
    doc = []
    with open(file_path, "r", encoding="utf-8") as fin:
        dataset = fin.readlines()
        data_size = len(dataset)
        for line in tqdm(dataset, desc='Cleaning the data...', total=data_size, bar_format="{l_bar}{bar:15}{r_bar}"):
            if args.input_corpus == 'namuwiki' and line == '\n':
                temp_dataset.append(doc)
                doc = []
                continue
            elif args.input_corpus == 'kowiki':
                if line.isspace() or len(line) == 0:
                    continue
                elif '</doc>' in line:
                    temp_dataset.append(doc)
                    doc = []
                    continue

            line = line.strip()
            line = clean_str(line)
            if line.isspace() or len(line.strip()) == 0:
                continue
            line = " ".join(line.split())
            line = line.replace('( )', '()')
            doc.append(line)


    file_name = os.path.basename(file_path)
    output_path = os.path.join(file_path.replace(file_name, 'clean-doc-' + file_name))
    num_docs = len(temp_dataset)
    with open(output_path, "w", encoding="utf-8") as fout:
        for doc in tqdm(temp_dataset, desc='Saving the data...', total=num_docs, bar_format="{l_bar}{bar:15}{r_bar}"):
            if len(doc) < 2:
                continue

            for line in doc:
                if line.isspace() or len(line) == 0:
                    continue
                else:
                    fout.write(line + '\n')
            fout.write('\n\n')
    print("done.\n")
