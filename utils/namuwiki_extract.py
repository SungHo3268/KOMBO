import json
import argparse
import os
from multiprocessing import Pool
from namuwiki.extractor import extract_text
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--host', type=str, default='dummy')
parser.add_argument('--port', type=int, default=56789)
parser.add_argument('--input_corpus', type=str, default="datasets/namuwiki/raw/namuwiki_20200302.json")
parser.add_argument('--output_dir', type=str, default="datasets/namuwiki/extracted/")
args = parser.parse_args()


def work(document):
    return {
        'title': document['title'],
        'content': extract_text(document['text'])
    }


print(f"Applying Namu Wiki Extractor to {os.path.basename(args.input_corpus)}")
with open(args.input_corpus, 'r', encoding='utf-8') as input_file:
    namu_wiki = json.load(input_file)

with Pool() as pool:
    documents = pool.map(work, namu_wiki)

print("done.\n")


with open(os.path.join(args.output_dir, os.path.basename(args.input_corpus).split('.')[0] + ".txt"), "w", encoding="utf-8") as fw:
    for article in tqdm(documents, desc="Saving the file into txt format..", bar_format="{l_bar}{bar:15}{r_bar}", total=len(documents)):
        title = article['title'].strip()
        content = article['content'].strip()
        for txt in [title, content]:
            if txt.isspace() or len(txt) == 0:
                continue
            else:
                fw.write(txt + '\n')
        fw.write('\n')
print("complete.")
