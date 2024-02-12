import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from nlu_tasks.srcs.preprocess import clean_text
import json


"""
Korean STS dataset (KorSTS)
[https://github.com/kakaobrain/KorNLUDatasets/tree/master/KorSTS]
It is translated from the STS-B dataset.
It comprises 8,628 sentence pairs—5,749 for training, 1,500 for development, and 1,379 for test.

When the two sentences are given, the task assesses the gradations of semantic similarity between two sentences with a scale from 0 to 5.

[  'genre' \t 'filename' \t 'year' \t 'id' \t 'score' \t 'sentence1' \t 'sentence2'\n,
   'genre' \t 'filename' \t 'year' \t 'id' \t 'score' \t 'sentence1' \t 'sentence2'\n,
       ...
   'genre' \t 'filename' \t 'year' \t 'id' \t 'score' \t 'sentence1' \t 'sentence2'
]                                                  
"""


def load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name = 'KorSTS'
    data_dir = f"datasets/nlu_tasks/{task_name}/"
    if do_hangeulize:
        data_path = os.path.join(data_dir, f'processed_data_{remain_lang}_hangeulized.json')
    else:
        data_path = os.path.join(data_dir, f'processed_data_{remain_lang}.json')
    if data_remove:
        data_path = data_path.replace(".json", "_dr.json")

    print(f"\n##### Loading the {task_name} dataset #####")
    print(f"Data path: {data_path}\n")
    if os.path.exists(data_path):
        total_dataset = json.load(open(data_path, "r"))
    else:
        total_dataset = {'train': dict(),
                         'dev': dict(),
                         'test': dict()
                         }

        for d_type in total_dataset:
            dataset = {'sentence1': [], 'sentence2': [], 'label': []}
            with open(os.path.join(data_dir, f'sts-{d_type}.tsv'), "r", encoding="utf8") as fr:
                raw_dataset = fr.readlines()[1:]
                for i, line in tqdm(enumerate(raw_dataset), desc=f"* {d_type.upper()} set...", bar_format="{l_bar}{bar:10}{r_bar}", total=len(raw_dataset)):
                    line = line.strip().split('\t')
                    if len(line) != 7:
                        print(f"[ERROR] {repr(line)}, line {i}")
                        continue
                    sentence1 = line[5]
                    sentence2 = line[6]
                    score = line[4]

                    sentence1 = clean_text(sentence1, remain_lang, do_hangeulize, data_remove)
                    sentence2 = clean_text(sentence2, remain_lang, do_hangeulize, data_remove)
                    if sentence1 is None or len(sentence1) == 0:
                        continue
                    if sentence2 is None or len(sentence2) == 0:
                        continue

                    dataset['sentence1'].append(sentence1)
                    dataset['sentence2'].append(sentence2)
                    dataset['label'].append(float(score))
                total_dataset[d_type] = dataset
                fr.close()

        for d_type in total_dataset:
            assert len(total_dataset[d_type]['sentence1']) == len(total_dataset[d_type]['sentence2']) == len(total_dataset[d_type]['label'])

        total_dataset['label_map'] = None
        json.dump(total_dataset, open(data_path, "w"))
    return total_dataset


if __name__ == '__main__':
    data = load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True)
