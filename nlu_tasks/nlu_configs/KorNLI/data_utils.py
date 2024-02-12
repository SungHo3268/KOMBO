import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from nlu_tasks.srcs.preprocess import clean_text
import json


"""
Korean Natural Language Inference (KorNLI)
[https://github.com/kakaobrain/KorNLUDatasets/tree/master/KorNLI]
It sourced from three different NLI datasets
    * Train: SNLI + MNLI
    * Dev: XNLI
    * Test: XNLI
    
It is composed of 950,354 sentence pairs: 942,854 for training, 2,490 for development, and 5,010 for test.

When a pair of sentences-a premise and a hypothesis-are given, the model classifies their relationship into one out of three categories: 
[entailment, contradiction, and neutral]

[  'sentence1' \t 'sentence2' \t gold_label,\n
   'sentence1' \t 'sentence2' \t gold_label,\n
       ...
   'sentence1' \t 'sentence2' \t gold_label
]                                                  
"""


def load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name = 'KorNLI'
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
        total_dataset = dict()

        label_map = {'entailment': 0,
                     'neutral': 1,
                     'contradiction': 2
                     }
        all_data_path = {'train': ['multinli.train.ko.tsv', 'snli_1.0_train.ko.tsv'],
                         'dev': ['xnli.dev.ko.tsv'],
                         'test': ['xnli.test.ko.tsv']
                         }

        for d_type in all_data_path:
            data = all_data_path[d_type]
            dataset = {'sentence1': [], 'sentence2': [], 'label': []}
            for d, d_path in enumerate(data):
                with open(os.path.join(data_dir, d_path), "r", encoding="utf8") as fr:
                    raw_dataset = fr.readlines()
                    for i, line in tqdm(enumerate(raw_dataset), desc=f"* {d_type.upper()} set {d+1}...", bar_format="{l_bar}{bar:10}{r_bar}", total=len(raw_dataset)):
                        if i == 0:
                            continue
                        line = line.strip().split('\t')
                        sentence1 = line[0]
                        sentence2 = line[1]
                        label = label_map[line[2]]

                        sentence1 = clean_text(sentence1, remain_lang, do_hangeulize, data_remove)
                        sentence2 = clean_text(sentence2, remain_lang, do_hangeulize, data_remove)
                        if sentence1 is None or len(sentence1) == 0:
                            continue
                        if sentence2 is None or len(sentence2) == 0:
                            continue

                        dataset['sentence1'].append(sentence1)
                        dataset['sentence2'].append(sentence2)
                        dataset['label'].append(label)
                    fr.close()
            total_dataset[d_type] = dataset

        for d_type in total_dataset:
            assert len(total_dataset[d_type]['sentence1']) == len(total_dataset[d_type]['sentence2']) == len(total_dataset[d_type]['label'])
        total_dataset['label_map'] = label_map

        json.dump(total_dataset, open(data_path, "w"))
    return total_dataset


if __name__ == '__main__':
    dataset = load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True)

