import os
import sys
import json
from tqdm import tqdm
sys.path.append(os.getcwd())
from typo.srcs.typo_generator import generate_typo
from pretraining.srcs.functions import BAR_FORMAT


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


def load_task_dataset(typo_ratio, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name = 'KorNLI'
    data_dir = f"datasets/nlu_tasks/{task_name}/"
    if do_hangeulize:
        data_path = os.path.join(data_dir, f'processed_data_{remain_lang}_hangeulized.json')
    else:
        data_path = os.path.join(data_dir, f'processed_data_{remain_lang}.json')
    if data_remove:
        data_path = data_path.replace(".json", "_dr.json")

    raw_dataset = json.load(open(data_path, "r"))
    total_dataset = dict()

    label_map = raw_dataset['label_map']
    all_data_path = ['dev', 'test']

    for d_type in all_data_path:
        dataset = raw_dataset[d_type]
        new_dataset = {'sentence1': [],
                       'sentence2': [],
                       'label': [],
                       }
        for i in range(len(dataset['label'])):
            sentence1 = dataset['sentence1'][i]
            sentence2 = dataset['sentence2'][i]
            label = dataset['label'][i]

            sentence1 = generate_typo(sentence1, typo_level="jamo", typo_ratio=typo_ratio)
            sentence2 = generate_typo(sentence2, typo_level="jamo", typo_ratio=typo_ratio)

            new_dataset['sentence1'].append(sentence1)
            new_dataset['sentence2'].append(sentence2)
            new_dataset['label'].append(label)

        total_dataset[d_type] = new_dataset

    total_dataset['label_map'] = label_map
    return total_dataset


if __name__ == '__main__':
    data = load_task_dataset(typo_ratio=0.1, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True)
