import os
import sys
import json
from tqdm import tqdm
sys.path.append(os.getcwd())
from typo.srcs.typo_generator import generate_typo
from pretraining.srcs.functions import BAR_FORMAT

"""
Naver sentiment movie corpus (NSMC)
[https://github.com/e9t/nsmc]
This is a movie review dataset in the Korean language. Reviews were scraped from Naver Movies.
The dataset construction is based on the method noted in Large movie review dataset from Maas et al., 2011.
It consists of 200K samples of which 150K are the training set and the rest 50K are the test set.
    Notice) Holding out 10 percent of the training data for development.
    
When a sentence(document) is given, the task is that the model classifies the sentiment of the sentence which is positive or negative.
Each sample is labeled with 0 (negative) or 1 (positive).

[  'id' \t 'document' \t 'label'\n,
   'id' \t 'document' \t 'label'\n,
       ...
   'id' \t 'document' \t 'label'\n,
]                                                  
"""

def load_task_dataset(typo_ratio, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name = 'NSMC'
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
    all_data_path = ['test']

    for d_type in all_data_path:
        dataset = raw_dataset[d_type]
        new_dataset = {'sentence': [],
                       'label': [],
                       }
        for i in range(len(dataset['label'])):
            sentence = dataset['sentence'][i]
            label = dataset['label'][i]

            sentence = generate_typo(sentence, typo_level="jamo", typo_ratio=typo_ratio)

            new_dataset['sentence'].append(sentence)
            new_dataset['label'].append(label)

        total_dataset[d_type] = new_dataset

    total_dataset['label_map'] = label_map
    return total_dataset


if __name__ == '__main__':
    dataset = load_task_dataset()
