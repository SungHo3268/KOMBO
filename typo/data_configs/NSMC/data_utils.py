import os
import sys
import json
sys.path.append(os.getcwd())
from typo.srcs.typo_generator import generate_typo

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


def load_task_dataset(typo_type, typo_rate, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name = 'NSMC'

    typo_dir = f"datasets/typo_tasks/{task_name}/typo_{typo_type}/typo_{typo_rate}%.json"
    if os.path.exists(typo_dir):
        total_dataset = json.load(open(typo_dir, "r"))

    else:
        os.makedirs(f"datasets/typo_tasks/{task_name}/typo_{typo_type}/", exist_ok=True)

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

                sentence = generate_typo(sentence, typo_type=typo_type, typo_rate=typo_rate)

                new_dataset['sentence'].append(sentence)
                new_dataset['label'].append(label)

            total_dataset[d_type] = new_dataset

        total_dataset['label_map'] = label_map
        json.dump(total_dataset, open(typo_dir, "w"), indent=4)
    return total_dataset


if __name__ == '__main__':
    data = load_task_dataset(typo_type="random", typo_rate=0.1, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True)
