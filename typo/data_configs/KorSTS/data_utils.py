import os
import sys
import json
sys.path.append(os.getcwd())
from typo.srcs.typo_generator import generate_typo

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


def load_task_dataset(typo_type, typo_rate, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name = 'KorSTS'

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
            new_dataset = {'sentence1': [],
                           'sentence2': [],
                           'label': [],
                           }
            for i in range(len(dataset['label'])):
                sentence1 = dataset['sentence1'][i]
                sentence2 = dataset['sentence2'][i]
                label = dataset['label'][i]

                sentence1 = generate_typo(sentence1, typo_type=typo_type, typo_rate=typo_rate)
                sentence2 = generate_typo(sentence2, typo_type=typo_type, typo_rate=typo_rate)

                new_dataset['sentence1'].append(sentence1)
                new_dataset['sentence2'].append(sentence2)
                new_dataset['label'].append(label)

            total_dataset[d_type] = new_dataset

        total_dataset['label_map'] = label_map
        json.dump(total_dataset, open(typo_dir, "w"), indent=4)
    return total_dataset


if __name__ == '__main__':
    data = load_task_dataset(typo_type="random", typo_rate=0.1, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True)
