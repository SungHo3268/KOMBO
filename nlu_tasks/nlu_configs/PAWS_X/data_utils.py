import os
import sys
sys.path.append(os.getcwd())
import json
from nlu_tasks.srcs.preprocess import clean_text
from tqdm import tqdm


"""
Paraphrase Identiﬁcation: (PAWS_X) Dataset
[https://github.com/google-research-datasets/paws]
This dataset contains 108,463 human-labeled and 656k noisily labeled pairs that feature the importance of 
modeling structure, context, and word order information for the problem of paraphrase identification.
    
When two sentences are given, the task is that the model clarify whether these two sentences are paraphrased.
Each sample is labeled with 0 (different meaning) or 1 (paraphrased).

[  'id' \t 'sentence1' \t 'sentence2' \t 'label'\n,
   'id' \t 'sentence1' \t 'sentence2' \t 'label'\n,
            ...
   'id' \t 'sentence1' \t 'sentence2' \t 'label'\n,
]                                                  
"""

def load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name = 'PAWS_X'
    data_dir = f"datasets/nlu_tasks/{task_name}/ko/"

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

        label_map = {'different': 0,
                     'paraphrased': 1}
        total_dataset['label_map'] = label_map

        for cur_type in ['train', 'dev', 'test']:
            dataset = {'sentence1': [], 'sentence2': [], 'label': []}
            with open(os.path.join(data_dir, f'{cur_type}.tsv'), "r", encoding="utf8") as fr:
                raw_dataset = fr.readlines()
                for i, line in tqdm(enumerate(raw_dataset), desc=f"* Loading {cur_type.upper()} set...", bar_format="{l_bar}{bar:10}{r_bar}", total=len(raw_dataset)):
                    if i == 0:
                        continue
                    line = line.strip().split('\t')
                    sentence1 = line[1]
                    sentence2 = line[2]
                    label = line[3]
                    if sentence1 == 'NS' or sentence2 == 'NS':
                        continue

                    sentence1 = clean_text(sentence1, remain_lang, do_hangeulize, data_remove)
                    sentence2 = clean_text(sentence2, remain_lang, do_hangeulize, data_remove)
                    if sentence1 is None or len(sentence1) == 0:
                        continue
                    if sentence2 is None or len(sentence2) == 0:
                        continue

                    dataset['sentence1'].append(sentence1)
                    dataset['sentence2'].append(sentence2)
                    dataset['label'].append(int(label))
                total_dataset[cur_type] = dataset
                fr.close()

        for d_type in total_dataset:
            if d_type == 'label_map':
                continue
            assert len(total_dataset[d_type]['sentence1']) == len(total_dataset[d_type]['sentence2']) == len(total_dataset[d_type]['label'])

        json.dump(total_dataset, open(data_path, "w"))
    return total_dataset


if __name__ == '__main__':
    data = load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True)
