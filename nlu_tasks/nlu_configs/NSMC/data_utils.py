import os
import sys
sys.path.append(os.getcwd())
import json
from tqdm import tqdm
from nlu_tasks.srcs.preprocess import clean_text
import numpy as np


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

# def clean_text(sentence):
#     sentence = sentence.strip().replace("&lt", "").replace("&gt", "")
#     sentence = sentence.replace('"""', '')
#     sentence = sentence.replace('""', '')
#     sentence = sentence.replace("'''", '')
#     sentence = sentence.replace("''", '')
#     sentence = sentence.replace("1234567890", '')
#     sentence = sentence.replace("123456789", '')
#     sentence = sentence.replace("0987654321", '')
#     sentence = sentence.replace("987654321", '')
#     reducing_list = ["ㅋㅋㅋㅋ", "ㄷㄷㄷㄷ", "ㄱㄱㄱㄱ", "ㅇㅇㅇㅇ", "ㅈㅈㅈㅈ", "ㅅㅅㅅㅅ", "ㅂㅂㅂㅂ",
#                      "ㅏㅏㅏㅏ", "ㅑㅑㅑㅑ", "!!!!", "????", "~~~~", "₩₩₩₩", "....", ",,,,",
#                      ";;;;", "::::", "ㅠㅠㅠㅠ", "ㅜㅜㅜㅜ", "////", "^^^^", "====",
#                      "----", ">>>>", "<<<<", "----", "ㅡㅡㅡㅡ", "____", "++++"]
#     for dummy in reducing_list:
#         if dummy in sentence:
#             sentence = sentence.replace(dummy, dummy[:2])
#     clean_flag = None
#     while not clean_flag:
#         clean_flag = True
#         for dummy in reducing_list:
#             if dummy in sentence:
#                 sentence = sentence.replace(dummy, dummy[:2])
#                 clean_flag = False
#     PUNCT = '\\'.join(string.punctuation)
#     PUNCT_MAPPING = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
#                      "—": "-", "–": "-", "’": "'", '”': '"', '“': '"', "£": "e", '∞': 'infinity',
#                      'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3',
#                      'π': 'pi',
#                      '\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': '', '·': '.'}
#     ONLY_REMAIN = re.compile(rf"[^A-Za-z0-9ㄱ-ㅣ가-힣{PUNCT}\s]")  # string.punctuation
#
#     for p in PUNCT_MAPPING:
#         sentence = sentence.replace(p, PUNCT_MAPPING[p])
#     sentence = re.sub(ONLY_REMAIN, "", sentence)
#     sentence = sentence.strip()
#     return sentence


def load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name = 'NSMC'
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

        label_map = {'negative': 0,
                     'positive': 1}

        dataset = {'sentence': [], 'label': []}
        with open(os.path.join(data_dir, 'ratings_train.txt'), "r", encoding="utf8") as fr:
            raw_dataset = fr.readlines()
            for i, line in tqdm(enumerate(raw_dataset), desc="* Training & Dev set...", bar_format="{l_bar}{bar:10}{r_bar}", total=len(raw_dataset)):
                if i == 0:
                    continue
                line = line.strip().split('\t')
                sentence = line[1]
                label = line[2]

                sentence = clean_text(sentence, remain_lang, do_hangeulize, data_remove)
                if sentence is None or len(sentence) == 0:
                    continue

                dataset['sentence'].append(sentence)
                dataset['label'].append(int(label))

            train_size = len(dataset['label'])
            dev_size = round(train_size * 0.1)

            per = np.random.permutation(np.arange(train_size))
            rand = per[:dev_size]

            train_dataset = {'sentence': [], 'label': []}
            dev_dataset = {'sentence': [], 'label': []}
            # for i in tqdm(range(train_size), desc="* Dev set...", bar_format="{l_bar}{bar:10}{r_bar}"):
            for i in range(train_size):
                if i in rand:
                    for key in dev_dataset:
                        dev_dataset[key].append(dataset[key][i])
                else:
                    for key in dev_dataset:
                        train_dataset[key].append(dataset[key][i])

            total_dataset['train'] = train_dataset
            total_dataset['dev'] = dev_dataset
            fr.close()

        dataset = {'sentence': [], 'label': []}
        with open(os.path.join(data_dir, 'ratings_test.txt'), "r", encoding="utf8") as fr:
            raw_dataset = fr.readlines()
            for i, line in tqdm(enumerate(raw_dataset), desc="* Test set...", bar_format="{l_bar}{bar:10}{r_bar}", total=len(raw_dataset)):
                if i == 0:
                    continue
                line = line.strip().split('\t')
                sentence = line[1]
                label = line[2]

                sentence = clean_text(sentence, remain_lang, do_hangeulize, data_remove)
                if sentence is None or len(sentence) == 0:
                    continue

                dataset['sentence'].append(sentence)
                dataset['label'].append(int(label))
            total_dataset['test'] = dataset
            fr.close()

        for d_type in total_dataset:
            assert len(total_dataset[d_type]['sentence']) == len(total_dataset[d_type]['label'])
        total_dataset['label_map'] = label_map

        json.dump(total_dataset, open(data_path, "w"))
    return total_dataset


if __name__ == '__main__':
    processed = load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True)
