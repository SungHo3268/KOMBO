import os
import json
import numpy as np
from tqdm import tqdm
from nlu_tasks.srcs.preprocess import clean_text


def load_task_dataset(args):
    data_dir = f"datasets/toxic/KOLD/"
    if args.do_hangeulize:
        data_path = os.path.join(data_dir, f'processed_data_{args.remain_lang}_hangeulized.json')
    else:
        data_path = os.path.join(data_dir, f'processed_data_{args.remain_lang}.json')
    if args.data_remove:
        data_path = data_path.replace(".json", "_dr.json")

    print(f"\n##### Loading the K_MHaS dataset #####")
    print(f"Data path: {data_path}\n")

    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as fr:
            trimmed_dataset = json.load(fr)
    else:
        raw_file_path = os.path.join(data_dir, "kold_v1.json")
        with open(raw_file_path, "r", encoding="utf-8") as fr:
            dataset = json.load(fr)
        off_list = list()
        tgt_list = list()
        grp_list = list()
        for data in dataset:
            off = data["OFF"]
            tgt = data["TGT"]
            grp = data["GRP"]
            if off is not None:
                off_list.append(off)
            if tgt is not None:
                tgt_list.append(tgt)
            if grp is not None:
                grp_list.append(grp)

        off_list = list(set(off_list))
        tgt_list = list(set(tgt_list))
        grp_list = list(set(grp_list))
        unique_grp = []
        for grp in grp_list:
            unique_grp.extend([g.split("-")[-1].strip() for g in grp.split('&')])
        grp_list = list(set(unique_grp))

        off_list.sort()
        tgt_list.sort()
        grp_list.sort()

        off_map = {off: i for i, off in enumerate(off_list)}
        tgt_map = {tgt: i for i, tgt in enumerate(tgt_list)}
        grp_map = {grp: i for i, grp in enumerate(grp_list)}

        grp_map["others"], grp_map["LGBTQ+"] = grp_map["LGBTQ+"], grp_map["others"]

        grp_map["southeast_asian"] = grp_map["homosexual"]
        grp_map["white"] = grp_map["queer"]
        grp_map["homosexual"] = grp_map["LGBTQ+"]
        grp_map["queer"] = grp_map["LGBTQ+"]
        grp_eye = np.eye(len(grp_list) - 2)     # 2 is for the "queer" and "homosexual".


        level_A = {'title': [],
                   'comment': [],
                   'label': [],
                   'label_map': off_map
                   }
        level_B = {'title': [],
                   'comment': [],
                   'label': [],
                   'label_map': tgt_map
                   }
        level_C = {'title': [],
                   'comment': [],
                   'label': [],
                   'label_map': grp_map
                   }
        for data in tqdm(dataset, desc="Preprocessing", total=len(dataset), bar_format="{l_bar}{bar:15}{r_bar}"):
            off = data["OFF"]
            tgt = data["TGT"]
            grp = data["GRP"]
            title = data["title"]
            comment = data["comment"]
            title = clean_text(title, args.remain_lang, args.do_hangeulize, args.data_remove).strip()
            comment = clean_text(comment, args.remain_lang, args.do_hangeulize, args.data_remove).strip()

            if off is not None:
                off_label = off_map[off]
                level_A['title'].append(title)
                level_A['comment'].append(comment)
                level_A['label'].append(off_label)
            if tgt is not None:
                tgt_label = tgt_map[tgt]
                level_B['title'].append(title)
                level_B['comment'].append(comment)
                level_B['label'].append(tgt_label)
            if grp is not None:
                grp_label = np.sum([grp_eye[grp_map[g.split("-")[-1].strip()]] for g in grp.split('&')], axis=0)
                level_C['title'].append(title)
                level_C['comment'].append(comment)
                level_C['label'].append([int(i) for i in grp_label])

        trimmed_dataset = {'A': level_A,
                           'B': level_B,
                           'C': level_C,
                           }
        with open(data_path, "w", encoding="utf-8") as fw:
            json.dump(trimmed_dataset, fw, indent=2)

    trimmed_dataset = trimmed_dataset[args.hparams.label_level]
    label_map = trimmed_dataset.pop("label_map")

    train_dataset, dev_dataset, test_dataset = args.split_dataset(trimmed_dataset, args.split_ratio)
    total_dataset = {
        'train': train_dataset,
        'dev': dev_dataset,
        'test': test_dataset,
        'label_map': label_map
    }
    return total_dataset


def split_dataset(dataset, split_ratio):
    # Get datasets for each steps
    split_ratio = list(map(float, split_ratio.split('_')))
    _shuffle_dataset = shuffle_dataset(dataset)

    train_size = int(len(_shuffle_dataset['label']) * split_ratio[0])
    dev_size = int(len(_shuffle_dataset['label']) * split_ratio[1])

    train_dataset = {
        'sentence1': _shuffle_dataset['title'][: train_size],
        'sentence2': _shuffle_dataset['comment'][: train_size],
        'label': _shuffle_dataset['label'][: train_size]
    }
    dev_dataset = {
        'sentence1': _shuffle_dataset['title'][train_size: train_size+dev_size],
        'sentence2': _shuffle_dataset['comment'][train_size: train_size+dev_size],
        'label': _shuffle_dataset['label'][train_size: train_size+dev_size]
    }
    test_dataset = {
        'sentence1': _shuffle_dataset['title'][train_size+dev_size:],
        'sentence2': _shuffle_dataset['comment'][train_size+dev_size:],
        'label': _shuffle_dataset['label'][train_size+dev_size:]
    }
    return train_dataset, dev_dataset, test_dataset


def shuffle_dataset(dataset):
    keys = list(dataset.keys())             # sentence1, sentence2, label, ...
    data_size = len(dataset[keys[0]])
    per = np.random.permutation(range(data_size))

    # self.logger.info("Shuffling the dataset")
    new_dataset = dict()
    for key in keys:
        new_dataset[key] = [dataset[key][idx] for idx in per]
    return new_dataset
