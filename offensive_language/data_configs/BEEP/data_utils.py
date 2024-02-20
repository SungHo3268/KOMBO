import os
import csv
import json
from nlu_tasks.srcs.preprocess import clean_text


def load_task_dataset(args):
    data_dir = "datasets/offensive_language/BEEP/"

    if args.do_hangeulize:
        data_path = os.path.join(data_dir, f'processed_data_{args.remain_lang}_hangeulized.json')
    else:
        data_path = os.path.join(data_dir, f'processed_data_{args.remain_lang}.json')
    if args.data_remove:
        data_path = data_path.replace(".json", "_dr.json")
    if args.binary:
        data_path = data_path.replace(".json", "_binary.json")

    print(f"\n##### Loading the BEEP dataset #####")
    print(f"Data path: {data_path}\n")

    if os.path.exists(data_path):
        total_dataset = json.load(open(data_path, "r"))

    else:
        os.makedirs(data_dir, exist_ok=True)
        total_dataset = dict()

        if args.binary:
            label_map = {'none': 0,
                         'offensive': 1,
                         'hate': 1
                         }
        else:
            label_map = {'none': 0,
                         'offensive': 1,
                         'hate': 2}

        dataset = {'sentence': [], 'label': []}
        len_sent = 0
        with open(os.path.join(data_dir, "train.tsv"), "r", encoding="utf8") as fr:
            raw_dataset = csv.reader(fr, delimiter="\t")
            for i, line in enumerate(raw_dataset):
                if i == 0:
                    continue

                sentence = line[0]
                label = label_map[line[3]]

                sentence = clean_text(sentence, args.remain_lang, args.do_hangeulize, args.data_remove)
                if sentence is None or len(sentence) == 0:
                    continue

                dataset['sentence'].append(sentence)
                dataset['label'].append(int(label))
                len_sent += len(sentence)

            total_dataset['train'] = dataset
            fr.close()

        dataset = {'sentence': [], 'label': []}
        len_sent = 0
        with open(os.path.join(data_dir, 'dev.tsv'), "r", encoding="utf8") as fr:
            raw_dataset = csv.reader(fr, delimiter="\t")
            for i, line in enumerate(raw_dataset):
                if i == 0:
                    continue
                sentence = line[0]
                label = label_map[line[3]]

                sentence = clean_text(sentence, args.remain_lang, args.do_hangeulize, args.data_remove)
                if sentence is None or len(sentence) == 0:
                    continue

                dataset['sentence'].append(sentence)
                dataset['label'].append(int(label))
                len_sent += len(sentence)

            total_dataset['dev'] = dataset
            total_dataset['test'] = dataset
            fr.close()

        total_dataset['label_map'] = label_map

        dev_size = len(dataset['label'])
        print(f"Average length: {len_sent / dev_size}")

        for d_type in total_dataset:
            if d_type == 'label_map':
                continue
            assert len(total_dataset[d_type]['sentence']) == len(total_dataset[d_type]['label'])

        json.dump(total_dataset, open(data_path, "w"))
    return total_dataset
