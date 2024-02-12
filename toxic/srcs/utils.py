import os
import sys
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
sys.path.append(os.getcwd())
from nlu_tasks.srcs.nlu_utils import get_config
from nlu_tasks.srcs.preprocess import clean_text
from toxic.srcs.models import KOLDModel


def load_data(file_path="datasets/toxic/kold_v1.json"):
    trimmed_file_path = os.path.join(os.path.dirname(file_path), "trimmed_kold_v1.json")
    if os.path.exists(trimmed_file_path):
        with open(trimmed_file_path, "r", encoding="utf-8") as fr:
            trimmed_dataset = json.load(fr)
    else:
        with open(file_path, "r", encoding="utf-8") as fr:
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
        # grp_eye = np.eye(len(grp_list))

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
            title = clean_text(title, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=False).strip()
            comment = clean_text(comment, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=False).strip()

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
        with open(trimmed_file_path, "w", encoding="utf-8") as fw:
            json.dump(trimmed_dataset, fw, indent=2)
    return trimmed_dataset


def get_task_model(args, tokenizer):
    if args.model_name == "klue-bert-base":
        config = args
        model = KOLDModel(config)
        if args.label_level == 'C':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        config = get_config(args)
        config.vocab_size = tokenizer.vocab_size
        if ('var' in config.tok_type) or ('distinct' in config.tok_type):
            config.update({"space_symbol_id": tokenizer.space_symbol_id,
                           "empty_jamo_id": tokenizer.empty_jamo_id,
                           })

        model = KOLDModel(config)
        if args.label_level == 'C':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # reload the checkpoint of the model
        if args.save_dir:
            print(f"Save directory: {args.save_dir.split('/')[-2]}")
            model_path = os.path.join(args.save_dir, "pytorch_model.bin")

            save_dict = torch.load(model_path)
            bert_state_dict = dict()
            classifier_state_dict = dict()
            for key in save_dict:
                if 'bert' in key:
                    bert_state_dict['.'.join(key.split(".")[1:])] = save_dict[key]
                elif 'classifier' in key:
                    classifier_state_dict['.'.join(key.split(".")[1:])] = save_dict[key]

            model.bert.load_state_dict(bert_state_dict)
            if len(classifier_state_dict) != 0:
                model.classifier.load_state_dict(classifier_state_dict)
            print("Complete to reload the checkpoint of the model from above save directory.")
    return config, model, criterion
