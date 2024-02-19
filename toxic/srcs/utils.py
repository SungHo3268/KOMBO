import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.getcwd())
from nlu_tasks.srcs.nlu_utils import get_config
from toxic.srcs.models import KOLDModel, KMHaSModel, BEEPModel


def get_task_model(args, tokenizer):
    config = get_config(args)
    config.vocab_size = tokenizer.vocab_size
    if ('var' in config.tok_type) or ('distinct' in config.tok_type):
        config.update({"space_symbol_id": tokenizer.space_symbol_id,
                       "empty_jamo_id": tokenizer.empty_jamo_id,
                       })

    if args.task_name.lower() == 'kold':
        model = KOLDModel(config)
    elif args.task_name.lower() == 'kmhas':
        model = KMHaSModel(config)
    elif args.task_name.lower() == 'beep':
        model = BEEPModel(config)
    else:
        raise ValueError(f"Invalid task name: {args.task_name}")

    if (((args.task_name.lower() == 'kold') and (args.label_level == 'C')) or
            (args.task_name.lower() == 'kmhas')):
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
