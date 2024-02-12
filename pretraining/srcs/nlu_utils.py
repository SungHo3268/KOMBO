import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from transformers import BertConfig
from token_fusing.srcs.models import KorNLIModel, KorSTSModel, NSMCModel, PAWS_XModel, KorQuADModel, ClinicalNERModel, CustomBertForPreTraining, KorFinASCModel
from token_fusing.srcs.tokenizers import BertTokenizer
from tokenization.srcs.functions import get_tokenizer


def get_bert_tokenizer(args):
    custom_tokenizer = get_tokenizer(args.tok_type, config_path="tokenization/utils/tok_info.cfg")
    if args.tok_type in ['subword', 'morphemeSubword']:
        custom_tokenizer.load_model(f"tokenization/resources/{args.tok_name}/tok.model")

    tokenizer = BertTokenizer(vocab_file=f"tokenization/resources/{args.tok_name}/tok.vocab",
                              custom_tokenizer=custom_tokenizer,
                              max_length=args.max_seq_len,
                              )
    return tokenizer

def get_config(args):
    prefix = args.model_name.split("-")[0].lower() + "_"
    prefix = prefix.replace("bert_", "")

    bert_config_path = f"token_fusing/utils/{prefix}bert_config.json"

    config = BertConfig.from_json_file(bert_config_path)
    temp_args = vars(args)
    for key in temp_args:
        if temp_args[key] is not None:
            config.update({key: temp_args[key]})
    return config


def get_task_model(args, tokenizer):
    config = get_config(args)
    config.vocab_size = tokenizer.vocab_size
    if ('var' in config.tok_type) or ('distinct' in config.tok_type) or ('position' in config.tok_type):
        config.update({"space_symbol_id": tokenizer.space_symbol_id,
                       "empty_jamo_id": tokenizer.empty_jamo_id,
                       })

    if args.task_name == 'KorNLI':
        model = KorNLIModel(config)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    elif args.task_name == 'KorSTS':
        model = KorSTSModel(config)
        criterion = nn.MSELoss()
    elif args.task_name == 'NSMC':
        model = NSMCModel(config)
        criterion = nn.CrossEntropyLoss()
    elif args.task_name == 'PAWS_X':
        model = PAWS_XModel(config)
        criterion = nn.CrossEntropyLoss()
    elif args.task_name == 'KorQuAD':
        model = KorQuADModel(config)
        criterion = None  # nn.CrossEntropyLoss() is already embedded in the KorQuADModel
    elif args.task_name == 'ClinicalNER':
        model = ClinicalNERModel(config)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    elif args.task_name == 'KorFinASC':
        model = KorFinASCModel(config)
        criterion = nn.CrossEntropyLoss()
    else:
        print("It's a Wrong Task Name. Please enter the right task name among [KorNLI, KorSTS, NSMC, PAWS_X]")
        raise ValueError

    # reload the checkpoint of the model
    if args.save_dir:
        if args.task_name == "KorQuAD" and hasattr(config, "embedding_type"):
            model_path = os.path.join(args.save_dir, "pytorch_model.bin")
            model.bert = CustomBertForPreTraining.from_pretrained(model_path, config=config)
            print("Complete to reload the checkpoint of the model from above save directory.")
        else:
            print(f"Save directory: {args.save_dir}")
            model_path = os.path.join(args.save_dir, "pytorch_model.bin")
            if 'bert' in args.model_name.lower():
                # pre_model = BertForPreTraining.from_pretrained(model_path, config=config)
                # model.bert.load_state_dict(pre_model.bert.state_dict())
                # del pre_model

                save_dict = torch.load(model_path)
                state_dict = dict()
                for key in save_dict:
                    if 'bert' in key:
                        state_dict['.'.join(key.split(".")[1:])] = save_dict[key]
                model.bert.load_state_dict(state_dict)
                print("Complete to reload the checkpoint of the model from above save directory.")
    return config, model, criterion

def get_optimizer(args, model):
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.optimizer == 'adamw':
        from torch.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
    elif args.optimizer == 'adamwscale':
        from charformer.srcs.functions import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
        )
    elif args.optimizer == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            relative_step=False,
        )
    else:
        raise NotImplementedError
    return optimizer

def get_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )
        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.num_warmup_steps,
            last_epoch=-1,
        )
        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.total_steps - args.num_warmup_steps,
            eta_min=args.final_cosine,
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.num_warmup_steps]
        )
    elif args.lr_scheduler == 'legacy':
        import math
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            LambdaLR,
        )
        num_steps_optimizer1 = math.ceil(args.total_steps * 0.9)
        iters_left_for_optimizer2 = args.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / args.learning_rate if step else 1e-2 / args.learning_rate
        )
        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                    min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.learning_rate
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif args.lr_scheduler == 'constant':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
        )
    elif args.lr_scheduler == 'linear':
        from transformers import get_linear_schedule_with_warmup

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps
        )
    else:
        raise NotImplementedError

    return lr_scheduler
