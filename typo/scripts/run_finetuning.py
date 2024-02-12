import torch
import os
import sys
import json
import numpy as np
sys.path.append(os.getcwd())
from typo.srcs.typo_trainer import Trainer
from pretraining.srcs.functions import init_random
from pretraining.utils.base_parser import ArgsBase
from pretraining.utils.logger import get_logger
from argparse import Namespace

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


parser = ArgsBase().add_typo_task_args()
temp_args = parser.parse_args()
temp = vars(temp_args)

config = json.load(open(f"nlu_tasks/data_configs/{temp['task_name']}/config.json"))
for arg in config:
    temp[arg] = config[arg]

if temp['tok_type'] in ['jamo', 'stroke', 'cji', 'bts']:
    temp['max_seq_len'] = 256
    temp['learning_rate'] = 5e-05
elif temp['tok_type'] in ['jamo_distinct']:
    temp['max_seq_len'] = 512
    temp['learning_rate'] = 5e-05
elif temp['tok_type'] in ['stroke_var', 'cji_var', 'bts_var']:
    temp['max_seq_len'] = 1024
    temp['learning_rate'] = 5e-05
else:
    pass

args = Namespace(**temp)


init_random(seed=args.random_seed)      # initialize the random_seed to get consistent results.

if args.tok_type in ['subword', 'morphemeSubword']:
    args.tok_name = f"{args.tok_type}_{args.lang}_{args.bpe_corpus}_{args.tok_vocab_size}"
else:
    args.tok_name = f"{args.tok_type}_{args.lang}_{args.tok_vocab_size}"

prefix = ""
if 'kombo' in args.model_name:
    if args.mlm_unit != 'token':
        prefix += f"span-{args.mlm_unit}-mlm_"
    if args.jamo_fusion:
        if 'trans' in args.jamo_fusion:
            temp_fusion = args.jamo_fusion.replace("trans", f"trans{args.jamo_trans_layer}")
        else:
            temp_fusion = args.jamo_fusion
        prefix += f"jamo-{temp_fusion}_"
        if args.jamo_residual:
            prefix = prefix[:-1] + "-res_"
        if args.cho_joong_first:
            prefix = prefix[:-1] + "-cjf_"
        if args.ignore_structure:
            prefix = prefix[:-1] + "-is_"
    if args.upsampling:
        prefix += f"{args.upsampling}-up_"
        if args.upsampling_residual:
            prefix = prefix[:-1] + "-res_"
    if args.num_hidden_layers:
        prefix += f"trans{args.num_hidden_layers}_"


args.log_dir = (f"logs/{args.model_name}/{args.tok_name}/typo/{args.task_name}/"
                f"{prefix}{args.max_seq_len}t_{args.batch_size}b_{args.gradient_accumulation_steps}s_{args.max_epochs}e_{args.learning_rate}lr_{args.random_seed}rs")
os.makedirs(args.log_dir, exist_ok=True)

logger = get_logger(log_path=os.path.join(args.log_dir, "train_logs.txt"))
logger.info("")
logger.info(f"* [sys] Current available # GPUs: {torch.cuda.device_count()}\n")
logger.info("Save the parser information")
with open(os.path.join(args.log_dir, 'argparse.json'), 'w') as fw:
    json.dump(args.__dict__, fw, indent=2)
    fw.close()


logger.info("========== fine_tuning ==========")
logger.info(f"task name                 : {args.task_name}")
logger.info(f"model                     : {args.model_name}")
logger.info(f"tokenizer                 : {args.tok_name}")
if 'kombo' in args.model_name:
    if args.jamo_fusion:
        logger.info(f"jamo_fusion       : {args.jamo_fusion}")
        logger.info(f"jamo_residual     : {bool(args.jamo_residual)}")
logger.info(f"random seed               : {args.random_seed}")
logger.info(f"total epochs              : {args.max_epochs}")
logger.info(f"batch size                : {args.batch_size}")
logger.info(f"gradient accum steps      : {args.gradient_accumulation_steps}")
logger.info(f"learning rate             : {args.learning_rate}")
logger.info(f"dropout prob              : {args.dropout_rate}")
logger.info(f"warmup ratio              : {args.warmup_ratio}")
logger.info(f"max seq len               : {args.max_seq_len}")
logger.info(f"typo rates                : {args.typo_rates}")
logger.info(f"pretrained seeds          : {args.pretrained_seed}")
logger.info(f"log_dir                   : {args.log_dir}\n")


total_scores = []
seeds = [int(seed) for seed in args.pretrained_seed.split("_")]
typo_rates = [float(ratio) for ratio in args.typo_rates.split("_")]
for seed in seeds:
    logger.info(f"========== Typo Evaluation of pre-trained model with ( {seed} ) random seed ==========")
    args.save_dir = (os.path.join(f"logs/{args.model_name}/{args.tok_name}/nlu_tasks/{args.task_name}/"
                     f"{prefix}{args.max_seq_len}t_{args.batch_size}b_{args.gradient_accumulation_steps}s_{args.max_epochs}e_{args.learning_rate}lr_{seed}rs", "ckpt"))
    try:
        os.path.exists(os.path.join(args.save_dir, "pytorch_model.bin"))
    except FileNotFoundError:
        print("Please confirm the 'ckpt' fine-tuned model on NLU tasks.")

    logger.info(f"Declare the Trainer ({args.model_name}) - with ({args.tok_name}) - on ({args.task_name})...")
    trainer = Trainer(args, logger)
    logger.info("Succeed to prepare the trainer.\n")

    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [p for n, p in trainer.model.named_parameters() if not any(nd in n for nd in no_decay)]

    total_trainable_params = sum(p.numel() for p in optimizer_grouped_parameters if p.requires_grad)
    logger.info(f"Model parameters: {total_trainable_params // 1000000}M\n")

    # logger.info(f"Start the fine-tuning on {args.task_name}.")
    scores = trainer.fine_tuning(typo_rates)
    total_scores.append(scores)
    print("\n")

total_scores = np.array(total_scores)
avg_scores = np.mean(total_scores, axis=0)
std = np.std(total_scores, axis=0)

logger.info(f"######### Total TYPO RESULTS #########")
for i, typo_ratio in enumerate(typo_rates):
    logger.info(f"Typo Ratio {typo_ratio * 100}%: {avg_scores[i] * 100:.2f} ± {std[i]:.2f} [%]")
logger.info(f"######################################")
print("\n")
