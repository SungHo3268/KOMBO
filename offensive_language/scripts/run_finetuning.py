import os
import sys
import json
import torch
from argparse import Namespace
sys.path.append(os.getcwd())
from pretraining.srcs.functions import init_random
from pretraining.utils.base_parser import ArgsBase
from pretraining.utils.logger import get_logger
from offensive_language.srcs.trainer import Trainer


parser = ArgsBase().add_offensive_task_args()
temp_args = parser.parse_args()
temp = vars(temp_args)

config = json.load(open(f"offensive_language/data_configs/{temp['task_name']}/config.json"))
for arg in config:
    temp[arg] = config[arg]


if temp['tok_type'] in ['jamo', 'stroke', 'cji', 'bts']:
    temp['max_seq_len'] = 256
elif temp['tok_type'] in ['jamo_distinct']:
    temp['max_seq_len'] = 512
elif temp['tok_type'] in ['stroke_var', 'cji_var', 'bts_var']:
    temp['max_seq_len'] = 1024
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
    if args.only_contextualization:
        prefix += f"jamo-context_"

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

args.log_dir = f"logs/{args.model_name}/{args.tok_name}/offensive_language/{args.task_name}/{prefix}{args.max_seq_len}t_{args.batch_size}b_{args.gradient_accumulation_steps}s_{args.max_epochs}e_{args.learning_rate}lr_{args.random_seed}rs_{args.label_level}"
args.ckpt_dir = os.path.join(args.log_dir, 'ckpt')
args.tb_dir = os.path.join(args.log_dir, 'tb')

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.tb_dir, exist_ok=True)

logger = get_logger(log_path=os.path.join(args.log_dir, "train_log.txt"))

logger.info("")
logger.info(f"* [sys] Current available # GPUs: {torch.cuda.device_count()}\n")
logger.info("Save the parser information")
logger.info(args.__dict__)
with open(os.path.join(args.log_dir, 'argparse.json'), 'w') as fw:
    json.dump(args.__dict__, fw, indent=2)
    fw.close()


print("")
logger.info(f"Declare the Trainer ({args.model_name}) - with ({args.tok_name}) - on ({args.task_name})...")
trainer = Trainer(args, logger)
logger.info("Succeed to prepare the trainer.\n")

no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

optimizer_grouped_parameters = [p for n, p in trainer.model.named_parameters() if not any(nd in n for nd in no_decay)]

total_trainable_params = sum(p.numel() for p in optimizer_grouped_parameters if p.requires_grad)
logger.info(f"Model parameters: {total_trainable_params // 1000000}M\n")

logger.info(f"Start the fine-tuning on {args.task_name}.\n")
trainer.fine_tuning()
