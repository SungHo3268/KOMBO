import os
import sys
import json
import logging
import torch
from argparse import Namespace
sys.path.append(os.getcwd())
from pretraining.srcs.functions import init_random
from pretraining.utils.base_parser import ArgsBase
from pretraining.utils.logger import get_logger
import nlu_tasks.srcs.korquad_trainer as korquad_trainer
from nlu_tasks.srcs.nlu_trainer import Trainer as NLU_Trainer


parser = ArgsBase().add_nlu_task_args()
temp_args = parser.parse_args()
temp = vars(temp_args)
config = json.load(open(f"nlu_tasks/data_configs/{temp['task_name']}/config.json"))
for arg in config:
    temp[arg] = config[arg]


if temp['tok_type'] in ['jamo', 'stroke', 'cji', 'bts']:
    temp['max_seq_len'] = 256
    temp['learning_rate'] = 5e-05
    if temp['task_name'] == 'KorQuAD':
        temp['max_seq_len'] = 512
        temp['doc_stride'] = 128
        temp['max_query_len'] = 64
        temp['max_answer_len'] = 30
elif temp['tok_type'] in ['jamo_distinct']:
    temp['max_seq_len'] = 512
    temp['learning_rate'] = 5e-05
    if temp['task_name'] == 'KorQuAD':
        temp['max_seq_len'] = 2048
        temp['doc_stride'] = 512
        temp['max_query_len'] = 256
        temp['max_answer_len'] = 128
        temp['batch_size'] = 4
        temp['gradient_accumulation_steps'] = 4
elif temp['tok_type'] in ['stroke_var', 'cji_var', 'bts_var']:
    temp['max_seq_len'] = 1024
    temp['learning_rate'] = 5e-05
    if temp['task_name'] == 'KorQuAD':
        temp['max_seq_len'] = 2048
        temp['doc_stride'] = 512
        temp['max_query_len'] = 512
        temp['max_answer_len'] = 256
        temp['batch_size'] = 4
        temp['gradient_accumulation_steps'] = 4
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

args.log_dir = f"logs/{args.model_name}/{args.tok_name}/nlu_tasks/{args.task_name}/{prefix}{args.max_seq_len}t_{args.batch_size}b_{args.gradient_accumulation_steps}s_{args.max_epochs}e_{args.learning_rate}lr_{args.random_seed}rs"

args.ckpt_dir = os.path.join(args.log_dir, 'ckpt')
args.tb_dir = os.path.join(args.log_dir, 'tb')

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.tb_dir, exist_ok=True)

if args.task_name == 'KorQuAD':
    logger = logging.getLogger(__name__)
    if args.doc_stride >= (args.max_seq_len - args.max_query_len):
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)
else:
    logger = get_logger(log_path=os.path.join(args.log_dir, "train_log.txt"))

logger.info("")
logger.info(f"* [sys] Current available # GPUs: {torch.cuda.device_count()}\n")
logger.info("Save the parser information")
logger.info(args.__dict__)
with open(os.path.join(args.log_dir, 'argparse.json'), 'w') as fw:
    json.dump(args.__dict__, fw, indent=2)
    fw.close()


if args.task_name in ['KorNLI', 'KorSTS', 'NSMC', 'PAWS_X']:
    print("")
    logger.info(f"Declare the Trainer ({args.model_name}) - with ({args.tok_name}) - on ({args.task_name})...")
    trainer = NLU_Trainer(args, logger)
    logger.info("Succeed to prepare the trainer.\n")

    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [p for n, p in trainer.model.named_parameters() if not any(nd in n for nd in no_decay)]

    total_trainable_params = sum(p.numel() for p in optimizer_grouped_parameters if p.requires_grad)
    logger.info(f"Model parameters: {total_trainable_params // 1000000}M\n")

    logger.info(f"Start the fine-tuning on {args.task_name}.\n")
    trainer.fine_tuning()

elif args.task_name == 'KorQuAD':
    korquad_trainer.fine_tuning(args, logger)
