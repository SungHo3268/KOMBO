import os
import sys
import json
from argparse import Namespace
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append(os.getcwd())
from tokenization.srcs.tokenizers import CharTokenizer
from nlu_tasks.srcs.nlu_trainer import Trainer as NLU_Trainer
from pretraining.srcs.functions import init_random
from pretraining.utils.base_parser import ArgsBase
from pretraining.utils.logger import get_logger

plt.rc('font', family='NanumBarunGothic')
mpl.rcParams['axes.unicode_minus'] = False

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


char_tokenizer = CharTokenizer(config_path="tokenization/utils/tok_info.cfg")


def load_model(tok_type, no_context=False):
    """
    Load the model and tokenizer.
    :param tok_type: str, tokenization type. ("jamo_distinct", "jamo_distinct_no_context", "char")
    """
    parser = ArgsBase().add_nlu_task_args()
    temp_args = parser.parse_args()

    temp = vars(temp_args)

    config = json.load(open(f"nlu_tasks/nlu_configs/{temp['task_name']}/config.json"))
    for arg in config:
        temp[arg] = config[arg]

    temp["tok_type"] = tok_type
    if temp["tok_type"] == 'char':
        temp["save_dir"] = "logs/bert-base/char_ko_2k/pretraining/128t_128b_1s_5e-05lr_42rs/ckpt"
        temp["model_name"] = "bert-base"
        temp["tok_vocab_size"] = "2k"
    elif temp["tok_type"] == 'jamo_distinct':
        temp[
            "save_dir"] = "logs/kombo-base/jamo_distinct_ko_200/pretraining/span-character-mlm_jamo-trans3_gru_conv1-cjf_repeat_gru-up-res_128t_128b_1s_5e-05lr_42rs/ckpt"
        temp["model_name"] = "kombo-base"
        temp["tok_vocab_size"] = "200"
        temp["lang"] = "ko"
        temp["mlm_unit"] = "character"
        temp["jamo_fusion"] = "trans_gru_conv1"
        temp["jamo_trans_layer"] = 3
        temp["jamo_residual"] = False
        temp["cho_joong_first"] = True
        temp["upsampling"] = "repeat_gru"
        temp["upsampling_residual"] = True
        if no_context:
            temp["save_dir"] = "logs/kombo-base/jamo_distinct_ko_200/pretraining/span-character-mlm_jamo-conv1-cjf_repeat_gru-up-res_128t_128b_1s_5e-05lr_42rs/ckpt"
            temp["jamo_fusion"] = "conv1"
    else:
        raise NotImplementedError

    args = Namespace(**temp)

    init_random(seed=args.random_seed)  # initialize the random_seed to get consistent results.

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

    args.log_dir = f"nlu_tasks/logs/{args.model_name}/{args.tok_name}/tasks/{args.task_name}/{prefix}{args.max_seq_len}t_{args.batch_size}b_{args.gradient_accumulation_steps}s_{args.max_epochs}e_{args.learning_rate}lr_{args.random_seed}rs"

    args.ckpt_dir = os.path.join(args.log_dir, 'ckpt')
    args.tb_dir = os.path.join(args.log_dir, 'tb')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    logger = get_logger(log_path=os.path.join(args.log_dir, "results.txt"))

    print("")
    trainer = NLU_Trainer(args, logger)

    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]
    optimizer_grouped_parameters = [p for n, p in trainer.model.named_parameters() if
                                    not any(nd in n for nd in no_decay)]

    total_trainable_params = sum(p.numel() for p in optimizer_grouped_parameters if p.requires_grad)
    print(f"Model parameters: {total_trainable_params // 1000000}M\n")
    return trainer


def get_vectors(trainer, tgt_words, tgt_sentence, investigate_layer):
    trainer.model.eval()

    bert = trainer.model.bert
    tokenizer = trainer.tokenizer

    inputs = tokenizer.encode(tgt_sentence, return_tensors="pt", padding=True)
    inputs = inputs.to(trainer.device)

    outputs = bert(inputs, output_hidden_states=True)[-1][investigate_layer].squeeze()

    src_vectors = []
    indices = []
    for tgt in tgt_words:
        target_idx = tgt_sentence.find(tgt) + 1
        if target_idx in indices:
            target_idx = tgt_sentence[::-1].find(tgt) + 1       # '+1' for reverse
            target_idx = len(tgt_sentence) - target_idx + 1     # '+1' for [CLS]
        indices.append(target_idx)
        src_vector = outputs[target_idx]
        src_vectors.append(src_vector)

    tgt_inputs = tokenizer.encode(tgt_sentence, return_tensors="pt", padding=True)
    tgt_inputs = tgt_inputs.to(trainer.device)

    tgt_vectors = bert(tgt_inputs, output_hidden_states=True)[-1][investigate_layer].squeeze()

    return src_vectors, tgt_vectors


def plot_similarity(src_vectors, tgt_vectors, tgt_tokens, tgt_words, activation=None):
    fig = plt.figure(figsize=(len(tgt_tokens) - 3, len(tgt_words)))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(len(tgt_words), 1),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_size=0.25,
                    cbar_pad=0.1
                    )

    for i, ax in enumerate(grid):
        similarities = [float(torch.nn.functional.cosine_similarity(src_vectors[i], tgt_vector, dim=0)) for tgt_vector
                        in tgt_vectors]

        if activation == 'relu':
            similarities = [max(sim, 0) for sim in similarities]
        elif activation == 'tanh':
            similarities = [(np.tanh(sim) + 1) / 2 for sim in similarities]
        elif activation == 'sigmoid':
            similarities = [1 / (1 + np.exp(-sim)) for sim in similarities]
        else:
            pass

        # min_sim = min(similarities)
        # max_sim = max(similarities)
        # similarities = [(sim - min_sim) / (max_sim - min_sim) for sim in similarities]

        similarities = np.power(similarities, 1/1.5)

        # ax.set_axis_off()
        if i == len(tgt_words) - 1:
            ax.set_xticks(np.arange(len(tgt_tokens)))
            manipulated = tgt_tokens.copy()
            ax.set_xticklabels(manipulated, fontsize=15, fontweight='normal')
            ax.set_yticks(np.arange(1))
            ax.set_yticklabels([f"{tgt_words[i]}"], fontsize=15)
        else:
            ax.set_yticks(np.arange(1))
            ax.set_yticklabels([f"{tgt_words[i]}"], fontsize=15)

        for j in range(len(tgt_tokens)):
            sim = similarities[j]

            if sim > 0.5:
                ax.text(j, 0, round(sim, 2), ha="center", va="center", color="w", fontsize=11, fontweight='bold')
            else:
                ax.text(j, 0, round(sim, 2), ha="center", va="center", color="black", fontsize=11, fontweight='bold')
        # im = ax.imshow(np.array(similarities).reshape(1, -1), cmap='Blues', vmin=0, vmax=1, aspect='auto')
        # im = ax.imshow(np.array(similarities).reshape(1, -1), cmap='Reds', vmin=0, vmax=1, aspect='auto')
        # im = ax.imshow(np.array(similarities).reshape(1, -1), cmap='Greens', vmin=0, vmax=1, aspect='auto')
        im = ax.imshow(np.array(similarities).reshape(1, -1), cmap='Purples', vmin=0, vmax=1, aspect='auto')

    # cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

    cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    cbar.ax.set_yticklabels(['low', 'medium', 'high'], fontsize=12)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    trainer = load_model(tok_type='char', no_context=False)    # tok_type = {"jamo_distinct", "char"}

    # tgt_sentence = "물이 많아 밥이 질어져 진밥을 먹게 되었다."
    # tgt_words = ['질', '진']
    # tgt_sentence = "흐르는 개천은 강물과 만나 바다로 흘러 간다."
    # tgt_words = ['흐', '흘']
    tgt_sentence = "다른 사람을 도와 주는 것은 나를 돕는 것이다."
    tgt_words = ['도', '돕']

    tgt_tokens = ['[CLS]'] + char_tokenizer.tokenize(tgt_sentence) + ['[SEP]']
    tgt_tokens = [token.replace("▁", " ") for token in tgt_tokens]

    src_vectors, tgt_vectors = get_vectors(trainer, tgt_words, tgt_sentence, investigate_layer=0)

    plot_similarity(src_vectors, tgt_vectors, tgt_tokens, tgt_words, activation="relu")
