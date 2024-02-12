import argparse
from distutils.util import strtobool as _bool


class ArgsBase:
    @staticmethod
    def add_base_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # system setting
        parser.add_argument('--mode', type=str, default='dummy')
        parser.add_argument('--host', type=str, default='dummy')
        parser.add_argument('--port', type=int, default=56789)
        parser.add_argument('--random_seed', type=int, default=42)
        parser.add_argument('--multi_gpu', type=_bool, default=False)
        parser.add_argument('--log_dir', type=str, help="logs (ckpt/ tb) directory path")
        parser.add_argument('--ckpt_dir', type=str, help="ckpt directory path (fine-tuned)")
        parser.add_argument('--tb_dir', type=str, help="tensorboard directory path")
        parser.add_argument('--save_dir', type=str, default="",
                            help="save directory path (pre-trained). "
                                 "Use this parameter when you want to reload or load the checkpoint of the model.")
        # tokenizer
        parser.add_argument('--tok_type', type=str, default='jamo_distinct', help="jamo  |  char  |  morpheme  |  subword  |  morphemeSubword  |  word  |  stroke  | cji  |  bts")
        parser.add_argument('--tok_vocab_size', type=str, default='200', help="200  |  2k  |  4k  |  8k  |  16k  |  32k  |  64k")
        parser.add_argument('--lang', type=str, default='ko', help="ko  |  en")
        parser.add_argument('--bpe_corpus', type=str, default='wiki', help="wiki  |  aihub")
        parser.add_argument('--tok_name', type=str, help="It will be calculated in each script files.")

        # model
        parser.add_argument('--model_name', type=str, default='tfm-bert-base',
                            help="bert-base  |  tfm-bert-base")
        parser.add_argument('--m_type', type=str, default='nano', help="hf  |  nano")   # ignore it.
        parser.add_argument('--max_seq_len', type=int, default=128)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        parser.add_argument('--max_grad_norm', type=int, default=1.)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--tb_interval', type=int, default=500, help="500")
        parser.add_argument('--num_hidden_layers', type=int)

        # TokenFusingModule
        parser.add_argument('--jamo_fusion', type=str)
        parser.add_argument('--jamo_residual', type=_bool)
        parser.add_argument('--cho_joong_first', type=_bool)
        parser.add_argument('--jamo_trans_layer', type=int)
        # For Funnel-Transformer and Hourglass-Transformer
        parser.add_argument('--ignore_structure', type=_bool, default=False)
        # For Jamo unit in TokenFusingModule
        parser.add_argument('--only_contextualization', type=_bool, default=False)

        parser.add_argument('--upsampling', type=str)
        parser.add_argument('--upsampling_residual', type=_bool)

        # Masking Strategy
        parser.add_argument('--mlm_unit', type=str, default="token", help="'character' or 'morpheme")
        return parser

    def add_bert_pretrain_args(self):
        parent_parser = argparse.ArgumentParser(description="Pre-training")
        parser = self.add_base_args(parent_parser)
        # preprocess
        parser.add_argument('--mlm_prob', type=float, default=0.15)
        parser.add_argument('--short_seq_prob', type=float, default=0.1)
        parser.add_argument('--nsp_prob', type=float, default=0.5)
        # train
        parser.add_argument('--cuda', type=int, default=0)
        parser.add_argument('--fp16', type=_bool, default=True)
        parser.add_argument('--num_worker', type=int, default=1)
        parser.add_argument('--total_steps', type=int, default=1_000_000, help="1_000_000")
        parser.add_argument('--num_warmup_steps', type=int, default=10_000, help="10_000")
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--adam_epsilon', type=float, default=1e-06)
        parser.add_argument('--save_interval', type=int, default=50_000, help="50_000")
        parser.add_argument('--sen_a_file', type=str, default="sentence_as.txt")
        parser.add_argument('--sen_b_file', type=str, default="sentence_bs.txt")
        parser.add_argument('--nsp_label_file', type=str, default="nsp_labels.txt")
        return parser

    def add_nlu_task_args(self):
        parent_parser = argparse.ArgumentParser(description="Fine_tuning NLU Tasks")
        parser = self.add_base_args(parent_parser)

        parser.add_argument('--task_name', type=str, default='KorSTS', help='KorNLI  |  KorQuAD  |  KorSTS  |  NSMC  |  PAWS_X')
        parser.add_argument('--max_epochs', type=int, default=2)
        parser.add_argument('--optimizer', type=str, default="adamw", help='adamw  ||  adamwscale  ||  adafactor')
        parser.add_argument('--lr_scheduler', type=str, default='linear', help='cosine  ||  legacy  ||  constant  ||  linear')
        parser.add_argument('--warmup_ratio', type=float, default=0.1)
        parser.add_argument('--total_steps', type=int, default=0)
        parser.add_argument('--num_warmup_steps', type=int, default=0)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--adam_epsilon', type=float, default=1e-06)
        parser.add_argument('--final_cosine', type=float, default=1e-05)

        # for data preprocessing (KorNLI, KorSTS, NSMC, PAWS_X)
        parser.add_argument('--remain_lang', type=str, default="ko_en_punc")
        parser.add_argument('--do_hangeulize', type=_bool, default=True)
        parser.add_argument('--data_remove', type=_bool, default=True)
        # for KorQuAD
        parser.add_argument('--max_query_len', type=int, default=64)
        parser.add_argument('--max_answer_len', type=int, default=30)
        return parser

    def add_typo_task_args(self):
        parent_parser = self.add_nlu_task_args()
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, description="Fine_tuning Typo Tasks")

        parser.add_argument('--typo_rates', type=str, default="0.0_0.05_0.10_0.15_0.20_0.25_0.30_0.35_0.40", required=True)
        parser.add_argument('--pretrained_seed', type=str, default="2739_7848_7295", required=True)
        return parser

    def add_toxic_task_args(self):
        parent_parser = self.add_nlu_task_args()
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, description="Fine_tuning Toxic Tasks")
        # for KOLD
        parser.add_argument('--label_level', type=str, default="A", required=True, help="A  |  B  |  C")
        parser.add_argument('--split_ratio', type=str, default="0.8_0.1_0.1", help="train_dev_test")
        return parser
