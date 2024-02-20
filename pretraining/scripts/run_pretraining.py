import os
import sys
import json
import torch
from copy import deepcopy
from dataclasses import dataclass
from transformers import BertForPreTraining
from transformers.data.data_collator import DataCollatorMixin
sys.path.append(os.getcwd())
from pretraining.srcs.models import CustomBertForPreTraining
from nlu_tasks.srcs.nlu_utils import get_bert_tokenizer, get_config
from pretraining.srcs.nsp_dataset import TextDatasetForNextSentencePrediction
from pretraining.srcs.functions import init_random, repeat_interleave
from pretraining.utils.base_parser import ArgsBase
from pretraining.utils.logger import get_logger
from pretraining.srcs.tokenizers import BertTokenizer
from tokenization.srcs.functions import get_tokenizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def set_parser():
    parser = ArgsBase().add_bert_pretrain_args()
    args = parser.parse_args()

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
    args.log_dir = f"logs/{args.model_name}/{args.tok_name}/pretraining/{prefix}{args.max_seq_len}t_{args.batch_size}b_{args.gradient_accumulation_steps}s_{args.learning_rate}lr_{args.random_seed}rs"

    args.ckpt_dir = os.path.join(args.log_dir, 'ckpt')
    args.tb_dir = os.path.join(args.log_dir, 'tb')
    os.makedirs(args.tb_dir, exist_ok=True)

    return args


def set_logger(args):
    logger = get_logger(log_path=os.path.join(args.log_dir, "train_log.txt"))
    logger.info("")
    logger.info("This train_log.txt inform the Running Progress of the Pre-training.\n")
    logger.info(f"Save the parser information to {args.log_dir}")
    logger.info(args.__dict__)
    with open(os.path.join(args.log_dir, 'argparse.json'), 'w') as fw:
        json.dump(args.__dict__, fw, indent=2)
        fw.close()
    return logger


def get_pretraining_model(args, tokenizer):
    config = get_config(args)
    config.vocab_size = tokenizer.vocab_size
    if ('var' in config.tok_type) or ('distinct' in config.tok_type) or ('position' in config.tok_type):
        config.update({"space_symbol_id": tokenizer.space_symbol_id,
                       "empty_jamo_id": tokenizer.empty_jamo_id,
                       })

    if hasattr(config, "embedding_type"):
        model = CustomBertForPreTraining(config=config)
    else:
        model = BertForPreTraining(config=config)
    return model, config


def set_dataset(args, logger, tokenizer):
    # Set paths for future uses
    data_dir = "datasets/pretraining"
    nsp_path = os.path.join(data_dir, (args.tok_name + '_' + str(args.max_seq_len) + 't'))
    wiki_file_path = os.path.join(data_dir, "concatenated.txt")

    # Set the dataset
    start_time = time.time()
    logger.info("Checking Dataset for NSP...")

    if args.mlm_unit in ['character', 'dynamic_character']:
        mlm_tokenizer = BertTokenizer(vocab_file=f"tokenization/resources/char_ko_2k/tok.vocab",
                                      custom_tokenizer=get_tokenizer("char", config_path="tokenization/utils/tok_info.cfg"),
                                      max_length=args.max_seq_len,
                                      )
    else:
        mlm_tokenizer = None

    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_unit=args.mlm_unit, mlm_tokenizer=mlm_tokenizer, mlm_probability=args.mlm_prob
    )
    dataset = TextDatasetForNextSentencePrediction(
        tokenizer=tokenizer,
        file_path=wiki_file_path,
        nsp_path=nsp_path,
        max_seq_len=args.max_seq_len,
        random_seed=args.random_seed,
        short_seq_probability=args.short_seq_prob,
        nsp_probability=args.nsp_prob,
        logger=logger,
        sen_a_file=args.sen_a_file,
        sen_b_file=args.sen_b_file,
        nsp_label_file=args.nsp_label_file,
    )
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    logger.info(f"Elapsed Time for NSP dataset: {elapsed_time}")
    return data_collator, dataset


def set_trainer(args, model, data_collator, dataset):
    training_args = TrainingArguments(
        seed=args.random_seed,
        data_seed=args.random_seed,
        fp16=args.fp16,

        output_dir=args.log_dir,
        overwrite_output_dir=False if args.save_dir else True,
        save_strategy='steps',
        save_steps=args.save_interval,
        save_total_limit=4,
        logging_dir=args.tb_dir,
        logging_strategy='steps',
        logging_steps=args.tb_interval,
        dataloader_num_workers=args.num_worker,

        warmup_steps=args.num_warmup_steps,
        max_steps=args.total_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type='linear',
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,

        resume_from_checkpoint=args.save_dir if args.save_dir else None
    )

    # Set the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    return trainer


from transformers.trainer import *
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

@dataclass
class CustomDataCollatorForLanguageModeling(DataCollatorMixin):
    """
        Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
        are not all of the same length.

        Args:
            tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
                The tokenizer used for encoding the data.
            mlm (`bool`, *optional*, defaults to `True`):
                Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
                with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
                tokens and the value to predict for the masked token.
            mlm_probability (`float`, *optional*, defaults to 0.15):
                The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.
            return_tensors (`str`):
                The type of Tensor to return. Allowable values are "np", "pt" and "tf".

        <Tip>

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
        [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

        </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_unit: str = "token"
    mlm_tokenizer: Optional[PreTrainedTokenizerBase] = None
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    @staticmethod
    def tf_bernoulli(shape, probability):
        import tensorflow as tf

        prob_matrix = tf.fill(shape, probability)
        return tf.cast(prob_matrix - tf.random.uniform(shape, 0, 1) >= 0, tf.bool)

    def tf_mask_tokens(
        self, inputs: Any, vocab_size, mask_token_id, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import tensorflow as tf

        mask_token_id = tf.cast(mask_token_id, inputs.dtype)

        input_shape = tf.shape(inputs)
        # 1 for a special token, 0 for a normal token in the special tokens mask
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        masked_indices = self.tf_bernoulli(input_shape, self.mlm_probability) & ~special_tokens_mask
        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        labels = tf.where(masked_indices, inputs, -100)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, 0.8) & masked_indices

        inputs = tf.where(indices_replaced, mask_token_id, inputs)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = self.tf_bernoulli(input_shape, 0.1) & masked_indices & ~indices_replaced
        random_words = tf.random.uniform(input_shape, minval=self.tokenizer._num_special_tokens, maxval=vocab_size, dtype=inputs.dtype)

        inputs = tf.where(indices_random, random_words, inputs)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        import tensorflow as tf

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="tf", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _tf_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in batch["input_ids"].numpy().tolist()
                ]
                # Cannot directly create as bool
                special_tokens_mask = tf.cast(tf.convert_to_tensor(special_tokens_mask, dtype=tf.int64), tf.bool)
            else:
                special_tokens_mask = tf.cast(special_tokens_mask, tf.bool)
            batch["input_ids"], batch["labels"] = self.tf_mask_tokens(
                tf.cast(batch["input_ids"], tf.int64),
                special_tokens_mask=special_tokens_mask,
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=len(self.tokenizer),
            )
        else:
            labels = batch["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                # Replace self.tokenizer.pad_token_id with -100
                labels = tf.where(labels == self.tokenizer.pad_token_id, -100, labels)
            else:
                labels = tf.identity(labels)  # Makes a copy, just in case
            batch["labels"] = labels
        return batch

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        if self.mlm_unit == 'token':
            """
            It is default option, which is for the original MLM task
            """
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                    labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens
        elif self.mlm_unit == 'character':
            """
            It is for the character-level span masking task
            """
            labels = inputs.clone()
            batch_size = labels.shape[0]
            probability_matrix = torch.full(size=(batch_size, (labels.shape[1]-3)//self.tokenizer.custom_tokenizer.char_len),
                                            fill_value=self.mlm_probability)
            masked_indices = torch.bernoulli(probability_matrix)
            # print(f"masked_indices.shape: \n{masked_indices.shape}")
            # print(f"masked_indices[30]: \n{masked_indices[30]}")
            repeats = torch.full(masked_indices.shape, fill_value=self.tokenizer.custom_tokenizer.char_len)
            masked_indices = repeat_interleave(masked_indices, repeats, dim=1)
            # print(f"masked_indices.shape: \n{masked_indices.shape}")
            # print(f"masked_indices[30]: \n{masked_indices[30]}")
            sep_loc_map = (inputs[:, 1:] == self.tokenizer.sep_token_id).bool()
            # print(f"sep_loc_map.shape: \n{sep_loc_map.shape}")
            org_sep_loc = torch.where(sep_loc_map == 1)
            # print(f"org_sep_loc[30].shape: \n{org_sep_loc[30].shape}")
            assert len(org_sep_loc[1]) == batch_size * 2

            sep_loc1, sep_loc2 = deepcopy(org_sep_loc)
            sep_loc2 += torch.LongTensor([-1, -2]).repeat(batch_size)
            sep_loc = (sep_loc1, sep_loc2)
            sep_repeats = torch.ones(masked_indices.shape)
            sep_repeats[sep_loc] = 2
            # print(f"sep_repeats.shape: \n{sep_repeats.shape}")
            # print(f"sep_repeats[30]: \n{sep_repeats[30]}")
            masked_indices = repeat_interleave(masked_indices, repeats=sep_repeats, dim=1)
            masked_indices[org_sep_loc] = 0
            # print(f"masked_indices.shape: \n{masked_indices.shape}")
            # print(f"masked_indices[30]: \n{masked_indices[30]}")
            masked_indices = torch.concat([torch.zeros(size=(batch_size, 1)), masked_indices], dim=1)
            masked_indices *= (inputs != self.tokenizer.pad_token_id)
            masked_indices = masked_indices.bool()
            # print(f"masked_indices.shape: \n{masked_indices.shape}")
            # print(f"masked_indices[30]: \n{masked_indices[30]}")

            labels[~masked_indices] = -100  # We only compute loss on masked tokens
            # print(f"inputs.shape: \n{inputs.shape}")
            # print(f"inputs[30]: \n{inputs[30]}")
            # print(f"labels.shape: \n{labels.shape}")
            # print(f"labels[30]: \n{labels[30]}")
        elif self.mlm_unit == 'dynamic_character':
            if random.random() < 0.5:
                """
                It is the entity-level span masking.
                """
                labels = inputs.clone()
                # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
                probability_matrix = torch.full(labels.shape, self.mlm_probability)
                if special_tokens_mask is None:
                    special_tokens_mask = [
                        self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                        labels.tolist()
                    ]
                    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
                else:
                    special_tokens_mask = special_tokens_mask.bool()

                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels[~masked_indices] = -100  # We only compute loss on masked tokens
            else:
                """
                It is the character-level span masking.
                """
                labels = inputs.clone()
                batch_size = labels.shape[0]
                probability_matrix = torch.full(
                    size=(batch_size, (labels.shape[1] - 3) // self.tokenizer.custom_tokenizer.char_len),
                    fill_value=self.mlm_probability)
                masked_indices = torch.bernoulli(probability_matrix)
                repeats = torch.full(masked_indices.shape, fill_value=self.tokenizer.custom_tokenizer.char_len)
                masked_indices = repeat_interleave(masked_indices, repeats, dim=1)
                sep_loc_map = (inputs[:, 1:] == self.tokenizer.sep_token_id).bool()
                org_sep_loc = torch.where(sep_loc_map == 1)
                assert len(org_sep_loc[1]) == batch_size * 2

                sep_loc1, sep_loc2 = deepcopy(org_sep_loc)
                sep_loc2 += torch.LongTensor([-1, -2]).repeat(batch_size)
                sep_loc = (sep_loc1, sep_loc2)
                sep_repeats = torch.ones(masked_indices.shape)
                sep_repeats[sep_loc] = 2

                masked_indices = repeat_interleave(masked_indices, repeats=sep_repeats, dim=1)
                masked_indices[org_sep_loc] = 0
                masked_indices = torch.concat([torch.zeros(size=(batch_size, 1)), masked_indices], dim=1)
                masked_indices *= (inputs != self.tokenizer.pad_token_id)
                masked_indices = masked_indices.bool()

                labels[~masked_indices] = -100  # We only compute loss on masked tokens
        else:
            raise NotImplementedError

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=self.tokenizer._num_special_tokens, high=len(self.tokenizer),
                                     size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(bool)

        probability_matrix[special_tokens_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=self.tokenizer._num_special_tokens, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# Modify the tensorboard to get seq_len of the model's output
class CustomTrainer(Trainer):
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        tr_seq_len = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._total_seq_len_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, tr_seq_len_step = self.training_step(model, inputs)
                else:
                    tr_loss_step, tr_seq_len_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    tr_seq_len += tr_seq_len_step
                else:
                    tr_loss += tr_loss_step
                    tr_seq_len += tr_seq_len_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, tr_seq_len, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                # STOP training by global_step counting
                # if self.state.global_step > 50_000:
                #     sys.exit(-111)
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, tr_seq_len, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        self._total_seq_len_scalar += tr_seq_len.item()
        train_loss = self._total_loss_scalar / self.state.global_step
        train_seq_len = self._total_seq_len_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        metrics["train_seq_len"] = train_seq_len

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, tr_seq_len, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_seq_len_scalar = self._nested_gather(tr_seq_len).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            tr_seq_len -= tr_seq_len

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["seq_len"] = round(tr_seq_len_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._total_seq_len_scalar += tr_seq_len_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            seq_len = outputs["hidden_states"][-2].shape[1]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            seq_len = seq_len / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach(), seq_len

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, output_hidden_states=True)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def main(args, config, logger, trainer):
    # Set GPU and get device info
    logger.info(f"* [sys] Current available # GPUs: {torch.cuda.device_count()}\n")
    device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')  # CUDA_VISIBLE_DEVICES must be stated at the command prompt
    torch.cuda.set_device(device)  # change allocation of current GPU

    logger.info("\n")
    logger.info("========== Pre-training ==========")
    logger.info(f"model                 : {args.model_name}")
    logger.info(f"tokenizer             : {args.tok_name}")
    logger.info(f"vocab size            : {config.vocab_size}")
    if 'kombo' in args.model_name:
        if config.jamo_fusion:
            logger.info(f"jamo_fusion           : {config.jamo_fusion}")
            logger.info(f"jamo_residual         : {bool(config.jamo_residual)}")
            logger.info(f"cho_joong_first       : {bool(config.cho_joong_first)}")
    logger.info(f"device                : {device}")
    logger.info(f"random seed           : {args.random_seed}")
    logger.info(f"total steps           : {args.total_steps}")
    logger.info(f"warmup steps          : {args.num_warmup_steps}")
    logger.info(f"batch size            : {args.batch_size}")
    logger.info(f"accumulation steps    : {args.gradient_accumulation_steps}")
    logger.info(f"learning rate         : {args.learning_rate}")
    logger.info(f"max seq len           : {args.max_seq_len}\n")

    if args.save_dir:
        logger.info(f"save dir              : {args.save_dir}")
    logger.info(f"* log dir             : {args.log_dir}")
    logger.info(f"* ckpt dir            : {args.ckpt_dir}")
    logger.info(f"  save_interval       : {args.save_interval} steps\n")
    logger.info(f"* tb dir              : {args.tb_dir}")
    logger.info(f"  tb interval         : {args.tb_interval} steps\n")

    # Run training
    logger.info("Start the Training !")
    train_result = trainer.train(resume_from_checkpoint=args.save_dir if args.save_dir else None)

    # Save model and metrics
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logger.info("Save the trained model and metrics !")
    trainer.save_model(args.ckpt_dir)
    trainer.save_metrics(split="train", metrics=train_result.metrics, combined=False)
    logger.info("Complete the Training !")


if __name__ == "__main__":
    # Set Argument Parser and Logger
    args = set_parser()
    logger = set_logger(args)

    # Initialize the random seed
    init_random(args.random_seed)  # initialize the random_seed to get consistent results.

    # Set tokenizer and model
    tokenizer = get_bert_tokenizer(args)
    model, config = get_pretraining_model(args, tokenizer)

    # Load or create the dataset
    data_collator, dataset = set_dataset(args, logger, tokenizer)

    # Set the training details
    trainer = set_trainer(args, model, data_collator, dataset)

    main(args, config, logger, trainer)
