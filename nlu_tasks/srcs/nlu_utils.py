import os
import sys
import math
from typing import Iterable, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import BertConfig
sys.path.append(os.getcwd())

from nlu_tasks.srcs.models import KorNLIModel, KorSTSModel, NSMCModel, PAWS_XModel, KorQuADModel
from pretraining.srcs.models import CustomBertForPreTraining
from pretraining.srcs.tokenizers import BertTokenizer
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
    prefix = args.model_name.split("-")[0].lower()

    bert_config_path = f"pretraining/utils/{prefix}_config.json"

    config = BertConfig.from_json_file(bert_config_path)
    temp_args = vars(args)
    for key in temp_args:
        if temp_args[key] is not None:
            config.update({key: temp_args[key]})
    return config


def get_task_model(args, tokenizer):
    config = get_config(args)
    config.vocab_size = tokenizer.vocab_size
    if ('var' in config.tok_type) or ('distinct' in config.tok_type):
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

class AdamWScale(Optimizer):
    """
    This AdamW implementation is copied from Huggingface.
    We modified it with Adagrad scaling by rms of a weight tensor

    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # /Adapt Step from Adagrad
                step_size = step_size * max(1e-3, self._rms(p.data))
                # /Adapt Step from Adagrad

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
