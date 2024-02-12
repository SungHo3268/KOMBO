import os
import math
import random
import functools
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


BAR_FORMAT = "{l_bar}{bar:15}{r_bar}"

def float_separator(num: int) -> str:
    num_seg = []
    while num > 1000:
        num_seg.append(num % 1000)
        num = num // 1000

    str_num = [num] + num_seg[::-1]
    temp = []
    for i, n in enumerate(str_num):
        if n == 0:
            temp.append('000')
        elif (i != 0) and (n < 100):
            temp.append('0' + str(n))
        else:
            temp.append(str(n))
    str_num = ','.join(temp)
    return str_num


def init_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)      # one gpu
    torch.cuda.manual_seed_all(seed)    # multi gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n* Set a random seed to {seed}\n")


def trim_pad(input_ids, inputs, pad_value=0, side="right"):
    compressed = torch.sum(input_ids, dim=0)       # (N, D) or (N, )

    drop_map = (compressed == pad_value).to(torch.long).to(input_ids.device)
    if torch.sum(drop_map) == 0:
        return input_ids, inputs

    if side == 'right':
        drop_start_idx = torch.argmax(drop_map * torch.arange(len(drop_map), 0, -1).to(input_ids.device))
        return input_ids[:, :drop_start_idx], inputs[:, :drop_start_idx]
    elif side == 'left':
        drop_start_idx = torch.argmax(drop_map * torch.arange(len(drop_map)).to(input_ids.device))
        return input_ids[:, drop_start_idx+1:], inputs[:, drop_start_idx+1:]
    else:
        raise NotImplementedError


def end_pad_to_divisible(inputs, divide_factor: List, pad_value=0):
    division_lcm = int(
        functools.reduce(lambda x, y: int((x * y) / math.gcd(x, y)), set(divide_factor), 1)
    )

    seq_len = inputs.shape[1]
    pad_seq_len = math.ceil(seq_len / division_lcm) * division_lcm
    if pad_seq_len == seq_len:
        return inputs
    remainder = pad_seq_len - seq_len

    if inputs.ndim == 2:
        pad_tuple = (0, remainder)
    elif inputs.ndim == 3:
        pad_tuple = (0, 0, 0, remainder)
    else:
        raise NotImplementedError

    return F.pad(inputs, pad_tuple, 'constant', pad_value)


def repeat_interleave(inputs, repeats, dim=None, batch_first=True, padding_value=0):
    """
    The 'inputs' and 'repeats' in this method must have the original form.
    It means that the shape of the 'inputs' and 'repeats' should not be reshaped.
    Then, in this function, the dim of the 'inputs' must be less than 2 and the dim of the 'repeats' should be  1.
    """

    # Calculate the shape of the output array and Determine the number of dimensions in the input
    output_shape = list(inputs.shape)
    device = inputs.device

    repeat_num = []
    if (type(repeats) == int) or (repeats.ndim == 1 and len(repeats) == 1):
        """
        In the case of the "candidates" and "block_scores" in getting latent subword representation.
        """
        # get the shape of the output
        output_shape[dim] *= repeats
        # reshape the inputs
        inputs = inputs.flatten(start_dim=0, end_dim=1)
    elif repeats.ndim == 2:
        if (inputs.ndim - repeats.ndim) == 1:  # 3dim(not flattened) - 2dim(not flattened)
            """
            In the case of the "context_inputs" of the "byte_merge" and "gru_outputs" of the "bts_merge".
            Both merged 'dim' are 1.
            """
            output_shape[dim] = int(torch.sum(repeats) / output_shape[0])
            # for padding
            repeat_num = torch.sum(repeats, dim=dim, keepdim=False)
            max_repeat_num = torch.max(repeat_num)
            # get the shape of the output
            output_shape[dim] = int(max_repeat_num)
            # reshape the inputs and repeats
            inputs = inputs.flatten(start_dim=0, end_dim=1)  # inputs.ndim = 2   (BxN, D)
            repeats = repeats.flatten()  # repeats.ndim = 1  (BxN, )
            assert inputs.shape[0] == repeats.shape[0]
        elif (inputs.ndim - repeats.ndim) == 0:  # 2dim(not flattened) - 2dim(not flattened)
            """
            In the case of the "context_input_ids" of the "byte_merge".
            """
            # for padding
            repeat_num = torch.sum(repeats, dim=dim, keepdim=False)
            max_repeat_num = torch.max(repeat_num)
            # get the shape of the output
            output_shape[dim] = int(max_repeat_num)
            # reshape the inputs and repeats
            inputs = inputs.unsqueeze(-1)
            inputs = inputs.flatten(start_dim=0, end_dim=1)  # inputs.ndim = 2   (BxN, 1)
            repeats = repeats.flatten()  # repeats.ndim = 1  (BxN, )
            assert inputs.shape[0] == repeats.shape[0]
        else:
            raise NotImplementedError
    elif repeats.ndim == 1:
        if (inputs.ndim - repeats.ndim) == 2:  # 3dim(not flattened) - 1dim(not flattened)
            output_shape[dim] = int(torch.sum(repeats))
            # reshape the inputs and repeats
            inputs = inputs.flatten(start_dim=0, end_dim=1)  # inputs.ndim = 2   (BxN, D)
            repeats = repeats.repeat(len(inputs) // len(repeats))
            assert inputs.shape[0] == repeats.shape[0]
        elif (inputs.ndim - repeats.ndim) == 1:  # 2dim(not flattened) - 1dim(not flattened)
            output_shape[dim] = int(torch.sum(repeats))
            # reshape the inputs and repeats
            inputs = inputs.unsqueeze(-1)
            inputs = inputs.flatten(start_dim=0, end_dim=1)  # inputs.ndim = 2   (BxN, D)
            repeats = repeats.repeat(len(inputs) // len(repeats))
            assert inputs.shape[0] == repeats.shape[0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # Create an array of repeated indices along the specified dim
    indices = np.repeat(np.arange(inputs.shape[0]), repeats=repeats, axis=0)
    indices = torch.LongTensor(indices).to(device)

    # take to select the elements based on the repeated indices
    output = inputs[indices]

    # Reshape the output array to match the desired shape
    try:
        # print("normal_reshape")
        output = output.reshape(output_shape)
        return output
    except RuntimeError:
        # print("padding_reshape")
        padded_output = []
        cur_idx = 0
        for repeat in repeat_num:
            repeat = int(repeat)
            padded_output.append(output[cur_idx: cur_idx + repeat])
            cur_idx += repeat
        padded_output = pad_sequence(padded_output, batch_first=batch_first, padding_value=padding_value)
        return padded_output.squeeze()
