# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

import math
from tokenize import Special
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from whisper.tokenizer import LANGUAGES as WhiserLanguages

WHISPER_LANGS = tuple(WhiserLanguages.keys())
IGNORE_ID = -1


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def add_blank(ys_pad: torch.Tensor, blank: int,
              ignore_id: int) -> torch.Tensor:
    """ Prepad blank for transducer predictor

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        blank (int): index of <blank>

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> blank = 0
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,   4,   5],
                [ 4,  5,  6,  -1,  -1],
                [ 7,  8,  9,  -1,  -1]], dtype=torch.int32)
        >>> ys_in = add_blank(ys_pad, 0, -1)
        >>> ys_in
        tensor([[0,  1,  2,  3,  4,  5],
                [0,  4,  5,  6,  0,  0],
                [0,  7,  8,  9,  0,  0]])
    """
    bs = ys_pad.size(0)
    _blank = torch.tensor([blank],
                          dtype=torch.long,
                          requires_grad=False,
                          device=ys_pad.device)
    _blank = _blank.repeat(bs).unsqueeze(1)  # [bs,1]
    out = torch.cat([_blank, ys_pad], dim=1)  # [bs, Lmax+1]
    return torch.where(out == ignore_id, blank, out)


def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def add_whisper_tokens(special_tokens, ys_pad: torch.Tensor, ignore_id: int,
                       task: str, no_timestamp: bool, language: str,
                       use_prev: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add whisper-style tokens.

    ([PREV] -> [previous text tokens or hotwords]).optional --
      ┌------------------------------------------------------↲
      ↓
    [sot] -> [language id] -> [transcribe] -> [begin time] -> [text tokens] -> [end time] -> ... -> [eot]    # noqa
        |          |                |-------> [no timestamps] -> [text tokens] ----------------------↑       # noqa
        |          |                                                                                 |       # noqa
        |          |--------> [translate]  -> [begin time] -> [text tokens] -> [end time] -> ... --->|       # noqa
        |                           |-------> [no timestamps] -> [text tokens] --------------------->|       # noqa
        |                                                                                            |       # noqa
        |--> [no speech(VAD)] ---------------------------------------------------------------------->|       # noqa

    Args:
        special_tokens: get IDs of special tokens
        ignore_id (int): index of padding
        no_timestamp (bool): whether to add timestamps tokens
        language (str): language tag

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + ?)
        ys_out (torch.Tensor) : (B, Lmax + ?)

    """
    if use_prev:
        # i.e., hotword list
        _prev = [special_tokens["sot_prev"]]
        # append hotword list to _prev
        # ...
        raise NotImplementedError
    else:
        _prev = []

    language_id = special_tokens["sot"] + 1 + WHISPER_LANGS.index(language)
    if task == "transcribe":
        task_id = special_tokens["transcribe"]
    elif task == "translate":
        task_id = special_tokens["translate"]
    elif task == "vad":
        task_id = special_tokens["no_speech"]
    else:
        raise NotImplementedError("unsupported task {}".format(task))
    _sot = _prev + [special_tokens["sot"], language_id, task_id]
    _eot = torch.tensor([special_tokens["eot"]],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys

    if task == "transcribe" or task == "translate":
        if no_timestamp:
            _sot.append(special_tokens["no_timestamps"])
        else:
            _sot.append(special_tokens["timestamp_begin"])
            # add subsequent tokens
            # ...
            raise NotImplementedError
    elif task == "vad":
        _sot.append(special_tokens["no_speech"])
    else:
        raise NotImplementedError

    _sot = torch.tensor(_sot,
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys_in = [torch.cat([_sot, y], dim=0) for y in ys]
    ys_out = [torch.cat([_sot[1:], y, _eot], dim=0) for y in ys]
    return pad_list(ys_in, special_tokens["eot"]), pad_list(ys_out, ignore_id)


def add_whisper_tokens_multi_language(
        special_tokens, ys_pad: torch.Tensor, ignore_id: int,
        task: str, no_timestamp: bool, language: list,
        use_prev: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add whisper-style tokens.

    ([PREV] -> [previous text tokens or hotwords]).optional --
      ┌------------------------------------------------------↲
      ↓
    [sot] -> [language id] -> [transcribe] -> [begin time] -> [text tokens] -> [end time] -> ... -> [eot]    # noqa
        |          |                |-------> [no timestamps] -> [text tokens] ----------------------↑       # noqa
        |          |                                                                                 |       # noqa
        |          |--------> [translate]  -> [begin time] -> [text tokens] -> [end time] -> ... --->|       # noqa
        |                           |-------> [no timestamps] -> [text tokens] --------------------->|       # noqa
        |                                                                                            |       # noqa
        |--> [no speech(VAD)] ---------------------------------------------------------------------->|       # noqa

    Args:
        special_tokens: get IDs of special tokens
        ignore_id (int): index of padding
        no_timestamp (bool): whether to add timestamps tokens
        language (str): language token id 

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + ?)
        ys_out (torch.Tensor) : (B, Lmax + ?)

    """
    if use_prev:
        # i.e., hotword list
        _prev = [special_tokens["sot_prev"]]
        # append hotword list to _prev
        # ...
        raise NotImplementedError
    else:
        _prev = []
    
    # default language set as 'zh'
    if language is None:
        language_id = torch.full((ys_in.shape[0], 1), fill_value=50260)
    else:
        language_id = torch.tensor(language).unsqueeze(-1).to(ys_pad.device)  # (B, 1)

    if task == "transcribe":
        task_id = special_tokens["transcribe"]
    elif task == "translate":
        task_id = special_tokens["translate"]
    elif task == "vad":
        task_id = special_tokens["no_speech"]
    else:
        raise NotImplementedError("unsupported task {}".format(task))
    
    _sot = _prev + [special_tokens["sot"], task_id]
    _eot = torch.tensor([special_tokens["eot"]],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys

    if task == "transcribe" or task == "translate":
        if no_timestamp:
            _sot.append(special_tokens["no_timestamps"])
        else:
            _sot.append(special_tokens["timestamp_begin"])
            # add subsequent tokens
            # ...
            raise NotImplementedError
    elif task == "vad":
        _sot.append(special_tokens["no_speech"])
    else:
        raise NotImplementedError

    _sot = torch.tensor(_sot,
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys_in = [torch.cat([_sot, y], dim=0) for y in ys]
    ys_out = [torch.cat([_sot, y, _eot], dim=0) for y in ys]

    for i in range(len(ys_in)):
        ys_in[i] = torch.cat([ys_in[i][0:1], language_id[i], ys_in[i][1:]])
        ys_out[i] = torch.cat([language_id[i], ys_out[i][1:]])

    return pad_list(ys_in, special_tokens["eot"]), pad_list(ys_out, ignore_id)


def reverse_pad_list(ys_pad: torch.Tensor,
                     ys_lens: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
    """Reverse padding for the list of tensors.

    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tokenmax).

    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])

    """
    r_ys_pad = pad_sequence([(torch.flip(y.int()[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True,
                            pad_value)
    return r_ys_pad


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def get_subsample(config):
    input_layer = config["encoder_conf"]["input_layer"]
    assert input_layer in ["conv2d", "conv2d6", "conv2d8"]
    if input_layer == "conv2d":
        return 4
    elif input_layer == "conv2d6":
        return 6
    elif input_layer == "conv2d8":
        return 8


def log_add(*args) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


if __name__ == "__main__":
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained('/data1/yumingdong/model/finetuned/whisper-large-v3-lora700+700-130000/')
    special_tokens = {'eot': 50258,
                      'no_speech': 50363,
                      'no_timestamps': 50364,
                      'sot': 50258,
                      'sot_prev': 50362,
                      'timestamp_begin': 50365,
                      'transcribe': 50360,
                      'translate': 50359
                      }
    ignore_id = -1
    ys_no_language = torch.tensor([
        [91,   2415, 237, 12136, 237, 12136],
        [91,   9455, 2131,  -1,    -1, -1],
        [1654, 2131, 2131, -1,    -1,  -1],
        [6404, 2131, -1,    -1,    -1, -1],
        [2131, -1,   -1,    -1,    -1, -1]
        ])
    

    ys_language = torch.tensor([
        [50352,   91,   2415, 237, 12136, 103, 103],
        [50353,  91,   9455, 2131,  -1,    -1, -1],
        [50354, 1654, 2131, 2131, -1,    -1,  -1],
        [50355, 6404, 2131, -1,    -1,    -1, -1],
        [50358, 2131, -1,   -1,    -1,    -1, -1]
        ])
    keys = [
        '50260|dadikefu00018696-0005730-0007210-S', 
        '50358|haoweilai00000786-0306986-0308474-C', 
        '50260|jingdongdigit00019295-0000000-0001480-S', 
        '50355|beijingranqi00120127-0131940-0133390-C', 
        '50358|dadikefu00026990-0052365-0053822-O',
    ]
    ys_pad = ys_no_language
    prev_len = ys_pad.size(1)
    ys_pad_lens = 7
    # ys_in_pad, ys_out_pad = add_whisper_tokens(special_tokens,
    #                                         ys_pad,
    #                                         ignore_id,
    #                                         task="transcribe",
    #                                         no_timestamp=True,
    #                                         language="zh",
    #                                         use_prev=False)
    ys_in_pad, ys_out_pad = add_whisper_tokens_multi_language(special_tokens,
                                            ys_pad,
                                            ignore_id,
                                            task="transcribe",
                                            no_timestamp=True,
                                            language="yue",
                                            use_prev=False,
                                            keys=keys)
    
    cur_len = ys_in_pad.size(1)
    ys_in_lens = ys_pad_lens + cur_len - prev_len
    print(f'prev_len: {ys_pad.size(1)}')
    print(f'ys_in_lens: {ys_in_lens}')
    print(f'ys_in_pad: {ys_in_pad.shape}')
    print(f'ys_out_pad: {ys_out_pad.shape}')
    print(ys_in_pad)

    ys_in = processor.batch_decode(ys_in_pad)
    ys_out = processor.batch_decode(ys_out_pad)
    # print(processor.batch_decode([[15, 4762, 12249, 91]]))
    # print(WHISPER_LANGS)
    for i in range(len(ys_in)):
        print([ys_in[i]])
        print(ys_out[i])
        
        
'''
['<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>|宏洏�']
<|zh|><|transcribe|><|notimestamps|>|宏洏�<|startoftranscript|>
['<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>|太好<|startoftranscript|><|startoftranscript|><|startoftranscript|>']
<|zh|><|transcribe|><|notimestamps|>|太好<|startoftranscript|>
['<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>我好好<|startoftranscript|><|startoftranscript|><|startoftranscript|>']
<|zh|><|transcribe|><|notimestamps|>我好好<|startoftranscript|>
['<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>也好<|startoftranscript|><|startoftranscript|><|startoftranscript|><|startoftranscript|>']
<|zh|><|transcribe|><|notimestamps|>也好<|startoftranscript|>
['<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>好<|startoftranscript|><|startoftranscript|><|startoftranscript|><|startoftranscript|><|startoftranscript|>']
<|zh|><|transcribe|><|notimestamps|>好<|startoftranscript|>
'''