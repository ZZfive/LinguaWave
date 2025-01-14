# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

import random
from typing import List

import numpy as np
import torch

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


def th_accuracy(
    pad_outputs: torch.Tensor,  # 模型预测输出 (B * Lmax, D)
    pad_targets: torch.Tensor,  # 目标标签 (B, Lmax)
    ignore_label: int          # 需要忽略的标签ID（通常是填充值）
) -> torch.Tensor:            # 返回准确率（0.0-1.0）
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)  # 将pad_outputs展平为(B, Lmax, D)，然后取最后一个维度上概率最大值，最终shape为(B, Lmax)的预测标签
    mask = pad_targets != ignore_label  # 创建一个掩码，用于忽略填充的标签
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))  # 计算预测标签与目标标签匹配的数量
    denominator = torch.sum(mask)  # 计算非填充标签的数量
    return (numerator / denominator).detach()  # 计算准确率，并返回


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)  # 使用核采样策略从概率分布中采样下一个token id
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()  # 计算最近win_size个token中重复出现的次数
    if rep_num >= win_size * tau_r:  # 如果重复次数超过阈值，则进行随机采样
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


# 核采样，或top-p采样
def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    '''
    weighted_scores: 模型输出的logits分数，形状为 [vocab_size]
    top_p: 概率阈值，默认0.8，表示采样概率总和的截断阈值
    top_k: 最大保留的token数量，默认25
    '''
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)  # 对概率分布进行排序，返回排序后的概率值和对应的索引
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:  # 如果累积概率小于top_p且保留的token数量小于top_k，则将当前token的概率和索引添加到列表中
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]  # 使用multinomial进行随机采样，replacement=True表示有放回采样
    return top_ids


# 基于概率分布的随机采样，不是均匀采样
def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)  # 使用softmax后的概率分布进行随机采样，replacement=True表示有放回采样
    return top_ids


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    if fade_in_mel.device == torch.device('cpu'):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 将布尔类型的掩码转换为注意力机制中使用的偏置值；对于需要被忽略的位置（原始mask为False），将其设置为该数据类型的最小值
def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)  # 将布尔mask转换为浮点类型
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * torch.finfo(dtype).min  # 将mask取反，并乘以最小浮点值，以实现注意力掩码的效果
    return mask
