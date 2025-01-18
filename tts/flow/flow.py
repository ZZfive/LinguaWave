# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from tts.utils.mask import make_pad_mask


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)  # 输入的speech tokens序列，shape: [1, token_len]
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)  # 输入的mel谱图特征，shape: [1, mel_len, 80]
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)  # 输入的说话人embedding特征，shape: [1, 192]

        # xvec projection
        embedding = F.normalize(embedding, dim=1)  # 对输入的说话人embedding特征进行归一化
        embedding = self.spk_embed_affine_layer(embedding)  # 将归一化后的说话人embedding特征映射到80维

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)  # 创建一个mask，用于对输入的speech tokens序列进行填充
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  # 将输入的speech tokens序列进行embedding，并乘以mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)  # [1, token_len, 512]
        h = self.encoder_proj(h)  # [1, token_len, 80]
        h, h_lengths = self.length_regulator(h, feat_len)  # 将输入的mel谱图特征进行插值，调整mel谱图序列长度

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:  # 50%的概率不进行条件注入
                continue
            index = random.randint(0, int(0.3 * j))  # 随机从mel的前30%位置中选一个索引index
            # 将原始mel谱图（feat）从开始到这个截断点的部分复制到条件张量（conds）中，创建一个"部分已知"的条件，模型需要基于这个部分信息来生成剩余的部分；这种渐进式生成策略，模型学习如何基于部分已知的mel谱图来预测和生成后续内容
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)  # 将输入的mel谱图特征进行插值，调整mel谱图序列长度与h一致
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,  # 预测出的speech tokens序列，shape: [1, token_len]
                  token_len,
                  prompt_token,  # 参考音频中提取的speech tokens序列，即使同一个参考音频提取的长度也不固定，shape: [1, prompt_token_len, 512]
                  prompt_token_len,
                  prompt_feat,  # 参考音频的mel谱图特征，shape: [1, mel_len1, 80]
                  prompt_feat_len,  
                  embedding,  # 说话人embedding特征，shape: [1, 192]
                  flow_cache):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)  # [1, 192]
        embedding = self.spk_embed_affine_layer(embedding)  # [1, 80]

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  # [1, token_len, 512]

        # text encode
        h, h_lengths = self.encoder(token, token_len)  # [1, token_len, 512]
        h = self.encoder_proj(h)  # [1, token_len, 80]
        # 没有使用时长预测器，而是直接基于预测的token 序列长度计算mel谱图长度，然后再使用下面的length_regulator进行插值，调整mel谱图序列长度
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        # h[:, :token_len1]是参考音频提取的speech tokens的编码结果，h[:, token_len1:]是预测的speech tokens编码结果，mel_len1, mel_len2分别对应前两者的mel谱图长度；输出[1, mel_len1 + mel_len2, 80]
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)  # [1, mel_len1 + mel_len2, 80]
        conds[:, :mel_len1] = prompt_feat  # 将参考音频的mel谱图特征复制到conds中
        conds = conds.transpose(1, 2)  # [1, 80, mel_len1 + mel_len2]

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)  # [1, mel_len1 + mel_len2]
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),  # [1, 80, mel_len1 + mel_len2]
            mask=mask.unsqueeze(1),  # [1, 1, mel_len1 + mel_len2]
            spks=embedding,  # [1, 80]
            cond=conds,  # [1, 80, mel_len1 + mel_len2]
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]  # [1, 80, mel_len2]；后半段是预测出的speech tokens对应的mel谱图特征
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)  # 对输入的说话人embedding特征进行归一化，[1, 192]
        embedding = self.spk_embed_affine_layer(embedding)  # 将归一化后的说话人embedding特征映射到80维，[1, 80]

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len  # 将参考音频speech tokens和预测的speech tokens拼接，[1, 55]
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)  # [1, 55, 1]
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  # [1, 55, 512]

        # text encode
        h, h_lengths = self.encoder(token, token_len)  # 如[1, 110, 512]；对speech tokens进行编码，中间长度维度会翻倍，因为此处encodr是UpsamplerConformerEncoder
        if finalize is False:  # finalize用于表示流式推理时的最后一次解码，非最后一次时finalize都为False，会截取pre_lookahead_len个tokens对应的向量场
            h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]  # pre_lookahead_len为3，token_mel_ratio为2，故最后6个token对应的向量特征被丢弃
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)  # 如[1, 110, 80]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)  # 如[1, 110, 80]
        conds[:, :mel_len1] = prompt_feat  # 将参考音频的mel谱图特征复制到conds中
        conds = conds.transpose(1, 2)  # 如[1, 80, 110]

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)  # 如[1, 110]
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),  # 如[1, 80, 110]
            mask=mask.unsqueeze(1),  # 如[1, 1, 110]
            spks=embedding,  # 如[1, 80]
            cond=conds,  # 如[1, 80, 110]
            n_timesteps=10  # 采样步数
        )  # 如[1, 80, 110]
        feat = feat[:, :, mel_len1:]  # 如[1, 80, 40]
        assert feat.shape[2] == mel_len2
        return feat.float(), None