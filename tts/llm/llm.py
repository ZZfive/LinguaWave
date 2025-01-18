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
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from tts.utils.common import IGNORE_ID
from tts.transformer.label_smoothing_loss import LabelSmoothingLoss
from tts.utils.common import th_accuracy


class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size  # 4096，speech token vocabulary size，4096也表示最终的结束token
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0  # sos对应start of sequence，eos对应end of sequence
        self.task_id = 1  # 对应cosyvoice论文图1中的turn of speech，将文本序列和音频序列区分开来
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)  # 将llm的输出映射到speech_token_size + 1维度的空间
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        '''
        sos_eos_emb: 开始/结束token的嵌入向量
        embedding: 说话人embedding
        text_token: 文本token序列
        task_id_emb: 任务id的嵌入向量
        speech_token: 语音token序列
        '''
        # 解除padding，将填充的部分，得到原始的文本token和语音token
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)  # batch_first=True表示输出中batch size为第一个维度
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        # 将开始/结束token、说话人embedding、文本token、任务id、语音token拼接在一起，得到lm_input
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)  # 计算lm_input的实际长度
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)  # 将lm_input填充到同一长度，并添加IGNORE_ID作为填充值
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)  # 音频内容文本token序列
        text_token_len = batch['text_token_len'].to(device)  # 音频内容文本token序列长度
        speech_token = batch['speech_token'].to(device)  # 音频内容语音token序列
        speech_token_len = batch['speech_token_len'].to(device)  # 音频内容语音token序列长度
        embedding = batch['embedding'].to(device)  # 音频内容说话人embedding

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))  # llm部分的主要前向计算
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)  # 计算交叉熵损失
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)  # 计算准确率
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,  # 模型输出的概率分布，[4097]
            decoded_tokens: List,  # 已经解码的token序列
            sampling: int,  # 采样策略参数
            ignore_eos: bool = True,  # 是否忽略eos
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)  # 根据采样策略从概率分布中采样下一个token id
            if (not ignore_eos) or (self.speech_token_size not in top_ids):  # 如果ignore_eos为False，或者top_ids不包含self.speech_token_size，则停止采样
                break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        if self.fp16 is True:
            embedding = embedding.half()

        device = text.device
        text = torch.concat([prompt_text, text], dim=1)  # 参考音频对应的文本prompt_text和输入目标文本text拼接
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)   # 使用text encoder进行文本特征编码，[batch_size, text_len, llm_input_size]

        # 2. encode embedding
        if embedding.shape[0] != 0:  # 此处的embedding是说话人embedding特征
            embedding = F.normalize(embedding, dim=1)  # 对说话人embedding特征进行归一化处理 [1, 192]
            embedding = self.spk_embed_affine_layer(embedding)  # 将说话人embedding特征映射到llm_input_size维度的空间 [1, 1024]
            embedding = embedding.unsqueeze(dim=1)  # 在第1维上增加一个维度，使得embedding的维度为 [batch_size, 1, llm_input_size] [1, 1, 1024]
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)  # 如果embedding为空，则初始化一个全0的embedding，维度为 [1, 0, 1024]

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)  # 获取sos和eos的token id对应的嵌入向量 [1, 1, 1024]
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)  # 获取task_id的token id对应的嵌入向量 [1, 1, 1024]
        if prompt_speech_token_len != 0:  # 如果prompt_speech_token_len不为0，则将prompt_speech_token转换为speech token embedding
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)  # 如果prompt_speech_token_len为0，则初始化一个全0的speech token embedding，维度为 [1, 0, 1024]
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)  # 将sos_eos_emb、embedding、text、task_id_emb、prompt_speech_token_emb拼接在一起，得到lm_input，维度为 [1, 1+1+text_len+speech_len, 1024]

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)  # 初始维度都是0，表示一开始缓存是空的，常见的缓存初始化方式，允许在推理过程中动态扩展缓存大小
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),  # 一个下三角矩阵，使当前位置数值只与自己和前面的数值进行注意力计算
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)  # [1, 4097]
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()  # 从forward_chunk中输出的概率分布中采样下一个token id
            if top_ids == self.speech_token_size:  # 如果采样到的token id是self.speech_token_size，则停止解码
                break
            # in stream mode, yield token one by one
            yield top_ids  # 用于流式输出
            out_tokens.append(top_ids)  # 用于下一次id的采样
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)  # 非第一次预测时，后续的token预测的输入是上一次预测的输出；因为此处top_ids起始就是一个id，直接通过索引获取对应向量更方便


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]  # 只取最后一个时间步的注意力掩码，shape从[1, text_len, text_len]变为[1, text_len]，如[1, 45, 45]->[1, 45]；Qwen2ForCausalLM内部会重新构建有效的causal mask
        outs = self.model(
            inputs_embeds=xs,  # 此处xs就是构建的lm_input，其中包含了sos_eos_emb、embedding、text tokens、task_id_emb、prompt speech tokens，不是单纯的text tokens，并且已转换为嵌入向量；将其传给inputs_embeds，在内部不会再调用Qwen2Model的embed_tokens层进行embedding操作
            attention_mask=input_masks,
            output_hidden_states=True,  # 输出所有层的隐藏状态
            return_dict=True,  # 以字典形式返回结果
            use_cache=True,  # 启用KV缓存
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]  # 提取最后一层的隐藏状态，shape如[1, 45, 896]
        new_cache = outs.past_key_values  # 更新KV缓存，记录的是上一次计算后Qwen2ForCausalLM模型每一层的K和V
        return xs, new_cache

'''
结构图：
Qwen2LM(
  (llm_embedding): Embedding(2, 896)
  (llm): Qwen2Encoder(
    (model): Qwen2ForCausalLM(
      (model): Qwen2Model(
        (embed_tokens): Embedding(151936, 896)
        (layers): ModuleList(
          (0-23): 24 x Qwen2DecoderLayer(
            (self_attn): Qwen2Attention(
              (q_proj): Linear(in_features=896, out_features=896, bias=True)
              (k_proj): Linear(in_features=896, out_features=128, bias=True)
              (v_proj): Linear(in_features=896, out_features=128, bias=True)
              (o_proj): Linear(in_features=896, out_features=896, bias=False)
              (rotary_emb): Qwen2RotaryEmbedding()
            )
            (mlp): Qwen2MLP(
              (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
              (up_proj): Linear(in_features=896, out_features=4864, bias=False)
              (down_proj): Linear(in_features=4864, out_features=896, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
            (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          )
        )
        (norm): Qwen2RMSNorm((896,), eps=1e-06)
      )
      (lm_head): Linear(in_features=896, out_features=151936, bias=False)
    )
  )
  (llm_decoder): Linear(in_features=896, out_features=6564, bias=True)
  (criterion_ce): LabelSmoothingLoss(
    (criterion): KLDivLoss()
  )
  (speech_embedding): Embedding(6564, 896)
)
'''
class Qwen2LM(torch.nn.Module):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm  # 一个上述的Qwen2Encoder对象
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            # 在前min_len个token预测时，只有当预测的top_ids不是speech_token_size才会跳出循环，如果每次采样的top_ids都是speech_token_size，可能进入死循环，故使用max_trials避免死循环出现
            # 这样设置的目的是让预测的前min_len个token不会是eos，避免出现预测的tokens数太少就结束了推理
            # 在预测的tokens数量超过max_len后，ignore_eos为False，采样时循环只会执行一次后直接跳出
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)  # 参考音频对应的文本prompt_text和输入目标文本text拼接
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)  # 用Qwen2Encoder对象中的embedding层进行文本特征，[1, text_len, llm_input_size]，如[1, 8, 896]

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)  # [1, 0, 896]；与cosyvoice中不同，cosyvoice2在speech tokens预测时不使用说话人embedding

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)  # [1, 1, 896]
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)  # [1, 1, 896]
        if prompt_speech_token_len != 0:  # 如果存在参考音频的speech tokens对齐进行编码
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)  # 如[1, 35, 896]
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)  # [1, 0, 896]
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)  # 如[1, 45, 896]；将sos_eos_emb、embedding、text、task_id_emb、prompt_speech_token_emb拼接在一起，得到lm_input

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)  # 基于参考音频文本长度和待生成文本长度计算单次推理最小预测长度
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)  # 基于参考音频文本长度和待生成文本长度计算单次推理最大预测长度

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),  # 一个下三角矩阵，使当前位置数值只与自己和前面的数值进行注意力计算
                                                      cache=cache)  # y_pred的shape未变，如[1, 45, 896]
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)  # [1, 6564]
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:  # 如果采样到的token id是self.speech_token_size，则停止采样
                break
            if top_ids > self.speech_token_size:  # 如果采样到的token id大于self.speech_token_size，则跳过不输出
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)  # 非第一次预测时，后续的token预测的输入是上一次预测的输出；因为此处top_ids起始就是一个id，直接通过索引获取对应向量更方便


class InternLM2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(pretrain_path, trust_remote_code=True)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]  # 只取最后一个时间步的注意力掩码，shape从[1, text_len, text_len]变为[1, text_len]，如[1, 45, 45]->[1, 45]；InternLM2ForCausalLM内部会重新构建有效的causal mask
        outs = self.model(
            inputs_embeds=xs,  # 此处xs就是构建的lm_input，其中包含了sos_eos_emb、embedding、text tokens、task_id_emb、prompt speech tokens，不是单纯的text tokens，并且已转换为嵌入向量；将其传给inputs_embeds，在内部不会再调用Qwen2Model的embed_tokens层进行embedding操作
            attention_mask=input_masks,
            output_hidden_states=True,  # 输出所有层的隐藏状态
            return_dict=True,  # 以字典形式返回结果
            use_cache=True,  # 启用KV缓存
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]  # 提取最后一层的隐藏状态，shape如[1, 45, hidden_size]
        new_cache = outs.past_key_values  # 更新KV缓存，记录的是上一次计算后InternLM2ForCausalLM模型每一层的K和V
        return xs, new_cache


"""
InternLM2LM(
  (llm_embedding): Embedding(2, 2048)
  (llm): InternLM2Encoder(
    (model): InternLM2ForCausalLM(
      (model): InternLM2Model(
        (tok_embeddings): Embedding(92544, 2048, padding_idx=2)
        (layers): ModuleList(
          (0-23): 24 x InternLM2DecoderLayer(
            (attention): InternLM2Attention(
              (wqkv): Linear(in_features=2048, out_features=4096, bias=False)
              (wo): Linear(in_features=2048, out_features=2048, bias=False)
              (rotary_emb): InternLM2DynamicNTKScalingRotaryEmbedding()
            )
            (feed_forward): InternLM2MLP(
              (w1): Linear(in_features=2048, out_features=8192, bias=False)
              (w3): Linear(in_features=2048, out_features=8192, bias=False)
              (w2): Linear(in_features=8192, out_features=2048, bias=False)
              (act_fn): SiLU()
            )
            (attention_norm): InternLM2RMSNorm()
            (ffn_norm): InternLM2RMSNorm()
          )
        )
        (norm): InternLM2RMSNorm()
      )
      (output): Linear(in_features=2048, out_features=92544, bias=False)
    )
  )
  (llm_decoder): Linear(in_features=2048, out_features=6564, bias=True)
  (criterion_ce): LabelSmoothingLoss(
    (criterion): KLDivLoss()
  )
  (speech_embedding): Embedding(6564, 2048)
)
"""
class InternLM2LM(torch.nn.Module):
    def __init__(
            self,
            llm_input_size: int = 2048,
            llm_output_size: int = 2048,
            speech_token_size: int = 6561,
            llm: InternLM2Encoder = None,
            sampling: Callable = None,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm  # 一个上述的InternLM2Encoder对象
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
    
    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids
    
    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)  # 参考音频对应的文本prompt_text和输入目标文本text拼接
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)  # 用Qwen2Encoder对象中的embedding层进行文本特征，[1, text_len, llm_input_size]，如[1, 8, 2048]

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)  # [1, 0, 2048]；与cosyvoice中不同，cosyvoice2在speech tokens预测时不使用说话人embedding

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)  # [1, 1, 2048]
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)  # [1, 1, 2048]
        if prompt_speech_token_len != 0:  # 如果存在参考音频的speech tokens对齐进行编码
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)  # 如[1, 35, 2048]
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)  # [1, 0, 2048]
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)  # 如[1, 45, 2048]；将sos_eos_emb、embedding、text、task_id_emb、prompt_speech_token_emb拼接在一起，得到lm_input

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)  # 基于参考音频文本长度和待生成文本长度计算单次推理最小预测长度
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)  # 基于参考音频文本长度和待生成文本长度计算单次推理最大预测长度

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),  # 一个下三角矩阵，使当前位置数值只与自己和前面的数值进行注意力计算
                                                      cache=cache)  # y_pred的shape未变，如[1, 45, 2048]
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)  # [1, 6564]
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:  # 如果采样到的token id是self.speech_token_size，则停止采样
                break
            if top_ids > self.speech_token_size:  # 如果采样到的token id大于self.speech_token_size，则跳过不输出
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)  # 非第一次预测时，后续的token预测的输入是上一次预测的输出；因为此处top_ids起始就是一个id，直接通过索引获取对应向量更方便