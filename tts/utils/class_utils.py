# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
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
import torch

from tts.transformer.activation import Swish
from tts.transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)
from tts.transformer.embedding import (PositionalEncoding,
                                             RelPositionalEncoding,
                                             WhisperPositionalEncoding,
                                             LearnablePositionalEncoding,
                                             NoPositionalEncoding)
from tts.transformer.attention import (MultiHeadedAttention,
                                             RelPositionMultiHeadedAttention)
from tts.transformer.embedding import EspnetRelPositionalEncoding
from tts.transformer.subsampling import LegacyLinearNoSubsampling
from tts.llm.llm import TransformerLM, Qwen2LM, InternLM2LM
from tts.flow.flow import MaskedDiffWithXvec, CausalMaskedDiffWithXvec
from tts.hifigan.generator import HiFTGenerator
from tts.cli.model import CosyVoiceModel, CosyVoice2Model, CosyVoice2ModelOtherLLM


COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}


def get_model_type(configs):
    # NOTE CosyVoice2Model inherits CosyVoiceModel
    if isinstance(configs['llm'], TransformerLM) and isinstance(configs['flow'], MaskedDiffWithXvec) and isinstance(configs['hift'], HiFTGenerator):
        return CosyVoiceModel
    if isinstance(configs['llm'], Qwen2LM) and isinstance(configs['flow'], CausalMaskedDiffWithXvec) and isinstance(configs['hift'], HiFTGenerator):
        return CosyVoice2Model
    if isinstance(configs['llm'], InternLM2LM) and isinstance(configs['flow'], CausalMaskedDiffWithXvec) and isinstance(configs['hift'], HiFTGenerator):
        return CosyVoice2ModelOtherLLM
    raise TypeError('No valid model type found!')
