import os
from functools import partial
from typing import List

from omegaconf import DictConfig

from tts.utils.common import ras_sampling
from tts.llm.llm import Qwen2Encoder, Qwen2LM
from tts.flow.decoder import ConditionalDecoder
from tts.flow.flow_matching import CausalConditionalCFM

def initialize_tokenize():
    pass
    

def initialize_llm(llm_input_size: int,
                   llm_output_size: int,
                   speech_token_size: int = 6561,
                   length_normalized_loss: bool = True,
                   lsm_weight: float = 0,
                   pretrain_path: str = "",
                   top_p: float = 0.8,
                   top_k: int = 25,
                   win_size: int = 10,
                   tau_r: float = 0.1
                   ) -> Qwen2LM:
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"pretrain_path {pretrain_path} not found")

    llm = Qwen2Encoder(pretrain_path)

    sampling = partial(ras_sampling, top_p=top_p, top_k=top_k, win_size=win_size, tau_r=tau_r)

    lm = Qwen2LM(llm_input_size, llm_output_size, speech_token_size, llm, sampling, length_normalized_loss, lsm_weight)

    return lm


def initialize_flow_decoder_estimator(in_channels: int = 320,
                                       out_channels: int = 80,
                                       causal: bool = True,
                                       channels: List[int] = [256],
                                       dropout: float = 0.0,
                                       attention_head_dim: int = 64,
                                       n_blocks: int = 4,
                                       num_mid_blocks: int = 12,
                                       num_heads: int = 8,
                                       act_fn: str = "gelu"
                                 ) -> ConditionalDecoder:
    estimator = ConditionalDecoder(in_channels, out_channels, causal, channels, dropout,
                                   attention_head_dim, n_blocks, num_mid_blocks, num_heads, act_fn)
    return estimator


def initialize_flow_decoder(estimator: ConditionalDecoder = None,
                            in_channels: int = 240,
                            n_spks: int = 1,
                            spk_emb_dim: int = 80,
                            sigma_min: float = 1e-06,
                            solver: str = "euler", 
                            t_scheduler: str = "cosine",
                            training_cfg_rate: float = 0.2,
                            inference_cfg_rate: float = 0.7,
                            reg_loss_type: str = "l1") -> CausalConditionalCFM:
    if estimator is None:
        raise ValueError("estimator is None")
    
    cfm_params = DictConfig(
        content={
            "sigma_min": sigma_min,
            "solver": solver,
            "t_scheduler": t_scheduler,
            "training_cfg_rate": training_cfg_rate,
            "inference_cfg_rate": inference_cfg_rate,
            "reg_loss_type": reg_loss_type
        }
    )
    decoder = CausalConditionalCFM(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
    return decoder


def initialize_flow_encoder():
    pass