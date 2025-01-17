import os
from functools import partial
from typing import List

from omegaconf import DictConfig

from tts.utils.common import ras_sampling
from tts.llm.llm import Qwen2Encoder, Qwen2LM
from tts.flow.decoder import ConditionalDecoder
from tts.flow.flow_matching import CausalConditionalCFM
from tts.transformer.upsample_encoder import UpsampleConformerEncoder
from tts.flow.flow import CausalMaskedDiffWithXvec


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


def initialize_causal_flow_decoder(estimator: ConditionalDecoder = None,
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
    flow_decoder = CausalConditionalCFM(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
    return flow_decoder


def initialize_causal_flow_encoder(output_size: int = 512,
                                   attention_heads: int = 8,
                                   linear_units: int = 2048,
                                   num_blocks: int = 6,
                                   dropout_rate: float = 0.1,
                                   positional_dropout_rate: float = 0.1,
                                   attention_dropout_rate: float = 0.1,
                                   normalize_before: bool = True,
                                   input_layer: str = 'linear',
                                   pos_enc_layer_type: str = 'rel_pos_espnet',
                                   selfattention_layer_type: str = 'rel_selfattn',
                                   input_size: int = 512,
                                   use_cnn_module: bool = False,
                                   macaron_style: bool = False
                                   ) -> UpsampleConformerEncoder:
    flow_encoder = UpsampleConformerEncoder(input_size=input_size,
                                            output_size=output_size,
                                            attention_heads=attention_heads,
                                            linear_units=linear_units,
                                            num_blocks=num_blocks,
                                            dropout_rate=dropout_rate,
                                            positional_dropout_rate=positional_dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            input_layer=input_layer,
                                            pos_enc_layer_type=pos_enc_layer_type,
                                            normalize_before=normalize_before,
                                            macaron_style=macaron_style,
                                            selfattention_layer_type=selfattention_layer_type,
                                            use_cnn_module=use_cnn_module)
    return flow_encoder


def initialize_causal_flow(encoder: UpsampleConformerEncoder = None,
                           decoder: CausalConditionalCFM = None,
                           input_size: int = 512,
                           output_size: int = 80,
                           spk_embed_dim: int = 192,
                           output_type: str = 'mel',
                           vocab_size: int = 6561,
                           input_frame_rate: int = 25,
                           only_mask_loss: bool = True,
                           token_mel_ratio: int = 2,
                           pre_lookahead_len: int = 3
                           ) -> CausalMaskedDiffWithXvec:
    if encoder is None:
        raise ValueError("encoder is None")
    if decoder is None:
        raise ValueError("decoder is None")
    
    flow = CausalMaskedDiffWithXvec(input_size, output_size, spk_embed_dim, output_type, vocab_size,
                                    input_frame_rate, only_mask_loss, token_mel_ratio, pre_lookahead_len,
                                    encoder, decoder)
    return flow


if __name__ == "__main__":
    flow_decoder_estimator = initialize_flow_decoder_estimator()
    print(flow_decoder_estimator)
    # flow_decoder = initialize_causal_flow_decoder(flow_decoder_estimator)
    # print(flow_decoder)
    # flow_encoder = initialize_causal_flow_encoder()
    # print(flow_encoder)
    # flow = initialize_causal_flow(flow_encoder, flow_decoder)
    # print(flow)