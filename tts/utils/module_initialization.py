import os
from functools import partial

from tts.utils.common import ras_sampling
from tts.llm.llm import Qwen2Encoder, Qwen2LM


def initialize_tokenize():
    pass
    

def initialize_llm(llm_input_size: int, llm_output_size: int, speech_token_size: int = 6561, length_normalized_loss: bool = True, lsm_weight: float = 0,
                   pretrain_path: str = "", top_p: float = 0.8, top_k: int = 25, win_size: int = 10, tau_r: float = 0.1) -> Qwen2LM:
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"pretrain_path {pretrain_path} not found")

    llm = Qwen2Encoder(pretrain_path)

    sampling = partial(ras_sampling, top_p=top_p, top_k=top_k, win_size=win_size, tau_r=tau_r)

    lm = Qwen2LM(llm_input_size, llm_output_size, speech_token_size, llm, sampling, length_normalized_loss, lsm_weight)

    return lm
