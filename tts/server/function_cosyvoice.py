import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("~/LinguaWave")
sys.path.append("/the/path/to/third_party/Matcha-TTS")
import random
from typing import List

import torch
import librosa
import torchaudio
import numpy as np
from scipy.io.wavfile import write
try:
    import noisereduce as nr
    nr_installed = True
except ImportError:
    print("noisereduce is not installed, please install it using pip install noisereduce")
    nr_installed = False

from tts.utils.file_utils import load_wav
from tts.cli.cosyvoice import CosyVoice2OtherLLM

video_save_dir = "~/audios"
prompt_sr, target_sr = 16000, 24000
max_val = 0.8


def set_all_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocess(speech: torch.Tensor, top_db: int = 60, hop_length: int = 220, win_length: int = 440) -> torch.Tensor:
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def save_audio_to_wav(audio_data_list: List[np.ndarray], sample_rate: int, file_path: str, apply_noise_reduction: bool = False,
                      prop_decrease: float = 0.6, n_std_thresh_stationary: float = 2) -> None:
    # 将多个音频数据合并
    audio_data = np.concatenate(audio_data_list)
    
    # 应用噪音消除
    if nr_installed and apply_noise_reduction:
        audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=prop_decrease,
                                     n_std_thresh_stationary=n_std_thresh_stationary)
    
    # 确保音频数据在合适的范围和格式中
    if audio_data.dtype != np.int16:
        # 如果音频数据不是 int16 格式，可能需要将其转换为适当的格式
        audio_data = (audio_data * 32767).astype(np.int16)  # 假设音频数据是归一化的浮点数
    
    # 保存为 wav 文件
    write(file_path, sample_rate, audio_data)


# 基于cosyvoice实现文本生音频
def cosyvoice_audio_clone(cosyvoice: torch.nn.Module, tts_text: str, prompt_text: str, prompt_wav_path: str, seed: int = None,
                          apply_noise_reduction: bool = True, prop_decrease: float = 0.6, n_std_thresh_stationary: float = 2,
                          cross_lingual: bool = False, save_dir: str = video_save_dir, name: str = None) -> str:
    save_path = ""
    try:
        if prompt_text == '' or prompt_text is None:
            raise ValueError('参考音色音频对应文本为空')
        
        if not os.path.exists(prompt_wav_path):
            raise ValueError(f'参考音色音频文件不存在:{prompt_wav_path}')
        
        try:
            info = torchaudio.info(prompt_wav_path)
        except Exception as e:
            raise RuntimeError(f"音频文件损坏:{e}")
        if info.sample_rate < prompt_sr:
            raise ValueError(f'参考音色音频采样率{info.sample_rate}低于{prompt_sr}')
        duration = info.num_frames / info.sample_rate
        if duration > 30.0:
            raise ValueError(f'参考音色音频时长{duration}高低于30.0s')
            
        prompt_speech_16k = postprocess(load_wav(prompt_wav_path, prompt_sr))
        seed = random.randint(1, 100000000) if seed is None else seed
        # seed = 425922
        set_all_random_seed(seed)
        
        audio_data_list = []
        if cross_lingual:
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k):
                audio_numpy = i['tts_speech'].numpy().flatten()
                # yield (target_sr,  audio_numpy)
                audio_data_list.append(audio_numpy)
        else:
            for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k):
                audio_numpy = i['tts_speech'].numpy().flatten()
                # yield (target_sr,  audio_numpy)
                audio_data_list.append(audio_numpy)
        
        audio_name = f'{name}.wav' if name is not None else f'audio_{seed}.wav'
        save_path = os.path.join(save_dir, audio_name)
        save_audio_to_wav(audio_data_list, target_sr, save_path, apply_noise_reduction,
                          prop_decrease, n_std_thresh_stationary)
        
        return save_path
    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    model = CosyVoice2OtherLLM(yaml_path='~/LinguaWave/tts/yamls/cosyvoice2_other_llm.yaml',
                               model_dir='/the/path/to/CosyVoice2-0.5B',
                               llm_not_loaded=True)
    tts_text = '你好呀，我的朋友'
    prompt_text = '我是通一实验室语音团队全新推出的深层式语音大模型提供舒适自然的语音合成能力'
    prompt_wav_path = '/the/path/to/cosyvoice.mp3'
    test_wav_path = cosyvoice_audio_clone(model, tts_text, prompt_text, prompt_wav_path, name='internlm2_5-1_8b-test')
    print(test_wav_path)