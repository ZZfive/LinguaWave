#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper


def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt])  # 加载音频
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)  # 重采样到16k
    if audio.shape[1] / 16000 > 30:
        logging.warning('do not support extract speech token for audio longer than 30s')
        speech_token = []
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)  # 提取梅尔频谱图
        speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                              ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()  # 提取语音token
    return utt, speech_token


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]  # 并行提交所有任务
    utt2speech_token = {}  # 用于收集结果
    for future in tqdm(as_completed(all_task)):  # 遍历所有任务
        utt, speech_token = future.result()  # 获取结果
        utt2speech_token[utt] = speech_token  # 记录utt与语音token的映射
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))  # 保存utt与语音token的映射


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    utt2wav = {}  # 用于记录utt与wav的映射
    with open('{}/wav.scp'.format(args.dir)) as f:  # 读取wav.scp文件
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    option = onnxruntime.SessionOptions()  # 创建onnx推理会话选项
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有优化
    option.intra_op_num_threads = 1  # 设置内部线程数
    providers = ["CUDAExecutionProvider"]  # 设置提供者
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)  # 创建onnx推理会话
    executor = ThreadPoolExecutor(max_workers=args.num_thread)  # 创建线程池

    main(args)
