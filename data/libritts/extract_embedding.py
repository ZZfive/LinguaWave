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
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt])  # 加载音频
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)  # 重采样到16k
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,  # 80维梅尔滤波器组
                       dither=0,  # 不使用抖动
                       sample_frequency=16000)  # 提取fbank特征
    feat = feat - feat.mean(dim=0, keepdim=True)  # 减去均值，特征归一化
    # 提取embedding
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()  # 提取embedding
    return utt, embedding


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]  # 并行提交所有任务
    utt2embedding, spk2embedding = {}, {}  # 用于收集结果
    for future in tqdm(as_completed(all_task)):  # 遍历所有任务
        utt, embedding = future.result()  # 获取结果
        utt2embedding[utt] = embedding  # 记录utt与对应音频中说话人的embedding的映射
        spk = utt2spk[utt]  # 获取当前音频文件对应的说话人id
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)  # 记录说话人id与其所有embedding的映射
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()  # 计算说话人embedding的均值
    torch.save(utt2embedding, "{}/utt2embedding.pt".format(args.dir))  # 保存utt与embedding的映射
    torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))  # 保存说话人id与其所有embedding的映射


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)  # 数据目录
    parser.add_argument("--onnx_path", type=str)  # onnx模型路径
    parser.add_argument("--num_thread", type=int, default=8)  # 线程数
    args = parser.parse_args()

    utt2wav, utt2spk = {}, {}  # 用于记录utt与wav/spk的映射
    with open('{}/wav.scp'.format(args.dir)) as f:  # 读取wav.scp文件
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/utt2spk'.format(args.dir)) as f:  # 读取utt2spk文件
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有优化
    option.intra_op_num_threads = 1  # 设置内部线程数
    providers = ["CPUExecutionProvider"]  # 设置提供者
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)  # 创建onnx推理会话
    executor = ThreadPoolExecutor(max_workers=args.num_thread)  # 创建线程池

    main(args)
