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
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch


def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    data_list = []
    for utt in tqdm(utt_list):
        data = open(utt2wav[utt], 'rb').read()  # 读取音频文件
        data_list.append(data)  # 将音频文件添加到data_list
    wav_list = [utt2wav[utt] for utt in utt_list]  # 记录每个utt对应的wav文件路径
    text_list = [utt2text[utt] for utt in utt_list]  # 记录每个utt对应的文本
    spk_list = [utt2spk[utt] for utt in utt_list]  # 记录每个utt对应的说话人id
    uttembedding_list = [utt2embedding[utt] for utt in utt_list]  # 记录每个utt对应的说话人embedding
    spkembedding_list = [spk2embedding[utt2spk[utt]] for utt in utt_list]  # 记录每个utt对应的说话人平均后embedding
    speech_token_list = [utt2speech_token[utt] for utt in utt_list]  # 记录每个utt对应的语音token序列

    # 保存到parquet,utt2parquet_file,spk2parquet_file
    df = pd.DataFrame()
    df['utt'] = utt_list
    df['wav'] = wav_list
    df['audio_data'] = data_list
    df['text'] = text_list
    df['spk'] = spk_list
    df['utt_embedding'] = uttembedding_list
    df['spk_embedding'] = spkembedding_list
    df['speech_token'] = speech_token_list
    df.to_parquet(parquet_file)  # 将数据保存到parquet文件
    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)  # 记录每个utt对应的parquet文件路径
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)  # 记录每个说话人对应的parquet文件路径
    logging.info('spend time {}'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        default='~/mll/tts/libritts/data/dev-clean',
                        type=str)
    parser.add_argument('--des_dir',
                        default='~/mll/tts/libritts/data/dev-clean/parquet',
                        type=str)
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir))  # 加载之前处理得到的utt对应的embedding向量，shape为[1,192]
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir))  # 加载之前处理得到的spk对应的embedding向量，shape为[1,192]
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir))  # 加载之前处理得到的每条utt中文本对应的speech tokens序列
    utts = list(utt2wav.keys())  # 获取所有utt

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)  # 创建进程池
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []  # 用于收集结果
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):  # 每1000个utt创建一个parquet文件
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))  # 创建parquet文件，每个parquet文件包含args.num_utts_per_parquet个utt
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file))  # 异步提交任务
    pool.close()  # 关闭进程池
    pool.join()  # 等待所有任务完成

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
