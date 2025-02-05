# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

import random
import json
import math
from functools import partial
from typing import Callable

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from tts.utils.file_utils import read_lists, read_json_lists


class Processor(IterableDataset):  # 继承自IterableDataset，提供一个可以链式处理数据的管道机制

    def __init__(self, source: IterableDataset, f: Callable, *args, **kw):
        assert callable(f)
        self.source = source  # 数据源
        self.f = f  # 处理函数
        self.args = args  # 处理函数的参数
        self.kw = kw  # 处理函数的参数

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)  # 每次迭代都会调用处理函数 f 来处理源数据

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)  # 通过 apply 方法支持链式调用


class DistributedSampler:  # 分布式采样器，用于在分布式训练中对数据进行采样

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle  # 是否打乱数据
        self.partition = partition  # 是否进行数据分区

    def update(self):
        assert dist.is_available()  # 确保分布式训练环境可用
        if dist.is_initialized():  # 如果分布式训练环境已初始化
            self.rank = dist.get_rank()  # 获取当前进程的rank
            self.world_size = dist.get_world_size()  # 获取分布式训练环境中的进程总数
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()  # 获取当前工作进程的信息
        if worker_info is None:  # 如果没有工作进程信息，则认为是在单进程模式下
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers  # 获取当前工作进程的编号和进程总数
        return dict(rank=self.rank,  # 返回一个包含当前进程信息和进程编号的字典
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):  # 对数据进行采样和分配
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))  # 创建一个包含数据索引的列表
        # force datalist even
        if self.partition:  # 如果需要进行数据分区
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)  # 基于epoch对数据进行打乱
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))  # 确保数据量不小于进程数
                data = data[:self.world_size]
            data = data[self.rank::self.world_size]  # 根据rank和world_size对数据进行分区
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))  # 确保数据量不小于工作进程数
            data = data[:self.num_workers]
        data = data[self.worker_id::self.num_workers]  # 根据worker_id和工作进程数对数据进行分区
        return data


'''
继承自IterableDataset不需要定义__getitem__方法和__len__方法，只需要定义__iter__方法
可流式访问数据，适用于大数据集和流式数据，按序生成数据，不用一次性将所有数据加载到内存中，通过DistributedSampler实现数据分片
每个进程只处理自己需要的数据，避免重复加载
'''
class DataList(IterableDataset):

    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists  # 数据列表，就是tar文件列表
        self.sampler = DistributedSampler(shuffle, partition)  # 创建分布式采样器

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()  # 获取分布式训练环境信息
        indexes = self.sampler.sample(self.lists)  # 对数据进行采样和分配
        for index in indexes:
            data = dict(src=self.lists[index])  # 创建一个包含数据源和索引的字典，key为"src"，value为self.lists[index]
            data.update(sampler_info)  # 将分布式训练环境信息添加到数据中
            yield data  # 一次只返回一个数据，数据量小，适合流式处理；此处返回的就是一个tar文件，其中包含的数据数量由数据预处理时决定，目前是1000


def Dataset(data_list_file,
            data_pipeline,
            mode='train',
            gan=False,
            shuffle=True,
            partition=True,
            tts_file='',
            prompt_utt2data=''):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert mode in ['train', 'inference']
    lists = read_lists(data_list_file)  # 读取数据列表，就是tar文件列表
    if mode == 'inference':  # 推理模式
        with open(tts_file) as f:
            tts_data = json.load(f)
        utt2lists = read_json_lists(prompt_utt2data)
        # filter unnecessary file in inference mode
        lists = list({utt2lists[utt] for utt in tts_data.keys() if utt2lists[utt] in lists})
    dataset = DataList(lists,
                       shuffle=shuffle,
                       partition=partition)  # 创建数据集
    if mode == 'inference':
        # map partial arg to parquet_opener func in inference mode
        data_pipeline[0] = partial(data_pipeline[0], tts_data=tts_data)  # 将tts_data传递给data_pipeline[0]
    if gan is True:
        # map partial arg to padding func in gan mode
        data_pipeline[-1] = partial(data_pipeline[-1], gan=gan)  # 将gan传递给data_pipeline[-1]
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)  # 将dataset和func传递给Processor
    return dataset