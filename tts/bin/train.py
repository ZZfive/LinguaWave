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

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from tts.utils.executor import Executor
from tts.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record  # 记录异常
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False  # Hifigan是GAN类模型，训练时要分别训练生成器和判别器

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}  # model对应的key会从override_dict中移除
    if gan is True:
        override_dict.pop('hift')
    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'llm_pretrain_path': args.llm_pretrain_path})  # 用传入的llm_pretrain_path覆盖yaml中的llm_pretrain_path
    except Exception:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))  # 将args中的参数更新到configs['train_conf']中

    # Init env for ddp
    init_distributed(args)  # 初始化分布式环境

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan)   # 初始化数据集和数据加载器

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)  # 校验配置

    # Tensorboard summary
    writer = init_summarywriter(args)   # 初始化TensorBoard写入器

    # load checkpoint
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:  # 断点续训，如果设置了之前训练的checkpoint，则加载checkpoint
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            if 'step' in state_dict:
                start_step = state_dict['step']  # 从checkpoint中获取上次训练停止时的step
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']  # 从checkpoint中获取上次训练停止时的epoch
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)  # 将模型从CPU移动到GPU

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)  # 初始化优化器和调度器
    scheduler.set_step(start_step)  # 设置调度器步数
    if scheduler_d is not None:
        scheduler_d.set_step(start_step)  # 设置调度器步数

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)  # 保存初始化模型

    # Get executor
    executor = Executor(gan=gan)  # 初始化训练执行器
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None  # 初始化混合精度训练的scaler
    print('start step {} start epoch {}'.format(start_step, start_epoch))
    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch  # epoch在此处更新，executor.step会在train_one_epoc内部更新
        train_dataset.set_epoch(epoch)
        dist.barrier()  # 进程同步；确保所有进程继续执行前都完成前面的操作，只有当所有进程都执行到dist.barrier()时，才会继续执行后面的代码
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))  # 创建一个分布式组
        if gan is True:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join)  # 训练一个epoch
        dist.destroy_process_group(group_join)  # 销毁分布式组


if __name__ == '__main__':
    main()
