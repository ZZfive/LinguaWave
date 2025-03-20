# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
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

import logging
import os
import torch
import json
import re
import datetime
import yaml

import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

from tts.dataset.dataset import Dataset
from tts.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR


def init_distributed(args):  # 初始化分布式训练环境
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 获取分布式训练中的进程总数
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 获取当前进程在本地机器上的GPU编号，是单机多卡中的编号
    rank = int(os.environ.get('RANK', 0))  # 全局进程编号，是进程在整个训练集群中的编号，可能是多机多卡的全局编号
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == 'torch_ddp':  # 使用PyTorch DDP进行分布式训练
        torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU
        dist.init_process_group(args.dist_backend)  # 初始化进程组
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)  # 使用DeepSpeed进行分布式训练
    return world_size, local_rank, rank


def init_distributed_debug(args):  # 初始化分布式训练环境
    # 尝试从环境变量获取分布式训练参数，如果不存在则使用默认值
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 获取分布式训练中的进程总数
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 获取当前进程在本地机器上的GPU编号，是单机多卡中的编号
    rank = int(os.environ.get('RANK', 0))  # 全局进程编号，是进程在整个训练集群中的全局编号
    
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    
    if args.train_engine == 'torch_ddp':  # 使用PyTorch DDP进行分布式训练
        torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU
        
        # 检查是否设置了必要的环境变量
        env_ready = all(var in os.environ for var in ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'])
        
        if env_ready:
            # 如果环境变量已设置，使用env://方式初始化
            dist.init_process_group(args.dist_backend)
        else:
            # 如果环境变量未设置，使用更明确的参数初始化
            # 设置默认的master地址和端口
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '12355')
            
            logging.warning(f"环境变量未完全设置，使用默认值初始化分布式环境: "
                           f"rank={rank}, world_size={world_size}, "
                           f"master_addr={master_addr}, master_port={master_port}")
            
            # 使用明确的参数初始化进程组
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=f'tcp://{master_addr}:{master_port}',
                world_size=world_size,
                rank=rank
            )
    else:
        # DeepSpeed初始化
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, gan):  # 初始化训练和验证数据集和数据加载器
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']
    train_dataset = Dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=True, partition=True)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=False, partition=False)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):  # 校验配置
    if args.train_engine == "torch_ddp":
        configs['train_conf']["dtype"] = 'fp32'  # 设置训练引擎为PyTorch DDP时，将dtype设置为fp32
    else:
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(configs['train_conf']['save_per_step'] *
                                                     configs['train_conf']['accum_grad'] / ds_configs["gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):  # 包装模型，使其能够在分布式环境中运行
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))  # 获取本地机器上的GPU数量，即单节点GPU数量   
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 获取分布式训练中的进程总数
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)   # 使用PyTorch DDP包装模型
    else:  # deepspeed分布式训练
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)  # 估算Zero2优化所需内存
    return model


def init_optimizer_and_scheduler(args, configs, model, gan):  # 初始化优化器和学习率调度器
    if gan is False:
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        # use deepspeed optimizer for speedup
        if args.train_engine == "deepspeed":
            def scheduler(opt):
                return scheduler_type(opt, **configs['train_conf']['scheduler_conf'])
            model, optimizer, _, scheduler = deepspeed.initialize(  # 使用DeepSpeed初始化模型、优化器和学习率调度器
                args=args,
                model=model,
                optimizer=None,
                lr_scheduler=scheduler,
                model_parameters=model.parameters())

        optimizer_d, scheduler_d = None, None

    else:
        # currently we wrap generator and discriminator in one model, so we cannot use deepspeed
        # 初始化生成器优化器
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        # 初始化生成器学习率调度器
        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        # 初始化判别器优化器
        if configs['train_conf']['optim_d'] == 'adam':
            optimizer_d = optim.Adam(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim_d'] == 'adamw':
            optimizer_d = optim.AdamW(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        # 初始化判别器学习率调度器
        if configs['train_conf']['scheduler_d'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler_d = WarmupLR(optimizer_d, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler_d'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler_d = NoamHoldAnnealing(optimizer_d, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler_d = ConstantLR(optimizer_d)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])
    return model, optimizer, scheduler, optimizer_d, scheduler_d


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)  # 创建模型保存目录
        writer = SummaryWriter(args.tensorboard_dir)  # 创建TensorBoard写入器
    return writer


def save_model(model, model_name, info_dict):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))  # 模型保存路径

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:  # 只在主进程保存模型
            torch.save({**model.module.state_dict(), 'epoch': info_dict['epoch'], 'step': info_dict['step']}, save_model_path)
    else:  # 使用deepspeed保存模型
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=model_name,
                                  client_state=info_dict)
    if rank == 0:  # 只在主进程保存额外的模型信息
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(rank, save_model_path))


def cosyvoice_join(group_join, info_dict):  # 处理分布式训练中的进程同步问题，特别是处理不同GPU上工作负载不均衡的情况
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 获取分布式训练中的进程总数
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 获取当前进程在本地机器上的GPU编号，是单机多卡中的编号
    rank = int(os.environ.get('RANK', 0))  # 全局进程编号，是进程在整个训练集群中的编号，可能是多机多卡的全局编号

    if info_dict["batch_idx"] != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(group=group_join,
                                   timeout=group_join.options._timeout)  # 同步所有进程，等待所有进程完成当前批次的数据处理
            return False
        except RuntimeError as e:
            logging.info("Detected uneven workload distribution: {}\n".format(e) +
                         "Break current worker to manually join all workers, " +
                         "world_size {}, current rank {}, current local_rank {}\n".
                         format(world_size, rank, local_rank))
            return True  # 如果检测到不均衡的工作负载，则返回True，表示需要手动同步所有进程
    else:
        return False


def batch_forward(model, batch, scaler, info_dict):
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if info_dict['train_engine'] == 'torch_ddp':
        autocast = torch.cuda.amp.autocast(enabled=scaler is not None)  # 启用混合精度训练
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)  # 启用混合精度训练

    with autocast:
        info_dict['loss_dict'] = model(batch, device)  # 前向计算
    return info_dict


def batch_backward(model, scaler, info_dict):
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict['loss_dict']['loss'])  # 反向传播，计算梯度
    else:
        scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']  # 计算梯度
        if scaler is not None:
            scaler.scale(scaled_loss).backward()  # 反向传播，计算梯度
        else:
            scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    grad_norm = 0.0
    if info_dict['train_engine'] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()  # 获取全局梯度范数
    elif (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:  # 梯度累积
        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)  # 取消缩放，即将梯度从FP16/BF16转换为FP32
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])  # 裁剪梯度
            # We don't check grad here since that if the gradient
            # has inf/nan values, scaler.step will skip
            # optimizer.step().
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            scaler.update()  # 更新缩放因子
        else:
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])  # 裁剪梯度
            if torch.isfinite(grad_norm):
                optimizer.step()
        optimizer.zero_grad()  # 清空梯度
        scheduler.step()  # 更新学习率
    info_dict["lr"] = optimizer.param_groups[0]['lr']  # 获取当前学习率
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
           (info_dict['train_engine'] == 'torch_ddp' and (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] == 0):
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        log_str = '{} Batch {}/{} '.format(tag, epoch, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += '{} {:.6f} '.format(name, value)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    logging.info(
        'Epoch {} Step {} CV info lr {} {} rank {}'.format(
            epoch, step + 1, lr, rank, ' '.join(['{}_{}'.format(k, v) for k, v in loss_dict.items()])))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)