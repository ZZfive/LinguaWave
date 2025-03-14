# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # 设置可用GPU
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')  # 获取GPU数量
job_id=1996  # 设置分布式训练的ID
dist_backend="nccl"  # 设置分布式训练的后端
num_workers=2  # 设置数据加载的线程数
prefetch=100  # 设置数据加载的预取数量
train_engine=torch_ddp  # 设置训练引擎
pretrained_model_dir=''  # 试情况调整

echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
if [ $train_engine == 'deepspeed' ]; then
echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi

# cat data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > data/train.data.list  # 将训练数据合并
# cat data/{dev-clean,dev-other}/parquet/data.list > data/dev.data.list  # 将测试数据合并

# NOTE will update llm/hift training later
for model in llm; do
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    tts/bin/train.py \
    --train_engine $train_engine \
    --config conf/cosyvoice2.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --llm_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --model $model \
    --checkpoint $pretrained_model_dir/$model.pt \
    --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
    --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --use_amp \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
done