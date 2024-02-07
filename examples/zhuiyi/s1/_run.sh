#!/bin/bash
# Created by yuanding on 2022/08/03
# CTC/AED模型调优训练

. ./path.sh || exit 1

stage=0
stop_stage=1
# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1
# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0

data_type=raw
train_set=train
dev_set=dev

cmvn=true

# use average_checkpoint will get better result
average_checkpoint=true
average_num=5
bpe_model=
gpus="7"
lr=0.0004
batch_size=16
epoch=20
warmup_steps=500
accum_grad=2
cpus=-1

. tools/parse_options.sh || exit 1

# if [ $# != 3 ]; then
#   echo "Usage: $0 [options] <data_dir> <model_dir> <out_dir>"
#   echo "data_dir: 调优数据文件夹, 需要包含train和dev."
#   echo "model_dir: 发版模型文件夹, 需包含self_learning文件夹, 部分旧模型不支持."
#   echo "out_dir: 调优模型保存路径."
#   echo "--average_num: 默认5."
#   echo "--gpus: 显卡编号, ','连接, 如'0,1,2,3'."
#   echo "--cpus: cpu训练时指定使用的cpu数量, 不指定则默认使用机器cpu核数的一半."
#   echo "--lr: 学习率, 默认0.0004."
#   echo "--batch_size: 默认16."
#   echo "--epoch: 默认20."
#   echo "--warmup_steps: 默认500"
#   echo "--accum_grad: 默认2"
#   exit 1
# fi

export CUDA_VISIBLE_DEVICES=$gpus
data_dir=/data2/yumingdong/data/raw/wenet_data_list/yue_50h+zh_50h_2
model_dir=/data1/yumingdong/model/wenet/asr_model_v4.5.0/
out_dir='./exp/baseline_20240207/'
tensorboard_dir=$out_dir/tensorboard

# codebook_path=/data1/yumingdong/official/wenet/examples/zhuiyi/distilation/codebook/whisper_largev3_conv2d4_zh_yue50+zh50.h5

self_learning=$model_dir/self_learning
train_config=$self_learning/exp/train.yaml
dict=$model_dir/lang_char.txt
checkpoint=$self_learning/exp/init.pt
cmvn_dir=$self_learning/exp/global_cmvn

if [ -f $self_learning/data/lang_char/bpe.model ]; then
  bpe_model=$self_learning/data/lang_char/bpe.model
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: 生成指定格式的数据."
  for x in ${train_set} ${dev_set}; do
    tools/make_raw_list.py $data_dir/$x/wav.scp $data_dir/$x/text \
      $data_dir/$x/data.list
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: 开始训练."
  mkdir -p $out_dir
  mkdir -p $tensorboard_dir
  # INIT_FILE is for DDP synchronization
  INIT_FILE=$out_dir/ddp_init
  if [ -f $INIT_FILE ]; then
    rm $INIT_FILE
  fi
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  world_size=$(expr $num_gpus \* $num_nodes)
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp ${cmvn_dir} ${out_dir}
  $cmvn && cmvn_opts="--cmvn ${out_dir}/global_cmvn"
  
  # CPU训练时的相关配置
  cp wenet/bin/train.py local/self_learning/
  if [ ${num_gpus} -eq 0 ]; then
    num_gpus=1
    world_size=-1
    if [ ${cpus} -ne -1 ]; then
      sed -i "/def main():/a\    torch.set_num_threads(${cpus})" local/self_learning/train.py
    fi
  fi

  for ((i = 0; i < $num_gpus; ++i)); do
  { 
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$(($i + 1)))
    if [ ${world_size} -eq -1 ]; then
      gpu_id=-1
    fi
    rank=$(expr $node_rank \* $num_gpus + $i)

    python local/self_learning/train.py --gpu $gpu_id \
      --config $train_config \
      --override_config "optim_conf.lr ${lr}" \
      --override_config "dataset_conf.batch_conf.batch_size ${batch_size}" \
      --override_config "max_epoch ${epoch}" \
      --override_config "scheduler_conf.warmup_steps ${warmup_steps}" \
      --override_config "accum_grad ${accum_grad}" \
      --data_type $data_type \
      --symbol_table $dict \
      ${bpe_model:+--bpe_model $bpe_model} \
      --train_data $data_dir/${train_set}/data.list \
      --cv_data $data_dir/$dev_set/data.list \
      ${codebook_path:+--codebook $codebook_path} \
      ${checkpoint:+--checkpoint $checkpoint} \
      ${tensorboard_dir:+--tensorboard_dir $tensorboard_dir} \
      --model_dir $out_dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 8 \
      $cmvn_opts \
      --pin_memory
  } &
  done
  wait
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  if [ ${average_checkpoint} == true ]; then
      decode_checkpoint=$out_dir/avg_${average_num}.pt
      echo "do model average and final checkpoint is $decode_checkpoint"
      python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $out_dir \
        --num ${average_num} \
        --val_best
  fi
fi