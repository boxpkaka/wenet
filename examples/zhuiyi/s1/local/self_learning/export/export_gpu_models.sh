#!/bin/bash
# Created by fangcheng on 2022/12/07
# 导出gpu推理需要的模型文件

. ./path.sh || exit 1

stage=0
stop_stage=0

conf_path=""
beam_size=10
num_decoding_left_chunks=5
online_chunk_size=16
offline_chunk_size=-1

average_num=5

. tools/parse_options.sh || exit 1

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <in_dir> <self_learning_dir> <out_dir>"
  echo "in_dir: 训练完成的模型文件夹路径,需要包含train.yaml以及pytorch模型文件路径."
  echo "self_learning_dir: 自学习文件夹路径."
  echo "out_dir: 导出的模型推理文件夹路径."
  echo "--average_num: 默认5."
  echo "--conf_path: 配置文件路径,用于获取参数,一般为conf/asr.yaml,默认为空."
  exit 1
fi

in_dir=$1
self_learning_dir=$2
out_dir=$3


if [[ $conf_path != "" ]]; then
  num_decoding_left_chunks=`sed '/  num_left_chunks:/!d;s/.*://' $conf_path`
  beam_size=`sed '/  beam:/!d;s/.*://' $conf_path`
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: 导出onnx model."

  # 导出流式模型
  onnx_dir=$out_dir/onnx/online_model
  python local/triton_model_repo/scripts/export_onnx_gpu.py \
    --config ${in_dir}/train.yaml \
    --cmvn_file ${in_dir}/global_cmvn \
    --checkpoint ${in_dir}/avg_${average_num}.pt \
    --decoding_chunk_size ${online_chunk_size} \
    --beam_size ${beam_size} \
    --output_onnx_dir ${out_dir} \
    --num_decoding_left_chunks ${num_decoding_left_chunks}

  # 导出非流式模型
  onnx_dir=$out_dir/onnx/offline_model
  python local/triton_model_repo/scripts/export_onnx_gpu.py \
    --config ${in_dir}/train.yaml \
    --cmvn_file ${in_dir}/global_cmvn \
    --checkpoint ${in_dir}/avg_${average_num}.pt \
    --decoding_chunk_size $offline_chunk_size \
    --beam_size ${beam_size} \
    --output_onnx_dir ${out_dir} \
    --num_decoding_left_chunks ${num_decoding_left_chunks}

fi
