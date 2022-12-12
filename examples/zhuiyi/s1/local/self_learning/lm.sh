#!/bin/bash
# Created by yuanding on 2022/08/08
# 语言模型调优

set -e
. ./path.sh || exit 1

stage=0
stop_stage=2

# use average_checkpoint will get better result
average_checkpoint=true
average_num=5

order=3
lambda=0.6
is_gpu_infer=false
is_kn_smooth=false

. tools/parse_options.sh || exit 1

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <self_learning_dir> <text> <out_dir>"
  echo "self_learning_dir: 自学习文件夹路径."
  echo "text: 调优需要的文本."
  echo "out_dir: 输出文件夹."
  echo "--order: 默认3."
  echo "--lambda: 默认0.6."
  echo "--is_gpu_infer: 是否为gpu推理需要的lm文件, 默认否."
  echo "--is_kn_smooth: 是否为kneserney平滑算法, 默认否, 即使用wittenbell平滑."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$gpus
self_learning=$1
text=$2
out_dir=$3
data=$self_learning/data/

lm=$out_dir/data/local/lm

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: train lm."
  mkdir -p $lm
  cp $text $lm/text
  local/train_lms.sh --is_kn_smooth ${is_kn_smooth} $lm/text $data/local/dict/lexicon.txt $lm
  mv ${lm}/lm.arpa ${lm}/part.arpa
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: interpolate ngrams."
    ngram -lm $data/local/lm/lm.arpa -order ${order} -mix-lm ${lm}/part.arpa -lambda ${lambda} -write-lm ${lm}/lm.arpa
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  if [[ ${is_gpu_infer} == true ]];
  then
    echo "$(date) stage 2: Build lm.bin."
    build_binary -s ${lm}/lm.arpa $out_dir/lm.bin || exit 1
  else
    echo "$(date) stage 2: Build decoding TLG."
    tools/fst/make_tlg.sh ${lm} $data/local/lang $out_dir/data/lang_test || exit 1
  fi
fi
