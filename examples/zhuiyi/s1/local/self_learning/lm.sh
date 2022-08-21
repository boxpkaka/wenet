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
lambda=0.85

. tools/parse_options.sh || exit 1

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <model_dir> <text> <out_dir>"
  echo "model_dir: 模型文件夹."
  echo "text: 调优需要的文本."
  echo "out_dir: 输出文件夹."
  echo "--order: 默认3."
  echo "--lambda: 默认0.85."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$gpus
model_dir=$1
text=$2
out_dir=$3
data=$model_dir/self_learning/data/

lm=$out_dir/data/local/lm
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: train lm."
  mkdir -p $lm
  cp $text $lm/text
  local/train_lms.sh $lm/text $data/local/dict/lexicon.txt $lm
  mv ${lm}/lm.arpa ${lm}/part.arpa
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: interpolate ngrams."
  ngram -lm $data/local/lm/lm.arpa -order ${order} -mix-lm ${lm}/part.arpa -lambda ${lambda} -write-lm ${lm}/lm.arpa
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$(date) stage 2: Build decoding TLG."
  tools/fst/make_tlg.sh ${lm} $data/local/lang $out_dir/data/lang_test || exit 1
fi
