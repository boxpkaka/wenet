#!/bin/bash
# Created by yuanding on 2022/08/08
# 语言模型调优

set -e
. ./path.sh || exit 1

stage=0
stop_stage=1

order=3
lambda=0.6

. tools/parse_options.sh || exit 1

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <self_learning_dir> <text> <out_dir>"
  echo "self_learning_dir: 自学习文件夹路径, 一般放在发版模型文件夹下, 部分旧模型不支持."
  echo "text: 调优需要的清洗后的文本."
  echo "out_dir: 输出文件夹."
  echo "--order: 默认3."
  echo "--lambda: 默认0.6."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$gpus
self_learning=$1
text=$2
out_dir=$3
data=$self_learning/data/

lm=$out_dir/data/local/lm
lexicon=$data/local/dict/lexicon.txt

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: train lm and write ngrams to intermediate files."
  mkdir -p $lm

  python3 local/self_learning/split_by_char.py $text $lm/text

  cat $lm/text | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | sort | uniq -c | \
   sort -nr > $lm/unigram.counts || exit 1;

  cat $lm/unigram.counts | awk '{print $2}' | cat - <(echo "<s>"; echo "</s>" ) > $lm/wordlist

  cp -r $lm/text $lm/train

  lmplz -o 3 --text $lm/train --discount_fallback --skip_symbols --intermediate $lm/lm_tune.intermediate
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: interpolate ngrams and generate lm.bin."
  lambda2=`awk 'BEGIN{print 1 - '${lambda}'}'`
  echo $lambda2
  interpolate -m $data/local/lm/lm.intermediate $lm/lm_tune.intermediate -w ${lambda} \
   ${lambda2} > ${lm}/lm.arpa

  mv ${lm}/lm.arpa ${lm}/lm.bin
fi
