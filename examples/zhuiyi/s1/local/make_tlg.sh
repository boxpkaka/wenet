#!/bin/bash
# Created by yuanding on 2022/07/26
# 构建TLG解码图.

set -e
. ./path.sh

stage=0
stop_stage=2

if [[ ! $# -eq 3 ]]; then
  echo "Usage: $0 <dict> <text> <lexicon>"
  echo "dict: 端到端模型对应的符号表, 一般为data/dict/lang_char.txt."
  echo "text: 语言模型语料, 需要经过分词."
  echo "lexicon: 词典文件, 需要和语料分词使用的词典保持一致, 一般为local/resource_zhuiyi/lexicon.txt."
  exit 1
fi

dict=$1
text=$2
lexicon=$3

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: Prepare dict."
  unit_file=$dict
  mkdir -p data/local/dict
  cp $unit_file data/local/dict/units.txt
  tools/fst/prepare_dict.py $unit_file $lexicon \
    data/local/dict/lexicon.txt
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: Train lm."
  lm=data/local/lm
  mkdir -p $lm
  cp $text $lm/text
  local/train_lms.sh $lm/text data/local/dict/lexicon.txt $lm
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$(date) stage 2: Build decoding TLG."
  tools/fst/compile_lexicon_token_fst.sh \
    data/local/dict data/local/tmp data/local/lang
  tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1
fi
