#!/bin/bash
# Created by yuanding on 2022/07/20
# 使用CTC/AED模型解码脚本

set -e
. ./path.sh || exit 1

export CUDA_VISIBLE_DEVICES=""

stage=0
stop_stage=2

data_type=raw
num_utts_per_shard=1000

test_set="test1 test2 test3 test4 test_dalianshuiwu test_longhudichan_yingxiao test_taipingcaixian test_zhongyizaixian"

dict=data/dict/lang_char.txt
dir=exp/unified_conformer

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=5
decode_modes="attention_rescoring"

. tools/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: 准备基础数据."
  for x in ${test_sets}; do
    python3 -m local.prepare_test_data $x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: 生成指定格式的数据."
  for x in ${test_sets}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 data/$x/wav.scp data/$x/text \
        $(realpath data/$x/shards) data/$x/data.list
    else
      tools/make_raw_list.py data/$x/wav.scp data/$x/text \
        data/$x/data.list
    fi
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$(date) stage 2: 解码."
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir \
      --num ${average_num} \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=16
  ctc_weight=0.5
  reverse_weight=0.5
  for mode in ${decode_modes}; do
  {
    echo "当前解码模式: "${decode_modes}
    for x in ${test_sets}; do
    {
      test_name=test_${mode}${decoding_chunk_size:+_chunk$decoding_chunk_size}
      test_dir=$dir/$test_name/${x}
      mkdir -p $test_dir
      echo "正在解码: ${x}, 解码日志: ${test_name}_${x}.log"
      python wenet/bin/recognize.py \
        --mode $mode \
        --config $dir/train.yaml \
        --data_type $data_type \
        --test_data data/$x/data.list \
        --checkpoint $decode_checkpoint \
        --beam_size 10 \
        --batch_size 1 \
        --penalty 0.0 \
        --dict $dict \
        --ctc_weight $ctc_weight \
        --result_file $test_dir/text \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
        >${test_name}_${x}.log 2>&1
      python tools/compute-wer.py --char=1 --v=1 \
        data/$x/text $test_dir/text >$test_dir/wer
    } &
    done
  }
  done
  wait
fi
