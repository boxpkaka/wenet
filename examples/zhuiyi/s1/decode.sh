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

dir=exp/unified_conformer
test_set_dir=/data2/yumingdong/data/raw/wenet/
test_sets="test1"
dict=/data1/yumingdong/model/wenet/asr_model_v4.5.0/lang_char.txt
data_dir=$test_set_dir/${test_sets}

average_num=5
decode_modes="attention_rescoring"

if [ ! -f "$data_dir/data.list" ]; then
    echo "data.list file no exist in $data_dir, try to create it"
    python tools/make_raw_list.py $data_dir/wav.scp $data_dir/text $data_dir/data.list
    echo "$data_dir/data.list has been created"
fi

if [[ ${average_checkpoint} == "true" ]]; then
  decode_checkpoint=$dir/avg_${average_num}.pt
else
  decode_checkpoint=$dir/final.pt
fi

# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#   echo "$(date) stage 0: 准备基础数据."
#   for x in ${test_sets}; do
#     python3 -m local.prepare_data.prepare_test_data $x
#   done
# fi

# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#   echo "$(date) stage 1: 生成指定格式的数据."
#   for x in ${test_sets}; do
#     if [ $data_type == "shard" ]; then
#       tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
#         --num_threads 16 data/$x/wav.scp data/$x/text \
#         $(realpath data/$x/shards) data/$x/data.list
#     else
#       tools/make_raw_list.py data/$x/wav.scp data/$x/text \
#         data/$x/data.list
#     fi
#   done
# fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$(date) stage 2: 解码."
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=16
  ctc_weight=0.5
  reverse_weight=0.5
  for mode in ${decode_modes}; do
  {
    echo "当前解码模式: "${decode_modes}
      test_name=test_${mode}${decoding_chunk_size:+_chunk$decoding_chunk_size}
      # mkdir -p $test_dir
      echo "正在解码: ${test_sets}, 解码日志: ${test_name}_${x}.log"
      python wenet/bin/recognize.py \
        --mode $mode \
        --config $dir/train.yaml \
        --data_type $data_type \
        --test_data $data_dir/data.list \
        --checkpoint $decode_checkpoint \
        --beam_size 10 \
        --batch_size 1 \
        --penalty 0.0 \
        --dict $dict \
        --ctc_weight $ctc_weight \
        --result_file $dir/text \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
        >${test_name}_${x}.log 2>&1
      python tools/compute-wer.py --char=1 --v=1 \
        $dir/text $data_dir/text >$dir/wer
    }
  done
fi
