#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi

count_wer_only=false
test_set_dir=/data2/yumingdong/data/raw/wenet/
test_set_name=aishell/test 
dir=/data1/yumingdong/offical/wenet/examples/aishell/whisper/exp/whisper-large-v3_multi_lang_yue_50h+zh_50h
gpu=2

decode_checkpoint=$dir/final.pt
data_dir=$test_set_dir/${test_set_name}

average_checkpoint=false
average_num=5
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ! -f "$data_dir/data.list" ]; then
    echo "data.list file no exist in $data_dir, try to create it"
    python tools/make_raw_list.py $data_dir/wav.scp $data_dir/text $data_dir/data.list
    echo "$data_dir/data.list has been created"
fi

if [ ${average_checkpoint} == true ]; then
  decode_checkpoint=$dir/avg_${average_num}.pt
  echo "do model average and final checkpoint is $decode_checkpoint"
  python wenet/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path $dir  \
    --num ${average_num} \
    --val_best
fi
# Please specify decoding_chunk_size for unified streaming and
# non-streaming model. The default value is -1, which is full chunk
# for non-streaming inference.
if [ ${count_wer_only} == false ]; then
  decoding_chunk_size=-1
  ctc_weight=0.3
  reverse_weight=0.5
  python wenet/bin/recognize.py --gpu $gpu \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type raw \
    --test_data $data_dir/data.list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 8 \
    --penalty 0.0 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $dir/$test_set_name \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
    # --simulate_streaming 
fi

for mode in ${decode_modes}; do
  python tools/compute-wer.py --char=1 --v=1 \
    $data_dir/text $dir/$test_set_name/$mode/text > $dir/$test_set_name/$mode/wer
done
