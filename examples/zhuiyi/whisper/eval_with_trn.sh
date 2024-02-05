#!/bin/bash

std_root_dir=/data2/yumingdong/data/raw/wenet
exp_dir=/data1/yumingdong/offical/wenet/examples/aishell/whisper/exp/whisper-large-v3_multi_lang_yue_50h+zh_50h_ctc_0_conv2d4
dataset=aishell/test
decode_mode=attention

# copy the std.trn to target
cp $std_root_dir/$dataset/std.trn $exp_dir/$dataset/$decode_mode/std.trn

cd /data1/yumingdong/whisper/whisper-eval/

# generate the reg.trn
python tools/prepare_std_trn_from_text.py $exp_dir/$dataset/$decode_mode/ reg.trn
python -m eval.eval_with_trn $exp_dir/$dataset/$decode_mode/

