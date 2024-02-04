#!/bin/bash

python save_h5.py  \
    --checkpoint /data1/yumingdong/offical/wenet/examples/aishell/whisper/exp/finetune_whisper_largev3_conv2d4_zh_yue50+zh50/final.pt \
    --save_path data/test.h5 \
    --quantizer_path quantizer/quantizer_yue_100.pt \
    --quantizer_in_dim 1280 \
    --quantizer_out_dim 8 \
    --quantizer_codebook_size 256 \
    --train_data /data2/yumingdong/data/raw/wenet_data_list/yue_50h+zh_50h_2/data.list \
    --config conf/batchsize_1.yaml \
    --save_codebook