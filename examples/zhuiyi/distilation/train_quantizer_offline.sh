#!/bin/bash

python train_quantizer.py  \
    --embedding_path /data1/yumingdong/offical/wenet/examples/aishell/distilation/data/yue100_for_quantizer_training.h5 \
    --save_path ./data \
    --quantizer_batch_size 512 \
    --quantizer_in_dim 1280 \
    --quantizer_out_dim 8 \
    --gpu 0 \