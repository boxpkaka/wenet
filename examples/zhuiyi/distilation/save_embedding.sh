#!/bin/bash

python save_h5.py  \
    --checkpoint /data1/yumingdong/model/huggingface/whisper-large-v3 \
    --encoder_type hubert \
    --save_path data/test.h5 \
    --train_data /data2/yumingdong/data/raw/wenet_data_list/yue_50h+zh_50h_2/data.list \
    --config conf/batchsize_1.yaml \
    --num_audio 2000