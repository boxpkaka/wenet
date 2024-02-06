#!/bin/bash

python train_quantizer.py  \
    --checkpoint /data1/yumingdong/model/finetuned/whisper-large-v3-lora700+700-130000 \
    --encoder_type huggingface \
    --train_data /data2/yumingdong/data/raw/wenet_data_list/yue_50h+zh_50h_2/data.list \
    --save_path ./quantizer/quantizer_whisper-large-v3-lora700+700-130000.pt \
    --quantizer_batch_size 512 \
    --quantizer_in_dim 1280 \
    --quantizer_out_dim 8 \
    --num_audio 1000 \
    --gpu 0 \
    --train_quantizer_online \