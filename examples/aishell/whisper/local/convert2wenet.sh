#!/bin/bash

python wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py \
    --whisper_ckpt $1 \
    --output_dir $2 \