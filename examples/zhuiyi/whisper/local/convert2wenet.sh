#!/bin/bash
. ./path.sh || exit 1;


python $WENET_DIR/wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py \
    --whisper_ckpt $1 \
    --output_dir $2 \