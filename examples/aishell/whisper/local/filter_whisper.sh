#!/bin/bash
. ./path.sh || exit 1;


python $WENET_DIR/tools/filter_ckpt.py \
  --filter_list "encoder.embed.conv" \
  --input_ckpt $1 \
  --output_ckpt $2