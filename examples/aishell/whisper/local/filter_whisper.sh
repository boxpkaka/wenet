#!/bin/bash


python local/filter_ckpt.py \
  --filter_list "encoder.embed.conv" \
  --input_ckpt $1 \
  --output_ckpt $2