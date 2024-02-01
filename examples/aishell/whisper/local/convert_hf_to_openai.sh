#!/bin/bash
. ./path.sh || exit 1;

python $WENET_DIR/tools/convert_hf_to_openai.py \
    --checkpoint $1 \
    --whisper_dump_path $2