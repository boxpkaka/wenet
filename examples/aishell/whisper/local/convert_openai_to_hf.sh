#!/bin/bash
. ./path.sh || exit 1;


python $WENET_DIR/tools/convert_openai_to_hf.py \
    --checkpoint_path $1 \
    --pytorch_dump_folder_path $2 \
    --convert_preprocessor True 
