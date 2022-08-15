#!/bin/bash
# Created by yuanding on 2022/07/20
# 使用TLG解码图解码

set -e
. ./path.sh || exit 1

export CUDA_VISIBLE_DEVICES=""

stage=0
stop_stage=1

nj=24

test_sets="test1 test2 test3 test4 test_dalianshuiwu test_longhudichan_yingxiao test_taipingcaixian test_zhongyizaixian"

. tools/parse_options.sh || exit 1

if [[ ! $# -eq 3 ]]; then
  echo "Usage: $0 <script_model> <lang_test> <result_dir>"
  echo "script_model: TorchScript模型路径, 一般位于exp/unified_conformer."
  echo "lang_test: 包含解码需要的TLG.fst和words.txt, 一般为data/lang_test."
  echo "result_dir: 结果文件夹."
  echo "解码前需要准备好解码图和解码程序, 解码图构建参考local/make_tlg.sh, 解码程序参考wenet runtime模块."
  exit 1
fi

script_mdoel=$1
lang_test=$2
result_dir=$3

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$(date) stage 0: 准备基础数据."
  for x in ${test_sets}; do
    python3 -m local.prepare_data.prepare_test_data ${x}
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(date) stage 1: 解码."
  for x in ${test_sets}; do
  {
    echo "正在解码测试集${x}"
    chunk_size=16
    ./local/decode.sh --nj ${nj} \
      --beam 10 --lattice_beam 5 --max_active 4000 \
      --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
      --acoustic_scale 4 \
      --chunk_size ${chunk_size} \
      --fst_path ${lang_test}/TLG.fst \
      data/${x}/wav.scp data/${x}/text.fmt ${script_mdoel} \
      ${lang_test}/words.txt ${result_dir}/lm_with_runtime_${x}
  }
  done
fi
