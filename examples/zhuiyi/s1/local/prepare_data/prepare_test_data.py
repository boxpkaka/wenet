#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by yuanding on 2022/07/20
"""wenet测试数据准备"""
import logging
from argparse import ArgumentParser
from pathlib import Path

from base_utils import dataset
from base_utils.dataset import CorpusConf, DbConf, db_name_parser
from base_utils.utils import LOGGER_FORMAT

from .prepare_data_for_raw import gen_data_by_wav_utts

_LOGGER = logging.getLogger(__file__)


def __cmd():
  desc = "准备生成测试集的raw格式数据."
  parser = ArgumentParser(description=desc, parents=[db_name_parser()])
  parser.add_argument("subsets", nargs="+", help="子集名称, 必须存在于AsrData表内.")
  parser.add_argument("--wavs_dir", type=Path, default=Path("data/wavs"),
                      help="wav音频生成目录, 默认data/wavs, 当路径存在时不会覆盖.")
  parser.add_argument("--pad_length", type=int, default=120,
                      help="生成音频时尾部padding静音时长, 默认120.")
  parser.add_argument("--is_english", default=False, action="store_true",
                      help="是否是英语, 默认否.")
  parser.add_argument("--nj", type=int, default=32, help="线程数, 默认32.")
  # TODO(fangcheng):
  args = parser.parse_args()

  corpus_conf = CorpusConf(DbConf(args.db_name))
  corpus = dataset.AsrCorpus(corpus_conf)
  wav_utts_list = corpus.get_wav_utts_list(args.subsets, filter_mty=True)
  gen_data_by_wav_utts(wav_utts_list, args.wavs_dir / args.subsets[0],
                       Path("data") / args.subsets[0],
                       pad_length=args.pad_length, nj=args.nj,
                       is_english=args.is_english)


if __name__ == '__main__':
  logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)
  __cmd()
