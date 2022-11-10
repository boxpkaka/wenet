#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by yuanding on 2022/07/20
"""wenet训练数据准备"""
import collections
import logging
import random
from argparse import ArgumentParser
from pathlib import Path

from base_utils import dataset
from base_utils.dataset import CorpusConf, DbConf, db_name_parser
from base_utils.utils import LOGGER_FORMAT

from .prepare_data_for_raw import gen_data_by_wav_utts

_LOGGER = logging.getLogger(__file__)


def combine_wav_utts(wav_utt_list):
  """组合wav和utt.

  Args:
      wav_utt_list: 一条wav和一个utt组成的列表.

  Returns:
      一条wav和多个utt组成的列表.
  """
  wav_to_utts = collections.defaultdict(list)
  for wav, utt in wav_utt_list:
    wav_to_utts[wav].append(utt)
  return list(wav_to_utts.items())


def __cmd():
  desc = "准备生成训练数据的raw格式数据到data/train."
  parser = ArgumentParser(description=desc, parents=[db_name_parser()])
  parser.add_argument("subsets", nargs="+", help="子集名称, 必须存在于AsrData表内.")
  parser.add_argument("--wavs_dir", type=Path, default=Path("data/wavs"),
                      help="wav音频生成目录, 默认data/wavs, 当路径存在时不会覆盖.")
  parser.add_argument("--dev_splits", type=float, default=0.05,
                      help="验证集划分比例, 默认0.05.")
  parser.add_argument("--bizs", nargs="+", help="选取特定的业务, 支持多个.")
  parser.add_argument("--nj", type=int, default=32, help="线程数, 默认32.")
  args = parser.parse_args()

  corpus_conf = CorpusConf(DbConf(args.db_name))
  corpus = dataset.AsrCorpus(corpus_conf)
  wav_utts_list = corpus.get_wav_utts_list(args.subsets,
                                           args.bizs,
                                           filter_mty=True)
  # 将一条wav对应多个utt转换成一条wav对应一个utt, 方便数据划分.
  wav_to_utt = [(k, v) for k, vs in wav_utts_list for v in vs]
  random.seed(777)
  random.shuffle(wav_to_utt)
  dev_nums = min(int(len(wav_to_utt) * args.dev_splits), 5000)
  gen_data_by_wav_utts(combine_wav_utts(wav_to_utt[:-dev_nums]),
                       args.wavs_dir / "train", Path("data/train"), nj=args.nj)
  gen_data_by_wav_utts(combine_wav_utts(wav_to_utt[-dev_nums:]),
                       args.wavs_dir / "dev", Path("data/dev"), nj=args.nj)


if __name__ == '__main__':
  logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)
  __cmd()
