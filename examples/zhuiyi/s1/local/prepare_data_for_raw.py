#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by yuanding on 2022/07/19
"""wenet数据准备"""
import logging
import wave
from argparse import ArgumentParser
from pathlib import Path
import re
from threading import Thread

import numpy as np
from base_utils import dataset
from base_utils.audio import WavInfo
from base_utils.dataset import CorpusConf, DbConf, db_name_parser
from base_utils.utils import LOGGER_FORMAT

_LOGGER = logging.getLogger(__file__)


def write_wav(wav_utts_list, out_dir: Path, pad_length):
  """生成音频数据到指定的路径.

  Args:
      wav_utts_list: wav和对应的utts列表.
      out_dir: 生成音频的路径.
  """
  silence = b'\x00\x00' * 8 * pad_length
  for wav, utts in wav_utts_list:
    wav_info = WavInfo(wav)
    for utt in utts:
      wav_data = wav_info.data(utt.begin, utt.end, utt.channel) + silence
      wav_path = out_dir / (utt.utt_id() + ".wav")
      with wave.open(str(wav_path), mode='wb') as wav:
        wav.setparams((1, int(wav_info.sample_width), wav_info.sample_rate, 0,
                       'NONE', 'NONE'))
        wav.writeframes(wav_data)


def write_wav_multithread(wav_utts_list, wav_dir: Path, pad_length, nj):
  """多线程生成音频数据.

  Args:
      wav_utts_list: wav和对应的utts的列表.
      wav_dir: 生成音频的路径.
      nj: 任务数.
  """
  splits = np.array_split(wav_utts_list, nj)
  jobs = []
  for idx in range(nj):
    t_obj = Thread(target=write_wav,
                   args=([splits[idx].tolist(), wav_dir, pad_length]))
    jobs.append(t_obj)
    t_obj.start()
  for job in jobs:
    job.join()


def format_text(text, remove_en_space=True):
  """格式化标注文本.

  Args:
      text: 格式化之前的文本.
      remove_en_space: 是否需要去掉英文之间的空格.

  Returns:
      格式化之后的文本.
  """
  text = text.upper().replace("{UM}", "呃")
  pattern_space = r'\s+' if remove_en_space else r'(?<![a-zA-Z])\s+(?![a-zA-Z])'
  return re.sub(pattern_space, '', text)


def write_data(wav_utts_list, data_dir):
  """生成data数据.

  Args:
      data_dir: 数据文件夹路径.
      wav_utts_list: wav和对应的utts列表
  """
  with (data_dir / "wav.scp").open("w", encoding="utf-8") as wav_scp_file, \
      (data_dir / "text").open("w", encoding="utf-8") as text_file, \
      (data_dir / "text.fmt").open("w", encoding="utf-8") as text_fmt_file:
    for wav, utts in wav_utts_list:
      for utt in utts:
        wav_scp_file.write(f"{utt.utt_id()} {wav.absolute()}\n")
        text_file.write(f"{utt.utt_id()} {format_text(utt.text)}\n")
        text_fmt_file.write(f"{utt.utt_id()} {format_text(utt.text, False)}\n")


def update_wav_utts(wav_utts_list, wav_dir: Path):
  """根据音频路径及utterances更新wav_utts列表.

  Args:
      wav_utts_list: 原始的wav和对应的utts的列表.
      wav_dir: 音频数据文件夹路径.

  Returns:
    更新后的wav和对应的utts的列表.
  """
  updated_wav_utts = []
  for _, utts in wav_utts_list:
    wav_utts = [(wav_dir / (utt.utt_id() + '.wav'), [utt]) for utt in utts]
    updated_wav_utts.extend(wav_utts)
  return updated_wav_utts


def gen_data_by_wav_utts(wav_utts,
                         wav_dir: Path,
                         data_dir: Path,
                         pad_length=0,
                         nj=32):
  """根据传入的wav_utts生成data数据.

  Args:
    wav_utts: wav和对应utts的列表.
    wav_dir: 音频文件夹.
    data_dir: 数据文件夹.
    nj: 任务数.
  """
  if wav_dir.exists():
    _LOGGER.info(f"{wav_dir} 已经存在, 如果需要重新生成请手动删除.")
  else:
    wav_dir.mkdir(parents=True)
    write_wav_multithread(wav_utts, wav_dir, pad_length, nj)

  if not data_dir.exists():
    data_dir.mkdir(parents=True)

  if (data_dir / "wav.scp").exists() and (data_dir / "text").exists():
    _LOGGER.info(f"{data_dir}下wav.scp和text已经存在, 如果需要重新生成请手动删除.")
  else:
    updated_wav_utts = update_wav_utts(wav_utts, wav_dir)
    write_data(updated_wav_utts, data_dir)


def gen_data_by_subsets(corpus_conf: dataset.CorpusConf,
                        subsets,
                        wav_dir: Path,
                        data_dir: Path,
                        bizs=None,
                        pad_length=0,
                        nj=32):
  """从数据库读取数据, 生成data数据.

  Args:
    corpus_conf: 语料库配置.
    subsets: 子集名称列表, 必须在AsrData表内.
    wav_dir: 音频文件夹.
    data_dir: 数据文件夹.
    bizs: 选取特定的业务, 默认None.
    nj: 任务数.
  """
  _LOGGER.info(f"{corpus_conf.db_conf.db_name} {subsets} to {data_dir}.")
  corpus = dataset.AsrCorpus(corpus_conf)
  wav_utts_list = corpus.get_wav_utts_list(subsets, bizs, filter_mty=True)
  gen_data_by_wav_utts(wav_utts_list, wav_dir, data_dir, pad_length, nj)


def __cmd():
  desc = "准备生成wenet raw格式数据."
  parser = ArgumentParser(description=desc, parents=[db_name_parser()])
  parser.add_argument("subsets", nargs="+", help="子集名称, 必须存在于AsrData表内.")
  parser.add_argument("wav_dir", type=Path, help="音频保存文件夹, 当路径存在时不会覆盖.")
  parser.add_argument("data_dir", type=Path, help="数据文件夹.")
  parser.add_argument("--biz", nargs="+", help="选取特定的业务, 支持多个.")
  parser.add_argument("--pad_length",
                      type=int,
                      default=0,
                      help="生成音频时尾部padding静音时长, 默认0.")
  parser.add_argument("--nj", type=int, default=32, help="线程数, 默认32.")
  args = parser.parse_args()

  corpus_conf = CorpusConf(DbConf(args.db_name))
  gen_data_by_subsets(corpus_conf,
                      args.subsets,
                      args.wav_dir,
                      args.data_dir,
                      bizs=args.biz,
                      pad_length=args.pad_length,
                      nj=args.nj)


if __name__ == '__main__':
  logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)
  __cmd()
