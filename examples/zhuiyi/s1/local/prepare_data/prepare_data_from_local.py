#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by yuanding on 2022/08/02
"""不经过数据库, 利用本地的wav文件夹和textgrid文件夹, 生成wenet数据."""
import datetime
import logging
from argparse import ArgumentParser
from pathlib import Path
import random

from base_utils import dataset
from base_utils.dataset import textgrid_parser, wav_parser
from base_utils.dataset.utils import (local_data_insert_many,
                                      local_wav_insert_many)
from base_utils.text import solve_with_oov
from base_utils.utils import LOGGER_FORMAT

from ..prepare_data_for_raw import gen_data_by_wav_utts
from .utils import (TextGrigConf, WavConf, get_parser, get_textgrid_parser,
                    get_wav_parser)

_LOGGER = logging.getLogger(__file__)


def process_text_with_wav(localdb, textgrid_conf, wav_out_dir, out_map_path):
  """textgrid文本规范检查及文本音频一致性检查，oov集外词会输出oov报告.
     最终的数据信息会存储至本地模拟数据库(localdb)的data表.

  Args:
    localdb: 本地模拟数据库, 存储和读取wav表，data表以及business表的数据.
    textgrid_conf: textgrid标注文件相关的配置.
    wav_out_dir: 输出音频文件夹路径.
    out_map_path: 音频输出映射文件.

  Returns:
    oov_obj: OovReport对象.
  """
  local_tp = textgrid_parser.LocalTextParser(textgrid_conf.paser_conf_path,
                                             localdb)

  tg_data_list, wav_files = local_tp.collect_textgrid_wav_info(
      textgrid_conf.tg_dir, textgrid_conf.tg_channel, wav_out_dir, out_map_path)

  wav_id_list, wav_num_chan_list = local_tp.check_textgrid_with_wav(
      tg_data_list, textgrid_conf.tg_channel, wav_files)

  all_data_list, oov_obj = textgrid_parser.get_data_list_with_oov_check(
      tg_data_list, wav_id_list, textgrid_conf.tg_channel, wav_num_chan_list,
      False, False)
  if oov_obj.has_oov():
    oov_file = Path(f"{textgrid_conf.tg_dir.name}.oov")
    _LOGGER.error(f"标注数据存在OOV, OOV信息: {textgrid_conf.tg_dir.name}.oov.")
    oov_obj.write(oov_file)
  else:
    local_data_insert_many(all_data_list, localdb)

  return oov_obj


def process_wav(localdb, wav_conf, biz_name, out_map_path, need_rename, nj):
  """音频格式检查及转码.
     音频默认进行重命名, 并生成对应的音频名-音频文件路径映射.
     最终的音频信息列表会存储至本地模拟数据库(localdb)的wav表.

  Args:
    localdb: 本地模拟数据库, 存储和读取wav表，data表以及business表的数据.
    wav_conf: 音频数据相关的配置.
    biz_name: 数据对应的业务名, 音频重命名时会基于业务名进行.
    out_map_path: 音频输出映射文件.
    need_rename: 是否需要对音频文件重命名.
    nj: 并发数量.
  """
  local_wav_paser = wav_parser.LocalWavParser(localdb)

  wav_in_out_paths = local_wav_paser.get_in_out_paths(wav_conf.wav_in_dir,
                                                      wav_conf.wav_out_dir,
                                                      biz_name,
                                                      wav_conf.sample_rate,
                                                      need_rename)

  wav_out_paths = wav_parser.format_wavs(wav_in_out_paths,
                                         wav_conf.wav_channels,
                                         wav_conf.sample_width,
                                         wav_conf.sample_rate,
                                         pools=nj)

  biz_id = local_wav_paser.get_biz_id_by_biz_name(biz_name)
  final_wav_paths, wav_data_list = local_wav_paser.filter_wav_paths(
      wav_out_paths, wav_conf.wav_channels, wav_conf.sample_rate, biz_id)

  wav_parser.write_map_file(wav_in_out_paths, final_wav_paths, out_map_path)

  local_wav_insert_many(wav_data_list, localdb)


def gen_local_db(wav_conf, textgrid_conf, biz_name, need_rename=False, nj=16):
  """处理本地标注文件和音频文件，并生成本地数据库.

  Args:
    wav_conf: 音频数据相关的配置.
    textgrid_conf: textgrid标注文件相关的配置.
    biz_name: 业务名.
    need_rename: 是否需要对音频文件重命名, 默认否.
    nj: 并发数量.

  Returns:
    localdb: LocalDB对象.
  """

  _LOGGER.info(f"标注文件: {textgrid_conf.tg_dir} and 音频文件:"
               f"{wav_conf.wav_in_dir} to "
               f"{wav_conf.wav_out_dir}.")

  # 音频文件重命名时, 基于时间戳及业务名定义映射文件
  time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  out_map_path = Path(biz_name + "_" + str(time_stamp) + ".txt")
  logging.info(f"音频名-音频文件路径映射存至 {out_map_path} ")

  localdb = dataset.local_database.LocalDB()

  # 音频格式检查及转码
  process_wav(localdb, wav_conf, biz_name, out_map_path, need_rename, nj)

  # 文本规范检查及文本音频一致性检查，oov词会输出oov_report
  oov_obj = process_text_with_wav(localdb, textgrid_conf, wav_conf.wav_out_dir,
                                  out_map_path)
  if oov_obj.has_oov():
    _LOGGER.info("OOV自动处理中, 默认覆盖原始标注文本")
    solve_with_oov(textgrid_conf.tg_dir,
                   Path(f"{textgrid_conf.tg_dir.name}.oov"))
    oov_obj = process_text_with_wav(localdb, textgrid_conf,
                                    wav_conf.wav_out_dir, out_map_path)
    if oov_obj.has_oov():
      _LOGGER.error(f"OOV处理失败, 请手动处理OOV, OOV信息: "
                    f"{textgrid_conf.tg_dir.name}.oov.")
    else:
      _LOGGER.info("OOV自动处理已完成, 标注文本可能存在空值")

  return localdb


def gen_wenet_data(data_dir: Path, wav_conf, textgrid_conf, biz_name,
                   need_rename, dev_splits, nj=16):
  """生成wenet需要的数据.

  Args:
      data_dir: 数据文件夹
      wav_conf: 音频数据相关的配置.
      textgrid_conf: textgrid标注文件相关的配置.
      biz_name: 业务名.
      need_rename: 是否需要重命名, 默认否.
      dev_splits: 验证集划分比例.
      nj: 并发数, 默认16.
  """

  localdb = gen_local_db(wav_conf, textgrid_conf, biz_name, need_rename)

  corpus = dataset.LocalCorpus(localdb, textgrid_conf.tg_channel)

  wav_utts_list = corpus.get_wav_utts_list(
      filter_mty=True, aim_channel=(textgrid_conf.tg_channel,))

  random.seed(777)
  random.shuffle(wav_utts_list)
  split = min(int(len(wav_utts_list) * dev_splits), 1000)
  train = wav_utts_list[:-split]
  dev = wav_utts_list[-split:]

  gen_data_by_wav_utts(train, data_dir / "wavs" / "train", data_dir / "train",
                       nj=nj)
  gen_data_by_wav_utts(dev, data_dir / "wavs" / "dev", data_dir / "dev", nj=nj)


def __cmd():
  desc = "利用本地的音频文件夹和标注文件文件夹生成wenet格式数据."
  parser = ArgumentParser(
      description=desc,
      parents=[get_parser(), get_wav_parser(), get_textgrid_parser()])
  parser.add_argument("--dev_splits", type=float, default=0.05,
                      help="验证集划分比例, 默认0.05.")
  parser.add_argument("--nj", type=int, default=16, help="线程数, 默认16.")
  args = parser.parse_args()

  gen_wenet_data(args.data_dir, WavConf(args), TextGrigConf(args),
                 biz_name=args.business_name, need_rename=args.need_rename,
                 dev_splits=args.dev_splits, nj=args.nj)


if __name__ == '__main__':
  logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)
  __cmd()
