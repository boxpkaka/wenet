#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by yuanding on 2022/08/02
"""_"""
from argparse import ArgumentParser
from pathlib import Path


# pylint: disable=too-many-instance-attributes
class WavConf:
  """textgrid以及wav相关参数解析类

  Attributes:
    wav_in_dir: 输入音频文件夹路径.
    wav_out_dir: 输出音频文件夹路径.
    sample_rate: 处理结束音频最后的采样率.
    sample_width: 处理结束音频最后的位深.
    wav_channels: 指定音频声道数, 如果音频真实声道数和指定值不符, 会报错.
  """

  def __init__(self, args):
    """初始化.

    Args:
      args: 参数命名空间, 存放textgrid以及wav相关参数
        (需要包含参数:
        wav_in_dir: 输入音频文件夹路径.
        wav_out_dir: 输出音频文件夹路径.
        sample_rate: 处理结束音频最后的采样率.
        sample_width: 处理结束音频最后的位深.
        wav_channels: 指定音频声道数, 如果音频真实声道数和指定值不符, 会报错.
    """

    self.wav_in_dir = args.wav_in_dir
    self.wav_out_dir = args.wav_out_dir
    self.sample_rate = args.sample_rate
    self.sample_width = args.sample_width
    self.wav_channels = args.wav_channels


# pylint: disable=too-many-instance-attributes
class TextGrigConf:
  """textgrid以及wav相关参数解析类

  Attributes:
    textgrid_dir: textgrid文件夹路径.
    paser_conf_path: textgird配置信息文件路径.
    textgrid_channel: textgrid文件对应声道信息.
    biz_name: 数据对应的业务名, 默认'selflearning'.
  """

  def __init__(self, args):
    """初始化.

    Args:
      args: 参数命名空间, 存放textgrid以及wav相关参数
        (需要包含参数:
        textgrid_dir: textgrid文件夹路径.
        paser_conf_path: textgird配置信息文件路径.
        textgrid_channel: textgrid文件对应声道信息.')
    """
    self.tg_dir = args.textgrid_dir
    self.paser_conf_path = args.paser_conf_path
    self.tg_channel = args.textgrid_channel


def get_wav_parser():
  """获取音频参数解析对象.

  Returns:
    音频参数解析对象.
  """
  parser = ArgumentParser(add_help=False)
  parser.add_argument("wav_in_dir", type=Path, help="输入音频文件夹路径.")
  parser.add_argument("wav_out_dir", type=Path, help="输出音频文件夹路径.")
  parser.add_argument("sample_rate", type=int, help="音频采样率.")
  parser.add_argument("sample_width", type=int, help="音频位深.")
  parser.add_argument("wav_channels", type=int, help="音频声道数.")
  return parser


def get_textgrid_parser():
  """获取textgrid标注参数解析对象.

  Returns:
    textgrid标注参数解析对象.
  """
  parser = ArgumentParser(add_help=False)
  parser.add_argument("textgrid_dir", type=Path, help="textgrid文件夹路径.")
  parser.add_argument("paser_conf_path", type=Path, help="textgird配置信息文件路径.")
  parser.add_argument("textgrid_channel", type=int, help="textgrid文件对应声道信息.")
  return parser


def get_parser():
  """获取参数解析对象.

  Returns:
    参数解析对象.
  """
  parser = ArgumentParser(add_help=False)
  parser.add_argument("data_dir", type=Path, help="生成的数据文件夹路径.")
  parser.add_argument("--need_rename", type=bool, default=False,
                      help="是否需要重命名, 默认选择否.")
  parser.add_argument("--business_name", type=str,
                      help="数据对应的业务名, 用于音频重命名, 默认'selflearning'",
                      default="selflearning")
  return parser
