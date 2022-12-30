#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by fangcheng on 2022/12/28
"""local模块."""
import logging
import yaml
from argparse import ArgumentParser
from pathlib import Path

from base_utils.utils import LOGGER_FORMAT


def parse_item(yaml_path, items):
  """获取配置文件参数值.
  Args:
     yaml_path: 配置文件路径.
     items: 列表, 需要读取的参数项, 需包含上层参数.

  Returns:
     参数值.
  """
  with open(yaml_path, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)

  for item in items:
    configs = configs[item]

  return configs


def __cmd():
  desc = "读取配置文件参数, 并打印具体的数值."
  parser = ArgumentParser(description=desc)
  parser.add_argument("yaml_path", type=Path, help="待读取的yaml配置文件路径.")
  parser.add_argument("item", nargs="+", help="需要获取的配置文件参数, 需包含上层参数.")
  args = parser.parse_args()
  item_value = parse_item(args.yaml_path, args.item)
  print(item_value)


if __name__ == '__main__':
  logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)
  __cmd()