#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by yuanding on 2022/08/09
"""local模块."""
from pathlib import Path
from argparse import ArgumentParser
from base_utils.text import get_sentformatter


def foramt_text(ori_text, format_text):
  """文本格式化

  Args:
     ori_text: 带格式化的文件.
     format_text: 格式化后的文件.
  """
  formatter = get_sentformatter()
  format_lines = []
  with ori_text.open("r", encoding="utf-8") as f_in:
    for line in f_in.readlines():
      text = formatter.format_without_filter_oovs(line)
      # 目前中文模型中对英文的处理, 需要转大写
      format_lines.append(text.upper().replace("{UM}", "呃"))
  with format_text.open("w", encoding="utf-8") as f_out:
    for line in format_lines:
      f_out.write(f"{line}\n")


def __cmd():
  desc = "对原始文本进行处理, 以便后续构建语言模型."
  parser = ArgumentParser(description=desc)
  parser.add_argument("ori_text", type=Path, help="待处理文本.")
  parser.add_argument("format_text", type=Path, help="处理后的文本.")
  args = parser.parse_args()
  foramt_text(args.ori_text, args.format_text)


if __name__ == '__main__':
  __cmd()
