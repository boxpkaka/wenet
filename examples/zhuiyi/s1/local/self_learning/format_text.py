#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by yuanding on 2022/08/09
"""local模块."""
import logging
import time
from argparse import ArgumentParser
from pathlib import Path

from base_utils.text import get_sentformatter
from base_utils.utils import LOGGER_FORMAT


def foramt_text(ori_text, format_text, dict_path, is_english=False,
                is_cantonese=False):
  """文本格式化

  Args:
     ori_text: 带格式化的文件.
     format_text: 格式化后的文件.
     dict_path: 分词使用的词典路径, 发音词典或者kaldi的words.txt.
     is_english: 是否是英语, 默认否.
     is_cantonese: 是否是粤语, 默认否.
  """
  formatter = get_sentformatter(dict_path=dict_path, is_english=is_english,
                                is_cantonese=is_cantonese)
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
  parser.add_argument("dict_path", type=Path,
                      help="分词使用的词典路径, 发音词典或者wenet模型文件夹下的\
                            lang_char.txt") # TODO(fangcheng): wenet模型文件夹
  parser.add_argument("--is_english", default=False, action="store_true",
                      help="是否是英语, 默认否.")
  parser.add_argument("--is_cantonese", default=False, action="store_true",
                      help="是否是粤语, 默认否.")
  args = parser.parse_args()

  logging.info(f"开始语料清洗.")
  begin_time = time.time()
  foramt_text(args.ori_text, args.format_text, args.dict_path, args.is_english,
              args.is_cantonese)
  logging.info(f"语料清洗完成.")
  logging.info(f"用时 {time.time()-begin_time}s.")


if __name__ == '__main__':
  logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)
  __cmd()
