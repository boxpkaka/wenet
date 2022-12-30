#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by fangcheng on 2022/12/28
"""local模块."""
import logging
import onnx
from argparse import ArgumentParser
from pathlib import Path

from base_utils.utils import LOGGER_FORMAT


def verify_quant(onnx_model):
  """验证模型是否为量化模型, 并返回验证结果.
  Args:
     onnx_model: onnx模型路径.

  Returns:
     验证结果.
  """
  rlt_verify = False
  for weight in onnx.load(onnx_model).graph.initializer:
    if "quantized" in weight.name:
      rlt_verify = True
      break

  return rlt_verify

def __cmd():
  desc = "验证onnx模型是否为量化模型, 是则打印True, 否则打印False."
  parser = ArgumentParser(description=desc)
  parser.add_argument("onnx_model", type=Path, help="待验证的onnx模型路径.")
  args = parser.parse_args()
  verify_rlt = verify_quant(args.onnx_model)
  print(f"is_quant={verify_rlt}")


if __name__ == '__main__':
  logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)
  __cmd()
