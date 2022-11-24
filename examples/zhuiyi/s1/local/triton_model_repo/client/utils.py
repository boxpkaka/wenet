import numpy as np


def _levenshtein_distance(ref, hyp):
  """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
  m = len(ref)
  n = len(hyp)

  # special case
  if ref == hyp:
    return 0
  if m == 0:
    return n
  if n == 0:
    return m

  if m < n:
    ref, hyp = hyp, ref
    m, n = n, m

  # use O(min(m, n)) space
  distance = np.zeros((2, n + 1), dtype=np.int32)

  # initialize distance matrix
  for j in range(n + 1):
    distance[0][j] = j

  # calculate levenshtein distance
  for i in range(1, m + 1):
    prev_row_idx = (i - 1) % 2
    cur_row_idx = i % 2
    distance[cur_row_idx][0] = i
    for j in range(1, n + 1):
      if ref[i - 1] == hyp[j - 1]:
        distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
      else:
        s_num = distance[prev_row_idx][j - 1] + 1
        i_num = distance[cur_row_idx][j - 1] + 1
        d_num = distance[prev_row_idx][j] + 1
        distance[cur_row_idx][j] = min(s_num, i_num, d_num)

  return distance[m % 2][n]


def cal_cer(references, predictions):
  errors = 0
  lengths = 0
  for ref, pred in zip(references, predictions):
    cur_ref = list(ref)
    cur_hyp = list(pred)
    cur_error = _levenshtein_distance(cur_ref, cur_hyp)
    errors += cur_error
    lengths += len(cur_ref)
  return float(errors) / lengths


def value_counts(values, vad_thr, level, difference):
  """根据vad阈值和需要统计的级别返回统计信息

  因为一般情况，延迟最少为一个vad阈值, 特殊情况会小于一个vad阈值，如音频结束。
  eg. vad_thr=350, level=3, difference=100时
  统计区间为[0-350, 350-450, 450+],
  为方便，区间左边界和右边界采用和中间一样的处理方式，所以此处返回值为[250-350, 350-450, 450-550]

  Args:
    values: 待统计延迟值列表
    vad_thr: vad阈值, 单位ms
    level: 统计级数
    difference: 每一级差值

  Returns:
    经过格式化的统计信息字符串
  """
  level_data = [0] * level
  for latency in values:
    index = min(max(0, ((latency - vad_thr) // difference) + 1), level - 1)
    level_data[index] += 1
  latency_level = list()
  for per_level in range(level):
    left = vad_thr + (per_level - 1) * 100
    latency_level.append((left, left + 100))
  result = ""
  for per_level, num in zip(latency_level, level_data):
    result += f"{per_level[0]} ms-{per_level[1]} ms : {num}\n"
  return result


def get_statistical_result(data_list, need_min=False):
  """根据列表获得数据数量, 平均值, 最大值

  Args:
    data_list: 数据列表
    need_min: 是否需要统计最小值, 为了输出信息简洁, 默认否.

  Returns:
    经过格式化的统计信息字符串, 如果待统计数据为空，则返回False.
  """
  if len(data_list) == 0:
    return False
  data_max = round(max(data_list), 2)
  data_num = len(data_list)
  data_avg = round(sum(data_list) / data_num, 2)
  if need_min:
    data_min = round(min(data_list), 2)
    result = f"num:{data_num}, min:{data_min}, max:{data_max}, avg:{data_avg}"
  else:
    result = f"num:{data_num}, max:{data_max}, avg:{data_avg}"
  return result


def print_performance(performence):
  """统计性能数据并打印

  Args:
    performence: 性能数据
  """
  performence = [item for sublist in performence for item in sublist]
  mid_latency = [item for sublist in performence for item in sublist[0]]
  last_latency = [item for sublist in performence for item in sublist[1]]
  compute_cost = [sublist[2] for sublist in performence]
  wavs_duration = [sublist[3] for sublist in performence]

  over_time_num = sum([latency - 300 > 0 for latency in last_latency])
  over_time_rate = round(over_time_num / len(last_latency), 4)
  # 统计rtf
  rtf = round(sum(compute_cost) / sum(wavs_duration), 4)

  result = (f"RTF:{rtf}, 中间片延迟: {get_statistical_result(mid_latency)}, "
            f"尾片延迟: {get_statistical_result(last_latency)}, "
            f"overtime_num:{over_time_num}, overtime_rate:{over_time_rate}")
  print(result)
  print(value_counts(last_latency, 0, 6, 100))
