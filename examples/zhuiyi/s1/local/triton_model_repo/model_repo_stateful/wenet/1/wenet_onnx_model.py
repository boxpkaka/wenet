# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import multiprocessing
import os

import numpy as np
import torch
import ctc_decoder
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


class WenetModel(object):

  def __init__(self, model_config, device):
    params = self.parse_model_parameters(model_config['parameters'])

    self.device = device
    print("Using device", device)
    print("Successfully load model !")

    # load sos and eos
    self.eos = self.sos = self.load_sos_eos(params["vocab_path"])

    # beam search setting
    self.beam_size = params.get("beam_size")

    num_processes = params["num_processes"]
    if params["num_processes"] == -1:
      num_processes = multiprocessing.cpu_count()

    self.decoder = ctc_decoder.CtcDecoder(params["vocab_path"],
                                          params["beam_size"], num_processes,
                                          params["cutoff_prob"])
    if os.path.exists(params["lm_path"]):
      self.decoder.LoadScorer(params["lm_path"], params["alpha"],
                              params["beta"])
      print("Successfully Load LM")

    if os.path.exists(params["context_path"]):
      self.decoder.LoadContextGraph(params["vocab_path"],
                                    params["context_path"],
                                    params["context_score"],
                                    params["incremental_context_score"],
                                    params["max_contexts"])
      print("Successfully Load ContextGraph")

    self.bidecoder = params.get('bidecoder')
    # rescore setting
    self.rescoring = params.get("rescoring", 0)
    print("Using rescoring:", bool(self.rescoring))
    print("Successfully load all parameters!")

    log_probs_config = pb_utils.get_input_config_by_name(
        model_config, "log_probs")
    # Convert Triton types to numpy types
    log_probs_dtype = pb_utils.triton_string_to_numpy(
        log_probs_config['data_type'])

    if log_probs_dtype == np.float32:
      self.dtype = torch.float32
    else:
      self.dtype = torch.float16

    self.prob_log = {}

  def generate_init_cache(self):
    encoder_out = None
    return encoder_out

  def load_sos_eos(self, vocab_file):
    sos_eos = None
    with open(vocab_file, "r", encoding="utf-8") as f:
      for line in f:
        char, idx = line.strip().split()
        if char == "<sos/eos>":
          sos_eos = int(idx)
      if not sos_eos:
        sos_eos = len(f.readlines()) - 1
    return sos_eos

  def parse_model_parameters(self, model_parameters):
    model_p = {
        "beam_size": 10,
        "cutoff_prob": 0.999,
        "vocab_path": None,
        "lm_path": None,
        "alpha": 2.0,
        "beta": 1.0,
        "context_path": None,
        "context_score": 3,
        "incremental_context_score": 0,
        "max_contexts": 5000,
        "rescoring": 0,
        "bidecoder": 1,
        "num_processes": -1,
    }
    # get parameter configurations
    for li in model_parameters.items():
      key, value = li
      true_value = value["string_value"]
      if key not in model_p:
        continue
      key_type = type(model_p[key])
      if key_type == type(None):
        model_p[key] = true_value
      else:
        model_p[key] = key_type(true_value)
    assert model_p["vocab_path"] is not None
    return model_p

  def infer(self, batch_log_probs, batch_log_probs_idx, seq_lens, rescore_index,
            batch_states):
    trie_vector, batch_start, batch_end, batch_encoder_hist, cur_encoder_out = batch_states
    score_hyps = self.batch_ctc_prefix_beam_search_cpu(batch_log_probs,
                                                       batch_log_probs_idx,
                                                       seq_lens, trie_vector,
                                                       batch_start, batch_end)

    if self.rescoring and len(rescore_index) != 0:
      # find the end of sequence
      rescore_encoder_hist = []
      rescore_encoder_lens = []
      rescore_hyps = []
      res_idx = list(rescore_index.keys())
      max_length = -1
      for idx in res_idx:
        hist_enc = batch_encoder_hist[idx]
        if hist_enc is None:
          cur_enc = cur_encoder_out[idx]
        else:
          cur_enc = torch.cat([hist_enc, cur_encoder_out[idx]], axis=0)
        rescore_encoder_hist.append(cur_enc)
        hist_enc_len = 0
        if hist_enc is not None:
          hist_enc_len = len(hist_enc)
        cur_mask_len = int(hist_enc_len + seq_lens[idx])
        rescore_encoder_lens.append(cur_mask_len)
        rescore_hyps.append(score_hyps[idx])
        if cur_enc.shape[0] > max_length:
          max_length = cur_enc.shape[0]
      best_index, score = self.batch_rescoring(rescore_hyps,
                                               rescore_encoder_hist,
                                               rescore_encoder_lens, max_length)

    best_sent = []
    best_score = []
    j = 0
    for idx, li in enumerate(score_hyps):
      best_idx = 0
      hyp_score = li[0][0]
      if idx in rescore_index and self.rescoring:
        best_idx = best_index[j]
        hyp_score = score[j]
        j += 1
      best_sent.append(li[best_idx][1])
      best_score.append(math.exp(hyp_score / max(len(li[best_idx][1]), 1)))

    final_result = self.decoder.MapSentBatch(best_sent)
    return final_result, best_score, cur_encoder_out

  def batch_ctc_prefix_beam_search_cpu(self, batch_log_probs_seq,
                                       batch_log_probs_idx, batch_len,
                                       batch_root, batch_start, batch_end):
    """
        Return: Batch x Beam_size elements, each element is a tuple
                (score, list of ids),
        """
    batch_len_list = batch_len
    batch_log_probs_seq_list = []
    batch_log_probs_idx_list = []
    for i in range(len(batch_len_list)):
      cur_len = int(batch_len_list[i])
      batch_log_probs_seq_list.append(
          batch_log_probs_seq[i][0:cur_len].tolist())
      batch_log_probs_idx_list.append(
          batch_log_probs_idx[i][0:cur_len].tolist())
    score_hyps = self.decoder.BeamSearchBatch(batch_log_probs_seq_list,
                                              batch_log_probs_idx_list,
                                              batch_root, batch_start,
                                              batch_end)
    return score_hyps

  def batch_rescoring(self, score_hyps, hist_enc, hist_mask_len, max_len):
    """
        score_hyps: [((ctc_score, (id1, id2, id3, ....)), (), ...), ....]
        hist_enc: [len1xF, len2xF, .....]
        hist_mask: [1x1xlen1, 1x1xlen2]
        return bzx1  best_index
        """
    bz = len(hist_enc)
    f = hist_enc[0].shape[-1]
    beam_size = self.beam_size
    encoder_lens = np.zeros((bz, 1), dtype=np.int32)
    encoder_out = torch.zeros((bz, max_len, f), dtype=self.dtype)
    hyps = []
    ctc_score = torch.zeros((bz, beam_size), dtype=self.dtype)
    max_seq_len = 0
    for i in range(bz):
      cur_len = hist_enc[i].shape[0]
      encoder_out[i, 0:cur_len] = hist_enc[i]
      encoder_lens[i, 0] = hist_mask_len[i]

      # process candidate
      if len(score_hyps[i]) < beam_size:
        to_append = (beam_size - len(score_hyps[i])) * [(-10000, ())]
        score_hyps[i] = list(score_hyps[i]) + to_append
      for idx, c in enumerate(score_hyps[i]):
        score, idlist = c
        if score < -10000:
          score = -10000
        ctc_score[i][idx] = score
        hyps.append(list(idlist))
        if len(hyps[-1]) > max_seq_len:
          max_seq_len = len(hyps[-1])

    max_seq_len += 2
    hyps_pad_sos_eos = np.ones((bz, beam_size, max_seq_len), dtype=np.int64)
    hyps_pad_sos_eos = hyps_pad_sos_eos * self.eos  # fill eos
    if self.bidecoder:
      r_hyps_pad_sos_eos = np.ones((bz, beam_size, max_seq_len), dtype=np.int64)
      r_hyps_pad_sos_eos = r_hyps_pad_sos_eos * self.eos

    hyps_lens_sos = np.ones((bz, beam_size), dtype=np.int32)
    bz_id = 0
    for idx, cand in enumerate(hyps):
      bz_id = idx // beam_size
      length = len(cand) + 2
      bz_offset = idx % beam_size
      pad_cand = [self.sos] + cand + [self.eos]
      hyps_pad_sos_eos[bz_id][bz_offset][0:length] = pad_cand
      if self.bidecoder:
        r_pad_cand = [self.sos] + cand[::-1] + [self.eos]
        r_hyps_pad_sos_eos[bz_id][bz_offset][0:length] = r_pad_cand
      hyps_lens_sos[bz_id][idx % beam_size] = len(cand) + 1
    in0 = pb_utils.Tensor.from_dlpack("encoder_out", to_dlpack(encoder_out))
    in1 = pb_utils.Tensor("encoder_out_lens", encoder_lens)
    in2 = pb_utils.Tensor("hyps_pad_sos_eos", hyps_pad_sos_eos)
    in3 = pb_utils.Tensor("hyps_lens_sos", hyps_lens_sos)
    input_tensors = [in0, in1, in2, in3]
    if self.bidecoder:
      in4 = pb_utils.Tensor("r_hyps_pad_sos_eos", r_hyps_pad_sos_eos)
      input_tensors.append(in4)
    in5 = pb_utils.Tensor.from_dlpack("ctc_score", to_dlpack(ctc_score))
    input_tensors.append(in5)
    request = pb_utils.InferenceRequest(
        model_name='decoder',
        requested_output_names=['best_index', 'score'],
        inputs=input_tensors)
    response = request.exec()
    best_index = pb_utils.get_output_tensor_by_name(response, 'best_index')
    best_index = from_dlpack(best_index.to_dlpack()).clone()
    best_index = best_index.numpy()[:, 0]

    score = pb_utils.get_output_tensor_by_name(response, 'score')
    score = from_dlpack(score.to_dlpack()).clone()
    score = score.numpy()[:, 0]
    return best_index, score

  def __del__(self):
    print("remove wenet model")
