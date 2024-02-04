# Copyright (c) 2023 Wenet Community. (authors: Xingchen Song)
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
#
# Modified from [Whisper](https://github.com/openai/whisper)

import torch

from typing import Dict, List, Optional, Tuple

from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.utils.common import IGNORE_ID, add_whisper_tokens, add_whisper_tokens_multi_language, th_accuracy
from wenet.transformer.search import (ctc_greedy_search,
                                      ctc_prefix_beam_search,
                                      attention_beam_search,
                                      attention_rescoring, DecodeResult)
from wenet.utils.context_graph import ContextGraph


class Whisper(ASRModel):

    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC = None,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: dict = None,
    ):
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight,
                         ignore_id, reverse_weight, lsm_weight,
                         length_normalized_loss, special_tokens)
        assert reverse_weight == 0.0
        self.sos = special_tokens["sot"]
        self.eos = special_tokens["eot"]

    # TODO(xcsong): time align
    def set_alignment_heads(self, dump: bytes):
        raise NotImplementedError

    @property
    def is_multilingual(self):
        return self.vocab_size >= 51865

    @property
    def num_languages(self):
        return self.vocab_size - 51765 - int(self.is_multilingual)

    @torch.jit.ignore(drop=True)
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                           text_lengths)
        else:
            loss_ctc, ctc_probs = None, None

        # 2b. Attention-decoder branch
        # use non blank (token level) embedding for decoder
        if self.apply_non_blank_embedding:
            assert self.ctc_weight != 0
            assert ctc_probs is not None
            encoder_out, encoder_mask = self.filter_blank_embedding(
                ctc_probs, encoder_out)
            
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths, batch['language'])
        else:
            loss_att = None
            acc_att = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "th_accuracy": acc_att,
        }
    
    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        language: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        # TODO(xcsong): add args for no_timestamp, language, etc

        prev_len = ys_pad.size(1)
        ys_in_pad, ys_out_pad = add_whisper_tokens_multi_language(self.special_tokens,
                                                                    ys_pad,
                                                                    self.ignore_id,
                                                                    task="transcribe",
                                                                    no_timestamp=True,
                                                                    language=language,
                                                                    use_prev=False,
                                                                    )
        cur_len = ys_in_pad.size(1)
        ys_in_lens = ys_pad_lens + cur_len - prev_len
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    
if __name__ == "__main__":
    encoder = TransformerEncoder(
        input_size=80
    )
    decoder = TransformerDecoder(
        vocab_size=32,
        encoder_output_size=80,
    )
    special_tokens = {'eot': 50258,
                    'no_speech': 50363,
                    'no_timestamps': 50364,
                    'sot': 50258,
                    'sot_prev': 50362,
                    'timestamp_begin': 50365,
                    'transcribe': 50360,
                    'translate': 50359
                    }
    model = Whisper(
        vocab_size=32,
        encoder=encoder,
        decoder=decoder,
        special_tokens=special_tokens 
    )
    ys_no_language = torch.tensor([
        [91,   2415, 237, 12136, 103, 103],
        [91,   9455, 2131,  -1,    -1, -1],
        [1654, 2131, 2131, -1,    -1,  -1],
        [6404, 2131, -1,    -1,    -1, -1],
        [2131, -1,   -1,    -1,    -1, -1]
        ])
    

    ys_language = torch.tensor([
        [50352,   91,   2415, 237, 12136, 103, 103],
        [50353,  91,   9455, 2131,  -1,    -1, -1],
        [50354, 1654, 2131, 2131, -1,    -1,  -1],
        [50355, 6404, 2131, -1,    -1,    -1, -1],
        [50358, 2131, -1,   -1,    -1,    -1, -1]
        ])
    ys_pad_lens = torch.tensor([
        [6],[3],[3],[2],[1]
    ])
    encoder_mask = torch.tensor([
        [True,True,True,True,True,True],
        [True,True,True,False,False,False],
        [True,True,True,False,False,False],
        [True,True,False,False,False,False],
        [True,False,False,False,False,False],
    ]).unsqueeze(1)
    loss_att, acc_att = model._calc_att_loss(
        encoder_out=torch.randn(5, 230, 80),
        encoder_mask=encoder_mask,
        ys_pad=ys_no_language,
        ys_pad_lens=ys_pad_lens,
    )

