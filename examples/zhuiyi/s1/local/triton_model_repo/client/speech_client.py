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

from tritonclient.utils import np_to_triton_dtype
import numpy as np
import math
import soundfile as sf
import time
import random

class OfflineSpeechClient(object):
    def __init__(self, triton_client, model_name, protocol_client, args):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name

    def recognize(self, wav_file, idx=0):
        waveform, sample_rate = sf.read(wav_file)
        samples = np.array([waveform], dtype=np.float32)
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        # better pad waveform to nearest length here
        target_seconds = math.ceil(len(waveform) / sample_rate)
        target_samples = np.zeros([1, target_seconds  * sample_rate], dtype=np.float32)
        target_samples[0][0: len(waveform)] = waveform
        samples = target_samples

        sequence_id = 10086 + idx
        result = ''
        inputs = [
            self.protocol_client.InferInput("WAV", samples.shape,
                                            np_to_triton_dtype(samples.dtype)),
            self.protocol_client.InferInput("WAV_LENS", lengths.shape,
                                            np_to_triton_dtype(lengths.dtype))
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        outputs = [self.protocol_client.InferRequestedOutput("TRANSCRIPTS"),
                   self.protocol_client.InferRequestedOutput("CONFIDENCE")]
        response = self.triton_client.infer(self.model_name,
                                            inputs,
                                            request_id=str(sequence_id),
                                            outputs=outputs)
        result = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
        score = response.as_numpy("CONFIDENCE")
        return [result, score]


class StreamingSpeechClient(object):
    def __init__(self, triton_client, model_name, protocol_client, args):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name
        chunk_size = args.chunk_size
        subsampling = args.subsampling
        context = args.context
        frame_shift_ms = args.frame_shift_ms
        frame_length_ms = args.frame_length_ms
        # for the first chunk
        # we need additional frames to generate
        # the exact first chunk length frames
        # since the subsampling will look ahead several frames
        first_chunk_length = (chunk_size - 1) * subsampling + context
        add_frames = math.ceil((frame_length_ms - frame_shift_ms) / frame_shift_ms)
        first_chunk_ms = (first_chunk_length + add_frames) * frame_shift_ms
        other_chunk_ms = chunk_size * subsampling * frame_shift_ms
        self.first_chunk_in_secs = first_chunk_ms / 1000
        self.other_chunk_in_secs = other_chunk_ms / 1000

    def recognize(self, wav_file, idx=0):
        waveform, sample_rate = sf.read(wav_file)
        wav_segs = []
        i = 0
        # 中间片数据, 尾片数据, 计算总耗时, 音频时长
        perf_metrics = [[], [], 0, len(waveform) / 8000]
        while i < len(waveform):
            if i == 0:
                stride = int(self.first_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[i: i + stride])
            else:
                stride = int(self.other_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[i: i + stride])
            i += len(wav_segs[-1])

        sequence_id = idx + 10086
        # time.sleep(random.uniform(0, 1))
        # simulate streaming
        for idx, seg in enumerate(wav_segs):
            # print(f"send {wav_file}: {idx} chunk, id: {sequence_id}")
            chunk_len = len(seg)
            if idx == 0:
                chunk_samples = int(self.first_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)
            else:
                chunk_samples = int(self.other_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)

            expect_input[0][0:chunk_len] = seg
            input0_data = expect_input

            input1_data = np.array([[chunk_len]], dtype=np.int32)

            inputs = [
                self.protocol_client.InferInput("WAV", input0_data.shape,
                                                np_to_triton_dtype(input0_data.dtype)),
                self.protocol_client.InferInput("WAV_LENS", input1_data.shape,
                                                np_to_triton_dtype(input1_data.dtype))]

            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            outputs = [self.protocol_client.InferRequestedOutput("TRANSCRIPTS"),
                       self.protocol_client.InferRequestedOutput("CONFIDENCE"),]
            end = False
            if idx == len(wav_segs) - 1:
                end = True

            # time.sleep(chunk_len / sample_rate)
            # start_time = time.time()
            response = self.triton_client.infer(self.model_name,
                                                inputs,
                                                outputs=outputs,
                                                sequence_id=sequence_id,
                                                sequence_start=idx == 0,
                                                sequence_end=end)

            # end_time = time.time()
            # compute_cost = end_time - start_time
            compute_cost = 1
            perf_metrics[2] += compute_cost
            if not end:
                perf_metrics[0].append(compute_cost)
                # if compute_cost > 0.3:
                    # print(f"{wav_file.split('/')[-1]} 中间片超时: {compute_cost}")
            else:
                perf_metrics[1].append(compute_cost)
                # time.sleep(2)
                # if compute_cost > 0.3:
                    # print(f"{wav_file.split('/')[-1]} 尾片超时: {compute_cost}")

            idx += 1
            result = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
            score = response.as_numpy("CONFIDENCE")
            # print("Get response from {}th chunk: {}".format(idx, result))
        return [result, score, perf_metrics]
