from transformers import WhisperModel, WhisperProcessor

import torch
import numpy as np

model_dir = '/data1/yumingdong/model/huggingface/whisper-large-v3'
model = WhisperModel.from_pretrained(model_dir).cuda().to(torch.bfloat16)
processor = WhisperProcessor.from_pretrained(model_dir)

wav = np.random.randn(1, 38900)
features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.cuda().to(torch.bfloat16)
print(features.shape)
encoder = model.encoder

output = model.encoder(features)[0]
print(output.shape)




