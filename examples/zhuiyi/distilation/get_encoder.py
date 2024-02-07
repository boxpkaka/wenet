from numpy import dtype
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import check_modify_and_save_config

import torch
import torch.nn as nn
from transformers import WhisperModel, AutoProcessor, HubertModel



class HuggingfaceWhisperEncoder(nn.Module):
    def __init__(self, args, configs=None):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(args.checkpoint).encoder
        self.processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    
    # NOTE(yumingdong): Processor only receives audio which samplerate=16k
    def forward(self, batch_dict, device, dtype, middle_layer=-1):
        wav = batch_dict['pcm'].squeeze(0)
        features = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        features = features.to(device).to(dtype)
        output = self.encoder(features, output_hidden_states=True)['hidden_states']
        return output[middle_layer]


class WenetWhisperEncoder(nn.Module):
    def __init__(self, args, configs):
        super().__init__()
        self.encoder = self.get_encoder_from_wenet(args, configs)
    
    def forward(self, batch_dict, device, dtype, middle_layer=-1):
        xs = batch_dict['feats'].to(device).to(dtype)
        xs_lens = batch_dict['feats_lengths'].to(device).to(dtype)
        output = self.encoder(xs, xs_lens, middle_layer=middle_layer)[0]
        return output
    
    @staticmethod
    def get_encoder_from_wenet(args, configs):
        tokenizer = init_tokenizer(configs)
        configs['vocab_size'] = tokenizer.vocab_size()
        configs = check_modify_and_save_config(args, configs, tokenizer.symbol_table)
        model, configs = init_model(args, configs)
        # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"Number of parameters: {params / 1e6:.2f}M")
        whisper_encoder = model.encoder
        return whisper_encoder


class HuBERT(nn.Module):
    def __init__(self, args=None, configs=None):
        super().__init__()
        self.model = HubertModel.from_pretrained('/data1/yumingdong/model/huggingface/hubert-chinese-large')

    # Example: pass 'layer=12' to get the 12th encoder output
    def forward(self, batch_dict, device, dtype, middle_layer=-1):
        wav = batch_dict['pcm'].to(device).to(dtype)
        output = self.model(wav, output_hidden_states=True)['hidden_states']
        return output[middle_layer]
        
        
ENCODER_TYPE = {
    'wenet-whisper':       WenetWhisperEncoder,
    'huggingface-whisper': HuggingfaceWhisperEncoder,
    'hubert':              HuBERT
}

def get_encoder(args, configs):
    encoder = ENCODER_TYPE[args.encoder_type](args, configs)
    return encoder


if __name__ == "__main__":
    device = torch.device('cuda')
    dtype = torch.float32
    model = HuBERT().to(device).to(dtype)
    inputs = {'pcm': torch.randn(1, 16000)}
    output = model(inputs, device, dtype, layer=13)
    print(output.shape)