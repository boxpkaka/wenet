from numpy import dtype
from wenet.utils.init_model import init_model
from wenet.whisper.whisper import Whisper
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import check_modify_and_save_config

import torch
from transformers import WhisperModel, AutoProcessor


class HuggingfaceWhisperEncoder(torch.nn.Module):
    def __init__(self, args, configs=None):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(args.checkpoint).encoder
        self.processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    
    def forward(self, batch_dict, device, dtype):
        wav = batch_dict['pcm'].squeeze(0)
        features = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        features = features.to(device).to(dtype)
        output = self.encoder(features)['last_hidden_state']
        return output


class WenetWhisperEncoder(torch.nn.Module):
    def __init__(self, args, configs):
        super().__init__()
        self.encoder = self.get_encoder_from_wenet(args, configs)
    
    def forward(self, batch_dict, device, dtype):
        xs = batch_dict['feats'].to(device).to(dtype)
        xs_lens = batch_dict['feats_lengths'].to(device).to(dtype)
        output = self.encoder(xs, xs_lens)[0]
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


ENCODER_TYPE = {
    'wenet': WenetWhisperEncoder,
    'huggingface': HuggingfaceWhisperEncoder
}

def get_encoder(args, configs):
    encoder = ENCODER_TYPE[args.encoder_type](args, configs)
    return encoder


if __name__ == "__main__":
    pass
    