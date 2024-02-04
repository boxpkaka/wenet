from wenet.utils.init_model import init_model
from wenet.whisper.whisper import Whisper
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import check_modify_and_save_config

import yaml
import torch
import argparse
from transformers import WhisperModel


def get_encoder_from_wenet(args, configs):
    tokenizer = init_tokenizer(configs)
    configs['vocab_size'] = tokenizer.vocab_size()
    configs = check_modify_and_save_config(args, configs, tokenizer.symbol_table)

    model, configs = init_model(args, configs)
    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of parameters: {params / 1e6:.2f}M")
    whisper_encoder = model.encoder
    
    return whisper_encoder


def get_encoder_from_huggingface(
        model_dir:str, 
        device: torch.device, 
        dtype: torch.dtype
):
    model = WhisperModel.from_pretrained(model_dir).to(device).to(dtype)
    encoder = model.encoder
    return encoder


if __name__ == "__main__":
    encoder = get_encoder_from_wenet(model_dir='../whisper/exp/finetune_whisper_largev3_conv2d4_zh_yue50+zh50/final.pt',
                                                  device=torch.device('cuda:7'),
                                                  dtype=torch.bfloat16)
    

    inputs = torch.randn(2, 300, 128).cuda().to(torch.bfloat16).to(torch.device('cuda:7'))
    inputs_len = torch.tensor([[300], [300]]).to(torch.device('cuda:7'))

    xs, xs_mask = encoder(inputs, inputs_len)
    print(xs.shape)