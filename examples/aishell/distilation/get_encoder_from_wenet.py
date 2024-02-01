from wenet.utils.init_model import init_model
from wenet.whisper.whisper import Whisper
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import check_modify_and_save_config

import yaml
import torch
import argparse


def get_whisper_encoder_from_checkpoint(
        model_dir:str, 
        device: torch.device, 
        dtype: torch.dtype
        ):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=model_dir)
    parser.add_argument('--deepspeed_config', type=str, default='../whisper/conf/ds_stage1.json')
    parser.add_argument('--train_engine', type=str, default='deepspeed')
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--model_dir', type=str, default='./')
    parser.add_argument('--save_states', type=str, default='only model')
    parser.add_argument('--jit', type=bool, default=False)
    args = parser.parse_args()

    config_path = './conf/wenet_whisper.yaml'
    with open(config_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    tokenizer = init_tokenizer(configs)
    configs['vocab_size'] = tokenizer.vocab_size()
    configs = check_modify_and_save_config(args, configs, tokenizer.symbol_table)

    model, configs = init_model(args, configs)
    model = model.to(device).to(dtype)

    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of parameters: {params / 1e6:.2f}M")
    whisper_encoder = model.encoder
    
    return whisper_encoder


if __name__ == "__main__":
    encoder = get_whisper_encoder_from_checkpoint(model_dir='../whisper/exp/finetune_whisper_largev3_conv2d4_zh_yue50+zh50/final.pt',
                                                  device=torch.device('cuda:7'),
                                                  dtype=torch.bfloat16)
    

    inputs = torch.randn(2, 300, 128).cuda().to(torch.bfloat16).to(torch.device('cuda:7'))
    inputs_len = torch.tensor([[300], [300]]).to(torch.device('cuda:7'))

    xs, xs_mask = encoder(inputs, inputs_len)
    print(xs.shape)