from wenet.utils.train_utils import init_dataset_and_dataloader
from wenet.utils.init_tokenizer import init_tokenizer
from examples.aishell.distilation.get_encoder_from_wenet import get_whisper_encoder_from_checkpoint

import torch
import yaml
import argparse
import h5py
import numpy as np


def get_args_configs(data_path: str, config_path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--train_data', type=str, default=data_path)
    parser.add_argument('--cv_data', type=str, default='/data2/yumingdong/data/raw/wenet/test_1000Cantonese/data.list')
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch', type=int, default=500)
    parser.add_argument('--jit', type=bool, default=False)
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
        
    return args, configs

def main():
    model_path = '/data1/yumingdong/wenet/examples/aishell/whisper/exp/finetune_whisper_largev3_conv2d4_20240119_yue_yue100_from_epoch_15/final.pt'
    save_path = './data/yue100_for_quantizer_training.h5'
    data_path = '/data2/yumingdong/data/raw/wenet_data_list/yue_100h+zh_100h/data.list'
    config_path = './conf/batchsize_1.yaml'
    device = torch.device('cuda:0')
    dtype = torch.float16
    num_audio = 2000

    args, configs = get_args_configs(data_path, config_path)
    tokenizer = init_tokenizer(configs)
    _, _, train_data_loader, _ = init_dataset_and_dataloader(args, configs, tokenizer)
    encoder = get_whisper_encoder_from_checkpoint(model_dir=model_path, device=device, dtype=dtype)

    with h5py.File(save_path, 'w') as hf:
        cnt = 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                '''
                keys, feats, target
                feats_lengths, target_lengths
                pcm, pcm_length
                '''
                dataset_name = batch_dict['keys'][0]
                xs = batch_dict['feats'].to(device).to(dtype)
                xs_lens = batch_dict['feats_lengths'].to(device).to(dtype)
                
                encoder_embedding = encoder(xs, xs_lens)[0]
                encoder_embedding = encoder_embedding.unsqueeze(0)  # (time, dim=1280)
                numpy_array = encoder_embedding.cpu().numpy().astype(np.float16)
                hf.create_dataset(dataset_name, data=numpy_array)
                cnt += 1
                print(f'[{cnt}/{num_audio}]')
                if cnt > num_audio:
                    break
    print('done')

if __name__ == "__main__":
    main()

