from multi_quantization import Quantizer, QuantizerTrainer
from wenet.utils.train_utils import init_dataset_and_dataloader
from wenet.utils.init_tokenizer import init_tokenizer
from get_encoder_from_wenet import get_whisper_encoder_from_checkpoint

import torch
import yaml
import argparse
import h5py
from tqdm import tqdm


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
    model_path = '/data1/yumingdong/offical/wenet/examples/aishell/whisper/exp/finetune_whisper_largev3_conv2d4_zh_yue50+zh50/final.pt'
    save_path = './data/test.h5'
    quantizer_fn = './quantizer/quantizer_yue_100.pt'
    data_path = '/data2/yumingdong/data/raw/wenet_data_list/yue_50h+zh_50h_2/data.list'
    config_path = 'conf/batchsize_1.yaml'
    device = torch.device('cuda:7')
    dtype = torch.bfloat16
    
    quantizer_input_dim = 1280
    quantizer_output_dim = 8
    quantizer_codebook_size = 256

    args, configs = get_args_configs(data_path, config_path)
    tokenizer = init_tokenizer(configs)
    _, _, train_data_loader, _ = init_dataset_and_dataloader(args, configs, tokenizer)

    encoder = get_whisper_encoder_from_checkpoint(model_dir=model_path, device=device, dtype=dtype)

    quantizer = Quantizer(dim=quantizer_input_dim, 
                          num_codebooks=quantizer_output_dim, 
                          codebook_size=quantizer_codebook_size).to(device).to(dtype)
    quantizer.load_state_dict(torch.load(quantizer_fn))
    
    with h5py.File(save_path, 'w') as hf:
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                xs = batch_dict['feats'].to(device).to(dtype)
                idx = batch_dict['keys'][0]
                xs_lens = batch_dict['feats_lengths'].to(device).to(dtype)
                encoder_embedding = encoder(xs, xs_lens)[0]
                codebook_indexes = quantizer.encode(encoder_embedding)
                numpy_array = codebook_indexes.cpu().numpy()
                hf.create_dataset(idx, data=numpy_array)
    
    print('done')


if __name__ == "__main__":
    main()