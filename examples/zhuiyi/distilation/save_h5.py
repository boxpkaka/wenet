from multi_quantization import Quantizer
from wenet.utils.train_utils import init_dataset_and_dataloader
from wenet.utils.init_tokenizer import init_tokenizer
from get_encoder import get_encoder

import torch
import yaml
import h5py
import logging
import argparse
import numpy as np

'''
保存h5文件, 用于生成<蒸馏码本>或<量化器训练所需的向量>
若生成蒸馏码本，必须通过 --quantizer_path 导入量化器
'''


def get_args_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',              type=str)
    parser.add_argument('--encoder_type',            type=str)
    parser.add_argument('--save_path',               type=str, required=True)
    parser.add_argument('--train_data',              type=str)
    parser.add_argument('--quantizer_path',          type=str, default='')
    parser.add_argument('--quantizer_in_dim',        type=int, default=1280)
    parser.add_argument('--quantizer_out_dim',       type=int, default=8)
    parser.add_argument('--quantizer_codebook_size', type=int, default=256)
    parser.add_argument('--quantizer_batch_size',    type=int, default=512)
    parser.add_argument('--train_quantizer_online',  action='store_true')
    parser.add_argument('--embedding_path',          type=str, default='')
    parser.add_argument('--config',                  type=str, default='./conf/batchsize_1.yaml')
    parser.add_argument('--gpu',                     type=str, default='0')
    parser.add_argument('--save_codebook',           action='store_true')
    parser.add_argument('--num_audio',               type=int, default=2000)
    parser.add_argument('--data_type',               type=str, default='raw')
    parser.add_argument('--cv_data',                 type=str, default='/data2/yumingdong/data/raw/wenet/test_1000Cantonese/data.list')
    parser.add_argument('--pin_memory',              type=bool, default=False)
    parser.add_argument('--num_workers',             type=int, default=8)
    parser.add_argument('--prefetch',                type=int, default=500)
    parser.add_argument('--jit',                     type=bool, default=False)
    parser.add_argument('--deepspeed_config',        type=str, default='../whisper/conf/ds_stage1.json')
    parser.add_argument('--train_engine',            type=str, default='deepspeed')
    parser.add_argument('--use_amp',                 type=bool, default=True)
    parser.add_argument('--model_dir',               type=str, default='./')
    parser.add_argument('--save_states',             type=str, default='only model')

    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
        
    return args, configs


def main():
    args, configs = get_args_configs()
    device = torch.device(f'cuda:{args.gpu}')
    dtype = torch.bfloat16
    logging.getLogger().setLevel(logging.INFO)
    
    tokenizer = init_tokenizer(configs)
    _, _, train_data_loader, _ = init_dataset_and_dataloader(args, configs, tokenizer)
    
    encoder = get_encoder(args, configs)
    encoder = encoder.to(device).to(dtype)
    
    if args.save_codebook:
        if args.quantizer_path == '':
            logging.error("should pass the quantizer ckpt path with \"--quantizer_path\"")
        quantizer = Quantizer(dim=args.quantizer_in_dim, 
                            num_codebooks=args.quantizer_out_dim, 
                            codebook_size=args.quantizer_codebook_size).to(device).to(dtype)
        quantizer.load_state_dict(torch.load(args.quantizer_path))
        logging.info("save codebook indexes for distillation")
    else:
        quantizer = None
        logging.info("save encoder embedding for quantizer training")
    
    cnt = 0
    with h5py.File(args.save_path, 'w') as hf:
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                idx = batch_dict['keys'][0]
                encoder_embedding = encoder(batch_dict, device, dtype)
                info = f'{idx} embedding: {encoder_embedding.shape} '
                
                if quantizer is None:
                    if cnt > args.num_audio:
                        break
                    numpy_array = encoder_embedding.to(torch.float16).cpu().numpy().astype(np.float16)
                    logging.info(f'[{cnt}]/[{args.num_audio}]' + info)
                    cnt += 1
                else:
                    codebook_indexes = quantizer.encode(encoder_embedding)
                    numpy_array = codebook_indexes.cpu().numpy()
                    logging.info(info + f'codebook_indexes: {codebook_indexes.shape}')
                
                hf.create_dataset(idx, data=numpy_array)
                
    logging.info('done')


if __name__ == "__main__":
    main()