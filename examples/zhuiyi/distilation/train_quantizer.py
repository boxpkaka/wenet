from save_h5 import get_args_configs
from wenet.utils.train_utils import init_dataset_and_dataloader
from wenet.utils.init_tokenizer import init_tokenizer
from get_encoder import get_encoder_from_wenet

import multi_quantization as quantization
import torch
import logging


'''
量化器训练
- 帧级训练, 所有embedding将被合并然后shuffle为一个Tensor: (*, in_dim)
online:
- 直接用预训练模型产生向量并训练量化器
offline:
- 从本地读取h5文件中的向量

'''

def main():
    args, configs = get_args_configs()
    logging.getLogger().setLevel(logging.INFO)

    B = args.quantizer_batch_size
    device = torch.device(f'cuda:{args.gpu}')
    dtype = torch.bfloat16
    
    trainer = quantization.QuantizerTrainer(dim=args.quantizer_in_dim,
                                            bytes_per_frame=args.quantizer_out_dim,
                                            device=device)
    
    # Get training tensor by online of offline
    if args.train_quantizer_online:
        logging.info("Online Training")
        tokenizer = init_tokenizer(configs)
        _, _, train_data_loader, _ = init_dataset_and_dataloader(args, configs, tokenizer)
        
        encoder = get_encoder_from_wenet(args, configs)
        encoder = encoder.to(device).to(dtype)
        
        total_tensor = []
        cnt = 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                xs = batch_dict['feats'].to(device).to(dtype)
                xs_lens = batch_dict['feats_lengths'].to(device).to(dtype)
                encoder_embedding = encoder(xs, xs_lens)[0]  # (1, T, D)
                total_tensor.append(encoder_embedding.squeeze(0))  # (T, D)
                cnt += 1
                if cnt > args.num_audio:
                    break
        
        total_tensor = torch.cat(total_tensor, dim=0)    # (*, D)
        logging.info(f'Total frames: {total_tensor.shape[0]}')
        # clean cuda memory
        del encoder
        torch.cuda.empty_cache()
        # shuffle tensor
        shuffle_idx = torch.randperm(total_tensor.shape[0])
        total_tensor = total_tensor[shuffle_idx]
    else:
        total_tensor, _ = quantization.read_hdf5_data(args.embedding_path)
    
    def minibatch_generator(data: torch.Tensor, repeat: bool):
        assert 3 * B < data.shape[0]
        cur_offset = 0
        while True if repeat else cur_offset + B <= data.shape[0]:
            start = cur_offset % (data.shape[0] + 1 - B)
            end = start + B
            cur_offset += B
            yield data[start:end, :].to(device).to(dtype=torch.float)

    for x in minibatch_generator(total_tensor, repeat=True):
        trainer.step(x)
        if trainer.done():
            break

    quantizer = trainer.get_quantizer()
    torch.save(quantizer.state_dict(), args.save_path)
    logging.info(f'save in {args.save_path}')


if __name__ == "__main__":
    main()
