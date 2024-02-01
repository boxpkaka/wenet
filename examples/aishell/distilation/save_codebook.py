from multi_quantization import Quantizer, QuantizerTrainer
from wenet.utils.train_utils import init_dataset_and_dataloader
from wenet.utils.init_tokenizer import init_tokenizer
from get_encoder_from_wenet import get_whisper_encoder_from_checkpoint

import torch
import yaml
import argparse
import logging
import h5py
from tqdm import tqdm
from lhotse.features.io import NumpyHdf5Writer


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
    save_path = './data/aishell_train_zh_yue50+zh50.h5'
    quantizer_fn = './quantizer/quantizer_yue_100.pt'
    data_path = '/data2/yumingdong/data/raw/wenet/aishell/train/data.list'
    config_path = './conf/batchsize_1.yaml'
    device = torch.device('cuda:7')
    dtype = torch.float32

    args, configs = get_args_configs(data_path, config_path)
    tokenizer = init_tokenizer(configs)
    _, _, train_data_loader, _ = init_dataset_and_dataloader(args, configs, tokenizer)

    encoder = get_whisper_encoder_from_checkpoint(model_dir=model_path, device=device, dtype=dtype)

    quantizer = Quantizer(dim=1280, num_codebooks=8, codebook_size=256).to(device)
    quantizer.load_state_dict(torch.load(quantizer_fn))
    
    with h5py.File(save_path, 'w') as hf:
        with torch.no_grad():
            for batch_idx, batch_dict in tqdm(enumerate(train_data_loader)):
                xs = batch_dict['feats'].to(device)
                idx = batch_dict['keys'][0]
                xs_lens = batch_dict['feats_lengths'].to(device)
                encoder_embedding = encoder(xs, xs_lens)[0]
                codebook_indexes = quantizer.encode(encoder_embedding)
                numpy_array = codebook_indexes.cpu().numpy()
                hf.create_dataset(idx, data=numpy_array)
    
    print('done')
    
    # with torch.no_grad():
    #     for batch_idx, batch_dict in enumerate(train_data_loader):
    #         print('=' * 90)
    #         xs = batch_dict['feats'].to(device)
    #         xs_lens = batch_dict['feats_lengths'].to(device)
    #         encoder_embedding = encoder(xs, xs_lens)[0]
    #         print(f'shape: {encoder_embedding.shape}')
    #         print(f'mean before encoding: {torch.mean(encoder_embedding)}')
    #         code = quantizer.encode(encoder_embedding)
    #         reconstruct = quantizer.decode(code)
    #         print(f'mean after decoding: {torch.mean(reconstruct)}')
    #         print(f'sum of diff {torch.sum(reconstruct - encoder_embedding)}')
    #         print(f'max of diff {torch.max(reconstruct - encoder_embedding)}')
    #         print(f'mse: {torch.nn.functional.mse_loss(reconstruct, encoder_embedding)}')
    #         print(f'cos: {torch.nn.functional.cosine_similarity(reconstruct, encoder_embedding, dim=-1)}')
    #         print(f'cos mean: {torch.mean(torch.nn.functional.cosine_similarity(reconstruct, encoder_embedding, dim=-1))}')
    #         print('=' * 90)
    
    # @torch.no_grad()
    # def extract_and_save_embedding(self):
    #     """
    #     The extract embedding is used to train quantizer.
    #     """
    #     if self.embedding_file_path.exists():
    #         warn_message = (
    #             f"{self.embedding_file_path} already exists."
    #             + " Skip extracting embeddings from teacher model"
    #         )
    #         logging.warn(warn_message)
    #         return

    #     logging.info("Start to extract embeddings for training the quantizer.")
    #     total_cuts = 0
    #     with NumpyHdf5Writer(self.embedding_file_path) as writer:
    #         for batch_idx, batch in tqdm(enumerate(self.quantizer_train_dl)):
    #             cut_list = batch["supervisions"]["cut"]
    #             (
    #                 encoder_embedding,
    #                 num_frames,
    #             ) = self.teacher_model.extract_embedding(batch)
    #             encoder_embedding = encoder_embedding.cpu().numpy()
    #             for idx, cut in enumerate(cut_list):
    #                 cut.encoder_embedding = writer.store_array(
    #                     key=cut.id,
    #                     value=encoder_embedding[idx][: num_frames[idx]],
    #                 )
    #             total_cuts += len(cut_list)
    #             logging.info(
    #                 f"Processed {total_cuts} output of {self.params.num_utts} cuts."
    #             )

    #     logging.info(f"Processed all {total_cuts} cuts.")


if __name__ == "__main__":
    main()