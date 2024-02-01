import multi_quantization as quantization

import torch
import logging
logging.getLogger().setLevel(logging.INFO)


def main():
    embedding_file_path = './data/yue100_for_quantizer_training.h5'
    device = torch.device('cuda:0')
    save_path = './quantizer/quantizer_yue_100.pt'
    
    B = 512    # Batch size
    dim = 1280
    bytes_pre_frame = 8

    trainer = quantization.QuantizerTrainer(dim=dim,
                                            bytes_per_frame=bytes_pre_frame,
                                            device=device)


    train, valid = quantization.read_hdf5_data(embedding_file_path)

    def minibatch_generator(data: torch.Tensor, repeat: bool):
        assert 3 * B < data.shape[0]
        cur_offset = 0
        while True if repeat else cur_offset + B <= data.shape[0]:
            start = cur_offset % (data.shape[0] + 1 - B)
            end = start + B
            cur_offset += B
            yield data[start:end, :].to(device).to(dtype=torch.float)
    cnt = 0
    for x in minibatch_generator(train, repeat=True):
        trainer.step(x)
        cnt += 1
        if trainer.done():
            break

    quantizer = trainer.get_quantizer()
    torch.save(quantizer.state_dict(), save_path)
    print('done')
    print(f'save in {save_path}')


if __name__ == "__main__":
    main()
