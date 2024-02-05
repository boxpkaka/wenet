import random
import json
import os
from transformers import WhisperTokenizer


def get_wav_table(wav_file: str, wav_table: dict) -> dict:
    with open(wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]
    return wav_table


def shuffle_data(file_path: str) -> None:
    with open(file_path, 'r', encoding='utf8') as file:
        lines = file.readlines()
    random.shuffle(lines)
    with open(file_path, 'w', encoding='utf8') as file:
        file.writelines(lines)


def make_data_list(root_dir_list: list, output_dir: list) -> None:
    # huggingface whisper-large-v3 模型目录路径，需包含 tokenizer.json, tokenizer_config.json
    model_dir = '/data1/yumingdong/model/huggingface/whisper-large-v3'
    tokenizer = WhisperTokenizer.from_pretrained(model_dir)
    vocab = tokenizer.get_vocab()

    wav_file_list = [os.path.join(item[0], 'wav.scp') for item in root_dir_list]
    text_file_list = [[os.path.join(item[0], 'text'), item[1]] for item in root_dir_list]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'data.list')

    wav_table = {}
    for wav_file in wav_file_list:
        wav_table = get_wav_table(wav_file, wav_table)
    
    with open(output_file, 'w', encoding='utf8') as fout:
        for item in text_file_list:
            text_file, language = item
            language_id = vocab[f'<|{language}|>']
            with open(text_file, 'r', encoding='utf8') as fin:
                for line in fin:
                    arr = line.strip().split(maxsplit=1)
                    key = f'{arr[0]}'
                    txt = f'{arr[1]}' if len(arr) > 1 else ''
                    assert key in wav_table
                    wav = wav_table[key]
                    line = dict(key=key, wav=wav, txt=txt, language=language_id)
                    json_line = json.dumps(line, ensure_ascii=False)
                    fout.write(json_line + '\n')
    shuffle_data(output_file)


def main():
    '''
    root_dir_list: <data_dir>, <language>
    data_dir: 包含wav.scp和text
    '''
    root_dir_list = [
        ['/data2/yumingdong/data/raw/wenet/mandarin_100h', 'zh'],
        ['/data2/yumingdong/data/raw/wenet/cantonese_100h', 'yue']
    ]
    output_dir = '/data2/yumingdong/data/raw/wenet_data_list/yue100+zh100'
    make_data_list(root_dir_list, output_dir)


if __name__ == '__main__':
    main()
