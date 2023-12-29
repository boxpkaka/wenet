import os
import sys
import json
from tqdm import tqdm
from typing import List


'''
args: root_dir/[wav.scp, text]
output: data.list {}
'''


def get_file(path: str) -> List:
    with open(path, 'r', encoding='utf-8') as f:
        file = f.readlines()
    f.close()
    file = [line.strip() for line in file]
    return file


def save_file(path: str, file: List) -> None:
    with open(path, 'w') as f:
        for item in tqdm(file):
            data = json.dumps(item, ensure_ascii=False)
            f.write(data + '\n')
    f.close()


def main(root_dir: str):
    wav = get_file(os.path.join(root_dir, 'wav.scp'))
    text = get_file(os.path.join(root_dir, 'text'))

    dic = {}
    res = []

    for line in wav:
        idx, path = line.split(" ")
        if dic.get(idx) is None:
            dic[idx] = [path]
    
    for line in text:
        item = line.split(" ")
        idx = item[0]
        trans = ''.join(item[1: ])
        if dic.get(idx) is not None:
            dic[idx].append(trans)
    
    for k, v in dic.items():
        res.append({"key": k, "wav": v[0], "txt": v[1]})

    save_file(os.path.join(root_dir, 'data.list'), res)


if __name__ == "__main__":
    root_dir = sys.argv[1]
    main(root_dir)





