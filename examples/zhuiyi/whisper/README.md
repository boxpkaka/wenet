# Preliminary
1. Download whisper ckpt from this [link](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17-L30)

2. We assume you have run stage0~stage3 using `aishell/s0/run.sh` and here we are simply creating a symbolic link
```sh
ln -s ../s0/data .
```

3. Run below command to convert openai-style ckpt to wenet-style ckpt:
```sh
mkdir -p exp
mkdir -p exp/whisper
mkdir -p exp/whisper/large-v3
. ./path.sh && python wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py \
  --whisper_ckpt downloaded-large-v3.pt \
  --output_dir exp/whisper/large-v3
python local/filter_ckpt.py \
  --filter_list "encoder.embed.conv" \
  --input_ckpt exp/whisper/large-v3/wenet_whisper.pt \
  --output_ckpt exp/whisper/large-v3/wenet_whisper.remove-subsample.pt
```

# Performance Record

## Whisper-largev2 (original) Result

| decoding mode             |  CER  |
|---------------------------|-------|
| attention decoder         | 8.47  |
| ctc greedy search         |  N/A  |
| ctc prefix beam search    |  N/A  |
| attention rescoring       |  N/A  |

## Whisper-largev3 (conv1d2, full-parameter tuning) Result

* Feature info: using log_mel_spectrogram feature, no cmvn
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 1, 8 * 3090 gpu, 30 epochs (about 16 hours), conf/finetune_whisper_largev3_onlyattn.yaml
* Decoding info: ctc_weight 0.0, average_num 2
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 2.57 % N=104765 C=102142 S=2529 D=94 I=74 |
| ctc greedy search         | N/A |
| ctc prefix beam search    | N/A |
| attention rescoring       | N/A |

* Feature info: using log_mel_spectrogram feature, no cmvn, no speed perturb
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 1, 8 * 3090 gpu, 80 epochs (about 43 hours), conf/finetune_whisper_largev3.yaml
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 2.78 % N=104765 C=101943 S=2711 D=111 I=87  |
| ctc greedy search         | 6.89 % N=104765 C=98386 S=6210 D=169 I=839  |
| ctc prefix beam search    | 6.86 % N=104765 C=98410 S=6194 D=161 I=830  |
| attention rescoring       | 5.00 % N=104765 C=99771 S=4874 D=120 I=245  |

## Whisper-largev3 (conv2d4, full-parameter tuning) Result

* Feature info: using log_mel_spectrogram feature, no cmvn
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 1, 8 * 3090 gpu, 30 epochs (about 14 hours), conf/finetune_whisper_largev3_conv2d4_onlyattn.yaml
* Decoding info: ctc_weight 0.0, average_num 2
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 2.63 % N=104765 C=102088 S=2579 D=98 I=79  |
| ctc greedy search         | N/A |
| ctc prefix beam search    | N/A |
| attention rescoring       | N/A |

* Feature info: using log_mel_spectrogram feature, no cmvn, no speed perturb
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 1, 8 * 3090 gpu, 80 epochs (about 36 hours), conf/finetune_whisper_largev3_conv2d4.yaml
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 3.38 % N=104765 C=101336 S=3305 D=124 I=111 |
| ctc greedy search         | 7.43 % N=104765 C=97759 S=6813 D=193 I=779  |
| ctc prefix beam search    | 7.42 % N=104765 C=97767 S=6806 D=192 I=777  |
| attention rescoring       | 5.65 % N=104765 C=99100 S=5538 D=127 I=259  |

# Frequently Asked Questions

- Q: Why are there so many insertion errors in the decoding results of CTC?
- A: Because Chinese characters are composed of multiple bytes, in Whisper's tokenizer, one Chinese character might be represented by multiple tokens (for example, 3 tokens). During the CTC decoding process, it's possible that only two of these tokens are decoded. This not only causes garbled text (see [#2308](https://github.com/wenet-e2e/wenet/issues/2308) ) but also leads to insertion errors.

# Self-learning

## 模型地址

​	`/data1/yumingdong/model`

## 格式转换

- 转换关系如图

<p align="center">
  <img src="./img/whisper-format-convert.png" alt="whisper-format-convert" style="width: 25%;">
</p>


- Transformers => OpenAI：`local/convert_hf_to_openai.sh` 一般用于transformers => wenet的中间格式
- OpenAI => Transformers: `local/convert_openai_to_hf.sh`
- OpenAI => Wenet: 
  -  `local/convert2wenet.sh` 仅修改网络层名，保留全部参数
  -  `local/filter_whisper.sh` 丢弃卷积下采样层参数

## 训练

- 数据准备：`local/make_raw_data_list_with_language.py` ，依赖本地的`tokenizer.json`文件
- 训练：`run.sh`，参照本文开头的官方内容
- 测试：`test.sh`，使用`--language`指定语种

## 注意事项

- 非官方支持多语种训练和指定语种推理
- wenet官方多语种支持进度参照：[Pull Request #2334](https://github.com/wenet-e2e/wenet/pull/2334)
- 涉及CTC解码产生乱码，参照[Frequently Asked Questions](#Frequently-Asked-Questions)
