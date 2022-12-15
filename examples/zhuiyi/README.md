# wenet

wenet的内部版本, 添加了我们自己的相关脚本, 最好能定期更新.

## 依赖

- Python==3.8
- base_utils: 内部的依赖库.
- pytorch==v1.11.0
- srilm

## 编译

参考wenet runtime相关文档, 主要是生成构建解码图时fst的相关程序和解码程序.

```bash
cd runtime/libtorch
mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
```

## 数据格式

wenet UIO支持两种格式的数据, SHARD格式数据支持读取本地数据读取, 网络分布式存储文件的读取, 对大规模数据更友好.

## 数据准备

主要是从数据库拉取数据并生成text和wav.scp, 同时将音频写入指定文件夹.

```bash
python3 -m local.prepare_data.prepare_train_data -h
```

注意生成训练数据需要较长时间且占一定存储空间, 生成时可以指定路径, 后续实验可以复用数据.

## 模型训练

包括计算CMVN, 生成字典, 准备wenet所需格式的数据, 模型训练等步骤.

```bash
./run.sh
```

模型结构及训练相关参数配置见`conf/train_unified_conformer.yaml`.

使用多卡训练并指定nccl作为dist_backend时, 若报错需要配置环境变量NCCL_P2P_DISABLE:

```bash
export NCCL_P2P_DISABLE=1
```

## 构建解码图

   ```bash
   ./local/make_tlg.sh
   ```

   需要准备

   1. 端到端模型对应的符号表.
   2. 语料, 需要经过分词.
   3. 词典, 要和语料分词使用的词典保持一致. 参考`local/resource_zhuiyi/lexicon.txt`.
   4. 需要指定srlim工具包路径到PATH变量, 一般使用kaldi tools中的srilm即可.

## 解码

1. CTC/AED模型解码

   ```bash
   ./decode.sh
   ```

2. TLG解码图解码

   ```bash
   ./decode_tlg.sh
   ```

3. 实验结果记录`experiments`项目.
4. 实验文件保存至`/data1/share/experiments/wenet`.

## 自学习平台

### self_learning准备

将模型对应版本的self_learning文件夹放到模型文件夹根目录, 若部分旧模型发版时已经存在self_learning, 进行覆盖即可.

### 声学模型调优

1. 数据准备

   ```bash
   python3 -m local.prepare_data.prepare_data_from_local -h
   usage: prepare_data_from_local.py [-h] [--need_rename NEED_RENAME]
                                     [--business_name BUSINESS_NAME]
                                     [--dev_splits DEV_SPLITS] [--nj NJ]
                                     data_dir wav_in_dir wav_out_dir sample_rate
                                     sample_width wav_channels textgrid_dir
                                     paser_conf_path textgrid_channel

   利用本地的音频文件夹和标注文件文件夹生成wenet格式数据.

   positional arguments:
      data_dir              生成的数据文件夹路径.
      wav_in_dir            输入音频文件夹路径.
      wav_out_dir           输出音频文件夹路径.
      sample_rate           音频采样率.
      sample_width          音频位深.
      wav_channels          音频声道数.
      textgrid_dir          textgrid文件夹路径.
      paser_conf_path       textgird配置信息文件路径.
      textgrid_channel      textgrid文件对应声道信息.

   optional arguments:
      -h, --help            show this help message and exit
      --need_rename NEED_RENAME
                            是否需要重命名, 默认选择否.
      --business_name BUSINESS_NAME
                            数据对应的业务名, 用于音频重命名, 默认'selflearning'
      --dev_splits DEV_SPLITS
                            验证集划分比例, 默认0.05.
      --nj NJ               线程数, 默认16.
   ```

2. 模型调优

   ```bash
   ./local/self_learning/run.sh
      Usage: ./local/self_learning/run.sh [options] <data_dir> <self_learning_dir> <out_dir>
      data_dir: 调优数据文件夹, 需要包含train和dev.
      self_learning_dir: 自学习文件夹路径, 一般放在发版模型文件夹下, 部分旧模型不支持.
      out_dir: 调优模型保存路径.
      --average_num: 默认5.
      --gpus: 显卡编号, ','连接, 如'0,1,2,3'.
      --cpus: cpu训练时指定使用的cpu数量, 不指定则默认使用机器cpu核数的一半.
      --lr: 学习率, 默认0.0004.
      --batch_size: 默认16.
      --epoch: 默认20.
      --warmup_steps: 默认500
      --accum_grad: 默认2
   ```

   - data_dir为第一步数据准备中生成的训练数据文件夹.

3. 模型导出
  1. cpu推理模型包导出:

   ```bash
   ./local/self_learning/export/export_cpu_models.sh
      Usage: local/self_learning/export/export_cpu_models.sh [options] <in_dir> <self_learning_dir> <out_dir>
      in_dir: 训练完成的模型文件夹路径,需要包含train.yaml以及pytorch模型文件路径.
      self_learning_dir: 自学习文件夹路径, 一般放在发版模型文件夹下, 部分旧模型不支持.
      out_dir: 导出的模型推理文件夹路径.
      --average_num: 默认5.
      --conf_path: 配置文件路径,用于获取参数,一般为conf/asr.yaml,默认为空.
   ```

   - in_dir为第二步调优后的模型文件夹.
   - libtorch模型文件为`$out_dir/asr.zip`, 替换`$model_dir/libtorch_model/asr.zip`.
   - onnx模型文件夹为`$out_dir/onnx_model/online_model`, `$out_dir/onnx_model/offline_model`, 分别替换`$model_dir/onnx_model/online_model`和`$model_dir/onnx_model/offline_model`下的相关文件.


### 语言模型调优

1. 数据准备

   准备好文本数据, 一句话一行格式.

   ```bash
   python3 -m local.self_learning.format_text -h
   usage: format_text.py [-h] [--is_english] [--is_cantonese]
                      ori_text format_text dict_path

   对原始文本进行处理, 以便后续构建语言模型.

   positional arguments:
   ori_text        待处理文本.
   format_text     处理后的文本.
   dict_path       分词使用的词典路径, 发音词典或者wenet模型文件夹下的lang_char.txt

   optional arguments:
   -h, --help      show this help message and exit
   --is_english    是否是英语, 默认否.
   --is_cantonese  是否是粤语, 默认否.
   ```
   - dict_path: `<model_dir>/lang_char.txt`

2. 构造解码图

   ```bash
   ./local/self_learning/lm.sh
      Usage: ./local/self_learning/lm.sh [options] <self_learning_dir> <text> <out_dir>
      self_learning_dir: 自学习文件夹路径, 一般放在发版模型文件夹下, 部分旧模型不支持.
      text: 调优需要的清洗后的文本.
      out_dir: 输出文件夹.
      --order: 默认3.
      --lambda: 默认0.6.
      --is_kn_smooth: 是否为kneserney平滑算法, 默认否, 即使用wittenbell平滑.
   ```

   - 将`$out_dir/data/lang_test/TLG.fst`替换`$model_dir/graph/TLG.fst`.
   - text 为第一步清洗后的文本
