# wenet

wenet的内部版本, 添加了我们自己的相关脚本, 最好能定期更新.

## 依赖

- Python==3.6
- base_utils: 内部的依赖库.
- pytorch==v1.10.1
- srilm

## 编译

参考wenet runtime相关文档, 主要是生成构建解码图时fst的相关程序和解码程序.

```bash
cd runtime/server/x86/build
cmake ..
make -j 16
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

### 声学模型调优

1. 数据准备

   ```bash
   python3 local.prepare_data.prepare_data_from_local -h
   ```

2. 模型调优

   ```bash
   ./local/self_learning/run.sh
      Usage: ./local/self_learning/run.sh [options] <data_dir> <model_dir> <out_dir>
      data_dir: 调优数据文件夹, 需要包含train和dev.
      model_dir: ASR模型文件夹, 需要包含self_learning文件夹.
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
   - libtorch模型文件为`$out_dir/asr.zip`, 替换`$model_dir/libtorch_model/asr.zip`.
   - onnx模型文件夹为`$out_dir/onnx/online_model`, `$out_dir/onnx/offline_model`, 分别替换`$model_dir/onnx_model/online_model`和`$model_dir/onnx_model/offline_model`下的相关文件.

### 语言模型调优

1. 数据准备

   准备好文本数据, 一句话一行格式.

2. 构造解码图

   ```bash
   ./local/self_learning/lm.sh
   Usage: ./local/self_learning/lm.sh [options] <model_dir> <text> <out_dir>
   model_dir: 模型文件夹.
   text: 调优需要的文本.
   out_dir: 输出文件夹.
   --order: 默认3.
   --lambda: 默认0.85.
   ```

   - 将`$out_dir/data/lang_test/TLG.fst`替换`$model_dir/graph/TLG.fst`.
