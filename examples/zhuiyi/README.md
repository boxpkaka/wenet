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
python3 -m local.prepare_train_data -h
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
