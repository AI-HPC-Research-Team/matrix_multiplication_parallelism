# Matrix Multiplication Parallelism

我们在mindspore框架上实现新的矩阵乘并行算法，提供更灵活的切分策略，并根据盘古模型测试了所提出的并行算法。我们的算法已经合并到[mindspore1.9](https://github.com/mindspore-ai/mindspore.git)版本中,具体代码实现在[matmul_info.cc](https://github.com/mindspore-ai/mindspore/blob/r1.9/mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc)文件。

## 算法介绍

我们的模型可以将任意的GPU个数拆分乘两个因子$m$和$n$相乘，即不一定要求$m=n$，使得空闲的GPU更少，能够充分利用GPU资源，并且在执行一次并行矩阵乘法之后，运算结果举证依旧是按照 $m \times n$块划分，无须重排布，减少了通信需求。

对于两个矩阵$A$,$B$相乘，$A\times B=C$,其中$A\in \mathbb{R}^{s \times h}, B\in \mathbb{R}^{h \times k}，C\in \mathbb{R}^{s \times k}$， 分别将矩阵 `A`，`B`按照行切分$m$块、按列切分$n$块。对于$A$矩阵，$GPU_{i,1},GPU_{i,2},...,GPU_{i,n}$相互通信，并按列拼接成小矩阵$A_i\in \mathbb{R}^{s/m \times h }$；对于$B$矩阵，$GPU_{1,j},GPU_{2,j},...,GPU_{m,j}$相互通信，并按行拼接成小矩阵$B_j\in \mathbb{R}^{h \times k/n }$，对于$GPU_{i,j}$,其运算结果为 $A_i \times B_j = C_{i.j}$，$C_{i,j}\in \mathbb{R}^{s/m \times k/n}$。如下图所示：

![1670469977985](image/README/1670469977985.png)

我们为transformer的并行训练提供了一种新的并行策略，以Pangu-alpha为例，其并行策略如下图所示：

![1670486295971](image/README/1670486295971.png)

## Installation

首先需要安装mindspore Ascend版本或GPU版本，其次根据[Pangu-alpha](https://gitee.com/mindspore/models.git)仓库指引安装以下python库

mindspore 1.9.0 or higher version

* jieba 0.42.1
* sentencepiece 0.1.94
* transformers >= 4.7.0

## Train

我们修改了模型的切分策略，当运行分布训练脚本时，会执行上述切分策略

```
bash scripts/run_distribute_train.sh DATASET RANK_TABLE RANK_SIZE TYPE MODE STAGE_NUM MICRO_SIZE PER_BATCH RANK_START
```

## Reference

[Pangu-alpha](https://gitee.com/mindspore/models.git)

[mindspore](https://github.com/mindspore-ai/mindspore.git)
