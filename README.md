# lit_span_re

关系抽取和实体抽取模型。

方案说明：
实体抽取模型：
实体抽取方案使用span做相对位置，产生的位置用于下一步。
关系抽取模型：
对实体表示的序列进行meanpool处理，此处对实体的全部表示进行mean处理用来作为实体在当前序列里的表示。
使用孪生网络进行分类任务，做分类任务来训练关系抽取模型。


# 方案2 基于记忆的关系模型
当前模型可以基于三元组数据做预训练，主体设计借鉴sentent bert的孪生网络结构。
使用均值池化方案做。

https://github.com/lucidrains/x-transformers#augmenting-self-attention-with-persistent-memory









快速开发Pytorch的简单例子。
包含了一个简单的神经网络，一个简单的训练和预测程序。


## about

# 核心

- 构建数据集（https://github.com/napoler/BulidDataset ）
- 训练模型（当前）
- 部署模型（https://github.com/napoler/tkit-bentoml-frameworks-expand ）

## 构建数据集
构建数据集（https://github.com/napoler/BulidDataset ）
包含多个构建预处理数据集的示例。

## 训练模型
当前，训练模型（当前）

## 部署模型
部署模型（https://github.com/napoler/tkit-bentoml-frameworks-expand ）

推荐使用Bentoml快速部署, 可以在线部署模型。

