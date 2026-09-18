# 配置git网络加速
## 方法一：
# 配置 HTTP 和 HTTPS 代理
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
## 方法二：
原地址：git clone https://github.com/username/project.git 

加速地址：git clone https://kgithub.com/username/project.git


## Day 02

### Dataset
管理数据集，定义样本数量以及如何读取单个样本。

### DataLoader
从 Dataset 中按 batch 取数据，可以实现 shuffle 等功能。

### Batch
模型一次处理的一组训练样本。

### Epoch
整个训练数据被模型完整学习一遍。

### Model
根据输入计算预测结果的神经网络。

### Loss
衡量模型预测与真实答案之间差距的数值。

### Optimizer
根据梯度更新模型参数。

### Learning Rate
学习率，控制每次参数更新的步长。

### Gradient
Loss 对模型参数的变化率。

### IoU
Intersection over Union，交并比，用于评价分割预测区域和真实区域的重合程度。

## day 03
### LLM
large language model,大语言模型
### token
大语言模型处理文本的基本单位，但是不一定等于一个字或者是一个单词
### tokenizer
分词器，负责把文字切分成token，并转换成模型可以处理的token ID
### input_ids
tokenier 将token映射成整数后得到的数字序列
### sequence length 
一条输入中包含的token数量
### chat template
按照模型训练时使用的聊天格式，对system、user、assistant消息进行组织
### inference
推理。使用训练好的模型生成结果，不进行参数的更新
### generation
模型根据已有的token，不断预测后续token的过程
### max_new_token
模型最多允许新增生成多少个token
### causal langage model
根据已有的token 预测下一个token 的语言模型