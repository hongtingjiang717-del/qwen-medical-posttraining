# day04：SFT 数据集 + Baseline 评测框架

git add :将改动放进待提交区
git commit:将待提交区中的内容正式保存成一个版本
json文件是一种存储结构化数据的文本格式
message:训练时的输入格式和推理时的输入格式最好保持一致

## 概念
SFT--Supervised Fine-Tuning：监督微调
PEFT---Parameter-Efficient Fine-Tuning：LoRA是PEFT最经典的方法之一。只训练少量的adapter参数，而不是把整个17亿参数模型全部更新
## 重新梳理一下项目的逻辑
baseline
   ↓
LoRA training
   ↓
fine-tuned model
   ↓
same questions
   ↓
evaluation
   ↓
comparison
## 今日任务
（1）构建医疗问答原始的数据
（2）将普通的json文本转换成message格式，实现SFT也就是监督学习。
该实现依靠三个角色：system：模型应该扮演什么角色/遵守什么总体原则
user:用户输入
assistant:理想答案，
不断给模型看依据如此的system和user的文本，理想答案是什么，从而实现监督学习。
（3）创建独立的评测集（测试集）也就是一些新的问题
data/eval/baseline_questions.json

