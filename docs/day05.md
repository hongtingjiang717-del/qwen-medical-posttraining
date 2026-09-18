# day05:LoRA微调
## 今日学习框架
Qwen3-1.7B
    ↓
medical_sft.json
    ↓
LoRA
    ↓
训练
    ↓
LoRA Adapter
    ↓
加载 Base + Adapter
    ↓
医疗问答推理

## 实验1
运行结果：
trainable params: XXXXX
all params: XXXXXXX
trainable%: X%
采用 LoRA 参数高效微调，在冻结基座模型参数的情况下仅更新低秩 Adapter 参数。

## 指令知识学习
检查某个.py文件有没有python语法错误：python -m py_compile scripts\train_lora.py