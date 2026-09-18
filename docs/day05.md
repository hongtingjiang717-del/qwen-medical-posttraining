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
（1）终于移植完成，并且现在源文件以及环境都还在，接下来开始运行
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# python -c "import torch; print('torch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); print('bf16:', torch.cuda.is_bf16_supported())"
torch: 2.14.0+cu130
GPU: NVIDIA GeForce RTX 4090 D
bf16: True
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# python -c "import transformers; print('transformers:', transformers.__version__)"
transformers: 4.51.3
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# nvidia-smi
Fri Sep 18 22:01:17 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.05              Driver Version: 580.76.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090 D      On  |   00000000:C1:00.0 Off |                  Off |
| 31%   37C    P8             18W /  425W |       0MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
（2）python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple peft==0.15.2
安装PEFT---LoRA
好神奇！！！在llm10d的环境上安装的peft，在qwen310的环境上竟然是安装好的？通用了？？？
(3)开始lora训练的smoke test(冒烟测试)
load model
✓

load dataset
✓

LoRA插入
✓

forward
✓

backward
✓

loss下降
✓

save adapter
✓
（4）运行结果：
trainable params: XXXXX
all params: XXXXXXX
trainable%: 0.1863（训练的参数占比）
采用 LoRA 参数高效微调，在冻结基座模型参数的情况下仅更新低秩 Adapter 参数。

(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# python scripts/train_lora.py
============================================================
Qwen3 Medical LoRA Training
============================================================
GPU: NVIDIA GeForce RTX 4090 D
Training dtype: bfloat16

Loading tokenizer...

Loading base model...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 65.85it/s]

Creating LoRA configuration...

Trainable parameters:
trainable params: 3,211,264 || all params: 1,723,786,240 || trainable%: 0.1863

Loading dataset...
Number of training samples: 12
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.

Starting training...
{'loss': 6.0443, 'grad_norm': 8.007596015930176, 'learning_rate': 0.0, 'epoch': 0.33}                                                                    
{'loss': 4.0836, 'grad_norm': 6.566772937774658, 'learning_rate': 0.0002, 'epoch': 0.67}                                                                 
{'loss': 4.0854, 'grad_norm': 3.874664306640625, 'learning_rate': 0.0001923879532511287, 'epoch': 1.0}                                                   
{'loss': 5.0324, 'grad_norm': 5.594372749328613, 'learning_rate': 0.00017071067811865476, 'epoch': 1.33}                                                 
{'loss': 3.4407, 'grad_norm': 5.258092403411865, 'learning_rate': 0.000138268343236509, 'epoch': 1.67}                                                   
{'loss': 3.0019, 'grad_norm': 4.901899337768555, 'learning_rate': 0.0001, 'epoch': 2.0}                                                                  
{'loss': 2.8664, 'grad_norm': 4.735697269439697, 'learning_rate': 6.173165676349103e-05, 'epoch': 2.33}                                                  
{'loss': 3.8069, 'grad_norm': 5.769436359405518, 'learning_rate': 2.9289321881345254e-05, 'epoch': 2.67}                                                 
{'loss': 2.5965, 'grad_norm': 3.9665727615356445, 'learning_rate': 7.612046748871327e-06, 'epoch': 3.0}                                                  
{'train_runtime': 9.8401, 'train_samples_per_second': 3.659, 'train_steps_per_second': 0.915, 'train_loss': 3.884228918287489, 'epoch': 3.0}             
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:09<00:00,  1.09s/it]

Training finished.
Final training loss: 3.884228918287489

Saving LoRA adapter to: /root/qwen-medical-posttraining/outputs/qwen3_medical_lora/adapter

Done.

(5)完整模型
几GB
vs.
Adapter
通常28MB级

（6）结果分析：
Base model: Qwen3-1.7B
GPU: RTX 4090 D
dtype: bfloat16

训练样本: 12
Epoch: 3
Optimizer steps: 9

Trainable params: 3,211,264
Total params: 1,723,786,240
Trainable ratio: 0.1863%

Final training loss: 3.8842
Runtime: 9.84 s

这说明 LoRA 的核心目的实现了：17.24 亿参数里只训练了约 321 万个参数，占 0.1863%。也就是说基座模型基本被冻结，只更新了很小一部分 LoRA 参数，这就是 PEFT 的意义。

## 指令知识学习
(1)检查某个.py文件有没有python语法错误：python -m py_compile scripts\train_lora.py
(2)服务器克隆github上的项目，然后将旧服务器上的模型移过来，但是不上传github（移动速度很快）
(qwen310) root@autodl-container-b9014e9b33-09af3076:~# git ls-remote https://github.com/hongtingjiang717-del/qwen-medical-posttraining.git
638488f26193c51d22f4f5e0ae4d32747749e5a0        HEAD
638488f26193c51d22f4f5e0ae4d32747749e5a0        refs/heads/main
(qwen310) root@autodl-container-b9014e9b33-09af3076:~# git clone https://github.com/hongtingjiang717-del/qwen-medical-posttraining.git
Cloning into 'qwen-medical-posttraining'...
remote: Enumerating objects: 64, done.
remote: Counting objects: 100% (64/64), done.
remote: Compressing objects: 100% (50/50), done.
remote: Total 64 (delta 12), reused 63 (delta 11), pack-reused 0 (from 0)
Unpacking objects: 100% (64/64), 109.24 KiB | 530.00 KiB/s, done.
(qwen310) root@autodl-container-b9014e9b33-09af3076:~# cd ~/qwen-medical-posttraining
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# ls scrpits
ls: cannot access 'scrpits': No such file or directory
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# ls scripts
prepare_dataset.py  __pycache__  pytorch_basics.py  qwen_inference.py  training_loop_demo.py  train_lora.py
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# ls data/processed
medical_sft.json
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# ls ~/qwen-medical-posttraining_old/models
Qwen3-1.7B
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# mkdir -p ~/qwen-medical-posttraining/models
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# mv ~/qwen-medical-posttraining_old/models/Qwen3-1.7B ~/qwen-medical-posttraining/models/
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# ls ~/qwen-medical-posttraining/models/Qwen3-1.7B | head
config.json
configuration.json
generation_config.json
LICENSE
merges.txt
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
README.md
tokenizer_config.json

(3).gitignor是一个文本文件，不是一个文件夹
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# touch .gitignore
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# echo "models/" >> .gitignore
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# echo "outputs/" >> .gitignore
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# echo "__pycache__/" >> .gitignore
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# echo "*.pyc" >> .gitignore
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# cat .gitignore
models/
outputs/
__pycache__/
*.pyc
(qwen310) root@autodl-container-b9014e9b33-09af3076:~/qwen-medical-posttraining# git status
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        deleted:    .gitignore/pytorch-nested-unet-master/.gitignore
        deleted:    .gitignore/pytorch-nested-unet-master/LICENSE
        deleted:    .gitignore/pytorch-nested-unet-master/README.md
        deleted:    .gitignore/pytorch-nested-unet-master/archs.py
        deleted:    .gitignore/pytorch-nested-unet-master/dataset.py
        deleted:    .gitignore/pytorch-nested-unet-master/losses.py
        deleted:    .gitignore/pytorch-nested-unet-master/metrics.py
        deleted:    .gitignore/pytorch-nested-unet-master/preprocess_dsb2018.py
        deleted:    .gitignore/pytorch-nested-unet-master/requirements.txt
        deleted:    .gitignore/pytorch-nested-unet-master/train.py
        deleted:    .gitignore/pytorch-nested-unet-master/utils.py
        deleted:    .gitignore/pytorch-nested-unet-master/val.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .gitignore

no changes added to commit (use "git add" and/or "git commit -a")


(4)curl -I https://github.com  感觉像是刷新github网站的方法，帮助push
(5)watch -n 0.5 nvidia-smi   训练过程中打开另一个终端，每0.5s更新一次，查看显存


# Day05 - Qwen3 LoRA SFT Smoke Test

## 实验配置

- Base Model: Qwen3-1.7B
- GPU: NVIDIA GeForce RTX 4090 D
- Precision: BF16
- Dataset: 12 medical QA samples
- Epochs: 3
- Batch Size: 1
- Gradient Accumulation: 4
- LoRA rank: 8
- LoRA alpha: 16
- Learning Rate: 2e-4

## LoRA 参数

- Trainable Parameters: 3,211,264
- Total Parameters: 1,723,786,240
- Trainable Ratio: 0.1863%

## 训练结果

- Optimizer Steps: 9
- Runtime: 9.84 s
- Final Training Loss: 3.8842
- No NaN / Inf
- No CUDA OOM
- LoRA Adapter successfully saved

## 结论

本次实验成功验证了医疗 SFT 数据从 Chat Template、Tokenization、
LoRA 参数注入、Trainer 训练到 Adapter 保存的完整流程。

本次仅为 smoke test，12 条样本不足以评价模型医疗能力，
后续需使用正式公开医疗数据集进行训练，并通过独立测试集比较 Base Model
与 LoRA Model 的表现。
