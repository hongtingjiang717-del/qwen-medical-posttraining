'''
① 找到 Qwen3-1.7B
↓
② 读取 medical_sft.json
↓
③ tokenizer 把文字变成 token
↓
④ 给 Qwen 添加 LoRA Adapter
↓
⑤ 只训练 LoRA 参数
↓
⑥ 保存 Adapter
↓
⑦ 输出训练结果
'''
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-1.7B"

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medical_sft.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "qwen3_medical_lora"
)


# 最大序列长度
MAX_LENGTH = 1024


# ============================================================
# 2. 读取 JSON 数据
# ============================================================

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 3. 构建 SFT Dataset
# ============================================================
#作用：将json中的一条样本，根据推理中的聊天模板，转化成messages；然后进入tokenizer生成inputs_ids，attention_mask和labels(正确答案)
#loss的计算：根据前面的token→预测下一个token，和labels也就是正确答案作比较，计算loss
class MedicalSFTDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        max_length=1024,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        messages = item["messages"]

        # -------------------------
        # Prompt部分
        # system + user
        # -------------------------

        prompt_messages = messages[:-1]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # -------------------------
        # 完整文本
        # system + user + assistant
        # -------------------------

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Prompt tokenization
        prompt_encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        # 完整文本 tokenization
        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        # labels 最开始复制 input_ids
        labels = input_ids.copy()

        # Prompt部分不参与loss
        prompt_length = min(
            len(prompt_encoding["input_ids"]),
            len(labels),
        )
        #这句话比较关键，他表示我们模型在学习的时候忽略-100这个位置（system和user的内容所在的位置）
        #主要学习根据system和user，assistant这个参考答案是怎么回答的
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# 4. Data Collator
#    一个batch中的不同句子需要padding到相同长度
# ============================================================

class SFTDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):

        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        input_ids = []
        attention_masks = []
        labels = []

        for feature in features:

            current_length = len(feature["input_ids"])

            padding_length = (
                max_length - current_length
            )

            input_ids.append(
                feature["input_ids"]
                + [self.tokenizer.pad_token_id]
                * padding_length
            )

            attention_masks.append(
                feature["attention_mask"]
                + [0] * padding_length
            )

            labels.append(
                feature["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 60)
    print("Qwen3 Medical LoRA Training")
    print("=" * 60)

    # --------------------------------------------------------
    # 检查GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Please run this script on the GPU server."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------------
    # 判断使用 bf16 还是 fp16
    # --------------------------------------------------------

    use_bf16 = torch.cuda.is_bf16_supported()

    if use_bf16:
        torch_dtype = torch.bfloat16
        print("Training dtype: bfloat16")
    else:
        torch_dtype = torch.float16
        print("Training dtype: float16")

    # --------------------------------------------------------
    # 加载 Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # 加载 Base Model
    # --------------------------------------------------------

    print("\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 训练时关闭KV Cache
    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA配置
    # --------------------------------------------------------

    print("\nCreating LoRA configuration...")

    lora_config = LoraConfig(
        r=8, #rank,表示低秩矩阵的秩；该值越大，表示学习能力越强，参数量越大，显存占用越大
        lora_alpha=16,#控制lora更新的缩放强度，目前是一个入门配置2*r

        #qwen的transformer attention里面：QKVO分别对应下面的参数，让lora插进这些层
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        lora_dropout=0.05,#训练时随机丢弃一部分lora激活，防止过拟合

        bias="none",

        task_type="CAUSAL_LM",
    )
    #由普通的qwen3模型升级为lora参数的模型，注意原模型的参数被冻结
    #qwen3+lora adapter
    #有关lora adapter
    '''
    1. 加载预训练底座模型，全部参数冻结；
    2. 在 Transformer 注意力层旁边挂上两组很小的矩阵 A、B；
    3. 只训练 A、B；训练结束保存下来的这一组 A/B 权重文件 / 文件夹，就叫 **LoRA Adapter**。

    注意区分名词：

   - LoRA：是整套**微调算法技术名字**
   - LoRA‑Adapter：是**训练完成后得到的轻量插件权重（产物）**（PEFT 库输出的 adapter 文件夹）
   - Base model：原始底座大模型
   训练结束输出目录下生成`adapter_config.json`、`adapter_model.safetensors`，这两个就是你的 **LoRA adapter（适配器）**，**不是完整大模型**，不能单独拿来推理，必须搭配原来的 base 底座模型一起加载。
    '''
    model = get_peft_model(
        model,
        lora_config,
    )

    # Gradient checkpointing时使用
    model.enable_input_require_grads()

    # 查看真正需要训练多少参数
    print("\nTrainable parameters:")

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # 加载训练数据
    # --------------------------------------------------------

    print("\nLoading dataset...")

    raw_data = load_json(TRAIN_FILE)

    print(
        f"Number of training samples: "
        f"{len(raw_data)}"
    )

    train_dataset = MedicalSFTDataset(
        raw_data,
        tokenizer,
        max_length=MAX_LENGTH,
    )

    data_collator = SFTDataCollator(
        tokenizer
    )

    # --------------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=str(
            OUTPUT_DIR / "checkpoints"
        ),

        per_device_train_batch_size=1,

        #为什么要进行梯度累积？因为开大batch显存不够，4就是积累4个batch更新一次，4个batch的loss求和会除以他们的个数
        gradient_accumulation_steps=4, #梯度累积，第一条计算梯度不更新；第二条累积梯度不更新；直到第四条累积梯度才更新，适合显存比较有限的大模型

        num_train_epochs=3,

        learning_rate=2e-4,

        warmup_ratio=0.1,

        lr_scheduler_type="cosine",

        logging_steps=1,

        save_strategy="epoch",

        save_total_limit=2,

        bf16=use_bf16,

        fp16=not use_bf16,

        gradient_checkpointing=True,#用计算时间换取显存，中间的有些结果不保存，需要的时候会重新计算

        optim="adamw_torch",

        report_to="none",

        remove_unused_columns=False,

        seed=42,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # 开始训练
    # --------------------------------------------------------

    print("\nStarting training...")

    train_result = trainer.train()

    print("\nTraining finished.")

    print(
        "Final training loss:",
        train_result.training_loss,
    )

    # --------------------------------------------------------
    # 保存LoRA Adapter
    # --------------------------------------------------------

    adapter_dir = (
        OUTPUT_DIR
        / "adapter"
    )

    adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving LoRA adapter to:",
        adapter_dir,
    )

    model.save_pretrained(
        adapter_dir
    )

    tokenizer.save_pretrained(
        adapter_dir
    )

    print("\nDone.")


# ============================================================
# 6. Python程序入口
# ============================================================

if __name__ == "__main__":
    main()