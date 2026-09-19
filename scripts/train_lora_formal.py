'''
train.json
    ↓
更新LoRA参数

validation.json
    ↓
不更新参数
只计算 eval_loss

每一个epoch都是：训练--验证--保存checkpoint
最后选择validation loss 最低的checkpoint加载回来，保存best_adapter

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
# 1. 路径和实验配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen3-1.7B"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "huatuo"
    / "train.json"
)

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "huatuo"
    / "validation.json"
)

# 注意：
# 正式训练与昨天的 smoke test 使用不同输出目录
# 不覆盖昨天的小实验
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "qwen3_huatuo_lora"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
)

METRICS_FILE = (
    RESULT_DIR
    / "formal_training_metrics.json"
)

MAX_LENGTH = 1024

SEED = 42


# ============================================================
# 2. 读取JSON
# ============================================================

def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# 3. SFT Dataset
# ============================================================

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

        # ----------------------------------------
        # prompt:
        # system + user
        # ----------------------------------------

        prompt_messages = messages[:-1]

        prompt_text = (
            self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

        # ----------------------------------------
        # 完整训练文本:
        # system + user + assistant
        # ----------------------------------------

        full_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )

        # ----------------------------------------
        # Tokenize prompt
        # ----------------------------------------

        prompt_encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        # ----------------------------------------
        # Tokenize完整文本
        # ----------------------------------------

        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = (
            full_encoding["input_ids"]
        )

        attention_mask = (
            full_encoding["attention_mask"]
        )

        # labels初始复制input_ids
        labels = input_ids.copy()

        # ----------------------------------------
        # system + user不计算loss
        # assistant答案才计算loss
        # ----------------------------------------

        prompt_length = min(
            len(
                prompt_encoding[
                    "input_ids"
                ]
            ),
            len(labels),
        )

        labels[:prompt_length] = (
            [-100] * prompt_length
        )

        return {
            "input_ids": input_ids,
            "attention_mask":
                attention_mask,
            "labels": labels,
        }


# ============================================================
# 4. Dynamic Padding
# ============================================================

class SFTDataCollator:

    def __init__(
        self,
        tokenizer,
    ):

        self.tokenizer = tokenizer

    def __call__(
        self,
        features,
    ):

        max_length = max(
            len(
                feature["input_ids"]
            )
            for feature in features
        )

        input_ids = []
        attention_masks = []
        labels = []

        for feature in features:

            current_length = len(
                feature["input_ids"]
            )

            padding_length = (
                max_length
                - current_length
            )

            input_ids.append(
                feature["input_ids"]
                + [
                    self.tokenizer.pad_token_id
                ]
                * padding_length
            )

            attention_masks.append(
                feature[
                    "attention_mask"
                ]
                + [0] * padding_length
            )

            labels.append(
                feature["labels"]
                + [-100]
                * padding_length
            )

        return {
            "input_ids":
                torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    attention_masks,
                    dtype=torch.long,
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long,
                ),
        }


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 70)
    print(
        "Qwen3-1.7B Huatuo "
        "Formal LoRA SFT"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # GPU检查
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    use_bf16 = (
        torch.cuda.is_bf16_supported()
    )

    if use_bf16:

        torch_dtype = (
            torch.bfloat16
        )

        print(
            "Training dtype: bfloat16"
        )

    else:

        torch_dtype = (
            torch.float16
        )

        print(
            "Training dtype: float16"
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # --------------------------------------------------------
    # Base Model
    # --------------------------------------------------------

    print(
        "\nLoading base model..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,

            torch_dtype=
                torch_dtype,

            trust_remote_code=True,

            low_cpu_mem_usage=True,
        )
    )

    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    print(
        "\nCreating LoRA adapter..."
    )

    lora_config = LoraConfig(

        r=8,

        lora_alpha=16,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        lora_dropout=0.05,

        bias="none",

        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.enable_input_require_grads()

    print(
        "\nTrainable parameters:"
    )

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # 数据
    # --------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    train_raw = load_json(
        TRAIN_FILE
    )

    val_raw = load_json(
        VAL_FILE
    )

    print(
        "Train samples:",
        len(train_raw),
    )

    print(
        "Validation samples:",
        len(val_raw),
    )

    train_dataset = (
        MedicalSFTDataset(
            train_raw,
            tokenizer,
            MAX_LENGTH,
        )
    )

    val_dataset = (
        MedicalSFTDataset(
            val_raw,
            tokenizer,
            MAX_LENGTH,
        )
    )

    data_collator = (
        SFTDataCollator(
            tokenizer
        )
    )

    # --------------------------------------------------------
    # 正式训练参数
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=str(
            OUTPUT_DIR
            / "checkpoints"
        ),

        # 单卡一次1条
        # 稳妥控制显存
        per_device_train_batch_size=1,

        # 验证可以一次2条
        per_device_eval_batch_size=2,

        # 8个mini batch后
        # 才执行一次参数更新
        gradient_accumulation_steps=8,

        # 正式实验先跑2 epoch
        num_train_epochs=2,

        learning_rate=2e-4,

        warmup_ratio=0.05,

        lr_scheduler_type="cosine",

        # 每20 optimizer step
        # 打印一次训练日志
        logging_steps=20,

        # 每个epoch验证一次
        eval_strategy="epoch",

        # 每个epoch保存一次
        save_strategy="epoch",

        save_total_limit=2,

        # 训练完成后自动加载
        # validation loss最低的checkpoint
        load_best_model_at_end=True,

        metric_for_best_model=
            "eval_loss",

        greater_is_better=False,

        bf16=use_bf16,

        fp16=not use_bf16,

        gradient_checkpointing=True,

        optim="adamw_torch",

        report_to="none",

        remove_unused_columns=False,

        # 明确告诉Trainer
        # labels字段是监督标签
        label_names=["labels"],

        seed=SEED,

        data_seed=SEED,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=
            train_dataset,

        eval_dataset=
            val_dataset,

        data_collator=
            data_collator,
    )

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------

    print(
        "\nStarting formal training..."
    )

    # 重置CUDA峰值显存统计
    torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    print(
        "\nTraining finished."
    )

    # --------------------------------------------------------
    # 使用best model再跑一次validation
    # --------------------------------------------------------

    print(
        "\nEvaluating best model..."
    )

    eval_metrics = (
        trainer.evaluate()
    )

    # --------------------------------------------------------
    # 峰值显存
    # --------------------------------------------------------

    peak_allocated_gb = (
        torch.cuda
        .max_memory_allocated()
        / 1024 ** 3
    )

    peak_reserved_gb = (
        torch.cuda
        .max_memory_reserved()
        / 1024 ** 3
    )

    print(
        f"\nFinal train loss: "
        f"{train_result.training_loss:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{eval_metrics['eval_loss']:.4f}"
    )

    print(
        "Best checkpoint:",
        trainer.state.best_model_checkpoint,
    )

    print(
        f"Peak allocated GPU memory: "
        f"{peak_allocated_gb:.2f} GB"
    )

    print(
        f"Peak reserved GPU memory: "
        f"{peak_reserved_gb:.2f} GB"
    )

    # --------------------------------------------------------
    # 保存最终best Adapter
    # --------------------------------------------------------

    adapter_dir = (
        OUTPUT_DIR
        / "best_adapter"
    )

    adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving best LoRA adapter to:",
        adapter_dir,
    )

    trainer.model.save_pretrained(
        adapter_dir
    )

    tokenizer.save_pretrained(
        adapter_dir
    )

    # --------------------------------------------------------
    # 保存实验指标
    # --------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {

        "base_model":
            "Qwen3-1.7B",

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(val_dataset),

        "epochs": 2,

        "max_length":
            MAX_LENGTH,

        "learning_rate":
            2e-4,

        "gradient_accumulation_steps":
            8,

        "train_loss":
            train_result.training_loss,

        "eval_loss":
            eval_metrics["eval_loss"],

        "best_checkpoint":
            trainer.state.best_model_checkpoint,

        "train_runtime_sec":
            train_result.metrics.get(
                "train_runtime"
            ),

        "peak_allocated_gpu_gb":
            peak_allocated_gb,

        "peak_reserved_gpu_gb":
            peak_reserved_gb,

        "log_history":
            trainer.state.log_history,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nMetrics saved to:",
        METRICS_FILE,
    )

    print(
        "\nFormal training complete."
    )


if __name__ == "__main__":
    main()