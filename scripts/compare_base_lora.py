import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-1.7B"

ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "qwen3_medical_lora"
    / "adapter"
)

EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "baseline_questions.json"
)

RESULT_DIR = PROJECT_ROOT / "results"

RESULT_FILE = (
    RESULT_DIR
    / "smoke_base_vs_lora.json"
)


# ============================================================
# 2. System Prompt
# ============================================================

SYSTEM_PROMPT = (
    "你是一名医疗健康知识助手。"
    "请使用准确、清晰、谨慎的语言回答用户问题，"
    "不得替代专业医生进行诊断或开具处方。"
)


# ============================================================
# 3. 加载评测问题
# ============================================================

def load_questions(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# 4. 单个问题推理
# ============================================================

def generate_answer(
    model,
    tokenizer,
    question,
):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # 将 messages 转成 Qwen 的聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Tokenization
    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    # 将输入移动到模型所在GPU
    device = next(model.parameters()).device

    model_inputs = {
        key: value.to(device)
        for key, value in model_inputs.items()
    }

    input_length = (
        model_inputs["input_ids"].shape[1]
    )

    # 推理阶段不计算梯度,只有正向传播
    #model.rain（）和model.eval（）的区别
    #model.train:训练，使用反向传播计算梯度用来更新参数；
    #model.eval：评估，只进行前向传播，不计算梯度
    with torch.inference_mode():

        generated_ids = model.generate(
            **model_inputs,

            max_new_tokens=256,

            # 为了公平比较，
            # 使用确定性生成
            #模型不是随机采样---确定性解码
            do_sample=False,

            pad_token_id=tokenizer.eos_token_id,
        )

    # model.generate返回：
    # 输入token + 新生成token
    #
    # 所以只截取真正生成出来的部分
    generated_ids = generated_ids[
        :,
        input_length:
    ]

    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return answer


# ============================================================
# 5. 对整个评测集推理
# ============================================================

def run_evaluation(
    model,
    tokenizer,
    questions,
    model_name,
):

    results = []

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Running evaluation: {model_name}"
    )

    print(
        f"{'=' * 60}"
    )

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]

        print(
            f"\n[{index}/{len(questions)}]"
        )

        print(
            "Question:",
            question,
        )

        start_time = time.time()

        answer = generate_answer(
            model,
            tokenizer,
            question,
        )

        elapsed_time = (
            time.time() - start_time
        )

        print(
            "Answer:",
            answer,
        )

        print(
            f"Time: {elapsed_time:.2f}s"
        )

        results.append(
            {
                "id": item["id"],
                "question": question,
                "answer": answer,
                "generation_time_sec": round(
                    elapsed_time,
                    2,
                ),
            }
        )

    return results


# ============================================================
# 6. 清理GPU
# ============================================================

def clear_gpu():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 7. 主程序
# ============================================================

def main():

    print("=" * 60)
    print("Base Qwen vs LoRA Qwen")
    print("=" * 60)

    # --------------------------------------------------------
    # CUDA检查
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
    # 精度
    # --------------------------------------------------------

    if torch.cuda.is_bf16_supported():

        torch_dtype = torch.bfloat16

        print(
            "dtype: bfloat16"
        )

    else:

        torch_dtype = torch.float16

        print(
            "dtype: float16"
        )

    # --------------------------------------------------------
    # 检查文件
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Base model not found: "
            f"{MODEL_PATH}"
        )

    if not ADAPTER_PATH.exists():

        raise FileNotFoundError(
            f"LoRA adapter not found: "
            f"{ADAPTER_PATH}"
        )

    if not EVAL_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation file not found: "
            f"{EVAL_FILE}"
        )

    # --------------------------------------------------------
    # 评测问题
    # --------------------------------------------------------

    questions = load_questions(
        EVAL_FILE
    )

    print(
        "Evaluation samples:",
        len(questions),
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # ========================================================
    # Part A：Base Model
    # ========================================================

    print(
        "\nLoading BASE model..."
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,

            torch_dtype=torch_dtype,

            trust_remote_code=True,

            low_cpu_mem_usage=True,
        )
        .to("cuda")
    )

    base_model.eval()

    base_results = run_evaluation(
        base_model,
        tokenizer,
        questions,
        model_name="Base Qwen3-1.7B",
    )

    # --------------------------------------------------------
    # 清理Base模型
    # --------------------------------------------------------

    print(
        "\nReleasing BASE model..."
    )

    del base_model

    clear_gpu()

    # ========================================================
    # Part B：LoRA Model
    # ========================================================

    print(
        "\nLoading base model again..."
    )

    lora_base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,

            torch_dtype=torch_dtype,

            trust_remote_code=True,

            low_cpu_mem_usage=True,
        )
        .to("cuda")
    )

    print(
        "Loading LoRA adapter..."
    )

    lora_model = (
        PeftModel.from_pretrained(
            lora_base_model,
            ADAPTER_PATH,
        )
    )

    lora_model.eval()

    print(
        "LoRA adapter loaded successfully."
    )

    lora_results = run_evaluation(
        lora_model,
        tokenizer,
        questions,
        model_name="Qwen3-1.7B + Medical LoRA",
    )

    # ========================================================
    # Part C：保存结果
    # ========================================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = []

    for base_item, lora_item in zip(
        base_results,
        lora_results,
    ):

        comparison.append(
            {
                "id": base_item["id"],

                "question":
                    base_item["question"],

                "base_answer":
                    base_item["answer"],

                "lora_answer":
                    lora_item["answer"],

                "base_time_sec":
                    base_item[
                        "generation_time_sec"
                    ],

                "lora_time_sec":
                    lora_item[
                        "generation_time_sec"
                    ],
            }
        )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparison,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "Evaluation finished."
    )

    print(
        "Result saved to:"
    )

    print(
        RESULT_FILE
    )

    print(
        "=" * 60
    )


# ============================================================
# Python入口
# ============================================================

if __name__ == "__main__":
    main()