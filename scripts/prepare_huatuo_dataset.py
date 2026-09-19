#===========================================
#该文件用于生成数据以及进行数据预处理和数据集划分
#===========================================

'''
下载/读取公开医疗问答
↓
随机抽样
↓
清洗空值
↓
去重
↓
转 messages
↓
train.json
validation.json
test.json
'''
#使用公开中文医疗数据集Huatuo26M-Lite。它是 Huatuo-26M 经过清洗、去重、筛选和答案重写后的精炼版本，大约 17.8 万条中文医疗 QA，Apache-2.0 许可；当前数据字段包括 question、answer、score、label、related_diseases 等。
#由于数量庞大，我们只使用其中的部分数据
'''
Train:       3000  训练参数
Validation:   300  训练过程中看模型有没有过拟合
Test:         300  最后模型训练完成以后才评价

需要学习：
真实公开数据
→ 清洗
→ 去重
→ 固定随机种子
→ train/validation/test
→ 正式LoRA
→ validation loss
→ Base vs LoRA
'''
import json
from pathlib import Path

from datasets import load_dataset


# ============================================================
# 1. 基本配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "huatuo"
)

TRAIN_FILE = OUTPUT_DIR / "train.json"
VAL_FILE = OUTPUT_DIR / "validation.json"
TEST_FILE = OUTPUT_DIR / "test.json"


DATASET_NAME = "FreedomIntelligence/Huatuo26M-Lite"

SEED = 42  #random seed 随机种子，固定随机种子，同样的数据，同样的代码，每次划分基本一致，这就叫做可复现性
#注意要先seed然后抽样，排除数据原始排序可能存在某种规律

TRAIN_SIZE = 3000
VAL_SIZE = 300
TEST_SIZE = 300

TOTAL_SIZE = (
    TRAIN_SIZE
    + VAL_SIZE
    + TEST_SIZE
)


SYSTEM_PROMPT = (
    "你是一名医疗健康知识助手。"
    "请使用准确、清晰、谨慎的语言回答用户问题，"
    "不得替代专业医生进行诊断或开具处方。"
)


# ============================================================
# 2. 数据清洗
# ============================================================

def clean_dataset(dataset):

    cleaned = []

    # 用于问题去重
    seen_questions = set()

    for item in dataset:

        question = str(
            item.get("question", "")
        ).strip()

        answer = str(
            item.get("answer", "")
        ).strip()

        # ----------------------------------------
        # 过滤空值
        # ----------------------------------------

        if not question or not answer:
            continue

        # ----------------------------------------
        # 简单长度过滤
        # 防止过短或异常长文本
        # ----------------------------------------

        if len(question) < 5:
            continue

        if len(question) > 250:
            continue

        if len(answer) < 20:
            continue

        if len(answer) > 2000:
            continue

        # ----------------------------------------
        # 按问题去重
        # ----------------------------------------

        if question in seen_questions:
            continue

        seen_questions.add(question)

        cleaned.append(
            {
                "question": question,
                "answer": answer,

                "label": item.get(
                    "label",
                    ""
                ),

                "related_diseases": item.get(
                    "related_diseases",
                    ""
                ),

                "score": item.get(
                    "score",
                    None
                ),
            }
        )

    return cleaned


# ============================================================
# 3. 转换成SFT messages格式
# ============================================================

def convert_to_sft(data):

    result = []

    for item in data:

        result.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": item["question"],
                    },
                    {
                        "role": "assistant",
                        "content": item["answer"],
                    },
                ],

                # 保留部分元数据
                # 方便以后做错误分析
                "metadata": {
                    "label": item["label"],
                    "related_diseases":
                        item["related_diseases"],
                    "score": item["score"],
                },
            }
        )

    return result


# ============================================================
# 4. 保存JSON
# ============================================================

def save_json(data, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 60)
    print("Preparing Huatuo26M-Lite Dataset")
    print("=" * 60)

    # --------------------------------------------------------
    # 下载原始数据
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print(
        "Original samples:",
        len(dataset),
    )

    print(
        "Columns:",
        dataset.column_names,
    )

    # --------------------------------------------------------
    # 清洗
    # --------------------------------------------------------

    print("\nCleaning dataset...")

    cleaned = clean_dataset(
        dataset
    )

    print(
        "Samples after cleaning:",
        len(cleaned),
    )

    if len(cleaned) < TOTAL_SIZE:

        raise RuntimeError(
            "Not enough samples after cleaning."
        )

    # --------------------------------------------------------
    # 固定随机种子并打乱
    # --------------------------------------------------------

    dataset = dataset.shuffle(
        seed=SEED
    )

    # 重新清洗shuffle后的数据
    cleaned = clean_dataset(
        dataset
    )

    # --------------------------------------------------------
    # 截取3600条
    # --------------------------------------------------------

    selected = cleaned[
        :TOTAL_SIZE
    ]

    # --------------------------------------------------------
    # 数据集划分
    # --------------------------------------------------------

    train_raw = selected[
        :TRAIN_SIZE
    ]

    val_raw = selected[
        TRAIN_SIZE:
        TRAIN_SIZE + VAL_SIZE
    ]

    test_raw = selected[
        TRAIN_SIZE + VAL_SIZE:
    ]

    # --------------------------------------------------------
    # 转SFT格式
    # --------------------------------------------------------

    train_data = convert_to_sft(
        train_raw
    )

    val_data = convert_to_sft(
        val_raw
    )

    test_data = convert_to_sft(
        test_raw
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_json(
        train_data,
        TRAIN_FILE,
    )

    save_json(
        val_data,
        VAL_FILE,
    )

    save_json(
        test_data,
        TEST_FILE,
    )

    print("\nDataset split finished.")

    print(
        f"Train: {len(train_data)}"
    )

    print(
        f"Validation: {len(val_data)}"
    )

    print(
        f"Test: {len(test_data)}"
    )

    print("\nSaved to:")

    print(
        TRAIN_FILE
    )

    print(
        VAL_FILE
    )

    print(
        TEST_FILE
    )

    print("\nExample:")

    print(
        json.dumps(
            train_data[0],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()