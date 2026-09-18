#数据预处理的pipeline
'''
medical_qa_raw.json
        ↓
Python读取
        ↓
数据检查
        ↓
转换格式
        ↓
medical_sft.json
'''
import json #导入python自带的JSON模块，负责将JSON文件转换成python对象
from pathlib import Path #处理路径的方法


# 1. 获取项目根目录qwen-medical-posttraining
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. 定义输入文件
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "medical_qa_raw.json"

# 3. 定义输出文件
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "medical_sft.json"


def load_raw_data(file_path):
    """读取原始医疗问答数据。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

#具体来说的说，json格式可能是
'''
    {
  "question": "什么是脑电图？",
  "answer": "脑电图是一种..."
}
'''
#而转换成message格式之后，变成如下的格式：
'''
{
  "question": "什么是脑电图？",
  "answer": "脑电图是一种..."
}
'''
def convert_to_sft_format(raw_data):
    """将 question-answer 格式转换为 messages 格式。"""

    sft_data = []

    for item in raw_data:

        question = item["question"].strip()
        answer = item["answer"].strip()

        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一名医疗健康知识助手。"
                        "请使用准确、清晰、谨慎的语言回答用户问题，"
                        "不得替代专业医生进行诊断或开具处方。"
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]
        }

        sft_data.append(sample)

    return sft_data


def save_data(data, file_path):
    """保存处理后的数据。"""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():

    print("Loading raw dataset...")

    raw_data = load_raw_data(INPUT_FILE)

    print(f"Raw samples: {len(raw_data)}")

    sft_data = convert_to_sft_format(raw_data)

    save_data(sft_data, OUTPUT_FILE)

    print(f"Processed samples: {len(sft_data)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()







