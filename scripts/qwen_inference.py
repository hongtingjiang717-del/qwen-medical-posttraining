#===================
#这个代码在干什么呢？
#输入一句中文
#tokrnizer将它切成token 
#token变成数字ID
#包装成qwen熟悉的聊天格式---chat template
#数字送进gpu上的qwen模型
#qwen一个token一个token的往后预测
#得到一串新的数字ID
#tokenizer再把数字翻译成中文
#============================


import time  #用来记录时间的工具包
import torch

from transformers import AutoTokenizer,AutoModelForCausalLM

#========================
#1.模型名称
#========================
#告诉hugging face,我要加载哪一个模型
#hugging face抱抱脸，ai界的github
#1.7b表示大约有17亿参数
# model_name="Qwen/Qwen3-1.7B"
model_name="/root/qwen-medical-posttraining/models/Qwen3-1.7B"

#========================
#2.检查GPU
#========================
print("="*50)
print("环境信息")
print("="*50)

print("Pytorch version:",torch.__version__)
print("CUDA available",torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name(0))

#=========================================
#3.加载Tokenizer（分词器）作用是将文字和数字之间相互转换
#=========================================
#根据 Qwen/Qwen3-1.7B，把它配套使用的 Tokenizer 下载并加载进来
#注意模型和tokenizer必须配套，一个tokemizer有他的文字和数字序列的对应规则
print("正在加载Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer加载完成")


#===================
#4.准备用户问题
#===================
prompt = "请用简单的语言解释EEG中的Alpha波是什么？"

#===================
#5.先观察普通的Tokenizer
#===================
#将中文句子转换成token id,得到{
    # "input_ids": [123, 456, 789, 111]
# }
plain_tokens = tokenizer(
    prompt,
    add_special_tokens=False
)
plain_ids = plain_tokens["input_ids"]
print("\n"+"="*50)
print("Tokenizer实验")
print("="*50)

print("原始文本：")
print(prompt)

print("\nToken IDs:")
print(plain_ids)

print("\nToken 数量:")
print(len(plain_ids))

print("\nTokens:")
#这句话展示数字ID对应的token长什么样子
#注意token既不等于完整的汉字也不等于单词
#tokenizer可以实现从文字向token id的双向转换
print(tokenizer.convert_ids_to_tokens(plain_ids))

#========================
#6.构造聊天格式
#========================
#一共三个角色：system、user、assistant
#分别说：你是一个脑电专家、alpha波是什么？、alpha波是...
messages = [
    {
        "role":"user",
        "content":prompt
    }
]
# 把普通问题包装成QWEN熟悉的聊天格式
text = tokenizer.apply_chat_template(
    messages,
    tokenizer = False, #意思就是先不要转换成数字，先展示一下template之后的文字是什么样子的
    add_generation_prompt=True,#在最后给模型加一个“现在轮到assistant回答”的标记
    enable_thinking=False  #暂时关闭思考模式
)
print("\n" + "=" * 50)
print("Chat Template 处理后的文本")
print("=" * 50)

print(text)

#==========================
#7.变成模型输入
#==========================
#进入torch 的世界，把文字真正变成数字tensor
#pt表示pytorch tensor
model_inputs=tokenizer(
    [text],
    return_tensor="pt"
)
print("\ninput_ids shape:")
print(model_inputs["input_ids"].shape) 
#假设输出torch.Size([1, 27])表示batch_size = 1;27 = sequence length表示这一批有一条文本，这一条文本有27个token

# =========================
# 8. 加载 Qwen
# =========================

print("\n正在加载 Qwen3-1.7B...")

start_load = time.time()

#加载qwen大模型
#from_pretrained表示使用他们训练好的参数
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto", #Transformers 根据模型和设备自己选择合适的数据类型
    device_map="auto" #自动决定模型放在哪里
)

load_time = time.time() - start_load

print("模型加载完成")
print(f"模型加载时间: {load_time:.2f} 秒")
print("模型所在设备:", model.device)


# 把输入数据放到和模型相同的设备
model_inputs = model_inputs.to(model.device)


if torch.cuda.is_available():
    memory_gb = torch.cuda.memory_allocated() / 1024**3 #PyTorch 当前已经分配了多少 GPU 显存，1024**3表示1gb
    print(f"当前 PyTorch GPU 显存占用: {memory_gb:.2f} GB")


# =========================
# 9. 开始推理
# =========================

print("\n" + "=" * 50)
print("开始生成")
print("=" * 50)

start_generate = time.time()

#只推理，不训练
with torch.inference_mode():
    #让qwen开始生成答案
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,#最多允许模型新生成 128 个 token
        do_sample=True,#按概率进行一定程度的随机采样，所以一个问题两次的答案不一定一样
        temperature=0.7, #控制随机程度，越大越随机
        top_p=0.8,#只保留累计概率达到某个范围的高概率候选
        top_k=20 #优先从概率最高的一小批候选token里选
    )

generation_time = time.time() - start_generate


# =========================
# 10. 只取模型新生成的部分
# =========================
#模型生成的包括原始输入 token + 新生成 token，所以这一步是删掉prompt，保留answer
new_tokens = generated_ids[
    0,
    model_inputs["input_ids"].shape[1]:
]


# =========================
# 11. Token IDs → 中文
# =========================
#把生成的数字重新变成人能看懂得文字
response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True #解码的时候，把这些特殊 token 隐藏掉
)


print("\n模型回答:")
print(response)

print("\n生成 token 数:")
print(len(new_tokens))

print(f"\n生成耗时: {generation_time:.2f} 秒")


if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"峰值 PyTorch GPU 显存: {peak_memory:.2f} GB")