# day06 ：LoRA模型推理+Base/LoRA同题对比
## 补充一下adapter的知识
adapter相当于一个依附在大模型身上的小模型，他不能脱离基座大模型自己跑。推理的时候必须记载原始的大模型，再把adapter挂上去一起进行前向运算。
两种形式：
- 方式 A：load_base_model + load_adapter 动态挂载；
- 方式 B：**merge（合并）**，把 adapter 权重合并进基座，输出一个独立完整新模型；合并之后就不再需要 adapter 文件。
