(1)一个debug思想，出现以下的情况，往往不是程序卡死
(llm10d) PS D:\desktop\work-projects\qwen-medical-posttraining> python scripts/pytorch_basics.py
(llm10d) PS D:\desktop\work-projects\qwen-medical-posttraining> python scripts/pytorch_basics.py
应该怀疑：
文件是不是没保存就开始编译？
运行的是不是不是当前文件？

（2）python scripts/pytorch_basics.py
运行结果如下：
2.14.0+cpu #表示安装的cpu版本的pytorch
# tensor是张量，也就是多维数字数组
# requires_grad=True表示x后面参与的所有计算，因为后边我要知道loss对x的梯度，pytorch开始要建立一张计算图
# requires_grad=True官方来说就是，要求 PyTorch 跟踪该 Tensor 的计算并计算梯度
x= tensor([1., 2., 3.], requires_grad=True)
# grad_fn=<PowBackward0>表示pytorch不光记得y数组还记得他是怎么来的
y= tensor([1., 4., 9.], grad_fn=<PowBackward0>)
# loss（1+4+9）/3
loss= tensor(4.6667, grad_fn=<MeanBackward0>)
# loss.backward():计算loss对每一个需要求梯度的变量的梯度 ∂Loss/ ，将结果放在x.grad---这一部分并不修改参数
# optimizer.step()更新参数的模型
x.grad= tensor([0.6667, 1.3333, 2.0000])