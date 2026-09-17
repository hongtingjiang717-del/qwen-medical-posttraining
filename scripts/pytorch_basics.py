import torch 
print(torch.__version__)
#创建张量，1，2，3默认会创建整数tensor，整数不能计算梯度
x = torch.tensor([1.0,2.0,3.0],requires_grad=True)
y=x**2
loss=y.mean()
print("x=",x)
print("y=",y)
print("loss=",loss)
loss.backward() #反向传播
print("x.grad=",x.grad)