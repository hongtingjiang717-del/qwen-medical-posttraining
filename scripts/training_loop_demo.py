import torch
from torch import nn #神经网络相关工具
from torch.utils.data import Dataset,DataLoader #管理训练数据

#创建一个假的数据集
#Dataset负责保存数据，并告诉程序第几个样本是什么
class SimpleDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
        ])
        self.y = torch.tensor([
            [2.0],
            [4.0],
            [6.0],
            [8.0],
            [10.0],
            [12.0],
        ])
    def __len__(self):
        return len(self.x)
    #提取第几个index的样本
    def __getitem__(self,index):
        return self.x[index],self.y[index]

#测试代码
dataset = SimpleDataset()
#加入dataloader
dataloader = DataLoader(
    dataset,
    batch_size = 2,  #每次训练只取两条数据
    shuffle = True  #每次训练之前把数据的顺序打乱
)
# for batch_x,batch_y in dataloader:
    # print("batch_x:",batch_x)
    # print("batch_y:",batch_y)

#加载一个简单的模型 
model = nn.Linear(1,1)  #y=wx+b,我们的目标是让模型自己学到w=2,b=0
#打印模型一开始运行的随机参数
print("训练前weight:",model.weight.item())
print("训练前bias:",model.bias.item())

#加入loss
loss_fn = nn.MSELoss()  #这里使用均方误差作为损失函数

#加入optimizer
optimizer = torch.optim.SGD(
    model.parameters(),  #SGD是一种模型优化算法，根据梯度决定参数怎么改
    lr = 0.01  #学习率，每次修改参数的时候迈多大一步
)

#training loop训练循环
epochs =100
for epoch in range(epochs):
    total_loss = 0
    for batch_x,batch_y in dataloader:
        prediction = model(batch_x)
        loss = loss_fn(prediction,batch_y)
        optimizer.zero_grad()   #把上一轮留下来的梯度清空
        loss.backward() #反向传播，计算每个参数的梯度
        optimizer.step() #根据刚才计算的梯度，真正的更新weight和bias
        total_loss += loss.item()#为什么是加和呢？注意是一个epoch中的所有的batch的loss加和，每次epoch之前都会归零

    if (epoch + 1) % 10 == 0:
        print(
            f"epoch{epoch+1},"
            f"loss={total_loss:.4f}"
        )
print("训练后weight:",model.weight.item())
print("训练后bias:",model.bias.item())



# print("数据集的长度：",len(dataset))
# print("第一个样本：",dataset[0])