# Day 02 - PyTorch 完整训练流程
## 今日目标
Dataset 是什么？
DataLoader 是什么？
batch 是什么？
model(x) 在干什么？
loss 是怎么产生的？
optimizer 为什么存在？
epoch 是什么？
IoU 是什么？
## 一、完整训练流程

数据
↓
Dataset
↓
DataLoader
↓
Batch
↓
Model
↓
Prediction
↓
Loss
↓
loss.backward()
↓
optimizer.step()
↓
模型参数改变

---

## 二、今天自己运行的代码

training_loop_demo.py

---

## 三、DSB2018 细胞分割代码对应关系

Dataset：
文件
dataset.py

DataLoader：
文件
train.py
代码：
   train_loader = torch.utils.data.DataLoader(
       train_dataset,
       batch_size=config['batch_size'],
       shuffle=True,
       num_workers=config['num_workers'],
       drop_last=True)
   val_loader = torch.utils.data.DataLoader(
       val_dataset,
       batch_size=config['batch_size'],
       shuffle=False,
       num_workers=config['num_workers'],
       drop_last=False)

Model：
文件：
train.py
代码：
# model
parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                    choices=ARCH_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ARCH_NAMES) +
                    ' (default: NestedUNet)')
parser.add_argument('--deep_supervision', default=False, type=str2bool)
parser.add_argument('--input_channels', default=3, type=int,
                    help='input channels')
parser.add_argument('--num_classes', default=1, type=int,
                    help='number of classes')
parser.add_argument('--input_w', default=96, type=int,
                    help='image width')
parser.add_argument('--input_h', default=96, type=int,
                    help='image height')

Loss：

Optimizer：

Training Loop：

IoU：

---

## 四、今天遇到的问题

（1）问题：配置git环境变量的时候,没有设置成长期变量
解决：
$gitPath = "D:\sofware\Git\cmd"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -notlike "*$gitPath*") {
    [Environment]::SetEnvironmentVariable(
        "Path",
        "$userPath;$gitPath",
        "User"
    )
}
（2）问题：github接口不稳定，导致push的时候上不去
解决：使用电脑的代理接口7897
 git config --global http.proxy http://127.0.0.1:7897
 （3）我注册git的时候使用的qq邮箱，所以在连接github的时候，自动连接的我的qq邮箱那个账号，而不是我目前在使用的谷歌账号的github,捣鼓了半天
 （4）记录一些有关git的命令：
 git status:查看还没有push 的文件
 git add -m "day02:update learning notes":更新信息“”中的内容是写在日志中的，用来备注
 git push -u origin main :上传

---

## 五、今天必须会回答的问题

1. Dataset 是什么？
2. DataLoader 是什么？
3. batch 是什么？
4. epoch 是什么？
5. loss.backward() 和 optimizer.step() 有什么区别？
6. optimizer.zero_grad() 为什么需要？
7. IoU 是什么？