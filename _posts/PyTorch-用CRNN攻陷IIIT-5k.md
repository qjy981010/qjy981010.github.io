---
title: PyTorch文字识别 用CRNN攻陷IIIT-5k
date: 2017-12-24 19:16:40
tags: ['pytorch','crnn','IIIT-5k','OCR','文字识别']
author: Jiyang Qi
---

CRNN是2015年提出的一种，端对端的，场景文字识别方法，它采用CNN与RNN的结合来进行学习。它相对于其他算法主要有以下两个特点：
1. 端对端训练，直接输入图片给出结果，而不是把多个训练好的模型进行组合来识别
2. 不需要对图片中的文字进行分割就可以进行识别，可以适应任意长度的序列

原论文在[这里](https://arxiv.org/abs/1507.05717)  
PS:**是CRNN，不是RCNN**，RCNN是一种物体检测算法，别混了。。

本文将重点介绍CRNN原理，以及如何用pytorch实现CRNN，并在IIIT-5k数据集上进行尝试

# CRNN解析与构建
首先让我们看看CRNN的网络总体架构，如下图：
![](/images/network_architecture.png)
自底向上步骤为：
1. 通过卷积层提取图像特征
2. 循环层，预测下一帧的字母
3. 转录，将预测序列转化为字母，得到单词

对于输入的图片，图片首先通过CNN网络，得到特征图。之后，如何将这个特征图送入RNN呢？CRNN将特征图的每一列像素作为一个特征向量，所有列组成一个特征序列，这一序列将作为RNN的输入，即RNN第i个特征向量为特征图第i列，如下图所示。
![](/images/receptive_field.png)
图中 Feature Sequence 就是特征序列， Receptive Field 就代表原输入图像中的一列（感受野），他们一一对应，且相对位置不变。即原图像上从左到右的每一列，映射到特征序列上，依然保持原来从左到右的顺序。因此特征序列就可以认为是原图像的一种表示。

也正因为这样一种机制，图片的宽度不一定相同，但高度必须相同。为了方便，我们可以调整输入的图片的高度为32，来保证卷积后得到的特征图的每一列都只有一个像素。

CRNN具体的网络结构如下：

注意：为了与论文保持一致，本文的宽高结构均用**宽 × 高**来表示，三维张量格式为**宽 × 高 × 通道数**  
*其中k表示卷积核大小(kernel\_size)，s表示步长(stride)，p表示填充(padding\_size)*

| Type | Configurations | Output Size |
| :---: | :---: | :---: |
| Input | W × 32 gray-scale image | W × 32 × 1 |
| Convolution | #maps:64, k:3 × 3, s:1, p:1 | W × 32 × 64 |
| MaxPooling | Window:2 × 2, s:2 | W/2 × 16 × 64 |
| Convolution | #maps:128, k:3 × 3, s:1, p:1 | W/2 × 16 × 128 |
| MaxPooling | Window:2 × 2, s:2 | W/4 × 8 × 128 |
| Convolution | #maps:256, k:3 × 3, s:1, p:1 | W/4 × 8 × 256 |
| Convolution | #maps:256, k:3 × 3, s:1, p:1 | W/4 × 8 × 256 |
| MaxPooling | Window:1 × 2, s:2 | W/4 × 4 × 256 |
| Convolution | #maps:512, k:3 × 3, s:1, p:1 | W/4 × 4 × 512 |
| BatchNormalization | - | W/4 × 4 × 512 |
| Convolution | #maps:512, k:3 × 3, s:1, p:1 | W/4 × 4 × 512 |
| BatchNormalization | - | W/4 × 4 × 512 |
| MaxPooling | Window:1 × 2, s:2 | W/4 × 2 × 512 |
| Convolution | #maps:512, k:2 × 2, s:1, p:0 | W/4-1 × 1 × 512 |
| Map-to-Sequence | - | W/4-1 × 512 |
| Bidirectional-LSTM | #hidden units:256 | W/4-1 × 256 |
| Bidirectional-LSTM | #hidden units:256 | W/4-1 × label_num |
| Transcription | - | str |

下面我们把每个步骤分开来看

### 卷积
从上表的配置可以看出，卷积层很像VGG-11。不同的地方主要有两个：
1. 增加了批归一化层
2. 池化层的大小从正方形变成了长方形

加入批归一化层可以加快训练。而用高为2宽为1的长方形更容易获取窄长英文字母的特征，这样更容易区分像i和l这样的字母。
### 循环
循环层采用深度双向LSTM模型，想多了解LSTM的朋友可以看一下[这个博客](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

了解了以上两个部分以后，我们就可以开始构建我们的CRNN网络了。
```python
import torch
import torch.nn as nn
import numpy as np

class CRNN(nn.Module):
    """
    CRNN模型

    Args:
        in_channels (int): 输入的通道数，如果是灰度图则为1，如果没有灰度化则为3
        out_channels (int): 输出的通道数（类别数），即样本里共有多少种字符
    """
    def __init__(self, in_channels, out_channels):
        super(CRNN, self).__init__()
        self.in_channels = in_channels
        hidden_size = 256
        # CNN 结构与参数
        self.cnn_struct = ((64, ), (128, ), (256, 256), (512, 512), (512, ))
        self.cnn_paras = ((3, 1, 1), (3, 1, 1),
                          (3, 1, 1), (3, 1, 1), (2, 1, 0))
        # 池化层结构
        self.pool_struct = ((2, 2), (2, 2), (2, 1), (2, 1), None)
        # 是否加入批归一化层
        self.batchnorm = (False, False, False, True, False)
        self.cnn = self._get_cnn_layers()
        # RNN 两层双向LSTM。pytorch中LSTM的输出通道数为hidden_size * num_directions,这里因为是双向的，所以num_directions为2
        self.rnn1 = nn.LSTM(self.cnn_struct[-1][-1], hidden_size, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True)
        # 最后一层全连接
        self.fc = nn.Linear(hidden_size*2, out_channels)
        # 初始化参数，不是很重要
        self._initialize_weights()

    def forward(self, x):           # input: height=32, width>=100
        x = self.cnn(x)             # batch, channel=512, height=1, width>=24
        x = x.squeeze(2)            # batch, channel=512, width>=24
        x = x.permute(2, 0, 1)      # width>=24, batch, channel=512
        x = self.rnn1(x)[0]         # length=width>=24, batch, channel=256*2
        x = self.rnn2(x)[0]         # length=width>=24, batch, channel=256*2
        l, b, h = x.size()
        x = x.view(l*b, h)          # length*batch, hidden_size*2
        x = self.fc(x)              # length*batch, output_size
        x = x.view(l, b, -1)        # length>=24, batch, output_size
        return x

    # 构建CNN层
    def _get_cnn_layers(self):
        cnn_layers = []
        in_channels = self.in_channels
        for i in range(len(self.cnn_struct)):
            for out_channels in self.cnn_struct[i]:
                cnn_layers.append(
                    nn.Conv2d(in_channels, out_channels, *(self.cnn_paras[i])))
                if self.batchnorm[i]:
                    cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if (self.pool_struct[i]):
                cnn_layers.append(nn.MaxPool2d(self.pool_struct[i]))
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
```
上面网络结构的定义可能有点不是很一目了然，但是感觉这样的代码比较容易维护，而且可复用。

### 转录
网络构建完了，接下来是我们最后的转录过程。在实际模型的训练中，我们需要计算损失，然后根据损失来更新参数。这里我们要用到的损失函数是CTC Loss，这一损失函数比较适合用于我们这种序列数据。

在我写这篇博客时pytorch官方还没有提供计算CTC Loss的API，但是pytorch开发人员已经基于百度的warp-ctc，实现了其pytorch版本，这就是我们本次要用的库。（当然你也可以选择其他库，不过缺点就是其他库速度会慢）

**另：在[我的github上](https://github.com/qjy981010/CRNN.IIIT-5K.pytorch)现以提供了warp-ctc编译好的库，和直接使用方法，不想折(bai)腾(fei)源(li)码(qi)的同学可以用那种方法，然后就可以跳过下面的安装部分了：）**

我们需要手动编译安装这个库，安装的过程可能会非常坑，大家要有耐心。下面的步骤如果出现奇怪的问题，可以看一下[这个库的pytorch-binding的README](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)，或者[原百度库的issue](https://github.com/baidu-research/warp-ctc/issues)
```bash
# 先把库clone下来
git clone https://github.com/SeanNaren/warp-ctc
```

要注意，这个库比较坑的第一点是，编译时必须保证gcc版本在6.0以下，如何降级请大家参考自己Linux发行版的教程。
```bash
cd warp-ctc
mkdir build
cd build
cmake .. # 有CUDA的朋友这一步应该能检测到CUDA
make
```

编译完以后，你需要把你的gcc版本还原回去，不然后面会出问题。然后，你要把`CUDA_HOME`这个环境变量设为你CUDA的安装位置，比如大部分人的安装位置应该是在`/usr/local/cuda`，archlinux是在`/opt/cuda`，所以把下面一句加到`~/.bashrc`
```bash
export CUDA_HOME="path/to/your/cuda"
```

然后开始安装pytorch依赖；
```bash
cd pytorch_binding
python setup.py install
```

然后为了保证warpctc_pytorch能被找到，将下面一行加到`~/.bashrc`
```bash
export LD_LIBRARY_PATH='/path/to/your/python3.6/site-packages/warpctc_pytorch'
```
然后可以试着import一下看能不能用：
```python
from warpctc_pytorch import CTCLoss
```
不OK的话可以看我github上的`README.md`，或者上面给的两个链接，或者根据报错适当的改一改他的源码。OK的话就非常棒了。

损失函数有了，但是我们数据集中的标签是字符串，这些字符串是无法直接计算损失的，想将他们转化为网络能用的真正的label，我们要将其按一定的格式编码为数字来进行训练。最后从网络中得到结果后，我们又要将这个结果解码，才得到我们想要的字符串。这个解码的过程就是最后的Transcription。

首先我们要知道有哪些字符需要我们编码，在IIIT-5K中，我们的label中的字符有A-Z，0-9，还有别忘了空字符。一共37个。

我们用一个类来实现编码解码。要注意的是，因为我们所用的warpctc库的实现中，默认将空字符编码为0，所以我们要为其余字符设置从1开始的编码。
```python
class LabelTransformer(object):
    """
    字符编码解码器

    Args:
        letters (str): 所有的字符组成的字符串
    """
    def __init__(self, letters):
        self.encode_map = {letter: idx+1 for idx, letter in enumerate(letters)}
        self.decode_map = ' ' + letters

    def encode(self, text):
        if isinstance(text, str):
            length = [len(text)]
            result = [self.encode_map[letter] for letter in text]
        else:
            length = []
            result = []
            for word in text:
                length.append(len(word))
                result.extend([self.encode_map[letter] for letter in word])
        return torch.IntTensor(result), torch.IntTensor(length)

    def decode(self, text_code):
        result = []
        for code in text_code:
            word = []
            for i in range(len(code)):
                if code[i] != 0 and (i == 0 or code[i] != code[i-1]):
                    word.append(self.decode_map[code[i]])
            result.append(''.join(word))
        return result
```
这样我们的CRNN的基本流程就搞定了，接下来我们在IIIT-5K上试一试。

# 加载数据
数据集在[这里](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)下载。

数据集下载下来是.mat文件，还好我大Python有专门的库来加载.mat。  
*默默说一句：内存小的小朋友一定要谨慎行事，如果内存只有4G的话（像我一样）就要小心了*
```python
import scipy.io as sio
data = sio.loadmat('traindata.mat')
```
可以先观察一波数据集，对训练集来说，有用的数据在`data['traindata'][0]`，一共2000条数据，测试集有3000条。其中，每条数据里存的有四项，第一项是图片的文件名，第二项是label（真实标签），第三项第四项分别是大小为50，和1000的字典。数据中的字典十分占内存，他们可以用在转录中过程，本文中并没有使用他们。

pytorch中没有找到现成的API来加载这样的数据，那么我们怎么把数据加载进来呢？比较优雅的做法是继承`torch.utils.data.Dataset`类，在继承这个类时，必须要重载的方法是`__len__`和`__getitem__`。
- `__len__`使我们的类支持Python内置的`len`函数
- `__getitem__`用来支持取下标运算

同时，我们要注意，CRNN要求传入的图片高度相同，宽度至少为100，比较合适的高度是32。所以我们在这里自己定义一个类用来对图片做缩放。类的定义方法参考`torchvision.transforms`中的类，如下，只需要重载`__call__`即可。
```python
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FixHeightResize(object):
    """
    对图片做固定高度的缩放
    """
    def __init__(self, height=32, minwidth=100):
        self.height = height
        self.minwidth = minwidth

    # img 为 PIL.Image 对象
    def __call__(self, img):
        w, h = img.size
        width = max(int(w * self.height / h), self.minwidth)
        return img.resize((width, self.height), Image.ANTIALIAS)


class IIIT5k(Dataset):
    """
    用于加载IIIT-5K数据集，继承于torch.utils.data.Dataset

    Args:
        root (string): 数据集所在的目录
        training (bool, optional): 为True时加载训练集，为False时加载测试集，默认为True
        fix_width (bool, optional): 为True时将图片缩放到固定宽度，为False时宽度不固定，默认为False
    """
    def __init__(self, root, training=True, fix_width=False):
        super(IIIT5k, self).__init__()
        data_str = 'traindata' if training else 'testdata'
        self.img, self.label = zip(*[(x[0][0], x[1][0]) for x in
            sio.loadmat(os.path.join(root, data_str+'.mat'))[data_str][0]])

        # 图片缩放 + 转化为灰度图 + 转化为张量
        transform = [transforms.Resize((32, 100), Image.ANTIALIAS)
                     if fix_width else FixHeightResize(32)]
        transform.extend([transforms.Grayscale(), transforms.ToTensor()])
        transform = transforms.Compose(transform)

        # 加载图片
        self.img = [transform(Image.open(root+'/'+img)) for img in self.img]

    # 以下两个方法必须要重载
    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]
```
这样就可以像调用`torchvision.datasets`里的数据集一样方便的调用我们的IIIT-5k了。这里我把图片的加载写在了`__init__`中，内存消耗较大，大家也可以将图片加载写在`__getitem__`中，节省内存，不过速度难免会慢一些。在刚才的类里，我们还给IIIT-5K加了一个`fix_width`参数，至于为什么我们后面会讲。

pytorch提供了一个`DataLoader`类。将我们之前定义的IIIT5k类的实例传入这个类，可以很方便的加载数据，支持多线程、数据打乱、批训练，何乐而不为呢。

其中，批训练可以明显加快训练过程。不过令人心凉的是，在用DataLoader进行批训练时，pytorch默认会将batch中的张量连接起来，而宽度不固定的图片是不能直接连接的。一个方便的做法是直接将所有图片缩放成统一大小的图片，这就是为什么我们上面加了`fix_width`这样一个参数。否则我们就只能一张一张的训练了。

为了加快以后加载数据的过程，可以将我们的IIIT-5k实例存入`.pkl`文件。这样以后加载数据时，省内存，加载更是一秒加载完。
```python
import pickle
from torch.utils.data import DataLoader

def load_data(root, training=True, fix_width=False):
    """
    用于加载IIIT-5K数据集，继承于torch.utils.data.Dataset

    Args:
        root (string): 数据集所在的目录
        training (bool, optional): 为True时加载训练集，为False时加载测试集，默认为True
        fix_width (bool, optional): 为True时将图片缩放到固定宽度，为False时宽度不固定，默认为False

    Return:
        加载的训练集或者测试集
    """
    if training:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(root, 'train'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=True, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
    else:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(root, 'test'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=False, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
```

# 开始训练

有了上面这些，我们就可以开始训练了。优化方法采用Adadelta，对这类自适应优化算法感兴趣的可以看[我的另一篇博客](https://qjy981010.github.io/2017/12/23/%E8%87%AA%E9%80%82%E5%BA%94%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E6%80%BB%E7%BB%93/)。（Adadelta算法本身并不需要学习速率，但pytorch给他增加了lr这一参数，这个lr其实就是每次迭代时在参数变化量前乘的系数，默认为1，当作学习速率用即可，但在我这里测试时，lr=1时效果不好，于是改用了0.1）。在固定宽度时，lr设为0.1，速度很快。
```python
import torch.optim as optim
from torch.autograd import Variable

def train(root, start_epoch, epoch_num, letters, net=None, lr=0.1, fix_width=False):
    """
    训练CRNN

    Args:
        root (str): 存放数据集的文件夹
        start_epoch (int): 开始训练的是第多少次epoch，便于对训练过程的追踪回顾。
        epoch_num (int): 将训练的epoch数目
        letters (str): 所有的字符组成的字符串
        net (CRNN, optional): 之前训练过的网络
        lr (float, optional): 学习速率，默认为0.1，这里注意adadelta本身没有学习速率
                              pytorch增加了这一参数作为每次迭代参数改变量的系数，一般为1，但设为1时测试效果并不好。
        fix_width (bool, optional): 是否固定宽度，默认固定

    Returns:
        CRNN: 训练好的模型
    """
    # 加载数据
    trainloader = load_data(root, training=True, fix_width=fix_width)
    # 判断GPU是否可用
    use_cuda = torch.cuda.is_available()
    if not net:
        # 如果没有之前训练好的模型，就新建一个
        net = CRNN(1, len(letters) + 1)
    # 损失函数
    criterion = CTCLoss()
    # 优化方法采用Adadelta
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    # 构建编码解码器
    labeltransformer = LabelTransformer(letters)

    print('====   Training..   ====')
    # .train() 对批归一化有一定的作用
    net.train()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        print('----    epoch: %d    ----' % (epoch, ))
        loss_sum = 0
        for i, (img, label) in enumerate(trainloader):
            label, label_length = labeltransformer.encode(label)
            if use_cuda:
                img = img.cuda()
            img, label = Variable(img), Variable(label)
            label_length = Variable(label_length)
            # 清空梯度
            optimizer.zero_grad()
            # 将图片输入
            outputs = net(img)
            output_length = Variable(torch.IntTensor([outputs.size(0)]*outputs.size(1)))
            # 计算损失
            loss = criterion(outputs, label, output_length, label_length)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            loss_sum += loss.data[0]
        print('loss = %f' % (loss_sum, ))
    print('Finished Training')
    return net
```

为了验证我们模型的效果，还要定义一个测试函数。
```python
def test(root, net, letters, fix_width=True):
    """
    测试CRNN模型

    Args:
        root (str): 存放数据集的文件夹
        letters (str): 所有的字符组成的字符串
        net (CRNN, optional): 训练好的网络
        fix_width (bool, optional): 是否固定宽度，默认固定
    """
    # 加载数据
    testloader = load_data(root, training=False, fix_width=fix_width)
    # 判断GPU是否可用
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    # 构建编码解码器
    labeltransformer = LabelTransformer(letters)

    print('====    Testing..   ====')
    # .eval() 对批归一化有一定的作用
    net.eval()
    correct = 0
    for i, (img, origin_label) in enumerate(testloader):
        if use_cuda:
            img = img.cuda()
        img = Variable(img)

        outputs = net(img) # length × batch × num_letters
        outputs = outputs.max(2)[1].transpose(0, 1)  # batch × length
        outputs = labeltransformer.decode(outputs.data)
        correct += sum([out == real for out, real in zip(outputs, origin_label)])
    # 计算准确率
    print('test accuracy: ', correct / 30, '%')
```

还有最后的main函数。
```python
def main(training=True, fix_width=False):
    """
    主函数，控制train与test的调用以及模型的加载存储等

    Args:
        training (bool, optional): 为True是训练，为False是测试，默认为True
        fix_width (bool, optional): 是否固定图片宽度，默认为False
    """
    file_name = ('fix_width_' if fix_width else '') + 'crnn.pkl'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    root = 'data/IIIT5K/'
    if training:
        net = None
        start_epoch = 0
        epoch_num = 2 # 每训练两个epoch进行一次测试
        lr = 0.1
        if os.path.exists(file_name):
            print('Pre-trained model detected.\nLoading model...')
            start_epoch, net = pickle.load(open(file_name, 'rb'))
        if torch.cuda.is_available():
            print('GPU detected.')
        for i in range(5):
            net = train(root, start_epoch, epoch_num, letters, net=net, lr=lr, fix_width=fix_width)
            start_epoch += epoch_num
            test(root, net, letters, fix_width=fix_width)
        # 将训练的epoch数与我们的模型保存起来，模型还可以加载出来继续训练
        pickle.dump((start_epoch, net), open(file_name, 'wb'), True)
    else:
        start_epoch, net = pickle.load(open(file_name, 'rb'))
        test(root, net, letters, fix_width=fix_width)
```

终于，我们可以愉快的训练了：）
```python
if __name__ == '__main__':
    main(training=True, fix_width=False)
```
全部代码在[我的github上](https://github.com/qjy981010/CRNN.IIIT-5K.pytorch)，欢迎issue和star！！

按论文上的说法，在IIIT-5K数据集上，无字典训练可以达到70%的准确率。我在测试时，如果固定图片高度进行批训练，速度就非常快了，学习速率设为0.1，很快就能把准确率提升到50%左右。不过毕竟数据太少，还没能达到论文的效果。

如果出现`Out of Memory`这类错误，请降低加载数据是的`batch_size`和`num_workers`。