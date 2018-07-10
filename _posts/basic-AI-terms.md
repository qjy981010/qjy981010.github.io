---
title: 基本AI术语
date: 2018-07-09 09:26:33
tags: ['人工智能','AI','术语','基本']
author: Jiyang Qi
mathjax: true
---

# 基本问题

#### 极大似然与负对数损失

极大似然中，一组参数在一堆数据下的似然值，等于每一条数据在这组参数下的条件概率之积。

而我们通过模型得到的置信度，其实就是那个条件概率，所以我们的目标是让他们的积最大。

但损失函数一般是每条数据的损失之和，为了把和变为上面的积，就对每一项取了对数。让对数之和最大，其实就是让负对数之和最小。于是我们便有了负对数损失(NLL Loss)。

$$L(y) = -\log(\hat{y_i})$$

_pytorch中的NLLLoss本身不进行对数运算，只做一个取负_

#### 负对数损失与交叉熵损失的关系

交叉熵损失定义如下，其中$Y$为真实label(0/1)，$\hat{Y}$为预测的结果

$$H(Y,\hat{Y})=-Y\cdot \log (\hat{Y})=-\sum_i y_i \cdot \log (\hat{y_i})$$

将最右边的式子展开，得到的其实就是负对数损失。


#### softmax与交叉熵

softmax中的$e^x$与交叉熵中的$\log(x)$抵消，使模型易于收敛。  
因此softmax通常作为输出层的激活函数

_pytorch中的CrossEntropyLoss自带softmax_


#### 为什么要用softmax

1. 当使用交叉熵损失时softmax中的$e^x$与交叉熵中的$\log(x)$抵消，使模型易于收敛。
2. softmax满足最大熵原则。


# 基础术语

下面主要记录一些基础的术语

## 信息论&概率论

#### 熵(Entropy)

熵在信息论中可以理解为，一个事件包含的信息量。熵越小，事件包含的信息量就越小。必然事件和不可能事件的熵为0（比如“我是我妈生的”这句话为必然事件，不包含任何信息，信息量为0）。熵定义如下：

$$S(x) = -\sum _{i}P(x_{i})log_{b}P(x_{i})$$

其中S为熵，x为事件，当x必然发生或必然不发生时，$P(x)$为1或0，此时S为0。


#### KL散度/KL距离/相对熵

KL散度一般用来计算两个概率分布之间的距离，但它与普通的距离计算不同，因为'A对B的KL距离'不一定等于'B对A的KL距离'。

对离散事件： $D_{KL}(A||B) = \sum _{i}P_{A}(x_i) log\bigg(\frac{P_{A}(x_i)}{P_{B}(x_i)} \bigg)$

对连续事件： $D_{KL}(A||B) = \int a(x) log\bigg(\frac{a(x)}{b(x)} \bigg)$

如果A与B的概率分布相同，即$P_A=P_B$，则距离为0。


#### 交叉熵

交叉熵同样用来衡量两个分布之间的差异，其定义如下：

$$H(A,B)= -\sum _{i}P_{A}(x_i)log(P_{B}(x_i))$$

可以推出来 $D_{KL}(A||B) = -S(A)+H(A,B)$


#### 交叉熵与KL散度的关系

- 相同点：a. 都不具备对称性 b. 都是非负的
- 当$S(A)$固定时，最小化KL散度与最小化交叉熵等价。而交叉熵公式更简洁，因此我们一般都使用交叉熵。

在机器学习中，我们要让模型的分布逼近数据集的分布，而数据集就是上面的A，因此$S(A)$固定



## 距离度量

#### 欧氏距离(Euclidean distance)

$$d_{12} = \sqrt{(x_1-x_2)^2+(y_1-y_2)^2}$$


#### 曼哈顿距离(Manhattan Distance)

$$d_{12} = |x_1-x_2|+|y_1-y_2|$$


#### 闵可夫斯基距离(Minkowski Distance)

可以看做是欧氏距离和曼哈顿距离的一种推广  

$$d_{12} = \sqrt[p]{(x_1-x_2)^p+(y_1-y_2)^p}$$

当$p=1$时，就是曼哈顿距离  
当$p=2$时，就是欧氏距离  
当$p \to \infty$时，就是切比雪夫距离  


#### 马氏距离(Mahalanobis Distance)

前面的距离度量均会受到变量量纲的影响，而马氏距离通过对数据的仿射变换，消除了这种影响。这是它的优点，而在某些情况下也会成为它的缺点。

定义：有m个样本向量$X_1, X_m$，协方差矩阵记为S，则其中向量$X_i$与$X_j$之间的马氏距离定义为：

$$D(X_i,X_j) = \sqrt{(X_i-X_j)^TS^{-1}(X_i-X_j)}$$


#### 编辑距离(Levenshtein Distance)

用来定义两个字符串之间的距离，指两个字串之间，由一个转成另一个所需的最少编辑操作次数。Standardization字符，删除字符。


## 数据预处理

#### 归一化(Normalization)

通过对原始数据进行线性变换把数据映射到[0,1]之间

$$x' = \frac{x-min}{max-min}$$


#### 标准化(Standardization)

使数据均值为0，标准差为1

$$x' = \frac{x-\mu}{\sigma}$$


## 评价标准

首先定义基本的评价标准

|  | 预测为真 | 预测为假 |
| --- | --- | --- |
| 实际为真 | TP(真正例) | FN(假反例) |
| 实际为假 | FP(假正例) | TN(真反例) |

接下来的评价标准基本都基于上面这四个

#### 精确率/查准率(precision)

$$P = \frac{TP}{TP+FP}$$

#### 召回率/查全率(recall)

$$R = \frac{TP}{TP+FN}$$

#### 准确率(accuracy)

$$Acc = \frac{TP+TN}{TP+FP+TN+FN}$$

#### F1 Score

通常我们需要综合精确率和召回率进行考虑，这时我们一般使用F1度量  

$$F_1 = \frac{2 \times P \times R}{P + R} =  \frac{2 \times TP}{2 \times TP + FP + FN}$$

当对精确率和召回率的重视程度不同时，我们可以采用更一般的$F_\beta$

$$F_\beta = \frac{(1+\beta^2) \times P \times R}{(\beta^2 \times P) + R}$$

#### P-R曲线

以召回率为横轴，精确率为纵轴，得到的曲线为P-R曲线。

<div align=center>
![](/images/P-R)
</div>

当对两个模型进行比较时，我们可以根据他们P-R曲线下的面积(AP)判断，面积越大性能越好。

#### ROC曲线

先定义真正例率和假正例率：

$$TPR = \frac{TP}{TP+FN}$$

$$FPR = \frac{FP}{TN+FP}$$

以FPR为横轴，TPR为纵轴即可得到ROC曲线。

<div align=center>
![](/images/roc.png)
</div>

#### AUC

ROC曲线下的面积即为AUC，越大说明模型越好。__可以无视正负样本不平衡的影响__

#### mAP(mean average precision)

常用于多类别物体检测。对每个类别均可以绘制一条P-R曲线，其中AP即指P-R曲线下的面积。mAP即为多个类别AP的平均值。

## 奇怪的卷积

#### 反卷积/转置卷积 (deconvolution / transposed convolution)

可以理解为一种特殊的卷积，只不过它会将feature map放大，达到上采样的效果。  

<div align=center>
![](/images/no_padding_no_strides_transposed.gif)
</div>

上图可以理解3x3的卷积核，padding=2，stride=1的卷积操作。而同时，在特征图大小变化上，与padding=0，stride=1的卷积操作相反。  
用pytorch实现时代码如下，其中kernel_size为3，stride与padding按其对应的卷积操作参数进行设置。

```python
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

当卷积操作stride大于1时，此时其对应的转置卷积的stride小于1，此时被称为分数卷积(fractionally-strided convolution)，如下图：

<div align=center>
![](/images/no_padding_strides_transposed.gif)
</div>

[for more .gif](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)


#### 空洞卷积/扩张卷积(Dilated Convolutions)

空洞卷积可以在基本不增加计算量的前提下，增大感受野。之前如果需要扩大感受野，都是采用池化，但池化会造成分辨率下降。

<div align=center>
![](/images/dilation.gif)
</div>

为了不损失数据的连续性，我们需要选择适当的各层空洞卷积参数，如下图的配置方案。

<div align=center>
![](/images/dilated.png)
</div>

pytorch中实现有一个空洞的卷积操作，代码如下：

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=2)
```

其中dilation参数默认为1，为1在pytorch中定义为没有空洞。不同框架中对此参数的定义不同，需要注意。
