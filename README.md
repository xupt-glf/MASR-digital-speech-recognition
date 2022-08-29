# 纯数字语音验证码识别研发
##  1. 纯数字语音识别算法简介
基于开源项目MASR深度学习算法进行开发，采用了主流的Pytorch深度学习框架对算法进行了实现。MASR使用的是门控卷积神经网络（Gated Convolutional Network），网络结构类似于Facebook在2016年提出的Wav2letter。但是使用的激活函数不是`ReLU`或者是`HardTanh`，而是`GLU`（门控线性单元）。因此称作门控卷积网络。采用卷积神经网络（CNN）、门控机制（GLU）和连接性时序分类（CTC）的方法，使用中文数字语音数据集进行训练，将数字语音转录为拼音，并通过语言模型，将拼音序列转换为数字。

训练方法：使用无重复纯语音数据集训练MASR模型，总迭代次数设置为：200。训练的硬件环境为：NVIDIA TITAN RTX显卡，显存大小24G。保存训练过程中在验证集上表现最优的模型为最终模型，模型大小为108.7M。

模型评价指标：采用语音识别中常用的字错误率CER来衡量模型的性能，CER=编辑距离/句子长度，越低越好，大致可以理解1-CER为识别准确率。
##  2. 数据集
基于Q学友重复语音数据集和无重复语音数据集进行研发，为了排除异常语音数据的干扰，预先对语音数据集进行数据处理。针对无重复语音数据集，剔除了时长小于2秒或大于15秒的语音数据，针对重复语音数据集，剔除了时长小于3秒或大于30秒的语音数据。最终获得共116,351条语音数据，按照70%、15%、15%划分训练集、验证集、测试集，具体划分：训练集：81,445、验证集：17,453、测试集：17,453。
## 3. 实验环境
本项目所有实验均在Ubuntu 18.04操作系统下完成和实现，使用了基于Python3.7的TensorFlow深度学习框架，英伟达显卡驱动版本为Driver Version: 440.82，CUDA版本为10.1。下面给出了利用该算法进行推理时依赖的第三方Python库。

- Ubuntu: 18.04 lts
-  Python 3.7.8
- Pytorch 1.6.0
- NVIDIA GPU + CUDA_10.0 CuDNN_7.5

## 4. 实验结果
硬件配置为：单张NVIDIA TITAN RTX显卡，显存大小24G。训练好的模型对整个测试集进行混合并行测试。Pytorch默认动态占用整块显卡的显存。对17,453个测试样本进行逐条测试，共花费了1002.41s时长，平均每个样本的时间消耗为：57.44ms。字错误率CER=0.104，即识别准确率为1-0. 044=0.896。
如果将每个语音完全识别正确定义为正确识别，则模型测试的识别率为：预测完全正确的样本数：15,819，共测试数据总数：17,453，这样样本完全测试正确的准确率为：0.856。

## 5. 训练与测试
### Train
````
MASR基于pytorch，`MASRModel`是`torch.nn.Module`的子类。这将给熟悉`pytorch`的用户带来极大的方便。

使用MASR的训练功能需要安装以下额外的依赖，既然你浏览到了这里，这些依赖你一定能自行搞定！

* `levenshtein-python`

  计算CER中的编辑距离

* `warpctc_pytorch`

  百度的高性能CTC正反向传播实现的pytorch接口

* `tqdm`

  进度显示

* `tensorboardX`

  为pytorch提供tensorboard支持

* `tensorboard`

  实时查看训练曲线

当然，相信你也有GPU，否则训练将会变得很慢。

**通常，神经网络的训练比搭建要困难得多，然而MASR为你搞定了所有复杂的东西，使用MASR进行训练非常方便。**

如果你只想要使用MASR内置的门卷积网络`GatedConv`的话，首先初始化一个`GatedConv`对象。

```python
from models.conv import GatedConv

model = GatedConv(vocabulary)
```

你需要传入向它`vocabulary`，这是一个字符串，包含你的数据集中所有的汉字。但是注意，`vocabulary[0]`应该被设置成一个无效字符，用于表示CTC中的空标记。

之后，使用`to_train`方法将`model`转化成一个可以训练的对象。

```python
model.to_train()
```

此时`model`则变成可训练的了，使用`fit`方法来进行训练。

```python
model.fit('train.index', 'dev.index', epoch=10)
```

`epoch`表示你想要训练几次，而`train.index`和`dev.index`应该分别为训练数据集和开发数据集（验证集或测试集）的索引文件。

索引文件应具有如下的简单格式：

```python
/path/to/audio/file0.wav,我爱你
/path/to/audio/file1.wav,你爱我吗
...
```

左边是音频文件路径，右边是对应的标注，用逗号（英文逗号）分隔。

`model.fit`方法还包含学习率、batch size、梯度裁剪等等参数，可以根据需要调整，建议使用默认参数。

完整的训练流程参见[train.py](/examples/train.py)。
````
### Test
````
识别自己的语音需要额外安装一个依赖：pyaudio

参考[pyaudio官网](https://people.csail.mit.edu/hubert/pyaudio/)把它装上，然后执行以下命令即可。

```sh
python examples/demo-record-recognize.py
```

请在看到提示「录音中」后开始说话，你有5秒钟的说话时间（可以自己在源码中更改）。
````
