# -*- encoding: utf-8 -*-
"""
@File: 3.3_SyntheticRegressionData.py
@Modify Time: 2024/6/27 13:44       
@Author: Kevin-Chen
@Descriptions: 
"""
from mxnet import autograd, gluon, np, npx
import tools_func as d2l
from mxnet.gluon import nn  # nn是神经网络的缩写
from mxnet import init

npx.set_np()


def load_array(data_arrays, size, is_train=True):
    """
    加载数据数组并创建一个Gluon数据迭代器。

    该函数用于将给定的数据数组转换为Gluon的数据集，并根据是否处于训练模式来决定是否打乱数据。

    参数:
    - data_arrays: 一个或多个数据数组，可以是任何形式的数组，如numpy数组或列表。
    - size: 迭代器每次返回的数据批大小。
    - is_train: 是否处于训练模式，默认为True。在训练模式下，数据集会被随机打乱。

    返回:
    - 一个Gluon数据迭代器，用于在训练或测试过程中批量加载数据。
    """
    # 在MXNet框架中，gluon.data.ArrayDataset 是用于创建数据集对象的一个非常实用的类, 其接受多个同等长度的数组并将它们组合成一个数据集。
    data_set = gluon.data.ArrayDataset(*data_arrays)  # 将数据数组转换为Gluon的数据集(类对象)
    # 根据是否处于训练模式，返回一个数据加载器，数据加载器负责批量加载数据并可选择打乱数据顺序
    return gluon.data.DataLoader(data_set, size, shuffle=is_train)  # 返回一个Gluon数据迭代器


if __name__ == '__main__':
    # 生成数据集
    true_w = np.array([2, -3.4])  # 真实权重
    true_b = 4.2  # 真实偏置
    # features是自变量, labels是因变量
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 生成1000个样本

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    # 定义了一个神经网络模型
    net = nn.Sequential()   # 创建一个有序的模块序列
    net.add(nn.Dense(1))    # 向序列中添加了一个全连接层（Dense层），该层有1个输出单元。

    # 初始化模型参数
    net.initialize(init.Normal(sigma=0.01)) # 使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数

    # 定义损失函数
    loss = gluon.loss.L2Loss()  # 平方损失又称L2范数损失

    # 定义优化算法
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})   # 使用小批量随机梯度下降作为优化算法

    # 训练模型
    '''
    这段代码是一个训练神经网络的循环。简单来说，它在每个epoch（轮次）上遍历数据集，并对每个batch进行以下操作：
    使用autograd.record()记录计算过程，以便之后进行反向传播。
    1. 计算当前batch的损失函数值l。
    2. 调用l.backward()进行反向传播，计算梯度。
    3. 使用trainer.step(batch_size)更新模型参数，即进行一次优化器的迭代步。 
    4. 在每个epoch结束后，还会计算整个数据集的损失函数值，并打印出当前epoch和损失值。
    '''
    num_epochs = 3  # 设定了模型训练的总轮次（epochs）
    for epoch in range(num_epochs):
        for X, y in data_iter:  # data_iter是一个数据迭代器，每次迭代会返回一个数据样本X和对应的标签y。
            with autograd.record(): # 创建了一个自动记录执行过程的上下文, 即计算图; 在这个上下文中，执行的操作都会被记录下来，以便后续进行反向传播计算梯度。
                l = loss(net(X), y)
            l.backward()    # 用于反向传播的函数，用于计算损失函数关于模型参数的梯度。调用该函数后，系统会从输出端开始，按照计算图的反向顺序计算梯度，并将梯度存储在模型参数的.grad属性中。
            trainer.step(batch_size)    # 执行一次训练迭代，利用SGD算法根据batch_size大小的当前批次数据计算出的梯度来更新模型参数。
        l = loss(net(features), labels) # 计算当前epoch的损失函数值
        print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')

    # 比较学到的参数和真实参数
    w = net[0].weight.data()
    print(w)
    print(f'w的估计误差： {true_w - w.reshape(true_w.shape)}')
    b = net[0].bias.data()
    print(b)
    print(f'b的估计误差： {true_b - b}')