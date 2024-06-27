# -*- encoding: utf-8 -*-
"""
@File: 3.2_ObjectOrientedDesign.py
@Modify Time: 2024/6/27 10:55       
@Author: Kevin-Chen
@Descriptions: 线性回归从零实现
"""
import random
from mxnet import autograd, np, npx

npx.set_np()


# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """
    创建一个数据迭代器，用于以指定的批量大小遍历数据集。

    参数:
    batch_size -- 每个批次的数据数量。
    features -- 数据集中的特征数组。
    labels -- 数据集中的标签数组。

    返回:
    一个生成器，每次产生一个包含批量特征和批量标签的元组。
    """
    # 获取数据集中的样本数量
    num_examples = len(features)
    # 为所有样本创建一个索引列表
    indices = list(range(num_examples))
    # 随机打乱索引顺序，以便在遍历数据时增加随机性
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    # 遍历数据集，每次取出一个批量大小的样本
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引，确保不超过数据集的边界
        batch_indices = np.array(
            indices[i: min(i + batch_size, num_examples)])
        # 根据批次索引获取对应的特征和标签，并作为生成器的输出
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """
    实现线性回归模型的预测函数。

    参数:
    X: 输入数据，形状为(n_features, )的数组，其中n_features是特征的数量。
    w: 权重参数，形状为(n_features, )的数组，表示每个特征的权重。
    b: 偏置参数，标量，表示模型的截距。

    返回值:
    预测结果，即输入数据X与权重w的点积加上偏置b的结果。
    """
    """线性回归模型"""
    return np.dot(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    使用小批量随机梯度下降法更新参数。

    参数:
    - params: 待更新的参数列表，每个参数都是一个Tensor。
    - lr: 学习率，控制参数更新的速度。
    - batch_size: 每个迭代步骤中使用的样本数量。

    该函数遍历参数列表，对每个参数进行更新。更新规则是：参数值减去学习率乘以参数的梯度值除以批量大小。
    这种更新方式旨在逐步减小损失函数，从而找到最优的参数值。
    """
    """小批量随机梯度下降"""
    for param in params:
        # 直接在参数数组上进行就地更新，避免创建新的数组副本
        param[:] = param - lr * param.grad / batch_size


if __name__ == '__main__':
    # 初始化真实权重和偏差，用于后续生成模拟数据
    true_w = np.array([2, -3.4])
    true_b = 4.2

    # 生成模拟数据集，包含1000个样本
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 初始化权重w和偏差b，使用随机值作为初始猜测
    w = np.random.normal(0, 0.01, (2, 1))
    b = np.zeros(1)

    # 启用自动求导，为权重和偏差记录梯度
    w.attach_grad()
    b.attach_grad()

    # 定义批量大小
    batch_size = 10
    lr = 0.05  # 学习率
    num_epochs = 5  # 训练轮数

    # 定义线性回归模型和平方损失函数
    net = linreg
    loss = squared_loss

    # 开始训练循环
    for epoch in range(num_epochs):
        # 在每个训练轮中，遍历数据集的每个批量
        for X, y in data_iter(batch_size, features, labels):
            # 记录运算图，以便之后可以反向传播计算梯度
            with autograd.record():
                l = loss(net(X, w, b), y)  # 计算当前批量数据的损失

            # 计算[w,b]的梯度并进行反向传播
            l.backward()
            # 使用梯度下降算法更新权重和偏差
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数

        # 计算当前训练轮结束后整个数据集的损失
        train_l = loss(net(features, w, b), labels)
        # 打印当前训练轮数和对应的损失值
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')
    print(f"估计值: \n {w}, \n {b}")
