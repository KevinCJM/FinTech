# -*- encoding: utf-8 -*-
"""
@File: 3.3_SyntheticRegressionData.py
@Modify Time: 2024/6/27 13:44       
@Author: Kevin-Chen
@Descriptions: 
"""
from mxnet import autograd, gluon, np, npx
import tools_func as d2l

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
    data_set = gluon.data.ArrayDataset(*data_arrays)    # 将数据数组转换为Gluon的数据集(类对象)
    # 根据是否处于训练模式，返回一个数据加载器，数据加载器负责批量加载数据并可选择打乱数据顺序
    return gluon.data.DataLoader(data_set, size, shuffle=is_train)  # 返回一个Gluon数据迭代器



if __name__ == '__main__':
    # 生成数据集
    true_w = np.array([2, -3.4])    # 真实权重
    true_b = 4.2    # 真实偏置
    # features是自变量, labels是因变量
    features, labels = d2l.synthetic_data(true_w, true_b, 1000) # 生成1000个样本

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    # print(next(iter(data_iter)))
    data_iter = iter(data_iter)
    while True:
        try:
            item = next(data_iter)
            print(item)
        except StopIteration:
            break