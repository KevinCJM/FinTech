import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置打印选项
# np.set_printoptions(precision=8, suppress=True)


def the_mse_loss(w, x, y):
    y_pred = x.dot(w)
    loss = np.sum(0.5 * (y_pred - y) ** 2) / len(y)
    return loss


def run_regression(x_value, y_value, iterations=1000, initial_learning_rate=5.0, min_learning_rate=0.99):
    """
    运行回归
    :param x_value: x值
    :param y_value: y值
    :param iterations: 迭代次数
    :param initial_learning_rate: 初始学习率
    :param min_learning_rate: 最小学习率
    :return: 权重
    """

    # 定义损失函数
    def mse_loss(x, y, w_i):
        n = len(y)
        y_pred = x.dot(w_i)
        loss = (1 / n) * np.sum(0.5 * (y_pred - y) ** 2)
        return loss

    # 定义梯度计算函数
    def gradients(x, y, w_i):
        n = len(y)
        y_pred = x.dot(w_i)
        the_dw = (1 / n) * np.dot(x.T, (y_pred - y))  # 梯度
        return the_dw

    learning_rate = initial_learning_rate  # 设置初始学习率
    weight_history = []  # 保存权重历史
    loss_history = []  # 保存损失历史
    # 初始化权重
    w = np.linalg.inv(x_value.T.dot(x_value)).dot(x_value.T).dot(y_value)
    w = np.maximum(w, 0)  # 确保权重非负
    # w = np.random.rand(x_value.shape[1])
    if w.sum() == 0:
        # 如果权重和为0，则重新初始化权重
        w = np.random.rand(x_value.shape[1])
    w /= w.sum()  # 确保初始权重和为1
    print(w)

    ''' 梯度下降方法求解 '''
    for i in range(iterations):
        # 计算当前学习率
        learning_rate = max(min_learning_rate, learning_rate - (initial_learning_rate - min_learning_rate) / iterations)
        dw = gradients(x_value, y_value, w)
        w -= learning_rate * dw
        # 重新标准化权重以满足约束
        w = np.maximum(w, 0)  # 确保权重非负
        w = np.minimum(w, 1)  # 权重小于1
        if w.sum() == 0:
            # 如果权重和为0，则重新初始化权重
            w = np.random.rand(x_value.shape[1])
        w /= w.sum()  # 确保权重和为1
        weight_history.append(w.copy())
        # 计算损失
        loss_history.append(mse_loss(x_value, y_value, w))

    # 找到最小的损失
    min_loss_index = np.argmin(loss_history)
    w = weight_history[min_loss_index]

    return w, loss_history, weight_history


def run_minimize(x_value, y_value):
    # 定义损失函数
    def mse_loss(w, x, y):
        y_pred = x.dot(w)
        loss = np.sum(0.5 * (y_pred - y) ** 2) / len(y)
        return loss

    # 自变量数量数
    num_of_x = x_value.shape[1]

    # 初始权重 - 计算无约束下的解析解
    initial_w = np.linalg.inv(x_value.T.dot(x_value)).dot(x_value.T).dot(y_value)
    print(initial_w)
    # initial_w = np.random.rand(num_of_x)
    initial_w /= np.sum(initial_w)  # 初始化使得权重和为1

    bnds = [(0, 1)] * num_of_x  # 动态创建边界条件
    # bnds = [(None, None)] * num_of_x  # 动态创建边界条件
    # bnds[1:] = [(0, 1)] * (num_of_x - 1)  # 除了第一个权重之外，其余权重非负

    # 约束条件：权重和为1
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    result = minimize(mse_loss, initial_w, args=(x_value, y_value), method='SLSQP',
                      bounds=bnds, constraints=cons, tol=1e-15)

    if result.success:
        # optimized_weights = result.x
        # print('优化后的权重: ', optimized_weights)
        # print('在优化后的权重下的损失: ', mse_loss(optimized_weights, x_value, y_value))
        return result.x
    else:
        return None


if __name__ == '__main__':
    # 设置随机种子，确保可复现性
    np.random.seed()
    # 数据数量
    num_of_data = 2520
    num_of_funds = 1
    num_of_index = 10

    # 生成基金的日收益率数据
    funds_returns = np.random.normal(0, 0.01, (num_of_data, num_of_funds))  # 假设均值为0，标准差为0.01
    # 生成指数的日收益率数据
    index_returns = np.random.normal(0, 0.01, (num_of_data, num_of_index))  # 假设均值为0，标准差为0.01

    # # 生成数据
    # index_returns = np.random.rand(num_of_data, 2)
    # # 给定一个拟合的y值
    # funds_returns = 0.7 * index_returns[:, 0] + 0.3 * index_returns[:, 1] + np.random.randn(num_of_data)  # 增加随机噪声

    # 将numpy数组转换为DataFrame以方便操作
    funds_returns = pd.DataFrame(funds_returns, columns=[f"Fund_{i + 1}" for i in range(num_of_funds)])
    index_returns = pd.DataFrame(index_returns, columns=[f"Index_{i + 1}" for i in range(num_of_index)])
    # # 为回归添加常数项
    # index_returns = sm.add_constant(index_returns)

    print('数据准备完毕', '=' * 20)

    # 运行回归
    s_t = time.time()
    the_res_w, l_h, w_h = run_regression(index_returns.values, funds_returns.values[:, 0],
                                     iterations=100, initial_learning_rate=2.99, min_learning_rate=0.01)
    the_res_loss = the_mse_loss(the_res_w, index_returns.values, funds_returns.values[:, 0])
    print('最终权重: ', the_res_w)
    print('权重损失: ', the_res_loss)
    print('回归耗时: ', time.time() - s_t)
    print('=' * 20)

    # 使用statsmodels库进行回归
    s_t = time.time()
    res_w = run_minimize(index_returns.values, funds_returns.values[:, 0])
    res_loss = the_mse_loss(res_w, index_returns.values, funds_returns.values[:, 0])
    print('最终权重: ', res_w)
    print('权重损失: ', res_loss)
    print('回归耗时: ', time.time() - s_t)

    # # 绘制 loss_history 图
    # plt.plot(l_h)
    # plt.show()
    # pass
