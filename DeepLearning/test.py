import time
import numpy as np

# 设置打印选项
np.set_printoptions(precision=4, suppress=True)
# 随机数据数量
num_of_data = 1000
# 设置随机种子
np.random.seed(0)

# 生成数据
x_value = np.random.rand(num_of_data, 2)
# 给定一个拟合的y值
y_value = 0.7 * x_value[:, 0] + 0.3 * x_value[:, 1] + np.random.randn(num_of_data)  # 增加随机噪声


# 定义损失函数
def mse_loss(w, x, y):
    y_pred = x.dot(w)
    loss = np.sum(0.5 * (y_pred - y) ** 2) / len(y)
    return loss


def run_regression(x_value, y_value, iterations=1000, initial_learning_rate=5.0, min_learning_rate=0.99, w=None):
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

    def mse_loss_with_penalty(w, x, y, kappa=1000):
        mse = np.mean(0.5 * (np.dot(x, w) - y) ** 2)
        penalty = kappa * (np.sum(w) - 1) ** 2  # 确保权重和为1的罚项
        return mse + penalty

    # 定义梯度计算函数
    def gradients(x, y, w_i):
        n = len(y)
        y_pred = x.dot(w_i)
        the_dw = (1 / n) * np.dot(x.T, (y_pred - y))  # 梯度
        # 计算单位向量
        # the_dw = the_dw / np.linalg.norm(the_dw)
        return the_dw

    learning_rate = initial_learning_rate  # 设置初始学习率
    weight_history = []  # 保存权重历史
    loss_history = []  # 保存损失历史
    # 初始化权重
    # w = np.random.rand(x_value.shape[1])
    if w is None:
        w = np.array([0.5, 0.5])
        w /= w.sum()  # 确保初始权重和为1
    else:
        w = w

    ''' 梯度下降方法求解 '''
    for i in range(iterations):
        # 计算当前学习率
        learning_rate = max(min_learning_rate, learning_rate - (initial_learning_rate - min_learning_rate) / iterations)
        dw = gradients(x_value, y_value, w)
        w -= learning_rate * dw
        # # 重新标准化权重以满足约束
        # w = np.maximum(w, 0)  # 确保权重非负
        # if w.sum() == 0:
        #     # 如果权重和为0，则重新初始化权重
        #     w = np.random.rand(x_value.shape[1])
        # w /= w.sum()  # 确保权重和为1
        # weight_history.append(w.copy())
        # 计算损失
        loss_history.append(mse_loss(x_value, y_value, w))

    # 找到最小的损失
    min_loss_index = np.argmin(loss_history)
    w = weight_history[min_loss_index]

    return w, loss_history, weight_history

s_t = time.time()
w, l_h, w_h = run_regression(x_value, y_value, iterations=200, initial_learning_rate=0.99, min_learning_rate=0.09)

print('批量梯度下降耗时: ', time.time() - s_t)
print('最终权重: ', w)
print('最小损失: ', mse_loss(w, x_value, y_value))
w /= w.sum()
print('归一权重: ', w)
print('归一最小损失: ', mse_loss(w, x_value, y_value))
print('=' * 50)

''' minimize方法求解 '''
# 使用minimize函数进行优化
from scipy.optimize import minimize

s_t = time.time()

# 初始权重
initial_w = np.random.rand(2)
initial_w /= np.sum(initial_w)  # 初始化使得权重和为1

bnds = ((0, None), (0, None))
# 约束条件：权重和为1
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
result = minimize(mse_loss, initial_w, args=(x_value, y_value), method='SLSQP', bounds=bnds,
                  constraints=cons, tol=1e-16)

if result.success:
    optimized_weights = result.x
    print('优化后的权重: ', optimized_weights)
    print('在优化后的权重下的损失: ', mse_loss(optimized_weights, x_value, y_value))
else:
    print('优化失败: ', result.message)

print('minimize耗时: ', time.time() - s_t)

import matplotlib.pyplot as plt
import numpy as np

# 创建权重网格
w1_range = np.linspace(-0.5, 1.5, 200)
w2_range = np.linspace(-0.5, 1.5, 200)
W1, W2 = np.meshgrid(w1_range, w2_range)
Loss = np.zeros(W1.shape)

# 计算每个组合的损失，这次不强制权重和为1
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w_test = np.array([W1[i, j], W2[i, j]])
        Loss[i, j] = mse_loss(w_test, x_value, y_value)

# 创建一个figure对象
fig = plt.figure(figsize=(20, 8))

# 第一个子图: 损失函数等高线图
ax1 = fig.add_subplot(1, 2, 1)  # 1行2列，第1个子图
contour = ax1.contourf(W1, W2, Loss, levels=100, cmap='viridis')
fig.colorbar(contour, ax=ax1)

# 绘制梯度下降路径
weight_history = np.array(w_h)
ax1.plot(weight_history[:, 0], weight_history[:, 1], 'r.-', label='Gradient Descent Path')
ax1.scatter(weight_history[-1, 0], weight_history[-1, 1], c='red', s=10, label='End Point of GD')

# 标记 minimize 方法的最优解
ax1.scatter(optimized_weights[0], optimized_weights[1], c='yellow', s=500, marker='*', label='Minimize Solution')
ax1.scatter(w[0], w[1], c='red', s=200, marker='*', label='Regression Solution')

ax1.set_title('Contour Plot of Loss Surface')
ax1.set_xlabel('Weight 1')
ax1.set_ylabel('Weight 2')
ax1.legend()

# 第二个子图: l_h 一维数组走势图
ax2 = fig.add_subplot(1, 2, 2)  # 1行2列，第2个子图
ax2.plot(l_h)
ax2.set_xlabel('times')
ax2.set_ylabel('loss')
ax2.set_title('loss change')

# 显示图像
plt.show()