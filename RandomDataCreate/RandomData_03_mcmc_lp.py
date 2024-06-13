# -*- encoding: utf-8 -*-
"""
@File: Markowitz_01_no_limit.py
@Modify Time: 2024/6/4 09:12
@Author: Kevin-Chen
@Descriptions: MCMC + 线性规划调整权重
"""

import numpy as np
import cvxpy as cp
import warnings
import time
from Markowitz_01_no_limit import show_random_weights

warnings.filterwarnings('ignore')


# 使用线性规划调整提议权重以确保满足约束条件，包括权重和为1的约束。
def linear_programming_adjustment(proposal, fund_codes_list, single_limits_list, multi_limits_dict):
    """
    使用线性规划调整提议权重以确保满足约束条件，包括权重和为1的约束。

    :param proposal: numpy array, 提议的权重数组。
    :param fund_codes_list: list, 资产代码列表。
    :param single_limits_list: list, 单个资产的权重下限和上限。
    :param multi_limits_dict: dict, 多个资产的组合权重约束。
    :return: numpy array, 调整后的权重。
    """
    num_assets = len(proposal)
    x = cp.Variable(num_assets)

    # 创建单个资产的约束
    constraints = [x >= [limit[0] for limit in single_limits_list],
                   x <= [limit[1] for limit in single_limits_list]]

    # 创建多资产的联合约束
    for codes, (lower, upper) in multi_limits_dict.items():
        indices = [fund_codes_list.index(code) for code in codes.split(',')]
        constraints.append(cp.sum(x[indices]) >= lower)
        constraints.append(cp.sum(x[indices]) <= upper)

    # 添加权重总和为1的约束
    constraints.append(cp.sum(x) == 1)

    # 目标函数：最小化与提议权重的欧几里得距离
    objective = cp.Minimize(cp.norm(x - proposal, 2))

    # 定义问题并求解
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    # 检查是否有可行解
    if result is not None and x.value is not None:
        return x.value
    else:
        return proposal  # 如果无解，则返回原提议


# 随机选择一些资产进行调整
def randomly_adjust_weights(the_current_weights, step_size, num_adjustments=1):
    """
    随机选择一些资产进行调整。

    :param the_current_weights: numpy array, 当前的权重。
    :param step_size: float, 调整的步长。
    :param num_adjustments: int, 要调整的资产数量。
    :return: numpy array, 调整后的权重。
    """
    indices = np.random.choice(len(the_current_weights), size=num_adjustments, replace=False)
    the_proposal = the_current_weights.copy()
    the_proposal[indices] += np.random.normal(0, step_size, size=num_adjustments)
    the_proposal = np.clip(the_proposal, 0, 1)
    the_proposal /= np.sum(the_proposal)
    return the_proposal


if __name__ == '__main__':
    fund_codes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    single_limits = [(0.0, 1.0), (0.1, 0.6), (0.0, 1.0), (0.1, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.1, 0.6), (0.0, 1.0), (0.0, 0.6), (0.0, 1.0)]
    multi_limits = {"001,002": (0.31, 0.31), "003,004": (0.21, 0.21), "005": (0.06, 0.06),
                    "006,007": (0.11, 0.11), "008,009": (0.11, 0.11), "010": (0.2, 0.2)}

    # 创建一个等权重的初始权重, 用了线性规划, 出发点无需在可行域内, 线性规划会把他拉到可行域上的
    num_of_asset = len(fund_codes)
    current_weights = np.array([1 / num_of_asset] * num_of_asset)
    # 随机游走的步长
    step_size = 0.01
    num_of_sample = 100

    s_t = time.time()
    final_weight = []
    for i in range(num_of_sample):
        # new_proposal = randomly_adjust_weights(current_weights, step_size, num_adjustments=1)
        new_proposal = current_weights + np.random.normal(0, step_size, len(current_weights))

        adjusted_weights = linear_programming_adjustment(new_proposal, fund_codes, single_limits, multi_limits)

        final_weight.append(adjusted_weights)
        current_weights = adjusted_weights

    final_weight = np.array(final_weight).round(6)
    print(final_weight)
    print("去重重复前的数量: ", len(final_weight))
    final_weight = np.unique(final_weight, axis=0)
    print("去重重复后的数量: ", len(final_weight))
    print("计算耗时: ", time.time() - s_t)

    ''' 画图 '''
    show_random_weights(final_weight, fund_codes)
