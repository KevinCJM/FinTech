# -*- encoding: utf-8 -*-
"""
@File: Markowitz_01_no_limit.py
@Modify Time: 2024/6/4 09:12       
@Author: Kevin-Chen
@Descriptions: mcmc
"""

import time
import warnings
import numpy as np
import cvxpy as cp
import pandas as pd
from numba.typed import List
from scipy.optimize import linprog
from scipy.optimize import minimize
from pyecharts import options as opts
from numba import njit, types, prange
from pyecharts.charts import Scatter, Line
from RandomData_01_rejection_sampling import *

warnings.filterwarnings('ignore')


# 使用线性规划调整提议权重以确保满足约束条件，包括权重和为1的约束。
def linear_programming_adjustment(proposal, fund_codes_list, single_limits_list, multi_limits_dict):
    """
    使用线性规划调整提议权重以确保满足约束条件，包括权重和为1的约束。

    :param proposal: numpy array, 初始提议的权重数组, 线性规划会将可行域内的结果往这个值取靠。
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


# 使用 MCMC 方法进行采样
def mcmc_sampling_one(current_weights, single_limits_array, multi_limits_array, fund_codes_array, num_of_samples=1000,
                      step_size=0.00001):
    """
    使用Markov Chain Monte Carlo (MCMC)方法进行采样。

    参数:
    current_weights: 当前权重数组。
    single_limits_array: 单个基金的权重限制数组。
    multi_limits_array: 多个基金的权重限制数组。
    fund_codes_array: 基金代码数组。
    num_of_samples: 期望的样本数量。
    step_size: 每次移动的步长。

    返回:
    一个包含多次采样结果的numpy数组。
    """
    # 初始化采样结果列表
    the_samples = []

    # 当采样数量未达到期望值时，继续循环
    # while len(the_samples) < num_of_samples:
    for i in range(num_of_samples):
        # 提出新的权重提案
        proposal = current_weights + np.random.normal(0, step_size, len(current_weights))
        # 确保权重在0到1之间
        proposal = np.clip(proposal, 0, 1)
        # 归一化提案，使其总和为1
        proposal /= np.sum(proposal)

        # 检查提案是否满足约束条件
        if is_feasible(proposal, single_limits_array, multi_limits_array, fund_codes_array):
            # 更新当前权重为提案，并添加到采样结果中
            current_weights = proposal
            the_samples.append(current_weights)

    # 将采样结果转换为numpy数组并返回
    return np.array(the_samples)


# 将权重调整到可行的范围内
def reflect_to_feasible(weights, single_limit_array, indices, start_indices, lengths, lower_limits, upper_limits):
    """
    将权重调整到可行的范围内。

    该函数首先确保每个单独的权重都在给定的限制范围内。然后，它检查由特定索引和长度定义的子集权重之和是否在给定的下限和上限范围内。
    如果子集权重之和小于下限，函数将增加这些权重；如果大于上限，函数将减少这些权重。这样做的目的是使整个权重向量满足所有约束条件。

    参数:
    weights: 待调整的权重向量。
    single_limit_array: 包含每个权重的单独限制范围的二维数组，每一行包含两个值，分别代表下限和上限。
    indices: 包含所有子集权重在weights中的索引的数组。
    start_indices: 每个子集在indices中的起始索引的数组。
    lengths: 每个子集的长度数组，即每个子集在indices中的索引数量。
    lower_limits: 每个子集权重之和的下限数组。
    upper_limits: 每个子集权重之和的上限数组。

    返回:
    调整后的权重向量。
    """
    """
    反射法确保权重满足所有约束条件。
    """
    # 确保每个单独的权重都在其允许的范围内
    for i in range(len(weights)):
        if weights[i] < single_limit_array[i, 0]:
            weights[i] = single_limit_array[i, 0] + (single_limit_array[i, 0] - weights[i])
        elif weights[i] > single_limit_array[i, 1]:
            weights[i] = single_limit_array[i, 1] - (weights[i] - single_limit_array[i, 1])

    # 确保每个子集的权重之和在允许的范围内
    for i in range(len(start_indices)):
        total_weight = 0.0
        for j in range(lengths[i]):
            total_weight += weights[indices[start_indices[i] + j]]
        if total_weight < lower_limits[i]:
            correction = lower_limits[i] - total_weight
            for j in range(lengths[i]):
                weights[indices[start_indices[i] + j]] += correction / lengths[i]
        elif total_weight > upper_limits[i]:
            correction = total_weight - upper_limits[i]
            for j in range(lengths[i]):
                weights[indices[start_indices[i] + j]] -= correction / lengths[i]

    return weights


def mcmc_sampling(initial_weights_list, single_limit_array, indices, start_indices, lengths, lower_limits, upper_limits,
                  num_samples=10000, step_size=0.01):
    samples = List.empty_list(types.float64[:])
    num_initial_weights = len(initial_weights_list)
    current_weights = initial_weights_list[np.random.randint(0, num_initial_weights)]

    while len(samples) < num_samples:
        proposal = current_weights + np.random.normal(0, step_size, len(current_weights))
        proposal = np.clip(proposal, 0, 1)
        proposal /= np.sum(proposal)

        proposal = reflect_to_feasible(proposal, single_limit_array, indices, start_indices, lengths, lower_limits,
                                       upper_limits)

        if is_feasible_njt(proposal, single_limit_array, indices, start_indices, lengths, lower_limits, upper_limits):
            current_weights = proposal
            samples.append(current_weights)

    return np.array(samples)


if __name__ == '__main__':
    fund_codes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    single_limits = [(0.0, 1.0), (0.1, 0.6), (0.0, 1.0), (0.1, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.1, 0.6), (0.0, 1.0), (0.0, 0.6), (0.0, 1.0)]
    multi_limits = {"001,002": (0.31, 0.31), "003,004": (0.21, 0.21), "005": (0.06, 0.06),
                    "006,007": (0.11, 0.11), "008,009": (0.11, 0.11), "010": (0.2, 0.2)}

    s_t = time.time()
    single_limits = np.array(single_limits)
    indices_array, start_indices_array, lengths_array, lower_limits_array, upper_limits_array = multi_limit_dict_to_array(
        multi_limits, fund_codes)

    ''' 使用穷举法, 生成初始随机资产配置权重 '''
    # initial_weights_array = []
    # while len(initial_weights_array) <= 1:
    #     random_weight = np.random.dirichlet(np.ones(3), 10)
    #     final_weight = filter_feasible_weights(random_weight, single_limits, indices_array, start_indices_array,
    #                                            lengths_array, lower_limits_array, upper_limits_array, 10000)
    #     initial_weights_array = initial_weights_array + list(final_weight)
    #
    # # 用一个数据作为初始权重
    # initial_weights_array = np.array(initial_weights_array)[0]

    ''' 使用线性规划法, 生成初始随机资产配置权重 '''
    initial_weights_array = linear_programming_adjustment([0.5 for _ in range(len(fund_codes))], fund_codes,
                                                          single_limits, multi_limits)

    print(initial_weights_array)
    print("初始权重数：", len(initial_weights_array))
    print("初始权重数耗时：", time.time()-s_t)
    num_samples = 10000
    step_size = 0.001

    ''' 使用 mcmc 进行数据采样 '''
    samples = mcmc_sampling_one(initial_weights_array, single_limits, multi_limits, fund_codes,
                                num_samples, step_size=step_size)

    samples = samples.round(6)
    print(samples)
    print("去重重复前的数量：", len(samples))
    samples = np.unique(samples, axis=0)
    print("去重重复后的数量：", len(samples))
    print(f"耗时：{(time.time() - s_t):.2f}秒")

    ''' 画图 '''
    # show_random_weights(samples, fund_codes)
