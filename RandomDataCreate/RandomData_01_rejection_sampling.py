# -*- encoding: utf-8 -*-
"""
@File: Markowitz_01_no_limit.py
@Modify Time: 2024/6/4 09:12       
@Author: Kevin-Chen
@Descriptions: 穷举法
"""

import time
import warnings
import numpy as np
import pandas as pd
from numba.typed import List
from numba import njit, types, prange
from scipy.optimize import linprog
from scipy.optimize import minimize
from pyecharts import options as opts
from pyecharts.charts import Scatter, Line
from Markowitz_01_no_limit import show_random_weights

warnings.filterwarnings('ignore')


# 检查资产配置是否满足单个资产和多个资产的限制条件
def is_feasible(weights, single_limit_array, multi_limit_dict, asset_code_list):
    """
    检查资产配置是否满足单个资产和多个资产的限制条件

    参数:
    weights: np.array, 资产的权重分配。
    single_limit_array: np.array, 单个资产的限制数组，每行包含两个值，分别代表下限和上限。
    multi_limit_array: dict, 多个资产的限制字典，键为资产代码组合，值为该组合的下限和上限。
    asset_code_list: list, 资产代码列表。

    返回:
    bool, 如果资产配置满足所有限制条件，则返回True，否则返回False。
    """
    # 检查每个资产的权重是否在单个资产限制范围内
    if not np.all((weights >= single_limit_array[:, 0]) & (weights <= single_limit_array[:, 1])):
        return False

    # 遍历多资产限制字典，检查资产组合的权重是否在限制范围内
    for key, (lower, upper) in multi_limit_dict.items():
        # 根据资产代码获取其在weights中的索引
        indices = [asset_code_list.index(code) for code in key.split(',')]
        # 检查资产组合的权重是否在限制范围内
        if not (lower <= np.sum(weights[indices]) <= upper):
            return False

    # 如果所有限制条件都满足，则返回True
    return True


# 检查资产配置是否满足单个资产和多个资产的限制条件 (使用njit加速)
@njit(types.boolean(types.float64[:], types.float64[:, :], types.int64[:], types.int64[:], types.int64[:],
                    types.float64[:], types.float64[:]))
def is_feasible_njt(weights, single_limit_array, indices, start_indices, lengths, lower_limits, upper_limits):
    """
    检查资产配置是否满足单个资产和多个资产的限制条件

    参数:
    weights: np.array, 资产的权重分配。
    single_limit_array: np.array, 单个资产的限制数组，每行包含两个值，分别代表下限和上限。
    indices: np.array, 所有组合的索引展平为一个一维数组。
    start_indices: np.array, 每个组合在 indices 中的开始位置。
    lengths: np.array, 每个组合的长度。
    lower_limits: np.array, 每个组合的下限。
    upper_limits: np.array, 每个组合的上限。

    返回:
    bool, 如果资产配置满足所有限制条件，则返回 True，否则返回 False。
    """
    # 检查每个资产的权重是否在单个资产限制范围内
    for i in range(len(weights)):
        if weights[i] < single_limit_array[i, 0] or weights[i] > single_limit_array[i, 1]:
            return False

    # 检查多资产组合的限制条件
    for i in range(len(start_indices)):
        total_weight = 0.0
        for j in range(lengths[i]):
            total_weight += weights[indices[start_indices[i] + j]]
        if total_weight < lower_limits[i] or total_weight > upper_limits[i]:
            return False

    # 如果所有限制条件都满足，则返回 True
    return True


# 将多资产限制字典转换为适合 numba 的格式
def multi_limit_dict_to_array(multi_limit_dict, asset_code_list):
    """
    将多资产限制字典转换为适合 numba 的格式

    :param multi_limit_dict: 多资产限制字典, 键为资产代码组合，值为该组合的下限和上限。
    :param asset_code_list:  资产代码列表。
    :return: 五个一维数组，分别为 indices, start_indices, lengths, lower_limits, upper_limits。
    """
    # 将 multi_limit_dict 转换为适合 numba 的格式
    indices = []  # 所有组合的索引展平为一个一维数组
    start_indices = []  # 每个组合在 indices 中的开始位置
    lengths = []  # 每个组合的长度
    lower_limits = []  # 每个组合的下限
    upper_limits = []  # 每个组合的上限

    start_idx = 0
    for key, (lower, upper) in multi_limit_dict.items():
        idx = [asset_code_list.index(code) for code in key.split(',')]
        indices.extend(idx)
        start_indices.append(start_idx)
        lengths.append(len(idx))
        lower_limits.append(lower)
        upper_limits.append(upper)
        start_idx += len(idx)

    indices = np.array(indices, dtype=np.int64)
    start_indices = np.array(start_indices, dtype=np.int64)
    lengths = np.array(lengths, dtype=np.int64)
    lower_limits = np.array(lower_limits, dtype=np.float64)
    upper_limits = np.array(upper_limits, dtype=np.float64)
    return indices, start_indices, lengths, lower_limits, upper_limits


@njit(types.ListType(types.float64[:])(types.float64[:, :], types.float64[:, :], types.int64[:], types.int64[:],
                                       types.int64[:], types.float64[:], types.float64[:], types.int64))
def filter_feasible_weights(random_weights, single_limit_array, indices, start_indices, lengths, lower_limits,
                            upper_limits, max_count):
    """
    过滤满足条件的权重，并行进行并发执行

    参数:
    random_weights: np.array, 随机生成的权重。
    single_limit_array: np.array, 单个资产的限制数组，每行包含两个值，分别代表下限和上限。
    indices: np.array, 所有组合的索引展平为一个一维数组。
    start_indices: np.array, 每个组合在 indices 中的开始位置。
    lengths: np.array, 每个组合的长度。
    lower_limits: np.array, 每个组合的下限。
    upper_limits: np.array, 每个组合的上限。
    max_count: int, 最大满足条件的权重数量。

    返回:
    numba.typed.List, 满足条件的权重数组。
    """
    feasible_weights = List.empty_list(types.float64[:])
    count = 0
    for i in prange(random_weights.shape[0]):
        if is_feasible_njt(random_weights[i], single_limit_array, indices, start_indices, lengths, lower_limits,
                           upper_limits):
            feasible_weights.append(random_weights[i])
            count += 1
            if count >= max_count:
                break
    return feasible_weights


if __name__ == '__main__':
    fund_codes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    single_limits = [(0.0, 1.0), (0.1, 0.6), (0.0, 1.0), (0.1, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.1, 0.6), (0.0, 1.0), (0.0, 0.6), (0.0, 1.0)]
    multi_limits = {"001,002": (0.31, 0.31), "003,004": (0.21, 0.21), "005": (0.06, 0.06),
                    "006,007": (0.11, 0.11), "008,009": (0.11, 0.11), "010": (0.2, 0.2)}

    single_limits = np.array(single_limits)
    print(single_limits)

    s_t = time.time()

    indices_array, start_indices_array, lengths_array, lower_limits_array, upper_limits_array = multi_limit_dict_to_array(
        multi_limits, fund_codes)

    random_weight = np.random.dirichlet(np.ones(len(fund_codes)), 10000)

    ''' 普通for循环'''
    # final_weight = []
    # for i_w in random_weight:
    #     # if is_feasible(i_w, single_limits, multi_limits, fund_codes):
    #     if is_feasible_njt(i_w, single_limits, indices_array, start_indices_array, lengths_array, lower_limits_array,
    #                          upper_limits_array):
    #         final_weight.append(i_w)
    #         if len(final_weight) == 10000:
    #             break

    ''' 用并发,绕开GIL '''
    final_weight = filter_feasible_weights(random_weight, single_limits, indices_array, start_indices_array,
                                           lengths_array, lower_limits_array, upper_limits_array, 100)
    final_weight = np.array(final_weight).round(6)
    print(final_weight)
    print("去重重复前的数量: ", len(final_weight))

    final_weight = np.unique(final_weight, axis=0)
    print("去重重复后的数量: ", len(final_weight))
    print("计算耗时: ", time.time() - s_t)

    ''' 画图 '''
    # show_random_weights(final_weight, fund_codes)
    pass
