# -*- encoding: utf-8 -*-
"""
@File: Markowitz_01_no_limit.py
@Modify Time: 2024/6/4 09:12
@Author: Kevin-Chen
@Descriptions: MCMC + 线性规划调整权重 + njt
"""

import time
import numpy as np
from Markowitz_01_no_limit import show_random_weights
from numba import njit, prange
from numba import float64, int64
from RandomData_01_rejection_sampling import multi_limit_dict_to_array, is_feasible_njt


@njit(float64[:](float64[:, :], float64[:], float64))
def manual_lst_sq(a, b, regularization=1e-5):
    """
    通过最小二乘法求解线性方程组的解，同时引入正则化项以避免过拟合。

    参数:
    a: 二维数组，表示线性方程组的系数矩阵。
    b: 一维数组，表示线性方程组的常数向量。
    regularization: 浮点数，表示正则化项的强度。

    返回:
    一维数组，表示线性方程组的解向量。
    """
    # 将矩阵a转置，为后续计算做准备
    a_t = a.T
    # 计算a的转置乘以a，得到ATA矩阵
    ata = np.dot(a_t, a)
    # 引入正则化项，防止奇异矩阵，矩阵加上对角线上为regularization的值
    ata = ata + regularization * np.eye(a.shape[1])
    # 计算a的转置乘以b，得到ATb向量
    atb = np.dot(a_t, b)
    # 解出正则化后的线性方程组的解 (np.linalg.solve函数用于求解 𝐴𝑥=𝑏 形式的方程组)
    correction = np.linalg.solve(ata, atb)
    return correction


@njit(float64[:](float64[:], float64[:, :], int64[:], int64[:], int64[:], float64[:], float64[:], int64))
def primal_dual_interior_point_njt(proposal, the_single_limits, indices_array, start_indices_array, lengths_array,
                                   lower_limits_array, upper_limits_array, max_iter=100):
    num_assets = len(proposal)
    num_constraints = 2 * num_assets + len(lower_limits_array) * 2 + 2

    A = np.zeros((num_constraints, num_assets))
    b = np.zeros(num_constraints)

    idx = 0
    for i in range(num_assets):
        A[idx, i] = 1
        b[idx] = the_single_limits[i][1]
        idx += 1
        A[idx, i] = -1
        b[idx] = -the_single_limits[i][0]
        idx += 1

    for i in range(len(lower_limits_array)):
        start = start_indices_array[i]
        length = lengths_array[i]
        for j in range(length):
            index = indices_array[start + j]
            A[idx, index] = 1
            A[idx + 1, index] = -1
        b[idx] = upper_limits_array[i]
        b[idx + 1] = -lower_limits_array[i]
        idx += 2

    A[idx, :] = 1
    b[idx] = 1
    A[idx + 1, :] = -1
    b[idx + 1] = -1

    x = np.copy(proposal)

    for _ in range(max_iter):
        Ax_b = A.dot(x) - b
        violating = Ax_b > 0

        # 如果没有违反任何约束，则优化结束
        if not np.any(violating):
            break

        delta_x = manual_lst_sq(A[violating], Ax_b[violating], regularization=1e-5)
        x -= delta_x

    return x


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


def chang_dict(old_multi_limits, fund_codes_list):
    # 新的多资产限制字典
    new_multi_limits = {}

    # 遍历原字典，转换键
    for key, value in old_multi_limits.items():
        # 将键分割为单独的资产代码
        codes = key.split(',')
        # 获取每个资产代码在 fund_codes 列表中的索引
        indices = tuple(fund_codes_list.index(code) for code in codes)
        # 在新字典中设置转换后的键和原始值
        new_multi_limits[indices] = value
    return new_multi_limits


@njit(float64[:, :](float64[:], float64[:, :], int64[:], int64[:], int64[:], float64[:], float64[:], float64, int64,
                    int64))
def mcmc_lp_sampling(the_current_weights, single_limit_array, indices, start_indices, lengths, lower_limits,
                     upper_limits, the_step_size, num_samples, max_iter):
    num_assets = len(the_current_weights)
    the_final_weight = np.zeros((num_samples, num_assets))  # 预先分配空间
    count = 0  # 有效样本计数器
    # new_proposal_array = np.random.dirichlet(np.ones(num_assets), num_samples)

    for i in range(num_samples):
        new_proposal = the_current_weights + np.random.normal(0, the_step_size, num_assets)
        # new_proposal = new_proposal_array[i]

        adjusted_weights = primal_dual_interior_point_njt(new_proposal, single_limit_array, indices,
                                                          start_indices, lengths,
                                                          lower_limits, upper_limits, max_iter)

        # 保留有效的调整后权重
        if adjusted_weights is not None and is_feasible_njt(adjusted_weights, single_limit_array, indices,
                                                            start_indices, lengths, lower_limits, upper_limits):
            the_final_weight[count] = adjusted_weights
            count += 1
        the_current_weights = adjusted_weights

    return the_final_weight[:count]  # 返回有效样本部分


@njit(float64[:, :](float64[:, :], int64))
def generate_initial_points(single_limit_array, num_points):
    num_assets = single_limit_array.shape[0]
    points = np.zeros((num_points, num_assets))
    # 生成边界点
    for i in range(num_assets):
        points[i % num_points, i] = single_limit_array[i, 1]  # 使用每个资产的上限
        if (i + 1) % num_points < num_points:
            points[(i + 1) % num_points, i] = single_limit_array[i, 0]  # 使用每个资产的下限
    # 添加随机点
    for i in range(num_assets, num_points):
        points[i, :] = np.random.dirichlet(np.ones(num_assets), 1)
    return points


@njit(float64[:, :](
    int64, float64[:, :], int64[:], int64[:], int64[:], float64[:], float64[:],
    float64, int64, int64, int64), parallel=True)
def run_multiple_mcmc_lp_sampling(num_assets, single_limit_array, indices, start_indices, lengths,
                                  lower_limits, upper_limits, the_step_size, num_samples, max_iter, num_runs=10):
    init_weights = generate_initial_points(single_limit_array, num_runs)
    max_possible_samples = num_samples * len(init_weights)  # 假设每次都生成 num_samples 个样本
    all_results = np.zeros((max_possible_samples, num_assets))  # 预先分配空间

    # for i in prange(num_runs):
    for i in prange(len(init_weights)):
        start_idx = i * num_samples
        result = mcmc_lp_sampling(
            init_weights[i],  # 确保每次调用都使用初始权重的副本
            single_limit_array,
            indices,
            start_indices,
            lengths,
            lower_limits,
            upper_limits,
            the_step_size,
            num_samples,
            max_iter
        )
        if result.shape[0] > 0:  # 确保不添加空数组
            # num_result_samples = result.shape[0]
            # all_results[result_count:result_count + num_result_samples, :] = result
            # result_count += num_result_samples
            end_idx = start_idx + result.shape[0]
            all_results[start_idx:end_idx, :] = result

    # 裁剪数组以匹配实际结果数量
    # final_results = all_results[:result_count, :]

    # 计算每行的总和
    row_sums = np.sum(all_results, axis=1)
    # 筛选出总和不为0的行
    filtered_results = all_results[row_sums != 0]

    return filtered_results


def filter_weights_by_gini(the_final_weight, threshold=0.4):
    # 计算每行的基尼不纯度
    gini_impurities = 1 - np.sum(the_final_weight ** 2, axis=1)

    # 找出基尼不纯度大于等于0.4的行索引
    valid_indices = gini_impurities >= threshold

    # 过滤出符合条件的权重组合
    filtered_weights = the_final_weight[valid_indices]

    return filtered_weights


if __name__ == '__main__':
    fund_codes = ['003816.OF', '660107.OF', '519510.OF', '202302.OF', '000013.OF', '000010.OF',
                  '018060.OF', '006011.OF', '012404.OF', '013706.OF',
                  '960033.OF', '019447.OF', '017092.OF', '017091.OF',
                  '019005.OF', '517520.OF', '018543.OF', '159985.OF']
    single_limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    # multi_limits = {"003816.OF,660107.OF,519510.OF,202302.OF,000013.OF,000010.OF": (0.743, 0.743),
    #                 "018060.OF,006011.OF,012404.OF,013706.OF": (0.216, 0.216),
    #                 "960033.OF,019447.OF,017092.OF,017091.OF": (0.009, 0.009),
    #                 "019005.OF,517520.OF,018543.OF,159985.OF": (0.032, 0.032)}
    multi_limits = {"003816.OF,660107.OF,519510.OF,202302.OF,000013.OF,000010.OF": (0.743205, 0.743205),
                    "018060.OF,006011.OF,012404.OF,013706.OF": (0.215744, 0.215744),
                    "960033.OF,019447.OF,017092.OF,017091.OF": (0.009177, 0.009177),
                    "019005.OF,517520.OF,018543.OF,159985.OF": (0.031874, 0.031874)}

    num_of_asset = len(fund_codes)
    single_limits = np.array(single_limits)
    # 字典转换
    indices_array, start_indices_array, lengths_array, lower_limits_array, upper_limits_array = (
        multi_limit_dict_to_array(multi_limits, fund_codes))

    # 创建一个等权重的初始权重, 用了线性规划, 出发点无需在可行域内, 线性规划会把他拉到可行域上的
    current_weights = np.array([1 / num_of_asset] * num_of_asset)
    # 随机游走的步长
    step_size = 0.5
    # 采样次数
    num_samples = 10000

    s_t = time.time()

    ''' 计算一次 '''
    # current_weights = clip_array(current_weights, single_limits[:, 0], single_limits[:, 1])
    # print(current_weights)
    # final_weight = primal_dual_interior_point_njt(current_weights, single_limits,
    #                                                   indices_array, start_indices_array, lengths_array,
    #                                                   lower_limits_array, upper_limits_array, max_iter=100)

    ''' 单线程 '''
    # final_weight = mcmc_lp_sampling(current_weights, single_limits, indices_array,
    #                                 start_indices_array, lengths_array,
    #                                 lower_limits_array, upper_limits_array,
    #                                 the_step_size=step_size, num_samples=num_samples, max_iter=10)

    ''' 多线程,绕开GIL锁 (效果一般) '''
    final_weight = run_multiple_mcmc_lp_sampling(num_of_asset, single_limits, indices_array,
                                                 start_indices_array, lengths_array,
                                                 lower_limits_array, upper_limits_array,
                                                 the_step_size=step_size, num_samples=600, max_iter=10,
                                                 num_runs=num_of_asset)

    ''' 打印信息 '''
    final_weight = np.array(final_weight).round(7)
    print(final_weight)
    unique_final_weight = np.unique(final_weight, axis=0)
    print(unique_final_weight)
    print("去除重复前的数量: ", len(final_weight))
    print("去除重复后的数量: ", len(unique_final_weight))
    print("计算耗时: ", time.time() - s_t)

    ''' 剔除不符合基尼不纯度的权重 '''
    unique_final_weight = filter_weights_by_gini(unique_final_weight, threshold=0.2)
    print("去除基尼不纯度后的数量: ", len(unique_final_weight))

    ''' 画图 '''
    show_random_weights(unique_final_weight, fund_codes)
