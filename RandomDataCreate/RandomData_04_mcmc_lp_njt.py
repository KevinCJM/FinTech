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
    # 引入正则化项，防止过拟合，矩阵加上对角线上为regularization的值
    ata_regularized = ata + regularization * np.eye(a.shape[1])
    # 计算a的转置乘以b，得到ATb向量
    atb = np.dot(a_t, b)
    # 使用高斯消元法，解出正则化后的线性方程组的解
    correction = np.linalg.solve(ata_regularized, atb)
    return correction


# 使用内点法优化投资组合
def primal_dual_interior_point(proposal, the_single_limits, the_multi_limits, max_iter=100):
    """
    使用原始对偶内点法优化投资组合。

    参数:
    proposal: 初始投资组合权重。
    the_single_limits: 单个资产的上下限。
    the_multi_limits: 多资产组合的上下限。
    max_iter: 最大迭代次数。

    返回:
    优化后的投资组合权重。
    """
    # 初始化资产数量和约束数量
    num_assets = len(proposal)
    # 计算约束条件的数量, 总约束数增加1，用于添加权重和约束
    num_constraints = 2 * num_assets + 2 * len(the_multi_limits) + 2

    # 初始化约束矩阵A和右侧向量b
    A = np.zeros((num_constraints, num_assets))
    b = np.zeros(num_constraints)

    # 填充单资产限制的约束条件
    # 填充单个资产的限制
    idx = 0
    for i_ in range(num_assets):
        A[idx, i_] = 1  # 单个资产上限
        b[idx] = the_single_limits[i_][1]
        idx += 1
        A[idx, i_] = -1  # 单个资产下限
        b[idx] = -the_single_limits[i_][0]
        idx += 1

    # 填充多资产组合限制的约束条件
    for indices, (lower, upper) in the_multi_limits.items():
        A[idx, indices] = 1  # 组合上限
        b[idx] = upper
        idx += 1
        A[idx, indices] = -1  # 组合下限
        b[idx] = -lower
        idx += 1

    # 添加权重和为1的约束条件
    A[idx, :] = 1
    b[idx] = 1
    A[idx + 1, :] = -1
    b[idx + 1] = -1

    # 初始化投资组合权重
    x = np.copy(proposal)  # 初始权重
    # 初始化迭代次数
    iter_count = 0

    # 主循环：最多进行max_iter次迭代
    # 使用内点法优化权重
    for _ in range(max_iter):
        # 计算当前投资组合违反约束的程度
        Ax_b = A.dot(x) - b
        violating = Ax_b > 0

        # 如果没有违反任何约束，则优化结束
        if not np.any(violating):
            break

        # 计算用于修正投资组合的校正向量
        # correction = np.linalg.lstsq(A[violating], Ax_b[violating], rcond=None)[0]
        correction = manual_lst_sq(A[violating], Ax_b[violating], regularization=1e-5)
        x -= correction

        # 确保权重在允许的范围内，并归一化
        '''
        在每一步跃进中, 我们可以通过执行 np.clip 和归一化的操作帮助提高算法的收敛速度和稳定性. 
        通过确保每一步权重的合法性和一致性, 可以避免算法在非法或不稳定的权重值区域中循环或发散, 从而更快地找到满足所有约束的最优解.
        '''
        # 确保权重在单个资产的限制范围内
        x = np.clip(x, a_min=[lim[0] for lim in the_single_limits], a_max=[lim[1] for lim in the_single_limits])
        # 归一化
        x /= np.sum(x)
        # 更新迭代次数
        iter_count += 1
        # 如果达到最大迭代次数仍未满足所有约束，则返回None
        if iter_count == max_iter - 1:
            return None

    return x


@njit(float64[:](float64[:], float64[:], float64[:]))
def clip_array(x, min_vals, max_vals):
    # 确保 x, min_vals, max_vals 的长度相同
    for i in range(len(x)):
        if x[i] < min_vals[i]:
            x[i] = min_vals[i]
        elif x[i] > max_vals[i]:
            x[i] = max_vals[i]
    return x


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
    iter_count = 0

    for _ in range(max_iter):
        Ax_b = A.dot(x) - b
        violating = Ax_b > 0

        # 如果没有违反任何约束，则优化结束
        if not np.any(violating):
            break

        correction = manual_lst_sq(A[violating], Ax_b[violating], regularization=1e-5)

        x -= correction
        # x = clip_array(x, the_single_limits[:, 0], the_single_limits[:, 1])
        # x /= np.sum(x)
        # iter_count += 1
        # if iter_count == max_iter - 1:
        #     # return np.zeros_like(proposal)  # 返回一个全0的数组，形状与proposal相同
        #     return x

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
    float64[:], float64[:, :], int64[:], int64[:], int64[:], float64[:], float64[:],
    float64, int64, int64, int64), parallel=True)
def run_multiple_mcmc_lp_sampling(the_current_weights, single_limit_array, indices, start_indices, lengths,
                                  lower_limits, upper_limits, the_step_size, num_samples, max_iter, num_runs=10):
    num_assets = len(the_current_weights)
    max_possible_samples = num_samples * num_runs  # 假设每次都生成 num_samples 个样本
    all_results = np.zeros((max_possible_samples, num_assets))  # 预先分配空间
    result_count = 0  # 实际收集的样本计数
    init_weights = generate_initial_points(single_limit_array, num_runs)

    for i in prange(len(init_weights)):
        result = mcmc_lp_sampling(
            # the_current_weights.copy(),  # 确保每次调用都使用初始权重的副本
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
            num_result_samples = result.shape[0]
            all_results[result_count:result_count + num_result_samples, :] = result
            result_count += num_result_samples

    # 裁剪数组以匹配实际结果数量
    final_results = all_results[:result_count, :]

    return final_results


if __name__ == '__main__':
    fund_codes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    single_limits = [(0.0, 1.0), (0.1, 0.6), (0.0, 1.0), (0.1, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.1, 0.6), (0.0, 1.0), (0.0, 0.6), (0.0, 1.0)]
    multi_limits = {"001,002": (0.31, 0.31), "003,004": (0.21, 0.21), "005": (0.06, 0.06),
                    "006,007": (0.11, 0.11), "008,009": (0.11, 0.11), "010": (0.2, 0.2)}


    num_of_asset = len(fund_codes)
    single_limits = np.array(single_limits)
    # 字典转换
    indices_array, start_indices_array, lengths_array, lower_limits_array, upper_limits_array = (
        multi_limit_dict_to_array(multi_limits, fund_codes))

    # 创建一个等权重的初始权重, 用了线性规划, 出发点无需在可行域内, 线性规划会把他拉到可行域上的
    current_weights = np.array([1 / num_of_asset] * num_of_asset)
    # 随机游走的步长
    step_size = 0.01
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
    final_weight = mcmc_lp_sampling(current_weights, single_limits, indices_array,
                                    start_indices_array, lengths_array,
                                    lower_limits_array, upper_limits_array,
                                    the_step_size=step_size, num_samples=num_samples, max_iter=100)

    ''' 多线程,绕开GIL锁 (效果一般) '''
    # final_weight = run_multiple_mcmc_lp_sampling(current_weights, single_limits, indices_array,
    #                                              start_indices_array, lengths_array,
    #                                              lower_limits_array, upper_limits_array,
    #                                              the_step_size=step_size, num_samples=10000, max_iter=100, num_runs=10)


    ''' 打印信息 '''
    final_weight = np.array(final_weight).round(6)
    print(final_weight)
    unique_final_weight = np.unique(final_weight, axis=0)
    print(unique_final_weight)
    print("去除重复前的数量: ", len(final_weight))
    print("去除重复后的数量: ", len(unique_final_weight))
    print("计算耗时: ", time.time() - s_t)

    ''' 画图 '''
    show_random_weights(unique_final_weight, fund_codes)
