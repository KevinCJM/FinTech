# -*- encoding: utf-8 -*-
"""
@File: Markowitz_01_no_limit.py
@Modify Time: 2024/6/4 09:12       
@Author: Kevin-Chen
@Descriptions: 无权重约束下的马科维兹有效前沿
"""

import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize import minimize
from pyecharts import options as opts
from pyecharts.charts import Scatter, Line


# 定义目标函数 (组合波动率)
def cal_portfolio_volatility(weights, the_cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(the_cov_matrix, weights)))


# 定义目标函数 (组合收益率的负值，因为我们使用 minimize 来最大化收益)
def negative_portfolio_return(weights, the_total_return):
    return -np.dot(weights, the_total_return)


# 使用 minimize 函数找到波动率最小的组合权重
def find_the_min_volatility(the_total_return, the_cov_matrix, the_initial_weights, the_bounds, the_constraints):
    """
    使用 minimize 函数找到波动率最小的组合权重

    :param the_total_return: 各个基金的累计收益率, 一维数组
    :param the_cov_matrix: 各个基金的协方差矩阵, 二维数组
    :param the_initial_weights: 权重的初始猜测, 一维数组
    :param the_bounds: 权重边界 (每个权重在0到1之间), 一维数组
    :param the_constraints: 约束条件
    :return: min_volatility, optimal_return, optimal_weights
    """
    min_volatility_result = minimize(cal_portfolio_volatility, x0=the_initial_weights, args=(the_cov_matrix,),
                                     method='SLSQP', bounds=the_bounds, constraints=the_constraints,
                                     tol=1e-15, options={'disp': False, 'maxiter': 300})
    # 最优权重
    optimal_weights = min_volatility_result.x
    # 最小波动率
    min_volatility = min_volatility_result.fun
    # 计算最优组合的预期收益率
    optimal_return = np.dot(optimal_weights, the_total_return)
    return min_volatility, optimal_return, optimal_weights


# 使用 minimize 函数找到收益最大的组合权重
def find_the_max_return(the_total_return, the_cov_matrix, the_initial_weights, the_bounds, the_constraints):
    """
    使用 minimize 函数找到收益最大的组合权重

    :param the_total_return: 各个基金的累计收益率, 一维数组
    :param the_cov_matrix: 各个基金的协方差矩阵, 二维数组
    :param the_initial_weights: 权重的初始猜测, 一维数组
    :param the_bounds: 权重边界 (每个权重在0到1之间), 一维数组
    :param the_constraints: 约束条件
    :return: max_return, max_return_volatility, max_return_weights
    """
    max_return_result = minimize(negative_portfolio_return, the_initial_weights, args=(the_total_return,),
                                 method='SLSQP', bounds=the_bounds, constraints=the_constraints,
                                 tol=1e-15, options={'disp': False, 'maxiter': 300})
    # 最优权重
    max_return_weights = max_return_result.x
    # 最大收益率
    max_return = -max_return_result.fun
    # 计算最优组合的波动率 (收益率最大)
    max_return_volatility = cal_portfolio_volatility(max_return_weights, the_cov_matrix)
    return max_return, max_return_volatility, max_return_weights


# 根据全局最小方差组合的波动率, 以及全局最大收益率组合对应的波动率, 得到马科维兹有效前沿的点位数据
def find_the_efficient_frontier(num_of_points, the_min_volatility, the_max_volatility, the_bounds, the_constraints,
                                the_initial_weights, the_total_return, the_cov_matrix):
    """
    根据全局最小方差组合的波动率, 以及全局最大收益率组合对应的波动率, 画出马科维兹有效前沿
    1. 根据 the_min_volatility 和 the_max_volatility 等分切分 num_of_points 个波动率的点
    2. 遍历每一个波动率点, 使用优化函数, 找到该波动率点对应的最大收益率组合
    3. 返回三个结果:
    3.1 一维数组, 代表有效前沿上点位的波动率
    3.2 一维数组, 代表有效前沿上点位的收益率
    3.3 二维数组, 代表有效前沿上点位的权重

    :param num_of_points: 计算的点数, int
    :param the_min_volatility: 全局最小方差组合的波动率, float
    :param the_max_volatility: 全局最大收益率组合对应的波动率, float
    :param the_bounds: 权重边界 (每个权重在0到1之间), 一维数组
    :param the_constraints: 约束条件
    :param the_initial_weights: 权重的初始猜测, 一维数组
    :param the_total_return: 各个基金的累计收益率, 一维数组
    :param the_cov_matrix: 各个基金的协方差矩阵, 二维数组
    :return: volatility_array, return_array, weight_2d_array
    """
    # 生成等分的波动率点
    volatility_points = np.linspace(the_min_volatility, the_max_volatility, num_of_points)

    # 存储结果的数组
    return_array = []
    weight_2d_array = []

    for target_volatility in volatility_points:
        # 目标函数，添加波动率约束条件
        def target_function(weights):
            return -np.dot(weights, the_total_return)

        def volatility_constraint(weights):
            return cal_portfolio_volatility(weights, the_cov_matrix) - target_volatility

        constraints_with_vol = the_constraints + ({'type': 'eq', 'fun': volatility_constraint},)

        result = minimize(target_function, the_initial_weights, method='SLSQP', bounds=the_bounds,
                          constraints=constraints_with_vol, tol=1e-15, options={'disp': False, 'maxiter': 300})

        if result.success:
            the_optimal_weights = result.x
            the_optimal_return = np.dot(the_optimal_weights, the_total_return)
            return_array.append(the_optimal_return)
            weight_2d_array.append(the_optimal_weights)

    volatility_array = volatility_points
    return_array = np.array(return_array)
    weight_2d_array = np.array(weight_2d_array)

    return volatility_array, return_array, weight_2d_array


# 根据全局最小方差组合的波动率, 以及全局最大收益率组合对应的波动率, 得到马科维兹有效前沿的点位数据 (使用最近随机权重作为x0起始权重)
def find_the_efficient_frontier_with_random_weights(num_of_points, the_min_volatility, the_max_volatility, the_bounds,
                                                    the_constraints, the_total_return, the_cov_matrix,
                                                    the_random_weights, the_random_returns, the_random_volatility):
    """
    根据全局最小方差组合的波动率, 以及全局最大收益率组合对应的波动率, 得到马科维兹有效前沿的点位数据 (使用最近随机权重作为x0起始权重)

    参数:
    num_of_points: 整数，有效前沿上的点数。
    the_min_volatility: 浮点数，最小目标波动性。
    the_max_volatility: 浮点数，最大目标波动性。
    the_bounds: 元组列表，每个资产的权重约束范围。
    the_constraints: 优化的约束条件。
    the_total_return: 数组，资产的预期回报率。
    the_cov_matrix: 数组，资产的协方差矩阵。
    the_random_weights: 数组，随机生成的资产权重。
    the_random_returns: 数组，对应随机权重的回报率。
    the_random_volatility: 数组，对应随机权重的波动性。

    返回:
    volatility_array: 数组，有效前沿上的波动性。
    return_array: 数组，有效前沿上的回报率。
    weight_2d_array: 数组，每个有效前沿点对应的资产权重。
    """
    # 生成等间距的波动性点
    volatility_points = np.linspace(the_min_volatility, the_max_volatility, num_of_points)

    # 初始化存储回报率和权重的列表
    return_array = []
    weight_2d_array = []

    # 遍历每个目标波动性
    for target_volatility in volatility_points:
        # 定义初始权重的查找函数
        def find_the_initial_weights():
            """
            找到随机权重中波动性小于等于目标波动性的最大回报的权重。
            """
            # 根据目标波动性筛选随机波动性
            # 从 the_random_volatility 中找到所有小于 target_volatility 的数据的索引位置
            indices = np.where(the_random_volatility <= target_volatility)
            # 在筛选后的随机回报中找到最大回报的索引
            # 从 the_random_returns 中找到 indices 索引位置的收益率, 并选出最大的那个, 得到其索引位置
            max_return_index = np.argmax(the_random_returns[indices])
            # 返回对应的最大回报的权重
            # 根据 max_return_index 从 random_weights 找到对应的权重
            start_weights = the_random_weights[indices][max_return_index]
            return start_weights

        # 定义目标函数为负的预期回报（为了最大化回报，目标函数取负值）
        def target_function(weights):
            return -np.dot(weights, the_total_return)

        # 定义波动性约束函数
        def volatility_constraint(weights):
            return cal_portfolio_volatility(weights, the_cov_matrix) - target_volatility

        # 将波动性约束添加到现有约束中
        constraints_with_vol = the_constraints + ({'type': 'eq', 'fun': volatility_constraint},)

        # 使用SLSQP方法优化目标函数，同时满足波动性约束
        result = minimize(target_function, find_the_initial_weights(), method='SLSQP', bounds=the_bounds,
                          constraints=constraints_with_vol)

        # 如果优化成功，添加结果到列表中
        if result.success:
            the_optimal_weights = result.x
            the_optimal_return = np.dot(the_optimal_weights, the_total_return)
            return_array.append(the_optimal_return)
            weight_2d_array.append(the_optimal_weights)

    # 将列表转换为数组并返回
    volatility_array = volatility_points
    return_array = np.array(return_array)
    weight_2d_array = np.array(weight_2d_array)

    return volatility_array, return_array, weight_2d_array


# 使用 pyecharts 画图, 马科维兹有效前沿 & 随机权重的散点图
def draw_efficient_frontier(portfolio_v, portfolio_r, v_array, r_array, html_file="markowitz_efficient_frontier.html"):
    """
    使用 pyecharts 画图, 马科维兹有效前沿 & 随机权重的散点图
    会将结果保存到指定路径的 html 文件中

    :param portfolio_v: 散点图, x轴数据, 波动率
    :param portfolio_r: 散点图, y轴数据, 收益率
    :param v_array: 折线图, x轴数据, 波动率
    :param r_array: 折线图, y轴数据, 收益率
    :param html_file: html文件的路径
    :return: None
    """
    scatter = (
        Scatter()
        .add_xaxis(portfolio_v.tolist())
        .add_yaxis("Portfolios", portfolio_r.tolist(),
                   symbol_size=3,  # 设置点的大小为最小
                   label_opts=opts.LabelOpts(is_show=False)  # 不展示标签
                   )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Markowitz Efficient Frontier"),
            xaxis_opts=opts.AxisOpts(name="Volatility", type_="value", min_=0.004),
            yaxis_opts=opts.AxisOpts(name="Return", type_="value", min_=0.02),
            tooltip_opts=opts.TooltipOpts(formatter="{c}"),
        )
    )

    # 添加有效前沿的线条
    line = (
        Line()
        .add_xaxis(v_array.tolist())
        .add_yaxis("Efficient Frontier", r_array.tolist(),
                   is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=2, color="red"),
                   label_opts=opts.LabelOpts(is_show=False))
    )

    # 合并散点图和线条图
    scatter.overlap(line)

    # 保存并展示图表
    scatter.render(html_file)
    print(f"保存html文件到: {html_file}")


# 展示各个产品数的权重分布
def show_random_weights(samples, fund_codes):
    from pandas.plotting import parallel_coordinates
    import matplotlib.pyplot as plt

    portfolios_df = pd.DataFrame(samples, columns=fund_codes)
    # 添加标签列
    portfolios_df['Index'] = range(len(portfolios_df))

    # 使用平行坐标图
    plt.figure(figsize=(12, 6))
    parallel_coordinates(portfolios_df, 'Index', color=plt.cm.tab20.colors, alpha=0.5)
    plt.xlabel('Assets')
    plt.ylabel('Weights')
    plt.title('Parallel Coordinates Plot of Portfolio Weights')
    plt.legend().remove()
    plt.show()

if __name__ == '__main__':
    ''' 读取数据, 并计算收益率&协方差矩阵 '''
    # 权重生成
    num_of_samples = 10000  # 生成的权重数量
    df = pd.read_pickle(
        "/Users/chenjunming/Desktop/hundsun_git/python_algorithm/KYP/pickle_file/all_funds_return_f.pkl",
        compression='gzip')
    df = df[['015830', '001628', '007859', '270023']]
    df = df[df.index >= pd.to_datetime('2021-01-01')]
    # df = pd.read_parquet("funds_return.parquet")  # df 的字段是产品代码, 索引是连续的自然日, 值是基金的每日收益率; 数据中有大量nan;
    pct_2d_array = df.values
    num_of_assets = pct_2d_array.shape[1]  # 产品数量

    # 计算区间内累计收益率
    total_return = np.nanprod(1 + pct_2d_array, axis=0) - 1
    # 计算协方差矩阵
    cov_matrix = df.cov().values
    s_t = time.time()

    ''' 生成随机权重 '''

    random_weights = np.random.dirichlet(np.ones(num_of_assets), num_of_samples)
    print(random_weights)

    ''' 根据随机权重生成组合收益率和组合风险 '''

    portfolio_returns = np.dot(random_weights, total_return)
    # 'ij'：表示 random_weights 的第一个维度是组合的索引（i），第二个维度是资产的索引（j）
    # 'jk'：表示 cov_matrix 的第一个维度是资产的索引（j），第二个维度也是资产的索引（k）
    # 'ik'：表示 random_weights 的第一个维度是组合的索引（i），第二个维度是资产的索引（k）
    # '->i'：表示将结果累加并简化为形状 (i,) 的一维数组，即最终的结果是每个组合的方差，并返回一个包含所有组合方差的数组
    # 'ij,jk,ik->i'：表示对 random_weights 乘以 cov_matrix 得到一个中间结果，然后再乘以 random_weights，最后得到一维数组
    portfolio_volatility = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))

    print(portfolio_returns)
    print(portfolio_volatility)

    ''' 使用 minimize 优化, 找到有效前沿 '''
    # 权重边界 (每个权重在0到1之间)
    bounds = tuple((0, 1) for _ in range(num_of_assets))
    # 初始猜测
    initial_weights = np.array(num_of_assets * [1. / num_of_assets])
    # 约束条件
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # 权重和为1
    )

    # 使用 minimize 函数找到波动率最小的组合权重
    min_std, min_std_ret, min_std_w = find_the_min_volatility(total_return, cov_matrix, initial_weights, bounds,
                                                              constraints)
    print(f"最小波动率: {min_std}, 最小波动率对应收益率: {min_std_ret}")
    # 使用 minimize 函数找到收益最大的组合权重
    max_ret, max_ret_std, max_ret_w = find_the_max_return(total_return, cov_matrix, initial_weights, bounds,
                                                          constraints)
    print(f"最大收益率: {max_ret}, 最大收益率对应波动率: {max_ret_std}")

    # 将波动率最小的组合以及收益率最大组合的 权重 加入到随机组合权重数据中
    random_weights = np.vstack([random_weights, min_std_w, max_ret_w])
    # 将波动率最小的组合以及收益率最大组合的 收益率 加入到随机组合收益数据中
    portfolio_returns = np.hstack([portfolio_returns, min_std_ret, max_ret])
    # 将波动率最小的组合以及收益率最大组合的 波动率 加入到随机组合波动率数据中
    portfolio_volatility = np.hstack([portfolio_volatility, min_std, max_ret_std])

    # 根据全局最小方差组合的波动率, 以及全局最大收益率组合对应的波动率, 得到马科维兹有效前沿的点位数据
    # std_array, ret_array, _ = find_the_efficient_frontier(100, min_std, max_ret_std, bounds, constraints,
    #                                                       initial_weights, total_return, cov_matrix)

    # 根据全局最小方差组合的波动率, 以及全局最大收益率组合对应的波动率, 得到马科维兹有效前沿的点位数据 (使用最近随机权重作为x0起始权重)
    std_array, ret_array, _ = find_the_efficient_frontier_with_random_weights(100, min_std, max_ret_std, bounds,
                                                                              constraints, total_return, cov_matrix,
                                                                              random_weights, portfolio_returns,
                                                                              portfolio_volatility)
    print(std_array)
    print(ret_array)

    ''' 画图 '''
    draw_efficient_frontier(portfolio_volatility, portfolio_returns, std_array, ret_array)
    print(time.time() - s_t)
    pass
