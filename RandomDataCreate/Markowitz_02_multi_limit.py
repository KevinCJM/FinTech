# -*- encoding: utf-8 -*-
"""
@File: Markowitz_02_multi_limit.py
@Modify Time: 2024/6/4 09:12       
@Author: Kevin-Chen
@Descriptions: 多个权重约束下的马科维兹有效前沿
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyecharts import options as opts
from pyecharts.charts import Scatter, Line

from RandomData_05_mcmc_lp_njt_mt import run_multiple_mcmc_lp_sampling, filter_weights_by_gini, \
    multi_limit_dict_to_array, show_random_weights
from Markowitz_01_no_limit import draw_efficient_frontier


# 组合收益率函数
def portfolio_return(the_weights, the_return):
    return np.dot(the_weights, the_return)


# 组合风险函数
def portfolio_volatility(the_weights, the_cov_matrix):
    # return np.sqrt(np.einsum('ij,jk,ik->i', the_weights, the_cov_matrix, the_weights))
    return np.sqrt(np.dot(the_weights.T, np.dot(the_cov_matrix, the_weights)))


# 生成随机权重
def create_random_weights(the_fund_codes, the_single_limits, the_multi_limits, the_annual_return, the_cov_matrix,
                          step_size=0.1):
    """

    :param the_fund_codes:
    :param the_single_limits:
    :param the_multi_limits:
    :param the_annual_return:
    :param the_cov_matrix:
    :param step_size:
    :return:
    """
    ''' 根据随机权重生成组合收益率和组合风险 '''
    s_t = time.time()
    num_of_asset = len(the_fund_codes)
    the_single_limits = np.array(the_single_limits)
    indices_array, start_indices_array, lengths_array, lower_limits_array, upper_limits_array = (
        multi_limit_dict_to_array(the_multi_limits, the_fund_codes))
    final_weight = run_multiple_mcmc_lp_sampling(num_of_asset, the_single_limits, indices_array,
                                                 start_indices_array, lengths_array,
                                                 lower_limits_array, upper_limits_array,
                                                 the_step_size=step_size, num_samples=1000, max_iter=100,
                                                 num_runs=num_of_asset + 1)
    unique_final_weight = np.unique(final_weight, axis=0)
    print("去除重复前的数量: ", len(final_weight))
    print("去除重复后的数量: ", len(unique_final_weight))

    ''' 剔除不符合基尼不纯度的随机权重 '''
    unique_final_weight = filter_weights_by_gini(unique_final_weight, threshold=0.2)
    print("去除基尼不纯度后的数量: ", len(unique_final_weight))
    print("随机权重生成耗时: ", time.time() - s_t)
    # show_random_weights(unique_final_weight, the_fund_codes)

    ''' 计算随机权重组合的收益率和组合风险 '''
    portfolio_r = portfolio_return(unique_final_weight, the_annual_return)
    portfolio_v = np.sqrt(np.einsum('ij,jk,ik->i', unique_final_weight, the_cov_matrix, unique_final_weight))
    # portfolio_v = portfolio_volatility(unique_final_weight.T, the_cov_matrix)

    ''' 找到最大收益率对应的风险,剔除大于该风险的随机数据 '''
    # 找到最大收益率的索引
    max_return_index = np.argmax(portfolio_r)
    # 获取最大收益率对应的风险
    max_return_risk = portfolio_v[max_return_index]
    # 找到风险值小于等于最大收益率对应风险的所有组合
    selected_indices = portfolio_v <= max_return_risk
    # 筛选出这些组合的收益率和风险
    portfolio_r = portfolio_r[selected_indices]
    portfolio_v = portfolio_v[selected_indices]
    unique_final_weight = unique_final_weight[selected_indices]
    return portfolio_r, portfolio_v, unique_final_weight


# 基尼不纯度
def gini_impurity(weights):
    return 1.0 - np.sum(weights ** 2)


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

    # 组合风险函数
    def the_portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(the_cov_matrix, weights)))

    min_volatility_result = minimize(the_portfolio_volatility, x0=the_initial_weights,
                                     method='SLSQP', bounds=the_bounds, constraints=the_constraints, tol=1e-15)
    # 最优权重
    optimal_weights = min_volatility_result.x
    # 最小波动率
    min_volatility = min_volatility_result.fun
    # 计算最优组合的预期收益率
    optimal_return = portfolio_return(optimal_weights, the_total_return)
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

    # 定义目标函数 (组合收益率的负值，因为我们使用 minimize 来最大化收益)
    def negative_portfolio_return(the_weights):
        return -np.dot(the_weights, the_total_return)

    max_return_result = minimize(negative_portfolio_return, the_initial_weights,
                                 method='SLSQP', bounds=the_bounds, constraints=the_constraints, tol=1e-15)
    # 最优权重
    max_return_weights = max_return_result.x
    # 最大收益率
    max_return = -max_return_result.fun
    # 计算最优组合的波动率 (收益率最大)
    max_return_volatility = portfolio_volatility(max_return_weights, the_cov_matrix)
    return max_return, max_return_volatility, max_return_weights


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
            the_indices = np.where(the_random_volatility <= target_volatility)
            # 在筛选后的随机回报中找到最大回报的索引: 从 the_random_returns 中找到 the_indices 索引位置的收益率, 并选出最大的那个, 得到其索引位置
            max_return_index = np.argmax(the_random_returns[the_indices])
            # 返回对应的最大回报的权重: 根据 max_return_index 从 random_weights 找到对应的权重
            start_weights = the_random_weights[the_indices][max_return_index]
            return start_weights

        # 定义目标函数为负的预期回报（为了最大化回报，目标函数取负值）
        def target_function(weights):
            return -np.dot(weights, the_total_return)

        def cal_portfolio_volatility(weights, t_cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(t_cov_matrix, weights)))

        # 定义波动性约束函数
        def volatility_constraint(weights):
            return cal_portfolio_volatility(weights, the_cov_matrix) - target_volatility

        # 将波动性约束添加到现有约束中
        constraints_with_vol = the_constraints + [{'type': 'eq', 'fun': volatility_constraint}]

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

def _get_weight_diff(weights, idx_list, limit, param=1.0):
    """ 计算权重的差值 """

    summ = 0
    for idx in idx_list:
        summ += weights[idx]
    summ = (summ - limit) * param
    return summ


if __name__ == '__main__':
    ''' 产品代码&权重约束 '''
    fund_codes = ['003816.OF', '660107.OF', '519510.OF', '202302.OF', '000013.OF', '000010.OF',
                  '018060.OF', '006011.OF', '012404.OF', '013706.OF',
                  '960033.OF', '019447.OF', '017092.OF', '017091.OF',
                  '019005.OF', '517520.OF', '018543.OF', '159985.OF']
    single_limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                     (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    multi_limits = {"003816.OF,660107.OF,519510.OF,202302.OF,000013.OF,000010.OF": (0.743205, 0.743205),
                    "018060.OF,006011.OF,012404.OF,013706.OF": (0.215744, 0.215744),
                    "960033.OF,019447.OF,017092.OF,017091.OF": (0.009177, 0.009177),
                    "019005.OF,517520.OF,018543.OF,159985.OF": (0.031874, 0.031874)}

    ''' 相关系数,风险,收益,协方差 '''
    correlation_matrix = [[1.0, -0.02491800228542878, -0.03529455505473209, -0.09390864735802824, -0.032035228358317014,
                           -0.03189013505545331, -0.007879795940440335, 0.01295049807127088, -0.007175845623551147,
                           -0.014645156189186721, -0.027039425384296947, 0.02090714400905664, 0.0029993787151943014,
                           -0.00021055437306913893, -0.0021580973249399433, 0.03359266965322585, -0.025336418376560373,
                           -0.001557529665025934],
                          [-0.02491800228542878, 1.0, 0.8999183354408088, 0.7143301856313493, 0.9689227794215808,
                           0.9688869621854727, 0.03776031561621979, 0.09136761278704197, 0.04104607333104881,
                           -0.036854214333885256, 0.002128517129545959, 0.1066651993538786, 0.0949436725485615,
                           0.09482367439580361, 0.09462269341662502, 0.01814121538517647, -0.009614470370408391,
                           0.06741676699471282],
                          [-0.03529455505473209, 0.8999183354408088, 1.0, 0.7350413655262488, 0.9169494734798375,
                           0.9169808766862765, 0.04462673323832432, 0.11269171481466823, 0.047643778916904765,
                           -0.03702376157033994, 0.011967851296772749, 0.04329743269651667, 0.052857413182512254,
                           0.053557835696111854, 0.03443112718678784, -0.0008519205856549475, -0.03664612052780514,
                           0.07294409256983381],
                          [-0.09390864735802824, 0.7143301856313493, 0.7350413655262488, 1.0, 0.7087972300550152,
                           0.7087618537713289, 0.05510085397938882, 0.10001729059222274, 0.0833719042191852,
                           -0.036168230805587454, 0.015143222449519619, 0.06807895031199149, 0.07622255838081586,
                           0.07536154643702647, 0.06385969846944947, -0.0013548103870576823, -0.03628642174424017,
                           0.03590924410907699],
                          [-0.032035228358317014, 0.9689227794215808, 0.9169494734798375, 0.7087972300550152, 1.0,
                           0.9999994278135985, 0.03507605985349203, 0.0947473915859849, 0.04406999041092052,
                           -0.0326056541331293, 0.018730670293442966, 0.11947072013963599, 0.0950195843748273,
                           0.09235256340187137, 0.08622970251661277, 0.013964950550719028, -0.015509293099898932,
                           0.07429262861655032],
                          [-0.03189013505545331, 0.9688869621854727, 0.9169808766862765, 0.7087618537713289,
                           0.9999994278135985, 1.0, 0.03503915212324654, 0.09482355116674673, 0.044070044751953766,
                           -0.03260151696555401, 0.018687911908593795, 0.11929294431653316, 0.09488086876590299,
                           0.09221379014764564, 0.08631091512514964, 0.013782072106778452, -0.015654008185337874,
                           0.07428071693695114],
                          [-0.007879795940440335, 0.03776031561621979, 0.04462673323832432, 0.05510085397938882,
                           0.03507605985349203, 0.03503915212324654, 1.0, 0.00498343237719333, -0.0011847931609068898,
                           -0.0015499820241593552, -0.0010148224054982603, 0.05368581071078418, 0.021642974947996566,
                           0.022449790184474943, -0.0019215907683746154, -0.0006963530108662673, -0.05639225931804887,
                           0.0004108645852545974],
                          [0.01295049807127088, 0.09136761278704197, 0.11269171481466823, 0.10001729059222274,
                           0.0947473915859849, 0.09482355116674673, 0.00498343237719333, 1.0, 0.0006800908256227725,
                           -0.01862366625127312, -0.04844644754437627, 0.01684453331701437, 0.007783391108928574,
                           0.009348714650046839, 0.015105674516316758, 0.0015065304738359339, 0.00392282296663726,
                           0.03784013412488183],
                          [-0.007175845623551147, 0.04104607333104881, 0.047643778916904765, 0.0833719042191852,
                           0.04406999041092052, 0.044070044751953766, -0.0011847931609068898, 0.0006800908256227725,
                           1.0,
                           -0.0018717778884775795, -0.0010683180093891771, -0.0012514159561372107,
                           -0.002246874734559819,
                           -0.0022085484609783426, -0.001852637994737397, -0.0006547724352789414,
                           -0.0015004464699282607,
                           -0.028902790340035435],
                          [-0.014645156189186721, -0.036854214333885256, -0.03702376157033994, -0.036168230805587454,
                           -0.0326056541331293, -0.03260151696555401, -0.0015499820241593552, -0.01862366625127312,
                           -0.0018717778884775795, 1.0, -0.001677884323328096, -0.0020325214083830928,
                           -0.00315139109561219,
                           -0.0033742546593141676, -0.0011914255151646025, 0.0006110743152879624,
                           -0.0026899429959251443,
                           0.026550867764937997],
                          [-0.027039425384296947, 0.002128517129545959, 0.011967851296772749, 0.015143222449519619,
                           0.018730670293442966, 0.018687911908593795, -0.0010148224054982603, -0.04844644754437627,
                           -0.0010683180093891771, -0.001677884323328096, 1.0, -0.0016385592761114978,
                           -0.0025311868683162166, -0.002585673776454634, -0.00212027466350292, -0.0005446751100294957,
                           -0.0012896420941646545, 0.01250979650684191],
                          [0.02090714400905664, 0.1066651993538786, 0.04329743269651667, 0.06807895031199149,
                           0.11947072013963599, 0.11929294431653316, 0.05368581071078418, 0.01684453331701437,
                           -0.0012514159561372107, -0.0020325214083830928, -0.0016385592761114978, 1.0,
                           0.6211366294686574,
                           0.6159250333526626, -0.04039668655047027, -0.006363247758968427, 0.06227345576080737,
                           -0.011556304873436965],
                          [0.0029993787151943014, 0.0949436725485615, 0.052857413182512254, 0.07622255838081586,
                           0.0950195843748273, 0.09488086876590299, 0.021642974947996566, 0.007783391108928574,
                           -0.002246874734559819, -0.00315139109561219, -0.0025311868683162166, 0.6211366294686574, 1.0,
                           0.9783667602297074, -0.04009027575110249, -0.02850389666278512, 0.10579710587987343,
                           -0.03160784792556082],
                          [-0.00021055437306913893, 0.09482367439580361, 0.053557835696111854, 0.07536154643702647,
                           0.09235256340187137, 0.09221379014764564, 0.022449790184474943, 0.009348714650046839,
                           -0.0022085484609783426, -0.0033742546593141676, -0.002585673776454634, 0.6159250333526626,
                           0.9783667602297074, 1.0, -0.04369110103323669, -0.03329726418765513, 0.11175085171621618,
                           -0.03350876000943746],
                          [-0.0021580973249399433, 0.09462269341662502, 0.03443112718678784, 0.06385969846944947,
                           0.08622970251661277, 0.08631091512514964, -0.0019215907683746154, 0.015105674516316758,
                           -0.001852637994737397, -0.0011914255151646025, -0.00212027466350292, -0.04039668655047027,
                           -0.04009027575110249, -0.04369110103323669, 1.0, 0.27172607948670907, 0.059572046679374074,
                           0.024531052051725847],
                          [0.03359266965322585, 0.01814121538517647, -0.0008519205856549475, -0.0013548103870576823,
                           0.013964950550719028, 0.013782072106778452, -0.0006963530108662673, 0.0015065304738359339,
                           -0.0006547724352789414, 0.0006110743152879624, -0.0005446751100294957, -0.006363247758968427,
                           -0.02850389666278512, -0.03329726418765513, 0.27172607948670907, 1.0, 0.04897890683824761,
                           -0.024876822929851272],
                          [-0.025336418376560373, -0.009614470370408391, -0.03664612052780514, -0.03628642174424017,
                           -0.015509293099898932, -0.015654008185337874, -0.05639225931804887, 0.00392282296663726,
                           -0.0015004464699282607, -0.0026899429959251443, -0.0012896420941646545, 0.06227345576080737,
                           0.10579710587987343, 0.11175085171621618, 0.059572046679374074, 0.04897890683824761, 1.0,
                           0.007358146146054095],
                          [-0.001557529665025934, 0.06741676699471282, 0.07294409256983381, 0.03590924410907699,
                           0.07429262861655032, 0.07428071693695114, 0.0004108645852545974, 0.03784013412488183,
                           -0.028902790340035435, 0.026550867764937997, 0.01250979650684191, -0.011556304873436965,
                           -0.03160784792556082, -0.03350876000943746, 0.024531052051725847, -0.024876822929851272,
                           0.007358146146054095, 1.0]]
    annual_return = [0.022765414170588638, 0.022378830508153147, 0.024618637503817764, 0.022859118967620606,
                     0.023383312104743537, 0.02328080902750207, 0.0, 0.05582564887934027, 0.0, 0.0,
                     0.04989848081639381, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.41646522585689727]
    annual_std = [0.0011155267122790625, 0.0010546071719261345, 0.0010586112324833703, 0.0008964159789360627,
                  0.0011189527772928775, 0.0011139989246014392, 0.7978194704882879, 0.022544444853959792,
                  0.5729283890378397, 0.22904958967825195, 0.02774974001131007, 0.061691336341861996,
                  0.08537674782711843, 0.08572199835470311, 0.04907077916527096, 0.06985965841295717,
                  0.04445344969948427, 0.14026399768160455]
    cov_matrix = correlation_matrix * np.outer(annual_std, annual_std)

    ''' 生成随机权重 '''
    random_r, random_v, random_weight = create_random_weights(fund_codes, single_limits, multi_limits, annual_return,
                                                              cov_matrix)
    random_min_v_index = np.argmin(random_v)
    random_min_v = random_v[random_min_v_index]
    random_min_v_r = random_r[random_min_v_index]
    random_min_v_weight = random_weight[random_min_v_index]
    print(f"随机组合的最小风险: {random_min_v}, 随机组合最小风险对应的收益率: {random_min_v_r}")
    random_max_r_index = np.argmax(random_r)
    random_max_r = random_r[random_max_r_index]
    random_max_r_v = random_v[random_max_r_index]
    random_max_r_weight = random_weight[random_max_r_index]
    print(f"随机组合的最大收益率: {random_max_r}, 随机组合最大收益率对应的风险: {random_max_r_v}")

    # scatter = (
    #     Scatter()
    #     .add_xaxis(random_v.tolist())
    #     .add_yaxis("Portfolios", random_r.tolist(),
    #                symbol_size=3,  # 设置点的大小为最小
    #                label_opts=opts.LabelOpts(is_show=False)  # 不展示标签
    #                )
    #     .set_global_opts(
    #         title_opts=opts.TitleOpts(title="Markowitz Efficient Frontier"),
    #         xaxis_opts=opts.AxisOpts(name="Volatility", type_="value"),
    #         yaxis_opts=opts.AxisOpts(name="Return", type_="value"),
    #         tooltip_opts=opts.TooltipOpts(formatter="{c}"),
    #     )
    # )
    # # 保存并展示图表
    # scatter.render('markowitz_efficient_frontier.html')
    # print(f"保存html文件到: markowitz_efficient_frontier.html")

    ''' 优化,找到最大收益率对应的风险&最小风险对应的收益率 '''
    s_t = time.time()
    # 产品权重上下限
    bounds = single_limits
    # 总权重约束
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'ineq', 'fun': lambda weights: gini_impurity(weights) - 0.4}]
    # for key, (min_sum, max_sum) in multi_limits.items():
    #     indices = [fund_codes.index(code) for code in key.split(',')]
    #     constraints.append({'type': 'eq', 'fun': lambda weights, idx=indices: np.sum(weights[idx]) - min_sum})

    # def _get_weight_diff_lower(weights, idx_list, lower_limit):
    #     # 这个函数计算权重和与下限的差，应当非负
    #     return np.sum(weights[idx_list]) - lower_limit
    #
    #
    # def _get_weight_diff_upper(weights, idx_list, upper_limit):
    #     # 这个函数计算上限与权重和的差，也应当非负
    #     return upper_limit - np.sum(weights[idx_list])
    #
    #
    #
    # for key, (min_sum, max_sum) in multi_limits.items():
    #     indices = [fund_codes.index(code) for code in key.split(',')]
    #     # 添加保证权重总和不低于下限的约束
    #     constraints.append({'type': 'ineq', 'fun': _get_weight_diff_lower, 'args': (indices, min_sum)})
    #     # 添加保证权重总和不超过上限的约束
    #     constraints.append({'type': 'ineq', 'fun': _get_weight_diff_upper, 'args': (indices, max_sum)})

    spec_weight_range = [[0, 1, 2, 3, 4, 5, (0.743205, 0.743205)], [0, 1, 2, 3, 4, 5, (0.743205, 0.743205)],
                         [14, 15, 16, 17, (0.031874, 0.031874)], [10, 11, 12, 13, (0.009177, 0.009177)],
                         [10, 11, 12, 13, (0.009177, 0.009177)], [14, 15, 16, 17, (0.031874, 0.031874)],
                         [6, 7, 8, 9, (0.215744, 0.215744)], [6, 7, 8, 9, (0.215744, 0.215744)],
                         [14, (0, 1)], [4, (0, 1)], [17, (0, 1)],
                         [16, (0, 1)], [0, (0, 1)], [1, (0, 1)], [2, (0, 1)], [3, (0, 1)], [5, (0, 1)], [13, (0, 1)],
                         [6, (0, 1)],
                         [7, (0, 1)], [15, (0, 1)], [9, (0, 1)], [10, (0, 1)], [11, (0, 1)], [12, (0, 1)], [8, (0, 1)]]
    # 设定产品权重
    for i in range(len(spec_weight_range)):
        sc_index = spec_weight_range[i][:-1]
        w = spec_weight_range[i][-1]

        if w[0] > 0:
            constraints.append({'type': 'ineq', 'fun': _get_weight_diff, 'args': (sc_index, w[0], 1.0)})
        if w[1] < 1:
            constraints.append({'type': 'ineq', 'fun': _get_weight_diff, 'args': (sc_index, w[1], -1.0)})

    # 最小化风险
    min_v, min_v_r, min_v_w = find_the_min_volatility(annual_return, cov_matrix, random_min_v_weight, bounds,
                                                      constraints)
    print(f"最小风险: {min_v}, 最小风险对应的收益率: {min_v_r}")
    # 最大化收益率
    max_r, max_r_v, max_r_w = find_the_max_return(annual_return, cov_matrix, random_max_r_weight, bounds, constraints)
    print(f"最大收益率对应的风险: {max_r_v}, 最大收益率: {max_r}")

    # 将波动率最小的组合以及收益率最大组合的 权重 加入到随机组合权重数据中
    random_weights = np.vstack([random_weight, min_v_w, max_r_w])
    # 将波动率最小的组合以及收益率最大组合的 收益率 加入到随机组合收益数据中
    random_returns = np.hstack([random_r, min_v_r, max_r])
    # 将波动率最小的组合以及收益率最大组合的 波动率 加入到随机组合波动率数据中
    random_volatility = np.hstack([random_v, min_v, max_r_v])

    std_array, ret_array, _ = find_the_efficient_frontier_with_random_weights(60, min_v, max_r_v,
                                                                              bounds, constraints, annual_return,
                                                                              cov_matrix, random_weights,
                                                                              random_returns, random_volatility)
    print(std_array)
    print(ret_array)

    ''' 画图 '''
    draw_efficient_frontier(random_volatility, random_returns, std_array, ret_array)

    # plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams.update({"font.size": 12})
    #
    # fig, ax = plt.subplots(figsize=(10, 5))
    # plt.figure(figsize=(10, 5))
    # ax.plot(std_array, ret_array, c='royalblue', marker='.')
    # ax.grid(True)
    # ax.set_xlabel('波动率')
    # ax.set_ylabel('期望收益')
    # ax.set_title('有效前沿曲线')
    # plt.show()

    print(time.time() - s_t)

    pass
