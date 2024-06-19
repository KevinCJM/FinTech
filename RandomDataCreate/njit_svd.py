# -*- encoding: utf-8 -*-
"""
@File: njit_svd.py
@Modify Time: 2024/6/18 10:48       
@Author: Kevin-Chen
@Descriptions: 手动实现奇异值分解(SVD)用于计算伪逆矩阵, 从而可用用numba的njit加速器, 以提高计算效率
"""
import time
import warnings
import numpy as np
from scipy.linalg import null_space
from numba import njit, float64, int64

warnings.filterwarnings('ignore')  # 不显示 warnings


@njit(float64(float64[:, :]))
def det_numba(matrix):
    """
    计算矩阵的行列式值。

    该函数使用Numba的Just-In-Time编译器优化，以提升数学运算的执行速度。
    它接受一个二维浮点数数组作为输入，并返回该矩阵的行列式值。

    参数:
    matrix: 二维浮点数数组，表示需要计算行列式的矩阵。

    返回值:
    浮点数，表示输入矩阵的行列式值。
    """
    # 使用NumPy的线性代数函数计算矩阵的行列式
    return np.linalg.det(matrix)


# 正常的行列式计算
def det_normal(matrix):
    return np.linalg.det(matrix)


# # 测量使用 njit 的时间
# start_njit = time.time()
# det_numba(A)
# end_njit = time.time()
# njit_time = end_njit - start_njit
#
# # 测量不使用 njit 的时间
# start_normal = time.time()
# det_normal(A)
# end_normal = time.time()
# normal_time = end_normal - start_normal
#
# print(det_numba(A))
# print(f"使用 njit 的时间: {njit_time} 秒")
# print(f"不使用 njit 的时间: {normal_time} 秒")

# 使用njit装饰器对函数进行编译优化，提高计算性能
# 使用njit装饰器对函数进行编译优化，提高计算性能
@njit(float64[:](float64[:, :]))
def compute_eigen_values(matrix):
    """
    计算给定矩阵的特征值。

    参数:
    matrix: float64[:, :] 类型的二维数组，表示输入的矩阵, 需要是方阵。

    返回值:
    float64[:] 类型的一维数组，包含输入矩阵的特征值。
    """
    # 使用numpy的linalg.eigvals函数计算矩阵的特征值
    eigen_values = np.linalg.eigvals(matrix)
    # 由于奇异值矩阵中奇异值的顺序是按降序排列的, 所以需要进行排序操作
    eigen_values[:] = np.sort(eigen_values)[::-1]  # 从大到小排序, 避免分配新的内存空间
    # 返回计算得到的特征值数组
    return eigen_values


# 根据给定的特征值数组，创建一个奇异值矩阵
@njit(float64[:, :](float64[:], int64, int64))
def create_sigma_matrix(the_eigen_values, m, n):
    """
    根据给定的特征值数组，创建一个奇异值矩阵。

    参数:
    the_eigen_values: 一维数组，包含特征值。
    m: 整数，指定奇异值矩阵的行数。
    n: 整数，指定奇异值矩阵的列数。

    返回:
    一个二维数组，为生成的奇异值矩阵。
    """
    # 计算特征值的平方根，得到奇异值
    singular_values = np.sqrt(the_eigen_values)
    # 初始化一个大小为(m, n)的零矩阵, 初值为0
    the_sigma_matrix = np.zeros((m, n))
    # 将计算得到的奇异值填充到矩阵的对角线上
    np.fill_diagonal(the_sigma_matrix, singular_values[:min(m, n)])
    return the_sigma_matrix


# 传入方阵, 计算特征值和特征向量
@njit((float64[:, :],))  # 指定输入参数 a 为二维浮点数数组
def create_eigen_matrix(a):
    """
    计算并返回给定方阵a的特征值和特征向量。

    参数:
    a: 二维浮点数数组，输入的矩阵必须是方阵。

    返回值:
    一个元组，包含两个数组：
    - 特征值数组，按降序排列；
    - 特征向量矩阵，对应的特征向量按相同的顺序排列。
    """
    # 使用numpy的linalg.eig函数计算矩阵a的特征值和特征向量
    the_eigen_values, the_eigen_vectors = np.linalg.eig(a)

    # 获取排序后的索引（从大到小）
    sorted_indices = np.argsort(the_eigen_values)[::-1]

    # 根据排序后的索引重新排序特征值和特征向量
    the_eigen_values[:] = the_eigen_values[sorted_indices]
    the_eigen_vectors[:, :] = the_eigen_vectors[:, sorted_indices]

    # 指定返回类型为两个浮点数组，一个是一维数组，另一个是二维数组
    return the_eigen_values, the_eigen_vectors


def find_eigenvectors(a, the_eigen_values):
    n = a.shape[0]
    eigenvectors = []

    for lamb in the_eigen_values:
        # 构建矩阵 A - λI
        matrix = a - np.eye(n) * lamb
        # 计算该矩阵的零空间，即寻找齐次线性方程的解
        ns = null_space(matrix)
        # 添加到结果列表中
        eigenvectors.append(ns)

    return np.hstack(eigenvectors)


@njit
def svd(matrix):
    u, s, vt = np.linalg.svd(matrix)
    return u, s, vt


@njit(float64[:, :](float64[:, :]))
def cal_pseudo_inverse(nd_array):
    """
    计算给定二维数组的伪逆矩阵。

    通过奇异值分解（SVD）方法计算矩阵的伪逆。此方法适用于处理包括奇异矩阵在内的各种矩阵。

    参数:
    nd_array: 二维numpy数组，表示需要计算伪逆的矩阵。

    返回:
    二维numpy数组，表示输入矩阵的伪逆。
    """
    # 对输入矩阵进行奇异值分解
    u_matrix, s_array, vt_matrix = svd(nd_array)

    # 创建一个与原矩阵形状相同但元素全为0的矩阵
    sigma = np.zeros_like(nd_array)  # 构建一个全是0的矩阵, 形状与A一致
    np.fill_diagonal(sigma, s_array)  # 将计算得到的奇异值填充到矩阵的对角线上

    # 对sigma矩阵的非零元素取倒数，并转置, 用于构建奇异值矩阵的伪逆矩阵
    sigma_plus = np.where(sigma != 0, 1.0 / sigma, 0).T

    # 计算伪逆矩阵: 通过vt_matrix的转置、sigma_plus和u_matrix的转置的乘积得到
    a_plus = vt_matrix.T @ sigma_plus @ u_matrix.T
    return a_plus


# @njit(float64[:, :](float64[:, :], float64[:]))
# def cal_regression_coefficients(x_nd_array, y_array):
#     # 对输入矩阵进行奇异值分解
#     u_matrix, s_array, vt_matrix = svd(x_nd_array)
#
#     # 创建一个与原矩阵形状相同但元素全为0的矩阵
#     sigma = np.zeros_like(x_nd_array)  # 构建一个全是0的矩阵, 形状与A一致
#     np.fill_diagonal(sigma, s_array)  # 将计算得到的奇异值填充到矩阵的对角线上
#
#     # 对sigma矩阵的非零元素取倒数，并转置, 用于构建奇异值矩阵的伪逆矩阵
#     sigma_plus = np.where(sigma != 0, 1.0 / sigma, 0).T
#
#     # 计算伪逆矩阵: 通过vt_matrix的转置、sigma_plus和u_matrix的转置的乘积得到
#     a_plus = vt_matrix.T @ sigma_plus @ u_matrix.T
#
#     # 计算回归系数
#     coefficients = a_plus @ y_array
#     return coefficients


if __name__ == '__main__':
    ''' 构建示例矩阵 A '''
    a_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(cal_pseudo_inverse(a_array))
    pass
