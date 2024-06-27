# -*- encoding: utf-8 -*-
"""
@File: tools_func.py
@Modify Time: 2024/6/13 18:28       
@Author: Kevin-Chen
@Descriptions: 
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline  # 用于在Jupyter中显示图表


def use_svg_display():
    """ 使用svg格式在Jupyter中显示绘图
    该函数的作用是在Jupyter Notebook中设置Matplotlib绘图的输出格式为SVG矢量图
    Matplotlib的输出格式设置为SVG，使得在Jupyter Notebook中显示的绘图是以SVG格式呈现的，具有较高的质量和可缩放性。
    """
    backend_inline.set_matplotlib_formats('svg')  # 用矢量图显示


def set_figsize(figsize=(3.5, 2.5)):  # @save
    """ 设置matplotlib的图表大小
    该函数用于设置matplotlib绘图的图表大小。函数内部首先调用use_svg_display()函数，使得图表以矢量图的形式显示，
    然后通过plt.rcParams['figure.figsize'] = figsize语句将图表大小设置为传入的figsize参数值。
    """
    use_svg_display()  # 用矢量图显示
    plt.rcParams['figure.figsize'] = figsize  # 设置图表大小


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """ 设置matplotlib的轴
    该函数用于设置matplotlib图形的轴属性，包括x轴和y轴的标签、比例尺、范围，以及图例和网格线的设置。通过传入相应的参数，可以自定义轴的显示效果。

    参数:
    - axes: matplotlib的Axes对象，表示图表的轴。
    - xlabel: 字符串，设置x轴的标签。
    - ylabel: 字符串，设置y轴的标签。
    - xlim: 元组，设置x轴的显示范围。
    - ylim: 元组，设置y轴的显示范围。
    - xscale: 字符串，设置x轴的尺度类型，如'linear'表示线性尺度。
    - yscale: 字符串，设置y轴的尺度类型，如'log'表示对数尺度。
    - legend: 列表或元组，包含图例的信息。

    返回值:
    无
    """
    axes.set_xlabel(xlabel)  # 设置x轴标签
    axes.set_ylabel(ylabel)  # 设置y轴标签
    axes.set_xscale(xscale)  # 设置x轴的比例尺
    axes.set_yscale(yscale)  # 设置y轴的比例尺
    axes.set_xlim(xlim)  # 设置x轴的范围
    axes.set_ylim(ylim)  # 设置y轴的范围

    # 如果legend不为空，则添加图例
    if legend:
        axes.legend(legend)  # 设置图例
    axes.grid()  # 设置网格线


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """
    绘制一个或多个数据集的图形。

    参数:
    - X: 数据集，可以是单个数据序列或多组数据序列。如果多组数据序列被提供，它们应该以列表的形式给出。
    - Y: 可选参数，用于指定要绘制的Y轴数据。如果没有提供，将假设Y数据与X数据相同。
    - xlabel, ylabel: 分别指定X轴和Y轴的标签。
    - legend: 图例标签列表，用于标识图中的不同数据序列。
    - xlim, ylim: 分别指定X轴和Y轴的显示范围。
    - xscale, yscale: 指定X轴和Y轴的缩放类型，如'linear'（线性）或'log'（对数）。
    - fmts: 数据序列的格式字符串列表，用于指定不同数据序列的绘制样式。
    - figsize: 指定生成图形的尺寸。
    - axes: 可选的matplotlib轴对象，用于在指定的轴上绘制图形。

    返回值:
    无返回值。此函数直接生成并显示图形。
    """

    if legend is None:
        legend = []

    # 设置图形尺寸
    set_figsize(figsize)
    # 获取当前活动的轴对象，如果没有指定axes参数
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        """
        判断输入X是否为单个一维序列。

        参数:
        - X: 输入的数据对象。

        返回值:
        - True如果X是一维序列或列表，否则返回False。
        """
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    # 根据X的情况调整Y的格式，确保X和Y的格式一致
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    # 确保X和Y的数据组数相同
    if len(X) != len(Y):
        X = X * len(Y)
    # 清理当前轴上的所有元素，为新的绘图做准备
    axes.cla()
    # 遍历数据序列和格式字符串列表，绘制每组数据
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    # 设置轴标签、范围和图例
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# 计算器工具
class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    x = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(x, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

