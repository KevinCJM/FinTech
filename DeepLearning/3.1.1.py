import numpy as np
# 导入3D模块包
import mpl_toolkits.mplot3d as axes3d

# 用meshgrid函数生成2组网格点坐标矩阵, 用于表示w0和w1在3D坐标上的变化, 因此w0和w1都是 500*500的矩阵
grid_w0, grid_w1 = np.meshgrid(np.linspace(0, 9, 500), # linespace函数用于线性等分(将0-9分为500个点)
np.linspace(0, 4, 500))
# 创建一个loss矩阵,结构与w0/w1相同
grid_loss = np.zeros_like(grid_w0)
# 遍历train_x与train_y元素, 用500个w0和500个w1矩阵去计算相应的loss值, 并添加入loss矩阵
for x, y in zip(train_x, train_y):
    grid_loss += ((grid_w0 + x*grid_w1 - y) ** 2) / 2
mp.figure('Loss Function') # 创建图表
ax = mp.gca(projection='3d') # 获取当前坐标系对象
mp.title('Loss Function', fontsize=20) # 定义图表标题
ax.set_xlabel('w0', fontsize=14) # 设置x轴名称
ax.set_ylabel('w1', fontsize=14) # 设置y轴名称
ax.set_zlabel('loss', fontsize=14) # 设置z轴名称
# 绘制3D平面图形: 用于表现各种w0和w1的组合产生的loss值
ax.plot_surface(grid_w0, # x轴数据用w0
grid_w1, # y轴数据用w1
grid_loss, # z轴数据用loss
rstride=10, # rstride（row）指定行的跨度
cstride=10, # cstride(column)指定列的跨度
cmap='jet', # 设置颜色映射
alpha=0.8) # 设置透明度
# 绘制3D点图形: 用于表现1000次迭代使w0和w1的变化路径
ax.plot(w0, # x轴坐标点
w1, # y轴坐标点
losses, # z轴坐标点
'-o', # 点的显示模式
c='orangered',
label='BGD')
mp.legend()
mp.show