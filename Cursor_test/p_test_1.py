import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from pykalman import KalmanFilter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 尝试设置中文字体
try:
    font = FontProperties(family='Heiti TC')  # 对于简体中文，可以尝试 'Heiti SC'
except:
    print("未找到指定字体，将使用系统默认字体")
    font = FontProperties()

# 从开源金融数据平台获取黄金指数数据
# 这里使用 yfinance 库作为示例，您可能需要先安装它：pip install yfinance

# import yfinance as yf

# # 定义黄金ETF的股票代码（以SPDR黄金信托ETF为例）
# gold_symbol = "GLD"

# # 获取黄金指数数据
# # 使用yfinance库下载黄金ETF数据
# gold_data = yf.download(
#     gold_symbol,  # 股票代码，这里是黄金ETF的代码
#     start="2010-01-01",  # 开始日期，从2010年1月1日开始
#     end=pd.Timestamp.today().strftime('%Y-%m-%d')  # 结束日期，使用当前日期
# )

# # gold_data 是一个包含以下列的DataFrame:
# # Date: 日期
# # Open: 开盘价
# # High: 最高价
# # Low: 最低价
# # Close: 收盘价
# # Adj Close: 调整后的收盘价
# # Volume: 交易量

# # 如果需要保存数据到CSV文件
# gold_data.to_parquet("gold_index_data.parquet")

# print("黄金指数数据已成功获取和保存。")


# 读取保存的黄金指数数据
gold_data = pd.read_parquet("/Users/chenjunming/Desktop/FinTech/Cursor_test/gold_index_data.parquet")
gold_data = gold_data[['Adj Close']]

# 应用高斯滤波
sigma = 30  # 标准差，控制平滑程度 (越大越平滑)
gold_data['Gaussian'] = gaussian_filter1d(gold_data['Adj Close'].values, sigma)

# 应用移动平均
window = 60  # 移动窗口大小 (越大越平滑)
gold_data['MA'] = gold_data['Adj Close'].rolling(window=window).mean()

# 应用卡尔曼滤波
transition_covariance = 0.01 # (越大越平滑)
observation_covariance = 1
kf = KalmanFilter(initial_state_mean=gold_data['Adj Close'].iloc[0],
                  n_dim_obs=1,
                  n_dim_state=1,
                  transition_matrices=[1],
                  observation_matrices=[1],
                  initial_state_covariance=1,
                  observation_covariance=observation_covariance,
                  transition_covariance=transition_covariance)

kalman_smoothed, _ = kf.smooth(gold_data['Adj Close'].values)
gold_data['Kalman'] = kalman_smoothed

# 创建交互式图表
fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter(x=gold_data.index, y=gold_data['Adj Close'], name='原始数据', mode='lines', 
                         line=dict(color='black'), opacity=0.4))
fig.add_trace(go.Scatter(x=gold_data.index, y=gold_data['Gaussian'], name='高斯滤波', mode='lines'))
fig.add_trace(go.Scatter(x=gold_data.index, y=gold_data['MA'], name='移动平均', mode='lines'))
fig.add_trace(go.Scatter(x=gold_data.index, y=gold_data['Kalman'], name='卡尔曼滤波', mode='lines'))

fig.update_layout(
    title='黄金ETF调整后收盘价 - 原始数据 vs 高斯滤波 vs 移动平均 vs 卡尔曼滤波',
    xaxis_title='日期',
    yaxis_title='价格',
    legend_title='数据类型',
    hovermode='x unified'
)

# 显示图表
fig.show()

# 如果需要保存为HTML文件
fig.write_html("gold_price_analysis.html")

# 计算并打印滞后效果
peak_original = gold_data['Adj Close'].idxmax()
peak_gaussian = gold_data['Gaussian'].idxmax()
peak_ma = gold_data['MA'].idxmax()
peak_kalman = gold_data['Kalman'].idxmax()

print(f"原始数据峰值日期: {peak_original}")
print(f"高斯滤波峰值日期: {peak_gaussian}")
print(f"移动平均峰值日期: {peak_ma}")
print(f"卡尔曼滤波峰值日期: {peak_kalman}")
print(f"高斯滤波滞后天数: {(peak_gaussian - peak_original).days}")
print(f"移动平均滞后天数: {(peak_ma - peak_original).days}")
print(f"卡尔曼滤波滞后天数: {(peak_kalman - peak_original).days}")