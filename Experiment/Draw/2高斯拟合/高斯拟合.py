"""
Author: yida
Time is: 2021/11/18 20:03 
this Code: 
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import rayleigh


# 加载数据
w = np.load("/Users/yida/PycharmProjects/MyPaper/CutSubImage/Init_Alor_R_W_Cutsubimg/0907/plt_img/w.npy")
point = np.where(w != 0)
# 统计w矩阵中的非零元素
value = w[point]
# print(value)
max_v = max(value)
min_v = min(value)
print("min : ", min(value), "max : ", max(value))
# plt.title("Weight Hist")
# n, bins, patches = plt.hist(value, bins=10)
# plt.savefig("w_hist.svg", dpi=300)
# print("频数n为 : ", n)
# print("bins : ", bins)
# plt.show()

x = value
# 这里填入你的数据list如果已经是array格式就不用转化了
# n, bins, patches = plt.hist(x, 20, density=1, facecolor='blue', alpha=0.75)  #第二个参数是直方图柱子的数量
mu = np.mean(x)  # 计算均值
sigma = np.std(x)
num_bins = 256  # 直方图柱子的数量
n, bins, patches = plt.hist(x, num_bins, density=1, alpha=0.75)
# 直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
# 模拟正态分布
y = norm.pdf(bins, mu, sigma)  # 拟合一条最佳正态分布曲线y
print("gass: m={} s={}".format(mu, sigma))
# 模拟relay分布
# y = rayleigh.pdf(bins, mu, sigma)  # 拟合一条最佳正态分布曲线y


plt.grid(True)
plt.plot(bins, y, 'r--')  # 绘制y的曲线
plt.xlabel('values')  # 绘制x轴
plt.ylabel('Probability')  # 绘制y轴
plt.xlim(0, 255)
plt.title('Histogram : $\mu$=' + str(round(mu, 2)) + ' $\sigma=$' + str(round(sigma, 2)))  # 中文标题 u'xxx'
# plt.subplots_adjust(left=0.15)#左边距
plt.savefig("w_hist.svg", dpi=600)

plt.show()


import scipy.stats


print(scipy.stats.kstest(x, 'norm', args=(mu, sigma)))
print(x.shape)
