"""
Author: yida
Time is: 2021/11/18 20:03 
this Code: 对直方图进行区间划分, 然后保存图像
"""
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
w = np.load("/Users/yida/PycharmProjects/MyPaper/CutSubImage/Init_Alor_R_W_Cutsubimg/0907/plt_img/w.npy")
point = np.where(w != 0)
# 统计w矩阵中的非零元素
value = w[point]
# print(value)
max_v = max(value)
min_v = min(value)
print("min : ", min(value), "max : ", max(value))

x = value
mu = np.mean(x)  # 计算均值
sigma = np.std(x)
num_bins = 256  # 直方图柱子的数量
n, bins, patches = plt.hist(x, num_bins, density=1, alpha=0.75)
plt.grid(True)
plt.xlabel('weights')  # 绘制x轴
plt.ylabel('Probability')  # 绘制y轴

# 区间限制
# 获取x数组
x = np.arange(min_v, max_v, (max_v - min_v) / 8)
print(x)
# 第一个点替换, 在加一个点
x[0] = x[0] + (x[1] - x[0])/2
x = np.append(x, (x[-1] - x[-2]) / 2 + x[-1])
# 获取y数组
y = np.arange(0, num_bins, num_bins//8)
y[0] = 16
y = np.append(y, num_bins - num_bins//8)
y = n[y]
# 遍历绘图
for i in range(len(x)):
    # if i == 0 :
    #     plt.vlines([x[i]], 0, y[i], colors='k')
    # elif i == len(x)-1:
    #     plt.vlines([x[i]], 0, y[i], colors='k')
    # else:
    #     plt.vlines([x[i]], 0, y[i], colors='r')
    plt.vlines([x[i]], 0, y[i], colors='r')

plt.title('Interval Limit = 8')
plt.savefig("w_hist.svg", dpi=600)

plt.show()



