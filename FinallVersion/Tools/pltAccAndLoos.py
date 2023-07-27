"""
Author: yida
Time is: 2021/12/18 09:49 
this Code: 读取npy中的数据, 绘制损失以及准确率识别曲线
"""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    path = '/Users/yida/Desktop/1730_my'

    loss_path = path + '/1var_loss.npy'
    acc_path = path + '/1var_acc.npy'

    plt.figure()
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    loss = np.load(loss_path)
    plt.plot(loss, 'g')
    plt.title(loss_path)
    plt.ylabel('Loss')
    plt.xlabel('Step')

    # 绘制acc曲线
    plt.subplot(2, 2, 4)
    acc = np.load(acc_path)
    plt.plot(acc, 'r')
    plt.title(acc_path)
    plt.ylabel('Acc')
    plt.xlabel('Step')
    plt.show()
