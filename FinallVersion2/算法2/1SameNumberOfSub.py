"""
Author: yida
Time is: 2021/11/28 20:21 
this Code:  为了让算法2生成相同子图数的数据, 输入算法1获取的土壤子图结果, 然后访问每张图像的子图数, 并且保存在字典中
"""
import os

import numpy as np

class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']  # 类标签

if __name__ == '__main__':
    # 训练集
    dict_train = {}
    root_path_train = '/Users/yida/Desktop/train/subimg'

    for i in class_img:
        path_class = os.path.join(root_path_train, i)  # i个类别

        file_class = os.listdir(path_class)
        if '.DS_Store' in file_class:
            file_class.remove('.DS_Store')
        for j in file_class:
            img_num = os.path.join(path_class, j)  # j张大图

            file_sub = os.listdir(img_num)
            if '.DS_Store' in file_sub:
                file_sub.remove('.DS_Store')
            m = len(file_sub)  # m张子图
            print("第{}的{}张土壤图像共有{}张土壤子图".format(i, j, m))
            dict_train['{}_{}'.format(i, j)] = m
    np.save('dict_train.npy', dict_train)
    dict_train = np.load('dict_train.npy', allow_pickle=True).item()  # 转换成字典
    print("训练集:", dict_train)
    print(type(dict_train))
