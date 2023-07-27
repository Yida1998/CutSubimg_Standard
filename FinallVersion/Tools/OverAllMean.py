"""
Author: yida
Time is: 2022/1/3 20:10 
this Code: 计算整体数据的均值和标准差
"""
import os

import cv2
import numpy as np

classes = ['9', '0', '7', '6', '1', '10', '8', '4', '3']
b_m = []
g_m = []
r_m = []
b_s = []
g_s = []
r_s = []

if __name__ == '__main__':
    root = '/Users/yida/Desktop/train/dataset_1'
    file = os.listdir(root)
    for i in classes:
        file_path = os.path.join(root, i)
        file_img = os.listdir(file_path)
        for item in file_img:
            if item.endswith('.jpg'):
                img_path = os.path.join(file_path, item)
                img = cv2.imread(img_path)
                img_b = np.mean(img[:, :, 0] / 255.)
                img_g = np.mean(img[:, :, 1] / 255.)
                img_r = np.mean(img[:, :, 2] / 255.)
                # 均值
                b_m.append(img_b)
                g_m.append(img_g)
                r_m.append(img_r)
                # 标准差
                img_b = np.std(img[:, :, 0] / 255.)
                img_g = np.std(img[:, :, 1] / 255.)
                img_r = np.std(img[:, :, 2] / 255.)
                # 均值
                b_s.append(img_b)
                g_s.append(img_g)
                r_s.append(img_r)
    print("mean[{}, {}, {}]".format(np.mean(r_m), np.mean(g_m), np.mean(b_m)))
    print("std[{}, {}, {}]".format(np.mean(r_s), np.mean(g_s), np.mean(b_s)))