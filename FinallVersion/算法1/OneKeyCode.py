"""
Author: yida
Time is: 2021/11/28 19:42 
this Code:
[最终版本]  :  输入路径, 一键启动, 移动步长式子图切割算法
2021年12月18日09:36:34 新增:将类中的img_size和move_step 放到init中
"""
import os
import shutil
import time

import numpy as np
from PIL import Image

# 土壤类别标签
# class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']
class_img = ['灰棕紫泥大眼泥土', '红棕紫泥红石骨子土', '灰棕紫泥石骨子土', '红棕紫泥红紫砂泥土', '灰棕紫泥砂土', '暗紫泥二泥土', '灰棕紫泥半砂半泥土', '暗紫泥油石骨子土', '暗紫泥夹砂土', '红棕紫泥红棕紫泥土', '暗紫泥大泥土']


class CutSbuImage:
    def __init__(self, path, img_size, move_step):
        self.path = path
        self.img_size = img_size
        self.move_step = move_step
        self.total = 0  # 标记总土壤图像数目

    def master(self):
        """
        :return:总的子图数, 为了计算单张子图的时间
        """
        self.save_img(self.path, True)  # 对输入图像进行切子图操作, 为True时才进行操作
        return self.total

    def save_img(self, img_path, mark):
        """
        输入:切子图,对输入的文件进行切子图操作
        然后保存在当前文件夹的subimg下
        mark:标记位,如果为True才开始
        :return:
        """
        if mark:
            print("切子图:[True]")
            img_save = os.path.join(img_path, 'subimg_1')  # 存放子图路径
            if os.path.exists(img_save):
                shutil.rmtree(img_save)
                print("文件夹已存在,正在删除...")
            # 开始遍历图像
            for i in class_img:
                sub_path = os.path.join(img_path, i)  # 子路径
                file_sub = os.listdir(sub_path)  # 子文件
                for item in file_sub:
                    if item.endswith('.jpg'):
                        img_sub = os.path.join(sub_path, item)  # 子土壤图像路径
                        img_sub_save = os.path.join(img_save, i, item.split('.')[0])  # 保存图像的路径
                        self.cut_img(img_sub, img_sub_save, i, item)
            print("************切子图已全部完成*************")
        else:
            print("切子图:[False]")

    def cut_img(self, img_path, img_save, class_i, img_item):
        """
        :param img_path: 图像路径
        :param img_save: 存储路径
        :param class_i:第i类
        :param img_item: 图像名称
        :return:
        """
        size = self.img_size  # 设置你要裁剪的小图的宽度
        step = self.move_step  # 设置移动步长

        path = img_path  # 读取图像路径
        im = Image.open(path)
        im = im.convert('RGB')  # 有保存信息, 转换成rgb试试
        img_size = im.size

        m = img_size[0]  # 读取图片的长度

        n = img_size[1]  # 读取图片的宽度

        num = 0  # 循环次数当做文件名
        path_now = img_save
        if os.path.exists(path_now) == False:  # 生成的文件夹放在当前目录下
            os.makedirs(path_now)  # 生成目录
        for i in range(m // step):
            for j in range(n // step):
                flag = 0  # 设置标记
                x = step * i
                y = step * j
                if y + size > n or x + size > m:
                    break
                region = im.crop((x, y, x + size, y + size))  # 裁剪区域 超过了范围会补黑色
                # 代优化!!!!!!!!!!! 用numpy判断
                # 判断有白色背景就不保存
                img = np.array(region)
                # numpy 加速
                point = np.sum((img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255))
                if point > 0:  # 存在白色点像素 直接flog = 1
                    flag = 1
                if flag == 0:  # 保存图像
                    region.save(path_now + '/' + str(num) + ".jpg")  # str(i)是裁剪后的编号，此处是0到99
                    num += 1
                    print("\r正在处理第{}张图".format(num), end='')
                    # 长移动i次 宽移动j次
        print("")
        print('切子图操作已完成...第{}类第{}张土壤图像共获得{}张子图'.format(class_i, img_item.split('.')[0], num))
        self.total += 1
        return path_now  # 返回子图路径


def main1(path_root1, img_sizes, move_steps):
    """
    切子图算法1
    输入:path
    类中:设置子图大小和移动步长
    """
    path = path_root1
    main1_start = time.time()  # 开始时间
    c = CutSbuImage(path, img_sizes, move_steps)
    total = c.master()  # 启动类
    main1_end = time.time()  # 结束时间
    total_time = main1_end - main1_start
    print("Total总土壤数->{} | Time总时间花费->{:.6f} | Single单张图像时间花费->{:.6f}".format(total, total_time, total_time / total))


def main2(path_root2):
    """
    汇总子图
    :param path_root2: 路径
    :return:
    """
    path = os.path.join(path_root2, 'subimg_1')
    target = os.path.abspath(os.path.join(path, '../dataset_1'))  # subimg当前目录
    # 有就删除, 没有就重新生成
    if os.path.exists(target):
        shutil.rmtree(target)
        os.makedirs(target)
    else:
        os.makedirs(target)
    print("target:", target)
    for i in class_img:
        sub_path = os.path.join(path, i)  # 子文件路径
        sub_file = os.listdir(sub_path)  # 子文件下的图像
        if '.DS_Store' in sub_file:
            sub_file.remove('.DS_Store')
        for j in sub_file:  # 第j张图
            sub_file_img = os.path.join(sub_path, j)  # 下一步获取子图
            file_subimg = os.listdir(sub_file_img)  # 获取子图
            for item in file_subimg:
                if item.endswith('.jpg'):
                    # 子图路径
                    target_name = '_' + str(i) + '_' + str(j) + '.jpg'
                    sub_img_path = os.path.join(sub_file_img, item)  # 原路径
                    sub_class = os.path.join(target, i)  # 生成路径
                    if not os.path.exists(sub_class):
                        os.makedirs(sub_class)  # 不存在就生成文件夹
                    target_path = os.path.join(sub_class, item.split('.')[0] + target_name)  # 移动路径
                    shutil.copy(sub_img_path, target_path)  # 开始移动
        print("正在移动第{}个类".format(i))
    print("任务完成...")


if __name__ == '__main__':
    start = time.time()
    path_root_ = "/Users/yida/Desktop/兆易杯/数据集/high/val"
    img_size = 224  # 子图大小
    move_step = 112  # 移动步长
    main1(path_root_, img_size, move_step)  # 子图切割

    main2(path_root_)  # 子图汇总

    end = time.time()
    print("运行时间:", end - start)
