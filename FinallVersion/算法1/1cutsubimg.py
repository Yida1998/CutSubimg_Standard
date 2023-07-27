"""
2021年07月09日19:19:32 步骤设计完整,每一步应该可以完整运行才行
输入:文件->n个类别->每个类别有m张图像
输出:子图文件夹./subimg, 每个土壤图像都一个子图文件夹
"""
import os
import shutil
import time

import numpy as np
from PIL import Image

# 土壤类别标签
class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']


class CutSbuImage:
    def __init__(self, path, img_size, move_step):
        self.path = path
        self.img_size = img_size
        self.move_step = move_step

    def master(self):
        self.save_img(self.path, True)  # 对输入图像进行切子图操作, 为True时才进行操作

    def save_img(self, img_path, mark):
        """
        输入:切子图,对输入的文件进行切子图操作
        然后保存在当前文件夹的subimg下
        mark:标记位,如果为True才开始
        :return:
        """
        if mark:
            print("切子图:[True]")
            img_save = os.path.join(img_path, 'subimg')  # 存放子图路径
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
        return path_now  # 返回子图路径


if __name__ == '__main__':
    """
    输入:path
    类中:设置子图大小和移动步长      修改: 把它设置到类外面来了
    """
    start = time.time()
    path = '/Users/yida/Desktop/train'
    img_size = 224  # 子图大小
    move_step = 224  # 移动步长
    c = CutSbuImage(path, img_size, move_step)
    c.master()  # 启动类
    end = time.time()
    print("时间消耗:", end-start)
