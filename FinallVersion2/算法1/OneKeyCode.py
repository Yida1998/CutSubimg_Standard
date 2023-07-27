"""
Author: yida
Time is: 2021/11/28 19:42 
this Code:算法1移动步长式切子图算法
[最终版本]  :  输入路径, 一键启动, 移动步长式子图切割算法
命令行参数:
    path_root: 以分割的图像路径
    img_size: 子图大小
    img_step: 移动步长
    bbox:是否获取最小外接矩形'Rectangle'
    土壤类别的标签 如果不算现在这个数据集需要修改 -> 以后及时修改即可

2022年03月02日16:03:21
有时候这个逻辑有问题的话, 子集很难发现, 但是以后调试的时候遇见也难得修改, 所以写代码的时候就把情况考虑清楚
应该用全局变量的就用全局变量
多测试, 脑子灵活点
"""
import argparse
import os
import shutil
import time

import numpy as np
from PIL import Image

from BoundingBox_Class import main as BBmain

# 土壤类别标签
# class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']
# class_img = ['暗', '灰', '红']
class_img = ['暗紫泥二泥土', '暗紫泥大泥土', '暗紫泥夹砂土', '暗紫泥油石骨子土', '灰棕紫泥半砂半泥土', '灰棕紫泥大眼泥土', '灰棕紫泥石骨子土', '灰棕紫泥砂土', '红棕紫泥红棕紫泥土', '红棕紫泥红石骨子土', '红棕紫泥红紫砂泥土']

class CutSbuImage:
    def __init__(self, path, img_size, move_step):
        self.path = path
        self.img_size = img_size
        self.move_step = move_step
        self.total = 0  # 标记总土壤图像数目
        self.total_subimgs = 0  # 标记总子图数

    def master(self):
        """
        :return:总的子图数, 为了计算单张子图的时间
        """
        self.save_img(self.path, True)  # 对输入图像进行切子图操作, 为True时才进行操作
        return self.total, self.total_subimgs

    def save_img(self, img_path, mark):
        """
        输入:切子图,对输入的文件进行切子图操作
        然后保存在当前文件夹的subimg下
        mark:标记位,如果为True才开始
        :return:
        """
        if mark:
            print("切子图:[True]")
            img_save = os.path.join(img_path, subimg_name)  # 存放子图路径
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
                    flag = 1  # 2022年03月08日15:40:17 从我现在的角度来看 你这个flag不是很多于, 不用flag直接break不就好了?
                if flag == 0:  # 保存图像
                    region.save(path_now + '/' + str(num) + ".jpg")  # str(i)是裁剪后的编号，此处是0到99
                    num += 1
                    print("\r正在处理第{}张图".format(num), end='')
                    # 长移动i次 宽移动j次
        print("")
        print('切子图操作已完成...第{}类第{}张土壤图像共获得{}张子图'.format(class_i, img_item.split('.')[0], num))
        self.total += 1
        self.total_subimgs += num

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
    total, total_subimg = c.master()  # 启动类
    main1_end = time.time()  # 结束时间
    total_time = main1_end - main1_start
    # print("Total总土壤数->{} | Time总时间花费->{:.6f} | Single单张图像时间花费->{:.6f}".format(total, total_time, total_time / total))
    return total, total_subimg


def main2(path_root2):
    """
    汇总子图
    :param path_root2: 路径
    :return:
    """
    path = os.path.join(path_root2, subimg_name)
    if opt.img_size == opt.img_step:
        dataset_name = 'TestSet1_' + str(opt.img_size) + '_' + str(opt.img_step)
    else:
        dataset_name = 'TrainSet1_' + str(opt.img_size) + '_' + str(opt.img_step)
    target = os.path.abspath(os.path.join(path, '../' + dataset_name))  # subimg当前目录
    print("target -> ", target)
    # 有就删除, 没有就重新生成
    if os.path.exists(target):
        shutil.rmtree(target)
        os.makedirs(target)
    else:
        os.makedirs(target)
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
    return dataset_name


# =======================后处理算法=========================
def dealingAfter(path_sub):
    """
    删除中间文件夹 subimg_name
    :param path_sub:
    :return:
    """
    if os.path.exists(path_sub):
        shutil.rmtree(path_sub)
        # print("删除 {} 成功".format(path_sub))    # 这句话可以不要


def move_to_upper_strata(flag, root):
    """
    当使用最小外接矩形的时候, 生成的子图以及, subimg都在最小外接矩形中
    所以要把它们移动到上一级的目录
    :flog: 标记是否使用最小外接矩形
    :return:
    """
    if flag:
        # 先判断目录路径是否存在 存在的话先删除在移动
        target1 = os.path.join(root, datase_name)
        target2 = os.path.join(root, subimg_name)
        if os.path.exists(target1):
            shutil.rmtree(target1)
        if os.path.exists(target2):
            shutil.rmtree(target2)
        src1 = os.path.join(root, bbox_name, datase_name)
        src2 = os.path.join(root, bbox_name, subimg_name)
        shutil.move(src1, root)
        shutil.move(src2, root)
        print("后处理完成...")


if __name__ == '__main__':
    # 初始化命令行参数
    parser = argparse.ArgumentParser()
    # --------------初始化参数--------------
    parser.add_argument('--path_root', type=str, default='/Users/yida/Desktop/实验数据/实验一/3实验一35val/val', help='文件夹的路径')
    parser.add_argument('--img_size', type=int, default=600, help='子图大小')
    parser.add_argument('--img_step', type=int, default=600, help='滑动窗口的移动步长')
    parser.add_argument('--bbox', action='store_false', default=True, help='是否主动去获取外接矩形, 平日调试时默认关闭, 用命令行进行测试时记得开启')
    parser.add_argument('--valBySame', action='store_true', default=False, help='验证集由相同的图像中随机选择')
    parser.add_argument('--val_rate', type=float, default=0.2, help='每一张大图中随机选择子图的比例构成训练集和验证集, 当valBySame开启时才用得上')
    opt = parser.parse_args()  # 实例化参数列表
    # --------------初始化参数--------------
    # 数据集名称
    subimg_name = 'subimg1_' + str(opt.img_size) + '_' + str(opt.img_step)
    bbox_name = 'Rectangle'
    # -------------------------------------
    # 1.输入路径获取最小外接矩形
    path_root_ = opt.path_root
    if opt.bbox:  # 默认要去主动走
        BBmain(path_root_, class_img)  # 调用最小外接矩形的函数
        path_root_ = os.path.join(path_root_, bbox_name)  # 最小外接矩形就默认保存在Rectangle
        print("获取最小外接矩形结束, 正在进行子图切割算法1...")
    start = time.time()
    # 2.设置参数
    img_size = opt.img_size  # 子图大小
    move_step = opt.img_step  # 移动步长
    # 3.算法开始
    total_img, total_subimgs = main1(path_root_, img_size, move_step)  # 子图切割
    # 4.合并子图
    datase_name = main2(path_root_)  # 子图汇总
    end = time.time()
    total_time = end - start
    print("Total总土壤数->{} | Time总时间花费->{:.6f} | Single单张图像时间花费->{:.6f} | 总子图数为->{} | 扩充倍率为->{:.2f}".format(total_img,
                                                                                                          total_time,
                                                                                                          total_time / total_img,
                                                                                                          total_subimgs,
                                                                                                          total_subimgs / total_img))
    # 5.处理多于的子图集
    # dealingAfter(os.path.join(path_root_, subimg_name))
    # 由于使用 Rectangle的路径, 生成的子集会在其中, 所以把它往上层移动
    move_to_upper_strata(flag=opt.bbox, root=opt.path_root)
