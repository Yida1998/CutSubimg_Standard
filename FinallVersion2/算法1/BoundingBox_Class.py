"""
重写成类的形式
2021年09月22日19:03:23
输入一个路径(仅包含土壤图像), 返回一张包含土壤外接矩阵的图像

2021年11月26日11:35:12
已完成类的形式, 只是中间可视化的结果还没有去掉, 这个可以先不急
"""
import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']
class_img = []  # 全局变量


class BoundingBox:
    def __init__(self, img_path, img_size=225):
        self.img = cv2.imread(img_path)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_size = img_size

    def master(self):
        """
        主函数,调用
        :return:
        """
        lable = self.lable_MAtrix1()
        point = self.find_boundary_point2(lable)
        new_img = self.get_rectangl3(lable, point)
        return new_img

    def lable_MAtrix1(self):
        """
       初始化标记矩阵lable, 土壤区域标记为255, 非土壤区域标记为0
       使用了保护边境措施,进行了腐蚀
       :return:可用土壤标记矩阵: [255, 0]
        """
        lable = np.copy(self.img_gray)
        img_size = self.img_size
        # 原图背景 255, 土壤0-255->背景为0, 土壤区域为255
        lable[lable == 255] = 0
        lable[lable != 0] = 255
        # 土壤区域标记为1
        lable[lable != 0] = 1

        # 膨胀, 去掉P图错误标记 2021年11月25日22:13:38: 修正, 修改 从2倍到3倍 差距变大防止填充不满土壤
        lable = cv2.erode(lable, None, iterations=3)
        lable = cv2.dilate(lable, None, iterations=9)

        return lable

    def find_boundary_point2(self, lable):
        """
        输入lable矩阵,返回4个边界点
        :param lable:
        :return:
        """
        plt.imshow(lable)  # cmap修改颜色
        # 第一行, 最后一行
        point = np.where(lable != 0)
        top = (point[1][0], point[0][0])
        bottom = (point[1][-1], point[0][-1])
        # 翻转
        lable_t = lable.transpose()
        point = np.where(lable_t != 0)
        left = (point[0][0], point[1][0])
        right = (point[0][-1], point[1][-1])
        # plt画图的行和列的起点不同, 注意
        plt.scatter(top[0], top[1], s=200)
        plt.scatter(bottom[0], bottom[1], s=200)
        plt.scatter(left[0], left[1], s=200)
        plt.scatter(right[0], right[1], s=200)
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.title("Points_label")
        # plt.savefig("Points_label.svg", dpi=600)
        # plt.show()

        lable = lable[top[1]:bottom[1], left[0]:right[0]]
        plt.imshow(lable)
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.title("Label")
        # plt.savefig("label.svg", dpi=600)
        # plt.show()
        # print("2寻找边界成功...")
        p = [top[1], bottom[1], left[0], right[0]]
        # print(p)
        return p

    def get_rectangl3(self, lable_, p):
        """
        输入原图 标记矩阵以及点
        :param lable_:
        :param p:
        :return:
        """
        img_ = self.img
        img_[:, :, 0] = img_[:, :, 0] * lable_
        img_[:, :, 1] = img_[:, :, 1] * lable_
        img_[:, :, 2] = img_[:, :, 2] * lable_
        # 注意这个标记矩阵, 它是有0就变成255了!
        img_[lable_ == 0] = 255
        img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        # 截取目标区域
        img_rgb = img_rgb[p[0]:p[1], p[2]:p[3]]
        img_ = img_[p[0]:p[1], p[2]:p[3]]
        plt.imshow(img_rgb)
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.savefig("img_rgb.svg", dpi=600)
        # plt.show()
        return img_


def main(img_path, class_img_):
    """
    当Rectangle存在的时候就不对它进行删除, 而是直接跳出
    :param img_path: 图像路径
    :param class_img_:图像类标签
    :return:
    """
    class_img = class_img_
    print("最小外接矩形: Circumscribed rectangle...")
    img_save = os.path.join(img_path, 'Rectangle')  # 存放子图路径
    if os.path.exists(img_save):
        return 0        # 当目录存在的时候, 直接跳出目录
        shutil.rmtree(img_save)
        print("文件夹已存在, 正在删除...")
    # 开始遍历图像
    print(class_img)
    for i in class_img:
        sub_path = os.path.join(img_path, i)  # 图像原图路径
        img_save_path = os.path.join(img_save, i)  # 图像存储路径
        # 创建路径
        os.makedirs(img_save_path)
        # 子文件
        file_sub = os.listdir(sub_path)

        for item in file_sub:
            if item.endswith('.jpg'):   # 我日, 不要写一些bug 注意细节
                # i : 类别   item: 图像名称.jpg
                img_num = item.split('.')[0]

                img_sub = os.path.join(sub_path, item)  # 待处理图像路径

                img_save_subpath = os.path.join(img_save_path, item)  # 保存图像的路径
                # 初始化类并调用方法
                b = BoundingBox(img_path=img_sub)
                img = b.master()
                # 保存图像
                cv2.imwrite(img_save_subpath, img)
                print("第{}类第{}张图像成功获取外接矩形...".format(i, img_num))

    print("************外接矩形获取已全部完成*************")


if __name__ == '__main__':
    start = time.time()
    path_root = "/Users/yida/Desktop/train"
    main(path_root, class_img)
    end = time.time()
    print("消耗时间:{}".format(end - start))
