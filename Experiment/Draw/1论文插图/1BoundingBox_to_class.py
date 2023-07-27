"""
重写成类的形式
2021年09月22日19:03:23
输入一个路径(仅包含土壤图像), 返回一张包含土壤外接矩阵的图像
"""
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

import numpy as np


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
        print("lable矩阵计算完成...")
        return lable

    def find_boundary_point2(self, lable):
        """
        输入lable矩阵,返回4个边界点
        :param lable:
        :return:
        """
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(lable)  # cmap修改颜色
        # plt.savefig(save_path + '/1NoPoints_label.svg', dpi=600)
        plt.savefig(save_path + '/1NoPoints_label.jpg', dpi=300)

        # plt.show()
        # plt.close()
        # 第一行, 最后一行
        point = np.where(lable != 0)
        top = (point[1][0], point[0][0])
        bottom = (point[1][-1], point[0][-1])
        # 翻转
        lable_t = lable.transpose()
        point = np.where(lable_t != 0)
        left = (point[0][0], point[1][0])
        right = (point[0][-1], point[1][-1])

        # 画点
        # plt画图的行和列的起点不同, 注意
        # plt.scatter(top[0], top[1], s=200)
        # plt.scatter(bottom[0], bottom[1], s=200)
        # plt.scatter(left[0], left[1], s=200)
        # plt.scatter(right[0], right[1], s=200)

        # 描述边界
        plt.text(top[0], top[1] - 30, 'a', fontdict={'size': 20, 'color': 'r'})
        plt.text(bottom[0], bottom[1] + 130, 'b', fontdict={'size': 20, 'color': 'r'})
        plt.text(left[0] - 100, left[1], 'c', fontdict={'size': 20, 'color': 'r'})
        plt.text(right[0] + 30, right[1], 'd', fontdict={'size': 20, 'color': 'r'})
        print(top, bottom, left, right)

        # 不设置刻度

        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.title("Points_label")
        # 画矩形
        currentAxis = plt.gca()
        rect = patches.Rectangle((left[0], top[1]), right[0] - left[0], bottom[1] - top[1], linewidth=2, edgecolor='r',
                                 facecolor='none')
        currentAxis.add_patch(rect)
        # 坐标太傻比了  x轴从左往右  y轴 从上到下
        # plt.savefig(save_path + "/2Points_label.svg", dpi=600)
        plt.savefig(save_path + "/2Points_label.jpg", dpi=300)
        plt.show()

        lable = lable[top[1]:bottom[1], left[0]:right[0]]
        plt.imshow(lable)
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.title("Label")
        # plt.savefig(save_path+"/label.svg", dpi=600)
        plt.show()
        print("2寻找边界成功...")
        p = [top[1], bottom[1], left[0], right[0]]
        print(p)
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
        # plt.savefig(save_path+ "/img_rgb.svg", dpi=600)
        plt.show()
        return img_


if __name__ == '__main__':
    # path = "/Users/yida/Desktop/土壤分割/result/87.jpg"
    path = "./plt_img/87.jpg"
    save_path = 'plt_img'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    b = BoundingBox(img_path=path)
    img = b.master()
    # 4.保存图像
    cv2.imwrite(save_path + "/new_img.jpg", img)
    print("保存成功...")
