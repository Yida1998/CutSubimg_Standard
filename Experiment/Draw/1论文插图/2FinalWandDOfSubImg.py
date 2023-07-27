"""
2021年09月07日10:05:35
基于2次均值滤波的子图切割算法
1.把获取标记矩阵和权重矩阵合在一起了 : 权重矩阵目前是用的均值滤波来获取的
2.w * d 求最大的点 放入备选点集合
3.切子图操作
现在的代码完全是可以运行的!!!

2021年10月09日11:01:25->进行代码升级
目标:
1.实现自适应子图切割
2.分区间获取土壤子图

2021年10月10日10:06:30
总结:
1.自适应分割已经实现
2.分区间获取子图已经完成
->
尚需:
3.完成注解, 审计代码, 去掉冗余及重复操作, 加速!!!!

2021年11月23日10:04:52:
1.更新区间限制的方式, 分成8个区间, 且将后两个区间对半分舍去第一个区间和最后一个区间
2.重新设定自适应数

2021年11月24日15:38:19:
检查一下代码,然后把它规范成类的形式,输入整个文件夹进行操作
目前这个是最终版本!!!!
修改: 删除存储图像信息  以及打印输出 仅输出基本信息, 中间过程需存储, 后期看 哪些图像不得行 -> 已完成

2022年01月06日10:44:30
用于论文中的插图显示, 去掉腐蚀和膨胀
"""
import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


class DoubleConvCutSubImg:
    def __init__(self, path, num_sum, imgsave_path, img_class, img_num, img_size=224, adaption=False, bins=None):
        """
        :param path: 图像路径
        :param num_sum: 切子图数
        :param imgsave_path: 子图存储路径
        :param img_class: 输入图像类别
        :param img_num: 图像大图编号
        :param img_size: 图像大小, 默认为:224
        :param adaption: 是否开启自适应获取土壤子图, 默认:False
        :param bins: 进行区间划分并依次取值, 默认为None:max(w*d)
        """
        # 第一部分:获取标记矩阵和权重矩阵
        self.path = path
        self.img = cv2.imread(path)
        self.img_gray = cv2.imread(path, 0)
        # 保证img_size为奇数,不然做卷积操作有问题
        if img_size % 2 == 0:
            self.img_size = img_size + 1
        else:
            self.img_size = img_size

        # 第二部分:构建w和d矩阵,迭代求解获得子图
        self.n = num_sum  # 子图数
        self.imgsave_path = imgsave_path  # 存储子图的路径
        self.img_class = img_class  # 类别
        self.img_num = img_num  # 图像编号

        # 标记矩阵和权重矩阵
        self.lable = None
        self.w = None
        # 2022年03月04日09:51:00权重矩阵 要用v通道
        self.v = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)[:, :, 2]

        # 初始化区间子图数
        self.region_subimg = None

        # 自适应获取数目比例
        self.adaption = adaption
        self.rate = 1e-4

        # 是否划分区间
        self.bins = bins

        # 初始化文件夹 -> 这都可以掉 怎么做到了
        if os.path.exists(self.imgsave_path):
            shutil.rmtree(self.imgsave_path)
            os.makedirs(self.imgsave_path)
        else:
            os.makedirs(self.imgsave_path)
        print("正在处理第{}类第{}张紫色土图像...子图大小为:{}...Adaption自适应:{}...Bins区间:{}".format(img_class, img_num, img_size, adaption,
                                                                                 bins))

    def master(self):
        """
        主函数:用于调用其他类函数
        :return:
        """
        # ============================= #
        #       一.初始化标记矩阵
        # ============================= #
        lable = self.lable_MAtrix1()
        self.lable = lable
        # ============================= #
        #       二.初始化权重矩阵
        # ============================= #
        w = self.weight_Matrix2(lable)
        self.w = w

    def lable_MAtrix1(self):
        """
       初始化标记矩阵lable, 土壤区域标记为255, 非土壤区域标记为0
       使用了保护边境措施,进行了腐蚀
       :return:可用土壤标记矩阵: [255, 0]
        """
        lable = np.copy(self.img_gray)
        # 原图背景 255, 土壤0-255->背景为0, 土壤区域为255
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        lable[lable == 255] = 0
        lable[lable != 0] = 255
        plt.imshow(lable)
        # plt.savefig(save_path + "/3img_label.svg", dpi=600)
        plt.savefig(save_path + "/3img_label.jpg", dpi=300)
        plt.show()

        # 2021年11月24日20:33:07通过膨胀再腐蚀的方法: 修正P图错误 , 2021年11月25日21:53:21:bug缩小腐蚀和膨胀的次数20->5, 避免与边缘连接
        # lable = cv2.dilate(lable, None, iterations=5)  # 膨胀, 去掉P图错误标记
        # lable = cv2.erode(lable, None, iterations=5)

        # 对标记矩阵进均均值滤波 为了解决子图切割包含背景的子图
        new_lable = cv2.blur(lable, (self.img_size, self.img_size))

        new_lable[new_lable != 255] = 0
        # 进行边界保护措施 - > 整体缩小20个单位
        # new_lable = cv2.erode(new_lable, None, iterations=10)  # 腐蚀 -> 缩小   原来设置20 -> 修正为10 看看有没有bug
        # 将图像转换到float32
        new_lable = new_lable.astype(np.float32)

        lable[lable != 0] = 1  # 这句号应该没有用的 可以删除
        # 设置自适应获取的子图总数
        if self.adaption:
            # self.n = 2 * self.img.shape[0] * self.img.shape[1] // (self.img_size ** 2)    # step=0.5
            self.n = self.img.shape[0] * self.img.shape[1] // (self.img_size ** 2)  # step=1
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(new_lable)
        # plt.savefig(save_path + "/4new_label.svg", dpi=600)
        plt.savefig(save_path + "/4new_label.jpg", dpi=300)
        plt.show()
        plt.close()
        return new_lable

    def weight_Matrix2(self, lable):
        """
        初始化权重矩阵:这个相当于领域信息,是子图选择的参考
        :return:权重矩阵
        """
        # 初始化图像,权重矩阵求的是灰度图的均值
        # img = np.copy(self.img_gray)
        img = np.copy(self.v)
        # 背景置0
        img[img == 255] = 0
        # 进均均值滤波
        img = img.astype(np.float32)
        img_blur = cv2.blur(img, (self.img_size, self.img_size))
        lable = np.where(lable == 255, 1, 0)
        weight = img_blur * lable  # 进行哈达玛积,排除包含背景的区域
        # 将图像转换到float32 方便计算
        weight = weight.astype(np.float32)

        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(weight)
        # plt.savefig(save_path + "/5img_weight.svg", dpi=600)
        plt.savefig(save_path + "/5img_weight.jpg", dpi=300)
        np.save(save_path + "/w.npy", weight)    # 保存权重矩阵
        plt.show()
        return weight


if __name__ == '__main__':
    start = time.time()
    # 输入图像路径
    img_path = "./plt_img/new_img.jpg"  # 51.06s
    save_path = './plt_img'  # 存储路径

    # 存储图像路径
    imgsave_path_ = "/Users/yida/Desktop/subimg"
    # 子图数
    num_sum_ = 10
    # 图像类别编号
    img_class_ = 2
    # 图像顺序标号
    img_num_ = 5
    # 初始化类
    r = DoubleConvCutSubImg(img_path, num_sum=num_sum_, imgsave_path=imgsave_path_, img_class=img_class_,
                            img_num=img_num_, adaption=False, bins=8)
    # 调用主类方法
    r.master()
    end = time.time()
    print("time:", end - start)
