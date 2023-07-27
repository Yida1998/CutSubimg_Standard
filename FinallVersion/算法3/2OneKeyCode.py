"""
[基于权重与距离矩阵的土壤子图选择算法-加载字典, 指定子图数目]


2021年11月26日11:59:40: 对三个类进行封装, 全部汇总到一个函数, 高内聚低耦合
还得优化下代码, 把所有参数都外置, 拿出来, 完善封装形式!!!

2021年11月26日21:28:36: 优雅, 永不过时!

一键启动

2021年12月17日16:58:55:
仅在2个一键运行代码中进行了修改, 新增是否获取外接矩形加速, 增加了超参数以及判断修改了路径, 这个应该不会造成bug
写代码的时候一定要考虑可拓展性和鲁棒性 因为后期不太好大规模的删减

2021年12月18日09:38:30: 新增 是否使用外接矩形加速, 设置超参数, 多设置了个判断
2022年01月25日11:01:35 新增计算单幅图像时间开销


"""
import argparse
import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from FinalWandDOfSubImg import DoubleConvCutSubImg

# class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']  # 类标签
class_img = ['暗', '灰', '红']


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
        # plt.imshow(lable)  # cmap修改颜色
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
        plt.close()

        lable = lable[top[1]:bottom[1], left[0]:right[0]]
        # plt.imshow(lable)
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.title("Label")
        # plt.savefig("label.svg", dpi=600)
        # plt.show()
        plt.close()
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
        # plt.imshow(img_rgb)
        # 不设置刻度
        plt.axis('off')
        # 取消背景白色框框
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.savefig("img_rgb.svg", dpi=600)
        # plt.show()
        plt.close()
        return img_


class CutSbuImage:

    def __init__(self, path, num_sum, adaption, bins):
        """
        :param path:数据集路径
        :param num_sum: 子图数量, 如需自定义子图数的话, 可以传一个列表进来 num_sum[类别][编号]进行访问
        :param adaption: 是否开启自适应子图切割
        :param bins: 划分区间
        """
        self.path = path
        self.num_sum = num_sum  # 子图总数
        self.adaption = adaption  # 是否开启自适应子图切割
        self.bins = bins  # 是否进行区间划分
        self.total = 0  # 新增单张图像时间开销

    def master(self):
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
            print("切子图:True")
            # img_save = os.path.join(img_path, 'subimg')  # 存放子图路径
            # 2021年12月17日16:49:21:新增 可选获取外接矩形加速
            if opt.acceleration:
                img_save = os.path.join(os.path.dirname(img_path), 'subimg_3')  # 存放子图路径
            else:
                img_save = os.path.join(img_path, 'subimg_3')
            #    新增功能 👆🏻
            if os.path.exists(img_save):
                shutil.rmtree(img_save)
                print("文件夹已存在, 正在删除...")
            # 开始遍历图像
            for i in class_img:
                sub_path = os.path.join(img_path, i)  # 子路径
                file_sub = os.listdir(sub_path)  # 子文件
                for item in file_sub:
                    if item.endswith('.jpg'):
                        # i : 类别   item: 图像名称.jpg
                        img_num = item.split('.')[0]
                        img_sub = os.path.join(sub_path, item)  # 带切割图像路径
                        img_sub_save = os.path.join(img_save, i, item.split('.')[0])  # 保存图像的路径
                        # 生成图像路径
                        os.makedirs(img_sub_save)
                        # 调用方法
                        # 如果输入是一个字典的话 就按照指定数量进行生成
                        if isinstance(self.num_sum, dict):
                            num_sum_i_num = self.num_sum['{}_{}'.format(i, img_num)]
                            r = DoubleConvCutSubImg(img_sub, num_sum=num_sum_i_num, imgsave_path=img_sub_save,
                                                    img_class=i,
                                                    img_num=img_num, adaption=self.adaption, bins=self.bins)
                            r.master()
                            self.total += 1
                        else:
                            r = DoubleConvCutSubImg(img_sub, num_sum=self.num_sum, imgsave_path=img_sub_save,
                                                    img_class=i,
                                                    img_num=img_num, adaption=self.adaption, bins=self.bins)
                            r.master()
                            self.total += 1
            print("************切子图已全部完成*************")
        else:
            print("切子图:False")


def main1(img_path):
    print("最小外接矩形: Circumscribed rectangle...")
    img_save = os.path.join(img_path, 'Rectangle')  # 存放子图路径
    if os.path.exists(img_save):
        shutil.rmtree(img_save)
        print("文件夹已存在, 正在删除...")
    # 开始遍历图像
    for i in class_img:
        sub_path = os.path.join(img_path, i)  # 图像原图路径
        img_save_path = os.path.join(img_save, i)  # 图像存储路径
        # 创建路径
        os.makedirs(img_save_path)
        # 子文件
        file_sub = os.listdir(sub_path)

        for item in file_sub:
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


def main2(path_root, num_sum, adaption, bins):
    #    path: 输入待处理的原图像路径
    # 2021年12月17日11:32:17:这儿需要修改一下, 新增外接矩形加速可选
    if opt.acceleration:
        print("使用外接矩形加速...")
        path_root = os.path.join(path_root, 'Rectangle')
    else:
        print("不使用外接矩形加速...")
        path_root = os.path.join(path_root)
    main2_start = time.time()
    c = CutSbuImage(path_root, num_sum, adaption, bins)
    total = c.master()  # 启动类
    main2_end = time.time()  # 结束时间
    total_time = main2_end - main2_start
    print("Total总土壤数->{}张 | Time总时间花费->{:.6f}s | Single单张图像时间花费->{:.6f}s".format(total, total_time, total_time / total))


def main3(path):
    path = os.path.join(path, 'subimg_3')
    target = os.path.abspath(os.path.join(path, '../dataset_3'))  # subimg当前目录
    # 有就删除, 没有就重新生成
    if os.path.exists(target):
        shutil.rmtree(target)
        os.makedirs(target)
    else:
        os.makedirs(target)
    print(target)
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
                    sub_img_path = os.path.join(sub_file_img, item)  # 原路径
                    sub_class = os.path.join(target, i)  # 生成路径
                    if not os.path.exists(sub_class):
                        os.makedirs(sub_class)  # 不存在就生成文件夹
                    # target_path = os.path.join(sub_class, item.split('.')[0] + target_name)     # 移动路径
                    target_path = os.path.join(sub_class, item)  # 移动路径不修改名称
                    shutil.copy(sub_img_path, target_path)  # 开始移动
        print("正在移动第{}个类...".format(i))
    print("任务完成...")


def main(root_path, num_sum, adaption, bins):
    start1 = time.time()
    main1(root_path)  # 获取外接矩形
    start2 = time.time()
    print("获取最小外接矩形花费时间为:{}".format(start2 - start1))
    main2(root_path, num_sum, adaption, bins)  # 子图切割
    main3(root_path)  # 合并
    end = time.time()
    print("消耗时间:", end - start2)


if __name__ == '__main__':
    # 设置系统参数
    parser = argparse.ArgumentParser()
    # 设置路径
    parser.add_argument('--path', type=str, default="/Users/yida/Desktop/final_dataset/0/train", help='加载数据集路径')
    parser.add_argument('--dict', type=str, default='dict_train.npy', help='加载对应子图数')
    parser.add_argument('--acceleration', action='store_false', default=True, help='是否使用外接矩形加速运算, 默认为True')
    opt = parser.parse_args()  # 实例化
    root_paths = opt.path  # 设置路径
    # ======子图切割======
    num_sum_ = np.load(opt.dict, allow_pickle=True).item()  # 子图总数字典
    adaption_ = False  # 是否开启自适应子图切割
    bins_ = 16  # 是否进行区间划分
    main(root_paths, num_sum_, adaption_, bins_)
