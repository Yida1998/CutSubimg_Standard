"""
2021年07月09日19:19:32 步骤设计完整,每一步应该可以完整运行才行
输入:文件->n个类别->每个类别有m张图像
输出:子图文件夹./subimg, 每个土壤图像都一个子图文件夹

2021年11月24日15:48:00
对原代码进行修改, 将子图切割操作 换成自适应的WD距离矩阵的切割方法 -> 已完成
"""
import os
import shutil
import time

from FinalWandDOfSubImg import DoubleConvCutSubImg

# 土壤类别标签
class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']


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
            print("切子图:True")
            # img_save = os.path.join(img_path, 'subimg')  # 存放子图路径
            img_save = os.path.join(os.path.dirname(img_path), 'subimg')    # 存放子图路径
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
                        r = DoubleConvCutSubImg(img_sub, num_sum=self.num_sum, imgsave_path=img_sub_save,
                                                img_class=i,
                                                img_num=img_num, adaption=self.adaption, bins=self.bins)
                        r.master()
            print("************切子图已全部完成*************")
        else:
            print("切子图:False")


if __name__ == '__main__':
    start = time.time()
    #    path: 输入待处理的原图像路径
    path_root = '/Users/yida/Desktop/train'
    num_sum = 10  # 子图总数
    adaption = True  # 是否开启自适应子图切割
    bins = 8  # 是否进行区间划分

    c = CutSbuImage(path_root, num_sum, adaption, bins)
    c.master()  # 启动类
    end = time.time()
    print("共计时长: {}".format(end - start))
