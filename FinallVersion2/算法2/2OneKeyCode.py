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

2022年03月03日10:34:22
1.不要分区间bins, 把它修改成比例, 然后全部按照这个比例来进行将w矩阵置为0
2.bins -> rate

2022年03月03日21:21:09
因为你这个w矩阵最小值判断错误的问题, 导致比例报错? 你怎么能够这么蠢??????我的天
调试了一下午? 居然没有发现问题, 明天去解决下一个问题.

2022年03月05日11:21:06
新增 opt.broke, 默认为False, 当他为字典类型时开始替换 {l:0.2} 替换那一边的子图, 以及替换的比例, 替换时是用随机数!

2022年03月06日21:01:39
新增 opt.random, 默认为False, 当为true时随机在土壤图像上进行子图选择 生成数据集_random

2022年04月10日17:26:56
修改不使用外接矩形, 最后时间未进行统计的bug

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
# class_img = ['暗', '灰', '红']
class_img = ['暗紫泥二泥土', '暗紫泥大泥土', '暗紫泥夹砂土', '暗紫泥油石骨子土', '灰棕紫泥半砂半泥土', '灰棕紫泥大眼泥土', '灰棕紫泥石骨子土', '灰棕紫泥砂土', '红棕紫泥红棕紫泥土', '红棕紫泥红石骨子土', '红棕紫泥红紫砂泥土']


class BoundingBox:
    def __init__(self, img_path, img_size=225):
        """
        2022年03月02日11:23:43
        写代码一定要把参数 含义 取值描述清楚, 一周能看明白 一个月两个月还能明白嘛???
        :param img_path:
        :param img_size: 这玩意参数 你都没有用上啊? 写的什么玩意 描述也不带
        """
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

    def __init__(self, path, num_sum, adaption, rate, rateflip, broke, random, alpha):
        """
        :param path:数据集路径
        :param num_sum: 子图数量, 如需自定义子图数的话, 可以传一个列表进来 num_sum[类别][编号]进行访问
        :param adaption: 是否开启自适应子图切割
        :param rate: 舍弃w矩阵两端的比例
        :param rateflip: 是否在子图选择的时候将rate进行取反, 默认为False
        :param broke: 默认为False 当判定为字典类型时, 进行破坏性插入
        :param random: 默认为False 当为True时, 随机在w矩阵中选择 土壤子图
        :param alpha: 自适应调节土壤子图因子
        """
        self.path = path
        self.num_sum = num_sum  # 子图总数
        self.adaption = adaption  # 是否开启自适应子图切割
        self.rate = rate  # 是否进行区间划分
        self.total = 0  # 新增单张图像时间开销
        self.rateflip = rateflip  # 是否对rate进行取反
        self.broke = broke  # 当且仅当broke判定为字典类型时, 进行破坏性插入
        self.random = random
        self.alpha = alpha

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
            if opt.bbox:
                img_save = os.path.join(os.path.dirname(img_path), subimg_name)  # 存放子图路径
            else:
                img_save = os.path.join(img_path, subimg_name)
            #    新增功能 👆🏻
            if os.path.exists(img_save):
                shutil.rmtree(img_save)
                print("文件夹已存在, 正在删除...")
            # 开始遍历图像
            global subimg_num  # 声明下全局变量 保存子图总数
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
                                                    img_num=img_num, adaption=self.adaption, rate=self.rate,
                                                    img_size=opt.img_size,  # 子图数大小
                                                    rateflip=self.rateflip,  # 对rate选择时进行取反
                                                    broke=self.broke,  # 进行破坏性插入
                                                    random=self.random,  # 是否开启随机子图选择 默认为False
                                                    alpha=self.alpha)  # 自适应子图调节因子
                            subimg_num += r.master()  # 这儿会返回子图总数 加起来
                            self.total += 1
                        else:
                            r = DoubleConvCutSubImg(img_sub, num_sum=self.num_sum, imgsave_path=img_sub_save,
                                                    img_class=i,
                                                    img_num=img_num, adaption=self.adaption, rate=self.rate,
                                                    img_size=opt.img_size,  # 子图数大小
                                                    rateflip=self.rateflip,  # 对rate选择时进行取反
                                                    broke=self.broke,  # 进行破坏性插入
                                                    random=self.random,  # 是否开启随机子图选择 默认为False
                                                    alpha=self.alpha)  # 自适应子图调节因子
                            subimg_num += r.master()  # 这儿会返回子图总数 加起来
                            self.total += 1
            print("************切子图已全部完成*************")
        else:
            print("切子图:False")


def main1(img_path):
    print("最小外接矩形: Circumscribed rectangle...")
    img_save = os.path.join(img_path, bbox_name)  # 存放子图路径
    if os.path.exists(img_save):
        print("最小外接矩形文件已存在, 不重复获取, 直接使用...")
        return 0  # 当外接矩形文件夹存在的时候就直接跳出, 直接使用这个文件夹
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
            if item.endswith('.jpg'):  # 新增JPG判断
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


def main2(path_root, num_sum, adaption, rate, rateflip, broke, random, alpha):
    #    path: 输入待处理的原图像路径
    # 2021年12月17日11:32:17:这儿需要修改一下, 新增外接矩形加速可选
    if opt.bbox:
        print("使用外接矩形加速...")
        path_root = os.path.join(path_root, bbox_name)
    else:
        print("不使用外接矩形加速...")
        path_root = os.path.join(path_root)
    main2_start = time.time()
    c = CutSbuImage(path_root, num_sum, adaption, rate, rateflip, broke, random, alpha)
    total = c.master()  # 启动类
    main2_end = time.time()  # 结束时间
    total_time = main2_end - main2_start
    print("Total总土壤数->{}张 | Time总时间花费->{:.6f}s | Single单张图像时间花费->{:.6f}s | 子图总数:->{}".format(total, total_time,
                                                                                             total_time / total,
                                                                                             subimg_num))


def main3(path):
    if not opt.valBySame:  # 不开启参数时, 正常合并子图
        path = os.path.join(path, subimg_name)
        target = os.path.abspath(os.path.join(path, '../' + dataset_name))  # subimg当前目录
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

    else:  # 开启参数后, 同一张大图里面拿部分子图出来构成训练集和验证集
        # 设置路径
        print("验证集由相同土壤图像的子图构成比例为:{}".format(opt.val_rate))
        path = os.path.join(path, subimg_name)
        test_rate = opt.val_rate  # 测试集比例
        target = os.path.abspath(os.path.join(path, '../' + dataset_name))  # subimg当前目录
        target_train = os.path.abspath(os.path.join(path, '../' + dataset_name + '/train'))
        target_val = os.path.abspath(os.path.join(path, '../' + dataset_name + '/val'))
        # 有就删除, 没有就重新生成
        if os.path.exists(target):
            shutil.rmtree(target)
            os.makedirs(target_train)
            os.makedirs(target_val)
        else:
            os.makedirs(target_train)
            os.makedirs(target_val)
        print(target)
        for i in class_img:
            sub_path = os.path.join(path, i)  # 子文件路径
            sub_file = os.listdir(sub_path)  # 子文件下的图像
            if '.DS_Store' in sub_file:
                sub_file.remove('.DS_Store')
            for j in sub_file:  # 第j张图
                sub_file_img = os.path.join(sub_path, j)  # 下一步获取子图
                file_subimg = os.listdir(sub_file_img)  # 获取子图
                img_num = 0  # 子图总数
                for item in file_subimg:
                    if item.endswith('.jpg'):
                        # 按照比例进行分配
                        if img_num <= test_rate * len(file_subimg):
                            # 子图路径 val
                            target_name = '_' + str(i) + '_' + str(j) + '.jpg'
                            sub_img_path = os.path.join(sub_file_img, item)  # 原路径
                            sub_class = os.path.join(target_val, i)  # 生成路径
                            if not os.path.exists(sub_class):
                                os.makedirs(sub_class)  # 不存在就生成文件夹
                            target_path = os.path.join(sub_class, item.split('.')[0] + target_name)  # 移动路径
                            shutil.copy(sub_img_path, target_path)  # 开始移动
                            img_num += 1
                        else:
                            # 子图路径 train
                            target_name = '_' + str(i) + '_' + str(j) + '.jpg'
                            sub_img_path = os.path.join(sub_file_img, item)  # 原路径
                            sub_class = os.path.join(target_train, i)  # 生成路径
                            if not os.path.exists(sub_class):
                                os.makedirs(sub_class)  # 不存在就生成文件夹
                            target_path = os.path.join(sub_class, item.split('.')[0] + target_name)  # 移动路径
                            shutil.copy(sub_img_path, target_path)  # 开始移动
            print("正在移动第{}个类".format(i))
        print("任务完成...")


def main(root_path, num_sum, adaption, rate, rateflip, broke, random, alpha):
    if opt.bbox:
        start1 = time.time()
        main1(root_path)  # 获取外接矩形
        start2 = time.time()
        print("获取最小外接矩形花费时间为:{}".format(start2 - start1))
    start2 = time.time()
    main2(root_path, num_sum, adaption, rate, rateflip, broke, random, alpha)  # 子图切割
    main3(root_path)  # 合并
    end = time.time()
    print("消耗时间:", end - start2)


def get_same_num_dict(subimg_path):
    """
    获取相同子图数的字典
    :param subimg_path: 待获取子图数的路径 , 这个玩意的路径我都删了啊 我还得再重新弄一下
    :return:
    """
    # 训练集
    print("正在获取目标子图集-> {} 的子图标记字典".format(subimg_path))
    dict_train = {}
    root_path_train = subimg_path
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
            # print("第{}的{}张土壤图像共有{}张土壤子图".format(i, j, m))
            dict_train['{}_{}'.format(i, j)] = m
    print("获取训练集标记成功:", dict_train)
    return dict_train


def init_datasetname(start: str = '', end: str = ''):
    """
    初始化文件夹名称, 可传入结尾符号
    :param start:
    :param end:
    :return:
    """
    # subimg_name = 'subimg2_' + str(opt.img_size)+'_rate'+str(opt.rate)+'_num'+str(opt.num_sum)
    # dataset_name = 'TrainSet2_' + str(opt.img_size)+'_rate'+str(opt.rate)+'_num'+str(opt.num_sum)
    file_name = start + str(opt.img_size) + '_drop' + [k for k in opt.rate.keys()][0] + str(
        [k for k in opt.rate.values()][0]) + ('_adaption' if opt.adaption else '_num' + str(opt.num_sum)) + str([
                                                                                                                    '_rateflip' if opt.rateflip else ''][
                                                                                                                    0]) + \
                str('_broke' + str([k for k in opt.broke.keys()][0]) + str(
                    [v for v in opt.broke.values()][0]) if isinstance(opt.broke, dict) else str('')) + str(
        ['_ramdom' if opt.random else ''][0]) + '_' + str(opt.alpha) + end
    return file_name


if __name__ == '__main__':
    # 设置系统参数
    parser = argparse.ArgumentParser()
    # 设置路径
    parser.add_argument('--path', type=str, default="/Users/yida/Desktop/纠错/high_4KFold/0/train", help='加载数据集路径')
    parser.add_argument('--bbox', action='store_false', default=True, help='是否使用外接矩形加速运算, 默认为True')
    parser.add_argument('--img_size', type=int, default=224, help='默认子图大小为224')
    parser.add_argument('--same_path', type=str, default='/Users/yida/Desktop/final_dataset/0/train/subimg1_224_224',
                        help='待获取相同子图数量的数据集路径')
    parser.add_argument('--adaption', action='store_false', default=True, help='是否开启自适应子图切割')
    # 由于命令行无法传入字典, 所以传入字符串, 把它改成字典以_为分隔符
    parser.add_argument('--rate', type=str, default='m_0',
                        help='m=0 时不进行区间限制 l左边 m两端 r右边 数字是比例; 去除w矩阵两端的比例 比例改为两端总比例, 改成字典')
    parser.add_argument('--rateflip', action='store_true', default=False,
                        help='将比例rate进行取反, 默认关闭, 开启的时候仅在舍弃的rate区域进行子图选择, 将其它区域w置为0')
    parser.add_argument('--num_sum', type=int, default=0, help='默认获取子图数为0时与算法1获取相同子图数, 但需要关闭自适应 当不为0的时候就指定该数目为获取的子图数')
    parser.add_argument('--broke', type=None, default=False,
                        help='留个破坏性的接口, 默认为False, 否则为字典类型{范围:m, 替换比例:0.01}, 当类型为字典时 进行破坏性子图插入 输入:{m:0.5}')
    parser.add_argument('--datasetname_end', type=str, default='', help='默认为空字符, 为了防止随机选择时出现数据集重叠, 因此留一个接口, 重命名数据集')
    parser.add_argument('--random', action='store_true', default=False, help='默认为False, 开启时在w矩阵进行不重复的随机土壤子图选择')
    parser.add_argument('--alpha', type=float, default=1.0, help='自适应获取土壤子图的alpha因子, 默认为1, 可以自适应调节')
    parser.add_argument('--valBySame', action='store_true', default=False, help='验证集由相同的图像中随机选择')
    parser.add_argument('--val_rate', type=float, default=0.2, help='每一张大图中随机选择子图的比例构成训练集和验证集, 当valBySame开启时才用得上')
    opt = parser.parse_args()  # 实例化
    #  2022年04月12日17:51:54 由于命令行无法传入字典, 我在这儿临时传入字符串作一个修改, 以'_'为分隔符转换成字典
    opt.rate = {opt.rate.split('_')[0]: float(opt.rate.split('_')[1])}
    # ==========数据集名称==========
    bbox_name = 'Rectangle'
    datasetname_end = opt.datasetname_end  # 自定义尾巴避免 随机选择重复
    # 定义数据集名称
    subimg_name = init_datasetname(start='subimg2_', end=datasetname_end)
    dataset_name = init_datasetname(start='TrainSet2_', end=datasetname_end)
    print("数据集信息: 最小外接矩形: -> {} 子集: -> {} 训练集: -> {}".format(bbox_name, subimg_name, dataset_name))
    # ==============子图选择算法==============
    subimg_num = 0  # 子图总数, 设为全局变量
    # --------------设置命令行参数---------------
    root_paths = opt.path  # 设置路径
    adaption_ = opt.adaption  # 是否开启自适应子图切割, 默认为False;  如果开启自适应子图的话, 初始化num_sum_; 否则加载字典
    rate_ = opt.rate  # 取消区间划分 将区间变成rate
    rateflip_ = opt.rateflip  # 将rate进行取反, 默认为False, 如果开启的话, 就仅在rate限制的区间进行选择
    broke_ = opt.broke
    if not isinstance(opt.broke, bool):     # 判断不是bool类型时 生效
        broke_ = {broke_.split('_')[0]: float(broke_.split('_')[1])}    # 是否进行破坏性替换, 默认为False, 当为字典类型时启动破坏性插入, 明暗程度定义为默认 转换成字典类型
    random_ = opt.random
    alpha_ = opt.alpha  # 自适应子图数调节因子
    # ------------------------------------------
    if opt.adaption:
        num_sum_ = 0
    else:
        if opt.num_sum != 0:  # 当这个参数不为0的时候, 子图数就使用指定的num_sum
            num_sum_ = opt.num_sum
        else:
            num_sum_ = get_same_num_dict(opt.same_path)  # 子图总数字典

    main(root_paths, num_sum_, adaption_, rate_, rateflip_, broke_, random_, alpha_)
