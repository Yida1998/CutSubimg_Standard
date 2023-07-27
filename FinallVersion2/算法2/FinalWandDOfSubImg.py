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

2022年01月25日11:22:41: 修改区间限制方法 放在interval_limit_add中:利用中心极限法则

2022年03月04日10:35:36:检查代码, 修正代码中错误使用权重矩阵的错误-> v通道
"""
import os
import shutil
import time

import cv2
import numpy as np


class DoubleConvCutSubImg:
    def __init__(self, path, num_sum, imgsave_path, img_class, img_num, img_size=224, adaption=False,
                 rate: dict = {'m': 0}, rateflip=False, broke: None = False, random: bool = False, alpha: float = 1.0):
        """
        :param path: 图像路径
        :param num_sum: 切子图数
        :param imgsave_path: 子图存储路径
        :param img_class: 输入图像类别
        :param img_num: 图像大图编号
        :param img_size: 图像大小, 默认为:224
        :param adaption: 是否开启自适应获取土壤子图, 默认:False
        :param rate: 舍弃区间限制, 切换成舍弃w矩阵两端比例, 把它改为字典方便灵活使用 {m:rate}两端 {l:}左边 {r:}右边
        :param rateflip: 默认为False, 对rate进行取反, 当取反时仅在rate的区间中进行选择
        :param broke: 默认为False, 当判定为字典类型时对图像进行破坏性插入
        :param random: 默认为False, 当为true是在w矩阵进行随机土壤子图选择
        :param alpha: 自适应子图调节因子
        """
        # 第一部分:获取标记矩阵和权重矩阵
        self.path = path
        self.img = cv2.imread(path)
        self.img_gray = cv2.imread(path, 0)  # 读取两次速度应该会变慢, 应该直接把图像转换成灰度图, 但是这样更快一点

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
        self.v = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)[:, :, 2]  # 权重矩阵用v通道

        # 初始化区间子图数
        self.region_subimg = None

        # 自适应获取数目比例
        self.adaption = adaption

        # 舍弃权重矩阵的比例
        self.rate = rate
        self.rateflip = rateflip  # rate取反之后进行选择

        # 破坏性插入
        self.broke = broke

        # 随机子图选择
        self.random = random

        # 初始化文件夹 -> 这都可以掉 怎么做到了
        if os.path.exists(self.imgsave_path):
            shutil.rmtree(self.imgsave_path)
            os.makedirs(self.imgsave_path)
        else:
            os.makedirs(self.imgsave_path)

        # 新增参数
        self.alpha = alpha  # 自适应子图调节因子
        print(
            "正在处理第{}类第{}张紫色土图像...子图大小为:{}...Adaption自适应:{}...W舍弃方式及比例:{}...alpha自适应因子:{}".format(img_class, img_num, img_size, adaption,
                                                                                                 rate, self.alpha))

    def master(self):
        """
        主函数:用于调用其他类函数
        :return:
        """
        if self.random:
            # 随机获取土壤子图
            n = self.master_randomchose()
        else:
            # 子图算好获取土壤子图
            n = self.master_wadoss()

        return n

    def master_wadoss(self):
        """
        把原来master中的代码 封装到这个函数中去
        为了新增随机子图选择, 在视觉上更具清晰
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
        # ============================= #
        #       三.初始化第一个max(W)
        # ============================= #
        # 进行破坏性子集选择, 自定义程度
        if isinstance(self.broke, dict):
            p_broke = self.make_broke_gather()
        # 进行区间限制, 利用中心极限定理, 舍弃w矩阵的两端
        w = self.interval_limit_add(w)
        max_ij = np.where(w == np.max(w))
        # 最大值所对应的横坐标和纵坐标(当取值相同时,取第一个)
        x, y = max_ij[0][0], max_ij[1][0]
        # ============================= #
        #       四.迭代求解max(W*D)
        # ============================= #
        # 2.初始化以x,y为中心的距离矩阵
        d_before = self.make_distance2(x, y, w)
        # 3.迭代求解
        for num in range(1, self.n + 1):
            # 1.将最大值标记为-num
            w, x, y = self.set_max_ij_1(w, x, y, num)
            # 2.更新距离矩阵d
            d_after = self.update_distance2(d_before, x, y, w)
            # 3.得到特征矩阵
            feature = self.get_feature3(w, d_after)
            # 4.寻找特征矩阵的最大值对应的坐标
            x, y = self.max_ij_in_feature4(feature)
            # 5.从新赋值d_after, 开始迭代
            d_before = d_after
        point = np.where(w < 0)
        # 开始展示图像, 输入坐标开始切子图
        if isinstance(self.broke, dict) and len(p_broke[0]) != 0:
            point = self.make_broke_replace(point, p_broke)
        self.get_sub_img5(point[0], point[1])
        return self.n  # 返回子图数

    def master_randomchose(self):
        """
        随机进行子图选择
        :return:
        """
        print("正在随机子图选择...")
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
        w = self.w
        n = self.n
        # 在w矩阵中随机进行选择n个点
        point = np.where(w > 0)
        # 随机选择n个坐标
        oder = np.random.choice(point[0], n, replace=False)
        x = point[0][oder]
        y = point[1][oder]
        self.get_sub_img5(x, y)
        return n

    def lable_MAtrix1(self):
        """
       初始化标记矩阵lable, 土壤区域标记为255, 非土壤区域标记为0
       使用了保护边境措施,进行了腐蚀
       :return:可用土壤标记矩阵: [255, 0]
        """
        lable = np.copy(self.img_gray)
        # 原图背景 255, 土壤0-255->背景为0, 土壤区域为255
        lable[lable == 255] = 0
        lable[lable != 0] = 255

        # 2021年11月24日20:33:07通过膨胀再腐蚀的方法: 修正P图错误 , 2021年11月25日21:53:21:bug缩小腐蚀和膨胀的次数20->5, 避免与边缘连接
        # lable = cv2.dilate(lable, None, iterations=5)  # 膨胀, 去掉P图错误标记
        # lable = cv2.erode(lable, None, iterations=5)

        # 对标记矩阵进均均值滤波 为了解决子图切割包含背景的子图 -> 未出错, 这个是把它当成二值图像来处理了嘛? 所以不会报错.
        new_lable = cv2.blur(lable, (self.img_size, self.img_size))
        # 2022年03月03日16:59:37先均值滤波之后在预处理把, 有一个就有一个吧 有时候这个图像它本身有点问题
        lable = cv2.dilate(lable, None, iterations=5)  # 膨胀, 去掉P图错误标记   从5->3
        lable = cv2.erode(lable, None, iterations=5)

        new_lable[new_lable != 255] = 0
        # 进行边界保护措施 - > 整体缩小20个单位
        new_lable = cv2.erode(new_lable, None,
                              iterations=20)  # 腐蚀 -> 缩小   原来设置20 -> 修正为10 看看有没有bug  2022年03月03日15:05:18 因为出现一点点白色边界所以又从10调整到20
        # 将图像转换到float32
        new_lable = new_lable.astype(np.float32)

        lable[lable != 0] = 1  # 这句号应该没有用的 可以删除
        # 设置自适应获取的子图总数
        if self.adaption:
            # self.n = 2 * self.img.shape[0] * self.img.shape[1] // (self.img_size ** 2)    # step=0.5
            self.n = int(self.alpha * self.img.shape[0] * self.img.shape[1] // (self.img_size ** 2))  # step=1
        # plt.imshow(new_lable)
        # plt.title("new_lable")
        # plt.show()
        # plt.close()
        return new_lable

    def weight_Matrix2(self, lable):
        """
        初始化权重矩阵:这个相当于领域信息,是子图选择的参考
        :return:权重矩阵
        """
        # 初始化图像,权重矩阵求的是灰度图的均值
        # img = np.copy(self.img_gray)    # 服了, 这可是系统性错误啊, 怎么用的灰度图, 需要使用v通道的.
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
        return weight

    def set_max_ij_1(self, w, i, j, num):
        """
        将max_ij所对应的坐标在w中设为-num
        为了统计子图标记点,以及切割子图的先后顺序
        :param w:
        :param i:
        :param j:
        :param num:_
        :return:w, i, j
        """
        # 将最大值进行标记
        w[i][j] = -num
        return w, i, j

    def make_distance2(self, i, j, w_m):
        """
        计算以i,j为中心的距离矩阵
        标记矩阵为0的就不用计算了
        :param i:
        :param j:
        :return:
        """
        # lable = self.lable
        d = np.zeros_like(w_m)  # 直接输入原矩阵就好
        # 计算以i,j为中心,lable为参考的的计算矩阵的距离
        # lable = 255 和 0, points是所有为1的点的坐标
        # 2022年03月03日20:03:44原来计算的label矩阵 这儿其实可以计算w矩阵中大于0的点
        # points = np.where(lable == 255)
        points = np.where(w_m > 0)  # 2022年03月03日20:38:31你是怎么能蠢到用label矩阵的
        point_x = points[0]  # 横坐标
        point_y = points[1]  # 纵坐标
        # 计算欧式距离
        dis = np.sqrt(np.power(point_x - i, 2) + np.power(point_y - j, 2))
        # 赋值给矩阵d ->old 方法
        # for i, v in enumerate(dis):
        #     d[point_x[i]][point_y[i]] = v
        # 快速操作赋值 -> 速度快了八倍
        d[points] = dis
        return d

    def update_distance2(self, d_before, x, y, w_m):
        """
        这个方法被称作:最大最小距离法
        更新距离矩阵:更新方式,选取2个矩阵中对应位置的最小值
        :param d_before:迭代前的距离矩阵
        :param x: 最大feature的点横坐标
        :param y: 最大feature的点横坐标
        :return:
        """
        # 计算当前点的距离矩阵
        d_after = self.make_distance2(x, y, w_m)
        # 计算2个矩阵的最小值作为新的矩阵
        d = np.where(d_before <= d_after, d_before, d_after)
        # d = d_before + d_after
        return d

    def get_feature3(self, w, d):
        """
        得到特征矩阵:f = w+d
        :param w:
        :param d:
        :return:
        """
        # 计算特征,w 和d 矩阵 的对应元素之和或者乘积
        feature = w * d
        return feature

    def get_feature_limits3(self, w, d, limits):
        """
        得到特征矩阵:f = w+d
        :param w:
        :param d:
        :param limits:进行区间范围限制,传入的是一个列表 limits[0]:左区间, limits[1]右区间
        :return:
        """
        # 计算特征,w 和d 矩阵 的对应元素之和或者乘积
        # 不在区间范围内的w设置为0
        limit_point = np.where((w < limits[0]) | (w > limits[1]))
        w_temp = np.copy(w)
        w_temp[limit_point] = 0
        # 得到feature矩阵 然后在里面寻找最大值
        feature = w_temp * d
        return feature

    def get_the_m3(self, m):
        """
        判断当前应该返回哪一个区间, 如果当前区间为0的话就转到下一个区间
        :param regin_sub: 区间子图数
        :param m: 当前的区间
        :return:
        """
        # 区间赋值
        if np.sum(self.region_subimg) == 0:
            return np.random.randint(0, self.bins)
        else:
            while True:
                if self.region_subimg[m] > 0:
                    self.region_subimg[m] -= 1
                    return m
                else:
                    m += 1
                    if m >= self.bins:
                        m = 0

    def max_ij_in_feature4(self, feature):
        """
        寻找特征矩阵中的最大值
        :param feature:
        :return:
        """
        point = np.where(feature == np.max(feature))
        x = point[0][0]  # 横坐标
        y = point[1][0]  # 纵坐标
        return x, y

    def get_sub_img5(self, x, y):
        """
        输入坐标点, 切割子图
        :param x:
        :param y:
        :return:
        """
        step = self.img_size // 2  # x, y 是图像的中心点
        for k in range(len(x)):
            # 获取子图像
            subimg = self.img[x[k] - step:x[k] + step, y[k] - step:y[k] + step]
            # 移动
            path = self.imgsave_path + "/" + str(k) + "_" + str(self.img_class) + "_" + str(self.img_num) + ".jpg"
            # 保存图像
            cv2.imwrite(path, subimg)
        print("任务已完成, 共计获得子图{}张...".format(self.n))

    def interval_limit_add(self, w):
        """
        修改, 将方法改为舍弃w矩阵两端的比例
        且 rate设为字典形式, 方便灵活使用m:两端  l:左边 r:右边
        :param w: w权重矩阵, 进行修正
        :return:
        """
        # 找到最大最小区间
        w_max = np.max(w)
        w_min = np.min(w[np.where(w != 0)])  # np.where(w)2022年03月03日21:15:41你这个最小值 怎么会写个w呢??????? 我的天?????

        rate = self.rate  # 取出字典rate{}
        # ------------------------rate取反----------------------------
        if self.rateflip:  # 当对rate进行取反时
            print("正在对ratefilp进行取反...")
            if 'm' in rate:
                ra = rate['m']
                u = 0.5 * ra * (w_max - w_min)  # 仅在两边进行选择
                # 更新w矩阵
                w[(w >= (w_min + u)) & (w <= (w_max - u))] = 0
            elif 'l' in rate:
                ra = rate['l']
                u = ra * (w_max - w_min)  # 仅在左边进行选择
                # 更新w矩阵
                w[w >= (w_min + u)] = 0
            elif 'r' in rate:
                ra = rate['r']
                u = ra * (w_max - w_min)  # 仅在右边进行选择
                # 更新w矩阵
                w[w <= (w_max - u)] = 0
            else:
                print("未正确设置rate格式...未限制权重矩阵...")
        # ------------------------rate不取反----------------------------
        else:
            if 'm' in rate:
                ra = rate['m']
                u = 0.5 * ra * (w_max - w_min)  # 舍弃两端在中间选择
                # 更新w矩阵
                w[w < (w_min + u)] = 0
                w[w > (w_max - u)] = 0
            elif 'l' in rate:
                ra = rate['l']
                u = ra * (w_max - w_min)  # 舍弃左边
                # 更新w矩阵
                w[w < (w_min + u)] = 0
            elif 'r' in rate:
                ra = rate['r']
                u = ra * (w_max - w_min)  # 舍弃右边
                # 更新w矩阵
                w[w > (w_max - u)] = 0
            else:
                print("未正确设置rate格式...未限制权重矩阵...")
        # -----------------------------------------------------------
        return w

    def make_broke_gather(self, exten=0.05):
        """
        当判定破坏性broke为字典类型时, 进行破坏性插入, 此时进行破坏性子集构建
        :param exten:自己定义的明暗子集程度, 大于整体的0.05为过亮区域, 小于整体的0.05为过暗区域
        :return:
        """
        w = self.w
        w_max = np.max(w)
        w_min = np.min(w[np.where(w != 0)])
        rate = self.broke  # 字典
        n = self.n  # 总子图数

        if 'm' in rate:
            ra = rate['m']  # 替换比例
            u = exten * (w_max - w_min)  # 仅在两边进行选择出的子图
            # 拿到符合条件的集合
            g1 = np.where((w < w_min + u) & (w > 0))
            g2 = np.where(w > w_max - u)
            # 所需子图数
            n_sub = int(0.5 * n * ra)
            p1 = self.make_broke_p(g1[0], g1[1], n_sub)
            p2 = self.make_broke_p(g2[0], g2[1], n_sub)
            p_x = np.append(p1[0], p2[0])
            p_y = np.append(p1[1], p2[1])
            print("正在进行破坏性子图插入...破坏程度exten={}...broke为{}...替换子图数为{}...".format(exten, rate, int(n * ra)))

            return [p_x, p_y]


        elif 'l' in rate:
            ra = rate['l']  # 替换比例
            u = exten * (w_max - w_min)  # 仅在两边进行选择出的子图
            # 拿到符合条件的集合
            g1 = np.where((w < w_min + u) & (w > 0))
            # 所需子图数
            n_sub = int(n * ra)
            p1 = self.make_broke_p(g1[0], g1[1], n_sub)
            print("正在进行破坏性子图插入...破坏程度exten={}...broke为{}...替换子图数为{}...".format(exten, rate, int(n * ra)))

            return p1

        elif 'r' in rate:
            ra = rate['r']  # 替换比例
            u = exten * (w_max - w_min)  # 仅在两边进行选择出的子图
            # 拿到符合条件的集合
            g2 = np.where(w > w_max - u)
            # 所需子图数
            n_sub = int(n * ra)
            p2 = self.make_broke_p(g2[0], g2[1], n_sub)
            print("正在进行破坏性子图插入...破坏程度exten={}...broke为{}...替换子图数为{}...".format(exten, rate, int(n * ra)))

            return p2

    def make_broke_p(self, x_p, y_p, sub_n):
        """
        传入集合和所需子图数, 返回坐标的集合
        :param x_p: 满足条件的集合x
        :param y_p:满足条件的纵坐标y
        :param sub_n: 所需要选择的子图数
        :return: 集合
        """
        oder = np.arange(len(x_p))  # 坐标 0-n
        k = np.random.choice(oder, sub_n, replace=False)  # 随机选择k个坐标
        x = x_p[k]
        y = y_p[k]
        return [x, y]

    def make_broke_replace(self, p1_src, p2_broke):
        """
        输入原子集p1, 替换的子集p2
        :param p1_src:
        :param p2_broke:
        :return:
        """
        oder = np.arange(len(p1_src[0]))
        target = np.random.choice(oder, len(p2_broke[0]), replace=False)
        p1_src[0][target] = p2_broke[0]
        p1_src[1][target] = p2_broke[1]
        return p1_src


if __name__ == '__main__':
    start = time.time()
    # 输入图像路径
    # img_path = "/Users/yida/Desktop/研究生学习/数据集/土壤数据集/完整土壤数据集/手工PS数据全集/暗紫泥大泥土/5.jpg"
    # img_path = "/Users/yida/Desktop/土壤_wwr/high/1/2.jpg"    # 125.99s
    img_path = "/CutSubImage/Init_Alor_R_W_Cutsubimg/1009/plt_img/new_img.jpg"  # 51.06s

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
