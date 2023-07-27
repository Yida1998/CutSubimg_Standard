"""
对图像进行交叉验证, 用于检验分类效果
对每个类别的n张图像进行交叉验证分类 获取数据集 从而在训练网络时进行交叉验证
输入:数据集路径 保存数据集的位置 k折交叉验证


输出:k个数据集
将一个数据集分成k份,其中由k-1份组成训练集
余下1份组成测试集

2022年04月14日20:03:14 -> 新增对目录的判断, 删除移除隐藏文件 代码不够严谨
"""
import os
import shutil
import time

from sklearn.model_selection import KFold


def make_path(path_targe):
    """
    输入一个路径, 如果存在就删除 不存在就生成
    :param path_targe:
    :return:
    """
    # 判断是否存在并重新创建文件夹
    if os.path.exists(path_targe):
        shutil.rmtree(path_targe)
        os.makedirs(path_targe)
        print("succeed : ", path_targe)
    else:
        os.makedirs(path_targe)
        print("succeed : ", path_targe)


if __name__ == '__main__':
    # ==================设置超参数===================== #
    #  root_data : 数据路径    格式:文件名->n个类别->m张图像(需符合pytorch读取训练数据集要求)
    root_data = "/Users/yida/Desktop/纠错/high3"
    #  save_path:存放数据路径
    save_path = "/Users/yida/Desktop"
    #  k_num : 设置交叉验证数
    k_num = 4
    # =============================================== #
    # 定义交叉验证:设置参数  shuffle=True 打乱, random_state随机数种子
    kf = KFold(n_splits=k_num, shuffle=True, random_state=0)

    # 生成存放文件目录
    # save_path = os.path.join(save_path, time.strftime("%Y-%m-%d"))
    save_path = os.path.join(save_path, root_data.split('/')[-1])
    file = os.listdir(root_data)
    start_time = time.time()  # 记录操作时间
    for i in file:
        root_sub = os.path.join(root_data, i)
        flag = os.path.isdir(root_sub)  # 验证root_sub是否为目录
        if flag:
            file_sub = os.listdir(root_sub)
            print(file_sub)
            for n, data in enumerate(kf.split(file_sub)):
                train, test = data
                save_path_sub_train = os.path.join(save_path, str(n), "train", i)
                save_path_sub_test = os.path.join(save_path, str(n), "val", i)
                # 生成路径
                make_path(save_path_sub_train)
                make_path(save_path_sub_test)
                print(len(train), len(test))
                for item in train:
                    img_name = file_sub[item]
                    # 图像路径
                    path_train = os.path.join(root_sub, img_name)
                    # 目标路径
                    targe_path = os.path.join(save_path_sub_train, img_name)
                    # 开始移动
                    shutil.copy(path_train, targe_path)
                    print("{} to {}...".format(path_train, targe_path))
                for item in test:
                    img_name = file_sub[item]
                    # 图像路径
                    path_train = os.path.join(root_sub, img_name)
                    # 目标路径
                    targe_path = os.path.join(save_path_sub_test, img_name)
                    # 开始移动
                    shutil.copy(path_train, targe_path)
                    print("{} to {}...".format(path_train, targe_path))
    print("End...{}折交叉验证已完成 ,  数据集路径为: {}".format(k_num, save_path))
    end_time = time.time()
