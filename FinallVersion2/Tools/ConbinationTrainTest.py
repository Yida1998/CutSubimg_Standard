"""
Author: yida
Time is: 2022/3/3 09:17 
this Code: 整理训练集和测试集及中间过程的subimg, 将其和并到上层文件夹
"""
import os
import shutil


def makedir(path):
    """
    新建文件夹, 传入路径
    :param path:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


if __name__ == '__main__':
    # 根目录
    root = '/Users/yida/Desktop/final_dataset/0/val'
    # 数据集名称
    subimg_name = 'subimg'
    dataset_name = 'Set'
    # 生成目标路径
    target_subimg = os.path.join(root, 'SUBIMGS')
    target_train = os.path.join(root, 'TRAINS')
    # 新建文件夹
    makedir(target_subimg)
    makedir(target_train)
    file = os.listdir(root)
    for item in file:
        if subimg_name in item:
            src = os.path.join(root, item)
            shutil.move(src, target_subimg)
            print("{} -> {}".format(src, target_subimg))
        elif dataset_name in item:
            src = os.path.join(root, item)
            shutil.move(src, target_train)
            print("{} -> {}".format(src, target_train))
