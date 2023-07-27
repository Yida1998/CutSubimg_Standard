"""
把1生成的subimg,一个类下面的所有子图进行汇总

汇总子图,唯一代码!!!!!!

没有其它意思就是汇总子图!!!!

2021年08月12日10:41:53->说得好,就是这么直接,将土壤图像subimg合并放在与subimg同级的test里面
"""
# 土壤类别标签
import os
import shutil

class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']

if __name__ == '__main__':
    path = "/Users/yida/Desktop/train/subimg"
    target = os.path.abspath(os.path.join(path, '../dataset_1'))  # subimg当前目录
    # 有就删除, 没有就重新生成
    if os.path.exists(target):
        shutil.rmtree(target)
        os.makedirs(target)
    else:
        os.makedirs(target)
    print(target)


    for i in class_img:
        sub_path = os.path.join(path, i)    # 子文件路径
        sub_file = os.listdir(sub_path)     # 子文件下的图像
        if '.DS_Store' in sub_file:
            sub_file.remove('.DS_Store')
        for j in sub_file:  # 第j张图
            sub_file_img = os.path.join(sub_path, j)    # 下一步获取子图
            file_subimg = os.listdir(sub_file_img)  # 获取子图
            for item in file_subimg:
                if item.endswith('.jpg'):
                    # 子图路径
                    target_name = '_' + str(i) + '_' + str(j) + '.jpg'
                    sub_img_path = os.path.join(sub_file_img, item)     # 原路径
                    sub_class = os.path.join(target, i)    # 生成路径
                    if not os.path.exists(sub_class):
                        os.makedirs(sub_class)      # 不存在就生成文件夹
                    target_path = os.path.join(sub_class, item.split('.')[0] + target_name)     # 移动路径
                    shutil.copy(sub_img_path, target_path)  # 开始移动
        print("正在移动第{}个类".format(i))
    print("任务完成...")

