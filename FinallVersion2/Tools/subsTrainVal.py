"""
Author: yida
Time is: 2021/12/18 09:49
this Code: 将每一张图像的部分子图分为验证集, 其它作为训练集
"""
import argparse
# 土壤类别标签
import os
import shutil

# class_img = ['0', '1', '3', '4', '6', '7', '8', '9', '10']
class_img = ['暗紫泥二泥土', '暗紫泥大泥土', '暗紫泥夹砂土', '暗紫泥油石骨子土', '灰棕紫泥半砂半泥土', '灰棕紫泥大眼泥土', '灰棕紫泥石骨子土', '灰棕紫泥砂土', '红棕紫泥红棕紫泥土', '红棕紫泥红石骨子土', '红棕紫泥红紫砂泥土']

if __name__ == '__main__':
    # 设置系统参数
    parser = argparse.ArgumentParser()
    # 设置路径
    parser.add_argument('--path', type=str, default="/Users/yida/Desktop/数据集8_2/train/subimg2_224_dropm0.04_adaption_1.0", help='加载数据集路径')
    parser.add_argument('--test_rate', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--data_name', type=str, default="dataset", help='数据集保存名称')
    opt = parser.parse_args()  # 实例化

    path = opt.path
    test_rate = opt.test_rate  # 测试集比例
    data_name = opt.data_name   # 多个文件夹的话 这个参数需要修改一下
    data_name = 'Val_' + opt.path.split('/')[-1]
    target = os.path.abspath(os.path.join(path, '../' + data_name))  # subimg当前目录
    target_train = os.path.abspath(os.path.join(path, '../' + data_name + '/train'))
    target_val = os.path.abspath(os.path.join(path, '../' + data_name + '/val'))
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
