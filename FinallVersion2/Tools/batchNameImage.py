# -*- coding:utf8 -*-
'''
--------------------------------运行说明-----------------------------------
代码无问题, 可直接运行!!!!!2022年04月01日11:05:32

该代码可以直接使用 如果文件夹内有重复的图像会以一个较大的随机数进行重命名

程序运行2遍即可完全重命名

陈嘿萌2021年03月19日10:09:50

参考:CSDN已作出修改

2021年04月08日10:07:48
如果修改图像结尾的话,导致命名不一致,无法进行判断条件,会造成图像覆盖
现在新增img_end 用来修改图像结尾,对应位置全部替换成img_end

2022年04月01日10:51:16
稍微调整一下代码, 把参数从外面传进类中去
--------------------------------运行说明-----------------------------------

使用说明:
最多运行程序两遍, 即可完全重命名图像
'''
import os
import random


class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''

    def __init__(self, path, img_end='.jpg', img_suffix='.jpg', img_start=1):
        self.path = path  # 表示需要命名处理的文件夹
        self.img_end = img_end  # 图像结尾标识 默认为'.jpg'
        self.img_suffix = img_suffix  # 图像后缀类型
        self.img_start = img_start  # 图像起始编号

    def rename(self):
        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        filelist = os.listdir(self.path)  # 获取文件列表
        total_num = len(filelist)  # 获取文件夹内所有文件个数
        i = self.img_start  # 表示文件的命名是从1开始的
        flog = 0  # 标记重复图像
        img_end = self.img_end  # 标注需要修改的结尾  !!!
        for item in filelist:  # 文件在文件夹内
            if item.endswith(self.img_suffix):  # 判断后缀
                # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的即可）
                src = os.path.join(os.path.abspath(self.path), item)  # 获取图像名字
                print(str())
                while str(i) + img_end in filelist:  # 如果图像的名字已经在文件夹中重复的话
                    r = random.randint(1000, 10000)  # 生成一个随机数 重新开始命名
                    i += r
                    # 这样就可以了!!!
                    flog = 1  # 标记图像有重复
                dst = os.path.join(os.path.abspath(self.path), str(i) + img_end)  # 修改的名字 可以任意修改结尾如jpeg
                os.rename(src, dst)  # 重命名
                print('converting %s to %s ...' % (src, dst))
                i = i + 1

        print('total %d to rename & the end of num is  %d' % (total_num, i - 1))
        if flog == 1:
            print("有重复图像张请重新运行一遍程序...")
        else:
            print("任务已完成...")


if __name__ == '__main__':
    path_ = '/Users/yida/Desktop/202202226吴蔚然'  # 待重命名数据路径
    img_end_ = '.jpg'  # 图像结尾标识
    img_suffix_ = '.jpg'  # 判断图像的类型
    img_start_ = 1  # 图像起始标号
    demo = BatchRename(path_, img_end_, img_suffix_, img_start_)
    demo.rename()
