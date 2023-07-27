"""
Author: yida
Time is: 2022/1/27 15:06 
this Code: 合并土壤的大类, 将同一类型下的土种进行合并
"""
import os
import shutil

class_img = {0: "暗紫泥大泥土",
             1: "暗紫泥二泥土",
             2: "暗紫泥夹砂土",
             3: "暗紫泥油石骨子土",
             4: "灰棕紫泥大眼泥土",
             5: "灰棕紫泥半砂半泥土",
             6: "灰棕紫泥砂土",
             7: "灰棕紫泥石骨子土",
             8: "红棕紫泥红棕紫泥土",
             9: "红棕紫泥红石骨子土",
             10: "红棕紫泥红紫砂泥土"}

if __name__ == '__main__':
    path = "/Users/yida/Desktop/研究生学习/数据集/土壤数据集/完整土壤数据集/high_华为"
    target = "/Users/yida/Desktop"  # 目标路径
    target = os.path.join(target, path.split('/')[-1])
    file = os.listdir(path)
    for i in file:
        if '.' not in i:  # 提出  .txt和 .Ds隐藏文件
            path_sub = os.path.join(path, i)
            file_sub = os.listdir(path_sub)
            for item in file_sub:
                if item.endswith('.jpg'):
                    src = os.path.join(path_sub, item)
                    if i in ['0', '1', '2', '3']:
                        dst = os.path.join(target, '0_暗')
                    elif i in ['4', '5', '6', '7']:
                        dst = os.path.join(target, '1_灰')
                    elif i in ['8', '9', '10']:
                        dst = os.path.join(target, '2_红')
                    # 生成路径
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                        print("{}正在被生成...")
                    dst = os.path.join(dst, i + '_' + item)
                    # copy文件
                    shutil.copy(src, dst)
                    print("{} to {}...".format(src, dst))
    print("操作完成...")