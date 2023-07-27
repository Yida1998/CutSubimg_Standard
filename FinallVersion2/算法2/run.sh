# 使用方法 sh xx.sh
# 训练集
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/train' --img_size=600  --bbox --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/train' --img_size=448  --bbox --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/train' --img_size=336  --bbox --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/train' --img_size=224  --bbox --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/train' --img_size=168  --bbox --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/train' --img_size=112  --bbox --adaption
# python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/train' --img_size=64 --adaption

# 测试集, 就不用了, 因为直接使用算法1获取的作为测试.
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/val' --img_size=600  --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/val' --img_size=448 --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/val' --img_size=336  --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/val' --img_size=224  --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/val' --img_size=168  --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/val' --img_size=112  --adaption
#python 2OneKeyCode.py --path='/Users/yida/Desktop/final_dataset/0/val' --img_size=64  --adaption

# 训练集 构建随机数据集random1-3
#python 2OneKeyCode.py
#python 2OneKeyCode.py --rate='m_0.02'
#python 2OneKeyCode.py --rate='m_0.32'

# 0502 构建不同阴影程度的数据集, 舍弃阴影或者亮度 2 4 8 16 32
#python 2OneKeyCode.py --rate='l_0.02' --adaption
#python 2OneKeyCode.py --rate='l_0.04' --adaption
#python 2OneKeyCode.py --rate='l_0.08' --adaption
#python 2OneKeyCode.py --rate='l_0.16' --adaption
#python 2OneKeyCode.py --rate='l_0.32' --adaption
#
#python 2OneKeyCode.py --rate='r_0.02' --adaption
#python 2OneKeyCode.py --rate='r_0.04' --adaption
#python 2OneKeyCode.py --rate='r_0.08' --adaption
#python 2OneKeyCode.py --rate='r_0.16' --adaption
#python 2OneKeyCode.py --rate='r_0.32' --adaption

# 0816 完成4折验证,舍弃不完整土壤子图, 初步实验
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/1/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/1/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/2/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/2/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/3/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/3/val'


# 0817对数据集1, 进行自适应因子调节

#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/train' --alpha=0.5
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/val'   --alpha=0.5
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/train' --alpha=1.5
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/val'   --alpha=1.5
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/train' --alpha=2.0
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/val'   --alpha=2.0

# 0818 子图选择舍弃两端的004比例构成的土壤子图
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/train' --rate='m_0.02'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/val'   --rate='m_0.02'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/train' --rate='m_0.06'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/val'   --rate='m_0.06'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/train' --rate='m_0.08'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFold/0/val'   --rate='m_0.08'

# 0821测试新的数据,构建最终数据集
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFoldNew/0/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFoldNew/0/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFoldNew/1/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFoldNew/1/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFoldNew/2/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFoldNew/2/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFoldNew/3/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_4KFoldNew/3/val'

# 0824重构土壤数据集
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/0/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/0/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/1/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/1/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/2/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/2/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/3/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/3/val'

# 0824不获取最小外接矩形
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/0/train'   --bbox
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/0/val'  --bbox
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/1/train' --bbox
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/1/val' --bbox
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/2/train' --bbox
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/2/val'  --bbox
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/3/train' --bbox
#python 2OneKeyCode.py --path='/Users/yida/Desktop/纠错/high_re/3/val'  --bbox

# 0825调试
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/test'

# 0828 获取子集,试一下重构验证集
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/train'

# 0828试一下训练测试 8:2
#python 2OneKeyCode.py --path='/Users/yida/Desktop/数据集8_2/train'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/数据集8_2/val'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/数据集8_2/train' --rate='m_0.04'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/数据集8_2/val' --rate='m_0.04'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/train' --rate='m_0.04'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/test' --rate='m_0.04'


# 0829开始最终实验, 实验二, 确定 alpha参数
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=0.5 --valBySame
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=0.5
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.5 --valBySame
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.5
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=2.0 --valBySame
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=2.0

#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=2.5 --valBySame
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=2.5

# 0831实验三, 获取不同大小的土壤子图
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=600
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=600
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=448
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=448
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=336
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=336
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=224
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=224
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=168
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --img_size=168
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=112
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=112
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=64
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --img_size=64



# 0904 实验4, 随机选择一组土壤子图数据集
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=600 --random
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --img_size=600 --random

# 人工制造
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0 --valBySame --img_size=600 --broke='l_0.5' --datasetname_end='_broke'
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --img_size=600  --broke='l_0.5' --datasetname_end='_broke'

# 0904对比时间 alpha 不使用外接矩形
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=0.5
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.5
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=2.0
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=2.5
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=600
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=448
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=336
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=224
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --img_size=168
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=112
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --img_size=64

# 时间2
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=0.5 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_224_dropm0.0_adaption_0.5'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_224_dropm0.0_adaption_1.0'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.5 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_224_dropm0.0_adaption_1.5'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=2.0 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_224_dropm0.0_adaption_2.0'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=2.5 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_224_dropm0.0_adaption_2.5'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=600 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_600_dropm0.0_adaption_1.0'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=448 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_448_dropm0.0_adaption_1.0'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=336 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_336_dropm0.0_adaption_1.0'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=224 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_224_dropm0.0_adaption_1.0'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --img_size=168 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_168_dropm0.0_adaption_1.0'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=112 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_112_dropm0.0_adaption_1.0'
#
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0 --img_size=64 --bbox --adaption --same_path='/Users/yida/Desktop/未命名文件夹/subimg2_64_dropm0.0_adaption_1.0'

#0906 实验四 人工制造样本集
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0  --img_size=600  --valBySame

# 0907补充两组时间
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0  --img_size=56 --valBySame
#python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/train' --alpha=1.0  --img_size=560 --valBySame
python 2OneKeyCode.py --path='/Users/yida/Desktop/最终数据集/微调/test' --alpha=1.0  --img_size=560 --bbox --adaption --same_path='/Users/yida/Desktop/最终数据集/微调/test/subimg2_560_dropm0.0_adaption_1.0'
