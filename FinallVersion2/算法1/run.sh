# 训练集
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/train' --img_size=600 --img_step=300 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/train' --img_size=448 --img_step=224 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/train' --img_size=336 --img_step=168 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/train' --img_size=224 --img_step=112 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/train' --img_size=168 --img_step=84 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/train' --img_size=112 --img_step=64 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/train' --img_size=64 --img_step=32 --bbox
#
#
## 测试集
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/val' --img_size=600 --img_step=300 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/val' --img_size=448 --img_step=224 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/val' --img_size=336 --img_step=168 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/val' --img_size=224 --img_step=112 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/val' --img_size=168 --img_step=84 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/val' --img_size=112 --img_step=64 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/final_dataset/0/val' --img_size=64 --img_step=32 --bbox

# 0818获取 train val的子集
#python OneKeyCode.py --path_root='/Users/yida/Desktop/纠错/high_4KFold/0/train' --img_size=224 --img_step=224 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/纠错/high_4KFold/0/val' --img_size=224 --img_step=112 --bbox
#python OneKeyCode.py --path_root='/Users/yida/Desktop/纠错/high_4KFold/0/val' --img_size=224 --img_step=224 --bbox

# 0826测试实验, 获取训练和测试子集
#python OneKeyCode.py --path_root='/Users/yida/Desktop/最终数据集/train' --img_size=224 --img_step=112 --bbox
python OneKeyCode.py --path_root='/Users/yida/Desktop/数据集8_2/train' --img_size=224 --img_step=112
python OneKeyCode.py --path_root='/Users/yida/Desktop/数据集8_2/val' --img_size=224 --img_step=112

