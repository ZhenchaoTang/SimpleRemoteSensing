from PIL import Image
from torch.utils.data import Dataset
import os
import gdal
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

    
class Mydataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        '''
        :param path: 存储有图片存放地址、对应标签的文件的地址；
        :param transform: 定义了各种包括随即裁剪、旋转、仿射等在内的对图像的预处理操作
        :param target_transform:
        '''
        data = pd.read_csv(path) # 获取csv中的数据
        imgs = []
        # 删除第一行
        for i in range(len(data)):
            imgs.append((data.iloc[i, 1], data.iloc[i, 2]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        # 定一个key（即索引）来获取对应的数据样本
        fn, label = self.imgs[item]
        img = Image.open(fn).convert('RGB')
        gt = np.array(Image.open(label))
        # 进行数据增强,并将数据格式转换为tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, gt
    
    def __len__(self):
        # 返回数据集的大小规模
        
        return len(self.imgs)

# 测试实例
if __name__ == "__main__":
    # 数据预处理设置
    normMean = [0.55759144, 0.7458663, 0.4823004]
    normStd = [0.084988184, 0.1494927, 0.28076953]
    normTransfrom = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose([
            transforms.ToTensor(),
            # normTransfrom,
        ]) # 对数据转tensor,再对其进行归一化[-1, 1]
    # 构建Mydataset实例
    train_data = Mydataset(path='F:/SEAICE_data/SEAICE98/train_path_list.csv', transform=transform)
    img, gt = train_data.__getitem__(0)
    print(img.shape, gt)
