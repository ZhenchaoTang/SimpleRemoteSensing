import numpy as np
import cv2
import gdal
import random
import pandas as pd

'''
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素归一化至 0-1 再计算
'''
# 读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset

if __name__ == "__main__":


    train_csv_path = 'F:/SEAICE_data/SEAICE92S/train_path_list.csv'


    CNum = 1000 # 挑选多少张图片进行计算

    img_h, img_w = 256, 256
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    data = pd.read_csv(train_csv_path) # 获取csv表中的数据
    data = data.sample(frac = 1.0) # shuffle, 随机抽样, 百分比抽样
    data = data.reset_index(drop = True) # 重新添加index, drop表示丢弃原有index一列

    for i in range(CNum):
        img_path = data.iloc[i, 1]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]
        if i == 0:
            imgs = img
        else:
            imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32) / 255

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel() # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print(means)    # BGR
    means.reverse() # RGB
    print(stdevs)
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

    '''
    train
    Num = 5000
    [0.55954075, 0.55954075, 0.55954075]
    [0.08008871, 0.08008871, 0.08008871]
    normMean = [0.55954075, 0.55954075, 0.55954075]
    normStd = [0.08008871, 0.08008871, 0.08008871]
    transforms.Normalize(normMean = [0.55954075, 0.55954075, 0.55954075], normStd = [0.08008871, 0.08008871, 0.08008871])
    val
    Num = 108
    [0.5679135, 0.5679135, 0.5679135]
    [0.07787021, 0.07787021, 0.07787021]
    normMean = [0.5679135, 0.5679135, 0.5679135]
    normStd = [0.07787021, 0.07787021, 0.07787021]
    transforms.Normalize(normMean = [0.5679135, 0.5679135, 0.5679135], normStd = [0.07787021, 0.07787021, 0.07787021])
    test
    Num = 112
    [0.57464457, 0.57464457, 0.57464457]
    [0.09158211, 0.09158211, 0.09158211]
    normMean = [0.57464457, 0.57464457, 0.57464457]
    normStd = [0.09158211, 0.09158211, 0.09158211]
    transforms.Normalize(normMean = [0.57464457, 0.57464457, 0.57464457], normStd = [0.09158211, 0.09158211, 0.09158211])
    '''
