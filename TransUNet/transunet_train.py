import torch
import numpy as np
import os
import time 
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from datetime import datetime
from utils.data_utils import Mydataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models.networks.TransUnet import get_transNet
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

class args:

    train_path = 'F:/ISPRS/Creat Datasets/train_path_list.csv'
    val_path = 'F:/ISPRS/Creat Datasets//val_path_list.csv'
    result_dir = 'E:/TransFormer/TransUNet/Result/'
    batch_size = 2
    learning_rate = 0.001
    max_epoch = 50

if __name__ == "__main__":

    best_train_acc = 0.80
    best_val_acc = 0.80

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

    log_dir = os.path.join(args.result_dir, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    #---------------------------1、加载数据---------------------------
    normMean = [0.46830434, 0.3182886, 0.31384888] # 统计不同波段的均值和方差进行归一化处理
    normStd = [0.21847846, 0.15521803, 0.14722987]
    normTransfrom = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose([
            transforms.ToTensor(),
            normTransfrom,
        ]) # 对数据转tensor,再对其进行归一化[-1, 1]
    # 构建Mydataset实例
    train_data = Mydataset(path = args.train_path, transform = transform)
    val_data = Mydataset(path = args.val_path, transform = transform)
    #构建DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)

    #---------------------------2、定义网络---------------------------
    net = get_transNet(n_classes = 6)
    net.cuda()
    # 将网络结构图传入tensorboard
    init_img = torch.randn((1, 3, 256, 256), device = 'cuda')
    writer.add_graph(net, init_img)

    #---------------------------3、初始化预训练权重、定义损失函数、优化器、设置超参数、---------------------------
    if torch.cuda.is_available(): # 类别权重用于计算损失函数
        w = torch.Tensor([0.71280016, 0.77837713, 0.93428148, 1.0756635, 16.18921045, 28.26338505]).cuda()
    else:
        w = torch.Tensor([0.71280016, 0.77837713, 0.93428148, 1.0756635, 16.18921045, 28.26338505])
    criterion = nn.CrossEntropyLoss(weight = w).cuda()
    optimizer = optim.SGD(net.parameters(), lr = args.learning_rate, momentum = 0.9, dampening = 0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    #---------------------------4、训练网络---------------------------
    for epoch in range(args.max_epoch):
        epoch_start = time.time() # epoch开始时间
        loss_sigma = 0.0
        acc_sigma = 0.0
        loss_val_sigma = 0.0
        acc_val_sigma = 0.0
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels = labels.long().cuda()
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            predicts = torch.argmax(outputs, dim = 1) # softmax和argmax的区别
            acc_train = accuracy_score(np.reshape(labels.cpu(), [-1]), np.reshape(predicts.cpu(), [-1]))
            loss.backward()
            optimizer.step()
            # 统计预测信息
            loss_sigma += loss.item()
            acc_sigma += acc_train
            if i % 100 == 99:
                loss_avg = loss_sigma / 100
                acc_avg = acc_sigma / 100
                loss_sigma = 0.0
                acc_sigma = 0.0
                tf.compat.v1.logging.info("Training:Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.4f}".format(
                    epoch + 1, args.max_epoch,i+1,len(train_loader),loss_avg,acc_avg))
                writer.add_scalar("LOSS", loss_avg, epoch)
                writer.add_scalar("ACCURACY", acc_avg, epoch)
                writer.add_scalar("LEARNING_RATE", optimizer.param_groups[0]["lr"], epoch)
                # 保存模型
                if (acc_avg) > best_train_acc:
                    # 保存精度最高的模型
                    net_save_path = os.path.join(log_dir, 'train_net_params.pkl')
                    torch.save(net.state_dict(), net_save_path)
                    best_train_acc = acc_avg
                    tf.compat.v1.logging.info('Save model successfully to "%s"!' % (log_dir + 'net_params.pkl'))

        net.eval()
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels = labels.long().cuda()
            with torch.no_grad():
                outputs = net.forward(inputs)
            predicts = torch.argmax(outputs, dim=1)
            acc_val = accuracy_score(np.reshape(labels.cpu(), [-1]), np.reshape(predicts.cpu(), [-1]))
            loss_val = criterion(outputs, labels)
            # 统计预测信息
            loss_val_sigma += loss_val.item()
            acc_val_sigma += acc_val
        tf.compat.v1.logging.info("After 1 epoch：acc_val:{:.4f},loss_val:{:.4f}".format(acc_val_sigma/(len(val_loader)), loss_val_sigma/(len(val_loader)))) 
        writer.add_scalar("VAL_LOSS", loss_val_sigma/(len(val_loader)), epoch)
        writer.add_scalar("VAL_ACCURACY", acc_val_sigma/(len(val_loader)), epoch)
        # 保存验证模型
        if (acc_val_sigma/(len(val_loader))) > best_val_acc:
            # 保存精度最高的模型
            net_save_path = os.path.join(log_dir, 'val_net_params.pkl')
            torch.save(net.state_dict(), net_save_path)
            best_val_acc = acc_val_sigma/(len(val_loader))
            tf.compat.v1.logging.info('Save model successfully to "%s"!' % (log_dir + 'val_net_params.pkl'))
        epoch_end = time.time() # 当前epoch结束时间点
        tf.compat.v1.logging.info("After 1 epoch：run_time:{:.4f},number of current epoch:{:0>4}".format(((epoch_end - epoch_start)/60), (epoch + 1)))
        acc_val_sigma = 0.0
        loss_val_sigma = 0.0
        scheduler.step() 

    writer.close()
    net_save_path = os.path.join(log_dir,'net_params_end.pkl')
    torch.save(net.state_dict(),net_save_path)