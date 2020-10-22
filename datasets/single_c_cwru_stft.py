import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.MatrixDatasets import dataset

from tqdm import tqdm


from datasets.matrix_aug import *

from scipy import signal
import cv2
import pickle
import torchvision.transforms as transforms


#Digital data was collected at 12,000 samples per second
#固定输入样本信号的长度为1024个采样点
signal_size = 1024

dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm

datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
axis = ["_DE_time", "_FE_time", "_BA_time"]

label = [i for i in range(0, 10)]


def STFT(fl):
    f, t, Zxx = signal.stft(fl, nperseg=64)
    img = np.abs(Zxx) / len(Zxx)
    return img



def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
                path1 = os.path.join(root, datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root, datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1, dataname[N[k]][n], label=label[n])
            data += data1
            lab += lab1

    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''

    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size

    fl = fl.reshape(-1, )
    while end <= fl.shape[0]:
        x = fl[start:end]
        imgs = STFT(x)  # STFT 出来的是个二维np数组，并不是标准的图像格式，但是也可是视为灰度图像

        # 新增的是，将STFT的结果转为3通道的灰度图像
        # imag = (imgs * (255 / (imgs.max() - imgs.min()))).astype(np.uint8)
        # # img = cv2.merge([imag, imag, imag]) #3通道的灰度图像
        # img = cv2.applyColorMap(imag, cv2.COLORMAP_JET)  # 3通道的彩色图像，JET底色为红色，HSV为蓝色， RAINBOW也为蓝色, 调用的是伪彩色生成函数

        data.append(imgs)
        lab.append(label)
        start += signal_size
        end += signal_size


    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class CWRU_STFT(object):
    num_classes = 10
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype

        self.data_transforms = {
            'train': Compose([  # Compose方法来自sequence_aug.py
                ReSize(size=10.0),
                Reshape(),
                # Normalize(self.normlize_type),
                RandomScale(),
                Retype(),
            ]),



            'val': Compose([
                ReSize(size=10.0),
                Reshape(),
                # Normalize(self.normlizetype),
                Retype(),
            ])

        }




    def data_split(self, transfer_learning=True):

        if transfer_learning:

            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)

            # if len(os.path.basename(self.data_dir).split('.')) == 2:
            #     with open(self.data_dir, 'rb') as fo:
            #         list_data = pickle.load(fo, encoding='bytes')
            # else:
            #     list_data = get_files(self.data_dir, test, 'STFT')  # 使用STFT得到时频域信号（二维的，可以视为图像信号）
            #     with open(os.path.join(self.data_dir, "CWRUSTFT.pkl"), 'wb') as fo:
            #         pickle.dump(list_data, fo)


            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

            trains_pd, test_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"]) # 用的sklearn中的测试集分割来实现训练集的验证集分割
            train_pd, val_pd = train_test_split(trains_pd, test_size=0.20, random_state=40, stratify=trains_pd['label'])

            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            source_test = dataset(list_data=test_pd, transform=self.data_transforms['val'])



            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.8, random_state=40, stratify=data_pd["label"])

            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])




            return source_train, source_val, source_test, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

            trains_pd, test_pd = train_test_split(data_pd, test_size=0.2, random_state=40,
                                                  stratify=data_pd["label"])  # 用的sklearn中的测试集分割来实现训练集的验证集分割
            train_pd, val_pd = train_test_split(trains_pd, test_size=0.20, random_state=40, stratify=trains_pd['label'])

            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            source_test = dataset(list_data=test_pd, transform=self.data_transforms['val'])



            # get target val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val']) # 没有目标域上的训练集只有测试集

            return source_train, source_val, source_test, target_val


