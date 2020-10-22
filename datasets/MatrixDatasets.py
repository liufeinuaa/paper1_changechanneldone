#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from datasets.matrix_aug import *

class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape(), # 仅对二维矩阵有用, 构成1通道图片，当使用3通道图像时得注释掉,
                Retype()
            ])
        else:
            self.transforms = transform


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label

