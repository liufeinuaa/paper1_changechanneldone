import os
from PIL import Image
from torchvision import transforms
import numpy as np
from datasets.matrix_aug import *
import torch
from torch.utils.data import Dataset

class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        super(dataset, self).__init__()
        self.test = test
        self.transform = transform
        if self.test:
            self.imgs = list_data['data'].tolist()
        else:
            self.imgs = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()

        if transform is None:
            self.transforms = Compose([
                Retype(),
            ])
        else:
            self.transforms = transform


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):

        if self.test:
            img = self.imgs[item]
            img = self.transforms(img)
            return img, item
        else:
            img = self.imgs[item]
            label = self.labels[item]
            img = self.transforms(img)

            return img, label
