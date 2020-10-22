
import numpy as np
import torch
import cv2
import random
from scipy.signal import resample
from PIL import Image
import scipy

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object): # 使用cv2生成三通道图片时，问题就出现在这里，不能使用这个reshape，这个是专门针对二维情形时的
    def __call__(self, seq):

        seq = seq[np.newaxis, :, :]
        return seq


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)

class ReSize(object):
    def __init__(self, size=1):
        self.size = size
    def __call__(self, seq):
        # seq = scipy.misc.imresize(seq, self.size, interp='bilinear', mode=None)

    # 因为高版本的scipy中将misc.imresize删除了，所以在使用colab进行训练时会报错，将代码修改如下
        if type(self.size) != tuple:
            size = np.array(seq.shape)
            if size.size == 2:
                size = tuple(size*int(self.size))
                seq = cv2.resize(seq, size, interpolation=cv2.INTER_LINEAR)
                # seq = np.array(Image.fromarray(seq).resize(size))
            elif size.size == 3:  # 当引入3通道图像时
                size3 = (size[0]*int(self.size), size[1]*int(self.size))
                seq = cv2.resize(seq, size3, interpolation=cv2.INTER_LINEAR)
        else:
            seq = cv2.resize(seq, self.size, interpolation=cv2.INTER_LINEAR)
            # seq = np.array(Image.fromarray(seq).resize(self.size))

        # seq = seq / 255
        return seq



class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        return seq*scale_factor


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1, 1))
            return seq*scale_factor

class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_height = seq.shape[1] - self.crop_len
            max_length = seq.shape[2] - self.crop_len
            random_height = np.random.randint(max_height)
            random_length = np.random.randint(max_length)
            seq[random_length:random_length+self.crop_len, random_height:random_height+self.crop_len] = 0
            return seq


class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif self.type == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq