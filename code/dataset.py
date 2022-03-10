import torch.utils.data as data
import torch
import numpy as np
import h5py
import os
from skimage import io, transform, color


def data_augment(im, num):  # 数据扩增，data augmentation
    # org_image = im.transpose(1,2,0)
    org_image = im
    if num == 0:
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num == 1:
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num == 2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)
        tranform = lrud_image
    elif num == 3:
        rotated_image1 = np.rot90(org_image)
        tranform = rotated_image1
    elif num == 4:
        rotated_image2 = np.rot90(org_image, -1)
        tranform = rotated_image2
    elif num == 5:
        rotated_image1 = np.rot90(org_image)
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num == 6:
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    # tranform = tranform.transpose(2,0,1)c
    return tranform


def random_crop(im, crop_size, mode='y'):
    if mode == 'y':
        h, w = im.shape
    else:
        h, w, c = im.shape
    x = np.random.randint(0, w-crop_size)
    y = np.random.randint(0, h-crop_size)
    if mode == 'y':
        return im[y:y+crop_size, x:x+crop_size]
    else:
        return im[y:y+crop_size, x:x+crop_size, :]


def random_scale(im, mode='y'):
    scale = np.random.uniform(0.5, 1.0)
    if mode == 'y':
        h, w = im.shape
    else:
        h, w, c = im.shape
    new_h = int(h*scale)
    new_w = int(w*scale)
    if mode == 'y':
        rim = transform.resize(im, (new_h, new_w), order=3)
    else:
        rim = transform.resize(im, (new_h, new_w, c), order=3)
    return rim


class DatasetFromImage(data.Dataset):
    def __init__(self, hr_path):
        super(DatasetFromImage, self).__init__()
        self.files = os.listdir(hr_path)
        self.HR = []
        for f in self.files:
            if f.endswith('.png'):
                self.HR.append(f)
        self.path = hr_path
        self.crop_size = 256  # [128,160,192,224,256]
        self.scale = 4
        self.mode = 'y'   # rgb2ycbcr，取出y channel

    def __getitem__(self, index):
        num = np.random.randint(0, 8)
        HR = io.imread(os.path.join(self.path, self.HR[index]))
        if self.mode == 'y':
            HR = color.rgb2ycbcr(HR)[:, :, 0]
        HR = random_scale(HR)
        HR_crop = random_crop(HR, crop_size=self.crop_size)
        LR_crop = transform.resize(HR_crop, (self.crop_size//self.scale,
                                   self.crop_size//self.scale), order=3)  # bicubic downsample
        HR_crop = data_augment(HR_crop, num)
        LR_crop = data_augment(LR_crop, num)
        if self.mode == 'y':
            HR_t = torch.from_numpy(HR_crop.astype(
                np.float32)).unsqueeze(0)/255.0
            LR_t = torch.from_numpy(LR_crop.astype(
                np.float32)).unsqueeze(0)/255.0
        else:
            HR_t = torch.from_numpy(HR_crop.astype(
                np.float32).transpose(2, 0, 1))/255.0
            LR_t = torch.from_numpy(LR_crop.astype(
                np.float32).transpose(2, 0, 1))/255.0
        return LR_t, HR_t

    def __len__(self):
        return len(self.HR)
