#!/usr/bin/env python
import os
from skimage import io, transform
from PIL import Image
import numpy as np

path1 = './data/Set5_bicubic'
path2 = './data/Set5_lr'

if not os.path.exists(path1):
    os.makedirs(path1)

if not os.path.exists(path2):
    os.makedirs(path2)

for filename in os.listdir('data/Set5'):
    if filename.endswith('.bmp'):
        img = io.imread('data/Set5/' + filename)
        h,w,c = img.shape
        img = transform.resize(img, (h//4, w//4, c),order=3)
        bicubic = transform.resize(img, (h, w, c),order=3)
        io.imsave(path1 + '/' + filename, bicubic)
        nearest = Image.fromarray(img.astype(np.uint8)).resize((h, w),Image.NEAREST)
        print(nearest)