#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: megaface.py
@time: 2018/12/24 16:29
@desc:
'''

import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
from PIL import Image

def img_loader(path):
    try:
        img = Image.open(path)
        """
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)"""
        #print(img.size)

        return img
    except IOError:
        print('Cannot load image ' + path)                                     

class MegaFace(data.Dataset):
    def __init__(self, megaface_dir, transform=None, loader=img_loader):

        self.transform = transform
        self.loader = loader

        test_image_file_list = []
        print('Scanning files under facescrub and megaface...')
        for root, dirs, files in os.walk(megaface_dir):
            for e in files:
                filename = os.path.join(root, e)
                ext = os.path.splitext(filename)[1].lower()
                if ext in ('.png', '.bmp', '.jpg', '.jpeg'):
                    test_image_file_list.append(filename)

        self.image_list = test_image_file_list

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = self.loader(img_path)

        #水平翻转图像
        #img = cv2.flip(img, 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        return img, img_path

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    #megaface = '/home/jimmyok/project/FaceScrub/facescrub_aligned/Dana_Delany'
    megaface = '/home/jimmyok/project/Megaface/001'
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(112),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    dataset = MegaFace(megaface, transform=transform)
    trainloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    print(len(trainloader))
    for i, data in enumerate(trainloader): #this returns tensor & address
        print(i) 
        print(data[0].shape)
        print(data[0][0].shape)
