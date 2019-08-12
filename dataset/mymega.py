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

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)

class Rescale(object):
    """Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

                            
    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.
        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        return image
                                                                                                                                                                                                                                                                                                                                                                                            

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
    megaface = '/home/jimmyok/project/FaceScrub/facescrub_aligned/Dana_Delany'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    dataset = MegaFace(megaface, transform=transform)
    trainloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    print(len(trainloader))
    print(type(trainloader))
    for i, data in enumerate(trainloader): #this returns tensor & address
        print(type(data[0]))
        print(i) 
        print(data[0].shape)
