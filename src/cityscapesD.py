import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from labels import labels
import json


name_to_num={'person': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7, '': 8}
num_to_name = {v: k for k, v in name_to_num.items()}


class Cityscapes(Dataset):
    def __init__(self, root_dir, split='train'):
        
        self.root_dir = root_dir
        self.split = split

        self.transform_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RGBShift(p=0.5),
            A.GaussNoise(p=0.5),
            A.Blur(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Resize(1536, 1536),
        ])

        self.transform_val = A.Compose([
            A.Resize(1536, 1536),
        ])
        
        
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.depth_dir = os.path.join(root_dir, 'disparity', split)
        self.pseudo_dir = os.path.join(root_dir, 'depth', split)
        
        self.image_files= sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(self.images_dir) for f in fn if f.endswith('_leftImg8bit.png')])
        self.depth_files = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(self.depth_dir) for f in fn if f.endswith('_disparity.png')])
        self.pseudo_files = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(self.pseudo_dir) for f in fn if f.endswith('_depth.png')])

    

    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, idx):

        # if self.task == 'depth':

        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)       #no GT for label of train_extra
        label =cv2.imread(self.pseudo_files[idx], cv2.IMREAD_UNCHANGED) #0-80 
        disparity = cv2.imread(self.depth_files[idx], cv2.IMREAD_UNCHANGED)

        #disparity could be dispatiy of GT of cityscapes or pseudo depth of the teacher
        # print(disparity.shape, disparity.max(), disparity.min()) [0-31209]
        # print(label.shape, label.max(), label.min()) [0-255]
        if disparity.max() > 300: #cityscapes disparity

            disparity[disparity>0] = (disparity[disparity>0] - 1.0) / 256.0

            b= 0.209313 #baseline in meters
            fx= 2262.52 #focal length in pixels
            depth = (fx * b) / (disparity + 1e-6) 
            depth[depth> 1e6] = 0
            depth = np.clip(depth, 0, 80) #clip to 80m
        else: #pseudo depth of teacher
            depth = disparity / disparity.max() * 80.0 #normalize to 0-80m


        if self.split != 'val':
            aug = self.transform_train(image=image, masks=[label, depth])
            image = aug['image']
            label, depth = aug['masks']

        if self.split == 'val':
            aug = self.transform_val(image=image, masks=[label, depth])
            image = aug['image']
            label, depth = aug['masks']
        
        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # C, H, W
        image = image / 255.0

        label = torch.tensor(label, dtype=torch.int64)
        depth = torch.tensor(depth, dtype=torch.float32)

        # image= image.to('cuda') 
        # label = label.to('cuda')
        # depth = depth.to('cuda')

        

        return image, label, depth
    
 

## Example usage for testin purposes
if __name__ == "__main__":

    transform = None
    dataset = Cityscapes(root_dir='/home/avalocal/thesis23/KD/ml-depth-pro/train_merge_1860_186', split='train')
    print(len(dataset))

    for i in range(len(dataset)):
        image, label, depth = dataset[i]
        print(image.shape, label.shape, depth.shape)
        breakpoint()
        # print(image.shape, label.shape, depth.shape)
        # print(np.unique(label))
        # print(np.unique(depth))
        # print(np.max(depth), np.min(depth))
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()
        # plt.imshow(depth)
        # plt.show()
        # break


