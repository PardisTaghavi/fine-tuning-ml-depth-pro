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
            A.Resize(518, 1036),
        ])

        self.transform_val = A.Compose([
            A.Resize(518, 1036),
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

        #disparity precomputed disparity depth maps. To obtain the disparity values, 
        # compute for each pixel p with p > 0: d = ( float(p) - 1. ) / 256.,
        #  while a value p = 0 is an invalid measurement.
        # disparity = (disparity - 1.0) / 256.0 # ifp=0 -> p-1/
        
        disparity[disparity>0] = (disparity[disparity>0] - 1.0) / 256.0
        b= 0.209313 #baseline in meters
        fx= 2262.52 #focal length in pixels

        depth = (fx * b) / (disparity + 1e-6) #depth = (fx * b) / disparity

        # print("max-min depth values", np.max(depth), np.min(depth))
        # print("unique values", np.sort(np.unique(depth)))
        #invalid measurements to zero
        depth[depth> 1e6] = 0
        depth = np.clip(depth, 0, 80) #clip to 80m

        #show depth 
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('TkAgg')
        # plt.imshow(depth)
        # plt.show()

        # breakpoint()

        # we might need to clip to [0-80m later] #TODO





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
    dataset = Cityscapes(root_dir='/media/avalocal/T7/pardis/pardis/perception_system/datasets/cityscapes', split='train_unlabeled', task='instance')
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    # print(len(batch))
    # print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4], batch[5])

