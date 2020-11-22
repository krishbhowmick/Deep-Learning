#TEST THE MODEL

import torch
import torch.nn.functional as F
import torch.nn as nn
from DataLoader_BCE import NucleiSeg                # Test Dataset Import
from UNet_Model import *                            # import Archietechture
from torchvision import transforms 
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import random
import os
import random
import cv2
import numpy as np


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Device or CUDA
device = torch.device('cpu')


# InterSection Over Union(IOU)-[output_mask-PredictedOutput, gt_mask- InputMask]
def IOU(output_mask,gt_mask):
    num=torch.sum((output_mask.float()*gt_mask.float())*1.0)
    den=output_mask.float() + gt_mask.float()
    den[den==2]=1
    den=torch.sum(den)
    return torch.tensor(float(num.item())/float(den.item()))


train_batch_size = 1                   # Test BatchSize
learning_rate = 10e-4                  # Learning Rate


model = torch.load('/home/krishna/MONUSEG/MONUSEG_UNet/MONUSEG_UNet_BCE/SaveModel/105th.pth').to(device)    # Trained Model Import 



"""
cuda = torch.cuda.is_available()
if cuda:
  model = model.cuda() """                                                                                # Put Model into Cuda


s='GC_test'                                                                                               # TensorBoard SerialNumber
tb=SummaryWriter('runs/'+s)                                                                               # TensorBoard Define


transforms_test = transforms.Compose([transforms.Resize(1024),transforms.ToTensor()])
dataset_test = NucleiSeg(path='/home/krishna/MONUSEG/data/test/images', transforms = transforms_test)     # Test Set
data_loader_test = DataLoader(dataset_test, batch_size=train_batch_size, shuffle=False, num_workers=4)    # TestDataLoader

 
ind=0                                                            # Counting 
for i, (images, masks) in enumerate(data_loader_test):           # From TestDataSet X-InputImage ,y-Mask ,image_path- 
    #if cuda:
    image = images.to(device).float()
    mask = masks.to(device).float()
    predicted_output = model(image)  # [N, 2, H, W]                    # fpred-Predicted from Model 
    predicted_output=(predicted_output>0.6).float()
    #fpred = F.softmax(prediction,dim=1)                          # Use Softmax in Predicted Image
    iou_score=IOU(torch.argmax(predicted_output,1),mask)                    # IOU Operation- Compair b/w output & mask 
    ind=ind+1

    tb.add_scalar('Test Average IOU',iou_score,ind)              # TensorBoard 
    print ("Batch num:",ind,"IOU",iou_score)
    save_image(predicted_output, f'predsiou/predsiou_{i}.png', normalize=True, nrow=1, padding=5, pad_value=1)


