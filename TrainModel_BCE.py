import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from SegNet_16 import *
import os
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from DataLoader_segnet import NucleiSeg
from torchvision.utils import save_image
#import pdb

  
transforms_train = transforms.Compose([transforms.Resize(1024), 
                transforms.ToTensor()])                                  

transforms_temp = transforms.Compose([transforms.ToTensor()])

batch_size = 1                                                                                       # Batch size                                                       
num_epochs = 201                                                                                     # Number of epochs
train_set = NucleiSeg(path='/home/krishna/MONUSEG/data/train/images', transforms = transforms_train) # TrainingSet from DataLoader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers = 4)            # Import From Pytorch
train_dataset_sizes =len(train_set)                                                                  # Size of Training DataSet
test_set = NucleiSeg(path='/home/krishna/MONUSEG/data/validation/images', transforms = transforms_train)   # TestSet from DataLoader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,num_workers = 4)              # Import from Pytorch
test_dataset_sizes =len(test_set)                                                                    # Size of Test DataSet



tb=SummaryWriter('runs/run2')


def train():
    criterion = nn.BCEWithLogitsLoss()                                       # loss function (binary_cross_entropy_with_logits_Loss) 
    cuda = torch.cuda.is_available()                                         # NVDIA driver
    net = SegNet()                                                     # UNet11 is Import from MODEL
    #print(net)
    #exit()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)                  # Optimizer
    if cuda:
        net = net.cuda()                                                     # Model put into Cuda

    iteration_count_train = 0                                                # ??????????? 
    iteration_count_val = 0                                                  # ????????? 


    for epoch in range(num_epochs):
        print(f'---------------EPOCH = {epoch}---------------------------')  # count epoch in Output 
        # net.train()                                                        # 
        running_loss = 0.0                                                   # 

        for i, (images, masks) in enumerate(train_loader):                   # Using TrainLoader enumerate call one by one image and mask

            if cuda:
                images = images.cuda()                                       # Image put into Cuda
                masks = masks.cuda()                                         # Mask put into Cuda
            optimizer.zero_grad()                                            # Optimizer as ZeroGradient
            outputs = net(images)                                            # Image goes into Model and give Output
            #print(outputs.shape)
            temp = masks.repeat(1,3,1,1)
            #fpdb.set_trace()                                     #
            cat_1 = torch.cat((images, temp), 0)                             # concatenation between images and temp
            #print(cat_1.size())
            temp = outputs.repeat(1,3,1,1)    # change
            temp = nn.Sigmoid()(temp)          # change
            cat_2 = torch.cat((cat_1, temp), 0)                              # concatenation between cat_1 and temp
            #print(cat_2.size())
            save_image(cat_2, f'predstrain/predstrain_{epoch}_{i}.png', normalize=True, nrow=1, padding=5, pad_value=1)   # SaveImage 
            train_iteration_loss = criterion(outputs, masks)                 # Calculate LOSS using output and mask --- then print it
            print(train_iteration_loss)
            train_iteration_loss.backward()                                  # Calculate gradient using Loss
            optimizer.step()                                                 # Update weight using Optimizer
            running_loss += train_iteration_loss.item() * images.size(0)
            tb.add_scalar('train_iteration_loss', train_iteration_loss.item(), iteration_count_train)
            iteration_count_train += 1
        train_epoch_loss = running_loss / train_dataset_sizes
        tb.add_scalar('train_epoch_loss',train_epoch_loss,epoch)

        if epoch % 5 == 0:
            with torch.no_grad():
                for j, (images, masks) in enumerate(test_loader):
                    if cuda:
                        images = images.cuda()
                        masks = masks.cuda()
                    outputs = net(images)
                    temp = masks.repeat(1,3,1,1)
                    cat_1 = torch.cat((images, temp), 0)
                    temp = outputs.repeat(1,3,1,1)
                    temp = nn.Sigmoid()(temp)
                    cat_2 = torch.cat((cat_1, temp), 0)
                    save_image(cat_2, f'predsval/predsval_{epoch}_{j}.png', normalize=True, nrow=1, padding=5, pad_value=1)
                    val_iteration_loss = criterion(outputs, masks)
                    print(val_iteration_loss)
                    running_loss += val_iteration_loss.item() * images.size(0)
                    tb.add_scalar('val_iteration_loss',val_iteration_loss.item(),iteration_count_val)
                    iteration_count_val += 1
                val_epoch_loss = running_loss / test_dataset_sizes
                tb.add_scalar('val_epoch_loss',val_epoch_loss,epoch)


        
        

    return net

train()