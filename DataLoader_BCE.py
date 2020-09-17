
#DATA LOADER______________________________________________


from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image
import os
import numpy as np
import glob

#os.listdir() method in python is used to get the list of all files and directories in the specified directory.
#The index() method searches an element in the list and returns its index.
#The open() function opens a file, and returns it as a file object.


class NucleiSeg(Dataset):
    def __init__(self, path='', transforms=None):
        self.path = path                                                     # Path
        images = glob.glob(path+'/*')                                        # GlobFunction for put Image in List
        images = [image for image in images if '.tif' in image]              # Put Image in List
        #print(images)
        masks = [s.replace('images', 'masks')[:-4]+'.png' for s in images]   # Put Mask in List
        #print(masks)
        self.list = [(images[i], masks[i]) for i in range(len(images))]      # Put image and Mask in tuple
        #print(self.list)
        self.transforms = transforms                                         # ???????????????????????????

    def __getitem__(self, index):
        image_path = self.list[index][0]
        mask_path = self.list[index][1]
        image = Image.open(image_path)        # open the image
        image = image.convert('RGB')          # convert it into RGB 
        mask = Image.open(mask_path)          # open mask 
        mask = mask.convert('L')              # ??????????????????????????????????
        if self.transforms is not None:       # ??????????????????????????????????
            image = self.transforms(image)
            mask = self.transforms(mask)
        return (image, mask)

    def __len__(self):
        return len(self.list)                                                  # of how many data(images?) you have


transforms = transforms.ToTensor()
ds = NucleiSeg(path='/home/krishna/MONUSEG/data/train/images', transforms = transforms)                              #OBJECT

"""
print(len(ds))

i, m = ds.__getitem__(1)

print(i.shape)
print(m.shape)

#print(images.size())   # not working (just tried)
#print(ds.size())
"""
#sftp://krishna@10.107.42.188/home/krishna/MONUSEG/MONUSEG_segnet/train/images