import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F



def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)            #[in-input image/ out-output image/ 3-kernel size/ zero padding]  
"""
def concat(xs):
    return torch.cat(xs, 4)                             #[xs- / instade of 1, anyvalue put it can run]

class Conv3BN(nn.Module):
    def __init__(self, in_, out, bn=False):
        super(Conv3BN,self).__init__()                  # The super() function returns an object that represents the parent class.
        self.conv = conv3x3(in_, out)                   # Convolution-
        self.bn = nn.BatchNorm2d(out) if bn else None   # Batch normalization is a technique for improving the speed, performance, and stability of artificial neural networks
        self.activation = nn.SELU(inplace=True)         # activation-    ?
 
    def forward(self, x):
        x = self.conv(x)                                # x first do convolution operation
        if self.bn is not None:
            x = self.bn(x)                              # then (if) do batchnormalization
        x = self.activation(x)                          # then do activation function return
        return x                                        # return x


class UNetModule(nn.Module):
    def __init__(self, in_, out):                       # [in- / out-]
        super(UNetModule,self).__init__()           
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out) 

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x                                        #exact value of x
"""

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)                            # ??????????????????????????

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        #print(x.size())
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):        #[in channel-1st/ middle channel-2nd/ out channel-3rd]
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)      
    
class UNet11(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 num_channels=3,
                ):                                             #[class- /,filter- kernel/,channel-colour channel]
        super(UNet11, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        encoder = models.vgg11(pretrained=True).features             #   ???
        self.relu = encoder[1]
        
        #self.mean = (0.485, 0.456, 0.406)
        #self.std = (0.229, 0.224, 0.225)
                
        
        # try to use 8-channels as first input
        if num_channels==3:
            self.conv1 = encoder[0]
        else:
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))            
        
        self.conv2 = encoder[3]
        self.conv3s = encoder[6]
        self.conv3 = encoder[8]
        self.conv4s = encoder[11]
        self.conv4 = encoder[13]
        self.conv5s = encoder[16]
        self.conv5 = encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        


        """self.dec1 = ConvRelu(num_filters * (2 + 1), num_classes)
        self.final = nn.ReLU() """

        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)




        self.Dropout = nn.Dropout(0.5)                                                           # ????????????????????
    """def require_encoder_grad(self,requires_grad):
        blocks = [self.conv1,
                  self.conv2,
                  self.conv3s,
                  self.conv3,
                  self.conv4s,
                  self.conv4,
                  self.conv5s,
                  self.conv5]
        
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad """
                
    def forward(self, x):
        conv1 = self.Dropout(self.relu(self.conv1(x)))
        #print(conv1.size())
        conv2 = self.Dropout(self.relu(self.conv2(self.pool(conv1))))
        #print(conv2.size())
        conv3s = self.Dropout(self.relu(self.conv3s(self.pool(conv2))))
        #print(conv3s.size())
        conv3 = self.Dropout(self.relu(self.conv3(conv3s)))
        #print(conv3.size())
        conv4s = self.Dropout(self.relu(self.conv4s(self.pool(conv3))))
        #print(conv4s.size())
        conv4 = self.Dropout(self.relu(self.conv4(conv4s)))
        #print(conv4.size())
        conv5s = self.Dropout(self.relu(self.conv5s(self.pool(conv4))))
        #print(conv5s.size())
        conv5 = self.Dropout(self.relu(self.conv5(conv5s)))
        #print(conv5.size())
        

        center = self.center(self.pool(conv5))
        #print(center.size())
        dec5 = self.dec5(torch.cat([center, conv5], 1))    # ??????????????????????????????????????
        #print(dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        #print(dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        #print(dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        #print(dec2.size())
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        #print(dec1.size())
        
        #return F.sigmoid(self.final(dec1))
        return self.final(dec1)

"""

Super--

nn.Sequential--allows you to build a neural net by specifying sequentially the building blocks (nn.Module’s) of that net.
               A sequential container. Modules will be added to it in the order they are passed in the constructor.

torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

in_channels (int) – Number of channels in the input image
out_channels (int) – Number of channels produced by the convolution
kernel_size (int or tuple) – Size of the convolving kernel
stride (int or tuple, optional) – Stride of the convolution. Default: 1
padding (int or tuple, optional) – dilation * (kernel_size - 1) - padding zero-padding will be added to both sides of each dimension in the input. Default: 0
output_padding (int or tuple, optional) – Additional size added to one side of each dimension in the output shape. Default: 0
groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1


Dropout--Dropout prevents overfitting due to too many iterations.

torch.cat--It is going to try to concatenate across dimension 2 – but dimension numbers, as tensor indexes start at 0 in PyTorch. 
Thus dim=2 refers to the 3rd dimension, and your tensors are only 2-dimensional.


"""

