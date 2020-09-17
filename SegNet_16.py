import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F



def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)            #[in-input image/ out-output image/ 3-kernel size/ zero padding]  


class Conv3BN(nn.Module):
    def __init__(self, in_, out, bn=False):
        super(Conv3BN,self).__init__()                  # The super() function returns an object that represents the parent class.
        self.conv = conv3x3(in_, out)                   # Convolution-
        self.bn = nn.BatchNorm2d(out) if bn else None   # Batch normalization is a technique for improving the speed, performance, and stability of artificial neural networks
        self.activation = nn.ReLU(inplace=True)         # activation-    ?
 
    def forward(self, x):
        x = self.conv(x)                                # x first do convolution operation
        if self.bn is not None:
            x = self.bn(x)                              # then (if) do batchnormalization
        x = self.activation(x)                          # then do activation function return
        return x                                        # return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):        #[in channel-1st/ middle channel-2nd/ out channel-3rd]
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv3BN(middle_channels, out_channels))

    def forward(self, x):
        return self.block(x)      
 

class SegNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_channels=3,
                ):                                             #[class- /,filter- kernel/,channel-colour channel]
        super(SegNet, self).__init__()
        encoder = models.vgg16(pretrained=True).features  
        self.pool = nn.MaxPool2d(2, 2)

        self.bn64=nn.BatchNorm2d(64) 
        self.bn128=nn.BatchNorm2d(128)
        self.bn256=nn.BatchNorm2d(256)
        self.bn512=nn.BatchNorm2d(512)

        #self.relu = F.relu(inplace=True)
                
        
        # try to use 8-channels as first input
        self.enc11 = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))            
        self.enc12 = encoder[2]

        self.enc21 = encoder[5]
        self.enc22 = encoder[7]

        self.enc31 = encoder[10]
        self.enc32 = encoder[12]
        self.enc33 = encoder[14]

        self.enc41 = encoder[17]
        self.enc42 = encoder[19]
        self.enc43 = encoder[21]

        self.enc51 = encoder[24]
        self.enc52 = encoder[26]
        self.enc53 = encoder[28]


        self.dec53 = DecoderBlock(512, 512, 512)
        self.dec52=Conv3BN(512,512)
        self.dec51=Conv3BN(512,512)

        self.dec43 = DecoderBlock(512, 512, 512)
        self.dec42=Conv3BN(512,512)
        self.dec41=Conv3BN(512,256)

        self.dec33 = DecoderBlock(256, 256, 256)
        self.dec32=Conv3BN(256,256)
        self.dec31=Conv3BN(256,128)

        self.dec22 = DecoderBlock(128, 128, 128)
        self.dec21=Conv3BN(128,64)

        self.dec12 = DecoderBlock(64, 64, 64)
        self.dec11=nn.Conv2d(64, num_classes, kernel_size=1) 

                
    def forward(self, x):
        enc11= (F.relu(self.bn64(self.enc11(x))))
        enc12= (F.relu(self.bn64(self.enc12(enc11))))

        enc21= (F.relu(self.bn128(self.enc21(self.pool(enc12)))))
        enc22= (F.relu(self.bn128(self.enc22(enc21))))

        enc31= (F.relu(self.bn256(self.enc31(self.pool(enc22)))))
        enc32= (F.relu(self.bn256(self.enc32(enc31))))
        enc33= (F.relu(self.bn256(self.enc33(enc32))))

        enc41= (F.relu(self.bn512(self.enc41(self.pool(enc33)))))
        enc42= (F.relu(self.bn512(self.enc42(enc41))))
        enc43= (F.relu(self.bn512(self.enc42(enc42))))

        enc51= (F.relu(self.bn512(self.enc51(self.pool(enc43)))))
        enc52= (F.relu(self.bn512(self.enc42(enc51))))
        enc53= (F.relu(self.bn512(self.enc42(enc52))))

        dec53=(self.dec53(self.pool(enc53)))
        dec52=(self.dec52(dec53))
        dec51=(self.dec51(dec52))

        dec43=(self.dec43(dec51))
        dec42=(self.dec42(dec43))
        dec41=(self.dec41(dec42))

        dec33=(self.dec33(dec41))
        dec32=(self.dec32(dec33))
        dec31=(self.dec31(dec32))

        dec22=(self.dec22(dec31))
        dec21=(self.dec21(dec22))

        dec12=(self.dec12(dec21))
        dec11=(self.dec11(dec12))
        
        #return F.sigmoid(dec11)
        return dec11

