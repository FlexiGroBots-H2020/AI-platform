import torchvision
import torch
import torch.nn as nn





class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(out_channels)) #,nn.Dropout())

    def forward(self, x):

        x = self.double_conv(x)
        return x



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, height, width))

    def forward(self, x):

        x = self.maxpool_conv(x)
        return x



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, height, width)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        #x2 = self.crop(x2, x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)

        return x

    def crop(self, enc_ftrs, x):

        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)

        return enc_ftrs

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x_conv = self.conv(x)
        #x_prob = self.softmax(x_conv)

        return x_conv


### 32 REZOLUCIJA
class UNet3(nn.Module):
    def __init__(self,  n_channels, n_classes, height, width,zscore):
        super(UNet3, self).__init__()
        self.zscore = zscore
        self.normalization = nn.InstanceNorm2d(n_channels,affine=True)
        # povecan broj ulaznih kanala sa 16 na 32
        # self.inc = DoubleConv(n_channels, 16, height, width)
        # self.down1 = Down(16, 24, int(height / 2), int(width / 2)) # 256x
        # self.down2 = Down(24, 32, int(height / 4), int(width / 4)) # 128x
        # self.down3 = Down(32, 48, int(height / 8), int(width / 8)) # 64x
        # self.up1 = Up(80, 32, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(56, 24, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(40, 16, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(16, n_classes)
############################################
        # self.inc = DoubleConv(n_channels, 16, height, width)
        # self.down1 = Down(16, 32, int(height / 2), int(width / 2)) # 256x
        # self.down2 = Down(32, 48, int(height / 4), int(width / 4)) # 128x
        # self.down3 = Down(48, 64, int(height / 8), int(width / 8)) # 64x
        # self.up1 = Up(112, 48, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(80, 32, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(48, 16, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(16, n_classes)
############################################

        self.inc = DoubleConv(n_channels, 16, height, width)
        self.down1 = Down(16, 32, int(height / 2), int(width / 2)) # 256x
        self.down2 = Down(32, 64, int(height / 4), int(width / 4)) # 128x
        self.down3 = Down(64, 128, int(height / 8), int(width / 8)) # 64x
        self.up1 = Up(192, 64, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(96, 32, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(48, 16, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(16, n_classes)

#############################################

        # self.inc = DoubleConv(n_channels, 32, height, width)
        # self.down1 = Down(32, 64, int(height / 2), int(width / 2)) # 256x
        # self.down2 = Down(64, 128, int(height / 4), int(width / 4)) # 128x
        # self.down3 = Down(128, 256, int(height / 8), int(width / 8)) # 64x
        # self.up1 = Up(384, 128, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(192, 64, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(96, 32, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(32, n_classes)


    def forward(self, x):
        if self.zscore == 0:
            x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x



### 32 REZOLUCIJA
class UNet3_modified(nn.Module):
    def __init__(self,  n_channels, n_classes, height, width, no_indices):
        super(UNet3_modified, self).__init__()
        ##########################################################################
        self.numerator = nn.Conv2d(n_channels, no_indices, kernel_size=1)  #
        self.denominator = nn.Conv2d(n_channels, no_indices, kernel_size=1)  #
        ##########################################################################

        self.normalization = nn.InstanceNorm2d(n_channels + no_indices,affine=True)
        self.inc = DoubleConv(n_channels + no_indices, 16, height, width)
        self.down1 = Down(16, 24, int(height / 2), int(width / 2)) # 256x
        self.down2 = Down(24, 32, int(height / 4), int(width / 4)) # 128x
        self.down3 = Down(32, 48, int(height / 8), int(width / 8)) # 64x
        self.up1 = Up(80, 32, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(56, 24, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(40, 16, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(16, n_classes)

        # self.normalization = nn.InstanceNorm2d(n_channels + no_indices,affine=True)
        # self.inc = DoubleConv(n_channels + no_indices, 16+16, height, width)
        # self.down1 = Down(16+16, 24+16, int(height / 2), int(width / 2)) # 256x
        # self.down2 = Down(24+16, 32+16, int(height / 4), int(width / 4)) # 128x
        # self.down3 = Down(32+16, 48+16, int(height / 8), int(width / 8)) # 64x
        # self.up1 = Up(80+32, 32+16, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(56+32, 24+16, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(40+32, 16+16, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(16+16, n_classes)

        # self.normalization = nn.InstanceNorm2d(n_channels + no_indices,affine=True)
        # self.inc = DoubleConv(n_channels + no_indices, 16*2, height, width)
        # self.down1 = Down(16*2, 24*2, int(height / 2), int(width / 2)) # 256x
        # self.down2 = Down(24*2, 32*2, int(height / 4), int(width / 4)) # 128x
        # self.down3 = Down(32*2, 48*2, int(height / 8), int(width / 8)) # 64x
        # self.up1 = Up(80*2, 32*2, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(56*2, 24*2, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(40*2, 16*2, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(16*2, n_classes)



    def forward(self, x):
        ##################################################################################
        num = self.numerator(x)  #
        denom = self.denominator(x)  #
        lvi = torch.div(num, denom + 0.000000001)  #
        fin = torch.cat((x, lvi), 1)  #
        x = self.normalization(fin)  #
        # x = self.normalization(x)
        ##################################################################################
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x





### 32 REZOLUCIJA
class UNetO(nn.Module):
    def __init__(self,  n_channels, n_classes, height, width):
        super(UNet3, self).__init__()

        self.inc = DoubleConv(n_channels, 64, height, width)
        self.down1 = Down(64, 128, int(height / 2), int(width / 2)) # 256x
        self.down2 = Down(128, 256, int(height / 4), int(width / 4)) # 128x
        self.down3 = Down(256,512, int(height / 8), int(width / 8)) # 64x
        self.up1 = Up(768, 256, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(384, 128, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(192, 64, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x


# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         init.xavier_uniform(m.weight, gain=numpy.sqrt(2.0))
#         init.constant(m.bias, 0.1)
