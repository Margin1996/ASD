import torch
import torch.nn as nn

class CA_layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_layer, self).__init__()
        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel//reduction),
            nn.Hardswish(),
            nn.Conv2d(channel//reduction, channel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        y = self.fc(self.gap(x))
        return x*y.expand_as(x)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
 
    def forward(self, x):
        B, C, H, W = x.shape
 
        chnls_per_group = C // self.groups
        
        assert C % self.groups == 0
 
        x = x.view(B, self.groups, chnls_per_group, H, W)  # (B,C,H,W)->(B,group,C,H,W)
 
        x = torch.transpose(x, 1, 2).contiguous() 

        x = x.view(B, -1, H, W)  # (B,C,H,W)
        return x
class SACM(nn.Module):
    def __init__(self, dim, kernel_size, bias=True):
        # The term 'kernel_size' refers to the size of either the width or the height of the input.
        super(SACM, self).__init__()
        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.bk_conv_1_H = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=0, groups=dim, bias=bias)
        self.bk_conv_1_W = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=0, groups=dim, bias=bias)

        self.bk_conv_2_H = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=0, groups=dim, bias=bias)
        self.bk_conv_2_W = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=0, groups=dim, bias=bias)

        self.mixer = ChannelShuffle(groups=2)
        
        self.ca = CA_layer(dim)
        self.gama1 = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.gama2 = nn.Parameter(torch.zeros(1),requires_grad=True)
    def forward(self,x_1,x_2):
        #spatial
        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)
        x_1_res, x_2_res = x_1, x_2
        # stage 1
        x_1_1 = self.bk_conv_1_H(torch.cat((x_1,x_1[:,:,:-1,:]),dim=2))
        x_2_1 = self.bk_conv_1_W(torch.cat((x_2,x_2[:,:,:,:-1]),dim=3))
        mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
        x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        # stage 2
        x_1_2 = self.bk_conv_2_W(torch.cat((x_1_1,x_1_1[:,:,:,:-1]),dim=3))
        x_2_2 = self.bk_conv_2_H(torch.cat((x_2_1,x_2_1[:,:,:-1,:]),dim=2))

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        #channel
        x_ffn = x_1 + x_2
        x = x_ffn + self.ca(x_ffn)

        return x