import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os
import cv2
from pdb import set_trace as stx
import numbers
from einops import rearrange
from scipy.signal import convolve2d
torch.backends.cudnn.benchmark = False
import scipy.signal as signal
from einops import rearrange
import os
import glob
from skimage import exposure
from scipy.fftpack import dct, idct
from .EfficientNet import EfficientNet
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)

class fusion(nn.Module): #gConv
    def __init__(self, channel):
        channel = channel
        super(fusion, self).__init__()
        self.g1 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=9, stride=1, padding=4)
        )
        self.g2 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=5, stride=1, padding=2)
        )
        self.g3 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=7, stride=1, padding=3)
        )
        self.g4 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.g5 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        # self.g6 = ConvBR(channel,64,kernel_size=3,stride=1,padding=1)

    def forward(self, x, xa, xb):
        xa = F.interpolate(xa,size=x.size()[2:],mode='bilinear')
        xb = xa * F.interpolate(xb,size=x.size()[2:],mode='bilinear')
        
        xx = torch.cat((x, xa, xb), 1)
        xx1 = self.g1(xx)       
        xx2 = self.g2(xx)
        xx3 = self.g3(xx)
        #xx4 = self.g4(xx)
        xxx = self.g5(torch.cat((xx1, xx2, xx3), 1))
        # xxx1 = self.g6(xxx)
        # print(xxx.shape)
        return xxx
    def initialize(self):
        weight_init(self)


class Attention(nn.Module):
    def __init__(self, dim, window_size, heads, mode, bias=False):
        super(Attention, self).__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x, f_x=None):
        b,c,h,w = x.shape
        if f_x is not None:
            q=self.qkv1conv(self.qkv_0(x))
            k=self.qkv2conv(self.qkv_1(f_x))
            v=self.qkv3conv(self.qkv_2(x))
        else:
            q=self.qkv1conv(self.qkv_0(x))
            k=self.qkv2conv(self.qkv_1(x))
            v=self.qkv3conv(self.qkv_2(x))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

####################################################################################################

class EEA_head(nn.Module):
    def __init__(self, dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, size=44,
                 LayerNorm_type='WithBias',mode='train'):
        super(EEA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, window_size=size/11, heads=num_heads,mode=mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, edge_f=None):
        x = x + self.attn(self.norm1(x), edge_f)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def initialize(self):
        weight_init(self)


class EEA_module(nn.Module):
    def __init__(self, dim=64,size=44,mode='train'):
        super(EEA_module, self).__init__()
        #self.B_TA = EEA_head()
        #self.F_TA = EEA_head()
        self.TAE = EEA_head(size=size,mode=mode,dim=dim)
        self.EAE = EEA_head(size=size,mode=mode,dim=dim)
        self.TA = EEA_head(size=size,mode=mode,dim=dim)
        self.EA = EEA_head(size=size,mode=mode,dim=dim)
        self.EMA = EEA_head(size=size,mode=mode,dim=dim)
        self.EFMA = EEA_head(size=size,mode=mode,dim=dim)
        self.FIA = EEA_head(size=size,mode=mode,dim=dim * 3)
        self.Fuse = nn.Sequential(nn.Conv2d(6 * dim, 3 * dim, kernel_size=1), nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1))
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        channel = dim
        self.g1 = nn.Sequential(
            ConvBR(channel * 6, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=9, stride=1, padding=4)
        )
        self.g2 = nn.Sequential(
            ConvBR(channel * 6, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=5, stride=1, padding=2)
        )
        self.g3 = nn.Sequential(
            ConvBR(channel * 6, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=7, stride=1, padding=3)
        )
        self.g5 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, side_x, mask, edge_f1, edge_f2):
        N, C, H, W = x.shape
        #print(x.shape, side_x.shape, mask.shape, edge_f1.shape, edge_f2.shape)
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)
        #xf = self.F_TA(x, mask_d)
       # xb = self.B_TA(x, 1 - mask_d)
        #xee = self.EAE(x, mask, edge_f1)
        xte = self.TAE(x, edge_f1)
        #xme = self.EMA(x, None, edge_f2)
        xmfe = self.EFMA(x, mask * edge_f2)
        xt = self.TA(x)
        side_x = F.interpolate(side_x, size=x.size()[2:], mode='bilinear')
        #side_x = F.interpolate(side_x,size=x.size()[2:],mode='bilinear')
        x = torch.cat((xt, xte, xmfe), 1)
        x = x.view(N, 3 * C, H, W)
        # x = self.Fuse(x)
        #xx1 = self.g1(x)
        #xx2 = self.g2(x)
        #xx3 = self.g3(x)

        x = self.FIA(x)
        x = self.g5(x)
        x = side_x + side_x * x

        return x

    def initialize(self):
        weight_init(self)

class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

class Conv_Bn(nn.Module):
    def __init__(self, channels):
        super(Conv_Bn, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1):
        fuse = self.bn1(self.conv1(input1))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

def get_edge(x,y):
    x = torch.where(x < 0.8, 0.0, x.to(torch.double))
    x = torch.where(x >= 1.0, 1.0, x.to(torch.double))
    y = torch.where(y < 0.8, 0.0, y.to(torch.double))
    y = torch.where(y >= 1.0, 1.0, y.to(torch.double))
    z = x - y
    pe = z.to(torch.float32)
    # print(pe.dtype)
    pe = torch.where(pe < 0.8, 0.0, pe.to(torch.double))
    # pe = torch.where(pe >= 0.8, 1.0, pe.to(torch.double))
    pe = pe.to(torch.float32)
    return pe
class BN(nn.Module):
    def __init__(self,channel):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm1d(channel)
    def forward(self,x):
        return self.bn(x)
    def initialize(self):
        weight_init(self)
class ch_resize(nn.Module):
    def __init__(self, channel_list, channel):
        super(ch_resize, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(channel_list[0] + int(channel/2), int(channel_list[0]/4), kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(int(channel_list[0]/4), channels, kernel_size=1, stride=1, bias=False))
        self.c2 = nn.Conv2d(channel_list[1] + int(channel/2), int(channel_list[1]/4), kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(channel_list[2] + int(channel/2), int(channel_list[2]/4), kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(channel_list[3] + int(channel/2), int(channel_list[3]/4), kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(channel + int(channel/2), channel, kernel_size=3, stride=1, padding=1)

    def forward(self,x5,x4,x3,x2,x1):
        return self.c5(x5), self.c4(x4),self.c3(x3),self.c2(x2),self.c1(x1)
    def initialize(self):
        weight_init(self)

def resize_tensors(E5, E4, E3, E2, E1):
    if E5.nelement() % 11 != 0 or E5.nelement() % 12 != 0:
        E5 = F.interpolate(E5, size=[88/2,88/2], mode='bilinear')
    if E4.nelement() % 11 != 0 or E4.nelement() % 12 != 0:
        E4 = F.interpolate(E4, size=[44/2,44/2], mode='bilinear')
    if E3.nelement() % 11 != 0 or E3.nelement() % 12 != 0:
        E3 = F.interpolate(E3, size=[44/2,44/2], mode='bilinear')
    if E2.nelement() % 11 != 0 or E2.nelement() % 12 != 0:
        E2 = F.interpolate(E2, size=[88/2,88/2], mode='bilinear')
    if E1.nelement() % 11 != 0 or E1.nelement() % 12 != 0:
        E1 = F.interpolate(E1, size=[176/2,176/2], mode='bilinear')

    return E5, E4, E3, E2, E1

def save_images(tensor: torch.Tensor,index):
    assert tensor.is_cuda, "Input tensor must be on GPU"

    mean_tensor = torch.mean(tensor, dim=1, keepdim=False)

    for i in range(mean_tensor.shape[0]):
        index += 1
        image = mean_tensor[i]
        image_np = image.cpu().detach().numpy()
        image_np = (image_np * 255).astype('uint8')
        from PIL import Image
        im = Image.fromarray(image_np)
        im.save(f'image/image_{index}.png')


def get_images(tensor: torch.Tensor):

    assert tensor.is_cuda, "Input tensor must be on GPU"


    mean_tensor = torch.mean(tensor, dim=1, keepdim=False)


    for i in range(mean_tensor.shape[0]):
        image = mean_tensor[i]
        return image

class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

from scipy.signal import butter, filtfilt
from scipy.signal import tf2zpk
from scipy.signal import cheby1, lfilter
from scipy.signal import cheby1
class Decoder(nn.Module):
    def __init__(self, channels, mode, channel_list):
        super(Decoder, self).__init__()

        #self.ch_resize = ch_resize(channel_list, channels)
        
        self.side_conv1 = nn.Sequential(nn.Conv2d(channel_list[3], int(channel_list[3]/4), kernel_size=1, stride=1),
                                 nn.Conv2d(int(channel_list[3]/4), channels, kernel_size=1, stride=1, bias=False))
                                 
        self.side_conv2 = nn.Sequential(nn.Conv2d(channel_list[2], int(channel_list[2]/4), kernel_size=1, stride=1),
                                 nn.Conv2d(int(channel_list[2]/4), channels, kernel_size=1, stride=1, bias=False))
                                 
        self.side_conv3 = nn.Sequential(nn.Conv2d(channel_list[1], int(channel_list[1]/4), kernel_size=1, stride=1),
                                 nn.Conv2d(int(channel_list[1]/4), channels, kernel_size=1, stride=1, bias=False))
                                 
        self.side_conv4 = nn.Sequential(nn.Conv2d(channel_list[0], int(channel_list[0]/4), kernel_size=1, stride=1),
                                 nn.Conv2d(int(channel_list[0]/4), channels, kernel_size=1, stride=1, bias=False))

        # get state feature of ES-CRF
        self.side_conv1x = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2x = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3x = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4x = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv1n = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2n = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3n = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4n = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.fuse1 = nn.Sequential(nn.Conv2d(int(channels*2), channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse2 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse3 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse4 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse5 = nn.Sequential(nn.Conv2d(int(channels), channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))

        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        
        self.conv_block = Conv_Block(channels)
        
        self.EEA5 = EEA_module(dim=channels,size=22,mode=mode)
        self.EEA4 = EEA_module(dim=channels,size=44,mode=mode)
        self.EEA3 = EEA_module(dim=channels,size=88,mode=mode)
        self.EEA2 = EEA_module(dim=channels,size=88,mode=mode)

        self.predtrans = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        
        # transition feature function of ES-CRF
        
        self.tffx = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.tffn = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.get_edge_feature5 = fusion(channels)
        self.get_edge_feature4 = fusion(channels)
        self.get_edge_feature3 = fusion(channels)
        self.get_edge_feature2 = fusion(channels)


        self.edge_out5 = ConvBR(channels, 1, kernel_size=3, stride=1, padding=1)
        self.edge_out2 = ConvBR(channels, 1, kernel_size=3, stride=1, padding=1)
        self.edge_out3 = ConvBR(channels, 1, kernel_size=3, stride=1, padding=1)
        self.edge_out4 = ConvBR(channels, 1, kernel_size=3, stride=1, padding=1)

        # get edge posibility map from each state

        self.mkx = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True)
                                 )
        self.mkn = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True)
                                 ) 
        self.mkx0 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
                                 
        self.mkn0 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.initialize()

        #self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        #self.efficientnet2 = EfficientNet.from_pretrained('efficientnet-b0')

        #self.efficientnet3 = EfficientNet.from_pretrained('efficientnet-b0')

        #self.efficientnet4 = EfficientNet.from_pretrained('efficientnet-b0')
        
        #self.change_lvbo = nn.Conv2d(320, 128, kernel_size=1, bias=False)
        #self.change_lvbo2 = nn.Conv2d(320, 128, kernel_size=1, bias=False)
        #self.change_lvbo3 = nn.Conv2d(320, 128, kernel_size=1, bias=False)
        #self.change_lvbo4 = nn.Conv2d(320, 128, kernel_size=1, bias=False)
 
    def lvbo1(self, D, original_img, shape):
        original_img = original_img * 255
        filtered_images = []
        D = F.interpolate(D,size=shape,mode='bilinear')
    
        original_img = torch.mean(original_img, dim=1, keepdim=True)
        for i in range(D.size(0)):
            D_sum = D[i]
            D_sum = (D_sum - D_sum.min()) / (D_sum.max() - D_sum.min() + 1e-10)
            D_resized = D_sum
            high_val_points = torch.stack((D_resized >= 0.5).nonzero(as_tuple=True))
            if high_val_points.shape[1] == 0:
                img_back = torch.zeros_like(original_img[i])
                filtered_images.append(img_back)
                print("high_val_points.shape[1] == 0",torch.max(D_resized))
                continue
            top = torch.min(high_val_points[1])
            bottom = torch.max(high_val_points[1])
            left = torch.min(high_val_points[2])
            right = torch.max(high_val_points[2])
            size = max(bottom-top, right-left)
            center_y = (top + bottom) // 2
            center_x = (left + right) // 2
            top = center_y - size // 2
            bottom = top + size
            left = center_x - size // 2
            right = left + size
            window = (original_img)[i, :, max(0, top):min(shape[0], bottom), max(0, left):min(shape[1], right)] 
            window = window.float()
            if window.numel() <= 0:
                img_back = torch.zeros_like(original_img[i])
                filtered_images.append(img_back)
                print("window.numel() <= 0")
                continue
            window_resized = F.interpolate(window.unsqueeze(0), size=shape, mode='bilinear').squeeze()
            f_window = np.fft.fft2(window_resized.cpu().detach().numpy())
            fshift_window = np.fft.fftshift(f_window)
            
            # Binarize the window's frequency domain with a threshold of 0.1
            fshift_window = (np.abs(fshift_window) >= 0.1).astype(np.float32)
            
            f = np.fft.fft2(original_img[i].cpu().detach().numpy())
            fshift = np.fft.fftshift(f)
            
            # Create a mask with the same size as the image
            mask = np.zeros_like(fshift)
            
            # Set the pixels within the window to the corresponding values in the window's frequency domain
            mask = fshift_window
            
            # Apply the mask to the image in the frequency domain
            fshift_masked = fshift * mask
            
            f_ishift = np.fft.ifftshift(fshift_masked)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back) + 1e-8)
            filtered_images.append(torch.from_numpy(img_back).cuda())
        filtered_images = torch.stack(filtered_images, dim=0).float().cuda()
        return filtered_images

            
    def lvbo(self, D, original_img, shape):
        b, g, r = original_img[:,0:1,:,:],original_img[:,1:2,:,:],original_img[:,2:3,:,:]
        #print(b.shape)
        b_filtered = self.lvbo1(D, b, shape)
        g_filtered = self.lvbo1(D, g, shape)
        r_filtered = self.lvbo1(D, r, shape)
        # Concatenate the filtered channels into a three-channel image
        filtered_image = torch.cat((b_filtered, g_filtered, r_filtered), dim=1)
        # Sum the channels into a one-channel image
        filtered_image = torch.stack((filtered_image,), dim=1).squeeze(1)
        filtered_image = filtered_image.mean(dim=1, keepdim=True)
        #filtered_image = edge_f = torch.zeros_like(filtered_image).cuda()
        return filtered_image
        
    def lvbo1_DI(self, D, original_img, shape):
        original_img = original_img * 255
        filtered_images = []
        D = F.interpolate(D,size=shape,mode='bilinear')
    
        original_img = torch.mean(original_img, dim=1, keepdim=True)
        for i in range(D.size(0)):
            D_sum = D[i]
            D_sum = (D_sum - D_sum.min()) / (D_sum.max() - D_sum.min() + 1e-10)
            D_resized = D_sum
            high_val_points = torch.stack((D_resized >= 0.5).nonzero(as_tuple=True))
            if high_val_points.shape[1] == 0:
                img_back = torch.zeros_like(original_img[i])
                filtered_images.append(img_back)
                print("high_val_points.shape[1] == 0",torch.max(D_resized))
                continue
            top = torch.min(high_val_points[1])
            bottom = torch.max(high_val_points[1])
            left = torch.min(high_val_points[2])
            right = torch.max(high_val_points[2])
            size = max(bottom-top, right-left)
            center_y = (top + bottom) // 2
            center_x = (left + right) // 2
            top = center_y - size // 2
            bottom = top + size
            left = center_x - size // 2
            right = left + size
            window = (original_img)[i, :, max(0, top):min(shape[0], bottom), max(0, left):min(shape[1], right)] 
            window = window.float()
            if window.numel() <= 0:
                img_back = torch.zeros_like(original_img[i])
                filtered_images.append(img_back)
                print("window.numel() <= 0")
                continue
            window_resized = F.interpolate(window.unsqueeze(0), size=shape, mode='bilinear').squeeze()
            f_window = np.fft.fft2(window_resized.cpu().detach().numpy())
            fshift_window = np.fft.fftshift(f_window)
            
            # Create a low-pass filter in the frequency domain
            rows, cols = fshift_window.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.uint8)
            r = 0.1 * crow  # I choose 0.1 as the cut-off frequency
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask[mask_area] = 1
            
            # Apply the low-pass filter to the window's frequency domain
            fshift_window = fshift_window * mask
            
            f = np.fft.fft2(original_img[i].cpu().detach().numpy())
            fshift = np.fft.fftshift(f)
            
            # Apply the mask to the image in the frequency domain
            fshift_masked = fshift * fshift_window
            
            f_ishift = np.fft.ifftshift(fshift_masked)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back) + 1e-8)
            filtered_images.append(torch.from_numpy(img_back).cuda())
        filtered_images = torch.stack(filtered_images, dim=0).float().cuda()
        return filtered_images
    
    
    def lvbo_DI(self, D, original_img, shape):
        b, g, r = original_img[:,0:1,:,:],original_img[:,1:2,:,:],original_img[:,2:3,:,:]
        #print(b.shape)
        b_filtered = self.lvbo1(D, b, shape)
        g_filtered = self.lvbo1(D, g, shape)
        r_filtered = self.lvbo1(D, r, shape)
        # Concatenate the filtered channels into a three-channel image
        filtered_image = torch.cat((b_filtered, g_filtered, r_filtered), dim=1)
        # Sum the channels into a one-channel image
        filtered_image = torch.stack((filtered_image,), dim=1).squeeze(1)
        filtered_image = filtered_image.mean(dim=1, keepdim=True)
        #filtered_image = edge_f = torch.zeros_like(filtered_image).cuda()
        return filtered_image

    def lvbo1_GAO(self, D, original_img, shape):
        original_img = original_img * 255
        filtered_images = []
        D = F.interpolate(D,size=shape,mode='bilinear')
    
        original_img = torch.mean(original_img, dim=1, keepdim=True)
        for i in range(D.size(0)):
            D_sum = D[i]
            D_sum = (D_sum - D_sum.min()) / (D_sum.max() - D_sum.min() + 1e-10)
            D_resized = D_sum
            high_val_points = torch.stack((D_resized >= 0.5).nonzero(as_tuple=True))
            if high_val_points.shape[1] == 0:
                img_back = torch.zeros_like(original_img[i])
                filtered_images.append(img_back)
                print("high_val_points.shape[1] == 0",torch.max(D_resized))
                continue
            top = torch.min(high_val_points[1])
            bottom = torch.max(high_val_points[1])
            left = torch.min(high_val_points[2])
            right = torch.max(high_val_points[2])
            size = max(bottom-top, right-left)
            center_y = (top + bottom) // 2
            center_x = (left + right) // 2
            top = center_y - size // 2
            bottom = top + size
            left = center_x - size // 2
            right = left + size
            window = (original_img)[i, :, max(0, top):min(shape[0], bottom), max(0, left):min(shape[1], right)] 
            window = window.float()
            if window.numel() <= 0:
                img_back = torch.zeros_like(original_img[i])
                filtered_images.append(img_back)
                print("window.numel() <= 0")
                continue
            window_resized = F.interpolate(window.unsqueeze(0), size=shape, mode='bilinear').squeeze()
            f_window = np.fft.fft2(window_resized.cpu().detach().numpy())
            fshift_window = np.fft.fftshift(f_window)
            
            # Create a high-pass filter in the frequency domain
            rows, cols = fshift_window.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.ones((rows, cols), np.uint8)
            r = 0.1 * crow  # I choose 0.1 as the cut-off frequency
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask[mask_area] = 0
            
            # Apply the high-pass filter to the window's frequency domain
            fshift_window = fshift_window * mask
            
            f = np.fft.fft2(original_img[i].cpu().detach().numpy())
            fshift = np.fft.fftshift(f)
            
            # Apply the mask to the image in the frequency domain
            fshift_masked = fshift * fshift_window
            
            f_ishift = np.fft.ifftshift(fshift_masked)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back) + 1e-8)
            filtered_images.append(torch.from_numpy(img_back).cuda())
        filtered_images = torch.stack(filtered_images, dim=0).float().cuda()
        return filtered_images
    
    
    def lvbo_GAO(self, D, original_img, shape):
        b, g, r = original_img[:,0:1,:,:],original_img[:,1:2,:,:],original_img[:,2:3,:,:]
        #print(b.shape)
        b_filtered = self.lvbo1(D, b, shape)
        g_filtered = self.lvbo1(D, g, shape)
        r_filtered = self.lvbo1(D, r, shape)
        # Concatenate the filtered channels into a three-channel image
        filtered_image = torch.cat((b_filtered, g_filtered, r_filtered), dim=1)
        # Sum the channels into a one-channel image
        filtered_image = torch.stack((filtered_image,), dim=1).squeeze(1)
        filtered_image = filtered_image.mean(dim=1, keepdim=True)
        #filtered_image = edge_f = torch.zeros_like(filtered_image).cuda()
        return filtered_image

    def lvbo1_DAI(self, D, original_img, shape):
        original_img = original_img * 255
        filtered_images = []
        D = F.interpolate(D,size=shape,mode='bilinear')
    
        original_img = torch.mean(original_img, dim=1, keepdim=True)
        for i in range(D.size(0)):
            D_sum = D[i]
            D_sum = (D_sum - D_sum.min()) / (D_sum.max() - D_sum.min() + 1e-10)
            D_resized = D_sum
            high_val_points = torch.stack((D_resized >= 0.5).nonzero(as_tuple=True))
            if high_val_points.shape[1] == 0:
                img_back = torch.zeros_like(original_img[i])
                filtered_images.append(img_back)
                print("high_val_points.shape[1] == 0",torch.max(D_resized))
                continue
            top = torch.min(high_val_points[1])
            bottom = torch.max(high_val_points[1])
            left = torch.min(high_val_points[2])
            right = torch.max(high_val_points[2])
            size = max(bottom-top, right-left)
            center_y = (top + bottom) // 2
            center_x = (left + right) // 2
            top = center_y - size // 2
            bottom = top + size
            left = center_x - size // 2
            right = left + size
            window = (original_img)[i, :, max(0, top):min(shape[0], bottom), max(0, left):min(shape[1], right)] 
            window = window.float()
            if window.numel() <= 0:
                img_back = torch.zeros_like(original_img[i])
                filtered_images.append(img_back)
                print("window.numel() <= 0")
                continue
            window_resized = F.interpolate(window.unsqueeze(0), size=shape, mode='bilinear').squeeze()
            f_window = np.fft.fft2(window_resized.cpu().detach().numpy())
            fshift_window = np.fft.fftshift(f_window)
            
            # Create a band-pass filter in the frequency domain
            rows, cols = fshift_window.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.uint8)
            r_out = 0.2 * crow  # Outer radius of the band-pass filter
            r_in = 0.1 * crow  # Inner radius of the band-pass filter
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = np.logical_and((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in**2, 
                                       (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out**2)
            mask[mask_area] = 1
            
            # Apply the band-pass filter to the window's frequency domain
            fshift_window = fshift_window * mask
            
            f = np.fft.fft2(original_img[i].cpu().detach().numpy())
            fshift = np.fft.fftshift(f)
            
            # Apply the mask to the image in the frequency domain
            fshift_masked = fshift * fshift_window
            
            f_ishift = np.fft.ifftshift(fshift_masked)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back) + 1e-8)
            filtered_images.append(torch.from_numpy(img_back).cuda())
        filtered_images = torch.stack(filtered_images, dim=0).float().cuda()
        return filtered_images

    
    
    def lvbo_DAI(self, D, original_img, shape):
        b, g, r = original_img[:,0:1,:,:],original_img[:,1:2,:,:],original_img[:,2:3,:,:]
        #print(b.shape)
        b_filtered = self.lvbo1(D, b, shape)
        g_filtered = self.lvbo1(D, g, shape)
        r_filtered = self.lvbo1(D, r, shape)
        # Concatenate the filtered channels into a three-channel image
        filtered_image = torch.cat((b_filtered, g_filtered, r_filtered), dim=1)
        # Sum the channels into a one-channel image
        filtered_image = torch.stack((filtered_image,), dim=1).squeeze(1)
        filtered_image = filtered_image.mean(dim=1, keepdim=True)
        #filtered_image = edge_f = torch.zeros_like(filtered_image).cuda()
        return filtered_image


    def forward(self, E4, E3, E2, E1, shape, original_img):
        E4, E3, E2, E1= self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)
        E4_o, E3_o, E2_o,E1_o = E4, E3, E2, E1
        if E3.size()[2:] != E2.size()[2:]:
            E31 = F.interpolate(E3, size=E2.size()[2:], mode='bilinear')
        if E4.size()[2:] != E2.size()[2:]:
            E41 = F.interpolate(E4, size=E2.size()[2:], mode='bilinear')

        E5 = self.conv_block(E41, E31, E2)

        E4 = torch.cat((E41, E5),1)
        E3 = torch.cat((E31, E5),1)
        E2 = torch.cat((E2, E5),1)

        E4 = F.relu(self.fuse1(E4), inplace=True)
        E3 = F.relu(self.fuse2(E3), inplace=True)
        E2 = F.relu(self.fuse3(E2), inplace=True)
        
        #state1
        # print(E5.shape,E4.shape,E3.shape,E2.shape)
        P5 = self.predtrans(E5)
        P51 = P5
        L51 = self.lvbo(P51,original_img,shape)
        
        
        Dmax5 = self.side_conv4x(E5)
        Pmax5 = self.tffx(Dmax5)
        Dmin5 = self.side_conv4n(E5)
        Pmin5 = self.tffn(Dmin5)
        Dmax5mk = self.mkx(F.interpolate(Dmax5,size=[88,88],mode='bilinear'))
        Dmin5mk = self.mkn(F.interpolate(Dmin5,size=[88,88],mode='bilinear'))
        E5e = get_edge(Pmax5,Pmin5)
        P5e = E5e
        D5e = Dmax5 - Dmin5
        E5e1 = E5e.detach()
        D5e = self.get_edge_feature5(D5e,E5, E5e1)
        E5e = self.edge_out5(F.interpolate(D5e,size=shape,mode='bilinear'))
        D5e = F.interpolate(D5e, size=(22,22), mode='bilinear')
        E5e1 = F.interpolate(E5e1, size=(22,22), mode='bilinear')
        #L52 = self.efficientnet.extract_endpoints(L51)['reduction_5']
        L5 = F.interpolate(L51,size=(22,22),mode='bilinear')
        #L5 = self.change_lvbo(L5)
        E5 = F.interpolate(E5, size=(22,22), mode='bilinear')
        E4 = F.interpolate(E4, size=(22,22), mode='bilinear')
        D4 = self.EEA5(E5, E4, E5e1, E5*L5, D5e)
        D4 = F.interpolate(D4, size=(22,22), mode='bilinear')
        P4 = self.predtrans(D4)
        P41 = P4
        #state2
        L41 = self.lvbo(P41,original_img,shape)
        #L42 = self.efficientnet.extract_endpoints(L41)['reduction_5']
        L4 = F.interpolate(L41,size=(44,44),mode='bilinear')
        #L4 = self.change_lvbo(L4)
        D4 = F.interpolate(D4, size=(44,44), mode='bilinear')
        E3 = F.interpolate(E3, size=(44,44), mode='bilinear')
        
        Dmax4 = self.side_conv3x(D4)
        Dmax5 = torch.cat([F.interpolate(Dmax4,size=Dmax5mk.size()[2:]),self.mkx0(Dmax5mk)],dim=1)
        Dmax4 = self.fuse(Dmax5)
        Pmax4 = self.tffx(Dmax4)
        Dmin4 = self.side_conv3n(Dmin5)
        Dmin5 = torch.cat([F.interpolate(Dmin4,size=Dmin5mk.size()[2:]),self.mkn0(Dmin5mk)],dim=1)
        Dmin4 = self.fuse(Dmin5)
        Pmin4 = self.tffn(Dmin4)
        Dmax4mk = self.mkx(F.interpolate(Dmax4,size=[88,88],mode='bilinear'))
        Dmin4mk = self.mkn(F.interpolate(Dmin4,size=[88,88],mode='bilinear'))
        E4e = get_edge(Pmax4,Pmin4)
        P4e = E4e
        D4e = Dmax4 - Dmin4
        E4e1 = E4e.detach()
        D5e = F.interpolate(D5e, size=D4e.size()[2:], mode='bilinear')
        D4e = self.get_edge_feature4(D4e, D4, E4e1)
        E4e = self.edge_out4(F.interpolate(D4e,size=shape,mode='bilinear'))
        D4e = F.interpolate(D4e,size=(44,44),mode='bilinear')
        E4e1 = F.interpolate(E4e1, size=(44,44), mode='bilinear')
        D3 = self.EEA4(D4, E3, E4e1, D4*L4, D4e)
        D3 = F.interpolate(D3, size=(44,44), mode='bilinear')
        P3 = self.predtrans(D3)
        P31 = P3
        
        #state3
        L31 = self.lvbo(P3,original_img,shape)
        #L32 = self.efficientnet.extract_endpoints(L31)['reduction_5']
        L3 = F.interpolate(L31,size=(88,88),mode='bilinear')
        #L3 = self.change_lvbo(L3)
        D3 = F.interpolate(D3, size=(88,88), mode='bilinear')
        E2 = F.interpolate(E2, size=(88,88), mode='bilinear')
        
        Dmax3 = self.side_conv2x(D3)
        Dmax4 = torch.cat([F.interpolate(Dmax3,size=Dmax4mk.size()[2:]),self.mkx0(Dmax4mk)],dim=1)
        Dmax3 = self.fuse(Dmax4)
        Pmax3 = self.tffx(Dmax3)
        Dmin3 = self.side_conv2n(Dmin4)
        Dmin4 = torch.cat([F.interpolate(Dmin3,size=Dmin4mk.size()[2:]),self.mkn0(Dmin4mk)],dim=1)
        Dmin3 = self.fuse(Dmin4)
        Pmin3 = self.tffn(Dmin3)
        Dmax3mk = self.mkx(F.interpolate(Dmax3, size=[88,88], mode='bilinear'))
        Dmin3mk = self.mkn(F.interpolate(Dmin3, size=[88,88], mode='bilinear'))
        E3e = get_edge(Pmax3,Pmin3)
        P3e = E3e
        D3e = Dmax3 - Dmin3
        E3e1 = E3e.detach()
        D4e = F.interpolate(D4e, size=D3e.size()[2:], mode='bilinear')
        D3e = self.get_edge_feature3(D3e, D3, E3e1)
        E3e = self.edge_out3(F.interpolate(D3e,size=(88,88),mode='bilinear'))
        D3e = F.interpolate(D3e,size=(88,88),mode='bilinear')
        E3e1 = F.interpolate(E3e1, size=(88,88), mode='bilinear')
        D2 = self.EEA3(D3, E2, E3e1, D3*L3, D3e)
        D2 = F.interpolate(D2, size=(88,88), mode='bilinear')
        P2 = self.predtrans(D2)
        P21 = P2
        
        #state4
        E1 = F.interpolate(E1,size=(88,88),mode='bilinear')
        L21 = self.lvbo(P2,original_img,shape)
        #L22 = self.efficientnet.extract_endpoints(L21)['reduction_5']
        L2 = F.interpolate(L21,size=(88,88),mode='bilinear')
        #L2 = self.change_lvbo(L2)
        D2 = F.interpolate(D2, size=(88,88), mode='bilinear')
        
        Dmax2 = self.side_conv1x(D2)
        Dmax3 = torch.cat([F.interpolate(Dmax2,size=Dmax3mk.size()[2:]),self.mkx0(Dmax3mk)],dim=1)
        Dmax2 = self.fuse(Dmax3)
        Pmax2 = self.tffx(Dmax2)
        Dmin2 = self.side_conv1n(Dmin3)
        Dmin3 = torch.cat([F.interpolate(Dmin2,size=Dmin3mk.size()[2:]),self.mkn0(Dmin3mk)],dim=1)
        Dmin2 = self.fuse(Dmin3)
        Pmin2 = self.tffn(Dmin2)
        E2e = get_edge(Pmax2,Pmin2)
        P2e = E2e
        D2e = Dmax2 - Dmin2
        E2e1 = E2e.detach()
        D3e = F.interpolate(D3e, size=D2e.size()[2:], mode='bilinear')
        D2e = self.get_edge_feature2(D2e, D2, E2e1)
        E2e = self.edge_out2(F.interpolate(D2e,size=(88,88),mode='bilinear'))
        E1 = F.interpolate(E1,size=(88,88),mode='bilinear')
        E2e1 = F.interpolate(E2e1, size=(88,88), mode='bilinear')
        D1 = self.EEA2(D2, E1, E2e1, D2*L2, D2e)
        P1 = self.predtrans(D1)
        #print(P5.shape,P4.shape,P3.shape,P2.shape,P1.shape)


        P1 = F.interpolate(P1, size=shape, mode='bilinear')
        P2 = F.interpolate(P2, size=shape, mode='bilinear')
        P3 = F.interpolate(P3, size=shape, mode='bilinear')
        P4 = F.interpolate(P4, size=shape, mode='bilinear')
        P5 = F.interpolate(P5, size=shape, mode='bilinear')

        P1 = torch.squeeze(P1,2)
        P2 = torch.squeeze(P2,2)
        P3 = torch.squeeze(P3,2)
        P4 = torch.squeeze(P4,2)
        P5 = torch.squeeze(P5,2)

        Pmax2 = F.interpolate(Pmax2, size=shape, mode='bilinear')
        Pmax3 = F.interpolate(Pmax3, size=shape, mode='bilinear')
        Pmax4 = F.interpolate(Pmax4, size=shape, mode='bilinear')
        Pmax5 = F.interpolate(Pmax5, size=shape, mode='bilinear')

        Pmin2 = F.interpolate(Pmin2, size=shape, mode='bilinear')
        Pmin3 = F.interpolate(Pmin3, size=shape, mode='bilinear')
        Pmin4 = F.interpolate(Pmin4, size=shape, mode='bilinear')
        Pmin5 = F.interpolate(Pmin5, size=shape, mode='bilinear')

        E5e = F.interpolate(E5e, size=shape, mode='bilinear')
        E4e = F.interpolate(E4e, size=shape, mode='bilinear')
        E3e = F.interpolate(E3e, size=shape, mode='bilinear')
        E2e = F.interpolate(E2e, size=shape, mode='bilinear')

        P5e = F.interpolate(P5e, size=shape, mode='bilinear')
        P4e = F.interpolate(P4e, size=shape, mode='bilinear')
        P3e = F.interpolate(P3e, size=shape, mode='bilinear')
        P2e = F.interpolate(P2e, size=shape, mode='bilinear')


        F1 = get_images(D1)

        F2 = get_images(D2)

        F3 = get_images(D3)

        F4 = get_images(D4)

        return [P5, P4, P3, P2, P1], [Pmax5, Pmax4, Pmax3, Pmax2], [Pmin5, Pmin4, Pmin3, Pmin2], [E5e, E4e, E3e, E2e], [P5e, P4e, P3e, P2e], [F1, F2, F3, F4], [L21,L31,L41,L51]

    def initialize(self):
        weight_init(self)

 
