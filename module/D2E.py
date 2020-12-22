import torch
import torch.nn as nn 
from torch.nn import init
from torch.nn.parameter import Parameter
import sys
sys.path.append('../')
import module.lreq as ln
from module.net import Blur,FromRGB
from torch.nn import functional as F

# 原D改E
class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False, fused_scale=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv_1 = ln.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = Blur(inputs)
        self.last = last
        self.fused_scale = fused_scale
        self.dense = ln.Linear(inputs, 512) #全连接
        self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True) # fused_scale
        #self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
            #print(x.shape)
            x = self.conv_1(x) + self.bias_1
            x = F.leaky_relu(x, 0.2)
            w1 = self.dense(x.view(x.shape[0], -1))
            w1 = F.leaky_relu(x1, 0.2)
            if not last:
                x = self.conv_2(self.blur(x))
                x = F.leaky_relu(x, 0.2)
            w2 = self.dense(x.view(x.shape[0], -1))
            w2 = F.leaky_relu(w2, 0.2)
            return x,w1,w2

class Discriminator(nn.Module):
    def __init__(self, startf=16, maxf=512, layer_count=9, channels=3):
        super(Discriminator, self).__init__()
        self.maxf = maxf # 512
        self.startf = startf # START_CHANNEL_COUNT: 16
        self.layer_count = layer_count # 9
        self.from_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        w = []
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)
            self.from_rgb.append(FromRGB(channels, inputs))
            fused_scale = resolution >= 128
            block = DiscriminatorBlock(inputs, outputs, i == self.layer_count - 1, fused_scale=fused_scale)
            resolution //= 2
            #print("encode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def forward(self, x, lod=8, blend=1): # 1024 -> lod = 8
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        for i in range(self.layer_count - lod - 1, self.layer_count): #range(0,9)
            print(i)
            x,x1,x2 = self.encode_block[i](x)
            w.append(x1)
            w.append(x2)
        return w


dense:
256 32
512 64
1024 128
2048 256
4096 512
8192 512
8192 512
8192 512
8192 512




#import torch
#import module.D2E as E
if __name__ == "__main__":
    e = Discriminator()
    x = torch.randn(1,3,1024,1024)
    w = e(x)
