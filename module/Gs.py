import torch
import torch.nn as nn 
from torch.nn import init
from torch.nn.parameter import Parameter
from dlutils.pytorch import count_parameters, millify
from module.net import DecodeBlock, ToRGB

class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels
        self.latent_size = latent_size

        mul = 2**(self.layer_count-1) # mul =  4, 8, 16, 32 ... | layer_count=6 -> 128*128

        inputs = min(self.maxf, startf * mul)
        self.const = Parameter(torch.Tensor(1, inputs, 4, 4)) #[1,512,4,4]
        self.zeros = torch.zeros(1, 1, 1, 1)
        init.ones_(self.const)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []
        to_rgb = nn.ModuleList()
        self.decode_block: nn.ModuleList[DecodeBlock] = nn.ModuleList()

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0
            fused_scale = resolution * 2 >= 128

            block = DecodeBlock(inputs, outputs, latent_size, has_first_conv, fused_scale=fused_scale) 

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            to_rgb.append(ToRGB(outputs, channels))

            #print("decode_block%d %s styles in: %dl out resolution: %d" % ((i + 1), millify(count_parameters(block)), outputs, resolution)) #输出参数大小
            self.decode_block.append(block)
            inputs = outputs
            mul //= 2 #逐渐递减到 1 

        self.to_rgb = to_rgb

    def forward(self, styles, lod, blend=1, remove_blob=True):
        x = self.const
        for i in range(lod+1):
            x = self.decode_block[i](x, styles[:, 2*i+0], styles[:,2*i+1])
        x = self.to_rgb[lod](x)
        return x