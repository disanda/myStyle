import torch
import numpy as np
from module.net import Generator, Mapping, Discriminator
from torchvision.utils import save_image
import module.BE as BE

#------------------随机数设置--------------
def set_seed(seed):
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

#-------测试G和pgE在PG下的分辨率情况------------
G = Generator(startf=16, maxf=512, layer_count=9, latent_size=512, channels=3)
G.load_state_dict(torch.load('./pre-model/Gs_dict.pth'))
Gm = Mapping(num_layers=18, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512)
Gm.load_state_dict(torch.load('./pre-model/Gm_dict.pth')) 

i=5
lod = 8
set_seed(i)
latents = torch.randn(3, 512)
latents = Gm(latents) 
img = G.forward(latents,lod=lod) # lod = 8 -> 1024
#save_image((img+1)/2, 'lod%d.png'%lod)


E = BE.BE()
print(img.shape)
c, w = E(img,block_num=lod+1)
print(c.shape)
print(w.shape)

