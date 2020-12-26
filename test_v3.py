import torch
import numpy as np
from module.net import Generator, Mapping, Discriminator
from torchvision.utils import save_image
import torchvision
from torch.nn import functional as F

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

i=9
lod = 8
set_seed(i)
latents = torch.randn(5, 512)

with torch.no_grad():
	latents = Gm(latents) 
	img = G.forward(latents,lod=lod) # lod = 8 -> 1024
# save_image((img+1)/2, 'lod%d.png'%lod)


# E = BE.BE()
# print(img.shape)
# c, w = E(img,block_num=lod+1)
# print(c.shape)
# print(w.shape)

#-----------测试不同结构BE的pre-model加载情况

#import module.BE as BE
# import module.BE_reTune as BE
# E = BE.BE()

# #E.load_state_dict(torch.load('/Users/apple/Desktop/myStyle/StyleGAN-v1/E_model_ep30000.pth',map_location=torch.device('cpu')),strict=False)

# #先分析两个模型的keys差异
# pretrained_dict = torch.load('/Users/apple/Desktop/myStyle/StyleGAN-v1/E_model_ep30000.pth',map_location=torch.device('cpu'))
# model_dict = E.state_dict()

#查找包含的差异keys的字符
# for k,v in pretrained_dict.items():
# 	if 'noise' in k:
# 		k=k.replace('noise','kkk') ##替换字符
# 		print(k)

# 更新
# model_dict.update(pretrained_dict2)
# E.load_state_dict(model_dict) # strict=False

# with torch.no_grad():
# 	latents = Gm(latents) 
# 	img1 = G.forward(latents,lod=lod)
# 	c, w = E(img1,block_num=lod+1)
# 	img2 = G.forward(w,lod=lod)

# img = torch.cat((img1,img2))
# save_image((img+1)/2, 'ED_lod%d_i%d.png'%(lod,i),nrow=5)


#---------------多种插值的上下采样方式--------
# from torch.nn import functional as F

# #img2_1 = F.interpolate(img,[256,256],mode='bilinear')
# img2_2 = F.avg_pool2d(img,2,2)
# #img2_3 = F.interpolate(img,[256,256])

# with torch.no_grad():
# 	latents = Gm(latents) 
# 	img1 = G.forward(latents,lod=lod)

# save_image((img+1)/2, 'ED_lod%d_i%d.png'%(lod,i),nrow=5)

#------------------裁剪区域----------------

img2 = img[:,:,:,128:-128]
print(img2.shape)
img3 = F.avg_pool2d(img2,2,2)
print(img3.shape)
save_image((img3+1)/2, 'column128_down_seed_%d.png'%i,nrow=5)




