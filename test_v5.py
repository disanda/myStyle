import torch
import numpy as np
from module.net import Generator, Mapping, Discriminator, Mapping2
from torchvision.utils import save_image
import torchvision
from torch.nn import functional as F
import module.BE_v2 as BE

#------------------随机数设置--------------
def set_seed(seed):
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

#-------------load single image 2 tensor--------------
# loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# from PIL import Image
# def image_loader(image_name):
# 	image = Image.open(image_name).convert('RGB')
# 	#image = image.resize((1024,1024))
# 	image = loader(image).unsqueeze(0)
# 	return image.to(torch.float)

# imgs1=image_loader('/Users/apple/Desktop/myStyle/model-result-v1/E/ty_align.png')

# imgs1 = imgs1*2-1

#-------实现一个大的Mapping网络，完成 512 -> 512*18 的映射 ------------
G = Generator(startf=16, maxf=512, layer_count=9, latent_size=512, channels=3)
G.load_state_dict(torch.load('./pre-model/Gs_dict.pth'))
Gm1 = Mapping2(num_layers=18, mapping_layers=8, latent_size=512)
Gm2 = Mapping2(num_layers=18, mapping_layers=8, latent_size=512, inverse=True)
#Gm1.load_state_dict(torch.load('./pre-model/Gm1.pth')) 

E = BE.BE()
E.load_state_dict(torch.load('/Users/apple/Desktop/myStyle/model-result-v1/E/pre-model/v8-E_model_ep20000.pth',map_location=torch.device('cpu')),strict=False)

set_seed(0)
with torch.no_grad(): #这里需要生成图片和变量
	latents = torch.randn(4, 512)
	w1 = Gm1(latents)
	imgs1 = G.forward(w1,8)
	const2,w2 = E(imgs1)
	imgs2 = G.forward(w2,8)

imgs = torch.cat((imgs1,imgs2))
save_image(imgs*0.5+0.5, 'img2.png',nrow=4)

