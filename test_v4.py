import torch
import numpy as np
from module.net import Generator, Mapping, Discriminator
from torchvision.utils import save_image
import torchvision
from torch.nn import functional as F
import module.BE as BE

#------------------随机数设置--------------
def set_seed(seed):
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

#-------------load single image 2 tensor--------------
loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

from PIL import Image
def image_loader(image_name):
	image = Image.open(image_name).convert('RGB')
	#image = image.resize((1024,1024))
	image = loader(image).unsqueeze(0)
	return image.to(torch.float)

imgs1=image_loader('./wwm_align.png')

imgs1 = imgs1*2-1

#-------测试G和pgE在PG下的分辨率情况------------
G = Generator(startf=16, maxf=512, layer_count=9, latent_size=512, channels=3)
G.load_state_dict(torch.load('./pre-model/Gs_dict.pth'))
#Gm = Mapping(num_layers=18, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512)
#Gm.load_state_dict(torch.load('./pre-model/Gm_dict.pth')) 
E = BE.BE()
E.load_state_dict(torch.load('./pre-model/v6_2_E_model_ep10000.pth',map_location=torch.device('cpu')),strict=False)

#set_seed(epoch%20000)
#latents = torch.randn(batch_size, 512).to('cuda') #[32, 512]

center_tensor = torch.load('./pre-model/v6_2_center_tensor_ep10000.pt',map_location=torch.device('cpu'))
layer_idx = torch.arange(18)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
coefs = torch.where(layer_idx < 8, 0.7 * ones, ones)

with torch.no_grad(): #这里需要生成图片和变量
	const2,w2 = E(imgs1)
	#w2 = torch.lerp(center_tensor, w2, coefs) # avg + (styles-avg) * [0.7,0.7,...,1,1,1] 
	imgs2 = G.forward(w2,8)

save_image(imgs2*0.5+0.5, 'wwm_align_r_v6_2_nolerp.png',nrow=1)