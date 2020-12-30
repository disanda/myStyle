#它这个实质上是能让编码器映射部分图片，但是这部分映射了，可能其他部分就不能很好的映射
import os
import torch
import torchvision
from module.net import * # Generator,Mapping
import module.BE_v2 as BE
from module.custom_adam import LREQAdam
import lpips
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def set_seed(seed): #随机数设置
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


def train(avg_tensor = None, coefs=0):
	Gs = Generator(startf=16, maxf=512, layer_count=9, latent_size=512, channels=3)
	Gs.load_state_dict(torch.load('./pre-model/Gs_dict.pth')) 
	Gm = Mapping(num_layers=18, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512)
	Gm.load_state_dict(torch.load('./pre-model/Gm_dict.pth')) 
	Gm1 = Mapping2(num_layers=18, mapping_layers=8, latent_size=512)
	Gm2 = Mapping2(num_layers=18, mapping_layers=8, latent_size=512, inverse=True)
	Gm1.load_state_dict(torch.load('./pre-model/Gm1.pth')) 
	Gm2.load_state_dict(torch.load('./pre-model/Gm2.pth')) 
	E = BE.BE()
	E.load_state_dict(torch.load('/_yucheng/myStyle/myStyle-v1/result/EB_V8_newE_noEw_Ebias/models/E_model_ep115000.pth'),strict=False)

	Gs.cuda()
	E.cuda()

	Gm1_optimizer = LREQAdam([{'params': Gm1.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0)
	Gm2_optimizer = LREQAdam([{'params': Gm2.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0)

	loss_mse = torch.nn.MSELoss()
	loss_kl = torch.nn.KLDivLoss()

batch_size=5
	for epoch in range(100000):
		set_seed(epoch%20000)
		z = torch.randn(batch_size, 512) #[32, 512]
		w1 = Gm(z,coefs_m=coefs).to('cuda') #[batch_size,18,512]
		with torch.no_grad(): #这里需要生成图片和变量
			imgs1 = Gs.forward(w1,8)
			const2,w2 = E(imgs1.cuda())

		w_m1 = Gm1(z) 
		z_m2 = Gm2(w2)
#loss1
		Gm1_optimizer.zero_grad()
		loss_m1_mse = loss_mse(w2,w_m1)
		loss_m1_mse_mean = loss_mse(w2.mean(),w_m1.mean())
		loss_m1_mse_std = loss_mse(w2.std(),w_m1.std())

		y1_w, y2_w = torch.nn.functional.softmax(w2),torch.nn.functional.softmax(w_m1)
		loss_kl_w = loss_kl(torch.log(y2_w),y1_w) #D_kl(True=y1_w||Fake=y2_w)
		loss_kl_w = torch.where(torch.isnan(loss_kl_w),torch.full_like(loss_kl_w,0), loss_kl_w)
		loss_kl_w = torch.where(torch.isinf(loss_kl_w),torch.full_like(loss_kl_w,1), loss_kl_w)

		loss_1 = loss_m1_mse + loss_m1_mse_mean + loss_m1_mse_std + loss_kl_w
		loss_1.backward()
		Gm1_optimizer.step()
#loss2
		Gm2_optimizer.zero_grad()
		loss_m2_mse = loss_mse(z,z_m2)
		loss_m2_mse_mean = loss_mse(z.mean(),z_m2.mean())
		loss_m2_mse_std = loss_mse(z.std(),z_m2.std())

		y1_z, y2_z = torch.nn.functional.softmax(z),torch.nn.functional.softmax(z_m2)
		loss_kl_z = loss_kl(torch.log(y2_z),y1_z) #D_kl(True=y1_z||Fake=y2_z)
		loss_kl_z = torch.where(torch.isnan(loss_kl_z),torch.full_like(loss_kl_z,0), loss_kl_z)
		loss_kl_z = torch.where(torch.isinf(loss_kl_z),torch.full_like(loss_kl_z,1), loss_kl_z)

		loss_2 = loss_m2_mse + loss_m2_mse_mean + loss_m2_mse_std + loss_kl_z
		loss_2.backward()
		Gm2_optimizer.step()

		loss_all =  loss_1  + loss_2 
		print('i_'+str(epoch)+'--loss_all__:'+str(loss_all.item())+'--loss_m1_mse:'+str(loss_m1_mse.item())+'--loss_m1_mse_mean:'+str(loss_m1_mse_mean.item())+'--loss_m1_mse_std:'+str(loss_m1_mse_std.item())+'--loss_kl_w:'+str(loss_kl_w.item()))
		print('--loss_m2_mse:'+str(loss_m2_mse.item())+'--loss_m2_mse_mean:'+str(loss_m2_mse_mean.item())+'--loss_m2_mse_std:'+str(loss_m2_mse_std.item())+'--loss_kl_z:'+str(loss_kl_z.item()))
		print('-')

		if epoch % 100 == 0:
			with torch.no_grad(): #这里需要生成图片和变量
				w = Gm1(z)
				z2 = Gm2(w)
				w_ = Gm1(z2) 
				imgs1 = Gs.forward(w,8)
				imgs2 = Gs.forward(w_,8)
			test_img = torch.cat(imgs1,imgs2)*0.5+0.5
			torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.jpg'%(epoch),nrow=2) # nrow=3
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print('i_'+str(epoch)+'--loss_all__:'+str(loss_all.item())+'--loss_m1_mse:'+str(loss_m1_mse.item())+'--loss_m1_mse_mean:'+str(loss_m1_mse_mean.item())+'--loss_m1_mse_std:'+str(loss_m1_mse_std.item())+'--loss_kl_w:'+str(loss_kl_w.item()),file=f)
				print('--loss_m2_mse:'+str(loss_m2_mse.item())+'--loss_m2_mse_mean:'+str(loss_m2_mse_mean.item())+'--loss_m2_mse_std:'+str(loss_m2_mse_std.item())+'--loss_kl_z:'+str(loss_kl_z.item()),file=f)
			if epoch % 5000 == 0:
				torch.save(Gm1.state_dict(), resultPath1_2+'/Gm1_model_ep%d.pth'%epoch)
				torch.save(Gm2.state_dict(), resultPath1_2+'/Gm2_model_ep%d.pth'%epoch)


if __name__ == "__main__":
	resultPath = "./result/Gm_1&2_V10"
	if not os.path.exists(resultPath): os.mkdir(resultPath)

	resultPath1_1 = resultPath+"/imgs"
	if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

	resultPath1_2 = resultPath+"/models"
	if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

	center_tensor = torch.load('./center_tensor.pt')
	layer_idx = torch.arange(18)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
	ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
	coefs = torch.where(layer_idx < 8, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1] 


	train(center_tensor,coefs)




