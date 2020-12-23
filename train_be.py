import os
import torch
import torchvision
from module.net import * # Generator,Mapping
import module.BE as BE
from module.custom_adam import LREQAdam
import lpips
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def set_seed(seed): #随机数设置
	np.random.seed(seed)
	#random.seed(seed)
	torch.manual_seed(seed) # cpu
	torch.cuda.manual_seed_all(seed)  # gpu
	torch.backends.cudnn.deterministic = True

def train():
	Gs = Generator(startf=16, maxf=512, layer_count=9, latent_size=512, channels=3)
	Gs.load_state_dict(torch.load('./pre-model/Gs_dict.pth'))
	Gm = Mapping(num_layers=18, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512)
	Gm.load_state_dict(torch.load('./pre-model/Gm_dict.pth')) 
	#Gs.requires_grad_(False)
	Gm.requires_grad_(False)
	E = BE.BE()
	Gs.cuda()
	Gm.cuda()
	E.cuda()
	const1 = Gs.const

	E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0)

	loss_all=0
	loss_mse = torch.nn.MSELoss()
	loss_lpips = lpips.LPIPS(net='vgg').to('cuda')
	loss_kl = torch.nn.KLDivLoss()
	#loss3 = torch.nn.KLDivLoss()

	batch_size = 3
	for epoch in range(120000):
		set_seed(epoch%30000)
		latents = torch.randn(batch_size, 512).to('cuda') #[32, 512]
		with torch.no_grad(): #这里需要生成图片和变量
			w1 = Gm(latents) #[batch_size,18,512]
			imgs1 = Gs.forward(w1,8)

		const2,w2 = E(imgs1.cuda())

		imgs2=Gs.forward(w2,8)

		E_optimizer.zero_grad()
		loss_mse = loss_mse(imgs1,imgs2)
		loss_lpips = loss_lpips(imgs1,imgs2).mean()

		loss_c = loss_mse(const1,const2) #没有这个const，梯度起初没法快速下降，很可能无法收敛，
		loss_c_m = loss_mse(const1.mean(dim=0),const2.mean(dim=0))
		loss_c_s = loss_mse(const1.std(dim=0),const2.std(dim=0))

		loss_w = loss_mse(w1,w2)
		loss_w_m = loss_mse(w1.mean(dim=0),w2.mean(dim=0))
		loss_w_s = loss_mse(w1.std(dim=0),w2.std(dim=0))

		y1, y2 = torch.nn.functional.softmax(const1),torch.nn.functional.softmax(const2)
		loss_kl_c = loss_kl(torch.log(y1),y2)
		loss_kl_c = torch.where(torch.isnan(loss_kl_c),torch.full_like(loss_kl_c,0), loss_kl_c)
		loss_kl_c = torch.where(torch.isinf(loss_kl_c),torch.full_like(loss_kl_c,1), loss_kl_c)

		loss_all = loss_mse+ loss_lpips  + 0.1*loss_c+loss_kl_c+loss_w+loss_w_m+loss_w_s+loss_c_m+loss_c_s
		loss_all.backward()
		E_optimizer.step()

		print('i_'+str(epoch)+'--loss_all__:'+str(loss_all.item())+'--loss_mse:'+str(loss_mse.item())+'--loss_lpips:'+str(loss_lpips.item())+'--loss_c:'+str(loss_c.item())+'--loss_kl_c:'+str(loss_kl_c.item()))
		print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item())+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item()))
		print('-')
		if epoch % 100 == 0:
			test_img = torch.cat((imgs1[:3],imgs2[:3]))*0.5+0.5
			torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.jpg'%(epoch), nrow=3)
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print('i_'+str(epoch)+'--loss_all__:'+str(loss_all.item())+'--loss_mse:'+str(loss_mse.item())+'--loss_lpips:'+str(loss_lpips.item())+'--loss_c:'+str(loss_c.item())+'--loss_kl_c:'+str(loss_kl_c.item()),file=f)
				print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item())+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item()),file=f)
			if epoch % 10000 == 0:
				torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)

if __name__ == "__main__":
	resultPath = "./result/EB_V3_GsginE"
	if not os.path.exists(resultPath): os.mkdir(resultPath)

	resultPath1_1 = resultPath+"/imgs"
	if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

	resultPath1_2 = resultPath+"/models"
	if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

	train()




