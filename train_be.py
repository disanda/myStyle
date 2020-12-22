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
	#Gs.cuda()
	#Gm.cuda()
	Gs.requires_grad_(False)
	Gm.requires_grad_(False)
	const1 = Gs.const
	E = BE.BE()
	E.cuda()

	E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0)

	loss_all=0
	loss_mse = torch.nn.MSELoss()
	loss_lpips = lpips.LPIPS(net='vgg').cuda()
	#loss3 = torch.nn.KLDivLoss()

	batch_size = 5
	for epoch in range(120000):
		with torch.no_grad(): #这里需要生成图片和变量
			set_seed(epoch%30000)
			latents = torch.randn(batch_size, 512) #[32, 512]
			w1 = Gm(latents) #[batch_size,18,512]
			imgs1 = Gs.forward(w1,8)

		const2,w2 = E(imgs1.cuda())

		with torch.no_grad():
			imgs2=Gs.forward(w2.to('cpu'),8)

		E_optimizer.zero_grad()
		loss_1 = loss_mse(imgs1,imgs2)
		loss_2_1 = loss_lpips(imgs1.cuda(),imgs2.cuda()).mean()
		loss_2_2 = loss_lpips(imgs1.cuda(),imgs2.cuda()).std()
		loss_w_1 = loss_mse(w1,w2)
		loss_w_2 = loss_lpips(w1.cuda(),w2).mean()
		loss_w_3 = loss_lpips(w1.cuda(),w2).std()
		loss_c = loss_mse(const1,const2)
		loss_all = loss_1+ loss_2_1 + loss_2_2 + loss_w_1 + loss_w_2 + loss_w_3 + loss_c
		loss_all.backward()
		E_optimizer.step()

		print('loss_all__:'+str(loss_all.item())+'--loss_1:'+str(loss_1.item())+'--loss_2_1:'+str(loss_2_1.item())+'--loss_2_2:'+str(loss_2_2.item())+'--loss_c:'+str(loss_c.item()))
		print('i_'+str(epoch)+'loss_w_1:'+str(loss_w_1.item())+'--loss_w_2:'+str(loss_w_2.item())+'--loss_w_3:'+str(loss_w_3.item())+'--loss_w_2:'+str(loss_w_3.item()))
		if epoch % 100 == 0:
			test_img = torch.cat((imgs1[:3],imgs2[:3]))*0.5+0.5
			torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.jpg'%(epoch), nrow=3)
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print('loss_all__:'+str(loss_all.item())+'--loss_1:'+str(loss_1.item())+'--loss_2_1:'+str(loss_2_1.item())+'--loss_2_2:'+str(loss_2_2.item())+'--loss_c:'+str(loss_c.item()),file=f)
				print('loss_w_1:'+str(loss_w_1.item())+'--loss_w_2:'+str(loss_w_2.item())+'--loss_w_3:'+str(loss_w_3.item())+'--loss_w_2:'+str(loss_w_3.item()),file=f)
			if epoch % 1000 == 0:
				torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)

if __name__ == "__main__":
	resultPath = "./result/EB_V1"
	if not os.path.exists(resultPath): os.mkdir(resultPath)

	resultPath1_1 = resultPath+"/imgs"
	if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

	resultPath1_2 = resultPath+"/models"
	if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

	train()


