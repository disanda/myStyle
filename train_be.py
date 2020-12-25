import os
import torch
import torchvision
from module.net import * # Generator,Mapping
import module.BE as BE
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

def train():
	Gs = Generator(startf=16, maxf=512, layer_count=9, latent_size=512, channels=3)
	Gs.load_state_dict(torch.load('./pre-model/Gs_dict.pth'))
	Gm = Mapping(num_layers=18, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512)
	Gm.load_state_dict(torch.load('./pre-model/Gm_dict.pth')) 
	#Gm.requires_grad_(False)
	#Gs.requires_grad_(False)
	E = BE.BE()
	E.load_state_dict(torch.load('/_yucheng/myStyle/myStyle-v1/result/EB_V4_residual_mse1C/models/E_model_ep10000.pth'),strict=False)
	Gs.cuda()
	Gm.cuda()
	E.cuda()
	const1 = Gs.const

	E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0)

	loss_all=0
	loss_mse = torch.nn.MSELoss()
	loss_lpips = lpips.LPIPS(net='vgg').to('cuda')
	loss_kl = torch.nn.KLDivLoss()

	batch_size = 8
	for epoch in range(120000):
		set_seed(epoch%12000)
		latents = torch.randn(batch_size, 512).to('cuda') #[32, 512]
		with torch.no_grad(): #这里需要生成图片和变量
			w1 = Gm(latents) #[batch_size,18,512]
			imgs1 = Gs.forward(w1,8)

		const2,w2 = E(imgs1.cuda())

		imgs2=Gs.forward(w2,8)

		E_optimizer.zero_grad()
		#loss_img_mse = loss_mse(imgs1,imgs2)
		loss_img_mse_c1 = loss_mse(imgs1[:,0],imgs2[:,0])
		loss_img_mse_c2 = loss_mse(imgs1[:,1],imgs2[:,1])
		loss_img_mse_c3 = loss_mse(imgs1[:,2],imgs2[:,2])
		loss_img_mse = max(loss_img_mse_c1,loss_img_mse_c2,loss_img_mse_c3)

		imgs_center1 = imgs1[:,:,128:640,256:-256]
		imgs_center2 = imgs1[:,:,128:640,256:-256]
		loss_img_mse_center = loss_mse(imgs_center1,imgs_center2)
		loss_img_lpips_center = loss_lpips(imgs_center1,imgs_center2).mean()

		imgs1_ = F.avg_pool2d(imgs1,2,2)
		imgs2_ = F.avg_pool2d(imgs2,2,2)
		loss_img_lpips = loss_lpips(imgs1_,imgs2_).mean()


		loss_c = loss_mse(const1,const2) #没有这个const，梯度起初没法快速下降，很可能无法收敛, 这个惩罚即乘0.1后,效果大幅提升！
		loss_c_m = loss_mse(const1.mean(),const2.mean())
		loss_c_s = loss_mse(const1.std(),const2.std())

		loss_w = loss_mse(w1,w2)
		loss_w_m = loss_mse(w1.mean(),w2.mean()) #初期一会很大10,一会很小0.0001
		loss_w_s = loss_mse(w1.std(),w2.std()) #后期一会很大，一会很小

		y1, y2 = torch.nn.functional.softmax(const1),torch.nn.functional.softmax(const2)
		loss_kl_c = loss_kl(torch.log(y2),y1)
		loss_kl_c = torch.where(torch.isnan(loss_kl_c),torch.full_like(loss_kl_c,0), loss_kl_c)
		loss_kl_c = torch.where(torch.isinf(loss_kl_c),torch.full_like(loss_kl_c,1), loss_kl_c)

		y1_imgs, y2_imgs = torch.nn.functional.softmax(imgs1_),torch.nn.functional.softmax(imgs2_)
		loss_kl_img = loss_kl(torch.log(y2_imgs),y1_imgs) #D_kl(True=y1_imgs||Fake=y2_imgs)
		loss_kl_img = torch.where(torch.isnan(loss_kl_img),torch.full_like(loss_kl_img,0), loss_kl_img)
		loss_kl_img = torch.where(torch.isinf(loss_kl_img),torch.full_like(loss_kl_img,1), loss_kl_img)

		w1_kl, w2_kl = torch.nn.functional.softmax(w1),torch.nn.functional.softmax(w2)
		loss_kl_w = loss_kl(torch.log(w2_kl),w1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
		loss_kl_w = torch.where(torch.isnan(loss_kl_w),torch.full_like(loss_kl_w,0), loss_kl_w)
		loss_kl_w = torch.where(torch.isinf(loss_kl_w),torch.full_like(loss_kl_w,1), loss_kl_w)

		print('i_'+str(epoch)+'--loss_all__:'+str(loss_all)+'--loss_mse:'+str(loss_img_mse)+'--loss_lpips:'+str(loss_img_lpips)+'--loss_c:'+str(loss_c)+'--loss_kl_c:'+str(loss_kl_c))
		print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item())+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item()))
		print('loss_m_center:'+str(loss_img_mse_center.item())+'--loss_lpips_center:'+str(loss_img_lpips_center.item())+'--loss_kl_imgs:'+str(loss_kl_imgs.item())+'--loss_kl_w:'+str(loss_kl_w.item()))
		print('-')

		loss_all = 13*loss_img_mse+ 5*loss_img_lpips  + 0.02*loss_c+loss_kl_c+0.02*loss_w+0.03*loss_w_m+0.03*loss_w_s+0.03*loss_c_m+0.03*loss_c_s \
		+ 31*loss_img_mse_center +17*loss_img_lpips_center + loss_kl_img + loss_kl_w

		loss_all.backward()
		E_optimizer.step()

		print('i_'+str(epoch)+'--loss_all__:'+str(loss_all.item())+'--loss_mse:'+str(loss_img_mse.item())+'--loss_lpips:'+str(loss_img_lpips.item())+'--loss_c:'+str(loss_c.item())+'--loss_kl_c:'+str(loss_kl_c.item()))
		print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item())+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item()))
		print('loss_m_center:'+str(loss_img_mse_center.item())+'--loss_lpips_center:'+str(loss_img_lpips_center.item())+'--loss_kl_imgs:'+str(loss_kl_imgs.item())+'--loss_kl_w:'+str(loss_kl_w.item()))
		print('-')
		if epoch % 100 == 0:
			test_img = torch.cat((imgs1[:3],imgs2[:3]))*0.5+0.5
			torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.jpg'%(epoch), nrow=3)
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print('i_'+str(epoch)+'--loss_all__:'+str(loss_all.item())+'--loss_mse:'+str(loss_img_mse.item())+'--loss_lpips:'+str(loss_img_lpips.item())+'--loss_c:'+str(loss_c.item())+'--loss_kl_c:'+str(loss_kl_c.item()),file=f)
				print('loss_w:'+str(loss_w.item())+'--loss_w_m:'+str(loss_w_m.item())+'--loss_w_s:'+str(loss_w_s.item())+'--loss_c_m:'+str(loss_c_m.item())+'--loss_c_s:'+str(loss_c_s.item()),file=f)
				print('loss_m_center:'+str(loss_img_mse_center.item())+'--loss_lpips_center:'+str(loss_img_lpips_center.item())+'--loss_kl_imgs:'+str(loss_kl_imgs.item())+'--loss_kl_w:'+str(loss_kl_w.item()),file=f)
			if epoch % 10000 == 0:
				torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)

if __name__ == "__main__":
	resultPath = "./result/EB_V5_center_kl_inverse_all_res11-89"
	if not os.path.exists(resultPath): os.mkdir(resultPath)

	resultPath1_1 = resultPath+"/imgs"
	if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

	resultPath1_2 = resultPath+"/models"
	if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

	train()




