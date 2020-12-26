import numpy as np
import torch
from defaults import get_cfg_defaults
import argparse
from module.model import Model
from nativeUtils.checkpointer import Checkpointer
import logging
from torchvision.utils import save_image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="StyleGAN")
parser.add_argument(
        "--config-file",
        default="configs/experiment_ffhq.yaml",
        metavar="FILE",
        type=str,) # args.config_file
args = parser.parse_args()
cfg = get_cfg_defaults()
cfg.merge_from_file(args.config_file)
cfg.freeze()

model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count= cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=3)
model.eval()

model_dict = {
        'generator_s': model.generator,
        'mapping_fl_s': model.mapping,
        'dlatent_avg': model.dlatent_avg,
    }

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
print(cfg)
checkpointer = Checkpointer(cfg, model_dict, logger=logger, save=True)
checkpointer.load(file_name='./pre-model/karras2019stylegan-ffhq.pth')

#-----------random-w----------------- 在同seed下随机生成图像和官方pre-trained一致
# rnd = np.random.RandomState(5)
# latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
# sample = torch.tensor(latents).float()

# with torch.no_grad():
#     save_image((model.generate(lod=8, blend_factor=1, z=sample, remove_blob= True)+1)/2, 'sample-R-blob.png') # model.generate()输入z:[-1,512], 经过Gmap后处理为[-1,18,512] 再经过Gs

#------------------trump w+------------ 用川普的潜码编辑
donald_trump = np.load('./direction/donald_trump_01.npy') #[18, 512]
i=5
donald_trump[i,:] = 0
#print(donald_trump)
donald_trump = donald_trump.reshape(1,18,512)
donald_trump=torch.tensor(donald_trump)

images=model.generator.forward(donald_trump,8) # lod=8,即1024,  和model.generate()不同,model.generator.forward()输入18层潜变量
images=(images + 1)/2
save_image(images,'trump_ls_layer%d=0.png'%(i))

#-----------------direnction with w+------- 用川普和属性的潜码编辑
# donald_trump = np.load('./direction/donald_trump_01.npy') #[18, 512]
# direction = np.load('./direction/age.npy')#[18, 512]

# i=-3 #属性向量系数
# j=1 #Latent code 层数: j+1
# seq = donald_trump
# seq[j] = (seq+i*direction)[j] # 选择第i-j层的潜码
# seq = seq.reshape(1,18,512)
# seq=torch.tensor(seq)
# with torch.no_grad():
# 	img = model.generator.forward(seq,8)

# save_image((img+1)/2,'-age-layer%d.png'%(j+1))
# print('done')

