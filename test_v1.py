import numpy as np
import torch
from defaults import get_cfg_defaults
import argparse
from module.model import Model
from checkpointer import Checkpointer
import logging
from torchvision.utils import save_image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="StyleGAN")
parser.add_argument(
        "--config-file",
        default="configs/experiment_ffhq.yaml",
        metavar="FILE",
        type=str,)
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

#-----------random-w-----------------
# rnd = np.random.RandomState(5)
# latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
# sample = torch.tensor(latents).float()

# with torch.no_grad():
#     save_image((model.generate(8, True, z=sample)+1)/2, 'sample2.png') #z: [-1,512]

#------------------trump w+------------
# donald_trump = np.load('stylegan/ffhq_dataset/latent_representations/donald_trump_01.npy') #[18, 512]
# donald_trump = donald_trump.reshape(1,18,512)
# donald_trump=torch.tensor(donald_trump)

# images=model.generator.forward(donald_trump,8) # lod=8,即1024
# images=(images + 1)/2
# save_image(images,'trump.png')

#-----------------direnction with w+-------
donald_trump = np.load('./direction/donald_trump_01.npy') #[18, 512]
donald_trump=torch.tensor(donald_trump)
direction = np.load('./direction/smile.npy')#[18, 512]

i=3
seq = donald_trump
seq[0:3] = (seq+i*direction)[0:3]
seq = seq.reshape(1,18,512)
with torch.no_grad():
	img = model.generator.forward(seq,8)

save_image((img+1)/2,'trump-smile.png')
print('done')

