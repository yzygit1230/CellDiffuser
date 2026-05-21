import os
from collections import OrderedDict
from functools import partial
import data
from options.test_options import TestOptions
from models.cell_model import CellModel
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm
from celldiffuser_utils.script_util import create_gaussian_diffusion
from celldiffuser_utils.script_util import create_model

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = CellModel(opt)
model.eval()

denoise_step = opt.denoise_step         
timestep_t = opt.timestep_t             
num_timesteps = opt.num_timesteps       
assert timestep_t % denoise_step == 0
skip = timestep_t // denoise_step
assert num_timesteps % skip == 0
timestep_respacing = f'ddim{num_timesteps // skip}'
respaced_timestep_t = timestep_t // skip
diffuser = create_gaussian_diffusion(timestep_respacing=timestep_respacing,
                                     **opt.diffuser_kwargs)
unet = create_model(**opt.unet_kwargs)

if opt.use_fp16:
    unet.convert_to_fp16()
unet.cuda().eval()
denoiser = partial(diffuser.ddim_sample_from_t_loop,
                   model=unet,
                   timestep_t=respaced_timestep_t)
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

print('Number of images: ', len(dataloader))
num_images = 0
for i, data_i in enumerate(tqdm(dataloader)):
    noisy_generated = model(data_i, mode='inference')
    generated = denoiser(x_t=noisy_generated)

    for b in range(generated.shape[0]):
        original_path = data_i['path'][b]
        sub_dir = original_path.split('/')[-3]  
        original_name = os.path.splitext(os.path.basename(original_path))[0]

        # save
        save_dir = os.path.join(webpage.get_image_dir(), sub_dir)  
        os.makedirs(save_dir, exist_ok=True) 
        visuals = OrderedDict([('synthesized_image', generated[b])])
        image_name = os.path.basename(original_path) 
        save_path = os.path.join(save_dir, image_name) 
        visualizer.save_images(webpage, visuals, [save_path])

webpage.save()

print('Inference Finished.')