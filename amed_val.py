import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import dnnlib
from solver_utils import get_schedule
from diffusers import StableDiffusionPipeline
from diffusers_amed_plugin_dpmpp import DPMSolverMultistepScheduler
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse, os, sys, glob
import accelerate
from pytorch_lightning import seed_everything
from pycocotools.coco import COCO
import random
from torch import autocast
from contextlib import contextmanager, nullcontext
import time
from tqdm import tqdm, trange

def read_amed_predictor(AMED_predictor, num_steps, sigma_min=0.0292, sigma_max=14.6146, device=None, schedule_type='polynomial', schedule_rho=7, net=None):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    schedule_type = AMED_predictor.schedule_type
    schedule_rho = AMED_predictor.schedule_rho
    ones = torch.tensor([1]).reshape(-1,)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, batch_size=1)
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        t_steps_amed = torch.cat((t_cur.reshape(-1,), t_mid.reshape(-1,), t_next.reshape(-1,))) if i == 0 else torch.cat((t_steps_amed, t_mid.reshape(-1,), t_next.reshape(-1,)))
        scale_dirs = torch.cat((ones, scale_dir.reshape(-1,), ones)) if i == 0 else torch.cat((scale_dirs, scale_dir.reshape(-1,), ones))
        scale_times = torch.cat((ones, scale_time.reshape(-1,), ones)) if i == 0 else torch.cat((scale_times, scale_time.reshape(-1,), ones))
    return t_steps_amed, scale_dirs, scale_times


def get_amed_prediction(AMED_predictor, t_cur, t_next, batch_size):
    unet_enc = torch.zeros((batch_size, 8, 8), device=t_cur.device)
    output = AMED_predictor(unet_enc, t_cur, t_next)
    output_list = [*output]
    if len(output_list) == 2:
        try:
            use_scale_time = AMED_predictor.module.scale_time
        except:
            use_scale_time = AMED_predictor.scale_time
        if use_scale_time:
            r, scale_time = output_list
            scale_dir = torch.ones_like(scale_time)
        else:
            r, scale_dir = output_list
            scale_time = torch.ones_like(scale_dir)
    elif len(output_list) == 3:
        r, scale_dir, scale_time = output_list
    else:
        r = output
        scale_dir = torch.ones_like(r)
        scale_time = torch.ones_like(r)
    return r, scale_dir, scale_time


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2)
    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


class SD_scheduler(torch.nn.Module):
    def __init__(self,
        alphas_cumprod,
        epsilon_t       = 1e-3,                 # Minimum t-value used during training.
        beta_d          = 9.0420,               # Extent of the noise level schedule.
        beta_min        = 0.8477,               # Initial slope of the noise level schedule.
    ):
        super().__init__()
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        
        log_alphas = 0.5 * torch.log(alphas_cumprod)
        self.M = len(log_alphas)
        self.t_array = torch.linspace(0., 1., self.M + 1)[1:].reshape((1, -1))
        self.log_alpha_array = log_alphas.reshape((1, -1,))

        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        
    def marginal_log_mean_coeff(self, t):
        t = torch.tensor(t)
        return self.interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def sigma(self, t):
        return self.marginal_std(t) / self.marginal_alpha(t)

    def sigma_inv(self, sigma):
        lamb = -(sigma.log())
        log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
        t = self.interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
        return t.reshape((-1,))
    
    def interpolate_fn(self, x, xp, yp):
        N, K = x.shape[0], xp.shape[1]
        all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
        sorted_all_x, x_indices = torch.sort(all_x, dim=2)
        x_idx = torch.argmin(x_indices, dim=2)
        cand_start_idx = x_idx - 1
        start_idx = torch.where(
            torch.eq(x_idx, 0),
            torch.tensor(1, device=x.device),
            torch.where(
                torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
            ),
        )
        end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
        start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
        end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
        start_idx2 = torch.where(
            torch.eq(x_idx, 0),
            torch.tensor(0, device=x.device),
            torch.where(
                torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
            ),
        )
        y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
        start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
        end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
        cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
        return cand

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_org"
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        default='./instances_val2014.json',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--use_free_net",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    if not os.path.exists("./exp/00024-ms_coco-5-16-dpmpp-dpmpp-2-discrete"):
        raise ValueError("Download the AMED predictor (00024-ms_coco-5-16-dpmpp-dpmpp-2-discrete) and unzip to './exp'. \
                        from 'https://drive.google.com/drive/folders/1KlS0mV3qKMBu1qghy9sXRrjL2Aic4_fN'")
    predictor_path = '25'
    device = torch.device('cpu')
    exp_path = './exp'

    # Load AMED predictor
    if not predictor_path.endswith('pt'):      # load by experiment number
        # find the directory with trained AMED predictor
        predictor_path_str = '0' * (5 - len(predictor_path)) + predictor_path
        for file_name in os.listdir(exp_path):
            if file_name.split('-')[0] == predictor_path_str:
                file_list = [f for f in os.listdir(os.path.join(exp_path, file_name)) if f.endswith("pt")]
                max_index = -1
                max_file = None
                for ckpt_name in file_list:
                    file_index = int(ckpt_name.split("-")[-1].split(".")[0])
                    if file_index > max_index:
                        max_index = file_index
                        max_file = ckpt_name
                predictor_path = os.path.join(exp_path, file_name, max_file)
                break
    print(f'Loading AMED predictor from "{predictor_path}"...')
    amed_dict = torch.load(predictor_path, map_location=torch.device('cpu'),weights_only=False)['model']

    AMED_kwargs = dnnlib.EasyDict()
    AMED_kwargs.update(img_resolution=64)
    AMED_kwargs.class_name = 'training.networks.AMED_predictor'
    AMED_kwargs.update(num_steps=amed_dict.num_steps, sampler_stu=amed_dict.sampler_stu, sampler_tea=amed_dict.sampler_tea, \
                        M=amed_dict.M, guidance_type=amed_dict.guidance_type, guidance_rate=amed_dict.guidance_rate, \
                        schedule_rho=amed_dict.schedule_rho, schedule_type=amed_dict.schedule_type, afs=amed_dict.afs, \
                        dataset_name=amed_dict.dataset_name, scale_dir=amed_dict.scale_dir, scale_time=amed_dict.scale_time, \
                        max_order=amed_dict.max_order, predict_x0=amed_dict.predict_x0, lower_order_final=amed_dict.lower_order_final)
    AMED_predictor = dnnlib.util.construct_class_by_name(**AMED_kwargs) # subclass of torch.nn.Module
    AMED_predictor.load_state_dict(amed_dict.state_dict())
    # Read AMED predictor
    if AMED_predictor.dataset_name == 'ms_coco':
        betas = make_beta_schedule('linear', 1000, linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
        alphas = 1. - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), device=device)
        scheduler = SD_scheduler(alphas_cumprod).to(device)
        sigma_min = 0.0292
        sigma_max = 14.6146
    else:
        scheduler = None
        sigma_min = 0.002
        sigma_max = 80

    t_steps_ori = get_schedule(2*AMED_predictor.num_steps-1, sigma_min, sigma_max, device=device, schedule_type=AMED_predictor.schedule_type, \
                           schedule_rho=AMED_predictor.schedule_rho, net=scheduler)
    t_steps, scale_dirs, scale_times = read_amed_predictor(AMED_predictor, AMED_predictor.num_steps, sigma_min, sigma_max, device=device, schedule_type=AMED_predictor.schedule_type, \
                              schedule_rho=AMED_predictor.schedule_rho, net=scheduler)
    t_steps_ori = [round(t.item()) for t in (1000 * scheduler.sigma_inv(t_steps_ori) - 1).data]
    t_steps = [round(t.item()) for t in (1000 * scheduler.sigma_inv(t_steps) - 1).data]
    scale_dirs = [round(t.item(), 4) for t in scale_dirs.data]
    scale_times = [round(t.item(), 4) for t in scale_times.data]
    print('Original   :', t_steps_ori)
    print('AMED       :', t_steps)
    print('Grad scale :', scale_dirs)
    print('Time scale :', scale_times)

    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())
    
    seed_everything(opt.seed)

    DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    pipe = StableDiffusionPipeline.from_single_file( "./v1-5-pruned-emaonly.safetensors")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.scale_dirs = scale_dirs
    pipe.scheduler.scale_times = scale_times
    pipe.to('cuda')
    prompt = "a photogragh of an astronaut riding a horse"
    seed = 42
    num_steps = 10
    sampling_schedule_amed = t_steps
    
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        coco_annotation_file_path = opt.from_file
        coco_caption_file_path = './captions_val2014.json'
        coco_annotation = COCO(annotation_file=coco_annotation_file_path)
        coco_caption = COCO(annotation_file=coco_caption_file_path)
        query_names = [] #['cup','broccoli','dining table','toaster','carrot','toilet','sink','fork','hot dog','knife','pizza','spoon','donut','clock','bowl','cake','vase','banana','scissors','couch','apple','sandwich','potted plant','microwave','orange','bed','oven']
        unselect_names = [] # ['person','airplane','bird','mouse','cat','dog','horse','clock']

        # 获取包含指定类别的图像ID
        query_ids = []
        img_ids = coco_annotation.getImgIds()
        # for query_name in query_names:
        # query_ids += coco_annotation.getCatIds(catNms=query_names)
        # for query_id in query_ids:
        #     img_ids += coco_annotation.getImgIds(catIds=query_id)

        # 获取包含不需要类别的图像ID
        unselect_id = []
        unselect_img_ids = []
        for unselect_name in unselect_names:
            unselect_id += coco_annotation.getCatIds(catNms=[unselect_name])
            unselect_img_ids += coco_annotation.getImgIds(catIds=unselect_id)

        # 过滤掉包含不需要类别的图像ID
        real_img_ids = [item for item in img_ids if item not in unselect_img_ids]
        random.shuffle(real_img_ids)
        
        real_img_ids = real_img_ids[0:10000]

        # 获取这些图像的caption ID
        caption_ids = coco_caption.getAnnIds(imgIds=real_img_ids)

        # 获取并显示这些图像的captions
        captions = coco_caption.loadAnns(caption_ids)
        tmp_caption = []
        for idx,caption in enumerate(captions):
            if idx % 5 != 0:
                continue
            tmp_caption.append(caption)
        captions = tmp_caption
        
        data = list(map(lambda x: x['caption'], captions))
        data = data[(0):10000]
        images = coco_caption.loadImgs(ids=real_img_ids)
        folder_name = 'E:\\txt2img-samples\\scls_coco_img_val_random'
        img_path = 'D:\\research_project\\archive(2)\\coco2014\\images\\val2014'
        # if not os.path.exists(folder_name):
        #     os.makedirs(name=folder_name,exist_ok=True)
        #     img_file_name = [ img['file_name'] for img in images ]
        #     for filename in os.listdir(path=img_path):
        #         if filename in img_file_name:
        #             shutil.copy(os.path.join(img_path, filename), folder_name)

    
    folder_name = f"samples-amed-{opt.ddim_steps}"
    if opt.use_free_net:
        folder_name += "-free"
    W, H = 512, 512
    sample_path = os.path.join(outpath, folder_name)
    os.makedirs(sample_path, exist_ok=True)
    
    
    base_count = len(os.listdir(sample_path))
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        # with precision_scope("cuda"):
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling", disable =not accelerator.is_main_process):
                for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                    
                    # torch.cuda.empty_cache()
                    intermediate_photos = list()
                    # prompts = prompts[0]
                            
                    # if isinstance(prompts, tuple) or isinstance(prompts, str):
                    #     prompts = list(prompts)
                    if isinstance(prompts, str):
                        prompts = prompts #+ 'high quality, best quality, masterpiece, 4K, highres, extremely detailed, ultra-detailed'
                        prompts = (prompts,)
                    if isinstance(prompts, tuple) or isinstance(prompts, str):
                        prompts = list(prompts)
                    x_samples_ddim = pipe(
                        prompts, num_images_per_prompt=1,
                        num_inference_steps=num_steps,
                        timesteps=sampling_schedule_amed,
                    ).images
                    # x_samples_ddim = pipe(prompt=prompts,num_inference_steps=opt.ddim_steps,guidance_scale=opt.scale,height=H,width=W).images
                    if True:
                        for x_sample in x_samples_ddim:
                            # x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            x_sample.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

    
if __name__ == "__main__":
    main()