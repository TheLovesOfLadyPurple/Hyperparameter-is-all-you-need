# Hyperparameters-are-all-you-need
the implementation of the paper 'Hyperparameters are all you need: Using five-step inference for an original diffusion model to generate images comparable to the latest distillation model.'
This repo implement a 8-step inference algorithm which FID performance is better than the results from the normal DPM++2m solver with 20-step inference. Meanwhile, it also support 5 and 6 step inference which generate result comparable with the latest diffussion distillation algorithm.
run the following to get the experiment result.  
```
python customed_timeschedule_sampler.py
python customed_timeschedule_sampler_xl.py
python customed_timeschedule_sampler_laion.py
python customed_timeschedule_sampler_xl_laion.py
```
 
## Requirements
This project use diffusers, which means you can simply install the environment by using pip install without confronting any conflict.  
```
conda env create --name hyper python=3.9
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt 
```
To find a proper version of torch, please use the following link:
https://pytorch.org/get-started/locally/

In this project, we also provide a upgrade implementation, which ultilize the golden noise to generate the images.  The checkpoint is in: https://1drv.ms/u/c/4e158dd7b255cd87/EaI2QngMC_lArhWGcjG5v7ABSm-3z8-Tm_sd2dN5nNIAYQ?e=tNKvzR and https://1drv.ms/u/c/4e158dd7b255cd87/EYzPIaAnN9dEpmxvHfys7M0Bv8_qsIGdt9wMf5yosMNq2w?e=t5Fd6b .  To run the code properly, you should also download coco 2014 and coco 2017 dataset from https://cocodataset.org/#home 
The vae of the sdxl is in https://huggingface.co/madebyollin/sdxl-vae-fp16-fix .

## Text to Image
using the following command to generate images from the new solver:
```
python txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --n_samples 4 --n_iter 4 --scale 5.0  --stop_steps 8
```

using the following command to generate images from the original solver:
```
python txt2imgOrg.py --prompt "a virus monster is playing guitar, oil on canvas" --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
```

## Text to Image ACGN version
using the following command to generate images from the new solver:
![inpainting](gen_img_val_v15/samples-customed-8-free-notNPNet-full-trick-7.5/00002.png =192)
```
python txt2imgACGN.py --prompt "((masterpiece,best quality)) , 1girl, ((school uniform)),brown blazer, black skirt,small breasts,necktie,red plaid skirt,looking at viewer" --ddim_steps 20 --n_samples 4 --n_iter 1 --scale 7.5 --W 768 --H 1024 --use_free
```

using the following command to generate images from the original solver:

![inpainting](gen_img_val_v15/samples-org-20-free-notNPNet/00002.png =192)
```
 python txt2imgOrg.py --prompt "((masterpiece,best quality)) , 1girl, ((school uniform)),brown blazer, black skirt,small breasts,necktie,red plaid skirt,looking at viewer" --ddim_steps 20 --n_samples 4 --n_iter 1 --scale 7.5 --W 768 --H 1024 --use_free --is_acgn
```