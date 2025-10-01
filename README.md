# Hyperparameters-are-all-you-need
the implementation of the paper 'Hyperparameters are all you need: Using five-step inference for an original diffusion model to generate images comparable to the latest distillation model.'
This repo implement a 8-step inference algorithm which FID performance is better than the results from the normal DPM++2m solver with 20-step inference. Meanwhile, it also support 5 and 6 step inference which generate result comparable with the latest diffussion distillation algorithm.
run the customed_timeschedule_sampler.py, customed_timeschedule_sampler_xl.py, customed_timeschedule_sampler_laion.py, customed_timeschedule_sampler_xl_laion.py to get the experiment result.  
 
## Requirements
This project use diffusers, which means you can simply install the environment by using pip install withoug confronting any conflict.  
```
conda env create --name hyper python=3.9
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt 
```
To find a proper version of torch, please use the following link:
https://pytorch.org/get-started/locally/

In this project, we also provide a upgrade implementation, which ultilize the golden noise to generate the images.  The checkpoint is in: https://1drv.ms/u/c/4e158dd7b255cd87/EaI2QngMC_lArhWGcjG5v7ABSm-3z8-Tm_sd2dN5nNIAYQ?e=tNKvzR and https://1drv.ms/u/c/4e158dd7b255cd87/EYzPIaAnN9dEpmxvHfys7M0Bv8_qsIGdt9wMf5yosMNq2w?e=t5Fd6b .  To run the code properly, you should also download coco 2014 and coco 2017 dataset from https://cocodataset.org/#home 
The vae of the sdxl is in https://huggingface.co/madebyollin/sdxl-vae-fp16-fix .