from pytorch_fid import fid_score
from basicsr.utils import scandir_SIDD
from PIL import Image
import numpy as np
import os

def resize_image(image_path, target_size):
    image = Image.open(image_path)
    return image.resize(target_size, Image.Resampling.BICUBIC)

def pad_image(image, size, fill_color=(0, 0, 0)):
    new_image = Image.new("RGB", size, fill_color)
    new_image.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return new_image

def preprocess_and_save_images(image_dir, target_size, output_dir):
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else: 
        return 
    
    # 处理每张图像并保存到输出目录
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        processed_image = resize_image(image_path, target_size)
        
        # 生成保存路径
        output_path = os.path.join(output_dir, image_name)
        
        # 保存图像
        processed_image.save(output_path)


if __name__ == '__main__':
    # LAION
    # gen_img_path = "D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\naf_dir_p_deblur_distill_all_cls_from_f6to8_laion" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\sd_org_befor_distill_all_cls" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\naf_dir_p_deblur_distill_all_cls\\denoised_group_12" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\png_naf_dir_p_deblur_distill\\denoised_group_11"
    # org_img_path = "D:\\research_project\\k-stable-diffusion\\downloaded_laion_images" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\scls_coco_img_val" #"D:\\research_project\\archive(2)\\coco2014\\images\\val2014" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\scls_coco_img_val"
    # output_path = "D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\crops_org_laion_all_cls_random" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\crops_org_coco_all_cls" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\crops_org_coco"
    #COCO
    gen_img_path = "./free/samples-org-20-7.5-v15"
    org_img_path = "./coco_val" 
    output_path = "./crops_org_coco_bicubic" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\crops_org_coco_all_cls" #"D:\\research_project\\k-stable-diffusion\\outputs\\txt2img-samples\\crops_org_coco"
    # output_gen_path = "./gen_img_val/idddXL6"
    preprocess_and_save_images(image_dir=org_img_path,target_size=(512,512),output_dir=output_path)
    # preprocess_and_save_images(image_dir=gen_img_path,target_size=(512,512),output_dir=output_gen_path)
    
    score = fid_score.calculate_fid_given_paths(paths=[gen_img_path, output_path],device='cuda:0',dims=2048,batch_size=128)
    print(score)