import os
import glob
from PIL import Image
import numpy as np
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import argparse

# Initialize CLIP score function
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch32")

def load_texts_from_folder(text_folder):
    """Load text prompts from text files."""
    texts = []
    text_files = sorted(glob.glob(os.path.join(text_folder, "*.txt")))
    
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            texts.append(text)
    
    return texts

def get_image_files(image_folder):
    """Get sorted list of image files."""
    return sorted(glob.glob(os.path.join(image_folder, "*.png")))

def load_image_batch(image_files, batch_start, batch_size):
    """Load a batch of images from file paths."""
    batch_end = min(batch_start + batch_size, len(image_files))
    images = []
    
    for i in range(batch_start, batch_end):
        img = Image.open(image_files[i]).convert('RGB')
        # Convert to numpy array with values in [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0  # Use float32 to save memory
        images.append(img_array)
    
    return np.array(images, dtype=np.float32)

def calculate_clip_score_batch(images, prompts):
    """Calculate CLIP score for a batch of images and prompts."""
    # Convert images to uint8 (0-255 range)
    images_int = (images * 255).astype("uint8")
    
    # Convert to torch tensor and rearrange dimensions from (B, H, W, C) to (B, C, H, W)
    images_tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2).to(device='cuda')
    
    # Calculate CLIP score
    score = clip_score_fn(images_tensor, prompts).detach()
    return score

def calculate_clip_score(image_files, prompts, batch_size=50):
    """Calculate CLIP score for images and prompts using batching."""
    total_score = 0.0
    total_samples = 0
    
    print(f"Processing {len(image_files)} images in batches of {batch_size}...")
    
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_prompts = prompts[batch_start:batch_end]
        
        # Load batch of images
        batch_images = load_image_batch(image_files, batch_start, batch_size)
        
        # Calculate CLIP score for this batch
        batch_score = calculate_clip_score_batch(batch_images, batch_prompts)
        
        # Accumulate weighted score
        batch_size_actual = batch_end - batch_start
        total_score += float(batch_score) * batch_size_actual
        total_samples += batch_size_actual
        
        print(f"Processed batch {batch_start//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}, "
              f"batch score: {float(batch_score):.4f}")
        
        # Clear memory
        del batch_images
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return average score
    average_score = total_score / total_samples if total_samples > 0 else 0.0
    return round(average_score, 4)

def main():
    parser = argparse.ArgumentParser(description='Calculate CLIP score for images and text pairs')
    parser.add_argument('--image_folder', type=str, required=False, default='D:\\2017val6Real\\gen_img_val_v15_2017\\samples-iDDD6-retrain-free-notNPNet-full-trick-4-5.5',
                       help='Path to folder containing images (e.g., gen_img_val_v15/samples-iDDD5-retrain-free-full-trick-4-5)')
    parser.add_argument('--text_folder', type=str, default='saved_txts_2017',
                       help='Path to folder containing text files (default: saved_txts)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    print(f"Loading texts from: {args.text_folder}")
    print(f"Loading images from: {args.image_folder}")
    
    # Load texts and get image file paths
    texts = load_texts_from_folder(args.text_folder)
    image_files = get_image_files(args.image_folder)
    
    print(f"Loaded {len(texts)} texts and found {len(image_files)} images")
    
    # Ensure we have matching numbers of texts and images
    min_samples = min(len(texts), len(image_files))
    if args.max_samples:
        min_samples = min(min_samples, args.max_samples)
    
    texts = texts[:min_samples]
    image_files = image_files[:min_samples]
    
    print(f"Evaluating {min_samples} text-image pairs")
    
    # Calculate CLIP score with batching
    print("Calculating CLIP score...")
    clip_score_result = calculate_clip_score(image_files, texts, batch_size=200)
    
    print(f"CLIP score: {clip_score_result}")
    
    # Save results to file
    results_file = f"clip_score_results_{os.path.basename(args.image_folder)}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Image folder: {args.image_folder}\n")
        f.write(f"Text folder: {args.text_folder}\n")
        f.write(f"Number of samples: {min_samples}\n")
        f.write(f"CLIP score: {clip_score_result}\n")
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()