import pandas as pd
import requests
import os
from urllib.parse import urlparse
import time
from pathlib import Path
import hashlib
import numpy as np

def save_urls_and_texts(parquet_file, urls_file='urls_10k.txt', texts_file='texts_10k.txt', sample_size=10000):
    """Extract URLs and texts from parquet file and save to separate files"""
    print("Loading parquet file...")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} rows")
    
    # Randomly sample specified number of rows
    actual_sample_size = min(sample_size, len(df))
    print(f"Randomly selecting {actual_sample_size} rows...")
    np.random.seed(42)  # For reproducible results
    df_sample = df.sample(n=actual_sample_size, random_state=42).reset_index(drop=True)
    print(f"Selected {len(df_sample)} rows")
    
    # Save URLs
    print(f"Saving URLs to {urls_file}...")
    with open(urls_file, 'w', encoding='utf-8') as f:
        for i, url in enumerate(df_sample['URL']):
            f.write(f"{i:05d}: {url}\n")
    
    # Save texts
    print(f"Saving texts to {texts_file}...")
    with open(texts_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(df_sample['TEXT']):
            f.write(f"Image {i:05d}: {text}\n\n")
    
    print(f"Saved {len(df_sample)} URLs and texts")
    return df_sample

def get_file_extension(url):
    """Extract file extension from URL"""
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(('.jpg', '.jpeg')):
        return '.jpg'
    elif path.endswith('.png'):
        return '.png'
    elif path.endswith('.gif'):
        return '.gif'
    elif path.endswith('.webp'):
        return '.webp'
    else:
        return '.jpg'  # Default to jpg

def download_images(df, download_dir='downloaded_images', max_images=None, start_index=0):
    """Download images from URLs"""
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    # Determine how many images to download
    total_images = len(df) if max_images is None else min(max_images, len(df) - start_index)
    end_index = start_index + total_images if max_images else len(df)
    
    print(f"Starting download of {total_images} images from index {start_index} to {end_index-1}...")
    
    downloaded = 0
    failed = 0
    
    # Create session for connection reuse
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    for i in range(start_index, end_index):
        try:
            url = df.iloc[i]['URL']
            text = df.iloc[i]['TEXT']
            
            # Generate filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            extension = get_file_extension(url)
            filename = f"image_{i:05d}_{url_hash}{extension}"
            filepath = os.path.join(download_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(filepath):
                print(f"Skipping {i+1}/{end_index}: {filename} already exists")
                downloaded += 1
                continue
            
            # Download image
            response = session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check if content is actually an image
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                print(f"Skipping {i+1}/{end_index}: Not an image content type: {content_type}")
                failed += 1
                continue
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Save corresponding text file
            text_filename = f"image_{i:05d}_{url_hash}.txt"
            text_filepath = os.path.join(download_dir, text_filename)
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write(f"Text: {text}\n")
            
            downloaded += 1
            
            if downloaded % 10 == 0:
                print(f"Downloaded {downloaded}/{total_images} images...")
            
            # Small delay to be respectful to servers
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed to download image {i+1}: {str(e)}")
            failed += 1
            continue
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Failed downloads: {failed}")
    print(f"Images saved to: {download_dir}")

def main():
    parquet_file = 'part-00127-cad4a140-cebd-46fa-b874-e8968f93e32e-c000.snappy.parquet'
    
    # Check if parquet file exists
    if not os.path.exists(parquet_file):
        print(f"Error: Parquet file '{parquet_file}' not found!")
        return
    
    # Step 1: Extract and save URLs and texts (10,000 samples)
    print("=== Step 1: Extracting URLs and texts (10,000 samples) ===")
    df = save_urls_and_texts(parquet_file, sample_size=10000)
    
    # Step 2: Download images
    print("\n=== Step 2: Downloading images ===")
    
    # Ask user how many images to download
    total_images = len(df)
    print(f"Total images available: {total_images}")
    
    while True:
        try:
            choice = input(f"How many images would you like to download? (Enter number, 'all' for all {total_images}, or 'quit' to exit): ").strip().lower()
            
            if choice == 'quit':
                print("Exiting...")
                return
            elif choice == 'all':
                max_images = None
                break
            else:
                max_images = int(choice)
                if max_images <= 0:
                    print("Please enter a positive number.")
                    continue
                if max_images > total_images:
                    print(f"Cannot download more than {total_images} images.")
                    continue
                break
        except ValueError:
            print("Please enter a valid number, 'all', or 'quit'.")
            continue
    
    # Ask for starting index
    start_index = 0
    while True:
        try:
            start_input = input(f"Starting index (0 to {total_images-1}, default 0): ").strip()
            if not start_input:
                start_index = 0
                break
            start_index = int(start_input)
            if 0 <= start_index < total_images:
                break
            else:
                print(f"Please enter a number between 0 and {total_images-1}.")
        except ValueError:
            print("Please enter a valid number.")
            continue
    
    download_images(df, max_images=max_images, start_index=start_index)

if __name__ == "__main__":
    main()
