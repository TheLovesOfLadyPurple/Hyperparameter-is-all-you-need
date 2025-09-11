import pandas as pd
import requests
import os
from urllib.parse import urlparse
import time
import hashlib
import numpy as np

def extract_and_download_10k():
    """Extract 10,000 random samples and download images"""
    parquet_file = 'part-00127-cad4a140-cebd-46fa-b874-e8968f93e32e-c000.snappy.parquet'
    
    print("=== Step 1: Loading and sampling data ===")
    print("Loading parquet file...")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} rows")
    
    # Randomly sample 10,000 rows
    sample_size = min(10000, len(df))
    print(f"Randomly selecting {sample_size} rows...")
    np.random.seed(42)  # For reproducible results
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"Selected {len(df_sample)} rows")
    
    # Save URLs and texts
    print("Saving URLs and texts...")
    with open('sampled_urls_10k.txt', 'w', encoding='utf-8') as f:
        for i, url in enumerate(df_sample['URL']):
            f.write(f"{i:05d}: {url}\n")
    
    with open('sampled_texts_10k.txt', 'w', encoding='utf-8') as f:
        for i, text in enumerate(df_sample['TEXT']):
            f.write(f"Image {i:05d}: {text}\n\n")
    
    # Save CSV for easy viewing
    df_sample[['URL', 'TEXT']].to_csv('sampled_data_10k.csv', index=True, encoding='utf-8')
    print("Files saved: sampled_urls_10k.txt, sampled_texts_10k.txt, sampled_data_10k.csv")
    
    print("\n=== Step 2: Downloading images ===")
    download_images(df_sample)

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

def download_images(df, download_dir='downloaded_images_10k'):
    """Download images from URLs"""
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    total_images = len(df)
    print(f"Starting download of {total_images} images...")
    
    downloaded = 0
    failed = 0
    
    # Create session for connection reuse
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    for i in range(total_images):
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
                print(f"Skipping {i+1}/{total_images}: {filename} already exists")
                downloaded += 1
                continue
            
            # Download image with timeout
            response = session.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Check if content is actually an image
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                print(f"Skipping {i+1}/{total_images}: Not an image content type: {content_type}")
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
                f.write(f"Index: {i}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Text: {text}\n")
            
            downloaded += 1
            
            if downloaded % 50 == 0:
                print(f"Downloaded {downloaded}/{total_images} images...")
            
            # Small delay to be respectful to servers
            time.sleep(0.2)
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed for image {i+1}: {str(e)}")
            failed += 1
            continue
        except Exception as e:
            print(f"Failed to download image {i+1}: {str(e)}")
            failed += 1
            continue
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Failed downloads: {failed}")
    print(f"Images saved to: {download_dir}")

if __name__ == "__main__":
    extract_and_download_10k()
