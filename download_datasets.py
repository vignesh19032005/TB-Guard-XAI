# Dataset Downloader for TB-Guard-XAI
# Downloads and organizes multiple TB datasets

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import kaggle

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets_raw"
DATA_DIR.mkdir(exist_ok=True)

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_shenzhen_montgomery():
    """Download Shenzhen and Montgomery TB datasets"""
    print("\n📦 Downloading Shenzhen & Montgomery datasets...")
    
    # These are from NIH
    urls = {
        "shenzhen": "https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets",
        "montgomery": "https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets"
    }
    
    print("⚠️  Manual download required:")
    print("1. Visit: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets")
    print("2. Download 'Shenzhen Hospital X-ray Set'")
    print("3. Download 'Montgomery County X-ray Set'")
    print(f"4. Extract to: {DATA_DIR}/shenzhen/ and {DATA_DIR}/montgomery/")
    print("\nPress Enter when done...")
    input()

def download_tbx11k():
    """Download TBX11K dataset"""
    print("\n📦 Downloading TBX11K dataset...")
    
    print("⚠️  Manual download required:")
    print("1. Visit: https://mmcheng.net/tb/")
    print("2. Download TBX11K dataset")
    print(f"3. Extract to: {DATA_DIR}/tbx11k/")
    print("\nPress Enter when done...")
    input()

def download_kaggle_tb():
    """Download Kaggle TB dataset"""
    print("\n📦 Downloading Kaggle TB Chest X-ray dataset...")
    
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
            path=DATA_DIR / "kaggle_tb",
            unzip=True
        )
        print("✅ Kaggle TB dataset downloaded")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Manual download:")
        print("1. Visit: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        print(f"2. Download and extract to: {DATA_DIR}/kaggle_tb/")

def download_nih_chestxray():
    """Download NIH ChestX-ray14"""
    print("\n📦 Downloading NIH ChestX-ray14...")
    
    print("⚠️  Large dataset - Manual download recommended:")
    print("1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC")
    print("2. Download images (112GB total)")
    print(f"3. Extract to: {DATA_DIR}/nih_chestxray/")
    print("\nSkip for now? (y/n): ")
    skip = input().lower()
    if skip == 'y':
        print("⏭️  Skipping NIH ChestX-ray14")

def download_padchest():
    """Download PadChest"""
    print("\n📦 PadChest dataset...")
    
    print("⚠️  Very large dataset - Manual download:")
    print("1. Visit: http://bimcv.cipf.es/bimcv-projects/padchest/")
    print("2. Request access and download")
    print(f"3. Extract to: {DATA_DIR}/padchest/")
    print("\nSkip for now? (y/n): ")
    skip = input().lower()
    if skip == 'y':
        print("⏭️  Skipping PadChest")

def main():
    print("="*60)
    print("TB-Guard-XAI Dataset Downloader")
    print("="*60)
    
    print("\n📋 Datasets to download:")
    print("1. Shenzhen TB (662 images)")
    print("2. Montgomery TB (138 images)")
    print("3. TBX11K (~11k images)")
    print("4. Kaggle TB (~4k images)")
    print("5. NIH ChestX-ray14 (112k images) - Optional")
    print("6. PadChest (160k images) - Optional")
    
    print("\n⚠️  Note: Some datasets require manual download due to licensing")
    print("\nContinue? (y/n): ")
    
    if input().lower() != 'y':
        print("❌ Cancelled")
        return
    
    download_shenzhen_montgomery()
    download_tbx11k()
    download_kaggle_tb()
    download_nih_chestxray()
    download_padchest()
    
    print("\n✅ Dataset download instructions complete!")
    print(f"\n📁 Raw datasets location: {DATA_DIR}")
    print("\n🔄 Next step: Run 'python prepare_datasets.py' to organize and split data")

if __name__ == "__main__":
    main()
