# download_covid_dataset.py
# Phase 3: Add "Other Lung Disease" class to eliminate false positives

import os
from pathlib import Path
import subprocess
import zipfile

# Point Kaggle to our local directory where kaggle.json is stored
os.environ['KAGGLE_CONFIG_DIR'] = str(Path().absolute())

DATASET_NAME = "tawsifurrahman/covid19-radiography-database"
DOWNLOAD_DIR = Path("datasets_raw/covid19_radiography")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def download_and_extract():
    print("="*60)
    print(f"DOWNLOADING {DATASET_NAME}")
    print("="*60)
    
    zip_path = DOWNLOAD_DIR / "covid19-radiography-database.zip"
    
    if zip_path.exists() or (DOWNLOAD_DIR / "COVID-19_Radiography_Dataset").exists():
        print("Dataset already downloaded or extracted!")
    else:
        # Download
        print("📥 Downloading via Kaggle API (Approx 700MB)...")
        subprocess.run([
            "venv/Scripts/kaggle", "datasets", "download", "-d", DATASET_NAME,
            "-p", str(DOWNLOAD_DIR)
        ], check=True)
        
    print("📦 Extracting specific classes (Normal & Lung Opacity)...")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Only extract what we need to save disk space
            members = zip_ref.namelist()
            to_extract = [m for m in members if "Lung_Opacity" in m or "Normal" in m]
            
            print(f"Found {len(to_extract)} relevant images out of {len(members)}. Extracting...")
            zip_ref.extractall(DOWNLOAD_DIR, members=to_extract)
            
        print("🗑️ Removing zip file...")
        os.remove(zip_path)
        
    print("\n✅ COVID-19 Radiography Database acquired successfully!")
    print("Next step: integrate 'Lung Opacity' images as the 'Other' class in dataset processing.")

if __name__ == "__main__":
    download_and_extract()
