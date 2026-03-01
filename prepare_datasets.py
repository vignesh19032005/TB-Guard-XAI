# Dataset Preparation and Splitting
# Organizes multiple datasets into train/val/test splits
# Handles ALL 5 datasets (~16k images total)

import os
import shutil
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "datasets_raw"
PROCESSED_DIR = BASE_DIR / "datasets_processed"

# Create structure
for split in ['train', 'val', 'test']:
    for cls in ['TB', 'Normal']:
        (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def collect_shenzhen():
    """Collect Shenzhen dataset using metadata CSV"""
    print("\n📂 Processing Shenzhen dataset...")
    
    shenzhen_dir = RAW_DIR / "datasets_shenzhe_Monto" / "shenzhen"
    if not shenzhen_dir.exists():
        print("⚠️  Shenzhen not found, skipping")
        return [], []
    
    tb_images = []
    normal_images = []
    
    # Read metadata CSV
    metadata_file = shenzhen_dir / "shenzhen_metadata.csv"
    images_dir = shenzhen_dir / "images" / "images"
    
    if metadata_file.exists() and images_dir.exists():
        df = pd.read_csv(metadata_file)
        for _, row in df.iterrows():
            img_path = images_dir / row['study_id']
            if img_path.exists():
                findings = str(row['findings']).lower()
                # TB cases have 'tb', 'ptb', 'stb' in findings
                if 'tb' in findings or 'ptb' in findings or 'stb' in findings:
                    tb_images.append(img_path)
                elif 'normal' in findings:
                    normal_images.append(img_path)
    
    print(f"  TB: {len(tb_images)}, Normal: {len(normal_images)}")
    return tb_images, normal_images

def collect_montgomery():
    """Collect Montgomery dataset using metadata CSV"""
    print("\n📂 Processing Montgomery dataset...")
    
    mont_dir = RAW_DIR / "datasets_shenzhe_Monto" / "montogomery"
    if not mont_dir.exists():
        print("⚠️  Montgomery not found, skipping")
        return [], []
    
    tb_images = []
    normal_images = []
    
    # Read metadata CSV
    metadata_file = mont_dir / "montgomery_metadata.csv"
    images_dir = mont_dir / "images" / "images"
    
    if metadata_file.exists() and images_dir.exists():
        df = pd.read_csv(metadata_file)
        for _, row in df.iterrows():
            img_path = images_dir / row['study_id']
            if img_path.exists():
                findings = str(row['findings']).lower()
                # TB cases have 'tb' in findings
                if 'tb' in findings:
                    tb_images.append(img_path)
                elif 'normal' in findings:
                    normal_images.append(img_path)
    
    print(f"  TB: {len(tb_images)}, Normal: {len(normal_images)}")
    return tb_images, normal_images

def collect_kaggle_tb():
    """Collect Kaggle TB dataset"""
    print("\n📂 Processing Kaggle TB dataset...")
    
    kaggle_dir = RAW_DIR / "kaggle_tb" / "TB_Chest_Radiography_Database"
    if not kaggle_dir.exists():
        print("⚠️  Kaggle TB not found, skipping")
        return [], []
    
    tb_images = []
    normal_images = []
    
    # Kaggle structure: Tuberculosis/ and Normal/ folders
    tb_folder = kaggle_dir / "Tuberculosis"
    normal_folder = kaggle_dir / "Normal"
    
    if tb_folder.exists():
        tb_images.extend(list(tb_folder.glob("*.png")))
        tb_images.extend(list(tb_folder.glob("*.jpg")))
        tb_images.extend(list(tb_folder.glob("*.jpeg")))
    
    if normal_folder.exists():
        normal_images.extend(list(normal_folder.glob("*.png")))
        normal_images.extend(list(normal_folder.glob("*.jpg")))
        normal_images.extend(list(normal_folder.glob("*.jpeg")))
    
    print(f"  TB: {len(tb_images)}, Normal: {len(normal_images)}")
    return tb_images, normal_images

def collect_tb_cxr_dataset():
    """Collect 'Dataset of Tuberculosis Chest X-rays Images' dataset
    
    Structure:
        Dataset of Tuberculosis Chest X-rays Images/
            Dataset of Tuberculosis Chest X-rays Images/
                TB Chest X-rays/       -> TB.1.jpg, TB.2.jpg, ...  (2494 images)
                Normal Chest X-rays/   -> others (1).jpg, ...       (514 images)
    """
    print("\n📂 Processing TB Chest X-rays Images dataset...")
    
    dataset_dir = RAW_DIR / "Dataset of Tuberculosis Chest X-rays Images" / "Dataset of Tuberculosis Chest X-rays Images"
    if not dataset_dir.exists():
        print("⚠️  TB Chest X-rays Images dataset not found, skipping")
        return [], []
    
    tb_images = []
    normal_images = []
    
    tb_folder = dataset_dir / "TB Chest X-rays"
    normal_folder = dataset_dir / "Normal Chest X-rays"
    
    if tb_folder.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            tb_images.extend(list(tb_folder.glob(ext)))
    
    if normal_folder.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            normal_images.extend(list(normal_folder.glob(ext)))
    
    print(f"  TB: {len(tb_images)}, Normal: {len(normal_images)}")
    return tb_images, normal_images

def collect_tbx11k():
    """Collect TBX11K dataset using annotation JSON files
    
    Tag categories:
        - healthy        -> Normal  (3800 images)
        - sick_but_non-tb -> Normal (3800 images) — included to maximize dataset
        - active_tb      -> TB      (630 images)
        - latent_tb      -> TB      (139 images)
        - active&latent_tb -> TB    (30 images)
        - uncertain_tb   -> Skipped (0 images)
    """
    print("\n📂 Processing TBX11K dataset...")
    
    tbx_dir = RAW_DIR / "tbx11k-DatasetNinja"
    if not tbx_dir.exists():
        print("⚠️  TBX11K not found, skipping")
        return [], []
    
    tb_images = []
    normal_images = []
    
    # Tags that indicate TB
    tb_tags = {'active_tb', 'latent_tb', 'active&latent_tb'}
    # Tags that indicate Normal (including sick_but_non-tb for maximizing data)
    normal_tags = {'healthy', 'sick_but_non-tb'}
    
    # Process train, val, test folders
    for subset in ['train', 'val', 'test']:
        img_dir = tbx_dir / subset / "img"
        ann_dir = tbx_dir / subset / "ann"
        
        if not img_dir.exists() or not ann_dir.exists():
            continue
        
        # Read each annotation file
        for ann_file in ann_dir.glob("*.json"):
            try:
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            
            # Get image filename (remove .json -> leaves e.g. "h0001.png")
            img_name = ann_file.stem
            img_path = img_dir / img_name
            
            if not img_path.exists():
                continue
            
            # Check tags to determine if TB or Normal
            tags = ann_data.get('tags', [])
            tag_names = {tag['name'] for tag in tags}
            
            # Classify based on tags
            if tag_names & tb_tags:
                tb_images.append(img_path)
            elif tag_names & normal_tags:
                normal_images.append(img_path)
            # Skip 'uncertain_tb' and any unrecognized tags
    
    print(f"  TB: {len(tb_images)}, Normal: {len(normal_images)}")
    return tb_images, normal_images

def collect_covid_radiography():
    """
    Collect subset of COVID-19 Radiography Database (Lung Opacity & Normal).
    Lung Opacity and Normal are both labeled as 'Normal' (i.e. Non-TB)
    for this binary TB classifier.
    """
    db_dir = RAW_DIR / "covid19_radiography" / "COVID-19_Radiography_Dataset"
    
    tb_images = []
    normal_images = []
    
    if not db_dir.exists():
        print(f"⚠️ Skipping COVID-19 Radiography Database: {db_dir} not found")
        return tb_images, normal_images
        
    print(f"\n📂 Processing COVID-19 Radiography Database...")
    
    # 1. Normal Cases (Sample 1500 to keep balance)
    normal_dir = db_dir / "Normal" / "images"
    if normal_dir.exists():
        normal_paths = list(normal_dir.glob("*.png"))
        # Take first 1500
        normal_images.extend(normal_paths[:1500])
        
    # 2. Lung Opacity Cases (Sample 3000 to heavily train the model to reject non-TB pneumonia/opacity)
    opacity_dir = db_dir / "Lung_Opacity" / "images"
    if opacity_dir.exists():
        opacity_paths = list(opacity_dir.glob("*.png"))
        # Take first 3000
        normal_images.extend(opacity_paths[:3000])
    
    print(f"  TB: {len(tb_images)}, Normal (Non-TB): {len(normal_images)}")
    return tb_images, normal_images

def split_and_copy(tb_images, normal_images, dataset_name):
    """Split dataset and copy to processed folder"""
    if not tb_images and not normal_images:
        return
        
    print(f"\n🔀 Splitting {dataset_name}...")
    print(f"  TB: {len(tb_images)}, Normal (Non-TB): {len(normal_images)}")
    
    # 70% train, 15% val, 15% test
    # Needs at least some images to split properly
    if len(tb_images) > 0:
        tb_train, tb_temp = train_test_split(tb_images, test_size=0.3, random_state=42)
        tb_val, tb_test = train_test_split(tb_temp, test_size=0.5, random_state=42)
    else:
        tb_train, tb_val, tb_test = [], [], []
        
    if len(normal_images) > 0:
        norm_train, norm_temp = train_test_split(normal_images, test_size=0.3, random_state=42)
        norm_val, norm_test = train_test_split(norm_temp, test_size=0.5, random_state=42)
    else:
        norm_train, norm_val, norm_test = [], [], []
    
    # Copy files
    splits = [
        ('train', tb_train, norm_train),
        ('val', tb_val, norm_val),
        ('test', tb_test, norm_test)
    ]
    
    for split_name, tb_list, norm_list in splits:
        for img_path in tqdm(tb_list, desc=f"Copying TB to {split_name}"):
            if img_path.exists():
                dst = PROCESSED_DIR / split_name / "TB" / f"{dataset_name}_{img_path.name}"
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                    
        for img_path in tqdm(norm_list, desc=f"Copying Normal to {split_name}"):
            if img_path.exists():
                dst = PROCESSED_DIR / split_name / "Normal" / f"{dataset_name}_{img_path.name}"
                if not dst.exists():
                    shutil.copy2(img_path, dst)

def main():
    print("="*60)
    print("DATASET PREPARATION")
    print("="*60)
    
    datasets_results = {
        "Shenzhen": collect_shenzhen(),
        "Montgomery": collect_montgomery(),
        "KaggleTB": collect_kaggle_tb(),
        "TBCXR": collect_tb_cxr_dataset(),
        "TBX11K": collect_tbx11k(),
        "COVID19_Radiography": collect_covid_radiography()
    }
    
    total_tb = 0
    total_normal = 0
    
    for name, (tb, normal) in datasets_results.items():
        total_tb += len(tb)
        total_normal += len(normal)
        if tb or normal:
            split_and_copy(tb, normal, name)
    
    print(f"\n📊 Total collected: TB={total_tb}, Normal={total_normal}, Grand Total={total_tb + total_normal}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics (Processed)")
    print("="*60)
    
    stats = {}
    grand_total = 0
    for split in ['train', 'val', 'test']:
        tb_count = len(list((PROCESSED_DIR / split / "TB").glob("*")))
        normal_count = len(list((PROCESSED_DIR / split / "Normal").glob("*")))
        total = tb_count + normal_count
        grand_total += total
        stats[split] = {'TB': tb_count, 'Normal': normal_count, 'Total': total}
        print(f"\n{split.upper()}:")
        print(f"  TB: {tb_count}")
        print(f"  Normal: {normal_count}")
        print(f"  Total: {total}")
    
    stats['grand_total'] = grand_total
    print(f"\n🎯 GRAND TOTAL: {grand_total} images")
    
    # Save stats
    with open(PROCESSED_DIR / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Processed data: {PROCESSED_DIR}")
    print("\n🔄 Next step: Run 'python train_ensemble.py' to start training")

if __name__ == "__main__":
    main()
