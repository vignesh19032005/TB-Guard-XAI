import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LungPreprocessor:
    """Advanced Preprocessing pipeline for chest X-rays (Phase 4 Arch)"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
    
    def remove_artifacts_and_segment(self, image):
        """Remove text artifacts, borders, and segment lung field using Otsu & Morphological ops"""
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # 1. Edge cropping (remove typical artifact zones)
        h, w = gray.shape
        border = int(min(h, w) * 0.03)
        cropped = gray[border:h-border, border:w-border]
        
        # 2. Basic Lung Segmentation (Thresholding + Morphology)
        # Apply slight blur to remove noise
        blur = cv2.GaussianBlur(cropped, (5, 5), 0)
        
        # Otsu's thresholding to separate lungs (dark) from tissue (bright)
        # Note: Lungs are usually dark in X-rays
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert so lungs are white (lungs are darker than tissue usually, so OTSU might make them black. If so invert)
        # Lungs are dark, so thresholding usually makes dark areas 0. 
        # We invert so lungs are 255 (white)
        mask = cv2.bitwise_not(thresh)
        
        # Morphological opening to remove small noise (like text/markers)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Morphological closing to fill holes in lungs (like heart shadow)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Smooth mask
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Apply mask but keep some context (blend with original)
        mask_float = mask.astype(float) / 255.0
        segmented = cropped * mask_float + cropped * 0.2 * (1 - mask_float) # Keep 20% background context
        segmented = segmented.astype(np.uint8)
        
        # Resize back to target size to standardize before CLAHE
        standardized = cv2.resize(segmented, (self.image_size, self.image_size))
        return standardized
    
    def apply_clahe(self, image):
        """Apply CLAHE for contrast enhancement"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def normalize_intensity(self, image):
        """Normalize intensity values out of extreme percentiles"""
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip(image, p2, p98)
        image = ((image - image.min()) / (max(1e-8, image.max() - image.min())) * 255).astype(np.uint8)
        return image
    
    def preprocess(self, image_path):
        """Full preprocessing pipeline"""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        image = self.remove_artifacts_and_segment(image)
        image = self.normalize_intensity(image)
        image = self.apply_clahe(image)
        
        return image

def get_train_transforms(image_size=224):
    """Training augmentations - Advanced Medical Safe (Phase 5 Arch)"""
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.Normalize(mean=0.485, std=0.229, max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_val_transforms(image_size=224):
    """Validation transforms"""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=0.485, std=0.229, max_pixel_value=255.0),
        ToTensorV2()
    ])

class PreprocessedDataset(torch.utils.data.Dataset):
    """Dataset with preprocessing"""
    def __init__(self, image_paths, labels, transforms=None, use_preprocessing=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.use_preprocessing = use_preprocessing
        self.preprocessor = LungPreprocessor() if use_preprocessing else None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        if self.use_preprocessing:
            image = self.preprocessor.preprocess(img_path)
        else:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (224, 224))
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            
        if image.shape[0] == 3:
            image = image.mean(dim=0, keepdim=True)
        elif image.dim() == 2:
            image = image.unsqueeze(0)
            
        return image, label
