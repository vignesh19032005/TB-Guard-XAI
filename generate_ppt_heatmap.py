import os
import cv2
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# TB-Guard specific imports
from ensemble_models import load_ensemble
from preprocessing import LungPreprocessor, get_val_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/ensemble_best.pth"

def select_image():
    root = tk.Tk()
    root.withdraw()
    
    # Try different ways to bring dialog to front
    try:
        root.attributes('-topmost', True)
    except tk.TclError:
        pass
        
    file_path = filedialog.askopenfilename(
        title="Select Chest X-ray Image for Heatmap Generation",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def generate_heatmap(image_path, model, preprocessor, save_dir):
    print(f"Processing image: {image_path}")
    
    # 1. Preprocess
    img_array = preprocessor.preprocess(image_path)
    transforms = get_val_transforms()
    augmented = transforms(image=img_array)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    # Ensure correct shape
    if image_tensor.shape[1] == 3:
        image_tensor = image_tensor.mean(dim=1, keepdim=True)
    elif image_tensor.shape[1] != 1:
        image_tensor = image_tensor[:, :1, :, :]
        
    # 2. Get Probability
    print("Running Model Prediction...")
    with torch.no_grad():
        mean_prob, std_prob = model.predict_with_uncertainty(image_tensor, n_samples=10)
        prob = mean_prob.item()
        
    prediction = "TB Positive" if prob >= 0.5 else "TB Negative"
    color = "red" if prob >= 0.5 else "blue"
    
    # 3. Generate Grad-CAM++
    print("Generating High-Clarity XAI Heatmap...")
    target_layer = model.densenet.model.features.denseblock4 # Standard target for TB-Guard DenseNet
    cam = GradCAMPlusPlus(model=model.densenet, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(0)])[0]
    
    # 4. Read original for plotting
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    h, w = original.shape[:2]
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    heatmap_resized = cv2.resize(grayscale_cam, (w, h))
    
    # To provide the 'absolute clarity' for PPT, we will create a high-quality side-by-side plot
    # and a high-quality single overlay image
    
    file_name = Path(image_path).stem
    out_overlay_path = save_dir / f"{file_name}_ppt_overlay.png"
    out_side_path = save_dir / f"{file_name}_ppt_sideby_side.png"
    
    # -- PLOT 1: High contrast JET transparent heatmap directly via Matplotlib --
    fig_side, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(original_rgb)
    ax1.set_title("Original Chest X-ray", fontsize=18, fontweight='bold', pad=15)
    ax1.axis('off')
    
    ax2.imshow(original_rgb)
    im2 = ax2.imshow(heatmap_resized, cmap='jet', alpha=0.45, vmin=0, vmax=1) # 45% transparency for balance
    ax2.set_title(f"Grad-CAM++ AI Attention\n{prediction} ({prob:.1%})", fontsize=18, fontweight='bold', color=color, pad=15)
    ax2.axis('off')
    
    # Add colorbar
    cbar = fig_side.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Focus Intensity', rotation=270, labelpad=25, fontsize=14)
    
    fig_side.tight_layout()
    fig_side.savefig(out_side_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig_side)
    
    # -- PLOT 2: Single Overlay with clearly visible Title for PPT -- 
    fig_overlay, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original_rgb)
    im_single = ax.imshow(heatmap_resized, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    ax.set_title(f"XAI Heatmap: {prediction} ({prob:.1%})", fontsize=20, fontweight='bold', color=color, pad=20)
    ax.axis('off')
    
    # Include colorbar here too as it makes PPT better
    cbar_ov = fig_overlay.colorbar(im_single, ax=ax, fraction=0.046, pad=0.04)
    cbar_ov.set_label('Attention Focus Intensity', rotation=270, labelpad=25, fontsize=14)
    
    fig_overlay.savefig(out_overlay_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig_overlay)
    
    print(f"\n✅ GENERATED HEATMAPS WITH ABSOLUTE CLARITY!")
    print(f"📸 Saved Side-by-side View: {out_side_path}")
    print(f"📸 Saved Full Overlay View: {out_overlay_path}\n")

    return out_side_path

def main():
    print("==============================================")
    print(" 🎨 PPT Heatmap Generator - TB Guard XAI 🎨 ")
    print("==============================================\n")
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ Could not find model at {MODEL_PATH}")
        sys.exit(1)
        
    print(f"Loading Ensemble Model onto {DEVICE}...")
    try:
        model = load_ensemble(MODEL_PATH, DEVICE)
        preprocessor = LungPreprocessor()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
        
    save_dir = Path("outputs/ppt_heatmaps")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nModel Loaded. Ready to Generate PPT Images!")
    
    while True:
        print("\n=> Opening File Selector...")
        img_path = select_image()
        
        if not img_path:
            print("No file selected. Exiting generator.")
            break
            
        try:
            generate_heatmap(img_path, model, preprocessor, save_dir)
        except Exception as e:
            print(f"❌ Failed to process image: {e}")
            import traceback
            traceback.print_exc()
            
        ans = input("Process another image? (y/n): ")
        if ans.lower() != 'y':
            break

if __name__ == "__main__":
    main()
