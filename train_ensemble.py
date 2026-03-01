# Train Ensemble Model with Multi-Dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import numpy as np

from ensemble_models import TBEnsemble
from preprocessing import PreprocessedDataset, get_train_transforms, get_val_transforms

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # TBEnsemble.forward() already applies sigmoid, so inputs are probabilities
        # Use binary_cross_entropy (NOT _with_logits) to avoid double-sigmoid
        inputs = inputs.clamp(1e-7, 1 - 1e-7)  # numerical stability
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Config
PROCESSED_DIR = Path("datasets_processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Training parameters
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

print(f"Device: {DEVICE}")

def load_dataset_split(split_dir):
    """Load images and labels from split directory"""
    image_paths = []
    labels = []
    
    # TB images
    tb_dir = split_dir / "TB"
    for img_path in tb_dir.glob("*"):
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(img_path)
            labels.append(1)
    
    # Normal images
    normal_dir = split_dir / "Normal"
    for img_path in normal_dir.glob("*"):
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(img_path)
            labels.append(0)
    
    return image_paths, labels

def train_epoch(model, loader, criterion, optimizer, scaler):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)
        
        optimizer.zero_grad()
        
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images).squeeze()
            # Compute loss outside autocast (binary_cross_entropy isn't autocast-safe)
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, threshold=0.5):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images).squeeze()
            # Ensemble already applies sigmoid, outputs are probabilities
            probs = outputs.cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    preds_binary = (all_preds > threshold).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds_binary, average='binary'
    )
    
    return {
        'accuracy': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }

def find_best_threshold(predictions, labels):
    """Find optimal threshold"""
    best_thresh = 0.5
    best_f1 = 0
    
    for thresh in np.arange(0.3, 0.7, 0.01):
        preds_binary = (predictions > thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds_binary, average='binary'
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1

def main():
    print("="*60)
    print("TB-Guard-XAI Ensemble Training")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_paths, train_labels = load_dataset_split(PROCESSED_DIR / "train")
    val_paths, val_labels = load_dataset_split(PROCESSED_DIR / "val")
    test_paths, test_labels = load_dataset_split(PROCESSED_DIR / "test")
    
    print(f"Train: {len(train_paths)} images")
    print(f"Val: {len(val_paths)} images")
    print(f"Test: {len(test_paths)} images")
    
    # Create datasets
    train_dataset = PreprocessedDataset(
        train_paths, train_labels,
        transforms=get_train_transforms(IMAGE_SIZE),
        use_preprocessing=True
    )
    val_dataset = PreprocessedDataset(
        val_paths, val_labels,
        transforms=get_val_transforms(IMAGE_SIZE),
        use_preprocessing=True
    )
    test_dataset = PreprocessedDataset(
        test_paths, test_labels,
        transforms=get_val_transforms(IMAGE_SIZE),
        use_preprocessing=True
    )
    
    # Calculate class distribution
    num_tb = sum(train_labels)
    num_normal = len(train_labels) - num_tb
    
    print(f"\nClass distribution - TB: {num_tb}, Normal: {num_normal}")
    print(f"Imbalance ratio: 1:{num_normal/num_tb:.2f}")
    print("Using Focal Loss to handle class imbalance")
    
    # Create dataloaders - use regular shuffle, no weighted sampling
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    
    # Initialize model
    print("\nInitializing ensemble model...")
    model = TBEnsemble(use_mc_dropout=True).to(DEVICE)
    
    # Use Focal Loss with alpha tuned for class imbalance
    # alpha=0.75 gives higher weight to TB (minority class) to combat imbalance
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    
    # Training loop
    best_val_auc = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        
        # Validate
        val_metrics = evaluate(model, val_loader)
        
        # Update scheduler
        scheduler.step(val_metrics['auc'])
        
        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        
        # Calculate per-class accuracy for better insight
        val_preds_binary = (val_metrics['predictions'] > 0.5).astype(int)
        val_labels_array = val_metrics['labels']
        tb_mask = val_labels_array == 1
        normal_mask = val_labels_array == 0
        tb_acc = (val_preds_binary[tb_mask] == val_labels_array[tb_mask]).mean()
        normal_acc = (val_preds_binary[normal_mask] == val_labels_array[normal_mask]).mean()
        print(f"Val TB Accuracy: {tb_acc:.4f}, Normal Accuracy: {normal_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append({
            'accuracy': val_metrics['accuracy'],
            'auc': val_metrics['auc'],
            'f1': val_metrics['f1']
        })
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), MODELS_DIR / "ensemble_best.pth")
            print("Best model saved!")
    
    # Threshold tuning
    print("\n" + "="*60)
    print("Threshold Tuning")
    print("="*60)
    
    model.load_state_dict(torch.load(MODELS_DIR / "ensemble_best.pth"))
    val_metrics = evaluate(model, val_loader)
    
    best_thresh, best_f1 = find_best_threshold(
        val_metrics['predictions'],
        val_metrics['labels']
    )
    
    print(f"Best Threshold: {best_thresh:.3f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    test_metrics = evaluate(model, test_loader, threshold=best_thresh)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Save results
    results = {
        'best_threshold': float(best_thresh),
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'auc': float(test_metrics['auc']),
            'f1': float(test_metrics['f1']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall'])
        },
        'history': history
    }
    
    with open(MODELS_DIR / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved: {MODELS_DIR}/ensemble_best.pth")
    print(f"📊 Results saved: {MODELS_DIR}/training_results.json")

if __name__ == "__main__":
    main()
