# Comprehensive Model Evaluation — Optimized for GPU

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from ensemble_models import load_ensemble
from preprocessing import PreprocessedDataset, get_val_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path("models")
PROCESSED_DIR = Path("datasets_processed")
OUTPUTS_DIR = Path("outputs/evaluation")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Maximize GPU utilization
BATCH_SIZE = 64  # Large batch to fill 16GB GPU
MC_SAMPLES = 20  # MC Dropout iterations

def load_dataset_split(split_dir):
    """Load images and labels"""
    image_paths = []
    labels = []
    
    for cls, label in [("TB", 1), ("Normal", 0)]:
        cls_dir = split_dir / cls
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                image_paths.append(img_path)
                labels.append(label)
    
    return image_paths, labels

def evaluate_with_uncertainty_batched(model, dataloader, n_samples=20):
    """Batched MC Dropout evaluation — fast, uses full GPU"""
    model.eval()
    model.dropout.train()  # Enable only dropout
    
    all_means = []
    all_stds = []
    all_labels = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            
            # Run MC Dropout samples in batch
            batch_preds = []
            for _ in range(n_samples):
                pred = model._forward_with_dropout(images)
                batch_preds.append(pred)
            
            # Stack: [n_samples, batch_size]
            batch_preds = torch.stack(batch_preds)
            
            mean_pred = batch_preds.mean(dim=0).cpu().numpy()
            std_pred = batch_preds.std(dim=0).cpu().numpy()
            
            all_means.extend(mean_pred)
            all_stds.extend(std_pred)
            all_labels.extend(labels.numpy())
    
    return np.array(all_means), np.array(all_stds), np.array(all_labels)

def calculate_calibration(predictions, labels, n_bins=10):
    """Calculate calibration metrics"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            accuracies.append(0)
            confidences.append(0)
            bin_counts.append(0)
    
    # Expected Calibration Error
    ece = np.sum([
        (bin_counts[i] / len(predictions)) * abs(accuracies[i] - confidences[i])
        for i in range(n_bins)
    ])
    
    return {
        'ece': ece,
        'accuracies': accuracies,
        'confidences': confidences,
        'bin_counts': bin_counts
    }

def plot_calibration(calibration_data, save_path):
    """Plot reliability diagram"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    confidences = calibration_data['confidences']
    accuracies = calibration_data['accuracies']
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(confidences, accuracies, 'o-', label=f'Model (ECE: {calibration_data["ece"]:.3f})')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_curve(labels, predictions, save_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC: {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_uncertainty_distribution(uncertainties, labels, save_path):
    """Plot uncertainty distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tb_uncertainties = uncertainties[labels == 1]
    normal_uncertainties = uncertainties[labels == 0]
    
    ax.hist(tb_uncertainties, bins=30, alpha=0.5, label='TB', color='red')
    ax.hist(normal_uncertainties, bins=30, alpha=0.5, label='Normal', color='blue')
    
    ax.set_xlabel('Uncertainty (Std Dev)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prediction Uncertainty Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def analyze_failure_cases(predictions, uncertainties, labels, image_paths, threshold=0.5):
    """Analyze failure cases"""
    preds_binary = (predictions > threshold).astype(int)
    failures = preds_binary != labels
    
    failure_indices = np.where(failures)[0]
    
    failure_cases = []
    for idx in failure_indices:
        failure_cases.append({
            "image": str(image_paths[idx]),
            "true_label": "TB" if labels[idx] == 1 else "Normal",
            "predicted_label": "TB" if preds_binary[idx] == 1 else "Normal",
            "probability": float(predictions[idx]),
            "uncertainty": float(uncertainties[idx])
        })
    
    # Sort by uncertainty
    failure_cases.sort(key=lambda x: x['uncertainty'], reverse=True)
    
    return failure_cases

def main():
    print("="*60)
    print("Comprehensive Model Evaluation")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = load_ensemble(MODELS_DIR / "ensemble_best.pth", DEVICE)
    
    # Load training results for threshold
    with open(MODELS_DIR / "training_results.json") as f:
        results = json.load(f)
        threshold = results.get("best_threshold", 0.5)
    
    print(f"Using threshold: {threshold:.3f}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"MC Dropout samples: {MC_SAMPLES}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_paths, test_labels = load_dataset_split(PROCESSED_DIR / "test")
    test_dataset = PreprocessedDataset(
        test_paths, test_labels,
        transforms=get_val_transforms(),
        use_preprocessing=True
    )
    
    # Use DataLoader for batched processing
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True, shuffle=False
    )
    
    predictions, uncertainties, labels = evaluate_with_uncertainty_batched(
        model, test_loader, n_samples=MC_SAMPLES
    )
    
    # Calculate metrics
    print("\nCalculating metrics...")
    preds_binary = (predictions > threshold).astype(int)
    
    acc = accuracy_score(labels, preds_binary)
    auc = roc_auc_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_binary, average='binary')
    cm = confusion_matrix(labels, preds_binary)
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    # Calibration
    print("Calculating calibration...")
    calibration_data = calculate_calibration(predictions, labels)
    
    # Results
    evaluation_results = {
        "test_metrics": {
            "accuracy": float(acc),
            "auc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1": float(f1)
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "calibration": {
            "ece": float(calibration_data['ece'])
        },
        "uncertainty": {
            "mean": float(uncertainties.mean()),
            "std": float(uncertainties.std()),
            "min": float(uncertainties.min()),
            "max": float(uncertainties.max())
        },
        "threshold": float(threshold)
    }
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall/Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nExpected Calibration Error: {calibration_data['ece']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn}, FP: {fp}")
    print(f"  FN: {fn}, TP: {tp}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_calibration(calibration_data, OUTPUTS_DIR / "calibration.png")
    plot_roc_curve(labels, predictions, OUTPUTS_DIR / "roc_curve.png")
    plot_uncertainty_distribution(uncertainties, labels, OUTPUTS_DIR / "uncertainty_dist.png")
    
    # Failure analysis
    print("\nAnalyzing failure cases...")
    failure_cases = analyze_failure_cases(predictions, uncertainties, labels, test_paths, threshold)
    
    print(f"Total failures: {len(failure_cases)}")
    if failure_cases:
        print(f"Top 5 uncertain failures:")
        for i, case in enumerate(failure_cases[:5], 1):
            print(f"  {i}. {Path(case['image']).name}")
            print(f"     True: {case['true_label']}, Pred: {case['predicted_label']}")
            print(f"     Prob: {case['probability']:.3f}, Uncertainty: {case['uncertainty']:.3f}")
    
    evaluation_results['failure_cases'] = failure_cases
    
    # Save results
    with open(OUTPUTS_DIR / "evaluation_results.json", 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {OUTPUTS_DIR}")
    print(f"📊 Plots: calibration.png, roc_curve.png, uncertainty_dist.png")
    print(f"📄 Full results: evaluation_results.json")

if __name__ == "__main__":
    main()
