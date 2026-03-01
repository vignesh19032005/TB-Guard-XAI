import torch
import cv2
from mistral_explainer import MistralExplainer
from ensemble_models import load_ensemble
from preprocessing import PreprocessedDataset, get_val_transforms
from torch.utils.data import DataLoader
from pathlib import Path

class DummyExplainer(MistralExplainer):
    def __init__(self, model_path=None):
        self.model = load_ensemble(model_path, "cuda")
        self.mistral = None
        self.rag = None
        self.preprocessor = PreprocessedDataset(None, None, use_preprocessing=True).preprocessor

# Initialize mistral
explainer = DummyExplainer("models/ensemble_best.pth")

print("--- Testing via MistralExplainer ---")
test_file = Path(r"datasets_raw/covid19_radiography/COVID-19_Radiography_Dataset/Normal/images/Normal-19.png")
if test_file.exists():
    print("Normal-19 (COVID) Pred 1:", explainer.predict_with_uncertainty(test_file, n_samples=20)["probability"])
else:
    print(f"File {test_file} not found")

test_file2 = Path(r"datasets_raw/kaggle_tb/TB_Chest_Radiography_Database/Normal/Normal-19.png")
if test_file2.exists():
    print("Normal-19 (Kaggle) Pred 2:", explainer.predict_with_uncertainty(test_file2, n_samples=20)["probability"])
else:
    print(f"File {test_file2} not found")
    print(f"File {test_file2} not found")

if test_file.exists() and test_file2.exists():
    print("--- Testing via Evaluate Logic ---")
    test_dataset = PreprocessedDataset([test_file, test_file2], [0, 1], transforms=get_val_transforms(), use_preprocessing=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    import numpy as np
    model = load_ensemble("models/ensemble_best.pth", "cuda")
    model.eval()
    model.dropout.train()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to("cuda")
            batch_preds = []
            for _ in range(20):
                pred = model._forward_with_dropout(images)
                batch_preds.append(pred)
            batch_preds = torch.stack(batch_preds)
            mean_pred = batch_preds.mean(dim=0).cpu().numpy()
            print("DataLoader Mean Preds:", mean_pred)
