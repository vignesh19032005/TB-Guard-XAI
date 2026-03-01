# Ensemble Model Architecture

import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchvision import models
import timm

class DenseNetTB(nn.Module):
    """DenseNet121 for TB detection"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model.op_threshs = None
        else:
            self.model = xrv.models.DenseNet(weights=None)
        
        # Binary classification
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
    
    def forward(self, x):
        return self.model(x)

class EfficientNetTB(nn.Module):
    """EfficientNet-B3 for TB detection"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=1, in_chans=1)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)

class ResNetTB(nn.Module):
    """ResNet50 for TB detection"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)
        
        # Modify first conv for grayscale
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)

class TBEnsemble(nn.Module):
    """Ensemble of multiple models with weighted voting"""
    
    def __init__(self, weights=None, use_mc_dropout=False, dropout_rate=0.3):
        super().__init__()
        
        self.densenet = DenseNetTB(pretrained=True)
        self.efficientnet = EfficientNetTB(pretrained=True)
        self.resnet = ResNetTB(pretrained=True)
        
        # Default weights
        if weights is None:
            self.weights = torch.tensor([0.4, 0.35, 0.25])
        else:
            self.weights = torch.tensor(weights)
        
        self.use_mc_dropout = use_mc_dropout
        if use_mc_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Get predictions from each model
        pred_densenet = torch.sigmoid(self.densenet(x)).squeeze()
        pred_efficientnet = torch.sigmoid(self.efficientnet(x)).squeeze()
        pred_resnet = torch.sigmoid(self.resnet(x)).squeeze()
        
        # Weighted average (no dropout during normal forward)
        ensemble_pred = (
            self.weights[0] * pred_densenet +
            self.weights[1] * pred_efficientnet +
            self.weights[2] * pred_resnet
        )
        
        return ensemble_pred
    
    def _forward_with_dropout(self, x):
        """Forward pass with dropout on logits for MC uncertainty estimation"""
        # Get raw logits from each model (before sigmoid)
        logit_densenet = self.densenet(x).squeeze()
        logit_efficientnet = self.efficientnet(x).squeeze()
        logit_resnet = self.resnet(x).squeeze()
        
        # Apply dropout to logits — proper MC Dropout location
        logit_densenet = self.dropout(logit_densenet)
        logit_efficientnet = self.dropout(logit_efficientnet)
        logit_resnet = self.dropout(logit_resnet)
        
        # Convert to probabilities after dropout
        pred_densenet = torch.sigmoid(logit_densenet)
        pred_efficientnet = torch.sigmoid(logit_efficientnet)
        pred_resnet = torch.sigmoid(logit_resnet)
        
        # Weighted average
        ensemble_pred = (
            self.weights[0] * pred_densenet +
            self.weights[1] * pred_efficientnet +
            self.weights[2] * pred_resnet
        )
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, x, n_samples=20):
        """MC Dropout uncertainty estimation"""
        # Keep model in eval mode (BatchNorm stays stable)
        # Only enable dropout manually
        self.eval()
        self.dropout.train()  # Enable dropout only
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self._forward_with_dropout(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred

def load_ensemble(checkpoint_path=None, device='cuda'):
    """Load ensemble model"""
    model = TBEnsemble(use_mc_dropout=True)
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model = model.to(device)
    model.eval()  # Always start in eval mode
    return model

if __name__ == "__main__":
    # Test ensemble
    model = TBEnsemble(use_mc_dropout=True)
    
    # Dummy input
    x = torch.randn(2, 1, 224, 224)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Uncertainty estimation
    mean, std = model.predict_with_uncertainty(x, n_samples=10)
    print(f"\nMean prediction: {mean}")
    print(f"Std prediction: {std}")
    
    print("\n✅ Ensemble model test passed")
