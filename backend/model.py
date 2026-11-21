"""
PyTorch Deep Learning Model Architecture for Dental OPG Disease Detection
Supports multi-label detection for:
1. Dental Caries
2. Periapical Lesion
3. Periodontitis
With built-in Grad-CAM explainable AI visualization for dental diagnostics.
"""

import sys
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    models = None
    TORCH_AVAILABLE = False

import numpy as np
import cv2
from PIL import Image

DISEASE_METADATA = {
    "Dental Caries": {
        "index": 0,
        "name": "Dental Caries",
        "category": "Hard Tissue Pathology",
        "description": "Demineralization of enamel and dentin causing coronal or interproximal radiolucencies.",
        "hallmarks": "Crown radiolucency, shadow under existing fillings, occlusal pit breakdown.",
        "severity_levels": {
            "mild": "Incipient enamel lesion (non-cavitated)",
            "moderate": "Dentin involvement without pulp exposure",
            "severe": "Deep dentinal caries approaching or exposing pulp chamber"
        },
        "clinical_guidance": "Bitewing confirmation, minimally invasive restoration (composite resin), fluoride varnish, or indirect pulp capping depending on depth."
    },
    "Periapical Lesion": {
        "index": 1,
        "name": "Periapical Lesion",
        "category": "Endodontic Pathology",
        "description": "Periapical radiolucency surrounding the root apex indicating pulpal necrosis, granuloma, or periapical abscess.",
        "hallmarks": "Loss of lamina dura at apex, widening of periodontal ligament (PDL) space, radiolucent halo around root tip.",
        "severity_levels": {
            "mild": "Widened apical PDL space (< 1mm)",
            "moderate": "Distinct periapical radiolucency (< 5mm, localized granuloma)",
            "severe": "Large periapical cyst or diffuse apical osteitis (> 5mm) with cortical expansion"
        },
        "clinical_guidance": "Pulp vitality testing (cold/electric), root canal treatment (endodontic therapy), or surgical apicoectomy if persistent."
    },
    "Periodontitis": {
        "index": 2,
        "name": "Periodontitis",
        "category": "Periodontal Pathology",
        "description": "Progressive loss of alveolar crest bone support around the teeth due to chronic periodontal infection.",
        "hallmarks": "Horizontal or vertical bone resorption >2mm apical to the cementoenamel junction (CEJ), furcation involvement.",
        "severity_levels": {
            "mild": "Stage I: 1-2 mm clinical attachment loss / mild crestal blunting",
            "moderate": "Stage II: 3-4 mm loss / horizontal bone resorption extending to coronal third",
            "severe": "Stage III/IV: >= 5 mm loss / vertical bone defects, furcation involvement, tooth mobility risk"
        },
        "clinical_guidance": "Comprehensive periodontal charting, Scaling and Root Planing (SRP), antimicrobials, and periodontal maintenance every 3 months."
    }
}

class DentalOPGNet(nn.Module if TORCH_AVAILABLE else object):
    """
    State-of-the-art Deep Convolutional Neural Network for Dental Radiography.
    Uses ResNet50 backbone with specialized multi-disease classification head.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(DentalOPGNet, self).__init__()
        
        # ResNet50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_model = models.resnet50(weights=weights)
        
        # Backbone feature extractor
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )
        
        # Adaptive pooling to handle varying panoramic aspect ratios
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Disease classification head
        in_features = base_model.fc.in_features  # 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        # Hooks for Grad-CAM
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Target layer is layer4 (the last convolutional block in ResNet50)
        target_layer = self.features[7]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits

    def generate_gradcam(self, input_tensor, target_class_idx):
        """
        Generate Grad-CAM activation heatmap for a specific disease class index (0, 1, or 2).
        """
        self.eval()
        self.zero_grad()
        
        # Forward pass
        logits = self.forward(input_tensor)
        score = logits[0, target_class_idx]
        
        # Backward pass
        score.backward(retain_graph=True)
        
        # Gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            # Fallback uniform map if hooks fail
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze(0)
        cam = F.relu(cam)  # Apply ReLU
        
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize [0, 1]
        
        # Resize CAM to match input tensor resolution
        cam_resized = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam_resized

def load_trained_model(weights_path=None, device=None):
    if device is None:
        device = "cuda" if (TORCH_AVAILABLE and torch and torch.cuda.is_available()) else "cpu"
    """Instantiate and load DentalOPGNet."""
    model = DentalOPGNet(num_classes=3, pretrained=True)
    if weights_path and torch.cuda.is_available():
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            print(f"[Model] Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"[Model] Could not load weights from {weights_path}: {e}. Initializing base model.")
    elif weights_path:
        try:
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state.get("model_state_dict", state))
            print(f"[Model] Successfully loaded weights from {weights_path} on CPU.")
        except Exception as e:
            print(f"[Model] Weight load note: {e}")
            
    model.to(device)
    model.eval()
    return model
