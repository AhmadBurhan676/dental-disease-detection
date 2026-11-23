"""
PyTorch Model Architecture for Dental OPG Disease Detection (DENTEX classes).
Multi-label ResNet50 for: Caries, Deep Caries, Periapical Lesion, Impacted.
No Periodontitis — not present in DENTEX ground-truth.
Built-in Grad-CAM for real gradient-based saliency maps.
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

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH — matches backend/dataset_prep.py exactly
# ---------------------------------------------------------------------------
DISEASE_CLASSES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
NUM_CLASSES = len(DISEASE_CLASSES)

DISEASE_METADATA = {
    "Caries": {
        "index": 0,
        "name": "Caries",
        "category": "Hard Tissue Pathology",
        "description": "Demineralization of enamel and dentin causing coronal or interproximal radiolucencies.",
        "hallmarks": "Crown radiolucency, shadow under existing fillings, occlusal pit breakdown.",
        "severity_levels": {
            "mild": "Incipient enamel lesion (non-cavitated)",
            "moderate": "Dentin involvement without pulp exposure",
            "severe": "Deep dentinal caries approaching or exposing pulp",
        },
        "clinical_guidance": (
            "Bitewing confirmation, minimally invasive restoration, fluoride varnish, "
            "or indirect pulp capping depending on depth."
        ),
    },
    "Deep Caries": {
        "index": 1,
        "name": "Deep Caries",
        "category": "Hard Tissue Pathology",
        "description": "Advanced caries approaching or reaching the pulp chamber.",
        "hallmarks": "Extensive radiolucency involving dentin, possible pulp horn exposure.",
        "severity_levels": {
            "moderate": "Dentin involvement nearing pulp",
            "severe": "Pulp exposure or near-exposure",
        },
        "clinical_guidance": (
            "Indirect or direct pulp capping, root canal treatment, or extraction "
            "depending on restorability and pulp vitality."
        ),
    },
    "Periapical Lesion": {
        "index": 2,
        "name": "Periapical Lesion",
        "category": "Endodontic Pathology",
        "description": "Periapical radiolucency indicating pulpal necrosis, granuloma, or abscess.",
        "hallmarks": "Loss of lamina dura, widened PDL space, radiolucent halo around root tip.",
        "severity_levels": {
            "mild": "Widened apical PDL space (< 1 mm)",
            "moderate": "Distinct periapical radiolucency < 5 mm",
            "severe": "Large periapical cyst or diffuse osteitis > 5 mm",
        },
        "clinical_guidance": (
            "Pulp vitality testing, root canal treatment, or surgical apicoectomy if persistent."
        ),
    },
    "Impacted": {
        "index": 3,
        "name": "Impacted",
        "category": "Developmental Pathology",
        "description": "Tooth that has failed to erupt into its normal position in the dental arch.",
        "hallmarks": "Tooth visible in bone but not erupted; often seen with third molars.",
        "severity_levels": {
            "mild": "Soft tissue impaction",
            "moderate": "Partial bony impaction",
            "severe": "Complete bony impaction",
        },
        "clinical_guidance": (
            "Clinical and radiographic assessment; surgical extraction depending on symptoms, "
            "angulation, and adjacent tooth risk."
        ),
    },
}


class DentalOPGNet(nn.Module if TORCH_AVAILABLE else object):
    """
    ResNet50-based multi-label classifier for panoramic dental radiographs.
    Outputs raw logits (apply sigmoid for probabilities).
    Supports real Grad-CAM via registered forward/backward hooks.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)

        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = base.fc.in_features  # 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        self._gradients = None
        self._activations = None
        self._register_cam_hooks()

    def _register_cam_hooks(self):
        def fwd(module, inp, out):
            self._activations = out

        def bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0]

        self.features[7].register_forward_hook(fwd)
        self.features[7].register_full_backward_hook(bwd)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        return self.classifier(flat)

    def generate_gradcam(self, input_tensor, target_class_idx: int) -> np.ndarray:
        """Compute real Grad-CAM heatmap. Returns float32 (H, W) in [0, 1]."""
        self.eval()
        self.zero_grad()
        logits = self.forward(input_tensor)
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        grads = self._gradients
        acts = self._activations
        if grads is None or acts is None:
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((H, W), dtype=np.float32)

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        return cv2.resize(cam, (W, H))


def load_trained_model(weights_path=None, device=None):
    """
    Load DentalOPGNet. Returns None if weights missing or invalid.
    NEVER silently falls back to fake predictions.
    """
    if not TORCH_AVAILABLE:
        return None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DentalOPGNet(num_classes=NUM_CLASSES, pretrained=(weights_path is None))

    if weights_path:
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            print(f"[Model] Loaded weights from {weights_path}")
        except Exception as e:
            print(f"[Model] WARNING: Could not load weights from {weights_path}: {e}")
            return None

    model.to(device)
    model.eval()
    return model
