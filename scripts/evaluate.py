#!/usr/bin/env python3
"""
scripts/evaluate.py
===================
Run a full evaluation pass on the trained DentalVision AI model and write
  models/metrics.json

Usage
-----
    python scripts/evaluate.py [--split val] [--model models/dental_model.pth]

The script computes per-class F1, precision, recall, specificity, overall
accuracy, and mean ROC-AUC on the requested dataset split.  It also captures
the per-epoch history already stored in models/training_metrics.json (if
present) so the convergence chart in the UI uses real training data.

If no trained model weights are found the script exits with a clear message
rather than writing placeholder numbers.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_PATH = MODELS_DIR / "metrics.json"
SCHEMA_VERSION = "dentalvision-metrics-v1"

CLASSES = ["Dental Caries", "Periapical Lesion", "Periodontitis"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DentalVision AI and write models/metrics.json")
    p.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate on (default: val)")
    p.add_argument("--model", default=str(MODELS_DIR / "dental_model.pth"), help="Path to trained model weights")
    p.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    p.add_argument("--baseline", default=str(MODELS_DIR / "baseline_metrics.json"),
                   help="Optional path to a baseline metrics.json from an earlier run, used for delta display in UI")
    p.add_argument("--validated", action="store_true", default=True,
                   help="Mark result as validated (shows 'Evaluation Results' badge in UI)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_binary_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    y_true, y_pred_prob: 1-D lists or numpy arrays of length N.
    Returns dict with tp, fp, fn, tn, precision, recall, f1, specificity.
    """
    import numpy as np
    y_true = np.array(y_true, dtype=int)
    y_pred = (np.array(y_pred_prob) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                specificity=round(specificity, 4))


def compute_auc(y_true, y_scores):
    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        return round(float(roc_auc_score(y_true, y_scores)), 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# History from existing training_metrics.json
# ---------------------------------------------------------------------------
def load_existing_history():
    p = MODELS_DIR / "training_metrics.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        h = data.get("history", {})
        if not h:
            return None
        return {
            "epochs":       h.get("epochs", []),
            "train_loss":   h.get("train_loss", []),
            "val_loss":     h.get("val_loss", []),
            "val_accuracy": h.get("val_accuracy", []),
            "per_class_f1": {
                "Dental Caries":    h.get("caries_f1", []),
                "Periapical Lesion":h.get("periapical_f1", []),
                "Periodontitis":    h.get("periodontitis_f1", []),
            }
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------
def load_baseline(path: str):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERROR] Model weights not found at {model_path}")
        print("  Train the model first with:  python backend/train.py")
        print("  Then re-run this script.")
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    sys.path.insert(0, str(BASE_DIR))

    try:
        import torch
        from backend.model import DentalOPGNet, DISEASE_METADATA
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DentalOPGNet(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    from backend.dataset_prep import get_dataloader
    loader = get_dataloader(split=args.split, batch_size=16, shuffle=False)

    all_probs = {c: [] for c in CLASSES}
    all_labels = {c: [] for c in CLASSES}
    correct_total = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            preds = outputs.cpu().numpy()
            lbl   = labels.cpu().numpy()
            for i, cls in enumerate(CLASSES):
                all_probs[cls].extend(preds[:, i].tolist())
                all_labels[cls].extend(lbl[:, i].tolist())

            # overall accuracy: image-level (at least one class correct at threshold)
            import numpy as np
            pred_binary = (preds >= args.threshold).astype(int)
            lbl_binary  = (lbl >= 0.5).astype(int)
            correct_total += int((pred_binary == lbl_binary).all(axis=1).sum())
            total_samples += images.shape[0]

    # ----------------------------------------------------------------
    # Per-class metrics
    # ----------------------------------------------------------------
    per_class = {}
    auc_values = []
    baseline = load_baseline(args.baseline)

    for cls in CLASSES:
        m = compute_binary_metrics(all_labels[cls], all_probs[cls], args.threshold)
        auc = compute_auc(all_labels[cls], all_probs[cls])
        if auc is not None:
            auc_values.append(auc)

        entry = {
            "f1":          m["f1"],
            "precision":   m["precision"],
            "recall":      m["recall"],
            "specificity": m["specificity"],
            "auc_roc":     auc,
            "tp":          m["tp"],
            "fp":          m["fp"],
            "fn":          m["fn"],
            # baselines (from optional baseline file)
            "baseline_f1": None,
        }
        if baseline and cls in baseline.get("per_class", {}):
            entry["baseline_f1"] = baseline["per_class"][cls].get("f1")
        per_class[cls] = entry

    overall_acc = round(100.0 * correct_total / total_samples, 1) if total_samples else None
    mean_auc    = round(sum(auc_values) / len(auc_values), 4) if auc_values else None

    # ----------------------------------------------------------------
    # Assemble output
    # ----------------------------------------------------------------
    history = load_existing_history()

    output = {
        "_schema": SCHEMA_VERSION,
        "_note": "Generated by scripts/evaluate.py. Do not edit manually.",
        "validated": args.validated,
        "evaluation_split": args.split,
        "dataset": "Orvile / Román et al. Sensors 2021 (Zenodo 4457648)",
        "n_samples_eval": total_samples,
        "baseline_label": "baseline model (epoch 1)" if baseline else None,
        "classes": CLASSES,
        "overall_accuracy": overall_acc,
        "mean_auc_roc": mean_auc,
        "baseline_overall_accuracy": baseline.get("overall_accuracy") if baseline else None,
        "baseline_mean_auc_roc": baseline.get("mean_auc_roc") if baseline else None,
        "per_class": per_class,
        "history": history,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"[OK] Wrote {OUTPUT_PATH}")
    print(f"     Overall accuracy : {overall_acc}%")
    print(f"     Mean AUC-ROC     : {mean_auc}")
    for cls in CLASSES:
        print(f"     {cls:22s} F1={per_class[cls]['f1']:.3f}")


if __name__ == "__main__":
    main()
