"""
Evaluate trained DentalOPGNet on the DENTEX test set (official 50-image validation with ground truth).
Writes models/metrics.json with real per-class and macro metrics + 95% bootstrap CIs.
Generates static/roc_curve.png and static/confusion_matrix.png.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    sys.exit("PyTorch required. Install: pip install torch torchvision")

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.model import DISEASE_CLASSES, load_trained_model
from backend.dataset_prep import setup_dataset, load_splits, VALIDATION_IMAGE_DIR

MODELS_DIR = Path("models")
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

EVAL_H, EVAL_W = 256, 512


def preprocess(img_path: Path) -> "torch.Tensor":
    transform = T.Compose([
        T.Resize((EVAL_H, EVAL_W)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def bootstrap_auc(labels, probs, n_boot=1000, ci=0.95):
    """95% bootstrap CI for AUC-ROC."""
    rng = np.random.default_rng(42)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        lb, pb = labels[idx], probs[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        try:
            aucs.append(roc_auc_score(lb, pb))
        except Exception:
            pass
    if not aucs:
        return None, None
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return round(float(lo), 4), round(float(hi), 4)


def evaluate():
    weights_path = MODELS_DIR / "dental_model.pth"
    if not weights_path.exists():
        print("ERROR: models/dental_model.pth not found.")
        print("Run 'python -m backend.train' first.")
        sys.exit(1)

    device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    model = load_trained_model(str(weights_path), device=device)
    if model is None:
        print("ERROR: Could not load model weights.")
        sys.exit(1)

    # Load test split
    try:
        splits = load_splits()
    except FileNotFoundError:
        splits = setup_dataset()
    test_recs = splits.get("test", [])
    if not test_recs:
        print("ERROR: No test records found in data/splits.json.")
        sys.exit(1)

    print(f"[Eval] Evaluating on {len(test_recs)} test images...")
    model.eval()
    all_probs, all_labels = [], []
    t0 = time.time()
    missing = 0

    for rec in test_recs:
        img_path = VALIDATION_IMAGE_DIR / rec["filename"]
        if not img_path.exists():
            missing += 1
            continue
        tensor = preprocess(img_path).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(rec["labels"])

    if missing > 0:
        print(f"[Eval] WARNING: {missing} images not found — skipped.")

    if not all_probs:
        print("ERROR: No images could be evaluated.")
        sys.exit(1)

    all_probs_np = np.array(all_probs, dtype=np.float32)
    all_labels_np = np.array(all_labels, dtype=np.float32)
    elapsed = time.time() - t0
    n_eval = len(all_probs)

    # --- Per-class metrics ---
    preds = (all_probs_np >= 0.5).astype(int)
    per_class = {}
    auc_values = []
    for i, cls in enumerate(DISEASE_CLASSES):
        y_true = all_labels_np[:, i].astype(int)
        y_pred = preds[:, i]
        y_prob = all_probs_np[:, i]
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        tp = int((y_pred * y_true).sum())
        fp = int((y_pred * (1 - y_true)).sum())
        fn = int(((1 - y_pred) * y_true).sum())
        tn = int(((1 - y_pred) * (1 - y_true)).sum())
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if y_true.sum() > 0:
            try:
                auc = roc_auc_score(y_true, y_prob)
                auc_lo, auc_hi = bootstrap_auc(y_true, y_prob)
                auc_values.append(auc)
            except Exception:
                auc, auc_lo, auc_hi = None, None, None
        else:
            auc, auc_lo, auc_hi = None, None, None
        per_class[cls] = {
            "auc": round(float(auc), 4) if auc is not None else None,
            "auc_ci_lo": auc_lo,
            "auc_ci_hi": auc_hi,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "specificity": round(float(specificity), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_positive": int(y_true.sum()),
        }

    macro_auc = round(float(np.mean(auc_values)), 4) if auc_values else None

    metrics = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples_eval": n_eval,
        "split": "test (DENTEX official validation — 50 images with ground truth)",
        "threshold": 0.5,
        "disease_classes": DISEASE_CLASSES,
        "macro_auc": macro_auc,
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "macro_precision": round(float(np.mean([v["precision"] for v in per_class.values()])), 4),
        "macro_recall": round(float(np.mean([v["recall"] for v in per_class.values()])), 4),
        "per_class": per_class,
        "elapsed_seconds": round(elapsed, 2),
        "is_fabricated": False,
        "is_validated": True,
    }

    out_path = MODELS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Saved metrics to {out_path}")
    print(f"[Eval] macro_auc={macro_auc} macro_f1={metrics['macro_f1']}")

    # --- ROC curve ---
    _plot_roc(all_labels_np, all_probs_np)
    # --- Confusion matrix (per class binary) ---
    _plot_confusion(all_labels_np, preds)

    return metrics


def _plot_roc(labels, probs):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    colors = ["#F59E0B", "#fb923c", "#F43F5E", "#6366F1"]
    for i, (cls, col) in enumerate(zip(DISEASE_CLASSES, colors)):
        y_true = labels[:, i].astype(int)
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs[:, i])
        try:
            auc = roc_auc_score(y_true, probs[:, i])
        except Exception:
            auc = 0.0
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})", color=col, linewidth=2)
    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curves — DENTEX Test Set", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(loc="lower right", facecolor="#334155", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
    out = STATIC_DIR / "roc_curve.png"
    fig.savefig(out, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Eval] Saved ROC curve to {out}")


def _plot_confusion(labels, preds):
    n_cls = len(DISEASE_CLASSES)
    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4))
    fig.patch.set_facecolor("#1e293b")
    if n_cls == 1:
        axes = [axes]
    for i, (cls, ax) in enumerate(zip(DISEASE_CLASSES, axes)):
        y_true = labels[:, i].astype(int)
        y_pred = preds[:, i]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(cls, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("True", color="white")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"], color="white")
        ax.set_yticklabels(["Neg", "Pos"], color="white")
        ax.tick_params(colors="white")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
        ax.set_facecolor("#1e293b")
    fig.suptitle("Confusion Matrices — DENTEX Test Set", color="white",
                 fontsize=14, fontweight="bold")
    out = STATIC_DIR / "confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Eval] Saved confusion matrix to {out}")


if __name__ == "__main__":
    evaluate()
