"""
neural_network_focal.py  --  MLP + Focal Loss (PyTorch GPU)
===========================================================
Why Focal Loss beats weighted CrossEntropy for rare classes:
  - Weighted CE: assigns a fixed multiplier per class.
    The model still sees easy examples at full loss.
  - Focal Loss: FL(pt) = -alpha_t * (1-pt)^gamma * log(pt)
    The (1-pt)^gamma term down-weights EASY examples dynamically.
    When the model is confident (pt high), the gradient is suppressed.
    Hard/rare examples with low confidence drive the gradient.
    This is exactly what we need for rare attack classes.

gamma=2.0 is the standard Focal Loss exponent (from RetinaNet paper).
alpha = inverse class frequency (same logic as weighted CE, but combined
        with the gamma down-weighting for a doubly-effective signal).

Architecture: same as neural_network.py (27->256->256->128->64->N)
Change: CrossEntropyLoss(weight=...) replaced by FocalLoss(alpha, gamma=2)
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_PATH   = "data/train.csv"
TEST_PATH    = "data/test.csv"
LABEL_COL    = "Label"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
RANDOM_STATE = 42

BATCH_SIZE   = 4096
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 60
PATIENCE     = 10
VAL_FRAC     = 0.15

FOCAL_GAMMA  = 2.0   # focusing parameter: higher = more focus on hard examples

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 62)
print("  IoT IDS -- Neural Network (Focal Loss) Training")
print("=" * 62)
print(f"\n  Device     : {DEVICE}")
print(f"  Focal gamma: {FOCAL_GAMMA}")
if DEVICE.type == "cuda":
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\nLoading data ...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train_full  = train_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_train_full  = train_df[LABEL_COL].values.astype(np.int64)
X_test        = test_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_test        = test_df[LABEL_COL].values.astype(np.int64)
feature_names = train_df.drop(columns=[LABEL_COL]).columns.tolist()

le          = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
class_names = le.classes_.tolist()
n_classes   = len(class_names)
n_features  = X_train_full.shape[1]

print(f"  Train : {X_train_full.shape[0]:,} rows x {n_features} features")
print(f"  Test  : {X_test.shape[0]:,} rows  x {n_features} features")
print(f"  Classes: {n_classes}")

# ── CLASS WEIGHTS (alpha for Focal Loss) ─────────────────────────────────────
unique, counts   = np.unique(y_train_full, return_counts=True)
class_freq       = counts / counts.sum()
class_wts        = 1.0 / (class_freq * n_classes)
class_wts        = class_wts / class_wts.mean()     # normalise mean to 1
alpha_t          = torch.tensor(class_wts, dtype=torch.float32).to(DEVICE)

print("\n  Class alpha weights (top 5 heaviest -> rare classes):")
top5_idx = np.argsort(class_wts)[::-1][:5]
for i in top5_idx:
    print(f"    {class_names[i]:<40} alpha={class_wts[i]:.3f}")

# ── FOCAL LOSS ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha  : Tensor of shape (n_classes,) — per-class weighting (like CE weight)
    gamma  : Focusing exponent. gamma=0 => standard weighted CE.
             gamma=2 is the standard from the RetinaNet paper.
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy loss per sample (unreduced)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (N,)

        # p_t = probability of the TRUE class
        pt = torch.exp(-ce_loss)  # (N,)  ∈ [0,1]

        # alpha_t = weight for the true class of each sample
        alpha_t = self.alpha[targets]  # (N,)

        # Focal weight: down-weight easy (high p_t) examples
        focal_loss = alpha_t * (1.0 - pt) ** self.gamma * ce_loss  # (N,)

        return focal_loss.mean()


# ── MODEL ARCHITECTURE (identical to neural_network.py) ──────────────────────
class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_in),

            nn.Linear(n_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── TRAIN / VAL SPLIT ─────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size    = VAL_FRAC,
    stratify     = y_train_full,
    random_state = RANDOM_STATE,
)
print(f"\n  Train subset : {len(X_tr):,}  |  Val : {len(X_val):,}")


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


train_loader = make_loader(X_tr,  y_tr,  shuffle=True)
val_loader   = make_loader(X_val, y_val, shuffle=False)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_model(train_ldr, val_ldr, max_epochs, patience, verbose=True):
    model     = MLP(n_features, n_classes).to(DEVICE)
    criterion = FocalLoss(alpha=alpha_t, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_f1 = -1.0
    best_state  = None
    best_epoch  = 0
    no_improve  = 0
    history     = {"train_loss": [], "val_loss": [], "val_f1": []}

    # Validation loss uses standard weighted CE for monitoring
    # (Focal loss values are not directly comparable to CE loss)
    val_criterion = nn.CrossEntropyLoss(weight=alpha_t)

    for epoch in range(1, max_epochs + 1):
        # -- train --
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_ldr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        scheduler.step()
        train_loss = epoch_loss / len(train_ldr.dataset)

        # -- validate --
        model.eval()
        val_loss   = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for Xb, yb in val_ldr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits   = model(Xb)
                val_loss += val_criterion(logits, yb).item() * len(yb)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_true.append(yb.cpu().numpy())
        val_loss   /= len(val_ldr.dataset)
        val_preds   = np.concatenate(all_preds)
        val_labels  = np.concatenate(all_true)
        val_f1      = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{max_epochs} | "
                  f"focal_loss={train_loss:.4f}  val_ce_loss={val_loss:.4f}  "
                  f"val_macro_F1={val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch  = epoch
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (best epoch={best_epoch})")
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_f1, history


# ── PHASE 1: FIND BEST EPOCH ──────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Phase 1: Training on subset to find best epoch")
print(f"  Loss     : Focal Loss (gamma={FOCAL_GAMMA})")
print(f"  Max epochs={MAX_EPOCHS}  Patience={PATIENCE}  LR={LR}")
print(f"  Batch size={BATCH_SIZE}")
print(f"{'='*62}\n")

t0 = time.time()
_, best_epoch, best_val_f1, history = train_model(
    train_loader, val_loader, MAX_EPOCHS, PATIENCE, verbose=True
)
phase1_time = time.time() - t0
print(f"\n  Phase 1 done in {phase1_time:.1f}s")
print(f"  Best epoch : {best_epoch}  |  Val macro-F1 : {best_val_f1:.4f}")

# ── PHASE 2: RETRAIN ON FULL TRAINING SET ─────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Phase 2: Retraining on full training set ({best_epoch} epochs)")
print(f"{'='*62}\n")

full_loader = make_loader(X_train_full, y_train_full, shuffle=True)
t0          = time.time()
final_model, _, _, _ = train_model(
    full_loader, val_loader,
    max_epochs = best_epoch,
    patience   = best_epoch + 1,   # effectively disable early stopping
    verbose    = True,
)
train_time = time.time() - t0
print(f"\n  Phase 2 done in {train_time:.1f}s")

# ── EVALUATE ON LOCKED TEST SET ───────────────────────────────────────────────
print(f"\n{'='*62}")
print("  Final evaluation on locked test set")
print(f"{'='*62}")

test_loader = make_loader(X_test, y_test, shuffle=False)

final_model.eval()
all_preds, all_probs = [], []
with torch.no_grad():
    for Xb, _ in test_loader:
        logits = final_model(Xb.to(DEVICE))
        probs  = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

y_pred      = np.concatenate(all_preds)
y_pred_prob = np.concatenate(all_probs)

macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
micro_f1    = f1_score(y_test, y_pred, average="micro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
accuracy    = accuracy_score(y_test, y_pred)
macro_prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_rec   = recall_score(y_test,   y_pred, average="macro", zero_division=0)

print(f"\n  Macro F1-score    : {macro_f1:.4f}   <- primary metric")
print(f"  Micro F1-score    : {micro_f1:.4f}")
print(f"  Weighted F1-score : {weighted_f1:.4f}")
print(f"  Accuracy          : {accuracy:.4f}")
print(f"  Macro Precision   : {macro_prec:.4f}")
print(f"  Macro Recall      : {macro_rec:.4f}")

try:
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    roc_auc    = roc_auc_score(y_test_bin, y_pred_prob, average="macro", multi_class="ovr")
    print(f"  ROC-AUC (macro)   : {roc_auc:.4f}")
except Exception:
    roc_auc = None
    print("  ROC-AUC           : could not compute")

benign_idx = class_names.index("BENIGN") if "BENIGN" in class_names else None
if benign_idx is not None:
    benign_mask = (y_test == benign_idx)
    fp_rate     = (y_pred[benign_mask] != benign_idx).sum() / benign_mask.sum()
    print(f"  Benign FP rate    : {fp_rate*100:.2f}%")
else:
    fp_rate = None

print(f"\n  Per-class report:")
report_str  = classification_report(y_test, y_pred, target_names=class_names, zero_division=0, digits=4)
print(report_str)
report_dict = classification_report(y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("Generating plots ...")

cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(20, 17))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.3, linecolor="gray", annot_kws={"size": 7})
ax.set_title(f"MLP (Focal Loss gamma={FOCAL_GAMMA}) -- Normalised Confusion Matrix\n"
             f"Macro F1={macro_f1:.4f}  Acc={accuracy:.4f}", fontsize=13)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True",      fontsize=11)
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "nn_focal_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/nn_focal_confusion_matrix.png")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep_range  = range(1, len(history["train_loss"]) + 1)
axes[0].plot(ep_range, history["train_loss"], label="Train Focal Loss")
axes[0].plot(ep_range, history["val_loss"],   label="Val CE Loss")
axes[0].set_title("Loss Curves (Phase 1)")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(ep_range, history["val_f1"], color="green", label="Val Macro F1")
axes[1].axhline(best_val_f1, color="red", linestyle="--",
                label=f"Best={best_val_f1:.4f} @ ep{best_epoch}")
axes[1].set_title("Validation Macro F1 (Phase 1)")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.suptitle(f"MLP Focal Loss (gamma={FOCAL_GAMMA}) -- Training Curves", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "nn_focal_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/nn_focal_training_curves.png")

per_class_f1  = [report_dict[c]["f1-score"]  for c in class_names]
per_class_rec = [report_dict[c]["recall"]     for c in class_names]
per_class_pre = [report_dict[c]["precision"]  for c in class_names]

fig, axes = plt.subplots(3, 1, figsize=(16, 16))
for ax, values, title, color in zip(
    axes,
    [per_class_f1, per_class_rec, per_class_pre],
    ["F1-score per class", "Recall per class", "Precision per class"],
    ["steelblue", "darkorange", "seagreen"],
):
    bars = ax.barh(class_names, values, color=color, edgecolor="white", height=0.7)
    ax.set_xlim(0, 1.08)
    ax.axvline(np.mean(values), color="red", linestyle="--", linewidth=1.2,
               label=f"Macro avg = {np.mean(values):.3f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)
plt.suptitle(f"MLP Focal Loss (gamma={FOCAL_GAMMA}) -- Per-class Metrics", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "nn_focal_per_class_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/nn_focal_per_class_metrics.png")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "nn_focal_model.pt")
torch.save({
    "model_state_dict": final_model.state_dict(),
    "n_features"      : n_features,
    "n_classes"       : n_classes,
    "best_epoch"      : best_epoch,
    "class_names"     : class_names,
    "focal_gamma"     : FOCAL_GAMMA,
}, model_path)
print(f"  Model saved -> {model_path}")

# ── SAVE METRICS ──────────────────────────────────────────────────────────────
LR_MACRO_F1 = 0.4401; RF_MACRO_F1 = 0.5861

metrics = {
    "model"               : "MLP_FocalLoss",
    "architecture"        : f"{n_features}->256->256->128->64->{n_classes} (BN+ReLU+Dropout)",
    "focal_gamma"         : FOCAL_GAMMA,
    "best_epoch"          : best_epoch,
    "phase1_time_s"       : round(phase1_time, 2),
    "train_time_phase2_s" : round(train_time, 2),
    "val_macro_f1"        : round(best_val_f1, 4),
    "test_macro_f1"       : round(macro_f1,    4),
    "test_micro_f1"       : round(micro_f1,    4),
    "test_weighted_f1"    : round(weighted_f1, 4),
    "test_accuracy"       : round(accuracy,    4),
    "test_macro_precision": round(macro_prec,  4),
    "test_macro_recall"   : round(macro_rec,   4),
    "test_roc_auc_macro"  : round(roc_auc, 4) if roc_auc else None,
    "benign_false_positive_rate": round(fp_rate, 4) if fp_rate is not None else None,
    "vs_lr_macro_f1_delta": round(macro_f1 - LR_MACRO_F1, 4),
    "vs_rf_macro_f1_delta": round(macro_f1 - RF_MACRO_F1, 4),
    "hyperparams": {
        "batch_size": BATCH_SIZE, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "focal_gamma": FOCAL_GAMMA,
    },
    "per_class": {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall"   : round(report_dict[cls]["recall"],    4),
            "f1"       : round(report_dict[cls]["f1-score"],  4),
            "support"  : int(report_dict[cls]["support"]),
        }
        for cls in class_names
    },
}

metrics_path = os.path.join(REPORTS_DIR, "nn_focal_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved -> {metrics_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  MLP FOCAL LOSS -- FINAL RESULTS")
print(f"{'='*62}")
print(f"  Loss function    : Focal Loss (gamma={FOCAL_GAMMA})")
print(f"  Architecture     : {n_features}->256->256->128->64->{n_classes}")
print(f"  Best epoch       : {best_epoch}")
print(f"  Test Macro F1    : {macro_f1:.4f}   <- primary metric")
print(f"  Test Accuracy    : {accuracy:.4f}")
print(f"  Test ROC-AUC     : {roc_auc:.4f}" if roc_auc else "  Test ROC-AUC     : N/A")
print(f"  Benign FP rate   : {fp_rate*100:.2f}%" if fp_rate is not None else "")
print(f"\n  vs LR  : Macro F1 {LR_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-LR_MACRO_F1:+.4f})")
print(f"  vs RF  : Macro F1 {RF_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-RF_MACRO_F1:+.4f})")
print(f"\n  Outputs:")
print(f"    models/nn_focal_model.pt")
print(f"    reports/nn_focal_metrics.json")
print(f"    reports/nn_focal_confusion_matrix.png")
print(f"    reports/nn_focal_per_class_metrics.png")
print(f"    reports/nn_focal_training_curves.png")
print(f"{'='*62}")
