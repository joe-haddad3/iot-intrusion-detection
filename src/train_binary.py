"""
train_binary.py  —  Binary classification: BENIGN vs ATTACK
============================================================
Uses LightGBM (GPU) instead of Logistic Regression for much
better recall on attack detection.

  0 = BENIGN
  1 = ATTACK  (any of the 33 attack types)
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TRAIN_PATH   = "data/train.csv"
TEST_PATH    = "data/test.csv"
LABEL_COL    = "Label"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
RANDOM_STATE = 42

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("  IoT IDS — Binary Classification (BENIGN vs ATTACK)")
print("  Model: LightGBM")
print("=" * 60)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
for path in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[LABEL_COL]).values.astype(np.float32) # Take the training dataset, remove the label column (because the model needs to learn it, not cheat), convert everything to a numeric matrix and store it as float32 for efficient ML training
# X_train will now only contain the inputs to the model
X_test  = test_df.drop(columns=[LABEL_COL]).values.astype(np.float32) # We also separate the label column from X_test to Y_test (X_test is like the exam questions)
feature_names = train_df.drop(columns=[LABEL_COL]).columns.tolist() # Extracting the names of the feature columns as a Python list (for debugging and interpretation)

# ── REMAP LABELS TO BINARY ─────────────────────────────────────────────────────
le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
class_names = le.classes_.tolist() # So classes_names is something like: ['BENIGN', 'DDoS', 'DoS', 'Mirai', 'Spoofing', ...]

if "BENIGN" not in class_names:
    raise ValueError("'BENIGN' not found in label encoder classes.")

benign_idx = class_names.index("BENIGN") #This returns the position of "BENIGN" inside the list of class names
print(f"\nBENIGN class index : {benign_idx}")
print(f"Total original classes: {len(class_names)}")

# Get the label column (ex: [0, 2, 5, 1, 9, etc.] and compares each to the BENIGN index, then converts each true and false to 0 and 1 (.astype(np.int32))
# If label == BENIGN => False => 0
# If label != BENIGN => True  => 1
y_train = (train_df[LABEL_COL].values != benign_idx).astype(np.int32)
y_test  = (test_df[LABEL_COL].values  != benign_idx).astype(np.int32)

print(f"\nBinary label distribution (train):")
for lbl, name in [(0, "BENIGN"), (1, "ATTACK")]:
    c = (y_train == lbl).sum() # Since True returns 1 and False returns 0, when summing we get c = the number of samples with that label
    print(f"  {name:<8}: {c:>8,}  ({c/len(y_train)*100:.1f}%)")
# => Example output
# Binary label distribution (train):
#   BENIGN  :   80,000  (80.0%)
#   ATTACK  :   20,000  (20.0%)
# => This tells you how imbalanced your dataset is.


print(f"\nTrain: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
print(f"Test : {X_test.shape[0]:,} rows  x {X_test.shape[1]} features")

# ── DETECT LGBM DEVICE ────────────────────────────────────────────────────────
def detect_lgbm_device():
    try:
        tp = {"objective": "binary", "device": "gpu", "num_leaves": 7, "verbosity": -1}
        td = lgb.Dataset(np.random.rand(200, 4).astype(np.float32),
                         label=np.random.randint(0, 2, 200))
        lgb.train(tp, td, num_boost_round=2)
        return "gpu"
    except Exception:
        return "cpu"

LGBM_DEVICE = detect_lgbm_device() # The LightGBM device simply tells the model where to run its computations.
print(f"\nLightGBM device: {LGBM_DEVICE}")

# ── TRAIN/VAL SPLIT FOR EARLY STOPPING ────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size    = 0.10,
    stratify     = y_train,
    random_state = RANDOM_STATE,
)

# Class weight: scale positive (ATTACK) weight inversely to frequency
neg = (y_tr == 0).sum() # neg = number of BENIGN samples after the train/validation split
pos = (y_tr == 1).sum()

# IMP lal projet: This tells you: How imbalanced your dataset is
print(f"\nClass balance — BENIGN: {neg:,}  ATTACK: {pos:,}")


# LightGBM does: Train on dtrain, after each iteration, evaluate on dval, if performance stops improving → stop training
dtrain = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feature_names, free_raw_data=False)
dval   = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain, free_raw_data=False)

# A decision tree is a model that makes predictions by asking a sequence of yes/no questions about the features.
params = {
    "objective"         : "binary",
    "metric"            : "auc",  # measures how well the model separates classes across all thresholds
    "device"            : LGBM_DEVICE,
    "num_leaves"        : 127,  # Controls how complex each tree can be. More leaves => more complex patterns => more questions => more detailed decisions. 127 is relatively high => allows capturing complex attack behavior Good for: non-linear patterns in network traffic
    "learning_rate"     : 0.05, # Lower = slower learning but more stable
    "min_child_samples" : 20, # Minimum samples required in a leaf (of the tree) => Preventing overfitting
    "feature_fraction"  : 0.8, # Each tree uses only 80% of features (like packet_size, duration, etc.) => adds randomness and reduces overfitting
    "bagging_fraction"  : 0.8, # Each tree uses 80% of the data. (If you have 100,000 rows: Tree 1 trains on 80,000 rows (random subset) and tree 2 trains on a different 80,000 rows)
    "bagging_freq"      : 5, # Row sampling happens every 5 trees

    # Regularization (penalties on complexity)
    "reg_alpha"         : 0.1, # L1 regularization
    "reg_lambda"        : 0.1, # L2 regularization
    "is_unbalance"      : True,  # Automatically adjusts weights so the model pays more attention to the minority class
    "n_jobs"            : 4, # Use 4 CPU threads for parallel computation.
    "seed"              : RANDOM_STATE,
    "verbosity"         : -1,
}

print(f"\n{'='*60}")
print("  Training LightGBM binary classifier ...")
print(f"{'='*60}")

t0 = time.time()
callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=False),
    lgb.log_evaluation(period=50),
]

model = lgb.train(
    params,
    dtrain,
    num_boost_round = 1000,
    valid_sets      = [dval],
    callbacks       = callbacks,
)
train_time = time.time() - t0
print(f"\n  Done in {train_time:.1f}s  |  Best iteration: {model.best_iteration}")

# ── EVALUATE ON TEST SET ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Final evaluation on locked test set")
print(f"{'='*60}")

# Find best threshold: maximize attack recall while keeping FP rate below 10%
val_probs  = model.predict(X_val)
val_benign = (y_val == 0)
val_attack = (y_val == 1)

best_thresh, best_recall = 0.5, 0.0
for t in np.arange(0.05, 0.95, 0.01):
    preds  = (val_probs >= t).astype(int)
    fp_rate = (preds[val_benign] == 1).sum() / val_benign.sum()
    recall  = (preds[val_attack] == 1).sum() / val_attack.sum()
    if fp_rate <= 0.10 and recall > best_recall:
        best_recall, best_thresh = recall, t

best_val_f1 = f1_score(y_val, (val_probs >= best_thresh).astype(int), average="macro", zero_division=0)
print(f"\n  Best threshold (FP<=10%, attack recall={best_recall:.4f}): {best_thresh:.2f}")

y_pred_prob = model.predict(X_test)
y_pred      = (y_pred_prob >= best_thresh).astype(int)

test_f1        = f1_score(y_test, y_pred,      zero_division=0)
test_accuracy  = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall    = recall_score(y_test, y_pred,    zero_division=0)
test_roc_auc   = roc_auc_score(y_test, y_pred_prob)

benign_mask = (y_test == 0)
attack_mask = (y_test == 1)
fp_rate = (y_pred[benign_mask] == 1).sum() / benign_mask.sum()
fn_rate = (y_pred[attack_mask] == 0).sum() / attack_mask.sum()

print(f"\n  F1-score         : {test_f1:.4f}")
print(f"  ROC-AUC          : {test_roc_auc:.4f}")
print(f"  Accuracy         : {test_accuracy:.4f}")
print(f"  Precision        : {test_precision:.4f}")
print(f"  Recall           : {test_recall:.4f}")
print(f"\n  False Positive Rate (BENIGN flagged as ATTACK): {fp_rate*100:.2f}%")
print(f"  False Negative Rate (ATTACK missed as BENIGN) : {fn_rate*100:.2f}%")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"],
                             zero_division=0, digits=4))

# ── FEATURE IMPORTANCE PLOT ───────────────────────────────────────────────────
print("Generating plots ...")

fi = pd.Series(model.feature_importance(importance_type="gain"),
               index=feature_names).sort_values(ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(9, 7))
fi.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Binary LightGBM — Top 20 Feature Importance (Gain)", fontsize=12)
ax.set_xlabel("Gain")
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "binary_lgbm_feature_importance.png"), dpi=150)
plt.close(fig)

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN","ATTACK"]).plot(
    ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Binary LightGBM — Confusion Matrix\nF1={test_f1:.4f}  Acc={test_accuracy:.4f}", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "binary_lr_confusion_matrix.png"), dpi=150)
plt.close(fig)

# ── ROC CURVE ─────────────────────────────────────────────────────────────────
fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr_arr, tpr_arr, color="steelblue", linewidth=2,
        label=f"LightGBM Binary (AUC = {test_roc_auc:.4f})")
ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Binary Classification (BENIGN vs ATTACK)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "binary_lr_roc_curve.png"), dpi=150)
plt.close(fig)
print("  Plots saved.")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "binary_lr_model.pkl")
joblib.dump(model, model_path)
joblib.dump(benign_idx, os.path.join(MODELS_DIR, "binary_benign_idx.pkl"))
joblib.dump(best_thresh, os.path.join(MODELS_DIR, "binary_threshold.pkl"))
print(f"\n  Model saved -> {model_path}")

# ── SAVE METRICS ─────────────────────────────────────────────────────────────
metrics = {
    "model"               : "LightGBM (binary)",
    "task"                : "BENIGN (0) vs ATTACK (1)",
    "lgbm_device"         : LGBM_DEVICE,
    "best_iteration"      : model.best_iteration,
    "train_time_s"        : round(train_time, 2),
    "test_f1"             : round(test_f1,        4),
    "test_roc_auc"        : round(test_roc_auc,   4),
    "test_accuracy"       : round(test_accuracy,  4),
    "test_precision"      : round(test_precision, 4),
    "test_recall"         : round(test_recall,    4),
    "false_positive_rate" : round(fp_rate,        4),
    "false_negative_rate" : round(fn_rate,        4),
}
with open(os.path.join(REPORTS_DIR, "binary_lr_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'='*60}")
print("  BINARY LGBM — FINAL RESULTS")
print(f"{'='*60}")
print(f"  F1-score   : {test_f1:.4f}")
print(f"  ROC-AUC    : {test_roc_auc:.4f}")
print(f"  Accuracy   : {test_accuracy:.4f}")
print(f"  Precision  : {test_precision:.4f}")
print(f"  Recall     : {test_recall:.4f}")
print(f"  FP rate    : {fp_rate*100:.2f}%  (BENIGN flagged as ATTACK)")
print(f"  FN rate    : {fn_rate*100:.2f}%  (ATTACK missed)")
print(f"{'='*60}")
