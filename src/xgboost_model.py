"""
xgboost_model.py  --  XGBoost (GPU) for IoT Intrusion Detection
================================================================
Strategy:
  1. Load preprocessed train/test (27 features, 34 classes)
  2. Hold out 20% of train as validation for early stopping
  3. Small grid search over max_depth x learning_rate
  4. Retrain best config on full training set
  5. Evaluate on locked test set
  6. Save model + full metrics report

GPU: uses device='cuda' (XGBoost >= 2.0)
Imbalance: scale_pos_weight not applicable for multiclass;
           we use sample_weight proportional to inverse class freq.
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
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
VAL_FRAC     = 0.20          # held-out val for early stopping
EARLY_STOP   = 30            # rounds without improvement
MAX_ROUNDS   = 1000          # upper bound; early stopping kicks in

PARAM_GRID = {
    "max_depth"    : [6, 8],
    "learning_rate": [0.1, 0.05],
    "subsample"    : [0.8],
    "colsample_bytree": [0.8],
}

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  IoT IDS -- XGBoost (GPU) Training")
print("=" * 62)

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train_full = train_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_train_full = train_df[LABEL_COL].values
X_test       = test_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_test       = test_df[LABEL_COL].values
feature_names = train_df.drop(columns=[LABEL_COL]).columns.tolist()

le          = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
class_names = le.classes_.tolist()
n_classes   = len(class_names)

print(f"\n  Train : {X_train_full.shape[0]:,} rows x {X_train_full.shape[1]} features")
print(f"  Test  : {X_test.shape[0]:,} rows  x {X_test.shape[1]} features")
print(f"  Classes: {n_classes}")

# ── CLASS WEIGHTS (inverse frequency) ────────────────────────────────────────
unique, counts = np.unique(y_train_full, return_counts=True)
freq = counts / counts.sum()
class_weight = 1.0 / (freq * n_classes)   # scale so mean weight ~ 1

def make_sample_weights(y, cw):
    return np.array([cw[c] for c in y], dtype=np.float32)

# ── TRAIN / VAL SPLIT ─────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size    = VAL_FRAC,
    stratify     = y_train_full,
    random_state = RANDOM_STATE,
)
w_tr  = make_sample_weights(y_tr,  class_weight)
w_val = make_sample_weights(y_val, class_weight)

print(f"\n  Train subset : {len(X_tr):,}  |  Val : {len(X_val):,}")

dtrain = xgb.DMatrix(X_tr,  label=y_tr,  weight=w_tr,  feature_names=feature_names)
dval   = xgb.DMatrix(X_val, label=y_val, weight=w_val, feature_names=feature_names)
dtest  = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

# ── HYPERPARAMETER SEARCH ─────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Grid search  max_depth={PARAM_GRID['max_depth']}")
print(f"               learning_rate={PARAM_GRID['learning_rate']}")
print(f"  Early stop : {EARLY_STOP} rounds  |  Max rounds: {MAX_ROUNDS}")
print(f"  Device     : cuda (GPU)")
print(f"{'='*62}\n")

combos = [
    {"max_depth": d, "learning_rate": lr,
     "subsample": s, "colsample_bytree": c}
    for d  in PARAM_GRID["max_depth"]
    for lr in PARAM_GRID["learning_rate"]
    for s  in PARAM_GRID["subsample"]
    for c  in PARAM_GRID["colsample_bytree"]
]

grid_results = {}
best_val_f1  = -1
best_params  = None
best_rounds  = None

for i, combo in enumerate(combos, 1):
    params = {
        "objective"        : "multi:softprob",
        "num_class"        : n_classes,
        "eval_metric"      : "mlogloss",
        "device"           : "cuda",
        "tree_method"      : "hist",
        "max_depth"        : combo["max_depth"],
        "learning_rate"    : combo["learning_rate"],
        "subsample"        : combo["subsample"],
        "colsample_bytree" : combo["colsample_bytree"],
        "min_child_weight" : 5,
        "gamma"            : 0.1,
        "reg_lambda"       : 1.0,
        "seed"             : RANDOM_STATE,
        "verbosity"        : 0,
    }

    label = (f"depth={combo['max_depth']} lr={combo['learning_rate']}")
    print(f"  [{i}/{len(combos)}] {label} ...", flush=True)
    t0 = time.time()

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round  = MAX_ROUNDS,
        evals            = [(dval, "val")],
        early_stopping_rounds = EARLY_STOP,
        evals_result     = evals_result,
        verbose_eval     = False,
    )

    y_val_prob = model.predict(dval).reshape(-1, n_classes)
    y_val_pred = y_val_prob.argmax(axis=1)
    val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    val_acc = accuracy_score(y_val, y_val_pred)
    best_iter = model.best_iteration
    elapsed = time.time() - t0

    grid_results[label] = {
        "params"    : combo,
        "val_macro_f1": val_f1,
        "val_accuracy": val_acc,
        "best_iter" : best_iter,
        "time_s"    : elapsed,
    }
    print(f"    val macro-F1={val_f1:.4f}  acc={val_acc:.4f}"
          f"  rounds={best_iter}  [{elapsed:.0f}s]")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_params = combo
        best_rounds = best_iter

# ── BEST CONFIG ───────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Best: depth={best_params['max_depth']}  lr={best_params['learning_rate']}")
print(f"  Val macro-F1 : {best_val_f1:.4f}")
print(f"  Rounds       : {best_rounds}")
print(f"{'='*62}\n")

# ── TRAIN FINAL MODEL ON FULL TRAINING SET ────────────────────────────────────
print("Training final model on full training set ...")
w_full = make_sample_weights(y_train_full, class_weight)
dfull  = xgb.DMatrix(X_train_full, label=y_train_full,
                     weight=w_full, feature_names=feature_names)

final_params = {
    "objective"        : "multi:softprob",
    "num_class"        : n_classes,
    "eval_metric"      : "mlogloss",
    "device"           : "cuda",
    "tree_method"      : "hist",
    "max_depth"        : best_params["max_depth"],
    "learning_rate"    : best_params["learning_rate"],
    "subsample"        : best_params["subsample"],
    "colsample_bytree" : best_params["colsample_bytree"],
    "min_child_weight" : 5,
    "gamma"            : 0.1,
    "reg_lambda"       : 1.0,
    "seed"             : RANDOM_STATE,
    "verbosity"        : 0,
}

t0 = time.time()
final_model = xgb.train(
    final_params,
    dfull,
    num_boost_round = best_rounds,
    verbose_eval    = False,
)
train_time = time.time() - t0
print(f"  Done in {train_time:.1f}s  |  {best_rounds} rounds")

# ── EVALUATE ON LOCKED TEST SET ───────────────────────────────────────────────
print(f"\n{'='*62}")
print("  Final evaluation on locked test set")
print(f"{'='*62}")

y_pred_prob = final_model.predict(dtest).reshape(-1, n_classes)
y_pred      = y_pred_prob.argmax(axis=1)

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
    fp_rate = (y_pred[benign_mask] != benign_idx).sum() / benign_mask.sum()
    print(f"  Benign FP rate    : {fp_rate*100:.2f}%")
else:
    fp_rate = None

print(f"\n  Per-class report:")
report_str  = classification_report(y_test, y_pred, target_names=class_names,
                                    zero_division=0, digits=4)
print(report_str)
report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                    zero_division=0, output_dict=True)

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
print("Generating plots ...")
cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(20, 17))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.3, linecolor="gray", annot_kws={"size": 7})
ax.set_title(f"XGBoost -- Normalised Confusion Matrix\n"
             f"Macro F1={macro_f1:.4f}  Acc={accuracy:.4f}", fontsize=13)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True",      fontsize=11)
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "xgb_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> reports/xgb_confusion_matrix.png")

# ── PER-CLASS BAR CHART ───────────────────────────────────────────────────────
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
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7)
plt.suptitle("XGBoost -- Per-class Metrics (Test Set)", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "xgb_per_class_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> reports/xgb_per_class_metrics.png")

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
scores = final_model.get_score(importance_type="gain")
fi_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
fi_names  = [x[0] for x in fi_sorted]
fi_vals   = [x[1] for x in fi_sorted]

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#c0392b" if v > np.percentile(fi_vals, 75) else "steelblue" for v in fi_vals]
ax.barh(fi_names[::-1], fi_vals[::-1], color=colors[::-1], edgecolor="white")
ax.set_xlabel("Gain", fontsize=11)
ax.set_title("XGBoost -- Feature Importances (Gain)\n(red = top 25%)", fontsize=12)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "xgb_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> reports/xgb_feature_importance.png")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "xgb_model.ubj")
final_model.save_model(model_path)
print(f"  Model saved -> {model_path}")

# ── SAVE METRICS ──────────────────────────────────────────────────────────────
LR_MACRO_F1 = 0.4401;  RF_MACRO_F1 = 0.5861
LR_ACCURACY = 0.5853;  RF_ACCURACY = 0.7642

metrics = {
    "model"               : "XGBoost",
    "best_params"         : {**best_params, "n_rounds": best_rounds},
    "train_time_s"        : round(train_time, 2),
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
    "grid_results"        : {
        k: {kk: round(vv, 4) if isinstance(vv, float) else vv
            for kk, vv in v.items() if kk != "params"}
        for k, v in grid_results.items()
    },
    "feature_importance_gain": {n: round(float(v), 4) for n, v in fi_sorted},
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

metrics_path = os.path.join(REPORTS_DIR, "xgb_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved -> {metrics_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  XGBOOST -- FINAL RESULTS")
print(f"{'='*62}")
print(f"  depth={best_params['max_depth']}  lr={best_params['learning_rate']}  rounds={best_rounds}")
print(f"  Test Macro F1    : {macro_f1:.4f}   <- primary metric")
print(f"  Test Accuracy    : {accuracy:.4f}")
print(f"  Test ROC-AUC     : {roc_auc:.4f}" if roc_auc else "  Test ROC-AUC     : N/A")
print(f"  Benign FP rate   : {fp_rate*100:.2f}%" if fp_rate is not None else "")
print(f"\n  vs LR  : Macro F1 {LR_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-LR_MACRO_F1:+.4f})")
print(f"  vs RF  : Macro F1 {RF_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-RF_MACRO_F1:+.4f})")
print(f"\n  Outputs:")
print(f"    models/xgb_model.ubj")
print(f"    reports/xgb_metrics.json")
print(f"    reports/xgb_confusion_matrix.png")
print(f"    reports/xgb_per_class_metrics.png")
print(f"    reports/xgb_feature_importance.png")
print(f"{'='*62}")
