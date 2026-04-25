"""
lightgbm_model.py  --  LightGBM (GPU) for IoT Intrusion Detection
==================================================================
Why LightGBM beats XGBoost here:
  - Leaf-wise tree growth finds better splits faster
  - 3-5x faster training, lower memory
  - Histogram-based algorithm is already in LightGBM's DNA

Strategy:
  1. Load preprocessed train.csv / test.csv
  2. Hold out 20% of train as validation for early stopping
  3. Grid search: num_leaves x learning_rate
  4. Retrain best config on full training set
  5. Evaluate on locked test set
  6. Save model + full metrics report

GPU: tries device='gpu', falls back to CPU if unavailable
Imbalance: inverse-frequency sample weights
"""

import os, json, time, warnings
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
VAL_FRAC     = 0.20
EARLY_STOP   = 50
MAX_ROUNDS   = 2000

PARAM_GRID = {
    "num_leaves"       : [63, 127],
    "learning_rate"    : [0.1, 0.05],
    "min_child_samples": [20],
    "feature_fraction" : [0.8],
    "bagging_fraction" : [0.8],
    "bagging_freq"     : [5],
}

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  IoT IDS -- LightGBM Training")
print("=" * 62)

for path in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train_full  = train_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_train_full  = train_df[LABEL_COL].values
X_test        = test_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_test        = test_df[LABEL_COL].values
feature_names = train_df.drop(columns=[LABEL_COL]).columns.tolist()

le          = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
class_names = le.classes_.tolist()
n_classes   = len(class_names)

print(f"\n  Train : {X_train_full.shape[0]:,} rows x {X_train_full.shape[1]} features")
print(f"  Test  : {X_test.shape[0]:,} rows  x {X_test.shape[1]} features")
print(f"  Classes: {n_classes}")

# ── CLASS WEIGHTS (inverse frequency) ────────────────────────────────────────
unique, counts = np.unique(y_train_full, return_counts=True)
freq             = counts / counts.sum()
class_weight_arr = 1.0 / (freq * n_classes)

def make_sample_weights(y, cw_arr):
    return cw_arr[y].astype(np.float32)

# ── TRAIN / VAL SPLIT ─────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size    = VAL_FRAC,
    stratify     = y_train_full,
    random_state = RANDOM_STATE,
)
w_tr  = make_sample_weights(y_tr,  class_weight_arr)
w_val = make_sample_weights(y_val, class_weight_arr)

print(f"\n  Train subset : {len(X_tr):,}  |  Val : {len(X_val):,}")

dtrain = lgb.Dataset(X_tr,  label=y_tr,  weight=w_tr,  feature_name=feature_names, free_raw_data=False)
dval   = lgb.Dataset(X_val, label=y_val, weight=w_val, feature_name=feature_names, reference=dtrain, free_raw_data=False)

# ── DETECT GPU ────────────────────────────────────────────────────────────────
def detect_lgbm_device():
    try:
        tparams = {
            "objective": "multiclass", "num_class": 2, "device": "gpu",
            "num_leaves": 7, "verbosity": -1, "num_threads": 1,
        }
        tdata = lgb.Dataset(
            np.random.rand(200, 4).astype(np.float32),
            label=np.random.randint(0, 2, 200),
            free_raw_data=True,
        )
        lgb.train(tparams, tdata, num_boost_round=3)
        print("  GPU available — using device='gpu'")
        return "gpu"
    except Exception as e:
        print(f"  GPU not available ({e.__class__.__name__}) — using CPU")
        return "cpu"

DEVICE = detect_lgbm_device()

# ── HYPERPARAMETER SEARCH ─────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Grid search  num_leaves={PARAM_GRID['num_leaves']}")
print(f"               learning_rate={PARAM_GRID['learning_rate']}")
print(f"  Early stop : {EARLY_STOP} rounds  |  Max rounds: {MAX_ROUNDS}")
print(f"  Device     : {DEVICE}")
print(f"{'='*62}\n")

combos = [
    {
        "num_leaves": nl, "learning_rate": lr,
        "min_child_samples": mcs, "feature_fraction": ff,
        "bagging_fraction": bf, "bagging_freq": bfreq,
    }
    for nl    in PARAM_GRID["num_leaves"]
    for lr    in PARAM_GRID["learning_rate"]
    for mcs   in PARAM_GRID["min_child_samples"]
    for ff    in PARAM_GRID["feature_fraction"]
    for bf    in PARAM_GRID["bagging_fraction"]
    for bfreq in PARAM_GRID["bagging_freq"]
]

grid_results = {}
best_val_f1  = -1.0
best_params  = None
best_rounds  = None

for i, combo in enumerate(combos, 1):
    params = {
        "objective"        : "multiclass",
        "num_class"        : n_classes,
        "metric"           : "multi_logloss",
        "device"           : DEVICE,
        "num_leaves"       : combo["num_leaves"],
        "learning_rate"    : combo["learning_rate"],
        "min_child_samples": combo["min_child_samples"],
        "feature_fraction" : combo["feature_fraction"],
        "bagging_fraction" : combo["bagging_fraction"],
        "bagging_freq"     : combo["bagging_freq"],
        "max_depth"        : -1,
        "min_split_gain"   : 0.0,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 0.1,
        "n_jobs"           : -1,
        "seed"             : RANDOM_STATE,
        "verbosity"        : -1,
    }

    label = f"leaves={combo['num_leaves']} lr={combo['learning_rate']}"
    print(f"  [{i}/{len(combos)}] {label} ...", flush=True)
    t0 = time.time()

    callbacks = [
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=-1),
    ]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round = MAX_ROUNDS,
        valid_sets      = [dval],
        callbacks       = callbacks,
    )

    y_val_prob = model.predict(X_val)
    y_val_pred = y_val_prob.argmax(axis=1)
    val_f1  = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    val_acc = accuracy_score(y_val, y_val_pred)
    best_iter = model.best_iteration
    elapsed   = time.time() - t0

    grid_results[label] = {
        "params"      : combo,
        "val_macro_f1": val_f1,
        "val_accuracy": val_acc,
        "best_iter"   : best_iter,
        "time_s"      : elapsed,
    }
    print(f"    val macro-F1={val_f1:.4f}  acc={val_acc:.4f}  rounds={best_iter}  [{elapsed:.0f}s]")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_params = combo
        best_rounds = best_iter

# ── BEST CONFIG ───────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Best: leaves={best_params['num_leaves']}  lr={best_params['learning_rate']}")
print(f"  Val macro-F1 : {best_val_f1:.4f}")
print(f"  Rounds       : {best_rounds}")
print(f"{'='*62}\n")

# ── TRAIN FINAL MODEL ON FULL TRAINING SET ────────────────────────────────────
print("Training final model on full training set ...")
w_full = make_sample_weights(y_train_full, class_weight_arr)
dfull  = lgb.Dataset(
    X_train_full, label=y_train_full,
    weight=w_full, feature_name=feature_names,
)

final_params = {
    "objective"        : "multiclass",
    "num_class"        : n_classes,
    "metric"           : "multi_logloss",
    "device"           : DEVICE,
    "num_leaves"       : best_params["num_leaves"],
    "learning_rate"    : best_params["learning_rate"],
    "min_child_samples": best_params["min_child_samples"],
    "feature_fraction" : best_params["feature_fraction"],
    "bagging_fraction" : best_params["bagging_fraction"],
    "bagging_freq"     : best_params["bagging_freq"],
    "max_depth"        : -1,
    "min_split_gain"   : 0.0,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "n_jobs"           : -1,
    "seed"             : RANDOM_STATE,
    "verbosity"        : -1,
}

t0 = time.time()
final_model = lgb.train(
    final_params,
    dfull,
    num_boost_round = best_rounds,
)
train_time = time.time() - t0
print(f"  Done in {train_time:.1f}s  |  {best_rounds} rounds")

# ── EVALUATE ON LOCKED TEST SET ───────────────────────────────────────────────
print(f"\n{'='*62}")
print("  Final evaluation on locked test set")
print(f"{'='*62}")

y_pred_prob = final_model.predict(X_test)
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
ax.set_title(f"LightGBM -- Normalised Confusion Matrix\n"
             f"Macro F1={macro_f1:.4f}  Acc={accuracy:.4f}", fontsize=13)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True",      fontsize=11)
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "lgbm_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/lgbm_confusion_matrix.png")

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
plt.suptitle("LightGBM -- Per-class Metrics (Test Set)", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "lgbm_per_class_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/lgbm_per_class_metrics.png")

fi_gain = final_model.feature_importance(importance_type="gain")
fi_idx  = np.argsort(fi_gain)[::-1]
fi_names_sorted = np.array(feature_names)[fi_idx]
fi_vals_sorted  = fi_gain[fi_idx]

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#c0392b" if v > np.percentile(fi_vals_sorted, 75) else "steelblue"
          for v in fi_vals_sorted]
ax.barh(fi_names_sorted[::-1], fi_vals_sorted[::-1], color=colors[::-1], edgecolor="white")
ax.set_xlabel("Gain", fontsize=11)
ax.set_title("LightGBM -- Feature Importances (Gain)\n(red = top 25%)", fontsize=12)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "lgbm_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/lgbm_feature_importance.png")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "lgbm_model.pkl")
joblib.dump(final_model, model_path)
print(f"  Model saved -> {model_path}")

# ── SAVE METRICS ──────────────────────────────────────────────────────────────
LR_MACRO_F1 = 0.4401; RF_MACRO_F1 = 0.5861; XGB_MACRO_F1 = 0.6200

metrics = {
    "model"               : "LightGBM",
    "device"              : DEVICE,
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
    "vs_lr_macro_f1_delta" : round(macro_f1 - LR_MACRO_F1,  4),
    "vs_rf_macro_f1_delta" : round(macro_f1 - RF_MACRO_F1,  4),
    "vs_xgb_macro_f1_delta": round(macro_f1 - XGB_MACRO_F1, 4),
    "grid_results": {
        k: {kk: round(vv, 4) if isinstance(vv, float) else vv
            for kk, vv in v.items() if kk != "params"}
        for k, v in grid_results.items()
    },
    "feature_importance_gain": {
        feature_names[i]: round(float(fi_gain[i]), 4) for i in fi_idx
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

metrics_path = os.path.join(REPORTS_DIR, "lgbm_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved -> {metrics_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  LIGHTGBM -- FINAL RESULTS")
print(f"{'='*62}")
print(f"  leaves={best_params['num_leaves']}  lr={best_params['learning_rate']}  rounds={best_rounds}")
print(f"  Test Macro F1    : {macro_f1:.4f}   <- primary metric")
print(f"  Test Accuracy    : {accuracy:.4f}")
print(f"  Test ROC-AUC     : {roc_auc:.4f}" if roc_auc else "  Test ROC-AUC     : N/A")
print(f"  Benign FP rate   : {fp_rate*100:.2f}%" if fp_rate is not None else "")
print(f"\n  vs LR  : Macro F1 {LR_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-LR_MACRO_F1:+.4f})")
print(f"  vs RF  : Macro F1 {RF_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-RF_MACRO_F1:+.4f})")
print(f"  vs XGB : Macro F1 {XGB_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-XGB_MACRO_F1:+.4f})")
print(f"\n  Outputs:")
print(f"    models/lgbm_model.pkl")
print(f"    reports/lgbm_metrics.json")
print(f"    reports/lgbm_confusion_matrix.png")
print(f"    reports/lgbm_per_class_metrics.png")
print(f"    reports/lgbm_feature_importance.png")
print(f"{'='*62}")
