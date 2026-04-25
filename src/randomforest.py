"""
train_rf.py  —  Random Forest (strong model) for IoT Intrusion Detection
=========================================================================
Strategy to maximise macro-F1 / recall / precision / accuracy:

  1. Load preprocessed train.csv (27 features, 463k rows, 34 classes)
  2. Tune max_depth and min_samples_leaf via StratifiedKFold CV (3-fold)
     — n_estimators fixed at 500 (stable, not wasteful)
     — class_weight='balanced_subsample' (best RF setting for imbalance)
  3. Re-train best config on full training set
  4. Evaluate on locked test.csv
  5. Feature importance plot + full metrics report

Key choices explained:
  - class_weight='balanced_subsample' : recomputes weights per bootstrap
    sample — far better than 'balanced' for RF on rare classes
  - max_features='sqrt'               : sqrt(27)≈5, reduces tree correlation
  - bootstrap=True                    : standard bagging for variance reduction
  - 3-fold CV not 5-fold              : RF on 463k rows is slow; 3 folds
    are still statistically reliable and 40% faster
  - oob_score=True on final model     : free out-of-bag validation estimate

Usage:
    python src/train_rf.py
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TRAIN_PATH   = "data/train.csv"
TEST_PATH    = "data/test.csv"
LABEL_COL    = "Label"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
RANDOM_STATE = 42
CV_FOLDS     = 3        # 3-fold: reliable + much faster than 5 for RF

# CV subsample: tune on a stratified fraction to avoid OOM on 2.9M rows.
# The final model is always trained on the full training set.
CV_SAMPLE_FRAC   = 0.15   # 15% ≈ 440k rows — large enough to be representative

# Fixed hyperparameters
N_ESTIMATORS_CV   = 100   # reduced during CV for speed/memory; full model uses 300
N_ESTIMATORS_FINAL = 300  # final model (balanced between quality and memory)
MAX_FEATURES  = "sqrt"    # sqrt(27) ≈ 5 features per split — standard for classification

# Hyperparameter grid to tune via CV
# Kept small and targeted to avoid days of runtime
PARAM_GRID = {
    "max_depth"        : [30, 50],  # None removed: too memory-heavy with 500 trees on 2.9M
    "min_samples_leaf" : [1, 2],    # 1 = essential for rare classes
}

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── STEP 1: LOAD DATA ──────────────────────────────────────────────────────────
print("=" * 62)
print("  IoT IDS — Random Forest Training")
print("=" * 62)

for path in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")

print("\nLoading data ...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train       = train_df.drop(columns=[LABEL_COL]).values
y_train       = train_df[LABEL_COL].values
X_test        = test_df.drop(columns=[LABEL_COL]).values
y_test        = test_df[LABEL_COL].values
feature_names = train_df.drop(columns=[LABEL_COL]).columns.tolist()

print(f"  Train : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"  Test  : {X_test.shape[0]:,} rows  × {X_test.shape[1]} features")
print(f"  Classes: {len(np.unique(y_train))}")

# ── STEP 2: LOAD LABEL ENCODER ────────────────────────────────────────────────
le          = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
class_names = le.classes_.tolist()
n_classes   = len(class_names)

# ── STEP 3: CLASS DISTRIBUTION ────────────────────────────────────────────────
print("\nClass distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for cls_idx, cnt in zip(unique, counts):
    flag = "  ⚠ rare" if cnt < 100 else ""
    print(f"  {class_names[cls_idx]:<40} {cnt:>7,}{flag}")

# ── STEP 4: SUBSAMPLE FOR CV (avoids OOM on 2.9 M rows) ──────────────────────
from sklearn.model_selection import train_test_split

n_cv_samples = int(len(X_train) * CV_SAMPLE_FRAC)
X_cv, _, y_cv, _ = train_test_split(
    X_train, y_train,
    train_size   = CV_SAMPLE_FRAC,
    stratify     = y_train,
    random_state = RANDOM_STATE,
)
print(f"\n  CV subsample: {len(X_cv):,} rows ({CV_SAMPLE_FRAC*100:.0f}% of training set)")
print(f"  Final model will be trained on full {len(X_train):,} rows\n")

# ── STEP 5: CROSS-VALIDATION HYPERPARAMETER SEARCH ────────────────────────────
print(f"{'='*62}")
print(f"  Tuning via {CV_FOLDS}-fold Stratified CV")
print(f"  max_depth        : {PARAM_GRID['max_depth']}")
print(f"  min_samples_leaf : {PARAM_GRID['min_samples_leaf']}")
print(f"  n_estimators     : {N_ESTIMATORS_CV} (CV) / {N_ESTIMATORS_FINAL} (final)")
print(f"  class_weight     : balanced_subsample (fixed)")
print(f"{'='*62}")

skf        = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
best_macro_f1 = -1
best_params   = {}

total_combos = len(PARAM_GRID["max_depth"]) * len(PARAM_GRID["min_samples_leaf"])
combo_num    = 0

for max_depth in PARAM_GRID["max_depth"]:
    for min_leaf in PARAM_GRID["min_samples_leaf"]:
        combo_num += 1
        depth_label = str(max_depth) if max_depth is not None else "None"
        key = f"depth={depth_label}_leaf={min_leaf}"

        print(f"\n  [{combo_num}/{total_combos}] max_depth={depth_label}"
              f"  min_samples_leaf={min_leaf} ...", flush=True)
        t0 = time.time()

        model = RandomForestClassifier(
            n_estimators     = N_ESTIMATORS_CV,
            max_depth        = max_depth,
            min_samples_leaf = min_leaf,
            max_features     = MAX_FEATURES,
            class_weight     = "balanced_subsample",  # best for RF + imbalance
            bootstrap        = True,
            oob_score        = False,   # disabled during CV for speed
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
        )

        scores = cross_validate(
            model,
            X_cv, y_cv,
            cv      = skf,
            scoring = {
                "macro_f1"  : "f1_macro",
                "accuracy"  : "accuracy",
                "macro_prec": "precision_macro",
                "macro_rec" : "recall_macro",
            },
            return_train_score = False,
            n_jobs = 1,
        )

        elapsed       = time.time() - t0
        mf1_mean      = scores["test_macro_f1"].mean()
        mf1_std       = scores["test_macro_f1"].std()
        acc_mean      = scores["test_accuracy"].mean()
        prec_mean     = scores["test_macro_prec"].mean()
        rec_mean      = scores["test_macro_rec"].mean()

        cv_results[key] = {
            "max_depth"        : max_depth,
            "min_samples_leaf" : min_leaf,
            "macro_f1_mean"    : mf1_mean,
            "macro_f1_std"     : mf1_std,
            "accuracy_mean"    : acc_mean,
            "precision_mean"   : prec_mean,
            "recall_mean"      : rec_mean,
            "time_s"           : elapsed,
        }

        print(f"    macro-F1={mf1_mean:.4f} (±{mf1_std:.4f})"
              f"  acc={acc_mean:.4f}"
              f"  prec={prec_mean:.4f}"
              f"  rec={rec_mean:.4f}"
              f"  [{elapsed:.0f}s]")

        if mf1_mean > best_macro_f1:
            best_macro_f1 = mf1_mean
            best_params   = {
                "max_depth"        : max_depth,
                "min_samples_leaf" : min_leaf,
            }

# ── STEP 5: REPORT BEST CONFIG ────────────────────────────────────────────────
best_key = (f"depth={best_params['max_depth']}_"
            f"leaf={best_params['min_samples_leaf']}")
best_cv  = cv_results[best_key]

print(f"\n{'='*62}")
print(f"  Best config:")
print(f"    max_depth        = {best_params['max_depth']}")
print(f"    min_samples_leaf = {best_params['min_samples_leaf']}")
print(f"  CV macro-F1  : {best_cv['macro_f1_mean']:.4f} (±{best_cv['macro_f1_std']:.4f})")
print(f"  CV accuracy  : {best_cv['accuracy_mean']:.4f}")
print(f"  CV precision : {best_cv['precision_mean']:.4f}")
print(f"  CV recall    : {best_cv['recall_mean']:.4f}")
print(f"{'='*62}")

# ── STEP 6: TRAIN FINAL MODEL ON FULL TRAINING SET ────────────────────────────
print(f"\nTraining final RF on full training set ...")
print(f"  n_estimators={N_ESTIMATORS_FINAL}, max_depth={best_params['max_depth']},"
      f" min_samples_leaf={best_params['min_samples_leaf']}")
t0 = time.time()

final_model = RandomForestClassifier(
    n_estimators     = N_ESTIMATORS_FINAL,
    max_depth        = best_params["max_depth"],
    min_samples_leaf = best_params["min_samples_leaf"],
    max_features     = MAX_FEATURES,
    class_weight     = "balanced_subsample",
    bootstrap        = True,
    oob_score        = True,    # free validation estimate on full model
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
)
final_model.fit(X_train, y_train)
train_time = time.time() - t0

print(f"  Training complete in {train_time:.1f}s")
print(f"  OOB accuracy (free estimate): {final_model.oob_score_:.4f}")

# ── STEP 7: EVALUATE ON LOCKED TEST SET ───────────────────────────────────────
print(f"\n{'='*62}")
print("  Final evaluation on locked test set")
print(f"{'='*62}")

y_pred      = final_model.predict(X_test)
y_pred_prob = final_model.predict_proba(X_test)

macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
micro_f1    = f1_score(y_test, y_pred, average="micro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
accuracy    = accuracy_score(y_test, y_pred)
macro_prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_rec   = recall_score(y_test, y_pred,    average="macro", zero_division=0)

print(f"\n  Macro F1-score    : {macro_f1:.4f}   <- primary metric")
print(f"  Micro F1-score    : {micro_f1:.4f}")
print(f"  Weighted F1-score : {weighted_f1:.4f}")
print(f"  Accuracy          : {accuracy:.4f}")
print(f"  Macro Precision   : {macro_prec:.4f}")
print(f"  Macro Recall      : {macro_rec:.4f}")
print(f"  OOB Score         : {final_model.oob_score_:.4f}")

# ROC-AUC
try:
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    roc_auc    = roc_auc_score(
        y_test_bin, y_pred_prob,
        average="macro", multi_class="ovr"
    )
    print(f"  ROC-AUC (macro)   : {roc_auc:.4f}")
except Exception:
    roc_auc = None
    print("  ROC-AUC           : could not compute")

# Benign false positive rate
benign_idx = class_names.index("BENIGN") if "BENIGN" in class_names else None
if benign_idx is not None:
    benign_mask  = (y_test == benign_idx)
    benign_preds = y_pred[benign_mask]
    fp_rate      = (benign_preds != benign_idx).sum() / benign_mask.sum()
    print(f"  Benign FP rate    : {fp_rate*100:.2f}%  (LR baseline was 67.48%)")
else:
    fp_rate = None

# Per-class report
print(f"\n  Per-class report:")
report_str  = classification_report(
    y_test, y_pred,
    target_names  = class_names,
    zero_division = 0,
    digits        = 4,
)
print(report_str)

report_dict = classification_report(
    y_test, y_pred,
    target_names  = class_names,
    zero_division = 0,
    output_dict   = True,
)

# ── STEP 8: CONFUSION MATRIX ───────────────────────────────────────────────────
print("Generating plots ...")

cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(20, 17))
sns.heatmap(
    cm_norm,
    annot       = True,
    fmt         = ".2f",
    cmap        = "Blues",
    xticklabels = class_names,
    yticklabels = class_names,
    ax          = ax,
    linewidths  = 0.3,
    linecolor   = "gray",
    annot_kws   = {"size": 7},
)
ax.set_title(
    f"Random Forest — Normalised Confusion Matrix (Test Set)\n"
    f"Macro F1={macro_f1:.4f}  Accuracy={accuracy:.4f}",
    fontsize=13, pad=14
)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label",      fontsize=11)
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()

cm_path = os.path.join(REPORTS_DIR, "rf_confusion_matrix.png")
fig.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {cm_path}")

# ── STEP 9: PER-CLASS METRICS BAR CHART ───────────────────────────────────────
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
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(
        x     = np.mean(values),
        color = "red", linestyle="--", linewidth=1.2,
        label = f"Macro avg = {np.mean(values):.3f}"
    )
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

plt.suptitle(
    "Random Forest — Per-class Metrics (Test Set)", fontsize=14, y=1.005
)
plt.tight_layout()

perclass_path = os.path.join(REPORTS_DIR, "rf_per_class_metrics.png")
fig.savefig(perclass_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {perclass_path}")

# ── STEP 10: FEATURE IMPORTANCE PLOT ──────────────────────────────────────────
importances  = final_model.feature_importances_
indices      = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in indices]
sorted_imps  = importances[indices]

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#c0392b" if imp > np.percentile(sorted_imps, 75) else "steelblue"
          for imp in sorted_imps]
ax.barh(sorted_names[::-1], sorted_imps[::-1], color=colors[::-1], edgecolor="white")
ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
ax.set_title("Random Forest — Feature Importances\n(red = top 25%)", fontsize=12)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()

fi_path = os.path.join(REPORTS_DIR, "rf_feature_importance.png")
fig.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {fi_path}")

# ── STEP 11: CV RESULTS HEATMAP ───────────────────────────────────────────────
depths = [str(d) for d in PARAM_GRID["max_depth"]]
leaves = [str(l) for l in PARAM_GRID["min_samples_leaf"]]

heatmap_data = np.zeros((len(depths), len(leaves)))
for i, d in enumerate(PARAM_GRID["max_depth"]):
    for j, l in enumerate(PARAM_GRID["min_samples_leaf"]):
        key = f"depth={d}_leaf={l}"
        heatmap_data[i, j] = cv_results[key]["macro_f1_mean"]

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    heatmap_data,
    annot       = True,
    fmt         = ".4f",
    cmap        = "YlGnBu",
    xticklabels = [f"leaf={l}" for l in leaves],
    yticklabels = [f"depth={d}" for d in depths],
    ax          = ax,
    linewidths  = 0.5,
)
ax.set_title("CV Macro F1 — Hyperparameter Grid", fontsize=12)
plt.tight_layout()

cv_path = os.path.join(REPORTS_DIR, "rf_cv_heatmap.png")
fig.savefig(cv_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {cv_path}")

# ── STEP 12: COMPARISON WITH LR BASELINE ──────────────────────────────────────
LR_MACRO_F1  = 0.3051
LR_ACCURACY  = 0.5591
LR_FP_RATE   = 0.6748

print(f"\n{'='*62}")
print("  RF vs LR Baseline Comparison")
print(f"{'='*62}")
print(f"  {'Metric':<25} {'LR Baseline':>12} {'Random Forest':>14} {'Δ':>8}")
print(f"  {'-'*60}")
print(f"  {'Macro F1':<25} {LR_MACRO_F1:>12.4f} {macro_f1:>14.4f}"
      f" {macro_f1-LR_MACRO_F1:>+8.4f}")
print(f"  {'Accuracy':<25} {LR_ACCURACY:>12.4f} {accuracy:>14.4f}"
      f" {accuracy-LR_ACCURACY:>+8.4f}")
if fp_rate is not None:
    print(f"  {'Benign FP rate':<25} {LR_FP_RATE:>11.2%} {fp_rate:>13.2%}"
          f" {fp_rate-LR_FP_RATE:>+8.2%}")

# ── STEP 13: SAVE MODEL ───────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
joblib.dump(final_model, model_path)
print(f"\n  Model saved -> {model_path}")

# ── STEP 14: SAVE FULL METRICS JSON ───────────────────────────────────────────
metrics = {
    "model"                    : "RandomForest",
    "best_params"              : {
        "n_estimators"     : N_ESTIMATORS_FINAL,
        "max_depth"        : best_params["max_depth"],
        "min_samples_leaf" : best_params["min_samples_leaf"],
        "max_features"     : MAX_FEATURES,
        "class_weight"     : "balanced_subsample",
    },
    "train_time_s"             : round(train_time, 2),
    "cv_folds"                 : CV_FOLDS,
    "cv_macro_f1_mean"         : round(best_cv["macro_f1_mean"],  4),
    "cv_macro_f1_std"          : round(best_cv["macro_f1_std"],   4),
    "cv_accuracy_mean"         : round(best_cv["accuracy_mean"],  4),
    "cv_precision_mean"        : round(best_cv["precision_mean"], 4),
    "cv_recall_mean"           : round(best_cv["recall_mean"],    4),
    "oob_score"                : round(final_model.oob_score_, 4),
    "test_macro_f1"            : round(macro_f1,    4),
    "test_micro_f1"            : round(micro_f1,    4),
    "test_weighted_f1"         : round(weighted_f1, 4),
    "test_accuracy"            : round(accuracy,    4),
    "test_macro_precision"     : round(macro_prec,  4),
    "test_macro_recall"        : round(macro_rec,   4),
    "test_roc_auc_macro"       : round(roc_auc, 4) if roc_auc else None,
    "benign_false_positive_rate": round(fp_rate, 4) if fp_rate is not None else None,
    "improvement_over_lr"      : {
        "macro_f1_delta"   : round(macro_f1 - LR_MACRO_F1, 4),
        "accuracy_delta"   : round(accuracy  - LR_ACCURACY, 4),
    },
    "feature_importances"      : {
        feature_names[i]: round(float(importances[i]), 6)
        for i in np.argsort(importances)[::-1]
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
    "cv_all_results": {
        k: {
            "macro_f1_mean": round(v["macro_f1_mean"], 4),
            "macro_f1_std" : round(v["macro_f1_std"],  4),
            "accuracy_mean": round(v["accuracy_mean"], 4),
            "time_s"       : round(v["time_s"],        1),
        }
        for k, v in cv_results.items()
    },
}

metrics_path = os.path.join(REPORTS_DIR, "rf_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved -> {metrics_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  RANDOM FOREST — FINAL RESULTS")
print(f"{'='*62}")
print(f"  Best max_depth        : {best_params['max_depth']}")
print(f"  Best min_samples_leaf : {best_params['min_samples_leaf']}")
print(f"  OOB Score             : {final_model.oob_score_:.4f}")
print(f"  Test Macro F1         : {macro_f1:.4f}   <- primary metric")
print(f"  Test Accuracy         : {accuracy:.4f}")
print(f"  Test Macro Recall     : {macro_rec:.4f}")
print(f"  Test Macro Precision  : {macro_prec:.4f}")
if roc_auc:
    print(f"  Test ROC-AUC          : {roc_auc:.4f}")
if fp_rate is not None:
    print(f"  Benign FP rate        : {fp_rate*100:.2f}%")
print(f"\n  vs LR Baseline:")
print(f"    Macro F1  : {LR_MACRO_F1:.4f} -> {macro_f1:.4f}"
      f"  ({macro_f1-LR_MACRO_F1:+.4f})")
print(f"    Accuracy  : {LR_ACCURACY:.4f} -> {accuracy:.4f}"
      f"  ({accuracy-LR_ACCURACY:+.4f})")
print(f"\n  Top 5 most important features:")
for i in range(min(5, len(feature_names))):
    idx = indices[i]
    print(f"    {i+1}. {feature_names[idx]:<30} {importances[idx]:.4f}")
print(f"\n  Outputs:")
print(f"    models/rf_model.pkl")
print(f"    reports/rf_metrics.json")
print(f"    reports/rf_confusion_matrix.png")
print(f"    reports/rf_per_class_metrics.png")
print(f"    reports/rf_feature_importance.png")
print(f"    reports/rf_cv_heatmap.png")
print(f"{'='*62}")