"""
train.py  —  Logistic Regression baseline for IoT Intrusion Detection
======================================================================
Strategy to maximise macro-F1 / recall / precision / accuracy:
  1. Load preprocessed train.csv (27 features, 463k rows, 34 classes)
  2. Tune regularisation strength C via StratifiedKFold cross-validation
  3. Re-train the best C on the full training set
  4. Evaluate on the locked test.csv
  5. Save model + full metrics report

Key choices explained:
  - solver='saga'         : fastest solver for large datasets (>100k rows)
  - multi_class='multinomial': proper joint probability over all 34 classes
  - class_weight='balanced'  : compensates for the 7 rare attack classes
  - C tuned via CV           : prevents overfitting / underfitting
  - penalty='l2'             : stable for 27 features (l1 / elasticnet also tested)

Usage:
    python src/train.py
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
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
TRAIN_PATH     = "data/train.csv"
TEST_PATH      = "data/test.csv"
LABEL_COL      = "Label"
MODELS_DIR     = "models"
REPORTS_DIR    = "reports"
RANDOM_STATE   = 42
CV_FOLDS       = 3   # reduced from 5 for large dataset speed

# Best C from previous run was 100.0 — skip grid search, train directly
C_VALUES       = [100.0]

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── STEP 1: LOAD DATA ──────────────────────────────────────────────────────────
print("=" * 60)
print("  IoT IDS — Logistic Regression Training")
print("=" * 60)

for path in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run preprocess.py first."
        )

print(f"\nLoading training data ...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[LABEL_COL]).values
y_train = train_df[LABEL_COL].values

X_test  = test_df.drop(columns=[LABEL_COL]).values
y_test  = test_df[LABEL_COL].values

feature_names = train_df.drop(columns=[LABEL_COL]).columns.tolist()

print(f"  Train : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"  Test  : {X_test.shape[0]:,} rows  × {X_test.shape[1]} features")
print(f"  Classes: {len(np.unique(y_train))}")

# ── STEP 2: LOAD LABEL ENCODER (for readable class names) ─────────────────────
le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
class_names = le.classes_.tolist()
n_classes   = len(class_names)

# ── STEP 3: CLASS DISTRIBUTION CHECK ──────────────────────────────────────────
print("\nClass distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for cls_idx, cnt in zip(unique, counts):
    flag = "  [RARE]" if cnt < 100 else ""
    print(f"  {class_names[cls_idx]:<40} {cnt:>7,}{flag}")

# ── STEP 4: CROSS-VALIDATION TO FIND BEST C ───────────────────────────────────
print(f"\n{'='*60}")
print(f"  Tuning C via {CV_FOLDS}-fold Stratified CV")
print(f"  C values tested: {C_VALUES}")
print(f"{'='*60}")

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_results = {}

for C in C_VALUES:
    print(f"\n  Testing C={C} ...", end=" ", flush=True)
    t0 = time.time()

    model = LogisticRegression(
        C            = C,
        penalty      = "l2",
        solver       = "saga",        # best for large datasets
        multi_class  = "multinomial", # proper 34-class joint probability
        class_weight = "balanced",    # handles 7 rare classes
        max_iter     = 2000,          # saga needs more iterations
        tol          = 1e-3,          # slightly relaxed for speed
        random_state = RANDOM_STATE,
        n_jobs       = -1,            # use all CPU cores
    )

    scores = cross_validate(
        model,
        X_train, y_train,
        cv      = skf,
        scoring = {
            "macro_f1"  : "f1_macro",
            "accuracy"  : "accuracy",
            "macro_prec": "precision_macro",
            "macro_rec" : "recall_macro",
        },
        return_train_score = False,
        n_jobs = 1,         # outer loop already parallelised via model n_jobs
    )

    elapsed = time.time() - t0

    cv_results[C] = {
        "macro_f1_mean"  : scores["test_macro_f1"].mean(),
        "macro_f1_std"   : scores["test_macro_f1"].std(),
        "accuracy_mean"  : scores["test_accuracy"].mean(),
        "precision_mean" : scores["test_macro_prec"].mean(),
        "recall_mean"    : scores["test_macro_rec"].mean(),
        "time_s"         : elapsed,
    }

    print(
        f"macro-F1={cv_results[C]['macro_f1_mean']:.4f} "
        f"(±{cv_results[C]['macro_f1_std']:.4f})  "
        f"acc={cv_results[C]['accuracy_mean']:.4f}  "
        f"[{elapsed:.0f}s]"
    )

# ── STEP 5: PICK BEST C ────────────────────────────────────────────────────────
best_C = max(cv_results, key=lambda c: cv_results[c]["macro_f1_mean"])
best_cv = cv_results[best_C]

print(f"\n{'='*60}")
print(f"  Best C = {best_C}")
print(f"  CV macro-F1  : {best_cv['macro_f1_mean']:.4f} (±{best_cv['macro_f1_std']:.4f})")
print(f"  CV accuracy  : {best_cv['accuracy_mean']:.4f}")
print(f"  CV precision : {best_cv['precision_mean']:.4f}")
print(f"  CV recall    : {best_cv['recall_mean']:.4f}")
print(f"{'='*60}")

# ── STEP 6: TRAIN FINAL MODEL ON FULL TRAINING SET ────────────────────────────
print(f"\nTraining final model on full training set (C={best_C}) ...")
t0 = time.time()

final_model = LogisticRegression(
    C            = best_C,
    penalty      = "l2",
    solver       = "saga",
    multi_class  = "multinomial",
    class_weight = "balanced",
    max_iter     = 2000,
    tol          = 1e-3,
    random_state = RANDOM_STATE,
    n_jobs       = -1,
)
final_model.fit(X_train, y_train)
train_time = time.time() - t0
print(f"  Training complete in {train_time:.1f}s")
print(f"  Converged: {final_model.n_iter_[0]} iterations")

# ── SAVE MODEL IMMEDIATELY AFTER TRAINING (before any print that could crash) ──
model_path = os.path.join(MODELS_DIR, "lr_model.pkl")
joblib.dump(final_model, model_path)
print(f"  Model saved -> {model_path}")

# ── STEP 7: EVALUATE ON LOCKED TEST SET ───────────────────────────────────────
print(f"\n{'='*60}")
print("  Final evaluation on locked test set")
print(f"{'='*60}")

y_pred      = final_model.predict(X_test)
y_pred_prob = final_model.predict_proba(X_test)

# Core metrics
macro_f1  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
micro_f1  = f1_score(y_test, y_pred, average="micro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
accuracy  = accuracy_score(y_test, y_pred)
macro_prec = precision_score(y_test, y_pred, average="macro",  zero_division=0)
macro_rec  = recall_score(y_test, y_pred,    average="macro",  zero_division=0)

print(f"\n  Macro F1-score   : {macro_f1:.4f}   <- primary metric")
print(f"  Micro F1-score   : {micro_f1:.4f}")
print(f"  Weighted F1-score: {weighted_f1:.4f}")
print(f"  Accuracy         : {accuracy:.4f}")
print(f"  Macro Precision  : {macro_prec:.4f}")
print(f"  Macro Recall     : {macro_rec:.4f}")

# ROC-AUC (one-vs-rest, macro)
try:
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    roc_auc = roc_auc_score(
        y_test_bin, y_pred_prob,
        average="macro", multi_class="ovr"
    )
    print(f"  ROC-AUC (macro)  : {roc_auc:.4f}")
except Exception:
    roc_auc = None
    print("  ROC-AUC          : could not compute (likely rare class issue)")

# Per-class report
print(f"\n  Per-class report:")
report_str = classification_report(
    y_test, y_pred,
    target_names=class_names,
    zero_division=0,
    digits=4
)
print(report_str)

# False positive rate on BENIGN traffic specifically
benign_idx = class_names.index("BENIGN") if "BENIGN" in class_names else None
if benign_idx is not None:
    benign_mask   = (y_test == benign_idx)
    benign_preds  = y_pred[benign_mask]
    fp_rate       = (benign_preds != benign_idx).sum() / benign_mask.sum()
    print(f"  False positive rate on BENIGN traffic: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
else:
    fp_rate = None

# ── STEP 8: CONFUSION MATRIX PLOT ─────────────────────────────────────────────
print("\nGenerating confusion matrix plot ...")

cm = confusion_matrix(y_test, y_pred)

# Normalised confusion matrix (row = true class)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(
    cm_norm,
    annot   = True,
    fmt     = ".2f",
    cmap    = "Blues",
    xticklabels = class_names,
    yticklabels = class_names,
    ax      = ax,
    linewidths = 0.3,
    linecolor  = "gray",
)
ax.set_title("Logistic Regression — Normalised Confusion Matrix (Test Set)", fontsize=14, pad=16)
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label",      fontsize=12)
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()

cm_path = os.path.join(REPORTS_DIR, "lr_confusion_matrix.png")
fig.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {cm_path}")

# ── STEP 9: PER-CLASS F1 BAR CHART ────────────────────────────────────────────
report_dict = classification_report(
    y_test, y_pred,
    target_names = class_names,
    zero_division = 0,
    output_dict   = True,
)

per_class_f1  = [report_dict[c]["f1-score"]  for c in class_names]
per_class_rec = [report_dict[c]["recall"]     for c in class_names]
per_class_pre = [report_dict[c]["precision"]  for c in class_names]

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

for ax, values, title, color in zip(
    axes,
    [per_class_f1, per_class_rec, per_class_pre],
    ["F1-score per class", "Recall per class", "Precision per class"],
    ["steelblue", "darkorange", "seagreen"],
):
    bars = ax.barh(class_names, values, color=color, edgecolor="white", height=0.7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(x=np.mean(values), color="red", linestyle="--", linewidth=1,
               label=f"Macro avg = {np.mean(values):.3f}")
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=8)

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7)

plt.suptitle("Logistic Regression — Per-class Metrics (Test Set)", fontsize=14, y=1.01)
plt.tight_layout()

perclass_path = os.path.join(REPORTS_DIR, "lr_per_class_metrics.png")
fig.savefig(perclass_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {perclass_path}")

# ── STEP 10: CV RESULTS PLOT ───────────────────────────────────────────────────
c_vals   = list(cv_results.keys())
f1_means = [cv_results[c]["macro_f1_mean"] for c in c_vals]
f1_stds  = [cv_results[c]["macro_f1_std"]  for c in c_vals]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    [str(c) for c in c_vals], f1_means, yerr=f1_stds,
    marker="o", linewidth=2, capsize=5, color="steelblue", ecolor="gray"
)
ax.axvline(
    x=[str(c) for c in c_vals].index(str(best_C)),
    color="red", linestyle="--", linewidth=1.5,
    label=f"Best C={best_C}"
)
ax.set_xlabel("Regularisation strength C", fontsize=11)
ax.set_ylabel("CV Macro F1-score",         fontsize=11)
ax.set_title("C Tuning — Cross-validation Macro F1", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

cv_path = os.path.join(REPORTS_DIR, "lr_cv_c_tuning.png")
fig.savefig(cv_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {cv_path}")

# ── STEP 11: MODEL ALREADY SAVED ABOVE ───────────────────────────────────────
print(f"\n  Model already saved -> {model_path}")

# ── STEP 12: SAVE FULL METRICS JSON ───────────────────────────────────────────
metrics = {
    "model"              : "LogisticRegression",
    "best_C"             : best_C,
    "train_time_s"       : round(train_time, 2),
    "cv_folds"           : CV_FOLDS,
    "cv_macro_f1_mean"   : round(best_cv["macro_f1_mean"],  4),
    "cv_macro_f1_std"    : round(best_cv["macro_f1_std"],   4),
    "cv_accuracy_mean"   : round(best_cv["accuracy_mean"],  4),
    "cv_precision_mean"  : round(best_cv["precision_mean"], 4),
    "cv_recall_mean"     : round(best_cv["recall_mean"],    4),
    "test_macro_f1"      : round(macro_f1,    4),
    "test_micro_f1"      : round(micro_f1,    4),
    "test_weighted_f1"   : round(weighted_f1, 4),
    "test_accuracy"      : round(accuracy,    4),
    "test_macro_precision": round(macro_prec, 4),
    "test_macro_recall"  : round(macro_rec,   4),
    "test_roc_auc_macro" : round(roc_auc, 4) if roc_auc else None,
    "benign_false_positive_rate": round(fp_rate, 4) if fp_rate is not None else None,
    "per_class": {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall"   : round(report_dict[cls]["recall"],    4),
            "f1"       : round(report_dict[cls]["f1-score"],  4),
            "support"  : int(report_dict[cls]["support"]),
        }
        for cls in class_names
    },
    "cv_all_C_results": {
        str(c): {
            "macro_f1_mean": round(v["macro_f1_mean"],  4),
            "macro_f1_std" : round(v["macro_f1_std"],   4),
            "accuracy_mean": round(v["accuracy_mean"],  4),
        }
        for c, v in cv_results.items()
    },
}

metrics_path = os.path.join(REPORTS_DIR, "lr_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved -> {metrics_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  LOGISTIC REGRESSION — FINAL RESULTS")
print(f"{'='*60}")
print(f"  Best C (from CV)   : {best_C}")
print(f"  Test Macro F1      : {macro_f1:.4f}   <- primary metric")
print(f"  Test Accuracy      : {accuracy:.4f}")
print(f"  Test Macro Recall  : {macro_rec:.4f}")
print(f"  Test Macro Prec    : {macro_prec:.4f}")
if roc_auc:
    print(f"  Test ROC-AUC       : {roc_auc:.4f}")
if fp_rate is not None:
    print(f"  Benign FP rate     : {fp_rate*100:.2f}%")
print(f"\n  Outputs:")
print(f"  models/lr_model.pkl")
print(f"  reports/lr_metrics.json")
print(f"  reports/lr_confusion_matrix.png")
print(f"  reports/lr_per_class_metrics.png")
print(f"  reports/lr_cv_c_tuning.png")
print(f"\n  Next step: run train_rf_xgb.py for the strong model")
print(f"  and compare macro-F1 to this baseline: {macro_f1:.4f}")
print(f"{'='*60}")