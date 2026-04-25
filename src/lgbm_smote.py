"""
lgbm_smote.py  --  SMOTE + LightGBM for IoT Intrusion Detection
================================================================
Root cause of low macro-F1: rare classes (UPLOADING_ATTACK ~1k,
RECON-PINGSWEEP ~1.8k, BACKDOOR_MALWARE ~2.6k) are underrepresented
and get crushed by majority classes during training.

Fix: SMOTE (Synthetic Minority Oversampling Technique) applied ONLY
to classes below SMOTE_THRESHOLD, synthetically up-sampling them to
SMOTE_TARGET samples. Large classes are left untouched.

Then train LightGBM on the rebalanced dataset. The combination of
SMOTE + LightGBM's leaf-wise growth targets the root cause directly.

Note: SMOTE is applied only to TRAINING data. Test set is never touched.
"""

import os, json, time, warnings

# Fix for Windows joblib subprocess error when counting physical cores
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_PATH      = "data/train.csv"
TEST_PATH       = "data/test.csv"
LABEL_COL       = "Label"
MODELS_DIR      = "models"
REPORTS_DIR     = "reports"
RANDOM_STATE    = 42
VAL_FRAC        = 0.20
EARLY_STOP      = 50
MAX_ROUNDS      = 2000

# SMOTE settings: classes with < SMOTE_THRESHOLD samples get upsampled
SMOTE_THRESHOLD = 5000   # classes below this count are minority
SMOTE_TARGET    = 5000   # target count after oversampling
SMOTE_K         = 5      # k-nearest neighbours for synthesis

# LightGBM best config (use best from lgbm_model.py if known, else defaults)
LGBM_NUM_LEAVES    = 127
LGBM_LR            = 0.05
LGBM_MIN_CHILD     = 20
LGBM_FEAT_FRAC     = 0.8
LGBM_BAG_FRAC      = 0.8
LGBM_BAG_FREQ      = 5

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  IoT IDS -- SMOTE + LightGBM Training")
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

# ── SHOW CLASS DISTRIBUTION BEFORE SMOTE ─────────────────────────────────────
print("\nClass distribution before SMOTE:")
unique_all, counts_all = np.unique(y_train_full, return_counts=True)
minority_classes = []
for cls_idx, cnt in zip(unique_all, counts_all):
    flag = "  <- WILL OVERSAMPLE" if cnt < SMOTE_THRESHOLD else ""
    print(f"  {class_names[cls_idx]:<45} {cnt:>7,}{flag}")
    if cnt < SMOTE_THRESHOLD:
        minority_classes.append(cls_idx)

print(f"\n  Minority classes to oversample : {len(minority_classes)}")
print(f"  SMOTE threshold               : {SMOTE_THRESHOLD:,}")
print(f"  SMOTE target per class        : {SMOTE_TARGET:,}")

# ── APPLY SMOTE ───────────────────────────────────────────────────────────────
# Build sampling_strategy: only upsample classes below the threshold.
# Respect that we can't downsample with SMOTE (only oversample).
sampling_strategy = {}
for cls_idx, cnt in zip(unique_all, counts_all):
    if cnt < SMOTE_THRESHOLD:
        # Ensure k_neighbors can be satisfied: need at least k+1 samples
        if cnt > SMOTE_K:
            sampling_strategy[int(cls_idx)] = SMOTE_TARGET
        else:
            print(f"  WARNING: class {class_names[cls_idx]} has {cnt} samples "
                  f"(<= k={SMOTE_K}), skipping SMOTE for this class.")

print(f"\nApplying SMOTE to {len(sampling_strategy)} classes ...")
t0 = time.time()

smote = SMOTE(
    sampling_strategy = sampling_strategy,
    k_neighbors       = SMOTE_K,
    random_state      = RANDOM_STATE,
)
X_resampled, y_resampled = smote.fit_resample(X_train_full, y_train_full)

smote_time = time.time() - t0
print(f"  SMOTE done in {smote_time:.1f}s")
print(f"  Training rows before SMOTE : {len(X_train_full):,}")
print(f"  Training rows after  SMOTE : {len(X_resampled):,}")
print(f"  Synthetic rows added       : {len(X_resampled) - len(X_train_full):,}")

# Show distribution after SMOTE
print("\nClass distribution after SMOTE:")
unique_re, counts_re = np.unique(y_resampled, return_counts=True)
for cls_idx, cnt in zip(unique_re, counts_re):
    orig_cnt = counts_all[list(unique_all).index(cls_idx)]
    delta    = cnt - orig_cnt
    marker   = f"  +{delta:,}" if delta > 0 else ""
    print(f"  {class_names[cls_idx]:<45} {cnt:>7,}{marker}")

# ── SAVE SMOTE STATS FOR COMPARISON ──────────────────────────────────────────
smote_stats = {
    "threshold"         : SMOTE_THRESHOLD,
    "target_per_class"  : SMOTE_TARGET,
    "k_neighbors"       : SMOTE_K,
    "rows_before"       : int(len(X_train_full)),
    "rows_after"        : int(len(X_resampled)),
    "synthetic_added"   : int(len(X_resampled) - len(X_train_full)),
    "classes_oversampled": len(sampling_strategy),
    "smote_time_s"      : round(smote_time, 2),
}

# ── CLASS WEIGHTS (mild, since SMOTE handles most of imbalance) ───────────────
# After SMOTE, rare classes are better represented. Use mild class weights
# (sqrt of inverse frequency rather than full inverse) to avoid over-correcting.
unique_re2, counts_re2 = np.unique(y_resampled, return_counts=True)
freq_re     = counts_re2 / counts_re2.sum()
cw_arr_re   = 1.0 / np.sqrt(freq_re * n_classes + 1e-8)  # mild: sqrt instead of full inverse

def make_sample_weights(y, cw_arr):
    return cw_arr[y].astype(np.float32)

# ── TRAIN / VAL SPLIT (on resampled data) ─────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_resampled, y_resampled,
    test_size    = VAL_FRAC,
    stratify     = y_resampled,
    random_state = RANDOM_STATE,
)
w_tr  = make_sample_weights(y_tr,  cw_arr_re)
w_val = make_sample_weights(y_val, cw_arr_re)

print(f"\n  Train subset (post-SMOTE): {len(X_tr):,}  |  Val: {len(X_val):,}")

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

# ── TRAIN LGBM ON SMOTE-REBALANCED DATA ───────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Training LightGBM on SMOTE-balanced data")
print(f"  num_leaves={LGBM_NUM_LEAVES}  lr={LGBM_LR}")
print(f"  Early stop: {EARLY_STOP} rounds  |  Max rounds: {MAX_ROUNDS}")
print(f"  Device: {DEVICE}")
print(f"{'='*62}\n")

dtrain = lgb.Dataset(X_tr,  label=y_tr,  weight=w_tr,  feature_name=feature_names, free_raw_data=False)
dval   = lgb.Dataset(X_val, label=y_val, weight=w_val, feature_name=feature_names, reference=dtrain, free_raw_data=False)

params = {
    "objective"        : "multiclass",
    "num_class"        : n_classes,
    "metric"           : "multi_logloss",
    "device"           : DEVICE,
    "num_leaves"       : LGBM_NUM_LEAVES,
    "learning_rate"    : LGBM_LR,
    "min_child_samples": LGBM_MIN_CHILD,
    "feature_fraction" : LGBM_FEAT_FRAC,
    "bagging_fraction" : LGBM_BAG_FRAC,
    "bagging_freq"     : LGBM_BAG_FREQ,
    "max_depth"        : -1,
    "min_split_gain"   : 0.0,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "n_jobs"           : -1,
    "seed"             : RANDOM_STATE,
    "verbosity"        : -1,
}

t0 = time.time()
callbacks = [
    lgb.early_stopping(EARLY_STOP, verbose=False),
    lgb.log_evaluation(period=-1),
]
val_model = lgb.train(
    params, dtrain,
    num_boost_round = MAX_ROUNDS,
    valid_sets      = [dval],
    callbacks       = callbacks,
)
best_rounds = val_model.best_iteration
val_elapsed = time.time() - t0

y_val_pred = val_model.predict(X_val).argmax(axis=1)
val_f1  = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"  Val macro-F1={val_f1:.4f}  acc={val_acc:.4f}  rounds={best_rounds}  [{val_elapsed:.0f}s]")

# ── RETRAIN ON FULL SMOTE-RESAMPLED DATA ─────────────────────────────────────
print(f"\nRetraining on full resampled training set ({best_rounds} rounds) ...")
w_full = make_sample_weights(y_resampled, cw_arr_re)
dfull  = lgb.Dataset(
    X_resampled, label=y_resampled,
    weight=w_full, feature_name=feature_names,
)

t0 = time.time()
final_model = lgb.train(params, dfull, num_boost_round=best_rounds)
train_time  = time.time() - t0
print(f"  Done in {train_time:.1f}s")

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

# Per-class report — highlight minority class improvements
print(f"\n  Per-class report:")
report_str  = classification_report(y_test, y_pred, target_names=class_names, zero_division=0, digits=4)
print(report_str)
report_dict = classification_report(y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)

# Highlight rare-class F1 improvements
print("\n  Minority class F1 scores (classes that were SMOTE'd):")
for cls_idx in sampling_strategy:
    cls_name = class_names[cls_idx]
    f1_val   = report_dict[cls_name]["f1-score"]
    print(f"  {cls_name:<45} F1={f1_val:.4f}")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("Generating plots ...")

cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(20, 17))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.3, linecolor="gray", annot_kws={"size": 7})
ax.set_title(f"SMOTE+LightGBM -- Normalised Confusion Matrix\n"
             f"Macro F1={macro_f1:.4f}  Acc={accuracy:.4f}", fontsize=13)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True",      fontsize=11)
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "lgbm_smote_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/lgbm_smote_confusion_matrix.png")

per_class_f1  = [report_dict[c]["f1-score"]  for c in class_names]
per_class_rec = [report_dict[c]["recall"]     for c in class_names]
per_class_pre = [report_dict[c]["precision"]  for c in class_names]

# Colour bars red for minority classes
colors_bar = []
for c in class_names:
    cls_idx = le.transform([c])[0]
    colors_bar.append("#e74c3c" if cls_idx in sampling_strategy else "steelblue")

fig, axes = plt.subplots(3, 1, figsize=(16, 16))
for ax, values, title, default_color in zip(
    axes,
    [per_class_f1, per_class_rec, per_class_pre],
    ["F1-score per class", "Recall per class", "Precision per class"],
    ["steelblue", "darkorange", "seagreen"],
):
    bar_colors = ["#e74c3c" if c in [class_names[i] for i in sampling_strategy] else default_color
                  for c in class_names]
    bars = ax.barh(class_names, values, color=bar_colors, edgecolor="white", height=0.7)
    ax.set_xlim(0, 1.08)
    ax.axvline(np.mean(values), color="black", linestyle="--", linewidth=1.2,
               label=f"Macro avg = {np.mean(values):.3f}")
    ax.set_title(f"{title} (red = SMOTE'd minority classes)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)
plt.suptitle("SMOTE+LightGBM -- Per-class Metrics (Test Set)", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "lgbm_smote_per_class_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/lgbm_smote_per_class_metrics.png")

# Class distribution before/after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, counts_arr, title in zip(
    axes,
    [counts_all, counts_re2],
    ["Before SMOTE", f"After SMOTE (target={SMOTE_TARGET:,})"],
):
    names_sorted = [class_names[i] for i in np.argsort(counts_arr)[::-1]]
    vals_sorted  = sorted(counts_arr, reverse=True)
    bar_colors   = ["#e74c3c" if n in [class_names[i] for i in sampling_strategy]
                    else "steelblue" for n in names_sorted]
    ax.barh(names_sorted[::-1], vals_sorted[::-1], color=bar_colors[::-1], edgecolor="white")
    ax.set_xlabel("Sample count")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(SMOTE_THRESHOLD, color="orange", linestyle="--", linewidth=1.5,
               label=f"SMOTE threshold={SMOTE_THRESHOLD:,}")
    ax.legend()
plt.suptitle("SMOTE+LightGBM -- Class Distribution (red = minority, oversampled)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "lgbm_smote_class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/lgbm_smote_class_distribution.png")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "lgbm_smote_model.pkl")
joblib.dump(final_model, model_path)
print(f"  Model saved -> {model_path}")

# ── SAVE METRICS ──────────────────────────────────────────────────────────────
LR_MACRO_F1 = 0.4401; RF_MACRO_F1 = 0.5861; XGB_MACRO_F1 = 0.6200

metrics = {
    "model"               : "LightGBM_SMOTE",
    "device"              : DEVICE,
    "smote_stats"         : smote_stats,
    "lgbm_params"         : {
        "num_leaves": LGBM_NUM_LEAVES, "learning_rate": LGBM_LR,
        "n_rounds": best_rounds, "device": DEVICE,
    },
    "train_time_s"        : round(train_time, 2),
    "smote_time_s"        : round(smote_time, 2),
    "val_macro_f1"        : round(val_f1,    4),
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
    "minority_class_f1": {
        class_names[i]: round(report_dict[class_names[i]]["f1-score"], 4)
        for i in sampling_strategy
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

metrics_path = os.path.join(REPORTS_DIR, "lgbm_smote_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved -> {metrics_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  SMOTE + LIGHTGBM -- FINAL RESULTS")
print(f"{'='*62}")
print(f"  SMOTE: {len(sampling_strategy)} classes oversampled to {SMOTE_TARGET:,} samples")
print(f"  Synthetic rows added : {len(X_resampled) - len(X_train_full):,}")
print(f"  Test Macro F1    : {macro_f1:.4f}   <- primary metric")
print(f"  Test Accuracy    : {accuracy:.4f}")
print(f"  Test ROC-AUC     : {roc_auc:.4f}" if roc_auc else "  Test ROC-AUC     : N/A")
print(f"  Benign FP rate   : {fp_rate*100:.2f}%" if fp_rate is not None else "")
print(f"\n  vs LR  : Macro F1 {LR_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-LR_MACRO_F1:+.4f})")
print(f"  vs RF  : Macro F1 {RF_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-RF_MACRO_F1:+.4f})")
print(f"  vs XGB : Macro F1 {XGB_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-XGB_MACRO_F1:+.4f})")
print(f"\n  Minority class F1 scores (SMOTE targets):")
for cls_idx in sampling_strategy:
    cls_name = class_names[cls_idx]
    print(f"    {cls_name:<43} F1={report_dict[cls_name]['f1-score']:.4f}")
print(f"\n  Outputs:")
print(f"    models/lgbm_smote_model.pkl")
print(f"    reports/lgbm_smote_metrics.json")
print(f"    reports/lgbm_smote_confusion_matrix.png")
print(f"    reports/lgbm_smote_per_class_metrics.png")
print(f"    reports/lgbm_smote_class_distribution.png")
print(f"{'='*62}")
