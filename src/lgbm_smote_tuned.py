"""
lgbm_smote_tuned.py  --  Hyperparameter-tuned SMOTE + LightGBM
===============================================================
Runs Optuna Bayesian optimization over LightGBM hyperparameters
on top of the SMOTE-balanced dataset to push beyond F1=0.647.

Saves the best model as:
  models/lgbm_smote_tuned_model.pkl
  reports/lgbm_smote_tuned_metrics.json
"""

import os, json, time, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_score, recall_score,
)

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_PATH      = "data/train.csv"
TEST_PATH       = "data/test.csv"
LABEL_COL       = "Label"
MODELS_DIR      = "models"
REPORTS_DIR     = "reports"
RANDOM_STATE    = 42
VAL_FRAC        = 0.15
N_TRIALS        = 8        # Optuna trials
EARLY_STOP      = 30
MAX_ROUNDS      = 500

SMOTE_THRESHOLD = 5000
SMOTE_TARGET    = 5000
SMOTE_K         = 5

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 62)
print("  IoT IDS -- Tuned SMOTE + LightGBM (Optuna)")
print("=" * 62)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
for path in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_full  = train_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_full  = train_df[LABEL_COL].values
X_test  = test_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_test  = test_df[LABEL_COL].values
feature_names = train_df.drop(columns=[LABEL_COL]).columns.tolist()

le          = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
class_names = le.classes_.tolist()
n_classes   = len(class_names)

print(f"\n  Train : {len(X_full):,} rows x {X_full.shape[1]} features")
print(f"  Test  : {len(X_test):,} rows  x {X_full.shape[1]} features")
print(f"  Classes: {n_classes}")

# ── DETECT GPU ────────────────────────────────────────────────────────────────
def detect_lgbm_device():
    try:
        tp = {"objective": "multiclass", "num_class": 2, "device": "gpu",
              "num_leaves": 7, "verbosity": -1}
        lgb.train(tp, lgb.Dataset(np.random.rand(200,4).astype(np.float32),
                  label=np.random.randint(0,2,200)), num_boost_round=2)
        return "gpu"
    except Exception:
        return "cpu"

LGBM_DEVICE = detect_lgbm_device()
print(f"  LightGBM device: {LGBM_DEVICE}")

# ── TRAIN/VAL SPLIT ───────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=VAL_FRAC, stratify=y_full, random_state=RANDOM_STATE
)

# ── SMOTE ─────────────────────────────────────────────────────────────────────
print(f"\n  Applying SMOTE (threshold={SMOTE_THRESHOLD}, target={SMOTE_TARGET}) ...")
t0 = time.time()

unique, counts = np.unique(y_tr, return_counts=True)
sampling = {cls: SMOTE_TARGET for cls, cnt in zip(unique, counts) if cnt < SMOTE_THRESHOLD}
print(f"  Classes to oversample: {len(sampling)}")

if sampling:
    smote = SMOTE(sampling_strategy=sampling, k_neighbors=SMOTE_K,
                  random_state=RANDOM_STATE)
    X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)
else:
    X_tr_s, y_tr_s = X_tr, y_tr

print(f"  After SMOTE: {len(X_tr_s):,} rows  [{time.time()-t0:.0f}s]")

dtrain = lgb.Dataset(X_tr_s, label=y_tr_s, feature_name=feature_names, free_raw_data=False)
dval   = lgb.Dataset(X_val,  label=y_val,  feature_name=feature_names,
                     reference=dtrain, free_raw_data=False)

# ── OPTUNA OBJECTIVE ──────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "objective"         : "multiclass",
        "num_class"         : n_classes,
        "metric"            : "multi_logloss",
        "device"            : "cpu",   # GPU causes split errors on multiclass
        "verbosity"         : -1,
        "seed"              : RANDOM_STATE,
        "n_jobs"            : 4,
        "num_leaves"        : trial.suggest_int("num_leaves", 63, 255),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_child_samples" : trial.suggest_int("min_child_samples", 10, 50),
        "feature_fraction"  : trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction"  : trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq"      : trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "max_depth"         : trial.suggest_int("max_depth", -1, 12),
    }
    cb = [lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(period=-1)]
    model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                      valid_sets=[dval], callbacks=cb)
    preds = model.predict(X_val).argmax(axis=1)
    return f1_score(y_val, preds, average="macro", zero_division=0)

# ── RUN OPTUNA ────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Running Optuna ({N_TRIALS} trials) ...")
print(f"{'='*62}")

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,))

best_params = study.best_params
best_val_f1 = study.best_value
print(f"\n  Best val macro-F1 : {best_val_f1:.4f}")
print(f"  Best params:")
for k, v in best_params.items():
    print(f"    {k:<22}: {v}")

# ── RETRAIN ON FULL SMOTE DATA WITH BEST PARAMS ───────────────────────────────
print(f"\n{'='*62}")
print("  Retraining on full SMOTE data with best params ...")
print(f"{'='*62}")

# Apply SMOTE to full training set
if sampling:
    X_full_s, y_full_s = SMOTE(
        sampling_strategy=sampling, k_neighbors=SMOTE_K,
        random_state=RANDOM_STATE
    ).fit_resample(X_full, y_full)
else:
    X_full_s, y_full_s = X_full, y_full

# Val split from full SMOTE data for early stopping
X_ft, X_fv, y_ft, y_fv = train_test_split(
    X_full_s, y_full_s, test_size=0.10, stratify=y_full_s, random_state=RANDOM_STATE
)
dft = lgb.Dataset(X_ft, label=y_ft, feature_name=feature_names, free_raw_data=False)
dfv = lgb.Dataset(X_fv, label=y_fv, feature_name=feature_names, reference=dft, free_raw_data=False)

final_params = {
    "objective"    : "multiclass",
    "num_class"    : n_classes,
    "metric"       : "multi_logloss",
    "device"       : "cpu",
    "verbosity"    : -1,
    "seed"         : RANDOM_STATE,
    "n_jobs"       : 4,
    **best_params,
}
cb = [lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(period=50)]
t0 = time.time()
final_model = lgb.train(final_params, dft, num_boost_round=MAX_ROUNDS,
                        valid_sets=[dfv], callbacks=cb)
print(f"  Done in {time.time()-t0:.0f}s  |  Best iteration: {final_model.best_iteration}")

# ── EVALUATE ──────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  Final evaluation on locked test set")
print(f"{'='*62}")

y_pred_prob = final_model.predict(X_test)
y_pred      = y_pred_prob.argmax(axis=1)

macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
micro_f1    = f1_score(y_test, y_pred, average="micro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
accuracy    = accuracy_score(y_test, y_pred)
precision   = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall      = recall_score(y_test, y_pred, average="macro",    zero_division=0)

print(f"\n  Macro F1    : {macro_f1:.4f}   <- primary metric")
print(f"  Micro F1    : {micro_f1:.4f}")
print(f"  Weighted F1 : {weighted_f1:.4f}")
print(f"  Accuracy    : {accuracy:.4f}")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")

OLD_F1 = 0.647
print(f"\n  vs baseline SMOTE+LGBM: {OLD_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-OLD_F1:+.4f})")

print(f"\n  Per-class report:")
print(classification_report(y_test, y_pred, target_names=class_names,
                             zero_division=0, digits=4))

# ── FEATURE IMPORTANCE PLOT ───────────────────────────────────────────────────
fi = pd.Series(final_model.feature_importance(importance_type="gain"),
               index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 8))
fi.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_title(f"Tuned LGBM+SMOTE — Feature Importance (Gain)\nMacro F1={macro_f1:.4f}", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "lgbm_smote_tuned_feature_importance.png"), dpi=150)
plt.close(fig)

# ── OPTUNA HISTORY PLOT ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
trials_f1 = [t.value for t in study.trials]
ax.plot(trials_f1, marker="o", markersize=4, linewidth=1.2, color="steelblue")
ax.axhline(OLD_F1, color="gray", linestyle="--", linewidth=1, label=f"Baseline F1={OLD_F1}")
ax.axhline(best_val_f1, color="red", linestyle="--", linewidth=1,
           label=f"Best val F1={best_val_f1:.4f}")
ax.set_xlabel("Trial"); ax.set_ylabel("Val Macro F1")
ax.set_title("Optuna Hyperparameter Search — LightGBM+SMOTE")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "lgbm_smote_tuned_optuna_history.png"), dpi=150)
plt.close(fig)
print("  Plots saved.")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "lgbm_smote_tuned_model.pkl")
joblib.dump(final_model, model_path)
print(f"\n  Model saved -> {model_path}")

# ── SAVE METRICS ─────────────────────────────────────────────────────────────
report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                     zero_division=0, output_dict=True)
metrics = {
    "model"           : "LightGBM+SMOTE (Optuna tuned)",
    "n_trials"        : N_TRIALS,
    "best_val_f1"     : round(best_val_f1,  4),
    "best_params"     : best_params,
    "best_iteration"  : final_model.best_iteration,
    "test_macro_f1"   : round(macro_f1,    4),
    "test_micro_f1"   : round(micro_f1,    4),
    "test_weighted_f1": round(weighted_f1, 4),
    "test_accuracy"   : round(accuracy,    4),
    "test_precision"  : round(precision,   4),
    "test_recall"     : round(recall,      4),
    "delta_vs_baseline": round(macro_f1 - OLD_F1, 4),
    "per_class": {
        cls: {
            "f1":        round(report_dict[cls]["f1-score"],  4),
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "support":   int(report_dict[cls]["support"]),
        } for cls in class_names
    },
}
with open(os.path.join(REPORTS_DIR, "lgbm_smote_tuned_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'='*62}")
print("  TUNED LGBM+SMOTE — FINAL RESULTS")
print(f"{'='*62}")
print(f"  Optuna trials  : {N_TRIALS}")
print(f"  Best val F1    : {best_val_f1:.4f}")
print(f"  Test Macro F1  : {macro_f1:.4f}  (baseline: {OLD_F1:.4f}, delta: {macro_f1-OLD_F1:+.4f})")
print(f"  Test Accuracy  : {accuracy:.4f}")
print(f"  Best params    : num_leaves={best_params.get('num_leaves')}  "
      f"lr={best_params.get('learning_rate'):.4f}  "
      f"min_child={best_params.get('min_child_samples')}")
print(f"\n  Outputs:")
print(f"    models/lgbm_smote_tuned_model.pkl")
print(f"    reports/lgbm_smote_tuned_metrics.json")
print(f"    reports/lgbm_smote_tuned_optuna_history.png")
print(f"    reports/lgbm_smote_tuned_feature_importance.png")
print(f"{'='*62}")
