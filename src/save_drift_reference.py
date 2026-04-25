"""
save_drift_reference.py
=======================
Computes reference statistics from the training set and saves them to
models/drift_reference.json.  Run once after preprocessing.

Saves:
  - per-feature mean and std (after full preprocessing pipeline)
  - training class distribution (fraction of each class)
  - binary ATTACK fraction
"""

import os, json
import numpy as np
import pandas as pd
import joblib

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
SAMPLE_N   = 100_000   # rows to sample for stable stats (full set = 2.9M, overkill)

print("Loading preprocessing artifacts...")
label_encoder     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
numeric_imputer   = joblib.load(os.path.join(MODELS_DIR, "numeric_imputer.pkl"))
numeric_scaler    = joblib.load(os.path.join(MODELS_DIR, "numeric_scaler.pkl"))
variance_selector = joblib.load(os.path.join(MODELS_DIR, "variance_selector.pkl"))

with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
    feature_info = json.load(f)
with open(os.path.join(MODELS_DIR, "log_transformed_cols.json")) as f:
    log_cols = json.load(f)

NUMERIC_COLS      = feature_info["numeric_cols"]
SELECTED_FEATURES = feature_info["selected_features"]
CLASS_NAMES       = label_encoder.classes_.tolist()
LABEL_COL         = "Label"

print(f"Loading {SAMPLE_N:,} rows from train.csv...")
df = pd.read_csv(TRAIN_PATH).sample(n=SAMPLE_N, random_state=42)

y = df[LABEL_COL].values

# Only keep columns that actually exist in this CSV
available_cols = [c for c in NUMERIC_COLS if c in df.columns]
X_raw = df[available_cols].copy()

# Pad any missing columns with 0
for col in NUMERIC_COLS:
    if col not in X_raw.columns:
        X_raw[col] = 0.0
X_raw = X_raw[NUMERIC_COLS]   # restore expected order

# Apply log transforms
for col in log_cols:
    if col in X_raw.columns:
        X_raw[col] = np.log1p(X_raw[col].clip(lower=0))

X_raw.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
X_imp   = numeric_imputer.transform(X_raw)
X_scl   = numeric_scaler.transform(X_imp)
X_final = variance_selector.transform(X_scl).astype(np.float32)

print(f"Processed shape: {X_final.shape}")

# Feature stats
feat_mean = X_final.mean(axis=0).tolist()
feat_std  = X_final.std(axis=0).tolist()
feat_std  = [max(s, 1e-6) for s in feat_std]   # avoid div-by-zero

# Class distribution
class_counts = {}
for i, name in enumerate(CLASS_NAMES):
    class_counts[name] = float((y == i).sum() / len(y))

# Binary ATTACK fraction
benign_idx      = CLASS_NAMES.index("BENIGN")
attack_fraction = float((y != benign_idx).sum() / len(y))

reference = {
    "feature_names" : SELECTED_FEATURES,
    "feature_mean"  : feat_mean,
    "feature_std"   : feat_std,
    "class_distribution": class_counts,
    "attack_fraction": attack_fraction,
    "sample_size"   : SAMPLE_N,
}

out_path = os.path.join(MODELS_DIR, "drift_reference.json")
with open(out_path, "w") as f:
    json.dump(reference, f, indent=2)

print(f"Saved -> {out_path}")
print(f"  Features : {len(SELECTED_FEATURES)}")
print(f"  Classes  : {len(CLASS_NAMES)}")
print(f"  Attack % : {attack_fraction*100:.1f}%")
