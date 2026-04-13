import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

# ── CONFIG ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/Merged01.csv"
LABEL_COL   = "Label"

TEST_SIZE    = 0.15
RANDOM_STATE = 42

# Missing value strategy:
# - "impute": keep rows and impute feature NaNs with median
# - "drop":   drop rows containing NaNs
MISSING_STRATEGY = "impute"

# Low-variance removal:
# 0.0  → removes only perfectly constant columns (too lenient)
# 0.01 → removes near-zero variance columns (recommended)
# None → disables the step entirely
VARIANCE_THRESHOLD = 0.01          # ← FIX 1: was 0.0

# Log-transform: features with |skew| above this threshold get log1p applied
# Set to None to disable
SKEW_THRESHOLD = 10

# Whether to stratify the train/test split on the label
STRATIFY_SPLIT = True

# If True, scale numeric features with RobustScaler after imputation
SCALE_NUMERIC = True

# Columns to force-treat as categorical even if pandas sees them as numeric
FORCE_CATEGORICAL_COLS = []

# Optional ordinal columns with explicit rank order
# Example: ORDINAL_COLUMNS = {"education": ["hs", "bachelor", "master", "phd"]}
ORDINAL_COLUMNS = {}

# Always drop rows whose label is missing
DROP_ROWS_WITH_MISSING_LABEL = True

# ── FOLDERS ────────────────────────────────────────────────────────────────────
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── STEP 1: LOAD & BASIC VALIDATION ───────────────────────────────────────────
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
print(f"Loaded: {df.shape}")

if df.empty:
    raise ValueError("Input dataset is empty.")

if LABEL_COL not in df.columns:
    raise ValueError(f"Missing label column '{LABEL_COL}' in dataset.")

if df.columns.duplicated().any():
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    raise ValueError(f"Duplicate column names found: {dup_cols}")

# ── STEP 2: CLEANING ──────────────────────────────────────────────────────────
df_clean    = df.copy()
rows_before = len(df_clean)

# Replace infinities with NaN
numeric_cols_all  = df_clean.select_dtypes(include=[np.number]).columns.tolist()
inf_count_before  = int(np.isinf(df_clean[numeric_cols_all]).sum().sum()) if numeric_cols_all else 0
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove duplicate rows
dupes_before = int(df_clean.duplicated().sum())
df_clean.drop_duplicates(inplace=True)

# Report missing values after inf replacement
missing_before       = int(df_clean.isna().sum().sum())
label_missing_before = int(df_clean[LABEL_COL].isna().sum())

# Drop rows with missing labels
if DROP_ROWS_WITH_MISSING_LABEL:
    df_clean = df_clean[df_clean[LABEL_COL].notna()].copy()

# Drop rows with any remaining NaN if strategy is "drop"
if MISSING_STRATEGY == "drop":
    df_clean.dropna(inplace=True)
elif MISSING_STRATEGY != "impute":
    raise ValueError("MISSING_STRATEGY must be 'impute' or 'drop'.")

rows_after_clean = len(df_clean)

print("\nCleaning summary:")
print(f"  Infinite values found (before replacement): {inf_count_before:,}")
print(f"  Missing values found (before handling):      {missing_before:,}")
print(f"  Missing labels dropped:                       {label_missing_before:,}")
print(f"  Duplicate rows removed:                       {dupes_before:,}")
print(f"  Rows before cleaning:                         {rows_before:,}")
print(f"  Rows after  cleaning:                         {rows_after_clean:,}")
print(f"  Rows removed total:                           {rows_before - rows_after_clean:,}")

if rows_after_clean == 0:
    raise ValueError("No rows left after cleaning.")

# ── STEP 3: SPLIT FEATURES / LABEL ────────────────────────────────────────────
X = df_clean.drop(columns=[LABEL_COL]).copy()
y = df_clean[LABEL_COL].copy()

if X.shape[1] == 0:
    raise ValueError("No feature columns found after removing label column.")

# ── STEP 4: DETECT COLUMN TYPES ───────────────────────────────────────────────
for col in FORCE_CATEGORICAL_COLS:
    if col in X.columns:
        X[col] = X[col].astype("object")

ordinal_cols              = [c for c in ORDINAL_COLUMNS.keys() if c in X.columns]
all_categorical_candidates = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
nominal_categorical_cols  = [c for c in all_categorical_candidates if c not in ordinal_cols]
numeric_cols              = [c for c in X.columns if c not in nominal_categorical_cols + ordinal_cols]

print("\nDetected feature groups:")
print(f"  Numeric columns              : {len(numeric_cols)}")
print(f"  Nominal categorical columns  : {len(nominal_categorical_cols)}")
print(f"  Ordinal categorical columns  : {len(ordinal_cols)}")

if not numeric_cols and not nominal_categorical_cols and not ordinal_cols:
    raise ValueError("No usable feature columns detected.")

# ── STEP 5: LABEL ENCODING ────────────────────────────────────────────────────
le        = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nLabel encoding map:")
for number, name in enumerate(le.classes_):
    print(f"  {number:2d} = {name}")

# Capture class distribution for metadata (FIX 3)
class_counts = pd.Series(y).value_counts().to_dict()

# ── STEP 6: TRAIN / TEST SPLIT ────────────────────────────────────────────────
use_stratify = STRATIFY_SPLIT and len(np.unique(y_encoded)) > 1
stratify_arg = y_encoded if use_stratify else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = stratify_arg
)

print("\nSplit sizes:")
print(f"  Train : {len(X_train):,} rows ({len(X_train)/rows_after_clean*100:.0f}%)")
print(f"  Test  : {len(X_test):,} rows ({len(X_test)/rows_after_clean*100:.0f}%)")

# ── STEP 7: LOG-TRANSFORM SKEWED FEATURES (FIX 2) ─────────────────────────────
# Skewness is computed on train only — same columns applied to test (no leakage).
log_cols = []

if SKEW_THRESHOLD is not None and numeric_cols:
    skew_vals = X_train[numeric_cols].skew().abs()
    log_cols  = [
        c for c in skew_vals[skew_vals > SKEW_THRESHOLD].index
        if X_train[c].min() >= 0      # log1p is only valid for non-negative values
    ]
    if log_cols:
        X_train[log_cols] = np.log1p(X_train[log_cols])
        X_test[log_cols]  = np.log1p(X_test[log_cols])
        print(f"\nLog-transformed {len(log_cols)} skewed features:")
        for c in log_cols:
            print(f"  {c}")
    else:
        print("\nNo features exceeded the skew threshold — no log-transform applied.")

# Save so api.py can replicate this transform at inference time
with open("models/log_transformed_cols.json", "w", encoding="utf-8") as f:
    json.dump(log_cols, f, indent=2)

# ── STEP 8: PREPROCESS EACH COLUMN GROUP (TRAIN-ONLY FIT) ─────────────────────
fitted_objects = {}

# ── 8a. Numeric ────────────────────────────────────────────────────────────────
if numeric_cols:
    X_train_num = X_train[numeric_cols].copy()
    X_test_num  = X_test[numeric_cols].copy()

    if MISSING_STRATEGY == "impute":
        num_imputer = SimpleImputer(strategy="median")
        X_train_num = pd.DataFrame(
            num_imputer.fit_transform(X_train_num),
            columns=numeric_cols, index=X_train_num.index
        )
        X_test_num = pd.DataFrame(
            num_imputer.transform(X_test_num),
            columns=numeric_cols, index=X_test_num.index
        )
        fitted_objects["numeric_imputer"] = num_imputer
        print("\nNumeric imputation done (median, fitted on train only).")

    if SCALE_NUMERIC:
        num_scaler = RobustScaler()
        X_train_num = pd.DataFrame(
            num_scaler.fit_transform(X_train_num),
            columns=numeric_cols, index=X_train_num.index
        )
        X_test_num = pd.DataFrame(
            num_scaler.transform(X_test_num),
            columns=numeric_cols, index=X_test_num.index
        )
        fitted_objects["numeric_scaler"] = num_scaler
        print("Numeric scaling done (RobustScaler fitted on train only).")
else:
    X_train_num = pd.DataFrame(index=X_train.index)
    X_test_num  = pd.DataFrame(index=X_test.index)

# ── 8b. Nominal categorical ────────────────────────────────────────────────────
cat_feature_names = []
if nominal_categorical_cols:
    X_train_cat = X_train[nominal_categorical_cols].copy()
    X_test_cat  = X_test[nominal_categorical_cols].copy()

    if MISSING_STRATEGY == "impute":
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat = pd.DataFrame(
            cat_imputer.fit_transform(X_train_cat),
            columns=nominal_categorical_cols, index=X_train_cat.index
        )
        X_test_cat = pd.DataFrame(
            cat_imputer.transform(X_test_cat),
            columns=nominal_categorical_cols, index=X_test_cat.index
        )
        fitted_objects["categorical_imputer"] = cat_imputer
        print("Nominal categorical imputation done (most_frequent, train only).")

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_enc   = onehot.fit_transform(X_train_cat)
    X_test_cat_enc    = onehot.transform(X_test_cat)
    cat_feature_names = onehot.get_feature_names_out(nominal_categorical_cols).tolist()

    X_train_cat = pd.DataFrame(X_train_cat_enc, columns=cat_feature_names, index=X_train.index)
    X_test_cat  = pd.DataFrame(X_test_cat_enc,  columns=cat_feature_names, index=X_test.index)
    fitted_objects["onehot_encoder"] = onehot
    print("Nominal categorical encoding done (OneHotEncoder, train only).")
else:
    X_train_cat = pd.DataFrame(index=X_train.index)
    X_test_cat  = pd.DataFrame(index=X_test.index)

# ── 8c. Ordinal categorical ────────────────────────────────────────────────────
ordinal_feature_names = []
if ordinal_cols:
    X_train_ord = X_train[ordinal_cols].copy()
    X_test_ord  = X_test[ordinal_cols].copy()

    if MISSING_STRATEGY == "impute":
        ordinal_imputer = SimpleImputer(strategy="most_frequent")
        X_train_ord = pd.DataFrame(
            ordinal_imputer.fit_transform(X_train_ord),
            columns=ordinal_cols, index=X_train_ord.index
        )
        X_test_ord = pd.DataFrame(
            ordinal_imputer.transform(X_test_ord),
            columns=ordinal_cols, index=X_test_ord.index
        )
        fitted_objects["ordinal_imputer"] = ordinal_imputer

    ordinal_encoder = OrdinalEncoder(
        categories=[ORDINAL_COLUMNS[col] for col in ordinal_cols],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    X_train_ord_enc       = ordinal_encoder.fit_transform(X_train_ord)
    X_test_ord_enc        = ordinal_encoder.transform(X_test_ord)
    ordinal_feature_names = ordinal_cols.copy()

    X_train_ord = pd.DataFrame(X_train_ord_enc, columns=ordinal_feature_names, index=X_train.index)
    X_test_ord  = pd.DataFrame(X_test_ord_enc,  columns=ordinal_feature_names, index=X_test.index)
    fitted_objects["ordinal_encoder"] = ordinal_encoder
    print("Ordinal categorical encoding done (OrdinalEncoder, train only).")
else:
    X_train_ord = pd.DataFrame(index=X_train.index)
    X_test_ord  = pd.DataFrame(index=X_test.index)

# ── STEP 9: MERGE ALL PREPROCESSED FEATURES ───────────────────────────────────
X_train_processed = pd.concat([X_train_num, X_train_ord, X_train_cat], axis=1)
X_test_processed  = pd.concat([X_test_num,  X_test_ord,  X_test_cat],  axis=1)

features_before_selection = X_train_processed.columns.tolist()
print(f"\nFeature count before variance filter: {len(features_before_selection)}")

if X_train_processed.shape[1] == 0:
    raise ValueError("No features left after preprocessing.")

# ── STEP 10: VARIANCE FILTER (FIX 1: threshold = 0.01) ───────────────────────
if VARIANCE_THRESHOLD is not None:
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)

    X_train_selected = selector.fit_transform(X_train_processed)
    X_test_selected  = selector.transform(X_test_processed)

    selected_mask     = selector.get_support()
    selected_features = [f for f, keep in zip(features_before_selection, selected_mask) if keep]
    removed_features  = [f for f, keep in zip(features_before_selection, selected_mask) if not keep]

    X_train_out = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_processed.index)
    X_test_out  = pd.DataFrame(X_test_selected,  columns=selected_features, index=X_test_processed.index)

    fitted_objects["variance_selector"] = selector

    print(f"VarianceThreshold={VARIANCE_THRESHOLD} applied.")
    print(f"  Features kept   : {len(selected_features)}")
    print(f"  Features removed: {len(removed_features)}")
    if removed_features:
        print(f"  Removed columns : {removed_features}")
else:
    selector          = None
    selected_features = features_before_selection
    X_train_out       = X_train_processed.copy()
    X_test_out        = X_test_processed.copy()

# ── STEP 11: SAVE TRAIN / TEST CSV ────────────────────────────────────────────
X_train_out["Label"] = y_train
X_test_out["Label"]  = y_test

X_train_out.to_csv("data/train.csv", index=False)
X_test_out.to_csv("data/test.csv",   index=False)

# ── STEP 12: SAVE ALL ARTIFACTS ───────────────────────────────────────────────
joblib.dump(le, "models/label_encoder.pkl")

artifact_map = {
    "numeric_imputer"    : "models/numeric_imputer.pkl",
    "numeric_scaler"     : "models/numeric_scaler.pkl",
    "categorical_imputer": "models/categorical_imputer.pkl",
    "onehot_encoder"     : "models/onehot_encoder.pkl",
    "ordinal_imputer"    : "models/ordinal_imputer.pkl",
    "ordinal_encoder"    : "models/ordinal_encoder.pkl",
    "variance_selector"  : "models/variance_selector.pkl",
}
for key, path in artifact_map.items():
    if key in fitted_objects:
        joblib.dump(fitted_objects[key], path)

# Feature group metadata
feature_info = {
    "numeric_cols"             : numeric_cols,
    "log_transformed_cols"     : log_cols,
    "nominal_categorical_cols" : nominal_categorical_cols,
    "ordinal_cols"             : ordinal_cols,
    "onehot_feature_names"     : cat_feature_names,
    "ordinal_feature_names"    : ordinal_feature_names,
    "selected_features"        : selected_features,
}
with open("models/feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(feature_info, f, indent=2)

# Full reproducibility metadata (FIX 3: includes class distribution + imbalance info)
imbalanced_count = sum(1 for v in class_counts.values() if v < 100)
metadata = {
    "input_path"                       : INPUT_PATH,
    "label_col"                        : LABEL_COL,
    "test_size"                        : TEST_SIZE,
    "random_state"                     : RANDOM_STATE,
    "missing_strategy"                 : MISSING_STRATEGY,
    "variance_threshold"               : VARIANCE_THRESHOLD,
    "skew_threshold"                   : SKEW_THRESHOLD,
    "scale_numeric"                    : SCALE_NUMERIC,
    "stratify_split"                   : use_stratify,
    "rows_before_cleaning"             : int(rows_before),
    "rows_after_cleaning"              : int(rows_after_clean),
    "num_numeric_features"             : int(len(numeric_cols)),
    "num_nominal_categorical_features" : int(len(nominal_categorical_cols)),
    "num_ordinal_features"             : int(len(ordinal_cols)),
    "num_log_transformed_features"     : int(len(log_cols)),
    "num_features_before_selection"    : int(len(features_before_selection)),
    "num_features_after_selection"     : int(len(selected_features)),
    "classes"                          : le.classes_.tolist(),
    "class_distribution"               : {k: int(v) for k, v in class_counts.items()},
    "num_imbalanced_classes_under_100" : int(imbalanced_count),
    "train_rows"                       : int(len(X_train_out)),
    "test_rows"                        : int(len(X_test_out)),
}
with open("models/preprocessing_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*58)
print("  Preprocessing complete!")
print("="*58)
print("\nFiles saved:")
print("  data/train.csv                         <- train set (85%)")
print("  data/test.csv                          <- locked test set (15%)")
print("  models/label_encoder.pkl               <- target LabelEncoder")
if "numeric_imputer"     in fitted_objects: print("  models/numeric_imputer.pkl             <- SimpleImputer (median)")
if "numeric_scaler"      in fitted_objects: print("  models/numeric_scaler.pkl              <- RobustScaler")
if "categorical_imputer" in fitted_objects: print("  models/categorical_imputer.pkl         <- categorical SimpleImputer")
if "onehot_encoder"      in fitted_objects: print("  models/onehot_encoder.pkl              <- OneHotEncoder")
if "ordinal_imputer"     in fitted_objects: print("  models/ordinal_imputer.pkl             <- ordinal SimpleImputer")
if "ordinal_encoder"     in fitted_objects: print("  models/ordinal_encoder.pkl             <- OrdinalEncoder")
if "variance_selector"   in fitted_objects: print("  models/variance_selector.pkl           <- VarianceThreshold")
print("  models/log_transformed_cols.json       <- log-transformed feature list")
print("  models/feature_columns.json            <- feature group metadata")
print("  models/preprocessing_metadata.json     <- full run metadata")
print(f"\n  Total features going into training : {len(selected_features)}")
print(f"  Classes                            : {len(le.classes_)}")
print(f"  Imbalanced classes (< 100 rows)    : {imbalanced_count}")
print(f"\n  NOTE: {imbalanced_count} rare classes detected.")
print("  Use class_weight='balanced' in train.py to handle this.")
print("="*58)
