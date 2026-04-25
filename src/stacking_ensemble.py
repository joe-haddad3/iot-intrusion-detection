"""
stacking_ensemble.py  --  Stacking Ensemble for IoT Intrusion Detection
=======================================================================
Architecture:
  Level-0 (base models, trained via 3-fold OOF):
    - Random Forest    (100 trees, fast)
    - XGBoost          (GPU, early stopping)
    - LightGBM         (GPU, early stopping)
    - MLP Neural Net   (15 epochs per fold, GPU)

  Level-1 (meta-learner, trained on OOF predictions):
    - LightGBM

Why stacking works:
  Each base model captures different inductive biases:
    RF  : many shallow trees, high variance, good diversity
    XGB : level-wise boosting, L1/L2 regularization
    LGBM: leaf-wise boosting, faster, better on rare splits
    NN  : non-linear feature interactions, global optimization

  The meta-learner learns which base model to trust per region.

Method: 3-Fold Stratified OOF
  - Split train.csv into 3 folds
  - Each base model is trained on 2 folds, predicts the 3rd
  - OOF probabilities form meta-features for the meta-learner
  - After OOF: retrain each base model on FULL train.csv
  - Use full-train models to predict test.csv → meta-test features
  - Evaluate meta-learner on meta-test features

Meta features: [RF_probs(C) | XGB_probs(C) | LGBM_probs(C) | NN_probs(C)]
  where C = number of classes → shape (N, 4*C)
"""

import os, json, time, warnings, gc
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Base models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Meta-learner
from sklearn.model_selection import StratifiedKFold, train_test_split
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
N_FOLDS      = 3

# Base model configs (faster than standalone — speed matters for K-fold)
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH    = 30
RF_MIN_LEAF     = 2

XGB_MAX_ROUNDS  = 500
XGB_EARLY_STOP  = 20
XGB_DEPTH       = 6
XGB_LR          = 0.1

LGBM_MAX_ROUNDS = 500
LGBM_EARLY_STOP = 20
LGBM_LEAVES     = 63
LGBM_LR         = 0.1

NN_EPOCHS       = 20          # epochs per fold (faster than standalone 60)
NN_PATIENCE     = 5
NN_BATCH_SIZE   = 4096
NN_LR           = 1e-3
NN_WEIGHT_DECAY = 1e-4

# Meta-learner LightGBM config
META_LEAVES      = 63
META_LR          = 0.05
META_ROUNDS      = 500
META_EARLY_STOP  = 30

OOF_CHECKPOINT      = os.path.join(MODELS_DIR, "stack_oof_checkpoint.npz")
OOF_FOLD_CHECKPOINT = os.path.join(MODELS_DIR, "stack_oof_fold_checkpoint.npz")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 62)
print("  IoT IDS -- Stacking Ensemble Training")
print("=" * 62)
print(f"\n  Device  : {DEVICE}")
print(f"  K-folds : {N_FOLDS}")
if DEVICE.type == "cuda":
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
for path in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")

train_df      = pd.read_csv(TRAIN_PATH)
test_df       = pd.read_csv(TEST_PATH)

X_train_full  = train_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_train_full  = train_df[LABEL_COL].values
X_test        = test_df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_test        = test_df[LABEL_COL].values
feature_names = train_df.drop(columns=[LABEL_COL]).columns.tolist()

le          = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
class_names = le.classes_.tolist()
n_classes   = len(class_names)
n_features  = X_train_full.shape[1]
n_train     = len(X_train_full)

print(f"\n  Train : {n_train:,} rows x {n_features} features")
print(f"  Test  : {len(X_test):,} rows  x {n_features} features")
print(f"  Classes: {n_classes}")
print(f"  Meta features per model: {n_classes}")
print(f"  Total meta features: {4 * n_classes} (4 models x {n_classes} classes)")

# ── SHARED UTILITIES ──────────────────────────────────────────────────────────
# Class weights (inverse frequency)
unique, counts = np.unique(y_train_full, return_counts=True)
freq           = counts / counts.sum()
cw_arr         = 1.0 / (freq * n_classes)  # shape (n_classes,)

def sample_weights(y, cw):
    return cw[y].astype(np.float32)

# Detect LightGBM GPU
def detect_lgbm_device():
    try:
        tp = {"objective": "multiclass", "num_class": 2, "device": "gpu",
              "num_leaves": 7, "verbosity": -1}
        td = lgb.Dataset(np.random.rand(200, 4).astype(np.float32),
                         label=np.random.randint(0, 2, 200), free_raw_data=True)
        lgb.train(tp, td, num_boost_round=2)
        return "gpu"
    except Exception:
        return "cpu"

LGBM_DEVICE = detect_lgbm_device()
XGB_DEVICE  = "cuda" if DEVICE.type == "cuda" else "cpu"
print(f"  LGBM device: {LGBM_DEVICE} | XGB device: {XGB_DEVICE}")

# ── MLP ARCHITECTURE ──────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_in),
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, n_out),
        )
    def forward(self, x):
        return self.net(x)

def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=NN_BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))

def train_nn(X_tr, y_tr, X_val, y_val, cw, max_epochs, patience):
    alpha_t   = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    model     = MLP(n_features, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=alpha_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    tl = make_loader(X_tr, y_tr, shuffle=True)
    vl = make_loader(X_val, y_val, shuffle=False)

    best_f1, best_state, no_imp = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yb in tl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for Xb, _ in vl:
                preds.append(model(Xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
        vf1 = f1_score(y_val, np.concatenate(preds), average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    model.load_state_dict(best_state)
    return model

def predict_nn_proba(model, X):
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64), shuffle=False)
    probs  = []
    with torch.no_grad():
        for Xb, _ in loader:
            p = torch.softmax(model(Xb.to(DEVICE)), dim=1)
            probs.append(p.cpu().numpy())
    return np.vstack(probs)

# ── OOF PREDICTION COLLECTION ────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# checkpoint: skip Phase 1 if OOF arrays already saved
_xgb_best_iter  = None
_lgbm_best_iter = None

# default params (overwritten inside OOF loop; also used if loading from checkpoint)
lgbm_params = {
    "objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss",
    "device": LGBM_DEVICE, "num_leaves": LGBM_LEAVES, "learning_rate": LGBM_LR,
    "min_child_samples": 20, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.1, "reg_lambda": 0.1,
    "n_jobs": -1, "seed": RANDOM_STATE, "verbosity": -1,
}

if os.path.exists(OOF_CHECKPOINT):
    # Full OOF checkpoint (all folds done) -- skip Phase 1 entirely
    print(f"\n  [CHECKPOINT] Loading full OOF from {OOF_CHECKPOINT} -- skipping Phase 1")
    _ck = np.load(OOF_CHECKPOINT)
    oof_rf   = _ck["oof_rf"]
    oof_xgb  = _ck["oof_xgb"]
    oof_lgbm = _ck["oof_lgbm"]
    oof_nn   = _ck["oof_nn"]
    _xgb_best_iter  = int(_ck["xgb_best_iter"])
    _lgbm_best_iter = int(_ck["lgbm_best_iter"])
    print(f"  OOF shapes: rf={oof_rf.shape}  xgb={oof_xgb.shape}  "
          f"lgbm={oof_lgbm.shape}  nn={oof_nn.shape}")
    print(f"  XGB best iter from ckpt: {_xgb_best_iter}  LGBM: {_lgbm_best_iter}")
else:
    # Load per-fold checkpoint if available (partial progress)
    _start_fold = 1
    oof_rf   = np.zeros((n_train, n_classes), dtype=np.float32)
    oof_xgb  = np.zeros((n_train, n_classes), dtype=np.float32)
    oof_lgbm = np.zeros((n_train, n_classes), dtype=np.float32)
    oof_nn   = np.zeros((n_train, n_classes), dtype=np.float32)

    if os.path.exists(OOF_FOLD_CHECKPOINT):
        _fck = np.load(OOF_FOLD_CHECKPOINT)
        _start_fold     = int(_fck["completed_folds"]) + 1
        _xgb_best_iter  = int(_fck["xgb_best_iter"])
        _lgbm_best_iter = int(_fck["lgbm_best_iter"])
        oof_rf   = _fck["oof_rf"]
        oof_xgb  = _fck["oof_xgb"]
        oof_lgbm = _fck["oof_lgbm"]
        oof_nn   = _fck["oof_nn"]
        print(f"\n  [FOLD CHECKPOINT] Resuming from fold {_start_fold}/{N_FOLDS} "
              f"(folds 1-{_start_fold-1} already done)")
    else:
        print(f"\n{'='*62}")
        print(f"  Phase 1: {N_FOLDS}-Fold OOF -- generating meta-train features")
        print(f"{'='*62}")

    total_oof_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        if fold_idx < _start_fold:
            continue  # already done, skip

        print(f"\n  -- Fold {fold_idx}/{N_FOLDS} ----------------------------------")
        X_tr, X_vl = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_vl = y_train_full[train_idx], y_train_full[val_idx]
        w_tr        = sample_weights(y_tr, cw_arr)

        # -- Random Forest --
        t0 = time.time()
        print(f"    [RF]   n_estimators={RF_N_ESTIMATORS} ...", flush=True)
        rf = RandomForestClassifier(
            n_estimators     = RF_N_ESTIMATORS,
            max_depth        = RF_MAX_DEPTH,
            min_samples_leaf = RF_MIN_LEAF,
            max_features     = "sqrt",
            class_weight     = "balanced_subsample",
            bootstrap        = True,
            random_state     = RANDOM_STATE,
            n_jobs           = 4,  # cap threads to limit peak RAM during predict_proba
        )
        rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf.predict_proba(X_vl)
        rf_f1 = f1_score(y_vl, rf.predict(X_vl), average="macro", zero_division=0)
        print(f"           fold macro-F1={rf_f1:.4f}  [{time.time()-t0:.0f}s]")
        del rf; gc.collect()

        # -- XGBoost (nthread=4 to cap peak RAM on large folds) --
        t0 = time.time()
        print(f"    [XGB]  max_depth={XGB_DEPTH}  lr={XGB_LR} ...", flush=True)
        w_xval = sample_weights(y_vl, cw_arr)
        dtrain_xgb = xgb.DMatrix(X_tr,  label=y_tr,  weight=w_tr,   feature_names=feature_names)
        dval_xgb   = xgb.DMatrix(X_vl,  label=y_vl,  weight=w_xval, feature_names=feature_names)
        xgb_params = {
            "objective": "multi:softprob", "num_class": n_classes,
            "eval_metric": "mlogloss", "device": XGB_DEVICE, "tree_method": "hist",
            "max_depth": XGB_DEPTH, "learning_rate": XGB_LR,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "gamma": 0.1, "reg_lambda": 1.0,
            "nthread": 4,  # limit threads to cap peak RAM
            "seed": RANDOM_STATE, "verbosity": 0,
        }
        xgb_model = xgb.train(
            xgb_params, dtrain_xgb, num_boost_round=XGB_MAX_ROUNDS,
            evals=[(dval_xgb, "val")],
            early_stopping_rounds=XGB_EARLY_STOP,
            verbose_eval=False,
        )
        xgb_proba = xgb_model.predict(xgb.DMatrix(X_vl, feature_names=feature_names)).reshape(-1, n_classes)
        oof_xgb[val_idx] = xgb_proba
        xgb_f1 = f1_score(y_vl, xgb_proba.argmax(axis=1), average="macro", zero_division=0)
        _xgb_best_iter = xgb_model.best_iteration
        print(f"           fold macro-F1={xgb_f1:.4f}  rounds={_xgb_best_iter}  [{time.time()-t0:.0f}s]")
        del dtrain_xgb, dval_xgb, xgb_model; gc.collect()

        # -- LightGBM --
        t0 = time.time()
        print(f"    [LGBM] leaves={LGBM_LEAVES}  lr={LGBM_LR} ...", flush=True)
        w_lval = sample_weights(y_vl, cw_arr)
        dtrain_lgb = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,   feature_name=feature_names, free_raw_data=False)
        dval_lgb   = lgb.Dataset(X_vl, label=y_vl, weight=w_lval, feature_name=feature_names, reference=dtrain_lgb, free_raw_data=False)
        lgbm_params = {
            "objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss",
            "device": LGBM_DEVICE, "num_leaves": LGBM_LEAVES, "learning_rate": LGBM_LR,
            "min_child_samples": 20, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "reg_alpha": 0.1, "reg_lambda": 0.1,
            "n_jobs": -1, "seed": RANDOM_STATE, "verbosity": -1,
        }
        lgbm_cb = [lgb.early_stopping(LGBM_EARLY_STOP, verbose=False), lgb.log_evaluation(period=-1)]
        lgbm_model = lgb.train(lgbm_params, dtrain_lgb, num_boost_round=LGBM_MAX_ROUNDS,
                               valid_sets=[dval_lgb], callbacks=lgbm_cb)
        lgbm_proba = lgbm_model.predict(X_vl)
        oof_lgbm[val_idx] = lgbm_proba
        lgbm_f1 = f1_score(y_vl, lgbm_proba.argmax(axis=1), average="macro", zero_division=0)
        _lgbm_best_iter = lgbm_model.best_iteration
        print(f"           fold macro-F1={lgbm_f1:.4f}  rounds={_lgbm_best_iter}  [{time.time()-t0:.0f}s]")
        del dtrain_lgb, dval_lgb, lgbm_model; gc.collect()

        # -- Neural Network --
        t0 = time.time()
        print(f"    [NN]   epochs={NN_EPOCHS}  patience={NN_PATIENCE} ...", flush=True)
        nn_cw = cw_arr / cw_arr.mean()
        nn_model = train_nn(X_tr, y_tr, X_vl, y_vl, nn_cw, NN_EPOCHS, NN_PATIENCE)
        nn_proba = predict_nn_proba(nn_model, X_vl)
        oof_nn[val_idx] = nn_proba
        nn_f1 = f1_score(y_vl, nn_proba.argmax(axis=1), average="macro", zero_division=0)
        print(f"           fold macro-F1={nn_f1:.4f}  [{time.time()-t0:.0f}s]")
        del nn_model; gc.collect()

        # Save per-fold checkpoint after every completed fold
        np.savez(
            OOF_FOLD_CHECKPOINT,
            oof_rf=oof_rf, oof_xgb=oof_xgb, oof_lgbm=oof_lgbm, oof_nn=oof_nn,
            completed_folds=fold_idx,
            xgb_best_iter=_xgb_best_iter or 200,
            lgbm_best_iter=_lgbm_best_iter or 200,
        )
        print(f"  [FOLD CHECKPOINT] Fold {fold_idx} saved -> {OOF_FOLD_CHECKPOINT}")

    print(f"\n  OOF generation done in {time.time()-total_oof_start:.0f}s")

    # Save full OOF checkpoint
    np.savez(
        OOF_CHECKPOINT,
        oof_rf=oof_rf, oof_xgb=oof_xgb, oof_lgbm=oof_lgbm, oof_nn=oof_nn,
        xgb_best_iter=_xgb_best_iter or 200,
        lgbm_best_iter=_lgbm_best_iter or 200,
    )
    print(f"  [CHECKPOINT] Full OOF saved -> {OOF_CHECKPOINT}")

# OOF performance summary
print("\n  OOF macro-F1 (before meta-learner):")
for name, oof in [("RF", oof_rf), ("XGB", oof_xgb), ("LGBM", oof_lgbm), ("NN", oof_nn)]:
    f1 = f1_score(y_train_full, oof.argmax(axis=1), average="macro", zero_division=0)
    print(f"    {name:<6} : {f1:.4f}")

# ── BUILD META FEATURES ───────────────────────────────────────────────────────
meta_train_X = np.hstack([oof_rf, oof_xgb, oof_lgbm, oof_nn])
meta_train_y = y_train_full
print(f"\n  Meta-train features: {meta_train_X.shape} = (n_train, 4 x {n_classes} classes)")

# ── PHASE 2: RETRAIN BASE MODELS ON FULL TRAINING DATA ────────────────────────
print(f"\n{'='*62}")
print(f"  Phase 2: Retraining all base models on full training set")
print(f"  (used for test set prediction and API inference)")
print(f"{'='*62}")

# Free Phase 1 memory before loading full dataset into new models
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

w_full = sample_weights(y_train_full, cw_arr)

# Resolve best iterations (either from live OOF or from checkpoint)
_xgb_rounds  = _xgb_best_iter  if _xgb_best_iter  else 200
_lgbm_rounds = _lgbm_best_iter if _lgbm_best_iter else 200

# -- RF final --
t0 = time.time()
print("\n  [RF] Training on full train ...", flush=True)
stack_rf = RandomForestClassifier(
    n_estimators     = RF_N_ESTIMATORS,
    max_depth        = RF_MAX_DEPTH,
    min_samples_leaf = RF_MIN_LEAF,
    max_features     = "sqrt",
    class_weight     = "balanced_subsample",
    bootstrap        = True,
    random_state     = RANDOM_STATE,
    n_jobs           = 4,
)
stack_rf.fit(X_train_full, y_train_full)
print(f"  [RF] Done in {time.time()-t0:.0f}s")

# -- XGB final --
# nthread=4 to cap memory; full 2.9M rows on all cores can OOM
t0 = time.time()
print(f"  [XGB] Training on full train ({_xgb_rounds} rounds) ...", flush=True)
gc.collect()
xgb_params_final = {
    "objective": "multi:softprob", "num_class": n_classes,
    "eval_metric": "mlogloss", "device": XGB_DEVICE, "tree_method": "hist",
    "max_depth": XGB_DEPTH, "learning_rate": XGB_LR,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 5, "gamma": 0.1, "reg_lambda": 1.0,
    "nthread": 4,  # limit threads to reduce peak RAM on full 2.9M row training
    "seed": RANDOM_STATE, "verbosity": 0,
}
dfull_xgb = xgb.DMatrix(X_train_full, label=y_train_full, weight=w_full, feature_names=feature_names)
stack_xgb = xgb.train(xgb_params_final, dfull_xgb, num_boost_round=_xgb_rounds, verbose_eval=False)
del dfull_xgb; gc.collect()
print(f"  [XGB] Done in {time.time()-t0:.0f}s  ({_xgb_rounds} rounds)")

# -- LGBM final --
t0 = time.time()
print(f"  [LGBM] Training on full train ({_lgbm_rounds} rounds) ...", flush=True)
dfull_lgb  = lgb.Dataset(X_train_full, label=y_train_full, weight=w_full, feature_name=feature_names)
stack_lgbm = lgb.train(lgbm_params, dfull_lgb, num_boost_round=_lgbm_rounds)
del dfull_lgb; gc.collect()
print(f"  [LGBM] Done in {time.time()-t0:.0f}s  ({_lgbm_rounds} rounds)")

# -- NN final --
t0 = time.time()
print("  [NN] Training on full train ...", flush=True)
nn_cw      = cw_arr / cw_arr.mean()
# Use validation set from a small split just for monitoring
X_tr_nn, X_vl_nn, y_tr_nn, y_vl_nn = train_test_split(
    X_train_full, y_train_full, test_size=0.10, stratify=y_train_full, random_state=RANDOM_STATE
)
stack_nn   = train_nn(X_tr_nn, y_tr_nn, X_vl_nn, y_vl_nn, nn_cw, NN_EPOCHS, NN_PATIENCE)
print(f"  [NN] Done in {time.time()-t0:.0f}s")

# ── GENERATE TEST META FEATURES ───────────────────────────────────────────────
print(f"\n{'='*62}")
print("  Generating test set meta-features ...")
print(f"{'='*62}")

dtest_xgb   = xgb.DMatrix(X_test, feature_names=feature_names)

test_rf     = stack_rf.predict_proba(X_test)
test_xgb    = stack_xgb.predict(dtest_xgb).reshape(-1, n_classes)
test_lgbm   = stack_lgbm.predict(X_test)
test_nn     = predict_nn_proba(stack_nn, X_test)

meta_test_X = np.hstack([test_rf, test_xgb, test_lgbm, test_nn])
print(f"  Meta-test features: {meta_test_X.shape}")

# ── PHASE 3: TRAIN META-LEARNER ───────────────────────────────────────────────
print(f"\n{'='*62}")
print("  Phase 3: Training LightGBM meta-learner")
print(f"  Meta features: {meta_train_X.shape[1]} ({N_FOLDS}-fold OOF from 4 base models)")
print(f"{'='*62}\n")

# Val split from meta-train for early stopping
X_mtr, X_mval, y_mtr, y_mval = train_test_split(
    meta_train_X, meta_train_y,
    test_size    = 0.20,
    stratify     = meta_train_y,
    random_state = RANDOM_STATE,
)

dmet_tr  = lgb.Dataset(X_mtr,  label=y_mtr,  free_raw_data=False)
dmet_val = lgb.Dataset(X_mval, label=y_mval, reference=dmet_tr, free_raw_data=False)

meta_params = {
    "objective"        : "multiclass",
    "num_class"        : n_classes,
    "metric"           : "multi_logloss",
    "device"           : "cpu",  # GPU can cause split errors on meta features
    "num_leaves"       : META_LEAVES,
    "learning_rate"    : META_LR,
    "min_child_samples": 50,
    "feature_fraction" : 0.8,
    "bagging_fraction" : 0.8,
    "bagging_freq"     : 5,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "n_jobs"           : 4,
    "seed"             : RANDOM_STATE,
    "verbosity"        : -1,
}

t0 = time.time()
meta_cb = [lgb.early_stopping(META_EARLY_STOP, verbose=False), lgb.log_evaluation(period=-1)]
meta_learner = lgb.train(
    meta_params, dmet_tr, num_boost_round=META_ROUNDS,
    valid_sets=[dmet_val], callbacks=meta_cb,
)
meta_time = time.time() - t0

val_meta_proba = meta_learner.predict(X_mval)
val_meta_f1    = f1_score(y_mval, val_meta_proba.argmax(axis=1), average="macro", zero_division=0)
print(f"  Meta-learner val macro-F1 : {val_meta_f1:.4f}  rounds={meta_learner.best_iteration}  [{meta_time:.0f}s]")

# Retrain meta-learner on all meta-train data
print("  Retraining meta-learner on full meta-train ...")
dmet_full = lgb.Dataset(meta_train_X, label=meta_train_y, free_raw_data=False)
t0 = time.time()
meta_learner_final = lgb.train(
    meta_params, dmet_full, num_boost_round=meta_learner.best_iteration or 100,
)
print(f"  Done in {time.time()-t0:.0f}s")

# ── EVALUATE ENSEMBLE ON LOCKED TEST SET ─────────────────────────────────────
print(f"\n{'='*62}")
print("  Final evaluation on locked test set")
print(f"{'='*62}")

y_pred_prob = meta_learner_final.predict(meta_test_X)
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

# Compare base model vs ensemble on test
print("\n  Per-model test macro-F1 vs ensemble:")
for name, proba in [("RF", test_rf), ("XGB", test_xgb), ("LGBM", test_lgbm), ("NN", test_nn)]:
    f1 = f1_score(y_test, proba.argmax(axis=1), average="macro", zero_division=0)
    print(f"    {name:<6} : {f1:.4f}")
print(f"    {'Ensemble':<6} : {macro_f1:.4f}  <- stacking result")

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
ax.set_title(f"Stacking Ensemble (RF+XGB+LGBM+NN -> LGBM) -- Confusion Matrix\n"
             f"Macro F1={macro_f1:.4f}  Acc={accuracy:.4f}", fontsize=12)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True",      fontsize=11)
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "ensemble_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/ensemble_confusion_matrix.png")

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
plt.suptitle("Stacking Ensemble -- Per-class Metrics (Test Set)", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "ensemble_per_class_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/ensemble_per_class_metrics.png")

# Model comparison bar chart
model_names = ["RF", "XGB", "LGBM", "NN", "Ensemble"]
model_f1s   = [
    f1_score(y_test, test_rf.argmax(1),   average="macro", zero_division=0),
    f1_score(y_test, test_xgb.argmax(1),  average="macro", zero_division=0),
    f1_score(y_test, test_lgbm.argmax(1), average="macro", zero_division=0),
    f1_score(y_test, test_nn.argmax(1),   average="macro", zero_division=0),
    macro_f1,
]
colors_cmp = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(model_names, model_f1s, color=colors_cmp, edgecolor="white", width=0.6)
ax.set_ylim(0, min(1.0, max(model_f1s) * 1.12))
ax.set_ylabel("Test Macro F1-score")
ax.set_title("Model Comparison — Test Macro F1\n(red = stacking ensemble)")
for bar, val in zip(bars, model_f1s):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.4f}",
            ha="center", va="bottom", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "ensemble_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/ensemble_model_comparison.png")

# ── SAVE MODELS ───────────────────────────────────────────────────────────────
joblib.dump(stack_rf,  os.path.join(MODELS_DIR, "stack_rf_model.pkl"))
stack_xgb.save_model(  os.path.join(MODELS_DIR, "stack_xgb_model.ubj"))
joblib.dump(stack_lgbm, os.path.join(MODELS_DIR, "stack_lgbm_model.pkl"))
torch.save({
    "model_state_dict": stack_nn.state_dict(),
    "n_features": n_features, "n_classes": n_classes, "class_names": class_names,
}, os.path.join(MODELS_DIR, "stack_nn_model.pt"))
joblib.dump(meta_learner_final, os.path.join(MODELS_DIR, "stack_meta_lgbm_model.pkl"))

print("\n  Models saved:")
print("    models/stack_rf_model.pkl")
print("    models/stack_xgb_model.ubj")
print("    models/stack_lgbm_model.pkl")
print("    models/stack_nn_model.pt")
print("    models/stack_meta_lgbm_model.pkl")

# ── SAVE METRICS ──────────────────────────────────────────────────────────────
LR_MACRO_F1 = 0.4401; RF_MACRO_F1 = 0.5861; XGB_MACRO_F1 = 0.6200

metrics = {
    "model"             : "StackingEnsemble",
    "base_models"       : ["RandomForest", "XGBoost", "LightGBM", "MLP"],
    "meta_learner"      : "LightGBM",
    "n_folds"           : N_FOLDS,
    "meta_features"     : 4 * n_classes,
    "val_macro_f1_meta" : round(val_meta_f1, 4),
    "test_macro_f1"     : round(macro_f1,    4),
    "test_micro_f1"     : round(micro_f1,    4),
    "test_weighted_f1"  : round(weighted_f1, 4),
    "test_accuracy"     : round(accuracy,    4),
    "test_macro_precision": round(macro_prec, 4),
    "test_macro_recall" : round(macro_rec,   4),
    "test_roc_auc_macro": round(roc_auc, 4) if roc_auc else None,
    "benign_false_positive_rate": round(fp_rate, 4) if fp_rate is not None else None,
    "vs_lr_macro_f1_delta" : round(macro_f1 - LR_MACRO_F1,  4),
    "vs_rf_macro_f1_delta" : round(macro_f1 - RF_MACRO_F1,  4),
    "vs_xgb_macro_f1_delta": round(macro_f1 - XGB_MACRO_F1, 4),
    "base_model_test_f1": {
        "rf"  : round(model_f1s[0], 4),
        "xgb" : round(model_f1s[1], 4),
        "lgbm": round(model_f1s[2], 4),
        "nn"  : round(model_f1s[3], 4),
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

metrics_path = os.path.join(REPORTS_DIR, "ensemble_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"\n  Metrics saved -> {metrics_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  STACKING ENSEMBLE -- FINAL RESULTS")
print(f"{'='*62}")
print(f"  Base: RF + XGB + LGBM + NN ({N_FOLDS}-fold OOF)")
print(f"  Meta: LightGBM ({meta_learner.best_iteration} rounds)")
print(f"  Test Macro F1    : {macro_f1:.4f}   <- primary metric")
print(f"  Test Accuracy    : {accuracy:.4f}")
print(f"  Test ROC-AUC     : {roc_auc:.4f}" if roc_auc else "  Test ROC-AUC     : N/A")
print(f"\n  Base model test F1:")
for n, f in zip(model_names[:-1], model_f1s[:-1]):
    print(f"    {n:<6}: {f:.4f}")
print(f"    Ensemble gain: {macro_f1 - max(model_f1s[:-1]):+.4f} over best base model")
print(f"\n  vs LR  : {LR_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-LR_MACRO_F1:+.4f})")
print(f"  vs RF  : {RF_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-RF_MACRO_F1:+.4f})")
print(f"  vs XGB : {XGB_MACRO_F1:.4f} -> {macro_f1:.4f}  ({macro_f1-XGB_MACRO_F1:+.4f})")
print(f"\n  Outputs:")
print(f"    models/stack_rf_model.pkl, stack_xgb_model.ubj,")
print(f"    stack_lgbm_model.pkl, stack_nn_model.pt, stack_meta_lgbm_model.pkl")
print(f"    reports/ensemble_metrics.json")
print(f"    reports/ensemble_confusion_matrix.png")
print(f"    reports/ensemble_per_class_metrics.png")
print(f"    reports/ensemble_model_comparison.png")
print(f"{'='*62}")
