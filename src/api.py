"""
api.py — FastAPI inference endpoint for IoT Intrusion Detection
===============================================================
Models served:
  /predict/lr           → Logistic Regression (multiclass)
  /predict/binary       → Binary LR (BENIGN vs ATTACK)
  /predict/rf           → Random Forest (multiclass)
  /predict/xgb          → XGBoost (multiclass)
  /predict/lgbm         → LightGBM (multiclass)
  /predict/lgbm_smote         → LightGBM + SMOTE (rare-class optimised)
  /predict/lgbm_smote_tuned  → LightGBM + SMOTE (Optuna tuned, best multiclass)
  /predict/nn                → MLP Neural Network (weighted CrossEntropy)
  /predict/nn_focal          → MLP Neural Network (Focal Loss)
  /predict/ensemble          → Stacking Ensemble (RF+XGB+LGBM+NN -> LGBM)
  /predict/all               → All models at once

Models are loaded lazily on startup — missing models are skipped
gracefully so the server can start even if not all models exist yet.

Run with:
    uvicorn src.api:app --reload
or from the project root:
    uvicorn api:app --reload --app-dir src
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any

warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
STATIC_DIR  = os.path.join(os.path.dirname(__file__), "static")

# ── LAZY MODEL LOADER ──────────────────────────────────────────────────────────
def try_load(path: str, loader=None):
    """Load a model file gracefully; return None if not found."""
    full = os.path.join(MODELS_DIR, path)
    if not os.path.exists(full):
        print(f"  [SKIP] {path} not found")
        return None
    try:
        if loader is not None:
            return loader(full)
        return joblib.load(full)
    except Exception as e:
        print(f"  [ERROR] loading {path}: {e}")
        return None

# ── PYTORCH MLP (must be defined to load nn_model.pt / nn_focal_model.pt) ─────
try:
    import torch
    import torch.nn as nn

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

    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE    = None
    print("  [WARN] PyTorch not installed — NN endpoints will be unavailable")

def load_nn_model(path: str):
    if not TORCH_AVAILABLE:
        return None
    ckpt    = torch.load(path, map_location=TORCH_DEVICE, weights_only=False)
    model   = MLP(ckpt["n_features"], ckpt["n_classes"]).to(TORCH_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def nn_predict_proba(model, X: np.ndarray) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=4096, shuffle=False)
    probs  = []
    with torch.no_grad():
        for (Xb,) in loader:
            p = torch.softmax(model(Xb.to(TORCH_DEVICE)), dim=1)
            probs.append(p.cpu().numpy())
    return np.vstack(probs)

# ── XGBOOST LOADER ────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("  [WARN] XGBoost not installed — XGB endpoint will be unavailable")

def load_xgb_model(path: str):
    if not XGB_AVAILABLE:
        return None
    m = xgb.Booster()
    m.load_model(path)
    return m

# ── LIGHTGBM AVAILABILITY ─────────────────────────────────────────────────────
import importlib.util as _ilu
LGBM_AVAILABLE = _ilu.find_spec("lightgbm") is not None
if not LGBM_AVAILABLE:
    print("  [WARN] LightGBM not installed — LGBM endpoints will be unavailable")

# ── LOAD ALL ARTIFACTS ─────────────────────────────────────────────────────────
print("Loading models and preprocessing artifacts...")

# Preprocessing pipeline
label_encoder     = try_load("label_encoder.pkl")
numeric_imputer   = try_load("numeric_imputer.pkl")
numeric_scaler    = try_load("numeric_scaler.pkl")
variance_selector = try_load("variance_selector.pkl")

with open(os.path.join(MODELS_DIR, "feature_columns.json"), encoding="utf-8") as f:
    feature_info = json.load(f)
with open(os.path.join(MODELS_DIR, "log_transformed_cols.json"), encoding="utf-8") as f:
    log_cols = json.load(f)

NUMERIC_COLS      = feature_info["numeric_cols"]
SELECTED_FEATURES = feature_info["selected_features"]
CLASS_NAMES       = label_encoder.classes_.tolist()
N_CLASSES         = len(CLASS_NAMES)

# Standard models
lr_model     = try_load("lr_model.pkl")
binary_model     = try_load("binary_lr_model.pkl")
binary_threshold = try_load("binary_threshold.pkl") or 0.5
rf_model     = try_load("rf_model.pkl")

# New models
xgb_model        = try_load("xgb_model.ubj",  loader=load_xgb_model)    if XGB_AVAILABLE  else None
lgbm_model       = try_load("lgbm_model.pkl")                             if LGBM_AVAILABLE else None
lgbm_smote_model       = try_load("lgbm_smote_model.pkl")              if LGBM_AVAILABLE else None
lgbm_smote_tuned_model = try_load("lgbm_smote_tuned_model.pkl")         if LGBM_AVAILABLE else None
nn_model         = try_load("nn_model.pt",     loader=load_nn_model)      if TORCH_AVAILABLE else None
nn_focal_model   = try_load("nn_focal_model.pt", loader=load_nn_model)    if TORCH_AVAILABLE else None

# Stacking ensemble base models + meta-learner
stack_rf_model   = try_load("stack_rf_model.pkl")
stack_xgb_model  = try_load("stack_xgb_model.ubj",  loader=load_xgb_model)  if XGB_AVAILABLE  else None
stack_lgbm_model = try_load("stack_lgbm_model.pkl")                          if LGBM_AVAILABLE else None
stack_nn_model   = try_load("stack_nn_model.pt",     loader=load_nn_model)   if TORCH_AVAILABLE else None
stack_meta_model = try_load("stack_meta_lgbm_model.pkl")                     if LGBM_AVAILABLE else None

ENSEMBLE_READY = all(m is not None for m in [
    stack_rf_model, stack_xgb_model, stack_lgbm_model, stack_nn_model, stack_meta_model
])

print(f"  Label encoder     : {N_CLASSES} classes")
print(f"  LR model          : {'ok' if lr_model          else 'missing'}")
print(f"  Binary LR model   : {'ok' if binary_model      else 'missing'}")
print(f"  Random Forest     : {'ok' if rf_model          else 'missing'}")
print(f"  XGBoost           : {'ok' if xgb_model         else 'missing'}")
print(f"  LightGBM          : {'ok' if lgbm_model        else 'missing'}")
print(f"  LightGBM+SMOTE    : {'ok' if lgbm_smote_model        else 'missing'}")
print(f"  LGBM+SMOTE Tuned  : {'ok' if lgbm_smote_tuned_model else 'missing'}")
print(f"  NN (CE Loss)      : {'ok' if nn_model          else 'missing'}")
print(f"  NN (Focal Loss)   : {'ok' if nn_focal_model    else 'missing'}")
print(f"  Stacking Ensemble : {'ok' if ENSEMBLE_READY    else 'missing (run stacking_ensemble.py)'}")
print("Ready.")

# ── DRIFT MONITOR ─────────────────────────────────────────────────────────────
_drift_ref_path = os.path.join(MODELS_DIR, "drift_reference.json")
_drift_ref = None
if os.path.exists(_drift_ref_path):
    with open(_drift_ref_path) as _f:
        _drift_ref = json.load(_f)
    print(f"  Drift reference  : loaded ({_drift_ref['sample_size']:,} training samples)")
else:
    print("  Drift reference  : missing (run src/save_drift_reference.py)")

WINDOW_SIZE = 500   # rolling window of recent requests

class DriftMonitor:
    def __init__(self):
        self.features    = deque(maxlen=WINDOW_SIZE)   # np arrays (n_features,)
        self.predictions = deque(maxlen=WINDOW_SIZE)   # class name strings
        self.timestamps  = deque(maxlen=WINDOW_SIZE)   # float epoch seconds

    def record(self, feat: np.ndarray, prediction: str):
        self.features.append(feat.flatten())
        self.predictions.append(prediction)
        self.timestamps.append(time.time())

    def reset(self):
        self.features.clear()
        self.predictions.clear()
        self.timestamps.clear()

    def stats(self) -> dict:
        n = len(self.predictions)
        if n == 0:
            return {"window_size": 0, "message": "No predictions recorded yet."}

        result: Dict[str, Any] = {"window_size": n}

        # ── 1. Alert rate ────────────────────────────────────────────────────
        now     = time.time()
        last_60 = sum(1 for t in self.timestamps if now - t <= 60)
        last_10 = sum(1 for t in self.timestamps if now - t <= 10)
        attacks_60 = sum(
            1 for t, p in zip(self.timestamps, self.predictions)
            if now - t <= 60 and p != "BENIGN"
        )
        result["alert_rate"] = {
            "requests_last_60s" : last_60,
            "requests_last_10s" : last_10,
            "attacks_last_60s"  : attacks_60,
            "attack_rate_pct"   : round(attacks_60 / max(last_60, 1) * 100, 1),
            "status"            : "HIGH" if attacks_60 / max(last_60, 1) > 0.5 else "NORMAL",
        }

        # ── 2. Predicted class mix ───────────────────────────────────────────
        from collections import Counter
        pred_counts = Counter(self.predictions)
        pred_dist   = {cls: round(cnt / n * 100, 1) for cls, cnt in pred_counts.items()}
        mix_drift   = []
        if _drift_ref:
            ref_dist = _drift_ref["class_distribution"]
            for cls, recent_frac in {c: cnt/n for c, cnt in pred_counts.items()}.items():
                ref_frac = ref_dist.get(cls, 0.0)
                delta    = recent_frac - ref_frac
                if abs(delta) > 0.05:   # >5% absolute shift is noteworthy
                    mix_drift.append({
                        "class"        : cls,
                        "recent_pct"   : round(recent_frac * 100, 1),
                        "training_pct" : round(ref_frac    * 100, 1),
                        "delta_pct"    : round(delta       * 100, 1),
                    })
            mix_drift.sort(key=lambda x: abs(x["delta_pct"]), reverse=True)

        result["class_mix"] = {
            "recent_distribution" : pred_dist,
            "drifted_classes"     : mix_drift,
            "status"              : "DRIFT DETECTED" if mix_drift else "NORMAL",
        }

        # ── 3. Feature drift ────────────────────────────────────────────────
        if _drift_ref and len(self.features) >= 10:
            arr         = np.array(list(self.features))  # (N, n_features)
            recent_mean = arr.mean(axis=0)
            ref_mean    = np.array(_drift_ref["feature_mean"])
            ref_std     = np.array(_drift_ref["feature_std"])
            z_scores    = np.abs((recent_mean - ref_mean) / ref_std)
            feat_names  = _drift_ref["feature_names"]
            drifted     = [
                {"feature": feat_names[i], "z_score": round(float(z_scores[i]), 2)}
                for i in np.where(z_scores > 2.0)[0]
            ]
            drifted.sort(key=lambda x: x["z_score"], reverse=True)
            result["feature_drift"] = {
                "drifted_features" : drifted,
                "max_z_score"      : round(float(z_scores.max()), 2),
                "status"           : "DRIFT DETECTED" if drifted else "NORMAL",
            }
        else:
            result["feature_drift"] = {
                "status"  : "INSUFFICIENT DATA" if _drift_ref else "NO REFERENCE",
                "message" : f"Need ≥10 samples (have {len(self.features)})" if _drift_ref else
                            "Run src/save_drift_reference.py first",
            }

        # Overall status
        statuses = [
            result["alert_rate"]["status"],
            result["class_mix"]["status"],
            result.get("feature_drift", {}).get("status", "NORMAL"),
        ]
        result["overall_status"] = "ALERT" if any(s != "NORMAL" for s in statuses) else "NORMAL"
        return result

drift_monitor = DriftMonitor()

# ── FASTAPI APP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IoT Intrusion Detection System",
    description=(
        "## Real-Time Network Intrusion Detection API\n\n"
        "Classifies IoT network traffic into **34 attack categories** using 8 ML models trained on 2.9M samples.\n\n"
        "### Models Available\n"
        "| Model | Macro F1 | Accuracy |\n"
        "|---|---|---|\n"
        "| LightGBM + SMOTE (Tuned) | **0.648** | 79.89% |\n"
        "| LightGBM + SMOTE | 0.647 | 79.69% |\n"
        "| LightGBM | 0.615 | 77.02% |\n"
        "| XGBoost | 0.611 | 76.53% |\n"
        "| Random Forest | 0.586 | 76.42% |\n"
        "| NN (CE Loss) | 0.562 | 71.14% |\n"
        "| NN (Focal Loss) | 0.541 | 67.26% |\n"
        "| Logistic Regression | 0.440 | 58.53% |\n\n"
        "### Quick Start\n"
        "Use `/predict/all` to run all models at once and get a majority vote result."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "predict", "description": "Run predictions on network traffic"},
        {"name": "info",    "description": "Model and system information"},
    ],
)

# ── STATIC FILES ────────────────────────────────────────────────────────────────
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── INPUT SCHEMA ───────────────────────────────────────────────────────────────
class TrafficRecord(BaseModel):
    Header_Length: float
    Protocol_Type: float
    Time_To_Live: float
    Rate: float
    fin_flag_number: float
    syn_flag_number: float
    rst_flag_number: float
    psh_flag_number: float
    ack_flag_number: float
    ece_flag_number: float
    cwr_flag_number: float
    ack_count: float
    syn_count: float
    fin_count: float
    rst_count: float
    HTTP: float
    HTTPS: float
    DNS: float
    Telnet: float
    SMTP: float
    SSH: float
    IRC: float
    TCP: float
    UDP: float
    DHCP: float
    ARP: float
    ICMP: float
    IGMP: float
    IPv: float
    LLC: float
    Tot_sum: float
    Min: float
    Max: float
    AVG: float
    Std: float
    Tot_size: float
    IAT: float
    Number: float
    Variance: float

# ── PREPROCESSING ──────────────────────────────────────────────────────────────
def preprocess(record: TrafficRecord) -> np.ndarray:
    """Apply the exact same pipeline used during training."""
    raw = {
        "Header_Length"  : record.Header_Length,
        "Protocol Type"  : record.Protocol_Type,
        "Time_To_Live"   : record.Time_To_Live,
        "Rate"           : record.Rate,
        "fin_flag_number": record.fin_flag_number,
        "syn_flag_number": record.syn_flag_number,
        "rst_flag_number": record.rst_flag_number,
        "psh_flag_number": record.psh_flag_number,
        "ack_flag_number": record.ack_flag_number,
        "ece_flag_number": record.ece_flag_number,
        "cwr_flag_number": record.cwr_flag_number,
        "ack_count"      : record.ack_count,
        "syn_count"      : record.syn_count,
        "fin_count"      : record.fin_count,
        "rst_count"      : record.rst_count,
        "HTTP"           : record.HTTP,
        "HTTPS"          : record.HTTPS,
        "DNS"            : record.DNS,
        "Telnet"         : record.Telnet,
        "SMTP"           : record.SMTP,
        "SSH"            : record.SSH,
        "IRC"            : record.IRC,
        "TCP"            : record.TCP,
        "UDP"            : record.UDP,
        "DHCP"           : record.DHCP,
        "ARP"            : record.ARP,
        "ICMP"           : record.ICMP,
        "IGMP"           : record.IGMP,
        "IPv"            : record.IPv,
        "LLC"            : record.LLC,
        "Tot sum"        : record.Tot_sum,
        "Min"            : record.Min,
        "Max"            : record.Max,
        "AVG"            : record.AVG,
        "Std"            : record.Std,
        "Tot size"       : record.Tot_size,
        "IAT"            : record.IAT,
        "Number"         : record.Number,
        "Variance"       : record.Variance,
    }
    df = pd.DataFrame([raw], columns=NUMERIC_COLS)
    df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    df_imp   = numeric_imputer.transform(df)
    df_scl   = numeric_scaler.transform(df_imp)
    df_final = variance_selector.transform(df_scl)
    return df_final.astype(np.float32)

# ── HELPERS ────────────────────────────────────────────────────────────────────
def top_probs(probs: np.ndarray, top_n: int = 5) -> list:
    indices = np.argsort(probs)[::-1][:top_n]
    return [{"class": CLASS_NAMES[i], "probability": round(float(probs[i]), 4)} for i in indices]

def require(model, name: str):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"{name} model not loaded. Run the corresponding training script first.",
        )

def sklearn_result(model, X: np.ndarray, model_name: str, record: bool = True) -> dict:
    Xdf        = pd.DataFrame(X, columns=SELECTED_FEATURES)
    idx        = int(model.predict(Xdf)[0])
    probs      = model.predict_proba(Xdf)[0]
    prediction = CLASS_NAMES[idx]
    if record:
        drift_monitor.record(X, prediction)
    return {
        "model"      : model_name,
        "prediction" : prediction,
        "confidence" : round(float(probs[idx]), 4),
        "top5"       : top_probs(probs),
    }

def xgb_result(model, X: np.ndarray, model_name: str, record: bool = True) -> dict:
    if not XGB_AVAILABLE:
        raise HTTPException(status_code=503, detail="XGBoost not installed")
    dm         = xgb.DMatrix(X, feature_names=SELECTED_FEATURES)
    probs      = model.predict(dm)[0]
    idx        = int(probs.argmax())
    prediction = CLASS_NAMES[idx]
    if record:
        drift_monitor.record(X, prediction)
    return {
        "model"      : model_name,
        "prediction" : prediction,
        "confidence" : round(float(probs[idx]), 4),
        "top5"       : top_probs(probs),
    }

def lgbm_result(model, X: np.ndarray, model_name: str, record: bool = True) -> dict:
    probs      = model.predict(X)[0]
    idx        = int(probs.argmax())
    prediction = CLASS_NAMES[idx]
    if record:
        drift_monitor.record(X, prediction)
    return {
        "model"      : model_name,
        "prediction" : prediction,
        "confidence" : round(float(probs[idx]), 4),
        "top5"       : top_probs(probs),
    }

def nn_result(model, X: np.ndarray, model_name: str, record: bool = True) -> dict:
    probs      = nn_predict_proba(model, X)[0]
    idx        = int(probs.argmax())
    prediction = CLASS_NAMES[idx]
    if record:
        drift_monitor.record(X, prediction)
    return {
        "model"      : model_name,
        "prediction" : prediction,
        "confidence" : round(float(probs[idx]), 4),
        "top5"       : top_probs(probs),
    }

def ensemble_predict(X: np.ndarray) -> dict:
    """Run all 4 stacking base models, concatenate probs, run meta-learner."""
    Xdf    = pd.DataFrame(X, columns=SELECTED_FEATURES)
    rf_p   = stack_rf_model.predict_proba(Xdf)                           # (1, C)
    xgb_p  = stack_xgb_model.predict(xgb.DMatrix(X, feature_names=SELECTED_FEATURES)).reshape(1, -1)  # (1, C)
    lgbm_p = stack_lgbm_model.predict(X)                                 # (1, C)
    nn_p   = nn_predict_proba(stack_nn_model, X)                         # (1, C)

    meta_X  = np.hstack([rf_p, xgb_p, lgbm_p, nn_p]).astype(np.float32)  # (1, 4C)
    probs   = stack_meta_model.predict(meta_X)[0]                          # (C,)
    idx     = int(probs.argmax())

    return {
        "model"      : "StackingEnsemble",
        "prediction" : CLASS_NAMES[idx],
        "confidence" : round(float(probs[idx]), 4),
        "top5"       : top_probs(probs),
        "base_model_predictions": {
            "rf"  : CLASS_NAMES[rf_p[0].argmax()],
            "xgb" : CLASS_NAMES[xgb_p[0].argmax()],
            "lgbm": CLASS_NAMES[lgbm_p[0].argmax()],
            "nn"  : CLASS_NAMES[nn_p[0].argmax()],
        },
    }

# ── ROOT + HEALTH ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {
        "service": "IoT Intrusion Detection API v2",
        "models_available": {
            "lr": lr_model is not None, "binary": binary_model is not None,
            "rf": rf_model is not None, "xgb": xgb_model is not None,
            "lgbm": lgbm_model is not None, "lgbm_smote": lgbm_smote_model is not None,
            "lgbm_smote_tuned": lgbm_smote_tuned_model is not None,
            "nn": nn_model is not None, "nn_focal": nn_focal_model is not None,
            "ensemble": ENSEMBLE_READY,
        },
        "docs": "/docs",
    }

@app.get("/health", tags=["info"])
def health():
    """Check if the API is running."""
    return {"status": "ok"}

@app.get("/models/info", tags=["info"])
def models_info():
    """Return loaded model availability and performance metrics."""
    return {
        "models_available": {
            "lr": lr_model is not None, "binary": binary_model is not None,
            "rf": rf_model is not None, "xgb": xgb_model is not None,
            "lgbm": lgbm_model is not None, "lgbm_smote": lgbm_smote_model is not None,
            "lgbm_smote_tuned": lgbm_smote_tuned_model is not None,
            "nn": nn_model is not None, "nn_focal": nn_focal_model is not None,
            "ensemble": ENSEMBLE_READY,
        },
        "performance": {
            "lgbm_smote_tuned":  {"test_macro_f1": 0.648, "test_accuracy": 0.7989},
            "lgbm_smote":        {"test_macro_f1": 0.647, "test_accuracy": 0.7969},
            "lgbm":              {"test_macro_f1": 0.615, "test_accuracy": 0.7702},
            "xgb":               {"test_macro_f1": 0.611, "test_accuracy": 0.7653},
            "rf":                {"test_macro_f1": 0.586, "test_accuracy": 0.7642},
            "nn":                {"test_macro_f1": 0.562, "test_accuracy": 0.7114},
            "nn_focal":          {"test_macro_f1": 0.541, "test_accuracy": 0.6726},
            "lr":                {"test_macro_f1": 0.440, "test_accuracy": 0.5853},
            "binary":            {"test_accuracy": 0.8367},
        },
        "dataset": {"train_samples": 2932834, "test_samples": 517560, "n_classes": 34},
        "classes": CLASS_NAMES,
    }

# ── INDIVIDUAL ENDPOINTS ───────────────────────────────────────────────────────
@app.post("/predict/lr", tags=["predict"])
def predict_lr(record: TrafficRecord):
    """Logistic Regression — 34-class multiclass prediction."""
    require(lr_model, "LR")
    try:
        return sklearn_result(lr_model, preprocess(record), "LogisticRegression")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/binary", tags=["predict"])
def predict_binary(record: TrafficRecord):
    """Binary LightGBM — BENIGN vs ATTACK."""
    require(binary_model, "BinaryLGBM")
    try:
        X            = preprocess(record)
        prob_attack  = float(binary_model.predict(X)[0])
        prob_benign  = 1.0 - prob_attack
        pred         = 1 if prob_attack >= binary_threshold else 0
        label        = "BENIGN" if pred == 0 else "ATTACK"
        confidence   = prob_benign if pred == 0 else prob_attack
        drift_monitor.record(X, label)
        return {
            "model"      : "BinaryLightGBM",
            "prediction" : label,
            "confidence" : round(confidence, 4),
            "prob_benign": round(prob_benign, 4),
            "prob_attack": round(prob_attack, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/rf", tags=["predict"])
def predict_rf(record: TrafficRecord):
    """Random Forest — 34-class multiclass prediction."""
    require(rf_model, "RandomForest")
    try:
        return sklearn_result(rf_model, preprocess(record), "RandomForest")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/xgb", tags=["predict"])
def predict_xgb(record: TrafficRecord):
    """XGBoost (GPU) — 34-class multiclass prediction."""
    require(xgb_model, "XGBoost")
    try:
        return xgb_result(xgb_model, preprocess(record), "XGBoost")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lgbm", tags=["predict"])
def predict_lgbm(record: TrafficRecord):
    """LightGBM (leaf-wise, GPU) — 34-class multiclass prediction."""
    require(lgbm_model, "LightGBM")
    try:
        return lgbm_result(lgbm_model, preprocess(record), "LightGBM")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lgbm_smote", tags=["predict"])
def predict_lgbm_smote(record: TrafficRecord):
    """LightGBM + SMOTE — optimised for rare attack classes."""
    require(lgbm_smote_model, "LightGBM_SMOTE")
    try:
        return lgbm_result(lgbm_smote_model, preprocess(record), "LightGBM_SMOTE")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lgbm_smote_tuned", tags=["predict"])
def predict_lgbm_smote_tuned(record: TrafficRecord):
    """LightGBM + SMOTE (Optuna tuned) — best multiclass model (Macro F1=0.648)."""
    require(lgbm_smote_tuned_model, "LightGBM_SMOTE_Tuned")
    try:
        return lgbm_result(lgbm_smote_tuned_model, preprocess(record), "LightGBM_SMOTE_Tuned")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/nn", tags=["predict"])
def predict_nn(record: TrafficRecord):
    """MLP Neural Network (weighted CrossEntropy) — 34-class prediction."""
    require(nn_model, "NeuralNetwork")
    try:
        return nn_result(nn_model, preprocess(record), "MLP_NeuralNetwork")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/nn_focal", tags=["predict"])
def predict_nn_focal(record: TrafficRecord):
    """MLP Neural Network (Focal Loss, gamma=2) — optimised for rare classes."""
    require(nn_focal_model, "NeuralNetwork_FocalLoss")
    try:
        return nn_result(nn_focal_model, preprocess(record), "MLP_FocalLoss")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/ensemble", tags=["predict"])
def predict_ensemble(record: TrafficRecord):
    """Stacking Ensemble: RF + XGB + LGBM + NN -> LightGBM meta-learner."""
    if not ENSEMBLE_READY:
        raise HTTPException(
            status_code=503,
            detail="Stacking ensemble not ready. Run src/stacking_ensemble.py first.",
        )
    try:
        return ensemble_predict(preprocess(record))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── DRIFT MONITORING ENDPOINTS ────────────────────────────────────────────────
@app.get("/monitor/drift", tags=["monitor"])
def get_drift():
    """
    Returns live drift statistics based on the last 500 predictions:
    - **alert_rate**: how many ATTACK predictions in the last 60s / 10s
    - **class_mix**: are certain attack types appearing more than during training?
    - **feature_drift**: are the input values shifting away from training distribution?
    """
    return drift_monitor.stats()

@app.post("/monitor/reset", tags=["monitor"])
def reset_drift():
    """Reset the rolling prediction window (clears drift history)."""
    drift_monitor.reset()
    return {"status": "ok", "message": "Drift monitor window cleared."}

# ── ALL MODELS ─────────────────────────────────────────────────────────────────
@app.post("/predict/all", tags=["predict"])
def predict_all(record: TrafficRecord):
    """Run all available models on the same input and return all results."""
    try:
        X       = preprocess(record)
        results: Dict[str, Any] = {}

        if lr_model:
            r = sklearn_result(lr_model, X, "LogisticRegression")
            results["logistic_regression"] = {"prediction": r["prediction"], "confidence": r["confidence"], "top5": r["top5"]}

        if binary_model:
            prob_attack = float(binary_model.predict(X)[0])
            prob_benign = 1.0 - prob_attack
            pred        = 1 if prob_attack >= binary_threshold else 0
            results["binary"] = {
                "prediction" : "BENIGN" if pred == 0 else "ATTACK",
                "confidence" : round(prob_benign if pred == 0 else prob_attack, 4),
                "prob_benign": round(prob_benign, 4),
                "prob_attack": round(prob_attack, 4),
            }

        if rf_model:
            r = sklearn_result(rf_model, X, "RandomForest")
            results["random_forest"] = {"prediction": r["prediction"], "confidence": r["confidence"], "top5": r["top5"]}

        if xgb_model:
            r = xgb_result(xgb_model, X, "XGBoost")
            results["xgboost"] = {"prediction": r["prediction"], "confidence": r["confidence"], "top5": r["top5"]}

        if lgbm_model:
            r = lgbm_result(lgbm_model, X, "LightGBM")
            results["lightgbm"] = {"prediction": r["prediction"], "confidence": r["confidence"], "top5": r["top5"]}

        if lgbm_smote_model:
            r = lgbm_result(lgbm_smote_model, X, "LightGBM_SMOTE")
            results["lightgbm_smote"] = {"prediction": r["prediction"], "confidence": r["confidence"], "top5": r["top5"]}

        if lgbm_smote_tuned_model:
            r = lgbm_result(lgbm_smote_tuned_model, X, "LightGBM_SMOTE_Tuned")
            results["lightgbm_smote_tuned"] = {"prediction": r["prediction"], "confidence": r["confidence"], "top5": r["top5"]}

        if nn_model:
            r = nn_result(nn_model, X, "MLP_NeuralNetwork")
            results["neural_network"] = {"prediction": r["prediction"], "confidence": r["confidence"], "top5": r["top5"]}

        if nn_focal_model:
            r = nn_result(nn_focal_model, X, "MLP_FocalLoss")
            results["neural_network_focal"] = {"prediction": r["prediction"], "confidence": r["confidence"], "top5": r["top5"]}

        if ENSEMBLE_READY:
            r = ensemble_predict(X)
            results["stacking_ensemble"] = {
                "prediction": r["prediction"], "confidence": r["confidence"],
                "top5": r["top5"], "base_predictions": r["base_model_predictions"],
            }

        # Simple majority vote across all available predictions
        all_preds = [v["prediction"] for v in results.values()
                     if isinstance(v, dict) and "prediction" in v]
        if all_preds:
            from collections import Counter
            vote_result = Counter(all_preds).most_common(1)[0]
            results["majority_vote"] = {
                "prediction": vote_result[0],
                "vote_count": vote_result[1],
                "total_models": len(all_preds),
            }

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
