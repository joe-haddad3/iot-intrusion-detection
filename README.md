# IoT Intrusion Detection System

A machine learning system for real-time detection of IoT network attacks using 34-class classification, binary BENIGN/ATTACK detection, drift monitoring, and a live REST API.

---

## Run with Docker (Recommended — no setup needed)

**Requires:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# Pull the image
docker pull joehaddad12/iot-ids

# Run the API
docker run -p 8002:8002 joehaddad12/iot-ids
```

Then open: **http://localhost:8002**

The dashboard UI loads automatically. All models are pre-loaded inside the image — no Python installation or training required.

---

## Project Structure

```
├── src/
│   ├── preprocess.py            # Data cleaning, scaling, train/test split
│   ├── logistic_regression.py   # Baseline LR model
│   ├── randomforest.py          # Random Forest model
│   ├── xgboost_model.py         # XGBoost model
│   ├── lightgbm_model.py        # LightGBM model
│   ├── lgbm_smote.py            # LightGBM + SMOTE (best multiclass)
│   ├── lgbm_smote_tuned.py      # Optuna-tuned LightGBM + SMOTE
│   ├── neural_network.py        # MLP Neural Network
│   ├── neural_network_focal.py  # MLP with Focal Loss
│   ├── stacking_ensemble.py     # Stacking ensemble
│   ├── train_binary.py          # Binary classifier (BENIGN vs ATTACK)
│   ├── save_drift_reference.py  # Computes drift reference stats
│   └── api.py                   # FastAPI inference server
├── stream_demo.py               # Live traffic simulation demo
├── build_live_demo.py           # Builds unseen demo data from raw CSVs
├── train_all.py                 # Runs all training scripts in order
├── reports/                     # Metrics JSON + plots (auto-generated)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- ~8 GB RAM minimum (16 GB recommended for training)
- GPU optional (LightGBM binary uses GPU if available, multiclass uses CPU)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/iot-ids.git
cd iot-ids
```

### 2. Create and activate virtual environment

```bash
# Windows
py -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU PyTorch, visit https://pytorch.org and install the version matching your CUDA.

---

## Data

Place the raw dataset CSVs in `data/raw/` before running preprocessing.  
The dataset used is the **CIC IoT Dataset 2023** (34 attack classes + BENIGN).

---

## Training

### Option A — Train everything at once

```bash
python train_all.py
```

This runs all scripts in order: preprocess → LR → binary → RF → XGBoost → NN → LightGBM → SMOTE → Focal NN → Stacking.

Skip the slow stacking ensemble (3-4 hours):
```bash
python train_all.py --skip-slow
```

### Option B — Train individually

```bash
# Step 1 — must run first
python src/preprocess.py

# Step 2 — individual models (any order after preprocess)
python src/logistic_regression.py
python src/train_binary.py
python src/randomforest.py
python src/xgboost_model.py
python src/lightgbm_model.py
python src/lgbm_smote.py
python src/lgbm_smote_tuned.py   # Optuna tuning (~40 min)
python src/neural_network.py
python src/neural_network_focal.py
python src/stacking_ensemble.py  # slow (~3-4 hrs)

# Step 3 — compute drift reference (required before starting API)
python src/save_drift_reference.py
```

All trained models are saved to `models/` and evaluation metrics/plots to `reports/`.

---

## Running the API

```bash
venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8002
```

Then open: [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)

### Available prediction endpoints

| Endpoint | Model |
|---|---|
| `POST /predict/binary` | LightGBM Binary (BENIGN vs ATTACK) |
| `POST /predict/lgbm_smote` | LightGBM + SMOTE (best multiclass) |
| `POST /predict/lgbm` | LightGBM |
| `POST /predict/xgb` | XGBoost |
| `POST /predict/rf` | Random Forest |
| `POST /predict/lr` | Logistic Regression |
| `POST /predict/nn` | Neural Network |
| `POST /predict/nn_focal` | Neural Network (Focal Loss) |
| `POST /predict/ensemble` | Stacking Ensemble |
| `POST /predict/all` | All models + majority vote |
| `GET  /monitor/drift` | Drift monitoring stats |
| `POST /monitor/reset` | Reset drift window |
| `GET  /health` | Health check |

### Example request

```bash
curl -X POST http://127.0.0.1:8002/predict/binary \
  -H "Content-Type: application/json" \
  -d '{"Header_Length": 20, "Protocol_Type": 6, "Time_To_Live": 64, "Rate": 100.0}'
```

---

## Live Stream Demo

Simulates real-time traffic by streaming rows to the API and printing predictions:

```bash
# Default (lgbm_smote model, 500 rows, 0.3s delay)
venv\Scripts\python.exe -X utf8 stream_demo.py

# Binary model with drift monitoring
venv\Scripts\python.exe -X utf8 stream_demo.py --model binary --show-drift

# Custom CSV, faster
venv\Scripts\python.exe -X utf8 stream_demo.py --csv data/live_demo.csv --delay 0.1

# Only show attack traffic
venv\Scripts\python.exe -X utf8 stream_demo.py --only-attacks --n 100
```

---

## Model Results Summary

| Model | Macro F1 | Accuracy |
|---|---|---|
| Logistic Regression | 0.4401 | 58.5% |
| Neural Network | 0.5616 | 71.1% |
| Random Forest | 0.5861 | 76.4% |
| XGBoost | 0.6105 | 76.5% |
| LightGBM | 0.6146 | 77.0% |
| LightGBM + SMOTE | 0.6466 | 79.7% |
| **LightGBM Binary** | **0.9725 F1** | **94.9%** |

Binary model metrics (BENIGN vs ATTACK):
- F1: 0.9725 — ROC-AUC: 0.9833
- False Positive Rate: 9.74% (BENIGN flagged as ATTACK)
- False Negative Rate: 4.82% (ATTACK missed)

---

## Drift Monitoring

The API tracks live predictions in a rolling window (500 predictions) and detects:
- **Alert rate** — spike in attack predictions in the last 60 seconds
- **Class mix drift** — shift in predicted class distribution vs training
- **Feature drift** — Z-score deviation of incoming features vs training baseline

Check drift status:
```
GET http://127.0.0.1:8002/monitor/drift
```
