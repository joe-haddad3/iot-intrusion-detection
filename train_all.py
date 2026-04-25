"""
train_all.py  --  Master training runner for IoT Intrusion Detection
====================================================================
Runs every training script in the correct order and prints a
final comparison table of all models' test macro-F1 scores.

Usage (from project root):
    python train_all.py              # run everything
    python train_all.py --skip-slow  # skip stacking (3-4 hrs)
    python train_all.py --from lgbm  # restart from a specific step

Steps:
    1. preprocess        (skip if train.csv / test.csv already exist)
    2. logistic_regression
    3. train_binary
    4. randomforest
    5. xgboost_model
    6. neural_network
    7. lightgbm_model
    8. lgbm_smote
    9. neural_network_focal
   10. stacking_ensemble  (skipped with --skip-slow)
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime, timedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPTS = [
    {
        "key"    : "preprocess",
        "module" : "src/preprocess.py",
        "desc"   : "Preprocessing (clean, split, scale, save artifacts)",
        "slow"   : False,
        "skip_if": ["data/train.csv", "data/test.csv"],   # skip if these exist
    },
    {
        "key"    : "logistic_regression",
        "module" : "src/logistic_regression.py",
        "desc"   : "Logistic Regression baseline",
        "slow"   : False,
        "skip_if": [],
    },
    {
        "key"    : "train_binary",
        "module" : "src/train_binary.py",
        "desc"   : "Binary LR (BENIGN vs ATTACK)",
        "slow"   : False,
        "skip_if": [],
    },
    {
        "key"    : "randomforest",
        "module" : "src/randomforest.py",
        "desc"   : "Random Forest (balanced_subsample, 300 trees)",
        "slow"   : False,
        "skip_if": [],
    },
    {
        "key"    : "xgboost_model",
        "module" : "src/xgboost_model.py",
        "desc"   : "XGBoost (GPU, early stopping)",
        "slow"   : False,
        "skip_if": [],
    },
    {
        "key"    : "neural_network",
        "module" : "src/neural_network.py",
        "desc"   : "MLP Neural Network (weighted CrossEntropy, 60 epochs)",
        "slow"   : False,
        "skip_if": [],
    },
    {
        "key"    : "lightgbm_model",
        "module" : "src/lightgbm_model.py",
        "desc"   : "LightGBM (leaf-wise, GPU, grid search)",
        "slow"   : False,
        "skip_if": [],
    },
    {
        "key"    : "lgbm_smote",
        "module" : "src/lgbm_smote.py",
        "desc"   : "LightGBM + SMOTE (minority class oversampling)",
        "slow"   : False,
        "skip_if": [],
    },
    {
        "key"    : "neural_network_focal",
        "module" : "src/neural_network_focal.py",
        "desc"   : "MLP Neural Network (Focal Loss gamma=2)",
        "slow"   : False,
        "skip_if": [],
    },
    {
        "key"    : "stacking_ensemble",
        "module" : "src/stacking_ensemble.py",
        "desc"   : "Stacking Ensemble (RF+XGB+LGBM+NN -> LGBM meta, 3-fold OOF)",
        "slow"   : True,   # ~2-3 hours
        "skip_if": [],
    },
]

# Metrics JSON files produced by each script
METRICS_FILES = {
    "logistic_regression" : "reports/lr_metrics.json",
    "train_binary"        : "reports/binary_lr_metrics.json",
    "randomforest"        : "reports/rf_metrics.json",
    "xgboost_model"       : "reports/xgb_metrics.json",
    "neural_network"      : "reports/nn_metrics.json",
    "lightgbm_model"      : "reports/lgbm_metrics.json",
    "lgbm_smote"          : "reports/lgbm_smote_metrics.json",
    "neural_network_focal": "reports/nn_focal_metrics.json",
    "stacking_ensemble"   : "reports/ensemble_metrics.json",
}

DISPLAY_NAMES = {
    "logistic_regression" : "Logistic Regression",
    "train_binary"        : "Binary LR",
    "randomforest"        : "Random Forest",
    "xgboost_model"       : "XGBoost",
    "neural_network"      : "NN (CE Loss)",
    "lightgbm_model"      : "LightGBM",
    "lgbm_smote"          : "LightGBM + SMOTE",
    "neural_network_focal": "NN (Focal Loss)",
    "stacking_ensemble"   : "Stacking Ensemble",
}

# ── CLI ARGS ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run all IoT IDS training scripts in order.")
parser.add_argument("--skip-slow", action="store_true",
                    help="Skip stacking_ensemble.py (~2-3 hours)")
parser.add_argument("--from", dest="start_from", default=None,
                    help="Skip all steps before this key (e.g. --from lgbm_smote)")
parser.add_argument("--only", default=None,
                    help="Run only this one step (e.g. --only stacking_ensemble)")
args = parser.parse_args()

# ── HELPERS ────────────────────────────────────────────────────────────────────
def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def banner(text: str, width: int = 62):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def read_macro_f1(key: str) -> float | None:
    path = METRICS_FILES.get(key)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        return m.get("test_macro_f1") or m.get("val_macro_f1")
    except Exception:
        return None

def run_script(script: dict, python: str) -> tuple[bool, float]:
    """Run a script. Returns (success, elapsed_seconds)."""
    module = script["module"]
    desc   = script["desc"]

    # Check skip_if condition
    skip_if = script.get("skip_if", [])
    if skip_if and all(os.path.exists(p) for p in skip_if):
        print(f"  [SKIP] {desc}")
        print(f"         ({', '.join(skip_if)} already exist)")
        return True, 0.0

    print(f"\n  Running: {desc}")
    print(f"  Command: python {module}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 62, flush=True)

    t0 = time.time()
    result = subprocess.run(
        [python, module],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] Finished in {fmt_duration(elapsed)}")
        return True, elapsed
    else:
        print(f"\n  [FAIL] Exit code {result.returncode}  ({fmt_duration(elapsed)})")
        return False, elapsed

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    python = sys.executable

    banner(f"IoT IDS — Full Training Pipeline  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Python : {python}")
    print(f"  CWD    : {os.getcwd()}")
    if args.skip_slow:
        print("  Mode   : --skip-slow (stacking_ensemble will be skipped)")
    if args.start_from:
        print(f"  Mode   : --from {args.start_from} (earlier steps skipped)")
    if args.only:
        print(f"  Mode   : --only {args.only}")

    results  = {}   # key -> (success, elapsed)
    reached  = args.start_from is None  # if --from given, wait until we see that key

    global_start = time.time()

    for script in SCRIPTS:
        key = script["key"]

        # --only mode: run a single step
        if args.only and key != args.only:
            continue

        # --from mode: skip until we reach the start key
        if not reached:
            if key == args.start_from:
                reached = True
            else:
                print(f"  [SKIP] {script['desc']}  (before --from {args.start_from})")
                continue

        # --skip-slow mode
        if args.skip_slow and script["slow"]:
            print(f"\n  [SKIP] {script['desc']}  (--skip-slow)")
            results[key] = ("skipped", 0.0)
            continue

        banner(f"Step: {script['desc']}")
        ok, elapsed = run_script(script, python)
        results[key] = (ok, elapsed)

        if not ok:
            print(f"\n  !! Script {key} FAILED — continuing with next step.")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - global_start

    banner("FINAL SUMMARY", width=70)
    print(f"\n  {'Model':<28} {'Macro F1':>9}  {'Status':<10}  {'Time':>8}")
    print(f"  {'-'*28}  {'-'*9}  {'-'*10}  {'-'*8}")

    best_f1   = -1.0
    best_name = ""

    for script in SCRIPTS:
        key = script["key"]
        if key == "preprocess" or key not in METRICS_FILES:
            continue
        if key not in results:
            continue

        status_flag, elapsed = results[key]
        if status_flag == "skipped":
            status = "SKIPPED"
            f1_str = "       -"
        elif not status_flag:
            status = "FAILED"
            f1_str = "       -"
        else:
            status = "OK"
            f1 = read_macro_f1(key)
            if f1 is not None:
                f1_str = f"{f1:.4f}"
                if f1 > best_f1 and key != "train_binary":
                    best_f1   = f1
                    best_name = DISPLAY_NAMES.get(key, key)
            else:
                f1_str = "  n/a  "

        name = DISPLAY_NAMES.get(key, key)
        time_str = fmt_duration(elapsed) if elapsed > 0 else "-"
        print(f"  {name:<28} {f1_str:>9}  {status:<10}  {time_str:>8}")

    print(f"\n  Total wall time : {fmt_duration(total_elapsed)}")
    if best_f1 > 0:
        print(f"  Best model      : {best_name}  (macro F1 = {best_f1:.4f})")

    # Dump summary JSON
    summary = {
        "run_date"   : datetime.now().isoformat(),
        "total_time_s": round(total_elapsed, 1),
        "results": {
            k: {
                "success": v[0],
                "elapsed_s": round(v[1], 1),
                "macro_f1": read_macro_f1(k),
            }
            for k, v in results.items()
        },
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n  Full summary saved -> reports/training_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
