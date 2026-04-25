"""
stream_demo.py — Live traffic simulation demo
==============================================
Streams rows from data/test.csv to the API one by one,
simulating real-time IoT network traffic classification.

Usage:
    python stream_demo.py                        # default: lgbm_smote, 0.3s delay
    python stream_demo.py --model lgbm           # pick model
    python stream_demo.py --delay 0.1            # faster
    python stream_demo.py --only-attacks         # skip BENIGN rows
    python stream_demo.py --n 200                # stop after 200 rows
    python stream_demo.py --show-drift           # print drift stats every 20 preds
"""

import sys
import argparse
import time
import json
import requests
import pandas as pd
import joblib
import os

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_BASE   = "http://127.0.0.1:8002"
TEST_CSV   = "data/test.csv"
MODELS_DIR = "models"
LABEL_COL  = "Label"

FEATURE_COLS = [
    "Header_Length", "Protocol_Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot_sum", "Min", "Max", "AVG", "Std", "Tot_size", "IAT", "Number", "Variance",
]

# Map test.csv column names → API field names (spaces/case differences)
COL_REMAP = {
    "Protocol Type": "Protocol_Type",
    "Tot sum"      : "Tot_sum",
    "Tot size"     : "Tot_size",
}

# ── COLORS ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def color_pred(pred, true_label, class_names, model):
    benign_idx = class_names.index("BENIGN") if "BENIGN" in class_names else -1
    is_attack  = (true_label != benign_idx)

    if model == "binary":
        true_binary = "BENIGN" if true_label == benign_idx else "ATTACK"
        correct = (pred == true_binary)
    else:
        correct = (pred == class_names[true_label]) if true_label >= 0 else None

    if correct is None:
        return f"{CYAN}{pred}{RESET}"
    elif correct and not is_attack:
        return f"{GREEN}{pred}{RESET}"
    elif correct and is_attack:
        return f"{RED}{pred}{RESET}"
    else:
        return f"{YELLOW}{pred} (wrong!){RESET}"

def print_drift(drift: dict):
    print(f"\n{'─'*60}")
    print(f"{BOLD}  DRIFT MONITOR{RESET}")
    ar = drift.get("alert_rate", {})
    cm = drift.get("class_mix", {})
    fd = drift.get("feature_drift", {})
    overall = drift.get("overall_status", "?")

    status_color = RED if overall == "ALERT" else GREEN
    print(f"  Overall : {status_color}{overall}{RESET}   (window={drift.get('window_size',0)} preds)")
    print(f"  Alert rate   : {ar.get('attacks_last_60s',0)} attacks in last 60s  "
          f"({ar.get('attack_rate_pct',0):.0f}%)  [{ar.get('status','?')}]")
    print(f"  Class mix    : {cm.get('status','?')}")
    for d in cm.get("drifted_classes", [])[:3]:
        print(f"    {d['class']:<35} recent={d['recent_pct']:.1f}%  "
              f"train={d['training_pct']:.1f}%  Δ={d['delta_pct']:+.1f}%")
    print(f"  Feature drift: {fd.get('status','?')}")
    for d in fd.get("drifted_features", [])[:3]:
        print(f"    {d['feature']:<35} z={d['z_score']:.1f}")
    print(f"{'─'*60}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="lgbm_smote_tuned",
                        choices=["lr","binary","rf","xgb","lgbm","lgbm_smote","lgbm_smote_tuned","nn","nn_focal","ensemble","all"])
    parser.add_argument("--delay",       type=float, default=0.3,  help="seconds between predictions")
    parser.add_argument("--n",           type=int,   default=500,  help="max rows to stream")
    parser.add_argument("--only-attacks",action="store_true",       help="skip BENIGN rows")
    parser.add_argument("--show-drift",  action="store_true",       help="print drift stats every 20 preds")
    parser.add_argument("--shuffle",     action="store_true",       help="shuffle test data first")
    parser.add_argument("--csv",         default=None,              help="custom CSV file (default: data/test.csv)")
    args = parser.parse_args()

    # ── Check API is up ───────────────────────────────────────────────────────
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        assert r.json()["status"] == "ok"
    except Exception:
        print(f"{RED}API is not running. Start it first:{RESET}")
        print("  venv\\Scripts\\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8002")
        return

    # ── Load class names ──────────────────────────────────────────────────────
    le          = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    class_names = le.classes_.tolist()
    benign_idx  = class_names.index("BENIGN") if "BENIGN" in class_names else -1

    # ── Load test data ────────────────────────────────────────────────────────
    csv_path = args.csv if args.csv else TEST_CSV
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COL_REMAP)

    # Handle string labels (e.g. "DDOS-SYN_FLOOD") vs integer labels
    if LABEL_COL in df.columns and df[LABEL_COL].dtype == object:
        label_to_idx = {name: i for i, name in enumerate(class_names)}
        df[LABEL_COL] = df[LABEL_COL].map(label_to_idx).fillna(-1).astype(int)

    if args.shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if args.only_attacks:
        df = df[df[LABEL_COL] != benign_idx].reset_index(drop=True)
        print(f"Filtered to attacks only: {len(df):,} rows")

    df = df.head(args.n)
    endpoint = f"{API_BASE}/predict/{args.model}"

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"  Model    : {args.model}")
    print(f"  Rows     : {len(df):,}")
    print(f"  Delay    : {args.delay}s")
    print(f"  Endpoint : {endpoint}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    print(f"  {'#':<5} {'True Label':<30} {'Predicted':<30} {'Conf':>6}  OK?")
    print(f"  {'-'*5} {'-'*30} {'-'*30} {'-'*6}  ---")

    correct = 0
    total   = 0

    for i, (_, row) in enumerate(df.iterrows()):
        # Build payload — only include columns the API knows about
        payload = {}
        for col in FEATURE_COLS:
            if col in row.index:
                payload[col] = float(row[col]) if pd.notna(row[col]) else 0.0
            else:
                payload[col] = 0.0

        true_idx   = int(row[LABEL_COL]) if LABEL_COL in row.index else -1
        true_label = class_names[true_idx] if 0 <= true_idx < len(class_names) else "?"

        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if args.model == "all":
                pred       = data.get("majority_vote", {}).get("prediction", "?")
                confidence = data.get("majority_vote", {}).get("vote_count", 0)
                conf_str   = f"{confidence} votes"
            else:
                pred       = data.get("prediction", "?")
                confidence = data.get("confidence", 0.0)
                conf_str   = f"{confidence:.0%}"

            # For binary model: any non-BENIGN true label counts as ATTACK
            if args.model == "binary":
                true_binary = "BENIGN" if true_label == "BENIGN" else "ATTACK"
                match = "OK" if pred == true_binary else "XX"
            else:
                match = "OK" if pred == true_label else "XX"
            if match == "OK":
                correct += 1
            total += 1

            pred_colored = color_pred(pred, true_idx, class_names, args.model)
            match_color  = GREEN if match == "✓" else YELLOW

            print(f"  {i+1:<5} {true_label:<30} {pred_colored:<39} {conf_str:>6}  "
                  f"{match_color}{match}{RESET}")

        except requests.exceptions.ConnectionError:
            print(f"{RED}  Connection lost — is the API still running?{RESET}")
            break
        except Exception as e:
            print(f"{YELLOW}  Row {i+1} error: {e}{RESET}")
            total += 1

        # Show drift stats periodically
        if args.show_drift and (i + 1) % 20 == 0:
            try:
                drift = requests.get(f"{API_BASE}/monitor/drift", timeout=3).json()
                print_drift(drift)
            except Exception:
                pass

        time.sleep(args.delay)

    # ── Final summary ─────────────────────────────────────────────────────────
    acc = correct / total if total > 0 else 0
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Rows processed : {total}")
    print(f"  Correct        : {correct}")
    print(f"  Accuracy       : {acc:.1%}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Final drift check
    try:
        drift = requests.get(f"{API_BASE}/monitor/drift", timeout=3).json()
        print_drift(drift)
    except Exception:
        pass

if __name__ == "__main__":
    main()
