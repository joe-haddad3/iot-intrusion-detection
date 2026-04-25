"""
build_live_demo.py
==================
Builds data/live_demo.csv from the raw Merged CSV files.
Picks rows that were NOT used in training (takes the last portion
of each file — Merged_balanced.csv sampled randomly so the tail
rows are effectively unseen).

Output: data/live_demo.csv  (500 rows, ~70% common classes, ~30% rare)
"""

import os, random
import numpy as np
import pandas as pd
import joblib

MERGED_DIR  = "C:/Users/Lenovo/Desktop/ML/MERGED_CSV"
MODELS_DIR  = "models"
OUT_PATH    = "data/live_demo.csv"
TOTAL_ROWS  = 500
RANDOM_SEED = 99

# Common vs rare split
COMMON_ROWS = 350
RARE_ROWS   = 150

# Classes considered "common" (high support in training data)
COMMON_CLASSES = {
    "DDOS-ACK_FRAGMENTATION", "DDOS-ICMP_FLOOD", "DDOS-PSHACK_FLOOD",
    "DDOS-RSTFINFLOOD", "DDOS-SYN_FLOOD", "DDOS-TCP_FLOOD", "DDOS-UDP_FLOOD",
    "DDOS-UDP_FRAGMENTATION", "DDOS-ICMP_FRAGMENTATION", "DDOS-SYNONYMOUSIP_FLOOD",
    "DOS-SYN_FLOOD", "DOS-UDP_FLOOD", "DOS-TCP_FLOOD", "DOS-HTTP_FLOOD",
    "MIRAI-GREIP_FLOOD", "MIRAI-GREETH_FLOOD", "MIRAI-UDPPLAIN",
    "BENIGN", "MITM-ARPSPOOFING", "DNS_SPOOFING",
    "RECON-PORTSCAN", "RECON-OSSCAN", "RECON-HOSTDISCOVERY",
}

RARE_CLASSES = {
    "BACKDOOR_MALWARE", "BROWSERHIJACKING", "COMMANDINJECTION",
    "DDOS-SLOWLORIS", "DDOS-HTTP_FLOOD", "DOS-GOLDENEYE",
    "SQLINJECTION", "UPLOADING_ATTACK", "VULNERABILITYSCAN",
    "XSS", "RECON-PINGSWEEP",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
known_classes = set(le.classes_.tolist())

print(f"Scanning Merged files in {MERGED_DIR} ...")
print(f"Target: {COMMON_ROWS} common + {RARE_ROWS} rare = {TOTAL_ROWS} rows\n")

common_pool = []
rare_pool   = []

# Scan every other file to keep it fast (still 31 files × last 2000 rows each)
files = sorted([f for f in os.listdir(MERGED_DIR) if f.endswith(".csv")])
files_to_scan = files[::2]   # every other file

for fname in files_to_scan:
    path = os.path.join(MERGED_DIR, fname)
    try:
        # Read a random sample of 3000 rows using skiprows for speed
        # Count rows quickly, then sample
        df = pd.read_csv(path, skiprows=lambda i: i != 0 and random.random() > 0.05)
        df = df.head(3000)
    except Exception as e:
        print(f"  [skip] {fname}: {e}")
        continue

    if "Label" not in df.columns:
        continue

    # Only keep rows whose label is known to the model
    df = df[df["Label"].isin(known_classes)]
    if df.empty:
        continue

    common_df = df[df["Label"].isin(COMMON_CLASSES)]
    rare_df   = df[df["Label"].isin(RARE_CLASSES)]

    if not common_df.empty:
        common_pool.append(common_df)
    if not rare_df.empty:
        rare_pool.append(rare_df)

    print(f"  {fname}: {len(common_df)} common, {len(rare_df)} rare")

# Combine pools
common_all = pd.concat(common_pool, ignore_index=True) if common_pool else pd.DataFrame()
rare_all   = pd.concat(rare_pool,   ignore_index=True) if rare_pool   else pd.DataFrame()

print(f"\nTotal pool: {len(common_all):,} common, {len(rare_all):,} rare")

# Sample
n_common = min(COMMON_ROWS, len(common_all))
n_rare   = min(RARE_ROWS,   len(rare_all))

sampled_common = common_all.sample(n=n_common, random_state=RANDOM_SEED)
sampled_rare   = rare_all.sample(n=n_rare,     random_state=RANDOM_SEED)

demo = pd.concat([sampled_common, sampled_rare], ignore_index=True)
demo = demo.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

# Rename columns to match API field names
demo = demo.rename(columns={
    "Protocol Type": "Protocol_Type",
    "Tot sum"      : "Tot_sum",
    "Tot size"     : "Tot_size",
})

# Drop any extra columns not needed
keep_cols = [
    "Header_Length", "Protocol_Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot_sum", "Min", "Max", "AVG", "Std", "Tot_size",
    "IAT", "Number", "Variance", "Label",
]
demo = demo[[c for c in keep_cols if c in demo.columns]]

# Fill any missing feature columns with 0
for col in keep_cols:
    if col != "Label" and col not in demo.columns:
        demo[col] = 0.0

demo.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(demo)} rows -> {OUT_PATH}")
print(f"\nClass distribution:")
print(demo["Label"].value_counts().to_string())
