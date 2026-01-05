#!/usr/bin/env python3
# run_feature_build.py

import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from helpers import *
from paths_and_stuff import *

from new_anomaly_detection_preprocessing import (
    apply_basic_casts_inplace,
    one_hot_encode_inplace,
    clean_numeric_inplace,
    pre_model_prep,
)
from new_anomaly_detection_feature_engineering import (
    engineer_tx_features,
    compute_account_features,
    compute_uniques_and_hhi,
    attach_sender_receiver_features,
)

# ===========================
# Config
# ===========================
csv_sep = ";"
output_dir = create_new_folder(folderPath, datetime.now().strftime("%Y-%m-%d"))

# Label column to preserve for evaluation (NOT used in modeling features)
LABEL_COL = "Is Laundering"

start = time.time()

# ---------------------------
# Load
# ---------------------------
df = read_csv_custom(filePath, nrows=5000)
df = df.sample(n=500)

# ---------------------------
# Preserve label early (guarantee it won't disappear)
# ---------------------------
label_series = None
if LABEL_COL in df.columns:
    # Keep as 0/1 numeric if possible; fallback to 0 for missing
    label_series = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype("uint8")

# Basic casts / sanitation (in-place)
apply_basic_casts_inplace(df)

# ---------------------------
# Tx features (semantic)
# ---------------------------
tx = engineer_tx_features(df)

# preprocessing responsibility
one_hot_encode_inplace(tx, "Payment Format", "PF")
one_hot_encode_inplace(tx, "Payment Currency", "PC")

# ---------------------------
# Account features
# ---------------------------
acc = compute_account_features(df)
acc = acc.merge(compute_uniques_and_hhi(df), on="Account", how="left")

clean_numeric_inplace(acc)
clean_numeric_inplace(tx)

# ---------------------------
# Attach sender/receiver features
# ---------------------------
tx_model = attach_sender_receiver_features(tx, acc)

# Keep timestamp stable if present
if "Timestamp" in tx_model.columns:
    tx_model["Timestamp"] = (
        pd.to_datetime(tx_model["Timestamp"], errors="coerce")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

# ---------------------------
# Re-attach label to tx_model if it isn't present
# ---------------------------
if label_series is not None and LABEL_COL not in tx_model.columns:
    # relies on row-order consistency (works if engineer_tx_features keeps same row order)
    tx_model[LABEL_COL] = label_series.values

# ---------------------------
# Pre-model matrices (exclude label to avoid leakage)
# ---------------------------
acc_for_model = acc.drop(columns=[LABEL_COL], errors="ignore")
tx_for_model = tx_model.drop(columns=[LABEL_COL], errors="ignore")

X_acc, acc_feat_names, _ = pre_model_prep(acc_for_model)
X_tx, tx_feat_names, _ = pre_model_prep(tx_for_model)

# ---------------------------
# Exports
# ---------------------------
out_dir = Path(output_dir)

# Raw-ish feature tables (label is preserved in tx_model if present)
acc.to_csv(out_dir / "account_features.csv", sep=csv_sep, index=False)
tx_model.to_csv(out_dir / "tx_with_sender_receiver_features.csv", sep=csv_sep, index=False)

# Pure model matrices
pd.DataFrame(X_acc, columns=acc_feat_names).to_csv(
    out_dir / "acc_pre_model.csv", sep=csv_sep, index=False
)

# TX pre-model export WITH label appended (but label was NOT used in pre_model_prep)
tx_pre_model_df = pd.DataFrame(X_tx, columns=tx_feat_names)

if LABEL_COL in tx_model.columns:
    tx_pre_model_df[LABEL_COL] = (
        pd.to_numeric(tx_model[LABEL_COL], errors="coerce").fillna(0).astype("uint8").values
    )
elif label_series is not None:
    tx_pre_model_df[LABEL_COL] = label_series.values

tx_pre_model_df.to_csv(
    out_dir / "tx_pre_model_with_account_context_pre_model.csv",
    sep=csv_sep,
    index=False,
)

print("\nExecution time:", time.time() - start, "seconds")
print("\nOutput dir:", out_dir)

print(f"\nHas '{LABEL_COL}' in tx_model?", LABEL_COL in tx_model.columns)
if LABEL_COL in tx_model.columns:
    print("\nLabel distribution (tx_model):")
    print(tx_model[LABEL_COL].value_counts(dropna=False).head())

print(f"\nHas '{LABEL_COL}' in tx_pre_model_with_account_context_pre_model.csv?",
      LABEL_COL in tx_pre_model_df.columns)
if LABEL_COL in tx_pre_model_df.columns:
    print("\nLabel distribution (tx_pre_model_df):")
    print(tx_pre_model_df[LABEL_COL].value_counts(dropna=False).head())
