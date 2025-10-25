# ===========================
# AML Tx Feature Engineering - Lean Ad-hoc Version
# ===========================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from helpers import *
from paths_and_stuff import *

# ---------------------------
# Config (change these two)
# ---------------------------
#filePath   = filePath  # <- set me
csv_sep    = ";"                               # ";" for your sample, change if needed
output_dir = create_new_folder(folderPath, datetime.now().strftime("%Y-%m-%d"))    

# ---------------------------
# Utilities
# ---------------------------
def _to_num(s): 
    return pd.to_numeric(s, errors="coerce")

def _ensure_columns(df: pd.DataFrame, cols):
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d

def _reorder_with_original_first(original_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    orig_cols = [c for c in original_df.columns if c in enriched_df.columns]
    new_cols  = [c for c in enriched_df.columns if c not in orig_cols]
    return enriched_df[orig_cols + new_cols]

# ---------------------------
# 1) Row-level (transaction) features
# ---------------------------
def engineer_tx_features(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_columns(df, ["From Bank", "To Bank", "Amount Paid", "Amount Received", "Payment Format"])

    # Safe casts
    d["From Bank"]      = d["From Bank"].astype(str)
    d["To Bank"]        = d["To Bank"].astype(str)
    amt_paid            = _to_num(d["Amount Paid"])
    amt_rec             = _to_num(d["Amount Received"])

    # Features
    d["Same_Bank"]      = (d["From Bank"] == d["To Bank"]).astype("Int8")
    d["Amount_Diff"]    = amt_paid - amt_rec
    d["Amount_Ratio"]   = np.where(amt_rec > 0, amt_paid / amt_rec, np.nan)
    d["Is_Reinvestment"] = d["Payment Format"].astype(str).str.contains("reinvest", case=False, na=False).astype("Int8")
     
    # Add one-hot encoded feature for payment format
    cats = (
        d["Payment Format"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unknown"})
    )
    
    pf_dummies = pd.get_dummies(cats, prefix="PF", dtype="uint8")
    d = pd.concat([d, pf_dummies], axis=1)
    d = d.drop(columns=["Payment Format"])

     # Add one-hot encoded feature for payment currency
    cats = (
        d["Payment Currency"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unknown"})
    )

    pf_dummies = pd.get_dummies(cats, prefix="PC", dtype="uint8")
    d = pd.concat([d, pf_dummies], axis=1)
    d = d.drop(columns=["Payment Currency"])

    return d

# ---------------------------
# 2) Account-level aggregates (sender+receiver)
# ---------------------------
def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_columns(df, ["Account", "Account.1", "Amount Paid", "Amount Received"])
    d["Account"]   = d["Account"].astype(str)
    d["Account.1"] = d["Account.1"].astype(str)

    out = d.groupby("Account", dropna=False).agg(
        total_out_tx=("Account.1", "count"),
        total_out_amt=("Amount Paid", lambda x: _to_num(x).sum()),
        avg_out_amt=("Amount Paid", lambda x: _to_num(x).mean()),
        max_out_amt=("Amount Paid", lambda x: _to_num(x).max()),
        min_out_amt=("Amount Paid", lambda x: _to_num(x).min()),
    )

    inb = d.groupby("Account.1", dropna=False).agg(
        total_in_tx=("Account", "count"),
        total_in_amt=("Amount Received", lambda x: _to_num(x).sum()),
        avg_in_amt=("Amount Received", lambda x: _to_num(x).mean()),
        max_in_amt=("Amount Received", lambda x: _to_num(x).max()),
        min_in_amt=("Amount Received", lambda x: _to_num(x).min()),
    )
    inb.index.name = "Account"

    acc = out.join(inb, how="outer").reset_index()
    acc["net_flow_amt"] = (acc["total_out_amt"].fillna(0) - acc["total_in_amt"].fillna(0))
    return acc

# ---------------------------
# 3) Uniques + HHI (per account)
#     HHI = sum_i (share_i^2) over counterparties
# ---------------------------
def compute_uniques_and_hhi(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_columns(df, ["Account", "Account.1"])
    d["Account"]   = d["Account"].astype(str)
    d["Account.1"] = d["Account.1"].astype(str)

    unique_receivers = d.groupby("Account", dropna=False)["Account.1"].nunique().rename("unique_receivers")
    unique_senders_r = d.groupby("Account.1", dropna=False)["Account"].nunique().rename("unique_senders")
    unique_senders   = unique_senders_r.reset_index().rename(columns={"Account.1": "Account"}).set_index("Account")["unique_senders"]

    pair_counts = d.groupby(["Account", "Account.1"], dropna=False).size().reset_index(name="tx_count")

    hhi_out = (
        pair_counts.groupby("Account")["tx_count"]
        .apply(lambda x: ((x / x.sum()) ** 2).sum() if x.sum() > 0 else np.nan)
        .rename("HHI_out")
    )
    hhi_in_raw = (
        pair_counts.groupby("Account.1")["tx_count"]
        .apply(lambda x: ((x / x.sum()) ** 2).sum() if x.sum() > 0 else np.nan)
        .rename("HHI_in")
    )
    hhi_in = hhi_in_raw.reset_index().rename(columns={"Account.1": "Account"}).set_index("Account")["HHI_in"]

    out = pd.concat([unique_receivers, unique_senders, hhi_out, hhi_in], axis=1).reset_index()
    return out

# ---------------------------
# 4) Attach sender/receiver account features to each transaction
# ---------------------------
def attach_sender_receiver_features(tx: pd.DataFrame,
                                   acc: pd.DataFrame,
                                   sender_suffix="_S",
                                   receiver_suffix="_R") -> pd.DataFrame:
    t = tx.copy()
    a = acc.copy()

    acc_S = a.copy()
    acc_S.columns = [f"{c}{sender_suffix}" for c in acc_S.columns]

    acc_R = a.copy()
    acc_R.columns = [f"{c}{receiver_suffix}" for c in acc_R.columns]

    t = t.merge(acc_S, how="left", left_on="Account",   right_on=f"Account{sender_suffix}")
    t = t.merge(acc_R, how="left", left_on="Account.1", right_on=f"Account{receiver_suffix}")
    return t

# ===========================
# MAIN
# ===========================
# Load (semicolon default; change `csv_sep` above if needed)

df = read_csv_custom(filePath, nrows=100000)

# Safe numeric casts for amounts (keep original text columns too if you want)s
for amount_col in ["Amount Paid", "Amount Received"]:
    if amount_col in df.columns:
        df[amount_col] = _to_num(df[amount_col])

# Ensure ID-like columns are stringss
for col in ["Account", "Account.1", "From Bank", "To Bank", "Payment Format"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# 1) Tx-level features
tx = engineer_tx_features(df)

# 2) Account aggregates
#acc = compute_account_features(df)

# 3) Uniques + HHI, merged into acc
#uniq_hhi = compute_uniques_and_hhi(df)
#acc = acc.merge(uniq_hhi, on="Account", how="left")

# 4) Tx table with sender/receiver aggregates
#tx_model = attach_sender_receiver_features(tx, acc, sender_suffix="_S", receiver_suffix="_R")

# Normalize timestamp if present (string format for easy CSV use)
#if "Timestamp" in tx_model.columns:
    #tx_model["Timestamp"] = pd.to_datetime(tx_model["Timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

# 5) Save outputs
today = datetime.now().strftime("%Y-%m-%d")
# out_dir = Path(output_dir) / today
# out_dir.mkdir(parents=True, exist_ok=True)
out_dir = Path(output_dir)

# a) Row-level features only
tx_out = _reorder_with_original_first(df, tx) if set(df.columns).issubset(tx.columns) else tx.copy()
tx_out.to_csv(out_dir / f"tx_features_only_{today}.csv", sep=csv_sep, index=False)

# b) Per-account aggregates (with uniques+HHI)
#acc.to_csv(out_dir / f"account_features_{today}.csv", sep=csv_sep, index=False)

# c) Uniques + HHI standalone (optional but handy)
#uniq_hhi.to_csv(out_dir / f"account_uniques_hhi_{today}.csv", sep=csv_sep, index=False)

# d) Full modeling table
#tx_model_out = _reorder_with_original_first(df, tx_model) if set(df.columns).issubset(tx_model.columns) else tx_model.copy()
#tx_model_out.to_csv(out_dir / f"tx_model_with_sender_receiver_features_{today}.csv", sep=csv_sep, index=False)

print("\n✅ Feature export completed. Files saved to:")
print(f"- {out_dir / f'tx_features_only_{today}.csv'}")
#print(f"- {out_dir / f'account_features_{today}.csv'}")
#print(f"- {out_dir / f'account_uniques_hhi_{today}.csv'}")
#print(f"- {out_dir / f'tx_model_with_sender_receiver_features_{today}.csv'}")
