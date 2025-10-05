import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from paths_and_stuff import * 
from helpers import *

# ============================================================
# Inline feature engineering functions
# ============================================================

def engineer_tx_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction-level (row-level) derived features.
    Safe for partially-missing columns.
    """
    d = df.copy()

    # Ensure key columns exist (create safe defaults if missing)
    for col in ["From Bank", "To Bank", "Amount Paid", "Amount Received", "Payment Format"]:
        if col not in d.columns:
            d[col] = np.nan

    # Same-bank flag
    d["Same_Bank"] = (d["From Bank"].astype(str) == d["To Bank"].astype(str)).astype("Int8")

    # Amount differences / ratios
    d["Amount_Diff"] = pd.to_numeric(d["Amount Paid"], errors="coerce") - pd.to_numeric(d["Amount Received"], errors="coerce")
    amt_rec = pd.to_numeric(d["Amount Received"], errors="coerce")
    amt_paid = pd.to_numeric(d["Amount Paid"], errors="coerce")
    d["Amount_Ratio"] = np.where(amt_rec > 0, amt_paid / amt_rec, np.nan)

    # Simple format tag (example)
    d["Is_Reinvestment"] = d["Payment Format"].astype(str).str.contains("reinvest", case=False, na=False).astype("Int8")

    return d


def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate features per account (both outbound and inbound).
    Key = 'Account' (sender). Inbound is aligned to that key.
    """
    d = df.copy()

    # Ensure columns exist
    for col in ["Account", "Account.1", "Amount Paid", "Amount Received"]:
        if col not in d.columns:
            d[col] = np.nan

    # Outbound (sender side) aggregates: group by Account
    out = d.groupby("Account", dropna=False).agg(
        total_out_tx=("Account.1", "count"),
        total_out_amt=("Amount Paid", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        avg_out_amt=("Amount Paid", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        max_out_amt=("Amount Paid", lambda x: pd.to_numeric(x, errors="coerce").max()),
        min_out_amt=("Amount Paid", lambda x: pd.to_numeric(x, errors="coerce").min()),
    )

    # Inbound (receiver side) aggregates: group by Account.1, then rename key -> Account
    inb = d.groupby("Account.1", dropna=False).agg(
        total_in_tx=("Account", "count"),
        total_in_amt=("Amount Received", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        avg_in_amt=("Amount Received", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        max_in_amt=("Amount Received", lambda x: pd.to_numeric(x, errors="coerce").max()),
        min_in_amt=("Amount Received", lambda x: pd.to_numeric(x, errors="coerce").min()),
    )
    inb.index.name = "Account"
    acc = out.join(inb, how="outer").reset_index()

    # Net flow (out - in)
    acc["net_flow_amt"] = (acc["total_out_amt"].fillna(0) - acc["total_in_amt"].fillna(0))

    return acc


def compute_uniques_and_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute unique counterparties + HHI on both sides, aligned to key 'Account'.
    - unique_receivers: # distinct Account.1 per Account
    - unique_senders:   # distinct Account per Account.1 (aligned back to Account)
    - HHI_out: Herfindahl index of a sender's distribution across receivers
    - HHI_in:  Herfindahl index of a receiver's distribution across senders
    """
    d = df.copy()
    for col in ["Account", "Account.1"]:
        if col not in d.columns:
            d[col] = np.nan

    # Unique counts
    unique_receivers = d.groupby("Account", dropna=False)["Account.1"].nunique().rename("unique_receivers")
    unique_senders_raw = d.groupby("Account.1", dropna=False)["Account"].nunique().rename("unique_senders")
    unique_senders = unique_senders_raw.reset_index().rename(columns={"Account.1": "Account"}).set_index("Account")["unique_senders"]

    # Pair counts for HHI
    pair_counts = d.groupby(["Account", "Account.1"], dropna=False).size().reset_index(name="tx_count")

    # HHI outbound (per Account over receivers)
    hhi_out = (
        pair_counts.groupby("Account")["tx_count"]
        .apply(lambda x: ((x / x.sum()) ** 2).sum() if x.sum() > 0 else np.nan)
        .rename("HHI_out")
    )

    # HHI inbound (per Account.1 over senders) -> realign to key 'Account'
    hhi_in_raw = (
        pair_counts.groupby("Account.1")["tx_count"]
        .apply(lambda x: ((x / x.sum()) ** 2).sum() if x.sum() > 0 else np.nan)
        .rename("HHI_in")
    )
    hhi_in = hhi_in_raw.reset_index().rename(columns={"Account.1": "Account"}).set_index("Account")["HHI_in"]

    out = pd.concat([unique_receivers, unique_senders, hhi_out, hhi_in], axis=1).reset_index()
    return out


def attach_sender_receiver_features(tx: pd.DataFrame,
                                   acc: pd.DataFrame,
                                   sender_suffix="_S",
                                   receiver_suffix="_R") -> pd.DataFrame:
    """
    Attach sender (by Account) and receiver (by Account.1) aggregates to each transaction.
    """
    t = tx.copy()
    a = acc.copy()

    # Prepare suffixed copies
    acc_S = a.add_suffix(sender_suffix)  # keys under Account_S
    acc_R = a.add_suffix(receiver_suffix)  # keys under Account_R

    # Join sender features (Account -> Account_S)
    if "Account" in t.columns and f"Account{sender_suffix}" in acc_S.columns:
        t = t.merge(acc_S, how="left", left_on="Account", right_on=f"Account{sender_suffix}")
    else:
        # If key column missing in acc_S, rebuild it
        acc_S = a.copy()
        acc_S.columns = [f"{c}{sender_suffix}" for c in acc_S.columns]
        t = t.merge(acc_S, how="left", left_on="Account", right_on=f"Account{sender_suffix}")

    # Join receiver features (Account.1 -> Account_R)
    if "Account.1" in t.columns and f"Account{receiver_suffix}" in acc_R.columns:
        t = t.merge(acc_R, how="left", left_on="Account.1", right_on=f"Account{receiver_suffix}")
    else:
        acc_R = a.copy()
        acc_R.columns = [f"{c}{receiver_suffix}" for c in acc_R.columns]
        t = t.merge(acc_R, how="left", left_on="Account.1", right_on=f"Account{receiver_suffix}")

    return t


# ============================================================
# 1) Load data (semicolon-separated)
# ============================================================
df = read_csv_custom(filePath, nrows=5000)


# Cast IDs to string to avoid dtype surprises
for col in ["Account", "Account.1", "From Bank", "To Bank", "Payment Format"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# ============================================================
# 2) Row-level features
# ============================================================
tx = engineer_tx_features(df)

# ============================================================
# 3) Per-account aggregates
# ============================================================
acc = compute_account_features(df)

# ============================================================
# 4) Uniques + HHI, then merge into acc
# ============================================================
uniq_hhi = compute_uniques_and_hhi(df)
acc = acc.merge(uniq_hhi, on="Account", how="left")

# ============================================================
# 5) Transaction table augmented with sender/receiver aggregates
# ============================================================
tx_model = attach_sender_receiver_features(
    tx,
    acc,
    sender_suffix="_S",
    receiver_suffix="_R"
)

# Normalize timestamp format if present
if "Timestamp" in tx_model.columns:
    tx_model["Timestamp"] = pd.to_datetime(tx_model["Timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# 6) Save all outputs (date-stamped)
# ============================================================
today = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path(create_new_folder(folderPath, today))
#OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _reorder_with_original_first(original_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Put original columns first, then the new feature columns."""
    orig_cols = [c for c in original_df.columns if c in enriched_df.columns]
    new_cols = [c for c in enriched_df.columns if c not in orig_cols]
    return enriched_df[orig_cols + new_cols]

# a) Row-level features only
tx_out = _reorder_with_original_first(df, tx) if set(df.columns).issubset(tx.columns) else tx.copy()
tx_out.to_csv(OUTPUT_DIR / f"tx_features_only_{today}.csv", sep=";", index=False)

# b) Per-account aggregates
acc.to_csv(OUTPUT_DIR / f"account_features_{today}.csv", sep=";", index=False)

# c) Uniques + HHI standalone
uniq_hhi.to_csv(OUTPUT_DIR / f"account_uniques_hhi_{today}.csv", sep=";", index=False)

# d) Full modeling table (tx + sender/receiver features)
tx_model_out = _reorder_with_original_first(df, tx_model) if set(df.columns).issubset(tx_model.columns) else tx_model.copy()
tx_model_out.to_csv(OUTPUT_DIR / f"tx_model_with_sender_receiver_features_{today}.csv", sep=";", index=False)

print("\n✅ Feature export completed. Files saved to:")
print(f"- {OUTPUT_DIR / f'tx_features_only_{today}.csv'}")
print(f"- {OUTPUT_DIR / f'account_features_{today}.csv'}")
print(f"- {OUTPUT_DIR / f'account_uniques_hhi_{today}.csv'}")
print(f"- {OUTPUT_DIR / f'tx_model_with_sender_receiver_features_{today}.csv'}")

