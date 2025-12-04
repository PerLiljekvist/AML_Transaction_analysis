# ===========================
# AML Tx Feature Engineering - Lean Ad-hoc Version
# ===========================
import pandas as pd 
import numpy as np
from pathlib import Path
from datetime import datetime
from helpers import *
from paths_and_stuff import *
from scipy.stats import entropy
import time 
from sklearn.preprocessing import RobustScaler

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

def shannon_entropy_binned(x, bins=10):
    """
    Shannon entropy for numeric series using binning (robust for floats/heavy tails).
    Returns 0.0 if too few values (no uncertainty / no diversification).
    """
    s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        # Previously np.nan -> now 0.0 to avoid NaNs downstream
        return 0.0

    # Use quantile bins to handle heavy-tailed AML amounts
    try:
        cats = pd.qcut(s, q=min(bins, s.nunique()), duplicates="drop")
        probs = cats.value_counts(normalize=True)
    except Exception:
        # Fallback to equal-width bins if qcut fails
        hist, _ = np.histogram(s, bins=min(bins, len(s)))
        probs = hist[hist > 0] / hist.sum()

    return entropy(probs, base=2)

def _ensure_columns(df: pd.DataFrame, cols):
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d

# ---------------------------
# PRE-MODEL PREP (lean, reusable)
# ---------------------------
def pre_model_prep(df: pd.DataFrame,
                   id_like=("Account", "Account.1", "From Bank", "To Bank", "Timestamp"),
                   nan_flag_cols=("entropy_out_amt", "entropy_in_amt", "HHI_out", "HHI_in",
                                  "unique_receivers", "unique_senders"),
                   impute_zero_if=("entropy", "HHI"),
                   log1p_cols=("total_out_amt", "total_in_amt",
                               "max_out_amt", "max_in_amt",
                               "net_flow_amt", "Amount Paid", "Amount Received")):
    """
    Minimal pre-model prep:
    - selects numeric features
    - adds NaN-missingness flags for key sparse features
    - imputes NaNs (0 for entropy/HHI; median otherwise, fallback 0 if median is NaN)
    - replaces any ±inf
    - log1p on heavy-tailed amount columns if present
    - robust scales everything (good for power-law / heavy tails)

    Returns:
        X_scaled (np.ndarray), feature_names (list[str]), scaler (fitted RobustScaler)
    """
    d = df.copy()

    # 1) numeric-only feature matrix (exclude id-like columns)
    num_cols = [
        c for c in d.columns
        if c not in id_like and pd.api.types.is_numeric_dtype(d[c])
    ]
    X = d[num_cols].copy()

    # 2) missingness indicator flags for structurally sparse features
    for c in nan_flag_cols:
        if c in X.columns:
            X[f"{c}_missing"] = X[c].isna().astype("uint8")

    # 3) log1p on heavy-tailed amount-ish columns (only if exists)
    for c in log1p_cols:
        if c in X.columns:
            # ensure no negative values before log, and guard against inf
            vals = X[c].copy()
            vals = vals.replace([np.inf, -np.inf], np.nan)
            vals = vals.clip(lower=0)
            X[c] = np.log1p(vals)

    # 4) handle NaNs and ±inf column by column
    for c in X.columns:
        col = X[c]

        # replace infinities with NaN first
        col = col.replace([np.inf, -np.inf], np.nan)

        if col.isna().any():
            if any(key in c for key in impute_zero_if):
                # entropy / HHI-type: use 0 as natural "no information" baseline
                col = col.fillna(0)
            else:
                med = col.median()
                # if median itself is NaN (e.g. column entirely NaN), fallback to 0
                if pd.isna(med):
                    med = 0
                col = col.fillna(med)

        X[c] = col

    # 5) final safety pass: no NaNs or inf left
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 6) robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, list(X.columns), scaler


# ---------------------------
# 1) Row-level (transaction) features
# ---------------------------
def engineer_tx_features(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_columns(df, ["Timestamp", "From Bank", "To Bank",
                             "Amount Paid", "Amount Received", "Payment Format"])

    # Safe casts
    d["From Bank"]      = d["From Bank"].astype(str)
    d["To Bank"]        = d["To Bank"].astype(str)
    d["Timestamp"]      = pd.to_datetime(df['Timestamp'], errors='coerce')
    amt_paid            = _to_num(d["Amount Paid"])
    amt_rec             = _to_num(d["Amount Received"])

    # Core tx-level features (compact but expressive)
    d["Same_Bank"]       = (d["From Bank"] == d["To Bank"]).astype("Int8")
    # Drop Amount_Diff (linear combo of Paid/Received)
    d["Amount_Ratio"]    = np.where(amt_rec > 0, amt_paid / amt_rec, np.nan)
    d["Is_Reinvestment"] = d["Payment Format"].astype(str).str.contains(
        "reinvest", case=False, na=False
    ).astype("Int8")
     
    # One-hot encoded feature for payment format
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

    # One-hot encoded feature for payment currency
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

    # Rolling window entropy feature (kept commented, not used in compact set)
    # d["amnt_paid_entropy_7"] = d["Amount Paid"].rolling(window=14).apply(shannon_entropy, raw=False)

    # Weekday & hour of transaction
    d["weekday_of_transaction"] = d["Timestamp"].dt.day_of_week
    d["hour_of_transaction"]    = d["Timestamp"].dt.hour

    return d    

# ---------------------------
# 2) Account-level aggregates (sender+receiver)
# ---------------------------
def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_columns(df, ["Account", "Account.1", "Amount Paid", "Amount Received"])
    d["Account"]   = d["Account"].astype(str)
    d["Account.1"] = d["Account.1"].astype(str)

    # Compact but informative outbound profile
    out = d.groupby("Account", dropna=False).agg(
        total_out_tx=("Account.1", "count"),
        total_out_amt=("Amount Paid", lambda x: _to_num(x).sum()),
        max_out_amt=("Amount Paid", lambda x: _to_num(x).max()),
        # Entropy = diversification / unpredictability of outbound amounts
        entropy_out_amt=("Amount Paid", lambda x: shannon_entropy_binned(_to_num(x))),
    )

    # Compact inbound profile
    inb = d.groupby("Account.1", dropna=False).agg(
        total_in_tx=("Account", "count"),
        total_in_amt=("Amount Received", lambda x: _to_num(x).sum()),
        max_in_amt=("Amount Received", lambda x: _to_num(x).max()),
        entropy_in_amt=("Amount Received", lambda x: shannon_entropy_binned(_to_num(x))),
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
    unique_senders   = unique_senders_r.reset_index().rename(
        columns={"Account.1": "Account"}
    ).set_index("Account")["unique_senders"]

    pair_counts = d.groupby(["Account", "Account.1"], dropna=False).size().reset_index(name="tx_count")

    hhi_out = (
        pair_counts.groupby("Account")["tx_count"]
        .apply(lambda x: ((x / x.sum()) ** 2).sum() if x.sum() > 0 else 0.0)
        .rename("HHI_out")
    )
    hhi_in_raw = (
        pair_counts.groupby("Account.1")["tx_count"]
        .apply(lambda x: ((x / x.sum()) ** 2).sum() if x.sum() > 0 else 0.0)
        .rename("HHI_in")
    )
    hhi_in = hhi_in_raw.reset_index().rename(
        columns={"Account.1": "Account"}
    ).set_index("Account")["HHI_in"]

    out = pd.concat([unique_receivers, unique_senders, hhi_out, hhi_in], axis=1).reset_index()

    # Ensure no NaNs left in these numeric HHI / unique columns
    for c in ["unique_receivers", "unique_senders", "HHI_out", "HHI_in"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)

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
start = time.time()

df = read_csv_custom(filePath, nrows=10000)
df = df.sample(n=1000)

# Safe numeric casts for amounts
for amount_col in ["Amount Paid", "Amount Received"]:
    if amount_col in df.columns:
        df[amount_col] = _to_num(df[amount_col])

# Ensure ID-like columns are strings
for col in ["Account", "Account.1", "From Bank", "To Bank", "Payment Format"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# 1) Tx-level features
tx = engineer_tx_features(df)

# 2) Account aggregates (with entropy)
acc = compute_account_features(df)

# 3) Uniques + HHI, merged into acc
uniq_hhi = compute_uniques_and_hhi(df)
acc = acc.merge(uniq_hhi, on="Account", how="left")

# 3b) Clean numeric columns in acc and tx (no NaN / inf in saved CSVs)
for _df in (acc, tx):
    num_cols = _df.select_dtypes(include=[np.number]).columns
    _df[num_cols] = _df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# 4) Tx table with sender/receiver aggregates (modeling base)
tx_model = attach_sender_receiver_features(tx, acc, sender_suffix="_S", receiver_suffix="_R")

# Normalize timestamp in tx_model for readability (optional; not used in pre-model)
if "Timestamp" in tx_model.columns:
    tx_model["Timestamp"] = pd.to_datetime(
        tx_model["Timestamp"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# Pre-model matrices (for anomaly detection codebase)
# ---------------------------
X_acc, acc_feat_names, acc_scaler = pre_model_prep(acc)
X_tx_model, tx_model_feat_names, tx_model_scaler = pre_model_prep(tx_model)

# 5) Save outputs (only modeling-relevant files)
today = datetime.now().strftime("%Y-%m-%d")
out_dir = Path(output_dir)

# a) Account-level feature table (compact, semantic)
acc.to_csv(out_dir / f"account_features_{today}.csv", sep=csv_sep, index=False)

# b) Tx-level + sender/receiver account features (semantic)
tx_model.to_csv(out_dir / f"tx_model_with_sender_receiver_features_{today}.csv",
                sep=csv_sep, index=False)

# c) Pre-model matrices (scaled, numeric-only, ready for anomaly models)
pd.DataFrame(X_acc, columns=acc_feat_names).to_csv(
    out_dir / f"acc_pre_model_{today}.csv", sep=csv_sep, index=False
)
pd.DataFrame(X_tx_model, columns=tx_model_feat_names).to_csv(
    out_dir / f"tx_model_pre_model_{today}.csv", sep=csv_sep, index=False
)

end = time.time()
length = end - start
print("Execution time:", length, "seconds!" )

print("\n✅ Export completed. Files saved to:")
print(f"- {out_dir / f'account_features_{today}.csv'}")
print(f"- {out_dir / f'tx_model_with_sender_receiver_features_{today}.csv'}")
print(f"- {out_dir / f'acc_pre_model_{today}.csv'}")
print(f"- {out_dir / f'tx_model_pre_model_{today}.csv'}")
