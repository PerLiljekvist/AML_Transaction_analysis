# feature_engineering.py
import pandas as pd
import numpy as np
from scipy.stats import entropy

from new_anomaly_detection_preprocessing import _to_num, _ensure_columns


def shannon_entropy_binned(x, bins: int = 10) -> float:
    s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0

    try:
        cats = pd.qcut(s, q=min(bins, s.nunique()), duplicates="drop")
        probs = cats.value_counts(normalize=True)
    except Exception:
        hist, _ = np.histogram(s, bins=min(bins, len(s)))
        probs = hist[hist > 0] / hist.sum()

    return float(entropy(probs, base=2))


# ---------------------------
# 1) Row-level (transaction) features
# ---------------------------
def engineer_tx_features(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_columns(
        df,
        [
            "Timestamp",
            "From Bank",
            "To Bank",
            "Amount Paid",
            "Amount Received",
            "Payment Format",
            "Payment Currency",
            "Account",
            "Account.1",
        ],
    )

    d["From Bank"] = d["From Bank"].astype(str)
    d["To Bank"] = d["To Bank"].astype(str)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")

    amt_paid = _to_num(d["Amount Paid"])
    amt_rec = _to_num(d["Amount Received"])

    d["Same_Bank"] = (d["From Bank"] == d["To Bank"]).astype("Int8")
    d["Amount_Ratio"] = np.where(amt_rec > 0, amt_paid / amt_rec, np.nan)
    d["Is_Reinvestment"] = (
        d["Payment Format"]
        .astype(str)
        .str.contains("reinvest", case=False, na=False)
        .astype("Int8")
    )

    d["weekday_of_transaction"] = d["Timestamp"].dt.day_of_week
    d["hour_of_transaction"] = d["Timestamp"].dt.hour

    return d


# ---------------------------
# 2) Account-level aggregates
# ---------------------------
def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_columns(df, ["Account", "Account.1", "Amount Paid", "Amount Received"])
    d["Account"] = d["Account"].astype(str)
    d["Account.1"] = d["Account.1"].astype(str)

    out = d.groupby("Account", dropna=False).agg(
        total_out_tx=("Account.1", "count"),
        total_out_amt=("Amount Paid", lambda x: _to_num(x).sum()),
        max_out_amt=("Amount Paid", lambda x: _to_num(x).max()),
        entropy_out_amt=("Amount Paid", lambda x: shannon_entropy_binned(_to_num(x))),
    )

    inb = d.groupby("Account.1", dropna=False).agg(
        total_in_tx=("Account", "count"),
        total_in_amt=("Amount Received", lambda x: _to_num(x).sum()),
        max_in_amt=("Amount Received", lambda x: _to_num(x).max()),
        entropy_in_amt=("Amount Received", lambda x: shannon_entropy_binned(_to_num(x))),
    )
    inb.index.name = "Account"

    acc = out.join(inb, how="outer").reset_index()
    acc["net_flow_amt"] = acc["total_out_amt"].fillna(0) - acc["total_in_amt"].fillna(0)
    return acc


# ---------------------------
# 3) Uniques + HHI
# ---------------------------
def compute_uniques_and_hhi(df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_columns(df, ["Account", "Account.1"])
    d["Account"] = d["Account"].astype(str)
    d["Account.1"] = d["Account.1"].astype(str)

    unique_receivers = d.groupby("Account")["Account.1"].nunique().rename("unique_receivers")
    unique_senders = (
        d.groupby("Account.1")["Account"]
        .nunique()
        .rename("unique_senders")
        .reset_index()
        .rename(columns={"Account.1": "Account"})
        .set_index("Account")["unique_senders"]
    )

    pair_counts = d.groupby(["Account", "Account.1"]).size().reset_index(name="tx_count")

    hhi_out = pair_counts.groupby("Account")["tx_count"].apply(
        lambda x: ((x / x.sum()) ** 2).sum() if x.sum() > 0 else 0.0
    )
    hhi_out.name = "HHI_out"

    hhi_in = pair_counts.groupby("Account.1")["tx_count"].apply(
        lambda x: ((x / x.sum()) ** 2).sum() if x.sum() > 0 else 0.0
    )
    hhi_in = (
        hhi_in.reset_index()
        .rename(columns={"Account.1": "Account", "tx_count": "HHI_in"})
        .set_index("Account")["HHI_in"]
    )

    out_df = pd.concat([unique_receivers, unique_senders, hhi_out, hhi_in], axis=1).reset_index()
    out_df.fillna(0, inplace=True)
    return out_df


# ---------------------------
# 4) Attach sender/receiver features
# ---------------------------
def attach_sender_receiver_features(
    tx: pd.DataFrame,
    acc: pd.DataFrame,
    sender_suffix="_S",
    receiver_suffix="_R",
) -> pd.DataFrame:
    acc_S = acc.add_suffix(sender_suffix)
    acc_R = acc.add_suffix(receiver_suffix)

    t = tx.merge(acc_S, how="left", left_on="Account", right_on=f"Account{sender_suffix}")
    t = t.merge(acc_R, how="left", left_on="Account.1", right_on=f"Account{receiver_suffix}")
    return t
