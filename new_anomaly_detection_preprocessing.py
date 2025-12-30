# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _ensure_columns(df: pd.DataFrame, cols):
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d


def apply_basic_casts_inplace(df: pd.DataFrame) -> None:
    for amount_col in ["Amount Paid", "Amount Received"]:
        if amount_col in df.columns:
            df[amount_col] = _to_num(df[amount_col])

    for col in [
        "Account",
        "Account.1",
        "From Bank",
        "To Bank",
        "Payment Format",
        "Payment Currency",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)


def one_hot_encode_inplace(
    df: pd.DataFrame,
    col: str,
    prefix: str,
) -> None:
    if col not in df.columns:
        return

    cats = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unknown"})
    )
    dummies = pd.get_dummies(cats, prefix=prefix, dtype="uint8")
    df.drop(columns=[col], inplace=True)
    df[dummies.columns] = dummies


def clean_numeric_inplace(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)


def pre_model_prep(
    df: pd.DataFrame,
    id_like=("Account", "Account.1", "From Bank", "To Bank", "Timestamp"),
    nan_flag_cols=("entropy_out_amt", "entropy_in_amt", "HHI_out", "HHI_in", "unique_receivers", "unique_senders"),
    impute_zero_if=("entropy", "HHI"),
    log1p_cols=("total_out_amt", "total_in_amt", "max_out_amt", "max_in_amt", "net_flow_amt", "Amount Paid", "Amount Received"),
):
    d = df.copy()

    num_cols = [
        c for c in d.columns
        if c not in id_like and pd.api.types.is_numeric_dtype(d[c])
    ]
    X = d[num_cols].copy()

    for c in nan_flag_cols:
        if c in X.columns:
            X[f"{c}_missing"] = X[c].isna().astype("uint8")

    for c in log1p_cols:
        if c in X.columns:
            vals = X[c].replace([np.inf, -np.inf], np.nan).clip(lower=0)
            X[c] = np.log1p(vals)

    for c in X.columns:
        col = X[c].replace([np.inf, -np.inf], np.nan)
        if col.isna().any():
            if any(k in c for k in impute_zero_if):
                col = col.fillna(0)
            else:
                col = col.fillna(col.median() if not pd.isna(col.median()) else 0)
        X[c] = col

    X = X.fillna(0)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, list(X.columns), scaler
