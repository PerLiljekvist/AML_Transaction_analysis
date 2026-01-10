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

    # keep label numeric (0/1) if present
    if "Is Laundering" in df.columns:
        df["Is Laundering"] = _to_num(df["Is Laundering"]).fillna(0).astype("uint8")

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
    label_cols=("Is Laundering", "Is Laundry", "Is_Laundering", "Is_Laundry"),
    nan_flag_cols=("entropy_out_amt", "entropy_in_amt", "HHI_out", "HHI_in", "unique_receivers", "unique_senders"),
    impute_zero_if=("entropy", "HHI", "unique_"),
    log1p_cols=("total_out_amt", "total_in_amt", "max_out_amt", "max_in_amt", "net_flow_amt", "Amount Paid", "Amount Received"),
    robust_scale=True,
    ensure_time_features=True,
):
    """
    Build a numeric model matrix X from df.

    - Optionally ensures weekday_of_transaction and hour_of_transaction exist (derived from Timestamp).
    - Excludes id-like + label columns from modeling.
    - Adds missingness flags for selected columns.
    - Applies log1p (signed log for net_flow).
    - Imputes remaining NaNs.
    - Drops constant columns EXCEPT the protected time features.
    - Optionally RobustScales.

    Returns:
      X_scaled (np.ndarray), feature_names (list[str]), scaler (RobustScaler|None)
    """
    d = df.copy()

    # ---------------------------
    # Ensure time-based features exist
    # ---------------------------
    if ensure_time_features and "Timestamp" in d.columns:
        ts = pd.to_datetime(d["Timestamp"], errors="coerce")
        if "weekday_of_transaction" not in d.columns:
            d["weekday_of_transaction"] = ts.dt.day_of_week
        if "hour_of_transaction" not in d.columns:
            d["hour_of_transaction"] = ts.dt.hour

    # Force time features to numeric if present (guards against accidental object dtype)
    for c in ("weekday_of_transaction", "hour_of_transaction"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # ---------------------------
    # Select numeric feature columns (exclude id-like + labels)
    # ---------------------------
    existing_label_cols = [c for c in label_cols if c in d.columns]
    exclude = set(id_like) | set(existing_label_cols)

    feat_cols = [
        c for c in d.columns
        if (c not in exclude) and pd.api.types.is_numeric_dtype(d[c])
    ]

    X = d[feat_cols].copy()

    # If nothing numeric, return empty matrix safely
    if X.shape[1] == 0:
        return np.zeros((len(d), 0), dtype=float), [], None

    # ---------------------------
    # Missingness flags
    # ---------------------------
    for c in nan_flag_cols:
        if c in X.columns:
            X[f"{c}_missing"] = X[c].isna().astype("uint8")

    # ---------------------------
    # Log transforms
    # ---------------------------
    for c in log1p_cols:
        if c in X.columns:
            vals = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

            if c == "net_flow_amt":
                v = vals.fillna(0)
                X[c] = np.sign(v) * np.log1p(np.abs(v))
            else:
                # amounts/features assumed non-negative; clip below 0
                X[c] = np.log1p(vals.clip(lower=0))

    # ---------------------------
    # Impute
    # ---------------------------
    for c in list(X.columns):
        col = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

        if col.isna().any():
            if any(k in c for k in impute_zero_if):
                col = col.fillna(0)
            else:
                med = col.median()
                col = col.fillna(med if not pd.isna(med) else 0)

        X[c] = col

    # Final safety
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------------------------
    # Drop constant columns (but protect the time features)
    # ---------------------------
    protected = {"weekday_of_transaction", "hour_of_transaction"}
    nunq = X.nunique(dropna=False)
    const_cols = [c for c in nunq.index if nunq[c] <= 1 and c not in protected]

    if const_cols:
        X = X.drop(columns=const_cols)

    # ---------------------------
    # Scale
    # ---------------------------
    scaler = None
    if robust_scale and X.shape[1] > 0:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.to_numpy(dtype=float, copy=False)

    return X_scaled, list(X.columns), scaler




