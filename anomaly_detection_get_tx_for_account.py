import pandas as pd

def _normalize_id_series(s: pd.Series) -> pd.Series:
    """
    Normalize identifier-like strings safely:
    - force pandas 'string' dtype
    - remove zero-width & BOM chars
    - Unicode NFKC normalization
    - trim whitespace
    """
    s = s.astype("string")
    # remove zero-width chars (ZWSP/ZWNJ/ZWJ) and BOM if present
    s = s.str.replace(r'[\u200B-\u200D\uFEFF]', '', regex=True)
    # normalize and strip
    s = s.str.normalize('NFKC').str.strip()
    return s

def load_rows_for_account(
    filepath,
    account,
    source_col="Account",
    target_col="Account.1",
    timestamp_col="Timestamp",
    sep=";",
    chunksize=1000,
    encoding="utf-8"
):
    """
    Loads rows from a CSV where either:
      - `source_col` equals the target account (outbound), OR
      - `target_col` equals the target account (inbound).
    Also computes dataset-wide stats: min/max timestamp and total #rows.

    Returns
    -------
    filtered_df : pandas.DataFrame
        All rows where `account` appears as sender or receiver.
    stats : dict
        {
          "total_rows": int,
          "dataset_min_date": pandas.Timestamp or None,
          "dataset_max_date": pandas.Timestamp or None
        }
    """
    filtered_chunks = []

    total_rows = 0
    dataset_min_date = None
    dataset_max_date = None

    # Stream the file with robust options; keep IDs as strings
    for chunk in pd.read_csv(
        filepath,
        sep=sep,
        chunksize=chunksize,
        dtype=str,                 # keep everything as str on read
        engine="python",           # more forgiving for ragged/malformed lines
        on_bad_lines="warn",
        encoding=encoding,
        encoding_errors="replace"
    ):
        # --- Force + normalize ID columns safely per chunk ---
        if source_col in chunk.columns:
            chunk[source_col] = _normalize_id_series(chunk[source_col])
        else:
            # ensure column exists to avoid KeyErrors downstream
            chunk[source_col] = pd.Series(pd.array([None] * len(chunk), dtype="string"))

        if target_col in chunk.columns:
            chunk[target_col] = _normalize_id_series(chunk[target_col])
        else:
            chunk[target_col] = pd.Series(pd.array([None] * len(chunk), dtype="string"))

        # Count total rows (dataset-level)
        total_rows += len(chunk)

        # Update dataset min/max date if timestamp column exists
        if timestamp_col in chunk.columns:
            ts = pd.to_datetime(chunk[timestamp_col], errors="coerce", utc=False)
            if not ts.isna().all():
                cmin = ts.min()
                cmax = ts.max()
                if dataset_min_date is None or (pd.notna(cmin) and cmin < dataset_min_date):
                    dataset_min_date = cmin
                if dataset_max_date is None or (pd.notna(cmax) and cmax > dataset_max_date):
                    dataset_max_date = cmax

        # Keep rows where the (normalized) account appears in either column
        # Normalize the probe `account` once for fair comparison
        if isinstance(account, str):
            probe = _normalize_id_series(pd.Series([account])).iloc[0]
        else:
            probe = account  # if already normalized/None

        mask = (chunk[source_col] == probe) | (chunk[target_col] == probe)
        filtered = chunk[mask]
        if not filtered.empty:
            filtered_chunks.append(filtered)

    # Build filtered dataframe (even if empty)
    if filtered_chunks:
        filtered_df = pd.concat(filtered_chunks, ignore_index=True)
    else:
        # Preserve header if file has one
        try:
            cols = pd.read_csv(
                filepath, sep=sep, nrows=0,
                engine="python", on_bad_lines="warn",
                encoding=encoding, encoding_errors="replace"
            ).columns
        except Exception:
            cols = []
        filtered_df = pd.DataFrame(columns=cols)

    stats = {
        "total_rows": int(total_rows),
        "dataset_min_date": dataset_min_date,
        "dataset_max_date": dataset_max_date,
    }

    return filtered_df, stats
