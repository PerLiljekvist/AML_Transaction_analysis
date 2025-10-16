import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from typing import Tuple, Dict, Any, Optional
from typing import Optional
import numpy as np
from typing import Optional

def make_df_from_file(file_path, column_name):
    df = pd.read_csv(file_path, sep=';')
    df.set_index(list(df)[0])
    df[column_name] = df[column_name].astype(str).str.replace(' ', '')
    df[column_name] = df[column_name].astype(str).str.replace(',', '.')
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    return df    

    if n > max_lines:
        n = max_lines  # or raise ValueError("n exceeds maximum allowed")
    lines = []
    with open(file_path, 'r', encoding=encoding) as f:
        for i, line in enumerate(f):
            lines.append(line.rstrip('\n'))
            if i + 1 >= n:
                break
    return lines

def get_file_head_as_df(file_path, n=250, encoding='utf-8'):
    """
    Returns the first n lines of a file as a pandas DataFrame.
    Each line is a row in the column 'line'.
    """
    lines = []
    with open(file_path, 'r', encoding=encoding) as f:
        for i, line in enumerate(f):
            lines.append(line.rstrip('\n'))
            if i + 1 >= n:
                break
    print(pd.DataFrame({'line': lines}))

def read_csv_custom(
    filepath,
    nrows=None,
    sep=None,
    account_cols=("Account", "Account.1"),
    encoding="utf-8",
    keep_default_na=False,
):
    """
    Reads a CSV file into a pandas DataFrame with options:
    - nrows: number of rows to read (None means all)
    - sep: delimiter; if None, auto-detects from a small sample
    - Forces `account_cols` to string and normalizes/strips them
    - Converts decimals with ',' as decimal separator (for non-account cols)
    - Attempts to parse object columns to datetimes and format as 'YYYY-MM-DD HH:MM:SS'
    """
    # --- 1) Auto-detect separator if not provided ---
    if sep is None:
        try:
            with open(filepath, 'r', newline='', encoding=encoding, errors='replace') as f:
                sample = f.read(4096)
                sniffer = csv.Sniffer()
                sep = sniffer.sniff(sample).delimiter
        except Exception:
            sep = ';'  # fallback if sniffer fails

    # --- 2) Build dtype mapping to force account cols to string ---
    dtype_map = {col: "string" for col in account_cols}

    # --- 3) Read CSV robustly ---
    df = pd.read_csv(
        filepath,
        sep=sep,
        decimal=',',
        nrows=nrows,
        encoding=encoding,
        encoding_errors="replace",
        dtype=dtype_map,
        engine="python",
        on_bad_lines="warn",
        keep_default_na=keep_default_na
    )

    # Ensure missing account columns exist
    present_account_cols = [c for c in account_cols if c in df.columns]

    # --- 4) Normalize/clean account columns ---
    if present_account_cols:
        for col in present_account_cols:
            s = df[col].astype("string")
            s = s.str.replace(r'[\u200B-\u200D\uFEFF]', '', regex=True)  # remove zero-width chars
            s = s.str.normalize('NFKC').str.strip()
            df[col] = s

    # --- 5) Numeric conversion for NON-account object columns ---
    obj_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_candidates = [c for c in obj_cols if c not in present_account_cols]

    for col in numeric_candidates:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() == 0:
            cleaned = (
                df[col]
                .str.replace('\u202F', '', regex=False)
                .str.replace('\u00A0', '', regex=False)
                .str.replace(' ', '', regex=False)
            )
            converted = pd.to_numeric(cleaned, errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted

    # --- 6) Datetime parsing for NON-account textual columns ---
    obj_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    date_candidates = [c for c in obj_cols if c not in present_account_cols]

    for col in date_candidates:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
            if parsed.notna().sum() > 0 and parsed.notna().sum() >= 0.5 * len(df[col]):
                df[col] = parsed.dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass

    return df

def plot_group_distributions(grouped_df):
    """
    Plots histograms and boxplots for unique_recipients and total_amount distributions.
    
    Args:
        grouped_df (pd.DataFrame): Output dataframe from preprocess_and_group
    """
    plt.figure(figsize=(14, 10))
    
    # Unique Recipients plots
    plt.subplot(2, 2, 1)
    sns.histplot(grouped_df['unique_recipients'], kde=True, bins=30, color='blue')
    plt.title('Unique Recipients Distribution')
    plt.xlabel('Unique Recipients per Group')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x=grouped_df['unique_recipients'], color='blue')
    plt.title('Unique Recipients Spread')
    
    # Total Amount plots
    plt.subplot(2, 2, 3)
    sns.histplot(grouped_df['total_amount'], kde=True, bins=30, color='green')
    plt.title('Total Amount Distribution')
    plt.xlabel('Total Amount per Group')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x=grouped_df['total_amount'], color='green')
    plt.title('Total Amount Spread')
    
    plt.tight_layout()
    plt.show()

def save_df_to_csv(df, file_name, file_path, index=False):
   
    full_path = os.path.join(file_path, file_name)
    df.to_csv(full_path, index=index)

def create_gephi_files_accounts(df, output_dir):
    """
    Exports nodes and edges files for Gephi network analysis.

    Parameters:
    - df: pandas DataFrame with transaction data.
    - output_dir: Path to the directory where files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create nodes: unique accounts from both sender and receiver columns
    accounts = pd.unique(df[['Account', 'Account.1']].values.ravel('K'))
    nodes = pd.DataFrame({'Id': accounts})
    nodes_path = os.path.join(output_dir, 'nodes.csv')
    nodes.to_csv(nodes_path, index=False)

    # Create edges: count number of transactions between each pair
    edge_counts = df.groupby(['Account', 'Account.1']).size().reset_index(name='Weight')
    edge_counts.columns = ['Source', 'Target', 'Weight']
    edges_path = os.path.join(output_dir, 'edges.csv')
    edge_counts.to_csv(edges_path, index=False)

    print(f"Nodes file saved to: {nodes_path}")
    print(f"Edges file saved to: {edges_path}")

def create_gephi_files_banks(df: pd.DataFrame, folder_path: str):
    # Stack both From and To bank+account pairs
    from_accounts = df[['From Bank', 'Account']].rename(columns={'From Bank': 'Bank', 'Account': 'Account'})
    to_accounts = df[['To Bank', 'Account.1']].rename(columns={'To Bank': 'Bank', 'Account.1': 'Account'})    
    all_accounts = pd.concat([from_accounts, to_accounts], axis=0, ignore_index=True)
    
    # Now count unique accounts per bank
    accounts_per_bank = all_accounts.drop_duplicates().groupby('Bank').agg(Number_of_Accounts=('Account', 'nunique')).reset_index()
    accounts_per_bank['Id'] = accounts_per_bank['Bank'].astype(str)
    accounts_per_bank['Label'] = accounts_per_bank['Bank'].astype(str)
    nodes = accounts_per_bank[['Id', 'Label', 'Number_of_Accounts']]

    # Edge file creation (same as earlier)
    df['Bank Pair'] = df.apply(lambda row: tuple(sorted([row['From Bank'], row['To Bank']])), axis=1)
    edges = df.groupby('Bank Pair').size().reset_index(name='Number_of_Transactions')
    edges[['Source', 'Target']] = pd.DataFrame(edges['Bank Pair'].tolist(), index=edges.index)
    edges = edges[['Source', 'Target', 'Number_of_Transactions']]

    # Save files
    nodes_file = f"{folder_path}/nodes.csv"
    edges_file = f"{folder_path}/edges.csv"
    nodes.to_csv(nodes_file, index=False)
    edges.to_csv(edges_file, index=False)
    return nodes_file, edges_file

def inspect_csv_file(
    file_path,
    *,
    encoding="utf-8",
    prefer_newline=None,
    delimiters=(",", ";", "\t"),
    na_values=None,
    infer_sample_rows=200_000,
    use_sniffer=True,
    strict_lengths=False
):
    """
    Inspect a CSV file with minimal memory usage.

    Prints/returns:
      - Number of lines (including header)
      - Detected column separator
      - Number of columns
      - Column-wise missing counts and percentages
      - Column-wise inferred data type (from a sample)
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    encoding : str
        Text encoding for the file (default: 'utf-8').
    prefer_newline : str | None
        If given ('\n' or '\r\n'), force this newline when reading, else auto.
    delimiters : tuple[str]
        Candidate delimiters to consider.
    na_values : set[str] | None
        Strings to treat as missing (case-insensitive). If None, uses a default set.
    infer_sample_rows : int
        Max number of data rows to use for type inference (keeps runtime bounded).
        Missing-value counts still scan the WHOLE file.
    use_sniffer : bool
        Use csv.Sniffer to detect delimiter (falls back to heuristic if it fails).
    strict_lengths : bool
        If True, rows with wrong number of columns raise; otherwise they’re counted as malformed.

    Returns
    -------
    dict
        {
          'file': str,
          'line_count': int,
          'separator': str,             # human-friendly name
          'delimiter': str,             # actual delimiter char
          'num_columns': int,
          'columns': list[str],
          'data_rows': int,
          'malformed_rows': int,
          'columns_info': {
              col_name: {
                 'missing_count': int,
                 'missing_pct': float,   # 0..100
                 'non_null_count': int,
                 'inferred_dtype': str   # 'int', 'float', 'bool', 'datetime', 'string'
              },
              ...
          }
        }
    """


    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return {'error': 'File not found', 'file': file_path}

    # Reasonable defaults for NA tokens (case-insensitive match)
    if na_values is None:
        na_values = {
            "", "na", "n/a", "nan", "null", "none", "?", "-", "--"
        }

    # Helper: human label for delimiter
    sep_names = {',': 'comma (,)', ';': 'semicolon (;)', '\t': 'tab (\\t)'}
    sep_label = lambda d: sep_names.get(d, repr(d))

    # --- Step 1: Read a sample to detect delimiter + header robustly ---
    # Read a small chunk for sniffer and to avoid loading entire file
    try:
        with open(file_path, 'r', encoding=encoding, newline=prefer_newline) as fh:
            sample = fh.read(1024 * 64)  # 64KB
            if not sample:
                print("Empty file.")
                return {
                    'file': file_path,
                    'line_count': 0,
                    'separator': None,
                    'delimiter': None,
                    'num_columns': 0,
                    'columns': [],
                    'data_rows': 0,
                    'malformed_rows': 0,
                    'columns_info': {}
                }
    except Exception as e:
        print(f"Error opening file: {e}")
        return {'error': str(e), 'file': file_path}

    # Detect delimiter
    delimiter = None
    if use_sniffer:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="".join(delimiters))
            delimiter = dialect.delimiter
        except Exception:
            delimiter = None

    if delimiter is None:
        # Heuristic: choose the candidate that yields most splits on the first (likely header) line
        first_line = sample.splitlines()[0] if sample else ""
        counts = {d: first_line.count(d) for d in delimiters}
        delimiter = max(counts, key=counts.get)

    # Now parse header properly using csv so quoted fields are handled
    with open(file_path, 'r', encoding=encoding, newline=prefer_newline) as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        try:
            header = next(reader, None)
        except Exception as e:
            print(f"Failed to read header: {e}")
            return {'error': f'Failed to read header: {e}', 'file': file_path}

        if header is None:
            print("No header found.")
            return {
                'file': file_path,
                'line_count': 0,
                'separator': sep_label(delimiter),
                'delimiter': delimiter,
                'num_columns': 0,
                'columns': [],
                'data_rows': 0,
                'malformed_rows': 0,
                'columns_info': {}
            }

        columns = [h.strip() for h in header]
        num_columns = len(columns)

        # --- Step 2: Initialize stats structures ---
        line_count = 1  # header included
        data_rows = 0
        malformed_rows = 0

        # Missing counters for entire file
        missing_counts = [0] * num_columns
        non_null_counts = [0] * num_columns

        # Type inference flags based on sample
        # We track what each column COULD be; if a value contradicts, we relax to a wider type
        could_be_int = [True] * num_columns
        could_be_float = [True] * num_columns
        could_be_bool = [True] * num_columns
        could_be_datetime = [True] * num_columns

        # Simple datetime pattern checks (ISO-like); avoids heavy parsers for speed
        # Matches: YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS][.ms][+TZ/Z]
        date_re = re.compile(
            r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?$"
        )

        def is_na(val: str) -> bool:
            return (val.strip().lower() in na_values)

        def looks_int(s: str) -> bool:
            s = s.strip()
            if s == "": return False
            # allow leading +/-
            if s[0] in "+-":
                s = s[1:]
            return s.isdigit()

        def looks_float(s: str) -> bool:
            # accept standard floats like -12.3, 1e6, .5, 5.
            s = s.strip()
            if s == "": return False
            try:
                float(s)
                return True
            except Exception:
                return False

        BOOL_TOKENS = {
            "true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"
        }
        def looks_bool(s: str) -> bool:
            return s.strip().lower() in BOOL_TOKENS

        def looks_datetime(s: str) -> bool:
            return bool(date_re.match(s.strip()))

        # --- Step 3: Stream rows once (low memory), count missing across ALL rows,
        #             infer types using ONLY the first `infer_sample_rows` data rows ---
        for row_idx, row in enumerate(reader, start=1):
            line_count += 1
            # Enforce column length (optionally)
            if len(row) != num_columns:
                malformed_rows += 1
                if strict_lengths:
                    raise ValueError(
                        f"Row {row_idx} has {len(row)} columns; expected {num_columns}"
                    )
                # normalize length for counting by padding/truncating
                if len(row) < num_columns:
                    row = list(row) + [""] * (num_columns - len(row))
                else:
                    row = row[:num_columns]

            data_rows += 1

            # Column-wise missing/non-null counts
            for j, val in enumerate(row):
                if is_na(val):
                    missing_counts[j] += 1
                else:
                    non_null_counts[j] += 1

            # Type inference limited to a sample of rows for speed on huge files
            if data_rows <= infer_sample_rows:
                for j, val in enumerate(row):
                    v = val.strip()
                    if v == "" or is_na(v):
                        # missing values don't inform type
                        continue

                    # If it already cannot be an int, don't bother re-checking
                    if could_be_int[j] and not looks_int(v):
                        could_be_int[j] = False
                    # Float is superset of int; if it's not float-like, mark False
                    if could_be_float[j] and not (looks_float(v) or looks_int(v)):
                        could_be_float[j] = False
                    if could_be_bool[j] and not looks_bool(v):
                        could_be_bool[j] = False
                    if could_be_datetime[j] and not looks_datetime(v):
                        could_be_datetime[j] = False

        # --- Step 4: Decide dtype per column from flags (most specific first) ---
        inferred_types = []
        for j in range(num_columns):
            # Order: bool -> int -> float -> datetime -> string
            # (you can reorder if you prefer dates over numbers when ambiguous)
            if could_be_bool[j] and non_null_counts[j] > 0:
                inferred_types.append("bool")
            elif could_be_int[j] and non_null_counts[j] > 0:
                inferred_types.append("int")
            elif could_be_float[j] and non_null_counts[j] > 0:
                inferred_types.append("float")
            elif could_be_datetime[j] and non_null_counts[j] > 0:
                inferred_types.append("datetime")
            else:
                inferred_types.append("string")

        # --- Step 5: Build results ---
        columns_info = {}
        for name, miss, nonnull, dtype_ in zip(
            columns, missing_counts, non_null_counts, inferred_types
        ):
            total = miss + nonnull
            pct = (miss / total * 100.0) if total else 0.0
            columns_info[name] = {
                "missing_count": miss,
                "missing_pct": round(pct, 4),
                "non_null_count": nonnull,
                "inferred_dtype": dtype_,
            }

        info = {
            "file": file_path,
            "line_count": line_count,                # includes header
            "separator": sep_label(delimiter),
            "delimiter": delimiter,
            "num_columns": num_columns,
            "columns": columns,
            "data_rows": data_rows,                  # excludes header
            "malformed_rows": malformed_rows,
            "columns_info": columns_info,
        }

        # --- Friendly printout ---
        print(f"File: {file_path}")
        print(f"Number of lines (incl. header): {line_count}")
        print(f"Detected column separator: {sep_label(delimiter)}")
        print(f"Number of columns: {num_columns}")
        print(f"Data rows: {data_rows}")
        if malformed_rows:
            print(f"⚠️ Malformed rows (wrong # of columns): {malformed_rows}")

        print("\nColumn summary:")
        for col in columns:
            c = columns_info[col]
            print(
                f"  - {col}: dtype={c['inferred_dtype']}, "
                f"missing={c['missing_count']} ({c['missing_pct']}%), "
                f"non-null={c['non_null_count']}"
            )

        return info

def create_new_folder(base_path, folder_name):
    """
    Creates a new folder with the given folder_name inside base_path.
    Returns the full path to the newly created folder.
    """
    # Construct the full path for the new folder
    new_folder_path = os.path.join(base_path, folder_name)
    # Create the new folder if it doesn't already exist
    os.makedirs(new_folder_path, exist_ok=True)
    # Return the full path
    return new_folder_path

def top_accounts_by_transactions(df: pd.DataFrame,
                                 top_n: int = 10,
                                 source_col: str = "Account",
                                 target_col: str = "Account.1",
                                 timestamp_col: str = "Timestamp"):
    """
    Count transactions separately for outbound and inbound roles,
    and return min/max transaction dates (computed from `timestamp_col`).
    """

    # ---- 0) Validate columns
    missing = [c for c in [source_col, target_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected column(s): {missing}. Got: {list(df.columns)}")

    # ---- 1) Timestamp min/max (only touch the timestamp col)
    min_date = max_date = None
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        if ts.notna().any():
            min_date = ts.min()
            max_date = ts.max()

    # ---- 2) Work on CLEANED string views of account columns
    # Strip whitespace, drop empties, keep as strings to prevent accidental datetime coercion
    src = (
        df[source_col]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        .dropna()
    )
    tgt = (
        df[target_col]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        .dropna()
    )

    # ---- 3) Counts
    outbound_df = (
        src.value_counts()
           .head(top_n)
           .rename_axis("Account")
           .reset_index(name="outbound_tx_count")
    )

    inbound_df = (
        tgt.value_counts()
           .head(top_n)
           .rename_axis("Account")
           .reset_index(name="inbound_tx_count")
    )

    return outbound_df, inbound_df, min_date, max_date

def convert_column(
    df: pd.DataFrame,
    column: str,
    target_dtype: str,
    *,
    inplace: bool = False,
    datetime_format: Optional[str] = None,   # e.g. "%Y-%m-%d %H:%M:%S"
    utc: bool = False,
    max_examples: int = 20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Try to convert `df[column]` to `target_dtype` and report failures.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column name to convert.
    target_dtype : str
        One of: "int", "Int64", "float", "bool", "datetime", "string", "category".
        - "int"   -> numpy int64 (fails on NA); use "Int64" for nullable integer.
        - "float" -> float64
        - "bool"  -> maps common truthy/falsey strings/numbers
        - "datetime" -> pandas datetime64[ns] (optionally `format` and `utc`)
        - "string"   -> pandas' nullable string dtype
        - "category" -> pandas category
    inplace : bool
        If True, modifies `df`; otherwise works on a copy.
    datetime_format : str or None
        Optional strptime-like format when target is "datetime".
    utc : bool
        If True and target is "datetime", localizes/convert to UTC.
    max_examples : int
        Max number of (index, value) examples included in the report.

    Returns
    -------
    (df_out, report) : (pd.DataFrame, dict)
        df_out: DataFrame with the converted column (or not, if everything failed).
        report: {
            'success': bool,
            'column': str,
            'target_dtype': str,
            'converted_dtype': str,
            'n_rows': int,
            'n_failed': int,
            'failure_rate': float,
            'failed_examples': list[(index, original_value)],
            'failed_value_counts': dict[value -> count],  # top offenders
            'notes': str
        }
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Work on a copy unless told otherwise
    out = df if inplace else df.copy()

    # Original series and non-null mask (for failure detection)
    s = out[column]
    notna = s.notna()

    failed_mask = pd.Series(False, index=s.index)
    converted = None
    notes = []

    t = target_dtype.lower()

    try:
        if t in ("int", "int64"):
            # Use to_numeric first (coerce invalids to NaN), then cast
            tmp = pd.to_numeric(s, errors="coerce")
            failed_mask = notna & tmp.isna()
            if t == "int":
                if tmp.isna().any():
                    # Cannot cast floats with NaN to numpy int; report and keep nullable Int64
                    notes.append("Nulls present after coercion; used pandas 'Int64' to retain nulls.")
                    converted = tmp.astype("Int64")
                else:
                    converted = tmp.astype(np.int64)
            else:
                converted = tmp.astype("Int64")

        elif t in ("float", "float64"):
            tmp = pd.to_numeric(s, errors="coerce")
            failed_mask = notna & tmp.isna()
            converted = tmp.astype(float)

        elif t in ("bool", "boolean"):
            # Normalize to strings for robust mapping, but accept numeric 1/0 as well.
            truthy = {"true", "t", "yes", "y", "1"}
            falsey = {"false", "f", "no", "n", "0"}

            def to_bool(val):
                if pd.isna(val):
                    return pd.NA
                if isinstance(val, (int, np.integer, float, np.floating)):
                    if pd.isna(val):
                        return pd.NA
                    if val == 1:
                        return True
                    if val == 0:
                        return False
                sval = str(val).strip().lower()
                if sval in truthy:
                    return True
                if sval in falsey:
                    return False
                return "FAIL"

            mapped = s.map(to_bool)
            failed_mask = mapped.eq("FAIL")
            # Replace "FAIL" with NA for a clean nullable boolean dtype
            mapped = mapped.where(~failed_mask, pd.NA)
            converted = mapped.astype("boolean")

        elif t == "datetime":
            tmp = pd.to_datetime(s, errors="coerce", format=datetime_format, utc=utc)
            failed_mask = notna & tmp.isna()
            converted = tmp

        elif t in ("string", "str"):
            # Pandas nullable string dtype
            converted = s.astype("string")

        elif t == "category":
            converted = s.astype("category")

        else:
            raise ValueError(
                "Unsupported target_dtype. "
                "Use one of: 'int','Int64','float','bool','datetime','string','category'."
            )

        # Assign back
        out[column] = converted

    except Exception as e:
        # Unexpected conversion error (schema/engine issue)
        notes.append(f"Unexpected error during conversion: {e}")
        # Mark all non-null as failed
        failed_mask = notna.copy()

    # Build the report
    n_rows = len(s)
    n_failed = int(failed_mask.sum())
    failure_rate = float(n_failed / n_rows) if n_rows else 0.0

    # Examples & offender counts
    failed_examples = []
    if n_failed:
        # Collect up to `max_examples` (index, original_value)
        bad_idx = failed_mask[failed_mask].index[:max_examples]
        failed_examples = [(idx, df.loc[idx, column]) for idx in bad_idx]  # original values

        # Count top failing raw values
        failed_values = df.loc[failed_mask, column]
        failed_value_counts = failed_values.value_counts(dropna=False).head(20).to_dict()
    else:
        failed_value_counts = {}

    report = {
        "success": n_failed == 0,
        "column": column,
        "target_dtype": target_dtype,
        "converted_dtype": str(out[column].dtype),
        "n_rows": n_rows,
        "n_failed": n_failed,
        "failure_rate": failure_rate,
        "failed_examples": failed_examples,
        "failed_value_counts": failed_value_counts,
        "notes": "; ".join(notes) if notes else "",
    }

    return out, report

# ---------- Pretty-print helpers (numbers & sectioning) ----------

def _pretty_number(x, ndigits=6):
    """
    Format ints with thousands separators and floats with rounding.
    Leaves booleans/None/NaN alone; returns strings for numbers.
    """
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return x
        if isinstance(x, (np.bool_, bool)):
            return x
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        if isinstance(x, (float, np.floating)):
            if np.isfinite(x):
                # If value looks like an integer, show as integer
                if float(x).is_integer():
                    return f"{int(x):,}"
                return f"{round(float(x), ndigits):,}"
            return x
        return x
    except Exception:
        return x

def _resolve_count_valid_section(present_metrics: set) -> str:
    """
    Heuristic to decide which section 'count_valid' belongs to.
    Preference order: Boolean > Datetime > Numeric, based on co-present metrics.
    """
    if {"true_count", "false_count"} & present_metrics:
        return "Boolean"
    if {"min_datetime", "max_datetime", "span_days"} & present_metrics:
        return "Datetime"
    if {"mean", "std", "median", "iqr"} & present_metrics:
        return "Numeric"
    return "Other"

def _add_section_labels(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'section' column based on metric names and co-present metrics.
    Works without touching your existing analytics.
    """
    present = set(report_df["metric"].astype(str))
    section_col = []
    for m in report_df["metric"].astype(str):
        if m in {
            "column","dtype","logical_type","rows_total","missing_count","missing_pct",
            "unique_count","unique_pct","unique_count_approx","memory_mb","workload_scope"
        }:
            section_col.append("Overview")
        elif m == "count_valid":
            section_col.append(_resolve_count_valid_section(present))
        elif m in {
            "min","q01","q05","q25","median","q75","q95","q99","max",
            "mean","std","mad","iqr","skew","kurtosis_fisher",
            "zeros_count","positive_count","negative_count",
            "outliers_iqr_count","outliers_zscore_count"
        }:
            section_col.append("Numeric")
        elif m in {"min_datetime","max_datetime","span_days","weekday_top","month_top"}:
            section_col.append("Datetime")
        elif m in {"true_count","false_count","true_pct","false_pct"}:
            section_col.append("Boolean")
        elif m in {
            "top_value","shannon_entropy_bits",
            "text_len_min","text_len_q25","text_len_median","text_len_mean",
            "text_len_q75","text_len_q95","text_len_max"
        }:
            section_col.append("Top Values / Text")
        else:
            section_col.append("Other")
    out = report_df.copy()
    out.insert(0, "section", section_col)
    return out

def _format_report_numbers(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply number formatting to value/count/pct columns and sort rows nicely.
    """
    df = report_df.copy()
    for col in ("value","count","pct"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _pretty_number(x, ndigits=6))
    # Stable sort by section -> metric -> rank (when present)
    sort_cols = [c for c in ["section","metric","rank"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")
    return df

def _with_blank_rows_between_sections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert a single blank row between section blocks for CSV readability.
    """
    parts = []
    for _, g in df.groupby("section", sort=False, as_index=False):
        parts.append(g)
        parts.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))
    if not parts:
        return df
    out = pd.concat(parts, ignore_index=True)
    # drop trailing blank
    return out.iloc[:-1] if len(out) else out


# ---------- Your function with output-only enhancements ----------

def univariate_eda(
    df: pd.DataFrame,
    column: str,
    top_k: int = 20,
    write_path: Optional[str] = None,
    approx: bool = False,
    sample_size: int = 1_000_000,
    iqr_outlier_factor: float = 1.5,
    z_thresh: float = 3.0,
) -> str:
    """
    Compute a univariate EDA report for `df[column]`.
    Returns the report as a CSV string. Optionally writes to `write_path`.

    Enhancements vs your original:
    - Adds a 'section' column (Overview/Numeric/Datetime/Boolean/Top Values / Text).
    - Formats numbers more readably (thousands separators, rounded floats).
    - Inserts blank rows between sections in the CSV for readability.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    s = df[column]
    n_total = len(s)

    # Optional sampling for heavy ops
    if approx and n_total > sample_size:
        s_work = s.sample(sample_size, random_state=42)
        note_sampling = f"sampled={sample_size} of {n_total}"
    else:
        s_work = s
        note_sampling = "full"

    # Basic / universal metrics
    report_rows = []
    add = report_rows.append

    dtype = s.dtype
    n_missing = int(s.isna().sum())
    pct_missing = (n_missing / n_total * 100.0) if n_total else np.nan

    # unique count may be heavy; use sample if approx
    try:
        n_unique = int(s_work.nunique(dropna=True))
        approx_unique = approx and n_total > len(s_work)
    except Exception:
        n_unique = np.nan
        approx_unique = True

    pct_unique = (
        n_unique / (len(s_work) - int(s_work.isna().sum())) * 100.0
        if len(s_work) else np.nan
    )

    mem_bytes = int(s.memory_usage(deep=True))
    mem_mb = mem_bytes / (1024 ** 2)

    # Heuristic logical type
    if pd.api.types.is_bool_dtype(dtype):
        logical = "boolean"
    elif pd.api.types.is_numeric_dtype(dtype):
        logical = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        logical = "datetime"
    elif pd.api.types.is_categorical_dtype(dtype):
        logical = "categorical"
    else:
        nn = len(s_work) - int(s_work.isna().sum())
        logical = "text" if (nn > 0 and n_unique/nn > 0.5) else "categorical"

    add({"metric": "column", "value": column})
    add({"metric": "dtype", "value": str(dtype)})
    add({"metric": "logical_type", "value": logical})
    add({"metric": "rows_total", "value": n_total})
    add({"metric": "missing_count", "value": n_missing})
    add({"metric": "missing_pct", "value": round(pct_missing, 6)})
    add({"metric": "unique_count", "value": n_unique})
    add({"metric": "unique_pct", "value": round(pct_unique, 6)})
    add({"metric": "unique_count_approx", "value": approx_unique})
    add({"metric": "memory_mb", "value": round(mem_mb, 6)})
    add({"metric": "workload_scope", "value": note_sampling})

    # ---- Numeric metrics ----
    if logical == "numeric":
        s_num = pd.to_numeric(s_work, errors="coerce")
        valid = s_num.dropna()
        n_valid = len(valid)

        if n_valid > 0:
            desc = valid.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            q25, q50, q75 = desc["25%"], desc["50%"], desc["75%"]
            iqr = q75 - q25
            mean = valid.mean()
            std = valid.std(ddof=1)
            mad = (valid - q50).abs().median()
            skew = valid.skew()
            kurt = valid.kurtosis()

            # Outlier counts
            iqr_low = q25 - iqr_outlier_factor * iqr
            iqr_high = q75 + iqr_outlier_factor * iqr
            out_iqr = int(((valid < iqr_low) | (valid > iqr_high)).sum())

            if std and std > 0:
                z_scores = (valid - mean) / std
                out_z = int((np.abs(z_scores) > z_thresh).sum())
            else:
                out_z = 0

            zeros = int((valid == 0).sum())
            pos = int((valid > 0).sum())
            neg = int((valid < 0).sum())

            add({"metric": "count_valid", "value": n_valid})
            for k in ["min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max", "mean", "std"]:
                key_map = {
                    "min": "min", "1%": "q01", "5%": "q05", "25%": "q25",
                    "50%": "median", "75%": "q75", "95%": "q95", "99%": "q99",
                    "max": "max", "mean": "mean", "std": "std"
                }
                v = desc[k] if k in desc.index else np.nan
                add({"metric": key_map[k], "value": round(float(v), 12)})

            add({"metric": "mad", "value": round(float(mad), 12)})
            add({"metric": "iqr", "value": round(float(iqr), 12)})
            add({"metric": "skew", "value": round(float(skew), 12) if pd.notna(skew) else np.nan})
            add({"metric": "kurtosis_fisher", "value": round(float(kurt), 12) if pd.notna(kurt) else np.nan})
            add({"metric": "zeros_count", "value": zeros})
            add({"metric": "positive_count", "value": pos})
            add({"metric": "negative_count", "value": neg})
            add({"metric": "outliers_iqr_count", "value": out_iqr})
            add({"metric": "outliers_zscore_count", "value": out_z})
        else:
            add({"metric": "count_valid", "value": 0})

    # ---- Datetime metrics ----
    elif logical == "datetime":
        s_dt = pd.to_datetime(s_work, errors="coerce")
        valid = s_dt.dropna()
        n_valid = len(valid)
        add({"metric": "count_valid", "value": n_valid})
        if n_valid > 0:
            vmin, vmax = valid.min(), valid.max()
            span = vmax - vmin
            add({"metric": "min_datetime", "value": str(vmin)})
            add({"metric": "max_datetime", "value": str(vmax)})
            add({"metric": "span_days", "value": round(span.total_seconds() / 86400.0, 6)})

            # Weekday distribution (named)
            try:
                wd = valid.dt.weekday.value_counts(dropna=False)
                wd_top = wd.head(min(7, top_k))
                weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
                for i, (wd_i, cnt) in enumerate(wd_top.items(), start=1):
                    add({
                        "metric": "weekday_top",
                        "rank": i,
                        "key": weekday_map.get(int(wd_i), str(wd_i)),
                        "count": int(cnt),
                        "pct": round(cnt / n_valid * 100.0, 6)
                    })
            except Exception:
                pass

            # Month distribution (YYYY-MM)
            try:
                m = valid.dt.to_period("M").astype(str).value_counts()
                m_top = m.head(min(12, top_k))
                for i, (mon, cnt) in enumerate(m_top.items(), start=1):
                    add({"metric": "month_top", "rank": i, "key": mon, "count": int(cnt),
                         "pct": round(cnt / n_valid * 100.0, 6)})
            except Exception:
                pass

    # ---- Boolean metrics ----
    elif logical == "boolean":
        valid = s_work.dropna()
        n_valid = len(valid)
        true_cnt = int((valid == True).sum())   # noqa: E712
        false_cnt = int((valid == False).sum()) # noqa: E712
        add({"metric": "count_valid", "value": n_valid})
        add({"metric": "true_count", "value": true_cnt})
        add({"metric": "false_count", "value": false_cnt})
        if n_valid:
            add({"metric": "true_pct", "value": round(true_cnt / n_valid * 100.0, 6)})
            add({"metric": "false_pct", "value": round(false_cnt / n_valid * 100.0, 6)})

    # ---- Categorical / Text metrics ----
    if logical in {"categorical", "text"}:
        valid = s_work.dropna()
        n_valid = len(valid)

        # Value counts (top_k); sample if approx for speed
        vc_source = valid
        if approx and len(valid) > sample_size:
            vc_source = valid.sample(sample_size, random_state=42)

        vc = vc_source.value_counts(dropna=False)
        top = vc.head(top_k)
        denom = float(len(vc_source)) if len(vc_source) else 1.0

        for i, (val, cnt) in enumerate(top.items(), start=1):
            add({
                "metric": "top_value",
                "rank": i,
                "key": str(val),
                "count": int(cnt),
                "pct": round(cnt / denom * 100.0, 6),
            })

        # Entropy (Shannon) on observed distribution
        try:
            p = (vc / denom).astype(float)
            entropy = float(-(p * np.log2(p + 1e-12)).sum())
            add({"metric": "shannon_entropy_bits", "value": round(entropy, 12)})
        except Exception:
            add({"metric": "shannon_entropy_bits", "value": np.nan})

        # Text length stats if looks like text
        if logical == "text":
            try:
                lens = valid.astype(str).str.len()
                if len(lens):
                    add({"metric": "text_len_min", "value": int(lens.min())})
                    add({"metric": "text_len_q25", "value": float(lens.quantile(0.25))})
                    add({"metric": "text_len_median", "value": float(lens.median())})
                    add({"metric": "text_len_mean", "value": float(lens.mean())})
                    add({"metric": "text_len_q75", "value": float(lens.quantile(0.75))})
                    add({"metric": "text_len_q95", "value": float(lens.quantile(0.95))})
                    add({"metric": "text_len_max", "value": int(lens.max())})
            except Exception:
                pass

    # ---- Finalize as DataFrame -> CSV string with sections & formatting ----
    report_df = pd.DataFrame(report_rows)

    # Order columns nicely; keep any extras (rank/key/count/pct) to the right
    base_cols = ["metric", "value"]
    extra_cols = [c for c in report_df.columns if c not in base_cols]
    preferred = ["rank", "key", "count", "pct"]
    ordered_extras = [c for c in preferred if c in extra_cols] + [c for c in extra_cols if c not in preferred]
    report_df = report_df[base_cols + ordered_extras]

    # Add sections, format numbers, and insert blank lines between sections
    report_df = _add_section_labels(report_df)
    report_df = _format_report_numbers(report_df)
    csv_df = _with_blank_rows_between_sections(report_df)

    csv_str = csv_df.to_csv(index=False)
    if write_path:
        csv_df.to_csv(write_path, index=False)

    return csv_str
