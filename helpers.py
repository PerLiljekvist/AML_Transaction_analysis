import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from typing import Tuple, Dict, Any, Optional

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

def read_csv_custom(filepath, nrows=None, sep=None):
    """
    Reads a CSV file into a pandas DataFrame with options:
    - nrows: number of rows to read (None means all)
    - sep: separator character; if None, auto-detects
    - Assumes first row is headers
    - Converts decimals with ',' as decimal separator
    - Converts date columns to 'yyyy-mm-dd hh:mm:ss' format
    
    Returns:
        pd.DataFrame
    """
    # Auto-detect separator if not provided
    if sep is None:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            sample = f.read(1024)
            sniffer = csv.Sniffer()
            sep = sniffer.sniff(sample).delimiter

    # Read CSV with pandas, specifying decimal=',' to handle decimal commas
    # Use nrows parameter to limit rows
    df = pd.read_csv(filepath, sep=sep, decimal=',', nrows=nrows, encoding='utf-8')

    # Attempt to convert columns to appropriate dtypes
    for col in df.columns:
        # Try converting to numeric (int or float)
        df[col] = pd.to_numeric(df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='ignore') \
            if df[col].dtype == object else df[col]

    # For date columns, try to parse and convert to 'yyyy-mm-dd hh:mm:ss'
    # We assume date columns contain strings with date-like patterns
    for col in df.select_dtypes(include=['object']).columns:
        try:
            parsed_dates = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
            if parsed_dates.notna().sum() > 0:
                df[col] = parsed_dates.dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            # If parsing fails, leave column as is
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





