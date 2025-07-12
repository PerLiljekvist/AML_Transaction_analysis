import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
    return pd.DataFrame({'line': lines})

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

def export_gephi_files_accounts(df, output_dir):
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

def export_gephi_files_banks(
    df: pd.DataFrame,
    output_dir: str,
    *,
    from_bank_col: str = "From Bank",
    to_bank_col: str = "To Bank",
    from_acct_col: str = "Account",
    to_acct_col: str = "Account.1",
    amount_col: str = "Amount Paid",     # or "Amount Received" if you prefer
):
    """
    Export Gephi-ready nodes and edges for a bank-level transaction network.

    Parameters
    ----------
    df : pandas.DataFrame
        Transaction data containing at least the columns specified below.
    output_dir : str
        Directory in which ``nodes.csv`` and ``edges.csv`` are written.
    from_bank_col, to_bank_col : str, default "From Bank", "To Bank"
        Column names holding the sender / receiver bank identifiers.
    from_acct_col, to_acct_col : str, default "Account", "Account.1"
        Column names holding the sender / receiver account identifiers.
    amount_col : str, default "Amount Paid"
        Column whose numeric values represent the transaction amount.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Build the node list -------------------------------------------------
    all_banks = pd.unique(df[[from_bank_col, to_bank_col]].values.ravel("K"))

    # 1) How many *unique* accounts does each bank hold?
    send_accts = (
        df[[from_bank_col, from_acct_col]]
        .rename(columns={from_bank_col: "Bank", from_acct_col: "Account"})
        .drop_duplicates()
    )
    recv_accts = (
        df[[to_bank_col, to_acct_col]]
        .rename(columns={to_bank_col: "Bank", to_acct_col: "Account"})
        .drop_duplicates()
    )
    acct_counts = (
        pd.concat([send_accts, recv_accts], ignore_index=True)
        .groupby("Bank")["Account"]
        .nunique()
    )

    # 2) Total amount in which the bank is involved (incoming + outgoing)
    amt_out = df.groupby(from_bank_col)[amount_col].sum()
    amt_in  = df.groupby(to_bank_col)[amount_col].sum()
    total_amt = amt_out.add(amt_in, fill_value=0)

    nodes = (
        pd.DataFrame({"Id": all_banks})
        .assign(
            Accounts=lambda x: x["Id"].map(acct_counts).fillna(0).astype(int),
            TotalAmount=lambda x: x["Id"].map(total_amt).fillna(0),
        )
    )

    # --- Build the edge list -------------------------------------------------
    edges = (
        df.groupby([from_bank_col, to_bank_col])
        .agg(
            Weight=("Timestamp", "size"),        # number of transactions
            TotalAmount=(amount_col, "sum"),     # total value between the pair
        )
        .reset_index()
        .rename(
            columns={
                from_bank_col: "Source",
                to_bank_col: "Target",
            }
        )
    )

    # --- Export --------------------------------------------------------------
    nodes_path = os.path.join(output_dir, "nodes.csv")
    edges_path = os.path.join(output_dir, "edges.csv")

    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    print(f"Nodes file saved to: {nodes_path}")
    print(f"Edges file saved to: {edges_path}")

def inspect_csv_file(file_path):
    """
    Inspect a CSV file and print information about it without loading it into memory.
    Prints:
      - Number of lines
      - Detected column separator
      - Number of columns (from header)
    Returns a dictionary with this information.
    """
    import os

    info = {}
    separators = [',', ';', '\t']
    sep_names = {',': 'comma (,)', ';': 'semicolon (;)', '\t': 'tab (\\t)'}
    detected_sep = None
    num_columns = None
    line_count = 0

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return {'error': 'File not found'}

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the first line (header) to detect separator and columns
            header = file.readline()
            line_count = 1 if header else 0

            # Detect separator by which one splits header into most columns
            sep_counts = {sep: header.count(sep) for sep in separators}
            detected_sep = max(sep_counts, key=sep_counts.get)
            num_columns = header.count(detected_sep) + 1 if header else 0

            # Count remaining lines efficiently
            for _ in file:
                line_count += 1

        info['line_count'] = line_count
        info['separator'] = sep_names[detected_sep]
        info['num_columns'] = num_columns

        print(f"File: {file_path}")
        print(f"Number of lines: {line_count}")
        print(f"Detected column separator: {sep_names[detected_sep]}")
        print(f"Number of columns (from header): {num_columns}")

        return info

    except Exception as e:
        print(f"Error reading file: {e}")
        return {'error': str(e)}
