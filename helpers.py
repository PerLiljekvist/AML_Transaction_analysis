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

def load_rows_for_account(filepath, account, account_column="Account", sep=";", chunksize=1000):
    """
    Loads rows from a CSV file where account_column equals the target account.
    
    Parameters:
        filepath (str): Path to the CSV file.
        account (str): The account value to filter on.
        account_column (str): Name of the column to filter (default: "Account").
        sep (str): Column separator in the CSV file (default: ";").
        chunksize (int): Number of rows per chunk (default: 1000).
        
    Returns:
        pandas.DataFrame: DataFrame with only rows matching the account.
    """
    filtered_chunks = []
    for chunk in pd.read_csv(
        filepath, 
        sep=sep, 
        chunksize=chunksize, 
        dtype=str
    ):
        filtered = chunk[chunk[account_column] == account]
        if not filtered.empty:
            filtered_chunks.append(filtered)
    if filtered_chunks:
        return pd.concat(filtered_chunks, ignore_index=True)
    else:
        # Return an empty DataFrame with the right columns
        return pd.DataFrame(columns=pd.read_csv(filepath, sep=sep, nrows=0).columns)

