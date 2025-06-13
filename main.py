import pandas as pd
import csv
import networkx as nx
from scipy import stats
from datetime import datetime

filePath = "/Users/perliljekvist/Documents/Python/IBM AML/Data/HI-Small_Trans.csv"

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

def detect_fan_out_patterns(df, time_freq='10min', z_threshold=3):
    """
    Detect fan-out patterns using z-score on unique recipient counts per source account and time window.
    
    Args:
        df (pd.DataFrame): Transaction data with columns:
            ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', ...]
        time_freq (str): Pandas time frequency string for grouping (default '1H' for 1 hour)
        z_threshold (float): Z-score threshold to flag anomalies (default 3)
    
    Returns:
        pd.DataFrame: Suspicious fan-out events with columns:
            ['From Bank', 'Account', 'Timestamp', 'unique_recipients', 'total_amount', 'z_score']
    """
    # Ensure Timestamp is datetime type
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Create unique destination account identifier
    df['Dest_Account'] = df['To Bank'].astype(str) + '_' + df['Account.1'].astype(str)
    
    # Group by source account and time window
    grouped = df.groupby(
        ['From Bank', 'Account', pd.Grouper(key='Timestamp', freq=time_freq)]
    ).agg(
        unique_recipients=('Dest_Account', 'nunique'),
        total_amount=('Amount Received', 'sum')
    ).reset_index()
    
    # Calculate z-score for unique recipient counts
    grouped['z_score'] = stats.zscore(grouped['unique_recipients'])
    
    # Filter for suspicious fan-out patterns exceeding the z-score threshold
    suspicious = grouped[grouped['z_score'] > z_threshold].copy()
    
    # Sort by descending z-score for priority review
    suspicious.sort_values('z_score', ascending=False, inplace=True)
    
    return suspicious

import pandas as pd
from scipy import stats

def detect_fan_out_groups(df, time_freq='1H', outlier_method='percentile', threshold=95):
    """
    Groups transactions by source account and time window.
    Flags outliers using z-score or percentile.
    
    Args:
        df (pd.DataFrame): Your transaction data.
        time_freq (str): Time window for grouping (e.g., '1H', '30min', '1D').
        outlier_method (str): 'zscore' or 'percentile'.
        threshold (float): For 'zscore', the z-score threshold; for 'percentile', the percentile (e.g., 95).
    
    Returns:
        pd.DataFrame: Aggregated groups with outlier flags.
    """
    # Ensure timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Create unique destination account identifier
    df['Dest_Account'] = df['To Bank'].astype(str) + '_' + df['Account.1'].astype(str)
    
    # Group by source account and time window
    grouped = df.groupby(
        ['From Bank', 'Account', pd.Grouper(key='Timestamp', freq=time_freq)]
    ).agg(
        unique_recipients=('Dest_Account', 'nunique'),
        total_amount=('Amount Received', 'sum')
    ).reset_index()
    
    # Flag outliers
    if outlier_method == 'zscore':
        grouped['z_score'] = stats.zscore(grouped['unique_recipients'])
        grouped['is_outlier'] = grouped['z_score'] > threshold
    elif outlier_method == 'percentile':
        threshold_value = grouped['unique_recipients'].quantile(threshold/100)
        grouped['is_outlier'] = grouped['unique_recipients'] > threshold_value
    
    return grouped


#################################Function calls##############################################

#df = get_file_head_as_df(filePath, n=10, encoding='utf-8')

df = read_csv_custom(filePath, nrows=50000)
#print(df)
                
#Detect suspicious patterns
# suspicious = detect_fan_out_patterns(df)
# print(f"Found {len(suspicious)} suspicious patterns")
# print(suspicious[['From Bank', 'Account', 'Timestamp', 'unique_recipients', 'z_score']])

# View suspicious groups
# Example usage
fan_out = detect_fan_out_groups(df, time_freq='1H', outlier_method='percentile', threshold=98)

# View suspicious groups
suspicious = fan_out[fan_out['is_outlier']]
print(suspicious[['From Bank', 'Account', 'Timestamp', 'unique_recipients', 'total_amount']])








