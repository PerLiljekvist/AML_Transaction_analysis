import pandas as pd
import csv
import networkx as nx
from scipy import stats
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
import numpy as np

#hhelp functions 

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

#fan-out

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

def preprocess_and_group(df, time_freq='1H'):
    """
    Preprocesses the dataframe and groups by source account and time window.
    
    Args:
        df (pd.DataFrame): Transaction data.
        time_freq (str): Time window for grouping (e.g., '1H', '30min', '1D').
    
    Returns:
        pd.DataFrame: Aggregated groups.
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Dest_Account'] = df['To Bank'].astype(str) + '_' + df['Account.1'].astype(str)
    grouped = df.groupby(
        ['From Bank', 'Account', pd.Grouper(key='Timestamp', freq=time_freq)]
    ).agg(
        unique_recipients=('Dest_Account', 'nunique'),
        total_amount=('Amount Received', 'sum')
    ).reset_index()
    return grouped

def plot_unique_recipient_histogram(df, account_col='Account', recipient_col='unique_recipients'):
    """
    Plots a histogram showing the distribution of how many accounts have 1, 2, 3, ... unique recipients.

    Parameters:
        df (pd.DataFrame): DataFrame with at least 'account' and 'unique_recipient' columns.
        account_col (str): Name of the column containing account identifiers.
        recipient_col (str): Name of the column containing unique recipient identifiers.
    """
    # If recipient_col is not present, assume each row is a unique recipient for the account
    # If you want to count unique recipients per account, use groupby and nunique
    # (but your sample data already has a 'unique_recipient' column, perhaps as a count)
    # If you want to use the column as-is:
    unique_recipient_counts = df.groupby(account_col)[recipient_col].max().value_counts().sort_index()
    # Or, if you want to actually count unique recipients (replace above with):
    # unique_recipient_counts = df.groupby(account_col)[recipient_col].nunique().value_counts().sort_index()
    # (but your data suggests recipient_col is already the count per account)

    print(unique_recipient_counts)

    plt.figure(figsize=(10, 6))
    unique_recipient_counts.plot(kind='bar', color='blue', edgecolor='black')
    plt.title('Distribution of Accounts by Number of Unique Recipients')
    plt.xlabel('Number of Unique Recipients per Account')
    plt.ylabel('Number of Accounts')
    plt.grid(True, alpha=0.3)
    plt.show()

def detect_fan_out_groups_zscore(df, time_freq='1H', threshold=3):
    """
    Groups transactions and flags outliers using z-score method.
    
    Args:
        df (pd.DataFrame): Transaction data.
        time_freq (str): Time window for grouping.
        threshold (float): Z-score threshold for outlier detection.
    
    Returns:
        pd.DataFrame: Aggregated groups with outlier flags.
    """
    grouped = preprocess_and_group(df, time_freq)
    grouped['z_score'] = stats.zscore(grouped['unique_recipients'])
    grouped['is_outlier'] = grouped['z_score'] > threshold
    return grouped

def detect_fan_out_groups_percentile(df, time_freq='1H', threshold=95):
    """
    Groups transactions and flags outliers using percentile method.
    
    Args:
        df (pd.DataFrame): Transaction data.
        time_freq (str): Time window for grouping.
        threshold (float): Percentile threshold (e.g., 95).
    
    Returns:
        pd.DataFrame: Aggregated groups with outlier flags.
    """
    grouped = preprocess_and_group(df, time_freq)
    threshold_value = grouped['unique_recipients'].quantile(threshold/100)
    grouped['is_outlier'] = grouped['unique_recipients'] > threshold_value
    return grouped


    """
    Detects fan-in AML patterns using percentile on unique senders per receiver per day.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns:
            ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', ...]
        threshold (float): Percentile threshold for flagging anomalies
    
    Returns:
        pd.DataFrame: Original DataFrame with added 'Is_Fan_In_P' flag column
    """
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert timestamp and extract date
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.floor('D')
    
    # Create composite keys
    df['Sender_Key'] = df['From Bank'].astype(str) + '_' + df['Account'].astype(str)
    df['Receiver_Key'] = df['To Bank'].astype(str) + '_' + df['Account.1'].astype(str)
    
    # Count unique senders per receiver per day
    group = df.groupby(['Receiver_Key', 'Date'])['Sender_Key'].nunique().reset_index(name='Unique_Senders')
    
    # Calculate percentile per receiver
    group['percentile'] = group.groupby('Receiver_Key')['Unique_Senders'].transform(
        lambda x: (x.rank(method='max') / len(x) * 100)
    )
    group['Is_Fan_In_P'] = group['percentile'] > threshold
    
    # Merge flags back to original DataFrame
    result = df.merge(
        group[['Receiver_Key', 'Date', 'Is_Fan_In_P']],
        on=['Receiver_Key', 'Date'],
        how='left'
    ).fillna(False)
    
    # Cleanup temporary columns
    return result.drop(columns=['Sender_Key', 'Receiver_Key'], errors='ignore')

#fan-in

def preprocess_and_group_fan_in(df, time_freq='1H'):
    """
    Preprocesses transaction data for fan-in detection by grouping incoming transactions.
    
    Args:
        df (pd.DataFrame): Transaction data with 'Timestamp' and 'Account.1' (receiver)
        time_freq (str): Time window for grouping (default: '1H')
    
    Returns:
        pd.DataFrame: Aggregated data with unique senders and total received amount per receiver
    """
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    # Group by time window and receiving account
    grouped = df.groupby([pd.Grouper(freq=time_freq), 'Account.1']).agg(
    unique_senders=('Account', 'nunique'),
    total_received=('Amount Received', 'sum'),
    transaction_count=('Amount Received', 'count')
).reset_index()
    
    return grouped

def detect_fan_in_groups_zscore(df, time_freq='1H', threshold=3):
    """
    Flags fan-in AML patterns using z-score outlier detection on unique senders.
    
    Args:
        df (pd.DataFrame): Transaction data
        time_freq (str): Time window for grouping
        threshold (float): Z-score cutoff for outliers
    
    Returns:
        pd.DataFrame: Aggregated groups with outlier flags
    """
    grouped = preprocess_and_group_fan_in(df, time_freq)
    grouped['z_score'] = stats.zscore(grouped['unique_senders'])
    grouped['is_outlier'] = grouped['z_score'] > threshold
    return grouped

def simple_fan_in_report(suspicious_df):
    print("\n" + "="*50)
    print("SIMPLE FAN-IN AML REPORT")
    print("="*50)
    print(f"Number of suspicious fan-in groups: {len(suspicious_df)}")
    
    # Filter for groups with more than one unique sender
    multi_sender_df = suspicious_df[suspicious_df['unique_senders'] > 1]
    print(f"Number of groups with >1 unique sender: {len(multi_sender_df)}")
    
    print("\nPreview of suspicious entries (showing unique_senders):")
    if not multi_sender_df.empty:
        print(multi_sender_df[['Timestamp', 'Account.1', 'unique_senders', 'transaction_count', 'total_received']].head(5))
    else:
        print("No suspicious groups with more than one unique sender.")
    
    print("="*50)

#scatter-gather

def preprocess_transactions(df):
    # Standardize column names
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    # Ensure correct data types
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Amount_Received'] = pd.to_numeric(df['Amount_Received'], errors='coerce')
    df['Amount_Paid'] = pd.to_numeric(df['Amount_Paid'], errors='coerce')
    return df

def detect_scatter_gather(df, percentile=95):
    # Scatter: Accounts sending to many unique recipients
    scatter_counts = df.groupby('Account')['Account.1'].nunique()
    scatter_threshold = np.percentile(scatter_counts, percentile)
    scatter_accounts = scatter_counts[scatter_counts >= scatter_threshold].index.tolist()

    # Gather: Accounts receiving from many unique senders
    gather_counts = df.groupby('Account.1')['Account'].nunique()
    gather_threshold = np.percentile(gather_counts, percentile)
    gather_accounts = gather_counts[gather_counts >= gather_threshold].index.tolist()

    # Find suspicious transactions
    suspicious = df[
        (df['Account'].isin(scatter_accounts)) |
        (df['Account.1'].isin(gather_accounts))
    ]
    return suspicious, scatter_accounts, gather_accounts, scatter_threshold, gather_threshold

def print_report(suspicious, scatter_accounts, gather_accounts, scatter_threshold, gather_threshold):
    print(f"\n--- Scatter-Gather Money Laundering Detection Report ---")
    print(f"Scatter threshold (unique recipients per sender): {scatter_threshold:.2f}")
    print(f"Gather threshold (unique senders per recipient): {gather_threshold:.2f}")
    print(f"Number of suspicious transactions found: {len(suspicious)}\n")
    print(f"Accounts flagged as scatterers (sending to many): {scatter_accounts}")
    print(f"Accounts flagged as gatherers (receiving from many): {gather_accounts}\n")
    print("Suspicious transactions:")
    #print(suspicious.to_string(index=False))

def detect_scatter_gather_money_laundering(df, percentile=95):
    df = preprocess_transactions(df)
    suspicious, scatter_accounts, gather_accounts, scatter_threshold, gather_threshold = detect_scatter_gather(df, percentile)
    print_report(suspicious, scatter_accounts, gather_accounts, scatter_threshold, gather_threshold)

#gather-scatter

def preprocess_transactions(df):
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Amount_Received'] = pd.to_numeric(df['Amount_Received'], errors='coerce')
    df['Amount_Paid'] = pd.to_numeric(df['Amount_Paid'], errors='coerce')
    return df

def detect_gather_scatter(df, percentile=95):
    # Gather: Accounts that receive from many unique senders
    gather_counts = df.groupby('Account.1')['Account'].nunique()
    gather_threshold = np.percentile(gather_counts, percentile)
    gather_accounts = gather_counts[gather_counts >= gather_threshold].index.tolist()
    
    # Scatter: Of those, which send to many unique recipients
    scatter_counts = df[df['Account'].isin(gather_accounts)].groupby('Account')['Account.1'].nunique()
    scatter_threshold = np.percentile(scatter_counts, percentile)
    gather_scatter_accounts = scatter_counts[scatter_counts >= scatter_threshold].index.tolist()
    
    # Find suspicious transactions: incoming and outgoing for these accounts
    suspicious_in = df[df['Account.1'].isin(gather_scatter_accounts)]
    suspicious_out = df[df['Account'].isin(gather_scatter_accounts)]
    suspicious = pd.concat([suspicious_in, suspicious_out]).drop_duplicates()
    
    return suspicious, gather_scatter_accounts, gather_threshold, scatter_threshold

def print_gather_scatter_report(suspicious, gather_scatter_accounts, gather_threshold, scatter_threshold):
    print(f"\n--- Gather-Scatter Money Laundering Detection Report ---")
    print(f"Gather threshold (unique senders per recipient): {gather_threshold:.2f}")
    print(f"Scatter threshold (unique recipients per sender): {scatter_threshold:.2f}")
    print(f"Number of suspicious transactions found: {len(suspicious)}\n")
    print(f"Accounts flagged as gather-scatter (central node): {gather_scatter_accounts}\n")
    print("Suspicious transactions:")
    #print(suspicious.to_string(index=False))

def detect_gather_scatter_money_laundering(df, percentile=95):
    df = preprocess_transactions(df)
    suspicious, gather_scatter_accounts, gather_threshold, scatter_threshold = detect_gather_scatter(df, percentile)
    print_gather_scatter_report(suspicious, gather_scatter_accounts, gather_threshold, scatter_threshold)

#################################Function calls##############################################
filePath = "/Users/perliljekvist/Documents/Python/IBM AML/Data/HI-Small_Trans.csv"
folderPath = "/Users/perliljekvist/Documents/Python/IBM AML/Data/"

df = read_csv_custom(filePath, nrows=10000)

detect_gather_scatter_money_laundering(df, percentile=98)

#results = detect_fan_in_groups_zscore(df, time_freq='1H', threshold=3)

# Filter flagged transactions
#suspicious = results[results['is_outlier']]

#simple_fan_in_report(results)

# Check distribution: Source account -> nof destination accounts before choosing method of detection
#plot_unique_recipient_histogram(grouped_df.where(grouped_df['unique_recipients'] > 1))
#plot_group_distributions(grouped_df)
          
#Detect suspicious patterns z-score based. No suitable for non-Gaussian distributed data 
# suspicious = detect_fan_out_patterns(df, time_freq='1H', z_threshold=3)
# print(f"Found {len(suspicious)} suspicious patterns")
# print(suspicious[['From Bank', 'Account', 'Timestamp', 'unique_recipients', 'z_score']])

#Detect suspicious patterns percentile based. Performs better on non-Gaussion distributed data. 
# df = detect_fan_out_groups_percentile(df, time_freq='1min', threshold=99.9)
# df = df.where(df.is_outlier == True)
# df = df.dropna(how='all') 
# print("Number of outliers detected (Percentile method):", df['is_outlier'].sum())
# print("\nPercentile Method Results:")
# print(df.head()) 
# save_df_to_csv(df, "percentile_result.csv", folderPath)

############################ generic micsh helping hand code ###################################
# df = (df.where(df['Account'] =='1004286A8').dropna(how='all'))
# save_df_to_csv(df, "account_with_many_transactions.csv", folderPath)
#save_df_to_csv(percentile_result, "percentile_test", folderPath, index=False)


# save_df_to_csv(grouped_df, "forocular.csv", folderPath)
#df = get_file_head_as_df(filePath, n=10, encoding='utf-8')

# grouped_df = preprocess_and_group(df, time_freq='10H')
# grouped_df = grouped_df.where(grouped_df['unique_recipients'] > 3)
# grouped_df = grouped_df.dropna(how='all') 












