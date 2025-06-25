import pandas as pd
import csv
import networkx as nx
from scipy import stats
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import io
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
''
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

#################################Function calls##############################################
filePath = "/Users/perliljekvist/Documents/Python/IBM AML/Data/HI-Small_Trans.csv"
folderPath = "/Users/perliljekvist/Documents/Python/IBM AML/Data/"

df = read_csv_custom(filePath, nrows=20000)
grouped_df = preprocess_and_group(df,'48H')

# Check distribution: Source account -> nof destination accounts before choosing method of detection
#plot_unique_recipient_histogram(grouped_df.where(grouped_df['unique_recipients'] > 1))
#plot_group_distributions(grouped_df)
          
#Detect suspicious patterns z-score based. No suitable for non-Gaussian distributed data 
# suspicious = detect_fan_out_patterns(df, time_freq='1H', z_threshold=3)
# print(f"Found {len(suspicious)} suspicious patterns")
# print(suspicious[['From Bank', 'Account', 'Timestamp', 'unique_recipients', 'z_score']])

#Detect suspicious patterns percentile based. Performs better on non-Gaussion distributed data. 
percentile_result = detect_fan_out_groups_percentile(df, time_freq='1min', threshold=99.9)
print("Number of outliers detected (Percentile method):", percentile_result['is_outlier'].sum())
print("\nPercentile Method Results:")
print(percentile_result.head()) 

############################ generic micsh helping hand code ###################################
# df = (df.where(df['Account'] =='1004286A8').dropna(how='all'))
# save_df_to_csv(df, "account_with_many_transactions.csv", folderPath)
#save_df_to_csv(percentile_result, "percentile_test", folderPath, index=False)


# save_df_to_csv(grouped_df, "forocular.csv", folderPath)
#df = get_file_head_as_df(filePath, n=10, encoding='utf-8')

# grouped_df = preprocess_and_group(df, time_freq='10H')
# grouped_df = grouped_df.where(grouped_df['unique_recipients'] > 3)
# grouped_df = grouped_df.dropna(how='all') 












