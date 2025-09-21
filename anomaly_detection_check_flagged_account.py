import pandas as pd
import matplotlib.pyplot as plt   # <-- fixed import
from paths_and_stuff import * 
from helpers import * 
from anomaly_detection_gettx_for_account import *

newDir = create_new_folder(folderPath, 'accounts_nodes_edges_2025-07-26')

outlier_account = "800C365B0"
outlier_accounts = [outlier_account]

# Load data (your helper decides how sep is used)
df = load_rows_for_account(filePath, outlier_account, sep=",")

# Parse / cast
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df['Amount Paid'] = pd.to_numeric(df['Amount Paid'], errors='coerce')
df['Amount Received'] = pd.to_numeric(df['Amount Received'], errors='coerce')
df['Account'] = df['Account'].astype(str)
df['Account.1'] = df['Account.1'].astype(str)
df['Payment Format'] = df['Payment Format'].astype(str)

# OUTBOUND subset (money sent by the outlier account)
filtered_out = df[df['Account'] == outlier_account].copy()

# INBOUND subset (money received by the outlier account)
filtered_in = df[df['Account.1'] == outlier_account].copy()

# Optional quick summary on outbound (kept from your code)
summary = filtered_out.groupby('Account').agg({
    'Amount Paid': ['count', 'sum', 'mean', 'max'],
    'Payment Format': lambda x: x.value_counts().to_dict()
})

# Plot daily totals for both outbound and inbound
for acct in outlier_accounts:
    # Outbound amounts by day
    if not filtered_out.empty:
        filtered_out['Date'] = filtered_out['Timestamp'].dt.date
        daily_out = filtered_out.groupby('Date')['Amount Paid'].sum()
    else:
        daily_out = pd.Series(dtype='float64')

    # Inbound amounts by day
    if not filtered_in.empty:
        filtered_in['Date'] = filtered_in['Timestamp'].dt.date
        daily_in = filtered_in.groupby('Date')['Amount Received'].sum()
    else:
        daily_in = pd.Series(dtype='float64')

    # Align date indices (include all dates that appear in either series)
    all_dates = sorted(set(daily_out.index).union(set(daily_in.index)))
    daily_out = daily_out.reindex(all_dates, fill_value=0)
    daily_in = daily_in.reindex(all_dates, fill_value=0)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.title(f"Payments for Account {acct}")
    plt.plot(all_dates, daily_out.values, marker='o', label='Outbound (Amount Paid)')
    plt.plot(all_dates, daily_in.values, marker='o', linestyle='--', label='Inbound (Amount Received)')
    plt.xlabel('Date')
    plt.ylabel('Total Amount')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
