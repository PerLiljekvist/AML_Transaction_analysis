import pandas as pd
import matplotlib as plt
from paths_and_stuff import * 
from helpers import * 

newDir = create_new_folder(folderPath, 'accounts_nodes_edges_2025-07-26')

outlier_account = "800043C00"
outlier_accounts = [outlier_account]

df = load_rows_for_account(filePath, outlier_account,sep=",")

df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='ignore')
df['Amount Paid'] = pd.to_numeric(df['Amount Paid'], errors='ignore')
df['Amount Received'] = pd.to_numeric(df['Amount Received'], errors='ignore')
df['Account'] = df['Account'].astype(str)
df['Payment Format'] = df['Payment Format'].astype(str)

filtered = df[
    df['Account'] == outlier_account
]

summary = filtered.groupby('Account').agg({
    'Amount Paid': ['count', 'sum', 'mean', 'max'],
    'Payment Format': lambda x: x.value_counts().to_dict()
})

#print(summary)

for acct in outlier_accounts:
    # Filter transactions for this account
    acct_data = filtered[filtered['Account'] == acct]
    # Convert timestamps to just the date (no time)
    acct_data['Date'] = acct_data['Timestamp'].dt.date
    # Group by date, summing the amounts
    daily_amount = acct_data.groupby('Date')['Amount Paid'].sum()
    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(daily_amount.index, daily_amount.values, marker='o')
    plt.xlabel('Date')
    plt.ylabel('Total Amount Paid')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
