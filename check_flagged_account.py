import pandas as pd
import matplotlib as plt
from paths_and_stuff import * 
from helpers import * 

newDir = create_new_folder(folderPath, 'accounts_nodes_edges_2025-07-26')

outlier_accounts = ['100428660']

df = read_csv_custom(filePath, nrows=400000)
df['Timestamp2'] = df['Timestamp'].astype('datetime64[ns]').dt.date
#df2 = df.groupby('Timestamp2')
print(df['Timestamp2'].unique())
quit()

filtered = df[
    (df.groupby['Account'].isin(outlier_accounts))
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
    plt.title(f'Total Amount Transferred per Day for Account {acct}')
    plt.xlabel('Date')
    plt.ylabel('Total Amount Paid')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()