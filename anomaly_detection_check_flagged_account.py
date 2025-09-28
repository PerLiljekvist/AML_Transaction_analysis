import pandas as pd
import matplotlib.pyplot as plt   # <-- fixed import
from paths_and_stuff import * 
from helpers import * 
from anomaly_detection_get_tx_for_account import *


def compute_account_stats(filtered_out: pd.DataFrame, filtered_in: pd.DataFrame) -> dict:
    # Counts
    outbound_tx = len(filtered_out)
    inbound_tx = len(filtered_in)
    total_tx = outbound_tx + inbound_tx

    # Amounts: outbound uses Amount Paid; inbound uses Amount Received
    amt_out = filtered_out['Amount Paid'] if 'Amount Paid' in filtered_out.columns else pd.Series(dtype='float64')
    amt_in  = filtered_in['Amount Received'] if 'Amount Received' in filtered_in.columns else pd.Series(dtype='float64')
    amounts = pd.concat([amt_out, amt_in], ignore_index=True)

    mean_amount = amounts.mean(skipna=True) if not amounts.empty else float('nan')
    max_amount  = amounts.max(skipna=True) if not amounts.empty else float('nan')

    # Unique counterparties: out -> Account.1, in -> Account
    cp_out = filtered_out['Account.1'].astype(str) if 'Account.1' in filtered_out.columns else pd.Series([], dtype='string')
    cp_in  = filtered_in['Account'].astype(str)   if 'Account'   in filtered_in.columns   else pd.Series([], dtype='string')
    counterparties = pd.concat([cp_out, cp_in], ignore_index=True).dropna()
    unique_counterparties = counterparties.nunique()

    return {
        "total_tx": int(total_tx),
        "outbound_tx": int(outbound_tx),
        "inbound_tx": int(inbound_tx),
        "mean_amount": None if pd.isna(mean_amount) else float(mean_amount),
        "max_single_tx": None if pd.isna(max_amount) else float(max_amount),
        "unique_counterparties": int(unique_counterparties),
    }

newDir = create_new_folder(folderPath, 'accounts_nodes_edges_2025-07-27')

outlier_account = "100428738"
outlier_accounts = [outlier_account]

# Load data (your helper decides how sep is used)
df, stats = load_rows_for_account(filePath, outlier_account, sep=",") 

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
        filtered_out = filtered_out.copy()
        filtered_out['Date'] = filtered_out['Timestamp'].dt.date
        daily_out = filtered_out.groupby('Date', dropna=True)['Amount Paid'].sum(min_count=1).fillna(0)
    else:
        daily_out = pd.Series(dtype='float64')

    # Inbound amounts by day
    if not filtered_in.empty:
        filtered_in = filtered_in.copy()
        filtered_in['Date'] = filtered_in['Timestamp'].dt.date
        daily_in = filtered_in.groupby('Date', dropna=True)['Amount Received'].sum(min_count=1).fillna(0)
    else:
        daily_in = pd.Series(dtype='float64')

    # Align date indices (include all dates that appear in either series)
    all_dates = sorted(set(daily_out.index).union(set(daily_in.index)))
    daily_out = daily_out.reindex(all_dates, fill_value=0)
    daily_in = daily_in.reindex(all_dates, fill_value=0)

    # --- NEW: compute aggregated stats for this account ---
    acc_stats = compute_account_stats(filtered_out, filtered_in)

    # Plot
# format helper
def _fmt(x):
    return "n/a" if (x is None or pd.isna(x)) else f"{x:,.2f}"

# build the multi-line stats string
stats_lines = [
    f"Total tx: {acc_stats['total_tx']:,}",
    f"Inbound: {acc_stats['inbound_tx']:,}   Outbound: {acc_stats['outbound_tx']:,}",
    f"Counterparties: {acc_stats['unique_counterparties']:,}",
    f"Mean amount: {_fmt(acc_stats['mean_amount'])}",
    f"Max single tx: {_fmt(acc_stats['max_single_tx'])}",
]
stats_text = "\n".join(stats_lines)   # <-- create the variable you pass below


# 1) Build a 2-column layout: left for text, right for the plot
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 3], wspace=0.05)

ax_text = fig.add_subplot(gs[0, 0])
ax_plot = fig.add_subplot(gs[0, 1])

# 2) Plot on ax_plot (change your plt.* to ax_plot.*)
ax_plot.plot(all_dates, daily_out.values, marker='o', label='Outbound (Amount Paid)')
ax_plot.plot(all_dates, daily_in.values, marker='o', linestyle='--', label='Inbound (Amount Received)')
ax_plot.set_title(f"Payments for Account {acct}")
ax_plot.set_xlabel('Date')
ax_plot.set_ylabel('Total Amount')
ax_plot.legend()
for label in ax_plot.get_xticklabels():
    label.set_rotation(45)

# 3) Put the stats in the left panel and hide its axes
ax_text.axis('off')
ax_text.text(
    0.0, 1.0, stats_text,
    transform=ax_text.transAxes,
    ha="left", va="top", fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.5", linewidth=1)
)

plt.show()



