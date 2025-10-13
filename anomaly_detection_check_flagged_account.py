import os
import pandas as pd
import matplotlib.pyplot as plt  # <-- fixed import
from datetime import datetime

from paths_and_stuff import *
from helpers import *
from anomaly_detection_get_tx_for_account import *


def compute_account_stats(filtered_out: pd.DataFrame, filtered_in: pd.DataFrame) -> dict:
    """
    Compute summary stats for a single account given outbound (sent) and inbound (received) subsets.
    """
    # ---------------- Counts ----------------
    outbound_tx = len(filtered_out)
    inbound_tx = len(filtered_in)
    total_tx = outbound_tx + inbound_tx

    # ---------------- Amounts ----------------
    # outbound uses Amount Paid; inbound uses Amount Received
    amt_out = filtered_out['Amount Paid'] if 'Amount Paid' in filtered_out.columns else pd.Series(dtype='float64')
    amt_in = filtered_in['Amount Received'] if 'Amount Received' in filtered_in.columns else pd.Series(dtype='float64')
    amounts = pd.concat([amt_out, amt_in], ignore_index=True)
    mean_amount = amounts.mean(skipna=True) if not amounts.empty else float('nan')
    max_amount = amounts.max(skipna=True) if not amounts.empty else float('nan')

    # ---------------- Counterparties ----------------
    # Outbound counterparties are the receivers (Account.1) in outbound rows
    cp_out = (
        filtered_out['Account.1'].astype(str)
        if 'Account.1' in filtered_out.columns else pd.Series([], dtype='string')
    )
    # Inbound counterparties are the senders (Account) in inbound rows
    cp_in = (
        filtered_in['Account'].astype(str)
        if 'Account' in filtered_in.columns else pd.Series([], dtype='string')
    )

    unique_outbound_cps = int(cp_out.dropna().nunique()) if not cp_out.empty else 0
    unique_inbound_cps = int(cp_in.dropna().nunique()) if not cp_in.empty else 0

    # Total unique counterparties (union of both directions)
    counterparties = pd.concat([cp_out, cp_in], ignore_index=True).dropna()
    unique_counterparties_total = int(counterparties.nunique()) if not counterparties.empty else 0

    return {
        "total_tx": int(total_tx),
        "outbound_tx": int(outbound_tx),
        "inbound_tx": int(inbound_tx),
        "mean_amount": None if pd.isna(mean_amount) else float(mean_amount),
        "max_single_tx": None if pd.isna(max_amount) else float(max_amount),
        "unique_counterparties_total": unique_counterparties_total,
        "unique_outbound_counterparties": unique_outbound_cps,
        "unique_inbound_counterparties": unique_inbound_cps,
    }


# -------- Your existing setup --------
newDir = create_new_folder(folderPath, datetime.now().strftime("%Y-%m-%d"))
outlier_account = "1004288A0"
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

# -------- Plot + Excel export --------
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

    # --- Stats for this account ---
    acc_stats = compute_account_stats(filtered_out, filtered_in)

    # --- Build the two DataFrames to save ---
    # 1) Stats sheet (one row)
    stats_df = pd.DataFrame([{
        "Account": acct,
        "Total Tx": acc_stats["total_tx"],
        "Inbound Tx": acc_stats["inbound_tx"],
        "Outbound Tx": acc_stats["outbound_tx"],
        # Split counterparties + keep total
        "Unique Counterparties (Total)": acc_stats["unique_counterparties_total"],
        "Unique Inbound Counterparties": acc_stats["unique_inbound_counterparties"],
        "Unique Outbound Counterparties": acc_stats["unique_outbound_counterparties"],
        "Mean Amount": acc_stats["mean_amount"],
        "Max Single Tx": acc_stats["max_single_tx"]
    }])

    # 2) Daily series sheet (Date, Inbound, Outbound)
    daily_df = pd.DataFrame({
        "Date": pd.to_datetime(all_dates),  # ensure datetime for Excel formatting
        "Inbound": [float(v) for v in daily_in.values] if len(daily_in) else [],
        "Outbound": [float(v) for v in daily_out.values] if len(daily_out) else [],
    })

    # -------------------------------------------------------
    # NEW: Distribution tables (Payment Format & Currency)
    # -------------------------------------------------------
    # Combine inbound + outbound rows for THIS account
    acct_df = pd.concat([filtered_out, filtered_in], ignore_index=True)

    def make_distribution_df(series: pd.Series, name: str) -> pd.DataFrame:
        series = series.dropna().astype(str)
        if series.empty:
            return pd.DataFrame(columns=[name, "Count", "Percent"])
        counts = series.value_counts().sort_values(ascending=False)
        total = int(counts.sum())
        df_dist = counts.reset_index()
        df_dist.columns = [name, "Count"]
        df_dist["Percent"] = (df_dist["Count"] / total * 100).round(2)
        return df_dist

    fmt_df = make_distribution_df(acct_df.get("Payment Format", pd.Series(dtype=str)), "Payment Format")

    # Prefer 'Payment Currency'; fallback to 'Receiving Currency'
    if "Payment Currency" in acct_df.columns and acct_df["Payment Currency"].notna().any():
        curr_series = acct_df["Payment Currency"]
    elif "Receiving Currency" in acct_df.columns:
        curr_series = acct_df["Receiving Currency"]
    else:
        curr_series = pd.Series(dtype=str)
    curr_df = make_distribution_df(curr_series, "Currency")

    # --- Write Excel file (now with 2 extra sheets) ---
    out_xlsx = os.path.join(newDir, f"account_{acct}_summary.xlsx")

    # Try xlsxwriter; if not installed, fallback to openpyxl (keeps things robust)
    try:
        engine_name = "xlsxwriter"
        import xlsxwriter  # noqa: F401
    except ImportError:
        engine_name = "openpyxl"

    with pd.ExcelWriter(out_xlsx, engine=engine_name, datetime_format="yyyy-mm-dd") as writer:
        stats_df.to_excel(writer, sheet_name="Stats", index=False)
        daily_df.to_excel(writer, sheet_name="DailySeries", index=False)
        fmt_df.to_excel(writer, sheet_name="PaymentFormatDist", index=False)
        curr_df.to_excel(writer, sheet_name="PaymentCurrencyDist", index=False)

        # Optional: widen columns (if engine supports it)
        if engine_name == "xlsxwriter":
            ws_stats = writer.sheets["Stats"]
            ws_daily = writer.sheets["DailySeries"]
            ws_fmt = writer.sheets["PaymentFormatDist"]
            ws_curr = writer.sheets["PaymentCurrencyDist"]
            for ws in [ws_stats, ws_daily, ws_fmt, ws_curr]:
                ws.set_column(0, 2, 24)

    print(f"Saved Excel summary for {acct}: {out_xlsx}")

    # --- Plot ---
    # format helper
    def _fmt(x):
        return "n/a" if (x is None or pd.isna(x)) else f"{x:,.2f}"

    # build the multi-line stats string (now with split counterparties)
    stats_lines = [
        f"Total tx: {acc_stats['total_tx']:,}",
        f"Inbound: {acc_stats['inbound_tx']:,}    Outbound: {acc_stats['outbound_tx']:,}",
        f"Counterparties (Total): {acc_stats['unique_counterparties_total']:,}",
        f"  ├─ Inbound CPs:  {acc_stats['unique_inbound_counterparties']:,}",
        f"  └─ Outbound CPs: {acc_stats['unique_outbound_counterparties']:,}",
        f"Mean amount: {_fmt(acc_stats['mean_amount'])}",
        f"Max single tx: {_fmt(acc_stats['max_single_tx'])}",
    ]
    stats_text = "\n".join(stats_lines)

    # 1) Build a 2-column layout: left for text, right for the plot
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 3], wspace=0.05)
    ax_text = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    # 2) Plot on ax_plot
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
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.5", linewidth=1)
    )

    plt.show()

    # -------------------------------------------------------
    # Bar charts for Payment Format & Payment Currency
    # -------------------------------------------------------
    def _plot_bar_distribution(df_dist: pd.DataFrame, cat_col: str, title: str):
        """Plot a simple bar chart from a distribution dataframe with Count/Percent columns."""
        if df_dist.empty:
            print(f"No data available for {title}")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df_dist[cat_col], df_dist["Count"])
        ax.set_title(title)
        ax.set_xlabel(cat_col)
        ax.set_ylabel("Count")
        # Annotate bars with count and percent
        for i, (cnt, pct) in enumerate(zip(df_dist["Count"], df_dist["Percent"])):
            ax.text(i, cnt, f"{int(cnt)} ({pct:.1f}%)", ha="center", va="bottom")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    _plot_bar_distribution(fmt_df, "Payment Format", f"Distribution of Payment Format for Account {acct}")
    _plot_bar_distribution(curr_df, "Currency", f"Distribution of Payment Currency for Account {acct}")
