# anomaly_detection_isolation_forest_scatter.py
# Minimal IsolationForest + Scatter Plot (outliers vs inliers colored)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from helpers import *
from paths_and_stuff import *

# ------------------------------------------------------------
# Config (adjust as needed)
# ------------------------------------------------------------
FILE_PATH = filePath        # assumes you define filePath externally
NROWS = 10000             # rows to read
CONTAMINATION = 0.01        # expected anomaly proportion
TOP_K_PAYMENT_FORMATS = 6   # None to keep all; collapse rare to "Other" otherwise

# ------------------------------------------------------------
# 1) Load
# ------------------------------------------------------------
df = read_csv_custom(FILE_PATH, nrows=NROWS)

# ------------------------------------------------------------
# 2) Rename & Feature Engineering
# ------------------------------------------------------------
df.rename(columns={
    'Account': 'From_Account',
    'Account.1': 'To_Account',
    'Amount Received': 'Amount_Received',
    'Amount Paid': 'Amount_Paid',
    'Receiving Currency': 'Receiving_Currency',
    'Payment Currency': 'Payment_Currency',
    'Payment Format': 'Payment_Format',
    'From Bank': 'From_Bank',
    'To Bank': 'To_Bank',
    'Is Laundering': 'Label'
}, inplace=True)

# Parse timestamps and derive simple features
df['Timestamp'] = pd.to_datetime(df.get('Timestamp'), errors='coerce')
df['Hour'] = df['Timestamp'].dt.hour
df['Amount_Diff'] = (df['Amount_Paid'] - df['Amount_Received']).abs()
df['Same_Account'] = (df['From_Account'] == df['To_Account']).astype(int)
df['Same_Bank'] = (df['From_Bank'] == df['To_Bank']).astype(int)

# Keep original Payment Format for summaries/facets under a SAFE name
df['PaymentFormatOrig'] = df['Payment_Format'].astype(str)

# Optionally collapse rare payment formats
if TOP_K_PAYMENT_FORMATS is not None:
    top_vals = df['PaymentFormatOrig'].value_counts().nlargest(TOP_K_PAYMENT_FORMATS).index
    df.loc[~df['PaymentFormatOrig'].isin(top_vals), 'PaymentFormatOrig'] = 'Other'

# One-hot encode Payment_Format for modeling
df = pd.get_dummies(df, columns=['Payment_Format'], drop_first=True)

# ------------------------------------------------------------
# 3) Build feature matrix (numeric only)
# ------------------------------------------------------------
payment_dummies = [c for c in df.columns if c.startswith('Payment_Format_')]

features = [
    'Amount_Received', 'Amount_Paid', 'Amount_Diff', 'Hour',
    'Same_Bank', 'Same_Account'
] + payment_dummies

X = df[features].copy()

# Guardrails
non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric:
    raise TypeError(f"Non-numeric columns leaked into X: {non_numeric}")

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# ------------------------------------------------------------
# 4) IsolationForest
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(contamination=CONTAMINATION, random_state=42)
df['iso_pred'] = iso.fit_predict(X_scaled)           # -1 anomaly, 1 normal
df['iso_score'] = iso.decision_function(X_scaled)    # higher = more normal
df['anomaly'] = df['iso_pred']                       # expected by the plot

n_out = int((df['anomaly'] == -1).sum())
n_in  = int((df['anomaly'] ==  1).sum())
print(f"[+] IsolationForest flagged anomalies: {n_out} / {len(df)}")

try:
    print(df)
except NameError:
    print("[i] print_anomalies not found; showing first 5 anomalies:")
    cols_preview = ['Timestamp', 'From_Bank', 'To_Bank', 'Amount_Paid', 'Amount_Received', 'iso_score']
    cols_preview = [c for c in cols_preview if c in df.columns]
    print(df.loc[df['anomaly'] == -1, cols_preview].head())

# ------------------------------------------------------------
# 5) Scatter Plot (Outliers vs Inliers colored)
# ------------------------------------------------------------
def scatter_outlier_plot(data, x_var, y_var, flag_col='anomaly',
                         alpha=0.6, point_size=25):
    """
    Scatter plot where outliers (-1) are one color and inliers (1) another.
    """
    if flag_col not in data.columns:
        raise KeyError(f"'{flag_col}' not found in data.")

    plot_df = data[[x_var, y_var, flag_col]].dropna(subset=[x_var, y_var]).copy()

    n_out = int((plot_df[flag_col] == -1).sum())
    n_in  = int((plot_df[flag_col] ==  1).sum())

    print(f"Scatter Plot: {x_var} vs {y_var}")
    print(f"Outliers: {n_out} | Inliers: {n_in} | Total: {len(plot_df)}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x=x_var, y=y_var,
        hue=flag_col,
        palette={-1: "red", 1: "blue"},
        alpha=alpha,
        s=point_size
    )

    plt.title(f"IsolationForest Outlier Detection\nOutliers={n_out}, Inliers={n_in}")
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend(title="Class", labels=["Outlier (-1)", "Inlier (1)"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 6) Use the plot (Amount_Paid vs Amount_Received)
# ------------------------------------------------------------
scatter_outlier_plot(
    data=df,
    x_var='Amount_Paid',
    y_var='Amount_Received',
    flag_col='anomaly'
)
