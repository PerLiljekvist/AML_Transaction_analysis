import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from paths_and_stuff import *
from helpers import *
from aml_functions import *
from io import StringIO

# ----------------------------
# 1. Load and Preprocess Data. 
# ----------------------------

df = read_csv_custom(filePath, nrows=10000)

# ----------------------------
# 2. Feature Engineering
# ----------------------------

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

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Amount_Diff'] = abs(df['Amount_Paid'] - df['Amount_Received'])
df['Same_Account'] = (df['From_Account'] == df['To_Account']).astype(int)
df['Same_Bank'] = (df['From_Bank'] == df['To_Bank']).astype(int)

df = pd.get_dummies(df, columns=['Payment_Format'], drop_first=True)

features = [
    'Amount_Received', 'Amount_Paid', 'Amount_Diff', 'Hour',
    'Same_Bank', 'Same_Account'
] + [col for col in df.columns if col.startswith('Payment_Format_')]

X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## 3.4 Isolation Forest
print("[+] Running Isolation Forest...")
iso = IsolationForest(contamination=0.2, random_state=42)
df['iso_pred'] = iso.fit_predict(X_scaled)
df['iso_score'] = iso.decision_function(X_scaled)

# ----------------------------
# 4. Show Results
# ----------------------------

print("\n[+] Detected anomalies:")

print("  - IsolationForest flagged:", (df['iso_pred'] == -1).sum())

# Show table of potential anomalies
anomalies_df = df[(df['iso_pred'] == -1)]

print("\n[+] Top anomalies found:")
print(anomalies_df[[ 'iso_score', 'iso_pred']])

# ----------------------------
# 5. Optional Plot (pairplot)
# ----------------------------

sns.pairplot(df, vars=['Amount_Paid', 'Amount_Diff', 'Hour'], hue='iso_pred', palette='coolwarm')
plt.suptitle('Isolation Forest Results (-1 = anomaly)', y=1.02)
plt.show()
