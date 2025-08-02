import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from paths_and_stuff import *
from helpers import *
from simple_aml_functions import *
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
iso = IsolationForest(contamination=0.01, random_state=42)
df['iso_pred'] = iso.fit_predict(X_scaled)
df['iso_score'] = iso.decision_function(X_scaled)

# ----------------------------
# 4. Show Results
# ----------------------------

print("\n[+] Detected anomalies:")

print("  - IsolationForest flagged:", (df['iso_pred'] == -1).sum())

# Show table of potential anomalies

print_anomalies(df)
# anomalies_df = df[(df['iso_pred'] == -1)]

# print("\n[+] Top anomalies found:")
# print(anomalies_df[[ 'iso_score', 'iso_pred']])

# ----------------------------
# 5. Optional Plot (2D Projection using PCA)
# ----------------------------

# Reduce features to 2D using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Attach the PCA components for plotting
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df['PCA1'], df['PCA2'],
    c=df['iso_pred'], cmap='coolwarm', alpha=0.7, edgecolor='k'
)
plt.title('Isolation Forest Results (PCA 2D projection)\n(-1 = anomaly, 1 = normal)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=['Anomaly (-1)', 'Normal (1)'])
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ----------------------------
# Optional Plot: 2D Projection using t-SNE
# ----------------------------

# Create a t-SNE instance with common parameters
#tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)

# Fit and transform the scaled features
# X_tsne = tsne.fit_transform(X_scaled)

# Attach t-SNE components to the DataFrame for plotting
# df['TSNE1'] = X_tsne[:, 0]
# df['TSNE2'] = X_tsne[:, 1]

# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     df['TSNE1'], df['TSNE2'],
#     c=df['iso_pred'], cmap='coolwarm', alpha=0.7, edgecolor='k'
# )
# plt.title('Isolation Forest Results (t-SNE 2D projection)\n(-1 = anomaly, 1 = normal)')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.legend(handles=scatter.legend_elements()[0], labels=['Anomaly (-1)', 'Normal (1)'])
# plt.grid(True)
# plt.tight_layout()
# plt.show()


