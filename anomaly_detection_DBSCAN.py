import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plts
import seaborn as sns
from paths_and_stuff import *
from helpers import *
from simple_aml_functions import *

# ----------------------------
# 1. Load and Preprocess Data. 
# ----------------------------

#folderOfTheDay = create_new_folder(folderPath, "20250715_account_nw")
df = read_csv_custom(filePath, nrows=40000)
# df = df.where(df['Is Laundering'] == 1).dropna().count()
# print(df['Is Laundering'])

# ----------------------------
# 2. Feature Engineerings
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

# ----------------------------
# 3. Apply Anomaly Detection
# ----------------------------

## 3.2 DBSCAN
print("[+] Running DBSCAN...")
dbscan = DBSCAN(eps=1.5, min_samples=2)
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

# ----------------------------
# 4. Show Results
# ----------------------------

print("\n[+] Detected anomalies:")
print("  - DBSCAN flagged:", (df['dbscan_cluster'] == -1).sum())

# Show table of potential anomalies
anomalies_df = df[
    (df['dbscan_cluster'] == -1) 
]

print("\n[+] Top anomalies found:")
print(anomalies_df[['dbscan_cluster']])

# ----------------------------
# 5. Optional Plot (pairplot)
# ----------------------------

plt.figure(figsize=(10, 6))

# Assign a color to each cluster (including noise, which is -1)
unique_clusters = df["dbscan_cluster"].unique()
palette = sns.color_palette("husl", n_colors=len(unique_clusters))
color_map = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}

for cluster_id in unique_clusters:
    cluster_points = df[df['dbscan_cluster'] == cluster_id]
    plt.scatter(cluster_points['Amount_Paid'], cluster_points['Amount_Diff'],
                s=100, label=f'Cluster {cluster_id}', color=color_map[cluster_id])

plt.title('DBSCAN Clustering Results')
plt.xlabel('Amount_Paid')
plt.ylabel('Amount_Diff')
plt.legend()
plt.show()


# ----------------------------
# 6. Optional Plot: PCA 2D Projection of all features
# ----------------------------

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df['PCA1'], df['PCA2'],
    c=df['dbscan_cluster'], cmap='tab10', alpha=0.7, edgecolor='k'
)
plt.title('DBSCAN Clustering Results (PCA 2D projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster ID (-1 = anomaly/noise)')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Optional Plot: t-SNE 2D Projection of all features
# ----------------------------

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)
df['TSNE1'] = X_tsne[:, 0]
df['TSNE2'] = X_tsne[:, 1]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df['TSNE1'], df['TSNE2'],
    c=df['dbscan_cluster'], cmap='tab10', alpha=0.7, edgecolor='k'
)
plt.title('DBSCAN Clustering Results (t-SNE 2D projection)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Cluster ID (-1 = anomaly/noise)')
plt.grid(True)
plt.tight_layout()
plt.show()
