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


# ----------------------------
# 1. Load and Preprocess Data. 
# ----------------------------

# Sample data file from your original message
from io import StringIO

data = """Timestamp;From Bank;Account;To Bank;Account.1;Amount Received;Receiving Currency;Amount Paid;Payment Currency;Payment Format;Is Laundering
2022-09-01 00:20:00;10;8000EBD30;10;8000EBD30;369734;US Dollar;369734;US Dollar;Reinvestment;0
2022-09-01 00:20:00;3208;8000F4580;1;8000F5340;1;US Dollar;1;US Dollar;Cheque;0
2022-09-01 00:00:00;3209;8000F4670;3209;8000F4670;1467557;US Dollar;1467557;US Dollar;Reinvestment;0
2022-09-01 00:02:00;12;8000F5030;12;8000F5030;280697;US Dollar;280697;US Dollar;Reinvestment;0
2022-09-01 00:06:00;10;8000F5200;10;8000F5200;3668297;US Dollar;3668297;US Dollar;Reinvestment;0
2022-09-01 00:03:00;1;8000F5AD0;1;8000F5AD0;616244;US Dollar;616244;US Dollar;Reinvestment;0
2022-09-01 00:08:00;1;8000EBAC0;1;8000EBAC0;1426;US Dollar;1426;US Dollar;Reinvestment;0
2022-09-01 00:16:00;1;8000EC1E0;1;8000EC1E0;1186;US Dollar;1186;US Dollar;Reinvestment;0
2022-09-01 00:26:00;12;8000EC280;2439;8017BF800;766;US Dollar;766;US Dollar;Credit Card;0
2022-09-01 00:21:00;1;8000EDEC0;211050;80AEF5310;38371;US Dollar;38371;US Dollar;Credit Card;0
2022-09-01 00:04:00;1;8000F4510;11813;8011305D0;982;US Dollar;982;US Dollar;Credit Card;0
2022-09-01 00:04:00;1;8000F47F0;1;8000F47F0;938;US Dollar;938;US Dollar;Reinvestment;0
2022-09-01 00:08:00;1;8000F4FE0;245335;812ED62E0;401;US Dollar;401;US Dollar;Credit Card;0
2022-09-01 00:17:00;10;80012FD90;36056;812ED6380;10670;US Dollar;10670;US Dollar;Credit Card;0
2022-09-01 00:11:00;12;80012FE00;13037;805B34210;54;US Dollar;54;US Dollar;Credit Card;0
2022-09-01 00:09:00;1;80012FE50;1;80012FE50;394423229;US Dollar;394423229;US Dollar;Reinvestment;0"""

#df = pd.read_csv(StringIO(data), delimiter=';')

#folderOfTheDay = create_new_folder(folderPath, "20250715_account_nw")
df = read_csv_custom(filePath, nrows=50000)
#df = df.where(df['Is Laundering'] == 1).dropna().count()
# print(df['Is Laundering'])


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

# ----------------------------
# 3. Apply Anomaly Detection
# ----------------------------

## 3.1 K-Means
print("\n[+] Running K-Means...")
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# Add anomaly score and flag based on distance to assigned cluster centroid
df['kmeans_anomaly_score'] = np.linalg.norm(
    X_scaled - kmeans.cluster_centers_[df['kmeans_cluster']], axis=1
)
top_n_kmeans = 3  # number to flag as anomaly
df['kmeans_anomaly_flag'] = 0
df.loc[df['kmeans_anomaly_score'].nlargest(top_n_kmeans).index, 'kmeans_anomaly_flag'] = 1

## 3.2 DBSCAN
print("[+] Running DBSCAN...")
dbscan = DBSCAN(eps=1.5, min_samples=2)
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

## 3.3 K-Nearest Neighbors (anomaly scores)
print("[+] Running KNN anomaly scoring...")
knn = NearestNeighbors(n_neighbors=3)
knn.fit(X_scaled)
distances, _ = knn.kneighbors(X_scaled)
df['knn_anomaly_score'] = distances.mean(axis=1)
top_n_knn = 3
df['knn_anomaly_flag'] = 0
df.loc[df['knn_anomaly_score'].nlargest(top_n_knn).index, 'knn_anomaly_flag'] = 1

## 3.4 Isolation Forest
print("[+] Running Isolation Forest...")
iso = IsolationForest(contamination=0.2, random_state=42)
df['iso_pred'] = iso.fit_predict(X_scaled)
df['iso_score'] = iso.decision_function(X_scaled)

# ----------------------------
# 4. Show Results
# ----------------------------

print("\n[+] Detected anomalies:")
print("  - KMeans anomaly (top {}):".format(top_n_kmeans), df['kmeans_anomaly_flag'].sum())
print("  - DBSCAN flagged:", (df['dbscan_cluster'] == -1).sum())
print("  - KNN scores (top {}):".format(top_n_knn), df['knn_anomaly_flag'].sum())
print("  - IsolationForest flagged:", (df['iso_pred'] == -1).sum())

# Show table of potential anomalies
anomalies_df = df[
    (df['dbscan_cluster'] == -1) |
    (df['knn_anomaly_flag'] == 1) |
    (df['iso_pred'] == -1) |
    (df['kmeans_anomaly_flag'] == 1)
]

print("\n[+] Top anomalies found:")
print(anomalies_df[['Amount_Paid', 'Amount_Diff', 'kmeans_anomaly_score', 'knn_anomaly_score', 
                    'iso_score', 'dbscan_cluster', 'iso_pred', 'kmeans_anomaly_flag']])

# ----------------------------
# 5. Optional Plot (pairplot)
# ----------------------------

sns.pairplot(df, vars=['Amount_Paid', 'Amount_Diff', 'Hour'], hue='iso_pred', palette='coolwarm')
plt.suptitle('Isolation Forest Results (-1 = anomaly)', y=1.02)
plt.show()
