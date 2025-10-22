# dbscan_tx_anomalies.py
# Run DBSCAN on a transaction features CSV and visualize results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from datetime import datetime
from helpers import *
from paths_and_stuff import *

# -------- Config (edit if needed) --------
OUTPUT_DIR = create_new_folder(folderPath, datetime.now().strftime("%Y-%m-%d"))    
INPUT_PATH = Path(OUTPUT_DIR + "/tx_features_only_2025-10-23.csv")   # your file
OUTPUT_PATH = Path(OUTPUT_DIR + "/tdbscan_results_.csv")
EPS = 8        # neighborhood radius in standardized space (↑ = fewer anomalies)
MIN_SAMPLES = 5   # min points to form a dense region (↑ = fewer/smaller clusters)

def main():
    # 1) Load CSV (let pandas sniff the delimiter: comma/semicolon/etc.)
    df = pd.read_csv(INPUT_PATH, sep=None, engine="python")

    # 2) Build feature matrix: numeric columns only, drop constants, fill NaNs with medians
    X = df.select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError("No numeric features found. Please include numeric columns in your file.")
    X = X.loc[:, X.nunique(dropna=False) > 1]           # drop zero-variance columns
    X = X.apply(lambda s: s.fillna(s.median()))         # simple, robust imputation
    if X.shape[1] == 0:
        raise ValueError("All numeric features were constant; DBSCAN needs variation.")

    # 3) Standardize features (DBSCAN is distance-based)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) Run DBSCAN (label -1 = anomaly/noise)
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    # 5) 2D projection for plotting only (PCA)
    pca = PCA(n_components=2, random_state=42)
    coords2d = pca.fit_transform(X_scaled)

    # 6) Attach results to original data and save
    out = df.copy()
    out["dbscan_label"] = labels
    out["is_anomaly"] = (labels == -1)
    out["pca_x"] = coords2d[:, 0]
    out["pca_y"] = coords2d[:, 1]
    out.to_csv(OUTPUT_PATH, index=False)

    # 7) Console table: show a compact anomalies view (top 25 rows)
    #    Show up to 10 non-numeric "context" columns + key result columns.
    non_num_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()[:10]
    cols_to_show = non_num_cols + ["dbscan_label", "is_anomaly", "pca_x", "pca_y"]
    anomalies = out[out["is_anomaly"]][cols_to_show].head(25)
    print("\n=== Anomalies preview (top 25) ===")
    if anomalies.empty:
        print("No anomalies found at current EPS/MIN_SAMPLES.")
    else:
        print(anomalies.to_string(index=False))

    # 8) Quick summary
    n_total = len(out)
    n_anom = int(out["is_anomaly"].sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("\n=== Summary ===")
    print(f"Total rows: {n_total}")
    print(f"Anomalies (-1): {n_anom} ({n_anom / max(n_total,1):.2%})")
    print(f"Clusters (excluding -1): {n_clusters}")
    print(f"Saved full results to: {OUTPUT_PATH.resolve()}")

    # 9) Simple scatter plot (2D PCA). Cross marker for anomalies.
    plt.figure(figsize=(8, 6))
    mask_anom = out["is_anomaly"]
    plt.scatter(out.loc[~mask_anom, "pca_x"], out.loc[~mask_anom, "pca_y"],
                s=10, alpha=0.6, label="Clustered")
    plt.scatter(out.loc[mask_anom, "pca_x"], out.loc[mask_anom, "pca_y"],
                s=30, alpha=0.9, marker="x", label="Anomalies (-1)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("DBSCAN on Transaction Features (2D PCA)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
