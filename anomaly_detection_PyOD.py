"""
AML/CFT Anomaly Detection Comparison (PyOD)
-------------------------------------------
- Loads transaction data
- Runs multiple anomaly detection algorithms
- Produces report:
    * # anomalies per algorithm
    * Overlaps (pairwise + consensus)
    * Output CSVs with flags
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
import itertools
import os
from helpers import * 
from paths_and_stuff import *

# ---------------- CONFIG ----------------
INPUT_FILE = "transactions_sample.csv"   # Update path
OUTPUT_DIR = "anomaly_results"
CONTAMINATION = 0.05   # Expected % anomalies (adjust to dataset size)
NUMERIC_FEATURES = ["Amount Received", "Amount Paid"]
CATEGORICAL_FEATURES = ["Payment Format"]
# ----------------------------------------


def load_and_prepare_data(filepath):
    """Load CSV and prepare features for modeling."""
    df = read_csv_custom(filePath, nrows=1000)

    # One-hot encode Payment Format
    if CATEGORICAL_FEATURES:
        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    # Feature matrix
    features = [col for col in NUMERIC_FEATURES if col in df.columns]
    features += [c for c in df.columns if c not in features and c not in [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Receiving Currency", "Payment Currency", "Is Laundering"
    ]]

    X = df[features].fillna(0).values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


def run_algorithms(X):
    """Run PyOD algorithms and return flags."""
    algorithms = {
        "IForest": IForest(contamination=CONTAMINATION, random_state=42),
        "LOF": LOF(contamination=CONTAMINATION),
        "COPOD": COPOD(contamination=CONTAMINATION),
        "PCA": PCA(contamination=CONTAMINATION),
        "KNN": KNN(contamination=CONTAMINATION),
        "HBOS": HBOS(contamination=CONTAMINATION)
    }

    results = {}
    for name, model in algorithms.items():
        model.fit(X)
        results[name] = model.labels_  # 0 = normal, 1 = anomaly
    return results


def analyze_results(df, results):
    """Summarize anomalies and overlaps."""
    res_df = df.copy()
    for name, labels in results.items():
        res_df[f"Anomaly_{name}"] = labels

    # Count anomalies per algorithm
    counts = {name: labels.sum() for name, labels in results.items()}

    # Overlap counts
    overlap = {}
    algo_names = list(results.keys())
    for r in range(2, len(algo_names) + 1):
        for combo in itertools.combinations(algo_names, r):
            mask = np.all([results[a] == 1 for a in combo], axis=0)
            overlap[" & ".join(combo)] = mask.sum()

    return res_df, counts, overlap


def save_report(res_df, counts, overlap):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save detailed results
    res_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_results.csv"), index=False)

    # Save summary
    summary = pd.DataFrame([
        {"Algorithm": k, "Anomalies Found": v} for k, v in counts.items()
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_counts.csv"), index=False)

    overlap_df = pd.DataFrame([
        {"Algorithms": k, "Shared Anomalies": v} for k, v in overlap.items()
    ])
    overlap_df.to_csv(os.path.join(OUTPUT_DIR, "summary_overlap.csv"), index=False)

    print("✅ Reports saved in:", OUTPUT_DIR)
    print("\n--- Summary Counts ---")
    print(summary)
    print("\n--- Overlaps ---")
    print(overlap_df)


if __name__ == "__main__":
    df, X = load_and_prepare_data(INPUT_FILE)
    results = run_algorithms(X)
    res_df, counts, overlap = analyze_results(df, results)
    save_report(res_df, counts, overlap)
