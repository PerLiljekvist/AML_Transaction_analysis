# dbscan_tx_anomalies_benchmarked.py
# Run DBSCAN on a transaction feature set, exclude target flag from features,
# and benchmark anomaly labels against known "Is Laundering" flag.
# Saves a human-friendly summaries CSV at the same destination as the full results.

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

# -------- Config --------
DATE_FOR_PATHS_AND_FILES = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = create_new_folder(folderPath, DATE_FOR_PATHS_AND_FILES)
INPUT_PATH = Path(OUTPUT_DIR + "/tx_features_only_" + DATE_FOR_PATHS_AND_FILES + ".csv")
OUTPUT_PATH = Path(OUTPUT_DIR + "/dbscan_results_.csv")
SUMMARIES_PATH = Path(OUTPUT_DIR + "/dbscan_summaries_.csv")
EPS = 8
MIN_SAMPLES = 5


def _append_summary(rows, section, measure, value, notes=None):
    """Add one human-friendly summary row."""
    rows.append({
        "section": section,
        "measure": measure,
        "value": value,
        "notes": notes if notes is not None else ""
    })


def main():
    # 1) Load CSV
    df = pd.read_csv(INPUT_PATH, sep=None, engine="python")

    summaries = []

    # 2) Distribution of Is Laundering
    if "Is Laundering" in df.columns:
        y = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).astype(int)
        counts = y.value_counts().sort_index()
        pct = (counts / counts.sum() * 100).round(2)

        print("\n=== Distribution of 'Is Laundering' ===")
        dist_table = pd.DataFrame({
            "Label": counts.index.map({0: "0 (not laundering)", 1: "1 (laundering)"}),
            "Count": counts.values,
            "Percentage": pct.values
        })
        print(dist_table.to_string(index=False))

        for k in counts.index:
            label = "Laundering" if k == 1 else "Not laundering"
            _append_summary(summaries, "Laundering label distribution",
                            f"{label} — count", int(counts.loc[k]))
            _append_summary(summaries, "Laundering label distribution",
                            f"{label} — percentage", float(pct.loc[k]),
                            notes="Share of total rows")
        _append_summary(summaries, "Laundering label distribution", "Total rows", int(counts.sum()))
    else:
        print("\n[Warning] 'Is Laundering' column not found in dataset.")

    # 3) Build feature matrix (exclude Is Laundering)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Is Laundering" in feature_cols:
        feature_cols.remove("Is Laundering")

    X = df[feature_cols].copy()
    if X.empty:
        raise ValueError("No numeric features found after removing Is Laundering.")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.apply(lambda s: s.fillna(s.median()))

    # 4) Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5) DBSCAN
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    # 6) Optional PCA (commented out)
    # pca = PCA(n_components=2, random_state=42)
    # coords2d = pca.fit_transform(X_scaled)

    # 7) Attach results
    out = df.copy()
    out["dbscan_label"] = labels
    out["is_anomaly"] = (labels == -1)
    # out["pca_x"] = coords2d[:, 0]
    # out["pca_y"] = coords2d[:, 1]
    out.to_csv(OUTPUT_PATH, index=False)

    # 8) Benchmark if labels exist
    if "Is Laundering" in df.columns:
        y = pd.to_numeric(out["Is Laundering"], errors="coerce").fillna(0).astype(int)
        pred = out["is_anomaly"].astype(int)

        cross_tab = pd.crosstab(y, pred, margins=True)
        print("\n=== Benchmark: 'Is Laundering' vs. detected anomalies ===")
        cross_pretty = cross_tab.copy()
        cross_pretty.index = cross_pretty.index.map({0: "Label 0 (not laundering)", 1: "Label 1 (laundering)", "All": "All"})
        cross_pretty.columns = ["Detected as NOT anomaly", "Detected as anomaly", "All"]
        print(cross_pretty.to_string())

        for i in cross_tab.index:
            for j in cross_tab.columns:
                _append_summary(
                    summaries,
                    "Comparison of labels and detected anomalies",
                    f"{i} vs {j}",
                    int(cross_tab.loc[i, j])
                )

        # Correlation
        corr = np.corrcoef(y, pred)[0, 1]
        print(f"\nCorrelation between 'Is Laundering' and detected anomalies: {corr:.3f}")
        _append_summary(summaries, "Alignment metric",
                        "Correlation between 'Is Laundering' and detected anomalies",
                        float(round(corr, 6)))

        # Human-friendly counts
        true_pos  = int(((y == 1) & (pred == 1)).sum())
        false_pos = int(((y == 0) & (pred == 1)).sum())
        false_neg = int(((y == 1) & (pred == 0)).sum())
        true_neg  = int(((y == 0) & (pred == 0)).sum())

        _append_summary(summaries, "Benchmark metrics — counts",
                        "Laundering correctly flagged as anomaly", true_pos,
                        "True positives — correct alerts")
        _append_summary(summaries, "Benchmark metrics — counts",
                        "Non-laundering incorrectly flagged as anomaly", false_pos,
                        "False positives — unnecessary alerts")
        _append_summary(summaries, "Benchmark metrics — counts",
                        "Laundering missed by algorithm", false_neg,
                        "False negatives — undetected laundering")
        _append_summary(summaries, "Benchmark metrics — counts",
                        "Non-laundering correctly not flagged", true_neg,
                        "True negatives — normal correctly left alone")

        # Derived metrics
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        _append_summary(summaries, "Benchmark metrics — quality",
                        "Precision — of anomalies flagged, share that were actual laundering",
                        float(round(precision, 6)))
        _append_summary(summaries, "Benchmark metrics — quality",
                        "Recall — of all laundering cases, share flagged as anomalies",
                        float(round(recall, 6)))
        _append_summary(summaries, "Benchmark metrics — quality",
                        "F1 score — balance between precision and recall",
                        float(round(f1, 6)))

    # 9) Compact anomaly preview
    non_num_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()[:10]
    cols_to_show = non_num_cols + ["dbscan_label", "is_anomaly"]
    anomalies = out[out["is_anomaly"]][cols_to_show].head(25)
    print("\n=== Anomalies preview (top 25) ===")
    print("No anomalies found." if anomalies.empty else anomalies.to_string(index=False))

    # 10) Overall summary
    n_total = len(out)
    n_anom = int(out["is_anomaly"].sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print("\n=== Overall run summary ===")
    print(f"Total rows: {n_total}")
    print(f"Rows flagged as anomalies: {n_anom} ({n_anom / max(n_total,1):.2%})")
    print(f"Clusters (excluding anomalies): {n_clusters}")
    print(f"Saved full results to: {OUTPUT_PATH.resolve()}")

    _append_summary(summaries, "Overall run summary", "Total rows", n_total)
    _append_summary(summaries, "Overall run summary", "Rows flagged as anomalies", n_anom)
    _append_summary(summaries, "Overall run summary",
                    "Percentage of rows flagged as anomalies",
                    float(round(n_anom / max(n_total, 1), 6)))
    _append_summary(summaries, "Overall run summary",
                    "Number of clusters (excluding anomalies)", n_clusters)
    _append_summary(summaries, "DBSCAN settings",
                    "Neighborhood radius (eps)", float(EPS),
                    "Higher values usually mean fewer anomalies")
    _append_summary(summaries, "DBSCAN settings",
                    "Minimum samples to form a dense region", int(MIN_SAMPLES),
                    "Higher values usually mean fewer/smaller clusters")

    # 11) Save human-friendly summary
    pd.DataFrame(summaries).to_csv(SUMMARIES_PATH, index=False)
    print(f"Saved human-friendly summaries to: {SUMMARIES_PATH.resolve()}")

    # 12) Optional scatter (enable if PCA)
    # plt.figure(figsize=(8,6))
    # mask_anom = out["is_anomaly"]
    # plt.scatter(out.loc[~mask_anom, "pca_x"], out.loc[~mask_anom, "pca_y"],
    #             s=10, alpha=0.6, label="Clustered")
    # plt.scatter(out.loc[mask_anom, "pca_x"], out.loc[mask_anom, "pca_y"],
    #             s=30, alpha=0.9, marker="x", label="Anomalies (-1)")
    # plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    # plt.title("DBSCAN on Transaction Features (2D PCA)")
    # plt.legend(); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
