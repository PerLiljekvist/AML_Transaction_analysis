#!/usr/bin/env python
"""
Run 3 pyOD anomaly detection algorithms on preprocessed AML data
and produce:
  - per-transaction anomaly columns (scores + flags + consensus)
  - a simple text report
  - CSV files with the full output and top consensus anomalies
  - aggregated CSV report comparing algorithms + laundering label
  - NEW: compact CSV report describing which features were used by the models
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from helpers import *
from paths_and_stuff import *
import time


# ===========================
# Config
# ===========================
start = time.time()
PATH = create_new_folder(folderPath, datetime.now().strftime("%Y-%m-%d"))

INPUT_FILE  = PATH + "/tx_pre_model_with_account_context_pre_model.csv"  # <- change me
OUTPUT_FILE = PATH + "/tx_with_pyod_anomalies.csv"                       # full output
TOP_FILE    = PATH + "/report_top_consensus_anomalies.csv"               # only top consensus anomalies

# Aggregated report
AGG_REPORT_FILE = PATH + "/report_anomaly_aggregate.csv"

# NEW: feature report
FEATURE_REPORT_FILE = PATH + "/report_features_used.csv"

CSV_SEP     = ";"         # your sample uses semicolon
CONTAM      = 0.005       # expected outlier share
RANDOM_SEED = 42


# ===========================
# Core ensemble logic
# ===========================
def run_pyod_ensemble(
    df: pd.DataFrame,
    feature_cols=None,
    contamination: float = 0.01,
    random_state: int = 42,
):
    """
    Run 3 pyOD anomaly detection algorithms on preprocessed & scaled features.

    Adds to returned DataFrame:
      - iforest_score, iforest_is_outlier
      - lof_score,     lof_is_outlier
      - copod_score,   copod_is_outlier
      - anomaly_votes             (0-3)
      - consensus_anomaly         (True if votes >= 2)

    Returns
    -------
    df_out : pd.DataFrame
        Original df with new columns added.
    summary_df : pd.DataFrame
        Summary: anomalies per algorithm + consensus.
    vote_dist : pd.DataFrame
        Distribution of anomaly_votes (0, 1, 2, 3).
    feature_info : dict
        Metadata about the features actually used (after exclusions + numeric filtering).
    """
    d = df.copy()

    # 1) Decide which columns to use as features
    if feature_cols is None:
        non_features = {
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Receiving Currency",
            "Is Laundering",  # label; don't use when fitting unsupervised models
        }
        requested_feature_cols = [c for c in d.columns if c not in non_features]
    else:
        requested_feature_cols = list(feature_cols)

    # 2) Keep only numeric subset of those requested feature columns
    numeric_feature_cols = list(d[requested_feature_cols].select_dtypes(include=[np.number]).columns)

    # 3) Build X
    X = d[numeric_feature_cols].to_numpy()
    n = len(d)

    # 4) Feature metadata
    feature_info = {
        "n_rows": int(n),
        "n_total_columns_in_input": int(d.shape[1]),
        "n_requested_feature_cols": int(len(requested_feature_cols)),
        "n_numeric_feature_cols_used": int(len(numeric_feature_cols)),
        "requested_feature_cols": requested_feature_cols,
        "numeric_feature_cols_used": numeric_feature_cols,
        "non_numeric_requested_but_excluded": [c for c in requested_feature_cols if c not in numeric_feature_cols],
    }

    # --------------------------
    # Helper: fit model & attach results
    # --------------------------
    def _fit_and_attach(model, name: str):
        """
        Fits a pyOD model and adds:
          <name>_score        : decision_scores_ (higher = more anomalous)
          <name>_is_outlier   : labels_ (1=outlier, 0=inlier) as bool
        """
        model.fit(X)
        scores = model.decision_scores_          # larger = more abnormal
        labels = model.labels_                   # 1 = outlier, 0 = inlier
        d[f"{name}_score"] = scores
        d[f"{name}_is_outlier"] = labels.astype(bool)

    # --------------------------
    # Algorithm 1: IForest
    # --------------------------
    iforest = IForest(
        contamination=contamination,
        random_state=random_state,
    )
    _fit_and_attach(iforest, "iforest")

    # --------------------------
    # Algorithm 2: LOF
    # --------------------------
    lof = LOF(
        contamination=contamination,
        n_neighbors=20,
    )
    _fit_and_attach(lof, "lof")

    # --------------------------
    # Algorithm 3: COPOD
    # --------------------------
    copod = COPOD(
        contamination=contamination,
    )
    _fit_and_attach(copod, "copod")

    # --------------------------
    # Consensus logic
    # --------------------------
    vote_cols = ["iforest_is_outlier", "lof_is_outlier", "copod_is_outlier"]
    d["anomaly_votes"] = d[vote_cols].sum(axis=1)  # int 0-3
    d["consensus_anomaly"] = d["anomaly_votes"] >= 2

    # --------------------------
    # Summary report
    # --------------------------
    rows = []
    for alg_name, col in [
        ("IForest", "iforest_is_outlier"),
        ("LOF", "lof_is_outlier"),
        ("COPOD", "copod_is_outlier"),
    ]:
        k = int(d[col].sum())
        rows.append(
            {
                "algorithm": alg_name,
                "n_anomalies": k,
                "share_of_data": k / n if n else np.nan,
            }
        )

    # Consensus row
    k_cons = int(d["consensus_anomaly"].sum())
    rows.append(
        {
            "algorithm": "Consensus (>=2 of 3)",
            "n_anomalies": k_cons,
            "share_of_data": k_cons / n if n else np.nan,
        }
    )

    summary_df = pd.DataFrame(rows)

    # Vote distribution (0,1,2,3)
    vote_dist = (
        d["anomaly_votes"]
        .value_counts()
        .sort_index()
        .rename_axis("votes")
        .reset_index(name="count")
    )
    vote_dist["share_of_data"] = vote_dist["count"] / n if n else np.nan

    return d, summary_df, vote_dist, feature_info


# ===========================
# Aggregated CSV report
# ===========================
def build_agg_report(df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated report comparing:
      - anomalies per algorithm
      - laundering count/rate among those anomalies (if Is Laundering exists)
      - vote-based breakdown (votes=2, votes=3, consensus>=2)

    Returns a DataFrame suitable for CSV export.
    """
    d = df_out.copy()
    n_total = len(d)

    # robust laundering flag (handles strings "1", floats, etc.)
    has_laundry = "Is Laundering" in d.columns
    if has_laundry:
        is_laundry = (
            pd.to_numeric(d["Is Laundering"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        )
    else:
        is_laundry = pd.Series(False, index=d.index)

    def _row(name: str, mask: pd.Series) -> dict:
        n_flag = int(mask.sum())
        n_lau = int((mask & is_laundry).sum()) if has_laundry else np.nan
        rate_lau = (n_lau / n_flag) if (has_laundry and n_flag > 0) else (0.0 if has_laundry else np.nan)
        return {
            "group": name,
            "n_flagged": n_flag,
            "share_of_data": (n_flag / n_total) if n_total > 0 else np.nan,
            "n_laundering": n_lau,
            "laundering_share_among_flagged": rate_lau,
        }

    rows = []
    rows.append(_row("IForest anomalies", d["iforest_is_outlier"].astype(bool)))
    rows.append(_row("LOF anomalies", d["lof_is_outlier"].astype(bool)))
    rows.append(_row("COPOD anomalies", d["copod_is_outlier"].astype(bool)))

    rows.append(_row("Votes = 2 (exactly)", d["anomaly_votes"].eq(2)))
    rows.append(_row("Votes = 3 (exactly)", d["anomaly_votes"].eq(3)))
    rows.append(_row("Consensus (>=2)", d["consensus_anomaly"].astype(bool)))

    if has_laundry:
        rows.append({
            "group": "All transactions (label prevalence)",
            "n_flagged": n_total,
            "share_of_data": 1.0,
            "n_laundering": int(is_laundry.sum()),
            "laundering_share_among_flagged": float(is_laundry.mean()) if n_total > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ===========================
# NEW: Compact feature report (CSV)
# ===========================
def build_feature_report(feature_info: dict) -> pd.DataFrame:
    """
    Produces a compact, stakeholder-friendly CSV describing what features were used.

    Output format: one row per feature used, plus a small summary section at the top.
    """
    # Summary rows (same columns as detail, but feature_name blank)
    summary_rows = [
        {"section": "SUMMARY", "key": "n_rows", "value": feature_info.get("n_rows"), "feature_name": ""},
        {"section": "SUMMARY", "key": "n_total_columns_in_input", "value": feature_info.get("n_total_columns_in_input"), "feature_name": ""},
        {"section": "SUMMARY", "key": "n_requested_feature_cols", "value": feature_info.get("n_requested_feature_cols"), "feature_name": ""},
        {"section": "SUMMARY", "key": "n_numeric_feature_cols_used", "value": feature_info.get("n_numeric_feature_cols_used"), "feature_name": ""},
        {"section": "SUMMARY", "key": "n_non_numeric_excluded", "value": len(feature_info.get("non_numeric_requested_but_excluded", [])), "feature_name": ""},
    ]

    used = feature_info.get("numeric_feature_cols_used", [])
    excluded = feature_info.get("non_numeric_requested_but_excluded", [])

    detail_used = [
        {"section": "FEATURE_USED", "key": "", "value": "", "feature_name": c}
        for c in used
    ]

    detail_excl = [
        {"section": "REQUESTED_BUT_EXCLUDED_NON_NUMERIC", "key": "", "value": "", "feature_name": c}
        for c in excluded
    ]

    return pd.DataFrame(summary_rows + detail_used + detail_excl)


# ===========================
# Main script
# ===========================
def main():
    # 1) Load preprocessed + (ideally) scaled data
    print(f"Reading input data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep=CSV_SEP)

    # 2) Optional: explicitly list feature columns
    feature_cols = None

    # 3) Run ensemble
    df_out, summary, vote_dist, feature_info = run_pyod_ensemble(
        df,
        feature_cols=feature_cols,
        contamination=CONTAM,
        random_state=RANDOM_SEED,
    )

    # 4) Print report to stdout
    print("\n=== Anomalies per algorithm + consensus ===")
    print(summary.to_string(index=False))

    print("\n=== Vote distribution (0–3 algorithms flagging anomaly) ===")
    print(vote_dist.to_string(index=False))

    # 4b) NEW: Save feature report CSV
    feat_report = build_feature_report(feature_info)
    print(f"\nSaving feature report to: {FEATURE_REPORT_FILE}")
    feat_report.to_csv(FEATURE_REPORT_FILE, sep=CSV_SEP, index=False)

    # 5) Save full, unaggregated output
    print(f"\nSaving full output with anomaly columns to: {OUTPUT_FILE}")
    df_out.to_csv(OUTPUT_FILE, sep=CSV_SEP, index=False)

    # 5c) Save aggregated CSV report
    agg_report = build_agg_report(df_out)
    print(f"\nSaving aggregated anomaly/laundering report to: {AGG_REPORT_FILE}")
    agg_report.to_csv(AGG_REPORT_FILE, sep=CSV_SEP, index=False)

    # 6) Save a small file with the "top" consensus anomalies
    top_consensus = df_out[df_out["consensus_anomaly"]].copy()

    if not top_consensus.empty:
        # ---- laundering sanity check ----
        if "Is Laundering" in top_consensus.columns:
            n_cons = len(top_consensus)
            is_laundry = (
                pd.to_numeric(top_consensus["Is Laundering"], errors="coerce")
                .fillna(0)
                .astype(int)
                .eq(1)
            )
            n_laundry = int(is_laundry.sum())
            laundry_rate = n_laundry / n_cons if n_cons > 0 else 0.0

            print("\n=== Consensus anomalies vs laundering label ===")
            print(f"Consensus anomalies        : {n_cons}")
            print(f"Labeled laundering (Is=1)  : {n_laundry}")
            print(f"Laundering share           : {laundry_rate:.2%}")

            # Add explicit boolean flag for clarity in output file
            top_consensus["is_laundry_flag"] = is_laundry
        else:
            print("\nWARNING: 'Is Laundering' column not found.")
            top_consensus["is_laundry_flag"] = False

        # ---- ranking ----
        top_consensus["avg_score"] = (
            top_consensus[["iforest_score", "lof_score", "copod_score"]]
            .mean(axis=1)
        )

        top_consensus = top_consensus.sort_values(
            ["anomaly_votes", "avg_score"],
            ascending=[False, False],
        )

        print(f"Saving top consensus anomalies to: {TOP_FILE}")
        top_consensus.to_csv(TOP_FILE, sep=CSV_SEP, index=False)
    else:
        print("No consensus anomalies (>=2 algos) found; not writing top file.")

    print("\nExecution time:", time.time() - start, "seconds")


if __name__ == "__main__":
    main()
