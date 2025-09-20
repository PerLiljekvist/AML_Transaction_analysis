import pandas as pd
from pathlib import Path
from paths_and_stuff import * 
from helpers import * 
from  anomaly_detection_gettx_for_account import *

import pandas as pd
from pathlib import Path

def build_ego_network_for_gephi(
    df: pd.DataFrame,
    suspicious_account: str,
    source_col: str = "Account",
    target_col: str = "Account.1",
    amount_paid_col: str = "Amount Paid",
    amount_received_col: str = "Amount Received",
    amount_pref: str = "paid",   # "paid", "received", or "both_sum"
    out_dir: str = ".",
    nodes_fname: str = "nodes.csv",
    edges_fname: str = "edges.csv"
):
    """
    Build an undirected ego network centered on `suspicious_account` and export:
      - nodes.csv : one column 'Id' listing counterparties only (plain node list)
      - edges.csv : columns 'Source','Target','Type','tx_count','amount_sum'
        * exactly one undirected edge per pair
        * tx_count = number of tx between the two nodes (both directions)
        * amount_sum = aggregated money over those tx (see amount_pref)

    amount_pref:
      - "paid"       -> sum of Amount Paid (falls back to Received if Paid missing)
      - "received"   -> sum of Amount Received (falls back to Paid if Received missing)
      - "both_sum"   -> sum of (Paid + Received) per row (useful if they differ)

    Assumes `df` contains at least the account columns; it can be pre-filtered or not.
    Self-transfers (source == target) are ignored in the edge list.
    """
    # Defensive copy & normalize types
    df = df.copy()
    suspicious_account = str(suspicious_account)
    df[source_col] = df[source_col].astype(str)
    df[target_col] = df[target_col].astype(str)

    # Keep only rows where suspicious appears on either side
    mask = (df[source_col] == suspicious_account) | (df[target_col] == suspicious_account)
    ego_df = df.loc[mask].copy()

    # Handle empty case early
    if ego_df.empty:
        nodes_df = pd.DataFrame(columns=["Id"])
        edges_df = pd.DataFrame(columns=["Source", "Target", "Type", "tx_count", "amount_sum"])
        out_path_nodes, out_path_edges = _write_gephi_csvs(nodes_df, edges_df, out_dir, nodes_fname, edges_fname)
        return nodes_df, edges_df, {"nodes_csv": str(out_path_nodes), "edges_csv": str(out_path_edges)}

    # Prepare amount columns as numeric
    for col in [amount_paid_col, amount_received_col]:
        if col in ego_df.columns:
            ego_df[col] = pd.to_numeric(ego_df[col], errors="coerce")
        else:
            ego_df[col] = 0.0  # ensure presence

    # Choose per-row value used for aggregation
    if amount_pref == "paid":
        # prefer paid; if NaN then use received
        ego_df["_value"] = ego_df[amount_paid_col].fillna(0)
        # if paid is zero but received is non-zero and paid was NaN, add received
        # simpler: take max of (paid, received) when one is missing—use where both provided equally
        missing_paid = ego_df[amount_paid_col].isna()
        ego_df.loc[missing_paid, "_value"] = ego_df.loc[missing_paid, amount_received_col].fillna(0)
    elif amount_pref == "received":
        missing_recv = ego_df[amount_received_col].isna()
        ego_df["_value"] = ego_df[amount_received_col].fillna(0)
        ego_df.loc[missing_recv, "_value"] = ego_df.loc[missing_recv, amount_paid_col].fillna(0)
    elif amount_pref == "both_sum":
        ego_df["_value"] = ego_df[amount_paid_col].fillna(0) + ego_df[amount_received_col].fillna(0)
    else:
        raise ValueError("amount_pref must be one of: 'paid', 'received', 'both_sum'")

    # Build undirected pairs by sorting endpoints per row
    # Also drop self-transfers for the edge list
    a = ego_df[source_col]
    b = ego_df[target_col]
    u = a.where(a <= b, b)  # lexicographically smaller first
    v = b.where(a <= b, a)

    pairs = pd.DataFrame({"u": u, "v": v})
    pairs["_value"] = ego_df["_value"]

    # Remove self-loops (u == v)
    pairs = pairs.loc[pairs["u"] != pairs["v"]]

    # Aggregate: one row per undirected edge
    agg = (
        pairs.groupby(["u", "v"], as_index=False)
             .agg(tx_count=("u", "size"),
                  amount_sum=("_value", "sum"))
    )

    # Build nodes list: counterparties connected to suspicious_account
    # From original ego_df, take the other side of each row where suspicious appears.
    cp_from_src = ego_df.loc[ego_df[source_col] == suspicious_account, target_col]
    cp_from_tgt = ego_df.loc[ego_df[target_col] == suspicious_account, source_col]
    counterparties = pd.concat([cp_from_src, cp_from_tgt], ignore_index=True)
    counterparties = (
        counterparties.dropna()
        .astype(str)
        .loc[lambda s: s != suspicious_account]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    nodes_df = pd.DataFrame({"Id": counterparties})

    # Build Gephi-style edges output (only edges touching the suspicious account)
    # Filter to edges where suspicious_account is one endpoint
    touching = agg[(agg["u"] == suspicious_account) | (agg["v"] == suspicious_account)].copy()
    if touching.empty:
        # If all rows were self-loops (shouldn't happen after filter), return empties
        edges_df = pd.DataFrame(columns=["Source", "Target", "Type", "tx_count", "amount_sum"])
    else:
        # Map to Source/Target with the other endpoint as counterparty
        touching["Source"] = suspicious_account
        touching["Target"] = touching.apply(lambda r: r["v"] if r["u"] == suspicious_account else r["u"], axis=1)
        edges_df = touching[["Source", "Target", "tx_count", "amount_sum"]].copy()
        edges_df.insert(2, "Type", "Undirected")

    # Write CSVs
    out_path_nodes, out_path_edges = _write_gephi_csvs(nodes_df, edges_df, out_dir, nodes_fname, edges_fname)

    # Clean temporary column
    if "_value" in ego_df.columns:
        ego_df.drop(columns=["_value"], inplace=True, errors="ignore")

    return nodes_df, edges_df, {"nodes_csv": str(out_path_nodes), "edges_csv": str(out_path_edges)}


def _write_gephi_csvs(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, out_dir: str, nodes_fname: str, edges_fname: str):
    """Helper to write nodes/edges CSVs and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / nodes_fname
    edges_path = out_dir / edges_fname
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    return nodes_path, edges_path


#--- Example usage ---
outlier_account = "100428660"
newDir = create_new_folder(folderPath, 'ego_nodes_edges_2025-09-20')

df = load_rows_for_account(filePath, outlier_account, sep=",")

nodes, edges, paths = build_ego_network_for_gephi(df, suspicious_account=outlier_account,
                                                  out_dir=newDir,
                                                  nodes_fname="nodes.csv",
                                                  edges_fname="edges.csv")
print(paths)


