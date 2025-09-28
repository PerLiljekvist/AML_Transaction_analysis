import pandas as pd
from collections import deque, defaultdict
from pathlib import Path
from paths_and_stuff import * 
from helpers import * 
from anomaly_detection_get_tx_for_account import *


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
    edges_fname: str = "edges.csv",
    k: int = 1,
    include_alter_alter_edges: bool = True
):
    """
    Build a k-step undirected ego network centered on `suspicious_account` and export:
      - nodes.csv : one column 'Id' listing nodes (excluding the ego itself)
      - edges.csv : columns 'Source','Target','Type','tx_count','amount_sum'
        * exactly one undirected edge per pair within the k-step ball
        * tx_count = number of tx between the two nodes (both directions)
        * amount_sum = aggregated money over those tx (see amount_pref)

    amount_pref:
      - "paid"       -> sum of Amount Paid (falls back to Received if Paid missing)
      - "received"   -> sum of Amount Received (falls back to Paid if Received missing)
      - "both_sum"   -> sum of (Paid + Received) per row (useful if they differ)

    Notes
    -----
    - Reachability is computed on an undirected version of the graph.
    - Set `include_alter_alter_edges=False` to only keep edges where one endpoint is the ego.
    - Self-transfers (source == target) are ignored.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    # Defensive copy & normalize types
    df = df.copy()
    suspicious_account = str(suspicious_account)
    df[source_col] = df[source_col].astype(str)
    df[target_col] = df[target_col].astype(str)

    # Handle empty case early
    if df.empty:
        nodes_df = pd.DataFrame(columns=["Id"])
        edges_df = pd.DataFrame(columns=["Source", "Target", "Type", "tx_count", "amount_sum"])
        out_path_nodes, out_path_edges = _write_gephi_csvs(nodes_df, edges_df, out_dir, nodes_fname, edges_fname)
        return nodes_df, edges_df, {"nodes_csv": str(out_path_nodes), "edges_csv": str(out_path_edges)}

    # Prepare amount columns as numeric (ensure presence)
    for col in (amount_paid_col, amount_received_col):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    # Choose per-row value used for aggregation
    if amount_pref == "paid":
        df["_value"] = df[amount_paid_col]
        missing_paid = df[amount_paid_col].isna()
        df.loc[missing_paid, "_value"] = df.loc[missing_paid, amount_received_col]
        df["_value"] = df["_value"].fillna(0)
    elif amount_pref == "received":
        df["_value"] = df[amount_received_col]
        missing_recv = df[amount_received_col].isna()
        df.loc[missing_recv, "_value"] = df.loc[missing_recv, amount_paid_col]
        df["_value"] = df["_value"].fillna(0)
    elif amount_pref == "both_sum":
        df["_value"] = df[amount_paid_col].fillna(0) + df[amount_received_col].fillna(0)
    else:
        raise ValueError("amount_pref must be one of: 'paid', 'received', 'both_sum'")

    # Build undirected endpoint pairs (u,v) with u < v; drop self-loops
    a = df[source_col]
    b = df[target_col]
    u = a.where(a <= b, b)
    v = b.where(a <= b, a)
    pairs = pd.DataFrame({"u": u, "v": v, "_value": df["_value"]})
    pairs = pairs.loc[pairs["u"] != pairs["v"]]

    if pairs.empty:
        nodes_df = pd.DataFrame(columns=["Id"])
        edges_df = pd.DataFrame(columns=["Source", "Target", "Type", "tx_count", "amount_sum"])
        out_path_nodes, out_path_edges = _write_gephi_csvs(nodes_df, edges_df, out_dir, nodes_fname, edges_fname)
        return nodes_df, edges_df, {"nodes_csv": str(out_path_nodes), "edges_csv": str(out_path_edges)}

    # ---- BFS to depth k on undirected graph ----
    # Adjacency (undirected)
    adj = defaultdict(set)
    for uu, vv in pairs[["u", "v"]].itertuples(index=False):
        adj[uu].add(vv)
        adj[vv].add(uu)

    # If ego not in graph at all, return empties
    if suspicious_account not in adj:
        nodes_df = pd.DataFrame(columns=["Id"])
        edges_df = pd.DataFrame(columns=["Source", "Target", "Type", "tx_count", "amount_sum"])
        out_path_nodes, out_path_edges = _write_gephi_csvs(nodes_df, edges_df, out_dir, nodes_fname, edges_fname)
        return nodes_df, edges_df, {"nodes_csv": str(out_path_nodes), "edges_csv": str(out_path_edges)}

    visited = {suspicious_account}
    depth = {suspicious_account: 0}
    q = deque([suspicious_account])

    while q:
        cur = q.popleft()
        if depth[cur] == k:
            continue
        for nbr in adj[cur]:
            if nbr not in visited:
                visited.add(nbr)
                depth[nbr] = depth[cur] + 1
                q.append(nbr)

    ego_ball = visited  # includes ego
    node_set = sorted(n for n in ego_ball if n != suspicious_account)  # exclude ego from nodes.csv

    # ---- Aggregate edges and restrict to the k-step node set ----
    # Aggregate globally first (fast), then filter to edges fully inside (ego ∪ node_set)
    agg = (
        pairs.groupby(["u", "v"], as_index=False)
             .agg(tx_count=("u", "size"), amount_sum=("_value", "sum"))
    )

    allowed = set(node_set) | {suspicious_account}
    sub_edges = agg[(agg["u"].isin(allowed)) & (agg["v"].isin(allowed))].copy()

    if not include_alter_alter_edges:
        sub_edges = sub_edges[(sub_edges["u"] == suspicious_account) | (sub_edges["v"] == suspicious_account)]

    # If nothing survived, return empties (but still write CSVs)
    if sub_edges.empty:
        nodes_df = pd.DataFrame({"Id": node_set})
        edges_df = pd.DataFrame(columns=["Source", "Target", "Type", "tx_count", "amount_sum"])
        out_path_nodes, out_path_edges = _write_gephi_csvs(nodes_df, edges_df, out_dir, nodes_fname, edges_fname)
        # Clean temp
        df.drop(columns=["_value"], inplace=True, errors="ignore")
        return nodes_df, edges_df, {"nodes_csv": str(out_path_nodes), "edges_csv": str(out_path_edges)}

    # Build Gephi-style outputs
    nodes_df = pd.DataFrame({"Id": node_set}).reset_index(drop=True)
    edges_df = sub_edges.rename(columns={"u": "Source", "v": "Target"})
    edges_df.insert(2, "Type", "Undirected")
    edges_df = edges_df[["Source", "Target", "Type", "tx_count", "amount_sum"]]

    # Write CSVs
    out_path_nodes, out_path_edges = _write_gephi_csvs(nodes_df, edges_df, out_dir, nodes_fname, edges_fname)

    # Clean temp
    df.drop(columns=["_value"], inplace=True, errors="ignore")

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
