"""
Lightweight one-file EDA runner (conceptual separation via sections)

Designed for generic tabular data, including AML tx-style datasets like:

Timestamp;From Bank;Account;To Bank;Account.1;Amount Received;Receiving Currency;Amount Paid;Payment Currency;Payment Format;Is Laundering
...

Outputs a folder with per-step subfolders containing CSV/JSON and optional PNG plots.

Dependencies: pandas, numpy, matplotlib
Optional: statsmodels (for VIF); if missing, VIF step will be skipped with a warning.
"""

# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib is used only if plots are enabled
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class EDAConfig:
    # Step selection
    only: Optional[List[str]] = None           # run ONLY these steps (if set)
    exclude: Optional[List[str]] = None        # skip these steps

    # Output / reporting
    save_plots: bool = True
    save_tables: bool = True
    save_json_summary: bool = True

    # Performance knobs
    sample_n_rows: Optional[int] = 200_000     # for plot-heavy steps; None = no sampling
    random_state: int = 42

    # Missingness
    missing_warn_threshold: float = 0.30       # warn if missing ratio >= this

    # Outliers (numeric only)
    outlier_method: str = "iqr"                # "iqr" or "mad"
    outlier_iqr_k: float = 1.5
    outlier_mad_z: float = 3.5
    outlier_min_nonnull: int = 30

    # Distributions
    dist_max_numeric_cols_plot: int = 12
    dist_max_categorical_cols_plot: int = 12
    dist_max_categories_show: int = 15
    dist_hist_bins: int = 50

    # Scaling / encoding recommendations
    id_like_unique_ratio: float = 0.98
    cat_low_cardinality: int = 20
    cat_high_cardinality: int = 200
    skew_abs_threshold: float = 1.0            # suggests log/robust scaling
    range_ratio_threshold: float = 1_000.0     # suggests scaling if max/min magnitude large

    # Multicollinearity (numeric only)
    corr_method: str = "pearson"               # "pearson" or "spearman"
    corr_abs_threshold: float = 0.85
    corr_max_numeric_cols: int = 60
    compute_vif: bool = False                  # can be expensive
    vif_max_numeric_cols: int = 25
    vif_nan_policy: str = "drop_rows"          # "drop_rows" or "fill_median"


# =============================================================================
# CONTEXT / TYPE PROFILING
# =============================================================================
@dataclass
class EDAContext:
    out_dir: Path
    run_id: str
    df_sample: pd.DataFrame

    # Column groups
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    bool_cols: List[str] = field(default_factory=list)
    id_like_cols: List[str] = field(default_factory=list)
    constant_cols: List[str] = field(default_factory=list)

    # Other metadata
    n_rows: int = 0
    n_cols: int = 0

    def log(self, msg: str) -> None:
        print(f"[EDA] {msg}")


def _safe_slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s[:120] if len(s) > 120 else s


def _infer_types(df: pd.DataFrame, cfg: EDAConfig) -> Dict[str, List[str]]:
    """
    Lightweight type inference:
    - numeric: pandas numeric dtypes
    - datetime: datetime64[ns] dtypes (caller can parse beforehand)
    - bool: boolean/bool dtypes
    - categorical: object/category + anything not numeric/datetime/bool
    - id-like: very high uniqueness or name heuristic
    - constant: nunique <= 1 (excluding NaN)
    """
    cols = list(df.columns)

    bool_cols = [c for c in cols if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_boolean_dtype(df[c])]
    datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c not in bool_cols]

    # everything else becomes categorical by default
    categorical_cols = [c for c in cols if c not in set(bool_cols + datetime_cols + numeric_cols)]

    # constant columns
    constant_cols = []
    for c in cols:
        s = df[c]
        # nunique(dropna=True) can be expensive but acceptable for lightweight EDA; you can sample if needed
        if s.dropna().nunique() <= 1:
            constant_cols.append(c)

    # id-like columns (high uniqueness or name pattern)
    id_name_re = re.compile(r"(?:^|[_\s])(?:id|uuid|guid|iban|account|acct|customer|client|transaction|tx)(?:$|[_\s])",
                            re.IGNORECASE)
    id_like_cols = []
    n = len(df)
    for c in cols:
        if c in numeric_cols or c in datetime_cols or c in bool_cols:
            continue
        s = df[c]
        nonnull = s.notna().sum()
        if nonnull == 0:
            continue
        uniq = s.dropna().nunique()
        uniq_ratio = uniq / max(nonnull, 1)
        if uniq_ratio >= cfg.id_like_unique_ratio or id_name_re.search(f" {c} ") is not None:
            id_like_cols.append(c)

    # keep id-like as subset of categorical (still categorical, but flagged)
    return dict(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        bool_cols=bool_cols,
        id_like_cols=sorted(set(id_like_cols)),
        constant_cols=sorted(set(constant_cols)),
    )


def make_context(df: pd.DataFrame, out_dir: Path, cfg: EDAConfig) -> EDAContext:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # sample for plot-heavy computations (keep full df for tables if you prefer; here we keep sample only in ctx)
    if cfg.sample_n_rows is not None and len(df) > cfg.sample_n_rows:
        df_sample = df.sample(n=cfg.sample_n_rows, random_state=cfg.random_state)
    else:
        df_sample = df

    info = _infer_types(df_sample, cfg)
    ctx = EDAContext(
        out_dir=out_dir,
        run_id=run_id,
        df_sample=df_sample,
        numeric_cols=info["numeric_cols"],
        categorical_cols=info["categorical_cols"],
        datetime_cols=info["datetime_cols"],
        bool_cols=info["bool_cols"],
        id_like_cols=info["id_like_cols"],
        constant_cols=info["constant_cols"],
        n_rows=len(df),
        n_cols=df.shape[1],
    )
    return ctx


# =============================================================================
# RESULT MODEL
# =============================================================================
@dataclass
class StepResult:
    name: str
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)  # store matplotlib Figure objects
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# STEPS
# =============================================================================
def step_missingness(df: pd.DataFrame, ctx: EDAContext, cfg: EDAConfig) -> StepResult:
    r = StepResult(name="missingness")

    miss_cnt = df.isna().sum()
    miss_ratio = miss_cnt / max(len(df), 1)

    tbl = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": miss_cnt.values,
        "missing_ratio": miss_ratio.values,
        "n_unique_nonnull": [df[c].dropna().nunique() for c in df.columns],
    }).sort_values(["missing_ratio", "missing_count"], ascending=False)

    r.tables["missing_by_column"] = tbl
    r.metrics["n_columns_with_missing"] = int((miss_cnt > 0).sum())

    bad = tbl[tbl["missing_ratio"] >= cfg.missing_warn_threshold]["column"].tolist()
    if bad:
        r.warnings.append(
            f"{len(bad)} columns have missing_ratio >= {cfg.missing_warn_threshold:.0%}: {bad[:15]}{'...' if len(bad)>15 else ''}"
        )
    return r


def _mad_zscore_outliers(x: pd.Series, z: float) -> Tuple[float, float, pd.Series]:
    """
    Robust z-score using MAD. Returns (center, scale, outlier_mask).
    """
    s = pd.to_numeric(x, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan, np.nan, pd.Series(False, index=x.index)

    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    if mad == 0:
        return med, 0.0, pd.Series(False, index=x.index)

    # 0.6745 makes MAD consistent with std under normality
    rz = 0.6745 * (pd.to_numeric(x, errors="coerce") - med) / mad
    mask = rz.abs() > z
    mask = mask.fillna(False)
    return med, mad, mask


def _iqr_outliers(x: pd.Series, k: float) -> Tuple[float, float, pd.Series]:
    s = pd.to_numeric(x, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan, np.nan, pd.Series(False, index=x.index)

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return q1, q3, pd.Series(False, index=x.index)

    lo = q1 - k * iqr
    hi = q3 + k * iqr
    v = pd.to_numeric(x, errors="coerce")
    mask = (v < lo) | (v > hi)
    mask = mask.fillna(False)
    return lo, hi, mask


def step_outliers(df: pd.DataFrame, ctx: EDAContext, cfg: EDAConfig) -> StepResult:
    r = StepResult(name="outliers")

    if not ctx.numeric_cols:
        r.warnings.append("No numeric columns detected; skipping outlier analysis.")
        return r

    rows = []
    for c in ctx.numeric_cols:
        s = df[c]
        nonnull = s.notna().sum()
        if nonnull < cfg.outlier_min_nonnull:
            continue

        if cfg.outlier_method.lower() == "mad":
            center, scale, mask = _mad_zscore_outliers(s, cfg.outlier_mad_z)
            method = f"MAD(z>{cfg.outlier_mad_z})"
            bound_lo, bound_hi = np.nan, np.nan
        else:
            bound_lo, bound_hi, mask = _iqr_outliers(s, cfg.outlier_iqr_k)
            method = f"IQR(k={cfg.outlier_iqr_k})"
            center, scale = np.nan, np.nan

        out_cnt = int(mask.sum())
        out_ratio = out_cnt / max(nonnull, 1)

        rows.append({
            "column": c,
            "method": method,
            "nonnull": int(nonnull),
            "outlier_count": out_cnt,
            "outlier_ratio": out_ratio,
            "bound_lo": bound_lo,
            "bound_hi": bound_hi,
            "mad_center": center,
            "mad_scale": scale,
            "min": pd.to_numeric(s, errors="coerce").min(),
            "max": pd.to_numeric(s, errors="coerce").max(),
        })

    if not rows:
        r.warnings.append("No numeric columns met minimum non-null requirement for outlier analysis.")
        return r

    tbl = pd.DataFrame(rows).sort_values("outlier_ratio", ascending=False)
    r.tables["outlier_summary_numeric"] = tbl
    r.metrics["numeric_cols_analyzed"] = int(len(tbl))
    return r


def step_scaling_encoding_reco(df: pd.DataFrame, ctx: EDAContext, cfg: EDAConfig) -> StepResult:
    r = StepResult(name="scaling_encoding")

    rows = []

    # Numeric recommendations
    for c in ctx.numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 10:
            continue

        # skewness (pandas uses Fisher-Pearson; ok for EDA)
        skew = float(x.skew()) if len(x) > 2 else np.nan

        # range ratio heuristic (avoid division by ~0; use percentiles for robustness)
        p01, p99 = float(x.quantile(0.01)), float(x.quantile(0.99))
        denom = max(abs(p01), 1e-12)
        range_ratio = abs(p99) / denom if denom > 0 else np.inf

        rec_scale = bool(range_ratio >= cfg.range_ratio_threshold)
        rec_robust = bool(abs(skew) >= cfg.skew_abs_threshold)
        rec_log = bool((x.min() >= 0) and (abs(skew) >= cfg.skew_abs_threshold))

        rationale = []
        if rec_scale:
            rationale.append(f"wide_scale(p99/p01≈{range_ratio:.1f})")
        if rec_robust:
            rationale.append(f"skew≈{skew:.2f}")
        if rec_log:
            rationale.append("nonnegative+skew→log1p_candidate")

        rows.append({
            "column": c,
            "type": "numeric",
            "suggest_scale": rec_scale or rec_robust,
            "suggest_robust_scaler": rec_robust,
            "suggest_log1p": rec_log,
            "skew": skew,
            "p01": p01,
            "p99": p99,
            "rationale": ";".join(rationale) if rationale else "",
        })

    # Categorical recommendations
    for c in ctx.categorical_cols + ctx.bool_cols:
        s = df[c]
        nonnull = s.notna().sum()
        if nonnull == 0:
            continue
        nunique = s.dropna().nunique()
        is_id_like = c in ctx.id_like_cols

        # heuristics for encoding approach
        if is_id_like:
            enc = "avoid_onehot_idlike"
            note = "High uniqueness / id-like; usually avoid one-hot; consider dropping or hashing if needed."
        elif nunique <= cfg.cat_low_cardinality:
            enc = "one_hot_ok"
            note = "Low cardinality; one-hot is usually fine."
        elif nunique >= cfg.cat_high_cardinality:
            enc = "high_cardinality"
            note = "High cardinality; avoid naive one-hot; consider target/hashing/frequency encoding."
        else:
            enc = "moderate_cardinality"
            note = "Moderate cardinality; one-hot possible but watch sparsity."

        rows.append({
            "column": c,
            "type": "categorical/bool",
            "n_unique_nonnull": int(nunique),
            "id_like": bool(is_id_like),
            "encoding_hint": enc,
            "note": note,
        })

    if not rows:
        r.warnings.append("No recommendations produced (maybe empty dataframe?).")
        return r

    r.tables["scaling_encoding_recommendations"] = pd.DataFrame(rows)
    return r


def step_distributions(df: pd.DataFrame, ctx: EDAContext, cfg: EDAConfig) -> StepResult:
    r = StepResult(name="distributions")

    # Summary stats tables
    if ctx.numeric_cols:
        num = df[ctx.numeric_cols].apply(pd.to_numeric, errors="coerce")
        desc = num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
        desc["skew"] = num.skew(numeric_only=True)
        desc["kurtosis"] = num.kurtosis(numeric_only=True)
        r.tables["numeric_summary"] = desc.reset_index().rename(columns={"index": "column"})
    else:
        r.warnings.append("No numeric columns detected; numeric distribution summary skipped.")

    cat_cols = ctx.categorical_cols + ctx.bool_cols
    if cat_cols:
        rows = []
        for c in cat_cols:
            s = df[c]
            nonnull = s.notna().sum()
            nunique = s.dropna().nunique()
            top = s.value_counts(dropna=True).head(cfg.dist_max_categories_show)
            rows.append({
                "column": c,
                "nonnull": int(nonnull),
                "n_unique_nonnull": int(nunique),
                "top_values": json.dumps(top.to_dict(), ensure_ascii=False),
            })
        r.tables["categorical_top_values"] = pd.DataFrame(rows).sort_values("n_unique_nonnull", ascending=False)
    else:
        r.warnings.append("No categorical/bool columns detected; categorical distribution summary skipped.")

    # Optional plots (use sample df from ctx for speed)
    if not cfg.save_plots:
        return r

    dps = ctx.df_sample

    # Numeric histograms (limited)
    for c in ctx.numeric_cols[: cfg.dist_max_numeric_cols_plot]:
        x = pd.to_numeric(dps[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 10:
            continue
        fig = plt.figure()
        plt.hist(x, bins=cfg.dist_hist_bins)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        r.figures[f"hist_{_safe_slug(c)}"] = fig

    # Categorical bar charts (limited; show top N)
    for c in cat_cols[: cfg.dist_max_categorical_cols_plot]:
        s = dps[c].dropna()
        if len(s) == 0:
            continue
        vc = s.value_counts().head(cfg.dist_max_categories_show)
        fig = plt.figure()
        plt.bar([str(k) for k in vc.index], vc.values)
        plt.title(f"Top categories: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        r.figures[f"bar_{_safe_slug(c)}"] = fig

    return r


def step_multicollinearity(df: pd.DataFrame, ctx: EDAContext, cfg: EDAConfig) -> StepResult:
    r = StepResult(name="multicollinearity")

    if not ctx.numeric_cols:
        r.warnings.append("No numeric columns detected; skipping multicollinearity.")
        return r

    num_cols = ctx.numeric_cols[: cfg.corr_max_numeric_cols]
    X = df[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Correlation matrix + high-corr pairs
    corr = X.corr(method=cfg.corr_method)
    r.tables["corr_matrix"] = corr

    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iat[i, j]
            if pd.isna(v):
                continue
            if abs(v) >= cfg.corr_abs_threshold:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "corr": float(v)})
    if pairs:
        r.tables["high_corr_pairs"] = pd.DataFrame(pairs).sort_values("corr", key=lambda s: s.abs(), ascending=False)
        r.warnings.append(f"Found {len(pairs)} high-correlation pairs with |corr| >= {cfg.corr_abs_threshold}.")
    else:
        r.notes.append(f"No pairs with |corr| >= {cfg.corr_abs_threshold} among first {len(num_cols)} numeric cols.")

    # Optional correlation heatmap (plots)
    if cfg.save_plots:
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(corr.values, aspect="auto")
        plt.colorbar()
        plt.title(f"Correlation heatmap ({cfg.corr_method})")
        plt.xticks(range(len(cols)), cols, rotation=90, fontsize=7)
        plt.yticks(range(len(cols)), cols, fontsize=7)
        plt.tight_layout()
        r.figures["corr_heatmap"] = fig

    # Optional VIF
    if not cfg.compute_vif:
        return r

    vif_cols = ctx.numeric_cols[: cfg.vif_max_numeric_cols]
    XV = df[vif_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    if cfg.vif_nan_policy == "fill_median":
        XV = XV.fillna(XV.median(numeric_only=True))
    else:
        XV = XV.dropna(axis=0)

    if len(XV) < 50 or XV.shape[1] < 2:
        r.warnings.append("VIF skipped: not enough rows after NaN handling or too few columns.")
        return r

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:
        r.warnings.append("statsmodels not installed; VIF skipped.")
        return r

    # Add constant column is not required for VIF in this function; it computes based on matrix columns.
    arr = XV.values.astype(float)
    vif_rows = []
    for i, c in enumerate(vif_cols):
        try:
            vif = float(variance_inflation_factor(arr, i))
        except Exception:
            vif = np.nan
        vif_rows.append({"column": c, "vif": vif})
    r.tables["vif"] = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    return r


# =============================================================================
# REPORT WRITER
# =============================================================================
def _write_step_result(step_dir: Path, res: StepResult, cfg: EDAConfig) -> Dict[str, Any]:
    step_dir.mkdir(parents=True, exist_ok=True)

    written = {"tables": [], "figures": [], "meta": {}}

    # Tables
    if cfg.save_tables and res.tables:
        for name, df in res.tables.items():
            p = step_dir / f"{_safe_slug(name)}.csv"
            df.to_csv(p, index=False)
            written["tables"].append(str(p.name))

    # Figures
    if cfg.save_plots and res.figures:
        for name, fig in res.figures.items():
            p = step_dir / f"{_safe_slug(name)}.png"
            try:
                fig.savefig(p, dpi=150, bbox_inches="tight")
                written["figures"].append(str(p.name))
            finally:
                # avoid memory leaks in long runs
                plt.close(fig)

    # Meta
    written["meta"] = {
        "metrics": res.metrics,
        "notes": res.notes,
        "warnings": res.warnings,
    }
    return written


def _write_summary(out_dir: Path, summary: Dict[str, Any]) -> None:
    p = out_dir / "summary.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


# =============================================================================
# RUNNER
# =============================================================================
StepFn = Callable[[pd.DataFrame, EDAContext, EDAConfig], StepResult]

STEPS: Dict[str, StepFn] = {
    "missingness": step_missingness,
    "outliers": step_outliers,
    "scaling_encoding": step_scaling_encoding_reco,
    "distributions": step_distributions,
    "multicollinearity": step_multicollinearity,
}


def _select_steps(cfg: EDAConfig) -> List[str]:
    names = list(STEPS.keys())

    if cfg.only:
        selected = [s for s in cfg.only if s in STEPS]
    else:
        selected = names

    if cfg.exclude:
        selected = [s for s in selected if s not in set(cfg.exclude)]

    return selected


def run_eda(
    df: pd.DataFrame,
    out_dir: str | Path,
    cfg: Optional[EDAConfig] = None,
) -> Dict[str, StepResult]:
    cfg = cfg or EDAConfig()
    out_dir = Path(out_dir)

    # Create run folder inside out_dir
    base_dir = out_dir
    ctx = make_context(df, base_dir, cfg)
    run_dir = base_dir / f"eda_{ctx.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx.out_dir = run_dir  # update to run folder

    ctx.log(f"Run folder: {run_dir}")
    ctx.log(f"Rows: {ctx.n_rows:,} | Cols: {ctx.n_cols}")
    ctx.log(f"Numeric: {len(ctx.numeric_cols)} | Categorical: {len(ctx.categorical_cols)} | Datetime: {len(ctx.datetime_cols)} | Bool: {len(ctx.bool_cols)}")
    if ctx.id_like_cols:
        ctx.log(f"ID-like cols: {ctx.id_like_cols[:10]}{'...' if len(ctx.id_like_cols)>10 else ''}")
    if ctx.constant_cols:
        ctx.log(f"Constant cols: {ctx.constant_cols[:10]}{'...' if len(ctx.constant_cols)>10 else ''}")

    selected = _select_steps(cfg)
    ctx.log(f"Selected steps: {selected}")

    results: Dict[str, StepResult] = {}
    artifacts: Dict[str, Any] = {}

    for step_name in selected:
        fn = STEPS[step_name]
        ctx.log(f"Running step: {step_name}")
        try:
            res = fn(df, ctx, cfg)
        except Exception as e:
            # fail-soft: capture error and continue
            res = StepResult(name=step_name)
            res.warnings.append(f"Step failed with error: {type(e).__name__}: {e}")
        results[step_name] = res

        step_dir = run_dir / step_name
        artifacts[step_name] = _write_step_result(step_dir, res, cfg)

    if cfg.save_json_summary:
        summary = {
            "run_id": ctx.run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rows": ctx.n_rows,
            "cols": ctx.n_cols,
            "column_groups": {
                "numeric_cols": ctx.numeric_cols,
                "categorical_cols": ctx.categorical_cols,
                "datetime_cols": ctx.datetime_cols,
                "bool_cols": ctx.bool_cols,
                "id_like_cols": ctx.id_like_cols,
                "constant_cols": ctx.constant_cols,
            },
            "selected_steps": selected,
            "artifacts": artifacts,
        }
        _write_summary(run_dir, summary)

    ctx.log("Done.")
    return results


# =============================================================================
# EXAMPLE USAGE (optional)
# =============================================================================
if __name__ == "__main__":
    # Minimal example for your AML-style CSV:
    #   - semicolon separator
    #   - parse Timestamp as datetime if present
    #
    # Edit the path and run: python eda_onefile.py
    path = None  # e.g. "transactions.csv"

    if path is None:
        print("Set `path` in __main__ to point to your CSV, or import and call run_eda(df, out_dir).")
    else:
        df = pd.read_csv(path, sep=";")
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        # Common: keep target/label column but it will be treated as numeric/bool; that’s fine for EDA.
        cfg = EDAConfig(
            # toggles
            exclude=None,                 # e.g. ["distributions"]
            only=None,                    # e.g. ["missingness", "multicollinearity"]

            # performance/plots
            sample_n_rows=200_000,
            save_plots=True,

            # multicollinearity
            corr_abs_threshold=0.85,
            compute_vif=False,
        )

        run_eda(df, out_dir="eda_output", cfg=cfg)
