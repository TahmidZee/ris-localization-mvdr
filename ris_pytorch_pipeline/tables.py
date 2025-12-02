# SPDX-License-Identifier: MIT
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Project config (for DR/jitter annotations)
from .configs import cfg

# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------
BENCH_DIR = Path("results_final/benches")
TABLE_DIR = Path("results_final/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# The order you want in tables/plots
METHOD_ORDER: List[str] = ["Hybrid", "Ramezani-MOD-MUSIC", "DCD-MUSIC", "NF-SubspaceNet"]

# Map group column → table ID prefix (for LaTeX labels / filenames)
GROUP_TABLE_ID = {
    "SNR": "B1",
    "K": "B2",
    "L": "B3",
    "codebook": "B4",
    "tag": "B5",
}

# Metrics we summarize
METRICS = ["phi", "theta", "rng"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_all_bench(mode: Optional[str] = None) -> pd.DataFrame:
    """
    Load all bench CSVs from BENCH_DIR and (optionally) filter by mode, e.g. "Blind-K" or "Oracle-K".
    Returns an empty DataFrame if no files exist.
    """
    if not BENCH_DIR.exists():
        return pd.DataFrame()

    files = sorted(list(BENCH_DIR.glob("*.csv")))
    if not files:
        return pd.DataFrame()

    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df["__source_file"] = p.name
            dfs.append(df)
        except Exception:
            # ignore broken files
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # normalize column names a bit (common variants)
    # keep originals if they already exist
    if "SNR" not in df.columns and "snr" in df.columns:
        df["SNR"] = df["snr"]
    if "who" not in df.columns and "method" in df.columns:
        df["who"] = df["method"]

    if mode and "mode" in df.columns:
        df = df[df["mode"] == mode].copy()

    return df


def _ensure_listlike(x):
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return x
    return [x]


def _dr_notes_from_cfg() -> str:
    """
    Fairness annotation string using cfg (DR/jitter knobs).
    """
    phase = getattr(cfg, "PHASE_JITTER", None)
    amp = getattr(cfg, "AMP_JITTER", None)
    drop = getattr(cfg, "DROPOUT_RATE", None)
    return f"phase={phase}, amp={amp}, drop={drop}"


def _summarize(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    blind_k: bool,
) -> pd.DataFrame:
    """
    Summarize (Median, IQR, N, K_Mode, DR_Notes, Status) by [group_col, who] for the given metric.
    - If a method has no finite values for that group, Status='N/A' and Median/IQR=None.
    - Respects METHOD_ORDER for consistent column ordering later.
    """
    if df.empty or group_col not in df.columns or metric not in df.columns or "who" not in df.columns:
        return pd.DataFrame(columns=[group_col, "Method", "Median", "IQR", "N", "K_Mode", "DR_Notes", "Status"])

    rows = []
    k_mode = "Blind-MDL" if blind_k else "Oracle"
    dr_notes = _dr_notes_from_cfg()

    # keep only relevant cols
    use_cols = [c for c in [group_col, "who", metric, "status"] if c in df.columns]
    d = df[use_cols].copy()

    for (gval, who), sdf in d.groupby([group_col, "who"], dropna=False):
        # Finite values for metric
        vals = pd.to_numeric(sdf[metric], errors="coerce")
        vals = vals[np.isfinite(vals)]
        n = int(vals.count())
        if n > 0:
            med = float(np.median(vals))
            q1 = float(np.quantile(vals, 0.25))
            q3 = float(np.quantile(vals, 0.75))
            iqr = q3 - q1
            status = "OK"
        else:
            med = None
            iqr = None
            status = "N/A"

        rows.append({
            group_col: gval,
            "Method": who,
            "Median": med,
            "IQR": iqr,
            "N": n,
            "K_Mode": k_mode,
            "DR_Notes": dr_notes,
            "Status": status,
        })

    out = pd.DataFrame(rows)

    # Order methods
    out["Method"] = pd.Categorical(out["Method"], categories=METHOD_ORDER, ordered=True)
    out = out.sort_values([group_col, "Method"]).reset_index(drop=True)
    return out


def _write_csv_and_tex(
    table_df: pd.DataFrame,
    group_col: str,
    metric: str,
    mode: str,
    table_id_prefix: str,
) -> Tuple[Path, Path]:
    """
    Save the long-form summary to CSV, and also a LaTeX pivot of medians only.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    safe_mode = mode.replace("-", "")
    stem = f"{table_id_prefix.lower()}_{metric}_{safe_mode}_{group_col.lower()}"

    csv_path = TABLE_DIR / f"{stem}.csv"
    table_df.to_csv(csv_path, index=False)

    # Pivot medians by group col × method
    pivot = table_df.pivot_table(index=group_col, columns="Method", values="Median")

    # LaTeX label and caption
    label = f"tab:{table_id_prefix}_{metric}_{safe_mode}_{group_col.lower()}"
    caption = f"{table_id_prefix}: RMSE({metric}) vs {group_col} [{mode}]"

    tex = pivot.to_latex(
        float_format=lambda x: f"{x:.3g}",
        na_rep="--",
        caption=caption,
        label=label,
        escape=True,
    )
    tex_path = TABLE_DIR / f"{stem}.tex"
    tex_path.write_text(tex, encoding="utf-8")

    print(f"✓ Wrote {csv_path.name} and {tex_path.name}")
    return csv_path, tex_path


def _available_group_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the list of grouping columns present in df (in preferred order).
    """
    candidates = ["SNR", "K", "L", "codebook", "tag"]
    return [c for c in candidates if c in df.columns]


def _summarize_and_write_for_groups(
    df: pd.DataFrame,
    mode: str,
    metrics: List[str],
    groups: List[str],
) -> List[Tuple[Path, Path]]:
    """
    For each metric and each available group col, write CSV + LaTeX tables.
    """
    written = []
    blind_k = (mode == "Blind-K")
    for metric in metrics:
        if metric not in df.columns:
            continue
        for g in groups:
            if g not in df.columns:
                continue
            summary = _summarize(df, g, metric, blind_k=blind_k)
            if summary.empty:
                continue
            table_id = GROUP_TABLE_ID.get(g, "BX")
            written.append(_write_csv_and_tex(summary, g, metric, mode, table_id))
    return written


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def generate_all_tables(mode_list: Optional[List[str]] = None) -> List[Tuple[Path, Path]]:
    """
    Generate tables for all metrics and grouping columns found in bench CSVs.
    - mode_list: e.g. ["Blind-K","Oracle-K"]. If None and a 'mode' column exists,
      defaults to ["Blind-K","Oracle-K"]. If 'mode' column doesn't exist, treats as single-mode.
    Returns list of (csv_path, tex_path) written.
    """
    written: List[Tuple[Path, Path]] = []
    df = _load_all_bench(mode=None)  # load all, we’ll filter per mode if exists

    if df.empty:
        print(f"(tables) No bench CSVs found in {BENCH_DIR}")
        return written

    groups = _available_group_cols(df)
    if not groups:
        print("(tables) No recognized grouping columns (SNR, K, L, codebook, tag) found; nothing to summarize.")
        return written

    if "who" not in df.columns:
        print("(tables) Missing 'who' column; nothing to summarize.")
        return written

    # Decide modes
    modes_to_do: List[str]
    if "mode" in df.columns:
        if mode_list is None:
            # do both if present
            modes_to_do = sorted([m for m in df["mode"].dropna().unique().tolist() if m in ("Blind-K","Oracle-K")])
            if not modes_to_do:
                modes_to_do = ["Blind-K"]
        else:
            modes_to_do = mode_list
        for mode in modes_to_do:
            dsub = df[df["mode"] == mode].copy()
            written += _summarize_and_write_for_groups(dsub, mode, METRICS, groups)
    else:
        # No mode column; treat as single-mode "Blind-K" for annotation purposes
        written += _summarize_and_write_for_groups(df, "Blind-K", METRICS, groups)

    return written


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Simple CLI to regenerate all tables
    out = generate_all_tables()
    print(f"(tables) Generated {len(out)} tables in {TABLE_DIR}")

