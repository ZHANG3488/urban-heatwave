
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare empirical UHI/UCI-stratified data for the NCC diurnal figure.

No original input file is modified. No synthetic/schematic data is generated.

Method-freeze final patch
-------------------------
Figure 4 panel a accepts only formal dT_day/dT_night inputs. Reconstructed
dTmax/dTmin extrema are not used as a fallback.

Figure 4 panel-b/c update
-------------------------
Only the panel-b/panel-c input definitions and the canonical UHI/UCI lookup
period are changed.

Canonical classification:
    UHI/UCI groups are taken from the annual rows of
    all_pair_period_metrics.csv. Pairs without an annual classification are
    excluded.

Panel b applies one common algorithm to all three outcomes:
    1) retain the original outcome model;
    2) apply a +1 degree perturbation to the model-relevant exposure;
    3) calculate the finite-difference marginal response separately for the
       urban and rural station;
    4) calculate the signed urban-minus-rural sensitivity contrast:
       Delta S = S_urban - S_rural;
    5) summarize contrasts by annual UHI/UCI group and HW/NHW phase using a
       station-pair bootstrap.

Original models and exposure periods:
    sleep: nighttime Tmin -> Minor et al. sleep-loss spline
    building CDH: 24-hour temperature -> CDH=max(T-26,0)
    labour loss: station-specific work-hour Tx shaded WBGT derived from common-time two-harmonic T/Td curves -> Dunne et al. model

Panel c keeps outcome-relevant temporal components:
    sleep: day=0, night=total
    building CDH: day+night=total
    labour loss: day=total, night=0 structural zero / not applicable

Positive panel-c burden values mean urban burden is larger than rural burden.
Panel-b values are signed urban-minus-rural marginal sensitivity contrasts.
Positive values indicate greater urban sensitivity; negative values indicate
greater rural sensitivity.

Outputs
-------
main_panel_a_hw_asymmetry_map.csv
main_panel_b_asymmetry_contrast.csv
main_panel_c_additive_decomposition.csv
main_model_reporting_coefficients.csv
main_model_robustness_interaction.csv
supp_daynight_delta_boxplot_data.csv
uhi_uci_run_summary.json
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import UNIFIED_ROOT as CONFIG_UNIFIED_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

UNIFIED_ROOT = Path(CONFIG_UNIFIED_ROOT)
PROJECT_ROOT = UNIFIED_ROOT  # reroute to unified outputs

FILES: Dict[str, Path] = {
    "cdh_daily_panel": UNIFIED_ROOT / "analysis/cdh_energy/all_pairs_daily_panel.csv",
    "hne_paired_panel": UNIFIED_ROOT / "analysis/hne_econ/paired/method_pooled/all_pairs_annual.csv",
    "labour_full": UNIFIED_ROOT / "analysis/labour/labour_loss_full.csv",
    "labour_integrated": UNIFIED_ROOT / "analysis/labour/integrated_fig_data/fig1_diurnal_labour_real.csv",
    "all_pair_period_metrics": UNIFIED_ROOT / "analysis/main_multiyear/robustness_percentile/all_pair_period_metrics.csv",
}

UHI_UCI_GROUPS = ("UHI", "UCI")

# Stable directory name.
# The actual labour-common pair count is detected dynamically at runtime.
LABOUR_COMMON_SUBDIR = "labour_common"

# Figure 4 panel-b outcome-specific marginal sensitivity.
# All sensitivities use a +1 degree perturbation while preserving the
# original outcome model and its relevant exposure period.
PANEL_B_DELTA_C = 1.0
CDH_THRESHOLD_C = 26.0

# Minor et al. (2022), Table S37 sleep-loss spline.
SLEEP_LOSS_BETA1 = -0.107
SLEEP_LOSS_BETA2 = -0.618
SLEEP_LOSS_KNOT1 = -20.0
SLEEP_LOSS_KNOT2 = 10.0
SLEEP_LOSS_CONST = 0.0

# Dunne et al. labour-capacity model used by the original labour pathway.
DUNNE_WBGT_THRESHOLD_C = 25.0
DUNNE_LOSS_COEFFICIENT = 25.0
DUNNE_EXPONENT = 2.0 / 3.0

# Tx/Tn period definition for hourly labour-loss aggregation.
# Keep these constants as the single source of truth.
LABOUR_TX_HOURS = tuple(range(8, 20))  # 08:00–19:59, daytime proxy
LABOUR_TN_HOURS = tuple(h for h in range(24) if h not in LABOUR_TX_HOURS)  # 20:00–07:59


# -----------------------------------------------------------------------------
# Unified uncertainty for main Figure b/c
# -----------------------------------------------------------------------------
# Error bars in the downstream NCC figure are based on the ci_low/ci_high columns
# written here. To keep panel b/c comparable across outcomes, all main-model CIs
# are now station-pair cluster bootstrap percentile 95% CIs.
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260529
BOOTSTRAP_MIN_SUCCESS = 100
BOOTSTRAP_CLUSTER_COL = "pair_id"
BOOTSTRAP_CI_METHOD = "station-pair cluster bootstrap percentile 95% CI"


def set_bootstrap_runtime_config(n_boot: Optional[int] = None, seed: Optional[int] = None, min_success: Optional[int] = None) -> None:
    """Update bootstrap settings from command-line arguments.

    This keeps the statistical definition unchanged, but allows fast pilot runs
    and final high-precision runs without editing the script.
    """
    global BOOTSTRAP_N, BOOTSTRAP_SEED, BOOTSTRAP_MIN_SUCCESS
    if n_boot is not None:
        BOOTSTRAP_N = int(n_boot)
    if seed is not None:
        BOOTSTRAP_SEED = int(seed)
    if min_success is not None:
        BOOTSTRAP_MIN_SUCCESS = int(min_success)


def _stable_bootstrap_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(key)) % 100000
    return int(BOOTSTRAP_SEED + offset)


def _percentile_ci(values, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return (
        float(np.nanpercentile(arr, 100.0 * alpha / 2.0)),
        float(np.nanpercentile(arr, 100.0 * (1.0 - alpha / 2.0))),
    )


def _bootstrap_percentile_dict(samples: List[dict]) -> Dict[str, dict]:
    """Convert a list of bootstrap statistic dictionaries into CI records."""
    if not samples:
        return {}

    keys = sorted({k for s in samples for k in s.keys()})
    out: Dict[str, dict] = {}
    for k in keys:
        vals = [s.get(k, np.nan) for s in samples]
        lo, hi = _percentile_ci(vals)
        out[k] = {
            "ci_low": lo,
            "ci_high": hi,
            "n_boot_success": int(np.isfinite(np.asarray(vals, dtype=float)).sum()),
            "n_boot_requested": int(BOOTSTRAP_N),
            "ci_method": BOOTSTRAP_CI_METHOD,
        }
    return out

def _bootstrap_worker_one(draw, grouped, stat_func, clusters):
    parts = []
    for j, cid in enumerate(draw):
        part = grouped[str(cid)].copy()
        part["_boot_cluster_id"] = f"boot{j}_{cid}"
        parts.append(part)

    boot = pd.concat(parts, ignore_index=True)

    try:
        s = stat_func(boot, "_boot_cluster_id")
    except Exception:
        return {}

    return s if isinstance(s, dict) else {}
    
def _cluster_bootstrap_statistics(
    df: pd.DataFrame,
    stat_func,
    *,
    cluster_col: str = BOOTSTRAP_CLUSTER_COL,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, dict]:
    """
    Station-pair cluster bootstrap.

    Resamples unique station-pair IDs with replacement and preserves all rows
    within each selected pair. A new _boot_cluster_id is assigned so duplicated
    pairs are treated as independent bootstrap clusters. stat_func must accept
    (boot_df, boot_cluster_col) and return a dictionary of scalar statistics.
    """
    if df is None or len(df) == 0:
        return {}

    rng = np.random.default_rng(seed)
    samples: List[dict] = []

    if cluster_col in df.columns:
        clusters = pd.Series(df[cluster_col]).dropna().astype(str).unique()
        if len(clusters) == 0:
            return {}
        grouped = {str(k): v for k, v in df.groupby(df[cluster_col].astype(str), observed=True)}

        try:
            from joblib import Parallel, delayed
            n_jobs = max(1, min(8, (os.cpu_count() or 2) - 1))

            draws = [
                rng.choice(clusters, size=len(clusters), replace=True)
                for _ in range(n_boot)
            ]

            samples = Parallel(n_jobs=n_jobs, backend="loky", batch_size=10)(
                delayed(_bootstrap_worker_one)(draw, grouped, stat_func, clusters)
                for draw in draws
            )
            samples = [s for s in samples if isinstance(s, dict) and s]

        except Exception:
            samples = []
            for _ in range(n_boot):
                draw = rng.choice(clusters, size=len(clusters), replace=True)
                s = _bootstrap_worker_one(draw, grouped, stat_func, clusters)
                if isinstance(s, dict) and s:
                    samples.append(s)
                    
    else:
        # Fallback only for tables without pair_id; still gives a transparent
        # row-bootstrap CI rather than silently reverting to normal-theory SEM.
        n = len(df)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot = df.iloc[idx].copy().reset_index(drop=True)
            boot["_boot_cluster_id"] = np.arange(len(boot)).astype(str)
            try:
                s = stat_func(boot, "_boot_cluster_id")
            except Exception:
                continue
            if isinstance(s, dict) and s:
                samples.append(s)

    if len(samples) < BOOTSTRAP_MIN_SUCCESS:
        print(
            f"  ! bootstrap warning: only {len(samples)} successful replicates "
            f"(<{BOOTSTRAP_MIN_SUCCESS}); CI will be reported when finite.",
            file=sys.stderr,
        )
    return _bootstrap_percentile_dict(samples)


def _ci_from_boot(ok: dict, key: str, fallback_low: float, fallback_high: float) -> Tuple[float, float]:
    rec = (ok.get("bootstrap_ci") or {}).get(key, {})
    lo = rec.get("ci_low", np.nan)
    hi = rec.get("ci_high", np.nan)
    if np.isfinite(lo) and np.isfinite(hi):
        return float(lo), float(hi)
    return float(fallback_low), float(fallback_high)


def _bootstrap_regression_ci(
    sub: pd.DataFrame,
    *,
    y: str,
    xd: str,
    xn: str,
    xdR: Optional[str],
    xnR: Optional[str],
    fe: Sequence[str],
    x_delta: bool,
    sign_flip: bool,
    seed: int,
) -> Dict[str, dict]:
    """Bootstrap the full FE estimator used for sleep rows."""
    def stat_func(boot: pd.DataFrame, boot_cluster_col: str) -> dict:
        boot_fe = [boot_cluster_col if f == BOOTSTRAP_CLUSTER_COL else f for f in fe]
        ok = panel_reg_fast_beta(
            boot,
            y,
            xd,
            xn,
            boot_cluster_col,
            boot_fe,
            xday_rural=xdR,
            xnight_rural=xnR,
            x_are_delta=x_delta,
        )
        if "error" in ok:
            return {}

        bd = float(ok["beta_day"])
        bn = float(ok["beta_night"])
        if sign_flip:
            bd, bn = -bd, -bn

        out = {"contrast": bn - bd}

        d = ok.get("delta_t_day_mean")
        n = ok.get("delta_t_night_mean")
        if d is not None and n is not None and np.isfinite(float(d)) and np.isfinite(float(n)):
            day = bd * float(d)
            night = bn * float(n)
            out.update({"day": day, "night": night, "total": day + night})
        return out

    return _cluster_bootstrap_statistics(sub, stat_func, seed=seed)


def _direct_cluster_bootstrap_from_sums(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    seed: int,
    statistic: str,
) -> Dict[str, dict]:
    """Fast cluster bootstrap for direct means.

    Instead of rebuilding a large bootstrapped DataFrame for every replicate,
    this function resamples station-pair aggregate sums/counts. It is
    algebraically equivalent for direct sample means and keeps the same
    station-pair cluster bootstrap estimand.
    """
    sums = np.asarray(sums, dtype=float)
    counts = np.asarray(counts, dtype=float)
    ok = np.isfinite(counts) & (counts > 0)
    if sums.ndim == 1:
        ok = ok & np.isfinite(sums)
    else:
        ok = ok & np.all(np.isfinite(sums), axis=1)
    sums = sums[ok]
    counts = counts[ok]
    n_cluster = len(counts)
    if n_cluster == 0:
        return {}

    rng = np.random.default_rng(seed)
    samples: List[dict] = []
    for _ in range(BOOTSTRAP_N):
        draw = rng.integers(0, n_cluster, size=n_cluster)
        denom = float(np.sum(counts[draw]))
        if denom <= 0 or not np.isfinite(denom):
            continue
        if statistic == "day_night":
            mean_vals = np.sum(sums[draw, :], axis=0) / denom
            day = float(mean_vals[0])
            night = float(mean_vals[1])
            samples.append({
                "day": day,
                "night": night,
                "total": day + night,
                "contrast": night - day,
            })
        elif statistic == "single_mean":
            day = float(np.sum(sums[draw]) / denom)
            samples.append({
                "day": day,
                "night": 0.0,
                "total": day,
                "contrast": -day,
            })

    if len(samples) < BOOTSTRAP_MIN_SUCCESS:
        print(
            f"  ! bootstrap warning: only {len(samples)} successful replicates "
            f"(<{BOOTSTRAP_MIN_SUCCESS}); CI will be reported when finite.",
            file=sys.stderr,
        )
    return _bootstrap_percentile_dict(samples)


def _bootstrap_direct_day_night_ci(
    sub: pd.DataFrame,
    *,
    day_col: str,
    night_col: str,
    seed: int,
) -> Dict[str, dict]:
    """Pair-equal bootstrap for direct empirical day/night means.

    Multiple rows within a station pair are first averaged. The bootstrap then
    resamples pair means with replacement, so each pair has equal weight.
    """
    if sub is None or len(sub) == 0:
        return {}

    if BOOTSTRAP_CLUSTER_COL in sub.columns:
        work = sub[[BOOTSTRAP_CLUSTER_COL, day_col, night_col]].copy()
        work[day_col] = pd.to_numeric(work[day_col], errors="coerce")
        work[night_col] = pd.to_numeric(work[night_col], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        if len(work) == 0:
            return {}

        pair_means = (
            work.groupby(
                work[BOOTSTRAP_CLUSTER_COL].astype(str),
                observed=True,
            )[[day_col, night_col]]
            .mean()
            .dropna()
        )
        if pair_means.empty:
            return {}

        # One row per pair: sums are the pair means and every count equals one.
        sums = pair_means.to_numpy(float)
        counts = np.ones(len(pair_means), dtype=float)
        return _direct_cluster_bootstrap_from_sums(
            sums,
            counts,
            seed=seed,
            statistic="day_night",
        )

    def stat_func(boot: pd.DataFrame, boot_cluster_col: str) -> dict:
        ok = clustered_direct_mean_2d(
            boot,
            day_col=day_col,
            night_col=night_col,
            cluster_col=boot_cluster_col,
        )
        if "error" in ok:
            return {}
        day = float(ok["mean_day"])
        night = float(ok["mean_night"])
        return {
            "day": day,
            "night": night,
            "total": day + night,
            "contrast": night - day,
        }

    return _cluster_bootstrap_statistics(sub, stat_func, seed=seed)

def _bootstrap_single_mean_ci(
    sub: pd.DataFrame,
    *,
    value_col: str,
    seed: int,
) -> Dict[str, dict]:
    """Fast station-pair cluster bootstrap for a single direct mean."""
    if sub is None or len(sub) == 0:
        return {}

    if BOOTSTRAP_CLUSTER_COL in sub.columns:
        work = sub[[BOOTSTRAP_CLUSTER_COL, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        if len(work) == 0:
            return {}
        g = work.groupby(work[BOOTSTRAP_CLUSTER_COL].astype(str), observed=True)
        sums = g[value_col].sum().to_numpy(float)
        counts = g.size().to_numpy(float)
        return _direct_cluster_bootstrap_from_sums(sums, counts, seed=seed, statistic="single_mean")

    def stat_func(boot: pd.DataFrame, boot_cluster_col: str) -> dict:
        vals = pd.to_numeric(boot[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            return {}
        day = float(vals.mean())
        return {"day": day, "night": 0.0, "total": day, "contrast": -day}

    return _cluster_bootstrap_statistics(sub, stat_func, seed=seed)



def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def safe_read(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"  ! read failed: {path}: {e}", file=sys.stderr)
        return None


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def reanchor_files(new_root: Path) -> None:
    global PROJECT_ROOT, FILES
    old = PROJECT_ROOT
    PROJECT_ROOT = new_root
    out = {}
    for k, p in FILES.items():
        try:
            out[k] = new_root / p.relative_to(old)
        except ValueError:
            out[k] = p
    FILES = out


def audit(out_dir: Path) -> dict:
    a = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "out_dir": str(out_dir),
        "matches": {},
    }

    print(f"[AUDIT] project root = {PROJECT_ROOT}")

    for k, p in FILES.items():
        recs = []

        if p.exists() and p.is_file():
            try:
                cols = pd.read_csv(p, nrows=0).columns.tolist()
                n = sum(1 for _ in open(p, "rb")) - 1
            except Exception as e:
                cols, n = [], -1
                print(f"  ! audit failed on {p}: {e}", file=sys.stderr)

            recs.append({
                "path": str(p),
                "n_rows": n,
                "n_cols": len(cols),
                "columns": cols,
            })

            print(f"  [{k:24s}] OK {p.name} ({n:,} rows x {len(cols):,} cols)")
        else:
            print(f"  [{k:24s}] MISSING {p}")

        a["matches"][k] = recs

    write_json(out_dir / "uhi_uci_data_audit.json", a)
    return a



def load_uhi_uci_lookup(
    audit_obj: dict,
    preferred_period: str = "annual",
) -> pd.DataFrame:
    """
    Load the strict canonical annual pair-level UHI/UCI classification.

    Rules shared by all downstream scripts:
      1) use all_pair_period_metrics.csv;
      2) retain hw_method == percentile when the column exists;
      3) retain period == annual only;
      4) require exactly one valid UHI/UCI label per pair;
      5) never fall back to warm-season, HW or NHW labels.
    """
    recs = audit_obj.get("matches", {}).get("all_pair_period_metrics", [])
    if not recs:
        raise FileNotFoundError(
            f"Missing UHI/UCI source: {FILES['all_pair_period_metrics']}"
        )

    src = Path(recs[0]["path"])
    g = pd.read_csv(
        src,
        usecols=lambda c: c in {"pair_id", "period", "group", "hw_method"},
    )

    required = {"pair_id", "period", "group"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"UHI/UCI lookup missing columns: {sorted(missing)}")

    if "hw_method" in g.columns:
        method = g["hw_method"].astype(str).str.lower().str.strip()
        g = g[method.eq("percentile")].copy()

    period = g["period"].astype(str).str.lower().str.strip()
    g = g[period.eq("annual")].copy()
    if g.empty:
        raise ValueError(
            "Canonical UHI/UCI source contains no annual percentile rows."
        )

    g["pair_id"] = g["pair_id"].astype(str)
    g["uhi_uci_group"] = g["group"].astype(str).str.upper().str.strip()
    g = g[g["uhi_uci_group"].isin(UHI_UCI_GROUPS)].copy()
    if g.empty:
        raise ValueError("Annual canonical source contains no valid UHI/UCI rows.")

    conflicts = (
        g.groupby("pair_id", observed=True)["uhi_uci_group"]
        .nunique()
    )
    conflict_ids = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflict_ids:
        examples = conflict_ids[:20]
        raise ValueError(
            "Conflicting annual UHI/UCI classifications for "
            f"{len(conflict_ids)} pair(s); examples={examples}"
        )

    g = g[["pair_id", "uhi_uci_group"]].drop_duplicates("pair_id")
    return g.reset_index(drop=True)


def add_group(df: pd.DataFrame, lookup: pd.DataFrame, context: str) -> pd.DataFrame:
    """Merge the canonical annual UHI/UCI lookup without silent ID loss."""
    if "pair_id" not in df.columns:
        raise ValueError(f"{context}: missing pair_id")
    if "pair_id" not in lookup.columns:
        raise ValueError(f"{context}: canonical lookup is missing pair_id")

    out = df.copy()
    lut = lookup.copy()

    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    lut["pair_id"] = lut["pair_id"].astype(str).str.strip()

    if lut["pair_id"].duplicated().any():
        dup = (
            lut.loc[lut["pair_id"].duplicated(keep=False), "pair_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"{context}: duplicated pair IDs in canonical lookup: {dup[:20]}"
        )

    out = (
        out.drop(columns=["uhi_uci_group"], errors="ignore")
        .merge(
            lut,
            on="pair_id",
            how="inner",
            validate="many_to_one",
        )
    )

    out = out[out["uhi_uci_group"].isin(UHI_UCI_GROUPS)].copy()
    if out.empty:
        raise ValueError(f"{context}: no rows left after UHI/UCI merge")

    return out



def phase_filter(df: pd.DataFrame, source: str, phase: str) -> Optional[pd.DataFrame]:
    if phase not in ("HW", "NHW"):
        raise ValueError("phase must be HW or NHW")

    if source == "hne":
        if "hw_flag" not in df.columns:
            return None

        warm = (
            (df["warm_season_flag"] == 1)
            if "warm_season_flag" in df.columns
            else pd.Series(True, index=df.index)
        )
        return df[(df["hw_flag"] == (1 if phase == "HW" else 0)) & warm].copy()

    if "period" not in df.columns:
        return None

    p = df["period"].astype(str).str.lower().str.strip()
    keep = (
        p.isin(["heatwave", "heat_wave", "hw"])
        if phase == "HW"
        else p.isin(["non_heatwave", "non-heatwave", "nonheatwave", "nhw"])
    )
    return df[keep].copy()


def group_phase(df: pd.DataFrame, source: str, group: str, phase: str) -> Optional[pd.DataFrame]:
    sub = phase_filter(df, source, phase)
    if sub is None or len(sub) == 0 or "uhi_uci_group" not in sub.columns:
        return sub
    return sub[sub["uhi_uci_group"] == group].copy()


def numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def within_demean(df: pd.DataFrame, y: str, xs: Sequence[str], fes: Sequence[str], iters: int = 100) -> pd.DataFrame:
    w = df[[y] + list(xs) + list(fes)].copy()

    for _ in range(iters):
        md = 0.0
        for fe in fes:
            for c in [y] + list(xs):
                old = w[c].copy()
                w[c] = w[c] - w.groupby(fe, observed=True)[c].transform("mean")
                d = float((w[c] - old).abs().max()) if len(w) else 0.0
                if np.isfinite(d):
                    md = max(md, d)
        if md < 1e-10:
            break

    return w


def ols_cluster(y: np.ndarray, X: np.ndarray, cluster: np.ndarray):
    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ (X.T @ y)
    resid = y - X @ beta

    n, k = X.shape
    G = len(np.unique(cluster))

    score = X * resid[:, None]
    sdf = pd.DataFrame(score)
    sdf["_c"] = cluster

    ssum = sdf.groupby("_c", observed=True).sum().values
    meat = ssum.T @ ssum

    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    vcov = corr * (inv @ meat @ inv)

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return beta, se, vcov, resid, n


def fit_stats(y: np.ndarray, X2: np.ndarray, resid: np.ndarray, raw: pd.DataFrame, xd: str, xn: str) -> dict:
    n, k = X2.shape

    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))

    r2 = None if sst <= 0 else float(1 - ssr / sst)
    adj = None if r2 is None or n <= k + 1 else float(1 - (1 - r2) * (n - 1) / max(n - k - 1, 1))
    rmse = float(np.sqrt(ssr / max(n - k, 1)))

    corr_raw = None
    corr_within = None
    vif = None

    try:
        v = float(raw[[xd, xn]].corr().iloc[0, 1])
        corr_raw = v if np.isfinite(v) else None
    except Exception:
        pass

    try:
        v = float(np.corrcoef(X2[:, 0], X2[:, 1])[0, 1])
        if np.isfinite(v):
            corr_within = v
            if abs(v) < 0.999999:
                vif = float(1 / (1 - v * v))
    except Exception:
        pass

    return {
        "within_r2": r2,
        "adj_within_r2": adj,
        "rmse_within": rmse,
        "x_day_night_corr_raw": corr_raw,
        "x_day_night_corr_within": corr_within,
        "x_day_night_vif_within": vif,
    }


def panel_reg(
    df: pd.DataFrame,
    ycol: str,
    xday: str,
    xnight: str,
    cluster_col: str,
    fe_cols: Sequence[str],
    xday_rural: Optional[str] = None,
    xnight_rural: Optional[str] = None,
    x_are_delta: bool = False,
    delta_source_label: Optional[str] = None,
) -> dict:
    needed = {ycol, xday, xnight, cluster_col, *fe_cols}
    miss = needed - set(df.columns)

    if miss:
        return {"error": f"missing columns: {sorted(miss)}", "n_obs": 0, "n_clusters": 0}

    work = numeric(df[list(needed)].copy(), [ycol, xday, xnight]).replace([np.inf, -np.inf], np.nan).dropna()

    if len(work) < 30:
        return {"error": f"only {len(work)} valid rows", "n_obs": int(len(work)), "n_clusters": 0}

    nc = int(work[cluster_col].nunique())
    if nc < 5:
        return {"error": f"only {nc} clusters", "n_obs": int(len(work)), "n_clusters": nc}

    dem = within_demean(work, ycol, [xday, xnight], fe_cols)

    y = dem[ycol].to_numpy(float)
    X = dem[[xday, xnight]].to_numpy(float)
    cl = work.loc[dem.index, cluster_col].astype("category").cat.codes.to_numpy()

    try:
        beta, se, vcov, resid, n = ols_cluster(y, X, cl)
    except Exception as e:
        return {"error": f"OLS failed: {e}", "n_obs": int(len(work)), "n_clusters": nc}

    out = {
        "beta_day": float(beta[0]),
        "beta_night": float(beta[1]),
        "se_day": float(se[0]),
        "se_night": float(se[1]),
        "cov_day_night": float(vcov[0, 1]),
        "n_obs": int(n),
        "n_clusters": nc,
        "x_day_urban_mean": float(work[xday].mean()),
        "x_night_urban_mean": float(work[xnight].mean()),
        "x_day_rural_mean": None,
        "x_night_rural_mean": None,
        "delta_t_day_mean": None,
        "delta_t_night_mean": None,
        "delta_t_source": "missing",
        "exposure_source": f"{xday}/{xnight}",
    }

    out.update(fit_stats(y, X, resid, work, xday, xnight))

    if xday_rural and xnight_rural and xday_rural in df.columns and xnight_rural in df.columns:
        sub = numeric(
            df[list(needed | {xday_rural, xnight_rural})].copy(),
            [xday, xnight, xday_rural, xnight_rural],
        )
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

        if len(sub):
            out.update({
                "x_day_rural_mean": float(sub[xday_rural].mean()),
                "x_night_rural_mean": float(sub[xnight_rural].mean()),
                "delta_t_day_mean": float((sub[xday] - sub[xday_rural]).mean()),
                "delta_t_night_mean": float((sub[xnight] - sub[xnight_rural]).mean()),
                "delta_t_source": "urban_minus_rural",
                "exposure_source": f"{xday}-{xday_rural}/{xnight}-{xnight_rural}",
            })

    elif x_are_delta:
        out.update({
            "delta_t_day_mean": float(work[xday].mean()),
            "delta_t_night_mean": float(work[xnight].mean()),
            "delta_t_source": delta_source_label or "delta_columns",
            "exposure_source": f"{xday}/{xnight}",
        })

    return out

def panel_reg_fast_beta(
    df: pd.DataFrame,
    ycol: str,
    xday: str,
    xnight: str,
    cluster_col: str,
    fe_cols: Sequence[str],
    xday_rural: Optional[str] = None,
    xnight_rural: Optional[str] = None,
    x_are_delta: bool = False,
) -> dict:
    """
    Fast FE regression for bootstrap only.

    It returns the same beta_day / beta_night point estimates as panel_reg(),
    but skips cluster SE, vcov, R2, VIF, fit_stats.
    """
    needed = {ycol, xday, xnight, cluster_col, *fe_cols}
    miss = needed - set(df.columns)
    if miss:
        return {"error": f"missing columns: {sorted(miss)}"}

    extra = set()
    if xday_rural and xnight_rural:
        if xday_rural in df.columns and xnight_rural in df.columns:
            extra = {xday_rural, xnight_rural}

    work = numeric(
        df[list(needed | extra)].copy(),
        [ycol, xday, xnight] + list(extra),
    ).replace([np.inf, -np.inf], np.nan).dropna(subset=[ycol, xday, xnight])

    if len(work) < 30:
        return {"error": f"only {len(work)} valid rows"}

    if work[cluster_col].nunique() < 5:
        return {"error": "too few clusters"}

    dem = within_demean(work, ycol, [xday, xnight], fe_cols)

    y = dem[ycol].to_numpy(float)
    X = dem[[xday, xnight]].to_numpy(float)

    try:
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    except Exception as e:
        return {"error": f"OLS failed: {e}"}

    out = {
        "beta_day": float(beta[0]),
        "beta_night": float(beta[1]),
        "delta_t_day_mean": None,
        "delta_t_night_mean": None,
    }

    if xday_rural and xnight_rural and xday_rural in work.columns and xnight_rural in work.columns:
        sub = work[[xday, xnight, xday_rural, xnight_rural]].dropna()
        if len(sub):
            out["delta_t_day_mean"] = float((sub[xday] - sub[xday_rural]).mean())
            out["delta_t_night_mean"] = float((sub[xnight] - sub[xnight_rural]).mean())

    elif x_are_delta:
        out["delta_t_day_mean"] = float(work[xday].mean())
        out["delta_t_night_mean"] = float(work[xnight].mean())

    return out

def panel_reg_interaction(df: pd.DataFrame, ycol: str, xday: str, xnight: str, cluster_col: str, fe_cols: Sequence[str]) -> dict:
    inter = "_day_x_night"

    needed = {ycol, xday, xnight, cluster_col, *fe_cols}
    miss = needed - set(df.columns)

    if miss:
        return {"error": f"missing columns: {sorted(miss)}", "n_obs": 0, "n_clusters": 0}

    work = numeric(df[list(needed)].copy(), [ycol, xday, xnight])
    work[inter] = work[xday] * work[xnight]
    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if len(work) < 30:
        return {"error": f"only {len(work)} valid rows", "n_obs": int(len(work)), "n_clusters": 0}

    nc = int(work[cluster_col].nunique())
    if nc < 5:
        return {"error": f"only {nc} clusters", "n_obs": int(len(work)), "n_clusters": nc}

    dem = within_demean(work, ycol, [xday, xnight, inter], fe_cols)

    y = dem[ycol].to_numpy(float)
    X = dem[[xday, xnight, inter]].to_numpy(float)
    cl = work.loc[dem.index, cluster_col].astype("category").cat.codes.to_numpy()

    beta, se, vcov, resid, n = ols_cluster(y, X, cl)

    out = {
        "beta_day": float(beta[0]),
        "beta_night": float(beta[1]),
        "beta_interaction": float(beta[2]),
        "se_day": float(se[0]),
        "se_night": float(se[1]),
        "se_interaction": float(se[2]),
        "n_obs": int(n),
        "n_clusters": nc,
    }
    out.update(fit_stats(y, X[:, :2], resid, work, xday, xnight))

    return out


def build_map(
    audit_obj: dict,
    out_dir: Path,
    lookup_override: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    """
    Build Figure 4 panel-a map input from the formal pair-period output of
    01_main_pair_period_metrics_common_time_dew2harm_fixed.py.

    Frozen method
    -------------
    The formal fixed-hour urban-rural contrasts are read directly from:

        dT_day_mean
        dT_night_mean

    These fields are produced by 01 from the urban-minus-rural difference
    between the separately reconstructed two-harmonic temperature curves,
    averaged over:

        day   = 08:00-19:59 local solar time
        night = 20:00-07:59 local solar time

    The map value is:

        (dT_night - dT_day)_HW
        -
        (dT_night - dT_day)_NHW

    HW/NHW rows are inherited directly from the formal percentile output of 01.
    Canonical UHI/UCI labels are inherited from the annual lookup. No heatwave
    detection, UHI/UCI reclassification, FFT reconstruction, dTmax/dTmin
    fallback, or CDH-table temperature fallback is performed here.
    """
    print("\n[MAIN A] map data by canonical annual UHI/UCI")

    recs = audit_obj.get("matches", {}).get(
        "all_pair_period_metrics",
        [],
    )
    if not recs:
        raise FileNotFoundError(
            "missing all_pair_period_metrics; Figure 4 panel a must read "
            "the formal dT_day_mean/dT_night_mean pair-period output from 01"
        )

    src = Path(recs[0]["path"])

    lookup = (
        lookup_override.copy()
        if lookup_override is not None
        else load_uhi_uci_lookup(audit_obj)
    )

    df = pd.read_csv(
        src,
        low_memory=False,
    )

    required = {
        "pair_id",
        "period",
        "dT_day_mean",
        "dT_night_mean",
        "lat_urban",
        "lon_urban",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Figure 4 panel-a requires the formal 01 pair-period fields "
            f"{sorted(required)}. Missing: {sorted(missing)}"
        )

    # Formal Figure 4 uses the percentile-method output only.
    if "hw_method" in df.columns:
        method_key = (
            df["hw_method"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df = df[
            method_key.eq("percentile")
        ].copy()

    df["pair_id"] = (
        df["pair_id"]
        .astype(str)
        .str.strip()
    )

    df["_period_key"] = (
        df["period"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Only accept formal upstream HW/NHW period rows.
    df = df[
        df["_period_key"].isin(
            [
                "heatwave",
                "non_heatwave",
            ]
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "Figure 4 panel-a found no formal percentile heatwave/"
            "non_heatwave rows in all_pair_period_metrics.csv"
        )

    # Attach the annual canonical UHI/UCI group.
    df = add_group(
        df,
        lookup,
        "map",
    )

    numeric_cols = [
        "dT_day_mean",
        "dT_night_mean",
        "lat_urban",
        "lon_urban",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Require one formal row per pair and period.
    dup = df.duplicated(
        subset=[
            "pair_id",
            "_period_key",
        ],
        keep=False,
    )

    if dup.any():
        examples = (
            df.loc[
                dup,
                [
                    "pair_id",
                    "period",
                ],
            ]
            .drop_duplicates()
            .head(20)
            .to_dict("records")
        )

        raise ValueError(
            "Figure 4 panel-a requires one formal row per pair and period; "
            f"duplicate examples={examples}"
        )

    optional_meta = [
        c
        for c in [
            "kg_group",
            "climate_zone_main",
            "continent",
        ]
        if c in df.columns
    ]

    hw_cols = [
        "pair_id",
        "uhi_uci_group",
        "lat_urban",
        "lon_urban",
        "dT_day_mean",
        "dT_night_mean",
        *optional_meta,
    ]

    nhw_cols = [
        "pair_id",
        "lat_urban",
        "lon_urban",
        "dT_day_mean",
        "dT_night_mean",
    ]

    # Retain optional period sample-size fields for diagnostics.
    if "urban_ndays" in df.columns:
        hw_cols.append("urban_ndays")
        nhw_cols.append("urban_ndays")

    if "rural_ndays" in df.columns:
        hw_cols.append("rural_ndays")
        nhw_cols.append("rural_ndays")

    hw = (
        df.loc[
            df["_period_key"].eq("heatwave"),
            hw_cols,
        ]
        .rename(
            columns={
                "lat_urban": "lat",
                "lon_urban": "lon",
                "dT_day_mean": "tx_hw",
                "dT_night_mean": "tn_hw",
                "urban_ndays": "urban_ndays_hw",
                "rural_ndays": "rural_ndays_hw",
            }
        )
        .copy()
    )

    nhw = (
        df.loc[
            df["_period_key"].eq("non_heatwave"),
            nhw_cols,
        ]
        .rename(
            columns={
                "lat_urban": "lat_nhw",
                "lon_urban": "lon_nhw",
                "dT_day_mean": "tx_nhw",
                "dT_night_mean": "tn_nhw",
                "urban_ndays": "urban_ndays_nhw",
                "rural_ndays": "rural_ndays_nhw",
            }
        )
        .copy()
    )

    if hw.empty or nhw.empty:
        raise ValueError(
            "Figure 4 panel-a requires both heatwave and non_heatwave rows"
        )

    out = hw.merge(
        nhw,
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )

    # Coordinates must be identical across the two periods.
    coordinate_ok = (
        np.isfinite(out["lat"])
        & np.isfinite(out["lon"])
        & np.isfinite(out["lat_nhw"])
        & np.isfinite(out["lon_nhw"])
        & np.isclose(
            out["lat"],
            out["lat_nhw"],
            rtol=0.0,
            atol=1e-10,
        )
        & np.isclose(
            out["lon"],
            out["lon_nhw"],
            rtol=0.0,
            atol=1e-10,
        )
    )

    if not bool(coordinate_ok.all()):
        bad_pairs = (
            out.loc[
                ~coordinate_ok,
                "pair_id",
            ]
            .astype(str)
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Figure 4 panel-a found inconsistent or missing pair coordinates "
            f"between HW and NHW rows; examples={bad_pairs}"
        )

    out = out.drop(
        columns=[
            "lat_nhw",
            "lon_nhw",
        ]
    )

    # A response is valid only if both periods and both day/night fields exist.
    # Missing scientific values are never converted to zero.
    out = (
        out.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "tx_hw",
                "tn_hw",
                "tx_nhw",
                "tn_nhw",
            ]
        )
        .copy()
    )

    if out.empty:
        raise ValueError(
            "Figure 4 panel-a has no matched pair with finite formal "
            "dT_day_mean/dT_night_mean in both HW and NHW"
        )

    # Frozen Figure 4 panel-a response:
    #
    #   (dT_night - dT_day)_HW
    #   -
    #   (dT_night - dT_day)_NHW
    #
    # Equivalent form:
    #
    #   (dT_night_HW - dT_night_NHW)
    #   -
    #   (dT_day_HW - dT_day_NHW)
    out["amp_day"] = (
        out["tx_hw"]
        - out["tx_nhw"]
    )

    out["amp_night"] = (
        out["tn_hw"]
        - out["tn_nhw"]
    )

    out["asym_hw"] = (
        out["tn_hw"]
        - out["tx_hw"]
    )

    out["asym_nhw"] = (
        out["tn_nhw"]
        - out["tx_nhw"]
    )

    out["asym_hw_minus_nhw"] = (
        out["asym_hw"]
        - out["asym_nhw"]
    )

    out["asym_hw_minus_nhw_check"] = (
        out["amp_night"]
        - out["amp_day"]
    )

    # Algebraic consistency check.
    if not np.allclose(
        out["asym_hw_minus_nhw"].to_numpy(dtype=float),
        out["asym_hw_minus_nhw_check"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError(
            "Figure 4 panel-a algebra check failed: "
            "asymmetry response is inconsistent with amp_night - amp_day"
        )

    out = out[
        np.isfinite(
            out["asym_hw_minus_nhw"]
        )
    ].copy()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    p = (
        out_dir
        / "main_panel_a_hw_asymmetry_map.csv"
    )

    out.to_csv(
        p,
        index=False,
    )

    write_json(
        out_dir
        / "main_panel_a_hw_asymmetry_map_metadata.json",
        {
            "value_column": "asym_hw_minus_nhw",

            "definition": (
                "(dT_night_mean-dT_day_mean)_HW - "
                "(dT_night_mean-dT_day_mean)_NHW"
            ),

            "equivalent_definition": (
                "(dT_night_mean_HW-dT_night_mean_NHW) - "
                "(dT_day_mean_HW-dT_day_mean_NHW)"
            ),

            "day_hours_local_solar_time": (
                "08:00-19:59"
            ),

            "night_hours_local_solar_time": (
                "20:00-07:59"
            ),

            "temperature_metric_source": (
                "01 formal two-harmonic pair-period reconstructed "
                "urban-minus-rural curves"
            ),

            "temperature_fields": [
                "dT_day_mean",
                "dT_night_mean",
            ],

            "period_source": (
                "01 formal percentile heatwave/non_heatwave pair-period rows"
            ),

            "unit": "deg C",

            "notes": (
                "Zero-centred difference metric for Figure 4 panel a; "
                "positive values indicate stronger HW amplification at night "
                "than during day. No dTmax/dTmin or CDH-table fallback is used."
            ),

            "stratification": (
                "annual canonical uhi_uci_group"
            ),

            "uhi_uci_classification_period": (
                "annual"
            ),

            "data_source": str(src),

            "uhi_uci_source": str(
                FILES[
                    "all_pair_period_metrics"
                ]
            ),

            "n_pairs": int(
                len(out)
            ),

            "synthetic_data_used": False,
        },
    )

    print(
        "  ✓ Figure 4 panel-a map input: "
        f"{len(out)} matched pairs from formal 01 HW/NHW rows"
    )

    return p
    
def append_contrast(rows, reports, *, outcome, group, phase, ok, ycol, xday, xnight, unit, src, fe, notes, sign_flip):
    bd, bn = float(ok["beta_day"]), float(ok["beta_night"])

    if sign_flip:
        bd, bn = -bd, -bn

    sd, sn, cov = float(ok["se_day"]), float(ok["se_night"]), float(ok["cov_day_night"])

    contrast = bn - bd
    se = float(np.sqrt(max(sn * sn + sd * sd - 2 * cov, 0)))
    fallback_low = contrast - 1.96 * se
    fallback_high = contrast + 1.96 * se
    ci_low, ci_high = _ci_from_boot(ok, "contrast", fallback_low, fallback_high)

    boot_rec = (ok.get("bootstrap_ci") or {}).get("contrast", {})

    base = {
        "outcome": outcome,
        "uhi_uci_group": group,
        "period_phase": phase,
        "beta_day": bd,
        "beta_night": bn,
        "beta_contrast_night_minus_day": contrast,
        "se_contrast": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": boot_rec.get("ci_method", BOOTSTRAP_CI_METHOD),
        "n_boot_success": boot_rec.get("n_boot_success"),
        "n_boot_requested": boot_rec.get("n_boot_requested", BOOTSTRAP_N),
        "unit": unit,
        "method": "UHI/UCI- and phase-stratified two-way FE OLS; point estimate unchanged; CI from station-pair cluster bootstrap",
        "n_obs": ok["n_obs"],
        "n_clusters": ok["n_clusters"],
        "data_source": str(src),
        "notes": notes,
    }

    rows.append(base)

    reports.append({
        **base,
        "ycol": ycol,
        "xday": xday,
        "xnight": xnight,
        "se_day": sd,
        "se_night": sn,
        "ci_day_low": bd - 1.96 * sd,
        "ci_day_high": bd + 1.96 * sd,
        "ci_night_low": bn - 1.96 * sn,
        "ci_night_high": bn + 1.96 * sn,
        "cov_day_night": cov,
        "fe_spec": fe,
        "cluster": "pair_id",
        "within_r2": ok.get("within_r2"),
        "adj_within_r2": ok.get("adj_within_r2"),
        "rmse_within": ok.get("rmse_within"),
        "x_day_night_corr_raw": ok.get("x_day_night_corr_raw"),
        "x_day_night_corr_within": ok.get("x_day_night_corr_within"),
        "x_day_night_vif_within": ok.get("x_day_night_vif_within"),
        "delta_t_day_mean": ok.get("delta_t_day_mean"),
        "delta_t_night_mean": ok.get("delta_t_night_mean"),
        "delta_t_source": ok.get("delta_t_source"),
        "exposure_source": ok.get("exposure_source"),
    })


def append_decomp(rows, *, outcome, group, phase, ok, unit, src, sign_flip):
    d = ok.get("delta_t_day_mean")
    n = ok.get("delta_t_night_mean")

    if d is None or n is None:
        return

    delta = np.array([d, n], float)
    beta = np.array([ok["beta_day"], ok["beta_night"]], float)

    if sign_flip:
        beta = -beta

    vcov = np.array([
        [float(ok["se_day"]) ** 2, float(ok["cov_day_night"])],
        [float(ok["cov_day_night"]), float(ok["se_night"]) ** 2],
    ])

    contrib = beta * delta
    uout = str(unit).split(" per ")[0]

    for i, comp in enumerate(["day", "night"]):
        se = abs(delta[i]) * np.sqrt(max(vcov[i, i], 0))
        est = float(contrib[i])
        fallback_low = est - 1.96 * se
        fallback_high = est + 1.96 * se
        ci_low, ci_high = _ci_from_boot(ok, comp, fallback_low, fallback_high)
        boot_rec = (ok.get("bootstrap_ci") or {}).get(comp, {})

        rows.append({
            "outcome": outcome,
            "uhi_uci_group": group,
            "period_phase": phase,
            "component": comp,
            "estimate": est,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_method": boot_rec.get("ci_method", BOOTSTRAP_CI_METHOD),
            "n_boot_success": boot_rec.get("n_boot_success"),
            "n_boot_requested": boot_rec.get("n_boot_requested", BOOTSTRAP_N),
            "unit": uout,
            "beta": float(beta[i]),
            "delta_mean": float(delta[i]),
            "delta_source": ok.get("delta_t_source"),
            "exposure_source": ok.get("exposure_source"),
            "n_obs": ok["n_obs"],
            "n_clusters": ok["n_clusters"],
            "data_source": str(src),
            "method": "additive beta*delta decomposition; CI from station-pair cluster bootstrap",
        })

    total = float(contrib.sum())
    se_total = float(np.sqrt(max(delta.T @ vcov @ delta, 0)))
    fallback_low = total - 1.96 * se_total
    fallback_high = total + 1.96 * se_total
    ci_low, ci_high = _ci_from_boot(ok, "total", fallback_low, fallback_high)
    boot_rec = (ok.get("bootstrap_ci") or {}).get("total", {})

    rows.append({
        "outcome": outcome,
        "uhi_uci_group": group,
        "period_phase": phase,
        "component": "total",
        "estimate": total,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": boot_rec.get("ci_method", BOOTSTRAP_CI_METHOD),
        "n_boot_success": boot_rec.get("n_boot_success"),
        "n_boot_requested": boot_rec.get("n_boot_requested", BOOTSTRAP_N),
        "unit": uout,
        "beta": None,
        "delta_mean": None,
        "delta_source": ok.get("delta_t_source"),
        "exposure_source": ok.get("exposure_source"),
        "n_obs": ok["n_obs"],
        "n_clusters": ok["n_clusters"],
        "data_source": str(src),
        "method": "additive total; CI from station-pair cluster bootstrap",
    })


def append_robust(rows, *, outcome, group, phase, ok, ycol, xday, xnight, unit, src, fe, notes, sign_flip):
    for term, bk, sk in [
        ("day", "beta_day", "se_day"),
        ("night", "beta_night", "se_night"),
        ("day_x_night", "beta_interaction", "se_interaction"),
    ]:
        b = float(ok[bk])
        s = float(ok[sk])

        if sign_flip:
            b = -b

        rows.append({
            "outcome": outcome,
            "uhi_uci_group": group,
            "period_phase": phase,
            "term": term,
            "estimate": b,
            "se": s,
            "ci_low": b - 1.96 * s,
            "ci_high": b + 1.96 * s,
            "unit": unit,
            "ycol": ycol,
            "xday": xday,
            "xnight": xnight,
            "n_obs": ok.get("n_obs"),
            "n_clusters": ok.get("n_clusters"),
            "within_r2": ok.get("within_r2"),
            "adj_within_r2": ok.get("adj_within_r2"),
            "rmse_within": ok.get("rmse_within"),
            "method": "two-way FE OLS with day×night interaction; pair-clustered CR1 SE",
            "fe_spec": fe,
            "cluster": "pair_id",
            "data_source": str(src),
            "notes": notes,
        })


def normalise_phase_label(x) -> Optional[str]:
    s = str(x).strip().upper()
    if s in {"HW", "HEATWAVE", "HEAT_WAVE"}:
        return "HW"
    if s in {"NHW", "NON_HEATWAVE", "NON-HEATWAVE", "NONHEATWAVE"}:
        return "NHW"
    return None


def labour_tx_tn_component(hour) -> Optional[str]:
    try:
        h = int(hour)
    except Exception:
        return None

    if h in LABOUR_TX_HOURS:
        return "day"
    if h in LABOUR_TN_HOURS:
        return "night"
    return None


def weighted_mean_and_se(values, ses, weights) -> Tuple[float, float, int, int]:
    """
    Weighted mean across hourly rows.

    Approximate SE:
        sqrt(sum((w_i * se_i)^2))

    This is a transparent plotting uncertainty approximation and is recorded
    in the output metadata.
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    s = pd.to_numeric(pd.Series(ses), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(float)

    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)

    if not ok.any():
        return np.nan, np.nan, 0, 0

    v = v[ok]
    w = w[ok]

    if len(s) == len(ok):
        s = s[ok]
    else:
        s = np.full_like(v, np.nan)

    ww = w / np.sum(w)
    est = float(np.sum(ww * v))

    if np.isfinite(s).any():
        ss = np.where(np.isfinite(s), s, 0.0)
        se = float(np.sqrt(np.sum((ww * ss) ** 2)))
    else:
        se = np.nan

    return est, se, int(np.sum(w)), int(len(v))

def _find_fig4_labour_paired_diffs_path(audit_obj: dict) -> Optional[Path]:
    """
    Locate labour_loss_final.py Figure-4 labour paired-difference output.

    This function does not recompute labour capacity.
    It only finds the upstream file generated from dlc_dunne_h00-h23.

    Expected upstream file:
        result_era/Figure4_data/fig4_labour_paired_diffs.csv
    """

    candidates = []

    # If you later add this key to FILES, this will automatically use it.
    for key in ["fig4_labour_paired_diffs", "labour_fig4_paired_diffs"]:
        rec = audit_obj.get("matches", {}).get(key, [])
        if rec:
            candidates.append(Path(rec[0]["path"]))

    # Current labour_loss_final.py default output path (unified).
    candidates.append(
        UNIFIED_ROOT / "shared/fig4_data/fig4_labour_paired_diffs.csv"
    )

    # Legacy fallback.
    candidates.append(
        UNIFIED_ROOT / "analysis/labour/fig4_labour_paired_diffs.csv"
    )

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    return None


def compute_sleep_loss_minutes(tmin):
    """
    Minor et al. (2022), Table S37 nighttime-Tmin spline.

    Parameters
    ----------
    tmin : array-like
        Nighttime minimum temperature in deg C.

    Returns
    -------
    numpy.ndarray
        Change in sleep duration in min/night.

        Negative values indicate reduced sleep duration.
        Missing input values remain NaN.
    """
    t = np.asarray(tmin, dtype=float)

    # First segment: -20 to 10 deg C.
    # Maximum segment width = 30 deg C.
    segment1 = np.clip(
        t - SLEEP_LOSS_KNOT1,
        0.0,
        SLEEP_LOSS_KNOT2 - SLEEP_LOSS_KNOT1,
    )

    # Second segment: temperature above 10 deg C.
    segment2 = np.maximum(
        t - SLEEP_LOSS_KNOT2,
        0.0,
    )

    return (
        SLEEP_LOSS_CONST
        + SLEEP_LOSS_BETA1 * segment1
        + SLEEP_LOSS_BETA2 * segment2
    )


def sleep_loss_amount_minutes(tmin):
    """
    Return positive sleep-loss burden in min/night.

    Positive values mean more minutes of sleep are lost.
    """
    sleep_change = compute_sleep_loss_minutes(tmin)
    return -sleep_change

def labour_capacity_pct_dunne(wbgt):
    """Dunne et al. labour capacity in percent."""
    w = np.asarray(wbgt, dtype=float)
    excess = np.maximum(w - DUNNE_WBGT_THRESHOLD_C, 0.0)
    capacity = 100.0 - DUNNE_LOSS_COEFFICIENT * np.power(
        excess,
        DUNNE_EXPONENT,
    )
    return np.clip(capacity, 0.0, 100.0)


def labour_loss_pct_dunne(wbgt):
    """Positive labour-loss amount implied by the Dunne capacity model."""
    return 100.0 - labour_capacity_pct_dunne(wbgt)

def finite_difference_sensitivity(model_func, values, delta=PANEL_B_DELTA_C):
    """Forward finite-difference marginal sensitivity for a nonlinear model."""
    x = np.asarray(values, dtype=float)
    return (model_func(x + delta) - model_func(x)) / float(delta)


def _pair_mean_bootstrap_summary(
    sub: pd.DataFrame,
    value_col: str,
    seed_key: str,
) -> dict:
    """
    Mean and station-pair bootstrap percentile CI.

    Multiple rows within a pair are first averaged so every pair receives equal
    weight. The bootstrap then resamples pair means with replacement.
    """
    if sub is None or len(sub) == 0 or value_col not in sub.columns:
        return {
            "estimate": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_pairs": 0,
        }

    tmp = sub[["pair_id", value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp["pair_id"] = tmp["pair_id"].astype(str)

    pair_vals = (
        tmp.dropna(subset=[value_col])
        .groupby("pair_id", observed=True)[value_col]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )

    n = int(len(pair_vals))
    if n == 0:
        return {
            "estimate": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_pairs": 0,
        }

    estimate = float(np.mean(pair_vals))
    se = (
        float(np.std(pair_vals, ddof=1) / np.sqrt(n))
        if n > 1
        else np.nan
    )

    if n == 1:
        ci_low = ci_high = estimate
    else:
        rng_local = np.random.default_rng(
            _stable_bootstrap_seed("panel_b_c_pair_mean", seed_key)
        )
        boots = np.empty(BOOTSTRAP_N, dtype=float)
        for i in range(BOOTSTRAP_N):
            boots[i] = np.mean(
                rng_local.choice(pair_vals, size=n, replace=True)
            )
        ci_low, ci_high = _percentile_ci(boots)

    return {
        "estimate": estimate,
        "se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_pairs": n,
    }



def _append_pair_phase_components(
    rows: List[dict],
    sub: pd.DataFrame,
    *,
    outcome: str,
    group: str,
    phase: str,
    component_columns: Dict[str, str],
    unit: str,
    data_source: Path,
) -> None:
    """Append one equal-weighted value per pair, phase and burden component."""
    if rows is None or sub is None or len(sub) == 0:
        return

    needed = {"pair_id", *component_columns.values()}
    missing = needed - set(sub.columns)
    if missing:
        raise ValueError(
            f"{outcome} paired amplification inputs missing columns: "
            f"{sorted(missing)}"
        )

    work = sub[["pair_id", *component_columns.values()]].copy()
    work["pair_id"] = work["pair_id"].astype(str)
    for col in component_columns.values():
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Require the displayed components to be jointly finite within a phase.
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(component_columns.values())
    )
    if work.empty:
        return

    pair_means = (
        work.groupby("pair_id", observed=True)[list(component_columns.values())]
        .mean()
        .reset_index()
    )

    for _, record in pair_means.iterrows():
        for component, col in component_columns.items():
            rows.append({
                "pair_id": str(record["pair_id"]),
                "uhi_uci_group": str(group),
                "period_phase": str(phase),
                "outcome": str(outcome),
                "component": str(component),
                "value": float(record[col]),
                "unit": str(unit),
                "data_source": str(data_source),
            })


def build_paired_hw_amplification_summary(
    pair_phase_rows: Sequence[dict],
    out_dir: Path,
) -> pd.DataFrame:
    """Summarize paired HW-NHW amplification without fitting a regression.

    For pair p, outcome k and phase q:
        DeltaY_pqk = Y_urban,pqk - Y_rural,pqk
        R_pk = DeltaY_p,HW,k - DeltaY_p,NHW,k

    The output reports the mean R_pk separately for annual UHI and UCI pairs,
    together with the UCI-minus-UHI difference. Confidence intervals use a
    bootstrap stratified by annual UHI/UCI group, with each selected pair
    carrying its matched NHW and HW values together.

    This function changes only the amplification summary layer. The original
    sleep, CDH and Dunne labour-loss calculations are not modified.
    """
    raw = pd.DataFrame(pair_phase_rows)
    required = {
        "pair_id",
        "uhi_uci_group",
        "period_phase",
        "outcome",
        "component",
        "value",
        "unit",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            "Paired HW amplification inputs missing columns: "
            f"{sorted(missing)}"
        )

    raw = raw.copy()
    raw["pair_id"] = raw["pair_id"].astype(str)
    raw["uhi_uci_group"] = raw["uhi_uci_group"].astype(str).str.upper()
    raw["period_phase"] = raw["period_phase"].astype(str).str.upper()
    raw["component"] = raw["component"].astype(str).str.lower()
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw[
        raw["uhi_uci_group"].isin(UHI_UCI_GROUPS)
        & raw["period_phase"].isin(["NHW", "HW"])
    ].replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])

    # Exactly one equal-weighted value per pair and phase.
    pair_phase = (
        raw.groupby(
            [
                "outcome",
                "component",
                "unit",
                "pair_id",
                "uhi_uci_group",
                "period_phase",
            ],
            observed=True,
        )["value"]
        .mean()
        .reset_index()
    )

    paired = (
        pair_phase.pivot_table(
            index=[
                "outcome",
                "component",
                "unit",
                "pair_id",
                "uhi_uci_group",
            ],
            columns="period_phase",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    paired.columns.name = None
    if not {"NHW", "HW"}.issubset(paired.columns):
        raise ValueError("Paired amplification inputs do not contain both NHW and HW.")

    paired = paired.dropna(subset=["NHW", "HW"]).copy()
    paired["R_Y"] = paired["HW"] - paired["NHW"]
    paired["response_definition"] = (
        "R_p=(Y_urban-Y_rural)_HW-(Y_urban-Y_rural)_NHW"
    )

    pair_path = out_dir / "main_hw_amplification_pair_values.csv"
    paired.to_csv(pair_path, index=False)

    total = paired[paired["component"].eq("total")].copy()
    if total.empty:
        raise ValueError("No total-burden rows available for paired amplification summary.")

    records: List[dict] = []
    ci_method = (
        "UHI/UCI-stratified station-pair bootstrap percentile 95% CI; "
        "each resampled pair retains matched NHW and HW values"
    )

    for outcome, outcome_data in total.groupby("outcome", observed=True):
        outcome_data = outcome_data.copy()
        unit = str(outcome_data["unit"].iloc[0])
        uhi = outcome_data[outcome_data["uhi_uci_group"].eq("UHI")].copy()
        uci = outcome_data[outcome_data["uhi_uci_group"].eq("UCI")].copy()

        if uhi.empty or uci.empty:
            raise ValueError(
                f"{outcome}: paired summary requires both UHI and UCI pairs."
            )

        def _means(frame: pd.DataFrame) -> Tuple[float, float, float]:
            return (
                float(frame["NHW"].mean()),
                float(frame["HW"].mean()),
                float(frame["R_Y"].mean()),
            )

        uhi_nhw, uhi_hw, uhi_r = _means(uhi)
        uci_nhw, uci_hw, uci_r = _means(uci)
        group_difference = uci_r - uhi_r

        rng = np.random.default_rng(
            _stable_bootstrap_seed("paired_hw_amplification", outcome)
        )
        boot_uhi = np.empty(BOOTSTRAP_N, dtype=float)
        boot_uci = np.empty(BOOTSTRAP_N, dtype=float)
        boot_diff = np.empty(BOOTSTRAP_N, dtype=float)
        uhi_vals = uhi["R_Y"].to_numpy(float)
        uci_vals = uci["R_Y"].to_numpy(float)

        for i in range(BOOTSTRAP_N):
            uhi_draw = rng.choice(uhi_vals, size=len(uhi_vals), replace=True)
            uci_draw = rng.choice(uci_vals, size=len(uci_vals), replace=True)
            boot_uhi[i] = float(np.mean(uhi_draw))
            boot_uci[i] = float(np.mean(uci_draw))
            boot_diff[i] = boot_uci[i] - boot_uhi[i]

        uhi_lo, uhi_hi = _percentile_ci(boot_uhi)
        uci_lo, uci_hi = _percentile_ci(boot_uci)
        diff_lo, diff_hi = _percentile_ci(boot_diff)

        common = {
            "outcome": str(outcome),
            "component": "total",
            "unit": unit,
            "response_definition": (
                "R_p=(Y_urban-Y_rural)_HW-(Y_urban-Y_rural)_NHW"
            ),
            "ci_method": ci_method,
            "n_boot_requested": int(BOOTSTRAP_N),
            "n_uhi": int(len(uhi)),
            "n_uci": int(len(uci)),
        }

        records.extend([
            {
                **common,
                "group": "UHI",
                "summary_statistic": "mean_amplification",
                "estimate": uhi_r,
                "ci_low": float(uhi_lo),
                "ci_high": float(uhi_hi),
                "matched_nhw_mean": uhi_nhw,
                "matched_hw_mean": uhi_hw,
                "n_pairs": int(len(uhi)),
            },
            {
                **common,
                "group": "UCI",
                "summary_statistic": "mean_amplification",
                "estimate": uci_r,
                "ci_low": float(uci_lo),
                "ci_high": float(uci_hi),
                "matched_nhw_mean": uci_nhw,
                "matched_hw_mean": uci_hw,
                "n_pairs": int(len(uci)),
            },
            {
                **common,
                "group": "UCI_minus_UHI",
                "summary_statistic": "difference_in_mean_amplification",
                "estimate": group_difference,
                "ci_low": float(diff_lo),
                "ci_high": float(diff_hi),
                "matched_nhw_mean": uci_nhw - uhi_nhw,
                "matched_hw_mean": uci_hw - uhi_hw,
                "n_pairs": int(len(uhi) + len(uci)),
            },
        ])

    summary_df = pd.DataFrame(records)
    summary_path = out_dir / "main_hw_amplification_paired_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ wrote paired amplification values:  {pair_path}")
    print(f"  ✓ wrote paired amplification summary: {summary_path}")
    return summary_df

def _panel_b_sensitivity_row(
    *,
    outcome: str,
    group: str,
    phase: str,
    result: dict,
    metric: str,
    exposure: str,
    unit: str,
    method: str,
    data_source: Path,
    sensitivity_day=np.nan,
    sensitivity_night=np.nan,
    notes: str = "",
) -> dict:
    """
    Build a generic panel-b row.

    Legacy beta/contrast column names are retained only so older consumers can
    still open the CSV. The plotting script reads panel_b_estimate and the
    panel_b_ci_* columns.
    """
    estimate = float(result["estimate"])
    return {
        "outcome": outcome,
        "uhi_uci_group": group,
        "period_phase": phase,
        "panel_b_estimate": estimate,
        "panel_b_ci_low": float(result["ci_low"]),
        "panel_b_ci_high": float(result["ci_high"]),
        "panel_b_metric": metric,
        "panel_b_exposure": exposure,
        "panel_b_unit": unit,
        "sensitivity_day": sensitivity_day,
        "sensitivity_night": sensitivity_night,
        # Backward-compatible aliases; this value is not universally a
        # night-minus-day contrast.
        "beta_day": sensitivity_day,
        "beta_night": sensitivity_night,
        "beta_contrast_night_minus_day": estimate,
        "se_contrast": result.get("se"),
        "ci_low": float(result["ci_low"]),
        "ci_high": float(result["ci_high"]),
        "ci_method": BOOTSTRAP_CI_METHOD,
        "n_boot_success": BOOTSTRAP_N if result["n_pairs"] > 1 else None,
        "n_boot_requested": BOOTSTRAP_N,
        "unit": unit,
        "method": method,
        "n_obs": int(result["n_pairs"]),
        "n_clusters": int(result["n_pairs"]),
        "data_source": str(data_source),
        "notes": notes,
    }


def build_sleep_panel_bc_outputs(
    audit_obj: dict,
    lookup: pd.DataFrame,
    paired_rows: Optional[List[dict]] = None,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Build Figure 4 sleep inputs using the original nighttime-only sleep model.

    Panel b uses the signed urban-minus-rural sensitivity contrast:
        Delta S_sleep = S_urban - S_rural,
    where each station sensitivity is the +1 degree C finite difference of the
    Minor et al. nighttime-Tmin sleep-loss model.

    Panel c:
        day = 0
        night = total = urban-minus-rural sleep-loss amount.
    """
    contrasts: List[dict] = []
    decomps: List[dict] = []
    skipped = {}

    recs = audit_obj.get("matches", {}).get("hne_paired_panel", [])
    if not recs:
        skipped["sleep_panel_bc"] = "missing hne_paired_panel input"
        return contrasts, decomps, skipped

    src = Path(recs[0]["path"])
    df = safe_read(src, parse_dates=["date"])
    if df is None or len(df) == 0:
        skipped["sleep_panel_bc"] = f"unreadable or empty source: {src}"
        return contrasts, decomps, skipped

    required = {
        "pair_id",
        "sleep_loss_min_U",
        "sleep_loss_min_R",
        "tmin_night_U",
        "tmin_night_R",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        skipped["sleep_panel_bc"] = (
            "missing sleep panel-b/c columns: " + ", ".join(missing)
        )
        return contrasts, decomps, skipped

    work = add_group(df, lookup, "sleep_panel_bc")
    work["pair_id"] = work["pair_id"].astype(str)

    # Existing source values are negative when sleep is reduced.
    work["_sleep_loss_amount_u"] = -pd.to_numeric(
        work["sleep_loss_min_U"], errors="coerce"
    )
    work["_sleep_loss_amount_r"] = -pd.to_numeric(
        work["sleep_loss_min_R"], errors="coerce"
    )
    work["_sleep_loss_delta"] = (
        work["_sleep_loss_amount_u"] - work["_sleep_loss_amount_r"]
    )

    t_u = pd.to_numeric(work["tmin_night_U"], errors="coerce")
    t_r = pd.to_numeric(work["tmin_night_R"], errors="coerce")
    work["_sleep_sensitivity_u"] = finite_difference_sensitivity(
        sleep_loss_amount_minutes,
        t_u.to_numpy(dtype=float),
    )
    work["_sleep_sensitivity_r"] = finite_difference_sensitivity(
        sleep_loss_amount_minutes,
        t_r.to_numpy(dtype=float),
    )
    work["_sleep_sensitivity_urban_minus_rural"] = (
        work["_sleep_sensitivity_u"]
        - work["_sleep_sensitivity_r"]
    )

    for group in UHI_UCI_GROUPS:
        for phase in ("NHW", "HW"):
            sub = group_phase(work, "hne", group, phase)
            if sub is None or len(sub) == 0:
                skipped[f"sleep_panel_bc::{group}::{phase}"] = "no rows"
                continue

            if paired_rows is not None:
                paired_sub = sub.copy()
                paired_sub["_paired_sleep_day"] = 0.0
                paired_sub["_paired_sleep_night"] = paired_sub["_sleep_loss_delta"]
                paired_sub["_paired_sleep_total"] = paired_sub["_sleep_loss_delta"]
                _append_pair_phase_components(
                    paired_rows,
                    paired_sub,
                    outcome="sleep",
                    group=group,
                    phase=phase,
                    component_columns={
                        "day": "_paired_sleep_day",
                        "night": "_paired_sleep_night",
                        "total": "_paired_sleep_total",
                    },
                    unit="min sleep loss/night",
                    data_source=src,
                )

            sensitivity = _pair_mean_bootstrap_summary(
                sub,
                "_sleep_sensitivity_urban_minus_rural",
                f"sleep_urban_minus_rural_sensitivity_{group}_{phase}",
            )
            burden = _pair_mean_bootstrap_summary(
                sub,
                "_sleep_loss_delta",
                f"sleep_burden_{group}_{phase}",
            )

            if np.isfinite(sensitivity["estimate"]):
                contrasts.append(
                    _panel_b_sensitivity_row(
                        outcome="sleep",
                        group=group,
                        phase=phase,
                        result=sensitivity,
                        metric=(
                            "urban_minus_rural_sleep_marginal_sensitivity"
                        ),
                        exposure="nighttime Tmin +1 deg C",
                        unit="min sleep loss night^-1 per deg C",
                        method=(
                            "Minor nighttime-Tmin sleep-loss model; +1 deg C "
                            "finite-difference sensitivity calculated separately "
                            "for urban and rural stations; panel-b contrast is "
                            "S_urban - S_rural; station-pair bootstrap"
                        ),
                        data_source=src,
                        sensitivity_day=0.0,
                        sensitivity_night=sensitivity["estimate"],
                        notes=(
                            "Positive means urban sleep loss is more sensitive "
                            "to a +1 deg C nighttime-Tmin increase than rural sleep loss."
                        ),
                    )
                )

            if not np.isfinite(burden["estimate"]):
                continue

            zero = {
                "estimate": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "n_pairs": burden["n_pairs"],
            }
            common = {
                "outcome": "sleep",
                "uhi_uci_group": group,
                "period_phase": phase,
                "unit": "min sleep loss/night",
                "method": (
                    "direct urban-rural sleep-loss amount from the existing "
                    "Minor et al. nighttime Tmin model"
                ),
                "data_source": str(src),
                "sign_definition": (
                    "positive = urban sleep loss > rural sleep loss"
                ),
            }
            for component, result, note in [
                ("day", zero, "structural zero; sleep model is nighttime-only"),
                ("night", burden, "nighttime Tmin sleep-loss pathway"),
                ("total", burden, "same as nighttime sleep-loss pathway"),
            ]:
                decomps.append({
                    **common,
                    "component": component,
                    "estimate": float(result["estimate"]),
                    "ci_low": float(result["ci_low"]),
                    "ci_high": float(result["ci_high"]),
                    "ci_method": BOOTSTRAP_CI_METHOD,
                    "n_boot_requested": BOOTSTRAP_N,
                    "n_obs": int(result["n_pairs"]),
                    "n_clusters": int(result["n_pairs"]),
                    "beta": None,
                    "delta_mean": None,
                    "delta_source": "direct_sleep_loss_amount_U_minus_R",
                    "exposure_source": note,
                })

    return contrasts, decomps, skipped

def build_cdh_panel_b_sensitivity(
    audit_obj: dict,
    lookup: pd.DataFrame,
) -> Tuple[List[dict], dict]:
    """
    Build panel-b CDH sensitivity with the urban-minus-rural contrast.

    The original CDH model is retained:
        CDH_h = max(T_h - 26 deg C, 0).

    For each station, a +1 degree C finite difference is calculated at every
    valid hour and averaged over the relevant hours. The signed
    urban-minus-rural sensitivity contrast (S_urban - S_rural) is then
    computed before group/phase bootstrap summarization.
    """
    rows: List[dict] = []
    skipped = {}

    recs = audit_obj.get("matches", {}).get("all_pair_period_metrics", [])
    if not recs:
        skipped["building_energy_panel_b"] = (
            "missing all_pair_period_metrics input"
        )
        return rows, skipped

    src = Path(recs[0]["path"])
    df = safe_read(src)
    if df is None or len(df) == 0:
        skipped["building_energy_panel_b"] = (
            f"unreadable or empty source: {src}"
        )
        return rows, skipped

    curve_cols = {
        *(f"urban_diurnal_h{h:02d}" for h in range(24)),
        *(f"rural_diurnal_h{h:02d}" for h in range(24)),
    }
    required = {"pair_id", "period", *curve_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        skipped["building_energy_panel_b"] = (
            "missing period-diurnal CDH sensitivity columns: "
            + ", ".join(missing)
        )
        return rows, skipped

    work = add_group(df, lookup, "building_energy_panel_b")
    work["pair_id"] = work["pair_id"].astype(str)

    all_hours = np.arange(24, dtype=int)
    day_hours = np.arange(8, 20, dtype=int)
    night_hours = np.array(
        list(range(20, 24)) + list(range(0, 8)),
        dtype=int,
    )

    def _cdh_sensitivity(curve: np.ndarray, hours: np.ndarray) -> float:
        vals = np.asarray(curve, dtype=float)[hours]
        valid = np.isfinite(vals)
        if not valid.any():
            return np.nan
        base = np.maximum(vals[valid] - CDH_THRESHOLD_C, 0.0)
        warm = np.maximum(
            vals[valid] + PANEL_B_DELTA_C - CDH_THRESHOLD_C,
            0.0,
        )
        # Common algorithm: average the hourly finite-difference response.
        return float(np.mean((warm - base) / PANEL_B_DELTA_C))

    def _row_sensitivity(row: pd.Series) -> pd.Series:
        u = np.array(
            [row.get(f"urban_diurnal_h{h:02d}", np.nan) for h in range(24)],
            dtype=float,
        )
        r = np.array(
            [row.get(f"rural_diurnal_h{h:02d}", np.nan) for h in range(24)],
            dtype=float,
        )

        u_day = _cdh_sensitivity(u, day_hours)
        r_day = _cdh_sensitivity(r, day_hours)
        u_night = _cdh_sensitivity(u, night_hours)
        r_night = _cdh_sensitivity(r, night_hours)
        u_total = _cdh_sensitivity(u, all_hours)
        r_total = _cdh_sensitivity(r, all_hours)

        def urban_minus_rural(a, b):
            return (
                a - b
                if np.isfinite(a) and np.isfinite(b)
                else np.nan
            )

        return pd.Series({
            "_cdh_sensitivity_day_urban_minus_rural": (
                urban_minus_rural(u_day, r_day)
            ),
            "_cdh_sensitivity_night_urban_minus_rural": (
                urban_minus_rural(u_night, r_night)
            ),
            "_cdh_sensitivity_total_urban_minus_rural": (
                urban_minus_rural(u_total, r_total)
            ),
        })

    derived = work.apply(_row_sensitivity, axis=1)
    work = pd.concat(
        [work.reset_index(drop=True), derived.reset_index(drop=True)],
        axis=1,
    )

    for group in UHI_UCI_GROUPS:
        for phase in ("NHW", "HW"):
            sub = group_phase(work, "cdh", group, phase)
            if sub is None or len(sub) == 0:
                skipped[
                    f"building_energy_panel_b::{group}::{phase}"
                ] = "no rows"
                continue

            total = _pair_mean_bootstrap_summary(
                sub,
                "_cdh_sensitivity_total_urban_minus_rural",
                f"cdh_urban_minus_rural_sensitivity_total_{group}_{phase}",
            )
            day = _pair_mean_bootstrap_summary(
                sub,
                "_cdh_sensitivity_day_urban_minus_rural",
                f"cdh_urban_minus_rural_sensitivity_day_{group}_{phase}",
            )
            night = _pair_mean_bootstrap_summary(
                sub,
                "_cdh_sensitivity_night_urban_minus_rural",
                f"cdh_urban_minus_rural_sensitivity_night_{group}_{phase}",
            )

            if not np.isfinite(total["estimate"]):
                continue

            rows.append(
                _panel_b_sensitivity_row(
                    outcome="building_energy",
                    group=group,
                    phase=phase,
                    result=total,
                    metric="urban_minus_rural_all_hour_cdh_marginal_sensitivity",
                    exposure="all-hour air temperature +1 deg C",
                    unit="mean hourly CDH sensitivity contrast per deg C",
                    method=(
                        "CDH=max(T-26,0); hourly +1 deg C finite-difference "
                        "response averaged over 24 h separately for each station; "
                        "panel-b contrast is S_urban - S_rural; "
                        "station-pair bootstrap"
                    ),
                    data_source=src,
                    sensitivity_day=day["estimate"],
                    sensitivity_night=night["estimate"],
                    notes=(
                        "Positive means urban CDH is more sensitive to a +1 deg C "
                        "temperature increase than rural CDH."
                    ),
                )
            )

    return rows, skipped


def build_integrated_labour_outputs2(
    audit_obj: dict,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """Disabled legacy hourly day/night labour pathway.

    The unified Figure 4 workflow must use ``build_integrated_labour_outputs``:
    Dunne labour loss at each station-specific work-hour Tx from the common-time two-harmonic temperature curve, panel-b
    urban-minus-rural marginal sensitivity contrast, and panel-c night
    structural zero.
    """
    raise RuntimeError(
        "Legacy hourly labour day/night pathway is disabled. "
        "Use build_integrated_labour_outputs()."
    )



def build_integrated_labour_outputs(
    audit_obj: dict,
    lookup: pd.DataFrame,
    paired_rows: Optional[List[dict]] = None,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """
    Build Figure 4 labour-loss inputs with the original Dunne model.

    Exposure basis:
        Tx_u = station-specific maximum of the urban common-time two-harmonic temperature curve within 08:00–19:59
        Tx_r = station-specific maximum of the rural common-time two-harmonic temperature curve within 08:00–19:59
        shaded WBGT is selected at each station's own Tx hour.

    Panel b uses the signed urban-minus-rural sensitivity contrast:
        Delta S_labour = S_urban - S_rural,
    where each S is the +1 degree C WBGT finite difference of the Dunne
    labour-loss function at the station-specific Tx-hour WBGT.

    Panel c:
        day = total = Loss_urban(Tx_u) - Loss_rural(Tx_r)
        night = 0, structural zero / not applicable.
    """
    contrasts: List[dict] = []
    reports: List[dict] = []
    decomps: List[dict] = []
    skipped = {}

    recs = audit_obj.get("matches", {}).get("labour_full", [])
    if not recs:
        skipped["labour_loss"] = "missing labour_full input in audit"
        return contrasts, reports, decomps, skipped

    src = Path(recs[0]["path"])
    df = safe_read(src)
    if df is None or len(df) == 0:
        skipped["labour_loss"] = f"failed to read or empty: {src}"
        return contrasts, reports, decomps, skipped

    required = {
        "pair_id",
        "period",
        "tx_hour_urban_dunne",
        "tx_hour_rural_dunne",
        "wbgt_tx_urban_dunne",
        "wbgt_tx_rural_dunne",
        "dloss_dunne_peak_t_diff",
        "urban_wbgt_input_ok",
        "rural_wbgt_input_ok",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        skipped["labour_loss"] = (
            "labour_loss_full.csv is missing formal common-time Dunne Tx fields: "
            + ", ".join(missing)
        )
        return contrasts, reports, decomps, skipped

    work = add_group(df, lookup, "labour_loss")
    work["pair_id"] = work["pair_id"].astype(str).str.strip()
    work["period_phase"] = work["period"].map(normalise_phase_label)
    work = work[
        work["uhi_uci_group"].isin(UHI_UCI_GROUPS)
        & work["period_phase"].isin(["HW", "NHW"])
    ].copy()

    def _row_labour(row: pd.Series) -> pd.Series:
        hour_u = pd.to_numeric(row.get("tx_hour_urban_dunne"), errors="coerce")
        hour_r = pd.to_numeric(row.get("tx_hour_rural_dunne"), errors="coerce")
        w_u = pd.to_numeric(row.get("wbgt_tx_urban_dunne"), errors="coerce")
        w_r = pd.to_numeric(row.get("wbgt_tx_rural_dunne"), errors="coerce")
        upstream_delta = pd.to_numeric(
            row.get("dloss_dunne_peak_t_diff"),
            errors="coerce",
        )
        urban_ok = (
            pd.to_numeric(row.get("urban_wbgt_input_ok", 0), errors="coerce") == 1
        )
        rural_ok = (
            pd.to_numeric(row.get("rural_wbgt_input_ok", 0), errors="coerce") == 1
        )

        valid = (
            urban_ok
            and rural_ok
            and np.isfinite(hour_u)
            and np.isfinite(hour_r)
            and np.isfinite(w_u)
            and np.isfinite(w_r)
            and np.isfinite(upstream_delta)
        )

        empty = {
            "_tx_hour_urban": np.nan,
            "_tx_hour_rural": np.nan,
            "_wbgt_urban_tx": np.nan,
            "_wbgt_rural_tx": np.nan,
            "_labour_loss_tx_u": np.nan,
            "_labour_loss_tx_r": np.nan,
            "_labour_loss_tx_delta": np.nan,
            "_labour_sensitivity_tx_u": np.nan,
            "_labour_sensitivity_tx_r": np.nan,
            "_labour_sensitivity_tx_urban_minus_rural": np.nan,
        }

        if not valid:
            return pd.Series(empty)

        if not (8 <= int(hour_u) <= 19 and 8 <= int(hour_r) <= 19):
            raise ValueError(
                "Upstream Dunne Tx hour is outside the frozen work window "
                "08:00–19:59 for "
                f"pair_id={row.get('pair_id')}, period={row.get('period')}: "
                f"urban={hour_u}, rural={hour_r}"
            )

        loss_u = float(labour_loss_pct_dunne(w_u))
        loss_r = float(labour_loss_pct_dunne(w_r))
        recomputed_delta = loss_u - loss_r

        if not np.isclose(
            recomputed_delta,
            upstream_delta,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError(
                "Upstream Dunne Tx labour-loss difference is inconsistent "
                "with the upstream Tx-hour WBGT values for "
                f"pair_id={row.get('pair_id')}, period={row.get('period')}: "
                f"upstream={upstream_delta}, recomputed={recomputed_delta}"
            )

        sens_u = float(
            finite_difference_sensitivity(labour_loss_pct_dunne, [w_u])[0]
        )
        sens_r = float(
            finite_difference_sensitivity(labour_loss_pct_dunne, [w_r])[0]
        )

        return pd.Series({
            "_tx_hour_urban": int(hour_u),
            "_tx_hour_rural": int(hour_r),
            "_wbgt_urban_tx": float(w_u),
            "_wbgt_rural_tx": float(w_r),
            "_labour_loss_tx_u": loss_u,
            "_labour_loss_tx_r": loss_r,
            # Formal pair contrast is inherited from 02 labour output.
            "_labour_loss_tx_delta": float(upstream_delta),
            "_labour_sensitivity_tx_u": sens_u,
            "_labour_sensitivity_tx_r": sens_r,
            "_labour_sensitivity_tx_urban_minus_rural": sens_u - sens_r,
        })

    derived = work.apply(
        _row_labour,
        axis=1,
    )

    work = pd.concat(
        [
            work.reset_index(drop=True),
            derived.reset_index(drop=True),
        ],
        axis=1,
    )

    # ============================================================
    # Labour-specific matched HW/NHW cohort
    # ============================================================
    # A pair is retained only when both the Dunne Tx-based labour-loss
    # estimate and its marginal sensitivity are finite in BOTH NHW and HW.
    #
    # This filtering is labour-specific. Sleep, CDH and other outcomes
    # are not affected.
    labour_required_cols = [
        "_labour_loss_tx_delta",
        "_labour_sensitivity_tx_urban_minus_rural",
    ]

    for col in labour_required_cols:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    work["_labour_row_valid"] = (
        np.isfinite(
            work["_labour_loss_tx_delta"]
            .to_numpy(dtype=float)
        )
        & np.isfinite(
            work[
                "_labour_sensitivity_tx_urban_minus_rural"
            ].to_numpy(dtype=float)
        )
    )

    valid_pair_phase = (
        work.loc[
            work["_labour_row_valid"],
            [
                "pair_id",
                "period_phase",
            ],
        ]
        .drop_duplicates(
            subset=[
                "pair_id",
                "period_phase",
            ]
        )
    )

    complete_pair_flags = (
        valid_pair_phase
        .groupby(
            "pair_id",
            observed=True,
        )["period_phase"]
        .apply(
            lambda phases: {
                "NHW",
                "HW",
            }.issubset(set(phases))
        )
    )

    complete_pair_ids = set(
        complete_pair_flags[
            complete_pair_flags
        ].index.astype(str)
    )

    n_before = int(
        work["pair_id"].nunique()
    )

    work = work[
        work["pair_id"]
        .astype(str)
        .isin(complete_pair_ids)
    ].copy()

    n_after = int(
        work["pair_id"].nunique()
    )

    matched_group_counts = (
        work[
            [
                "pair_id",
                "uhi_uci_group",
            ]
        ]
        .drop_duplicates("pair_id")
        ["uhi_uci_group"]
        .value_counts()
        .reindex(
            UHI_UCI_GROUPS,
            fill_value=0,
        )
        .astype(int)
        .to_dict()
    )

    phase_counts = (
        work[
            [
                "pair_id",
                "uhi_uci_group",
                "period_phase",
            ]
        ]
        .drop_duplicates(
            [
                "pair_id",
                "period_phase",
            ]
        )
        .groupby(
            [
                "uhi_uci_group",
                "period_phase",
            ],
            observed=True,
        )["pair_id"]
        .nunique()
        .to_dict()
    )

    print(
        "\n  [Labour matched HW/NHW cohort]"
    )

    print(
        f"    candidate pairs before matching : {n_before}"
    )

    print(
        f"    retained pairs in both periods  : {n_after}"
    )

    print(
        f"    excluded incomplete pairs       : "
        f"{n_before - n_after}"
    )

    print(
        f"    matched group counts             : "
        f"{matched_group_counts}"
    )

    print(
        f"    phase counts                     : "
        f"{phase_counts}"
    )

    # Strict internal validation: each group must use exactly the same
    # number of pairs in NHW and HW.
    for group in UHI_UCI_GROUPS:
        n_nhw = int(
            phase_counts.get(
                (group, "NHW"),
                0,
            )
        )

        n_hw = int(
            phase_counts.get(
                (group, "HW"),
                0,
            )
        )

        if n_nhw != n_hw:
            raise ValueError(
                "Labour HW/NHW matched filtering failed for "
                f"{group}: NHW={n_nhw}, HW={n_hw}"
            )

    work["_labour_loss_night"] = 0.0

    print(f"  [labour_loss] source: {src}")
    print(
        "  [labour_loss] primary model: Dunne et al. labour-capacity "
        "model converted to positive labour loss"
    )
    print(
        "  [labour_loss] exposure basis: shaded WBGT at each station's "
        "own FFT Tx hour; night is structural zero / not applicable"
    )
    print(
        "  [labour_loss] panel b: signed urban-minus-rural +1 deg C WBGT "
        "marginal sensitivity contrast"
    )

    for group in UHI_UCI_GROUPS:
        for phase in ("NHW", "HW"):
            sub = work[
                (work["uhi_uci_group"] == group)
                & (work["period_phase"] == phase)
            ].copy()
            if len(sub) == 0:
                skipped[f"labour_loss::{group}::{phase}"] = "no rows"
                continue

            if paired_rows is not None:
                paired_sub = sub.copy()
                paired_sub["_paired_labour_day"] = paired_sub["_labour_loss_tx_delta"]
                paired_sub["_paired_labour_night"] = 0.0
                paired_sub["_paired_labour_total"] = paired_sub["_labour_loss_tx_delta"]
                _append_pair_phase_components(
                    paired_rows,
                    paired_sub,
                    outcome="labour_loss",
                    group=group,
                    phase=phase,
                    component_columns={
                        "day": "_paired_labour_day",
                        "night": "_paired_labour_night",
                        "total": "_paired_labour_total",
                    },
                    unit="% labour loss",
                    data_source=src,
                )

            burden = _pair_mean_bootstrap_summary(
                sub,
                "_labour_loss_tx_delta",
                f"labour_loss_dunne_tx_{group}_{phase}",
            )
            sensitivity = _pair_mean_bootstrap_summary(
                sub,
                "_labour_sensitivity_tx_urban_minus_rural",
                f"labour_urban_minus_rural_sensitivity_dunne_tx_{group}_{phase}",
            )
            night = _pair_mean_bootstrap_summary(
                sub,
                "_labour_loss_night",
                f"labour_night_zero_{group}_{phase}",
            )

            if np.isfinite(sensitivity["estimate"]):
                contrasts.append(
                    _panel_b_sensitivity_row(
                        outcome="labour_loss",
                        group=group,
                        phase=phase,
                        result=sensitivity,
                        metric=(
                            "urban_minus_rural_tx_wbgt_marginal_"
                            "labour_loss_sensitivity_dunne"
                        ),
                        exposure=(
                            "station-specific work-hour Tx shaded WBGT derived from common-time two-harmonic T/Td curves "
                            "+1 deg C"
                        ),
                        unit="% labour loss per deg C WBGT",
                        method=(
                            "Dunne et al. labour-capacity model converted "
                            "to positive loss; urban and rural Tx-hour +1 "
                            "deg C WBGT sensitivities calculated separately; "
                            "panel-b labour value is S_urban - S_rural; "
                            "station-pair bootstrap"
                        ),
                        data_source=src,
                        sensitivity_day=sensitivity["estimate"],
                        sensitivity_night=0.0,
                        notes=(
                            "Labour only: panel-b value is the signed "
                            "urban-minus-rural marginal sensitivity contrast "
                            "S_urban - S_rural. Positive means greater urban "
                            "sensitivity; negative means greater rural "
                            "sensitivity. Night is not used."
                        ),
                    )
                )

            if not np.isfinite(burden["estimate"]):
                skipped[f"labour_loss::{group}::{phase}"] = (
                    "non-finite Dunne Tx-based labour-loss estimate"
                )
                continue

            common = {
                "outcome": "labour_loss",
                "uhi_uci_group": group,
                "period_phase": phase,
                "unit": "% labour loss",
                "method": (
                    "direct pair-level Dunne Tx-based labour-loss "
                    "difference from shaded WBGT"
                ),
                "temperature_basis": (
                    "shaded WBGT at station-specific work-hour Tx from the common-time two-harmonic temperature curve"
                ),
                "data_source": str(src),
                "notes": (
                    "Positive values mean urban labour loss > rural labour "
                    "loss. Night is not applicable and is stored as a "
                    "structural zero."
                ),
                "sign_definition": (
                    "positive = urban labour loss > rural labour loss"
                ),
            }

            reports.append({
                **common,
                "beta_day": burden["estimate"],
                "beta_night": 0.0,
                "beta_contrast_night_minus_day": np.nan,
                "panel_b_estimate": sensitivity["estimate"],
                "panel_b_ci_low": sensitivity["ci_low"],
                "panel_b_ci_high": sensitivity["ci_high"],
                "panel_b_metric": (
                    "urban_minus_rural_tx_wbgt_marginal_labour_loss_"
                    "sensitivity_dunne"
                ),
                "panel_b_unit": "% labour loss per deg C WBGT",
                "se_contrast": sensitivity["se"],
                "ci_low": sensitivity["ci_low"],
                "ci_high": sensitivity["ci_high"],
                "n_obs": int(burden["n_pairs"]),
                "n_clusters": int(burden["n_pairs"]),
                "ycol": "Dunne Tx-based labour loss",
                "xday": "station-specific work-hour Tx shaded WBGT derived from common-time two-harmonic T/Td curves",
                "xnight": "not_applicable_structural_zero",
                "se_day": burden["se"],
                "se_night": 0.0,
                "ci_day_low": burden["ci_low"],
                "ci_day_high": burden["ci_high"],
                "ci_night_low": 0.0,
                "ci_night_high": 0.0,
                "cov_day_night": np.nan,
                "fe_spec": "not_applicable_direct_pair_level_labour_loss",
                "cluster": "pair_id",
                "within_r2": None,
                "adj_within_r2": None,
                "rmse_within": None,
                "x_day_night_corr_raw": None,
                "x_day_night_corr_within": None,
                "x_day_night_vif_within": None,
                "delta_t_day_mean": None,
                "delta_t_night_mean": None,
                "delta_t_source": "station_specific_FFT_Tx_hour_WBGT",
                "exposure_source": (
                    "station-specific work-hour Tx shaded WBGT derived from common-time two-harmonic T/Td curves; "
                    "night not applicable"
                ),
                "source_day_column": "wbgt_urban/rural_h00-h23 at Tx",
                "source_night_column": "structural_zero",
                "source_work_column": "not_applicable_Tx_based",
                "work_mean_estimate": burden["estimate"],
                "work_mean_ci_low": burden["ci_low"],
                "work_mean_ci_high": burden["ci_high"],
            })

            total = burden
            for component, result, source_column, exposure_note in [
                (
                    "day",
                    burden,
                    "Dunne loss at station-specific FFT Tx-hour WBGT",
                    "Tx-hour daytime labour-loss pathway",
                ),
                (
                    "night",
                    night,
                    "structural_zero",
                    "structural zero; nighttime labour endpoint is not applicable",
                ),
                (
                    "total",
                    total,
                    "Dunne loss at station-specific FFT Tx-hour WBGT",
                    "same as Tx-hour daytime labour-loss pathway",
                ),
            ]:
                decomps.append({
                    **common,
                    "component": component,
                    "estimate": float(result["estimate"]),
                    "ci_low": float(result["ci_low"]),
                    "ci_high": float(result["ci_high"]),
                    "ci_method": BOOTSTRAP_CI_METHOD,
                    "n_boot_requested": BOOTSTRAP_N,
                    "beta": None,
                    "delta_mean": None,
                    "delta_source": "station_specific_FFT_Tx_hour_WBGT",
                    "exposure_source": exposure_note,
                    "n_obs": int(result["n_pairs"]),
                    "n_clusters": int(result["n_pairs"]),
                    "source_value_column": source_column,
                    "source_group_column": "annual canonical lookup",
                    "source_period_column": "period",
                    "source_hour_column": (
                        "station-specific work-hour Tx from the common-time two-harmonic temperature curve"
                        if component != "night"
                        else "not_applicable"
                    ),
                    "n_hours": 1 if component != "night" else 0,
                    "tx_hours": "station-specific",
                    "tn_hours": "",
                })

    print(
        f"  [labour_loss] generated rows: "
        f"panel_b={len(contrasts)}, reports={len(reports)}, "
        f"panel_c={len(decomps)}"
    )
    return contrasts, reports, decomps, skipped

def clustered_direct_mean_2d(
    df: pd.DataFrame,
    *,
    day_col: str,
    night_col: str,
    cluster_col: str = "pair_id",
) -> dict:
    """Equal-pair mean and covariance for direct day/night components.

    Multiple rows within each station pair are averaged first. Point estimates
    and uncertainty are then calculated across pair means, ensuring that every
    station pair has equal weight.
    """
    needed = {day_col, night_col, cluster_col}
    miss = needed - set(df.columns)
    if miss:
        return {
            "error": f"missing columns: {sorted(miss)}",
            "n_obs": 0,
            "n_clusters": 0,
        }

    work = df[[cluster_col, day_col, night_col]].copy()
    work[day_col] = pd.to_numeric(work[day_col], errors="coerce")
    work[night_col] = pd.to_numeric(work[night_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    n_obs = int(len(work))

    pair_means = (
        work.groupby(work[cluster_col].astype(str), observed=True)[
            [day_col, night_col]
        ]
        .mean()
        .dropna()
    )
    n_pairs = int(len(pair_means))
    if n_pairs < 5:
        return {
            "error": f"only {n_pairs} valid station pairs",
            "n_obs": n_obs,
            "n_clusters": n_pairs,
        }

    vals = pair_means[[day_col, night_col]].to_numpy(float)
    means = vals.mean(axis=0)
    vcov = np.cov(vals, rowvar=False, ddof=1) / float(n_pairs)
    vcov = np.atleast_2d(vcov)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    return {
        "mean_day": float(means[0]),
        "mean_night": float(means[1]),
        "se_day": float(se[0]),
        "se_night": float(se[1]),
        "cov_day_night": float(vcov[0, 1]),
        "n_obs": n_obs,
        "n_clusters": n_pairs,
        "vcov": vcov,
    }

def build_cdh_energy_outputs(
    audit_obj: dict,
    lookup: pd.DataFrame,
    paired_rows: Optional[List[dict]] = None,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """
    Build building-energy rows directly from day/night CDH columns.

    Energy representation:
        day   = CDH_day_u   - CDH_day_r
        night = CDH_night_u - CDH_night_r

    This replaces the previous residential/commercial energy regression using
    E_resi_u and E_comm_u.

    Sign:
        positive = urban CDH burden > rural CDH burden

    Output schema intentionally keeps beta_day/beta_night field names for
    compatibility with downstream plotting, but these are direct empirical
    mean U-R CDH differences, not regression coefficients.
    """
    contrasts: List[dict] = []
    reports: List[dict] = []
    decomps: List[dict] = []
    skipped = {}

    rec = audit_obj.get("matches", {}).get("cdh_daily_panel", [])
    if not rec:
        skipped["building_energy"] = "missing cdh_daily_panel input in audit"
        print("  ! cdh_daily_panel missing in audit", file=sys.stderr)
        return contrasts, reports, decomps, skipped

    src = Path(rec[0]["path"])
    df = safe_read(src, parse_dates=["local_date"])

    if df is None or len(df) == 0:
        skipped["building_energy"] = f"unreadable or empty source: {src}"
        print(f"  ! cdh_daily_panel unreadable or empty: {src}", file=sys.stderr)
        return contrasts, reports, decomps, skipped

    day_u = first_existing_col(df, ["CDH_day_u", "cdh_day_u"])
    day_r = first_existing_col(df, ["CDH_day_r", "cdh_day_r"])
    night_u = first_existing_col(df, ["CDH_night_u", "cdh_night_u"])
    night_r = first_existing_col(df, ["CDH_night_r", "cdh_night_r"])

    required = {
        "pair_id": "pair_id" if "pair_id" in df.columns else None,
        "period": "period" if "period" in df.columns else None,
        "CDH_day_u": day_u,
        "CDH_day_r": day_r,
        "CDH_night_u": night_u,
        "CDH_night_r": night_r,
    }
    miss = [k for k, v in required.items() if v is None]

    if miss:
        skipped["building_energy"] = (
            f"missing required CDH energy columns: {miss}; "
            f"available columns={list(df.columns)}"
        )
        print(f"  ! building_energy missing columns: {miss}", file=sys.stderr)
        return contrasts, reports, decomps, skipped

    df = add_group(df, lookup, "building_energy")

    df["_cdh_day_delta"] = pd.to_numeric(df[day_u], errors="coerce") - pd.to_numeric(df[day_r], errors="coerce")
    df["_cdh_night_delta"] = pd.to_numeric(df[night_u], errors="coerce") - pd.to_numeric(df[night_r], errors="coerce")

    print("  [building_energy] source:", src)
    print("  [building_energy] raw rows:", len(df))
    print("  [building_energy] source columns:", {
        "day_u": day_u,
        "day_r": day_r,
        "night_u": night_u,
        "night_r": night_r,
    })

    for gg in UHI_UCI_GROUPS:
        for ph in ("NHW", "HW"):
            sub = group_phase(df, "cdh", gg, ph)

            if sub is None or len(sub) < 30:
                skipped[f"building_energy::{gg}::{ph}"] = "too few rows after UHI/UCI and phase filters"
                continue

            if paired_rows is not None:
                paired_sub = sub.copy()
                paired_sub["_paired_cdh_total_delta"] = (
                    paired_sub["_cdh_day_delta"]
                    + paired_sub["_cdh_night_delta"]
                )
                _append_pair_phase_components(
                    paired_rows,
                    paired_sub,
                    outcome="building_energy",
                    group=gg,
                    phase=ph,
                    component_columns={
                        "day": "_cdh_day_delta",
                        "night": "_cdh_night_delta",
                        "total": "_paired_cdh_total_delta",
                    },
                    unit="CDH",
                    data_source=src,
                )

            ok = clustered_direct_mean_2d(
                sub,
                day_col="_cdh_day_delta",
                night_col="_cdh_night_delta",
                cluster_col="pair_id",
            )

            if "error" in ok:
                skipped[f"building_energy::{gg}::{ph}"] = ok["error"]
                continue

            day_est = float(ok["mean_day"])
            night_est = float(ok["mean_night"])
            day_se = float(ok["se_day"])
            night_se = float(ok["se_night"])
            cov_dn = float(ok["cov_day_night"])

            contrast = night_est - day_est
            se_contrast = float(np.sqrt(max(night_se ** 2 + day_se ** 2 - 2 * cov_dn, 0.0)))
            boot_ci = _bootstrap_direct_day_night_ci(
                sub,
                day_col="_cdh_day_delta",
                night_col="_cdh_night_delta",
                seed=_stable_bootstrap_seed("building_energy", gg, ph),
            )
            c_lo, c_hi = _ci_from_boot({"bootstrap_ci": boot_ci}, "contrast", contrast - 1.96 * se_contrast, contrast + 1.96 * se_contrast)

            unit = "CDH"
            method = (
                "direct pair-day mean of urban-rural building cooling degree-hour "
                "difference; no E_comm/E_resi energy regression"
            )
            notes = (
                f"Energy day={day_u}-{day_r}; night={night_u}-{night_r}. "
                "Positive means urban CDH burden > rural CDH burden."
            )

            contrasts.append({
                "outcome": "building_energy",
                "uhi_uci_group": gg,
                "period_phase": ph,
                "beta_day": day_est,
                "beta_night": night_est,
                "beta_contrast_night_minus_day": contrast,
                "se_contrast": se_contrast,
                "ci_low": c_lo,
                "ci_high": c_hi,
                "ci_method": BOOTSTRAP_CI_METHOD,
                "n_boot_success": (boot_ci.get("contrast") or {}).get("n_boot_success"),
                "n_boot_requested": (boot_ci.get("contrast") or {}).get("n_boot_requested", BOOTSTRAP_N),
                "unit": unit,
                "method": method + "; CI from station-pair cluster bootstrap",
                "n_obs": ok["n_obs"],
                "n_clusters": ok["n_clusters"],
                "data_source": str(src),
                "notes": notes,
            })

            reports.append({
                "outcome": "building_energy",
                "uhi_uci_group": gg,
                "period_phase": ph,
                "beta_day": day_est,
                "beta_night": night_est,
                "beta_contrast_night_minus_day": contrast,
                "se_contrast": se_contrast,
                "ci_low": c_lo,
                "ci_high": c_hi,
                "ci_method": BOOTSTRAP_CI_METHOD,
                "n_boot_success": (boot_ci.get("contrast") or {}).get("n_boot_success"),
                "n_boot_requested": (boot_ci.get("contrast") or {}).get("n_boot_requested", BOOTSTRAP_N),
                "unit": unit,
                "method": method + "; CI from station-pair cluster bootstrap",
                "n_obs": ok["n_obs"],
                "n_clusters": ok["n_clusters"],
                "data_source": str(src),
                "notes": notes,
                "ycol": "direct_CDH_day_night_delta",
                "xday": f"{day_u}-{day_r}",
                "xnight": f"{night_u}-{night_r}",
                "se_day": day_se,
                "se_night": night_se,
                "ci_day_low": day_est - 1.96 * day_se,
                "ci_day_high": day_est + 1.96 * day_se,
                "ci_night_low": night_est - 1.96 * night_se,
                "ci_night_high": night_est + 1.96 * night_se,
                "cov_day_night": cov_dn,
                "fe_spec": "not_applicable_direct_pair_day_cdh",
                "cluster": "pair_id",
                "within_r2": None,
                "adj_within_r2": None,
                "rmse_within": None,
                "x_day_night_corr_raw": None,
                "x_day_night_corr_within": None,
                "x_day_night_vif_within": None,
                "delta_t_day_mean": day_est,
                "delta_t_night_mean": night_est,
                "delta_t_source": "urban_minus_rural_direct_CDH",
                "exposure_source": f"{day_u}-{day_r}/{night_u}-{night_r}",
            })

            vcov = np.array(ok["vcov"], dtype=float)

            component_specs = [
                ("day", day_est, day_se, np.array([1.0, 0.0])),
                ("night", night_est, night_se, np.array([0.0, 1.0])),
                ("total", day_est + night_est, None, np.array([1.0, 1.0])),
            ]

            for comp, est, se_direct, weight in component_specs:
                if comp == "total":
                    se = float(np.sqrt(max(weight.T @ vcov @ weight, 0.0)))
                else:
                    se = float(se_direct)

                b_lo, b_hi = _ci_from_boot({"bootstrap_ci": boot_ci}, comp, float(est - 1.96 * se), float(est + 1.96 * se))
                b_rec = boot_ci.get(comp) or {}

                decomps.append({
                    "outcome": "building_energy",
                    "uhi_uci_group": gg,
                    "period_phase": ph,
                    "component": comp,
                    "estimate": float(est),
                    "ci_low": b_lo,
                    "ci_high": b_hi,
                    "ci_method": BOOTSTRAP_CI_METHOD,
                    "n_boot_success": b_rec.get("n_boot_success"),
                    "n_boot_requested": b_rec.get("n_boot_requested", BOOTSTRAP_N),
                    "unit": unit,
                    "beta": None,
                    "delta_mean": float(est) if comp in {"day", "night"} else None,
                    "delta_source": "urban_minus_rural_direct_CDH",
                    "exposure_source": f"{day_u}-{day_r}/{night_u}-{night_r}",
                    "n_obs": ok["n_obs"],
                    "n_clusters": ok["n_clusters"],
                    "data_source": str(src),
                    "method": (method if comp != "total" else "direct total = day CDH delta + night CDH delta") + "; CI from station-pair cluster bootstrap",
                    "source_day_u_column": day_u,
                    "source_day_r_column": day_r,
                    "source_night_u_column": night_u,
                    "source_night_r_column": night_r,
                    "sign_definition": "positive = urban CDH burden > rural CDH burden",
                })

    print(
        f"  [building_energy] generated rows: "
        f"contrasts={len(contrasts)}, reports={len(reports)}, decomps={len(decomps)}"
    )

    return contrasts, reports, decomps, skipped


def write_unified_panel_bc_pair_audit(
    audit_obj: dict,
    out_dir: Path,
    lookup: pd.DataFrame,
) -> Path:
    """Write one pair-level availability audit for Figure 4 panels b/c."""
    audit = lookup.rename(columns={"uhi_uci_group": "annual_group"}).copy()
    audit["pair_id"] = audit["pair_id"].astype(str)

    for c in [
        "sleep_nhw_valid", "sleep_hw_valid",
        "cdh_nhw_valid", "cdh_hw_valid",
        "labour_nhw_valid", "labour_hw_valid",
    ]:
        audit[c] = False

    def _merge_flags(flags: pd.DataFrame) -> None:
        nonlocal audit
        if flags is None or flags.empty:
            return
        flags = flags.copy()
        flags["pair_id"] = flags["pair_id"].astype(str)
        audit = audit.merge(flags, on="pair_id", how="left", suffixes=("", "_new"))
        for c in list(flags.columns):
            if c == "pair_id":
                continue
            nc = f"{c}_new"
            if nc in audit.columns:
                audit[c] = audit[c].fillna(False) | audit[nc].fillna(False)
                audit = audit.drop(columns=[nc])

    # Sleep availability.
    rec = audit_obj.get("matches", {}).get("hne_paired_panel", [])
    if rec:
        df = safe_read(Path(rec[0]["path"]))
        req = {"pair_id", "hw_flag", "sleep_loss_min_U", "sleep_loss_min_R", "tmin_night_U", "tmin_night_R"}
        if df is not None and req.issubset(df.columns):
            w = df[list(req)].copy()
            valid = w[["sleep_loss_min_U", "sleep_loss_min_R", "tmin_night_U", "tmin_night_R"]].apply(
                pd.to_numeric, errors="coerce"
            ).notna().all(axis=1)
            w = w.loc[valid].copy()
            flags = pd.DataFrame({"pair_id": w["pair_id"].astype(str)})
            flags["sleep_nhw_valid"] = pd.to_numeric(w["hw_flag"], errors="coerce").eq(0).to_numpy()
            flags["sleep_hw_valid"] = pd.to_numeric(w["hw_flag"], errors="coerce").eq(1).to_numpy()
            flags = flags.groupby("pair_id", as_index=False).max()
            _merge_flags(flags)

    # CDH availability for both panel-b hourly sensitivity and panel-c burden.
    rec = audit_obj.get("matches", {}).get("cdh_daily_panel", [])
    if rec:
        df = safe_read(Path(rec[0]["path"]))
        if df is not None and {"pair_id", "period"}.issubset(df.columns):
            required_cdh = [c for c in ["CDH_day_u", "CDH_day_r", "CDH_night_u", "CDH_night_r"] if c in df.columns]
            if len(required_cdh) == 4:
                w = df[["pair_id", "period", *required_cdh]].copy()
                valid = w[required_cdh].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
                w = w.loc[valid].copy()
                p = w["period"].astype(str).str.lower().str.strip()
                flags = pd.DataFrame({"pair_id": w["pair_id"].astype(str)})
                flags["cdh_nhw_valid"] = p.isin(["non_heatwave", "non-heatwave", "nonheatwave", "nhw"]).to_numpy()
                flags["cdh_hw_valid"] = p.isin(["heatwave", "heat_wave", "hw"]).to_numpy()
                flags = flags.groupby("pair_id", as_index=False).max()
                _merge_flags(flags)

    # Labour availability.
    rec = audit_obj.get("matches", {}).get("labour_full", [])
    if rec:
        df = safe_read(Path(rec[0]["path"]))
        needed = [
            *(f"urban_diurnal_h{h:02d}" for h in range(24)),
            *(f"rural_diurnal_h{h:02d}" for h in range(24)),
            *(f"wbgt_urban_h{h:02d}" for h in range(24)),
            *(f"wbgt_rural_h{h:02d}" for h in range(24)),
        ]
        if df is not None and {"pair_id", "period", *needed}.issubset(df.columns):
            w = df[["pair_id", "period", *needed]].copy()
            valid = w[needed].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
            w = w.loc[valid].copy()
            p = w["period"].astype(str).str.lower().str.strip()
            flags = pd.DataFrame({"pair_id": w["pair_id"].astype(str)})
            flags["labour_nhw_valid"] = p.isin(["non_heatwave", "non-heatwave", "nonheatwave", "nhw"]).to_numpy()
            flags["labour_hw_valid"] = p.isin(["heatwave", "heat_wave", "hw"]).to_numpy()
            flags = flags.groupby("pair_id", as_index=False).max()
            _merge_flags(flags)

    bool_cols = [c for c in audit.columns if c.endswith("_valid")]
    audit[bool_cols] = audit[bool_cols].fillna(False).astype(bool)
    audit["reference_hw_nhw_available"] = (
        (audit["sleep_nhw_valid"] | audit["cdh_nhw_valid"] | audit["labour_nhw_valid"])
        & (audit["sleep_hw_valid"] | audit["cdh_hw_valid"] | audit["labour_hw_valid"])
    )
    audit["all_three_outcomes_complete_hw_nhw"] = audit[bool_cols].all(axis=1)

    path = out_dir / "annual_uhi_uci_panel_bc_pair_audit.csv"
    audit.to_csv(path, index=False)
    return path

def build_models(
    audit_obj: dict,
    out_dir: Path,
    lookup_override: Optional[pd.DataFrame] = None,
):
    print("\n[MAIN B/C + REPORTING] models by UHI/UCI and HW/NHW")

    lookup = (
        lookup_override.copy()
        if lookup_override is not None
        else load_uhi_uci_lookup(audit_obj)
    )

    lookup["pair_id"] = (
        lookup["pair_id"]
        .astype(str)
        .str.strip()
    )
    write_unified_panel_bc_pair_audit(audit_obj, out_dir, lookup)

    contrasts, reports, decomps, robust = [], [], [], []
    paired_rows: List[dict] = []
    skipped = {}

    def run_spec(
        df,
        source,
        src,
        outcome,
        y,
        xd,
        xn,
        xdR,
        xnR,
        unit,
        fe,
        source_type,
        sign_flip=False,
        x_delta=False,
        note="",
        emit_panel_b=True,
        emit_panel_c=True,
    ):
        for gg in UHI_UCI_GROUPS:
            for ph in ("NHW", "HW"):
                sub = group_phase(df, source_type, gg, ph)

                if sub is None or len(sub) < 30:
                    skipped[f"{outcome}::{gg}::{ph}"] = "too few rows after filters"
                    continue

                ok = panel_reg(
                    sub,
                    y,
                    xd,
                    xn,
                    "pair_id",
                    fe,
                    xday_rural=xdR,
                    xnight_rural=xnR,
                    x_are_delta=x_delta,
                    delta_source_label=None,
                )

                if "error" in ok:
                    skipped[f"{outcome}::{gg}::{ph}"] = ok["error"]
                    continue

                ok["bootstrap_ci"] = _bootstrap_regression_ci(
                    sub,
                    y=y,
                    xd=xd,
                    xn=xn,
                    xdR=xdR,
                    xnR=xnR,
                    fe=fe,
                    x_delta=x_delta,
                    sign_flip=sign_flip,
                    seed=_stable_bootstrap_seed(outcome, gg, ph),
                )
                ok["ci_method"] = BOOTSTRAP_CI_METHOD

                _tmp_panel_b = []
                append_contrast(
                    _tmp_panel_b,
                    reports,
                    outcome=outcome,
                    group=gg,
                    phase=ph,
                    ok=ok,
                    ycol=y,
                    xday=xd,
                    xnight=xn,
                    unit=unit,
                    src=src,
                    fe=" + ".join(fe),
                    notes=note,
                    sign_flip=sign_flip,
                )
                if emit_panel_b:
                    contrasts.extend(_tmp_panel_b)

                if emit_panel_c:
                    append_decomp(
                        decomps,
                        outcome=outcome,
                        group=gg,
                        phase=ph,
                        ok=ok,
                        unit=unit,
                        src=src,
                        sign_flip=sign_flip,
                    )

                rb = panel_reg_interaction(sub, y, xd, xn, "pair_id", fe)

                if "error" not in rb:
                    append_robust(
                        robust,
                        outcome=outcome,
                        group=gg,
                        phase=ph,
                        ok=rb,
                        ycol=y,
                        xday=xd,
                        xnight=xn,
                        unit=unit,
                        src=src,
                        fe=" + ".join(fe),
                        notes=note,
                        sign_flip=sign_flip,
                    )
                else:
                    skipped[f"{outcome}_robust::{gg}::{ph}"] = rb["error"]

    # The former sleep TWFE reporting model is intentionally not run here.
    # Formal inference now uses one paired within-station-pair HW–NHW contrast
    # model for the total burden of every outcome.
    skipped["legacy_sleep_twfe_reporting"] = (
        "not run; superseded by the paired HW-NHW amplification model"
    )

    # Figure 4 panel b/c sleep inputs use the original nighttime-only
    # sleep-loss model: day=0 in panel c, and panel b is the marginal
    # nighttime-Tmin urban-minus-rural marginal sensitivity contrast.
    sleep_c, sleep_d, sleep_skipped = build_sleep_panel_bc_outputs(
        audit_obj,
        lookup,
        paired_rows=paired_rows,
    )
    contrasts.extend(sleep_c)
    decomps.extend(sleep_d)
    skipped.update(sleep_skipped)

    # Building energy
    # Do not use E_comm_u or E_resi_u.
    # Energy rows are built directly from CDH_day_u/r and CDH_night_u/r.
    energy_c, energy_r, energy_d, energy_skipped = build_cdh_energy_outputs(
        audit_obj,
        lookup,
        paired_rows=paired_rows,
    )

    # Keep the existing CDH reports and panel-c day/night decomposition.
    # Panel b uses outcome-specific marginal sensitivity instead of the former
    # night-minus-day CDH contrast.
    energy_sens, energy_sens_skipped = build_cdh_panel_b_sensitivity(
        audit_obj,
        lookup,
    )
    contrasts.extend(energy_sens)
    reports.extend(energy_r)
    decomps.extend(energy_d)
    skipped.update(energy_skipped)
    skipped.update(energy_sens_skipped)

    # Labour loss
    # Panel b uses urban-minus-rural Dunne Tx-hour marginal WBGT sensitivity contrast.
    # Panel c uses Dunne Tx-based labour loss with night=0 structural zero.
    lab_c, lab_r, lab_d, lab_skipped = build_integrated_labour_outputs(
        audit_obj,
        lookup,
        paired_rows=paired_rows,
    )

    contrasts.extend(lab_c)
    reports.extend(lab_r)
    decomps.extend(lab_d)
    skipped.update(lab_skipped)

    if not contrasts and not decomps:
        raise ValueError("no model/decomposition rows produced: " + json.dumps(skipped, indent=2))

    cdf = pd.DataFrame(contrasts)
    rdf = pd.DataFrame(reports)
    ddf = pd.DataFrame(decomps)
    bdf = pd.DataFrame(robust)

    paired_summary_df = build_paired_hw_amplification_summary(
        paired_rows,
        out_dir,
    )

    cdf.to_csv(out_dir / "main_panel_b_asymmetry_contrast.csv", index=False)
    cdf.to_csv(out_dir / "main_panel_c_asymmetry_contrast.csv", index=False)

    ddf.to_csv(out_dir / "main_panel_c_additive_decomposition.csv", index=False)
    ddf.to_csv(out_dir / "main_panel_d_additive_decomposition.csv", index=False)

    rdf.to_csv(out_dir / "main_model_reporting_coefficients.csv", index=False)
    bdf.to_csv(out_dir / "main_model_robustness_interaction.csv", index=False)

    write_json(
        out_dir / "main_model_outputs_metadata.json",
        {
            "synthetic_data_used": False,
            "stratification": "annual canonical uhi_uci_group",
            "uhi_uci_classification_period": "annual",
            "skipped": skipped,
            "amplification_summary": (
                "formal inference uses the paired within-station-pair contrast "
                "R_p=(Y_urban-Y_rural)_HW-(Y_urban-Y_rural)_NHW. Mean amplification "
                "is reported separately for annual UHI and UCI pairs, together with "
                "their difference. Confidence intervals use UHI/UCI-stratified "
                "station-pair bootstrap; no regression model is fitted."
            ),
            "figure_interpretation": (
                "panel b is an auxiliary +1 degree finite-difference marginal-sensitivity "
                "contrast; panel c contains direct outcome-specific day/night components."
            ),
            "robustness": (
                "the legacy sleep TWFE reporting model is not run; formal amplification "
                "is summarized directly from matched pair-level HW-NHW contrasts."
            ),
            "energy_source": str(FILES["cdh_daily_panel"]),
            "energy_method": (
                "Building energy is represented by CDH_day_u-CDH_day_r and "
                "CDH_night_u-CDH_night_r. No E_comm_u, E_resi_u, commercial-energy, "
                "or residential-energy representation is used."
            ),
            "energy_day_columns": ["CDH_day_u", "CDH_day_r"],
            "energy_night_columns": ["CDH_night_u", "CDH_night_r"],
            "energy_sign_definition": "positive = urban CDH burden > rural CDH burden",
            "labour_source": str(FILES["labour_full"]),
            "labour_method": (
                "Dunne et al. labour-capacity model converted to positive labour "
                "loss at station-specific work-hour Tx shaded WBGT derived from common-time two-harmonic T/Td curves."
            ),
            "labour_loss_definition": (
                "Loss_urban - Loss_rural, positive when urban labour loss is larger"
            ),
            "labour_sign_definition": "positive = urban labour loss > rural labour loss",
            "labour_temperature_basis": "station-specific work-hour Tx shaded WBGT derived from common-time two-harmonic T/Td curves",
            "labour_tx_definition": "station-specific maximum of FFT-reconstructed temperature curve",
            "labour_tx_hours": "station-specific",
            "labour_tn_hours": [],
            "uncertainty_method": BOOTSTRAP_CI_METHOD,
            "bootstrap_n_requested": BOOTSTRAP_N,
            "bootstrap_cluster": BOOTSTRAP_CLUSTER_COL,
            "labour_se_note": "Analytical SEs are retained in reporting columns, but main figure ci_low/ci_high use station-pair cluster bootstrap where pair_id is available.",
        },
    )

    (out_dir / "main_model_reporting_summary.md").write_text(
        "# Main burden reporting summary\n\n"
        "- Synthetic data used: `false`\n"
        "- Formal burden estimand: paired within-station-pair HW-NHW contrast `R_p=(Y_urban-Y_rural)_HW-(Y_urban-Y_rural)_NHW`. Mean amplification is reported separately for annual UHI and UCI pairs, together with their difference; no regression model is fitted.\n"
        "- Building energy rows: direct U-R CDH day/night differences from `CDH_day_u`, `CDH_day_r`, `CDH_night_u`, `CDH_night_r`.\n"
        "- Energy no longer uses `E_comm_u`, `E_resi_u`, residential energy, or commercial energy.\n"
        "- Labour-loss rows: Dunne Tx-based loss from shaded WBGT at each station-specific work-hour Tx from the common-time two-harmonic temperature curve.\n"
        "- Labour sign: positive means urban labour loss > rural labour loss.\n"
        "- Panel b: original-model +1 degree finite-difference sensitivity, signed urban-minus-rural contrast S_urban - S_rural.\n"
        "- Panel c: sleep day=0; labour night=0; CDH retains day and night.\n"
        "- Panel b is an auxiliary +1 degree marginal-sensitivity diagnostic; panel c is a descriptive day-night burden decomposition; neither defines the formal amplification estimator.\n\n"
        "## Skipped combinations\n```json\n"
        + json.dumps(skipped, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )

    print(
        f"  ✓ wrote model outputs; contrast rows={len(cdf)}, "
        f"decomposition rows={len(ddf)}, robustness rows={len(bdf)}, "
        f"paired-summary rows={len(paired_summary_df)}"
    )

# =============================================================================
# Labour-common cohort
# =============================================================================

def load_exact_labour_common_lookup(
    audit_obj: dict,
    standard_out_dir: Path,
) -> pd.DataFrame:
    """
    Extract the exact matched labour cohort from the formal standard run.

    No fixed number of pairs is assumed. The actual cohort size is determined
    dynamically from main_hw_amplification_pair_values.csv.

    The formal labour rows in that file have already passed:
      1. finite Dunne Tx-based labour loss;
      2. finite labour marginal sensitivity;
      3. valid NHW and HW rows for the same pair;
      4. annual canonical UHI/UCI classification.
    """
    pair_values_path = (
        standard_out_dir
        / "main_hw_amplification_pair_values.csv"
    )

    if not pair_values_path.exists():
        raise FileNotFoundError(
            "The standard model outputs must be generated before "
            "the labour-common cohort is constructed: "
            f"{pair_values_path}"
        )

    paired = pd.read_csv(pair_values_path)

    required = {
        "pair_id",
        "uhi_uci_group",
        "outcome",
        "component",
        "NHW",
        "HW",
    }

    missing = required - set(paired.columns)

    if missing:
        raise ValueError(
            "Labour-common source is missing required columns: "
            f"{sorted(missing)}"
        )

    paired["pair_id"] = (
        paired["pair_id"]
        .astype(str)
        .str.strip()
    )

    paired["outcome"] = (
        paired["outcome"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    paired["component"] = (
        paired["component"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    paired["uhi_uci_group"] = (
        paired["uhi_uci_group"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    paired["NHW"] = pd.to_numeric(
        paired["NHW"],
        errors="coerce",
    )

    paired["HW"] = pd.to_numeric(
        paired["HW"],
        errors="coerce",
    )

    labour = paired[
        paired["outcome"].isin(
            ["labour_loss", "labor_loss"]
        )
        & paired["component"].eq("total")
        & paired["uhi_uci_group"].isin(UHI_UCI_GROUPS)
    ].copy()

    labour = (
        labour.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "pair_id",
                "NHW",
                "HW",
            ]
        )
    )

    if labour.empty:
        raise ValueError(
            "No matched labour total-burden rows were found in "
            f"{pair_values_path}"
        )

    group_conflicts = (
        labour.groupby(
            "pair_id",
            observed=True,
        )["uhi_uci_group"]
        .nunique()
    )

    conflict_ids = (
        group_conflicts[
            group_conflicts > 1
        ]
        .index.astype(str)
        .tolist()
    )

    if conflict_ids:
        raise ValueError(
            "Conflicting UHI/UCI labels in the matched labour cohort. "
            f"Examples: {conflict_ids[:20]}"
        )

    labour_ids = set(
        labour["pair_id"]
        .dropna()
        .astype(str)
    )

    actual_labour_n = len(labour_ids)

    if actual_labour_n == 0:
        raise ValueError(
            "The dynamically detected matched labour cohort is empty."
        )

    canonical_lookup = load_uhi_uci_lookup(
        audit_obj
    ).copy()

    canonical_lookup["pair_id"] = (
        canonical_lookup["pair_id"]
        .astype(str)
        .str.strip()
    )

    common_lookup = canonical_lookup[
        canonical_lookup["pair_id"].isin(
            labour_ids
        )
    ].copy()

    common_lookup = common_lookup.drop_duplicates(
        subset=["pair_id"]
    )

    retained_ids = set(
        common_lookup["pair_id"]
        .astype(str)
    )

    missing_from_canonical = sorted(
        labour_ids - retained_ids
    )

    if missing_from_canonical:
        raise ValueError(
            "Some matched labour pairs are missing from the annual "
            "canonical UHI/UCI lookup. Examples: "
            f"{missing_from_canonical[:20]}"
        )

    if len(common_lookup) != actual_labour_n:
        raise ValueError(
            "The labour-common lookup is not one row per pair: "
            f"labour pair IDs={actual_labour_n}, "
            f"lookup rows={len(common_lookup)}"
        )

    group_counts = (
        common_lookup.groupby(
            "uhi_uci_group",
            observed=True,
        )["pair_id"]
        .nunique()
        .reindex(
            UHI_UCI_GROUPS,
            fill_value=0,
        )
        .astype(int)
        .to_dict()
    )

    print(
        "\n[LABOUR-COMMON COHORT]"
    )
    print(
        f"  dynamically detected pairs : {actual_labour_n}"
    )
    print(
        f"  canonical group counts     : {group_counts}"
    )
    print(
        f"  source                     : {pair_values_path}"
    )

    return common_lookup.reset_index(drop=True)


def validate_labour_common_outputs(
    common_out_dir: Path,
    common_lookup: pd.DataFrame,
) -> None:
    """
    Verify that panel a and all panel-b/c outcomes use exactly the dynamically
    detected labour cohort.

    Because every outcome is restricted to common_lookup, an outcome with the
    full expected count must also contain the same pair IDs.
    """
    expected_ids = set(
        common_lookup["pair_id"]
        .astype(str)
        .str.strip()
    )

    if not expected_ids:
        raise ValueError(
            "The labour-common validation lookup is empty."
        )

    expected_by_group = (
        common_lookup.groupby(
            "uhi_uci_group",
            observed=True,
        )["pair_id"]
        .nunique()
        .reindex(
            UHI_UCI_GROUPS,
            fill_value=0,
        )
        .astype(int)
        .to_dict()
    )

    # -----------------------------------------------------------------
    # Panel a: exact pair-ID comparison
    # -----------------------------------------------------------------
    panel_a_path = (
        common_out_dir
        / "main_panel_a_hw_asymmetry_map.csv"
    )

    if not panel_a_path.exists():
        raise FileNotFoundError(
            f"Missing labour-common panel-a file: {panel_a_path}"
        )

    panel_a = pd.read_csv(panel_a_path)

    if "pair_id" not in panel_a.columns:
        raise ValueError(
            "The labour-common panel-a file has no pair_id column."
        )

    panel_a_ids = set(
        panel_a["pair_id"]
        .astype(str)
        .str.strip()
    )

    if panel_a_ids != expected_ids:
        missing_ids = sorted(
            expected_ids - panel_a_ids
        )

        extra_ids = sorted(
            panel_a_ids - expected_ids
        )

        raise ValueError(
            "Panel a does not use the exact labour-common cohort: "
            f"missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}; "
            f"missing examples={missing_ids[:20]}; "
            f"extra examples={extra_ids[:20]}"
        )

    # -----------------------------------------------------------------
    # Paired HW-NHW values: exact pair-ID comparison for every outcome
    # -----------------------------------------------------------------
    pair_values_path = (
        common_out_dir
        / "main_hw_amplification_pair_values.csv"
    )

    if not pair_values_path.exists():
        raise FileNotFoundError(
            f"Missing labour-common paired values: {pair_values_path}"
        )

    pair_values = pd.read_csv(pair_values_path)

    required_pair_columns = {
        "pair_id",
        "outcome",
        "component",
        "NHW",
        "HW",
    }

    missing = required_pair_columns - set(
        pair_values.columns
    )

    if missing:
        raise ValueError(
            "Labour-common paired-value output is missing columns: "
            f"{sorted(missing)}"
        )

    pair_values["pair_id"] = (
        pair_values["pair_id"]
        .astype(str)
        .str.strip()
    )

    pair_values["outcome"] = (
        pair_values["outcome"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    pair_values["component"] = (
        pair_values["component"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    total_values = pair_values[
        pair_values["component"].eq("total")
    ].copy()

    required_outcomes = {
        "sleep",
        "building_energy",
        "labour_loss",
    }

    observed_outcomes = set(
        total_values["outcome"]
    )

    missing_outcomes = sorted(
        required_outcomes - observed_outcomes
    )

    if missing_outcomes:
        raise ValueError(
            "Labour-common paired output is missing outcomes: "
            f"{missing_outcomes}"
        )

    pair_errors = []

    for outcome in sorted(required_outcomes):
        outcome_ids = set(
            total_values.loc[
                total_values["outcome"].eq(outcome),
                "pair_id",
            ]
            .astype(str)
            .str.strip()
        )

        if outcome_ids != expected_ids:
            missing_ids = sorted(
                expected_ids - outcome_ids
            )

            extra_ids = sorted(
                outcome_ids - expected_ids
            )

            pair_errors.append(
                f"{outcome}: "
                f"missing={len(missing_ids)}, "
                f"extra={len(extra_ids)}; "
                f"missing examples={missing_ids[:10]}; "
                f"extra examples={extra_ids[:10]}"
            )

    if pair_errors:
        raise ValueError(
            "Not every outcome uses the exact labour-common pair IDs:\n"
            + "\n".join(
                f"  - {msg}"
                for msg in pair_errors
            )
        )

    # -----------------------------------------------------------------
    # Panel b and panel c: group-specific count comparison
    # -----------------------------------------------------------------
    table_checks = [
        (
            "panel b",
            common_out_dir
            / "main_panel_b_asymmetry_contrast.csv",
        ),
        (
            "panel c",
            common_out_dir
            / "main_panel_c_additive_decomposition.csv",
        ),
    ]

    count_errors = []

    for table_label, path in table_checks:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing labour-common {table_label} file: {path}"
            )

        table = pd.read_csv(path)

        required = {
            "outcome",
            "uhi_uci_group",
            "period_phase",
            "n_clusters",
        }

        missing = required - set(table.columns)

        if missing:
            raise ValueError(
                f"{table_label} is missing columns required for "
                f"sample validation: {sorted(missing)}"
            )

        table["uhi_uci_group"] = (
            table["uhi_uci_group"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        table["period_phase"] = (
            table["period_phase"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        table["n_clusters"] = pd.to_numeric(
            table["n_clusters"],
            errors="coerce",
        )

        for _, row in table.iterrows():
            group = str(
                row["uhi_uci_group"]
            )

            if group not in expected_by_group:
                continue

            expected_n = int(
                expected_by_group[group]
            )

            observed_raw = row["n_clusters"]

            if not np.isfinite(observed_raw):
                count_errors.append(
                    f"{table_label}: "
                    f"{row.get('outcome')} / {group} / "
                    f"{row.get('period_phase')} has invalid n_clusters"
                )
                continue

            observed_n = int(
                round(float(observed_raw))
            )

            if observed_n != expected_n:
                component = row.get(
                    "component",
                    "not_applicable",
                )

                count_errors.append(
                    f"{table_label}: "
                    f"{row.get('outcome')} / {group} / "
                    f"{row.get('period_phase')} / {component}: "
                    f"expected={expected_n}, observed={observed_n}"
                )

    if count_errors:
        raise ValueError(
            "Some panel-b/c rows silently lost labour-common pairs:\n"
            + "\n".join(
                f"  - {msg}"
                for msg in count_errors[:100]
            )
        )

    print(
        "\n[ASSERT PASS] LABOUR-COMMON OUTPUTS"
    )
    print(
        f"  dynamically detected common pairs : {len(expected_ids)}"
    )
    print(
        f"  group counts                      : {expected_by_group}"
    )
    print(
        "  panel a IDs                       : exact"
    )
    print(
        "  sleep/CDH/labour paired IDs       : exact"
    )
    print(
        "  panel b/c counts                  : exact"
    )


def build_labour_common_figure_inputs(
    audit_obj: dict,
    standard_out_dir: Path,
) -> Path:
    """
    Generate an independent Figure 4 data set in which panel a and all
    panel-b/c outcomes use the exact dynamically detected labour cohort.
    """
    common_lookup = load_exact_labour_common_lookup(
        audit_obj=audit_obj,
        standard_out_dir=standard_out_dir,
    )

    actual_n = int(
        common_lookup["pair_id"].nunique()
    )

    if actual_n == 0:
        raise ValueError(
            "The dynamically detected labour-common cohort is empty."
        )

    common_out_dir = (
        standard_out_dir
        / LABOUR_COMMON_SUBDIR
    )

    common_out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    common_lookup.to_csv(
        common_out_dir
        / "labour_common_pair_ids.csv",
        index=False,
    )

    # Same panel-a definition, restricted only to the common pair IDs.
    build_map(
        audit_obj,
        common_out_dir,
        lookup_override=common_lookup,
    )

    # Same outcome models and bootstrap definitions, but restricted before
    # estimation to the exact common pair IDs.
    build_models(
        audit_obj,
        common_out_dir,
        lookup_override=common_lookup,
    )

    validate_labour_common_outputs(
        common_out_dir,
        common_lookup,
    )

    group_counts = (
        common_lookup.groupby(
            "uhi_uci_group",
            observed=True,
        )["pair_id"]
        .nunique()
        .reindex(
            UHI_UCI_GROUPS,
            fill_value=0,
        )
        .astype(int)
        .to_dict()
    )

    write_json(
        common_out_dir
        / "labour_common_cohort_metadata.json",
        {
            "cohort_definition": (
                "Dynamically detected matched labour-loss cohort with "
                "finite Dunne Tx-based labour loss and finite marginal "
                "sensitivity in both NHW and HW."
            ),
            "master_outcome": "labour_loss",
            "pair_count_mode": "runtime_dynamic",
            "observed_n_pairs": actual_n,
            "group_counts": group_counts,
            "pair_id_file": str(
                common_out_dir
                / "labour_common_pair_ids.csv"
            ),
            "panel_a_pair_restriction": (
                "exact dynamically detected labour-common pair IDs"
            ),
            "panel_b_pair_restriction": (
                "exact dynamically detected labour-common pair IDs "
                "before sensitivity calculation and bootstrap"
            ),
            "panel_c_pair_restriction": (
                "exact dynamically detected labour-common pair IDs "
                "before burden calculation and bootstrap"
            ),
            "fixed_expected_pair_count_used": False,
            "synthetic_data_used": False,
        },
    )

    print(
        "\n[LABOUR-COMMON PREPARATION COMPLETE]"
    )
    print(
        f"  actual common pair count : {actual_n}"
    )
    print(
        f"  group counts             : {group_counts}"
    )
    print(
        f"  output directory         : {common_out_dir}"
    )

    return common_out_dir

def append_delta(rows, sub, *, var, outcome, group, phase, day, night, unit, src, key, notes):
    tmp = pd.DataFrame({
        "pair_id": sub["pair_id"].values,
        "day": pd.to_numeric(day, errors="coerce").values,
        "night": pd.to_numeric(night, errors="coerce").values,
    })

    for ep in ["day", "night"]:
        vals = tmp[["pair_id", ep]].dropna()

        for _, r in vals.iterrows():
            rows.append({
                "variable": var,
                "outcome_group": outcome,
                "uhi_uci_group": group,
                "period_phase": phase,
                "exposure_period": ep,
                "pair_id": r["pair_id"],
                "delta_value": float(r[ep]),
                "unit": unit,
                "source_key": key,
                "data_source": str(src),
                "notes": notes,
            })

def build_supp(audit_obj: dict, out_dir: Path):
    print("\n[SUPP] day/night exposure delta data by UHI/UCI")

    lookup = load_uhi_uci_lookup(audit_obj)
    rows, skipped = [], {}

    # Sleep
    rec = audit_obj["matches"].get("hne_paired_panel", [])
    if rec:
        src = Path(rec[0]["path"])
        df = safe_read(src)

        req = {"pair_id", "hw_flag", "tmax_day_U", "tmax_day_R", "tmin_night_U", "tmin_night_R"}

        if df is not None and req.issubset(df.columns):
            df = add_group(df, lookup, "supp sleep")

            for gg in UHI_UCI_GROUPS:
                for ph in ("NHW", "HW"):
                    sub = group_phase(df, "hne", gg, ph)

                    if sub is not None and len(sub):
                        append_delta(
                            rows,
                            sub,
                            var="Sleep air temperature",
                            outcome="sleep",
                            group=gg,
                            phase=ph,
                            day=sub["tmax_day_U"] - sub["tmax_day_R"],
                            night=sub["tmin_night_U"] - sub["tmin_night_R"],
                            unit="deg C",
                            src=src,
                            key="hne_paired_panel",
                            notes="Day=tmax_day_U-tmax_day_R; night=tmin_night_U-tmin_night_R.",
                        )
        else:
            skipped["sleep"] = "missing columns or unreadable source"

    # Building energy CDH
    # Do not use Tmax/Tmin here for cooling-energy representation.
    # Energy exposure boxplot is now direct U-R CDH day/night difference.
    rec = audit_obj["matches"].get("cdh_daily_panel", [])
    if rec:
        src = Path(rec[0]["path"])
        df = safe_read(src)

        day_u = first_existing_col(df, ["CDH_day_u", "cdh_day_u"]) if df is not None else None
        day_r = first_existing_col(df, ["CDH_day_r", "cdh_day_r"]) if df is not None else None
        night_u = first_existing_col(df, ["CDH_night_u", "cdh_night_u"]) if df is not None else None
        night_r = first_existing_col(df, ["CDH_night_r", "cdh_night_r"]) if df is not None else None

        req_ok = (
            df is not None
            and "pair_id" in df.columns
            and "period" in df.columns
            and day_u is not None
            and day_r is not None
            and night_u is not None
            and night_r is not None
        )

        if req_ok:
            df = add_group(df, lookup, "supp building energy")

            for gg in UHI_UCI_GROUPS:
                for ph in ("NHW", "HW"):
                    sub = group_phase(df, "cdh", gg, ph)

                    if sub is not None and len(sub):
                        append_delta(
                            rows,
                            sub,
                            var="Building cooling degree-hours",
                            outcome="building_energy",
                            group=gg,
                            phase=ph,
                            day=pd.to_numeric(sub[day_u], errors="coerce") - pd.to_numeric(sub[day_r], errors="coerce"),
                            night=pd.to_numeric(sub[night_u], errors="coerce") - pd.to_numeric(sub[night_r], errors="coerce"),
                            unit="CDH",
                            src=src,
                            key="cdh_daily_panel",
                            notes=(
                                f"Day={day_u}-{day_r}; night={night_u}-{night_r}. "
                                "Positive means urban CDH burden > rural CDH burden."
                            ),
                        )
        else:
            skipped["building_energy"] = (
                "missing CDH_day/CDH_night urban-rural columns or unreadable source; "
                f"available columns={list(df.columns) if df is not None else None}"
            )

    skipped["labour_heat_stress"] = (
        "omitted because the main labour outcome is Tx-based Dunne labour loss, not a separate dWBGT exposure panel"
    )

    if not rows:
        write_json(
            out_dir / "supp_daynight_delta_boxplot_data_metadata.json",
            {
                "synthetic_data_used": False,
                "skipped": skipped,
            },
        )
        print("  ⚠ no supplementary rows produced")
        return

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "supp_daynight_delta_boxplot_data.csv", index=False)

    write_json(
        out_dir / "supp_daynight_delta_boxplot_data_metadata.json",
        {
            "synthetic_data_used": False,
            "stratification": "annual canonical uhi_uci_group",
            "n_rows": int(len(out)),
            "energy_method": (
                "Building energy supplementary delta uses direct U-R CDH differences: "
                "CDH_day_u-CDH_day_r and CDH_night_u-CDH_night_r."
            ),
            "skipped": skipped,
        },
    )

    print(f"  ✓ wrote supp_daynight_delta_boxplot_data.csv ({len(out):,} rows)")

def summary(out_dir: Path):
    files = [
        "main_panel_a_hw_asymmetry_map.csv",
        "main_panel_b_asymmetry_contrast.csv",
        "main_panel_c_additive_decomposition.csv",
        "main_model_reporting_coefficients.csv",
        "main_model_robustness_interaction.csv",
        "supp_daynight_delta_boxplot_data.csv",
        "annual_uhi_uci_panel_bc_pair_audit.csv",
        "main_hw_amplification_pair_values.csv",
        "main_hw_amplification_paired_summary.csv",
    ]

    fig4_labour_source = FILES["labour_full"]

    s = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "original_files_modified": False,
        "synthetic_data_used": False,
        "uhi_uci_source": str(FILES["all_pair_period_metrics"]),
        "uhi_uci_classification_period": "annual",
        "formal_amplification_summary": (
            "For each outcome, compute pair-level R_p=(Y_urban-Y_rural)_HW-"
            "(Y_urban-Y_rural)_NHW, report its mean separately for annual UHI and "
            "UCI pairs, and report the UCI-minus-UHI difference. Bootstrap resampling "
            "is stratified by annual UHI/UCI group and preserves the matched NHW/HW "
            "values of each selected pair; no regression model is fitted."
        ),
        "panel_b_algorithm": (
            "Auxiliary interpretation only: apply a +1 degree perturbation to each "
            "original outcome function and compute S_urban-S_rural."
        ),
        "energy_source": str(FILES["cdh_daily_panel"]),
        "energy_representation": (
            "building_energy = direct urban-rural CDH day/night difference; "
            "day=CDH_day_u-CDH_day_r; night=CDH_night_u-CDH_night_r"
        ),
        "energy_sign_definition": (
            "positive = urban CDH burden > rural CDH burden"
        ),
        "labour_source": str(fig4_labour_source),
        "labour_outcome": "labour_loss",
        "labour_model": "Dunne et al. labour-capacity model",
        "labour_loss_definition": (
            "dLoss_Tx = Loss_urban(Tx_urban) - Loss_rural(Tx_rural), "
            "where loss=100-capacity and capacity is evaluated with the Dunne "
            "model at each station-specific work-hour Tx shaded WBGT derived from common-time two-harmonic T/Td curves."
        ),
        "labour_loss_sign_definition": (
            "positive = urban labour loss > rural labour loss"
        ),
        "labour_panel_b_definition": (
            "Dunne loss sensitivity at urban Tx-hour WBGT minus "
            "Dunne loss sensitivity at rural Tx-hour WBGT; "
            "Delta S_labour = S_urban - S_rural"
        ),
        "labour_night_component": (
            "structural zero / not applicable"
        ),
        "labour_in_panel_b": True,
        "labour_in_panel_c": True,
        "labour_temperature_basis": "Tx_FFT shaded WBGT",
        "labour_tx_definition": (
            "station-specific maximum of FFT-reconstructed temperature curve"
        ),
        "labour_tx_hours": "station-specific",
        "labour_tn_hours": [],
        "files": {f: str(out_dir / f) for f in files},
        "successful": [f for f in files if (out_dir / f).exists()],
        "missing": [f for f in files if not (out_dir / f).exists()],
    }

    write_json(out_dir / "uhi_uci_run_summary.json", s)

    (out_dir / "uhi_uci_run_report.md").write_text(
        "# UHI/UCI run summary\n\n"
        "- Original files modified: `false`\n"
        "- Synthetic data used: `false`\n"
        "- UHI/UCI classification: annual canonical group only; pairs without annual classification are excluded.\n"
        "- Panel b: original-model +1 degree finite-difference sensitivity, signed urban-minus-rural contrast S_urban - S_rural.\n"
        "- Building energy day definition: `CDH_day_u - CDH_day_r`\n"
        "- Building energy night definition: `CDH_night_u - CDH_night_r`\n"
        "- Labour outcome: `labour_loss`\n"
        "- Labour model: Dunne et al. labour-capacity model, converted to positive loss.\n"
        "- Labour exposure: shaded WBGT at each station-specific work-hour Tx from the common-time two-harmonic temperature curve.\n"
        "- Labour panel c: day = total = urban-rural labour loss; night = 0 structural zero.\n"
        "- Labour sign: positive means urban labour loss > rural labour loss.\n\n"
        + "\n".join(
            [
                f"- `{f}`: {'OK' if (out_dir / f).exists() else 'MISSING'}"
                for f in files
            ]
        ),
        encoding="utf-8",
    )

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--project-root", type=str, default=None)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(UNIFIED_ROOT / "plot_data/ncc_diurnal"),
    )
    ap.add_argument("--audit-only", action="store_true")

    ap.add_argument(
        "--skip",
        type=str,
        default="",
        help="comma-separated: map,models,labour_common,supp",
    )

    ap.add_argument(
        "--n-boot",
        type=int,
        default=BOOTSTRAP_N,
        help="Number of station-pair cluster bootstrap replicates. Use 199/499 for pilot runs; 999+ for final figures.",
    )
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--bootstrap-min-success", type=int, default=BOOTSTRAP_MIN_SUCCESS)

    args = ap.parse_args()
    set_bootstrap_runtime_config(args.n_boot, args.bootstrap_seed, args.bootstrap_min_success)
    print(
        f"[BOOTSTRAP] n={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, "
        f"min_success={BOOTSTRAP_MIN_SUCCESS}, cluster={BOOTSTRAP_CLUSTER_COL}"
    )

    if args.project_root:
        reanchor_files(Path(args.project_root).resolve())

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    a = audit(out_dir)

    if args.audit_only:
        return 0

    skip = {x.strip().lower() for x in args.skip.split(",") if x.strip()}

    if "map" not in skip:
        try:
            build_map(a, out_dir)
        except Exception as e:
            print(f"  ✗ map failed: {e}", file=sys.stderr)

    models_completed = False
    models_failed = False
    models_were_skipped = "models" in skip

    if not models_were_skipped:
        try:
            build_models(
                a,
                out_dir,
            )
            models_completed = True

        except Exception as e:
            models_failed = True

            print(
                f"  ✗ models failed: {e}",
                file=sys.stderr,
            )


    if "labour_common" not in skip:
        try:
            standard_pair_values = (
                out_dir
                / "main_hw_amplification_pair_values.csv"
            )

            # If the current standard-model run failed, never use an older
            # pair-values file left in the output directory.
            if models_failed:
                raise RuntimeError(
                    "The current standard model run failed. "
                    "Labour-common preparation is stopped to prevent reuse "
                    "of a stale main_hw_amplification_pair_values.csv."
                )

            # When models were explicitly skipped, using an existing standard
            # output is permitted, but it must exist.
            if (
                not models_completed
                and not standard_pair_values.exists()
            ):
                raise RuntimeError(
                    "The standard model outputs must exist before "
                    "building the labour-common cohort. Run without "
                    "'--skip models' or provide an existing "
                    "main_hw_amplification_pair_values.csv."
                )

            common_dir = build_labour_common_figure_inputs(
                audit_obj=a,
                standard_out_dir=out_dir,
            )

            print(
                "\n[LABOUR-COMMON DONE]"
            )
            print(
                f"  outputs: {common_dir}"
            )

        except Exception as e:
            print(
                f"  ✗ labour-common preparation failed: {e}",
                file=sys.stderr,
            )
            raise

    if "supp" not in skip:
        try:
            build_supp(a, out_dir)
        except Exception as e:
            print(f"  ✗ supp failed: {e}", file=sys.stderr)

    summary(out_dir)

    print(f"\n[DONE] empirical UHI/UCI data in: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
