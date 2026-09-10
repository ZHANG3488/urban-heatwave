#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze figure4 response states.

Original scientific calculations retained; see README.md for inputs and execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import stats

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib_scalar_vector_cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Frozen/default paths and analysis constants
# =============================================================================

DEFAULT_UNIFIED_ROOT = str(Path(__file__).resolve().parents[1] / "outputs")
DEFAULT_CODE_DIR = str(Path(__file__).resolve().parents[1] / "01_data_processing")
DEFAULT_ISD_BASE = str(Path(__file__).resolve().parents[1] / "data/isd_lite")
DEFAULT_OUTPUT_SUBDIR = "analysis/figure4_response_states"

EARLY_START = 2015
PRIMARY_CUT_YEAR = 2019
LATE_END = 2024
PRIMARY_BLOCK_DEG = 10.0
PRIMARY_FOLDS = 5
DEFAULT_CV_REPEATS = 100
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 20260807
DEFAULT_ANGLE_MIN_MAGNITUDE = 0.10
VALID_GROUPS = ("UHI", "UCI")
EXPECTED_CONTINENTS = (
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Oceania",
)
CONTINENT_ALIAS = {
    "africa": "Africa",
    "asia": "Asia",
    "europe": "Europe",
    "north america": "North America",
    "south america": "South America",
    "oceania": "Oceania",
    "australia": "Oceania",
    "australasia": "Oceania",
}

MODEL_ORDER = ("M0", "M1", "M2", "M3", "M4")
MODEL_LABELS = {
    "M0": "Intercept only",
    "M1": r"Daytime scalar, $R_x^E$",
    "M2": "Optimal linear rank-one",
    "M3": "Mean-amplitude vector",
    "M4": "Day-night vector",
}
MODEL_COLORS = {
    "M0": "#9e9e9e",
    "M1": "#4d4d4d",
    "M2": "#e69f00",
    "M3": "#7b3294",
    "M4": "#1f78b4",
}
COLOR_UHI = "#d62728"
COLOR_UCI = "#1f77b4"
COLOR_ZERO = "#333333"

MM = 1.0 / 25.4
W_DOUBLE = 180.0 * MM

_REF = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Independent cross-period scalar-versus-vector predictive test.",
    )
    p.add_argument("--unified-root", default=DEFAULT_UNIFIED_ROOT)
    p.add_argument("--reference-script", default=None)
    p.add_argument("--metrics-csv", default=None)
    p.add_argument("--daily-flags-csv", default=None)
    p.add_argument("--isd-base-dir", default=DEFAULT_ISD_BASE)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cut-year", type=int, default=PRIMARY_CUT_YEAR)
    p.add_argument(
        "--alternative-cut-years",
        type=int,
        nargs="*",
        default=[2018, 2020],
        help="Additional early/late cut years for sensitivity analyses.",
    )
    p.add_argument("--min-events", type=int, default=1)
    p.add_argument("--reliability-min-events", type=int, default=2)
    p.add_argument("--block-size", type=float, default=PRIMARY_BLOCK_DEG)
    p.add_argument("--n-folds", type=int, default=PRIMARY_FOLDS)
    p.add_argument("--n-cv-repeats", type=int, default=DEFAULT_CV_REPEATS)
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--angle-min-magnitude",
        type=float,
        default=DEFAULT_ANGLE_MIN_MAGNITUDE,
    )
    p.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--skip-pca", action="store_true")
    p.add_argument("--skip-alternative-cuts", action="store_true")
    p.add_argument("--skip-reverse", action="store_true")
    p.add_argument("--skip-loco", action="store_true")
    p.add_argument("--reuse-period-metrics", action="store_true")
    p.add_argument(
        "--resume-existing-output",
        action="store_true",
        help="Allow writing into an existing output directory. Required with --reuse-period-metrics.",
    )
    p.add_argument(
        "--final-analysis",
        action="store_true",
        help="Enforce the complete 100-repeat/1000-bootstrap robustness specification.",
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


# =============================================================================
# General utilities and input resolution
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_continent(value) -> str:
    if pd.isna(value):
        return "Unknown"
    key = " ".join(str(value).strip().lower().split())
    return CONTINENT_ALIAS.get(key, str(value).strip())


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    try:
        lon1, lat1, lon2, lat2 = map(float, (lon1, lat1, lon2, lat2))
    except Exception:
        return np.nan
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def block_id(lon, lat, size: float) -> str:
    if not np.isfinite(float(lon)) or not np.isfinite(float(lat)):
        return "missing"
    bx = int(math.floor((float(lon) + 180.0) / size))
    by = int(math.floor((float(lat) + 90.0) / size))
    return f"{size:g}deg_{bx:03d}_{by:03d}"


def find_reference_script(code_dir: Path) -> Path:
    patterns = (
        "01_compute_pair_period_metrics.py",
        "01_compute_pair_period_metrics.py",
    )
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(code_dir.glob(pattern)))
    unique = []
    seen = set()
    for p in candidates:
        if p.resolve() not in seen:
            unique.append(p)
            seen.add(p.resolve())
    if not unique:
        raise FileNotFoundError(
            f"Could not locate frozen analysis 01 in {code_dir}. "
            "Pass --reference-script explicitly."
        )
    return unique[-1]


def resolve_inputs(args: argparse.Namespace) -> Dict[str, Path]:
    root = Path(args.unified_root).expanduser().resolve()
    metrics_candidates = [
        root / "analysis/main_multiyear/robustness_percentile/all_pair_period_metrics.csv",
        root / "analysis/main_multiyear/all_pair_period_metrics.csv",
    ]
    flags_candidates = [
        root / "analysis/main_multiyear/robustness_percentile/daily_heatwave_flags.csv",
        root / "analysis/main_multiyear/daily_heatwave_flags.csv",
    ]
    metrics = Path(args.metrics_csv).expanduser().resolve() if args.metrics_csv else next(
        (p for p in metrics_candidates if p.exists()), metrics_candidates[0]
    )
    flags = Path(args.daily_flags_csv).expanduser().resolve() if args.daily_flags_csv else next(
        (p for p in flags_candidates if p.exists()), flags_candidates[0]
    )
    reference = (
        Path(args.reference_script).expanduser().resolve()
        if args.reference_script
        else find_reference_script(Path(DEFAULT_CODE_DIR))
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    return {
        "root": root,
        "metrics": metrics,
        "flags": flags,
        "reference": reference,
        "isd": Path(args.isd_base_dir).expanduser().resolve(),
        "output": output,
    }


def validate_paths(paths: Mapping[str, Path]) -> None:
    for key in ("metrics", "flags", "reference", "isd"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Required {key} path not found: {paths[key]}")
    out = paths["output"]
    protected = {paths["metrics"].parent.resolve(), paths["reference"].parent.resolve()}
    if out.resolve() in protected:
        raise ValueError("Output directory must be separate from frozen input/code directories.")


def validate_final_analysis_args(args: argparse.Namespace) -> None:
    """Fail early if a run labelled final omits a prespecified robustness test."""
    if not args.final_analysis:
        return
    problems = []
    if args.n_cv_repeats < 100:
        problems.append("--n-cv-repeats must be at least 100")
    if args.n_bootstrap < 1000:
        problems.append("--n-bootstrap must be at least 1000")
    if args.skip_alternative_cuts:
        problems.append("alternative cut years cannot be skipped")
    if not {2018, 2020}.issubset(set(args.alternative_cut_years)):
        problems.append("--alternative-cut-years must include 2018 and 2020")
    if args.skip_reverse:
        problems.append("reverse-direction analysis cannot be skipped")
    if args.skip_loco:
        problems.append("leave-one-continent-out analysis cannot be skipped")
    if args.skip_pca:
        problems.append("functional PCA validation cannot be skipped")
    if args.reliability_min_events < 2:
        problems.append("--reliability-min-events must be at least 2")
    if problems:
        raise ValueError("Invalid --final-analysis specification: " + "; ".join(problems))


def protect_output_directory(path: Path, resume: bool) -> None:
    """Prevent silent replacement of an earlier completed analysis."""
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {path}. "
            "Choose a new --output-dir, or pass --resume-existing-output deliberately."
        )


def load_reference_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_analysis01_scalar_vector_ref", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import frozen reference script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "parse_pair_id",
        "load_multiyear_station_ALL",
        "subset_by_dates",
        "compute_fft",
        "reconstruct_diurnal",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Reference script is missing required functions: {missing}")
    if int(getattr(module, "N_HARMONICS", -1)) != 2:
        raise ValueError("Frozen reference must use exactly N_HARMONICS=2.")
    return module


def _worker_init(reference_script: str) -> None:
    global _REF
    _REF = load_reference_module(Path(reference_script))


def read_canonical_metadata(metrics_path: Path, block_size: float) -> pd.DataFrame:
    df = pd.read_csv(metrics_path, low_memory=False)
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().eq("percentile")].copy()
    if "period" in df.columns and df["period"].astype(str).eq("annual").any():
        df = df[df["period"].astype(str).eq("annual")].copy()
    required = ["pair_id", "group", "lon_urban", "lat_urban", "lon_rural", "lat_rural"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Canonical metrics CSV missing columns: {missing}")
    conflict = df.groupby("pair_id")["group"].nunique(dropna=True)
    if (conflict > 1).any():
        raise ValueError(f"Conflicting fixed annual groups for {int((conflict > 1).sum())} pairs.")
    keep = required + [
        c for c in (
            "continent", "kg_code", "kg_group", "climate_zone_main",
            "urban_lcz", "urban_lcz_raw", "urban_lcz_corrected", "urban_lcz_class",
            "delta_tx_annual_synth", "uhi_definition", "uhi_classification_metric",
            "delta_LAT", "delta_LON",
        ) if c in df.columns
    ]
    meta = df[keep].drop_duplicates("pair_id", keep="first").copy()
    meta["pair_id"] = meta["pair_id"].astype(str)
    meta["group"] = meta["group"].astype(str).str.upper()
    meta = meta[meta["group"].isin(VALID_GROUPS)].copy()
    meta["continent"] = meta.get("continent", "Unknown").map(normalize_continent)
    meta["abs_lat"] = pd.to_numeric(meta["lat_urban"], errors="coerce").abs()
    meta["pair_distance_km"] = [
        haversine_km(*vals)
        for vals in meta[["lon_urban", "lat_urban", "lon_rural", "lat_rural"]].to_numpy()
    ]
    meta["spatial_block"] = [
        block_id(lon, lat, block_size)
        for lon, lat in meta[["lon_urban", "lat_urban"]].to_numpy()
    ]
    return meta.sort_values("pair_id").reset_index(drop=True)


def read_daily_flags(flags_path: Path, valid_pairs: Sequence[str]) -> pd.DataFrame:
    dtype = {
        "pair_id": str,
        "urban_usaf": str,
        "urban_wban": str,
        "rural_usaf": str,
        "rural_wban": str,
    }
    flags = pd.read_csv(flags_path, dtype=dtype, low_memory=False)
    required = [
        "pair_id", "date", "warm_season_year", "is_warm_season",
        "hw_flag_percentile_warm_season", "nhw_flag_percentile_warm_season",
    ]
    missing = [c for c in required if c not in flags.columns]
    if missing:
        raise KeyError(f"Daily flags CSV missing columns: {missing}")
    flags["pair_id"] = flags["pair_id"].astype(str)
    flags = flags[flags["pair_id"].isin(set(map(str, valid_pairs)))].copy()
    flags["date"] = pd.to_datetime(flags["date"], errors="coerce")
    flags["warm_season_year"] = pd.to_numeric(flags["warm_season_year"], errors="coerce")
    for c in ("is_warm_season", "hw_flag_percentile_warm_season", "nhw_flag_percentile_warm_season"):
        flags[c] = pd.to_numeric(flags[c], errors="coerce").fillna(0).astype(int)
    flags = flags.dropna(subset=["date", "warm_season_year"])
    flags["warm_season_year"] = flags["warm_season_year"].astype(int)
    flags = flags[(flags["warm_season_year"] >= EARLY_START) & (flags["warm_season_year"] <= LATE_END)]
    flags = flags.sort_values(["pair_id", "date"]).drop_duplicates(["pair_id", "date"], keep="last")
    overlap = (
        (flags["hw_flag_percentile_warm_season"] == 1)
        & (flags["nhw_flag_percentile_warm_season"] == 1)
    )
    if overlap.any():
        raise ValueError(f"HW and NHW flags overlap on {int(overlap.sum())} rows.")
    return flags.reset_index(drop=True)


def count_hw_events(sub: pd.DataFrame) -> int:
    """Count HW runs exactly as frozen analysis 01 counts consecutive positions.

    Frozen analysis 01 applies ``detect_heatwave`` to the ordered sequence of
    available warm-season daily maxima separately within warm-season year.  A
    missing calendar date therefore does not split a run.  The exported daily
    flags already contain the final HW membership, so the identical event
    count is the number of 0-to-1 transitions in that ordered available-day
    sequence.  This function is used only for the minimum-event reliability
    filter; it never redetects heatwaves.
    """
    warm = sub[sub["is_warm_season"].eq(1)].copy()
    if warm.empty:
        return 0
    total = 0
    for _, g in warm.groupby("warm_season_year", sort=False):
        hw = (
            g.sort_values("date")
            .drop_duplicates("date", keep="last")["hw_flag_percentile_warm_season"]
            .eq(1)
        )
        total += int((hw & ~hw.shift(fill_value=False)).sum())
    return total


def count_hw_events_calendar_gap(sub: pd.DataFrame) -> int:
    """Legacy calendar-gap counter retained only for a transparent audit."""
    hw = sub[sub["hw_flag_percentile_warm_season"].eq(1)][["date", "warm_season_year"]].copy()
    if hw.empty:
        return 0
    total = 0
    for _, g in hw.groupby("warm_season_year"):
        dates = pd.Series(pd.to_datetime(g["date"].drop_duplicates().sort_values()).to_numpy())
        if len(dates):
            total += int((dates.diff().dt.days.fillna(99) != 1).sum())
    return total


def event_count_method_audit(flags: pd.DataFrame, cut_years: Sequence[int]) -> pd.DataFrame:
    """Quantify the cohort-filter impact of the corrected event counter."""
    rows = []
    for pair_id, pair_flags in flags.groupby("pair_id", sort=False):
        masks = {"FULL": pair_flags["is_warm_season"].eq(1)}
        for cut in sorted(set(map(int, cut_years))):
            masks[f"cut{cut}_E"] = pair_flags["warm_season_year"].between(EARLY_START, cut) & pair_flags["is_warm_season"].eq(1)
            masks[f"cut{cut}_L"] = pair_flags["warm_season_year"].between(cut + 1, LATE_END) & pair_flags["is_warm_season"].eq(1)
        for block, mask in masks.items():
            sub = pair_flags[mask]
            frozen_count = count_hw_events(sub)
            calendar_count = count_hw_events_calendar_gap(sub)
            rows.append({
                "pair_id": str(pair_id),
                "block": block,
                "frozen_position_event_count": frozen_count,
                "legacy_calendar_gap_event_count": calendar_count,
                "difference": frozen_count - calendar_count,
                "count_changed": bool(frozen_count != calendar_count),
            })
    return pd.DataFrame(rows)


def make_block_date_spec(flags: pd.DataFrame, cut_years: Sequence[int]) -> Dict[str, Dict[str, object]]:
    specs: Dict[str, Dict[str, object]] = {}
    for cut in sorted(set(map(int, cut_years))):
        for side, mask in (
            ("E", flags["warm_season_year"].between(EARLY_START, cut)),
            ("L", flags["warm_season_year"].between(cut + 1, LATE_END)),
        ):
            sub = flags[mask & flags["is_warm_season"].eq(1)].copy()
            hw = sub.loc[sub["hw_flag_percentile_warm_season"].eq(1), "date"]
            nhw = sub.loc[sub["nhw_flag_percentile_warm_season"].eq(1), "date"]
            specs[f"cut{cut}_{side}"] = {
                "cut_year": cut,
                "side": side,
                "hw_dates": tuple(pd.to_datetime(hw).dt.date.unique().tolist()),
                "nhw_dates": tuple(pd.to_datetime(nhw).dt.date.unique().tolist()),
                "n_hw_days": int(hw.nunique()),
                "n_nhw_days": int(nhw.nunique()),
                "n_hw_events": count_hw_events(sub),
            }
    full = flags[flags["is_warm_season"].eq(1)].copy()
    hw = full.loc[full["hw_flag_percentile_warm_season"].eq(1), "date"]
    nhw = full.loc[full["nhw_flag_percentile_warm_season"].eq(1), "date"]
    specs["FULL"] = {
        "cut_year": np.nan,
        "side": "FULL",
        "hw_dates": tuple(pd.to_datetime(hw).dt.date.unique().tolist()),
        "nhw_dates": tuple(pd.to_datetime(nhw).dt.date.unique().tolist()),
        "n_hw_days": int(hw.nunique()),
        "n_nhw_days": int(nhw.nunique()),
        "n_hw_events": count_hw_events(full),
    }
    return specs


# =============================================================================
# Frozen period-specific metric reconstruction
# =============================================================================


def shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return np.nan
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def period_metrics_from_subsets(sub_u: pd.DataFrame, sub_r: pd.DataFrame) -> Optional[Dict[str, object]]:
    fft_u = _REF.compute_fft(sub_u)
    fft_r = _REF.compute_fft(sub_r)
    if fft_u is None or fft_r is None:
        return None
    u_curve = np.asarray(
        _REF.reconstruct_diurnal(fft_u["mean"], fft_u["amplitudes"], fft_u["phases"]),
        dtype=float,
    )
    r_curve = np.asarray(
        _REF.reconstruct_diurnal(fft_r["mean"], fft_r["amplitudes"], fft_r["phases"]),
        dtype=float,
    )
    if len(u_curve) != 24 or len(r_curve) != 24:
        raise ValueError("Frozen two-harmonic reconstruction must return 24 local-hour bins.")
    delta_curve = u_curve - r_curve
    d_tmean = float(fft_u["mean"] - fft_r["mean"])
    d_amp1 = float(fft_u["amplitudes"][0] - fft_r["amplitudes"][0])
    d_amp2 = float(fft_u["amplitudes"][1] - fft_r["amplitudes"][1])
    d_tx = float(np.max(u_curve) - np.max(r_curve))
    d_tn = float(np.min(u_curve) - np.min(r_curve))
    sx = float(d_tx - (d_tmean + d_amp1))
    sn = float(d_tn - (d_tmean - d_amp1))
    area = shoelace_area(r_curve, delta_curve)
    out: Dict[str, object] = {
        "dTmean": d_tmean,
        "dAmp1": d_amp1,
        "dAmp2": d_amp2,
        "dTx": d_tx,
        "dTn": d_tn,
        "Sx": sx,
        "Sn": sn,
        "AH": area,
        "urban_Tmean": float(fft_u["mean"]),
        "rural_Tmean": float(fft_r["mean"]),
        "urban_Amp1": float(fft_u["amplitudes"][0]),
        "rural_Amp1": float(fft_r["amplitudes"][0]),
        "urban_Amp2": float(fft_u["amplitudes"][1]),
        "rural_Amp2": float(fft_r["amplitudes"][1]),
        "urban_Phase1": float(fft_u["phases"][0]),
        "rural_Phase1": float(fft_r["phases"][0]),
        "urban_Phase2": float(fft_u["phases"][1]),
        "rural_Phase2": float(fft_r["phases"][1]),
        "urban_ndays": int(fft_u["n_days"]),
        "rural_ndays": int(fft_r["n_days"]),
    }
    for h in range(24):
        out[f"urban_h{h:02d}"] = float(u_curve[h])
        out[f"rural_h{h:02d}"] = float(r_curve[h])
        out[f"delta_h{h:02d}"] = float(delta_curve[h])
    return out


def _worker_process_pair(task: Mapping[str, object]) -> Dict[str, object]:
    pair_id = str(task["pair_id"])
    try:
        usaf_u, wban_u, usaf_r, wban_r = _REF.parse_pair_id(pair_id)
        years = list(range(EARLY_START, LATE_END + 1))
        u_all, valid_u = _REF.load_multiyear_station_ALL(
            usaf_u, wban_u, years, str(task["isd_base"]), float(task["lon_urban"]), verbose=False
        )
        r_all, valid_r = _REF.load_multiyear_station_ALL(
            usaf_r, wban_r, years, str(task["isd_base"]), float(task["lon_rural"]), verbose=False
        )
        if u_all is None or r_all is None:
            return {
                "pair_id": pair_id,
                "rows": [],
                "error": "frozen_station_loader_returned_none",
                "valid_years_u": valid_u,
                "valid_years_r": valid_r,
            }
        rows = []
        for block_name, spec in task["date_specs"].items():
            row = {
                "pair_id": pair_id,
                "block": block_name,
                "cut_year": spec["cut_year"],
                "side": spec["side"],
                "n_hw_days": spec["n_hw_days"],
                "n_nhw_days": spec["n_nhw_days"],
                "n_hw_events": spec["n_hw_events"],
                "n_valid_years_urban": len(valid_u),
                "n_valid_years_rural": len(valid_r),
            }
            if not spec["hw_dates"] or not spec["nhw_dates"]:
                row["status"] = "missing_hw_or_nhw_dates"
                rows.append(row)
                continue
            hw_u = _REF.subset_by_dates(u_all, set(spec["hw_dates"]))
            hw_r = _REF.subset_by_dates(r_all, set(spec["hw_dates"]))
            nhw_u = _REF.subset_by_dates(u_all, set(spec["nhw_dates"]))
            nhw_r = _REF.subset_by_dates(r_all, set(spec["nhw_dates"]))
            hw_m = period_metrics_from_subsets(hw_u, hw_r)
            nhw_m = period_metrics_from_subsets(nhw_u, nhw_r)
            if hw_m is None or nhw_m is None:
                row["status"] = "fft_failed"
                rows.append(row)
                continue
            for period, metrics in (("HW", hw_m), ("NHW", nhw_m)):
                prow = dict(row)
                prow["period"] = period
                prow["status"] = "PASS"
                prow.update(metrics)
                rows.append(prow)
        return {
            "pair_id": pair_id,
            "rows": rows,
            "error": "",
            "valid_years_u": valid_u,
            "valid_years_r": valid_r,
        }
    except Exception as exc:
        return {
            "pair_id": pair_id,
            "rows": [],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }


def compute_period_metrics(
    meta: pd.DataFrame,
    flags: pd.DataFrame,
    cut_years: Sequence[int],
    paths: Mapping[str, Path],
    n_workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped_flags = {pid: g.copy() for pid, g in flags.groupby("pair_id", sort=False)}
    tasks = []
    for rec in meta.to_dict("records"):
        pid = str(rec["pair_id"])
        if pid not in grouped_flags:
            continue
        tasks.append({
            "pair_id": pid,
            "lon_urban": rec["lon_urban"],
            "lat_urban": rec["lat_urban"],
            "lon_rural": rec["lon_rural"],
            "lat_rural": rec["lat_rural"],
            "isd_base": str(paths["isd"]),
            "date_specs": make_block_date_spec(grouped_flags[pid], cut_years),
        })
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(n_workers)),
        initializer=_worker_init,
        initargs=(str(paths["reference"]),),
    ) as ex:
        futs = {ex.submit(_worker_process_pair, task): task["pair_id"] for task in tasks}
        for k, fut in enumerate(as_completed(futs), start=1):
            result = fut.result()
            rows.extend(result.get("rows", []))
            if result.get("error"):
                errors.append({
                    "pair_id": result.get("pair_id", futs[fut]),
                    "error": result.get("error", ""),
                    "traceback": result.get("traceback", ""),
                })
            if k % 25 == 0 or k == len(futs):
                print(f"  Period metrics: {k}/{len(futs)} pairs completed")
    long_df = pd.DataFrame(rows)
    err_df = pd.DataFrame(errors)
    if long_df.empty:
        raise RuntimeError("No period-specific metrics were generated.")
    return long_df, err_df


def build_response_table(period_long: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    pass_df = period_long[period_long["status"].eq("PASS")].copy()
    key = ["pair_id", "block", "cut_year", "side"]
    response_rows = []
    for keys, g in pass_df.groupby(key, dropna=False, sort=False):
        if set(g["period"]) != {"HW", "NHW"}:
            continue
        h = g.set_index("period")
        rec = dict(zip(key, keys))
        rec.update({
            "n_hw_days": int(h.loc["HW", "n_hw_days"]),
            "n_nhw_days": int(h.loc["HW", "n_nhw_days"]),
            "n_hw_events": int(h.loc["HW", "n_hw_events"]),
            "Rmean": float(h.loc["HW", "dTmean"] - h.loc["NHW", "dTmean"]),
            "RAmp": float(h.loc["HW", "dAmp1"] - h.loc["NHW", "dAmp1"]),
            "RAmp2": float(h.loc["HW", "dAmp2"] - h.loc["NHW", "dAmp2"]),
            "Rx": float(h.loc["HW", "dTx"] - h.loc["NHW", "dTx"]),
            "Rn": float(h.loc["HW", "dTn"] - h.loc["NHW", "dTn"]),
            "RSx": float(h.loc["HW", "Sx"] - h.loc["NHW", "Sx"]),
            "RSn": float(h.loc["HW", "Sn"] - h.loc["NHW", "Sn"]),
            "dAH": float(h.loc["HW", "AH"] - h.loc["NHW", "AH"]),
            "AH_HW": float(h.loc["HW", "AH"]),
            "AH_NHW": float(h.loc["NHW", "AH"]),
        })
        rec["A_dn"] = rec["Rn"] - rec["Rx"]
        rec["shape_asymmetry"] = rec["RSn"] - rec["RSx"]
        rec["vector_magnitude"] = float(np.hypot(rec["Rmean"], rec["RAmp"]))
        rec["vector_angle_deg"] = float(np.degrees(np.arctan2(rec["RAmp"], rec["Rmean"])))
        for hbin in range(24):
            hw_delta = h.loc["HW", f"delta_h{hbin:02d}"]
            nhw_delta = h.loc["NHW", f"delta_h{hbin:02d}"]
            rec[f"R_h{hbin:02d}"] = float(hw_delta - nhw_delta)
        response_rows.append(rec)
    response = pd.DataFrame(response_rows)
    response = response.merge(meta, on="pair_id", how="left", validate="many_to_one")
    return response.sort_values(["block", "pair_id"]).reset_index(drop=True)


def frozen_full_period_interface_audit(
    metrics_path: Path,
    response: pd.DataFrame,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Compare 09 FULL responses with the corresponding frozen 01 outputs."""
    source = pd.read_csv(metrics_path, low_memory=False)
    if "hw_method" in source.columns:
        source = source[source["hw_method"].astype(str).str.lower().eq("percentile")]
    required = {"pair_id", "period", "dTmean", "dAmp1", "dAmp2", "dTx", "dTn"}
    if not required.issubset(source.columns):
        return pd.DataFrame([{
            "check": "frozen_FULL_numeric_interface",
            "status": "NOT_RUN",
            "n": 0,
            "max_abs_error": np.nan,
            "tolerance": tolerance,
            "details": "Frozen metrics CSV lacks one or more required period/metric columns.",
        }])
    aliases = {
        "heatwave": "HW", "hw": "HW",
        "non_heatwave": "NHW", "non-heatwave": "NHW", "nhw": "NHW",
    }
    source = source.copy()
    source["period_key"] = source["period"].astype(str).str.strip().str.lower().map(aliases)
    source = source[source["period_key"].isin(["HW", "NHW"])]
    if source.empty:
        return pd.DataFrame([{
            "check": "frozen_FULL_numeric_interface", "status": "NOT_RUN", "n": 0,
            "max_abs_error": np.nan, "tolerance": tolerance,
            "details": "No frozen heatwave/non-heatwave rows found.",
        }])
    source = source.drop_duplicates(["pair_id", "period_key"], keep="last")
    wide = source.pivot(index="pair_id", columns="period_key", values=["dTmean", "dAmp1", "dAmp2", "dTx", "dTn"])
    rows = []
    for pair_id in wide.index:
        try:
            rec = {"pair_id": str(pair_id)}
            for metric, out_name in (
                ("dTmean", "Rmean"), ("dAmp1", "RAmp"), ("dAmp2", "RAmp2"),
                ("dTx", "Rx"), ("dTn", "Rn"),
            ):
                rec[f"frozen_{out_name}"] = float(wide.loc[pair_id, (metric, "HW")] - wide.loc[pair_id, (metric, "NHW")])
            rows.append(rec)
        except (KeyError, TypeError, ValueError):
            continue
    frozen = pd.DataFrame(rows)
    current = response[response["block"].eq("FULL")][["pair_id", "Rmean", "RAmp", "RAmp2", "Rx", "Rn"]]
    merged = current.merge(frozen, on="pair_id", how="inner", validate="one_to_one")
    audit_rows = []
    for metric in ("Rmean", "RAmp", "RAmp2", "Rx", "Rn"):
        err = (merged[metric] - merged[f"frozen_{metric}"]).abs()
        finite = err[np.isfinite(err)]
        max_err = float(finite.max()) if len(finite) else np.nan
        audit_rows.append({
            "check": f"FULL_{metric}_equals_frozen_01",
            "status": "PASS" if len(finite) and max_err <= tolerance else "FAIL",
            "n": len(finite),
            "max_abs_error": max_err,
            "tolerance": tolerance,
            "details": "09 recomputation minus frozen 01 HW-NHW response",
        })
    return pd.DataFrame(audit_rows)


def algebraic_identity_audit(response: pd.DataFrame, tolerance: float = 1e-10) -> pd.DataFrame:
    checks = {
        "Rx_equals_Rmean_plus_RAmp_plus_RSx": response["Rx"] - (response["Rmean"] + response["RAmp"] + response["RSx"]),
        "Rn_equals_Rmean_minus_RAmp_plus_RSn": response["Rn"] - (response["Rmean"] - response["RAmp"] + response["RSn"]),
        "Adn_exact_decomposition": response["A_dn"] - (-2.0 * response["RAmp"] + response["RSn"] - response["RSx"]),
    }
    rows = []
    for name, error in checks.items():
        finite = np.abs(pd.to_numeric(error, errors="coerce").dropna().to_numpy(float))
        max_err = float(np.max(finite)) if len(finite) else np.nan
        rows.append({
            "check": name,
            "status": "PASS" if len(finite) and max_err <= tolerance else "FAIL",
            "n": len(finite), "max_abs_error": max_err, "tolerance": tolerance,
        })
    return pd.DataFrame(rows)


def make_predictive_cohort(response: pd.DataFrame, cut_year: int, min_events: int) -> pd.DataFrame:
    e = response[response["block"].eq(f"cut{cut_year}_E")].copy()
    l = response[response["block"].eq(f"cut{cut_year}_L")].copy()
    value_cols = [
        "Rmean", "RAmp", "RAmp2", "Rx", "Rn", "RSx", "RSn", "dAH",
        "A_dn", "shape_asymmetry",
        "n_hw_days", "n_nhw_days", "n_hw_events", "vector_magnitude", "vector_angle_deg",
    ]
    e = e[["pair_id"] + value_cols].rename(columns={c: f"{c}_E" for c in value_cols})
    l = l[["pair_id"] + value_cols].rename(columns={c: f"{c}_L" for c in value_cols})
    meta_cols = [
        "pair_id", "group", "continent", "lon_urban", "lat_urban", "lon_rural", "lat_rural",
        "spatial_block", "abs_lat", "pair_distance_km",
    ] + [c for c in ("kg_code", "kg_group", "climate_zone_main", "urban_lcz_corrected", "urban_lcz_class") if c in response.columns]
    meta = response[[c for c in meta_cols if c in response.columns]].drop_duplicates("pair_id")
    cohort = meta.merge(e, on="pair_id", how="inner").merge(l, on="pair_id", how="inner")
    required = ["Rmean_E", "RAmp_E", "Rx_E", "Rn_E", "Rx_L", "Rn_L"]
    finite = np.isfinite(cohort[required].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    event_ok = (cohort["n_hw_events_E"] >= min_events) & (cohort["n_hw_events_L"] >= min_events)
    cohort = cohort[finite & event_ok].copy()
    cohort["D_E"] = cohort["Rmean_E"] - cohort["RAmp_E"]
    cohort["target_magnitude"] = np.hypot(cohort["Rx_L"], cohort["Rn_L"])
    cohort["cut_year"] = int(cut_year)
    cohort["min_events"] = int(min_events)
    return cohort.sort_values("pair_id").reset_index(drop=True)


# =============================================================================
# Rank-one/rank-two models and repeated spatial cross-validation
# =============================================================================


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.scale

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.scale + self.mean


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.asarray(x, dtype=float)) @ np.asarray(y, dtype=float)


def fit_rrr_rank_one(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduced-rank regression under ordinary squared-error loss.

    x and y must already be centered/standardized on the training data.
    Returns the rank-one coefficient matrix and its predictor-space latent
    direction. The latter is used only by the complementary timing test.
    """
    b_ols = fit_ols(x, y)
    fitted = np.asarray(x, dtype=float) @ b_ols
    _, singular_values, vt = np.linalg.svd(fitted, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] <= 1e-14:
        return np.zeros_like(b_ols), np.array([1.0, 0.0], dtype=float)
    v1 = vt[:1, :].T
    b_rank1 = b_ols @ v1 @ v1.T
    u, s, _ = np.linalg.svd(b_rank1, full_matrices=False)
    w = u[:, 0] if len(s) and s[0] > 1e-14 else np.array([1.0, 0.0], dtype=float)
    if w[0] < 0:
        w = -w
    return b_rank1, w


def _univariate_fit_predict(train_x, train_y, test_x) -> np.ndarray:
    xs = Standardizer.fit(np.asarray(train_x, dtype=float))
    ys = Standardizer.fit(np.asarray(train_y, dtype=float).reshape(-1, 1))
    xtr = xs.transform(np.asarray(train_x, dtype=float))
    xte = xs.transform(np.asarray(test_x, dtype=float))
    ytr = ys.transform(np.asarray(train_y, dtype=float).reshape(-1, 1))
    coef = fit_ols(xtr, ytr)
    return ys.inverse(xte @ coef).ravel()


def fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictor_suffix: str,
    outcome_suffix: str,
) -> pd.DataFrame:
    """Fit M0-M4 on one training fold and predict one held-out fold."""
    state_cols = [f"Rmean_{predictor_suffix}", f"RAmp_{predictor_suffix}"]
    daynight_cols = [f"Rx_{predictor_suffix}", f"Rn_{predictor_suffix}"]
    target_cols = [f"Rx_{outcome_suffix}", f"Rn_{outcome_suffix}"]
    timing_col = f"dAH_{outcome_suffix}"

    x_state_tr = train[state_cols].to_numpy(float)
    x_state_te = test[state_cols].to_numpy(float)
    x_dn_tr = train[daynight_cols].to_numpy(float)
    x_dn_te = test[daynight_cols].to_numpy(float)
    y_tr = train[target_cols].to_numpy(float)

    sx_state = Standardizer.fit(x_state_tr)
    sx_dn = Standardizer.fit(x_dn_tr)
    sy = Standardizer.fit(y_tr)
    state_tr_z = sx_state.transform(x_state_tr)
    state_te_z = sx_state.transform(x_state_te)
    dn_tr_z = sx_dn.transform(x_dn_tr)
    dn_te_z = sx_dn.transform(x_dn_te)
    y_tr_z = sy.transform(y_tr)

    predictions: Dict[str, np.ndarray] = {}
    predictions["M0"] = np.repeat(sy.mean.reshape(1, -1), len(test), axis=0)

    # M1: conventional daytime scalar.
    b1 = fit_ols(dn_tr_z[:, :1], y_tr_z)
    predictions["M1"] = sy.inverse(dn_te_z[:, :1] @ b1)

    # M2: optimal linear rank-one state learned entirely inside training data.
    b2, latent_w = fit_rrr_rank_one(state_tr_z, y_tr_z)
    predictions["M2"] = sy.inverse(state_te_z @ b2)

    # M3: full two-dimensional mean-amplitude vector.
    b3 = fit_ols(state_tr_z, y_tr_z)
    predictions["M3"] = sy.inverse(state_te_z @ b3)

    # M4: complete two-dimensional day-night representation.
    b4 = fit_ols(dn_tr_z, y_tr_z)
    predictions["M4"] = sy.inverse(dn_te_z @ b4)

    # Timing outcome. Rank constraints are meaningless for a single outcome;
    # M2 therefore reuses the rank-one state direction learned from the joint
    # day/night target inside the same training fold.
    timing_available = timing_col in train.columns and np.isfinite(train[timing_col]).sum() >= 10
    timing_pred: Dict[str, np.ndarray] = {m: np.full(len(test), np.nan) for m in MODEL_ORDER}
    if timing_available:
        ok = np.isfinite(train[timing_col].to_numpy(float))
        ttrain = train.loc[ok, timing_col].to_numpy(float)
        timing_pred["M0"] = np.repeat(np.mean(ttrain), len(test))
        timing_pred["M1"] = _univariate_fit_predict(
            dn_tr_z[ok, :1], ttrain, dn_te_z[:, :1]
        )
        ztr = state_tr_z[ok] @ latent_w.reshape(-1, 1)
        zte = state_te_z @ latent_w.reshape(-1, 1)
        timing_pred["M2"] = _univariate_fit_predict(ztr, ttrain, zte)
        timing_pred["M3"] = _univariate_fit_predict(
            state_tr_z[ok], ttrain, state_te_z
        )
        timing_pred["M4"] = _univariate_fit_predict(
            dn_tr_z[ok], ttrain, dn_te_z
        )

    out_rows = []
    observed = test[target_cols].to_numpy(float)
    timing_obs = test[timing_col].to_numpy(float) if timing_col in test.columns else np.full(len(test), np.nan)
    for model in MODEL_ORDER:
        pred = predictions[model]
        for j, (_, rec) in enumerate(test.iterrows()):
            out_rows.append({
                "pair_id": rec["pair_id"],
                "spatial_block": rec["spatial_block"],
                "continent": rec.get("continent", "Unknown"),
                "group": rec.get("group", ""),
                "model": model,
                "Rx_obs": observed[j, 0],
                "Rn_obs": observed[j, 1],
                "Rx_pred": pred[j, 0],
                "Rn_pred": pred[j, 1],
                "dAH_obs": timing_obs[j],
                "dAH_pred": timing_pred[model][j],
            })
    return pd.DataFrame(out_rows)


def balanced_group_folds(
    groups: pd.Series,
    n_folds: int,
    rng: np.random.Generator,
) -> Dict[str, int]:
    counts = groups.value_counts().to_dict()
    group_names = list(counts)
    rng.shuffle(group_names)
    jitter = {g: rng.random() for g in group_names}
    group_names.sort(key=lambda g: (-counts[g], jitter[g]))
    totals = np.zeros(n_folds, dtype=int)
    mapping: Dict[str, int] = {}
    for group in group_names:
        candidate = np.flatnonzero(totals == totals.min())
        fold = int(rng.choice(candidate))
        mapping[str(group)] = fold
        totals[fold] += int(counts[group])
    return mapping


def evaluation_scales(oof: pd.DataFrame) -> Tuple[float, float]:
    obs = oof.drop_duplicates("pair_id")[["Rx_obs", "Rn_obs"]]
    # pandas 2.x/3.x may expose a read-only NumPy view (notably with
    # copy-on-write enabled).  A private writable copy is required because
    # the next line replaces degenerate scales in place.  This is a runtime
    # compatibility fix only; it does not change the estimand.
    scales = obs.std(ddof=1).to_numpy(dtype=float, copy=True)
    scales[~np.isfinite(scales) | (scales <= 1e-12)] = 1.0
    return float(scales[0]), float(scales[1])


def angular_error_deg(obs: np.ndarray, pred: np.ndarray, threshold: float) -> np.ndarray:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    obs_mag = np.linalg.norm(obs, axis=1)
    pred_mag = np.linalg.norm(pred, axis=1)
    ok = (obs_mag >= threshold) & (pred_mag > 1e-12)
    result = np.full(len(obs), np.nan)
    if ok.any():
        cosine = np.sum(obs[ok] * pred[ok], axis=1) / (obs_mag[ok] * pred_mag[ok])
        result[ok] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return result


def score_oof_predictions(
    oof: pd.DataFrame,
    angle_threshold: float,
    scenario: str,
    repeat: Optional[int] = None,
) -> pd.DataFrame:
    if oof.empty:
        return pd.DataFrame()
    sx, sn = evaluation_scales(oof)
    m0 = oof[oof["model"].eq("M0")][["pair_id", "Rx_pred", "Rn_pred", "dAH_pred"]].rename(
        columns={"Rx_pred": "Rx_m0", "Rn_pred": "Rn_m0", "dAH_pred": "dAH_m0"}
    )
    rows = []
    for model in MODEL_ORDER:
        d = oof[oof["model"].eq(model)].merge(m0, on="pair_id", how="left", validate="one_to_one")
        if d.empty:
            continue
        rx_res = d["Rx_obs"].to_numpy(float) - d["Rx_pred"].to_numpy(float)
        rn_res = d["Rn_obs"].to_numpy(float) - d["Rn_pred"].to_numpy(float)
        rx_base = d["Rx_obs"].to_numpy(float) - d["Rx_m0"].to_numpy(float)
        rn_base = d["Rn_obs"].to_numpy(float) - d["Rn_m0"].to_numpy(float)
        joint_sse = float(np.sum((rx_res / sx) ** 2 + (rn_res / sn) ** 2))
        joint_base = float(np.sum((rx_base / sx) ** 2 + (rn_base / sn) ** 2))
        rx_base_sse = float(np.sum(rx_base ** 2))
        rn_base_sse = float(np.sum(rn_base ** 2))
        obs = d[["Rx_obs", "Rn_obs"]].to_numpy(float)
        pred = d[["Rx_pred", "Rn_pred"]].to_numpy(float)
        global_mean = np.mean(obs, axis=0)
        global_res = obs - global_mean
        joint_global_base = float(np.sum((global_res[:, 0] / sx) ** 2 + (global_res[:, 1] / sn) ** 2))
        global_rx_base = float(np.sum(global_res[:, 0] ** 2))
        global_rn_base = float(np.sum(global_res[:, 1] ** 2))
        angles = angular_error_deg(obs, pred, angle_threshold)
        rx_cal = stats.linregress(pred[:, 0], obs[:, 0]) if np.std(pred[:, 0]) > 1e-12 else None
        rn_cal = stats.linregress(pred[:, 1], obs[:, 1]) if np.std(pred[:, 1]) > 1e-12 else None
        redistribution_obs = obs[:, 1] - obs[:, 0]
        redistribution_pred = pred[:, 1] - pred[:, 0]
        rec = {
            "scenario": scenario,
            "repeat": repeat,
            "model": model,
            "n": len(d),
            "Q2_joint": 1.0 - joint_sse / joint_base if joint_base > 0 else np.nan,
            "sRMSE_joint": math.sqrt(joint_sse / (2.0 * len(d))),
            "Q2_Rx": 1.0 - float(np.sum(rx_res ** 2)) / rx_base_sse if rx_base_sse > 0 else np.nan,
            "Q2_Rn": 1.0 - float(np.sum(rn_res ** 2)) / rn_base_sse if rn_base_sse > 0 else np.nan,
            "Q2_global_joint": 1.0 - joint_sse / joint_global_base if joint_global_base > 0 else np.nan,
            "Q2_global_Rx": 1.0 - float(np.sum(rx_res ** 2)) / global_rx_base if global_rx_base > 0 else np.nan,
            "Q2_global_Rn": 1.0 - float(np.sum(rn_res ** 2)) / global_rn_base if global_rn_base > 0 else np.nan,
            "RMSE_Rx": float(np.sqrt(np.mean(rx_res ** 2))),
            "RMSE_Rn": float(np.sqrt(np.mean(rn_res ** 2))),
            "MAE_Rx": float(np.mean(np.abs(rx_res))),
            "MAE_Rn": float(np.mean(np.abs(rn_res))),
            "bias_Rx": float(np.mean(-rx_res)),
            "bias_Rn": float(np.mean(-rn_res)),
            "calibration_intercept_Rx": float(rx_cal.intercept) if rx_cal else np.nan,
            "calibration_slope_Rx": float(rx_cal.slope) if rx_cal else np.nan,
            "calibration_intercept_Rn": float(rn_cal.intercept) if rn_cal else np.nan,
            "calibration_slope_Rn": float(rn_cal.slope) if rn_cal else np.nan,
            "sign_agreement_Rx": float(np.mean(np.sign(pred[:, 0]) == np.sign(obs[:, 0]))),
            "sign_agreement_Rn": float(np.mean(np.sign(pred[:, 1]) == np.sign(obs[:, 1]))),
            "quadrant_agreement": float(np.mean(
                (np.sign(pred[:, 0]) == np.sign(obs[:, 0]))
                & (np.sign(pred[:, 1]) == np.sign(obs[:, 1]))
            )),
            "redistribution_direction_agreement": float(np.mean(
                np.sign(redistribution_pred) == np.sign(redistribution_obs)
            )),
            "angular_error_deg": float(np.nanmean(angles)) if np.isfinite(angles).any() else np.nan,
            "n_angle": int(np.isfinite(angles).sum()),
        }
        ok_t = np.isfinite(d["dAH_obs"]) & np.isfinite(d["dAH_pred"]) & np.isfinite(d["dAH_m0"])
        if ok_t.sum() >= 5:
            t_res = d.loc[ok_t, "dAH_obs"].to_numpy(float) - d.loc[ok_t, "dAH_pred"].to_numpy(float)
            t_base = d.loc[ok_t, "dAH_obs"].to_numpy(float) - d.loc[ok_t, "dAH_m0"].to_numpy(float)
            denom = float(np.sum(t_base ** 2))
            rec["Q2_dAH"] = 1.0 - float(np.sum(t_res ** 2)) / denom if denom > 0 else np.nan
            rec["RMSE_dAH"] = float(np.sqrt(np.mean(t_res ** 2)))
            rec["n_dAH"] = int(ok_t.sum())
        else:
            rec.update({"Q2_dAH": np.nan, "RMSE_dAH": np.nan, "n_dAH": int(ok_t.sum())})
        rows.append(rec)
    return pd.DataFrame(rows)


def repeated_spatial_cv(
    cohort: pd.DataFrame,
    predictor_suffix: str,
    outcome_suffix: str,
    n_folds: int,
    n_repeats: int,
    seed: int,
    angle_threshold: float,
    scenario: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cohort["spatial_block"].nunique() < n_folds:
        raise ValueError(f"Scenario {scenario}: fewer spatial blocks than folds.")
    rng = np.random.default_rng(seed)
    all_oof = []
    all_metrics = []
    for repeat in range(int(n_repeats)):
        mapping = balanced_group_folds(cohort["spatial_block"], n_folds, rng)
        fold_ids = cohort["spatial_block"].map(mapping).astype(int)
        fold_rows = []
        for fold in range(n_folds):
            test = cohort[fold_ids.eq(fold)].copy()
            train = cohort[~fold_ids.eq(fold)].copy()
            if len(test) == 0 or len(train) < 10:
                continue
            pred = fit_predict_fold(train, test, predictor_suffix, outcome_suffix)
            pred["repeat"] = repeat
            pred["fold"] = fold
            pred["scenario"] = scenario
            fold_rows.append(pred)
        if not fold_rows:
            continue
        repeat_oof = pd.concat(fold_rows, ignore_index=True)
        expected = len(cohort) * len(MODEL_ORDER)
        if len(repeat_oof) != expected:
            raise RuntimeError(
                f"Scenario {scenario}, repeat {repeat}: OOF rows {len(repeat_oof)} != {expected}."
            )
        all_oof.append(repeat_oof)
        all_metrics.append(score_oof_predictions(repeat_oof, angle_threshold, scenario, repeat))
    if not all_oof:
        raise RuntimeError(f"No OOF predictions generated for {scenario}.")
    return pd.concat(all_oof, ignore_index=True), pd.concat(all_metrics, ignore_index=True)


def average_oof_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "pair_id", "spatial_block", "continent", "group", "model"]
    values = ["Rx_obs", "Rn_obs", "Rx_pred", "Rn_pred", "dAH_obs", "dAH_pred"]
    return oof.groupby(keys, as_index=False, dropna=False)[values].mean()


def spatial_block_bootstrap(
    oof_mean: pd.DataFrame,
    n_boot: int,
    seed: int,
    angle_threshold: float,
    scenario: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = oof_mean[["spatial_block"]].drop_duplicates()["spatial_block"].tolist()
    if len(blocks) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    metric_rows = []
    for b in range(int(n_boot)):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        parts = []
        for occurrence, block in enumerate(sampled):
            part = oof_mean[oof_mean["spatial_block"].eq(block)].copy()
            part["pair_id"] = part["pair_id"].astype(str) + f"__boot{occurrence:04d}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        scored = score_oof_predictions(sample, angle_threshold, scenario, repeat=None)
        scored["bootstrap"] = b
        metric_rows.append(scored)
    boot = pd.concat(metric_rows, ignore_index=True)
    point = score_oof_predictions(oof_mean, angle_threshold, scenario, repeat=None)
    summary_rows = []
    for _, p in point.iterrows():
        d = boot[boot["model"].eq(p["model"])]
        for metric in (
            "Q2_joint", "sRMSE_joint", "Q2_Rx", "Q2_Rn",
            "Q2_global_joint", "Q2_global_Rx", "Q2_global_Rn",
            "RMSE_Rx", "RMSE_Rn", "MAE_Rx", "MAE_Rn", "bias_Rx", "bias_Rn",
            "calibration_intercept_Rx", "calibration_slope_Rx",
            "calibration_intercept_Rn", "calibration_slope_Rn",
            "sign_agreement_Rx", "sign_agreement_Rn", "quadrant_agreement",
            "redistribution_direction_agreement", "angular_error_deg", "Q2_dAH", "RMSE_dAH",
        ):
            vals = pd.to_numeric(d[metric], errors="coerce").dropna().to_numpy(float)
            summary_rows.append({
                "scenario": scenario,
                "model": p["model"],
                "metric": metric,
                "estimate": p[metric],
                "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "n_boot_valid": len(vals),
            })
    summary = pd.DataFrame(summary_rows)

    comparison_rows = []
    for comparator in ("M1", "M2", "M4"):
        for metric in (
            "Q2_joint", "Q2_Rx", "Q2_Rn", "Q2_dAH",
            "Q2_global_joint", "Q2_global_Rx", "Q2_global_Rn",
            "redistribution_direction_agreement",
        ):
            wide = boot.pivot(index="bootstrap", columns="model", values=metric)
            if "M3" not in wide or comparator not in wide:
                continue
            delta = (wide["M3"] - wide[comparator]).dropna().to_numpy(float)
            pwide = point.set_index("model")
            estimate = pwide.loc["M3", metric] - pwide.loc[comparator, metric]
            comparison_rows.append({
                "scenario": scenario,
                "comparison": f"M3_minus_{comparator}",
                "metric": metric,
                "estimate": estimate,
                "ci_low": float(np.quantile(delta, 0.025)) if len(delta) else np.nan,
                "ci_high": float(np.quantile(delta, 0.975)) if len(delta) else np.nan,
                "n_boot_valid": len(delta),
            })
    return boot, summary, pd.DataFrame(comparison_rows)


def leave_one_continent_out(
    cohort: pd.DataFrame,
    predictor_suffix: str,
    outcome_suffix: str,
    angle_threshold: float,
    scenario: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for continent in EXPECTED_CONTINENTS:
        test = cohort[cohort["continent"].eq(continent)].copy()
        train = cohort[~cohort["continent"].eq(continent)].copy()
        if len(test) < 3 or len(train) < 20:
            continue
        pred = fit_predict_fold(train, test, predictor_suffix, outcome_suffix)
        pred["held_out_continent"] = continent
        pred["scenario"] = scenario
        rows.append(pred)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    oof = pd.concat(rows, ignore_index=True)
    metrics = []
    metrics.append(score_oof_predictions(oof, angle_threshold, scenario, repeat=None).assign(held_out_continent="POOLED"))
    for continent, d in oof.groupby("held_out_continent"):
        metrics.append(score_oof_predictions(d, angle_threshold, scenario, repeat=None).assign(held_out_continent=continent))
    return oof, pd.concat(metrics, ignore_index=True)


# =============================================================================
# Functional PCA/SVD basis validation
# =============================================================================


def functional_pca_analysis(
    response: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = response[response["block"].eq("FULL")].copy()
    curve_cols = [f"R_h{h:02d}" for h in range(24)]
    full = full.dropna(subset=curve_cols + ["Rmean", "RAmp", "spatial_block"])
    if len(full) < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    matrix = full[curve_cols].to_numpy(float)
    mean_curve = matrix.mean(axis=0)
    centered = matrix - mean_curve
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s ** 2
    ratio = variance / variance.sum() if variance.sum() > 0 else np.full_like(variance, np.nan)
    constant_basis = np.ones(24, dtype=float)
    constant_basis /= np.linalg.norm(constant_basis)
    daynight_basis = np.array([1.0 if 8 <= h <= 19 else -1.0 for h in range(24)])
    daynight_basis /= np.linalg.norm(daynight_basis)
    summary_rows = []
    score_rows = []
    scores = u * s
    for k in range(min(10, len(s))):
        loading = vt[k]
        summary_rows.append({
            "component": k + 1,
            "variance_explained": ratio[k],
            "cumulative_variance": np.nansum(ratio[: k + 1]),
            "alignment_abs_mean_basis": abs(float(np.dot(loading, constant_basis))),
            "alignment_abs_daynight_basis": abs(float(np.dot(loading, daynight_basis))),
            "corr_score_Rmean": stats.pearsonr(scores[:, k], full["Rmean"])[0],
            "corr_score_RAmp": stats.pearsonr(scores[:, k], full["RAmp"])[0],
            "n": len(full),
        })
        for h in range(24):
            score_rows.append({
                "record_type": "loading",
                "component": k + 1,
                "pair_id": "",
                "local_hour": h,
                "value": loading[h],
            })
    for i, rec in full.reset_index(drop=True).iterrows():
        for k in range(min(3, scores.shape[1])):
            score_rows.append({
                "record_type": "score",
                "component": k + 1,
                "pair_id": rec["pair_id"],
                "local_hour": np.nan,
                "value": scores[i, k],
            })

    # Spatially held-out representational sufficiency. The held-out curve is
    # projected onto a basis learned in training blocks; this is a
    # reconstruction test, not a future forecast.
    rng = np.random.default_rng(seed)
    fold_map = balanced_group_folds(full["spatial_block"], n_folds, rng)
    fold_ids = full["spatial_block"].map(fold_map).astype(int)
    cv_rows = []
    for fold in range(n_folds):
        train = matrix[~fold_ids.eq(fold).to_numpy()]
        test = matrix[fold_ids.eq(fold).to_numpy()]
        test_ids = full.loc[fold_ids.eq(fold), "pair_id"].tolist()
        if len(test) == 0 or len(train) < 5:
            continue
        train_mean = train.mean(axis=0)
        _, _, train_vt = np.linalg.svd(train - train_mean, full_matrices=False)
        for rank in (1, 2, 3):
            basis = train_vt[:rank]
            test_centered = test - train_mean
            reconstruction = train_mean + (test_centered @ basis.T) @ basis
            rmse = np.sqrt(np.mean((test - reconstruction) ** 2, axis=1))
            for pair_id, value in zip(test_ids, rmse):
                cv_rows.append({
                    "pair_id": pair_id,
                    "fold": fold,
                    "rank": rank,
                    "curve_RMSE_degC": float(value),
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(score_rows), pd.DataFrame(cv_rows)


# =============================================================================
# Reporting, figure construction and audit outputs
# =============================================================================


def sample_retention_table(
    meta: pd.DataFrame,
    response: pd.DataFrame,
    cut_years: Sequence[int],
    event_thresholds: Sequence[int],
) -> pd.DataFrame:
    rows = []
    total = int(meta["pair_id"].nunique())
    for cut in sorted(set(map(int, cut_years))):
        e = response[response["block"].eq(f"cut{cut}_E")]
        l = response[response["block"].eq(f"cut{cut}_L")]
        for min_events in sorted(set(map(int, event_thresholds))):
            cohort = make_predictive_cohort(response, cut, min_events)
            rows.append({
                "cut_year": cut,
                "early_years": f"{EARLY_START}-{cut}",
                "late_years": f"{cut + 1}-{LATE_END}",
                "min_hw_events_each_block": min_events,
                "canonical_pairs": total,
                "pairs_with_early_response": int(e["pair_id"].nunique()),
                "pairs_with_late_response": int(l["pair_id"].nunique()),
                "final_predictive_cohort": int(len(cohort)),
                "n_UHI": int(cohort["group"].eq("UHI").sum()),
                "n_UCI": int(cohort["group"].eq("UCI").sum()),
                "n_spatial_blocks": int(cohort["spatial_block"].nunique()),
                "n_continents": int(cohort["continent"].isin(EXPECTED_CONTINENTS).groupby(cohort["continent"]).any().sum()),
            })
    return pd.DataFrame(rows)


def summarize_cv_repeats(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Q2_joint", "sRMSE_joint", "Q2_Rx", "Q2_Rn",
        "Q2_global_joint", "Q2_global_Rx", "Q2_global_Rn",
        "RMSE_Rx", "RMSE_Rn", "MAE_Rx", "MAE_Rn", "bias_Rx", "bias_Rn",
        "calibration_intercept_Rx", "calibration_slope_Rx",
        "calibration_intercept_Rn", "calibration_slope_Rn",
        "sign_agreement_Rx", "sign_agreement_Rn", "quadrant_agreement",
        "redistribution_direction_agreement", "angular_error_deg", "Q2_dAH", "RMSE_dAH",
    ]
    rows = []
    for (scenario, model), g in repeat_metrics.groupby(["scenario", "model"]):
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(float)
            rows.append({
                "scenario": scenario,
                "model": model,
                "metric": metric,
                "mean_across_repeats": float(np.mean(vals)) if len(vals) else np.nan,
                "sd_across_repeats": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "q025_across_repeats": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "q975_across_repeats": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "n_repeats": len(vals),
                "note": "Fold-repeat distribution is descriptive; spatial-block bootstrap provides inferential intervals.",
            })
    return pd.DataFrame(rows)


def residual_diagnostic(
    primary_cohort: pd.DataFrame,
    primary_oof_mean: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred = primary_oof_mean[primary_oof_mean["model"].isin(["M1", "M3"])].copy()
    pred = pred.merge(primary_cohort[["pair_id", "D_E"]], on="pair_id", how="left", validate="many_to_one")
    pred["night_residual"] = pred["Rn_obs"] - pred["Rn_pred"]
    rows = []
    for model, g in pred.groupby("model"):
        d = g[["D_E", "night_residual"]].dropna()
        if len(d) >= 3:
            fit = stats.linregress(d["D_E"], d["night_residual"])
            rows.append({
                "model": model,
                "n": len(d),
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r": fit.rvalue,
                "p": fit.pvalue,
                "slope_se": fit.stderr,
            })
    return pred, pd.DataFrame(rows)


def temporal_component_stability(cohort: pd.DataFrame) -> pd.DataFrame:
    """Describe which response components transfer across the early/late split."""
    rows = []
    for quantity in ("Rmean", "RAmp", "Rx", "Rn", "RSx", "RSn", "dAH"):
        cols = [f"{quantity}_E", f"{quantity}_L"]
        if not set(cols).issubset(cohort.columns):
            continue
        d = cohort[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 3:
            continue
        fit = stats.linregress(d.iloc[:, 0], d.iloc[:, 1])
        rows.append({
            "quantity": quantity,
            "n": len(d),
            "pearson_r": fit.rvalue,
            "p_value": fit.pvalue,
            "late_on_early_slope": fit.slope,
            "late_on_early_intercept": fit.intercept,
        })
    return pd.DataFrame(rows)


def prediction_residual_associations(
    cohort: pd.DataFrame,
    oof_mean: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether errors track event support or omitted late shape terms."""
    candidate_columns = [
        "RSx_L", "RSn_L", "shape_asymmetry_L", "n_hw_events_E", "n_hw_events_L",
        "n_hw_days_E", "n_hw_days_L", "D_E",
    ]
    available = [c for c in candidate_columns if c in cohort.columns]
    merged = oof_mean.merge(
        cohort[["pair_id"] + available], on="pair_id", how="left", validate="many_to_one"
    )
    merged["Rx_residual"] = merged["Rx_obs"] - merged["Rx_pred"]
    merged["Rn_residual"] = merged["Rn_obs"] - merged["Rn_pred"]
    rows = []
    for model, gm in merged.groupby("model"):
        for residual in ("Rx_residual", "Rn_residual"):
            for covariate in available:
                d = gm[[residual, covariate]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(d) < 3 or d[covariate].nunique() < 2:
                    continue
                fit = stats.linregress(d[covariate], d[residual])
                rows.append({
                    "model": model,
                    "residual": residual,
                    "covariate": covariate,
                    "n": len(d),
                    "slope": fit.slope,
                    "pearson_r": fit.rvalue,
                    "p_value": fit.pvalue,
                })
    return pd.DataFrame(rows)


def prediction_influence_diagnostics(
    cohort: pd.DataFrame,
    oof_all_repeats: pd.DataFrame,
    oof_mean: pd.DataFrame,
    angle_threshold: float,
    scenario: str,
) -> pd.DataFrame:
    """Quantify scored-OOF influence and persistent extrapolation without refitting.

    ``delta_Q2_when_removed`` is a deletion diagnostic on the already held-out
    predictions, not a leave-one-pair-out refit.  It identifies reporting and
    inspection priorities; it is never used to exclude a station pair.
    """
    base_scores = score_oof_predictions(oof_mean, angle_threshold, scenario).set_index("model")
    cohort_cols = [
        c for c in (
            "pair_id", "group", "continent", "spatial_block", "RSx_L", "RSn_L",
            "n_hw_events_E", "n_hw_events_L", "n_hw_days_E", "n_hw_days_L",
        ) if c in cohort.columns
    ]
    meta = cohort[cohort_cols].drop_duplicates("pair_id")
    rows = []
    for model in MODEL_ORDER:
        mean_m = oof_mean[oof_mean["model"].eq(model)].copy()
        if mean_m.empty:
            continue
        rx_sd = float(mean_m["Rx_obs"].std(ddof=1))
        rn_sd = float(mean_m["Rn_obs"].std(ddof=1))
        if not np.isfinite(rx_sd) or rx_sd <= 1e-12:
            rx_sd = 1.0
        if not np.isfinite(rn_sd) or rn_sd <= 1e-12:
            rn_sd = 1.0
        mean_m["Rx_residual"] = mean_m["Rx_obs"] - mean_m["Rx_pred"]
        mean_m["Rn_residual"] = mean_m["Rn_obs"] - mean_m["Rn_pred"]
        mean_m["standardised_SSE"] = (
            (mean_m["Rx_residual"] / rx_sd) ** 2 + (mean_m["Rn_residual"] / rn_sd) ** 2
        )
        total_sse = float(mean_m["standardised_SSE"].sum())
        for _, rec in mean_m.iterrows():
            pair_id = rec["pair_id"]
            without = oof_mean[~oof_mean["pair_id"].eq(pair_id)]
            deleted = score_oof_predictions(without, angle_threshold, scenario)
            dq = np.nan
            if not deleted.empty and model in set(deleted["model"]):
                q_without = float(deleted.set_index("model").loc[model, "Q2_joint"])
                dq = q_without - float(base_scores.loc[model, "Q2_joint"])
            rows.append({
                "pair_id": pair_id,
                "model": model,
                "Rx_obs": rec["Rx_obs"],
                "Rn_obs": rec["Rn_obs"],
                "Rx_pred": rec["Rx_pred"],
                "Rn_pred": rec["Rn_pred"],
                "Rx_residual": rec["Rx_residual"],
                "Rn_residual": rec["Rn_residual"],
                "standardised_SSE": rec["standardised_SSE"],
                "SSE_share": rec["standardised_SSE"] / total_sse if total_sse > 0 else np.nan,
                "delta_Q2_when_removed": dq,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.merge(meta, on="pair_id", how="left", validate="many_to_one")
    result = result.merge(
        pd.concat([
            oof_all_repeats[oof_all_repeats["model"].eq(model)]
            .groupby("pair_id").agg(
                Rx_pred_repeat_sd=("Rx_pred", "std"),
                Rn_pred_repeat_sd=("Rn_pred", "std"),
                min_Rx_pred=("Rx_pred", "min"),
                max_Rx_pred=("Rx_pred", "max"),
                min_Rn_pred=("Rn_pred", "min"),
                max_Rn_pred=("Rn_pred", "max"),
                n_repeat_predictions=("repeat", "nunique"),
                n_Rn_pred_below_minus3=("Rn_pred", lambda s: int((s < -3.0).sum())),
                n_Rn_pred_above_plus3=("Rn_pred", lambda s: int((s > 3.0).sum())),
            ).reset_index().assign(model=model)
            for model in MODEL_ORDER
        ], ignore_index=True),
        on=["pair_id", "model"], how="left", validate="one_to_one",
    )
    return result.sort_values(["model", "SSE_share"], ascending=[True, False]).reset_index(drop=True)


def apply_nature_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def _metric_lookup(summary: pd.DataFrame, scenario: str, model: str, metric: str):
    d = summary[
        summary["scenario"].eq(scenario)
        & summary["model"].eq(model)
        & summary["metric"].eq(metric)
    ]
    if d.empty:
        return np.nan, np.nan, np.nan
    r = d.iloc[0]
    return float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])


def plot_predictive_closure(
    primary_scenario: str,
    performance_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    primary_cohort: pd.DataFrame,
    primary_oof_mean: pd.DataFrame,
    residual_points: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    apply_nature_style()
    fig = plt.figure(figsize=(W_DOUBLE, 132.0 * MM))
    gs = fig.add_gridspec(2, 3, left=0.065, right=0.985, bottom=0.09, top=0.955, wspace=0.38, hspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    axa, axb, axc, axd, axe, axf = axes

    # a: analysis design
    axa.axis("off")
    axa.plot([0.08, 0.92], [0.72, 0.72], color="#444444", lw=1.0, transform=axa.transAxes)
    axa.scatter([0.14, 0.50, 0.86], [0.72] * 3, s=20, color=["#4d4d4d", "#7b3294", "#1f78b4"], transform=axa.transAxes, zorder=3)
    axa.text(0.14, 0.82, "2015", ha="center", transform=axa.transAxes)
    axa.text(0.50, 0.82, "2019/20", ha="center", transform=axa.transAxes)
    axa.text(0.86, 0.82, "2024", ha="center", transform=axa.transAxes)
    axa.text(0.31, 0.57, "Estimate early\nresponse state", ha="center", va="center", transform=axa.transAxes)
    axa.text(0.69, 0.57, "Predict late\n$R_x$ and $R_n$", ha="center", va="center", transform=axa.transAxes)
    axa.annotate("", xy=(0.80, 0.39), xytext=(0.20, 0.39), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.9))
    axa.text(0.50, 0.27, "Rank one  vs  rank two\n10° spatial-block CV", ha="center", va="center", transform=axa.transAxes)

    # b: joint performance
    y_positions = np.arange(len(MODEL_ORDER))[::-1]
    for y, model in zip(y_positions, MODEL_ORDER):
        est, lo, hi = _metric_lookup(performance_summary, primary_scenario, model, "Q2_joint")
        axb.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o", ms=4.2, color=MODEL_COLORS[model], capsize=2, lw=0.9)
    axb.axvline(0, color=COLOR_ZERO, lw=0.75, ls="--")
    axb.set_yticks(y_positions, [MODEL_LABELS[m] for m in MODEL_ORDER])
    axb.set_xlabel(r"Joint held-out $Q^2$")

    # c: outcome-specific skill
    x = np.arange(2)
    offsets = np.linspace(-0.24, 0.24, 4)
    for off, model in zip(offsets, ("M1", "M2", "M3", "M4")):
        vals, lows, highs = [], [], []
        for metric in ("Q2_Rx", "Q2_Rn"):
            est, lo, hi = _metric_lookup(performance_summary, primary_scenario, model, metric)
            vals.append(est); lows.append(est - lo); highs.append(hi - est)
        axc.errorbar(x + off, vals, yerr=[lows, highs], fmt="o", ms=3.8, lw=0.85, capsize=1.8, color=MODEL_COLORS[model], label=model)
    axc.axhline(0, color=COLOR_ZERO, lw=0.75, ls="--")
    axc.set_xticks(x, [r"Daytime $R_x$", r"Nighttime $R_n$"])
    axc.set_ylabel(r"Held-out $Q^2$")
    axc.legend(frameon=False, ncol=2, loc="best", handletextpad=0.3, columnspacing=0.8)

    # d: matched late-night predictions
    for model, marker in (("M1", "o"), ("M3", "^")):
        d = primary_oof_mean[primary_oof_mean["model"].eq(model)]
        axd.scatter(d["Rn_obs"], d["Rn_pred"], s=10, alpha=0.45, edgecolor="none", marker=marker, color=MODEL_COLORS[model], label=model)
    vals = primary_oof_mean[["Rn_obs", "Rn_pred"]].to_numpy(float)
    lim = [np.nanmin(vals), np.nanmax(vals)]
    pad = 0.05 * max(lim[1] - lim[0], 1e-6)
    axd.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], color="#444444", ls="--", lw=0.8)
    axd.set_xlim(lim[0] - pad, lim[1] + pad); axd.set_ylim(lim[0] - pad, lim[1] + pad)
    axd.set_xlabel(r"Observed late $R_n$ (°C)")
    axd.set_ylabel(r"Predicted late $R_n$ (°C)")
    axd.legend(frameon=False, loc="upper left")

    # e: scalar information-loss diagnostic
    for model in ("M1", "M3"):
        d = residual_points[residual_points["model"].eq(model)].dropna(subset=["D_E", "night_residual"])
        ax_e_color = MODEL_COLORS[model]
        ax_e_alpha = 0.22 if model == "M1" else 0.18
        axe.scatter(d["D_E"], d["night_residual"], s=8, alpha=ax_e_alpha, edgecolor="none", color=ax_e_color)
        if len(d) >= 3:
            fit = stats.linregress(d["D_E"], d["night_residual"])
            xx = np.linspace(d["D_E"].min(), d["D_E"].max(), 100)
            axe.plot(xx, fit.intercept + fit.slope * xx, color=ax_e_color, lw=1.2, label=model)
    axe.axhline(0, color=COLOR_ZERO, lw=0.75, ls="--")
    axe.set_xlabel(r"Early orthogonal coordinate, $D^E$ (°C)")
    axe.set_ylabel(r"Late-night residual (°C)")
    axe.legend(frameon=False)

    # f: robustness of decisive M3-M2 contrast
    forest = comparison_summary[
        comparison_summary["comparison"].eq("M3_minus_M2")
        & comparison_summary["metric"].eq("Q2_joint")
    ].copy()
    if not forest.empty:
        labels = forest["scenario"].str.replace("_", " ", regex=False).tolist()
        yy = np.arange(len(forest))[::-1]
        for y, (_, r) in zip(yy, forest.iterrows()):
            axf.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]], fmt="o", color=MODEL_COLORS["M3"], ms=3.8, capsize=1.8, lw=0.85)
        axf.set_yticks(yy, labels)
    axf.axvline(0, color=COLOR_ZERO, lw=0.75, ls="--")
    axf.set_xlabel(r"Vector gain, $\Delta Q^2$ (M3−M2)")

    for label, ax in zip("abcdef", axes):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="top")
    fig.savefig(out_dir / "Fig_scalar_to_vector_predictive_closure.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "Fig_scalar_to_vector_predictive_closure.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    path: Path,
    primary_scenario: str,
    cohort: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    residual_summary: pd.DataFrame,
    loco_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    influence: pd.DataFrame,
    event_audit: pd.DataFrame,
) -> str:
    def fmt_triplet(model, metric):
        est, lo, hi = _metric_lookup(summary, primary_scenario, model, metric)
        return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"

    decisive = comparisons[
        comparisons["scenario"].eq(primary_scenario)
        & comparisons["comparison"].eq("M3_minus_M2")
        & comparisons["metric"].eq("Q2_joint")
    ]
    if decisive.empty:
        status = "NOT_EVALUABLE"
        delta_text = "NA"
    else:
        d = decisive.iloc[0]
        delta_text = f"{d['estimate']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"
        if d["ci_low"] > 0:
            night = comparisons[
                comparisons["scenario"].eq(primary_scenario)
                & comparisons["comparison"].eq("M3_minus_M2")
                & comparisons["metric"].eq("Q2_Rn")
            ]
            status = "STRONG_VECTOR_CLOSURE" if (not night.empty and night.iloc[0]["ci_low"] > 0) else "MODERATE_VECTOR_CLOSURE"
        elif d["estimate"] > 0:
            status = "MODEST_UNCERTAIN_VECTOR_GAIN"
        else:
            status = "NO_VECTOR_ADVANTAGE"
    lines = [
        "# Scalar-to-vector predictive test report",
        "",
        f"Run time (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Primary scenario: `{primary_scenario}`",
        f"Cohort: n={len(cohort)}; UHI={int(cohort.group.eq('UHI').sum())}; UCI={int(cohort.group.eq('UCI').sum())}; spatial blocks={cohort.spatial_block.nunique()}",
        "",
        "## Primary held-out performance",
        "",
        f"- M1 conventional daytime scalar, joint Q2: {fmt_triplet('M1', 'Q2_joint')}",
        f"- M2 optimal linear rank-one state, joint Q2: {fmt_triplet('M2', 'Q2_joint')}",
        f"- M3 mean-amplitude vector, joint Q2: {fmt_triplet('M3', 'Q2_joint')}",
        f"- M4 complete day-night vector, joint Q2: {fmt_triplet('M4', 'Q2_joint')}",
        f"- Decisive M3-M2 joint Q2 difference: {delta_text}",
        "",
        "## Pre-specified interpretation",
        "",
        f"Decision status: **{status}**",
        "",
        "A positive point estimate for M3-M2 indicates additional held-out information in the two-coordinate state. If its interval includes zero, the correct conclusion is suggestive rather than decisive vector gain. It does not identify heat capacity, forcing, conductance or heat storage uniquely.",
        "",
        "## Q2 definition sensitivity",
        "",
        f"- M3 cross-fitted-baseline joint Q2: {fmt_triplet('M3', 'Q2_joint')}",
        f"- M3 global-mean-baseline joint Q2: {fmt_triplet('M3', 'Q2_global_joint')}",
        "- The cross-fitted M0 baseline is primary because every held-out prediction, including the null prediction, is derived without its test fold. The global-mean version is a conventional sensitivity check.",
        "",
        "## Direct scalar-information-loss diagnostic",
        "",
    ]
    if not residual_summary.empty:
        for _, r in residual_summary.iterrows():
            lines.append(f"- {r['model']}: residual-vs-D slope={r['slope']:.3f}, r={r['r']:.3f}, p={r['p']:.3g}, n={int(r['n'])}.")
    lines.extend(["", "## Continental transfer", ""])
    if loco_metrics.empty:
        lines.append("- Not run or insufficient continent-specific samples.")
    else:
        pooled = loco_metrics[loco_metrics["held_out_continent"].eq("POOLED")]
        for model in ("M1", "M2", "M3", "M4"):
            d = pooled[pooled["model"].eq(model)]
            if not d.empty:
                lines.append(f"- {model} pooled leave-one-continent-out joint Q2: {d.iloc[0]['Q2_joint']:.3f}.")
    lines.extend(["", "## Component stability and extrapolation audit", ""])
    if not stability.empty:
        for quantity in ("Rmean", "RAmp", "Rx", "Rn", "RSx", "RSn"):
            d = stability[stability["quantity"].eq(quantity)]
            if not d.empty:
                lines.append(f"- {quantity}: early-late r={d.iloc[0]['pearson_r']:.3f} (n={int(d.iloc[0]['n'])}).")
    extreme = influence[
        influence["model"].eq("M3")
        & ((influence["min_Rn_pred"] < -3.0) | (influence["max_Rn_pred"] > 3.0))
    ] if not influence.empty else pd.DataFrame()
    lines.append(f"- Persistent M3 nighttime extrapolation beyond +/-3 C occurred for {extreme['pair_id'].nunique() if not extreme.empty else 0} pair(s); no pair was removed on this basis.")
    changed = int(event_audit["count_changed"].sum()) if not event_audit.empty else 0
    affected = int(event_audit.loc[event_audit["count_changed"], "pair_id"].nunique()) if changed else 0
    lines.append(f"- Matching the frozen position-based event counter changed {changed} pair-block counts across {affected} unique pair(s) relative to the legacy calendar-gap audit counter.")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The analysis tests out-of-period and out-of-space transferability of the observed response state. Weak prediction can reflect limited event support or temporal non-stationarity and does not by itself prove that the physics is one-dimensional.",
    ])
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return status


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    outputs: Sequence[Path],
    decision: str,
) -> None:
    data = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "numpy_rng": "default_rng/PCG64",
        },
        "arguments": vars(args),
        "inputs": {
            key: {"path": str(paths[key]), "sha256": sha256_file(paths[key])}
            for key in ("metrics", "flags", "reference")
        },
        "frozen_contract": {
            "heatwave_flags_reused": True,
            "heatwave_redetected": False,
            "event_counter": "0-to-1 transitions in ordered available warm-season days; matches frozen analysis 01",
            "harmonics": 2,
            "fixed_annual_group_descriptive_only": True,
            "primary_temporal_blocks": ["2015-2019", "2020-2024"],
        },
        "decision_status": decision,
        "outputs": [str(p) for p in outputs if p.exists()],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_storyline_evidence_summary(
    primary_scenario: str,
    performance: pd.DataFrame,
    comparisons: pd.DataFrame,
    pca_summary: pd.DataFrame,
    pca_cv: pd.DataFrame,
    residual_summary: pd.DataFrame,
) -> pd.DataFrame:
    """One-row-per-claim bridge from outputs to the final manuscript logic."""
    rows = []
    if not pca_summary.empty:
        pc2 = pca_summary[pca_summary["component"].eq(2)]
        if not pc2.empty:
            rows.append({
                "claim_id": "C1_low_dimensional_response_space",
                "quantity": "variance_explained_by_first_two_functional_modes",
                "estimate": float(pc2.iloc[0]["cumulative_variance"]),
                "ci_low": np.nan, "ci_high": np.nan,
                "interpretation": "Basis validation; not itself a novelty claim.",
            })
    if not pca_cv.empty:
        rank2 = pca_cv[pca_cv["rank"].eq(2)]["curve_RMSE_degC"].dropna()
        if len(rank2):
            rows.append({
                "claim_id": "C1_low_dimensional_response_space",
                "quantity": "held_out_rank2_curve_RMSE_degC",
                "estimate": float(rank2.mean()),
                "ci_low": np.nan, "ci_high": np.nan,
                "interpretation": "Spatially held-out reconstruction error.",
            })
    for comparison, label in (("M3_minus_M1", "vector_minus_daytime_scalar_joint_Q2"),
                              ("M3_minus_M2", "vector_minus_best_rank1_joint_Q2")):
        d = comparisons[
            comparisons["scenario"].eq(primary_scenario)
            & comparisons["comparison"].eq(comparison)
            & comparisons["metric"].eq("Q2_joint")
        ]
        if not d.empty:
            r = d.iloc[0]
            rows.append({
                "claim_id": "C2_cross_period_scalar_insufficiency",
                "quantity": label,
                "estimate": float(r["estimate"]),
                "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
                "interpretation": "Claim is decisive only if the interval excludes zero.",
            })
    for metric in ("Q2_joint", "Q2_Rn", "angular_error_deg", "redistribution_direction_agreement"):
        d = performance[
            performance["scenario"].eq(primary_scenario)
            & performance["model"].eq("M3")
            & performance["metric"].eq(metric)
        ]
        if not d.empty:
            r = d.iloc[0]
            rows.append({
                "claim_id": "C2_cross_period_scalar_insufficiency",
                "quantity": f"M3_{metric}",
                "estimate": float(r["estimate"]),
                "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
                "interpretation": "Held-out vector performance or direction score.",
            })
    if not residual_summary.empty:
        for _, r in residual_summary.iterrows():
            rows.append({
                "claim_id": "C2_cross_period_scalar_insufficiency",
                "quantity": f"{r['model']}_night_residual_vs_orthogonal_coordinate_r",
                "estimate": float(r["r"]),
                "ci_low": np.nan, "ci_high": np.nan,
                "interpretation": "Direct diagnostic of information lost by a scalar daytime response.",
            })
    return pd.DataFrame(rows)


def run_self_test() -> None:
    rng = np.random.default_rng(12345)
    n = 120
    x = rng.normal(size=(n, 2))
    y = np.column_stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1])) + rng.normal(scale=0.25, size=(n, 2))
    cohort = pd.DataFrame({
        "pair_id": [f"P{i:03d}" for i in range(n)],
        "spatial_block": [f"B{i // 4:02d}" for i in range(n)],
        "continent": [EXPECTED_CONTINENTS[i % len(EXPECTED_CONTINENTS)] for i in range(n)],
        "group": ["UHI" if i % 2 == 0 else "UCI" for i in range(n)],
        "Rmean_E": x[:, 0], "RAmp_E": x[:, 1],
        "Rx_E": x[:, 0] + x[:, 1], "Rn_E": x[:, 0] - x[:, 1],
        "Rx_L": y[:, 0], "Rn_L": y[:, 1],
        "dAH_L": 0.4 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=0.3, size=n),
    })
    oof, metrics = repeated_spatial_cv(
        cohort, "E", "L", 5, 5, 123, DEFAULT_ANGLE_MIN_MAGNITUDE, "self_test"
    )
    mean_oof = average_oof_predictions(oof)
    _, summary, comparisons = spatial_block_bootstrap(
        mean_oof, 50, 124, DEFAULT_ANGLE_MIN_MAGNITUDE, "self_test"
    )
    point = summary[summary["metric"].eq("Q2_joint")].set_index("model")["estimate"]
    if not (point["M3"] > point["M2"] and point["M3"] > point["M1"]):
        raise AssertionError(f"Synthetic two-dimensional test failed: {point.to_dict()}")
    if oof.groupby(["repeat", "model"])["pair_id"].nunique().min() != n:
        raise AssertionError("OOF coverage failed in self-test.")
    if comparisons.empty or metrics.empty:
        raise AssertionError("Metric/bootstrap output missing in self-test.")
    # Frozen analysis 01 treats the ordered available-day sequence as
    # consecutive even when a calendar date is absent.
    event_test = pd.DataFrame({
        "date": pd.to_datetime(["2019-06-01", "2019-06-02", "2019-06-04", "2019-06-05"]),
        "warm_season_year": [2019] * 4,
        "is_warm_season": [1] * 4,
        "hw_flag_percentile_warm_season": [1, 1, 1, 0],
    })
    if count_hw_events(event_test) != 1 or count_hw_events_calendar_gap(event_test) != 2:
        raise AssertionError("Frozen-position event counting self-test failed.")
    if "Q2_global_joint" not in metrics.columns:
        raise AssertionError("Global-mean Q2 sensitivity missing in self-test.")
    print("SELF_TEST_PASS")


# =============================================================================
# Main orchestration
# =============================================================================


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    validate_final_analysis_args(args)
    paths = resolve_inputs(args)
    validate_paths(paths)
    if args.reuse_period_metrics and not args.resume_existing_output:
        raise ValueError("--reuse-period-metrics requires --resume-existing-output.")
    protect_output_directory(paths["output"], args.resume_existing_output)
    ensure_dir(paths["output"])
    tables_dir = paths["output"] / "tables"
    figures_dir = paths["output"] / "figures"
    plot_data_dir = paths["output"] / "plot_data"
    ensure_dir(tables_dir); ensure_dir(figures_dir); ensure_dir(plot_data_dir)

    print("=" * 80)
    print("Scalar-to-vector cross-period predictive test")
    print(f"Frozen metrics: {paths['metrics']}")
    print(f"Frozen HW flags: {paths['flags']}")
    print(f"Frozen analysis-01 interface: {paths['reference']}")
    print(f"Independent output: {paths['output']}")
    print("=" * 80)

    # Interface preflight in the parent process.
    ref = load_reference_module(paths["reference"])
    print(f"Reference harmonics: {ref.N_HARMONICS}")
    meta = read_canonical_metadata(paths["metrics"], args.block_size)
    flags = read_daily_flags(paths["flags"], meta["pair_id"])
    print(f"Canonical fixed-group pairs: {len(meta)}")
    print(f"Daily flag rows: {len(flags)}")

    cut_years = [args.cut_year]
    if not args.skip_alternative_cuts:
        cut_years.extend(args.alternative_cut_years)
    cut_years = sorted(set(y for y in cut_years if EARLY_START <= y < LATE_END))

    event_audit = event_count_method_audit(flags, cut_years)
    event_audit_path = tables_dir / "event_count_method_audit.csv"
    event_audit.to_csv(event_audit_path, index=False)

    period_path = tables_dir / "period_specific_metrics_long.csv"
    errors_path = tables_dir / "period_metric_errors.csv"
    if args.reuse_period_metrics and period_path.exists():
        period_long = pd.read_csv(period_path, low_memory=False)
        errors = pd.read_csv(errors_path) if errors_path.exists() else pd.DataFrame()
        print(f"Reused period metrics: {period_path}")
    else:
        period_long, errors = compute_period_metrics(
            meta, flags, cut_years, paths, args.n_workers
        )
        period_long.to_csv(period_path, index=False)
        errors.to_csv(errors_path, index=False)

    response = build_response_table(period_long, meta)
    response_path = tables_dir / "block_specific_response_states.csv"
    response.to_csv(response_path, index=False)

    algebra_audit = algebraic_identity_audit(response)
    frozen_audit = frozen_full_period_interface_audit(paths["metrics"], response)
    interface_audit = pd.concat([algebra_audit, frozen_audit], ignore_index=True)
    interface_audit.to_csv(tables_dir / "frozen_interface_numeric_audit.csv", index=False)
    failures = interface_audit[interface_audit["status"].eq("FAIL")]
    if not failures.empty:
        raise RuntimeError(
            "Frozen-interface or algebraic identity audit failed: "
            + ", ".join(failures["check"].astype(str))
        )

    retention = sample_retention_table(
        meta, response, cut_years, [args.min_events, args.reliability_min_events]
    )
    retention_path = tables_dir / "sample_retention_flow.csv"
    retention.to_csv(retention_path, index=False)

    scenario_specs: List[Tuple[str, pd.DataFrame, str, str]] = []
    primary_cohort = make_predictive_cohort(response, args.cut_year, args.min_events)
    primary_scenario = f"cut{args.cut_year}_EtoL_min{args.min_events}"
    scenario_specs.append((primary_scenario, primary_cohort, "E", "L"))

    if args.reliability_min_events != args.min_events:
        reliable = make_predictive_cohort(response, args.cut_year, args.reliability_min_events)
        if len(reliable) >= max(30, args.n_folds * 5):
            scenario_specs.append((
                f"cut{args.cut_year}_EtoL_min{args.reliability_min_events}",
                reliable, "E", "L",
            ))

    if not args.skip_reverse:
        scenario_specs.append((
            f"cut{args.cut_year}_LtoE_min{args.min_events}",
            primary_cohort, "L", "E",
        ))

    if not args.skip_alternative_cuts:
        for cut in cut_years:
            if cut == args.cut_year:
                continue
            cohort = make_predictive_cohort(response, cut, args.min_events)
            if len(cohort) >= max(30, args.n_folds * 5):
                scenario_specs.append((f"cut{cut}_EtoL_min{args.min_events}", cohort, "E", "L"))

    all_oof = []
    all_repeat_metrics = []
    all_oof_mean = []
    all_boot = []
    all_perf_summary = []
    all_comparisons = []
    cohort_rows = []
    scenario_cohorts: Dict[str, pd.DataFrame] = {}

    for sidx, (scenario, cohort, predictor_suffix, outcome_suffix) in enumerate(scenario_specs):
        print(f"\n[{scenario}] n={len(cohort)}, blocks={cohort.spatial_block.nunique()}")
        if len(cohort) < max(30, args.n_folds * 5):
            print("  Skipped: insufficient cohort.")
            continue
        scenario_cohorts[scenario] = cohort
        csave = cohort.copy(); csave["scenario"] = scenario
        cohort_rows.append(csave)
        oof, repeat_metrics = repeated_spatial_cv(
            cohort,
            predictor_suffix,
            outcome_suffix,
            args.n_folds,
            args.n_cv_repeats,
            args.seed + sidx * 1000,
            args.angle_min_magnitude,
            scenario,
        )
        oof_mean = average_oof_predictions(oof)
        boot, perf_summary, comparisons = spatial_block_bootstrap(
            oof_mean,
            args.n_bootstrap,
            args.seed + 50000 + sidx * 1000,
            args.angle_min_magnitude,
            scenario,
        )
        all_oof.append(oof); all_repeat_metrics.append(repeat_metrics); all_oof_mean.append(oof_mean)
        all_boot.append(boot); all_perf_summary.append(perf_summary); all_comparisons.append(comparisons)

    if not all_oof:
        raise RuntimeError("No predictive scenario completed.")

    oof_all = pd.concat(all_oof, ignore_index=True)
    repeat_all = pd.concat(all_repeat_metrics, ignore_index=True)
    oof_mean_all = pd.concat(all_oof_mean, ignore_index=True)
    boot_all = pd.concat(all_boot, ignore_index=True)
    performance_summary = pd.concat(all_perf_summary, ignore_index=True)
    comparison_summary = pd.concat(all_comparisons, ignore_index=True)
    cohorts_all = pd.concat(cohort_rows, ignore_index=True)
    repeat_summary = summarize_cv_repeats(repeat_all)

    cohorts_all.to_csv(tables_dir / "predictive_cohorts.csv", index=False)
    repeat_all.to_csv(tables_dir / "cv_repeat_metrics.csv", index=False)
    repeat_summary.to_csv(tables_dir / "cv_repeat_metrics_summary.csv", index=False)
    oof_all.to_csv(tables_dir / "cv_oof_predictions_all_repeats.csv", index=False)
    oof_mean_all.to_csv(tables_dir / "cv_oof_predictions_mean.csv", index=False)
    boot_all.to_csv(tables_dir / "spatial_block_bootstrap_metrics.csv", index=False)
    performance_summary.to_csv(tables_dir / "model_performance_summary.csv", index=False)
    comparison_summary.to_csv(tables_dir / "model_comparison_summary.csv", index=False)
    performance_summary[
        performance_summary["metric"].isin([
            "Q2_joint", "Q2_Rx", "Q2_Rn", "Q2_global_joint", "Q2_global_Rx", "Q2_global_Rn"
        ])
    ].to_csv(tables_dir / "q2_definition_sensitivity.csv", index=False)
    oof_all[["scenario", "repeat", "fold", "pair_id", "spatial_block"]].drop_duplicates().to_csv(
        tables_dir / "spatial_cv_fold_assignments.csv", index=False
    )

    primary_oof_mean = oof_mean_all[oof_mean_all["scenario"].eq(primary_scenario)].copy()
    residual_points, residual_summary = residual_diagnostic(primary_cohort, primary_oof_mean)
    residual_points.to_csv(tables_dir / "scalar_information_loss_residuals.csv", index=False)
    residual_summary.to_csv(tables_dir / "scalar_information_loss_summary.csv", index=False)

    stability = temporal_component_stability(primary_cohort)
    stability.to_csv(tables_dir / "early_late_component_stability.csv", index=False)
    residual_associations = prediction_residual_associations(primary_cohort, primary_oof_mean)
    residual_associations.to_csv(tables_dir / "prediction_residual_associations.csv", index=False)
    primary_oof_all = oof_all[oof_all["scenario"].eq(primary_scenario)].copy()
    influence = prediction_influence_diagnostics(
        primary_cohort, primary_oof_all, primary_oof_mean,
        args.angle_min_magnitude, primary_scenario,
    )
    influence.to_csv(tables_dir / "prediction_influence_diagnostics.csv", index=False)
    influence.groupby("model", group_keys=False).head(10).to_csv(
        tables_dir / "top10_prediction_influence_by_model.csv", index=False
    )

    loco_oof = pd.DataFrame(); loco_metrics = pd.DataFrame()
    if not args.skip_loco:
        loco_oof, loco_metrics = leave_one_continent_out(
            primary_cohort, "E", "L", args.angle_min_magnitude, "leave_one_continent_out"
        )
        loco_oof.to_csv(tables_dir / "leave_one_continent_out_predictions.csv", index=False)
        loco_metrics.to_csv(tables_dir / "leave_one_continent_out_metrics.csv", index=False)
        if not loco_metrics.empty:
            loco_wide = loco_metrics.pivot_table(
                index="held_out_continent", columns="model",
                values=["Q2_joint", "Q2_Rx", "Q2_Rn", "Q2_dAH"], aggfunc="first"
            )
            loco_comparisons = []
            for continent in loco_wide.index:
                for comparator in ("M1", "M2", "M4"):
                    for metric in ("Q2_joint", "Q2_Rx", "Q2_Rn", "Q2_dAH"):
                        try:
                            value = loco_wide.loc[continent, (metric, "M3")] - loco_wide.loc[continent, (metric, comparator)]
                        except KeyError:
                            value = np.nan
                        loco_comparisons.append({
                            "held_out_continent": continent,
                            "comparison": f"M3_minus_{comparator}",
                            "metric": metric,
                            "difference": value,
                        })
            pd.DataFrame(loco_comparisons).to_csv(
                tables_dir / "leave_one_continent_out_model_comparisons.csv", index=False
            )

    pca_summary = pd.DataFrame(); pca_details = pd.DataFrame(); pca_cv = pd.DataFrame()
    if not args.skip_pca:
        pca_summary, pca_details, pca_cv = functional_pca_analysis(
            response, args.n_folds, args.seed + 90000
        )
        pca_summary.to_csv(tables_dir / "functional_pca_summary.csv", index=False)
        pca_details.to_csv(tables_dir / "functional_pca_loadings_and_scores.csv", index=False)
        pca_cv.to_csv(tables_dir / "functional_pca_spatial_reconstruction.csv", index=False)

    storyline_summary = build_storyline_evidence_summary(
        primary_scenario, performance_summary, comparison_summary,
        pca_summary, pca_cv, residual_summary,
    )
    storyline_summary.to_csv(tables_dir / "storyline_claims_1_2_evidence_summary.csv", index=False)

    # Explicit panel-source tables.
    performance_summary[performance_summary["scenario"].eq(primary_scenario)].to_csv(
        plot_data_dir / "panel_bc_performance.csv", index=False
    )
    primary_oof_mean.to_csv(plot_data_dir / "panel_d_oof_predictions.csv", index=False)
    residual_points.to_csv(plot_data_dir / "panel_e_scalar_residuals.csv", index=False)
    comparison_summary[
        comparison_summary["comparison"].eq("M3_minus_M2")
        & comparison_summary["metric"].eq("Q2_joint")
    ].to_csv(plot_data_dir / "panel_f_robustness.csv", index=False)

    plot_predictive_closure(
        primary_scenario,
        performance_summary,
        comparison_summary,
        primary_cohort,
        primary_oof_mean,
        residual_points,
        figures_dir,
        args.dpi,
    )

    report_path = paths["output"] / "scalar_to_vector_predictive_report.md"
    decision = write_markdown_report(
        report_path,
        primary_scenario,
        primary_cohort,
        performance_summary,
        comparison_summary,
        residual_summary,
        loco_metrics,
        stability,
        influence,
        event_audit,
    )

    outputs = [p for p in paths["output"].rglob("*") if p.is_file()]
    manifest_path = paths["output"] / "run_manifest.json"
    write_manifest(manifest_path, args, paths, outputs, decision)

    print("\n" + "=" * 80)
    print(f"Completed. Decision status: {decision}")
    print(f"Report: {report_path}")
    print(f"Figure: {figures_dir / 'Fig_scalar_to_vector_predictive_closure.png'}")
    print(f"All outputs: {paths['output']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
