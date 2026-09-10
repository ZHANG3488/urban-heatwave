#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze figure3 timing.

Original scientific calculations retained; see README.md for inputs and execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "figure3_state_process_timing_analysis"
DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20260813
DEFAULT_BLOCK_DEG = 10.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs/analysis/main_multiyear/robustness_percentile/all_pair_period_metrics.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs/analysis/figure3_timing"
PRIMARY_TIMING_METRIC = "h1_forward_contrast_background_change_h"
PRIMARY_CONTROLS = ("annual_group", "continent")
PERIOD_NHW = "non_heatwave"
PERIOD_HW = "heatwave"
PERIOD_ANNUAL = "annual"
PERIOD_ORDER = (PERIOD_ANNUAL, PERIOD_NHW, PERIOD_HW)
GROUP_ORDER = ("UHI", "UCI")
HOURS = np.arange(24, dtype=float)
TWO_PI = 2.0 * np.pi
EPS = 1.0e-12


CORE_REQUIRED = {
    "pair_id",
    "period",
    "dTmean",
    "dAmp1",
    "dTx",
    "dTn",
    "urban_Amp1",
    "urban_Phase1",
    "rural_Amp1",
    "rural_Phase1",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [json_ready(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def write_json(path: Path, obj: Mapping) -> None:
    path.write_text(
        json.dumps(json_ready(dict(obj)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_period(value: object) -> str:
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "nhw": PERIOD_NHW,
        "nonheatwave": PERIOD_NHW,
        "non_heat_wave": PERIOD_NHW,
        "hw": PERIOD_HW,
        "heat_wave": PERIOD_HW,
        "year": PERIOD_ANNUAL,
        "yearly": PERIOD_ANNUAL,
    }
    return aliases.get(s, s)


def first_existing(columns: Iterable[str], aliases: Sequence[str], label: str) -> str:
    present = [c for c in aliases if c in columns]
    if not present:
        raise ValueError(f"Missing {label}; expected one of: {list(aliases)}")
    return present[0]


def circular_lon_midpoint(lon1: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    a = np.deg2rad(lon1.astype(float))
    b = np.deg2rad(lon2.astype(float))
    z = np.exp(1j * a) + np.exp(1j * b)
    out = np.rad2deg(np.angle(z))
    ambiguous = np.abs(z) < EPS
    out[ambiguous] = ((lon1[ambiguous] + lon2[ambiguous] + 180.0) % 360.0) - 180.0
    return out


def wrap_signed_hours(value: np.ndarray | float, period: float = 24.0):
    x = np.asarray(value, dtype=float)
    out = ((x + period / 2.0) % period) - period / 2.0
    if np.ndim(value) == 0:
        return float(out)
    return out


def circular_difference_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def curve_columns(prefix: str) -> List[str]:
    return [f"{prefix}_diurnal_h{h:02d}" for h in range(24)]


def dft_coefficients(curve: Sequence[float]) -> np.ndarray:
    arr = np.asarray(curve, dtype=float)
    if arr.shape != (24,) or not np.isfinite(arr).all():
        raise ValueError("A complete finite 24-hour curve is required.")
    return np.fft.fft(arr) / 24.0


def harmonic_peak_time_from_coefficient(c: complex, harmonic: int = 1) -> float:
    if not (np.isfinite(c.real) and np.isfinite(c.imag)) or abs(c) <= EPS:
        return np.nan
    omega = TWO_PI * harmonic / 24.0
    return float((-np.angle(c) / omega) % (24.0 / harmonic))


def two_harmonic_peak_time(curve: Sequence[float], points: int = 2400) -> float:
    coeff = dft_coefficients(curve)
    t = np.linspace(0.0, 24.0, points, endpoint=False)
    recon = np.full(points, float(np.real(coeff[0])), dtype=float)
    for k in (1, 2):
        recon += 2.0 * np.real(coeff[k] * np.exp(1j * TWO_PI * k * t / 24.0))
    if not np.isfinite(recon).any():
        return np.nan
    return float(t[int(np.nanargmax(recon))])


def signed_loop_area(x: Sequence[float], y: Sequence[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[valid]
    yy = yy[valid]
    if len(xx) < 4:
        return np.nan
    return float(
        0.5
        * (
            np.dot(xx, np.roll(yy, -1))
            - np.dot(yy, np.roll(xx, -1))
        )
    )


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new directory to preserve prior results."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_and_validate_input(path: Path, method: str) -> Tuple[pd.DataFrame, Dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, low_memory=False)
    missing = sorted(CORE_REQUIRED.difference(raw.columns))
    missing += [c for c in curve_columns("urban") + curve_columns("rural") if c not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(set(missing))}")

    df = raw.copy()
    df["pair_id"] = df["pair_id"].astype(str).str.strip()
    df["period"] = df["period"].map(normalize_period)

    method_counts: Dict[str, int] = {}
    if "hw_method" in df.columns:
        method_norm = df["hw_method"].astype(str).str.strip().str.lower()
        method_counts = method_norm.value_counts(dropna=False).to_dict()
        df = df[method_norm.eq(method.lower())].copy()
        if df.empty:
            raise ValueError(f"No rows remain after selecting hw_method={method!r}.")

    df = df[df["period"].isin(PERIOD_ORDER)].copy()
    if df.empty:
        raise ValueError("No annual/NHW/HW rows remain after period normalization.")

    duplicated = df.duplicated(["pair_id", "period"], keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, ["pair_id", "period"]].head(20).to_dict("records")
        raise ValueError(f"Duplicate pair-period rows detected; examples: {examples}")

    counts = df["period"].value_counts().to_dict()
    absent = [p for p in PERIOD_ORDER if counts.get(p, 0) == 0]
    if absent:
        raise ValueError(f"Required periods absent: {absent}")

    qc_versions: List[str] = []
    if "temperature_diurnal_qc_version" in df.columns:
        qc_versions = sorted(
            df["temperature_diurnal_qc_version"].dropna().astype(str).str.strip().unique().tolist()
        )
        if len(qc_versions) > 1:
            raise ValueError(f"Mixed temperature QC versions in input: {qc_versions}")

    qc_pass_field = None
    for candidate in (
        "temperature_diurnal_qc_pass_pair",
        "raw_day_night_support_qc_pass_pair",
        "raw_day_night_support_pass_pair",
    ):
        if candidate in df.columns:
            qc_pass_field = candidate
            passed = pd.to_numeric(df[candidate], errors="coerce")
            bad = passed.ne(1) | passed.isna()
            if bad.any():
                examples = df.loc[bad, ["pair_id", "period", candidate]].head(20).to_dict("records")
                raise ValueError(
                    f"Selected input contains rows that do not pass {candidate}; examples: {examples}"
                )
            break

    numeric_cols = [
        "dTmean", "dAmp1", "dTx", "dTn", "urban_Amp1", "urban_Phase1",
        "rural_Amp1", "rural_Phase1",
    ] + curve_columns("urban") + curve_columns("rural")
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    info = {
        "input_rows_raw": int(len(raw)),
        "input_rows_selected": int(len(df)),
        "method_counts_before_selection": method_counts,
        "period_counts_selected": counts,
        "qc_versions": qc_versions,
        "qc_pass_field": qc_pass_field,
        "columns": list(df.columns),
    }
    return df, info


def build_matched_state_cohort(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attrition: List[Dict] = []
    annual = df[df["period"].eq(PERIOD_ANNUAL)].copy()
    annual["annual_group"] = np.where(annual["dTx"] >= 0.0, "UHI", "UCI")
    invalid_annual = annual[~np.isfinite(annual["dTx"])]["pair_id"].tolist()
    for pair_id in invalid_annual:
        attrition.append({"pair_id": pair_id, "stage": "annual_group", "reason": "nonfinite_annual_dTx"})
    annual = annual[np.isfinite(annual["dTx"])].copy()

    nhw_ids = set(df.loc[df["period"].eq(PERIOD_NHW), "pair_id"])
    hw_ids = set(df.loc[df["period"].eq(PERIOD_HW), "pair_id"])
    annual_ids = set(annual["pair_id"])
    all_ids = set(df["pair_id"])
    matched_ids = annual_ids & nhw_ids & hw_ids

    for pair_id in sorted(all_ids - matched_ids):
        missing_periods = []
        if pair_id not in annual_ids:
            missing_periods.append("annual")
        if pair_id not in nhw_ids:
            missing_periods.append("non_heatwave")
        if pair_id not in hw_ids:
            missing_periods.append("heatwave")
        attrition.append(
            {
                "pair_id": pair_id,
                "stage": "matched_periods",
                "reason": "missing_" + "+".join(missing_periods),
            }
        )

    selected = df[df["pair_id"].isin(matched_ids)].copy()
    group_map = annual.set_index("pair_id")["annual_group"]
    selected["annual_group"] = selected["pair_id"].map(group_map)

    nhw = selected[selected["period"].eq(PERIOD_NHW)].set_index("pair_id", drop=False)
    hw = selected[selected["period"].eq(PERIOD_HW)].set_index("pair_id", drop=False)
    ann = selected[selected["period"].eq(PERIOD_ANNUAL)].set_index("pair_id", drop=False)

    lon_u_col = first_existing(df.columns, ("lon_urban", "urban_lon"), "urban longitude")
    lat_u_col = first_existing(df.columns, ("lat_urban", "urban_lat"), "urban latitude")
    lon_r_col = first_existing(df.columns, ("lon_rural", "rural_lon"), "rural longitude")
    lat_r_col = first_existing(df.columns, ("lat_rural", "rural_lat"), "rural latitude")

    records: List[Dict] = []
    for pair_id in sorted(matched_ids):
        a = ann.loc[pair_id]
        n = nhw.loc[pair_id]
        h = hw.loc[pair_id]
        required_state = np.array([n["dTmean"], n["dAmp1"], h["dTmean"], h["dAmp1"]], dtype=float)
        if not np.isfinite(required_state).all():
            attrition.append(
                {"pair_id": pair_id, "stage": "state_complete_case", "reason": "nonfinite_state_coordinate"}
            )
            continue

        lon_u = float(a[lon_u_col])
        lat_u = float(a[lat_u_col])
        lon_r = float(a[lon_r_col])
        lat_r = float(a[lat_r_col])
        coords = np.array([lon_u, lat_u, lon_r, lat_r], dtype=float)
        if not np.isfinite(coords).all():
            attrition.append(
                {"pair_id": pair_id, "stage": "spatial_metadata", "reason": "nonfinite_pair_coordinates"}
            )
            continue

        r_mean = float(h["dTmean"] - n["dTmean"])
        r_amp = float(h["dAmp1"] - n["dAmp1"])
        r_day = float(h["dTx"] - n["dTx"])
        r_night = float(h["dTn"] - n["dTn"])
        rho = float(np.hypot(r_mean, r_amp))
        theta = float(np.degrees(np.arctan2(r_amp, r_mean)))

        continent = str(a.get("continent", n.get("continent", "Unknown"))).strip()
        if continent.lower() in {"", "nan", "none"}:
            continent = "Unknown"

        records.append(
            {
                "pair_id": pair_id,
                "annual_group": str(a["annual_group"]),
                "continent": continent,
                "lon_urban": lon_u,
                "lat_urban": lat_u,
                "lon_rural": lon_r,
                "lat_rural": lat_r,
                "delta_Tmean_NHW": float(n["dTmean"]),
                "delta_Amp_NHW": float(n["dAmp1"]),
                "delta_Tmean_HW": float(h["dTmean"]),
                "delta_Amp_HW": float(h["dAmp1"]),
                "delta_Tx_NHW": float(n["dTx"]),
                "delta_Tx_HW": float(h["dTx"]),
                "delta_Tn_NHW": float(n["dTn"]),
                "delta_Tn_HW": float(h["dTn"]),
                "R_mean": r_mean,
                "R_Amp": r_amp,
                "R_day_observed": r_day,
                "R_night_observed": r_night,
                "rho": rho,
                "theta_deg": theta,
                "D_orthogonal": r_mean - r_amp,
            }
        )

    state = pd.DataFrame(records)
    if state.empty:
        raise ValueError("Matched state cohort is empty after complete-case and coordinate checks.")
    return state, selected, pd.DataFrame(attrition, columns=["pair_id", "stage", "reason"])


def add_response_decomposition(
    state: pd.DataFrame, attrition: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add the exact observed-coordinate decomposition used only in Fig. 3c.

    The low-order identities are not assumed to be exact.  Their mismatch is
    retained as the shape/phase residual.  Day and night use one shared
    complete-case cohort within each UHI/UCI regime.
    """
    out = state.copy()
    valid = np.isfinite(
        out[["R_mean", "R_Amp", "R_day_observed", "R_night_observed"]]
    ).all(axis=1)
    out["decomposition_valid"] = valid.astype(int)

    out["day_mean_component"] = out["R_mean"]
    out["day_amplitude_component"] = out["R_Amp"]
    out["day_shape_phase_residual"] = (
        out["R_day_observed"] - out["R_mean"] - out["R_Amp"]
    )
    out["night_mean_component"] = out["R_mean"]
    out["night_amplitude_component"] = -out["R_Amp"]
    out["night_shape_phase_residual"] = (
        out["R_night_observed"] - out["R_mean"] + out["R_Amp"]
    )

    invalid_ids = out.loc[~valid, "pair_id"].astype(str).tolist()
    if invalid_ids:
        extra = pd.DataFrame(
            {
                "pair_id": invalid_ids,
                "stage": "day_night_decomposition",
                "reason": "nonfinite_day_or_night_response",
            }
        )
        attrition = pd.concat([attrition, extra], ignore_index=True)

    complete = out.loc[valid]
    if len(complete) < 20:
        raise ValueError(
            f"Only {len(complete)} pairs have complete day/night responses; "
            "panel c requires at least 20."
        )
    for group in GROUP_ORDER:
        if not complete["annual_group"].eq(group).any():
            raise ValueError(f"Panel c has no complete pairs in {group}.")

    day_error = np.abs(
        complete["R_day_observed"]
        - complete["day_mean_component"]
        - complete["day_amplitude_component"]
        - complete["day_shape_phase_residual"]
    )
    night_error = np.abs(
        complete["R_night_observed"]
        - complete["night_mean_component"]
        - complete["night_amplitude_component"]
        - complete["night_shape_phase_residual"]
    )
    if max(float(day_error.max()), float(night_error.max())) > 1.0e-10:
        raise RuntimeError("Day/night decomposition identity failed numerical validation.")
    return out, attrition


def assign_spatial_blocks(state: pd.DataFrame, block_deg: float) -> pd.DataFrame:
    if not np.isfinite(block_deg) or block_deg <= 0 or 180 % block_deg != 0:
        raise ValueError("block_deg must be a positive divisor of 180.")
    out = state.copy()
    lon_mid = circular_lon_midpoint(out["lon_urban"].to_numpy(), out["lon_rural"].to_numpy())
    lat_mid = 0.5 * (out["lat_urban"].to_numpy(float) + out["lat_rural"].to_numpy(float))
    lon_mid = ((lon_mid + 180.0) % 360.0) - 180.0
    lat_mid = np.clip(lat_mid, -90.0, np.nextafter(90.0, -np.inf))
    lon_idx = np.floor((lon_mid + 180.0) / block_deg).astype(int)
    lat_idx = np.floor((lat_mid + 90.0) / block_deg).astype(int)
    out["pair_lon_mid"] = lon_mid
    out["pair_lat_mid"] = lat_mid
    out["block_id"] = [f"lat{la:02d}_lon{lo:02d}" for la, lo in zip(lat_idx, lon_idx)]
    return out


def validate_fft_consistency(selected: pd.DataFrame, state_ids: set[str]) -> Dict:
    rows = selected[
        selected["pair_id"].isin(state_ids) & selected["period"].isin([PERIOD_NHW, PERIOD_HW])
    ]
    amp_errors: List[float] = []
    phase_errors: List[float] = []
    damp_errors: List[float] = []
    for _, row in rows.iterrows():
        for prefix in ("urban", "rural"):
            curve = row[curve_columns(prefix)].to_numpy(dtype=float)
            if not np.isfinite(curve).all():
                continue
            c1 = dft_coefficients(curve)[1]
            amp = 2.0 * abs(c1)
            phase = float(np.angle(c1))
            amp_errors.append(abs(amp - float(row[f"{prefix}_Amp1"])))
            phase_errors.append(
                abs(float(np.deg2rad(circular_difference_degrees(
                    np.array([np.degrees(phase)]),
                    np.array([np.degrees(float(row[f"{prefix}_Phase1"]))]),
                )[0])))
            )
        if np.isfinite(row["dAmp1"]):
            damp_errors.append(
                abs(float(row["dAmp1"]) - (float(row["urban_Amp1"]) - float(row["rural_Amp1"])))
            )
    diagnostics = {
        "n_pair_period_rows_checked": int(len(rows)),
        "max_amp_reconstruction_abs_error": float(np.nanmax(amp_errors)) if amp_errors else np.nan,
        "max_phase_reconstruction_abs_error_rad": float(np.nanmax(phase_errors)) if phase_errors else np.nan,
        "max_dAmp1_identity_abs_error": float(np.nanmax(damp_errors)) if damp_errors else np.nan,
    }
    if diagnostics["max_amp_reconstruction_abs_error"] > 1.0e-5:
        raise ValueError(f"Stored amplitude and reconstructed curve disagree: {diagnostics}")
    if diagnostics["max_phase_reconstruction_abs_error_rad"] > 1.0e-5:
        raise ValueError(f"Stored phase and reconstructed curve disagree: {diagnostics}")
    if diagnostics["max_dAmp1_identity_abs_error"] > 1.0e-5:
        raise ValueError(f"dAmp1 is inconsistent with urban_Amp1-rural_Amp1: {diagnostics}")
    return diagnostics


def timing_metrics_for_rows(n: pd.Series, h: pd.Series) -> Dict:
    metrics: Dict[str, float | str | int] = {}
    period_results: Dict[str, Dict[str, float]] = {}
    for label, row in (("NHW", n), ("HW", h)):
        u = row[curve_columns("urban")].to_numpy(dtype=float)
        r = row[curve_columns("rural")].to_numpy(dtype=float)
        if not (np.isfinite(u).all() and np.isfinite(r).all()):
            raise ValueError(f"nonfinite_24h_curve_{label}")
        delta = u - r
        cu = dft_coefficients(u)
        cr = dft_coefficients(r)
        cd = cu - cr
        if abs(cd[1]) <= EPS or abs(cr[1]) <= EPS:
            raise ValueError(f"undefined_first_harmonic_phase_{label}")

        t_delta_h1 = harmonic_peak_time_from_coefficient(cd[1])
        t_rural_h1 = harmonic_peak_time_from_coefficient(cr[1])
        lag_h1 = float((t_delta_h1 - t_rural_h1) % 24.0)

        t_delta_full = two_harmonic_peak_time(delta)
        t_rural_full = two_harmonic_peak_time(r)
        lag_full = float((t_delta_full - t_rural_full) % 24.0)

        area_signed = signed_loop_area(r, delta)
        period_results[label] = {
            "lag_h1": lag_h1,
            "lag_full": lag_full,
            "contrast_A1": float(2.0 * abs(cd[1])),
            "rural_A1": float(2.0 * abs(cr[1])),
            "loop_area_signed": area_signed,
            "loop_area_abs": abs(area_signed),
        }

    for label in ("NHW", "HW"):
        for key, value in period_results[label].items():
            metrics[f"{key}_{label}"] = value

    metrics[PRIMARY_TIMING_METRIC] = wrap_signed_hours(
        period_results["HW"]["lag_h1"] - period_results["NHW"]["lag_h1"]
    )
    metrics["two_harmonic_peak_offset_change_h"] = wrap_signed_hours(
        period_results["HW"]["lag_full"] - period_results["NHW"]["lag_full"]
    )
    metrics["signed_hysteresis_area_change_C2"] = (
        period_results["HW"]["loop_area_signed"] - period_results["NHW"]["loop_area_signed"]
    )
    metrics["absolute_hysteresis_area_change_C2"] = (
        period_results["HW"]["loop_area_abs"] - period_results["NHW"]["loop_area_abs"]
    )
    metrics["min_contrast_A1"] = min(
        period_results["NHW"]["contrast_A1"], period_results["HW"]["contrast_A1"]
    )
    metrics["timing_primary_valid"] = 1
    metrics["timing_exclusion_reason"] = ""
    return metrics


def derive_timing_metrics(
    state: pd.DataFrame, selected: pd.DataFrame, attrition: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nhw = selected[selected["period"].eq(PERIOD_NHW)].set_index("pair_id")
    hw = selected[selected["period"].eq(PERIOD_HW)].set_index("pair_id")
    timing_rows: List[Dict] = []
    extra_attrition: List[Dict] = []
    for pair_id in state["pair_id"]:
        try:
            values = timing_metrics_for_rows(nhw.loc[pair_id], hw.loc[pair_id])
        except Exception as exc:
            values = {
                PRIMARY_TIMING_METRIC: np.nan,
                "two_harmonic_peak_offset_change_h": np.nan,
                "signed_hysteresis_area_change_C2": np.nan,
                "absolute_hysteresis_area_change_C2": np.nan,
                "min_contrast_A1": np.nan,
                "timing_primary_valid": 0,
                "timing_exclusion_reason": str(exc),
            }
            extra_attrition.append(
                {"pair_id": pair_id, "stage": "timing_metric", "reason": str(exc)}
            )
        timing_rows.append({"pair_id": pair_id, **values})

    timing = pd.DataFrame(timing_rows)
    out = state.merge(timing, on="pair_id", how="left", validate="one_to_one")
    if extra_attrition:
        attrition = pd.concat([attrition, pd.DataFrame(extra_attrition)], ignore_index=True)
    return out, attrition


def make_panel_a_binned_flow(state: pd.DataFrame, bins: int = 10, min_count: int = 2) -> pd.DataFrame:
    x = np.concatenate([state["delta_Tmean_NHW"], state["delta_Tmean_HW"]])
    y = np.concatenate([state["delta_Amp_NHW"], state["delta_Amp_HW"]])
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymin == ymax:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    x_edges = np.linspace(xmin, xmax, bins + 1)
    y_edges = np.linspace(ymin, ymax, bins + 1)
    rows: List[Dict] = []
    for group in GROUP_ORDER:
        g = state[state["annual_group"].eq(group)].copy()
        g["xb"] = pd.cut(g["delta_Tmean_NHW"], x_edges, labels=False, include_lowest=True)
        g["yb"] = pd.cut(g["delta_Amp_NHW"], y_edges, labels=False, include_lowest=True)
        summary = (
            g.dropna(subset=["xb", "yb"])
            .groupby(["xb", "yb"], observed=True)
            .agg(
                x0=("delta_Tmean_NHW", "mean"),
                y0=("delta_Amp_NHW", "mean"),
                dx=("R_mean", "mean"),
                dy=("R_Amp", "mean"),
                n=("pair_id", "size"),
            )
            .reset_index()
        )
        summary = summary[summary["n"] >= min_count].copy()
        summary["annual_group"] = group
        rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def build_block_draws(state: pd.DataFrame, n_boot: int, seed: int) -> Tuple[List[str], np.ndarray]:
    blocks = sorted(state["block_id"].dropna().astype(str).unique().tolist())
    if len(blocks) < 2:
        raise ValueError("At least two occupied spatial blocks are required.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(n_boot, len(blocks)), endpoint=False)
    return blocks, draws


def weights_from_draw(block_series: pd.Series, blocks: Sequence[str], draw: np.ndarray) -> np.ndarray:
    counts = np.bincount(draw, minlength=len(blocks)).astype(float)
    block_to_idx = {b: i for i, b in enumerate(blocks)}
    return block_series.astype(str).map(lambda b: counts[block_to_idx[b]]).to_numpy(float)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_lstsq(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if weights is not None:
        mask &= np.isfinite(weights) & (weights > 0)
    Xv = X[mask]
    yv = y[mask]
    if len(yv) <= X.shape[1]:
        raise ValueError("Insufficient weighted observations for model fit.")
    if weights is None:
        return np.linalg.lstsq(Xv, yv, rcond=None)[0]
    sw = np.sqrt(weights[mask])
    return np.linalg.lstsq(Xv * sw[:, None], yv * sw, rcond=None)[0]


def category_levels(series: pd.Series) -> List[str]:
    values = sorted(series.fillna("Unknown").astype(str).unique().tolist())
    return values or ["Unknown"]


def model_design(
    df: pd.DataFrame,
    mu_mean: float,
    sd_mean: float,
    mu_amp: float,
    sd_amp: float,
    group_levels: Sequence[str],
    continent_levels: Sequence[str],
    controls: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    z_mean = (df["R_mean"].to_numpy(float) - mu_mean) / sd_mean
    damping = -(df["R_Amp"].to_numpy(float) - mu_amp) / sd_amp
    cols = [np.ones(len(df)), z_mean]
    names = ["intercept", "z_R_mean"]
    if controls:
        group = df["annual_group"].fillna("Unknown").astype(str)
        continent = df["continent"].fillna("Unknown").astype(str)
        for level in group_levels[1:]:
            cols.append(group.eq(level).to_numpy(float))
            names.append(f"annual_group[{level}]")
        for level in continent_levels[1:]:
            cols.append(continent.eq(level).to_numpy(float))
            names.append(f"continent[{level}]")
    W = np.column_stack(cols)
    X = np.column_stack([W, damping])
    names.append("damping_minus_z_R_Amp")
    return X, W, names


def fit_timing_model(
    df: pd.DataFrame,
    outcome: str,
    weights: Optional[np.ndarray] = None,
    scaling: Optional[Tuple[float, float, float, float]] = None,
    group_levels: Optional[Sequence[str]] = None,
    continent_levels: Optional[Sequence[str]] = None,
    controls: bool = True,
) -> Dict:
    cols = ["pair_id", "R_mean", "R_Amp", outcome, "annual_group", "continent", "block_id"]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["R_mean", "R_Amp", outcome]).copy()
    if len(d) < 20:
        raise ValueError(f"Fewer than 20 complete pairs for timing outcome {outcome}.")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[d.index.to_numpy()]

    if scaling is None:
        mu_mean = float(d["R_mean"].mean())
        sd_mean = float(d["R_mean"].std(ddof=1))
        mu_amp = float(d["R_Amp"].mean())
        sd_amp = float(d["R_Amp"].std(ddof=1))
    else:
        mu_mean, sd_mean, mu_amp, sd_amp = scaling
    if min(sd_mean, sd_amp) <= 0 or not np.isfinite([mu_mean, sd_mean, mu_amp, sd_amp]).all():
        raise ValueError("Invalid predictor standardization.")

    if group_levels is None:
        group_levels = category_levels(d["annual_group"])
    if continent_levels is None:
        continent_levels = category_levels(d["continent"])
    X, W, names = model_design(
        d, mu_mean, sd_mean, mu_amp, sd_amp, group_levels, continent_levels, controls=controls
    )
    y = d[outcome].to_numpy(float)
    beta = weighted_lstsq(X, y, weights)
    y_sd = float(np.std(y, ddof=1))

    damping = X[:, -1]
    if weights is None:
        gamma_y = weighted_lstsq(W, y)
        gamma_x = weighted_lstsq(W, damping)
    else:
        gamma_y = weighted_lstsq(W, y, weights)
        gamma_x = weighted_lstsq(W, damping, weights)
    y_res = y - W @ gamma_y
    x_res = damping - W @ gamma_x

    return {
        "data": d,
        "beta": beta,
        "names": names,
        "scaling": (mu_mean, sd_mean, mu_amp, sd_amp),
        "group_levels": list(group_levels),
        "continent_levels": list(continent_levels),
        "y_sd": y_sd,
        "x_res": x_res,
        "y_res": y_res,
        "beta_mean_h_per_sd": float(beta[names.index("z_R_mean")]),
        "beta_damping_h_per_sd": float(beta[names.index("damping_minus_z_R_Amp")]),
        "beta_mean_standardized": float(beta[names.index("z_R_mean")] / y_sd),
        "beta_damping_standardized": float(beta[names.index("damping_minus_z_R_Amp")] / y_sd),
    }


def quantile_bin_ids(x: np.ndarray, q: int = 5) -> np.ndarray:
    series = pd.Series(x)
    try:
        bins = pd.qcut(series, q=q, labels=False, duplicates="drop")
    except ValueError:
        ranks = series.rank(method="average", pct=True)
        bins = np.floor(np.minimum(ranks.to_numpy(), 0.999999) * q)
    return np.asarray(bins, dtype=float)


def bootstrap_all(
    state: pd.DataFrame,
    timing: pd.DataFrame,
    blocks: Sequence[str],
    draws: np.ndarray,
    full_model: Dict,
    amplitude_support_model: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_data = full_model["data"]
    scaling = full_model["scaling"]
    group_levels = full_model["group_levels"]
    continent_levels = full_model["continent_levels"]
    bin_ids = quantile_bin_ids(full_model["x_res"], q=5)
    replicate_rows: List[Dict] = []
    bin_rows: List[Dict] = []

    for rep, draw in enumerate(draws):
        w_state = weights_from_draw(state["block_id"], blocks, draw)
        row: Dict[str, float | int] = {
            "replicate": rep,
            "timing_fit_valid": 0,
            "timing_amplitude_support_fit_valid": 0,
        }
        for group in GROUP_ORDER:
            mask = state["annual_group"].eq(group).to_numpy()
            w = w_state * mask.astype(float)
            for variable in (
                "delta_Tmean_NHW", "delta_Amp_NHW", "delta_Tmean_HW", "delta_Amp_HW",
                "R_mean", "R_Amp",
            ):
                row[f"{group}_{variable}"] = weighted_mean(state[variable].to_numpy(float), w)
            row[f"{group}_theta_deg"] = float(
                np.degrees(np.arctan2(row[f"{group}_R_Amp"], row[f"{group}_R_mean"]))
            )

            decomp_mask = mask & state["decomposition_valid"].eq(1).to_numpy()
            w_decomp = w_state * decomp_mask.astype(float)
            for target in ("day", "night"):
                for component, variable in (
                    ("mean", f"{target}_mean_component"),
                    ("amplitude", f"{target}_amplitude_component"),
                    ("shape_phase_residual", f"{target}_shape_phase_residual"),
                    ("observed", f"R_{target}_observed"),
                ):
                    row[f"{group}_{target}_{component}"] = weighted_mean(
                        state[variable].to_numpy(float), w_decomp
                    )

        try:
            w_model = weights_from_draw(model_data["block_id"], blocks, draw)
            fit = fit_timing_model(
                model_data.reset_index(drop=True),
                PRIMARY_TIMING_METRIC,
                weights=w_model,
                scaling=scaling,
                group_levels=group_levels,
                continent_levels=continent_levels,
                controls=True,
            )
            row["timing_fit_valid"] = 1
            row["beta_mean_h_per_sd"] = fit["beta_mean_h_per_sd"]
            row["beta_damping_h_per_sd"] = fit["beta_damping_h_per_sd"]
            row["beta_mean_standardized"] = fit["beta_mean_standardized"]
            row["beta_damping_standardized"] = fit["beta_damping_standardized"]

            for bin_id in sorted(set(bin_ids[np.isfinite(bin_ids)].astype(int))):
                mask = bin_ids == bin_id
                w_bin = w_model * mask.astype(float)
                bin_rows.append(
                    {
                        "replicate": rep,
                        "bin_id": int(bin_id),
                        "x_mean": weighted_mean(full_model["x_res"], w_bin),
                        "y_mean": weighted_mean(full_model["y_res"], w_bin),
                    }
                )
        except Exception:
            row["beta_mean_h_per_sd"] = np.nan
            row["beta_damping_h_per_sd"] = np.nan
            row["beta_mean_standardized"] = np.nan
            row["beta_damping_standardized"] = np.nan

        if amplitude_support_model is not None:
            try:
                amp_data = amplitude_support_model["data"]
                w_amp = weights_from_draw(amp_data["block_id"], blocks, draw)
                amp_fit = fit_timing_model(
                    amp_data.reset_index(drop=True),
                    PRIMARY_TIMING_METRIC,
                    weights=w_amp,
                    scaling=amplitude_support_model["scaling"],
                    group_levels=amplitude_support_model["group_levels"],
                    continent_levels=amplitude_support_model["continent_levels"],
                    controls=True,
                )
                row["timing_amplitude_support_fit_valid"] = 1
                row["beta_damping_amplitude_support_h_per_sd"] = (
                    amp_fit["beta_damping_h_per_sd"]
                )
            except Exception:
                row["beta_damping_amplitude_support_h_per_sd"] = np.nan
        replicate_rows.append(row)

    return pd.DataFrame(replicate_rows), pd.DataFrame(bin_rows)


def unwrap_bootstrap_angles(values: np.ndarray, point: float) -> np.ndarray:
    return point + circular_difference_degrees(values, np.full_like(values, point))


def state_group_summary(state: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    variables = (
        "delta_Tmean_NHW", "delta_Amp_NHW", "delta_Tmean_HW", "delta_Amp_HW",
        "R_mean", "R_Amp",
    )
    for group in GROUP_ORDER:
        g = state[state["annual_group"].eq(group)]
        if g.empty:
            continue
        row: Dict[str, float | int | str] = {"annual_group": group, "n": int(len(g))}
        for variable in variables:
            point = float(g[variable].mean())
            vals = boot[f"{group}_{variable}"].dropna().to_numpy(float)
            lo, hi = np.quantile(vals, [0.025, 0.975]) if len(vals) else (np.nan, np.nan)
            row[variable] = point
            row[f"{variable}_ci_low"] = float(lo)
            row[f"{variable}_ci_high"] = float(hi)
        theta = float(np.degrees(np.arctan2(row["R_Amp"], row["R_mean"])))
        theta_boot = boot[f"{group}_theta_deg"].dropna().to_numpy(float)
        theta_unwrapped = unwrap_bootstrap_angles(theta_boot, theta) if len(theta_boot) else np.array([])
        tlo, thi = np.quantile(theta_unwrapped, [0.025, 0.975]) if len(theta_unwrapped) else (np.nan, np.nan)
        row["theta_deg"] = theta
        row["theta_ci_low"] = float(tlo)
        row["theta_ci_high"] = float(thi)
        row["rho_group_mean_vector"] = float(np.hypot(row["R_mean"], row["R_Amp"]))
        rows.append(row)
    return pd.DataFrame(rows)


def day_night_decomposition_summary(
    state: pd.DataFrame, boot: pd.DataFrame
) -> pd.DataFrame:
    """Return pair-level group means and block-bootstrap CIs for Fig. 3c."""
    valid = state[state["decomposition_valid"].eq(1)].copy()
    rows: List[Dict] = []
    component_specs = (
        ("mean", "Mean", "mean_component"),
        ("amplitude", "Signed amplitude", "amplitude_component"),
        ("shape_phase_residual", "Shape/phase residual", "shape_phase_residual"),
        ("observed", "Observed response", None),
    )
    for group in GROUP_ORDER:
        g = valid[valid["annual_group"].eq(group)].copy()
        for target, target_label, observed_col in (
            ("day", "Day", "R_day_observed"),
            ("night", "Night", "R_night_observed"),
        ):
            for order, (component, label, suffix) in enumerate(component_specs):
                variable = observed_col if component == "observed" else f"{target}_{suffix}"
                point = float(g[variable].mean())
                values = boot[f"{group}_{target}_{component}"].dropna().to_numpy(float)
                lo, hi = np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)
                rows.append(
                    {
                        "annual_group": group,
                        "target": target,
                        "target_label": target_label,
                        "component": component,
                        "component_label": label,
                        "component_order": order,
                        "estimate": point,
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "n_pairs": int(len(g)),
                        "n_blocks": int(g["block_id"].nunique()),
                        "bootstrap_valid": int(np.isfinite(values).sum()),
                        "bootstrap_requested": int(len(boot)),
                        "coordinate_identity": (
                            "R_day=R_mean+R_Amp+residual"
                            if target == "day"
                            else "R_night=R_mean-R_Amp+residual"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def amplitude_support_sensitivity_summary(
    full_model: Dict,
    amplitude_support_model: Dict,
    boot: pd.DataFrame,
    cutoff: float,
) -> pd.DataFrame:
    """Formal block-bootstrap sensitivity for phase stability at low H1 amplitude."""
    rows: List[Dict] = []
    specs = (
        (
            "primary_all_timing_valid",
            full_model,
            boot.loc[boot["timing_fit_valid"].eq(1), "beta_damping_h_per_sd"],
            np.nan,
        ),
        (
            "exclude_lowest_10pct_contrast_A1",
            amplitude_support_model,
            boot.loc[
                boot["timing_amplitude_support_fit_valid"].eq(1),
                "beta_damping_amplitude_support_h_per_sd",
            ],
            cutoff,
        ),
    )
    for label, model, values, threshold in specs:
        vals = values.dropna().to_numpy(float)
        lo, hi = np.quantile(vals, [0.025, 0.975]) if len(vals) else (np.nan, np.nan)
        rows.append(
            {
                "analysis": label,
                "estimate_h_per_sd": float(model["beta_damping_h_per_sd"]),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_pairs": int(len(model["data"])),
                "n_blocks": int(model["data"]["block_id"].nunique()),
                "bootstrap_valid": int(len(vals)),
                "bootstrap_requested": int(len(boot)),
                "predefined_min_contrast_A1_cutoff": threshold,
            }
        )
    return pd.DataFrame(rows)


def coefficient_summary(full_model: Dict, boot: pd.DataFrame) -> pd.DataFrame:
    valid = boot[boot["timing_fit_valid"].eq(1)].copy()
    rows: List[Dict] = []
    for term, point_key, boot_col in (
        ("R_mean", "beta_mean_h_per_sd", "beta_mean_h_per_sd"),
        ("amplitude_damping", "beta_damping_h_per_sd", "beta_damping_h_per_sd"),
        ("R_mean_standardized", "beta_mean_standardized", "beta_mean_standardized"),
        ("amplitude_damping_standardized", "beta_damping_standardized", "beta_damping_standardized"),
    ):
        vals = valid[boot_col].dropna().to_numpy(float)
        lo, hi = np.quantile(vals, [0.025, 0.975]) if len(vals) else (np.nan, np.nan)
        rows.append(
            {
                "term": term,
                "estimate": float(full_model[point_key]),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_pairs": int(len(full_model["data"])),
                "n_blocks": int(full_model["data"]["block_id"].nunique()),
                "bootstrap_valid": int(len(valid)),
                "bootstrap_requested": int(len(boot)),
                "outcome": PRIMARY_TIMING_METRIC,
                "controls": "+".join(PRIMARY_CONTROLS),
            }
        )
    return pd.DataFrame(rows)


def fwl_outputs(full_model: Dict, boot: pd.DataFrame, bin_boot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = full_model["data"].copy().reset_index(drop=True)
    pair = d[["pair_id", "annual_group", "continent", "block_id", "R_mean", "R_Amp", PRIMARY_TIMING_METRIC]].copy()
    pair["fwl_damping_residual"] = full_model["x_res"]
    pair["fwl_timing_residual_h"] = full_model["y_res"]
    pair["fwl_bin_id"] = quantile_bin_ids(full_model["x_res"], q=5)

    x_min = float(np.nanmin(full_model["x_res"]))
    x_max = float(np.nanmax(full_model["x_res"]))
    grid = np.linspace(x_min, x_max, 200)
    beta = full_model["beta_damping_h_per_sd"]
    beta_boot = boot.loc[boot["timing_fit_valid"].eq(1), "beta_damping_h_per_sd"].dropna().to_numpy(float)
    pred_boot = beta_boot[:, None] * grid[None, :]
    lo, hi = np.quantile(pred_boot, [0.025, 0.975], axis=0)
    curve = pd.DataFrame(
        {
            "x_fwl_damping": grid,
            "fit_timing_h": beta * grid,
            "ci_low": lo,
            "ci_high": hi,
        }
    )

    rows: List[Dict] = []
    for bin_id, g in pair.groupby("fwl_bin_id", dropna=True):
        reps = bin_boot[bin_boot["bin_id"].eq(int(bin_id))]
        yvals = reps["y_mean"].dropna().to_numpy(float)
        xvals = reps["x_mean"].dropna().to_numpy(float)
        ylo, yhi = np.quantile(yvals, [0.025, 0.975]) if len(yvals) else (np.nan, np.nan)
        xlo, xhi = np.quantile(xvals, [0.025, 0.975]) if len(xvals) else (np.nan, np.nan)
        rows.append(
            {
                "bin_id": int(bin_id),
                "n": int(len(g)),
                "x_mean": float(g["fwl_damping_residual"].mean()),
                "x_ci_low": float(xlo),
                "x_ci_high": float(xhi),
                "y_mean": float(g["fwl_timing_residual_h"].mean()),
                "y_ci_low": float(ylo),
                "y_ci_high": float(yhi),
            }
        )
    return pair, curve, pd.DataFrame(rows)


def fit_sensitivity_row(
    timing: pd.DataFrame,
    outcome: str,
    label: str,
    controls: bool = True,
    subset: Optional[pd.Series] = None,
) -> Dict:
    d = timing.copy()
    if subset is not None:
        d = d.loc[subset].copy()
    try:
        fit = fit_timing_model(d.reset_index(drop=True), outcome, controls=controls)
        return {
            "analysis": label,
            "outcome": outcome,
            "n": int(len(fit["data"])),
            "beta_damping_outcome_units_per_sd": fit["beta_damping_h_per_sd"],
            "beta_damping_standardized": fit["beta_damping_standardized"],
            "beta_mean_outcome_units_per_sd": fit["beta_mean_h_per_sd"],
            "status": "ok",
        }
    except Exception as exc:
        return {
            "analysis": label,
            "outcome": outcome,
            "n": 0,
            "beta_damping_outcome_units_per_sd": np.nan,
            "beta_damping_standardized": np.nan,
            "beta_mean_outcome_units_per_sd": np.nan,
            "status": f"failed:{exc}",
        }


def sensitivity_analyses(timing: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    for outcome, label in (
        (PRIMARY_TIMING_METRIC, "primary_controlled"),
        (PRIMARY_TIMING_METRIC, "primary_no_categorical_controls"),
        ("two_harmonic_peak_offset_change_h", "two_harmonic_peak_offset"),
        ("signed_hysteresis_area_change_C2", "signed_hysteresis_area"),
        ("absolute_hysteresis_area_change_C2", "absolute_hysteresis_area"),
    ):
        rows.append(
            fit_sensitivity_row(
                timing,
                outcome,
                label,
                controls=(label != "primary_no_categorical_controls"),
            )
        )

    for group in GROUP_ORDER:
        rows.append(
            fit_sensitivity_row(
                timing,
                PRIMARY_TIMING_METRIC,
                f"within_{group}",
                controls=True,
                subset=timing["annual_group"].eq(group),
            )
        )

    finite_amp = timing["min_contrast_A1"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_amp):
        cutoff = float(finite_amp.quantile(0.10))
        row = fit_sensitivity_row(
            timing,
            PRIMARY_TIMING_METRIC,
            "exclude_lowest_10pct_contrast_A1",
            controls=True,
            subset=timing["min_contrast_A1"] > cutoff,
        )
        row["predefined_cutoff"] = cutoff
        rows.append(row)

    loco_rows: List[Dict] = []
    for continent in sorted(timing["continent"].dropna().astype(str).unique()):
        row = fit_sensitivity_row(
            timing,
            PRIMARY_TIMING_METRIC,
            f"leave_out_{continent}",
            controls=True,
            subset=~timing["continent"].astype(str).eq(continent),
        )
        row["continent_left_out"] = continent
        loco_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(loco_rows)


def hysteresis_inset_table(selected: pd.DataFrame, timing_ids: set[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    sub = selected[
        selected["pair_id"].isin(timing_ids) & selected["period"].isin([PERIOD_NHW, PERIOD_HW])
    ].copy()
    for group in GROUP_ORDER:
        for period, label in ((PERIOD_NHW, "NHW"), (PERIOD_HW, "HW")):
            g = sub[sub["annual_group"].eq(group) & sub["period"].eq(period)]
            if g.empty:
                continue
            rural = g[curve_columns("rural")].to_numpy(float)
            urban = g[curve_columns("urban")].to_numpy(float)
            contrast = urban - rural
            for hour in range(24):
                rows.append(
                    {
                        "annual_group": group,
                        "state": label,
                        "hour": hour,
                        "rural_temperature": float(np.nanmean(rural[:, hour])),
                        "urban_rural_contrast": float(np.nanmean(contrast[:, hour])),
                        "n": int(len(g)),
                    }
                )
    return pd.DataFrame(rows)


def jacobian_direction_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "channel": "persistent_mean_loading",
                "label": "Persistent mean loading  F0 up",
                "sign_R_mean": "+",
                "sign_R_Amp": "0",
                "angle_min_deg": 0.0,
                "angle_max_deg": 0.0,
                "identified_quantity": "direction_only",
            },
            {
                "channel": "relative_amplitude_damping",
                "label": "Relative damping  F1 down and/or C up",
                "sign_R_mean": "0",
                "sign_R_Amp": "-",
                "angle_min_deg": -90.0,
                "angle_max_deg": -90.0,
                "identified_quantity": "direction_only_F1_C_degenerate",
            },
            {
                "channel": "weaker_effective_removal",
                "label": "Weaker effective removal  lambda down",
                "sign_R_mean": "+",
                "sign_R_Amp": "+",
                "angle_min_deg": 0.0,
                "angle_max_deg": 90.0,
                "identified_quantity": "quadrant_only_exact_slope_not_identified",
            },
        ]
    )


def rom_pi_sensitivity_table() -> pd.DataFrame:
    pi = np.logspace(-2, 2, 700)
    omega = TWO_PI / 24.0
    return pd.DataFrame(
        {
            "Pi_1": pi,
            "equivalent_tau_h": pi / omega,
            "forcing_amplitude_weight": np.ones_like(pi),
            "removal_weight_w_lambda": 1.0 / (1.0 + pi**2),
            "storage_weight_w_C": pi**2 / (1.0 + pi**2),
            "timing_weight_w_phi": pi / (1.0 + pi**2),
            "second_harmonic_storage_weight": (2.0 * pi) ** 2 / (1.0 + (2.0 * pi) ** 2),
        }
    )


def diagnostics_table(state: pd.DataFrame, timing: pd.DataFrame, full_model: Dict, boot: pd.DataFrame) -> pd.DataFrame:
    y = full_model["data"][PRIMARY_TIMING_METRIC].to_numpy(float)
    r = float(np.corrcoef(timing["R_mean"], -timing["R_Amp"])[0, 1])
    vif = float(1.0 / (1.0 - r**2)) if abs(r) < 1.0 else np.inf
    values = [
        ("state_pairs", len(state)),
        ("timing_pairs", len(full_model["data"])),
        ("occupied_blocks_state", state["block_id"].nunique()),
        ("occupied_blocks_timing", full_model["data"]["block_id"].nunique()),
        ("bootstrap_requested", len(boot)),
        ("bootstrap_valid_timing", int(boot["timing_fit_valid"].sum())),
        ("bootstrap_valid_fraction", float(boot["timing_fit_valid"].mean())),
        ("timing_abs_ge_10h_fraction", float(np.mean(np.abs(y) >= 10.0))),
        ("timing_abs_ge_11_5h_fraction", float(np.mean(np.abs(y) >= 11.5))),
        ("predictor_correlation_Rmean_vs_damping", r),
        ("two_predictor_VIF", vif),
    ]
    return pd.DataFrame(values, columns=["diagnostic", "value"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the frozen Figure 3 state-process-timing analysis."
    )
    p.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to all_pair_period_metrics.csv (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"New, empty output directory (default: {DEFAULT_OUTDIR})",
    )
    p.add_argument("--hw-method", default="percentile")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--block-deg", type=float, default=DEFAULT_BLOCK_DEG)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if args.bootstrap < 200:
        raise ValueError("At least 200 bootstrap replicates are required; use 5000 for final analysis.")
    prepare_output_dir(outdir)

    input_hash_before = sha256_file(input_path)
    df, input_info = load_and_validate_input(input_path, args.hw_method)
    state, selected, attrition = build_matched_state_cohort(df)
    state = assign_spatial_blocks(state, args.block_deg)
    fft_diagnostics = validate_fft_consistency(selected, set(state["pair_id"]))
    state, attrition = derive_timing_metrics(state, selected, attrition)
    state, attrition = add_response_decomposition(state, attrition)

    timing = state[
        state["timing_primary_valid"].eq(1)
        & np.isfinite(state[PRIMARY_TIMING_METRIC])
    ].copy().reset_index(drop=True)
    if len(timing) < 20:
        raise ValueError(f"Only {len(timing)} timing-valid pairs; panel d requires at least 20.")

    finite_amp = timing["min_contrast_A1"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_amp) < 20:
        raise ValueError("Insufficient finite first-harmonic amplitudes for timing sensitivity.")
    amplitude_support_cutoff = float(finite_amp.quantile(0.10))
    amplitude_support_timing = timing[
        timing["min_contrast_A1"] > amplitude_support_cutoff
    ].copy().reset_index(drop=True)
    if len(amplitude_support_timing) < 20:
        raise ValueError("Amplitude-support timing sensitivity has fewer than 20 pairs.")

    blocks, draws = build_block_draws(state, args.bootstrap, args.seed)
    full_model = fit_timing_model(timing, PRIMARY_TIMING_METRIC, controls=True)
    amplitude_support_model = fit_timing_model(
        amplitude_support_timing, PRIMARY_TIMING_METRIC, controls=True
    )
    boot, bin_boot = bootstrap_all(
        state,
        timing,
        blocks,
        draws,
        full_model,
        amplitude_support_model=amplitude_support_model,
    )
    valid_fraction = float(boot["timing_fit_valid"].mean())
    if valid_fraction < 0.95:
        raise RuntimeError(
            f"Only {valid_fraction:.1%} of timing bootstrap fits were valid; main-text inference stopped."
        )
    amplitude_support_valid_fraction = float(
        boot["timing_amplitude_support_fit_valid"].mean()
    )
    if amplitude_support_valid_fraction < 0.95:
        raise RuntimeError(
            f"Only {amplitude_support_valid_fraction:.1%} of amplitude-support "
            "bootstrap fits were valid; timing sensitivity inference stopped."
        )

    group_summary = state_group_summary(state, boot)
    binned_flow = make_panel_a_binned_flow(state)
    decomposition = day_night_decomposition_summary(state, boot)
    coefficients = coefficient_summary(full_model, boot)
    fwl_pair, fwl_curve, fwl_bins = fwl_outputs(full_model, boot, bin_boot)
    amplitude_support = amplitude_support_sensitivity_summary(
        full_model, amplitude_support_model, boot, amplitude_support_cutoff
    )
    sensitivity, loco = sensitivity_analyses(timing)
    loop_table = hysteresis_inset_table(selected, set(timing["pair_id"]))
    jacobian = jacobian_direction_table()
    rom_pi = rom_pi_sensitivity_table()
    diagnostics = diagnostics_table(state, timing, full_model, boot)

    output_frames: Dict[str, pd.DataFrame] = {
        "fig3_pair_level_state_timing.csv": state,
        "fig3_attrition_table.csv": attrition,
        "fig3_group_state_summary.csv": group_summary,
        "fig3_panel_a_binned_flow.csv": binned_flow,
        "fig3_panel_b_jacobian_directions.csv": jacobian,
        "fig3_panel_b_rom_pi_sensitivity.csv": rom_pi,
        "fig3_panel_c_day_night_decomposition.csv": decomposition,
        "fig3_panel_d_coefficients.csv": coefficients,
        "fig3_panel_d_fwl_pair_level.csv": fwl_pair,
        "fig3_panel_d_fwl_curve.csv": fwl_curve,
        "fig3_panel_d_fwl_bins.csv": fwl_bins,
        "fig3_panel_d_amplitude_support_sensitivity.csv": amplitude_support,
        "fig3_hysteresis_context_SI.csv": loop_table,
        "fig3_timing_sensitivity.csv": sensitivity,
        "fig3_leave_one_continent_out.csv": loco,
        "fig3_analysis_diagnostics.csv": diagnostics,
    }
    for name, frame in output_frames.items():
        frame.to_csv(outdir / name, index=False)
    boot.to_csv(outdir / "fig3_bootstrap_replicates.csv.gz", index=False, compression="gzip")

    input_hash_after = sha256_file(input_path)
    if input_hash_after != input_hash_before:
        raise RuntimeError("Input SHA-256 changed during analysis; outputs are not frozen.")

    output_names = list(output_frames) + ["fig3_bootstrap_replicates.csv.gz"]
    output_hashes = {name: sha256_file(outdir / name) for name in output_names}
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_path": str(input_path),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "input_info": input_info,
        "frozen_analysis": {
            "hw_method": args.hw_method,
            "periods": list(PERIOD_ORDER),
            "annual_group_definition": "annual two-harmonic dTx: UHI if dTx>=0; UCI if dTx<0",
            "state_coordinates": ["R_mean", "R_Amp"],
            "response_magnitude_name": "rho",
            "D_reserved_definition": "R_mean-R_Amp",
            "primary_timing_metric": PRIMARY_TIMING_METRIC,
            "timing_convention": "period forward offset in [0,24); HW-NHW change wrapped to [-12,12)",
            "primary_model": "Y ~ z(R_mean) + [-z(R_Amp)] + annual_group + continent",
            "primary_controls": list(PRIMARY_CONTROLS),
            "spatial_block_degrees": args.block_deg,
            "spatial_block_coordinate": "urban-rural pair midpoint",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "main_amplitude_threshold": "none; only undefined/nonfinite phase excluded",
            "timing_amplitude_support_sensitivity": (
                "predefined exclusion of the lowest 10% min contrast H1 amplitude; "
                "same spatial-block bootstrap draws"
            ),
            "timing_amplitude_support_cutoff": amplitude_support_cutoff,
            "jacobian_scope": "sign/direction constraints only; no station parameter inversion",
            "day_night_decomposition": (
                "R_day=R_mean+R_Amp+shape_phase_residual; "
                "R_night=R_mean-R_Amp+shape_phase_residual"
            ),
        },
        "cohort_counts": {
            "state_pairs": len(state),
            "decomposition_pairs": int(state["decomposition_valid"].sum()),
            "timing_pairs": len(timing),
            "timing_amplitude_support_pairs": len(amplitude_support_timing),
            "state_blocks": state["block_id"].nunique(),
            "timing_blocks": timing["block_id"].nunique(),
            "state_groups": state["annual_group"].value_counts().to_dict(),
            "timing_groups": timing["annual_group"].value_counts().to_dict(),
        },
        "fft_consistency": fft_diagnostics,
        "output_sha256": output_hashes,
    }
    manifest_path = outdir / "fig3_analysis_manifest.json"
    write_json(manifest_path, manifest)
    success = {
        "status": "SUCCESS",
        "script_version": SCRIPT_VERSION,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "input_sha256": input_hash_before,
        "state_pairs": len(state),
        "timing_pairs": len(timing),
        "bootstrap_valid_fraction": valid_fraction,
        "amplitude_support_bootstrap_valid_fraction": amplitude_support_valid_fraction,
    }
    write_json(outdir / "_SUCCESS.json", success)

    print(json.dumps(json_ready(success), indent=2))
    print(f"Analysis outputs: {outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
        raise
