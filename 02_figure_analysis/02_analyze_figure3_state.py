#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze figure3 state.

Original scientific calculations retained; see README.md for inputs and execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "figure3_state_theory_reconstruction_analysis"
DEFAULT_BOOTSTRAP = 5000
MIN_TEST_BOOTSTRAP = 200
DEFAULT_SEED = 20260813
DEFAULT_BLOCK_DEG = 10.0
GRID_BINS = 10
GRID_MIN_COUNT = 2
PERIOD_ANNUAL = "annual"
PERIOD_NHW = "non_heatwave"
PERIOD_HW = "heatwave"
PERIOD_ORDER = (PERIOD_ANNUAL, PERIOD_NHW, PERIOD_HW)
GROUP_ORDER = ("UHI", "UCI")
TWO_PI = 2.0 * np.pi
EPS = 1.0e-12

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs/analysis/main_multiyear/robustness_percentile/all_pair_period_metrics.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs/analysis/figure3_state"

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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if not isinstance(value, (str, bool)) and pd.isna(value):
        return None
    return value


def write_json(path: Path, obj: Mapping) -> None:
    path.write_text(
        json.dumps(json_ready(dict(obj)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Use a new directory to preserve the frozen result bundle."
        )
    path.mkdir(parents=True, exist_ok=True)


def normalize_period(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "nhw": PERIOD_NHW,
        "nonheatwave": PERIOD_NHW,
        "non_heat_wave": PERIOD_NHW,
        "hw": PERIOD_HW,
        "heat_wave": PERIOD_HW,
        "year": PERIOD_ANNUAL,
        "yearly": PERIOD_ANNUAL,
    }
    return aliases.get(text, text)


def curve_columns(prefix: str) -> List[str]:
    return [f"{prefix}_diurnal_h{hour:02d}" for hour in range(24)]


def first_existing(columns: Iterable[str], aliases: Sequence[str], label: str) -> str:
    present = [name for name in aliases if name in columns]
    if not present:
        raise ValueError(f"Missing {label}; expected one of {list(aliases)}")
    return present[0]


def load_and_validate_input(path: Path, hw_method: str) -> Tuple[pd.DataFrame, Dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, low_memory=False)
    missing = sorted(CORE_REQUIRED.difference(raw.columns))
    missing.extend(
        name
        for name in curve_columns("urban") + curve_columns("rural")
        if name not in raw.columns
    )
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(set(missing))}")

    df = raw.copy()
    df["pair_id"] = df["pair_id"].astype(str).str.strip()
    df["period"] = df["period"].map(normalize_period)

    method_counts: Dict[str, int] = {}
    if "hw_method" in df.columns:
        method = df["hw_method"].astype(str).str.strip().str.lower()
        method_counts = {str(k): int(v) for k, v in method.value_counts(dropna=False).items()}
        df = df[method.eq(hw_method.lower())].copy()
        if df.empty:
            raise ValueError(f"No rows remain after selecting hw_method={hw_method!r}.")

    df = df[df["period"].isin(PERIOD_ORDER)].copy()
    if df.empty:
        raise ValueError("No annual/NHW/HW rows remain after period normalization.")
    duplicated = df.duplicated(["pair_id", "period"], keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, ["pair_id", "period"]].head(20).to_dict("records")
        raise ValueError(f"Duplicate pair-period rows detected; examples: {examples}")

    period_counts = {str(k): int(v) for k, v in df["period"].value_counts().items()}
    absent = [period for period in PERIOD_ORDER if period_counts.get(period, 0) == 0]
    if absent:
        raise ValueError(f"Required periods absent: {absent}")

    qc_versions: List[str] = []
    if "temperature_diurnal_qc_version" in df.columns:
        qc_versions = sorted(
            df["temperature_diurnal_qc_version"]
            .dropna().astype(str).str.strip().unique().tolist()
        )
        if len(qc_versions) > 1:
            raise ValueError(f"Mixed temperature QC versions in selected rows: {qc_versions}")

    qc_pass_fields: List[str] = []
    for candidate in (
        "temperature_diurnal_qc_pass_pair",
        "raw_day_night_support_qc_pass_pair",
        "raw_day_night_support_pass_pair",
    ):
        if candidate not in df.columns:
            continue
        qc_pass_fields.append(candidate)
        passed = pd.to_numeric(df[candidate], errors="coerce")
        bad = passed.ne(1) | passed.isna()
        if bad.any():
            examples = df.loc[bad, ["pair_id", "period", candidate]].head(20).to_dict("records")
            raise ValueError(
                f"Selected rows include failures of {candidate}; examples: {examples}"
            )

    numeric_columns = [
        "dTmean", "dAmp1", "dTx", "dTn", "urban_Amp1", "urban_Phase1",
        "rural_Amp1", "rural_Phase1",
    ] + curve_columns("urban") + curve_columns("rural")
    coordinate_aliases = (
        "lon_urban", "urban_lon", "lat_urban", "urban_lat",
        "lon_rural", "rural_lon", "lat_rural", "rural_lat",
    )
    numeric_columns.extend([name for name in coordinate_aliases if name in df.columns])
    for name in dict.fromkeys(numeric_columns):
        df[name] = pd.to_numeric(df[name], errors="coerce")

    info = {
        "input_rows_raw": int(len(raw)),
        "input_rows_selected": int(len(df)),
        "method_counts_before_selection": method_counts,
        "period_counts_selected": period_counts,
        "temperature_diurnal_qc_versions": qc_versions,
        "qc_pass_fields_checked": qc_pass_fields,
        "raw_support_qc_field_present": any("raw_day_night_support" in x for x in qc_pass_fields),
    }
    return df, info


def circular_lon_midpoint(lon1: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    a = np.deg2rad(lon1.astype(float))
    b = np.deg2rad(lon2.astype(float))
    z = np.exp(1j * a) + np.exp(1j * b)
    midpoint = np.rad2deg(np.angle(z))
    ambiguous = np.abs(z) < EPS
    midpoint[ambiguous] = (
        (lon1[ambiguous] + lon2[ambiguous] + 180.0) % 360.0
    ) - 180.0
    return midpoint


def build_matched_state_cohort(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attrition: List[Dict] = []
    annual = df[df["period"].eq(PERIOD_ANNUAL)].copy()
    annual_valid = np.isfinite(annual["dTx"])
    for pair_id in annual.loc[~annual_valid, "pair_id"].astype(str):
        attrition.append(
            {"pair_id": pair_id, "stage": "annual_group", "reason": "nonfinite_annual_dTx"}
        )
    annual = annual.loc[annual_valid].copy()
    annual["annual_group"] = np.where(annual["dTx"] >= 0.0, "UHI", "UCI")

    ids_by_period = {
        period: set(df.loc[df["period"].eq(period), "pair_id"].astype(str))
        for period in PERIOD_ORDER
    }
    annual_ids = set(annual["pair_id"].astype(str))
    matched_ids = annual_ids & ids_by_period[PERIOD_NHW] & ids_by_period[PERIOD_HW]
    for pair_id in sorted(set(df["pair_id"].astype(str)) - matched_ids):
        missing = []
        if pair_id not in annual_ids:
            missing.append(PERIOD_ANNUAL)
        if pair_id not in ids_by_period[PERIOD_NHW]:
            missing.append(PERIOD_NHW)
        if pair_id not in ids_by_period[PERIOD_HW]:
            missing.append(PERIOD_HW)
        attrition.append(
            {
                "pair_id": pair_id,
                "stage": "matched_periods",
                "reason": "missing_" + "+".join(missing),
            }
        )

    selected = df[df["pair_id"].isin(matched_ids)].copy()
    group_map = annual.set_index("pair_id")["annual_group"]
    selected["annual_group"] = selected["pair_id"].map(group_map)
    annual_indexed = selected[selected["period"].eq(PERIOD_ANNUAL)].set_index("pair_id")
    nhw_indexed = selected[selected["period"].eq(PERIOD_NHW)].set_index("pair_id")
    hw_indexed = selected[selected["period"].eq(PERIOD_HW)].set_index("pair_id")

    lon_u = first_existing(df.columns, ("lon_urban", "urban_lon"), "urban longitude")
    lat_u = first_existing(df.columns, ("lat_urban", "urban_lat"), "urban latitude")
    lon_r = first_existing(df.columns, ("lon_rural", "rural_lon"), "rural longitude")
    lat_r = first_existing(df.columns, ("lat_rural", "rural_lat"), "rural latitude")

    records: List[Dict] = []
    for pair_id in sorted(matched_ids):
        ann = annual_indexed.loc[pair_id]
        nhw = nhw_indexed.loc[pair_id]
        hw = hw_indexed.loc[pair_id]
        state_values = np.array(
            [nhw["dTmean"], nhw["dAmp1"], hw["dTmean"], hw["dAmp1"]],
            dtype=float,
        )
        if not np.isfinite(state_values).all():
            attrition.append(
                {
                    "pair_id": pair_id,
                    "stage": "state_complete_case",
                    "reason": "nonfinite_state_coordinate",
                }
            )
            continue
        coordinates = np.array([ann[lon_u], ann[lat_u], ann[lon_r], ann[lat_r]], dtype=float)
        if not np.isfinite(coordinates).all():
            attrition.append(
                {
                    "pair_id": pair_id,
                    "stage": "spatial_metadata",
                    "reason": "nonfinite_pair_coordinates",
                }
            )
            continue

        r_mean = float(hw["dTmean"] - nhw["dTmean"])
        r_amp = float(hw["dAmp1"] - nhw["dAmp1"])
        r_day_observed = float(hw["dTx"] - nhw["dTx"])
        r_night_observed = float(hw["dTn"] - nhw["dTn"])
        r_day_reconstructed = r_mean + r_amp
        r_night_reconstructed = r_mean - r_amp
        decomposition_valid = bool(
            np.isfinite(
                [r_day_observed, r_night_observed,
                 r_day_reconstructed, r_night_reconstructed]
            ).all()
        )
        if not decomposition_valid:
            attrition.append(
                {
                    "pair_id": pair_id,
                    "stage": "day_night_reconstruction",
                    "reason": "nonfinite_observed_day_or_night_response",
                }
            )

        records.append(
            {
                "pair_id": pair_id,
                "annual_group": str(ann["annual_group"]),
                "lon_urban": float(ann[lon_u]),
                "lat_urban": float(ann[lat_u]),
                "lon_rural": float(ann[lon_r]),
                "lat_rural": float(ann[lat_r]),
                "delta_Tmean_NHW": float(nhw["dTmean"]),
                "delta_Amp_NHW": float(nhw["dAmp1"]),
                "delta_Tmean_HW": float(hw["dTmean"]),
                "delta_Amp_HW": float(hw["dAmp1"]),
                "R_mean": r_mean,
                "R_Amp": r_amp,
                "rho": float(np.hypot(r_mean, r_amp)),
                "theta_deg": float(np.degrees(np.arctan2(r_amp, r_mean))),
                "D_orthogonal": r_mean - r_amp,
                "R_day_observed": r_day_observed,
                "R_night_observed": r_night_observed,
                "R_day_reconstructed": r_day_reconstructed,
                "R_night_reconstructed": r_night_reconstructed,
                "day_residual": r_day_observed - r_day_reconstructed,
                "night_residual": r_night_observed - r_night_reconstructed,
                "reconstruction_valid": int(decomposition_valid),
            }
        )

    state = pd.DataFrame(records)
    if state.empty:
        raise ValueError("Matched state cohort is empty after complete-case checks.")
    if not all(state["annual_group"].eq(group).any() for group in GROUP_ORDER):
        raise ValueError("Both UHI and UCI groups are required for Figure 3.")
    reconstruction = state[state["reconstruction_valid"].eq(1)]
    if len(reconstruction) < 20:
        raise ValueError("Fewer than 20 pairs have complete observed day/night responses.")
    if not all(reconstruction["annual_group"].eq(group).any() for group in GROUP_ORDER):
        raise ValueError("Both UHI and UCI groups require valid day/night reconstruction data.")
    attrition_table = pd.DataFrame(attrition, columns=["pair_id", "stage", "reason"])
    return state, selected, attrition_table


def assign_spatial_blocks(state: pd.DataFrame, block_deg: float) -> pd.DataFrame:
    if not np.isfinite(block_deg) or block_deg <= 0 or not np.isclose(180 % block_deg, 0):
        raise ValueError("block_deg must be a positive divisor of 180.")
    out = state.copy()
    lon_mid = circular_lon_midpoint(
        out["lon_urban"].to_numpy(float), out["lon_rural"].to_numpy(float)
    )
    lat_mid = 0.5 * (
        out["lat_urban"].to_numpy(float) + out["lat_rural"].to_numpy(float)
    )
    lon_mid = ((lon_mid + 180.0) % 360.0) - 180.0
    lat_mid = np.clip(lat_mid, -90.0, np.nextafter(90.0, -np.inf))
    lon_index = np.floor((lon_mid + 180.0) / block_deg).astype(int)
    lat_index = np.floor((lat_mid + 90.0) / block_deg).astype(int)
    out["pair_lon_mid"] = lon_mid
    out["pair_lat_mid"] = lat_mid
    out["block_id"] = [
        f"lat{lat:02d}_lon{lon:02d}" for lat, lon in zip(lat_index, lon_index)
    ]
    return out


def dft_coefficients(curve: Sequence[float]) -> np.ndarray:
    values = np.asarray(curve, dtype=float)
    if values.shape != (24,) or not np.isfinite(values).all():
        raise ValueError("A complete finite 24-hour curve is required.")
    return np.fft.fft(values) / 24.0


def circular_difference_degrees(a: float, b: float) -> float:
    return float((a - b + 180.0) % 360.0 - 180.0)


def validate_fft_consistency(selected: pd.DataFrame, state_ids: set[str]) -> Dict:
    rows = selected[
        selected["pair_id"].isin(state_ids)
        & selected["period"].isin((PERIOD_NHW, PERIOD_HW))
    ]
    amplitude_errors: List[float] = []
    phase_errors: List[float] = []
    contrast_errors: List[float] = []
    checked_curves = 0
    for _, row in rows.iterrows():
        for prefix in ("urban", "rural"):
            curve = row[curve_columns(prefix)].to_numpy(float)
            if not np.isfinite(curve).all():
                continue
            coefficient = dft_coefficients(curve)[1]
            reconstructed_amplitude = 2.0 * abs(coefficient)
            reconstructed_phase = float(np.angle(coefficient))
            stored_amplitude = float(row[f"{prefix}_Amp1"])
            stored_phase = float(row[f"{prefix}_Phase1"])
            amplitude_errors.append(abs(reconstructed_amplitude - stored_amplitude))
            phase_errors.append(
                abs(
                    np.deg2rad(
                        circular_difference_degrees(
                            np.degrees(reconstructed_phase), np.degrees(stored_phase)
                        )
                    )
                )
            )
            checked_curves += 1
        if np.isfinite(row["dAmp1"]):
            contrast_errors.append(
                abs(
                    float(row["dAmp1"])
                    - (float(row["urban_Amp1"]) - float(row["rural_Amp1"]))
                )
            )
    diagnostics = {
        "pair_period_rows_checked": int(len(rows)),
        "complete_curves_checked": int(checked_curves),
        "max_amplitude_abs_error": float(np.max(amplitude_errors)) if amplitude_errors else np.nan,
        "max_phase_abs_error_rad": float(np.max(phase_errors)) if phase_errors else np.nan,
        "max_dAmp1_identity_abs_error": float(np.max(contrast_errors)) if contrast_errors else np.nan,
    }
    for key in (
        "max_amplitude_abs_error",
        "max_phase_abs_error_rad",
        "max_dAmp1_identity_abs_error",
    ):
        value = diagnostics[key]
        if np.isfinite(value) and value > 1.0e-5:
            raise ValueError(f"FFT/stored-metric consistency failed: {diagnostics}")
    return diagnostics


def make_panel_a_binned_flow(
    state: pd.DataFrame, bins: int = GRID_BINS, min_count: int = GRID_MIN_COUNT
) -> pd.DataFrame:
    """Frozen grid algorithm: equal-width bins of the NHW start state."""
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
        subset = state[state["annual_group"].eq(group)].copy()
        subset["xb"] = pd.cut(
            subset["delta_Tmean_NHW"], x_edges, labels=False, include_lowest=True
        )
        subset["yb"] = pd.cut(
            subset["delta_Amp_NHW"], y_edges, labels=False, include_lowest=True
        )
        summary = (
            subset.dropna(subset=["xb", "yb"])
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


def build_block_draw_counts(
    state: pd.DataFrame, n_bootstrap: int, seed: int
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    blocks = sorted(state["block_id"].astype(str).unique().tolist())
    if len(blocks) < 2:
        raise ValueError("At least two occupied spatial blocks are required.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(blocks), size=(n_bootstrap, len(blocks)))
    counts = np.zeros((n_bootstrap, len(blocks)), dtype=np.uint16)
    row_ids = np.repeat(np.arange(n_bootstrap), len(blocks))
    np.add.at(counts, (row_ids, sampled.ravel()), 1)
    block_index = pd.Categorical(state["block_id"], categories=blocks).codes
    if (block_index < 0).any():
        raise RuntimeError("Spatial-block indexing failed.")
    return blocks, counts, block_index


def bootstrap_mean_vector(
    values: np.ndarray,
    row_mask: np.ndarray,
    block_index: np.ndarray,
    draw_counts: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = row_mask & np.isfinite(values)
    sums = np.bincount(
        block_index[valid], weights=values[valid], minlength=draw_counts.shape[1]
    )
    sizes = np.bincount(block_index[valid], minlength=draw_counts.shape[1]).astype(float)
    numerator = draw_counts @ sums
    denominator = draw_counts @ sizes
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(draw_counts), np.nan, dtype=float),
        where=denominator > 0,
    )


def bootstrap_all(
    state: pd.DataFrame,
    draw_counts: np.ndarray,
    block_index: np.ndarray,
) -> pd.DataFrame:
    output: Dict[str, np.ndarray] = {"replicate": np.arange(len(draw_counts), dtype=int)}
    state_variables = (
        "delta_Tmean_NHW", "delta_Amp_NHW", "delta_Tmean_HW", "delta_Amp_HW",
        "R_mean", "R_Amp",
    )
    reconstruction_variables = (
        "R_day_observed", "R_day_reconstructed", "day_residual",
        "R_night_observed", "R_night_reconstructed", "night_residual",
    )
    for group in GROUP_ORDER:
        group_mask = state["annual_group"].eq(group).to_numpy()
        for variable in state_variables:
            output[f"{group}_{variable}"] = bootstrap_mean_vector(
                state[variable].to_numpy(float), group_mask, block_index, draw_counts
            )
        output[f"{group}_theta_deg"] = np.degrees(
            np.arctan2(output[f"{group}_R_Amp"], output[f"{group}_R_mean"])
        )
        reconstruction_mask = group_mask & state["reconstruction_valid"].eq(1).to_numpy()
        for variable in reconstruction_variables:
            output[f"{group}_{variable}"] = bootstrap_mean_vector(
                state[variable].to_numpy(float),
                reconstruction_mask,
                block_index,
                draw_counts,
            )
    return pd.DataFrame(output)


def percentile_interval(values: Sequence[float]) -> Tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan, np.nan, 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(len(finite))


def unwrap_bootstrap_angles(values: np.ndarray, point: float) -> np.ndarray:
    return point + ((values - point + 180.0) % 360.0 - 180.0)


def state_group_summary(state: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    variables = (
        "delta_Tmean_NHW", "delta_Amp_NHW", "delta_Tmean_HW", "delta_Amp_HW",
        "R_mean", "R_Amp",
    )
    for group in GROUP_ORDER:
        subset = state[state["annual_group"].eq(group)]
        row: Dict[str, float | int | str] = {
            "annual_group": group,
            "n_pairs": int(len(subset)),
            "n_blocks": int(subset["block_id"].nunique()),
        }
        for variable in variables:
            point = float(subset[variable].mean())
            low, high, valid = percentile_interval(bootstrap[f"{group}_{variable}"])
            row[variable] = point
            row[f"{variable}_ci_low"] = low
            row[f"{variable}_ci_high"] = high
            row[f"{variable}_bootstrap_valid"] = valid
        theta = float(np.degrees(np.arctan2(row["R_Amp"], row["R_mean"])))
        theta_values = bootstrap[f"{group}_theta_deg"].dropna().to_numpy(float)
        theta_values = unwrap_bootstrap_angles(theta_values, theta)
        low, high, valid = percentile_interval(theta_values)
        row["theta_deg"] = theta
        row["theta_ci_low"] = low
        row["theta_ci_high"] = high
        row["theta_bootstrap_valid"] = valid
        row["rho_group_mean_vector"] = float(np.hypot(row["R_mean"], row["R_Amp"]))
        rows.append(row)
    return pd.DataFrame(rows)


def reconstruction_summary(state: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    valid = state[state["reconstruction_valid"].eq(1)].copy()
    rows: List[Dict] = []
    for group in GROUP_ORDER:
        subset = valid[valid["annual_group"].eq(group)]
        for target in ("day", "night"):
            observed = f"R_{target}_observed"
            reconstructed = f"R_{target}_reconstructed"
            residual = f"{target}_residual"
            row: Dict[str, float | int | str] = {
                "annual_group": group,
                "target": target,
                "target_label": "Day" if target == "day" else "Night",
                "n_pairs": int(len(subset)),
                "n_blocks": int(subset["block_id"].nunique()),
                "bootstrap_requested": int(len(bootstrap)),
                "low_order_coordinate": (
                    "R_mean+R_Amp" if target == "day" else "R_mean-R_Amp"
                ),
            }
            for output_name, variable in (
                ("observed", observed),
                ("reconstructed", reconstructed),
                ("residual", residual),
            ):
                point = float(subset[variable].mean())
                low, high, n_valid = percentile_interval(bootstrap[f"{group}_{variable}"])
                row[f"{output_name}_estimate"] = point
                row[f"{output_name}_ci_low"] = low
                row[f"{output_name}_ci_high"] = high
                row[f"{output_name}_bootstrap_valid"] = n_valid
            identity_error = abs(
                row["observed_estimate"]
                - row["reconstructed_estimate"]
                - row["residual_estimate"]
            )
            if identity_error > 1.0e-10:
                raise RuntimeError("Observed = reconstructed + residual identity failed.")
            rows.append(row)
    return pd.DataFrame(rows)


def rom_pi_sensitivity_table() -> pd.DataFrame:
    pi = np.logspace(-2, 2, 700)
    omega = TWO_PI / 24.0
    return pd.DataFrame(
        {
            "Pi_1": pi,
            "equivalent_tau_h": pi / omega,
            "removal_weight_w_lambda": 1.0 / (1.0 + pi**2),
            "storage_weight_w_C": pi**2 / (1.0 + pi**2),
            "timing_weight_w_phi": pi / (1.0 + pi**2),
        }
    )


def jacobian_direction_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "channel": "persistent_mean_loading",
                "label": "F0 up",
                "sign_R_mean": "+",
                "sign_R_Amp": "0",
                "angle_min_deg": 0.0,
                "angle_max_deg": 0.0,
                "identification": "sign_direction_only",
            },
            {
                "channel": "relative_amplitude_damping",
                "label": "F1 down and/or C up",
                "sign_R_mean": "0",
                "sign_R_Amp": "-",
                "angle_min_deg": -90.0,
                "angle_max_deg": -90.0,
                "identification": "sign_direction_only_F1_C_not_separable",
            },
            {
                "channel": "weaker_effective_removal",
                "label": "lambda down",
                "sign_R_mean": "+",
                "sign_R_Amp": "+",
                "angle_min_deg": 0.0,
                "angle_max_deg": 90.0,
                "identification": "quadrant_only_slope_not_identified",
            },
        ]
    )


def diagnostics_table(
    state: pd.DataFrame,
    bootstrap: pd.DataFrame,
    fft_diagnostics: Mapping,
) -> pd.DataFrame:
    reconstruction = state[state["reconstruction_valid"].eq(1)]
    values = [
        ("state_pairs", len(state)),
        ("reconstruction_pairs", len(reconstruction)),
        ("UHI_state_pairs", int(state["annual_group"].eq("UHI").sum())),
        ("UCI_state_pairs", int(state["annual_group"].eq("UCI").sum())),
        ("occupied_blocks_state", state["block_id"].nunique()),
        ("occupied_blocks_reconstruction", reconstruction["block_id"].nunique()),
        ("bootstrap_requested", len(bootstrap)),
        ("max_reconstruction_identity_error", float(np.max(np.abs(
            state.loc[state["reconstruction_valid"].eq(1), "R_day_observed"]
            - state.loc[state["reconstruction_valid"].eq(1), "R_day_reconstructed"]
            - state.loc[state["reconstruction_valid"].eq(1), "day_residual"]
        )))),
    ]
    values.extend((f"fft_{key}", value) for key, value in fft_diagnostics.items())
    return pd.DataFrame(values, columns=["diagnostic", "value"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the revised Figure 3 state-theory-reconstruction analysis."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--hw-method", default="percentile")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--block-deg", type=float, default=DEFAULT_BLOCK_DEG)
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Permit 200-4999 bootstrap replicates for code/visual QA only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_mode:
        if args.bootstrap < MIN_TEST_BOOTSTRAP:
            raise ValueError(f"Test mode requires at least {MIN_TEST_BOOTSTRAP} replicates.")
        run_mode = "TEST"
    else:
        if args.bootstrap != DEFAULT_BOOTSTRAP:
            raise ValueError(
                f"Formal analysis requires exactly {DEFAULT_BOOTSTRAP} bootstrap replicates. "
                "Use --test-mode only for code or visual QA."
            )
        run_mode = "FORMAL"

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    input_hash_before = sha256_file(input_path)
    prepare_output_dir(outdir)

    df, input_info = load_and_validate_input(input_path, args.hw_method)
    state, selected, attrition = build_matched_state_cohort(df)
    state = assign_spatial_blocks(state, args.block_deg)
    fft_diagnostics = validate_fft_consistency(selected, set(state["pair_id"].astype(str)))
    blocks, draw_counts, block_index = build_block_draw_counts(
        state, args.bootstrap, args.seed
    )
    bootstrap = bootstrap_all(state, draw_counts, block_index)
    group_summary = state_group_summary(state, bootstrap)
    binned_flow = make_panel_a_binned_flow(state)
    reconstruction = reconstruction_summary(state, bootstrap)
    rom_pi = rom_pi_sensitivity_table()
    jacobian = jacobian_direction_table()
    diagnostics = diagnostics_table(state, bootstrap, fft_diagnostics)

    output_frames: Dict[str, pd.DataFrame] = {
        "fig3_pair_level_state_reconstruction.csv": state,
        "fig3_attrition_table.csv": attrition,
        "fig3_group_state_summary.csv": group_summary,
        "fig3_panel_a_binned_flow.csv": binned_flow,
        "fig3_panel_b_rom_pi_sensitivity.csv": rom_pi,
        "fig3_panel_c_jacobian_directions.csv": jacobian,
        "fig3_panel_d_day_night_reconstruction.csv": reconstruction,
        "fig3_analysis_diagnostics.csv": diagnostics,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(outdir / filename, index=False)
    bootstrap_filename = "fig3_bootstrap_replicates.csv.gz"
    bootstrap.to_csv(outdir / bootstrap_filename, index=False, compression="gzip")

    input_hash_after = sha256_file(input_path)
    if input_hash_after != input_hash_before:
        raise RuntimeError("Input SHA-256 changed during analysis; outputs are not frozen.")

    output_names = list(output_frames) + [bootstrap_filename]
    output_hashes = {name: sha256_file(outdir / name) for name in output_names}
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": run_mode,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_path": str(input_path),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "input_info": input_info,
        "frozen_analysis": {
            "hw_method": args.hw_method,
            "periods": list(PERIOD_ORDER),
            "annual_group_definition": "UHI if annual dTx>=0; UCI if annual dTx<0",
            "state_coordinates": ["R_mean", "R_Amp"],
            "response_magnitude_name": "rho",
            "D_reserved_definition": "R_mean-R_Amp",
            "day_reconstruction": "R_mean+R_Amp",
            "night_reconstruction": "R_mean-R_Amp",
            "residual_interpretation": "phase/shape/higher-harmonic remainder",
            "spatial_block_degrees": args.block_deg,
            "spatial_block_coordinate": "urban-rural pair midpoint",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "shared_bootstrap_draws": True,
            "panel_a_grid_bins": GRID_BINS,
            "panel_a_grid_min_count": GRID_MIN_COUNT,
            "theory_identification_limit": (
                "No city-specific C, lambda, Pi, or forcing component is inferred."
            ),
        },
        "cohort": {
            "state_pairs": int(len(state)),
            "reconstruction_pairs": int(state["reconstruction_valid"].sum()),
            "UHI_pairs": int(state["annual_group"].eq("UHI").sum()),
            "UCI_pairs": int(state["annual_group"].eq("UCI").sum()),
            "occupied_blocks": int(len(blocks)),
        },
        "fft_consistency": fft_diagnostics,
        "output_sha256": output_hashes,
    }
    manifest_path = outdir / "fig3_analysis_manifest.json"
    write_json(manifest_path, manifest)
    success = {
        "status": "SUCCESS",
        "script_version": SCRIPT_VERSION,
        "run_mode": run_mode,
        "state_pairs": int(len(state)),
        "reconstruction_pairs": int(state["reconstruction_valid"].sum()),
        "bootstrap_replicates": int(args.bootstrap),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(outdir / "_SUCCESS.json", success)

    print(f"SUCCESS: {outdir}")
    print(
        f"mode={run_mode}; state n={len(state)}; "
        f"reconstruction n={int(state['reconstruction_valid'].sum())}; "
        f"blocks={len(blocks)}; bootstrap={args.bootstrap}"
    )


if __name__ == "__main__":
    main()
