#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze figure3 dynamics.

Original scientific calculations retained; see README.md for inputs and execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


SCRIPT_VERSION = "figure3_dynamic_mechanism_analysis"
STATE_VERSION = "figure3_state_theory_reconstruction_analysis"
TIMING_VERSION = "figure3_state_process_timing_analysis"

DEFAULT_BOOTSTRAP = 5000
MIN_TEST_BOOTSTRAP = 200
DEFAULT_SEED = 20260813
DEFAULT_MATCH_MAX_DAYS = 30
DEFAULT_WORKERS = min(6, max(1, (os.cpu_count() or 2) - 2))

YEARS = tuple(range(2015, 2025))
PHASES = (
    "pre_m2",
    "pre_m1",
    "onset",
    "final_hw",
    "recovery_p1",
    "recovery_p2",
    "recovery_p3",
)
PHASE_LABELS = {
    "pre_m2": "Pre -2 d",
    "pre_m1": "Pre -1 d",
    "onset": "HW onset",
    "final_hw": "Final HW day",
    "recovery_p1": "Recovery +1 d",
    "recovery_p2": "Recovery +2 d",
    "recovery_p3": "Recovery +3 d",
}
PHASE_OFFSETS = {
    "pre_m2": ("start", -2),
    "pre_m1": ("start", -1),
    "onset": ("start", 0),
    "final_hw": ("end", 0),
    "recovery_p1": ("end", 1),
    "recovery_p2": ("end", 2),
    "recovery_p3": ("end", 3),
}
GROUP_ORDER = ("UHI", "UCI")
COMPONENTS = ("R_mean", "R_Amp")

TEMP_QC_NIGHT_HOURS = tuple(range(0, 8))
TEMP_QC_COMPLEMENT_HOURS = tuple(range(8, 24))
TEMP_QC_MIN_RAW_WINDOW_DAY_FRAC = 0.50
TEMP_QC_MIN_RAW_HOUR_BINS_PER_WINDOW = 1
TEMP_QC_MIN_SUPPORTED_HOUR_BINS = 8
EPS = 1.0e-12

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "outputs/analysis"
DEFAULT_STATE_DIR = ANALYSIS_ROOT / "figure3_state"
DEFAULT_TIMING_DIR = ANALYSIS_ROOT / "figure3_timing"
DEFAULT_DAILY_FLAGS = (
    ANALYSIS_ROOT
    / "main_multiyear/robustness_percentile/daily_heatwave_flags.csv"
)
DEFAULT_STATION_YEARS = ANALYSIS_ROOT / "main_multiyear/station_valid_years.csv"
DEFAULT_ISD_BASE = PROJECT_ROOT / "data/isd_lite"
DEFAULT_OUTDIR = ANALYSIS_ROOT / "figure3_dynamic_mechanism"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def composite_sha256(records: Sequence[Tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for path, file_hash in sorted(records):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if not isinstance(value, (str, bool)) and pd.isna(value):
        return None
    return value


def write_json(path: Path, obj: Mapping) -> None:
    path.write_text(
        json.dumps(json_ready(dict(obj)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_bundle(
    directory: Path,
    expected_version: str,
    required_files: Sequence[str],
) -> Tuple[Dict, Dict]:
    missing = [name for name in required_files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete bundle {directory}; missing: {missing}")
    manifest_path = directory / "fig3_analysis_manifest.json"
    success_path = directory / "_SUCCESS.json"
    manifest = load_json(manifest_path)
    success = load_json(success_path)
    if success.get("status") != "SUCCESS":
        raise ValueError(f"Bundle is not successful: {directory}")
    if manifest.get("script_version") != expected_version:
        raise ValueError(
            f"Unexpected bundle version in {directory}: "
            f"{manifest.get('script_version')!r}; expected {expected_version!r}."
        )
    if success.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"Manifest SHA-256 mismatch: {directory}")
    for name, expected_hash in manifest.get("output_sha256", {}).items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-listed file missing: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Manifest-listed file changed: {path}")
    return manifest, success


def validate_and_load_frozen_sources(
    state_dir: Path,
    timing_dir: Path,
    test_mode: bool,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    state_required = (
        "fig3_pair_level_state_reconstruction.csv",
        "fig3_group_state_summary.csv",
        "fig3_panel_a_binned_flow.csv",
        "fig3_panel_b_rom_pi_sensitivity.csv",
        "fig3_panel_c_jacobian_directions.csv",
        "fig3_panel_d_day_night_reconstruction.csv",
        "fig3_bootstrap_replicates.csv.gz",
        "fig3_analysis_manifest.json",
        "_SUCCESS.json",
    )
    timing_required = (
        "fig3_pair_level_state_timing.csv",
        "fig3_panel_d_coefficients.csv",
        "fig3_panel_d_fwl_pair_level.csv",
        "fig3_panel_d_fwl_curve.csv",
        "fig3_panel_d_fwl_bins.csv",
        "fig3_analysis_manifest.json",
        "_SUCCESS.json",
    )
    state_manifest, state_success = validate_bundle(
        state_dir, STATE_VERSION, state_required
    )
    timing_manifest, timing_success = validate_bundle(
        timing_dir, TIMING_VERSION, timing_required
    )
    if not test_mode:
        if str(state_manifest.get("run_mode", "")).upper() != "FORMAL":
            raise ValueError("The state bundle is not marked FORMAL.")
        if int(state_success.get("bootstrap_replicates", 0)) != DEFAULT_BOOTSTRAP:
            raise ValueError("The state bundle does not contain 5000 formal replicates.")

    tables = {
        "state_pair": pd.read_csv(state_dir / "fig3_pair_level_state_reconstruction.csv"),
        "state_group": pd.read_csv(state_dir / "fig3_group_state_summary.csv"),
        "state_flow": pd.read_csv(state_dir / "fig3_panel_a_binned_flow.csv"),
        "theory_pi": pd.read_csv(state_dir / "fig3_panel_b_rom_pi_sensitivity.csv"),
        "jacobian": pd.read_csv(state_dir / "fig3_panel_c_jacobian_directions.csv"),
        "reconstruction_si": pd.read_csv(
            state_dir / "fig3_panel_d_day_night_reconstruction.csv"
        ),
        "state_bootstrap": pd.read_csv(
            state_dir / "fig3_bootstrap_replicates.csv.gz"
        ),
        "timing_pair_state": pd.read_csv(
            timing_dir / "fig3_pair_level_state_timing.csv"
        ),
        "timing_coefficient": pd.read_csv(
            timing_dir / "fig3_panel_d_coefficients.csv"
        ),
        "timing_fwl_pair": pd.read_csv(
            timing_dir / "fig3_panel_d_fwl_pair_level.csv"
        ),
        "timing_fwl_curve": pd.read_csv(
            timing_dir / "fig3_panel_d_fwl_curve.csv"
        ),
        "timing_fwl_bins": pd.read_csv(
            timing_dir / "fig3_panel_d_fwl_bins.csv"
        ),
    }
    state_ids = set(tables["state_pair"]["pair_id"].astype(str))
    timing_ids = set(tables["timing_pair_state"]["pair_id"].astype(str))
    if state_ids != timing_ids:
        raise ValueError(
            "State and timing bundles use different pair cohorts; mixing stopped. "
            f"state_only={len(state_ids - timing_ids)}, "
            f"timing_only={len(timing_ids - state_ids)}"
        )
    if tables["state_pair"]["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id values in the state bundle.")
    coefficient = tables["timing_coefficient"]
    primary = coefficient[coefficient["term"].astype(str).eq("amplitude_damping")]
    if len(primary) != 1:
        raise ValueError("Expected one formal amplitude_damping coefficient row.")
    primary_row = primary.iloc[0]
    if int(primary_row["n_pairs"]) != len(state_ids):
        raise ValueError("The timing coefficient and state cohort have different n.")
    if not test_mode and int(primary_row["bootstrap_requested"]) != DEFAULT_BOOTSTRAP:
        raise ValueError("The timing coefficient is not based on 5000 replicates.")

    provenance = {
        "state_directory": str(state_dir),
        "state_manifest_sha256": sha256_file(state_dir / "fig3_analysis_manifest.json"),
        "state_run_mode": state_manifest.get("run_mode"),
        "timing_directory": str(timing_dir),
        "timing_manifest_sha256": sha256_file(timing_dir / "fig3_analysis_manifest.json"),
        "state_pairs": len(state_ids),
        "timing_beta_A": float(primary_row["estimate"]),
        "timing_beta_A_ci_low": float(primary_row["ci_low"]),
        "timing_beta_A_ci_high": float(primary_row["ci_high"]),
    }
    return tables, provenance


def wrap_degrees(value: np.ndarray | float):
    array = np.asarray(value, dtype=float)
    wrapped = (array + 180.0) % 360.0 - 180.0
    return float(wrapped) if np.ndim(value) == 0 else wrapped


def group_angle_contrast(
    group_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = group_summary.set_index("annual_group")
    point = wrap_degrees(
        float(rows.loc["UHI", "theta_deg"]) - float(rows.loc["UCI", "theta_deg"])
    )
    required = {"UHI_theta_deg", "UCI_theta_deg"}
    if not required.issubset(bootstrap.columns):
        raise ValueError("State bootstrap table lacks group angle replicates.")
    values = wrap_degrees(
        bootstrap["UHI_theta_deg"].to_numpy(float)
        - bootstrap["UCI_theta_deg"].to_numpy(float)
    )
    values = point + wrap_degrees(values - point)
    values = values[np.isfinite(values)]
    low, high = np.quantile(values, [0.025, 0.975])
    return pd.DataFrame(
        [
            {
                "contrast": "UHI_minus_UCI_theta",
                "estimate_deg": point,
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_valid": int(len(values)),
            }
        ]
    )


def parse_years(value: object) -> List[int]:
    if pd.isna(value):
        return []
    output: List[int] = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            output.append(int(float(token)))
        except ValueError:
            continue
    return sorted(set(output))


def load_station_year_map(path: Path, state_ids: set[str]) -> Dict[str, Dict[str, List[int]]]:
    table = pd.read_csv(path, low_memory=False)
    required = {"pair_id", "station_type", "valid_years_all"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Station-year table missing: {sorted(missing)}")
    table["pair_id"] = table["pair_id"].astype(str)
    table = table[table["pair_id"].isin(state_ids)].copy()
    if table.duplicated(["pair_id", "station_type"]).any():
        raise ValueError("Duplicate pair/station_type rows in station_valid_years.csv")
    result: Dict[str, Dict[str, List[int]]] = {}
    for row in table.itertuples(index=False):
        result.setdefault(str(row.pair_id), {})[str(row.station_type)] = parse_years(
            row.valid_years_all
        )
    missing_pairs = [
        pair_id
        for pair_id in state_ids
        if not {"urban", "rural"}.issubset(result.get(pair_id, {}))
    ]
    if missing_pairs:
        raise ValueError(
            f"Station-year metadata missing for {len(missing_pairs)} state pairs."
        )
    return result


def consecutive_runs(dates: Sequence[pd.Timestamp]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted(pd.Timestamp(value).normalize() for value in dates)
    if not dates:
        return []
    runs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = dates[0]
    for current in dates[1:]:
        if (current - previous).days != 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return runs


def event_phase_date(start: pd.Timestamp, end: pd.Timestamp, phase: str) -> pd.Timestamp:
    anchor, offset = PHASE_OFFSETS[phase]
    base = start if anchor == "start" else end
    return base + pd.Timedelta(days=offset)


def valid_flag_window(
    lookup: Mapping[pd.Timestamp, Mapping],
    start: pd.Timestamp,
    duration: int,
    warm_year: int,
    actual_event: bool,
) -> bool:
    end = start + pd.Timedelta(days=duration - 1)
    extended = pd.date_range(start - pd.Timedelta(days=2), end + pd.Timedelta(days=3))
    for date in extended:
        row = lookup.get(date.normalize())
        if row is None:
            return False
        if int(row["is_warm_season"]) != 1:
            return False
        if int(row["warm_season_year"]) != int(warm_year):
            return False
    event_dates = pd.date_range(start, end)
    prepost_dates = [
        start - pd.Timedelta(days=2),
        start - pd.Timedelta(days=1),
        end + pd.Timedelta(days=1),
        end + pd.Timedelta(days=2),
        end + pd.Timedelta(days=3),
    ]
    if actual_event:
        return all(int(lookup[d.normalize()]["hw"]) == 1 for d in event_dates) and all(
            int(lookup[d.normalize()]["hw"]) == 0 for d in prepost_dates
        )
    return all(int(lookup[d.normalize()]["hw"]) == 0 for d in extended)


def match_events_for_pair(
    flags: pd.DataFrame,
    valid_years: set[int],
    match_max_days: int,
) -> Tuple[pd.DataFrame, List[Dict]]:
    attrition: List[Dict] = []
    data = flags.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"])
    data = data[data["date"].dt.year.isin(valid_years)].copy()
    data["hw"] = pd.to_numeric(
        data["hw_flag_percentile_warm_season"], errors="coerce"
    ).fillna(0).astype(int)
    data["is_warm_season"] = pd.to_numeric(
        data["is_warm_season"], errors="coerce"
    ).fillna(0).astype(int)
    data["warm_season_year"] = pd.to_numeric(
        data["warm_season_year"], errors="coerce"
    )
    if data.duplicated("date").any():
        raise ValueError(f"Duplicate daily flags for pair {data['pair_id'].iloc[0]}")
    lookup = {
        row.date: {
            "hw": row.hw,
            "is_warm_season": row.is_warm_season,
            "warm_season_year": row.warm_season_year,
        }
        for row in data.itertuples(index=False)
    }
    event_rows: List[Dict] = []
    warm = data[data["is_warm_season"].eq(1)]
    for warm_year, subset in warm.groupby("warm_season_year"):
        hw_dates = subset.loc[subset["hw"].eq(1), "date"].tolist()
        for start, end in consecutive_runs(hw_dates):
            duration = int((end - start).days + 1)
            if duration < 3:
                continue
            if not valid_flag_window(lookup, start, duration, int(warm_year), True):
                attrition.append(
                    {
                        "stage": "event_boundary_support",
                        "reason": "pre_or_recovery_window_not_clean_NHW",
                        "event_start": start.strftime("%Y-%m-%d"),
                    }
                )
                continue
            event_rows.append(
                {
                    "warm_season_year": int(warm_year),
                    "start": start,
                    "end": end,
                    "duration_days": duration,
                }
            )
    if not event_rows:
        return pd.DataFrame(), attrition

    events = pd.DataFrame(event_rows)
    # A segment can be NHW by the binary flag while still lying inside the
    # pre-event or post-event recovery window of another qualifying heatwave.
    # Freeze an exclusion mask before matching so the pseudo-event control is
    # not contaminated by any observed -2 through +3 day event trajectory.
    protected_event_dates: set[pd.Timestamp] = set()
    for event in events.itertuples(index=False):
        protected_event_dates.update(
            pd.date_range(
                event.start - pd.Timedelta(days=2),
                event.end + pd.Timedelta(days=3),
            ).normalize()
        )
    matches: List[Dict] = []
    for (warm_year, duration), event_group in events.groupby(
        ["warm_season_year", "duration_days"], sort=True
    ):
        season_dates = sorted(
            data.loc[
                data["warm_season_year"].eq(warm_year)
                & data["is_warm_season"].eq(1),
                "date",
            ].tolist()
        )
        candidates = []
        for date in season_dates:
            if not valid_flag_window(
                lookup, date, int(duration), int(warm_year), False
            ):
                continue
            candidate_end = date + pd.Timedelta(days=int(duration) - 1)
            candidate_extended = set(
                pd.date_range(
                    date - pd.Timedelta(days=2),
                    candidate_end + pd.Timedelta(days=3),
                ).normalize()
            )
            if candidate_extended & protected_event_dates:
                continue
            candidates.append(date)
        candidates = sorted(set(candidates))
        if not candidates:
            for row in event_group.itertuples(index=False):
                attrition.append(
                    {
                        "stage": "event_matching",
                        "reason": "no_clean_same_year_NHW_segment",
                        "event_start": row.start.strftime("%Y-%m-%d"),
                    }
                )
            continue
        event_list = list(event_group.itertuples(index=False))
        cost = np.full((len(event_list), len(candidates)), 1.0e6, dtype=float)
        for i, event in enumerate(event_list):
            for j, candidate in enumerate(candidates):
                distance = abs((candidate - event.start).days)
                if distance <= match_max_days:
                    cost[i, j] = distance + j * 1.0e-8
        row_indices, col_indices = linear_sum_assignment(cost)
        assigned_events = set()
        for row_index, col_index in zip(row_indices, col_indices):
            if cost[row_index, col_index] >= 1.0e5:
                continue
            event = event_list[row_index]
            pseudo_start = candidates[col_index]
            pseudo_end = pseudo_start + pd.Timedelta(days=int(duration) - 1)
            matches.append(
                {
                    "warm_season_year": int(warm_year),
                    "event_start": event.start,
                    "event_end": event.end,
                    "duration_days": int(duration),
                    "pseudo_start": pseudo_start,
                    "pseudo_end": pseudo_end,
                    "match_distance_days": int(abs((pseudo_start - event.start).days)),
                }
            )
            assigned_events.add(row_index)
        for i, event in enumerate(event_list):
            if i not in assigned_events:
                attrition.append(
                    {
                        "stage": "event_matching",
                        "reason": f"no_unique_NHW_match_within_{match_max_days}d",
                        "event_start": event.start.strftime("%Y-%m-%d"),
                    }
                )
    return pd.DataFrame(matches), attrition


def build_event_matches(
    daily_flags: pd.DataFrame,
    state: pd.DataFrame,
    station_year_map: Mapping[str, Mapping[str, List[int]]],
    match_max_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "pair_id",
        "date",
        "warm_season_year",
        "is_warm_season",
        "hw_flag_percentile_warm_season",
    }
    missing = required.difference(daily_flags.columns)
    if missing:
        raise ValueError(f"Daily heatwave flags missing: {sorted(missing)}")
    daily_flags = daily_flags.copy()
    daily_flags["pair_id"] = daily_flags["pair_id"].astype(str)
    grouped = {key: value for key, value in daily_flags.groupby("pair_id", sort=False)}
    match_frames: List[pd.DataFrame] = []
    attrition_rows: List[Dict] = []
    for pair_id in state["pair_id"].astype(str):
        if pair_id not in grouped:
            attrition_rows.append(
                {"pair_id": pair_id, "stage": "daily_flags", "reason": "pair_absent"}
            )
            continue
        valid_years = set(station_year_map[pair_id]["urban"]) & set(
            station_year_map[pair_id]["rural"]
        )
        matches, attrition = match_events_for_pair(
            grouped[pair_id], valid_years, match_max_days
        )
        for row in attrition:
            attrition_rows.append({"pair_id": pair_id, **row})
        if matches.empty:
            attrition_rows.append(
                {"pair_id": pair_id, "stage": "event_matching", "reason": "no_matched_events"}
            )
            continue
        matches.insert(0, "pair_id", pair_id)
        matches["event_id"] = [
            f"{pair_id}__{start:%Y%m%d}__{end:%Y%m%d}"
            for start, end in zip(matches["event_start"], matches["event_end"])
        ]
        match_frames.append(matches)
    if not match_frames:
        raise ValueError("No event/NHW matches were available for the state cohort.")
    return pd.concat(match_frames, ignore_index=True), pd.DataFrame(attrition_rows)


def parse_pair_id(pair_id: str) -> Tuple[str, str, str, str]:
    pieces = str(pair_id).split("__")
    if len(pieces) != 2:
        raise ValueError(f"Cannot parse pair_id: {pair_id}")
    urban = pieces[0].split("_")
    rural = pieces[1].split("_")
    if len(urban) < 2 or len(rural) < 2:
        raise ValueError(f"Cannot parse pair_id: {pair_id}")
    return urban[0], urban[1], rural[0], rural[1]


def isd_path(base: Path, usaf: str, wban: str, year: int) -> Path:
    return base / str(year) / f"{usaf}-{wban}-{year}.gz"


def read_clean_station_year(path: Path, year: int, longitude: float) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["year", "month", "day", "hour", "temp_C", "dewpoint_C"],
        na_values={"temp_C": -9999, "dewpoint_C": -9999},
        engine="c",
        on_bad_lines="skip",
    )
    raw["temp_C"] = pd.to_numeric(raw["temp_C"], errors="coerce") / 10.0
    raw["datetime"] = pd.to_datetime(
        raw[["year", "month", "day", "hour"]], errors="coerce"
    )
    raw = (
        raw.dropna(subset=["datetime"])
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
    )
    full_index = pd.date_range(
        pd.Timestamp(year, 1, 1, 0), pd.Timestamp(year, 12, 31, 23), freq="1h"
    )
    indexed = raw.set_index("datetime")
    hourly = indexed[["temp_C"]].resample("1h").mean().reindex(full_index)
    observed = (
        indexed[["temp_C"]]
        .resample("1h")
        .count()
        .reindex(full_index)
        .fillna(0)
        .rename(columns={"temp_C": "temp_obs_count"})
    )
    hourly["temp_obs_count"] = observed["temp_obs_count"].astype("int16")
    hourly["temp_C"] = hourly["temp_C"].interpolate(
        method="time", limit=2, limit_area="inside"
    )
    valid_hour_count = hourly["temp_C"].groupby(hourly.index.date).transform(
        lambda series: series.notna().sum()
    )
    invalid_day = valid_hour_count < 8
    hourly.loc[invalid_day, "temp_C"] = np.nan
    hourly.loc[invalid_day, "temp_obs_count"] = 0
    hourly["temp_is_raw_observed"] = hourly["temp_obs_count"].gt(0).astype("int8")
    output = hourly.reset_index().rename(columns={"index": "datetime"})
    output["local_datetime"] = output["datetime"] + pd.Timedelta(hours=longitude / 15.0)
    output["local_hour"] = (
        output["local_datetime"].dt.hour
        + output["local_datetime"].dt.minute / 60.0
    )
    output["local_date"] = output["local_datetime"].dt.normalize()
    return output[["local_datetime", "local_date", "local_hour", "temp_C", "temp_is_raw_observed"]]


def load_station(
    base: Path,
    usaf: str,
    wban: str,
    years: Sequence[int],
    longitude: float,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    frames: List[pd.DataFrame] = []
    hashes: List[Tuple[str, str]] = []
    for year in sorted(set(int(value) for value in years if int(value) in YEARS)):
        path = isd_path(base, usaf, wban, year)
        if not path.is_file():
            continue
        before = sha256_file(path)
        frame = read_clean_station_year(path, year, longitude)
        after = sha256_file(path)
        if before != after:
            raise RuntimeError(f"Raw ISD input changed during reading: {path}")
        hashes.append((str(path.resolve()), before))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), hashes
    return pd.concat(frames, ignore_index=True), hashes


def curve_for_dates(
    station: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
) -> Tuple[np.ndarray | None, Dict]:
    dates = sorted(set(pd.Timestamp(value).normalize() for value in dates))
    selected = station[station["local_date"].isin(dates)].copy()
    diagnostics = {
        "n_requested_dates": len(dates),
        "n_rows": int(len(selected)),
        "supported_hour_bins": 0,
        "raw_night_hour_bins": 0,
        "raw_complement_hour_bins": 0,
        "raw_night_day_frac": 0.0,
        "raw_complement_day_frac": 0.0,
    }
    if not dates or selected.empty:
        return None, diagnostics
    selected = selected.dropna(subset=["temp_C"])
    if selected.empty:
        return None, diagnostics
    selected["h_bin"] = selected["local_hour"].round().astype(int) % 24
    hourly = selected.groupby("h_bin")["temp_C"].mean().reindex(range(24))
    raw = selected["temp_is_raw_observed"].eq(1)
    raw_night = raw & selected["h_bin"].isin(TEMP_QC_NIGHT_HOURS)
    raw_complement = raw & selected["h_bin"].isin(TEMP_QC_COMPLEMENT_HOURS)
    diagnostics.update(
        {
            "supported_hour_bins": int(hourly.notna().sum()),
            "raw_night_hour_bins": int(selected.loc[raw_night, "h_bin"].nunique()),
            "raw_complement_hour_bins": int(
                selected.loc[raw_complement, "h_bin"].nunique()
            ),
            "raw_night_day_frac": float(
                selected.loc[raw_night, "local_date"].nunique() / len(dates)
            ),
            "raw_complement_day_frac": float(
                selected.loc[raw_complement, "local_date"].nunique() / len(dates)
            ),
        }
    )
    passed = (
        diagnostics["supported_hour_bins"] >= TEMP_QC_MIN_SUPPORTED_HOUR_BINS
        and diagnostics["raw_night_hour_bins"] >= TEMP_QC_MIN_RAW_HOUR_BINS_PER_WINDOW
        and diagnostics["raw_complement_hour_bins"] >= TEMP_QC_MIN_RAW_HOUR_BINS_PER_WINDOW
        and diagnostics["raw_night_day_frac"] >= TEMP_QC_MIN_RAW_WINDOW_DAY_FRAC
        and diagnostics["raw_complement_day_frac"] >= TEMP_QC_MIN_RAW_WINDOW_DAY_FRAC
    )
    if not passed:
        return None, diagnostics
    hourly = hourly.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if hourly.isna().any():
        return None, diagnostics
    return hourly.to_numpy(float), diagnostics


def curve_metrics(curve: np.ndarray) -> Tuple[float, float]:
    coefficients = np.fft.fft(np.asarray(curve, dtype=float)) / 24.0
    return float(np.real(coefficients[0])), float(2.0 * abs(coefficients[1]))


def process_pair_dynamic(task: Mapping) -> Dict:
    pair_id = str(task["pair_id"])
    try:
        usaf_u, wban_u, usaf_r, wban_r = parse_pair_id(pair_id)
        years = sorted(set(task["urban_years"]) & set(task["rural_years"]))
        urban, hashes_u = load_station(
            Path(task["isd_base"]), usaf_u, wban_u, years, float(task["lon_urban"])
        )
        rural, hashes_r = load_station(
            Path(task["isd_base"]), usaf_r, wban_r, years, float(task["lon_rural"])
        )
        if urban.empty or rural.empty:
            return {
                "pair_id": pair_id,
                "rows": [],
                "attrition": [{"stage": "raw_hourly", "reason": "station_data_unavailable"}],
                "hashes": hashes_u + hashes_r,
            }
        matches = pd.DataFrame(task["matches"])
        for column in ("event_start", "event_end", "pseudo_start", "pseudo_end"):
            matches[column] = pd.to_datetime(matches[column]).dt.normalize()
        rows: List[Dict] = []
        attrition: List[Dict] = []
        for phase in PHASES:
            actual_dates = [
                event_phase_date(row.event_start, row.event_end, phase)
                for row in matches.itertuples(index=False)
            ]
            pseudo_dates = [
                event_phase_date(row.pseudo_start, row.pseudo_end, phase)
                for row in matches.itertuples(index=False)
            ]
            curves: Dict[str, np.ndarray] = {}
            diagnostics: Dict[str, Dict] = {}
            for label, station, dates in (
                ("actual_urban", urban, actual_dates),
                ("actual_rural", rural, actual_dates),
                ("pseudo_urban", urban, pseudo_dates),
                ("pseudo_rural", rural, pseudo_dates),
            ):
                curve, diag = curve_for_dates(station, dates)
                diagnostics[label] = diag
                if curve is not None:
                    curves[label] = curve
            if len(curves) != 4:
                failed = sorted(set(diagnostics) - set(curves))
                attrition.append(
                    {
                        "stage": "phase_raw_support",
                        "phase": phase,
                        "reason": "failed_" + "+".join(failed),
                    }
                )
                continue
            actual_u_mean, actual_u_amp = curve_metrics(curves["actual_urban"])
            actual_r_mean, actual_r_amp = curve_metrics(curves["actual_rural"])
            pseudo_u_mean, pseudo_u_amp = curve_metrics(curves["pseudo_urban"])
            pseudo_r_mean, pseudo_r_amp = curve_metrics(curves["pseudo_rural"])
            r_mean = (actual_u_mean - actual_r_mean) - (pseudo_u_mean - pseudo_r_mean)
            r_amp = (actual_u_amp - actual_r_amp) - (pseudo_u_amp - pseudo_r_amp)
            rows.append(
                {
                    "pair_id": pair_id,
                    "phase": phase,
                    "phase_order": PHASES.index(phase),
                    "phase_label": PHASE_LABELS[phase],
                    "R_mean": r_mean,
                    "R_Amp": r_amp,
                    "rho": float(np.hypot(r_mean, r_amp)),
                    "theta_deg": float(np.degrees(np.arctan2(r_amp, r_mean))),
                    "n_matched_events": int(len(matches)),
                    "min_raw_night_day_frac": float(
                        min(item["raw_night_day_frac"] for item in diagnostics.values())
                    ),
                    "min_raw_complement_day_frac": float(
                        min(item["raw_complement_day_frac"] for item in diagnostics.values())
                    ),
                    "min_supported_hour_bins": int(
                        min(item["supported_hour_bins"] for item in diagnostics.values())
                    ),
                }
            )
        return {
            "pair_id": pair_id,
            "rows": rows,
            "attrition": attrition,
            "hashes": hashes_u + hashes_r,
        }
    except Exception as error:
        return {
            "pair_id": pair_id,
            "rows": [],
            "attrition": [{"stage": "pair_processing", "reason": repr(error)}],
            "hashes": [],
        }


def run_dynamic_extraction(
    state: pd.DataFrame,
    matches: pd.DataFrame,
    station_year_map: Mapping[str, Mapping[str, List[int]]],
    isd_base: Path,
    workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]]]:
    state_index = state.set_index("pair_id")
    tasks: List[Dict] = []
    for pair_id, group in matches.groupby("pair_id", sort=False):
        row = state_index.loc[str(pair_id)]
        tasks.append(
            {
                "pair_id": str(pair_id),
                "lon_urban": float(row["lon_urban"]),
                "lon_rural": float(row["lon_rural"]),
                "urban_years": station_year_map[str(pair_id)]["urban"],
                "rural_years": station_year_map[str(pair_id)]["rural"],
                "isd_base": str(isd_base),
                "matches": group.to_dict("records"),
            }
        )
    rows: List[Dict] = []
    attrition: List[Dict] = []
    hashes: List[Tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process_pair_dynamic, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.extend(result["rows"])
            hashes.extend(result["hashes"])
            for item in result["attrition"]:
                attrition.append({"pair_id": result["pair_id"], **item})
            if index % 25 == 0 or index == len(futures):
                print(f"  Dynamic extraction: {index}/{len(futures)} pairs")
    if not rows:
        raise ValueError("No pair-phase dynamic metrics passed raw-support QC.")
    dynamic = pd.DataFrame(rows)
    metadata = state[
        ["pair_id", "annual_group", "block_id", "lon_urban", "lat_urban",
         "lon_rural", "lat_rural"]
    ]
    dynamic = dynamic.merge(metadata, on="pair_id", how="left", validate="many_to_one")
    return dynamic, pd.DataFrame(attrition), sorted(set(hashes))


def build_block_draw_counts(
    state: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> Tuple[List[str], np.ndarray]:
    blocks = sorted(state["block_id"].astype(str).unique().tolist())
    if len(blocks) < 2:
        raise ValueError("At least two spatial blocks are required.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(blocks), size=(n_bootstrap, len(blocks)))
    counts = np.zeros((n_bootstrap, len(blocks)), dtype=np.uint16)
    row_ids = np.repeat(np.arange(n_bootstrap), len(blocks))
    np.add.at(counts, (row_ids, sampled.ravel()), 1)
    return blocks, counts


def bootstrap_mean(
    table: pd.DataFrame,
    value: str,
    blocks: Sequence[str],
    draw_counts: np.ndarray,
) -> np.ndarray:
    block_map = {block: index for index, block in enumerate(blocks)}
    data = table[["block_id", value]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return np.full(len(draw_counts), np.nan)
    block_index = data["block_id"].astype(str).map(block_map).to_numpy()
    if pd.isna(block_index).any():
        raise ValueError("Dynamic data contains a spatial block absent from the state cohort.")
    block_index = block_index.astype(int)
    values = data[value].to_numpy(float)
    sums = np.bincount(block_index, weights=values, minlength=len(blocks))
    sizes = np.bincount(block_index, minlength=len(blocks)).astype(float)
    numerator = draw_counts @ sums
    denominator = draw_counts @ sizes
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(draw_counts), np.nan),
        where=denominator > 0,
    )


def summarize_dynamics(
    dynamic: pd.DataFrame,
    blocks: Sequence[str],
    draw_counts: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict] = []
    bootstrap_columns: Dict[str, np.ndarray] = {
        "replicate": np.arange(len(draw_counts), dtype=int)
    }
    for group in GROUP_ORDER:
        for component in COMPONENTS:
            for phase in PHASES:
                subset = dynamic[
                    dynamic["annual_group"].eq(group) & dynamic["phase"].eq(phase)
                ].copy()
                values = bootstrap_mean(subset, component, blocks, draw_counts)
                column = f"{group}_{component}_{phase}"
                bootstrap_columns[column] = values
                finite = values[np.isfinite(values)]
                low, high = (
                    np.quantile(finite, [0.025, 0.975])
                    if len(finite)
                    else (np.nan, np.nan)
                )
                summary_rows.append(
                    {
                        "annual_group": group,
                        "component": component,
                        "phase": phase,
                        "phase_order": PHASES.index(phase),
                        "phase_label": PHASE_LABELS[phase],
                        "estimate": float(subset[component].mean()) if len(subset) else np.nan,
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "n_pairs": int(subset["pair_id"].nunique()),
                        "n_blocks": int(subset["block_id"].nunique()),
                        "n_matched_events": int(subset["n_matched_events"].sum()),
                        "bootstrap_valid": int(len(finite)),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_columns)


def pretrend_tests(
    dynamic: pd.DataFrame,
    blocks: Sequence[str],
    draw_counts: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for group in GROUP_ORDER:
        for component in COMPONENTS:
            subset = dynamic[
                dynamic["annual_group"].eq(group)
                & dynamic["phase"].isin(("pre_m2", "pre_m1"))
            ]
            wide = subset.pivot(index="pair_id", columns="phase", values=component).dropna()
            metadata = (
                subset[["pair_id", "block_id"]]
                .drop_duplicates("pair_id")
                .set_index("pair_id")
            )
            wide = wide.join(metadata, how="inner")
            if len(wide) < 5:
                rows.append(
                    {
                        "annual_group": group,
                        "component": component,
                        "test": "joint_pretrend_zero",
                        "n_pairs": int(len(wide)),
                        "wald_statistic": np.nan,
                        "df": 2,
                        "p_value": np.nan,
                        "status": "insufficient_pairs",
                    }
                )
                continue
            boot_vectors = []
            for phase in ("pre_m2", "pre_m1"):
                boot_vectors.append(
                    bootstrap_mean(
                        wide.reset_index(), phase, blocks, draw_counts
                    )
                )
            boot_matrix = np.column_stack(boot_vectors)
            valid = np.isfinite(boot_matrix).all(axis=1)
            covariance = np.cov(boot_matrix[valid], rowvar=False, ddof=1)
            point = wide[["pre_m2", "pre_m1"]].mean().to_numpy(float)
            try:
                statistic = float(point @ np.linalg.pinv(covariance) @ point)
                p_value = float(chi2.sf(statistic, df=2))
                status = "estimated"
            except Exception:
                statistic = p_value = np.nan
                status = "failed"
            rows.append(
                {
                    "annual_group": group,
                    "component": component,
                    "test": "joint_pretrend_zero",
                    "pre_m2_estimate": float(point[0]),
                    "pre_m1_estimate": float(point[1]),
                    "n_pairs": int(len(wide)),
                    "n_blocks": int(wide["block_id"].nunique()),
                    "wald_statistic": statistic,
                    "df": 2,
                    "p_value": p_value,
                    "bootstrap_valid": int(valid.sum()),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def fit_zero_asymptote_tau(values: np.ndarray) -> Dict:
    values = np.asarray(values, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return {"tau_h": np.nan, "amplitude": np.nan, "r2": np.nan, "at_boundary": 1}
    times = np.array([0.0, 24.0, 48.0, 72.0])
    tau_grid = np.geomspace(1.0, 240.0, 700)
    basis = np.exp(-times[None, :] / tau_grid[:, None])
    denominator = np.sum(basis**2, axis=1)
    amplitudes = basis @ values / denominator
    residual = values[None, :] - amplitudes[:, None] * basis
    sse = np.sum(residual**2, axis=1)
    index = int(np.argmin(sse))
    total = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - float(sse[index]) / total if total > EPS else np.nan
    return {
        "tau_h": float(tau_grid[index]),
        "amplitude": float(amplitudes[index]),
        "r2": r2,
        "at_boundary": int(index in (0, len(tau_grid) - 1)),
    }


def recovery_timescale_summary(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    phases = ("final_hw", "recovery_p1", "recovery_p2", "recovery_p3")
    rows: List[Dict] = []
    for group in GROUP_ORDER:
        for component in COMPONENTS:
            selected = summary[
                summary["annual_group"].eq(group)
                & summary["component"].eq(component)
            ].set_index("phase")
            point_values = np.array([selected.loc[phase, "estimate"] for phase in phases])
            fit = fit_zero_asymptote_tau(point_values)
            boot_values = np.column_stack(
                [bootstrap[f"{group}_{component}_{phase}"].to_numpy(float) for phase in phases]
            )
            tau_values: List[float] = []
            for vector in boot_values:
                boot_fit = fit_zero_asymptote_tau(vector)
                if (
                    np.isfinite(boot_fit["tau_h"])
                    and boot_fit["at_boundary"] == 0
                    and np.isfinite(boot_fit["r2"])
                    and boot_fit["r2"] >= 0.50
                ):
                    tau_values.append(float(boot_fit["tau_h"]))
            valid_fraction = len(tau_values) / len(bootstrap)
            supported = bool(
                fit["at_boundary"] == 0
                and np.isfinite(fit["r2"])
                and fit["r2"] >= 0.50
                and valid_fraction >= 0.80
            )
            low, high = (
                np.quantile(tau_values, [0.025, 0.975])
                if tau_values
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "annual_group": group,
                    "component": component,
                    "model": "zero_asymptote_exponential_effective_recovery",
                    "tau_eff_h": fit["tau_h"] if supported else np.nan,
                    "tau_eff_ci_low": float(low) if supported else np.nan,
                    "tau_eff_ci_high": float(high) if supported else np.nan,
                    "amplitude": fit["amplitude"],
                    "r2": fit["r2"],
                    "point_fit_at_boundary": fit["at_boundary"],
                    "bootstrap_valid": int(len(tau_values)),
                    "bootstrap_valid_fraction": float(valid_fraction),
                    "inference_status": "SUPPORTED" if supported else "NOT_SUPPORTED",
                    "identification_note": "effective recovery only; not C/lambda",
                }
            )
    return pd.DataFrame(rows)


def diagnostics_table(
    state: pd.DataFrame,
    matches: pd.DataFrame,
    dynamic_all: pd.DataFrame,
    dynamic_complete: pd.DataFrame,
    pretrend: pd.DataFrame,
    raw_hashes: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    rows = [
        ("state_pairs", len(state)),
        ("pairs_with_matched_events", matches["pair_id"].nunique()),
        ("matched_events", len(matches)),
        ("pairs_with_any_dynamic_phase", dynamic_all["pair_id"].nunique()),
        ("pairs_complete_all_phases", dynamic_complete["pair_id"].nunique()),
        ("raw_isd_files_hashed", len(raw_hashes)),
        ("raw_isd_composite_sha256", composite_sha256(raw_hashes)),
        (
            "pretrend_tests_p_ge_0_05_fraction",
            float(np.mean(pretrend["p_value"].dropna() >= 0.05))
            if pretrend["p_value"].notna().any()
            else np.nan,
        ),
    ]
    return pd.DataFrame(rows, columns=["diagnostic", "value"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the event-centred dynamic-mechanism Figure 3 bundle."
    )
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--timing-dir", default=str(DEFAULT_TIMING_DIR))
    parser.add_argument("--daily-flags", default=str(DEFAULT_DAILY_FLAGS))
    parser.add_argument("--station-years", default=str(DEFAULT_STATION_YEARS))
    parser.add_argument("--isd-base", default=str(DEFAULT_ISD_BASE))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--match-max-days", type=int, default=DEFAULT_MATCH_MAX_DAYS)
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Permit reduced bootstrap and non-formal upstream bundles for QA only.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="TEST mode only: limit pairs for code/visual QA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_mode:
        if args.bootstrap < MIN_TEST_BOOTSTRAP:
            raise ValueError(f"TEST mode requires at least {MIN_TEST_BOOTSTRAP} replicates.")
        run_mode = "TEST"
    else:
        if args.bootstrap != DEFAULT_BOOTSTRAP:
            raise ValueError("Formal analysis requires exactly 5000 bootstrap replicates.")
        if args.match_max_days != DEFAULT_MATCH_MAX_DAYS:
            raise ValueError("Formal event matching uses the frozen 30-day maximum distance.")
        if args.max_pairs is not None:
            raise ValueError("--max-pairs is available only in TEST mode.")
        run_mode = "FORMAL"
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")

    state_dir = Path(args.state_dir).expanduser().resolve()
    timing_dir = Path(args.timing_dir).expanduser().resolve()
    daily_flags_path = Path(args.daily_flags).expanduser().resolve()
    station_years_path = Path(args.station_years).expanduser().resolve()
    isd_base = Path(args.isd_base).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    for path in (daily_flags_path, station_years_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not isd_base.is_dir():
        raise NotADirectoryError(isd_base)

    source_hashes_before = {
        "daily_heatwave_flags": sha256_file(daily_flags_path),
        "station_valid_years": sha256_file(station_years_path),
    }
    prepare_output_dir(outdir)
    tables, provenance = validate_and_load_frozen_sources(
        state_dir, timing_dir, args.test_mode
    )
    state = tables["state_pair"].copy()
    state["pair_id"] = state["pair_id"].astype(str)
    if args.max_pairs is not None:
        if not args.test_mode:
            raise ValueError("--max-pairs requires --test-mode.")
        state = state.sort_values("pair_id").head(args.max_pairs).copy()
        selected_ids = set(state["pair_id"])
        tables["state_flow"] = tables["state_flow"].copy()
        # The fixed flow table cannot be rebuilt after pair limitation. It is
        # retained only for syntax tests; QA plots should be interpreted as TEST.
    else:
        selected_ids = set(state["pair_id"])

    station_year_map = load_station_year_map(station_years_path, selected_ids)
    daily_flags = pd.read_csv(daily_flags_path, low_memory=False)
    daily_flags = daily_flags[daily_flags["pair_id"].astype(str).isin(selected_ids)].copy()
    matches, matching_attrition = build_event_matches(
        daily_flags, state, station_year_map, args.match_max_days
    )
    dynamic_all, extraction_attrition, raw_hashes = run_dynamic_extraction(
        state,
        matches,
        station_year_map,
        isd_base,
        workers=args.workers,
    )
    attrition = pd.concat(
        [matching_attrition, extraction_attrition], ignore_index=True, sort=False
    )
    phase_counts = dynamic_all.groupby("pair_id")["phase"].nunique()
    complete_pair_ids = set(phase_counts[phase_counts.eq(len(PHASES))].index.astype(str))
    incomplete_pair_ids = sorted(
        set(dynamic_all["pair_id"].astype(str)) - complete_pair_ids
    )
    if incomplete_pair_ids:
        incomplete_rows = pd.DataFrame(
            {
                "pair_id": incomplete_pair_ids,
                "stage": "complete_dynamic_trajectory",
                "reason": "fewer_than_seven_supported_phases",
            }
        )
        attrition = pd.concat([attrition, incomplete_rows], ignore_index=True, sort=False)
    dynamic = dynamic_all[
        dynamic_all["pair_id"].astype(str).isin(complete_pair_ids)
    ].copy()
    if dynamic.empty:
        raise ValueError("No pair has raw-supported data for all seven dynamic phases.")
    missing_groups = [
        group for group in GROUP_ORDER
        if not dynamic["annual_group"].astype(str).eq(group).any()
    ]
    if missing_groups:
        raise ValueError(
            "The complete seven-phase dynamic cohort lacks regimes: "
            f"{missing_groups}"
        )
    blocks, draw_counts = build_block_draw_counts(state, args.bootstrap, args.seed)
    dynamic_summary, dynamic_bootstrap = summarize_dynamics(
        dynamic, blocks, draw_counts
    )
    pretrend = pretrend_tests(dynamic, blocks, draw_counts)
    recovery = recovery_timescale_summary(dynamic_summary, dynamic_bootstrap)
    angle_contrast = group_angle_contrast(
        tables["state_group"], tables["state_bootstrap"]
    )
    diagnostics = diagnostics_table(
        state, matches, dynamic_all, dynamic, pretrend, raw_hashes
    )

    # Consolidated, plot-ready bundle. Existing results are not refitted.
    output_frames: Dict[str, pd.DataFrame] = {
        "fig3_panel_a_pair_state.csv": state,
        "fig3_panel_a_binned_flow.csv": tables["state_flow"],
        "fig3_panel_b_group_summary.csv": tables["state_group"],
        "fig3_panel_b_jacobian.csv": tables["jacobian"],
        "fig3_panel_b_group_angle_contrast.csv": angle_contrast,
        "fig3_panel_c_fwl_pair.csv": tables["timing_fwl_pair"],
        "fig3_panel_c_fwl_curve.csv": tables["timing_fwl_curve"],
        "fig3_panel_c_fwl_bins.csv": tables["timing_fwl_bins"],
        "fig3_panel_c_timing_coefficient.csv": tables["timing_coefficient"],
        "fig3_panel_d_event_matches.csv": matches,
        "fig3_panel_d_pair_phase_state.csv": dynamic,
        "fig3_panel_d_pair_phase_state_all_available.csv": dynamic_all,
        "fig3_panel_d_event_state_summary.csv": dynamic_summary,
        "fig3_panel_d_pretrend_tests.csv": pretrend,
        "fig3_panel_d_recovery_timescales.csv": recovery,
        "fig3_dynamic_attrition.csv": attrition,
        "fig3_dynamic_diagnostics.csv": diagnostics,
        "fig3_standalone_finite_frequency_sensitivity.csv": tables["theory_pi"],
        "fig3_SI_low_order_day_night_reconstruction.csv": tables["reconstruction_si"],
    }
    for name, frame in output_frames.items():
        frame.to_csv(outdir / name, index=False)
    bootstrap_name = "fig3_dynamic_bootstrap_replicates.csv.gz"
    dynamic_bootstrap.to_csv(
        outdir / bootstrap_name, index=False, compression="gzip"
    )

    source_hashes_after = {
        "daily_heatwave_flags": sha256_file(daily_flags_path),
        "station_valid_years": sha256_file(station_years_path),
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A tabular input changed during dynamic analysis.")
    # Every raw file was hashed immediately before and after its read in the
    # worker. The unique path/hash list is retained for a reproducible audit.
    raw_hash_table = pd.DataFrame(raw_hashes, columns=["path", "sha256"])
    raw_hash_name = "fig3_raw_isd_input_sha256.csv.gz"
    raw_hash_table.to_csv(outdir / raw_hash_name, index=False, compression="gzip")

    output_names = list(output_frames) + [bootstrap_name, raw_hash_name]
    output_hashes = {name: sha256_file(outdir / name) for name in output_names}
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": run_mode,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_provenance": provenance,
        "input_paths": {
            "daily_heatwave_flags": str(daily_flags_path),
            "station_valid_years": str(station_years_path),
            "isd_base": str(isd_base),
        },
        "input_sha256_before": source_hashes_before,
        "input_sha256_after": source_hashes_after,
        "raw_isd_files": len(raw_hashes),
        "raw_isd_composite_sha256": composite_sha256(raw_hashes),
        "frozen_analysis": {
            "event_definition": "consecutive frozen percentile HW flags; minimum 3 days",
            "event_boundary_requirement": "2 clean pre-HW and 3 clean post-HW warm-season days",
            "NHW_matching": (
                "same pair, warm-season year and duration; unique minimum-distance "
                "segment; pseudo extended window cannot overlap any qualifying "
                "heatwave -2 through +3 day event window"
            ),
            "maximum_match_distance_days": args.match_max_days,
            "phases": list(PHASES),
            "pair_weighting": "event curves averaged within pair before global inference",
            "dynamic_inference_cohort": (
                "same complete-case pair cohort with raw-supported curves in all seven phases"
            ),
            "raw_support_QC": {
                "night_hours": list(TEMP_QC_NIGHT_HOURS),
                "complement_hours": list(TEMP_QC_COMPLEMENT_HOURS),
                "minimum_raw_window_day_fraction": TEMP_QC_MIN_RAW_WINDOW_DAY_FRAC,
                "minimum_raw_hour_bins_per_window": TEMP_QC_MIN_RAW_HOUR_BINS_PER_WINDOW,
                "minimum_supported_hour_bins": TEMP_QC_MIN_SUPPORTED_HOUR_BINS,
            },
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "spatial_blocks": "frozen 10-degree pair-midpoint blocks from state bundle",
            "tau_identification_limit": "tau_eff is not identified as C/lambda",
        },
        "cohort": {
            "state_pairs": int(len(state)),
            "pairs_with_matched_events": int(matches["pair_id"].nunique()),
            "matched_events": int(len(matches)),
            "pairs_with_any_dynamic_data": int(dynamic_all["pair_id"].nunique()),
            "pairs_with_complete_dynamic_data": int(dynamic["pair_id"].nunique()),
            "occupied_state_blocks": int(len(blocks)),
        },
        "output_sha256": output_hashes,
    }
    manifest_path = outdir / "fig3_dynamic_analysis_manifest.json"
    write_json(manifest_path, manifest)
    success = {
        "status": "SUCCESS",
        "script_version": SCRIPT_VERSION,
        "run_mode": run_mode,
        "state_pairs": int(len(state)),
        "dynamic_pairs": int(dynamic["pair_id"].nunique()),
        "matched_events": int(len(matches)),
        "bootstrap_replicates": int(args.bootstrap),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(outdir / "_SUCCESS.json", success)
    print(f"SUCCESS: {outdir}")
    print(
        f"mode={run_mode}; state pairs={len(state)}; "
        f"dynamic pairs={dynamic['pair_id'].nunique()}; "
        f"matched events={len(matches)}; bootstrap={args.bootstrap}"
    )


if __name__ == "__main__":
    main()
