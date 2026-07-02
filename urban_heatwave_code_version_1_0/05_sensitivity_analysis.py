#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_analysis_v7.py
==============================================================
Independent sensitivity-analysis v7 script for multi-year UHI/UCI
diurnal-cycle analysis.

Purpose
-------
This v7 script keeps the core methods of the original analysis unchanged:
  1) ISD-Lite hourly reindexing and small-gap interpolation.
  2) Compatibility with native hourly and 3-hourly station records.
  3) ERA5 long-term Tmax DOY percentile threshold.
  4) ERA5 -> ISD empirical quantile-mapping bias correction.
  5) Heatwave definition: Tmax exceeds threshold for >= 3 consecutive days.
  6) NHW definition: JJA dates excluding HW dates.
  7) FFT and harmonic reconstruction method.

What is new
-----------
Only sensitivity loops and plotting-data outputs are added.

Sensitivity axes
----------------
A. Data integrity:
   MIN_YEAR_VALID_FRAC values: 0.70, 0.80, 0.90
   By default this is applied to warm-season/JJA valid-year selection,
   matching the requested MIN_YEAR_VALID_FRAC sensitivity test.
   FULL_YEAR_MIN_VALID_FRAC is kept at 0.80 unless
   APPLY_DATA_QUALITY_TO_FULL_YEAR is set to True.

B. Heatwave definitions:
   - percentile_P85
   - percentile_P90
   - percentile_P95
   - absolute_35C

Filtering rule
--------------
For every scenario loop, a pair is retained only if both heatwave and
non_heatwave period records are successfully generated.

Outputs
-------
No figures are generated. Only plotting/diagnostic data are saved.

Output directory:
  <UNIFIED_ROOT>/analysis/sensitivity
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from scipy.fft import fft as sp_fft

import geopandas as gpd
from shapely.geometry import Point
import rasterio

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from config import (
    BV_CSV_PATH,
    CONTINENT_SHP,
    ERA5_STATION_DIR,
    ISD_BASE_DIR,
    KG_TIF,
    PAIR_CSV_PATH,
    SENSITIVITY_OUTPUT_DIR as OUTPUT_DIR,
    STATION_META_PATH,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 1. Paths and global constants
# ══════════════════════════════════════════════════════════════

YEARS = list(range(2015, 2025))
# Hemisphere-aware warm season, synced with main analysis:
#   NH: JJA = Jun-Jul-Aug
#   SH: DJF = Dec-Jan-Feb
NH_WARM_MONTHS = [6, 7, 8]
SH_WARM_MONTHS = [12, 1, 2]

# Backward-compatible nominal value only.
# Actual expected days are calculated by hemisphere and season-year.
ANALYSIS_MONTHS = NH_WARM_MONTHS
WARM_SEASON_DAYS_EXPECTED = 92

# File-system paths are centralized in config.py.
PAIR_FALLBACK_PATH = PAIR_CSV_PATH
BV_RADIUS = "1000m"

# Original full-year threshold kept fixed by default.
FULL_YEAR_MIN_VALID_FRAC = 0.80

# If True, DATA_VALID_FRAC_SETTINGS will also be applied to full-year loading.
# Default False follows the user's request to vary MIN_YEAR_VALID_FRAC.
APPLY_DATA_QUALITY_TO_FULL_YEAR = False

DATA_VALID_FRAC_SETTINGS = [0.70, 0.80, 0.90]

# Lightweight extra branch: ISD year-window sensitivity.
# This branch is NOT crossed with all data-quality/HW-definition scenarios.
# It runs only under the baseline setting valid=0.80 and percentile_P90.
ENABLE_ISD_YEAR_WINDOW_SENSITIVITY = True
ISD_YEAR_WINDOW_SETTINGS = [
    {"isd_n_years": 1,  "analysis_years": [2024]},
    {"isd_n_years": 3,  "analysis_years": [2022, 2023, 2024]},
    {"isd_n_years": 5,  "analysis_years": [2020, 2021, 2022, 2023, 2024]},
    {"isd_n_years": 10, "analysis_years": list(range(2015, 2025))},
]
BASELINE_YEAR_WINDOW_VALID_FRAC = 0.80
BASELINE_YEAR_WINDOW_HW_PERCENTILE = 90

HEATWAVE_DEFINITION_SETTINGS = [
    {
        "hw_def": "percentile_P85",
        "hw_method": "percentile",
        "hw_percentile": 85,
        "abs_mode": "",
        "abs_threshold": np.nan,
    },
    {
        "hw_def": "percentile_P90",
        "hw_method": "percentile",
        "hw_percentile": 90,
        "abs_mode": "",
        "abs_threshold": np.nan,
    },
    {
        "hw_def": "percentile_P95",
        "hw_method": "percentile",
        "hw_percentile": 95,
        "abs_mode": "",
        "abs_threshold": np.nan,
    },
    {
        "hw_def": "absolute_35C",
        "hw_method": "absolute",
        "hw_percentile": np.nan,
        "abs_mode": "fixed",
        "abs_threshold": 35.0,
    },
]

MAX_CONSEC_NAN = 2
MAX_INTERP_GAP = 1

HW_MIN_DAYS = 3
HW_WINDOW_HALF = 7
RESTRICT_NON_HW_TO_HW_PAIRS = True

N_HARMONICS = 2
DAY_HOURS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]        # 08:00–19:59, 12 h
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]            # 20:00–07:59, 12 h
TROPICAL_NIGHT_THRESHOLD = 20.0

BOOT_N = 1000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

USE_ERA5_CLIMATOLOGY = True
USE_QM_BIAS_CORRECTION = True
USE_MONTHLY_MEAN_BIAS_CORRECTION = False
QM_N_QUANTILES = 1001
QM_MIN_OVERLAP_DAYS_PER_MONTH = 30
QM_MIN_OVERLAP_DAYS_ANNUAL = 100
MIN_LOYO_REF_YEARS = 5

HNE_REF_MODE = "full_year_all_pairs"
HW_REF_MODE_BASE = "ERA5_longterm_Tmax_P{P}_DOY_±7day_quantile_mapping_corrected"

STATION_VARS = [
    "LAT", "LON", "U10", "P_mm", "Rn", "BLH",
    "SM_Volumetric", "NDVI", "Albedo", "Pop_Density"
]
PAIR_FALLBACK_MAP = {
    "U10":           ("day_wspd_mean_u",       "day_wspd_mean_r"),
    "P_mm":          ("precip_u",              "precip_r"),
    "Rn":            ("Rn_mean_Wm2_u",         "Rn_mean_Wm2_r"),
    "BLH":           ("blh_u",                 "blh_r"),
    "SM_Volumetric": ("smap_soil_moisture_u",  "smap_soil_moisture_r"),
    "NDVI":          ("ndvi_median_u",         "ndvi_median_r"),
    "Albedo":        ("albedo_u",              "albedo_r"),
}

CONTINENT_THRESHOLD = {
    "Europe": 32.0,
    "Asia": 35.0,
    "North America": 35.0,
    "South America": 33.0,
    "Africa": 40.0,
    "Australia": 38.0,
    "Oceania": 38.0,
    "Antarctica": 20.0,
}
DEFAULT_THRESHOLD = 30.0

KG_INT2CODE = {
    1: "Af",  2: "Am",  3: "Aw",
    4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc",
    17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd",
    25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd",
    29: "ET", 30: "EF",
}
KG_GROUP_MAP = {
    "A": "Tropical",
    "B": "Arid",
    "C": "Temperate",
    "D": "Cold",
    "E": "Polar",
}


# ══════════════════════════════════════════════════════════════
# 2. Small utility functions
# ══════════════════════════════════════════════════════════════

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_pair_id(pair_id: str):
    parts = pair_id.split("__")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse pair_id: {pair_id}")
    u = parts[0].strip().split("_")
    r = parts[1].strip().split("_")
    if len(u) < 2 or len(r) < 2:
        raise ValueError(f"Cannot parse pair_id: {pair_id}")
    return u[0], u[1], r[0], r[1]


def isd_path(usaf: str, wban: str, year: int, data_dir: str) -> str:
    return os.path.join(data_dir, f"{usaf}-{wban}-{year}.gz")


def is_leap_year(yr: int) -> bool:
    return (yr % 4 == 0) and (yr % 100 != 0 or yr % 400 == 0)


def hemisphere_from_lat(lat) -> str:
    """Return NH / SH according to latitude."""
    try:
        return "SH" if float(lat) < 0 else "NH"
    except Exception:
        return "NH"


def warm_months_for_lat(lat) -> list:
    """Latitude-dependent warm-season months."""
    return SH_WARM_MONTHS if hemisphere_from_lat(lat) == "SH" else NH_WARM_MONTHS


def warm_season_label_for_lat(lat) -> str:
    """Readable warm-season label."""
    return "DJF" if hemisphere_from_lat(lat) == "SH" else "JJA"


def warm_months_string(lat) -> str:
    """CSV-friendly month list."""
    return ",".join(map(str, warm_months_for_lat(lat)))


def analysis_start_year(years=None) -> int:
    """First calendar year loaded by this scenario."""
    years = YEARS if years is None else years
    return int(min(years))


def analysis_end_year(years=None) -> int:
    """Last calendar year loaded by this scenario."""
    years = YEARS if years is None else years
    return int(max(years))


def warm_season_year_from_date(date_value, lat, years=None) -> int:
    """
    Assign warm-season year.

    SH boundary rule:
      - Jan-Feb of first loaded year are retained as DJF of that same year.
      - Dec of the last loaded year is excluded because the following Jan-Feb
        are outside the loaded range.
    """
    ts = pd.Timestamp(date_value)
    if hemisphere_from_lat(lat) == "SH" and ts.month == 12:
        return int(ts.year + 1)
    return int(ts.year)


def is_warm_season_date_for_analysis(date_value, lat, years=None) -> bool:
    """
    Hemisphere-aware warm-season filter inside the loaded scenario years.
    NH: JJA within years.
    SH: Jan-Feb within years; Dec only if following Jan-Feb are also inside years.
    """
    ts = pd.Timestamp(date_value)
    y0 = analysis_start_year(years)
    y1 = analysis_end_year(years)

    if hemisphere_from_lat(lat) == "SH":
        if ts.month in (1, 2):
            return y0 <= ts.year <= y1
        if ts.month == 12:
            return y0 <= ts.year < y1
        return False

    return (ts.month in NH_WARM_MONTHS) and (y0 <= ts.year <= y1)


def expected_warm_season_days(season_year: int, lat, years=None) -> int:
    """
    Expected warm-season days under the same boundary rule.

    NH: JJA = 92 days.
    SH:
      - first loaded season year: Jan-Feb only, 59 or 60 days.
      - later complete DJF years: Dec + Jan + Feb = 90 or 91 days.
    """
    y0 = analysis_start_year(years)

    if hemisphere_from_lat(lat) != "SH":
        return 92

    feb_days = 29 if is_leap_year(int(season_year)) else 28

    if int(season_year) == y0:
        return 31 + feb_days  # Jan + Feb only; Dec previous year not loaded

    return 31 + 31 + feb_days  # Dec previous year + Jan + Feb


def bootstrap_mean_ci(values, n_boot=BOOT_N, alpha=0.05):
    arr = pd.Series(values).dropna().values.astype(float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    boot_means = [
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return float(np.mean(arr)), float(lo), float(hi)


def normalize_lcz(val):
    try:
        v = float(val)
    except Exception:
        return np.nan
    if 51 <= v <= 56:
        return int(v - 50)
    return np.nan


def lcz_compactness_class(lcz):
    if pd.isna(lcz):
        return np.nan
    lcz = int(lcz)
    if lcz in [1]:
        return "compact"
    if lcz in [6]:
        return "open"
    return np.nan


def assign_lat_group(lat: float) -> str:
    if pd.isna(lat):
        return "Unknown"
    if abs(lat) < 23.5:
        return "Tropical"
    if abs(lat) < 40:
        return "Subtropical"
    if abs(lat) < 60:
        return "Temperate"
    return "Polar"


def kg_group(code):
    if not isinstance(code, str) or len(code) == 0:
        return np.nan
    return code[0]


def kg_main_climate(code):
    g = kg_group(code)
    return KG_GROUP_MAP.get(g, np.nan)


def wrap_angle_rad(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def phase_to_peak_hour(phase_rad, harmonic=1):
    if pd.isna(phase_rad) or harmonic <= 0:
        return np.nan
    return float(((-phase_rad) / (2 * np.pi * harmonic) * 24.0) % 24.0)


def circular_hour_diff(h1, h2):
    if pd.isna(h1) or pd.isna(h2):
        return np.nan
    d = (h1 - h2 + 12.0) % 24.0 - 12.0
    return float(d)


# ══════════════════════════════════════════════════════════════
# 3. External metadata loading
# ══════════════════════════════════════════════════════════════

def _read_table_flexible(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in [".xls", ".xlsx"]:
        for engine in [None, "xlrd", "openpyxl"]:
            try:
                kw = {} if engine is None else {"engine": engine}
                return pd.read_excel(path, dtype={"USAF": str, "WBAN": str}, **kw)
            except Exception:
                pass
    for sep in [",", "\t", r"\s+"]:
        try:
            return pd.read_csv(
                path, sep=sep, engine="python",
                dtype={"USAF": str, "WBAN": str}
            )
        except Exception:
            pass
    raise RuntimeError(f"Cannot read table: {path}")


@lru_cache(maxsize=1)
def load_station_meta() -> pd.DataFrame:
    df = _read_table_flexible(STATION_META_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if c.lower() == "usaf":
            rename_map[c] = "USAF"
        elif c.lower() == "wban":
            rename_map[c] = "WBAN"
    df = df.rename(columns=rename_map)
    for c in STATION_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["USAF", "WBAN"] + [c for c in STATION_VARS if c in df.columns]
    df = (
        df[keep]
        .drop_duplicates(subset=["USAF", "WBAN"])
        .set_index(["USAF", "WBAN"])
    )
    return df


@lru_cache(maxsize=1)
def load_pair_fallback() -> pd.DataFrame:
    df = pd.read_csv(PAIR_FALLBACK_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@lru_cache(maxsize=1)
def _load_bv() -> pd.DataFrame:
    df = pd.read_csv(BV_CSV_PATH, dtype={"USAF": str, "WBAN": str})
    df["USAF"] = df["USAF"].astype(str).str.strip()
    df["WBAN"] = df["WBAN"].astype(str).str.strip()
    return df.set_index(["USAF", "WBAN"])


@lru_cache(maxsize=1)
def _load_continents() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(CONTINENT_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def get_bv(usaf: str, wban: str) -> float:
    col = f"volume_m3_{BV_RADIUS}"
    try:
        return float(_load_bv().loc[(usaf.strip(), wban.strip()), col])
    except Exception:
        return np.nan


def get_station_var(usaf: str, wban: str, var: str) -> float:
    try:
        val = load_station_meta().loc[(usaf.strip(), wban.strip()), var]
        return float(val) if pd.notna(val) else np.nan
    except Exception:
        return np.nan


def _get_continent(lon: float, lat: float) -> str:
    gdf = _load_continents()
    pt = Point(lon, lat)
    hits = gdf[gdf.geometry.contains(pt)]
    if len(hits) == 0:
        gdf2 = gdf.copy()
        gdf2["_dist"] = gdf2.geometry.distance(pt)
        hits = gdf2.nsmallest(1, "_dist")
    if len(hits) == 0:
        return ""
    col = next((c for c in hits.columns if c.strip().lower() == "continent"), None)
    return str(hits.iloc[0][col]).strip() if col else ""


def extract_koppen_codes(lons, lats, tif_path=KG_TIF):
    out = []
    with rasterio.open(tif_path) as src:
        coords = [(float(x), float(y)) for x, y in zip(lons, lats)]
        vals = list(src.sample(coords))
        for i, v in enumerate(vals):
            try:
                iv = int(v[0])
                code = KG_INT2CODE.get(iv, None)
                if code is not None:
                    out.append(code)
                else:
                    found = None
                    for dlat in [-0.05, 0, 0.05]:
                        for dlon in [-0.05, 0, 0.05]:
                            if dlat == 0 and dlon == 0:
                                continue
                            v2 = list(src.sample([(
                                float(lons[i]) + dlon,
                                float(lats[i]) + dlat
                            )]))
                            iv2 = int(v2[0][0])
                            c2 = KG_INT2CODE.get(iv2, None)
                            if c2 is not None:
                                found = c2
                                break
                        if found:
                            break
                    out.append(found if found else np.nan)
            except Exception:
                out.append(np.nan)
    return out


# ══════════════════════════════════════════════════════════════
# 4. ISD-Lite reading, hourly normalization, and data loading
# ══════════════════════════════════════════════════════════════

def read_isd_lite(filepath: str):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            usecols=[0, 1, 2, 3, 4, 5],
            names=["year", "month", "day", "hour", "temp_C", "dewpoint_C"],
            na_values={"temp_C": -9999, "dewpoint_C": -9999},
            engine="c",
            on_bad_lines="skip",
        )
        df["temp_C"] = df["temp_C"] / 10.0
        df["dewpoint_C"] = df["dewpoint_C"] / 10.0
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]],
            errors="coerce"
        )
        df = df.dropna(subset=["datetime"]).drop_duplicates(subset="datetime")
        df = df.sort_values("datetime").reset_index(drop=True)
        if df.empty:
            return None
        df["date"] = df["datetime"].dt.date
        return df[["datetime", "date", "temp_C", "dewpoint_C"]]
    except Exception as exc:
        print(f"    Warning: read failed {filepath}: {exc}")
        return None


def detect_native_resolution(df):
    diffs = df["datetime"].diff().dt.total_seconds().dropna() / 3600.0
    diffs = diffs[(diffs > 0) & (diffs <= 6)]
    if len(diffs) == 0:
        return "unknown"
    mode = int(round(diffs.mode().iloc[0]))
    if mode <= 1:
        return "hourly"
    elif mode <= 3:
        return "3-hourly"
    else:
        return "coarser"


def clean_year_data_to_hourly(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Same logic as the original script:
      - resample to hourly
      - reindex to full-year hourly axis
      - interpolate only small internal gaps, limit=2
      - drop days with fewer than 8 valid temperature hours
    """
    t_start = pd.Timestamp(year, 1, 1, 0)
    t_end = pd.Timestamp(year, 12, 31, 23)
    full_idx = pd.date_range(t_start, t_end, freq="1h")

    df_idx = df.set_index("datetime").sort_index()

    hourly = df_idx[["temp_C", "dewpoint_C"]].resample("1h").mean()
    hourly = hourly.reindex(full_idx)

    hourly["temp_C"] = hourly["temp_C"].interpolate(
        method="time",
        limit=2,
        limit_area="inside"
    )
    hourly["dewpoint_C"] = hourly["dewpoint_C"].interpolate(
        method="time",
        limit=2,
        limit_area="inside"
    )

    valid_hour_count = hourly["temp_C"].groupby(hourly.index.date).transform(
        lambda s: s.notna().sum()
    )
    hourly.loc[valid_hour_count < 8, ["temp_C", "dewpoint_C"]] = np.nan

    result = hourly.reset_index().rename(columns={"index": "datetime"})
    result["date"] = result["datetime"].dt.date
    return result


def to_local_time(df: pd.DataFrame, lon: float) -> pd.DataFrame:
    offset = pd.Timedelta(hours=lon / 15.0)
    out = df.copy()
    out["local_datetime"] = out["datetime"] + offset
    out["local_hour"] = (
        out["local_datetime"].dt.hour
        + out["local_datetime"].dt.minute / 60.0
    )
    out["local_date"] = out["local_datetime"].dt.date
    out["local_month"] = pd.to_datetime(out["local_datetime"]).dt.month
    return out


def load_multiyear_station_ALL(
        usaf: str,
        wban: str,
        years,
        base_dir: str,
        lon: float,
        min_valid_frac: float,
        verbose: bool = False):
    combined = []
    valid_years = []
    native_resolution_by_year = {}

    for yr in years:
        year_dir = os.path.join(base_dir, str(yr))
        fpath = isd_path(usaf, wban, yr, year_dir)

        df_raw = read_isd_lite(fpath)
        if df_raw is None:
            if verbose:
                print(f"    [{yr}] file missing: {fpath}")
            continue

        native_resolution_by_year[yr] = detect_native_resolution(df_raw)

        df_clean = clean_year_data_to_hourly(df_raw, yr)
        if df_clean is None:
            continue

        df_local = to_local_time(df_clean, lon)
        year_days = 366 if is_leap_year(yr) else 365
        valid_days = df_local.dropna(subset=["temp_C"])["local_date"].nunique()
        valid_frac = valid_days / year_days

        if valid_frac < min_valid_frac:
            if verbose:
                print(
                    f"    [{yr}] full-year validity {valid_frac:.1%} "
                    f"< {min_valid_frac:.0%} -> skip"
                )
            continue

        combined.append(df_local)
        valid_years.append(yr)

    if not combined:
        return None, [], native_resolution_by_year

    return pd.concat(combined, ignore_index=True), valid_years, native_resolution_by_year


def slice_warm_season(
        df_all: pd.DataFrame,
        lat,
        min_year_valid_frac: float,
        years=None):
    """
    Hemisphere-aware warm-season slicing.

    Pair-level analyses should pass the urban-station latitude so that urban
    and rural periods use the same warm-season calendar:
      NH: JJA
      SH: DJF, with Dec assigned to the following season year.
    """
    if df_all is None:
        return None, []

    df_warm = df_all[
        df_all["local_date"].apply(
            lambda d: is_warm_season_date_for_analysis(d, lat, years=years)
        )
    ].copy()

    if df_warm.empty:
        return None, []

    df_warm["warm_season_year"] = df_warm["local_date"].apply(
        lambda d: warm_season_year_from_date(d, lat, years=years)
    )

    valid_yrs = []
    for season_year, grp in df_warm.groupby("warm_season_year"):
        expected_days = expected_warm_season_days(int(season_year), lat, years=years)
        valid_days = grp.dropna(subset=["temp_C"])["local_date"].nunique()

        if expected_days > 0 and (valid_days / expected_days) >= min_year_valid_frac:
            valid_yrs.append(int(season_year))

    if not valid_yrs:
        return None, []

    return df_warm[df_warm["warm_season_year"].isin(valid_yrs)].copy(), valid_yrs

def subset_by_dates(df_loc: pd.DataFrame, dates):
    if df_loc is None or not dates:
        return None
    return df_loc[df_loc["local_date"].isin(dates)].copy()


# ══════════════════════════════════════════════════════════════
# 5. Heatwave detection and ERA5/ISD calibration
# ══════════════════════════════════════════════════════════════

def compute_hw_threshold_from_tmax(
        tmax_series: pd.Series,
        q: int,
        window_half_width: int = HW_WINDOW_HALF):
    ref = tmax_series.reset_index()
    ref.columns = ["local_date", "tmax"]
    ref["local_date"] = pd.to_datetime(ref["local_date"])
    ref["doy"] = ref["local_date"].apply(
        lambda x: (
            pd.Timestamp(2001, x.month, x.day).dayofyear
            if not (x.month == 2 and x.day == 29) else 59
        )
    )
    ref = ref.dropna(subset=["tmax"])

    target_dates = tmax_series.index.tolist()
    thresholds = {}
    valid_counts = {}

    for d in target_dates:
        d_ts = pd.Timestamp(d)
        doy = (
            pd.Timestamp(2001, d_ts.month, d_ts.day).dayofyear
            if not (d_ts.month == 2 and d_ts.day == 29) else 59
        )
        low, high = doy - window_half_width, doy + window_half_width

        if low < 1:
            mask = (ref["doy"] >= 365 + low) | (ref["doy"] <= high)
        elif high > 365:
            mask = (ref["doy"] >= low) | (ref["doy"] <= high - 365)
        else:
            mask = (ref["doy"] >= low) & (ref["doy"] <= high)

        vals = ref.loc[mask, "tmax"].dropna().values.astype(float)
        valid_counts[d] = len(vals)
        thresholds[d] = float(np.nanpercentile(vals, q)) if len(vals) > 0 else np.nan

    return pd.Series(thresholds).sort_index(), pd.Series(valid_counts).sort_index()


def detect_heatwave(daily_tmax: pd.Series, threshold) -> pd.Series:
    if np.isscalar(threshold):
        above = (daily_tmax > float(threshold)).values
    else:
        thr_series = pd.Series(threshold).reindex(daily_tmax.index)
        above = (daily_tmax > thr_series).fillna(False).values

    hw = np.zeros(len(above), dtype=bool)
    n, i = len(above), 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= HW_MIN_DAYS:
                hw[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(hw, index=daily_tmax.index)


def get_absolute_hw_threshold(continent: str) -> float:
    return CONTINENT_THRESHOLD.get(continent, DEFAULT_THRESHOLD)


def detect_heatwave_absolute(
        daily_tmax: pd.Series,
        threshold: float,
        min_days: int = HW_MIN_DAYS) -> pd.Series:
    above = (daily_tmax > threshold).values
    hw = np.zeros(len(above), dtype=bool)
    n, i = len(above), 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_days:
                hw[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(hw, index=daily_tmax.index)

def detect_heatwave_warm_season_by_year(
        daily_tmax: pd.Series,
        threshold,
        lat,
        years=None) -> pd.Series:
    """
    Detect heatwaves only within the valid hemisphere-specific warm season,
    with consecutive-day detection performed separately for each
    warm-season year.

    This follows the same boundary logic as the main analysis:

    Northern Hemisphere
    -------------------
    JJA dates belong to their calendar year.

    Southern Hemisphere
    -------------------
    December belongs to the following DJF season year.

    For the loaded analysis-year window:
      - January-February of the first loaded year are retained;
      - December of the final loaded year is excluded because the following
        January-February are outside the loaded data window;
      - heatwave runs are not allowed to connect February directly to the
        following December;
      - heatwave runs are not allowed to cross different warm-season years.

    Parameters
    ----------
    daily_tmax
        Daily urban Tmax indexed by local date.

    threshold
        Either:
          - a scalar threshold, such as the fixed 35 degC sensitivity; or
          - a date-indexed threshold Series, such as the ERA5/QM-corrected
            percentile threshold.

    lat
        Urban-station latitude used to define the pair-level warm-season
        calendar.

    years
        Calendar years loaded by the current sensitivity scenario. This is
        required for the ISD year-window sensitivity so that first-year and
        final-year DJF boundaries follow the actual scenario window.

    Returns
    -------
    pandas.Series
        Boolean heatwave mask indexed by the valid warm-season daily Tmax
        dates supplied to this function.
    """
    s0 = pd.Series(
        daily_tmax,
        dtype=float,
    ).sort_index()

    out = pd.Series(
        False,
        index=s0.index,
        dtype=bool,
    )

    s = s0.dropna()

    if s.empty:
        return out

    valid_warm_dates = [
        d
        for d in s.index
        if is_warm_season_date_for_analysis(
            d,
            lat,
            years=years,
        )
    ]

    if not valid_warm_dates:
        return out

    s_warm = (
        s.reindex(
            pd.Index(
                sorted(valid_warm_dates)
            )
        )
        .dropna()
    )

    if s_warm.empty:
        return out

    season_year = pd.Series(
        [
            warm_season_year_from_date(
                d,
                lat,
                years=years,
            )
            for d in s_warm.index
        ],
        index=s_warm.index,
        dtype=int,
    )

    for _, season_dates in season_year.groupby(
            season_year,
            sort=True):

        idx_dates = pd.Index(
            sorted(
                season_dates.index.tolist()
            )
        )

        tmax_part = (
            s_warm.reindex(idx_dates)
            .dropna()
        )

        if tmax_part.empty:
            continue

        if np.isscalar(threshold):
            threshold_part = float(threshold)
        else:
            threshold_part = (
                pd.Series(threshold)
                .reindex(tmax_part.index)
            )

        hw_part = detect_heatwave(
            tmax_part,
            threshold_part,
        )

        out.loc[hw_part.index] = (
            hw_part.astype(bool)
        )

    return out.fillna(False).astype(bool)

def load_era5_tmax_series(usaf: str, wban: str, station_type: str = "urban"):
    fpath = os.path.join(
        ERA5_STATION_DIR,
        f"{usaf.strip()}_{wban.strip()}_{station_type}.csv"
    )

    if not os.path.exists(fpath):
        return None

    try:
        df = pd.read_csv(fpath, parse_dates=["date"], index_col="date")
        if "tmax_c" in df.columns:
            tmax_col = "tmax_c"
        elif "tmax" in df.columns:
            tmax_col = "tmax"
        else:
            return None

        s = pd.to_numeric(df[tmax_col], errors="coerce").dropna()
        if len(s) < MIN_LOYO_REF_YEARS * 30:
            return None
        return s
    except Exception:
        return None


def compute_hw_doy_thr_era5(
        era5_tmax: pd.Series,
        q: int,
        window_half_width: int = HW_WINDOW_HALF) -> dict:
    ref = era5_tmax.copy()
    ref.index = pd.to_datetime(ref.index)
    ref_df = pd.DataFrame({"tmax": ref.values, "date": ref.index})
    ref_df["doy"] = ref_df["date"].apply(
        lambda x: (
            pd.Timestamp(2001, x.month, x.day).dayofyear
            if not (x.month == 2 and x.day == 29) else 59
        )
    )
    ref_df = ref_df.dropna(subset=["tmax"])

    doy_thr = {}
    for doy in range(1, 366):
        low = doy - window_half_width
        high = doy + window_half_width
        if low < 1:
            mask = (ref_df["doy"] >= 365 + low) | (ref_df["doy"] <= high)
        elif high > 365:
            mask = (ref_df["doy"] >= low) | (ref_df["doy"] <= high - 365)
        else:
            mask = (ref_df["doy"] >= low) & (ref_df["doy"] <= high)

        vals = ref_df.loc[mask, "tmax"].dropna().values
        doy_thr[doy] = float(np.nanpercentile(vals, q)) if len(vals) > 0 else np.nan

    return doy_thr


def apply_doy_thr_to_index(date_index, doy_thr: dict) -> pd.Series:
    dates = pd.to_datetime(date_index)
    doys = dates.map(
        lambda x: (
            pd.Timestamp(2001, x.month, x.day).dayofyear
            if not (x.month == 2 and x.day == 29) else 59
        )
    )
    thr_vals = [doy_thr.get(int(d), np.nan) for d in doys]
    return pd.Series(thr_vals, index=date_index)


def quantile_mapping_values(values, sim_train, obs_train, n_quantiles=1001):
    values_arr = np.asarray(values, dtype=float)

    sim = pd.Series(sim_train).dropna().astype(float).values
    obs = pd.Series(obs_train).dropna().astype(float).values

    if len(sim) < 2 or len(obs) < 2:
        return values_arr.copy()

    quantiles = np.linspace(0, 100, n_quantiles)

    sim_q = np.nanpercentile(sim, quantiles)
    obs_q = np.nanpercentile(obs, quantiles)

    valid = np.isfinite(sim_q) & np.isfinite(obs_q)
    sim_q = sim_q[valid]
    obs_q = obs_q[valid]

    if len(sim_q) < 2:
        return values_arr.copy()

    sim_q_unique, unique_idx = np.unique(sim_q, return_index=True)
    obs_q_unique = obs_q[unique_idx]

    if len(sim_q_unique) < 2:
        return values_arr.copy()

    corrected = np.interp(
        values_arr,
        sim_q_unique,
        obs_q_unique,
        left=obs_q_unique[0],
        right=obs_q_unique[-1],
    )
    return corrected


def apply_quantile_mapping_bias_correction(
        hw_thr_series_raw: pd.Series,
        isd_tmax: pd.Series,
        era5_tmax: pd.Series,
        min_overlap_days_per_month: int = QM_MIN_OVERLAP_DAYS_PER_MONTH,
        min_overlap_days_annual: int = QM_MIN_OVERLAP_DAYS_ANNUAL,
        n_quantiles: int = QM_N_QUANTILES):
    isd = pd.Series(isd_tmax).copy()
    era = pd.Series(era5_tmax).copy()
    thr = pd.Series(hw_thr_series_raw).copy()

    isd.index = pd.to_datetime(isd.index)
    era.index = pd.to_datetime(era.index)

    overlap = pd.concat(
        [isd.rename("isd"), era.rename("era5")],
        axis=1,
        join="inner"
    ).dropna()

    diagnostics = {
        "qm_ref_mode": "monthly_QM_with_annual_fallback",
        "qm_n_quantiles": int(n_quantiles),
        "overlap_days": int(len(overlap)),
        "monthly_sample_counts": {},
        "monthly_methods": {},
        "annual_sample_count": int(len(overlap)),
    }

    if overlap.empty:
        for m in range(1, 13):
            diagnostics["monthly_sample_counts"][m] = 0
            diagnostics["monthly_methods"][m] = "no_correction_no_overlap"
        return thr.copy(), diagnostics

    has_annual_qm = len(overlap) >= min_overlap_days_annual

    corrected = thr.copy()
    thr_dt = pd.to_datetime(thr.index)

    for m in range(1, 13):
        month_mask_thr = thr_dt.month == m
        month_values = thr.loc[month_mask_thr].values

        sub = overlap[overlap.index.month == m]
        diagnostics["monthly_sample_counts"][m] = int(len(sub))

        if len(month_values) == 0:
            diagnostics["monthly_methods"][m] = "no_threshold_values"
            continue

        if len(sub) >= min_overlap_days_per_month:
            corrected_values = quantile_mapping_values(
                values=month_values,
                sim_train=sub["era5"],
                obs_train=sub["isd"],
                n_quantiles=n_quantiles,
            )
            diagnostics["monthly_methods"][m] = "monthly_QM"

        elif has_annual_qm:
            corrected_values = quantile_mapping_values(
                values=month_values,
                sim_train=overlap["era5"],
                obs_train=overlap["isd"],
                n_quantiles=n_quantiles,
            )
            diagnostics["monthly_methods"][m] = "annual_QM_fallback"

        else:
            corrected_values = month_values.copy()
            diagnostics["monthly_methods"][m] = "no_correction_insufficient_samples"

        corrected.loc[month_mask_thr] = corrected_values

    diagnostics["isd_tmax_mean_overlap"] = float(overlap["isd"].mean())
    diagnostics["era5_tmax_mean_overlap"] = float(overlap["era5"].mean())

    return corrected, diagnostics


# ══════════════════════════════════════════════════════════════
# 6. FFT and diurnal metrics
# ══════════════════════════════════════════════════════════════

def reconstruct_diurnal(mean_t, amps, phases, n_points=24):
    t = np.arange(n_points)
    sig = np.full(n_points, mean_t, dtype=float)
    for k, (a, p) in enumerate(zip(amps, phases), start=1):
        sig += a * np.cos(k * 2 * np.pi * t / n_points + p)
    return sig


def compute_fft(df_local: pd.DataFrame, n_harm=N_HARMONICS):
    """
    Same FFT logic as the original script:
      - group local hour to 24 hourly means
      - require at least 8 valid hourly bins
      - fill remaining hourly bins by linear interpolation
      - compute FFT amplitudes/phases and harmonic reconstruction
      - dewpoint curve is direct hourly mean/interpolation, not FFT
    """
    if df_local is None or len(df_local) == 0:
        return None
    df_local = df_local.dropna(subset=["temp_C"])
    if len(df_local) == 0:
        return None

    tmp = df_local.copy()
    tmp["h_bin"] = tmp["local_hour"].round().astype(int) % 24

    hourly_t = tmp.groupby("h_bin")["temp_C"].mean().reindex(range(24))
    if hourly_t.notna().sum() < 8:
        return None

    hourly_t = (
        hourly_t
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    if hourly_t.isna().any():
        return None

    vals = hourly_t.values.astype(float)
    coef = sp_fft(vals) / len(vals)
    mean_t = float(np.real(coef[0]))
    amps, phases = [], []
    for k in range(1, n_harm + 1):
        amps.append(float(2 * np.abs(coef[k])))
        phases.append(float(np.angle(coef[k])))

    reconstructed = reconstruct_diurnal(mean_t, amps, phases)

    if "dewpoint_C" in tmp.columns:
        hourly_d = tmp.groupby("h_bin")["dewpoint_C"].mean().reindex(range(24))
        if hourly_d.notna().sum() >= 3:
            hourly_d = (
                hourly_d
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
        dew_curve = hourly_d.values.astype(float)
    else:
        dew_curve = np.full(24, np.nan)

    return dict(
        mean=mean_t,
        amplitudes=amps,
        phases=phases,
        Tmax_fft=float(np.max(reconstructed)),
        Tmin_fft=float(np.min(reconstructed)),
        hourly_obs=vals,
        hourly_dew=dew_curve,
        n_days=int(tmp["local_date"].nunique()),
    )


def compute_daily_minmax(df_loc: pd.DataFrame):
    if df_loc is None or len(df_loc) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = df_loc.groupby("local_date")["temp_C"]
    return g.min().dropna(), g.max().dropna()


def curve_delta_metrics(u_curve, r_curve):
    delta = np.array(u_curve) - np.array(r_curve)
    return {
        "dT_day_mean": float(np.mean(delta[DAY_HOURS])),
        "dT_night_mean": float(np.mean(delta[NIGHT_HOURS])),
        "daytime_uci_flag": int(np.max(u_curve) - np.max(r_curve) < 0),
        "nighttime_uhi_flag": int(np.min(u_curve) - np.min(r_curve) > 0),
        "delta_peak_hour": int(np.argmax(delta)),
        "delta_min_hour": int(np.argmin(delta)),
        "delta_peak_value": float(np.max(delta)),
        "delta_min_value": float(np.min(delta)),
        "urban_peak_hour": int(np.argmax(u_curve)),
        "rural_peak_hour": int(np.argmax(r_curve)),
        "urban_min_hour": int(np.argmin(u_curve)),
        "rural_min_hour": int(np.argmin(r_curve)),
    }



# ─────────────────────────────────────────────────────────────
# UHI/UCI regime definition
# ─────────────────────────────────────────────────────────────
REGIME_DEFINITION = "annual_two_harmonic_synthesised_Tx_urban_minus_rural"
REGIME_CLASSIFICATION_METRIC = "delta_tx_annual_synth"


def classify_uhi_uci_regime_from_annual_harmonic_tx(df_u_all, df_r_all):
    """
    Classify the fixed manuscript-level UHI/UCI regime from the annual
    two-harmonic synthesised diurnal temperature cycle.

    The regime is based on ΔTx = max(Tu_syn(t)) - max(Tr_syn(t)), where
    Tu_syn and Tr_syn are the annual 24-h curves reconstructed from the first
    two FFT harmonics. This is intentionally different from averaging raw daily
    Tmax differences; it matches the mean–amplitude framework in the paper.
    """
    fft_u = compute_fft(df_u_all)
    fft_r = compute_fft(df_r_all)
    if fft_u is None or fft_r is None:
        return None

    u_curve = reconstruct_diurnal(
        fft_u["mean"], fft_u["amplitudes"], fft_u["phases"]
    )
    r_curve = reconstruct_diurnal(
        fft_r["mean"], fft_r["amplitudes"], fft_r["phases"]
    )

    u_tx = float(np.nanmax(u_curve))
    r_tx = float(np.nanmax(r_curve))
    d_tx = u_tx - r_tx

    return {
        "group": "UHI" if d_tx >= 0 else "UCI",
        "regime_definition": REGIME_DEFINITION,
        "regime_classification_metric": REGIME_CLASSIFICATION_METRIC,
        "regime_reference_period": "annual",
        "regime_temperature_source": "two_harmonic_synthesised_diurnal_cycle",
        "urban_tx_annual_synth": u_tx,
        "rural_tx_annual_synth": r_tx,
        "delta_tx_annual_synth": float(d_tx),
        "urban_tx_hour_annual_synth": int(np.nanargmax(u_curve)),
        "rural_tx_hour_annual_synth": int(np.nanargmax(r_curve)),
        "urban_tn_annual_synth": float(np.nanmin(u_curve)),
        "rural_tn_annual_synth": float(np.nanmin(r_curve)),
        "delta_tn_annual_synth": float(np.nanmin(u_curve) - np.nanmin(r_curve)),
    }


def compute_daily_risk_metrics(df_u_loc, df_r_loc):
    empty = dict(
        urban_tropical_night_freq=np.nan,
        rural_tropical_night_freq=np.nan,
        delta_tropical_night_freq=np.nan,
        urban_hotnight_excess=np.nan,
        rural_hotnight_excess=np.nan,
        delta_hotnight_excess=np.nan,
        urban_daily_tmin_mean=np.nan,
        rural_daily_tmin_mean=np.nan,
        delta_daily_tmin_mean=np.nan,
        risk_common_days=0,
    )
    if df_u_loc is None or df_r_loc is None:
        return empty

    u_tmin, _ = compute_daily_minmax(df_u_loc)
    r_tmin, _ = compute_daily_minmax(df_r_loc)
    common = u_tmin.index.intersection(r_tmin.index)
    if len(common) == 0:
        return empty

    u, r = u_tmin[common], r_tmin[common]
    return dict(
        urban_tropical_night_freq=float((u > TROPICAL_NIGHT_THRESHOLD).mean()),
        rural_tropical_night_freq=float((r > TROPICAL_NIGHT_THRESHOLD).mean()),
        delta_tropical_night_freq=float(
            (u > TROPICAL_NIGHT_THRESHOLD).mean()
            - (r > TROPICAL_NIGHT_THRESHOLD).mean()
        ),
        urban_hotnight_excess=float(
            np.maximum(u - TROPICAL_NIGHT_THRESHOLD, 0).mean()
        ),
        rural_hotnight_excess=float(
            np.maximum(r - TROPICAL_NIGHT_THRESHOLD, 0).mean()
        ),
        delta_hotnight_excess=float(
            (
                np.maximum(u - TROPICAL_NIGHT_THRESHOLD, 0)
                - np.maximum(r - TROPICAL_NIGHT_THRESHOLD, 0)
            ).mean()
        ),
        urban_daily_tmin_mean=float(u.mean()),
        rural_daily_tmin_mean=float(r.mean()),
        delta_daily_tmin_mean=float((u - r).mean()),
        risk_common_days=int(len(common)),
    )


# ══════════════════════════════════════════════════════════════
# 7. Scenario processing helpers
# ══════════════════════════════════════════════════════════════

def _years_label(years):
    years = sorted(int(y) for y in years)
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def _analysis_years_csv(years):
    return ",".join(map(str, sorted(int(y) for y in years)))


def _scenario_year_fields(scenario, fallback_years=YEARS):
    years = sorted(int(y) for y in scenario.get("analysis_years", fallback_years))
    return {
        "sensitivity_axis": scenario.get("sensitivity_axis", "data_quality_hw_definition"),
        "analysis_years": _analysis_years_csv(years),
        "isd_n_years": int(scenario.get("isd_n_years", len(years))),
        "isd_years_label": scenario.get("isd_years_label", _years_label(years)),
        "isd_start_year": int(scenario.get("isd_start_year", years[0])),
        "isd_end_year": int(scenario.get("isd_end_year", years[-1])),
    }


def _make_hw_ref_mode(hw_def, hw_min_days=HW_MIN_DAYS):
    if hw_def["hw_method"] == "percentile":
        return HW_REF_MODE_BASE.format(P=int(hw_def["hw_percentile"]))
    if hw_def["abs_mode"] == "fixed":
        return (
            f"absolute_Tmax_{hw_def['abs_threshold']:.1f}C_"
            f"consecutive_ge_{hw_min_days}_days"
        )
    raise ValueError(f"Unsupported heatwave definition in v7: {hw_def}")


def make_scenarios():
    scenarios = []

    # A. Original sensitivity branch:
    #    data integrity × heatwave definition, using full 2015-2024 ISD data.
    for frac in DATA_VALID_FRAC_SETTINGS:
        for hw_def in HEATWAVE_DEFINITION_SETTINGS:
            sid = f"valid{int(round(frac * 100)):03d}_{hw_def['hw_def']}"
            full_year_frac = (
                frac if APPLY_DATA_QUALITY_TO_FULL_YEAR
                else FULL_YEAR_MIN_VALID_FRAC
            )

            scenario = dict(hw_def)
            scenario.update({
                "scenario_id": sid,
                "sensitivity_axis": "data_quality_hw_definition",

                "analysis_years": YEARS,
                "isd_n_years": len(YEARS),
                "isd_years_label": _years_label(YEARS),
                "isd_start_year": YEARS[0],
                "isd_end_year": YEARS[-1],

                "data_quality_min_year_valid_frac": float(frac),
                "warm_min_year_valid_frac": float(frac),
                "full_year_min_valid_frac": float(full_year_frac),
                "apply_data_quality_to_full_year": int(APPLY_DATA_QUALITY_TO_FULL_YEAR),
                "hw_min_days": HW_MIN_DAYS,
                "hw_window_half": HW_WINDOW_HALF,
            })
            scenario["hw_ref_mode"] = _make_hw_ref_mode(scenario)
            scenarios.append(scenario)

    # B. Lightweight ISD year-window branch:
    #    only baseline valid=0.80 and percentile_P90, avoiding heavy crossing.
    if ENABLE_ISD_YEAR_WINDOW_SENSITIVITY:
        baseline_hw = {
            "hw_def": "percentile_P90",
            "hw_method": "percentile",
            "hw_percentile": BASELINE_YEAR_WINDOW_HW_PERCENTILE,
            "abs_mode": "",
            "abs_threshold": np.nan,
        }

        for spec in ISD_YEAR_WINDOW_SETTINGS:
            years = sorted(int(y) for y in spec["analysis_years"])
            n_years = int(spec["isd_n_years"])

            scenario = dict(baseline_hw)
            scenario.update({
                "scenario_id": f"isdwin{n_years:02d}yr_valid080_percentile_P90",
                "sensitivity_axis": "isd_year_window",

                "analysis_years": years,
                "isd_n_years": n_years,
                "isd_years_label": _years_label(years),
                "isd_start_year": years[0],
                "isd_end_year": years[-1],

                "data_quality_min_year_valid_frac": BASELINE_YEAR_WINDOW_VALID_FRAC,
                "warm_min_year_valid_frac": BASELINE_YEAR_WINDOW_VALID_FRAC,
                "full_year_min_valid_frac": FULL_YEAR_MIN_VALID_FRAC,
                "apply_data_quality_to_full_year": 0,
                "hw_min_days": HW_MIN_DAYS,
                "hw_window_half": HW_WINDOW_HALF,
            })
            scenario["hw_ref_mode"] = _make_hw_ref_mode(scenario)
            scenarios.append(scenario)

    return scenarios


def get_station_meta_values(pair_id, usaf_u, wban_u, usaf_r, wban_r):
    pair_fallback_df = load_pair_fallback()
    pair_match = pair_fallback_df[pair_fallback_df["pair_id"] == pair_id]
    pair_row = pair_match.iloc[0] if len(pair_match) > 0 else None

    station_meta_vals = {}
    for var in STATION_VARS:
        u_val = get_station_var(usaf_u, wban_u, var)
        r_val = get_station_var(usaf_r, wban_r, var)
        if pair_row is not None and var in PAIR_FALLBACK_MAP:
            u_col, r_col = PAIR_FALLBACK_MAP[var]
            if pd.isna(u_val) and u_col in pair_row.index:
                try:
                    u_val = float(pair_row[u_col]) if pd.notna(pair_row[u_col]) else np.nan
                except Exception:
                    pass
            if pd.isna(r_val) and r_col in pair_row.index:
                try:
                    r_val = float(pair_row[r_col]) if pd.notna(pair_row[r_col]) else np.nan
                except Exception:
                    pass
        station_meta_vals[f"urban_{var}"] = u_val
        station_meta_vals[f"rural_{var}"] = r_val
        station_meta_vals[f"delta_{var}"] = (
            u_val - r_val if pd.notna(u_val) and pd.notna(r_val) else np.nan
        )

    return station_meta_vals


def get_koppen_info(lon, lat):
    try:
        codes = extract_koppen_codes([lon], [lat])
        code = codes[0] if codes else np.nan
    except Exception:
        code = np.nan
    group = kg_group(code) if isinstance(code, str) else ""
    climate = KG_GROUP_MAP.get(group, "") if group else ""
    return code, group, climate


def prepare_heatwave_mask_and_threshold(
        scenario: dict,
        u_tmax_all: pd.Series,
        usaf_u: str,
        wban_u: str,
        continent: str):
    """
    Prepare a full-year threshold series and a full-year diagnostic
    heatwave mask for one sensitivity scenario.

    Important
    ---------
    The full-year heatwave mask returned by this function is retained for:

      - annual heatwave-day diagnostics;
      - annual non-heatwave-day diagnostics;
      - full-year heatwave-intensity diagnostics;
      - backward-compatible output fields.

    The formal warm-season HW/NHW dates are not determined here. They are
    detected later in process_single_pair_scenario(), separately within each
    hemisphere-specific warm-season year.

    Supported methods
    -----------------
    percentile
        ERA5 long-term DOY percentile threshold, optionally corrected to ISD
        using empirical quantile mapping. If ERA5 climatology is disabled,
        the threshold is estimated from the available ISD Tmax series.

    absolute
        Fixed or continent-dependent absolute Tmax threshold.

    Returns
    -------
    hw_mask_all : pandas.Series or None
        Boolean full-year heatwave mask indexed like u_tmax_all.

    hw_thr_series : pandas.Series or None
        Date-indexed threshold series. For absolute scenarios, the scalar
        threshold is expanded to a constant date-indexed Series so that the
        downstream warm-season grouped detector can use the same interface.

    threshold_info : dict or None
        Threshold and bias-correction diagnostics.

    error : dict or None
        Structured error information. A successful call returns error=None.
    """
    hw_method = str(
        scenario.get("hw_method", "")
    ).strip().lower()

    # ============================================================
    # A. Percentile-threshold sensitivity scenarios
    # ============================================================
    if hw_method == "percentile":
        q = int(scenario["hw_percentile"])

        # --------------------------------------------------------
        # A1. ERA5 long-term climatological threshold
        # --------------------------------------------------------
        if USE_ERA5_CLIMATOLOGY:
            era5_tmax_u = load_era5_tmax_series(
                usaf_u,
                wban_u,
                "urban",
            )

            if era5_tmax_u is None or len(era5_tmax_u) == 0:
                return None, None, None, {
                    "fail_step": "ERA5_missing",
                    "missing_data": (
                        "ERA5 Tmax CSV 缺失或数据量不足"
                    ),
                }

            doy_thresholds = compute_hw_doy_thr_era5(
                era5_tmax_u,
                q=q,
                window_half_width=HW_WINDOW_HALF,
            )

            hw_thr_series_raw = apply_doy_thr_to_index(
                u_tmax_all.index,
                doy_thresholds,
            )

            # Preserve the existing diagnostic definition.
            valid_count_series = pd.Series(
                len(era5_tmax_u),
                index=u_tmax_all.index,
                dtype=float,
            )

            # ----------------------------------------------------
            # A1a. ERA5-to-ISD quantile mapping
            # ----------------------------------------------------
            if USE_QM_BIAS_CORRECTION:
                (
                    hw_thr_series_corrected,
                    bias_diag,
                ) = apply_quantile_mapping_bias_correction(
                    hw_thr_series_raw=hw_thr_series_raw,
                    isd_tmax=u_tmax_all,
                    era5_tmax=era5_tmax_u,
                    min_overlap_days_per_month=(
                        QM_MIN_OVERLAP_DAYS_PER_MONTH
                    ),
                    min_overlap_days_annual=(
                        QM_MIN_OVERLAP_DAYS_ANNUAL
                    ),
                    n_quantiles=QM_N_QUANTILES,
                )

                bias_correction_method = (
                    "quantile_mapping_empirical_cdf"
                )

                qm_diagnostics = bias_diag
                era5_isd_bias_annual = np.nan

            # ----------------------------------------------------
            # A1b. Raw ERA5 threshold without QM
            # ----------------------------------------------------
            else:
                hw_thr_series_corrected = (
                    pd.Series(hw_thr_series_raw)
                    .copy()
                )

                qm_diagnostics = {}
                era5_isd_bias_annual = np.nan

                bias_correction_method = (
                    "none_raw_era5_threshold"
                )

            hw_thr_series = (
                pd.Series(hw_thr_series_corrected)
                .reindex(u_tmax_all.index)
                .astype(float)
            )

        # --------------------------------------------------------
        # A2. ISD short-term threshold fallback branch
        # --------------------------------------------------------
        else:
            (
                hw_thr_series,
                valid_count_series,
            ) = compute_hw_threshold_from_tmax(
                u_tmax_all,
                q=q,
                window_half_width=HW_WINDOW_HALF,
            )

            hw_thr_series = (
                pd.Series(hw_thr_series)
                .reindex(u_tmax_all.index)
                .astype(float)
            )

            hw_thr_series_raw = (
                hw_thr_series.copy()
            )

            hw_thr_series_corrected = (
                hw_thr_series.copy()
            )

            qm_diagnostics = {}
            era5_isd_bias_annual = np.nan

            bias_correction_method = (
                "ISD_shortterm_threshold_no_ERA5"
            )

        # --------------------------------------------------------
        # A3. Validate the resulting threshold
        # --------------------------------------------------------
        hw_thr_mean = (
            float(
                pd.Series(hw_thr_series)
                .dropna()
                .mean()
            )
            if pd.Series(hw_thr_series).notna().any()
            else np.nan
        )

        if pd.isna(hw_thr_mean):
            return None, None, None, {
                "fail_step": (
                    "compute_hw_threshold_from_tmax"
                ),
                "missing_data": (
                    "城市Tmax数据过少，无法计算热浪阈值"
                ),
            }

        # Full-year detection is retained for diagnostics only.
        hw_mask_all = detect_heatwave(
            u_tmax_all,
            hw_thr_series,
        )

        raw_reindexed = (
            pd.Series(hw_thr_series_raw)
            .reindex(u_tmax_all.index)
            .astype(float)
        )

        corr_reindexed = (
            pd.Series(hw_thr_series_corrected)
            .reindex(u_tmax_all.index)
            .astype(float)
        )

        threshold_info = {
            "hw_threshold_mean": hw_thr_mean,

            "hw_threshold_raw_mean": (
                float(raw_reindexed.mean())
                if raw_reindexed.notna().any()
                else np.nan
            ),

            "hw_threshold_corrected_mean": (
                float(corr_reindexed.mean())
                if corr_reindexed.notna().any()
                else np.nan
            ),

            "hw_threshold_qm_mean": (
                float(corr_reindexed.mean())
                if (
                    USE_QM_BIAS_CORRECTION
                    and corr_reindexed.notna().any()
                )
                else np.nan
            ),

            "bias_correction_method": (
                bias_correction_method
            ),

            "era5_isd_bias_annual": (
                era5_isd_bias_annual
            ),

            "n_valid_ref_days": (
                float(
                    pd.Series(valid_count_series)
                    .dropna()
                    .mean()
                )
                if (
                    valid_count_series is not None
                    and len(valid_count_series) > 0
                    and pd.Series(
                        valid_count_series
                    ).notna().any()
                )
                else np.nan
            ),

            "qm_ref_mode": (
                qm_diagnostics.get(
                    "qm_ref_mode",
                    "",
                )
            ),

            "qm_n_quantiles": (
                qm_diagnostics.get(
                    "qm_n_quantiles",
                    np.nan,
                )
            ),

            "qm_monthly_sample_counts": ";".join(
                [
                    (
                        f"{m}:"
                        f"{qm_diagnostics.get('monthly_sample_counts', {}).get(m, 0)}"
                    )
                    for m in range(1, 13)
                ]
            ),

            "qm_monthly_methods": ";".join(
                [
                    (
                        f"{m}:"
                        f"{qm_diagnostics.get('monthly_methods', {}).get(m, '')}"
                    )
                    for m in range(1, 13)
                ]
            ),

            "isd_tmax_mean_overlap": (
                qm_diagnostics.get(
                    "isd_tmax_mean_overlap",
                    np.nan,
                )
            ),

            "era5_tmax_mean_overlap": (
                qm_diagnostics.get(
                    "era5_tmax_mean_overlap",
                    np.nan,
                )
            ),

            "era5_isd_overlap_days": (
                qm_diagnostics.get(
                    "overlap_days",
                    0,
                )
            ),

            "exceed_days_raw": int(
                (
                    u_tmax_all
                    > raw_reindexed
                )
                .fillna(False)
                .sum()
            ),

            "exceed_days_corrected": int(
                (
                    u_tmax_all
                    > corr_reindexed
                )
                .fillna(False)
                .sum()
            ),

            "exceed_days_qm": (
                int(
                    (
                        u_tmax_all
                        > corr_reindexed
                    )
                    .fillna(False)
                    .sum()
                )
                if USE_QM_BIAS_CORRECTION
                else np.nan
            ),
        }

        return (
            hw_mask_all,
            hw_thr_series,
            threshold_info,
            None,
        )

    # ============================================================
    # B. Absolute-threshold sensitivity scenarios
    # ============================================================
    elif hw_method == "absolute":

        abs_mode = str(
            scenario.get("abs_mode", "")
        ).strip().lower()

        if abs_mode == "fixed":
            abs_thr = float(
                scenario["abs_threshold"]
            )
        else:
            # Preserve the existing continent-dependent fallback behavior.
            abs_thr = float(
                get_absolute_hw_threshold(
                    continent
                )
            )

        # Full-year detection is retained for diagnostics only.
        hw_mask_all = detect_heatwave_absolute(
            u_tmax_all,
            abs_thr,
        )

        threshold_info = {
            "hw_threshold_mean": abs_thr,

            "hw_threshold_raw_mean": np.nan,

            "hw_threshold_corrected_mean": np.nan,

            "hw_threshold_qm_mean": np.nan,

            "bias_correction_method": (
                "not_applicable_absolute_threshold"
            ),

            "era5_isd_bias_annual": np.nan,

            "n_valid_ref_days": np.nan,

            "qm_ref_mode": "",

            "qm_n_quantiles": np.nan,

            "qm_monthly_sample_counts": "",

            "qm_monthly_methods": "",

            "isd_tmax_mean_overlap": np.nan,

            "era5_tmax_mean_overlap": np.nan,

            "era5_isd_overlap_days": 0,

            "exceed_days_raw": np.nan,

            "exceed_days_corrected": int(
                (
                    u_tmax_all
                    > abs_thr
                )
                .fillna(False)
                .sum()
            ),

            "exceed_days_qm": np.nan,
        }

        # Expand the scalar threshold to a date-indexed Series so that
        # percentile and absolute scenarios share the same downstream
        # warm-season grouped-detection interface.
        threshold_series = pd.Series(
            abs_thr,
            index=u_tmax_all.index,
            dtype=float,
        )

        return (
            hw_mask_all,
            threshold_series,
            threshold_info,
            None,
        )

    # ============================================================
    # C. Unsupported or misspelled method
    # ============================================================
    else:
        return None, None, None, {
            "fail_step": "unsupported_hw_method",
            "missing_data": (
                "Unsupported hw_method="
                f"{scenario.get('hw_method')!r}; "
                "expected 'percentile' or 'absolute'."
            ),
        }

def build_period_records(
        scenario: dict,
        pair_id,
        row,
        group,
        continent,
        combined_u_all,
        combined_r_all,
        combined_u_warm,
        combined_r_warm,
        has_warm,
        hw_dates_warm,
        nhw_dates_warm,
        hw_thr_mean,
        hw_dates_all,
        nhw_dates_all,
        hw_intensity,
        delta_tmax_ann,
        regime_info,
        valid_yrs_u_all,
        valid_yrs_r_all,
        valid_yrs_u_warm,
        valid_yrs_r_warm,
        bv_u,
        bv_r,
        bv_delta,
        station_meta_vals,
        kg_code: str = "",
        kg_group_val: str = "",
        climate_zone_main: str = "",
) -> list:
    if regime_info is None:
        regime_info = {}

    periods = {"annual": (combined_u_all, combined_r_all)}

    if has_warm:
        periods["warm_season"] = (combined_u_warm, combined_r_warm)

    if has_warm and len(hw_dates_warm) > 0:
        periods["heatwave"] = (
            subset_by_dates(combined_u_warm, hw_dates_warm),
            subset_by_dates(combined_r_warm, hw_dates_warm),
        )

    if has_warm and ((not RESTRICT_NON_HW_TO_HW_PAIRS) or len(hw_dates_warm) > 0):
        periods["non_heatwave"] = (
            subset_by_dates(combined_u_warm, nhw_dates_warm),
            subset_by_dates(combined_r_warm, nhw_dates_warm),
        )

    records = []
    for period_name, (sub_u, sub_r) in periods.items():
        if sub_u is None or sub_r is None:
            continue
        if len(sub_u) == 0 or len(sub_r) == 0:
            continue

        fft_u = compute_fft(sub_u)
        fft_r = compute_fft(sub_r)
        if fft_u is None or fft_r is None:
            continue

        u_curve = reconstruct_diurnal(
            fft_u["mean"], fft_u["amplitudes"], fft_u["phases"]
        )
        r_curve = reconstruct_diurnal(
            fft_r["mean"], fft_r["amplitudes"], fft_r["phases"]
        )

        data_source = "full-year" if period_name == "annual" else "warm-season"

        urban_phase1 = fft_u["phases"][0] if len(fft_u["phases"]) > 0 else np.nan
        rural_phase1 = fft_r["phases"][0] if len(fft_r["phases"]) > 0 else np.nan
        urban_phase1_peak_hour = phase_to_peak_hour(urban_phase1, harmonic=1)
        rural_phase1_peak_hour = phase_to_peak_hour(rural_phase1, harmonic=1)
        dphase1_rad = wrap_angle_rad(urban_phase1 - rural_phase1) if pd.notna(urban_phase1) and pd.notna(rural_phase1) else np.nan

        rec = {
            "scenario_id": scenario["scenario_id"],
            **_scenario_year_fields(scenario),
            "data_quality_min_year_valid_frac": scenario["data_quality_min_year_valid_frac"],
            "warm_min_year_valid_frac": scenario["warm_min_year_valid_frac"],
            "full_year_min_valid_frac": scenario["full_year_min_valid_frac"],
            "apply_data_quality_to_full_year": scenario["apply_data_quality_to_full_year"],
            "hw_def": scenario["hw_def"],
            "hemisphere": hemisphere_from_lat(float(row["lat_urban"])),
            "warm_season_label": warm_season_label_for_lat(float(row["lat_urban"])),
            "warm_season_months": warm_months_string(float(row["lat_urban"])),
            "warm_season_reference": "urban_station_latitude_pair_calendar",
            "hw_method": scenario["hw_method"],
            "hw_percentile": scenario["hw_percentile"],
            "hw_abs_mode": scenario["abs_mode"],
            "hw_abs_threshold": scenario["abs_threshold"],
            "hw_min_days": scenario["hw_min_days"],
            "hw_window_half": scenario["hw_window_half"],
            "hne_ref_mode": HNE_REF_MODE,
            "hw_ref_mode": scenario["hw_ref_mode"],
            "hemisphere": hemisphere_from_lat(float(row["lat_urban"])),
            "warm_season_label": warm_season_label_for_lat(float(row["lat_urban"])),
            "warm_season_months": warm_months_string(float(row["lat_urban"])),
            "warm_season_reference": "urban_station_latitude_pair_calendar",

            "pair_id": pair_id,
            "period": period_name,
            "data_source": data_source,
            "group": group,
            # Fixed manuscript-level UHI/UCI regime:
            # annual two-harmonic synthesised Tx urban-minus-rural.
            "uhi_definition": regime_info.get("regime_definition", REGIME_DEFINITION),
            "uhi_classification_metric": regime_info.get("regime_classification_metric", REGIME_CLASSIFICATION_METRIC),
            "regime_reference_period": regime_info.get("regime_reference_period", "annual"),
            "regime_temperature_source": regime_info.get("regime_temperature_source", "two_harmonic_synthesised_diurnal_cycle"),
            "continent": continent,
            "kg_code": kg_code,
            "kg_group": kg_group_val,
            "climate_zone_main": climate_zone_main,
            "lat_group": row.get("lat_group", np.nan),

            "urban_lcz_raw": row["urban_lcz"],
            "urban_lcz_corrected": row["urban_lcz_corrected"],
            "urban_lcz_class": row["urban_lcz_class"],
            "lon_urban": float(row["lon_urban"]),
            "lat_urban": float(row["lat_urban"]),
            "lon_rural": float(row["lon_rural"]),
            "lat_rural": float(row["lat_rural"]),

            "hw_threshold_mean": hw_thr_mean,
            "n_hw_days_annual": len(hw_dates_all),
            "n_nhw_days_annual": len(nhw_dates_all),
            "n_hw_days_warm": len(hw_dates_warm),
            "n_nhw_days_warm": len(nhw_dates_warm),
            "hw_intensity": hw_intensity,
            # Backward-compatible alias: now equals annual two-harmonic synthesised ΔTx.
            "delta_tmax_ann": delta_tmax_ann,
            "delta_tx_annual_synth": regime_info.get("delta_tx_annual_synth", np.nan),
            "urban_tx_annual_synth": regime_info.get("urban_tx_annual_synth", np.nan),
            "rural_tx_annual_synth": regime_info.get("rural_tx_annual_synth", np.nan),
            "urban_tx_hour_annual_synth": regime_info.get("urban_tx_hour_annual_synth", np.nan),
            "rural_tx_hour_annual_synth": regime_info.get("rural_tx_hour_annual_synth", np.nan),
            "delta_tn_annual_synth": regime_info.get("delta_tn_annual_synth", np.nan),
            "delta_tmax_daily_ann_mean": regime_info.get("delta_tmax_daily_ann_mean", np.nan),

            "n_valid_years_urban_all": len(valid_yrs_u_all),
            "n_valid_years_rural_all": len(valid_yrs_r_all),
            "n_valid_years_urban_warm": len(valid_yrs_u_warm) if has_warm else 0,
            "n_valid_years_rural_warm": len(valid_yrs_r_warm) if has_warm else 0,

            "urban_BV_m3": bv_u,
            "rural_BV_m3": bv_r,
            "delta_BV_m3": bv_delta,
            "log10_BV": (
                np.log10(bv_u)
                if (not np.isnan(bv_u) and bv_u > 0) else np.nan
            ),

            "urban_Tmean": fft_u["mean"],
            "urban_Tmax_fft": fft_u["Tmax_fft"],
            "urban_Tmin_fft": fft_u["Tmin_fft"],
            "urban_ndays": fft_u["n_days"],
            "rural_Tmean": fft_r["mean"],
            "rural_Tmax_fft": fft_r["Tmax_fft"],
            "rural_Tmin_fft": fft_r["Tmin_fft"],
            "rural_ndays": fft_r["n_days"],
            "dTmean": fft_u["mean"] - fft_r["mean"],
            "dAmp1": fft_u["amplitudes"][0] - fft_r["amplitudes"][0],
            "dAmp2": (
                fft_u["amplitudes"][1] - fft_r["amplitudes"][1]
                if len(fft_u["amplitudes"]) > 1 else np.nan
            ),
            "dTx": fft_u["Tmax_fft"] - fft_r["Tmax_fft"],
            "dTn": fft_u["Tmin_fft"] - fft_r["Tmin_fft"],

            "urban_phase1_peak_hour": urban_phase1_peak_hour,
            "rural_phase1_peak_hour": rural_phase1_peak_hour,
            "dPhase1_rad": dphase1_rad,
            "dPhase1_hours": float(dphase1_rad / (2 * np.pi) * 24.0) if pd.notna(dphase1_rad) else np.nan,
            "phase1_peak_hour_diff": circular_hour_diff(
                urban_phase1_peak_hour,
                rural_phase1_peak_hour
            ),
        }

        rec.update(curve_delta_metrics(u_curve, r_curve))
        rec.update(compute_daily_risk_metrics(sub_u, sub_r))
        rec.update(station_meta_vals)

        for k in range(N_HARMONICS):
            rec[f"urban_Amp{k + 1}"] = (
                fft_u["amplitudes"][k] if k < len(fft_u["amplitudes"]) else np.nan
            )
            rec[f"urban_Phase{k + 1}"] = (
                fft_u["phases"][k] if k < len(fft_u["phases"]) else np.nan
            )
            rec[f"rural_Amp{k + 1}"] = (
                fft_r["amplitudes"][k] if k < len(fft_r["amplitudes"]) else np.nan
            )
            rec[f"rural_Phase{k + 1}"] = (
                fft_r["phases"][k] if k < len(fft_r["phases"]) else np.nan
            )

        for h in range(24):
            rec[f"urban_diurnal_h{h:02d}"] = float(u_curve[h])
            rec[f"rural_diurnal_h{h:02d}"] = float(r_curve[h])
            rec[f"delta_diurnal_h{h:02d}"] = float(u_curve[h] - r_curve[h])

        for h in range(24):
            u_dew = fft_u["hourly_dew"][h] if fft_u.get("hourly_dew") is not None else np.nan
            r_dew = fft_r["hourly_dew"][h] if fft_r.get("hourly_dew") is not None else np.nan
            rec[f"urban_dew_h{h:02d}"] = float(u_dew) if not np.isnan(u_dew) else np.nan
            rec[f"rural_dew_h{h:02d}"] = float(r_dew) if not np.isnan(r_dew) else np.nan

        records.append(rec)

    return records


def _error_with_scenario(error_info, scenario, analysis_years=None):
    """Attach scenario metadata to an error record without changing the error cause."""
    if error_info is None:
        return None
    years = analysis_years if analysis_years is not None else scenario.get("analysis_years", YEARS)
    for k, v in _scenario_year_fields(scenario, years).items():
        error_info.setdefault(k, v)
    for k in [
        "data_quality_min_year_valid_frac",
        "warm_min_year_valid_frac",
        "full_year_min_valid_frac",
        "hw_def",
        "hw_method",
        "hw_percentile",
        "abs_mode",
        "abs_threshold",
    ]:
        if k in scenario:
            out_key = "hw_abs_mode" if k == "abs_mode" else "hw_abs_threshold" if k == "abs_threshold" else k
            error_info.setdefault(out_key, scenario[k])
    return error_info


def process_single_pair_scenario(args):
    scenario, idx, row = args
    pair_id = row["pair_id"]
    lon_u = float(row["lon_urban"])
    lat_u = float(row["lat_urban"])
    lon_r = float(row["lon_rural"])
    lat_r = float(row["lat_rural"])

    # Scenario-specific ISD years.
    # Original branch uses 2015-2024.
    # ISD year-window branch uses 1/3/5/10 latest-year windows.
    analysis_years = sorted(int(y) for y in scenario.get("analysis_years", YEARS))

    records_out = []
    threshold_out = []
    station_year_out = []
    error_info = None

    try:
        usaf_u, wban_u, usaf_r, wban_r = parse_pair_id(pair_id)
    except ValueError as e:
        error_info = {
            "scenario_id": scenario["scenario_id"],
            "pair_id": pair_id,
            "fail_step": "parse_pair_id",
            "missing_data": f"ID解析失败: {e}",
        }
        return records_out, threshold_out, station_year_out, error_info

    combined_u_all, valid_yrs_u_all, native_u = load_multiyear_station_ALL(
        usaf_u,
        wban_u,
        analysis_years,
        ISD_BASE_DIR,
        lon_u,
        min_valid_frac=scenario["full_year_min_valid_frac"],
        verbose=False
    )
    combined_r_all, valid_yrs_r_all, native_r = load_multiyear_station_ALL(
        usaf_r,
        wban_r,
        analysis_years,
        ISD_BASE_DIR,
        lon_r,
        min_valid_frac=scenario["full_year_min_valid_frac"],
        verbose=False
    )

    if combined_u_all is None or combined_r_all is None:
        missing = []
        if combined_u_all is None:
            missing.append("urban_full_year")
        if combined_r_all is None:
            missing.append("rural_full_year")
        error_info = {
            "scenario_id": scenario["scenario_id"],
            "pair_id": pair_id,
            "fail_step": "load_multiyear_station_ALL",
            "missing_data": " & ".join(missing),
        }
        return records_out, threshold_out, station_year_out, error_info

    # Use the urban station latitude as the pair-level warm-season calendar.
    # This keeps urban and rural slices exactly synchronized.
    pair_hemisphere = hemisphere_from_lat(lat_u)
    pair_warm_label = warm_season_label_for_lat(lat_u)
    pair_warm_months = warm_months_string(lat_u)

    combined_u_warm, valid_yrs_u_warm = slice_warm_season(
        combined_u_all,
        lat=lat_u,
        min_year_valid_frac=scenario["warm_min_year_valid_frac"],
        years=analysis_years,
    )
    combined_r_warm, valid_yrs_r_warm = slice_warm_season(
        combined_r_all,
        lat=lat_u,
        min_year_valid_frac=scenario["warm_min_year_valid_frac"],
        years=analysis_years,
    )
    has_warm = (combined_u_warm is not None and combined_r_warm is not None)

    n_total = len(analysis_years)
    for station_type, usaf_id, wban_id, yrs_all, yrs_warm, native_map in [
        ("urban", usaf_u, wban_u, valid_yrs_u_all, valid_yrs_u_warm if has_warm else [], native_u),
        ("rural", usaf_r, wban_r, valid_yrs_r_all, valid_yrs_r_warm if has_warm else [], native_r),
    ]:
        native_values = [native_map.get(y, "missing") for y in analysis_years]
        station_year_out.append({
            "scenario_id": scenario["scenario_id"],
            **_scenario_year_fields(scenario, analysis_years),
            "data_quality_min_year_valid_frac": scenario["data_quality_min_year_valid_frac"],
            "hw_def": scenario["hw_def"],
            "pair_id": pair_id,
            "station_type": station_type,
            "usaf": usaf_id,
            "wban": wban_id,
            "valid_years_all": ",".join(map(str, yrs_all)),
            "n_valid_years_all": len(yrs_all),
            "valid_frac_all": round(len(yrs_all) / n_total, 4),
            "valid_years_warm": ",".join(map(str, yrs_warm)),
            "n_valid_years_warm": len(yrs_warm),
            "valid_frac_warm": round(len(yrs_warm) / n_total, 4),
            "native_resolution_by_year": ";".join(
                [f"{y}:{native_map.get(y, 'missing')}" for y in analysis_years]
            ),
            "native_resolution_summary": ";".join(
                sorted(set([v for v in native_values if v != "missing"]))
            ),
        })

    if not has_warm:
        error_info = {
            "scenario_id": scenario["scenario_id"],
            "pair_id": pair_id,
            "fail_step": "slice_warm_season",
            "missing_data": "城市或乡村JJA有效年份不足",
        }
        return records_out, threshold_out, station_year_out, error_info

    u_tmin_all, u_tmax_all = compute_daily_minmax(combined_u_all)
    r_tmin_all, r_tmax_all = compute_daily_minmax(combined_r_all)
    common_dates_all = u_tmax_all.index.intersection(r_tmax_all.index)

    if len(common_dates_all) == 0:
        error_info = {
            "scenario_id": scenario["scenario_id"],
            "pair_id": pair_id,
            "fail_step": "compute_daily_minmax",
            "missing_data": "城乡站点没有共同的Tmax有效日期",
        }
        return records_out, threshold_out, station_year_out, error_info

    # Diagnostic only: raw daily Tmax contrast. Do not use this to define UHI/UCI.
    delta_tmax_daily_ann_mean = float(
        (u_tmax_all[common_dates_all] - r_tmax_all[common_dates_all]).mean()
    )

    regime_info = classify_uhi_uci_regime_from_annual_harmonic_tx(
        combined_u_all, combined_r_all
    )
    if regime_info is None:
        error_info = {
            "scenario_id": scenario["scenario_id"],
            "pair_id": pair_id,
            "fail_step": "classify_uhi_uci_regime_from_annual_harmonic_tx",
            "missing_data": "annual two-harmonic synthesised Tx could not be reconstructed",
        }
        return records_out, threshold_out, station_year_out, error_info

    regime_info["delta_tmax_daily_ann_mean"] = delta_tmax_daily_ann_mean
    group = regime_info["group"]
    # Backward-compatible variable name used by downstream tables; value is now synthesised ΔTx.
    delta_tmax_ann = regime_info["delta_tx_annual_synth"]

    continent = _get_continent(lon_u, lat_u)
    kg_code_val, kg_group_val, climate_zone_val = get_koppen_info(lon_u, lat_u)

    hw_mask_all, hw_thr_series, threshold_info, hw_error = prepare_heatwave_mask_and_threshold(
        scenario=scenario,
        u_tmax_all=u_tmax_all,
        usaf_u=usaf_u,
        wban_u=wban_u,
        continent=continent,
    )
    if hw_error is not None:
        error_info = {
            "scenario_id": scenario["scenario_id"],
            "pair_id": pair_id,
            **hw_error,
        }
        return records_out, threshold_out, station_year_out, error_info

    # Full-year event detection is retained only for annual diagnostics and
    # backward-compatible annual metadata.
    hw_dates_all = set(
        hw_mask_all[
            hw_mask_all
        ].index.tolist()
    )

    nhw_dates_all = set(
        hw_mask_all[
            ~hw_mask_all
        ].index.tolist()
    )

    # Pair-specific warm-season common dates.
    # The urban-station latitude defines the pair-level warm-season calendar,
    # consistently with slice_warm_season().
    warm_dates_u = set(
        combined_u_warm[
            "local_date"
        ].unique()
    )

    warm_dates_r = set(
        combined_r_warm[
            "local_date"
        ].unique()
    )

    warm_common = (
        warm_dates_u
        & warm_dates_r
    )

    # Main sensitivity-analysis warm-season HW/NHW detection.
    #
    # Important:
    #   Do not derive warm-season HW dates by intersecting the full-year event
    #   mask with warm_common. Consecutive-day detection must be performed inside
    #   each warm_season_year, following the same NH-JJA / SH-DJF boundary rule
    #   as the main analysis.
    if len(warm_common) > 0:
        warm_index = pd.Index(
            sorted(warm_common)
        )

        # Use only warm-season common dates with finite urban daily Tmax.
        u_tmax_warm_common = (
            u_tmax_all
            .reindex(warm_index)
            .dropna()
        )

        # hw_thr_series is:
        #   - a date-indexed percentile threshold for P85/P90/P95; or
        #   - a constant-valued Series for the absolute-threshold scenario.
        hw_thr_warm_common = (
            pd.Series(hw_thr_series)
            .reindex(
                u_tmax_warm_common.index
            )
        )

        hw_mask_warm = (
            detect_heatwave_warm_season_by_year(
                daily_tmax=u_tmax_warm_common,
                threshold=hw_thr_warm_common,
                lat=lat_u,
                years=analysis_years,
            )
        )

        hw_dates_warm = set(
            hw_mask_warm[
                hw_mask_warm
            ].index.tolist()
        )

        nhw_dates_warm = set(
            hw_mask_warm[
                ~hw_mask_warm
            ].index.tolist()
        )

    else:
        hw_mask_warm = pd.Series(
            dtype=bool
        )

        hw_dates_warm = set()
        nhw_dates_warm = set()

    # Required filtering rule: keep only pairs with both HW and NHW dates.
    if len(hw_dates_warm) == 0 or len(nhw_dates_warm) == 0:
        error_info = {
            "scenario_id": scenario["scenario_id"],
            "pair_id": pair_id,
            "fail_step": "require_hw_and_nhw_dates",
            "missing_data": (
                f"HW或NHW日期为空: n_hw_days_warm={len(hw_dates_warm)}, "
                f"n_nhw_days_warm={len(nhw_dates_warm)}"
            ),
        }
        return records_out, threshold_out, station_year_out, error_info

    hw_thr_mean = threshold_info["hw_threshold_mean"]

    hw_intensity = float(
        (
            u_tmax_all[hw_mask_all]
            - pd.Series(hw_thr_series).reindex(u_tmax_all.index)[hw_mask_all]
        ).mean()
    ) if hw_mask_all.any() else np.nan

    threshold_out.append({
        "scenario_id": scenario["scenario_id"],
        **_scenario_year_fields(scenario, analysis_years),
        "data_quality_min_year_valid_frac": scenario["data_quality_min_year_valid_frac"],
        "warm_min_year_valid_frac": scenario["warm_min_year_valid_frac"],
        "full_year_min_valid_frac": scenario["full_year_min_valid_frac"],
        "hw_def": scenario["hw_def"],
        "hemisphere": pair_hemisphere,
        "warm_season_label": pair_warm_label,
        "warm_season_months": pair_warm_months,
        "warm_season_reference": "urban_station_latitude_pair_calendar",
        "hw_method": scenario["hw_method"],
        "hw_percentile": scenario["hw_percentile"],
        "hw_abs_mode": scenario["abs_mode"],
        "hw_abs_threshold": scenario["abs_threshold"],
        "hw_ref_mode": scenario["hw_ref_mode"],
        "pair_id": pair_id,
        "urban_usaf": usaf_u,
        "urban_wban": wban_u,
        "urban_lon": lon_u,
        "urban_lat": lat_u,
        "continent": continent,
        "kg_code": kg_code_val,
        "kg_group": kg_group_val,
        "group": group,
        "uhi_definition": regime_info.get("regime_definition", REGIME_DEFINITION),
        "uhi_classification_metric": regime_info.get("regime_classification_metric", REGIME_CLASSIFICATION_METRIC),
        "delta_tmax_ann": delta_tmax_ann,
        "delta_tx_annual_synth": regime_info.get("delta_tx_annual_synth", np.nan),
        "delta_tmax_daily_ann_mean": regime_info.get("delta_tmax_daily_ann_mean", np.nan),
        "n_hw_days_annual": len(hw_dates_all),
        "n_nhw_days_annual": len(nhw_dates_all),
        "n_hw_days_warm": len(hw_dates_warm),
        "n_nhw_days_warm": len(nhw_dates_warm),
        "hw_intensity": hw_intensity,
        "valid_urban_years_all": ",".join(map(str, valid_yrs_u_all)),
        "valid_rural_years_all": ",".join(map(str, valid_yrs_r_all)),
        "valid_urban_years_warm": ",".join(map(str, valid_yrs_u_warm)),
        "valid_rural_years_warm": ",".join(map(str, valid_yrs_r_warm)),
        **threshold_info,
    })

    station_meta_vals = get_station_meta_values(pair_id, usaf_u, wban_u, usaf_r, wban_r)

    bv_u = get_bv(usaf_u, wban_u)
    bv_r = get_bv(usaf_r, wban_r)
    bv_delta = bv_u - bv_r if (not np.isnan(bv_u) and not np.isnan(bv_r)) else np.nan

    records_out = build_period_records(
        scenario=scenario,
        pair_id=pair_id,
        row=row,
        group=group,
        continent=continent,
        combined_u_all=combined_u_all,
        combined_r_all=combined_r_all,
        combined_u_warm=combined_u_warm,
        combined_r_warm=combined_r_warm,
        has_warm=has_warm,
        hw_dates_warm=hw_dates_warm,
        nhw_dates_warm=nhw_dates_warm,
        hw_thr_mean=hw_thr_mean,
        hw_dates_all=hw_dates_all,
        nhw_dates_all=nhw_dates_all,
        hw_intensity=hw_intensity,
        delta_tmax_ann=delta_tmax_ann,
        regime_info=regime_info,
        valid_yrs_u_all=valid_yrs_u_all,
        valid_yrs_r_all=valid_yrs_r_all,
        valid_yrs_u_warm=valid_yrs_u_warm,
        valid_yrs_r_warm=valid_yrs_r_warm,
        bv_u=bv_u,
        bv_r=bv_r,
        bv_delta=bv_delta,
        station_meta_vals=station_meta_vals,
        kg_code=str(kg_code_val) if isinstance(kg_code_val, str) else "",
        kg_group_val=kg_group_val,
        climate_zone_main=climate_zone_val,
    )

    generated_periods = {r["period"] for r in records_out}
    if not {"heatwave", "non_heatwave"}.issubset(generated_periods):
        error_info = {
            "scenario_id": scenario["scenario_id"],
            "pair_id": pair_id,
            "fail_step": "require_hw_and_nhw_records",
            "missing_data": (
                "HW或NHW period record未成功生成，通常是FFT或有效小时覆盖不足"
            ),
        }
        return [], threshold_out, station_year_out, error_info

    return records_out, threshold_out, station_year_out, error_info


# ══════════════════════════════════════════════════════════════
# 8. Plotting-data output builders
# ══════════════════════════════════════════════════════════════

SCENARIO_COLS = [
    "scenario_id",
    "sensitivity_axis",

    "analysis_years",
    "isd_n_years",
    "isd_years_label",
    "isd_start_year",
    "isd_end_year",

    "data_quality_min_year_valid_frac",
    "warm_min_year_valid_frac",
    "full_year_min_valid_frac",
    "hw_def",
    "hw_method",
    "hw_percentile",
    "hw_abs_mode",
    "hw_abs_threshold",
    "hw_min_days",
    "hw_window_half",
    "hw_ref_mode",
]


def make_diurnal_long(all_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        c for c in (
            SCENARIO_COLS
            + [
                "pair_id", "period", "group", "continent",
                "kg_code", "kg_group", "climate_zone_main",
                "lat_group",
                "n_hw_days_warm", "n_nhw_days_warm",
            ]
        )
        if c in all_df.columns
    ]

    rows = []
    for _, row in all_df.iterrows():
        base = {c: row[c] for c in id_cols}
        for h in range(24):
            rows.append({
                **base,
                "hour": h,
                "urban_temp": row.get(f"urban_diurnal_h{h:02d}", np.nan),
                "rural_temp": row.get(f"rural_diurnal_h{h:02d}", np.nan),
                "delta_temp": row.get(f"delta_diurnal_h{h:02d}", np.nan),
                "urban_dewpoint": row.get(f"urban_dew_h{h:02d}", np.nan),
                "rural_dewpoint": row.get(f"rural_dew_h{h:02d}", np.nan),
            })
    return pd.DataFrame(rows)


def summarize_diurnal_group_mean(diurnal_long: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        c for c in (
            SCENARIO_COLS
            + [
                "period", "group", "hour",
                "kg_group", "climate_zone_main",
            ]
        )
        if c in diurnal_long.columns
    ]

    value_cols = ["urban_temp", "rural_temp", "delta_temp", "urban_dewpoint", "rural_dewpoint"]
    rows = []

    for key, g in diurnal_long.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n_pairs"] = int(g["pair_id"].nunique()) if "pair_id" in g.columns else int(len(g))

        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{col}_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            row[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{col}_q25"] = float(vals.quantile(0.25)) if len(vals) else np.nan
            row[f"{col}_q75"] = float(vals.quantile(0.75)) if len(vals) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def make_metric_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "dTmean", "dAmp1", "dAmp2", "dTx", "dTn",
        "dT_day_mean", "dT_night_mean",
        "delta_peak_hour", "delta_min_hour",
        "delta_peak_value", "delta_min_value",
        "dPhase1_rad", "dPhase1_hours", "phase1_peak_hour_diff",
        "delta_tropical_night_freq", "delta_hotnight_excess",
        "delta_daily_tmin_mean",
        "daytime_uci_flag", "nighttime_uhi_flag",
        "hw_intensity", "n_hw_days_warm", "n_nhw_days_warm",
    ]
    metrics = [m for m in metrics if m in all_df.columns]

    group_cols = [
        c for c in (
            SCENARIO_COLS
            + ["period", "group", "kg_group", "climate_zone_main"]
        )
        if c in all_df.columns
    ]

    rows = []
    for key, g in all_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        base["n_pairs"] = int(g["pair_id"].nunique())

        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            mean, lo, hi = bootstrap_mean_ci(vals)
            rows.append({
                **base,
                "metric": m,
                "n_nonmissing": int(len(vals)),
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "median": float(vals.median()) if len(vals) else np.nan,
                "q25": float(vals.quantile(0.25)) if len(vals) else np.nan,
                "q75": float(vals.quantile(0.75)) if len(vals) else np.nan,
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            })

    return pd.DataFrame(rows)


def make_uhi_uci_distribution(all_df: pd.DataFrame) -> pd.DataFrame:
    annual = all_df[all_df["period"] == "annual"].copy()
    group_cols = [c for c in SCENARIO_COLS if c in annual.columns]
    rows = []

    for key, g in annual.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        total_pairs = int(g["pair_id"].nunique())
        for grp in ["UHI", "UCI"]:
            n = int(g.loc[g["group"] == grp, "pair_id"].nunique())
            rows.append({
                **base,
                "group": grp,
                "n_pairs": n,
                "total_pairs": total_pairs,
                "proportion": float(n / total_pairs) if total_pairs else np.nan,
            })
    return pd.DataFrame(rows)


def make_period_sample_counts(all_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in (SCENARIO_COLS + ["period", "group"]) if c in all_df.columns]
    rows = []
    for key, g in all_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({
            **dict(zip(group_cols, key)),
            "n_pairs": int(g["pair_id"].nunique()),
            "n_records": int(len(g)),
        })
    return pd.DataFrame(rows)


def make_hw_nhw_pairwise_metric_difference(all_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "dTmean", "dAmp1", "dAmp2", "dTx", "dTn",
        "dT_day_mean", "dT_night_mean",
        "delta_peak_hour", "delta_min_hour",
        "delta_peak_value", "delta_min_value",
        "dPhase1_rad", "dPhase1_hours", "phase1_peak_hour_diff",
        "delta_tropical_night_freq", "delta_hotnight_excess",
        "delta_daily_tmin_mean",
    ]
    metrics = [m for m in metrics if m in all_df.columns]

    id_cols = [
        c for c in (
            SCENARIO_COLS
            + [
                "pair_id", "group", "continent",
                "kg_code", "kg_group", "climate_zone_main",
                "lat_group", "n_hw_days_warm", "n_nhw_days_warm"
            ]
        )
        if c in all_df.columns
    ]

    sub = all_df[all_df["period"].isin(["heatwave", "non_heatwave"])].copy()
    rows = []

    for key, g in sub.groupby([c for c in id_cols if c != "n_hw_days_warm" and c != "n_nhw_days_warm"], dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base_cols = [c for c in id_cols if c != "n_hw_days_warm" and c != "n_nhw_days_warm"]
        base = dict(zip(base_cols, key))

        hw = g[g["period"] == "heatwave"]
        nhw = g[g["period"] == "non_heatwave"]
        if hw.empty or nhw.empty:
            continue

        row = base.copy()
        row["n_hw_days_warm"] = int(hw["n_hw_days_warm"].iloc[0]) if "n_hw_days_warm" in hw.columns else np.nan
        row["n_nhw_days_warm"] = int(hw["n_nhw_days_warm"].iloc[0]) if "n_nhw_days_warm" in hw.columns else np.nan

        for m in metrics:
            hw_v = float(hw[m].iloc[0]) if pd.notna(hw[m].iloc[0]) else np.nan
            nhw_v = float(nhw[m].iloc[0]) if pd.notna(nhw[m].iloc[0]) else np.nan
            row[f"{m}_heatwave"] = hw_v
            row[f"{m}_non_heatwave"] = nhw_v
            row[f"{m}_hw_minus_nhw"] = (
                hw_v - nhw_v if pd.notna(hw_v) and pd.notna(nhw_v) else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def save_output_tables(
        all_records,
        threshold_rows,
        station_year_rows,
        error_records,
        scenario_manifest):
    ensure_dir(OUTPUT_DIR)

    scenario_df = pd.DataFrame(scenario_manifest)
    scenario_df.to_csv(os.path.join(OUTPUT_DIR, "scenario_manifest.csv"), index=False)

    all_df = pd.DataFrame(all_records)
    if all_df.empty:
        print("No valid records generated in any scenario.")
        if error_records:
            pd.DataFrame(error_records).to_csv(
                os.path.join(OUTPUT_DIR, "skipped_pairs_log_by_scenario.csv"),
                index=False
            )
        return

    all_df.to_csv(
        os.path.join(OUTPUT_DIR, "all_sensitivity_pair_period_metrics.csv"),
        index=False
    )

    fft_cols = [
        "scenario_id", "data_quality_min_year_valid_frac",
        "warm_min_year_valid_frac", "full_year_min_valid_frac",
        "hw_def", "hw_method", "hw_percentile", "hw_abs_mode", "hw_abs_threshold",
        "pair_id", "period", "data_source", "group",
        "urban_Tmean", "rural_Tmean", "dTmean",
        "urban_Amp1", "rural_Amp1", "dAmp1",
        "urban_Phase1", "rural_Phase1", "dPhase1_rad", "dPhase1_hours",
        "urban_phase1_peak_hour", "rural_phase1_peak_hour", "phase1_peak_hour_diff",
        "urban_Amp2", "rural_Amp2", "dAmp2",
        "urban_Phase2", "rural_Phase2",
        "delta_peak_hour", "delta_min_hour",
        "delta_peak_value", "delta_min_value",
        "n_hw_days_warm", "n_nhw_days_warm",
    ]
    fft_cols = [c for c in fft_cols if c in all_df.columns]
    all_df[fft_cols].to_csv(
        os.path.join(OUTPUT_DIR, "station_fft_parameters_sensitivity.csv"),
        index=False
    )

    diurnal_long = make_diurnal_long(all_df)
    diurnal_long.to_csv(
        os.path.join(OUTPUT_DIR, "pair_diurnal_curves_long.csv"),
        index=False
    )

    group_mean_diurnal = summarize_diurnal_group_mean(diurnal_long)
    group_mean_diurnal.to_csv(
        os.path.join(OUTPUT_DIR, "group_mean_diurnal_curves.csv"),
        index=False
    )

    metric_summary = make_metric_summary(all_df)
    metric_summary.to_csv(
        os.path.join(OUTPUT_DIR, "metric_summary_by_scenario.csv"),
        index=False
    )

    uhi_uci_distribution = make_uhi_uci_distribution(all_df)
    uhi_uci_distribution.to_csv(
        os.path.join(OUTPUT_DIR, "uhi_uci_distribution_by_scenario.csv"),
        index=False
    )

    period_counts = make_period_sample_counts(all_df)
    period_counts.to_csv(
        os.path.join(OUTPUT_DIR, "period_sample_counts_by_scenario.csv"),
        index=False
    )

    hw_nhw_diff = make_hw_nhw_pairwise_metric_difference(all_df)
    hw_nhw_diff.to_csv(
        os.path.join(OUTPUT_DIR, "hw_nhw_pairwise_metric_difference.csv"),
        index=False
    )

    sensitivity_key_cols = [
        "scenario_id", "data_quality_min_year_valid_frac",
        "warm_min_year_valid_frac", "full_year_min_valid_frac",
        "hw_def", "hw_method", "hw_percentile", "hw_abs_mode", "hw_abs_threshold",
        "pair_id", "period", "group", "continent", "kg_code", "kg_group",
        "climate_zone_main", "lat_group",
        "n_hw_days_warm", "n_nhw_days_warm",
        "hw_threshold_mean", "hw_intensity",
        "dTmean", "dAmp1", "dAmp2", "dTx", "dTn",
        "dT_day_mean", "dT_night_mean",
        "delta_peak_hour", "delta_min_hour",
        "dPhase1_rad", "dPhase1_hours", "phase1_peak_hour_diff",
        "delta_tropical_night_freq", "delta_hotnight_excess",
        "daytime_uci_flag", "nighttime_uhi_flag",
        "urban_ndays", "rural_ndays",
    ]
    sensitivity_key_cols = [c for c in sensitivity_key_cols if c in all_df.columns]
    all_df[sensitivity_key_cols].to_csv(
        os.path.join(OUTPUT_DIR, "sensitivity_pair_key_metrics.csv"),
        index=False
    )

    if threshold_rows:
        pd.DataFrame(threshold_rows).to_csv(
            os.path.join(OUTPUT_DIR, "station_threshold_summary_by_scenario.csv"),
            index=False
        )

    if station_year_rows:
        pd.DataFrame(station_year_rows).drop_duplicates(
            subset=["scenario_id", "pair_id", "station_type"]
        ).to_csv(
            os.path.join(OUTPUT_DIR, "station_valid_years_by_scenario.csv"),
            index=False
        )

    if error_records:
        error_df = pd.DataFrame(error_records)
        error_df.to_csv(
            os.path.join(OUTPUT_DIR, "skipped_pairs_log_by_scenario.csv"),
            index=False
        )
        err_summary = (
            error_df
            .groupby(["scenario_id", "fail_step"], dropna=False)
            .size()
            .reset_index(name="n_pairs")
        )
        err_summary.to_csv(
            os.path.join(OUTPUT_DIR, "skipped_pairs_summary_by_scenario.csv"),
            index=False
        )

    output_readme = os.path.join(OUTPUT_DIR, "README_outputs.txt")
    with open(output_readme, "w", encoding="utf-8") as f:
        f.write(
            "Sensitivity analysis v6 plotting-data outputs\n"
            "================================================\n\n"
            "No figures are generated by this script.\n\n"
            "Main files:\n"
            "1. scenario_manifest.csv\n"
            "   Scenario definitions: data integrity threshold and heatwave definition.\n\n"
            "2. all_sensitivity_pair_period_metrics.csv\n"
            "   Full pair-period metrics for retained pairs only. Each scenario retains only\n"
            "   pairs with both heatwave and non_heatwave records.\n\n"
            "3. pair_diurnal_curves_long.csv\n"
            "   Long-format pair-level diurnal curves. Columns include hour, urban_temp,\n"
            "   rural_temp, delta_temp, urban_dewpoint, rural_dewpoint.\n\n"
            "4. group_mean_diurnal_curves.csv\n"
            "   Plot-ready mean diurnal curves by scenario, period, UHI/UCI group, KG group,\n"
            "   and hour. This is the main data file for HW/NHW urban-rural mean comparison.\n\n"
            "5. uhi_uci_distribution_by_scenario.csv\n"
            "   UHI/UCI sample counts and proportions by scenario.\n\n"
            "6. period_sample_counts_by_scenario.csv\n"
            "   Retained sample counts by scenario, period, and UHI/UCI group.\n\n"
            "7. metric_summary_by_scenario.csv\n"
            "   Summary statistics and bootstrap CI for dAmp1, dTx, dTn, dT_day_mean,\n"
            "   dT_night_mean, phase/peak-hour diagnostics, and risk metrics.\n\n"
            "8. hw_nhw_pairwise_metric_difference.csv\n"
            "   Pairwise heatwave minus non_heatwave differences.\n\n"
            "9. station_threshold_summary_by_scenario.csv\n"
            "   Heatwave threshold and ERA5->ISD quantile-mapping diagnostics by scenario.\n\n"
            "10. station_valid_years_by_scenario.csv\n"
            "    Full-year and warm-season valid-year coverage, including native hourly/\n"
            "    3-hourly resolution diagnostics.\n\n"
            "11. skipped_pairs_log_by_scenario.csv / skipped_pairs_summary_by_scenario.csv\n"
            "    Dropped-pair diagnostics.\n"
        )

    print("\nSaved output files:")
    for fname in [
        "scenario_manifest.csv",
        "all_sensitivity_pair_period_metrics.csv",
        "station_fft_parameters_sensitivity.csv",
        "pair_diurnal_curves_long.csv",
        "group_mean_diurnal_curves.csv",
        "metric_summary_by_scenario.csv",
        "uhi_uci_distribution_by_scenario.csv",
        "period_sample_counts_by_scenario.csv",
        "hw_nhw_pairwise_metric_difference.csv",
        "sensitivity_pair_key_metrics.csv",
        "station_threshold_summary_by_scenario.csv",
        "station_valid_years_by_scenario.csv",
        "skipped_pairs_log_by_scenario.csv",
        "skipped_pairs_summary_by_scenario.csv",
        "README_outputs.txt",
    ]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"  - {path}")


# ══════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("Sensitivity Analysis v7: Data integrity x Heatwave definitions")
    print(f"YEARS={YEARS[0]}-{YEARS[-1]}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Data integrity settings: {DATA_VALID_FRAC_SETTINGS}")
    print("Heatwave settings:")
    for s in HEATWAVE_DEFINITION_SETTINGS:
        print(f"  - {s['hw_def']}")
    if ENABLE_ISD_YEAR_WINDOW_SENSITIVITY:
        print("ISD year-window sensitivity: 1/3/5/10 years under valid=0.80 + P90 only")
    print(f"HW logic: Tmax > threshold for >= {HW_MIN_DAYS} consecutive days")
    print("UHI/UCI grouping: mean urban-minus-rural daily Tmax; UHI if > 0, else UCI")
    print("NHW logic: JJA common dates excluding HW dates")
    print("FFT logic: unchanged from original script")
    print("=" * 78)

    ensure_dir(OUTPUT_DIR)

    print(f"\nReading pair table: {PAIR_CSV_PATH}")
    pair_df = pd.read_csv(PAIR_CSV_PATH)
    pair_df["urban_lcz_corrected"] = pair_df["urban_lcz"].apply(normalize_lcz)
    pair_df["urban_lcz_class"] = pair_df["urban_lcz_corrected"].apply(
        lcz_compactness_class
    )
    if "lat_group" not in pair_df.columns:
        pair_df["lat_group"] = pair_df["lat_urban"].apply(assign_lat_group)

    total_pairs = len(pair_df)
    print(f"  Total pairs: {total_pairs}")

    print("Loading continents / metadata / BV ...")
    _load_continents()
    load_station_meta()
    load_pair_fallback()
    _load_bv()
    print("  Done")

    scenarios = make_scenarios()
    scenario_manifest = []
    for s in scenarios:
        row = dict(s)
        row["analysis_years"] = _analysis_years_csv(row.get("analysis_years", YEARS))
        scenario_manifest.append(row)

    pd.DataFrame(scenario_manifest).to_csv(
        os.path.join(OUTPUT_DIR, "scenario_manifest.csv"),
        index=False
    )

    all_records = []
    threshold_rows = []
    station_year_rows = []
    error_records = []

    n_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"\nStarting sensitivity loops with {n_cores} CPU cores.")

    for si, scenario in enumerate(scenarios, start=1):
        print("\n" + "-" * 78)
        print(
            f"[{si}/{len(scenarios)}] Scenario: {scenario['scenario_id']} | "
            f"axis={scenario.get('sensitivity_axis', 'data_quality_hw_definition')} | "
            f"years={scenario.get('isd_years_label', _years_label(scenario.get('analysis_years', YEARS)))} | "
            f"warm_valid>={scenario['warm_min_year_valid_frac']:.0%} | "
            f"full_year_valid>={scenario['full_year_min_valid_frac']:.0%} | "
            f"HW={scenario['hw_def']}"
        )
        print("-" * 78)

        scenario_records = []
        scenario_thresholds = []
        scenario_station_years = []
        scenario_errors = []

        tasks = [(scenario, idx, row) for idx, row in pair_df.iterrows()]

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = {
                executor.submit(process_single_pair_scenario, args): args
                for args in tasks
            }

            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    rec_out, thr_out, sy_out, err_out = future.result()
                    if rec_out:
                        scenario_records.extend(rec_out)
                    if thr_out:
                        scenario_thresholds.extend(thr_out)
                    if sy_out:
                        scenario_station_years.extend(sy_out)
                    if err_out is not None:
                        scenario_errors.append(
                            _error_with_scenario(
                                err_out,
                                scenario,
                                scenario.get("analysis_years", YEARS)
                            )
                        )
                except Exception as exc:
                    failed_args = futures[future]
                    failed_row = failed_args[2]
                    scenario_errors.append(_error_with_scenario({
                        "scenario_id": scenario["scenario_id"],
                        "pair_id": failed_row["pair_id"],
                        "fail_step": "Process Exception",
                        "missing_data": f"进程崩溃: {exc}",
                    }, scenario, scenario.get("analysis_years", YEARS)))

                if done % 200 == 0 or done == total_pairs:
                    print(
                        f"  progress: {done}/{total_pairs}, "
                        f"kept_records={len(scenario_records)}, "
                        f"errors={len(scenario_errors)}"
                    )

        # Final safety filter per scenario: retain only pair_ids with HW and NHW.
        if scenario_records:
            tmp = pd.DataFrame(scenario_records)
            complete_pair_ids = []
            for pid, g in tmp.groupby("pair_id"):
                periods = set(g["period"].unique())
                if {"heatwave", "non_heatwave"}.issubset(periods):
                    complete_pair_ids.append(pid)

            tmp = tmp[tmp["pair_id"].isin(complete_pair_ids)].copy()
            kept_pair_ids = set(complete_pair_ids)
            scenario_records = tmp.to_dict("records")

            scenario_thresholds = [
                r for r in scenario_thresholds
                if r.get("pair_id") in kept_pair_ids
            ]
            scenario_station_years = [
                r for r in scenario_station_years
                if r.get("pair_id") in kept_pair_ids
            ]

            print(
                f"Scenario kept pairs: {len(kept_pair_ids)} | "
                f"kept records: {len(scenario_records)}"
            )

            if len(kept_pair_ids) > 0:
                dist = (
                    tmp[tmp["period"] == "annual"]
                    .groupby("group")["pair_id"]
                    .nunique()
                )
                print("UHI/UCI distribution among retained pairs:")
                for grp in ["UHI", "UCI"]:
                    n = int(dist.get(grp, 0))
                    pct = n / len(kept_pair_ids) if kept_pair_ids else np.nan
                    print(f"  {grp}: {n} ({pct:.1%})")

        else:
            kept_pair_ids = set()
            print("Scenario kept pairs: 0")

        all_records.extend(scenario_records)
        threshold_rows.extend(scenario_thresholds)
        station_year_rows.extend(scenario_station_years)
        error_records.extend(scenario_errors)

    print("\n" + "=" * 78)
    print("Saving sensitivity-analysis plotting data ...")
    save_output_tables(
        all_records=all_records,
        threshold_rows=threshold_rows,
        station_year_rows=station_year_rows,
        error_records=error_records,
        scenario_manifest=scenario_manifest,
    )

    print("\nCompleted.")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
