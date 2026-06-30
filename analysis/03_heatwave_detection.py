#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Open-source path configuration (scientific calculations below are unchanged)
# -----------------------------------------------------------------------------
from pathlib import Path as _OpenSourcePath
import sys as _open_source_sys
_OPEN_SOURCE_REPO_ROOT = _OpenSourcePath(__file__).resolve().parents[1]
if str(_OPEN_SOURCE_REPO_ROOT) not in _open_source_sys.path:
    _open_source_sys.path.insert(0, str(_OPEN_SOURCE_REPO_ROOT))
from project_paths import PATHS as _OPEN_SOURCE_PATHS
# -*- coding: utf-8 -*-
"""
heatwave_detection_all_stations.py  (v2 — with diurnal cycle reconstruction)
"""

import os
import warnings
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.fft import fft as sp_fft

import rasterio
import geopandas as gpd
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings("ignore")

# ============================================================================
# §0  路径配置
# ============================================================================
PAIR_CSV_PATH    = str(_OPEN_SOURCE_PATHS.pair_csv)
ISD_BASE_DIR     = str(_OPEN_SOURCE_PATHS.isd_dir)
ERA5_STATION_DIR = str(_OPEN_SOURCE_PATHS.era5_station_dir)
CONTINENT_SHP    = str(_OPEN_SOURCE_PATHS.continent_shp)
KG_TIF           = str(_OPEN_SOURCE_PATHS.kg_tif)
OUTPUT_DIR       = str(_OPEN_SOURCE_PATHS.heatwave_flags_output)

# ============================================================================
# §1  全局参数
# ============================================================================
YEARS                    = list(range(2015, 2025))
FULL_YEAR_MIN_VALID_FRAC = 0.80
HW_PERCENTILE            = 90
HW_WINDOW_HALF           = 7
HW_MIN_DAYS              = 3
USE_ERA5_CLIMATOLOGY     = True
MIN_LOYO_REF_YEARS       = 5
N_HARMONICS              = 2        # FFT 谐波数（与主脚本一致）
# Hemisphere-aware warm season, synced with analysis_multiyear.py
NH_WARM_MONTHS            = [6, 7, 8]
SH_WARM_MONTHS            = [12, 1, 2]
JJA_MONTHS                = NH_WARM_MONTHS  # backward-compatible default only

# analysis_multiyear.py daily HW flags; used first for urban station flags when available.
ANALYSIS_HW_FLAGS_CSV = str(_OPEN_SOURCE_PATHS.daily_hw_flags_csv)

# ── ERA5 → ISD bias correction, consistent with analysis_multiyear.py ──
HW_REF_MODE = "ERA5_longterm_Tmax_P90_DOY_±7day_quantile_mapping_corrected"

USE_QM_BIAS_CORRECTION = True
QM_N_QUANTILES = 1001
QM_MIN_OVERLAP_DAYS_PER_MONTH = 30
QM_MIN_OVERLAP_DAYS_ANNUAL = 100




# ============================================================================
# §1b  Heatwave-sync helpers
# ============================================================================
_ANALYSIS_HW_FLAGS_CACHE = None


def hemisphere_from_lat(lat) -> str:
    try:
        return "SH" if float(lat) < 0 else "NH"
    except Exception:
        return "NH"


def warm_months_for_lat(lat) -> list:
    return SH_WARM_MONTHS if hemisphere_from_lat(lat) == "SH" else NH_WARM_MONTHS


def warm_season_label_for_lat(lat) -> str:
    return "DJF" if hemisphere_from_lat(lat) == "SH" else "JJA"


def warm_months_string(lat) -> str:
    return ",".join(map(str, warm_months_for_lat(lat)))


def analysis_start_year() -> int:
    """First calendar year loaded by this script."""
    return int(min(YEARS))


def analysis_end_year() -> int:
    """Last calendar year loaded by this script."""
    return int(max(YEARS))


def is_leap_year_hw(yr: int) -> bool:
    return (yr % 4 == 0) and (yr % 100 != 0 or yr % 400 == 0)


def warm_season_year_from_date(date_value, lat) -> int:
    """
    Assign warm-season year.

    SH boundary rule for YEARS=2015-2024:
      - Jan-Feb 2015 are retained as DJF-2015.
      - Dec 2024 is excluded because DJF-2025 would need Jan-Feb 2025.
      - No attempt is made to read Dec 2014 or dates outside YEARS.
    """
    ts = pd.Timestamp(date_value)
    if hemisphere_from_lat(lat) == "SH" and ts.month == 12:
        return int(ts.year + 1)
    return int(ts.year)


def is_warm_season_date_for_analysis(date_value, lat) -> bool:
    """
    Hemisphere-aware warm-season date filter inside loaded YEARS.

    NH:
      JJA within YEARS.
    SH:
      Jan-Feb within YEARS are retained.
      Dec is retained only if the following Jan-Feb are also inside YEARS;
      therefore Dec of max(YEARS) is excluded.
    """
    ts = pd.Timestamp(date_value)
    y0 = analysis_start_year()
    y1 = analysis_end_year()

    if hemisphere_from_lat(lat) == "SH":
        if ts.month in (1, 2):
            return y0 <= ts.year <= y1
        if ts.month == 12:
            return y0 <= ts.year < y1
        return False

    return (ts.month in NH_WARM_MONTHS) and (y0 <= ts.year <= y1)


def expected_warm_season_days(season_year: int, lat) -> int:
    """
    Expected warm-season days under the same boundary rule.

    For SH first season year, only Jan-Feb are expected because Dec of the
    previous calendar year is intentionally abandoned.
    """
    season_year = int(season_year)
    if hemisphere_from_lat(lat) == "SH":
        if season_year == analysis_start_year():
            return 60 if is_leap_year_hw(season_year) else 59
        return 91 if is_leap_year_hw(season_year) else 90
    return 92

def _load_analysis_hw_flags():
    global _ANALYSIS_HW_FLAGS_CACHE
    if _ANALYSIS_HW_FLAGS_CACHE is not None:
        return _ANALYSIS_HW_FLAGS_CACHE
    if not os.path.exists(ANALYSIS_HW_FLAGS_CSV):
        _ANALYSIS_HW_FLAGS_CACHE = None
        return None
    try:
        df = pd.read_csv(ANALYSIS_HW_FLAGS_CSV)
        if "pair_id" not in df.columns or "date" not in df.columns:
            _ANALYSIS_HW_FLAGS_CACHE = None
            return None
        df["pair_id"] = df["pair_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"])
        for c in [
            "is_warm_season",
            "hw_flag_percentile_warm_season",
            "nhw_flag_percentile_warm_season",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        _ANALYSIS_HW_FLAGS_CACHE = df
        return df
    except Exception:
        _ANALYSIS_HW_FLAGS_CACHE = None
        return None


def get_analysis_hw_flags_for_pair(pair_id):
    df = _load_analysis_hw_flags()
    if df is None or len(df) == 0:
        return None
    sub = df[df["pair_id"].astype(str) == str(pair_id)].copy()
    if sub.empty:
        return None
    return sub.drop_duplicates(subset=["date"]).set_index("date").sort_index()


def align_analysis_hw_flags(analysis_hw_flags, date_index):
    if analysis_hw_flags is None or len(analysis_hw_flags) == 0:
        return None
    idx = pd.to_datetime(pd.Index(date_index), errors="coerce").normalize()
    ext = analysis_hw_flags.copy()
    ext.index = pd.to_datetime(ext.index, errors="coerce").normalize()
    aligned = ext.reindex(idx)
    needed = ["is_warm_season", "hw_flag_percentile_warm_season"]
    if not all(c in aligned.columns for c in needed):
        return None
    if aligned[needed].dropna(how="all").empty:
        return None
    return aligned


def detect_heatwave_warm_season(daily_tmax: pd.Series, threshold, lat) -> pd.Series:
    """
    Fallback HW detection when analysis_multiyear.py daily flags are unavailable.

    Same HW algorithm as the main script, with the boundary rule:
      - SH Jan-Feb of min(YEARS) are retained.
      - SH Dec of max(YEARS) is excluded.
      - Consecutive-day detection is applied within each warm_season_year.
    """
    s0 = pd.Series(daily_tmax).sort_index()
    out = pd.Series(False, index=pd.Series(daily_tmax).index)

    s = s0.dropna()
    if len(s) == 0:
        return out.astype(bool)

    warm_idx = [
        d for d in s.index
        if is_warm_season_date_for_analysis(d, lat)
    ]
    if len(warm_idx) == 0:
        return out.astype(bool)

    s_warm = s.reindex(pd.Index(sorted(warm_idx))).dropna()
    if len(s_warm) == 0:
        return out.astype(bool)

    for _, idx_dates in (
        pd.Series(s_warm.index, index=s_warm.index)
          .groupby(lambda d: warm_season_year_from_date(d, lat))
    ):
        idx_dates = pd.Index(sorted(idx_dates.tolist()))
        s_part = s_warm.reindex(idx_dates).dropna()
        if len(s_part) == 0:
            continue

        if np.isscalar(threshold):
            thr_part = threshold
        else:
            thr_part = pd.Series(threshold).reindex(s_part.index)

        hw_part = detect_heatwave_flag(s_part, thr_part)
        out.loc[hw_part.index] = hw_part.astype(bool)

    return out.fillna(False).astype(bool)

CONTINENT_THRESHOLD = {
    "Europe": 32.0, "Asia": 35.0, "North America": 35.0,
    "South America": 33.0, "Africa": 40.0,
    "Australia": 38.0, "Oceania": 38.0, "Antarctica": 20.0,
}
DEFAULT_THRESHOLD = 30.0

# ============================================================================
# §2  Köppen-Geiger
# ============================================================================
KG_INT2CODE = {
    1:"Af", 2:"Am", 3:"Aw",
    4:"BWh", 5:"BWk", 6:"BSh", 7:"BSk",
    8:"Csa", 9:"Csb", 10:"Csc",
    11:"Cwa", 12:"Cwb", 13:"Cwc",
    14:"Cfa", 15:"Cfb", 16:"Cfc",
    17:"Dsa", 18:"Dsb", 19:"Dsc", 20:"Dsd",
    21:"Dwa", 22:"Dwb", 23:"Dwc", 24:"Dwd",
    25:"Dfa", 26:"Dfb", 27:"Dfc", 28:"Dfd",
    29:"ET", 30:"EF",
}
KG_GROUP_MAP = {
    "A": "Tropical", "B": "Arid", "C": "Temperate",
    "D": "Cold",     "E": "Polar",
}

def extract_koppen_code(lon: float, lat: float) -> str:
    try:
        with rasterio.open(KG_TIF) as src:
            for dlat, dlon in [(0,0),(-0.05,0),(0.05,0),(0,-0.05),(0,0.05)]:
                v    = list(src.sample([(lon + dlon, lat + dlat)]))
                code = KG_INT2CODE.get(int(v[0][0]))
                if code:
                    return code
    except Exception:
        pass
    return ""

# ============================================================================
# §3  大陆归属
# ============================================================================
@lru_cache(maxsize=1)
def _load_continents() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(CONTINENT_SHP)
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    return gdf

def get_continent(lon: float, lat: float) -> str:
    gdf  = _load_continents()
    pt   = Point(lon, lat)
    hits = gdf[gdf.geometry.contains(pt)]
    if len(hits) == 0:
        gdf2 = gdf.copy()
        gdf2["_d"] = gdf2.geometry.distance(pt)
        hits = gdf2.nsmallest(1, "_d")
    col = next((c for c in hits.columns if c.strip().lower() == "continent"), None)
    return str(hits.iloc[0][col]).strip() if (col and len(hits)) else ""

def get_abs_threshold(continent: str) -> float:
    return CONTINENT_THRESHOLD.get(continent, DEFAULT_THRESHOLD)

# ============================================================================
# §4  ISD-Lite 读取与清洗
# ============================================================================
def read_isd_lite(filepath: str) -> "pd.DataFrame | None":
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(
            filepath, sep=r"\s+", header=None,
            usecols=[0, 1, 2, 3, 4],
            names=["year", "month", "day", "hour", "temp_C"],
            na_values={"temp_C": -9999},
            engine="c", on_bad_lines="skip",
        )
        df["temp_C"]  = df["temp_C"] / 10.0
        df["datetime"] = pd.to_datetime(df[["year","month","day","hour"]], errors="coerce")
        df = (df.dropna(subset=["datetime"])
                .drop_duplicates(subset="datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
        return df[["datetime", "temp_C"]] if not df.empty else None
    except Exception as exc:
        print(f"    Warning: read failed {filepath}: {exc}")
        return None

def clean_to_hourly(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """重采样到 hourly；小缺口（≤2h）插值；<8h 有效观测的日期整日置空。"""
    full_idx = pd.date_range(pd.Timestamp(year,1,1,0),
                              pd.Timestamp(year,12,31,23), freq="1h")
    hourly = (df.set_index("datetime")[["temp_C"]]
                .resample("1h").mean()
                .reindex(full_idx))
    hourly["temp_C"] = hourly["temp_C"].interpolate(
        method="time", limit=2, limit_area="inside")
    valid_cnt = hourly["temp_C"].groupby(hourly.index.date).transform(
        lambda s: s.notna().sum())
    hourly.loc[valid_cnt < 8, "temp_C"] = np.nan
    result = hourly.reset_index().rename(columns={"index": "datetime"})
    result["date"] = result["datetime"].dt.date
    return result

def load_station_multiyear(
        usaf: str, wban: str, lon: float,
        min_valid_frac: float = FULL_YEAR_MIN_VALID_FRAC
) -> "tuple[pd.DataFrame | None, list]":
    frames, valid_years = [], []
    offset = pd.Timedelta(hours=lon / 15.0)
    for yr in YEARS:
        fpath = os.path.join(ISD_BASE_DIR, str(yr), f"{usaf}-{wban}-{yr}.gz")
        df_raw = read_isd_lite(fpath)
        if df_raw is None:
            continue
        df_h = clean_to_hourly(df_raw, yr)
        df_h["local_dt"]    = df_h["datetime"] + offset
        df_h["local_date"]  = df_h["local_dt"].dt.date
        df_h["local_month"] = df_h["local_dt"].dt.month
        df_h["local_hour"]  = (df_h["local_dt"].dt.hour
                                + df_h["local_dt"].dt.minute / 60.0)
        year_days  = 366 if (yr%4==0 and (yr%100!=0 or yr%400==0)) else 365
        valid_days = df_h.dropna(subset=["temp_C"])["local_date"].nunique()
        if valid_days / year_days < min_valid_frac:
            continue
        frames.append(df_h)
        valid_years.append(yr)
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), valid_years

# ============================================================================
# §5  ERA5 工具
# ============================================================================
def load_era5_tmax(usaf: str, wban: str, station_type: str) -> "pd.Series | None":
    """
    Load ERA5 station-level Tmax series.

    Compatible with both naming conventions:
      1) {USAF}_{WBAN}_{station_type}.csv
      2) {USAF}-{WBAN}_{station_type}_tmax.csv

    Accepted Tmax column names:
      - tmax_c
      - tmax
    """
    usaf = str(usaf).strip()
    wban = str(wban).strip()

    candidates = [
        os.path.join(ERA5_STATION_DIR, f"{usaf}_{wban}_{station_type}.csv"),
        os.path.join(ERA5_STATION_DIR, f"{usaf}-{wban}_{station_type}_tmax.csv"),
    ]

    for fpath in candidates:
        if not os.path.exists(fpath):
            continue

        try:
            df = pd.read_csv(fpath, parse_dates=["date"], index_col="date")

            if "tmax_c" in df.columns:
                tmax_col = "tmax_c"
            elif "tmax" in df.columns:
                tmax_col = "tmax"
            else:
                continue

            s = pd.to_numeric(df[tmax_col], errors="coerce").dropna()
            if len(s) < MIN_LOYO_REF_YEARS * 30:
                continue

            return s

        except Exception:
            continue

    return None


def compute_doy_threshold_era5(era5_tmax: pd.Series) -> dict:
    ref = pd.DataFrame({"tmax": era5_tmax.values,
                         "date": pd.to_datetime(era5_tmax.index)})
    ref["doy"] = ref["date"].apply(
        lambda x: (pd.Timestamp(2001,x.month,x.day).dayofyear
                   if not (x.month==2 and x.day==29) else 59))
    ref = ref.dropna(subset=["tmax"])
    hw  = HW_WINDOW_HALF
    doy_thr = {}
    for doy in range(1, 366):
        lo, hi = doy - hw, doy + hw
        if lo < 1:
            mask = (ref["doy"] >= 365+lo) | (ref["doy"] <= hi)
        elif hi > 365:
            mask = (ref["doy"] >= lo) | (ref["doy"] <= hi-365)
        else:
            mask = (ref["doy"] >= lo) & (ref["doy"] <= hi)
        vals = ref.loc[mask, "tmax"].dropna().values
        doy_thr[doy] = float(np.nanpercentile(vals, HW_PERCENTILE)) if len(vals) else np.nan
    return doy_thr

def apply_doy_threshold(date_index, doy_thr: dict) -> pd.Series:
    dates = pd.to_datetime(date_index)
    doys  = dates.map(lambda x: (pd.Timestamp(2001,x.month,x.day).dayofyear
                                  if not (x.month==2 and x.day==29) else 59))
    return pd.Series([doy_thr.get(int(d), np.nan) for d in doys], index=date_index)

def quantile_mapping_values(values, sim_train, obs_train, n_quantiles=1001):
    """
    Empirical quantile mapping:
        corrected = F_obs^{-1}(F_sim(values))

    Here:
      values    = raw ERA5-derived thresholds
      sim_train = ERA5 Tmax samples during overlap
      obs_train = ISD Tmax samples during overlap
    """
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
        min_overlap_days_per_month: int = 30,
        min_overlap_days_annual: int = 100,
        n_quantiles: int = 1001):
    """
    Quantile mapping bias correction for ERA5-derived heatwave thresholds.

    Monthly QM is preferred. If monthly overlap is insufficient,
    annual QM is used. If annual overlap is also insufficient,
    raw ERA5 thresholds are retained.
    """
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
        "isd_tmax_mean_overlap": np.nan,
        "era5_tmax_mean_overlap": np.nan,
    }

    if overlap.empty:
        for m in range(1, 13):
            diagnostics["monthly_sample_counts"][m] = 0
            diagnostics["monthly_methods"][m] = "no_correction_no_overlap"
        return thr.copy(), diagnostics

    diagnostics["isd_tmax_mean_overlap"] = float(overlap["isd"].mean())
    diagnostics["era5_tmax_mean_overlap"] = float(overlap["era5"].mean())

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

    return corrected, diagnostics


def compute_isd_threshold(tmax_series: pd.Series) -> pd.Series:
    ref = pd.DataFrame({"tmax": tmax_series.values,
                         "date": pd.to_datetime(tmax_series.index)})
    ref["doy"] = ref["date"].apply(
        lambda x: (pd.Timestamp(2001,x.month,x.day).dayofyear
                   if not (x.month==2 and x.day==29) else 59)).values
    ref = ref.dropna(subset=["tmax"])
    hw  = HW_WINDOW_HALF
    thresholds = {}
    for d in tmax_series.index:
        d_ts = pd.Timestamp(d)
        doy  = (pd.Timestamp(2001,d_ts.month,d_ts.day).dayofyear
                if not (d_ts.month==2 and d_ts.day==29) else 59)
        lo, hi = doy - hw, doy + hw
        if lo < 1:
            mask = (ref["doy"] >= 365+lo) | (ref["doy"] <= hi)
        elif hi > 365:
            mask = (ref["doy"] >= lo) | (ref["doy"] <= hi-365)
        else:
            mask = (ref["doy"] >= lo) & (ref["doy"] <= hi)
        vals = ref.loc[mask, "tmax"].dropna().values
        thresholds[d] = float(np.nanpercentile(vals, HW_PERCENTILE)) if len(vals) else np.nan
    return pd.Series(thresholds).sort_index()

# ============================================================================
# §6  热浪检测
# ============================================================================
def detect_heatwave_flag(daily_tmax: pd.Series, threshold) -> pd.Series:
    if np.isscalar(threshold):
        above = (daily_tmax > float(threshold)).values
    else:
        thr_s = pd.Series(threshold).reindex(daily_tmax.index)
        above = (daily_tmax > thr_s).fillna(False).values
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

def compute_daily_tmax(df_station: pd.DataFrame) -> pd.Series:
    return df_station.groupby("local_date")["temp_C"].max().dropna()

# ============================================================================
# §7b  FFT 日循环重构  ← v2 新增
# ============================================================================
def compute_fft_diurnal(
        df_sub: pd.DataFrame,
        n_harm: int = N_HARMONICS,
) -> "dict | None":
    """
    对给定时段的逐小时子集做 FFT，返回重构日循环相关字段。

    Parameters
    ----------
    df_sub  : 包含 local_hour、temp_C 列的 DataFrame（已按时段过滤）
    n_harm  : 保留谐波数（默认 4，与主脚本 N_HARMONICS 一致）

    Returns
    -------
    dict 含以下键，或 None（数据不足）：
      mean, amplitudes[list], phases[list],
      Tmax_fft, Tmin_fft,
      reconstructed[np.ndarray 24值],
      n_days
    """
    if df_sub is None or len(df_sub) == 0:
        return None
    tmp = df_sub.dropna(subset=["temp_C"]).copy()
    if len(tmp) == 0:
        return None

    tmp["h_bin"] = tmp["local_hour"].round().astype(int) % 24
    hourly_t = tmp.groupby("h_bin")["temp_C"].mean().reindex(range(24))

    # 至少需要 8 个有覆盖的小时 bin（与主脚本 compute_fft 一致）
    if hourly_t.notna().sum() < 8:
        return None

    hourly_t = (hourly_t
                .interpolate(method="linear", limit_direction="both")
                .ffill().bfill())
    if hourly_t.isna().any():
        return None

    vals  = hourly_t.values.astype(float)
    coef  = sp_fft(vals) / len(vals)
    mean_t = float(np.real(coef[0]))

    amps, phases = [], []
    for k in range(1, n_harm + 1):
        amps.append(float(2 * np.abs(coef[k])))
        phases.append(float(np.angle(coef[k])))

    # 重构 24 点曲线
    t   = np.arange(24)
    sig = np.full(24, mean_t, dtype=float)
    for k, (a, p) in enumerate(zip(amps, phases), start=1):
        sig += a * np.cos(k * 2 * np.pi * t / 24 + p)

    return {
        "mean":          mean_t,
        "amplitudes":    amps,
        "phases":        phases,
        "Tmax_fft":      float(np.max(sig)),
        "Tmin_fft":      float(np.min(sig)),
        "reconstructed": sig,
        "n_days":        int(tmp["local_date"].nunique()),
    }


def build_diurnal_row(
        fft_result: dict,
        base_meta:  dict,
        period:     str,
) -> "dict | None":
    """
    将 compute_fft_diurnal() 的返回值展开为一行宽表，
    供 station_diurnal_reconstructed.csv 输出。

    Parameters
    ----------
    fft_result : compute_fft_diurnal() 的返回值
    base_meta  : pair_id, station_type, usaf, wban, lon, lat,
                 kg_code, kg_group, climate_zone, continent 等固定字段
    period     : "annual" / "JJA" / "HW" / "NHW"
    """
    if fft_result is None:
        return None

    row = {**base_meta, "period": period}
    row["n_days"]    = fft_result["n_days"]
    row["Tmean_fft"] = fft_result["mean"]
    row["Tmax_fft"]  = fft_result["Tmax_fft"]
    row["Tmin_fft"]  = fft_result["Tmin_fft"]

    for k, (a, p) in enumerate(
            zip(fft_result["amplitudes"], fft_result["phases"]), start=1):
        row[f"Amp{k}"]   = a
        row[f"Phase{k}"] = p

    for h, val in enumerate(fft_result["reconstructed"]):
        row[f"h{h:02d}"] = float(val)

    return row


def reconstruct_period_diurnal(
        df_all: pd.DataFrame,
        hw_flag_pct: pd.Series,
        base_meta: dict,
        diurnal_rows: list,
) -> None:
    """
    Compute annual / warm_season / HW / NHW reconstructed diurnal cycles.

    Warm season is hemisphere-aware:
      NH = JJA; SH = DJF.
    """
    lat = base_meta.get("lat", np.nan)
    # Warm-season date filtering follows analysis_multiyear.py boundary rule:
    # keep SH Jan-Feb of min(YEARS), exclude SH Dec of max(YEARS).

    # ── Annual：全年所有有效小时 ─────────────────────────────────
    fft_ann = compute_fft_diurnal(df_all)
    row_ann = build_diurnal_row(fft_ann, base_meta, "annual")
    if row_ann:
        diurnal_rows.append(row_ann)

    # ── Warm season：NH=JJA, SH=DJF with boundary-year filtering ──
    df_warm = df_all[
        df_all["local_date"].apply(
            lambda d: is_warm_season_date_for_analysis(d, lat)
        )
    ].copy()
    fft_warm = compute_fft_diurnal(df_warm)
    row_warm = build_diurnal_row(fft_warm, base_meta, "warm_season")
    if row_warm:
        diurnal_rows.append(row_warm)

    if len(df_warm) == 0:
        return

    warm_dates = set(df_warm["local_date"].unique())

    # ── HW：warm-season heatwave days ───────────────────────────
    hw_dates = set(hw_flag_pct[hw_flag_pct].index.tolist()) & warm_dates
    if hw_dates:
        df_hw = df_warm[df_warm["local_date"].isin(hw_dates)].copy()
        fft_hw = compute_fft_diurnal(df_hw)
        row_hw = build_diurnal_row(fft_hw, base_meta, "HW")
        if row_hw:
            diurnal_rows.append(row_hw)

    # ── NHW：warm-season non-heatwave days ─────────────────────
    nhw_dates = warm_dates - set(hw_flag_pct[hw_flag_pct].index.tolist())
    if nhw_dates:
        df_nhw = df_warm[df_warm["local_date"].isin(nhw_dates)].copy()
        fft_nhw = compute_fft_diurnal(df_nhw)
        row_nhw = build_diurnal_row(fft_nhw, base_meta, "NHW")
        if row_nhw:
            diurnal_rows.append(row_nhw)


# ============================================================================
# §7  单站点热浪分析（v2：新增 diurnal_rows 输出）
# ============================================================================
def analyze_single_station(
        usaf: str, wban: str, lon: float, lat: float,
        station_type: str,
        pair_id: str,
        continent: str,
        kg_code: str,
) -> "tuple[list, list, list]":
    """
    Returns
    -------
    summary_rows  : list[dict]  → station_heatwave_summary.csv
    daily_rows    : list[dict]  → station_hw_daily_flags.csv
    diurnal_rows  : list[dict]  → station_diurnal_reconstructed.csv  [v2 新增]
    """
    summary_rows, daily_rows, diurnal_rows = [], [], []

    # 1. 加载多年 ISD 数据
    df_all, valid_years = load_station_multiyear(usaf, wban, lon)
    if df_all is None or len(df_all) == 0:
        return summary_rows, daily_rows, diurnal_rows

    tmax_daily = compute_daily_tmax(df_all)
    if len(tmax_daily) < 30:
        return summary_rows, daily_rows, diurnal_rows

    # 2. Percentile threshold: ERA5 climatology + ERA5→ISD quantile mapping
    era5_s = load_era5_tmax(usaf, wban, station_type) if USE_ERA5_CLIMATOLOGY else None

    hw_threshold_raw_mean = np.nan
    hw_threshold_corrected_mean = np.nan
    bias_correction_method = "none"
    qm_diagnostics = {
        "overlap_days": 0,
        "isd_tmax_mean_overlap": np.nan,
        "era5_tmax_mean_overlap": np.nan,
        "monthly_sample_counts": {},
        "monthly_methods": {},
    }

    if era5_s is not None and len(era5_s) > 0:
        doy_thr = compute_doy_threshold_era5(era5_s)
        thr_pct_raw = apply_doy_threshold(tmax_daily.index, doy_thr)

        hw_threshold_raw_mean = (
            float(thr_pct_raw.mean()) if thr_pct_raw.notna().any() else np.nan
        )

        if USE_QM_BIAS_CORRECTION:
            thr_pct, qm_diagnostics = apply_quantile_mapping_bias_correction(
                hw_thr_series_raw=thr_pct_raw,
                isd_tmax=tmax_daily,
                era5_tmax=era5_s,
                min_overlap_days_per_month=QM_MIN_OVERLAP_DAYS_PER_MONTH,
                min_overlap_days_annual=QM_MIN_OVERLAP_DAYS_ANNUAL,
                n_quantiles=QM_N_QUANTILES,
            )
            ref_mode = HW_REF_MODE
            bias_correction_method = "quantile_mapping_empirical_cdf"
        else:
            thr_pct = thr_pct_raw.copy()
            ref_mode = "ERA5_longterm_Tmax_P90_DOY_raw"
            bias_correction_method = "none_raw_era5_threshold"

    else:
        # Keep the original fallback behaviour for minimum disruption.
        thr_pct = compute_isd_threshold(tmax_daily)
        thr_pct_raw = thr_pct.copy()
        ref_mode = "ISD_P90_short_fallback"
        bias_correction_method = "none_isd_short_record"

    thr_pct_mean = float(thr_pct.mean()) if thr_pct.notna().any() else np.nan
    hw_threshold_corrected_mean = thr_pct_mean

    if pd.isna(thr_pct_mean):
        return summary_rows, daily_rows, diurnal_rows


    # Canonical rule: all downstream urban/rural station records use the
    # pair-level daily HW/NHW flags exported by analysis_multiyear.py.
    # Missing or mismatched canonical flags cause this station record to be
    # skipped; this script must not redefine HW/NHW dates.
    analysis_hw_flags = get_analysis_hw_flags_for_pair(pair_id)
    ext_aligned = align_analysis_hw_flags(analysis_hw_flags, tmax_daily.index)

    required_flag_cols = [
        "is_warm_season",
        "hw_flag_percentile_warm_season",
        "nhw_flag_percentile_warm_season",
    ]
    if (
        ext_aligned is None
        or any(c not in ext_aligned.columns for c in required_flag_cols)
        or ext_aligned[required_flag_cols].dropna(how="any").empty
    ):
        print(
            f"    [SKIP] {pair_id}/{station_type}: canonical "
            "analysis_multiyear daily HW/NHW flags missing or mismatched"
        )
        return summary_rows, daily_rows, diurnal_rows

    valid_flags = ext_aligned[required_flag_cols].notna().all(axis=1)
    hw_flag_pct = pd.Series(False, index=tmax_daily.index)
    hw_flag_pct.loc[valid_flags] = (
        ext_aligned.loc[
            valid_flags, "hw_flag_percentile_warm_season"
        ].astype(int).astype(bool).values
    )
    ref_mode = "analysis_multiyear_daily_heatwave_flags"

    if "hw_threshold_corrected" in ext_aligned.columns:
        _ext_thr = pd.to_numeric(
            ext_aligned["hw_threshold_corrected"], errors="coerce"
        )
        if _ext_thr.notna().any():
            thr_pct = pd.Series(_ext_thr.values, index=tmax_daily.index)
            thr_pct_mean = (
                float(thr_pct.mean()) if thr_pct.notna().any() else np.nan
            )
            hw_threshold_corrected_mean = thr_pct_mean

    abs_thr = get_abs_threshold(continent)
    hw_flag_abs = detect_heatwave_warm_season(tmax_daily, abs_thr, lat)

    # 4. summary
    hw_days_pct = int(hw_flag_pct.sum())
    hw_days_abs = int(hw_flag_abs.sum())

    def hw_periods_str(flag: pd.Series) -> str:
        dates = pd.to_datetime(flag[flag].index.tolist())
        if len(dates) == 0:
            return ""
        periods, start, prev = [], dates[0], dates[0]
        for d in dates[1:]:
            if (d - prev).days > 1:
                periods.append(f"{start.date()}~{prev.date()}")
                start = d
            prev = d
        periods.append(f"{start.date()}~{prev.date()}")
        return "; ".join(periods)

    kg_group_val = str(kg_code)[0] if isinstance(kg_code, str) and kg_code else ""
    climate_zone = KG_GROUP_MAP.get(kg_group_val, "")

    intensity_pct = (float((tmax_daily[hw_flag_pct] - thr_pct[hw_flag_pct]).mean())
                     if hw_days_pct > 0 else np.nan)
    intensity_abs = (float((tmax_daily[hw_flag_abs] - abs_thr).mean())
                     if hw_days_abs > 0 else np.nan)

    summary_rows.append({
        "pair_id":          pair_id,
        "station_type":     station_type,
        "usaf":             usaf,
        "wban":             wban,
        "lon":              lon,
        "lat":              lat,
        "continent":        continent,
        "kg_code":          kg_code,
        "kg_group":         kg_group_val,
        "climate_zone":     climate_zone,
        "n_valid_years":    len(valid_years),
        "valid_years":      ",".join(map(str, valid_years)),
        "n_obs_days":       int(len(tmax_daily)),
        "hemisphere":       hemisphere_from_lat(lat),
        "warm_season_label": warm_season_label_for_lat(lat),
        "warm_season_months": warm_months_string(lat),
        "tmax_mean":        float(tmax_daily.mean()),
        "tmax_p90_mean":    thr_pct_mean,

        "hw_ref_mode":                  ref_mode,
        "bias_correction_method":       bias_correction_method,
        "hw_threshold_raw_mean":        hw_threshold_raw_mean,
        "hw_threshold_corrected_mean":  hw_threshold_corrected_mean,
        "era5_isd_overlap_days":        qm_diagnostics.get("overlap_days", 0),
        "isd_tmax_mean_overlap":        qm_diagnostics.get("isd_tmax_mean_overlap", np.nan),
        "era5_tmax_mean_overlap":       qm_diagnostics.get("era5_tmax_mean_overlap", np.nan),
        "qm_n_quantiles":               qm_diagnostics.get("qm_n_quantiles", np.nan),
        "qm_monthly_sample_counts":     ";".join(
            [f"{m}:{qm_diagnostics.get('monthly_sample_counts', {}).get(m, 0)}"
            for m in range(1, 13)]
        ),
        "qm_monthly_methods":           ";".join(
            [f"{m}:{qm_diagnostics.get('monthly_methods', {}).get(m, '')}"
            for m in range(1, 13)]
        ),


        "abs_threshold":    abs_thr,
        "hw_days_pct":      hw_days_pct,
        "hw_frac_pct":      round(hw_days_pct / len(tmax_daily), 4),
        "hw_intensity_pct": intensity_pct,
        "hw_periods_pct":   hw_periods_str(hw_flag_pct),
        "has_hw_pct":       int(hw_days_pct > 0),
        "hw_days_abs":      hw_days_abs,
        "hw_frac_abs":      round(hw_days_abs / len(tmax_daily), 4),
        "hw_intensity_abs": intensity_abs,
        "hw_periods_abs":   hw_periods_str(hw_flag_abs),
        "has_hw_abs":       int(hw_days_abs > 0),
    })

    # 5. 逐日标记
    for date, tmax_val in tmax_daily.items():
        thr_val = (float(thr_pct.reindex([date]).iloc[0])
                   if hasattr(thr_pct, "reindex") else np.nan)
        daily_rows.append({
            "pair_id":      pair_id,
            "station_type": station_type,
            "usaf":         usaf,
            "wban":         wban,
            "date":         date,
            "tmax":         float(tmax_val),
            "thr_pct":      thr_val,
            "thr_abs":      abs_thr,
            "is_hw_pct":    int(hw_flag_pct.get(date, False)),
            "is_hw_abs":    int(hw_flag_abs.get(date, False)),
            "is_warm_season": int(pd.Timestamp(date).month in warm_months_for_lat(lat)),
            "hemisphere": hemisphere_from_lat(lat),
            "warm_season_label": warm_season_label_for_lat(lat),
            "warm_season_months": warm_months_string(lat),
        })

    # ── v2：日循环重构 ────────────────────────────────────────────
    base_meta = {
        "pair_id":      pair_id,
        "station_type": station_type,
        "usaf":         usaf,
        "wban":         wban,
        "lon":          lon,
        "lat":          lat,
        "continent":    continent,
        "kg_code":      kg_code,
        "kg_group":     kg_group_val,
        "climate_zone": climate_zone,
        "hw_ref_mode":  ref_mode,
        "hemisphere": hemisphere_from_lat(lat),
        "warm_season_label": warm_season_label_for_lat(lat),
        "warm_season_months": warm_months_string(lat),
        "bias_correction_method": bias_correction_method,
        "era5_isd_overlap_days": qm_diagnostics.get("overlap_days", 0),
        "hw_threshold_raw_mean": hw_threshold_raw_mean,
        "hw_threshold_corrected_mean": hw_threshold_corrected_mean,
        "n_valid_years": len(valid_years),
    }

    reconstruct_period_diurnal(df_all, hw_flag_pct, base_meta, diurnal_rows)

    return summary_rows, daily_rows, diurnal_rows


# ============================================================================
# §8  pair 级联合热浪状态
# ============================================================================
def compute_pair_hw_status(
        pair_id: str,
        urban_daily: "pd.DataFrame | None",
        rural_daily: "pd.DataFrame | None",
) -> dict:
    empty = {
        "pair_id": pair_id,
        "n_common_days_pct": 0, "n_common_days_abs": 0,
        "urban_hw_days_pct": np.nan, "rural_hw_days_pct": np.nan,
        "both_hw_days_pct": np.nan, "either_hw_days_pct": np.nan,
        "jaccard_pct": np.nan,
        "urban_hw_days_abs": np.nan, "rural_hw_days_abs": np.nan,
        "both_hw_days_abs": np.nan, "either_hw_days_abs": np.nan,
        "jaccard_abs": np.nan,
        "rural_has_hw_pct": np.nan, "rural_has_hw_abs": np.nan,
        "urban_only_hw_pct": np.nan, "rural_only_hw_pct": np.nan,
        "urban_only_hw_abs": np.nan, "rural_only_hw_abs": np.nan,
    }
    if urban_daily is None or rural_daily is None:
        return empty
    if len(urban_daily) == 0 or len(rural_daily) == 0:
        return empty

    u = urban_daily.set_index("date")[["is_hw_pct", "is_hw_abs"]]
    r = rural_daily.set_index("date")[["is_hw_pct", "is_hw_abs"]]
    common = u.index.intersection(r.index)
    if len(common) == 0:
        return empty

    uc, rc = u.loc[common], r.loc[common]

    def jaccard(a, b):
        union = (a | b).sum()
        return float((a & b).sum() / union) if union else np.nan

    def pct_overlap(a_b, b_b):
        u_hw = a_b.sum()
        return float((a_b & b_b).sum() / u_hw) if u_hw else np.nan

    u_hw_pct = uc["is_hw_pct"].astype(bool)
    r_hw_pct = rc["is_hw_pct"].astype(bool)
    u_hw_abs = uc["is_hw_abs"].astype(bool)
    r_hw_abs = rc["is_hw_abs"].astype(bool)

    return {
        "pair_id":                  pair_id,
        "n_common_days_pct":        int(len(common)),
        "n_common_days_abs":        int(len(common)),
        "urban_hw_days_pct":        int(u_hw_pct.sum()),
        "rural_hw_days_pct":        int(r_hw_pct.sum()),
        "both_hw_days_pct":         int((u_hw_pct & r_hw_pct).sum()),
        "either_hw_days_pct":       int((u_hw_pct | r_hw_pct).sum()),
        "only_urban_hw_pct":        int((u_hw_pct & ~r_hw_pct).sum()),
        "only_rural_hw_pct":        int((~u_hw_pct & r_hw_pct).sum()),
        "jaccard_pct":              jaccard(u_hw_pct, r_hw_pct),
        "rural_has_hw_pct":         int(r_hw_pct.any()),
        "urban_has_hw_pct":         int(u_hw_pct.any()),
        "rural_hw_given_urban_pct": pct_overlap(u_hw_pct, r_hw_pct),
        "urban_hw_days_abs":        int(u_hw_abs.sum()),
        "rural_hw_days_abs":        int(r_hw_abs.sum()),
        "both_hw_days_abs":         int((u_hw_abs & r_hw_abs).sum()),
        "either_hw_days_abs":       int((u_hw_abs | r_hw_abs).sum()),
        "only_urban_hw_abs":        int((u_hw_abs & ~r_hw_abs).sum()),
        "only_rural_hw_abs":        int((~u_hw_abs & r_hw_abs).sum()),
        "jaccard_abs":              jaccard(u_hw_abs, r_hw_abs),
        "rural_has_hw_abs":         int(r_hw_abs.any()),
        "urban_has_hw_abs":         int(u_hw_abs.any()),
        "rural_hw_given_urban_abs": pct_overlap(u_hw_abs, r_hw_abs),
    }

# ============================================================================
# §9  单 pair 多进程入口
# ============================================================================
def process_pair(args: tuple) -> dict:
    idx, row = args
    pair_id  = row["pair_id"]

    try:
        parts   = pair_id.split("__")
        u_parts = parts[0].strip().split("_")
        r_parts = parts[1].strip().split("_")
        if len(parts) != 2 or len(u_parts) < 2 or len(r_parts) < 2:
            raise ValueError(f"pair_id 格式错误: {pair_id}")
        usaf_u, wban_u = u_parts[0], u_parts[1]
        usaf_r, wban_r = r_parts[0], r_parts[1]
    except Exception as e:
        return {"summary_rows": [], "daily_rows": [], "diurnal_rows": [],
                "pair_status": {}, "error": f"[{pair_id}] parse error: {e}"}

    lon_u, lat_u = float(row["lon_urban"]), float(row["lat_urban"])
    lon_r, lat_r = float(row["lon_rural"]), float(row["lat_rural"])

    continent = get_continent(lon_u, lat_u)
    kg_code_u = extract_koppen_code(lon_u, lat_u)
    kg_code_r = extract_koppen_code(lon_r, lat_r)

    sum_u, daily_u, diurn_u = analyze_single_station(
        usaf_u, wban_u, lon_u, lat_u, "urban",
        pair_id, continent, kg_code_u)

    sum_r, daily_r, diurn_r = analyze_single_station(
        usaf_r, wban_r, lon_r, lat_r, "rural",
        pair_id, continent, kg_code_r)

    urban_daily_df = pd.DataFrame(daily_u) if daily_u else None
    rural_daily_df = pd.DataFrame(daily_r) if daily_r else None
    pair_status    = compute_pair_hw_status(pair_id, urban_daily_df, rural_daily_df)

    hw_u = sum_u[0]["hw_days_pct"] if sum_u else "N/A"
    hw_r = sum_r[0]["hw_days_pct"] if sum_r else "N/A"
    dn_u = len(diurn_u)
    dn_r = len(diurn_r)
    print(f"  [{idx:4d}] {pair_id}  "
          f"urban hw_pct={hw_u}d diurnal_periods={dn_u} | "
          f"rural hw_pct={hw_r}d diurnal_periods={dn_r}")

    return {
        "summary_rows": sum_u + sum_r,
        "daily_rows":   daily_u + daily_r,
        "diurnal_rows": diurn_u + diurn_r,   # v2 新增
        "pair_status":  pair_status,
        "error":        None,
    }

# ============================================================================
# §10 主函数
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  Heatwave Detection + Diurnal Reconstruction – ALL Stations  v2")
    print(f"  Years       : {YEARS[0]}–{YEARS[-1]}")
    print(f"  Method      : P{HW_PERCENTILE} ± {HW_WINDOW_HALF}-day DOY,"
          f" ≥{HW_MIN_DAYS} consec. days")
    print(f"  ERA5 mode   : {'ON (fallback→ISD P90)' if USE_ERA5_CLIMATOLOGY else 'OFF'}")
    print(f"  FFT harmonics: {N_HARMONICS}")
    print(f"  Diurnal periods: annual / warm_season(NH=JJA, SH=DJF) / HW / NHW")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print("=" * 72)

    pair_df = pd.read_csv(PAIR_CSV_PATH)
    print(f"\nTotal pairs: {len(pair_df)}")

    print("Pre-loading continent shapefile ...")
    _load_continents()
    print("  Done\n")

    tasks   = [(idx, row) for idx, row in pair_df.iterrows()]
    n_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting {n_cores} worker processes ...\n")

    all_summary_rows  = []
    all_daily_rows    = []
    all_diurnal_rows  = []   # v2 新增
    all_pair_status   = []
    error_list        = []

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(process_pair, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                res = future.result()
                all_summary_rows.extend(res["summary_rows"])
                all_daily_rows.extend(res["daily_rows"])
                all_diurnal_rows.extend(res["diurnal_rows"])   # v2
                if res["pair_status"]:
                    all_pair_status.append(res["pair_status"])
                if res["error"]:
                    error_list.append(res["error"])
            except Exception as exc:
                t = futures[future]
                error_list.append(f"[{t[1]['pair_id']}] crash: {exc}")

    print(f"\nFinished. summary={len(all_summary_rows)}  "
          f"daily={len(all_daily_rows)}  "
          f"diurnal_rows={len(all_diurnal_rows)}  "
          f"errors={len(error_list)}")

    # ── 保存 station_heatwave_summary.csv ────────────────────────
    sum_df = pd.DataFrame(all_summary_rows)
    sum_df.to_csv(os.path.join(OUTPUT_DIR, "station_heatwave_summary.csv"), index=False)
    print(f"  Saved: station_heatwave_summary.csv  ({len(sum_df)} rows)")

    # ── 保存 station_hw_daily_flags.csv ──────────────────────────
    daily_df = pd.DataFrame(all_daily_rows)
    daily_df.to_csv(os.path.join(OUTPUT_DIR, "station_hw_daily_flags.csv"), index=False)
    print(f"  Saved: station_hw_daily_flags.csv  ({len(daily_df)} rows)")

    # ── 保存 pair_heatwave_status.csv ────────────────────────────
    pair_df_out = pd.DataFrame(all_pair_status)
    pair_df_out.to_csv(os.path.join(OUTPUT_DIR, "pair_heatwave_status.csv"), index=False)
    print(f"  Saved: pair_heatwave_status.csv  ({len(pair_df_out)} rows)")

    # ── v2 保存 station_diurnal_reconstructed.csv ─────────────────
    if all_diurnal_rows:
        diurnal_df = pd.DataFrame(all_diurnal_rows)

        meta_cols = [
            "pair_id", "station_type", "usaf", "wban", "period",
            "lon", "lat", "continent",
            "kg_code", "kg_group", "climate_zone",
            "hw_ref_mode",
            "bias_correction_method",
            "era5_isd_overlap_days",
            "hw_threshold_raw_mean",
            "hw_threshold_corrected_mean",
            "n_valid_years",
            "n_days", "Tmean_fft", "Tmax_fft", "Tmin_fft",
        ]

        amp_cols   = [f"Amp{k}"   for k in range(1, N_HARMONICS + 1)]
        phase_cols = [f"Phase{k}" for k in range(1, N_HARMONICS + 1)]
        hour_cols  = [f"h{h:02d}" for h in range(24)]

        ordered_cols = (
            [c for c in meta_cols  if c in diurnal_df.columns]
            + [c for c in amp_cols   if c in diurnal_df.columns]
            + [c for c in phase_cols if c in diurnal_df.columns]
            + [c for c in hour_cols  if c in diurnal_df.columns]
        )
        diurnal_df = diurnal_df[ordered_cols]
        diurnal_path = os.path.join(OUTPUT_DIR, "station_diurnal_reconstructed.csv")
        diurnal_df.to_csv(diurnal_path, index=False)
        print(f"  Saved: station_diurnal_reconstructed.csv  ({len(diurnal_df)} rows)")

        # 覆盖度诊断
        print("\n── 日循环重构覆盖度 ──")
        for period in ["annual", "JJA", "HW", "NHW"]:
            sub = diurnal_df[diurnal_df["period"] == period]
            n_u = (sub["station_type"] == "urban").sum()
            n_r = (sub["station_type"] == "rural").sum()
            print(f"  {period:8s}: urban={n_u}  rural={n_r}  total={len(sub)}")

        # 每种时期的平均样本天数
        print("\n── 各时期平均 n_days（有效日数）──")
        for period in ["annual", "JJA", "HW", "NHW"]:
            sub = diurnal_df[diurnal_df["period"] == period]["n_days"]
            if len(sub):
                print(f"  {period:8s}: mean={sub.mean():.1f}  "
                      f"min={sub.min()}  max={sub.max()}")
    else:
        print("  Warning: no diurnal reconstruction rows generated.")

    # ── 误差日志 ──────────────────────────────────────────────────
    if error_list:
        with open(os.path.join(OUTPUT_DIR, "errors.txt"), "w") as f:
            f.write("\n".join(error_list))
        print(f"  Saved: errors.txt  ({len(error_list)} entries)")

    # ── 控制台统计摘要 ────────────────────────────────────────────
    if not sum_df.empty:
        print("\n── 热浪覆盖统计（按站点类型）──")
        for stype in ["urban", "rural"]:
            sub = sum_df[sum_df["station_type"] == stype]
            if sub.empty:
                continue
            n   = len(sub)
            nhp = int(sub["has_hw_pct"].sum())
            nha = int(sub["has_hw_abs"].sum())
            print(f"  {stype:6s}  n={n:>5d}  "
                  f"has_hw_pct={nhp}({nhp/n:.1%})  "
                  f"avg_days={sub['hw_days_pct'].mean():.1f}  |  "
                  f"has_hw_abs={nha}({nha/n:.1%})  "
                  f"avg_days={sub['hw_days_abs'].mean():.1f}")

        if not pair_df_out.empty:
            print("\n── pair 级城乡热浪一致性（百分位法）──")
            tp = len(pair_df_out)
            both   = int(pair_df_out["both_hw_days_pct"].gt(0).sum())
            u_only = int(pair_df_out["only_urban_hw_pct"].gt(0).sum())
            r_only = int(pair_df_out["only_rural_hw_pct"].gt(0).sum())
            print(f"  两站均有热浪: {both}({both/tp:.1%})  "
                  f"仅城市: {u_only}({u_only/tp:.1%})  "
                  f"仅农村: {r_only}({r_only/tp:.1%})")
            vj = pair_df_out["jaccard_pct"].dropna()
            print(f"  Jaccard: mean={vj.mean():.3f}  median={vj.median():.3f}")

    print(f"\n所有文件已保存至: {OUTPUT_DIR}")
    print("输出文件一览:")
    print("  station_heatwave_summary.csv         ← 每站点热浪汇总")
    print("  station_hw_daily_flags.csv           ← 逐日Tmax + 热浪标记")
    print("  pair_heatwave_status.csv             ← pair级城乡联合状态")
    print("  station_diurnal_reconstructed.csv    ← [v2新增] annual/JJA/HW/NHW")
    print("                                          日循环重构宽表（h00-h23）")
    if error_list:
        print("  errors.txt                           ← 失败记录")


if __name__ == "__main__":
    main()


############################################################
