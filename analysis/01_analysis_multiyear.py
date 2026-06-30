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
analysis_multiyear.py  (v5 — KG Climate Zone Full Integration)
==============================================================
Multi-year (2015-2024) UHI/UCI Diurnal Cycle Analysis.

v2 修改内容（相比原版）：
  [1] read_isd_lite()    → 同时解析 Field 6 露点温度 (dewpoint_C)
  [2] clean_year_data()  → 同时对 dewpoint_C 做 QC + 线性插值
  [3] compute_fft()      → 同时输出露点24小时均值曲线 hourly_dew
  [4] main() rec 字典    → 追加 urban_dew_h00..h23 / rural_dew_h00..h23

v3 新增（相比 v2）：
  [5] get_absolute_hw_threshold()   → 按洲际返回固定绝对阈值
  [6] detect_heatwave_absolute()    → 绝对阈值版热浪检测
  [7] build_period_records()        → 将 period/FFT/rec 构建逻辑独立封装，
                                      支持两种方法复用
  [8] main() 循环内同时跑百分位 & 绝对阈值，各加 hw_method 列标记
  [9] 按方法分 robustness_percentile/ robustness_absolute/ 子目录独立输出
  [10] station_valid_years.csv      → 记录每站全年 & 暖季有效年份覆盖

v4 新增（相比 v3）：
  [11] build_period_records() 中新增 warm_season 时期
       → 使用完整 JJA 暖季数据（不区分热浪/非热浪），
         与 annual(平时) / heatwave(热浪) 构成三时期对比

v5 新增（相比 v4）——KG 气候分区全面接入：
  [12] build_period_records()
       → rec 中新增 hne_ref_mode、hw_ref_mode 两个版本信息字段
  [13] save_kggroup_robustness()
       → 输出 kggroup_robustness_summary.csv（按 KG 主分组汇总）
  [14] save_latgroup_robustness() 后
       → main() robustness 输出段调用 save_kggroup_robustness()
  [15] save_mediation_analysis()
       → subset_dict 中循环追加 annual_kg_A/B/C/D/E 子样本
  [16] save_mechanism_regressions()
       → subsets 中追加 annual_kg_* 分层键值
  [17] robustness_comparison.csv
       → compare_cols 保证含 kg_code、kg_group
  [18] station_threshold_summary.csv
       → 输出时补写 hw_ref_mode 字段
  [19] 终端输出说明
       → 注明 KG 已作为主气候分组，lat_group 作为兼容性保留字段

ERA5-MOD 新增（相比 v5）：
  [20] USE_ERA5_CLIMATOLOGY 开关（默认 True）
       → True  时用 ERA5 长期气候态估算热浪 DOY 阈值
       → False 时回退至原始 ISD 短期 P90 估计
  [21] ERA5_STATION_DIR / load_era5_tmax_series /
       compute_hw_doy_thr_era5 / apply_doy_thr_to_index 四个 ERA5 工具
  [22] process_single_pair() 中的 [ERA5-MOD] 替换段

数据分层策略：
┌──────────────────────────────────────────────────────────────────────┐
│ 数据层        │ 月份范围  │ 用途                                      │
├──────────────────────────────────────────────────────────────────────┤
│ 全年数据      │ 1-12月    │ annual FFT、UHI/UCI 分组、协变量           │
│ 全年Tmax      │ 1-12月    │ HW 阈值计算（P90, ±7天窗口）               │
│ 暖季数据      │ JJA(6-8月)│ warm_season / heatwave / non_heatwave FFT │
└──────────────────────────────────────────────────────────────────────┘
4.16新增
保留所有年份

统一重采样到 hourly

只对小缺口插值

FFT 阶段按小时覆盖度判定是否可用

不要因为原生是 3-hourly 就删除
"""

import os
import gzip
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from scipy.fft import fft as sp_fft
from scipy import stats

import geopandas as gpd
from shapely.geometry import Point

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ══════════════════════════════════════════════════════════════
# KG climate zone utilities  (Beck et al. 2023)
# ══════════════════════════════════════════════════════════════
import rasterio

KG_TIF = str(_OPEN_SOURCE_PATHS.kg_tif)

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

# v5: KG 主分组（单字母）列表，用于机制回归/中介分析分层
KG_GROUPS_ALL = ["A", "B", "C", "D", "E"]


def lat_group_fallback(lat):
    """纬度带粗分组，保留作兼容性字段（不替代 KG）。"""
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
                    # 就近偏移搜索（±0.05°以内）
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


def add_climate_zone_columns(df, lon_col="lon_urban", lat_col="lat_urban"):
    """
    Add:
      kg_code           e.g. Cfa
      kg_group          e.g. C
      climate_zone_main e.g. Temperate
      lat_group         fallback latitude band (kept for compatibility)
    """
    out = df.copy()
    out["kg_code"] = extract_koppen_codes(out[lon_col].values, out[lat_col].values)
    out["kg_group"] = out["kg_code"].apply(kg_group)
    out["climate_zone_main"] = out["kg_code"].apply(kg_main_climate)

    if "lat_group" not in out.columns:
        out["lat_group"] = out[lat_col].apply(lat_group_fallback)
    else:
        out["lat_group"] = out["lat_group"].fillna(
            out[lat_col].apply(lat_group_fallback)
        )
    return out


# ─────────────────────────────────────────────────────────────
# 1. Global Config
# ─────────────────────────────────────────────────────────────
YEARS = list(range(2015, 2025))

# Hemisphere-aware warm season:
#   Northern Hemisphere: JJA = Jun-Jul-Aug
#   Southern Hemisphere: DJF = Dec-Jan-Feb
NH_WARM_MONTHS = [6, 7, 8]
SH_WARM_MONTHS = [12, 1, 2]

# Keep this name for backward compatibility with old code comments / old scripts.
# Do not use it directly for hemisphere-aware filtering.
ANALYSIS_MONTHS = NH_WARM_MONTHS

# Kept for backward compatibility only.
# Actual expected warm-season days should be calculated by latitude/year.
WARM_SEASON_DAYS_EXPECTED = 92

MIN_YEAR_VALID_FRAC      = 0.80
FULL_YEAR_MIN_VALID_FRAC = 0.80

# v5: 版本信息常量，写入每行记录，便于多版本溯源
HNE_REF_MODE = "full_year_all_pairs"   # 热夜超值计算的参考口径

HW_REF_MODE = "ERA5_longterm_Tmax_P90_DOY_±7day_quantile_mapping_corrected"

USE_QM_BIAS_CORRECTION = True
USE_MONTHLY_MEAN_BIAS_CORRECTION = False
QM_N_QUANTILES = 1001
QM_MIN_OVERLAP_DAYS_PER_MONTH = 30
QM_MIN_OVERLAP_DAYS_ANNUAL = 100


# ── ERA5-MOD：ERA5 气候态热浪阈值开关及工具常量 ─────────────────────
USE_ERA5_CLIMATOLOGY = True   # ← 默认开启；改为 False 还原为 ISD 短期估计

ERA5_STATION_DIR = str(_OPEN_SOURCE_PATHS.era5_station_dir)
# ERA5 站点 Tmax CSV 命名约定：{USAF}-{WBAN}_{station_type}_tmax.csv
# 必须包含 "date"（YYYY-MM-DD）和 "tmax"（℃）两列

MIN_LOYO_REF_YEARS = 5
# ERA5 序列最短有效年数（×30 天 = 最低观测量），低于此阈值回退 ISD

def load_era5_tmax_series(usaf: str, wban: str, station_type: str = "urban"):
    fpath = os.path.join(
        ERA5_STATION_DIR,
        f"{usaf.strip()}_{wban.strip()}_{station_type}.csv"
    )

    if not os.path.exists(fpath):
        return None

    try:
        df = pd.read_csv(fpath, parse_dates=["date"], index_col="date")

        # 自动识别列名
        if "tmax_c" in df.columns:
            tmax_col = "tmax_c"
        elif "tmax" in df.columns:
            tmax_col = "tmax"
        else:
            return None

        s = pd.to_numeric(df[tmax_col], errors="coerce").dropna()

        # 数据量检查
        if len(s) < MIN_LOYO_REF_YEARS * 30:
            return None

        return s

    except Exception:
        return None


def compute_hw_doy_thr_era5(
        era5_tmax: pd.Series,
        q: int = HW_PERCENTILE if False else 90,          # 占位，见下方正文
        window_half_width: int = 7) -> dict:
    """
    基于 ERA5 长序列 Tmax，计算每个 DOY 的热浪温度阈值（P_q，滑动窗口）。

    本函数直接引用模块级常量 HW_PERCENTILE / HW_WINDOW_HALF，
    参数默认值仅作文档占位，实际调用时无需传参。

    Returns
    -------
    dict  {doy (1–365): threshold_°C}
    """
    # 使用模块级常量（避免函数签名与常量不同步）
    _q    = HW_PERCENTILE
    _half = HW_WINDOW_HALF

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
        low  = doy - _half
        high = doy + _half
        if low < 1:
            mask = (ref_df["doy"] >= 365 + low) | (ref_df["doy"] <= high)
        elif high > 365:
            mask = (ref_df["doy"] >= low) | (ref_df["doy"] <= high - 365)
        else:
            mask = (ref_df["doy"] >= low) & (ref_df["doy"] <= high)

        vals = ref_df.loc[mask, "tmax"].dropna().values
        doy_thr[doy] = float(np.nanpercentile(vals, _q)) if len(vals) > 0 else np.nan

    return doy_thr


def apply_doy_thr_to_index(
        date_index,
        doy_thr: dict) -> pd.Series:
    """
    将 DOY → 阈值字典映射到任意日期索引，返回对齐的阈值 Series。

    Parameters
    ----------
    date_index : pandas DatetimeIndex 或可转为 Timestamp 的日期序列
    doy_thr    : compute_hw_doy_thr_era5() 的返回值

    Returns
    -------
    pd.Series（index 与 date_index 相同，值为对应 DOY 的阈值）
    """
    dates = pd.to_datetime(date_index)
    doys  = dates.map(
        lambda x: (
            pd.Timestamp(2001, x.month, x.day).dayofyear
            if not (x.month == 2 and x.day == 29) else 59
        )
    )
    thr_vals = [doy_thr.get(int(d), np.nan) for d in doys]
    return pd.Series(thr_vals, index=date_index)

def apply_monthly_mean_bias_correction(
        hw_thr_series_raw: pd.Series,
        isd_tmax: pd.Series,
        era5_tmax: pd.Series,
        min_overlap_days_per_month: int = 15):
    """
    Monthly mean bias correction:
      threshold_corrected(date)
      = ERA5_DOY_P90(date)
        + [mean(ISD_Tmax_month) - mean(ERA5_Tmax_month)]

    Bias is estimated only from overlapping years/dates between ISD and ERA5.
    If monthly overlap is insufficient, fallback to annual mean bias.
    """
    isd = pd.Series(isd_tmax).copy()
    era = pd.Series(era5_tmax).copy()

    isd.index = pd.to_datetime(isd.index)
    era.index = pd.to_datetime(era.index)

    overlap = pd.concat(
        [isd.rename("isd"), era.rename("era5")],
        axis=1,
        join="inner"
    ).dropna()

    if overlap.empty:
        return hw_thr_series_raw.copy(), {}, np.nan, {
            "isd_tmax_mean": float(isd.mean()) if len(isd) else np.nan,
            "era5_tmax_mean": float(era.mean()) if len(era) else np.nan,
            "overlap_days": 0,
        }

    annual_bias = float(overlap["isd"].mean() - overlap["era5"].mean())

    monthly_bias = {}
    for m in range(1, 13):
        sub = overlap[overlap.index.month == m]
        if len(sub) >= min_overlap_days_per_month:
            monthly_bias[m] = float(sub["isd"].mean() - sub["era5"].mean())
        else:
            monthly_bias[m] = annual_bias

    thr = hw_thr_series_raw.copy()
    original_index = thr.index
    thr_dt = pd.to_datetime(thr.index)

    bias_vals = [monthly_bias.get(int(d.month), annual_bias) for d in thr_dt]
    hw_thr_series_corrected = pd.Series(
        thr.values + np.array(bias_vals, dtype=float),
        index=original_index
    )

    diagnostics = {
        "isd_tmax_mean": float(overlap["isd"].mean()),
        "era5_tmax_mean": float(overlap["era5"].mean()),
        "overlap_days": int(len(overlap)),
    }

    return hw_thr_series_corrected, monthly_bias, annual_bias, diagnostics

def quantile_mapping_values(values, sim_train, obs_train, n_quantiles=1001):
    """
    Empirical Quantile Mapping:
        y = F_obs^{-1}(F_sim(x))

    values    : ERA5 values to be corrected
    sim_train : ERA5 training samples
    obs_train : observed ISD training samples
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

    # Remove duplicated simulated quantiles to avoid interpolation problems
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
    Quantile Mapping bias correction for ERA5-derived heatwave thresholds.

    Uses overlapping ERA5 Tmax and ISD Tmax samples to build empirical CDF mapping:
        corrected = F_obs^{-1}(F_era5(x))

    Monthly QM is preferred. If monthly samples are insufficient,
    annual QM is used. If annual samples are also insufficient,
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

# ── ERA5-MOD 工具函数结束 ────────────────────────────────────────────

PAIR_CSV_PATH = str(_OPEN_SOURCE_PATHS.pair_csv)
STATION_META_PATH = str(_OPEN_SOURCE_PATHS.station_meta)
PAIR_FALLBACK_PATH = PAIR_CSV_PATH
ISD_BASE_DIR = str(_OPEN_SOURCE_PATHS.isd_dir)
OUTPUT_DIR = str(_OPEN_SOURCE_PATHS.main_multiyear_output)
CONTINENT_SHP = str(_OPEN_SOURCE_PATHS.continent_shp)
BV_CSV_PATH = str(_OPEN_SOURCE_PATHS.building_volume_csv)
BV_RADIUS = "1000m"

MAX_CONSEC_NAN = 2
MAX_INTERP_GAP = 1

HW_PERCENTILE  = 90
HW_MIN_DAYS    = 3
HW_WINDOW_HALF = 7
RESTRICT_NON_HW_TO_HW_PAIRS = True

N_HARMONICS  = 2
DAY_HOURS    = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]       # 08:00–19:59, 12 h
NIGHT_HOURS  = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]           # 20:00–07:59, 12 h
TROPICAL_NIGHT_THRESHOLD = 20.0

BOOT_N      = 1000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

MEDIATION_MIN_N    = 30
MEDIATION_CI_LEVEL = 95
MEDIATION_COVARIATES = [
    "urban_Rn", "urban_BLH", "urban_Pop_Density",
    "urban_SM_Volumetric", "urban_NDVI", "urban_Albedo", "urban_P_mm",
]

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
    "NDVI":          ("ndvi_median_u",          "ndvi_median_r"),
    "Albedo":        ("albedo_u",               "albedo_r"),
}

# 绝对阈值（v3，按洲际）
CONTINENT_THRESHOLD = {
    "Europe": 32.0, "Asia": 35.0, "North America": 35.0,
    "South America": 33.0, "Africa": 40.0,
    "Australia": 38.0, "Oceania": 38.0, "Antarctica": 20.0,
}
DEFAULT_THRESHOLD = 30.0


# ─────────────────────────────────────────────────────────────
# 2. Utility Functions
# ─────────────────────────────────────────────────────────────
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
    """Return NH / SH according to station latitude."""
    try:
        lat = float(lat)
    except Exception:
        return "NH"
    return "SH" if lat < 0 else "NH"


def warm_months_for_lat(lat) -> list:
    """Latitude-dependent warm-season months."""
    return SH_WARM_MONTHS if hemisphere_from_lat(lat) == "SH" else NH_WARM_MONTHS


def warm_season_label_for_lat(lat) -> str:
    """Readable warm-season label."""
    return "DJF" if hemisphere_from_lat(lat) == "SH" else "JJA"


def analysis_start_year() -> int:
    """First calendar year actually loaded by this script."""
    return int(min(YEARS))


def analysis_end_year() -> int:
    """Last calendar year actually loaded by this script."""
    return int(max(YEARS))


def warm_season_year_from_date(date_value, lat) -> int:
    """
    Assign each warm-season date to a season year.

    NH JJA:
        Jun-Aug belong to the same calendar year.

    SH DJF:
        Dec belongs to the next DJF season year.
        Example: 2019-12 belongs to DJF-2020.

    Boundary rule for YEARS=2015-2024:
        - SH Jan-Feb 2015 are retained as DJF-2015.
        - SH Dec 2024 is excluded because DJF-2025 would require Jan-Feb 2025.
        - No attempt is made to read Dec 2014 or dates outside YEARS.
    """
    ts = pd.Timestamp(date_value)

    if hemisphere_from_lat(lat) == "SH" and ts.month == 12:
        return int(ts.year + 1)

    return int(ts.year)


def is_warm_season_date_for_analysis(date_value, lat) -> bool:
    """
    True only for warm-season dates that are inside the available analysis window.

    NH:
        JJA within YEARS.

    SH:
        Jan-Feb within YEARS are retained.
        Dec is retained only if its DJF season is completed inside YEARS.
        Therefore, with YEARS=2015-2024:
            include 2015-01/02;
            include 2015-12 ... 2023-12;
            exclude 2024-12.
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
    Expected warm-season days for validity filtering.

    NH JJA:
        92 days for every season year.

    SH DJF:
        First season year keeps only Jan-Feb because Dec of the previous
        calendar year is outside the loaded YEARS and is intentionally abandoned.
        Later complete DJF seasons use Dec(previous year)+Jan+Feb.
        Dec of max(YEARS) is excluded and therefore DJF-(max(YEARS)+1)
        is not counted.
    """
    season_year = int(season_year)
    if hemisphere_from_lat(lat) == "SH":
        if season_year == analysis_start_year():
            return 60 if is_leap_year(season_year) else 59
        return 91 if is_leap_year(season_year) else 90

    return 92


def warm_months_string(lat) -> str:
    """Store warm-season months in output CSV, e.g. '6,7,8' or '12,1,2'."""
    return ",".join(map(str, warm_months_for_lat(lat)))

def safe_qcut(series: pd.Series, q=4, labels=None):
    s = series.copy()
    if s.dropna().nunique() < q:
        return pd.Series([np.nan] * len(s), index=s.index)
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        return pd.Series([np.nan] * len(s), index=s.index)


def bootstrap_mean_ci(values, n_boot=BOOT_N, alpha=0.05):
    arr = pd.Series(values).dropna().values.astype(float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        return arr[0], arr[0], arr[0]
    boot_means = [
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return float(np.mean(arr)), float(lo), float(hi)


def stars_from_p(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""


def normalize_lcz(val):
    try:
        v = float(val)
    except Exception:
        return np.nan
    if 51 <= v <= 56:
        return int(v - 50)
    return np.nan


def lcz_compactness_class(lcz):
    if pd.isna(lcz): return np.nan
    lcz = int(lcz)
    if lcz in [1]: return "compact"
    if lcz in [6]: return "open"
    return np.nan


def assign_lat_group(lat: float) -> str:
    if pd.isna(lat): return "Unknown"
    if abs(lat) < 23.5: return "Tropical"
    if abs(lat) < 40:   return "Subtropical"
    if abs(lat) < 60:   return "Temperate"
    return "Polar"


# ─────────────────────────────────────────────────────────────
# 3. Load External Metadata
# ─────────────────────────────────────────────────────────────
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
        if c.lower() == "usaf":   rename_map[c] = "USAF"
        elif c.lower() == "wban": rename_map[c] = "WBAN"
    df = df.rename(columns=rename_map)
    for c in STATION_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["USAF", "WBAN"] + [c for c in STATION_VARS if c in df.columns]
    df = (df[keep]
          .drop_duplicates(subset=["USAF", "WBAN"])
          .set_index(["USAF", "WBAN"]))
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


# ─────────────────────────────────────────────────────────────
# 4. Continent lookup
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_continents() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(CONTINENT_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _get_continent(lon: float, lat: float) -> str:
    gdf = _load_continents()
    pt  = Point(lon, lat)
    hits = gdf[gdf.geometry.contains(pt)]
    if len(hits) == 0:
        gdf2 = gdf.copy()
        gdf2["_dist"] = gdf2.geometry.distance(pt)
        hits = gdf2.nsmallest(1, "_dist")
    if len(hits) == 0:
        return ""
    col = next(
        (c for c in hits.columns if c.strip().lower() == "continent"), None
    )
    return str(hits.iloc[0][col]).strip() if col else ""


# ─────────────────────────────────────────────────────────────
# 5. ISD-Lite Reader  [v2 新增露点解析]
# ─────────────────────────────────────────────────────────────
def _parse_isd_token(token: str, scale=10.0, missing=-9999) -> float:
    try:
        v = int(token)
    except ValueError:
        return np.nan
    return np.nan if v == missing else v / scale


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
            df[["year", "month", "day", "hour"]], errors="coerce"
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


# ─────────────────────────────────────────────────────────────
# 6. Data Loading  [v2 clean_year_data 同时处理露点]
# ─────────────────────────────────────────────────────────────
def clean_year_data_to_hourly(df: pd.DataFrame, year: int) -> pd.DataFrame:
    t_start = pd.Timestamp(year, 1, 1, 0)
    t_end   = pd.Timestamp(year, 12, 31, 23)
    full_idx = pd.date_range(t_start, t_end, freq="1h")

    df_idx = df.set_index("datetime").sort_index()

    # 先聚合到 hourly：如果原始有更高频/重复值，取每小时平均
    hourly = df_idx[["temp_C", "dewpoint_C"]].resample("1h").mean()

    # 对齐到完整全年 hourly 轴
    hourly = hourly.reindex(full_idx)

    # 记录哪些值是“真实观测支持”的，避免无限脑补
    obs_hourly = (
        df_idx[["temp_C"]]
        .resample("1h")
        .count()
        .reindex(full_idx)
        .fillna(0)
        .rename(columns={"temp_C": "obs_count"})
    )

    # 只允许补小缺口：例如最多补连续 2 小时
    # method="time" uses linear interpolation along the time axis.
    hourly["temp_C"] = hourly["temp_C"].interpolate(
        method="time", limit=2, limit_area="inside"
    )
    hourly["dewpoint_C"] = hourly["dewpoint_C"].interpolate(
        method="time", limit=2, limit_area="inside"
    )

    # 可选：如果某天完全没有足够观测，就整天置空
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
    out["local_hour"]     = (
        out["local_datetime"].dt.hour
        + out["local_datetime"].dt.minute / 60.0
    )
    out["local_date"]  = out["local_datetime"].dt.date
    out["local_month"] = pd.to_datetime(out["local_datetime"]).dt.month
    return out


def load_multiyear_station_ALL(
        usaf: str, wban: str, years, base_dir: str, lon: float,
        min_valid_frac: float = FULL_YEAR_MIN_VALID_FRAC,
        verbose: bool = True):
    combined    = []
    valid_years = []

    for yr in years:
        year_dir = os.path.join(base_dir, str(yr))
        fpath    = isd_path(usaf, wban, yr, year_dir)

        df_raw = read_isd_lite(fpath)
        if df_raw is None:
            if verbose:
                print(f"    [{yr}] file missing: {fpath}")
            continue

        df_clean = clean_year_data_to_hourly(df_raw, yr)
        if df_clean is None:
            continue

        df_local   = to_local_time(df_clean, lon)
        year_days  = 366 if is_leap_year(yr) else 365
        valid_days = df_local.dropna(subset=["temp_C"])["local_date"].nunique()
        valid_frac = valid_days / year_days

        if valid_frac < min_valid_frac:
            if verbose:
                print(
                    f"    [{yr}] full-year validity {valid_frac:.1%} "
                    f"< {min_valid_frac:.0%} → skip"
                )
            continue

        if verbose:
            print(
                f"    [{yr}] full-year validity {valid_frac:.1%} "
                f"({valid_days}/{year_days} days) → included"
            )
        combined.append(df_local)
        valid_years.append(yr)

    if not combined:
        return None, []
    return pd.concat(combined, ignore_index=True), valid_years


def subset_by_dates(df_loc: pd.DataFrame, dates):
    if df_loc is None or not dates:
        return None
    return df_loc[df_loc["local_date"].isin(dates)].copy()


# ─────────────────────────────────────────────────────────────
# 7. Heatwave Detection
# ─────────────────────────────────────────────────────────────
def compute_hw_threshold_from_tmax(
        tmax_series: pd.Series,
        q: int = HW_PERCENTILE,
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
    thresholds   = {}
    valid_counts = {}

    for d in target_dates:
        d_ts = pd.Timestamp(d)
        doy  = (
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
        thresholds[d]   = float(np.nanpercentile(vals, q)) if len(vals) > 0 else np.nan

    return (
        pd.Series(thresholds).sort_index(),
        pd.Series(valid_counts).sort_index()
    )


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

def detect_heatwave_warm_season_by_year(daily_tmax: pd.Series, threshold, lat) -> pd.Series:
    """
    Detect HW only inside valid hemisphere-specific warm-season dates.

    Consecutive-day detection is applied separately within each warm_season_year.
    This avoids incorrectly connecting SH February directly to the following
    December while still keeping Jan-Feb of the first analysis year.
    """
    s0 = pd.Series(daily_tmax).sort_index()
    out = pd.Series(False, index=s0.index)

    s = s0.dropna()
    if len(s) == 0:
        return out.astype(bool)

    valid_idx = [
        d for d in s.index
        if is_warm_season_date_for_analysis(d, lat)
    ]
    if len(valid_idx) == 0:
        return out.astype(bool)

    s_warm = s.reindex(pd.Index(sorted(valid_idx))).dropna()
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

        hw_part = detect_heatwave(s_part, thr_part)
        out.loc[hw_part.index] = hw_part.astype(bool)

    return out.fillna(False).astype(bool)


def detect_heatwave_absolute_warm_season_by_year(daily_tmax: pd.Series, threshold: float, lat) -> pd.Series:
    """
    Absolute-threshold HW detection using the same warm-season boundary and
    season-year grouping as the percentile HW branch.
    """
    s0 = pd.Series(daily_tmax).sort_index()
    out = pd.Series(False, index=s0.index)

    s = s0.dropna()
    if len(s) == 0:
        return out.astype(bool)

    valid_idx = [
        d for d in s.index
        if is_warm_season_date_for_analysis(d, lat)
    ]
    if len(valid_idx) == 0:
        return out.astype(bool)

    s_warm = s.reindex(pd.Index(sorted(valid_idx))).dropna()
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

        hw_part = detect_heatwave_absolute(s_part, threshold)
        out.loc[hw_part.index] = hw_part.astype(bool)

    return out.fillna(False).astype(bool)

def _series_get(series_like, key, default=np.nan):
    """Safe scalar getter for date-indexed Series."""
    try:
        s = pd.Series(series_like)
        if key in s.index:
            return s.loc[key]
        return s.reindex([key]).iloc[0]
    except Exception:
        return default


def build_daily_heatwave_export_rows(
        pair_id,
        usaf_u, wban_u, usaf_r, wban_r,
        lon_u, lat_u, lon_r, lat_r,
        u_tmax_all, r_tmax_all,
        hw_thr_series_raw,
        hw_thr_series_corrected,
        hw_mask_all,
        hw_mask_warm,
        warm_common,
        continent,
        kg_code,
        kg_group_val,
        climate_zone_main,
        bias_correction_method,
        qm_diagnostics):
    """
    Daily heatwave table exported by analysis_multiyear.py.

    Downstream scripts should read this file instead of recalculating HW.
    Main unified flag:
        hw_flag_percentile_warm_season
    """
    rows = []

    warm_common = set(warm_common) if warm_common is not None else set()
    months_str = warm_months_string(lat_u)
    hemi = hemisphere_from_lat(lat_u)
    season_label = warm_season_label_for_lat(lat_u)

    all_dates = sorted(pd.Index(u_tmax_all.index).unique())

    hw_mask_all = pd.Series(hw_mask_all).reindex(all_dates).fillna(False)
    hw_mask_warm = pd.Series(hw_mask_warm).reindex(all_dates).fillna(False)

    r_tmax_aligned = pd.Series(r_tmax_all).reindex(all_dates)
    thr_raw_aligned = pd.Series(hw_thr_series_raw).reindex(all_dates)
    thr_corr_aligned = pd.Series(hw_thr_series_corrected).reindex(all_dates)

    for d in all_dates:
        d_ts = pd.Timestamp(d)
        is_warm = d in warm_common

        u_tx = _series_get(u_tmax_all, d, np.nan)
        r_tx = _series_get(r_tmax_aligned, d, np.nan)
        thr_raw = _series_get(thr_raw_aligned, d, np.nan)
        thr_corr = _series_get(thr_corr_aligned, d, np.nan)

        hw_warm = bool(hw_mask_warm.loc[d]) if is_warm else False

        rows.append({
            "pair_id": pair_id,
            "date": d_ts.strftime("%Y-%m-%d"),
            "year": int(d_ts.year),
            "month": int(d_ts.month),
            "warm_season_year": int(warm_season_year_from_date(d, lat_u)),
            "hemisphere": hemi,
            "warm_season_label": season_label,
            "warm_season_months": months_str,
            "is_warm_season": int(is_warm),

            "urban_usaf": usaf_u,
            "urban_wban": wban_u,
            "rural_usaf": usaf_r,
            "rural_wban": wban_r,
            "lon_urban": lon_u,
            "lat_urban": lat_u,
            "lon_rural": lon_r,
            "lat_rural": lat_r,

            "urban_tmax": float(u_tx) if pd.notna(u_tx) else np.nan,
            "rural_tmax": float(r_tx) if pd.notna(r_tx) else np.nan,
            "delta_tmax": (
                float(u_tx - r_tx)
                if pd.notna(u_tx) and pd.notna(r_tx) else np.nan
            ),

            "hw_threshold_raw": float(thr_raw) if pd.notna(thr_raw) else np.nan,
            "hw_threshold_corrected": (
                float(thr_corr) if pd.notna(thr_corr) else np.nan
            ),
            "hw_exceed_corrected": (
                int(u_tx > thr_corr)
                if pd.notna(u_tx) and pd.notna(thr_corr) else 0
            ),

            # Diagnostic only: full-year continuous-event detection.
            "hw_flag_percentile_all_year_detection": int(bool(hw_mask_all.loc[d])),

            # Main unified downstream flag.
            "hw_flag_percentile_warm_season": int(hw_warm),
            "nhw_flag_percentile_warm_season": int(is_warm and not hw_warm),

            "hw_ref_mode": HW_REF_MODE,
            "hw_percentile": HW_PERCENTILE,
            "hw_window_half": HW_WINDOW_HALF,
            "hw_min_days": HW_MIN_DAYS,
            "bias_correction_method": bias_correction_method,
            "qm_ref_mode": qm_diagnostics.get("qm_ref_mode", ""),
            "qm_overlap_days": qm_diagnostics.get("overlap_days", np.nan),

            "continent": continent,
            "kg_code": kg_code,
            "kg_group": kg_group_val,
            "climate_zone_main": climate_zone_main,
        })

    return rows

# ─────────────────────────────────────────────────────────────
# 7b. 绝对阈值热浪检测（v3）
# ─────────────────────────────────────────────────────────────
def get_absolute_hw_threshold(continent: str) -> float:
    """根据洲际返回绝对温度阈值（°C），用于稳健性对比。"""
    return CONTINENT_THRESHOLD.get(continent, DEFAULT_THRESHOLD)


def detect_heatwave_absolute(
        daily_tmax: pd.Series,
        threshold: float,
        min_days: int = HW_MIN_DAYS) -> pd.Series:
    """绝对阈值版热浪检测：Tmax > threshold 且连续 ≥ min_days 天。"""
    above = (daily_tmax > threshold).values
    hw    = np.zeros(len(above), dtype=bool)
    n, i  = len(above), 0
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


# ─────────────────────────────────────────────────────────────
# 8. FFT & Diurnal Metrics  [v2 新增露点日循环输出]
# ─────────────────────────────────────────────────────────────
def reconstruct_diurnal(mean_t, amps, phases, n_points=24):
    t   = np.arange(n_points)
    sig = np.full(n_points, mean_t, dtype=float)
    for k, (a, p) in enumerate(zip(amps, phases), start=1):
        sig += a * np.cos(k * 2 * np.pi * t / n_points + p)
    return sig


def compute_fft(df_local: pd.DataFrame, n_harm=N_HARMONICS):
    """
    [v2] 在原有温度 FFT 基础上，新增露点温度24小时均值曲线计算。
    露点不做 FFT 重建，直接取各小时多年平均值。
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
    # Use linear interpolation (not PCHIP) to fill any missing hourly bins
    # after grouping by hour. This is a minor cosmetic fill for the FFT input;
    # PCHIP was replaced with linear to unify all interpolation across the pipeline.
    hourly_t = (
        hourly_t
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    if hourly_t.isna().any():
        return None

    vals  = hourly_t.values.astype(float)
    coef  = sp_fft(vals) / len(vals)
    mean_t = float(np.real(coef[0]))
    amps, phases = [], []
    for k in range(1, n_harm + 1):
        amps.append(float(2 * np.abs(coef[k])))
        phases.append(float(np.angle(coef[k])))

    reconstructed = reconstruct_diurnal(mean_t, amps, phases)

    # Dewpoint diurnal curve: use linear interpolation (not PCHIP)
    # to fill sparse hourly bins. This curve is used only for WBGT/labour,
    # not for temperature-based main analysis.
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
        # Dewpoint QC diagnostics (for WBGT/labour use only)
        dew_valid_hours = int(hourly_d.notna().sum())
        dew_valid_frac = float(dew_valid_hours / 24.0)
    else:
        dew_curve = np.full(24, np.nan)
        dew_valid_hours = 0
        dew_valid_frac = 0.0

    return dict(
        mean        = mean_t,
        amplitudes  = amps,
        phases      = phases,
        Tmax_fft    = float(np.max(reconstructed)),
        Tmin_fft    = float(np.min(reconstructed)),
        hourly_obs  = vals,
        hourly_dew  = dew_curve,
        n_days      = int(tmp["local_date"].nunique()),
        # Dewpoint QC diagnostics (used only for WBGT/labour filtering)
        dewpoint_valid_hour_count = dew_valid_hours,
        dewpoint_valid_frac       = dew_valid_frac,
    )

def detect_native_resolution(df):
    diffs = df["datetime"].diff().dt.total_seconds().dropna() / 3600
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


def compute_daily_minmax(df_loc: pd.DataFrame):
    if df_loc is None or len(df_loc) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = df_loc.groupby("local_date")["temp_C"]
    return g.min().dropna(), g.max().dropna()


def curve_delta_metrics(u_curve, r_curve):
    delta = np.array(u_curve) - np.array(r_curve)
    return {
        "dT_day_mean":        float(np.mean(delta[DAY_HOURS])),
        "dT_night_mean":      float(np.mean(delta[NIGHT_HOURS])),
        "daytime_uci_flag":   int(np.max(u_curve) - np.max(r_curve) < 0),
        "nighttime_uhi_flag": int(np.min(u_curve) - np.min(r_curve) > 0),
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
            (np.maximum(u - TROPICAL_NIGHT_THRESHOLD, 0)
             - np.maximum(r - TROPICAL_NIGHT_THRESHOLD, 0)).mean()
        ),
        urban_daily_tmin_mean=float(u.mean()),
        rural_daily_tmin_mean=float(r.mean()),
        delta_daily_tmin_mean=float((u - r).mean()),
        risk_common_days=int(len(common)),
    )


# ─────────────────────────────────────────────────────────────
# 9. Regression / Mechanism & Mediation Outputs
# ─────────────────────────────────────────────────────────────
def linear_relation_table(df, x, y, subset_name):
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return {
            "subset": subset_name, "x": x, "y": y, "n": len(sub),
            "slope": np.nan, "intercept": np.nan, "r": np.nan,
            "r2": np.nan, "p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
        }
    lr = stats.linregress(sub[x], sub[y])
    rho, rho_p = stats.spearmanr(sub[x], sub[y], nan_policy="omit")
    return {
        "subset": subset_name, "x": x, "y": y, "n": int(len(sub)),
        "slope": float(lr.slope), "intercept": float(lr.intercept),
        "r": float(lr.rvalue), "r2": float(lr.rvalue ** 2),
        "p": float(lr.pvalue),
        "spearman_rho": float(rho) if not pd.isna(rho) else np.nan,
        "spearman_p":   float(rho_p) if not pd.isna(rho_p) else np.nan,
    }


def save_mechanism_regressions(all_df, output_dir):
    """
    [v5] 在原有 annual/heatwave/non_heatwave/warm_season 子集基础上，
    新增 annual_kg_A / B / C / D / E 分层，
    便于审查机制链条在不同气候背景下的一致性。
    """
    annual    = all_df[all_df["period"] == "annual"].copy()
    relations = [
        ("urban_BV_m3", "dAmp1"),
        ("dAmp1",        "dTx"),
        ("dAmp1",        "dTn"),
        ("urban_BV_m3",  "dTx"),
        ("urban_BV_m3",  "dT_night_mean"),
        ("urban_BV_m3",  "delta_tropical_night_freq"),
    ]
    subsets = {
        "annual_all":       annual,
        "annual_UHI":       annual[annual["group"] == "UHI"],
        "annual_UCI":       annual[annual["group"] == "UCI"],
        "heatwave_all":     all_df[all_df["period"] == "heatwave"],
        "non_heatwave_all": all_df[all_df["period"] == "non_heatwave"],
        "warm_season_all":  all_df[all_df["period"] == "warm_season"],
    }

    # v5：按 KG 主分组追加分层子集
    if "kg_group" in annual.columns:
        for kg in KG_GROUPS_ALL:
            sub_kg = annual[annual["kg_group"] == kg]
            if len(sub_kg) >= 5:
                subsets[f"annual_kg_{kg}"] = sub_kg

    rows = [
        linear_relation_table(s, x, y, n)
        for n, s in subsets.items()
        for x, y in relations
    ]
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "mechanism_regression_table.csv"), index=False
    )
    print("  Saved: mechanism_regression_table.csv")


def _standardize_series(s):
    s   = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def _ols_lstsq(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def bootstrap_single_mediation(
        df, x, m, y, covariates=None,
        n_boot=BOOT_N, ci_level=MEDIATION_CI_LEVEL,
        seed=RANDOM_SEED, min_n=MEDIATION_MIN_N,
        standardize=True):
    covariates = covariates or []
    cols = [x, m, y] + covariates
    sub  = df[cols].dropna().copy()
    if len(sub) < min_n:
        return None
    if standardize:
        for c in cols:
            sub[c] = _standardize_series(sub[c])

    n  = len(sub)
    xa = sub[x].values.astype(float)
    ma = sub[m].values.astype(float)
    ya = sub[y].values.astype(float)
    ca = sub[covariates].values.astype(float) if covariates else np.empty((n, 0))

    beta_a  = _ols_lstsq(np.column_stack([np.ones(n), xa, ca]), ma)
    a       = float(beta_a[1])
    beta_b  = _ols_lstsq(np.column_stack([np.ones(n), xa, ma, ca]), ya)
    c_prime = float(beta_b[1])
    b       = float(beta_b[2])
    beta_c  = _ols_lstsq(np.column_stack([np.ones(n), xa, ca]), ya)
    c_total = float(beta_c[1])
    indirect = float(a * b)

    rng_l   = np.random.default_rng(seed)
    boot_ab = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng_l.integers(0, n, size=n)
        xb, mb, yb = xa[idx], ma[idx], ya[idx]
        cb = ca[idx, :] if ca.size else np.empty((len(idx), 0))
        a_i = _ols_lstsq(np.column_stack([np.ones(len(idx)), xb, cb]), mb)[1]
        b_i = _ols_lstsq(np.column_stack([np.ones(len(idx)), xb, mb, cb]), yb)[2]
        boot_ab[i] = a_i * b_i

    alpha_v  = (100 - ci_level) / 200
    ci_low, ci_high = np.quantile(boot_ab, [alpha_v, 1 - alpha_v])
    pm = 100.0 * indirect / c_total if abs(c_total) > 1e-12 else np.nan
    return {
        "n": n, "x": x, "m": m, "y": y,
        "a": a, "b": b, "c_total": c_total, "c_prime": c_prime,
        "ab_indirect": indirect,
        "ab_ci_low":   float(ci_low),
        "ab_ci_high":  float(ci_high),
        "ab_sig":      int(not (ci_low <= 0 <= ci_high)),
        "pm_percent":  pm,
    }


def save_mediation_analysis(all_df, output_dir):
    """
    [v5] 在原有结构化子集基础上，新增 annual_kg_A/B/C/D/E 分层，
    检验机制链条（BV → dAmp1 → dTx/dTn）在不同气候带下的异质性。
    """
    annual = all_df[all_df["period"] == "annual"].copy()
    if annual.empty:
        return
    annual["log10_BV"] = np.log10(annual["urban_BV_m3"].replace(0, np.nan))
    valid_covs = [
        c for c in MEDIATION_COVARIATES
        if c in annual.columns and annual[c].notna().mean() >= 0.50
    ]
    print(f"  Mediation covariates: {valid_covs}")

    subset_dict = {
        "annual_all":     annual,
        "annual_UHI":     annual[annual["group"] == "UHI"],
        "annual_UCI":     annual[annual["group"] == "UCI"],
        "annual_compact": annual[annual["urban_lcz_class"] == "compact"],
        "annual_open":    annual[annual["urban_lcz_class"] == "open"],
    }

    # v5：按 KG 主分组追加子样本
    if "kg_group" in annual.columns:
        for kg in KG_GROUPS_ALL:
            sub_kg = annual[annual["kg_group"] == kg]
            if len(sub_kg) >= MEDIATION_MIN_N:
                subset_dict[f"annual_kg_{kg}"] = sub_kg

    rows = []
    for sname, sdf in subset_dict.items():
        for outcome in ["dTx", "dTn"]:
            res = bootstrap_single_mediation(
                sdf, "log10_BV", "dAmp1", outcome,
                covariates=valid_covs, n_boot=BOOT_N,
                ci_level=MEDIATION_CI_LEVEL, seed=RANDOM_SEED,
                min_n=MEDIATION_MIN_N, standardize=True,
            )
            if res is None:
                rows.append({
                    "subset": sname, "outcome": outcome,
                    "n": int(sdf[["log10_BV", "dAmp1", outcome]].dropna().shape[0]),
                    "ab_indirect": np.nan, "ab_ci_low": np.nan,
                    "ab_ci_high": np.nan, "ab_sig": np.nan,
                    "pm_percent": np.nan,
                    "covariates": ";".join(valid_covs),
                })
            else:
                res["subset"]     = sname
                res["outcome"]    = outcome
                res["covariates"] = ";".join(valid_covs)
                rows.append(res)

    out_csv = os.path.join(output_dir, "mediation_BV_dAmp1_to_dTx_dTn.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")


def save_hw_pressure_test(all_df, output_dir):
    metrics = [
        "dTmean", "dAmp1", "dTx", "dTn",
        "dT_day_mean", "dT_night_mean",
        "delta_tropical_night_freq", "delta_hotnight_excess",
    ]
    sub = all_df[all_df["period"].isin(["heatwave", "non_heatwave"])].copy()
    if sub.empty:
        return

    pivot_rows = []
    for group in ["UHI", "UCI"]:
        for pair_id, gp in sub[sub["group"] == group].groupby("pair_id"):
            row = {"pair_id": pair_id, "group": group}
            ok  = True
            for m in metrics:
                hw  = gp.loc[gp["period"] == "heatwave",     m]
                nhw = gp.loc[gp["period"] == "non_heatwave", m]
                if len(hw) == 0 or len(nhw) == 0:
                    ok = False
                    continue
                row[f"{m}_heatwave"]     = float(hw.iloc[0])
                row[f"{m}_non_heatwave"] = float(nhw.iloc[0])
                row[f"{m}_hw_minus_nhw"] = float(hw.iloc[0] - nhw.iloc[0])
            if ok:
                pivot_rows.append(row)

    if not pivot_rows:
        print("  Warning: no complete HW/non-HW pairs")
        return

    hwcmp = pd.DataFrame(pivot_rows)
    hwcmp.to_csv(
        os.path.join(output_dir, "hw_pressure_test_pairwise.csv"), index=False
    )

    summary_rows = []
    for group, gp in hwcmp.groupby("group"):
        for m in metrics:
            vals = gp.get(f"{m}_hw_minus_nhw", pd.Series(dtype=float))
            mn, lo, hi = bootstrap_mean_ci(vals)
            summary_rows.append({
                "group": group, "metric": m,
                "n": int(vals.dropna().shape[0]),
                "mean_hw_minus_nhw": mn, "ci_low": lo, "ci_high": hi,
            })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "hw_pressure_test_summary.csv"), index=False
    )
    print("  Saved: hw_pressure_test_pairwise.csv / hw_pressure_test_summary.csv")


def save_nighttime_risk_outputs(all_df, output_dir):
    metrics = [
        "delta_tropical_night_freq", "delta_hotnight_excess",
        "delta_daily_tmin_mean", "dTn", "dT_night_mean",
    ]
    rows = []
    for (period, group), g in all_df.groupby(["period", "group"]):
        for m in metrics:
            mn, lo, hi = bootstrap_mean_ci(g[m])
            rows.append({
                "period": period, "group": group, "metric": m,
                "n": int(g[m].dropna().shape[0]),
                "mean": mn, "ci_low": lo, "ci_high": hi,
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "nighttime_heat_risk_summary.csv"), index=False
    )
    print("  Saved: nighttime_heat_risk_summary.csv")


def save_bv_quartile_analysis(all_df, output_dir):
    annual = all_df[all_df["period"] == "annual"].copy()
    if annual.empty:
        return
    annual["BV_quartile"] = safe_qcut(
        annual["urban_BV_m3"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    rows = []
    for q, g in annual.groupby("BV_quartile", dropna=False):
        if pd.isna(q) or len(g) == 0:
            continue
        rows.append({
            "BV_quartile":                 str(q),
            "n":                           len(g),
            "urban_BV_mean":               g["urban_BV_m3"].mean(),
            "dAmp1_mean":                  g["dAmp1"].mean(),
            "dTx_mean":                    g["dTx"].mean(),
            "dTn_mean":                    g["dTn"].mean(),
            "daytime_UCI_probability":     g["daytime_uci_flag"].mean(),
            "delta_tropical_night_freq_mean": g["delta_tropical_night_freq"].mean(),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "bv_quartile_summary.csv"), index=False
    )
    print("  Saved: bv_quartile_summary.csv")


def save_latgroup_robustness(all_df, output_dir):
    """纬度带汇总——保留作兼容性输出，主分组已由 KG 承担（v5）。"""
    annual = all_df[all_df["period"] == "annual"].copy()
    if "lat_group" not in annual.columns or annual.empty:
        return
    rows = []
    for latg, g in annual.groupby("lat_group"):
        rows.append({
            "lat_group":               latg,
            "n":                       len(g),
            "dAmp1_mean":              g["dAmp1"].mean(),
            "dTx_mean":                g["dTx"].mean(),
            "dTn_mean":                g["dTn"].mean(),
            "daytime_UCI_probability": g["daytime_uci_flag"].mean(),
        })
    pd.DataFrame(rows).sort_values("lat_group").to_csv(
        os.path.join(output_dir, "latgroup_robustness_summary.csv"), index=False
    )
    print("  Saved: latgroup_robustness_summary.csv  "
          "[注：lat_group 为兼容性保留字段，主气候分组已改用 KG]")


def save_kggroup_robustness(all_df, output_dir):
    """
    [v5 新增] 按 Köppen-Geiger 主分组（A/B/C/D/E）输出稳健性汇总。
    同时附 kg_code 细分行，便于进一步下钻。

    输出文件：kggroup_robustness_summary.csv
    """
    annual = all_df[all_df["period"] == "annual"].copy()
    if "kg_group" not in annual.columns or annual.empty:
        print("  Warning: kg_group not found, skip save_kggroup_robustness()")
        return

    metrics = ["dAmp1", "dTx", "dTn", "dT_day_mean", "dT_night_mean",
               "delta_tropical_night_freq", "delta_hotnight_excess",
               "daytime_uci_flag"]

    rows = []

    # ── A. 按 KG 主组（单字母）汇总 ─────────────────────────────
    for kg, g in annual.groupby("kg_group", dropna=True):
        if len(g) == 0:
            continue
        row = {
            "level":          "kg_group",
            "kg_group":       kg,
            "kg_code":        "ALL",
            "climate_zone":   KG_GROUP_MAP.get(str(kg), kg),
            "n":              len(g),
            "n_UHI":          int((g["group"] == "UHI").sum()),
            "n_UCI":          int((g["group"] == "UCI").sum()),
        }
        for m in metrics:
            if m in g.columns:
                mn, lo, hi = bootstrap_mean_ci(g[m])
                row[f"{m}_mean"]    = mn
                row[f"{m}_ci_low"]  = lo
                row[f"{m}_ci_high"] = hi
        rows.append(row)

    # ── B. 按 KG 细码（如 Cfa、Dfb）汇总（n ≥ 5 才输出）─────────
    if "kg_code" in annual.columns:
        for kgc, g in annual.groupby("kg_code", dropna=True):
            if len(g) < 5:
                continue
            row = {
                "level":        "kg_code",
                "kg_group":     str(kgc)[0] if (isinstance(kgc, str) and len(kgc) > 0) else np.nan,
                "kg_code":      kgc,
                "climate_zone": KG_GROUP_MAP.get(str(kgc)[0], kgc),
                "n":            len(g),
                "n_UHI":        int((g["group"] == "UHI").sum()),
                "n_UCI":        int((g["group"] == "UCI").sum()),
            }
            for m in metrics:
                if m in g.columns:
                    mn, lo, hi = bootstrap_mean_ci(g[m])
                    row[f"{m}_mean"]    = mn
                    row[f"{m}_ci_low"]  = lo
                    row[f"{m}_ci_high"] = hi
            rows.append(row)

    if not rows:
        print("  Warning: no KG groups found, kggroup_robustness_summary.csv skipped")
        return

    out_path = os.path.join(output_dir, "kggroup_robustness_summary.csv")
    pd.DataFrame(rows).sort_values(["level", "kg_group", "kg_code"]).to_csv(
        out_path, index=False
    )
    print(f"  Saved: kggroup_robustness_summary.csv  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────
# 10. build_period_records（v5：增加版本信息字段 + KG 字段）
# ─────────────────────────────────────────────────────────────
def build_period_records(
        pair_id, row, group, continent,
        combined_u_all, combined_r_all,
        combined_u_warm, combined_r_warm,
        has_warm,
        hw_dates_warm, nhw_dates_warm,
        hw_thr_mean, hw_dates_all, nhw_dates_all,
        hw_intensity, delta_tmax_ann,
        valid_yrs_u_all, valid_yrs_r_all,
        valid_yrs_u_warm, valid_yrs_r_warm,
        bv_u, bv_r, bv_delta,
        station_meta_vals,
        hw_method: str,
        abs_threshold: float,
        kg_code: str = "",
        kg_group_val: str = "",
        climate_zone_main: str = "",
) -> list:
    """
    [v5] rec 中新增：
      - hne_ref_mode   : 热夜超值参考口径标识
      - hw_ref_mode    : 热浪阈值计算口径标识
      - kg_code        : Köppen-Geiger 细码（如 Cfa）
      - kg_group       : KG 主分组字母（如 C）
      - climate_zone_main : KG 主气候带名称（如 Temperate）

    时期说明（v4 延续）：
      annual       → 全年平均（背景基准）
      warm_season  → JJA 全部日期（不区分热浪）
      heatwave     → JJA 中的热浪日
      non_heatwave → JJA 中的非热浪日
    """
    has_hw_warm = len(hw_dates_warm) > 0

    periods = {"annual": (combined_u_all, combined_r_all)}

    if has_warm:
        periods["warm_season"] = (combined_u_warm, combined_r_warm)

    if has_warm and has_hw_warm:
        periods["heatwave"] = (
            subset_by_dates(combined_u_warm, hw_dates_warm),
            subset_by_dates(combined_r_warm, hw_dates_warm),
        )
    if has_warm and ((not RESTRICT_NON_HW_TO_HW_PAIRS) or has_hw_warm):
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

        rec = {
            # ── 标识 ──────────────────────────────────────────────
            "pair_id":              pair_id,
            "period":               period_name,
            "data_source":          data_source,
            "hw_method":            hw_method,
            "hw_abs_threshold":     abs_threshold,
            # v5 新增：版本信息字段，便于多版本溯源
            "hne_ref_mode":         HNE_REF_MODE,
            "hw_ref_mode":          HW_REF_MODE,
            "group":                group,
            "continent":            continent,
            # Hemisphere-aware warm-season metadata
            "hemisphere":           hemisphere_from_lat(row["lat_urban"]),
            "warm_season_label":    warm_season_label_for_lat(row["lat_urban"]),
            "warm_season_months":   warm_months_string(row["lat_urban"]),
            # v5 新增：KG 气候分区字段
            "kg_code":              kg_code,
            "kg_group":             kg_group_val,
            "climate_zone_main":    climate_zone_main,
            # 纬度带——保留作兼容性字段
            "lat_group":            row.get("lat_group", np.nan),
            "urban_lcz_raw":        row["urban_lcz"],
            "urban_lcz_corrected":  row["urban_lcz_corrected"],
            "urban_lcz_class":      row["urban_lcz_class"],
            "lon_urban":            float(row["lon_urban"]),
            "lat_urban":            float(row["lat_urban"]),
            "lon_rural":            float(row["lon_rural"]),
            "lat_rural":            float(row["lat_rural"]),
            # ── 热浪诊断 ──────────────────────────────────────────
            "hw_threshold_mean":    hw_thr_mean,
            "n_hw_days_annual":     len(hw_dates_all),
            "n_nhw_days_annual":    len(nhw_dates_all),
            "n_hw_days_warm":       len(hw_dates_warm),
            "n_nhw_days_warm":      len(nhw_dates_warm),
            "hw_intensity":         hw_intensity,
            "delta_tmax_ann":       delta_tmax_ann,
            # ── 有效年数 ──────────────────────────────────────────
            "n_valid_years_urban_all":  len(valid_yrs_u_all),
            "n_valid_years_rural_all":  len(valid_yrs_r_all),
            "n_valid_years_urban_warm": len(valid_yrs_u_warm) if has_warm else 0,
            "n_valid_years_rural_warm": len(valid_yrs_r_warm) if has_warm else 0,
            # ── 建筑体积 ──────────────────────────────────────────
            "urban_BV_m3":  bv_u,
            "rural_BV_m3":  bv_r,
            "delta_BV_m3":  bv_delta,
            "log10_BV": (
                np.log10(bv_u) if (not np.isnan(bv_u) and bv_u > 0) else np.nan
            ),
            # ── FFT 结果 ──────────────────────────────────────────
            "urban_Tmean":    fft_u["mean"],
            "urban_Tmax_fft": fft_u["Tmax_fft"],
            "urban_Tmin_fft": fft_u["Tmin_fft"],
            "urban_ndays":    fft_u["n_days"],
            "rural_Tmean":    fft_r["mean"],
            "rural_Tmax_fft": fft_r["Tmax_fft"],
            "rural_Tmin_fft": fft_r["Tmin_fft"],
            "rural_ndays":    fft_r["n_days"],
            "dTmean": fft_u["mean"]          - fft_r["mean"],
            "dAmp1":  fft_u["amplitudes"][0]  - fft_r["amplitudes"][0],
            "dAmp2":  (
                fft_u["amplitudes"][1] - fft_r["amplitudes"][1]
                if len(fft_u["amplitudes"]) > 1 else np.nan
            ),
            "dTx": fft_u["Tmax_fft"] - fft_r["Tmax_fft"],
            "dTn": fft_u["Tmin_fft"] - fft_r["Tmin_fft"],
        }

        rec.update(curve_delta_metrics(u_curve, r_curve))
        rec.update(compute_daily_risk_metrics(sub_u, sub_r))
        rec.update(station_meta_vals)

        for k in range(N_HARMONICS):
            rec[f"urban_Amp{k+1}"]   = (
                fft_u["amplitudes"][k] if k < len(fft_u["amplitudes"]) else np.nan
            )
            rec[f"urban_Phase{k+1}"] = (
                fft_u["phases"][k] if k < len(fft_u["phases"]) else np.nan
            )
            rec[f"rural_Amp{k+1}"]   = (
                fft_r["amplitudes"][k] if k < len(fft_r["amplitudes"]) else np.nan
            )
            rec[f"rural_Phase{k+1}"] = (
                fft_r["phases"][k] if k < len(fft_r["phases"]) else np.nan
            )

        for h in range(24):
            rec[f"urban_diurnal_h{h:02d}"] = float(u_curve[h])
            rec[f"rural_diurnal_h{h:02d}"] = float(r_curve[h])

        for h in range(24):
            u_dew = fft_u["hourly_dew"][h] if fft_u.get("hourly_dew") is not None else np.nan
            r_dew = fft_r["hourly_dew"][h] if fft_r.get("hourly_dew") is not None else np.nan
            rec[f"urban_dew_h{h:02d}"] = float(u_dew) if not np.isnan(u_dew) else np.nan
            rec[f"rural_dew_h{h:02d}"] = float(r_dew) if not np.isnan(r_dew) else np.nan

        # Dewpoint QC diagnostics (for WBGT/labour filtering only;
        # do NOT use for temperature-based main analysis filtering)
        dew_u_valid = fft_u.get("dewpoint_valid_hour_count", 0)
        dew_r_valid = fft_r.get("dewpoint_valid_hour_count", 0)
        rec["dewpoint_valid_hour_count_urban"] = dew_u_valid
        rec["dewpoint_valid_hour_count_rural"] = dew_r_valid
        rec["dewpoint_valid_frac_urban"]       = fft_u.get("dewpoint_valid_frac", 0.0)
        rec["dewpoint_valid_frac_rural"]       = fft_r.get("dewpoint_valid_frac", 0.0)
        rec["has_dewpoint_for_wbgt"] = int(
            (dew_u_valid >= 12) and (dew_r_valid >= 12)
        )
        rec["wbgt_qc_pass"] = int(rec["has_dewpoint_for_wbgt"])

        records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────
# 11. process_single_pair（v5：提取 KG 字段并传入 build_period_records）
# ─────────────────────────────────────────────────────────────
def process_single_pair(args):
    """处理单个站点对的所有逻辑，用于多进程调用。"""
    idx, row = args
    pair_id = row["pair_id"]
    lon_u   = float(row["lon_urban"])
    lat_u   = float(row["lat_urban"])
    lon_r   = float(row["lon_rural"])
    lat_r   = float(row["lat_rural"])

    records_out      = []
    threshold_out    = []
    station_year_out = []
    hw_daily_out     = []
    error_info       = None

    print(f"[{idx}] Start processing: {pair_id}")

    try:
        usaf_u, wban_u, usaf_r, wban_r = parse_pair_id(pair_id)
    except ValueError as e:
        error_info = {"pair_id": pair_id, "fail_step": "parse_pair_id",
                      "missing_data": f"ID解析失败: {e}"}
        return records_out, threshold_out, station_year_out, hw_daily_out, error_info

    # ── 1. 加载全年数据 ─────────────────────────────────────────
    combined_u_all, valid_yrs_u_all = load_multiyear_station_ALL(
        usaf_u, wban_u, YEARS, ISD_BASE_DIR, lon_u, verbose=False
    )
    combined_r_all, valid_yrs_r_all = load_multiyear_station_ALL(
        usaf_r, wban_r, YEARS, ISD_BASE_DIR, lon_r, verbose=False
    )

    if combined_u_all is None or combined_r_all is None:
        missing = []
        if combined_u_all is None: missing.append("urban_full_year(城市全年数据有效率不足)")
        if combined_r_all is None: missing.append("rural_full_year(乡村全年数据有效率不足)")
        error_info = {"pair_id": pair_id, "fail_step": "load_multiyear_station_ALL",
                      "missing_data": " & ".join(missing)}
        return records_out, threshold_out, station_year_out, hw_daily_out, error_info

    # ── 2. 在内存中切分暖季数据 ──────────────────────────────────
    # ── 2. 在内存中切分暖季数据：NH=JJA, SH=DJF ─────────────────────
    # Use the urban station latitude to define the pair-level warm season.
    # This keeps urban/rural paired dates consistent.
    pair_hemisphere = hemisphere_from_lat(lat_u)
    pair_warm_label = warm_season_label_for_lat(lat_u)
    pair_warm_months = warm_months_for_lat(lat_u)

    def slice_warm_season(df_all, lat_for_season):
        if df_all is None:
            return None, []

        df_warm = df_all[
            df_all["local_date"].apply(
                lambda d: is_warm_season_date_for_analysis(d, lat_for_season)
            )
        ].copy()
        if df_warm.empty:
            return None, []

        df_warm["warm_season_year"] = df_warm["local_date"].apply(
            lambda d: warm_season_year_from_date(d, lat_for_season)
        )

        valid_yrs = []
        for season_year, grp in df_warm.groupby("warm_season_year"):
            expected_days = expected_warm_season_days(season_year, lat_for_season)
            valid_days = grp.dropna(subset=["temp_C"])["local_date"].nunique()

            if (valid_days / expected_days) >= MIN_YEAR_VALID_FRAC:
                valid_yrs.append(int(season_year))

        if not valid_yrs:
            return None, []

        df_warm_filtered = df_warm[
            df_warm["warm_season_year"].isin(valid_yrs)
        ].copy()

        return df_warm_filtered, valid_yrs

    combined_u_warm, valid_yrs_u_warm = slice_warm_season(combined_u_all, lat_u)
    combined_r_warm, valid_yrs_r_warm = slice_warm_season(combined_r_all, lat_u)
    has_warm = (combined_u_warm is not None and combined_r_warm is not None)

    # ── 3. 收集站点有效年份信息 ──────────────────────────────────
    n_total = len(YEARS)
    for station_type, usaf_id, wban_id, yrs_all, yrs_warm in [
        ("urban", usaf_u, wban_u, valid_yrs_u_all,
         valid_yrs_u_warm if has_warm else []),
        ("rural", usaf_r, wban_r, valid_yrs_r_all,
         valid_yrs_r_warm if has_warm else []),
    ]:
        station_year_out.append({
            "pair_id":            pair_id,
            "station_type":       station_type,
            "usaf":               usaf_id,
            "wban":               wban_id,
            "valid_years_all":    ",".join(map(str, yrs_all)),
            "n_valid_years_all":  len(yrs_all),
            "valid_frac_all":     round(len(yrs_all) / n_total, 4),
            "valid_years_warm":   ",".join(map(str, yrs_warm)),
            "n_valid_years_warm": len(yrs_warm),
            "valid_frac_warm":    round(len(yrs_warm) / n_total, 4),
            "hemisphere":         pair_hemisphere,
            "warm_season_label":  pair_warm_label,
            "warm_season_months": ",".join(map(str, pair_warm_months)),
        })

    u_tmin_all, u_tmax_all = compute_daily_minmax(combined_u_all)
    r_tmin_all, r_tmax_all = compute_daily_minmax(combined_r_all)
    common_dates_all = u_tmax_all.index.intersection(r_tmax_all.index)

    if len(common_dates_all) == 0:
        error_info = {"pair_id": pair_id, "fail_step": "compute_daily_minmax",
                      "missing_data": "城乡站点没有共同的Tmax有效日期(common_dates为空)"}
        return records_out, threshold_out, station_year_out, hw_daily_out, error_info

    delta_tmax_ann = float(
        (u_tmax_all[common_dates_all] - r_tmax_all[common_dates_all]).mean()
    )
    group = "UHI" if delta_tmax_ann > 0 else "UCI"

    _era5_tmax_u = load_era5_tmax_series(usaf_u, wban_u, "urban") \
        if USE_ERA5_CLIMATOLOGY else None

    if _era5_tmax_u is None or len(_era5_tmax_u) == 0:
        error_info = {
            "pair_id": pair_id,
            "fail_step": "ERA5_missing",
            "missing_data": "ERA5 Tmax CSV 缺失或数据量不足，跳过该站点"
        }
        return records_out, threshold_out, station_year_out, hw_daily_out, error_info

    hw_thr_series_raw = apply_doy_thr_to_index(
        u_tmax_all.index,
        compute_hw_doy_thr_era5(_era5_tmax_u)
    )

    valid_count_series = pd.Series(len(_era5_tmax_u), index=u_tmax_all.index)

    if USE_QM_BIAS_CORRECTION:
        hw_thr_series_corrected, bias_diag = apply_quantile_mapping_bias_correction(
            hw_thr_series_raw=hw_thr_series_raw,
            isd_tmax=u_tmax_all,
            era5_tmax=_era5_tmax_u,
            min_overlap_days_per_month=QM_MIN_OVERLAP_DAYS_PER_MONTH,
            min_overlap_days_annual=QM_MIN_OVERLAP_DAYS_ANNUAL,
            n_quantiles=QM_N_QUANTILES,
        )
        bias_correction_method = "quantile_mapping_empirical_cdf"
        qm_diagnostics = bias_diag
        monthly_bias = {}
        era5_isd_bias_annual = np.nan

    elif USE_MONTHLY_MEAN_BIAS_CORRECTION:
        hw_thr_series_corrected, monthly_bias, era5_isd_bias_annual, bias_diag = (
            apply_monthly_mean_bias_correction(
                hw_thr_series_raw=hw_thr_series_raw,
                isd_tmax=u_tmax_all,
                era5_tmax=_era5_tmax_u,
                min_overlap_days_per_month=15,
            )
        )
        bias_correction_method = "monthly_mean_bias_overlap_years_fallback_annual"
        qm_diagnostics = {}

    else:
        hw_thr_series_corrected = hw_thr_series_raw.copy()
        monthly_bias = {}
        era5_isd_bias_annual = np.nan
        bias_diag = {
            "overlap_days": 0,
            "isd_tmax_mean_overlap": np.nan,
            "era5_tmax_mean_overlap": np.nan,
        }
        qm_diagnostics = {}
        bias_correction_method = "none_raw_era5_threshold"

    hw_thr_series = hw_thr_series_corrected


    hw_threshold_raw_mean = (
        float(hw_thr_series_raw.mean())
        if hw_thr_series_raw.notna().any() else np.nan
    )
    hw_threshold_corrected_mean = (
        float(hw_thr_series_corrected.mean())
        if hw_thr_series_corrected.notna().any() else np.nan
    )

    hw_threshold_qm_mean = (
        hw_threshold_corrected_mean
        if USE_QM_BIAS_CORRECTION else np.nan
    )


    hw_thr_mean = hw_threshold_corrected_mean
    n_valid_ref_days = float(valid_count_series.mean()) if len(valid_count_series) else np.nan

    if pd.isna(hw_thr_mean):
        error_info = {
            "pair_id": pair_id,
            "fail_step": "compute_hw_threshold_from_tmax",
            "missing_data": "城市Tmax数据过少，无法计算热浪阈值(hw_thr_mean为NaN)"
        }
        return records_out, threshold_out, station_year_out, hw_daily_out, error_info

    exceed_days_raw = int(
        (u_tmax_all > pd.Series(hw_thr_series_raw).reindex(u_tmax_all.index))
        .fillna(False)
        .sum()
    )
    exceed_days_corrected = int(
        (u_tmax_all > pd.Series(hw_thr_series_corrected).reindex(u_tmax_all.index))
        .fillna(False)
        .sum()
    )

    exceed_days_qm = (
        exceed_days_corrected
        if USE_QM_BIAS_CORRECTION else np.nan
    )

    # ── Pair-specific warm-season common dates ─────────────────────
    if has_warm:
        warm_dates_u = set(combined_u_warm["local_date"].unique())
        warm_dates_r = set(combined_r_warm["local_date"].unique())
        warm_common = warm_dates_u & warm_dates_r
    else:
        warm_common = set()

    # Diagnostic: full-year continuous-event detection.
    # Kept for annual metadata only.
    hw_mask_all = detect_heatwave(u_tmax_all, hw_thr_series_corrected)
    hw_dates_all = set(hw_mask_all[hw_mask_all].index.tolist())
    nhw_dates_all = set(hw_mask_all[~hw_mask_all].index.tolist())

    # Main unified warm-season heatwave detection.
    # This is the flag downstream scripts should use.
    # SH boundary rule:
    #   Jan-Feb of the first analysis year are retained;
    #   Dec of the final analysis year is excluded because the next Jan-Feb
    #   are outside YEARS.
    # Consecutive-day detection is grouped by warm_season_year.
    if has_warm and len(warm_common) > 0:
        warm_index = pd.Index(sorted(warm_common))
        u_tmax_warm_common = u_tmax_all.reindex(warm_index).dropna()
        hw_thr_warm_common = pd.Series(hw_thr_series_corrected).reindex(
            u_tmax_warm_common.index
        )

        hw_mask_warm_pct = detect_heatwave_warm_season_by_year(
            u_tmax_warm_common,
            hw_thr_warm_common,
            lat_u
        )
        hw_dates_warm_pct = set(hw_mask_warm_pct[hw_mask_warm_pct].index.tolist())
        nhw_dates_warm_pct = set(hw_mask_warm_pct[~hw_mask_warm_pct].index.tolist())
    else:
        hw_mask_warm_pct = pd.Series(dtype=bool)
        hw_dates_warm_pct = set()
        nhw_dates_warm_pct = set()

    avg_lon = lon_u
    avg_lat = lat_u
    continent = _get_continent(avg_lon, avg_lat)

    hw_intensity = float(
        (
            u_tmax_all[hw_mask_all]
            - pd.Series(hw_thr_series_corrected).reindex(u_tmax_all.index)[hw_mask_all]
        ).mean()
    ) if hw_mask_all.any() else np.nan


    # ── v5：提取 KG 分区信息 ────────────────────────────────────
    try:
        kg_codes = extract_koppen_codes([avg_lon], [avg_lat])
        _kg_code = kg_codes[0] if kg_codes else np.nan
    except Exception:
        _kg_code = np.nan

    _kg_group       = kg_group(_kg_code) if isinstance(_kg_code, str) else ""
    _climate_zone   = KG_GROUP_MAP.get(_kg_group, "") if _kg_group else ""


    # ── Daily HW flags for downstream scripts ─────────────────────
    hw_daily_out.extend(
        build_daily_heatwave_export_rows(
            pair_id=pair_id,
            usaf_u=usaf_u, wban_u=wban_u,
            usaf_r=usaf_r, wban_r=wban_r,
            lon_u=lon_u, lat_u=lat_u,
            lon_r=lon_r, lat_r=lat_r,
            u_tmax_all=u_tmax_all,
            r_tmax_all=r_tmax_all,
            hw_thr_series_raw=hw_thr_series_raw,
            hw_thr_series_corrected=hw_thr_series_corrected,
            hw_mask_all=hw_mask_all,
            hw_mask_warm=hw_mask_warm_pct,
            warm_common=warm_common,
            continent=continent,
            kg_code=_kg_code,
            kg_group_val=_kg_group,
            climate_zone_main=_climate_zone,
            bias_correction_method=bias_correction_method,
            qm_diagnostics=qm_diagnostics,
        )
    )
    # ── v5：station_threshold_summary 附加 hw_ref_mode ──────────
    threshold_out.append({
        "pair_id":              pair_id,
        "urban_usaf":           usaf_u,
        "urban_wban":           wban_u,
        "urban_lon":            lon_u,
        "urban_lat":            lat_u,
        "hw_threshold_mean":              hw_thr_mean,
        "hw_threshold_raw_mean":          hw_threshold_raw_mean,
        "hw_threshold_corrected_mean":    hw_threshold_corrected_mean,
        "era5_isd_bias_annual":           era5_isd_bias_annual,
        "bias_correction_method": bias_correction_method,

        "qm_ref_mode": qm_diagnostics.get("qm_ref_mode", ""),
        "qm_n_quantiles": qm_diagnostics.get("qm_n_quantiles", np.nan),
        "qm_monthly_sample_counts": ";".join(
            [f"{m}:{qm_diagnostics.get('monthly_sample_counts', {}).get(m, 0)}"
            for m in range(1, 13)]
        ),
        "qm_monthly_methods": ";".join(
            [f"{m}:{qm_diagnostics.get('monthly_methods', {}).get(m, '')}"
            for m in range(1, 13)]
        ),

        "era5_isd_monthly_bias": ";".join(
            [f"{m}:{monthly_bias.get(m, np.nan):.3f}" for m in range(1, 13)]
        ) if monthly_bias else "",

        "isd_tmax_mean_overlap": bias_diag.get(
            "isd_tmax_mean_overlap",
            bias_diag.get("isd_tmax_mean", np.nan)
        ),
        "era5_tmax_mean_overlap": bias_diag.get(
            "era5_tmax_mean_overlap",
            bias_diag.get("era5_tmax_mean", np.nan)
        ),
        "era5_isd_overlap_days": bias_diag.get("overlap_days", 0),

        "exceed_days_raw": exceed_days_raw,
        "exceed_days_corrected": exceed_days_corrected,
        "exceed_days_qm": exceed_days_qm,

        "hw_threshold_qm_mean": hw_threshold_qm_mean,

        "final_hw_days":                  len(hw_dates_all),
        "final_nhw_days":                 len(nhw_dates_all),

        "hw_percentile":        HW_PERCENTILE,
        "hw_ref_mode":          HW_REF_MODE,       # v5 新增
        "n_valid_ref_days":     n_valid_ref_days,
        "n_hw_days_annual":     len(hw_dates_all),
        "n_nhw_days_annual":    len(nhw_dates_all),
        "n_hw_days_warm":       len(hw_dates_warm_pct),
        "n_nhw_days_warm":      len(nhw_dates_warm_pct),
        "kg_code":              _kg_code,           # v5 新增
        "kg_group":             _kg_group,          # v5 新增
        "continent":            continent,
        "valid_urban_years_all":  ",".join(map(str, valid_yrs_u_all)),
        "valid_rural_years_all":  ",".join(map(str, valid_yrs_r_all)),
        "valid_urban_years_warm": ",".join(map(str, valid_yrs_u_warm)) if has_warm else "",
        "valid_rural_years_warm": ",".join(map(str, valid_yrs_r_warm)) if has_warm else "",
    })

    pair_fallback_df = load_pair_fallback()
    pair_match = pair_fallback_df[pair_fallback_df["pair_id"] == pair_id]
    pair_row   = pair_match.iloc[0] if len(pair_match) > 0 else None

    station_meta_vals = {}
    for var in STATION_VARS:
        u_val = get_station_var(usaf_u, wban_u, var)
        r_val = get_station_var(usaf_r, wban_r, var)
        if pair_row is not None and var in PAIR_FALLBACK_MAP:
            u_col, r_col = PAIR_FALLBACK_MAP[var]
            if pd.isna(u_val) and u_col in pair_row.index:
                try: u_val = float(pair_row[u_col]) if pd.notna(pair_row[u_col]) else np.nan
                except: pass
            if pd.isna(r_val) and r_col in pair_row.index:
                try: r_val = float(pair_row[r_col]) if pd.notna(pair_row[r_col]) else np.nan
                except: pass
        station_meta_vals[f"urban_{var}"] = u_val
        station_meta_vals[f"rural_{var}"] = r_val
        station_meta_vals[f"delta_{var}"] = (
            u_val - r_val if pd.notna(u_val) and pd.notna(r_val) else np.nan
        )

    bv_u     = get_bv(usaf_u, wban_u)
    bv_r     = get_bv(usaf_r, wban_r)
    bv_delta = bv_u - bv_r if (not np.isnan(bv_u) and not np.isnan(bv_r)) else np.nan

    common_kwargs = dict(
        pair_id=pair_id, row=row, group=group, continent=continent,
        combined_u_all=combined_u_all, combined_r_all=combined_r_all,
        combined_u_warm=combined_u_warm, combined_r_warm=combined_r_warm,
        has_warm=has_warm, hw_thr_mean=hw_thr_mean,
        hw_dates_all=hw_dates_all, nhw_dates_all=nhw_dates_all,
        hw_intensity=hw_intensity, delta_tmax_ann=delta_tmax_ann,
        valid_yrs_u_all=valid_yrs_u_all, valid_yrs_r_all=valid_yrs_r_all,
        valid_yrs_u_warm=valid_yrs_u_warm if has_warm else [],
        valid_yrs_r_warm=valid_yrs_r_warm if has_warm else [],
        bv_u=bv_u, bv_r=bv_r, bv_delta=bv_delta,
        station_meta_vals=station_meta_vals,
        # v5 新增：KG 字段传入
        kg_code=str(_kg_code) if isinstance(_kg_code, str) else "",
        kg_group_val=_kg_group,
        climate_zone_main=_climate_zone,
    )

    # ── 方法 A：百分位阈值 ──────────────────────────────────────
    recs_pct = build_period_records(
        **common_kwargs,
        hw_dates_warm=hw_dates_warm_pct,
        nhw_dates_warm=nhw_dates_warm_pct,
        hw_method="percentile",
        abs_threshold=np.nan,
    )
    if recs_pct:
        records_out.extend(recs_pct)

    # ── 方法 B：绝对阈值 ────────────────────────────────────────
    abs_thr = get_absolute_hw_threshold(continent)
    hw_mask_abs_all   = detect_heatwave_absolute(u_tmax_all, abs_thr)
    hw_dates_abs_all  = set(hw_mask_abs_all[hw_mask_abs_all].index.tolist())
    nhw_dates_abs_all = set(hw_mask_abs_all[~hw_mask_abs_all].index.tolist())

    if has_warm and len(warm_common) > 0:
        warm_index = pd.Index(sorted(warm_common))
        u_tmax_warm_common_abs = u_tmax_all.reindex(warm_index).dropna()
        hw_mask_warm_abs = detect_heatwave_absolute_warm_season_by_year(
            u_tmax_warm_common_abs,
            abs_thr,
            lat_u
        )

        hw_dates_warm_abs = set(
            hw_mask_warm_abs[hw_mask_warm_abs].index.tolist()
        )
        nhw_dates_warm_abs = set(
            hw_mask_warm_abs[~hw_mask_warm_abs].index.tolist()
        )
    else:
        hw_dates_warm_abs = nhw_dates_warm_abs = set()

    hw_intensity_abs = float(
        (u_tmax_all[hw_mask_abs_all] - abs_thr).mean()
    ) if hw_mask_abs_all.any() else np.nan

    recs_abs = build_period_records(
        **common_kwargs,
        hw_dates_warm=hw_dates_warm_abs,
        nhw_dates_warm=nhw_dates_warm_abs,
        hw_method="absolute",
        abs_threshold=abs_thr,
    )
    for rec in recs_abs:
        rec["hw_intensity"]      = hw_intensity_abs
        rec["n_hw_days_warm"]    = len(hw_dates_warm_abs)
        rec["n_nhw_days_warm"]   = len(nhw_dates_warm_abs)
        rec["n_hw_days_annual"]  = len(hw_dates_abs_all)
        rec["n_nhw_days_annual"] = len(nhw_dates_abs_all)

    if recs_abs:
        records_out.extend(recs_abs)

    if not records_out:
        error_info = {
            "pair_id": pair_id,
            "fail_step": "build_period_records / compute_fft",
            "missing_data": "插值后仍存在缺失，FFT计算失败或未生成有效时段记录"
        }

    return records_out, threshold_out, station_year_out, hw_daily_out, error_info


# ─────────────────────────────────────────────────────────────
# 12. Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print(f" Multi-Year UHI/UCI Diurnal Analysis  v5(+KG full integration)  "
          f"YEARS={YEARS[0]}-{YEARS[-1]}")
    print(f" HW threshold : full-year Tmax  P{HW_PERCENTILE} ±{HW_WINDOW_HALF}-day window"
          f"  [{HW_REF_MODE}]")
    print(f" ERA5 climatology : {'ENABLED' if USE_ERA5_CLIMATOLOGY else 'DISABLED (ISD fallback)'}"
          f"  [USE_ERA5_CLIMATOLOGY={USE_ERA5_CLIMATOLOGY}]")
    print(f" Annual FFT   : full-year data  (valid ≥{FULL_YEAR_MIN_VALID_FRAC:.0%}/yr)")

    print(
        f" Warm-season  : NH=JJA / SH=DJF full data "
        f"(valid ≥{MIN_YEAR_VALID_FRAC:.0%} of hemisphere-specific warm season/yr)"
    )
    print(
        f" HW/NHW FFT   : hemisphere-specific warm season "
        f"(NH=JJA, SH=DJF; valid ≥{MIN_YEAR_VALID_FRAC:.0%}/yr)"
    )

    print(f" Robustness   : percentile + absolute threshold (by continent)")
    print(f" Periods      : annual / warm_season / heatwave / non_heatwave")
    print(f" Climate zone : Köppen-Geiger (主分组) + lat_group (兼容性保留字段)")
    print("=" * 78)

    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "plots"))

    print(f"\nReading pair table: {PAIR_CSV_PATH}")
    pair_df = pd.read_csv(PAIR_CSV_PATH)
    pair_df["urban_lcz_corrected"] = pair_df["urban_lcz"].apply(normalize_lcz)
    pair_df["urban_lcz_class"]     = pair_df["urban_lcz_corrected"].apply(
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

    all_records       = []
    threshold_rows    = []
    station_year_rows = []
    daily_hw_rows     = []
    error_records     = []
    processed = skipped = 0

    tasks   = [(idx, row) for idx, row in pair_df.iterrows()]
    n_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"\n🚀 Starting Multi-Processing with {n_cores} CPU cores...\n")

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(process_single_pair, args): args for args in tasks}

        for future in as_completed(futures):
            try:
                res = future.result()

                if len(res) == 5:
                    rec_out, thr_out, sy_out, hw_day_out, err_out = res
                elif len(res) == 4:
                    rec_out, thr_out, sy_out, err_out = res
                    hw_day_out = []
                else:
                    raise ValueError(f"Unexpected process_single_pair return length: {len(res)}")

                if rec_out:
                    all_records.extend(rec_out)
                    processed += 1
                else:
                    skipped += 1

                if err_out is not None:
                    error_records.append(err_out)

                threshold_rows.extend(thr_out)
                station_year_rows.extend(sy_out)
                daily_hw_rows.extend(hw_day_out)

            except Exception as exc:
                print(f"Error processing pair: {exc}")
                skipped += 1
                failed_args = futures[future]
                error_records.append({
                    "pair_id": failed_args[1]["pair_id"],
                    "fail_step": "Process Exception",
                    "missing_data": f"进程崩溃: {exc}"
                })

    # ── 汇总输出 ─────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"Done: processed={processed}, skipped={skipped}")

    if error_records:
        error_df       = pd.DataFrame(error_records)
        error_out_path = os.path.join(OUTPUT_DIR, "skipped_pairs_log.csv")
        error_df.to_csv(error_out_path, index=False)
        print(f"  Saved: skipped_pairs_log.csv  (total dropped pairs={len(error_df)})")
        print("\nSkipped Reasons Summary:")
        print(error_df["fail_step"].value_counts().to_string())

    if not all_records:
        print("No valid records generated.")
        return

    all_df  = pd.DataFrame(all_records)
    all_out = os.path.join(OUTPUT_DIR, "all_pair_period_metrics.csv")
    all_df.to_csv(all_out, index=False)
    print(f"  Saved: {all_out}  (total rows={len(all_df)})")

    # ════════════════════════════════════════════════════════════════════
    # [新增] 独立提取并保存每个站点的 FFT 参数 (Tmean, 第1/2谐波的 Amp 和 Phase)
    # ════════════════════════════════════════════════════════════════════
    fft_cols = [
        "pair_id", "period", "hw_method", "data_source", "group",
        "urban_Tmean", "rural_Tmean", "dTmean",
        "urban_Amp1", "rural_Amp1", "urban_Phase1", "rural_Phase1",
        "urban_Amp2", "rural_Amp2", "urban_Phase2", "rural_Phase2"
    ]
    # 确保提取的列在 all_df 中实际存在，防止 KeyError
    avail_fft_cols = [c for c in fft_cols if c in all_df.columns]
    
    if avail_fft_cols:
        fft_df = all_df[avail_fft_cols].copy()
        fft_out_path = os.path.join(OUTPUT_DIR, "station_fft_parameters.csv")
        fft_df.to_csv(fft_out_path, index=False)
        print(f"  Saved: {fft_out_path}  (独立 FFT 参数文件)")
    # ════════════════════════════════════════════════════════════════════

    dew_cols  = [c for c in all_df.columns if "_dew_h" in c]
    dew_cover = all_df[dew_cols].notna().mean().mean() if dew_cols else 0.0
    print(f"  Dewpoint columns: {len(dew_cols)}  coverage: {dew_cover:.1%}")

    # ── 版本信息核验 ─────────────────────────────────────────────
    for vfield in ["hne_ref_mode", "hw_ref_mode"]:
        if vfield in all_df.columns:
            print(f"  {vfield}: {all_df[vfield].unique().tolist()}")

    # ── v5：KG 分布统计 ──────────────────────────────────────────
    annual_all = all_df[
        (all_df["period"] == "annual") & (all_df["hw_method"] == "percentile")
    ]
    if "kg_group" in all_df.columns and not annual_all.empty:
        print("\nKG group distribution (annual × percentile):")
        kg_dist = annual_all["kg_group"].value_counts(dropna=False)
        for grp, cnt in kg_dist.items():
            label = KG_GROUP_MAP.get(str(grp), grp)
            print(f"  {grp} ({label}): {cnt}")

    # ── 按 hw_method 分别保存到子目录 ────────────────────────────
    for method in ["percentile", "absolute"]:
        sub_m = all_df[all_df["hw_method"] == method].copy()
        if sub_m.empty:
            print(f"  [robustness/{method}] no records, skipped")
            continue

        method_dir = os.path.join(OUTPUT_DIR, f"robustness_{method}")
        ensure_dir(method_dir)

        sub_m.to_csv(
            os.path.join(method_dir, "all_pair_period_metrics.csv"), index=False
        )
        for g in ["UHI", "UCI"]:
            sg = sub_m[sub_m["group"] == g]
            if len(sg):
                sg.to_csv(
                    os.path.join(method_dir, f"{g}_group_fft_results.csv"),
                    index=False
                )

        save_hw_pressure_test(sub_m, method_dir)
        save_nighttime_risk_outputs(sub_m, method_dir)
        save_bv_quartile_analysis(sub_m, method_dir)
        save_latgroup_robustness(sub_m, method_dir)
        save_kggroup_robustness(sub_m, method_dir)      # v5 新增
        save_mechanism_regressions(sub_m, method_dir)
        save_mediation_analysis(sub_m, method_dir)

        n_annual      = (sub_m["period"] == "annual").sum()
        n_warm_season = (sub_m["period"] == "warm_season").sum()
        n_heatwave    = (sub_m["period"] == "heatwave").sum()
        print(
            f"  [robustness/{method}] "
            f"n_annual={n_annual}  n_warm_season={n_warm_season}  "
            f"n_heatwave={n_heatwave}  "
            f"hw_abs_thr_mean={sub_m['hw_abs_threshold'].mean():.1f}"
        )

    # ── 合并对比文件（v5：含 kg_code / kg_group）────────────────
    compare_cols = [
        "pair_id", "group", "continent", "lat_group",
        "kg_code", "kg_group", "climate_zone_main",   # v5 新增
        "hw_method", "hw_abs_threshold",
        "hne_ref_mode", "hw_ref_mode",                # v5 新增
        "dTmean", "dAmp1", "dTx", "dTn",
        "dT_day_mean", "dT_night_mean",
        "delta_tropical_night_freq", "delta_hotnight_excess",
        "n_hw_days_warm", "n_nhw_days_warm",
    ]
    annual_both = all_df[all_df["period"] == "annual"][
        [c for c in compare_cols if c in all_df.columns]
    ]
    robustness_compare_path = os.path.join(OUTPUT_DIR, "robustness_comparison.csv")
    annual_both.to_csv(robustness_compare_path, index=False)
    print(f"  Saved: robustness_comparison.csv  ({len(annual_both)} rows)")

    # ── 阈值诊断表（v5：含 hw_ref_mode / kg 字段，已在子进程写入）
    if threshold_rows:
        pd.DataFrame(threshold_rows).to_csv(
            os.path.join(OUTPUT_DIR, "station_threshold_summary.csv"), index=False
        )
        print("  Saved: station_threshold_summary.csv  "
              "[含 hw_ref_mode / kg_code / kg_group]")

    # ── Daily HW flags for downstream scripts ─────────────────────
    if daily_hw_rows:
        daily_hw_df = pd.DataFrame(daily_hw_rows)

        daily_hw_out_path = os.path.join(OUTPUT_DIR, "daily_heatwave_flags.csv")
        daily_hw_df.to_csv(daily_hw_out_path, index=False)

        # Also save a percentile-only copy inside robustness_percentile
        # because downstream scripts usually point to robustness_percentile.
        pct_dir = os.path.join(OUTPUT_DIR, "robustness_percentile")
        ensure_dir(pct_dir)
        daily_hw_df.to_csv(
            os.path.join(pct_dir, "daily_heatwave_flags.csv"),
            index=False
        )

        print(
            "  Saved: daily_heatwave_flags.csv  "
            "[downstream unified HW flags; main flag = hw_flag_percentile_warm_season]"
        )
    # ── 站点有效年份汇总 ──────────────────────────────────────────
    if station_year_rows:
        station_yr_df = pd.DataFrame(station_year_rows)
        station_yr_df = station_yr_df.drop_duplicates(
            subset=["pair_id", "station_type"]
        )
        station_yr_path = os.path.join(OUTPUT_DIR, "station_valid_years.csv")
        station_yr_df.to_csv(station_yr_path, index=False)
        print(f"  Saved: station_valid_years.csv  ({len(station_yr_df)} rows)")

        print(
            f"\nStation valid-year coverage summary "
            f"(out of {len(YEARS)} years, {YEARS[0]}-{YEARS[-1]}):"
        )
        for stype in ["urban", "rural"]:
            sub = station_yr_df[station_yr_df["station_type"] == stype]
            if sub.empty:
                continue
            frac = sub["valid_frac_all"]
            print(
                f"  {stype:6s}  n={len(sub):>5d}  "
                f"mean_coverage={frac.mean():.1%}  "
                f"≥80%: {(frac >= 0.8).sum()}  "
                f"100%: {(frac == 1.0).sum()}"
            )

    # ── 数据来源核验 ──────────────────────────────────────────────
    print("\nData source verification:")
    for period in ["annual", "warm_season", "heatwave", "non_heatwave"]:
        for method in ["percentile", "absolute"]:
            sub = all_df[
                (all_df["period"] == period) & (all_df["hw_method"] == method)
            ]
            src = sub["data_source"].unique().tolist() if "data_source" in sub.columns else ["?"]
            print(f"  {period:15s} [{method:10s}]: n={len(sub):>5d}, source={src}")

    # ── 关键变量可用性检查 ────────────────────────────────────────
    print("\nChecking key variable availability ...")
    check_cols = [
        "dAmp1", "dTx", "dTn", "log10_BV",
        "urban_Rn", "urban_NDVI",
        "urban_dew_h12", "rural_dew_h12",
        "hw_method", "hw_abs_threshold",
         "hemisphere", "warm_season_label", "warm_season_months",
        "kg_code", "kg_group", "climate_zone_main",   # v5 新增检查
        "hne_ref_mode", "hw_ref_mode",                 # v5 新增检查
    ]
    for c in check_cols:
        if c in all_df.columns:
            print(f"  {c}: {all_df[c].notna().sum()}/{len(all_df)} non-missing")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(
        "Output structure (v5 + ERA5-MOD):\n"
        f"  {OUTPUT_DIR}/\n"
        "  ├── all_pair_period_metrics.csv        ← 全量 + KG字段 + 版本字段\n"
        "  ├── robustness_comparison.csv          ← 含 kg_code/kg_group/hne_ref_mode/hw_ref_mode\n"
        "  ├── station_threshold_summary.csv      ← 含 hw_ref_mode/kg_code/kg_group\n"
        "  ├── station_valid_years.csv\n"
        "  ├── skipped_pairs_log.csv\n"
        "  ├── robustness_percentile/\n"
        "  │   ├── all_pair_period_metrics.csv\n"
        "  │   ├── UHI_group_fft_results.csv\n"
        "  │   ├── UCI_group_fft_results.csv\n"
        "  │   ├── kggroup_robustness_summary.csv  ← v5 新增（主分组 A-E + 细码）\n"
        "  │   ├── latgroup_robustness_summary.csv ← 兼容性保留，主分组已改用KG\n"
        "  │   ├── mechanism_regression_table.csv  ← 含 annual_kg_* 分层\n"
        "  │   ├── mediation_BV_dAmp1_to_dTx_dTn.csv  ← 含 annual_kg_* 分层\n"
        "  │   └── ...\n"
        "  └── robustness_absolute/\n"
        "      └── （同上结构）\n"
        "\n注：Köppen-Geiger (KG) 已作为主气候分区依据，lat_group 仅保留作兼容性字段。\n"
        "    hne_ref_mode / hw_ref_mode 字段内嵌于每行记录，便于多版本数据溯源。\n"
        "    USE_ERA5_CLIMATOLOGY=True 时热浪阈值优先使用 ERA5 长气候态；\n"
        "    ERA5 文件缺失时自动回退至 ISD 短期 P90 估计。"
    )


if __name__ == "__main__":
    main()


############################################################
