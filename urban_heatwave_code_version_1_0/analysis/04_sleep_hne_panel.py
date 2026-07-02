#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_hne_panel.py  (v11-econ — HNE + 睡眠损失 + 经济损失估算)
==================================================================================
在 v10-econ 基础上新增：

  [ECON-1] 工资弹性法（主要方法，sleep loss 渠道）
           来源：Gibson & Shrader (2018), Rev. Econ. Stat.
                 Costa-Font, Fleche & Pagan (2024), J. Health Econ.
           公式：ΔProductivity% = (β_wage / 60) × |ΔSleep_min|
           参数：短期 β = 1.1%/h，长期 β = 5.0%/h，小时弹性 β = 4.2%/h
           输出：econ_loss_pct_short / _long / _hourly

  [ECON-2] 出勤效率法（辅助，sleep loss 渠道）
           来源：Hafner et al. (2016), RAND Health Q.
           输出：econ_loss_pct_present

  [ECON-3] USD 转换（sleep loss → USD）
           输出：econ_loss_usd_sleep

  [ECON-4] 劳动力加总
           公式：Total_loss_USD = mean(loss_usd) × n_workers × working_days

  [ECON-5] ★ 直接高温劳动损失（新增）
           来源：Graff Zivin & Neidell (2014), JPE 122(5):1391-1432
                 Seppanen et al. (2006)（室内劳动者综合）
           公式：heat_loss_pct = α × max(Tmax - Tref, 0)
                 α=0.5 %/°C（全劳动力加权），Tref=29°C
           输出：heat_loss_pct, econ_loss_usd_heat

  [ECON-6] ★ 合并总损失（新增）
           公式：total_loss_pct = econ_loss_pct_short + heat_loss_pct
                 total_loss_usd = econ_loss_usd_sleep + econ_loss_usd_heat
           输出：total_loss_pct, total_loss_usd

  [ERA5]   ERA5 1991-2020 气候态阈值（HW + HNE 双渠道替换 ISD 短期估计）
           USE_ERA5_CLIMATOLOGY = True  ← 本文件默认开启

新增输出列（逐日面板和配对差值均含）：
  ─ sleep loss 渠道 ─────────────────────────────────────────────────
  sleep_loss_min           夜间睡眠损失（min/night，负值=减少）
  econ_loss_pct_short      短期工资弹性生产力损失（%）
  econ_loss_pct_long       长期工资弹性生产力损失（%）
  econ_loss_pct_hourly     小时工资弹性生产力损失（%）
  econ_loss_pct_present    出勤效率法生产力损失（%）
  econ_loss_usd_sleep      sleep渠道 USD/人/工作日

  ─ 直接高温渠道 ────────────────────────────────────────────────────
  heat_loss_pct            直接高温生产力损失（%）
  econ_loss_usd_heat       直接高温 USD/人/工作日

  ─ 合并总损失 ──────────────────────────────────────────────────────
  total_loss_pct           合并生产力损失（%）
  total_loss_usd           合并 USD/人/工作日

新增经济损失输出文件（economic_loss/ 目录）：
  total_annual_loss_sleep_loss_{pooled|loyo}.csv
  total_annual_loss_direct_heat_{pooled|loyo}.csv
  total_annual_loss_total_{pooled|loyo}.csv
  total_annual_loss_ALL_{pooled|loyo}.csv    ← 三类拼合宽表（论文对比用）
==================================================================================
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from config import (
    ERA5_STATION_DIR,
    ISD_BASE_DIR,
    KG_TIF,
    PAIR_CSV_PATH,
    STATION_FEATURES_CSV,
    UNIFIED_ROOT,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# KG climate zone utilities  (Beck et al. 2023)
# ══════════════════════════════════════════════════════════════
import rasterio


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


# ══════════════════════════════════════════════════════════════
# §0. 全局配置
# ══════════════════════════════════════════════════════════════
# File-system paths are centralized in config.py.
OUTPUT_DIR      = UNIFIED_ROOT + "/analysis/hne_econ"
CANONICAL_GROUP_CSV = (
    UNIFIED_ROOT + "/analysis/main_multiyear/"
    "robustness_percentile/all_pair_period_metrics.csv"
)


# ══════════════════════════════════════════════════════════════
# Figure 4 data export path
# ══════════════════════════════════════════════════════════════
FIG4_DATA_DIR = UNIFIED_ROOT + "/shared/fig4_data"

# Figure 4 exposure settings
# 主图使用 exposure-based metrics，而不是直接画 ΔAmp → sleep。
FIG4_PRIMARY_HNE_METHOD = "pooled"

# Fixed local-time windows for figure-level exposure.
# 这些用于统一 Figure 4 的可解释 exposure 指标；
# 原脚本的 solar-night HNE 也会同时导出为 night_hne_exposure。
FIG4_NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]   # 20:00–07:59, 12 h
FIG4_DAY_HOURS = list(range(8, 20))                              # 08:00–19:59, 12 h

FIG4_THETA_NIGHT_TEMP = 26.0   # °C, nighttime heat exposure threshold
FIG4_THETA_DAY_TEMP = 30.0     # °C, daytime heat exposure threshold

# ══════════════════════════════════════════════════════════════
# Figure 4 data export path
# ══════════════════════════════════════════════════════════════
MIN_OBS_SUPPORT_HOURS = 6   # 兼容 3-hourly
MIN_USABLE_HOURS      = 18  # 最终 hourly 可用小时
MAX_INTERP_GAP_HOURS  = 2   # 允许最多补连续 2 小时python

YEARS           = list(range(2015, 2025))

# Hemisphere-aware warm season, synced with analysis_multiyear.py:
#   NH: JJA = Jun-Jul-Aug
#   SH: DJF = Dec-Jan-Feb
NH_WARM_MONTHS = [6, 7, 8]
SH_WARM_MONTHS = [12, 1, 2]
ANALYSIS_MONTHS = NH_WARM_MONTHS  # backward-compatible default only

# analysis_multiyear.py daily HW flags; used first when available.
ANALYSIS_HW_FLAGS_CSV = (
    UNIFIED_ROOT
    + "/analysis/main_multiyear/robustness_percentile/daily_heatwave_flags.csv"
)

MAX_CONSEC_NAN        = 2
MAX_INTERP_GAP        = 1
MIN_VALID_FRAC_ANNUAL = 0.80
MIN_VALID_FRAC_WARM   = 0.80
WARM_SEASON_DAYS_EXPECTED = 92
MIN_LOYO_REF_YEARS    = 3

# ── HNE 参数 ──────────────────────────────────────────────────
HNE_PERCENTILE_TMIN = 95
HNE_THR_SEASON      = "warm_season"

# ── HW 参数 ──────────────────────────────────────────────────
HW_PERCENTILE_TMAX = 90
HW_WINDOW_HALF     = 7
HW_MIN_DAYS        = 3

# ── 睡眠损失样条参数（Minor et al. 2022, Table S37）──────────
SLEEP_LOSS_BETA1 = -0.107
SLEEP_LOSS_BETA2 = -0.618
SLEEP_LOSS_KNOT1 = -20.0
SLEEP_LOSS_KNOT2 =  10.0
SLEEP_LOSS_CONST =   0.0


# ══════════════════════════════════════════════════════════════
# §0b. 经济损失模型参数
# ══════════════════════════════════════════════════════════════

# ── [ECON-1] 工资弹性法 ───────────────────────────────────────
ECON_WAGE_ELAST_SHORT  = 1.1   # %/hour（短期，Gibson & Shrader 2018）
ECON_WAGE_ELAST_LONG   = 5.0   # %/hour（长期，Gibson & Shrader 2018）
ECON_HOURLY_WAGE_ELAST = 4.2   # %/hour（小时工资弹性，Costa-Font 2024）

# ── [ECON-2] 出勤效率法 ──────────────────────────────────────
ECON_PRESENTEEISM_BASE_HOURS    = 7.5
ECON_PRESENTEEISM_SHORT_PENALTY = 2.4
ECON_PRESENTEEISM_MID_PENALTY   = 1.5
ECON_WORKING_DAYS_PER_YEAR      = 250
ECON_WORKDAY_HOURS              = 8.0

# ── [ECON-3] USD 转换参数（全局默认，逐站由 station_features 覆盖）
ECON_WAGE_USD_DEFAULT = 0.0

# ── [ECON-5] ★ 直接高温劳动损失参数（Graff Zivin & Neidell 2014）
# 公式：heat_loss_pct = HEAT_ALPHA × max(Tmax - HEAT_TREF, 0)  [%/workday]
# HEAT_ALPHA 参考值：
#   0.5 %/°C — 全劳动力加权均值（推荐，保守）
#   1.0 %/°C — 户外/重体力劳动者
#   0.2 %/°C — 室内空调工作者
HEAT_TREF  = 29.0   # °C，高温损失阈值
HEAT_ALPHA = 0.5    # %/°C

# ── [ECON-4] 劳动力数据 ──────────────────────────────────────
# ★ 指向 download_labour_features.py 输出的 05_station_features.csv
LABOUR_FORCE_CSV = STATION_FEATURES_CSV   # 向后兼容

# ══════════════════════════════════════════════════════════════
# ERA5 气候态开关（True = 使用 ERA5 1991-2020 气候态阈值）
# ══════════════════════════════════════════════════════════════

USE_ERA5_CLIMATOLOGY = True

USE_QM_BIAS_CORRECTION = True
USE_MONTHLY_MEAN_BIAS_CORRECTION = False

QM_N_QUANTILES = 1001
QM_MIN_OVERLAP_DAYS_PER_MONTH = 30
QM_MIN_OVERLAP_DAYS_ANNUAL = 100

HW_REFERENCE_MODE = "full_year_Tmax_P90_7day_window_ERA5_QM_corrected_v11_econ"

MIN_ERA5_VALID_DAYS = MIN_LOYO_REF_YEARS * 30


# ─────────────────────────────────────────────────────────────
# Heatwave-sync helpers: read analysis_multiyear.py daily HW flags when available
# ─────────────────────────────────────────────────────────────
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


def detect_heatwave_warm_season(daily_tmax, threshold, lat) -> pd.Series:
    """
    Fallback HW detection when analysis_multiyear.py daily flags are unavailable.

    Same HW algorithm as the main script, with the boundary rule:
      - SH Jan-Feb of min(YEARS) are retained.
      - SH Dec of max(YEARS) is excluded.
      - Consecutive-day detection is applied within each warm_season_year.
    """
    s0 = pd.Series(daily_tmax).sort_index()
    out = pd.Series(0, index=pd.Series(daily_tmax).index, name="hw_flag")

    s = s0.dropna()
    if len(s) == 0:
        return out.astype(int)

    warm_idx = [
        d for d in s.index
        if is_warm_season_date_for_analysis(d, lat)
    ]
    if len(warm_idx) == 0:
        return out.astype(int)

    s_warm = s.reindex(pd.Index(sorted(warm_idx))).dropna()
    if len(s_warm) == 0:
        return out.astype(int)

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
        out.loc[hw_part.index] = hw_part.astype(int)

    return out.fillna(0).astype(int)

class ERA5MissingOrInvalid(Exception):
    pass


def era5_station_path(usaf, wban, label):
    return os.path.join(
        ERA5_STATION_DIR,
        f"{str(usaf).strip()}_{str(wban).strip()}_{label}.csv"
    )


def load_era5_daily(usaf, wban, label, require_tmax=True, require_tmin=True):
    """
    ERA5 文件命名：
        {USAF}-{WBAN}_{urban/rural}.csv

    优先读取列：
        tmax_c, tmin_c

    若 ERA5 文件缺失、列名不对、数据不足，直接抛出 ERA5MissingOrInvalid。
    """
    p = era5_station_path(usaf, wban, label)

    if not os.path.exists(p):
        raise ERA5MissingOrInvalid(f"ERA5_missing_or_invalid: file not found: {p}")

    try:
        df = pd.read_csv(p, parse_dates=["date"])
    except Exception as e:
        raise ERA5MissingOrInvalid(f"ERA5_missing_or_invalid: cannot read {p}: {e}")

    if "date" not in df.columns:
        raise ERA5MissingOrInvalid(f"ERA5_missing_or_invalid: missing date column: {p}")

    tmax_col = "tmax_c" if "tmax_c" in df.columns else ("tmax" if "tmax" in df.columns else None)
    tmin_col = "tmin_c" if "tmin_c" in df.columns else ("tmin" if "tmin" in df.columns else None)

    if require_tmax and tmax_col is None:
        raise ERA5MissingOrInvalid(f"ERA5_missing_or_invalid: missing tmax_c/tmax column: {p}")

    if require_tmin and tmin_col is None:
        raise ERA5MissingOrInvalid(f"ERA5_missing_or_invalid: missing tmin_c/tmin column: {p}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], errors="coerce")

    if tmax_col is not None:
        out["tmax_c"] = pd.to_numeric(df[tmax_col], errors="coerce")

    if tmin_col is not None:
        out["tmin_c"] = pd.to_numeric(df[tmin_col], errors="coerce")

    out = out.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    need_cols = []
    if require_tmax:
        need_cols.append("tmax_c")
    if require_tmin:
        need_cols.append("tmin_c")

    valid_n = out.dropna(subset=need_cols).shape[0]
    if valid_n < MIN_ERA5_VALID_DAYS:
        raise ERA5MissingOrInvalid(
            f"ERA5_missing_or_invalid: insufficient ERA5 records: {p}, valid_n={valid_n}"
        )

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day

    return out


def load_era5_tmax_series(usaf, wban, label):
    df = load_era5_daily(usaf, wban, label, require_tmax=True, require_tmin=False)
    s = df.set_index("date")["tmax_c"].dropna()
    s.index = pd.to_datetime(s.index)
    return s


def compute_hw_doy_thr_era5(era5_tmax, q=HW_PERCENTILE_TMAX, w=HW_WINDOW_HALF):
    ref = pd.DataFrame({"date": era5_tmax.index, "tmax": era5_tmax.values})
    ref["date"] = pd.to_datetime(ref["date"])
    ref["doy"] = ref["date"].apply(
        lambda x: pd.Timestamp(2001, x.month, x.day).dayofyear
        if not (x.month == 2 and x.day == 29) else 59
    )
    ref = ref.dropna(subset=["tmax"])

    if len(ref) < MIN_ERA5_VALID_DAYS:
        raise ERA5MissingOrInvalid("ERA5_missing_or_invalid: insufficient tmax reference data")

    doy_thr = {}
    for doy in range(1, 366):
        lo, hi = doy - w, doy + w

        if lo < 1:
            mask = (ref["doy"] >= 365 + lo) | (ref["doy"] <= hi)
        elif hi > 365:
            mask = (ref["doy"] >= lo) | (ref["doy"] <= hi - 365)
        else:
            mask = (ref["doy"] >= lo) & (ref["doy"] <= hi)

        vals = ref.loc[mask, "tmax"].dropna().values

        if len(vals) == 0:
            raise ERA5MissingOrInvalid(f"ERA5_missing_or_invalid: empty DOY window: doy={doy}")

        doy_thr[doy] = float(np.nanpercentile(vals, q))

    return doy_thr


def apply_doy_thr_to_index(date_index, doy_thr_dict):
    thr = {}
    for d in date_index:
        d_ts = pd.Timestamp(d)
        doy = (
            pd.Timestamp(2001, d_ts.month, d_ts.day).dayofyear
            if not (d_ts.month == 2 and d_ts.day == 29) else 59
        )
        thr[d] = doy_thr_dict.get(doy, np.nan)

    s = pd.Series(thr).sort_index()

    if s.isna().all():
        raise ERA5MissingOrInvalid("ERA5_missing_or_invalid: all mapped HW thresholds are NaN")

    return s

def quantile_mapping_values(values, sim_train, obs_train, n_quantiles=1001):
    """
    Empirical Quantile Mapping:
        y = F_obs^{-1}(F_sim(x))
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

    return np.interp(
        values_arr,
        sim_q_unique,
        obs_q_unique,
        left=obs_q_unique[0],
        right=obs_q_unique[-1],
    )


def apply_quantile_mapping_bias_correction(
        hw_thr_series_raw,
        isd_tmax,
        era5_tmax,
        min_overlap_days_per_month=30,
        min_overlap_days_annual=100,
        n_quantiles=1001):
    """
    Quantile Mapping bias correction for ERA5-derived HW thresholds.

    corrected = F_obs^{-1}(F_era5(x))
    """
    thr = pd.Series(hw_thr_series_raw).copy()
    isd = pd.Series(isd_tmax).copy()
    era = pd.Series(era5_tmax).copy()

    thr.index = pd.to_datetime(thr.index)
    isd.index = pd.to_datetime(isd.index)
    era.index = pd.to_datetime(era.index)

    overlap = pd.concat(
        [isd.rename("isd"), era.rename("era5")],
        axis=1,
        join="inner"
    ).dropna()

    diagnostics = {
        "bias_correction_method": "quantile_mapping_empirical_cdf",
        "qm_ref_mode": "monthly_QM_with_annual_fallback",
        "qm_n_quantiles": int(n_quantiles),
        "overlap_days": int(len(overlap)),
        "monthly_sample_counts": {},
        "monthly_methods": {},
        "annual_sample_count": int(len(overlap)),
    }

    corrected = thr.copy()

    if len(overlap) == 0:
        for m in range(1, 13):
            diagnostics["monthly_sample_counts"][m] = 0
            diagnostics["monthly_methods"][m] = "no_correction_no_overlap"
        return corrected, diagnostics

    has_annual_qm = len(overlap) >= min_overlap_days_annual
    thr_month = corrected.index.month

    for m in range(1, 13):
        mask = thr_month == m
        values = corrected.loc[mask].values
        sub = overlap[overlap.index.month == m]

        diagnostics["monthly_sample_counts"][m] = int(len(sub))

        if len(values) == 0:
            diagnostics["monthly_methods"][m] = "no_threshold_values"
            continue

        if len(sub) >= min_overlap_days_per_month:
            corrected_values = quantile_mapping_values(
                values, sub["era5"], sub["isd"], n_quantiles
            )
            diagnostics["monthly_methods"][m] = "monthly_QM"

        elif has_annual_qm:
            corrected_values = quantile_mapping_values(
                values, overlap["era5"], overlap["isd"], n_quantiles
            )
            diagnostics["monthly_methods"][m] = "annual_QM_fallback"

        else:
            corrected_values = values.copy()
            diagnostics["monthly_methods"][m] = "no_correction_insufficient_samples"

        corrected.loc[mask] = corrected_values

    diagnostics["isd_tmax_mean_overlap"] = float(overlap["isd"].mean())
    diagnostics["era5_tmax_mean_overlap"] = float(overlap["era5"].mean())

    return corrected, diagnostics

def compute_hne_thr_era5(usaf, wban, label, months=None, q=HNE_PERCENTILE_TMIN):
    if months is None:
        months = ANALYSIS_MONTHS

    df = load_era5_daily(usaf, wban, label, require_tmax=False, require_tmin=True)
    vals = df[df["month"].isin(months)]["tmin_c"].dropna().values

    if len(vals) < MIN_ERA5_VALID_DAYS:
        raise ERA5MissingOrInvalid(
            f"ERA5_missing_or_invalid: insufficient warm-season tmin: {usaf}-{wban}_{label}"
        )

    return float(np.nanpercentile(vals, q))


def build_hne_loyo_thr_era5(
    usaf, wban, label,
    analysis_years=None,
    months=None,
    q=HNE_PERCENTILE_TMIN,
    min_ref_years=MIN_LOYO_REF_YEARS,
):
    if analysis_years is None:
        analysis_years = YEARS
    if months is None:
        months = ANALYSIS_MONTHS

    df = load_era5_daily(usaf, wban, label, require_tmax=False, require_tmin=True)
    df_warm = df[df["month"].isin(months)].copy()

    era5_years = sorted(df_warm["year"].dropna().unique())

    if len(era5_years) < min_ref_years:
        raise ERA5MissingOrInvalid(
            f"ERA5_missing_or_invalid: insufficient ERA5 reference years: {usaf}-{wban}_{label}"
        )

    thr_by_year = {}

    for yr in analysis_years:
        ref = df_warm[df_warm["year"] != yr]["tmin_c"].dropna().values
        ref_years = [y for y in era5_years if y != yr]

        if len(ref_years) < min_ref_years or len(ref) == 0:
            raise ERA5MissingOrInvalid(
                f"ERA5_missing_or_invalid: insufficient LOYO ERA5 tmin: {usaf}-{wban}_{label}, year={yr}"
            )

        thr_by_year[yr] = float(np.nanpercentile(ref, q))

    return thr_by_year


# ══════════════════════════════════════════════════════════════
# §1. 工具函数
# ══════════════════════════════════════════════════════════════
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_pair_id(pair_id):
    parts = pair_id.split("__")
    u = parts[0].strip().split("_")
    r = parts[1].strip().split("_")
    return u[0], u[1], r[0], r[1]

def isd_path(usaf, wban, year, base):
    return os.path.join(base, str(year), f"{usaf}-{wban}-{year}.gz")

def is_leap(yr):
    return (yr % 4 == 0) and (yr % 100 != 0 or yr % 400 == 0)

def lat_group(lat):
    if abs(lat) < 23.5: return "Tropical"
    if abs(lat) < 40:   return "Subtropical"
    if abs(lat) < 60:   return "Temperate"
    return "Polar"

def norm_period_for_integrated(x):
    mp = {
        "annual": "annual",
        "warm_season": "JJA",
        "heatwave": "HW",
        "non_heatwave": "NHW",
        "JJA": "JJA",
        "HW": "HW",
        "NHW": "NHW",
    }
    return mp.get(str(x), str(x))

# ══════════════════════════════════════════════════════════════
# §2. 向量化日出/日落
# ══════════════════════════════════════════════════════════════
def solar_angles_vec(lat_deg, doy_arr):
    """
    计算日出/日落时刻，并显式标记极昼/极夜。

    Parameters
    ----------
    lat_deg : float
        纬度（度）
    doy_arr : array-like
        年积日 day-of-year

    Returns
    -------
    sunrise : np.ndarray
        日出本地太阳时（小时），普通情况有效；极昼/极夜时为 NaN
    sunset : np.ndarray
        日落本地太阳时（小时），普通情况有效；极昼/极夜时为 NaN
    polar_day : np.ndarray(bool)
        极昼标记（全天太阳不落）
    polar_night : np.ndarray(bool)
        极夜标记（全天太阳不升）
    """
    doy = np.asarray(doy_arr, dtype=float)
    decl = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))
    lat = math.radians(lat_deg)

    x = -math.tan(lat) * np.tan(decl)

    sunrise = np.full(len(doy), np.nan, dtype=float)
    sunset = np.full(len(doy), np.nan, dtype=float)
    polar_day = np.zeros(len(doy), dtype=bool)
    polar_night = np.zeros(len(doy), dtype=bool)

    normal_mask = (x > -1.0) & (x < 1.0)
    polar_day_mask = x <= -1.0
    polar_night_mask = x >= 1.0

    if normal_mask.any():
        h_deg = np.degrees(np.arccos(x[normal_mask]))
        sunrise[normal_mask] = 12.0 - h_deg / 15.0
        sunset[normal_mask] = 12.0 + h_deg / 15.0

    polar_day[polar_day_mask] = True
    polar_night[polar_night_mask] = True

    return sunrise, sunset, polar_day, polar_night


def add_is_night_column(df, lat):
    """
    基于太阳几何定义夜间：
      - 普通情况：日落到日出为夜间
      - 极昼：全天非夜间
      - 极夜：全天夜间
    """
    df = df.copy()
    doy = df["local_dt"].dt.dayofyear.values.astype(float)
    sunrise, sunset, polar_day, polar_night = solar_angles_vec(lat, doy)
    h = df["local_hour"].values.astype(float)

    is_night = np.zeros(len(df), dtype=bool)

    normal_mask = ~(polar_day | polar_night)
    is_night[normal_mask] = (
        (h[normal_mask] >= sunset[normal_mask]) |
        (h[normal_mask] < sunrise[normal_mask])
    )

    # 极昼：没有夜间
    is_night[polar_day] = False

    # 极夜：全天都算夜间
    is_night[polar_night] = True

    df["is_night"] = is_night.astype(bool)
    df["polar_day"] = polar_day
    df["polar_night"] = polar_night
    df["sunrise_hour"] = sunrise
    df["sunset_hour"] = sunset
    return df


# ══════════════════════════════════════════════════════════════
# §3. ISD-Lite 读取
# ══════════════════════════════════════════════════════════════
def read_isd_lite(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(
            filepath,
            sep=r'\s+',
            header=None,
            usecols=[0, 1, 2, 3, 4, 5],
            names=['year', 'month', 'day', 'hour', 'temp_C', 'dew_C'],
            na_values={'temp_C': -9999, 'dew_C': -9999},
            engine='c',
            on_bad_lines='skip'
        )
        df['temp_C'] = df['temp_C'] / 10.0
        df['dew_C']  = df['dew_C'] / 10.0
        df['datetime'] = pd.to_datetime(
            df[['year', 'month', 'day', 'hour']], errors='coerce'
        )
        df = df.dropna(subset=['datetime']).drop_duplicates(subset='datetime')
        df = df.sort_values('datetime').reset_index(drop=True)
        if df.empty:
            return None
        return df[['datetime', 'temp_C', 'dew_C']]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# §3b. QC + 插值
# ══════════════════════════════════════════════════════════════
def _qc_interp_vec(s, full_i,
                   max_consec_nan=MAX_CONSEC_NAN,
                   max_interp_gap=MAX_INTERP_GAP):
    s = s[~s.index.duplicated(keep="first")].reindex(full_i)
    if s.notna().all():
        return s.interpolate("linear", limit=max_interp_gap, limit_area="inside")
    is_nan       = s.isna()
    nan_group_id = (~is_nan).cumsum()
    nan_run_len  = (is_nan.groupby(nan_group_id)
                          .transform("sum")
                          .where(is_nan, 0))
    date_col  = pd.Series(full_i.date, index=full_i, name="date")
    daily_max = nan_run_len.groupby(date_col).max()
    bad_dates = set(daily_max[daily_max > max_consec_nan].index)
    if bad_dates:
        bad_mask = date_col.isin(bad_dates).values
        s        = s.copy()
        s.iloc[bad_mask] = np.nan
    return s.interpolate("linear", limit=max_interp_gap, limit_area="inside")


def clean_and_localise(df_raw, year, lon):
    full_1h_i = pd.date_range(
        pd.Timestamp(year, 1, 1, 0),
        pd.Timestamp(year, 12, 31, 23),
        freq="1h"
    )

    df_i = df_raw.set_index("datetime").sort_index()

    # 原始 hourly 观测支持数（不是插值）
    obs_hourly = (
        df_i[["temp_C"]]
        .resample("1h")
        .count()
        .reindex(full_1h_i)
        .fillna(0)
    )

    # 统一到 hourly
    hourly = (
        df_i[["temp_C", "dew_C"]]
        .resample("1h")
        .mean()
        .reindex(full_1h_i)
    )

    # 只补小缺口
    hourly["temp_C"] = hourly["temp_C"].interpolate(
        method="time",
        limit=MAX_INTERP_GAP_HOURS,
        limit_area="inside"
    )
    hourly["dew_C"] = hourly["dew_C"].interpolate(
        method="time",
        limit=MAX_INTERP_GAP_HOURS,
        limit_area="inside"
    )

    # 双门槛：原始观测支持 + 最终可用小时
    obs_support_count = obs_hourly["temp_C"].groupby(obs_hourly.index.date).transform(
        lambda s: (s > 0).sum()
    )
    usable_count = hourly["temp_C"].groupby(hourly.index.date).transform(
        lambda s: s.notna().sum()
    )

    bad_days = (
        (obs_support_count < MIN_OBS_SUPPORT_HOURS) |
        (usable_count < MIN_USABLE_HOURS)
    )

    hourly.loc[bad_days, ["temp_C", "dew_C"]] = np.nan

    offset   = pd.Timedelta(hours=lon / 15.0)
    local_dt = full_1h_i + offset

    out = pd.DataFrame({
        "datetime":   full_1h_i,
        "local_dt":   local_dt,
        "temp_C":     hourly["temp_C"].values,
        "dew_C":      hourly["dew_C"].values,
    })
    out["local_date"]  = out["local_dt"].dt.date
    out["local_hour"]  = out["local_dt"].dt.hour + out["local_dt"].dt.minute / 60.0
    out["local_month"] = out["local_dt"].dt.month
    out["year"]        = out["local_dt"].dt.year
    return out


# ══════════════════════════════════════════════════════════════
# §4. 湿球温度
# ══════════════════════════════════════════════════════════════
def dewpoint_to_rh(T_C, Td_C):
    a, b = 17.625, 243.04
    return np.clip(100 * np.exp(a*Td_C/(b+Td_C) - a*T_C/(b+T_C)), 0, 100)

def wet_bulb(T_C, RH):
    return (T_C * np.arctan(0.151977 * (RH + 8.313659)**0.5)
            + np.arctan(T_C + RH)
            - np.arctan(RH - 1.676331)
            + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
            - 4.686035)


# ══════════════════════════════════════════════════════════════
# §5-A. 睡眠损失模型（Minor et al. 2022, Table S37）
# ══════════════════════════════════════════════════════════════
def compute_sleep_loss_minutes(
    tmin,
    beta1=SLEEP_LOSS_BETA1,
    beta2=SLEEP_LOSS_BETA2,
    knot1=SLEEP_LOSS_KNOT1,
    knot2=SLEEP_LOSS_KNOT2,
    const=SLEEP_LOSS_CONST,
):
    """
    Minor et al. (2022), Table S37 nighttime-temperature spline.

    Parameters
    ----------
    tmin : array-like
        Nighttime minimum air temperature in deg C.

    beta1 : float
        Slope from knot1 (-20 deg C) to knot2 (10 deg C).

    beta2 : float
        Slope above knot2 (10 deg C).

    knot1 : float
        Lower spline knot, default -20 deg C.

    knot2 : float
        Upper spline knot, default 10 deg C.

    const : float
        Model constant, default 0.

    Returns
    -------
    numpy.ndarray
        Change in sleep duration in min/night.

        Negative values indicate reduced sleep duration.
        The mathematical reference value is zero at and below knot1.
        Missing temperature values remain NaN.

    Piecewise definition
    --------------------
    T <= -20:
        DeltaSleep = 0

    -20 < T <= 10:
        DeltaSleep = beta1 * (T + 20)

    T > 10:
        DeltaSleep = beta1 * 30 + beta2 * (T - 10)
    """
    tmin = np.asarray(tmin, dtype=float)

    # Exposure within the first spline segment:
    # starts at -20 deg C and stops increasing at 10 deg C.
    segment1 = np.clip(
        tmin - knot1,
        0.0,
        knot2 - knot1,
    )

    # Additional exposure above 10 deg C.
    segment2 = np.maximum(
        tmin - knot2,
        0.0,
    )

    return (
        const
        + beta1 * segment1
        + beta2 * segment2
    )
# ══════════════════════════════════════════════════════════════
# §5-B. 经济损失模型（sleep loss 渠道）
# ══════════════════════════════════════════════════════════════
def compute_economic_loss(
    sleep_loss_min,
    wage_usd_per_hour = ECON_WAGE_USD_DEFAULT,
    elast_short       = ECON_WAGE_ELAST_SHORT,
    elast_long        = ECON_WAGE_ELAST_LONG,
    elast_hourly      = ECON_HOURLY_WAGE_ELAST,
    ref_hours         = ECON_PRESENTEEISM_BASE_HOURS,
    workday_hours     = ECON_WORKDAY_HOURS,
    working_days_yr   = ECON_WORKING_DAYS_PER_YEAR,
):
    """
    将夜间睡眠损失（分钟/夜）转换为经济生产力损失。

    方法 1：工资弹性法（Gibson & Shrader 2018; Costa-Font 2024）
        ΔProductivity% = (β / 60) × |sleep_loss_min|

    方法 2：出勤效率法（Hafner et al. 2016）
        连续线性近似，日级

    方法 3：USD 转换
        econ_loss_usd_sleep = (econ_loss_pct_short / 100) × wage × workday_hours

    返回 dict：
        pct_short, pct_long, pct_hourly, pct_present, usd
    """
    sl = np.asarray(sleep_loss_min, dtype=float)
    loss_hours = np.maximum(-sl, 0.0) / 60.0

    pct_short  = elast_short  * loss_hours
    pct_long   = elast_long   * loss_hours
    pct_hourly = elast_hourly * loss_hours

    sleep_hrs = ref_hours + sl / 60.0
    annual_pp = np.where(
        sleep_hrs < 6.0, 2.4,
        np.where(sleep_hrs < 7.0, 1.5 * (7.0 - sleep_hrs), 0.0)
    )
    pct_present = annual_pp / working_days_yr * 100.0

    if wage_usd_per_hour > 0:
        usd = (pct_short / 100.0) * wage_usd_per_hour * workday_hours
    else:
        usd = np.full_like(pct_short, np.nan)

    return {
        "pct_short":   pct_short,
        "pct_long":    pct_long,
        "pct_hourly":  pct_hourly,
        "pct_present": pct_present,
        "usd":         usd,
    }


# ══════════════════════════════════════════════════════════════
# §5-C. 劳动力总量经济损失加总（需外部数据）
# ══════════════════════════════════════════════════════════════
def aggregate_total_economic_loss(
    panel_df,
    labour_df,
    econ_col        = "econ_loss_usd",
    join_col        = "country_iso3",
    working_days_yr = ECON_WORKING_DAYS_PER_YEAR,
):
    """
    将逐日逐站点损失（USD/人/日）聚合为全年总量（USD/年）。

    公式：total_annual_loss_USD = mean(econ_loss_usd) × n_workers × working_days_yr

    参数
    ────
    panel_df   : 逐日面板（含 econ_loss_usd 列和 country_iso3 列）
    labour_df  : 含 country_iso3, n_workers 的劳动力元数据
    econ_col   : 损失列名
    join_col   : 连接键

    返回
    ────
    DataFrame：[country_iso3, mean_daily_loss_usd, n_workers,
                total_annual_loss_usd, total_annual_loss_bn_usd]
    """
    if join_col not in panel_df.columns:
        raise ValueError(f"panel_df 缺少列 '{join_col}'。")
    if econ_col not in panel_df.columns or panel_df[econ_col].isna().all():
        raise ValueError(f"panel_df 的 '{econ_col}' 列全为 NaN。")

    mean_loss = (panel_df.groupby(join_col)[econ_col]
                         .mean()
                         .reset_index()
                         .rename(columns={econ_col: "mean_daily_loss_usd"}))

    merged = mean_loss.merge(labour_df[[join_col, "n_workers"]],
                             on=join_col, how="left")
    merged["total_annual_loss_usd"] = (
        merged["mean_daily_loss_usd"] * merged["n_workers"] * working_days_yr
    )
    merged["total_annual_loss_bn_usd"] = merged["total_annual_loss_usd"] / 1e9
    return merged.sort_values("total_annual_loss_usd", ascending=False)


# ══════════════════════════════════════════════════════════════
# §5-D. ★ 直接高温劳动损失（Graff Zivin & Neidell 2014）
# ══════════════════════════════════════════════════════════════
def compute_direct_heat_labour_loss(
    tmax_arr,
    wage_usd_per_hour = 0.0,
    tref              = HEAT_TREF,
    alpha             = HEAT_ALPHA,
    workday_hours     = ECON_WORKDAY_HOURS,
):
    """
    直接高温导致的劳动生产力损失。

    来源：Graff Zivin & Neidell (2014) JPE 122(5):1391-1432
          Seppanen et al. (2006)（室内劳动者，>25°C 每°C -2%，折算后综合）

    公式：
        heat_loss_pct      = alpha × max(Tmax - tref, 0)   [%/workday]
        econ_loss_usd_heat = (heat_loss_pct/100) × wage × workday_hours

    参数
    ────
    tmax_arr          : array-like，当日最高气温 [°C]
    wage_usd_per_hour : 当地时均工资（USD）；0 则 USD 列为 NaN
    tref              : 阈值温度，默认 29°C
    alpha             : 损失系数，默认 0.5 %/°C（全劳动力加权）

    返回 dict：
        "pct" : 生产力损失（%），shape 同输入
        "usd" : USD/人/工作日，shape 同输入
    """
    tmax = np.asarray(tmax_arr, dtype=float)
    pct  = alpha * np.maximum(tmax - tref, 0.0)

    if wage_usd_per_hour > 0:
        usd = (pct / 100.0) * wage_usd_per_hour * workday_hours
    else:
        usd = np.full_like(pct, np.nan)

    return {"pct": pct, "usd": usd}


# ══════════════════════════════════════════════════════════════
# §6. HNE 阈值计算
# ══════════════════════════════════════════════════════════════
def _select_df_for_thr(df_all, df_warm, season=HNE_THR_SEASON):
    if season == "warm_season" and df_warm is not None and len(df_warm) > 0:
        return df_warm
    return df_all

def compute_hne_thr_pooled(df_for_thr, q=HNE_PERCENTILE_TMIN):
    tmin = df_for_thr.groupby("local_date")["temp_C"].min().dropna()
    if len(tmin) == 0:
        return np.nan
    return float(np.nanpercentile(tmin.values, q))

def compute_hne_thr_loyo(df_for_thr, target_year, q=HNE_PERCENTILE_TMIN,
                          min_ref_years=MIN_LOYO_REF_YEARS):
    df_ref = df_for_thr[df_for_thr["year"] != target_year]
    if df_ref["year"].nunique() < min_ref_years:
        return np.nan
    tmin = df_ref.groupby("local_date")["temp_C"].min().dropna()
    if len(tmin) == 0:
        return np.nan
    return float(np.nanpercentile(tmin.values, q))

def build_hne_loyo_thr_dict(df_for_thr, q=HNE_PERCENTILE_TMIN):
    available_years = sorted(df_for_thr["year"].unique())
    thr_by_year     = {}
    for yr in available_years:
        thr_by_year[yr] = compute_hne_thr_loyo(df_for_thr, target_year=yr, q=q)
        ref_n = len([y for y in available_years if y != yr])
        print(f"    [HNE-LOYO] {yr}: thr={thr_by_year[yr]:.2f}°C  (ref={ref_n}yr)")
    all_dates = sorted(df_for_thr["local_date"].unique())
    return {
        pd.Timestamp(d): thr_by_year.get(pd.Timestamp(d).year, np.nan)
        for d in all_dates
    }


# ══════════════════════════════════════════════════════════════
# §7. HNE 逐日计算 + 睡眠损失 + 经济损失（双渠道）
# ══════════════════════════════════════════════════════════════
def compute_daily_hne_vec(df_loc, lat, hne_thr_src,
                           wage_usd_per_hour=ECON_WAGE_USD_DEFAULT):
    """
    逐日计算：HNE（He 2022）+ 睡眠损失（Minor 2022）
              + sleep loss 经济损失（Gibson & Shrader 2018; Hafner 2016）
              + 直接高温劳动损失（Graff Zivin & Neidell 2014）★ 新增
              + 合并总损失 ★ 新增

    输出列（每行 = 一日）：
    ─────────────────────
    hne_d, tmin_night, tmax_day, hne_thr_used,
    Tw_min_night, RH_night_mean,
    sleep_loss_min,
    econ_loss_pct_short, _long, _hourly, _present,
    econ_loss_usd_sleep,
    heat_loss_pct,          ← ★ 直接高温渠道
    econ_loss_usd_heat,     ← ★ 直接高温渠道
    total_loss_pct,         ← ★ 合并
    total_loss_usd          ← ★ 合并
    """
    df = df_loc.copy()
    if len(df) == 0:
        return pd.DataFrame()

    if "dew_C" not in df.columns:
        df["dew_C"] = np.nan
    valid_dew = df["dew_C"].notna()
    rh_arr    = np.where(valid_dew,
                         dewpoint_to_rh(df["temp_C"].values, df["dew_C"].values),
                         np.nan)
    df["RH"] = rh_arr
    df["Tw"] = np.where(~np.isnan(rh_arr),
                        wet_bulb(df["temp_C"].values, rh_arr),
                        np.nan)

    df = add_is_night_column(df, lat)

    if isinstance(hne_thr_src, dict):
        date_ts  = df["local_dt"].dt.normalize()
        thr_arr  = date_ts.map(hne_thr_src).values.astype(float)
    else:
        thr_arr  = np.full(len(df), float(hne_thr_src))

    df["hne_thr"] = thr_arr
    df = df.dropna(subset=["hne_thr"])
    if len(df) == 0:
        return pd.DataFrame()

    df["hne_hourly"] = np.where(
        df["is_night"],
        np.maximum(df["temp_C"].values - df["hne_thr"].values, 0.0),
        0.0
    )
    df["T_night"] = np.where(df["is_night"],  df["temp_C"], np.nan)
    df["T_day"]   = np.where(~df["is_night"], df["temp_C"], np.nan)
    df["Tw_n"]    = np.where(df["is_night"],  df["Tw"],     np.nan)
    df["RH_n"]    = np.where(df["is_night"],  df["RH"],     np.nan)

    g   = df.groupby("local_date")
    agg = pd.DataFrame({
        "hne_d":         g["hne_hourly"].sum(),
        "tmin_night":    g["T_night"].min(),
        "tmax_day":      g["T_day"].max(),
        "hne_thr_used":  g["hne_thr"].first(),
        "Tw_min_night":  g["Tw_n"].min() if "Tw" in df.columns else np.nan,
        "RH_night_mean": g["RH_n"].mean() if "RH" in df.columns else np.nan,
    }).reset_index()

    # 每行是一日；temp_h00 ... temp_h23 是该日本地时对应小时的平均气温。
    df["local_hour_int"] = np.floor(df["local_hour"]).astype(int) % 24

    temp_hourly = (
        df.groupby(["local_date", "local_hour_int"])["temp_C"]
          .mean()
          .unstack("local_hour_int")
    )

    temp_hourly = temp_hourly.reindex(columns=range(24))
    temp_hourly.columns = [f"temp_h{h:02d}" for h in range(24)]

    agg = agg.merge(
        temp_hourly.reset_index(),
        on="local_date",
        how="left"
    )

    agg["n_night_obs"] = g["is_night"].sum().values
    agg["n_day_obs"]   = (~df["is_night"]).groupby(df["local_date"]).sum().values

    # ── [ECON-1~2] Sleep loss → 经济损失 ────────────────────
    agg["sleep_loss_min"] = compute_sleep_loss_minutes(agg["tmin_night"].values)

    econ = compute_economic_loss(
        agg["sleep_loss_min"].values,
        wage_usd_per_hour=wage_usd_per_hour,
    )
    agg["econ_loss_pct_short"]   = econ["pct_short"]
    agg["econ_loss_pct_long"]    = econ["pct_long"]
    agg["econ_loss_pct_hourly"]  = econ["pct_hourly"]
    agg["econ_loss_pct_present"] = econ["pct_present"]
    agg["econ_loss_usd_sleep"]   = econ["usd"]
    agg["econ_loss_usd"]         = econ["usd"]   # 向后兼容

    # ── [ECON-5] ★ 直接高温 → 劳动损失 ─────────────────────
    heat = compute_direct_heat_labour_loss(
        agg["tmax_day"].values,
        wage_usd_per_hour=wage_usd_per_hour,
    )
    agg["heat_loss_pct"]      = heat["pct"]
    agg["econ_loss_usd_heat"] = heat["usd"]

    # ── [ECON-6] ★ 合并总损失 ───────────────────────────────
    agg["total_loss_pct"] = agg["econ_loss_pct_short"] + agg["heat_loss_pct"]
    if wage_usd_per_hour > 0:
        agg["total_loss_usd"] = (
            agg["econ_loss_usd_sleep"].fillna(0)
            + agg["econ_loss_usd_heat"].fillna(0)
        )
    else:
        agg["total_loss_usd"] = np.nan

    return agg


# ══════════════════════════════════════════════════════════════
# §8. HW 阈值计算
# ══════════════════════════════════════════════════════════════
def compute_hw_threshold_from_tmax(tmax_series, q=HW_PERCENTILE_TMAX,
                                    w=HW_WINDOW_HALF):
    ref = tmax_series.reset_index()
    ref.columns = ["local_date", "tmax"]
    ref["local_date"] = pd.to_datetime(ref["local_date"])
    ref["doy"] = ref["local_date"].apply(
        lambda x: (pd.Timestamp(2001, x.month, x.day).dayofyear
                   if not (x.month == 2 and x.day == 29) else 59)
    )
    ref = ref.dropna(subset=["tmax"])
    thresholds, valid_counts = {}, {}
    for d in tmax_series.index:
        d_ts = pd.Timestamp(d)
        doy  = (pd.Timestamp(2001, d_ts.month, d_ts.day).dayofyear
                if not (d_ts.month == 2 and d_ts.day == 29) else 59)
        lo, hi = doy - w, doy + w
        if lo < 1:
            mask = (ref["doy"] >= 365 + lo) | (ref["doy"] <= hi)
        elif hi > 365:
            mask = (ref["doy"] >= lo) | (ref["doy"] <= hi - 365)
        else:
            mask = (ref["doy"] >= lo) & (ref["doy"] <= hi)
        vals            = ref.loc[mask, "tmax"].dropna().values
        valid_counts[d] = len(vals)
        thresholds[d]   = float(np.nanpercentile(vals, q)) if len(vals) > 0 else np.nan
    return (pd.Series(thresholds).sort_index(),
            pd.Series(valid_counts).sort_index())


# ══════════════════════════════════════════════════════════════
# §9. HW 检测
# ══════════════════════════════════════════════════════════════
def detect_heatwave(daily_tmax, thr_series, min_days=HW_MIN_DAYS):
    thr_aligned = thr_series.reindex(daily_tmax.index)
    above       = (daily_tmax > thr_aligned).fillna(False).values
    flag        = np.zeros(len(above), dtype=int)
    n, i        = len(above), 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_days:
                flag[i:j] = 1
            i = j
        else:
            i += 1
    return pd.Series(flag, index=daily_tmax.index, name="hw_flag")


# ══════════════════════════════════════════════════════════════
# §10. 数据加载
# ══════════════════════════════════════════════════════════════
def load_station_multiyear(usaf, wban, years, base_dir, lon,
                            min_valid_frac=MIN_VALID_FRAC_ANNUAL,
                            verbose=True):
    frames, valid_years = [], []
    for yr in years:
        fp      = isd_path(usaf, wban, yr, base_dir)
        raw = read_isd_lite(fp)
        if raw is None:
            if verbose: print(f"    [{yr}] 文件缺失，跳过")
            continue
        cleaned = clean_and_localise(raw, yr, lon)
        year_days  = 366 if is_leap(yr) else 365
        valid_days = cleaned.dropna(subset=["temp_C"])["local_date"].nunique()
        frac       = valid_days / year_days
        if frac < min_valid_frac:
            if verbose: print(f"    [{yr}] 全年有效率 {frac:.1%} < {min_valid_frac:.0%}，跳过")
            continue
        if verbose: print(f"    [{yr}] 全年有效率 {frac:.1%} ({valid_days}/{year_days}天) ✓")
        frames.append(cleaned)
        valid_years.append(yr)
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), valid_years


def load_station_multiyear_warmseason(usaf, wban, years, base_dir, lon,
                                       months=None,
                                       lat=None,
                                       min_valid_frac=MIN_VALID_FRAC_WARM,
                                       verbose=True):
    """
    Load hemisphere-specific warm-season data.

    NH uses JJA; SH uses DJF. For SH, December is assigned to the next
    warm-season year, matching analysis_multiyear.py.
    """
    if months is None:
        months = warm_months_for_lat(lat)

    frames_all = []
    for yr in years:
        fp = isd_path(usaf, wban, yr, base_dir)
        raw = read_isd_lite(fp)
        if raw is None:
            continue
        frames_all.append(clean_and_localise(raw, yr, lon))

    if not frames_all:
        return None, []

    df_all = pd.concat(frames_all, ignore_index=True)
    df_warm = df_all[
        df_all["local_date"].apply(
            lambda d: is_warm_season_date_for_analysis(d, lat)
        )
    ].copy()

    if len(df_warm) == 0:
        if verbose:
            print("    暖季无数据，跳过")
        return None, []

    df_warm["warm_season_year"] = df_warm["local_date"].apply(
        lambda d: warm_season_year_from_date(d, lat)
    )

    valid_years = []
    keep_frames = []

    for season_year, grp in df_warm.groupby("warm_season_year"):
        expected_days = expected_warm_season_days(int(season_year), lat)
        valid_days = grp.dropna(subset=["temp_C"])["local_date"].nunique()
        frac = valid_days / expected_days

        if frac >= min_valid_frac:
            valid_years.append(int(season_year))
            keep_frames.append(grp)
            if verbose:
                print(
                    f"    [{int(season_year)} {warm_season_label_for_lat(lat)}] "
                    f"暖季有效率 {frac:.1%} ({valid_days}/{expected_days}天) ✓"
                )
        elif verbose:
            print(
                f"    [{int(season_year)} {warm_season_label_for_lat(lat)}] "
                f"暖季有效率 {frac:.1%} < {min_valid_frac:.0%}，跳过"
            )

    if not keep_frames:
        return None, []

    return pd.concat(keep_frames, ignore_index=True), valid_years

# ══════════════════════════════════════════════════════════════
# §11. 单站点面板构建
# ══════════════════════════════════════════════════════════════
def build_station_panel(df_all, df_warm, lat, station_label,
                         hne_method="both",
                         wage_usd_per_hour=ECON_WAGE_USD_DEFAULT,
                         era5_usaf=None, era5_wban=None, era5_label=None,
                         analysis_hw_flags=None):
    df_for_thr = _select_df_for_thr(df_all, df_warm)
    tmax_all = (df_all.dropna(subset=["temp_C"])
                      .groupby("local_date")["temp_C"].max().dropna())

    # HW/NHW flags and HW threshold diagnostics are supplied by
    # 01_main_pair_period_metrics.py; no downstream HW threshold reconstruction.

    ext_aligned = align_analysis_hw_flags(
        analysis_hw_flags, tmax_all.index
    )
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
            f"    [SKIP-{station_label}] canonical analysis_multiyear "
            "HW/NHW flags missing or mismatched; fallback disabled"
        )
        return {
            "pooled": {
                "panel": None,
                "hw_flag": pd.Series(dtype=int),
            },
            "loyo": {
                "panel": None,
                "hw_flag": pd.Series(dtype=int),
            },
        }

    def _num_series_from_01(col):
        if col not in ext_aligned.columns:
            return pd.Series(np.nan, index=tmax_all.index, dtype=float)
        return pd.to_numeric(ext_aligned[col], errors="coerce").reindex(tmax_all.index)

    hw_thr_series_raw = _num_series_from_01("hw_threshold_raw")
    hw_thr_series = _num_series_from_01("hw_threshold_corrected")
    if hw_thr_series.notna().sum() == 0:
        hw_thr_series = hw_thr_series_raw.copy()

    def _first_nonempty_01(col, default=""):
        if col not in ext_aligned.columns:
            return default
        vals = ext_aligned[col].dropna().astype(str).str.strip()
        vals = vals[vals.ne("")]
        return vals.iloc[0] if len(vals) else default

    bias_correction_method_01 = _first_nonempty_01(
        "bias_correction_method", "from_01_main_pair_period_metrics"
    )
    qm_diagnostics = {
        "qm_ref_mode": _first_nonempty_01("qm_ref_mode", ""),
        "qm_n_quantiles": QM_N_QUANTILES,
        "overlap_days": (
            float(pd.to_numeric(ext_aligned["qm_overlap_days"], errors="coerce").dropna().iloc[0])
            if "qm_overlap_days" in ext_aligned.columns
            and pd.to_numeric(ext_aligned["qm_overlap_days"], errors="coerce").notna().any()
            else np.nan
        ),
    }

    valid_flags = ext_aligned[required_flag_cols].notna().all(axis=1)
    hw_flag_series = pd.Series(
        np.nan, index=tmax_all.index, name="hw_flag"
    )
    hw_flag_series.loc[valid_flags] = (
        ext_aligned.loc[
            valid_flags, "hw_flag_percentile_warm_season"
        ].astype(int).values
    )
    hw_source = "analysis_multiyear_daily_heatwave_flags"

    print(f"    [{station_label}] HW_thr_mean={hw_thr_series.mean():.2f}°C  "
          f"HW_days={int(hw_flag_series.sum())}  source={hw_source}")

    results = {}
    for method in ["pooled", "loyo"]:
        if hne_method not in [method, "both"]:
            results[method] = {"panel": None, "hw_flag": hw_flag_series}
            continue

        if method == "pooled":
            # [ERA5-MOD start] no ISD fallback
            if USE_ERA5_CLIMATOLOGY:
                hne_thr = (compute_hne_thr_era5(era5_usaf, era5_wban, era5_label,
                                             months=warm_months_for_lat(lat))
                           if era5_usaf else np.nan)
            else:
                hne_thr = compute_hne_thr_pooled(df_for_thr)
            # [ERA5-MOD end]

            if np.isnan(hne_thr):
                print(f"    [SKIP-{station_label}/pooled] ERA5 HNE threshold missing/invalid")
                results[method] = {"panel": None, "hw_flag": hw_flag_series}
                continue
            print(f"    [{station_label}/pooled] HNE_thr={hne_thr:.2f}°C")

        else:
            # [ERA5-MOD start] no ISD fallback
            if USE_ERA5_CLIMATOLOGY and era5_usaf:
                _loyo_yr = build_hne_loyo_thr_era5(era5_usaf, era5_wban, era5_label,
                                                months=warm_months_for_lat(lat))
                if not _loyo_yr:
                    print(f"    [SKIP-{station_label}/loyo] ERA5 LOYO HNE threshold missing/invalid")
                    results[method] = {"panel": None, "hw_flag": hw_flag_series}
                    continue
                _all_dates = sorted(df_for_thr["local_date"].unique())
                hne_thr = {
                    pd.Timestamp(d): _loyo_yr.get(pd.Timestamp(d).year, np.nan)
                    for d in _all_dates
                }
            else:
                hne_thr = build_hne_loyo_thr_dict(df_for_thr)
            # [ERA5-MOD end]

        hne_df = compute_daily_hne_vec(df_all, lat, hne_thr,
                                        wage_usd_per_hour=wage_usd_per_hour)
        if len(hne_df) == 0:
            results[method] = {"panel": None, "hw_flag": hw_flag_series}
            continue

        hne_df["date_dt"] = pd.to_datetime(hne_df["local_date"])
        hne_df = hne_df.set_index("date_dt")
        _ext_hne = align_analysis_hw_flags(
            analysis_hw_flags, hne_df.index
        )
        if (
            _ext_hne is None
            or any(c not in _ext_hne.columns for c in required_flag_cols)
            or _ext_hne[required_flag_cols].dropna(how="any").empty
        ):
            print(
                f"    [SKIP-{station_label}/{method}] canonical "
                "HW/NHW flags do not align; fallback disabled"
            )
            results[method] = {
                "panel": None,
                "hw_flag": hw_flag_series,
            }
            continue

        valid_hne_flags = (
            _ext_hne[required_flag_cols].notna().all(axis=1)
        )
        hne_df = hne_df.loc[valid_hne_flags].copy()
        _ext_hne = _ext_hne.loc[valid_hne_flags].copy()
        if hne_df.empty:
            results[method] = {
                "panel": None,
                "hw_flag": hw_flag_series,
            }
            continue

        hne_df["hw_flag"] = (
            _ext_hne["hw_flag_percentile_warm_season"]
            .astype(int).values
        )
        hne_df["warm_season_flag"] = (
            _ext_hne["is_warm_season"].astype(int).values
        )
        if "hw_threshold_corrected" in _ext_hne.columns:
            _ext_thr = pd.to_numeric(
                _ext_hne["hw_threshold_corrected"], errors="coerce"
            )
            if _ext_thr.notna().any():
                hne_df["hw_thr_tmax"] = _ext_thr.values
            else:
                hne_df["hw_thr_tmax"] = (
                    hw_thr_series.reindex(hne_df.index).values
                )
        else:
            hne_df["hw_thr_tmax"] = (
                hw_thr_series.reindex(hne_df.index).values
            )

        hne_df["hw_source"] = hw_source
        hne_df["bias_correction_method"] = bias_correction_method_01
        hne_df["qm_ref_mode"] = qm_diagnostics.get("qm_ref_mode", "")
        hne_df["qm_n_quantiles"] = qm_diagnostics.get("qm_n_quantiles", np.nan)
        hne_df["qm_overlap_days"] = qm_diagnostics.get("overlap_days", np.nan)
        hne_df["hw_threshold_raw_mean"] = float(hw_thr_series_raw.mean())
        hne_df["hw_threshold_qm_mean"] = (
            float(hw_thr_series.mean()) if USE_QM_BIAS_CORRECTION else np.nan
        )

        hne_df["hw_tmax_excess"] = np.where(
            hne_df["hw_flag"] == 1,
            hne_df["tmax_day"] - hne_df["hw_thr_tmax"], 0.0)
        hne_df["month"]            = hne_df.index.month
        hne_df["year"]             = hne_df.index.year
        if "warm_season_flag" not in hne_df.columns:
            hne_df["warm_season_flag"] = pd.to_datetime(
                hne_df.index
            ).map(lambda d: is_warm_season_date_for_analysis(d, lat)).astype(int)
        hne_df["hne_method"]       = method
        hne_df["station"]          = station_label

        panel = hne_df.reset_index().rename(columns={"date_dt": "date"})
        results[method] = {"panel": panel, "hw_flag": hw_flag_series}

    return results

# ══════════════════════════════════════════════════════════════
# §12. 暖季分期切分
# ══════════════════════════════════════════════════════════════
def build_warm_season_periods(hne_daily_df, hw_flag_series,
                               pair_id, row, method, output_type):
    df = hne_daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    warm_mask = df["warm_season_flag"] == 1
    df_warm   = df[warm_mask].copy()
    if len(df_warm) == 0:
        return {"warm_season": None, "heatwave": None, "non_heatwave": None}
    hw_aligned = hw_flag_series.reindex(pd.to_datetime(df_warm["date"]), fill_value=0)
    df_warm["hw_flag_urban"] = hw_aligned.values
    hw_dates  = set(df_warm.loc[df_warm["hw_flag_urban"] == 1, "date"])
    nhw_dates = set(df_warm.loc[df_warm["hw_flag_urban"] == 0, "date"])

    def _tag(sub, period_name):
        if sub is None or len(sub) == 0:
            return None
        sub = sub.copy()
        sub["period"]      = period_name
        sub["pair_id"]     = pair_id
        sub["lat_urban"]   = float(row["lat_urban"])
        sub["lon_urban"]   = float(row["lon_urban"])
        sub["lat_group"]   = row.get("lat_group", "Unknown")
        sub["hne_method"]  = method
        sub["output_type"] = output_type
        if "continent" in row:
            sub["continent"] = row["continent"]
        return sub

    return {
        "warm_season":  _tag(df_warm, "warm_season"),
        "heatwave":     _tag(df_warm[df_warm["date"].isin(hw_dates)],  "heatwave"),
        "non_heatwave": _tag(df_warm[df_warm["date"].isin(nhw_dates)], "non_heatwave"),
    }


# ══════════════════════════════════════════════════════════════
# §13. Output A：独立站点
# ══════════════════════════════════════════════════════════════
def build_independent_output(pair_id, row,
                              wage_usd_per_hour=ECON_WAGE_USD_DEFAULT):
    try:
        usaf_u, wban_u, usaf_r, wban_r = parse_pair_id(pair_id)
    except Exception as e:
        print(f"  [SKIP] {e}")
        return [], []

    lon_u, lat_u = float(row["lon_urban"]),  float(row["lat_urban"])
    lon_r, lat_r = float(row["lon_rural"]),  float(row["lat_rural"])
    annual_panels, warm_period_panels = [], []
    analysis_hw_flags = get_analysis_hw_flags_for_pair(pair_id)
    if analysis_hw_flags is None or len(analysis_hw_flags) == 0:
        print(
            f"  [SKIP-indep] {pair_id}: canonical analysis_multiyear "
            "HW/NHW flags missing; fallback disabled"
        )
        return [], []

    for (usaf, wban, lon, lat, label) in [
        (usaf_u, wban_u, lon_u, lat_u, "urban"),
        (usaf_r, wban_r, lon_r, lat_r, "rural"),
    ]:
        print(f"  [indep/{label}] 全年数据加载:")
        df_all, yrs_all = load_station_multiyear(usaf, wban, YEARS, ISD_BASE_DIR, lon)
        print(f"  [indep/{label}] 暖季数据加载:")
        df_warm, yrs_warm = load_station_multiyear_warmseason(
            usaf, wban, YEARS, ISD_BASE_DIR, lon, lat=lat)
        if df_all is None or len(yrs_all) == 0:
            print(f"  [SKIP-indep/{label}] 无有效全年数据")
            continue
        # [ERA5-MOD start]
        if USE_ERA5_CLIMATOLOGY:
            station_res = build_station_panel(
                df_all, df_warm, lat, label, "both",
                wage_usd_per_hour=wage_usd_per_hour,
                era5_usaf=usaf, era5_wban=wban, era5_label=label,
                analysis_hw_flags=analysis_hw_flags,
            )
        else:
            station_res = build_station_panel(df_all, df_warm, lat, label, "both",
                                               wage_usd_per_hour=wage_usd_per_hour,
                                               analysis_hw_flags=analysis_hw_flags)
        # [ERA5-MOD end]
        for method, res in station_res.items():
            panel = res["panel"]
            hw_flag_ser = res["hw_flag"]
            if panel is None:
                continue
            panel["pair_id"]          = pair_id
            panel["lat_urban"]        = float(row["lat_urban"])
            panel["lon_urban"]        = float(row["lon_urban"])
            panel["lat_group"]        = row.get("lat_group", "Unknown")
            panel["kg_code"]          = row.get("kg_code",          np.nan)
            panel["kg_group"]         = row.get("kg_group",         np.nan)
            panel["climate_zone_main"]= row.get("climate_zone_main",np.nan)
            panel["output_type"]      = "independent"
            panel["period"]           = "annual"
            panel["n_valid_yrs_all"]  = len(yrs_all)
            panel["n_valid_yrs_warm"] = len(yrs_warm)
            if "continent" in row:
                panel["continent"] = row["continent"]
            annual_panels.append(panel)
            if df_warm is not None and len(yrs_warm) > 0:
                periods = build_warm_season_periods(
                    panel, hw_flag_ser, pair_id, row, method, "independent")
                for p_name, sub in periods.items():
                    if sub is not None:
                        sub["n_valid_yrs_warm"] = len(yrs_warm)
                        warm_period_panels.append(sub)
    return annual_panels, warm_period_panels


# ══════════════════════════════════════════════════════════════
# §14. Output B：配对分析
# ══════════════════════════════════════════════════════════════
def build_paired_output(pair_id, row,
                         wage_usd_per_hour=ECON_WAGE_USD_DEFAULT):
    try:
        usaf_u, wban_u, usaf_r, wban_r = parse_pair_id(pair_id)
    except Exception as e:
        print(f"  [SKIP] {e}")
        return [], []

    lon_u, lat_u = float(row["lon_urban"]), float(row["lat_urban"])
    lon_r, lat_r = float(row["lon_rural"]), float(row["lat_rural"])

    print(f"  [paired] 全年数据加载:")
    df_u_all, yrs_u_all = load_station_multiyear(
        usaf_u, wban_u, YEARS, ISD_BASE_DIR, lon_u)
    df_r_all, yrs_r_all = load_station_multiyear(
        usaf_r, wban_r, YEARS, ISD_BASE_DIR, lon_r)
    if df_u_all is None or df_r_all is None:
        print("  [SKIP-paired] 城乡全年数据不足")
        return [], []

    print(f"  [paired] 暖季数据加载:")
    df_u_warm, yrs_u_warm = load_station_multiyear_warmseason(
        usaf_u, wban_u, YEARS, ISD_BASE_DIR, lon_u, lat=lat_u)
    df_r_warm, yrs_r_warm = load_station_multiyear_warmseason(
        usaf_r, wban_r, YEARS, ISD_BASE_DIR, lon_r, lat=lat_u)
    has_warm = (df_u_warm is not None and df_r_warm is not None
                and len(yrs_u_warm) > 0 and len(yrs_r_warm) > 0)

    analysis_hw_flags = get_analysis_hw_flags_for_pair(pair_id)
    if analysis_hw_flags is None or len(analysis_hw_flags) == 0:
        print(
            f"  [SKIP-paired] {pair_id}: canonical analysis_multiyear "
            "HW/NHW flags missing; fallback disabled"
        )
        return [], []

    tmax_u_all = (df_u_all.dropna(subset=["temp_C"])
                          .groupby("local_date")["temp_C"].max().dropna())

    # Canonical HW/NHW flags and HW thresholds come only from 01.
    _ext_pair_hw = align_analysis_hw_flags(analysis_hw_flags, tmax_u_all.index)
    required_flag_cols = [
        "is_warm_season",
        "hw_flag_percentile_warm_season",
        "nhw_flag_percentile_warm_season",
    ]
    if (
        _ext_pair_hw is None
        or any(c not in _ext_pair_hw.columns for c in required_flag_cols)
        or _ext_pair_hw[required_flag_cols].dropna(how="any").empty
    ):
        print(
            f"  [SKIP-paired] {pair_id}: canonical 01 HW/NHW flags "
            "do not align; fallback disabled"
        )
        return [], []

    valid_pair_flags = _ext_pair_hw[required_flag_cols].notna().all(axis=1)
    hw_flag_u = pd.Series(np.nan, index=tmax_u_all.index, name="hw_flag")
    hw_flag_u.loc[valid_pair_flags] = (
        _ext_pair_hw.loc[valid_pair_flags, "hw_flag_percentile_warm_season"]
        .astype(int).values
    )
    hw_source = "01_main_pair_period_metrics_daily_heatwave_flags"

    def _num_pair_series_from_01(col):
        if col not in _ext_pair_hw.columns:
            return pd.Series(np.nan, index=tmax_u_all.index, dtype=float)
        return pd.to_numeric(_ext_pair_hw[col], errors="coerce").reindex(tmax_u_all.index)

    hw_thr_u_raw = _num_pair_series_from_01("hw_threshold_raw")
    hw_thr_u = _num_pair_series_from_01("hw_threshold_corrected")
    if hw_thr_u.notna().sum() == 0:
        hw_thr_u = hw_thr_u_raw.copy()

    qm_diagnostics = {
        "qm_ref_mode": (
            _ext_pair_hw["qm_ref_mode"].dropna().astype(str).iloc[0]
            if "qm_ref_mode" in _ext_pair_hw.columns
            and _ext_pair_hw["qm_ref_mode"].notna().any()
            else ""
        ),
        "qm_n_quantiles": QM_N_QUANTILES,
        "overlap_days": (
            float(pd.to_numeric(_ext_pair_hw["qm_overlap_days"], errors="coerce").dropna().iloc[0])
            if "qm_overlap_days" in _ext_pair_hw.columns
            and pd.to_numeric(_ext_pair_hw["qm_overlap_days"], errors="coerce").notna().any()
            else np.nan
        ),
    }

    df_u_for_thr = _select_df_for_thr(df_u_all, df_u_warm)
    df_r_for_thr = _select_df_for_thr(df_r_all, df_r_warm)

    annual_panels, warm_period_panels = [], []

    for method in ["pooled", "loyo"]:
        if method == "pooled":
            # [ERA5-MOD start] no ISD fallback
            if USE_ERA5_CLIMATOLOGY:
                hne_thr_u = compute_hne_thr_era5(
                    usaf_u, wban_u, "urban", months=warm_months_for_lat(lat_u)
                )
                hne_thr_r = compute_hne_thr_era5(
                    usaf_r, wban_r, "rural", months=warm_months_for_lat(lat_u)
                )
            else:
                hne_thr_u = compute_hne_thr_pooled(df_u_for_thr)
                hne_thr_r = compute_hne_thr_pooled(df_r_for_thr)
            # [ERA5-MOD end]

            if np.isnan(hne_thr_u) or np.isnan(hne_thr_r):
                print("  [SKIP-paired/pooled] ERA5 HNE threshold missing/invalid")
                continue

        else:
            # [ERA5-MOD start] no ISD fallback
            if USE_ERA5_CLIMATOLOGY:
                _loyo_u = build_hne_loyo_thr_era5(
                    usaf_u, wban_u, "urban", months=warm_months_for_lat(lat_u)
                )
                _loyo_r = build_hne_loyo_thr_era5(
                    usaf_r, wban_r, "rural", months=warm_months_for_lat(lat_u)
                )
                if not _loyo_u or not _loyo_r:
                    print("  [SKIP-paired/loyo] ERA5 LOYO HNE threshold missing/invalid")
                    continue

                _dates_u = sorted(df_u_for_thr["local_date"].unique())
                _dates_r = sorted(df_r_for_thr["local_date"].unique())

                hne_thr_u = {
                    pd.Timestamp(d): _loyo_u.get(pd.Timestamp(d).year, np.nan)
                    for d in _dates_u
                }
                hne_thr_r = {
                    pd.Timestamp(d): _loyo_r.get(pd.Timestamp(d).year, np.nan)
                    for d in _dates_r
                }
            else:
                hne_thr_u = build_hne_loyo_thr_dict(df_u_for_thr)
                hne_thr_r = build_hne_loyo_thr_dict(df_r_for_thr)
            # [ERA5-MOD end]

        hne_u = compute_daily_hne_vec(df_u_all, lat_u, hne_thr_u,
                                       wage_usd_per_hour=wage_usd_per_hour)
        hne_r = compute_daily_hne_vec(df_r_all, lat_r, hne_thr_r,
                                       wage_usd_per_hour=wage_usd_per_hour)
        if len(hne_u) == 0 or len(hne_r) == 0:
            continue

        hne_u["date_dt"] = pd.to_datetime(hne_u["local_date"])
        hne_r["date_dt"] = pd.to_datetime(hne_r["local_date"])

        u = hne_u.set_index("date_dt").add_suffix("_U")
        r = hne_r.set_index("date_dt").add_suffix("_R")
        panel = u.join(r, how="inner")
        if len(panel) == 0:
            continue

        _ext_panel_hw = align_analysis_hw_flags(
            analysis_hw_flags, panel.index
        )
        if (
            _ext_panel_hw is None
            or any(c not in _ext_panel_hw.columns for c in required_flag_cols)
            or _ext_panel_hw[required_flag_cols].dropna(how="any").empty
        ):
            print(
                f"  [SKIP-paired/{method}] {pair_id}: canonical "
                "HW/NHW flags do not align; fallback disabled"
            )
            continue

        valid_panel_flags = (
            _ext_panel_hw[required_flag_cols].notna().all(axis=1)
        )
        panel = panel.loc[valid_panel_flags].copy()
        _ext_panel_hw = _ext_panel_hw.loc[valid_panel_flags].copy()
        if panel.empty:
            continue

        panel["hw_flag"] = (
            _ext_panel_hw["hw_flag_percentile_warm_season"]
            .astype(int).values
        )
        panel["warm_season_flag"] = (
            _ext_panel_hw["is_warm_season"].astype(int).values
        )
        if "hw_threshold_corrected" in _ext_panel_hw.columns:
            _ext_thr = pd.to_numeric(
                _ext_panel_hw["hw_threshold_corrected"],
                errors="coerce",
            )
            if _ext_thr.notna().any():
                panel["hw_thr_tmax_U"] = _ext_thr.values
            else:
                panel["hw_thr_tmax_U"] = (
                    hw_thr_u.reindex(panel.index).values
                )
        else:
            panel["hw_thr_tmax_U"] = (
                hw_thr_u.reindex(panel.index).values
            )

        panel["hw_source"] = hw_source
        panel["bias_correction_method"] = (
            "quantile_mapping_empirical_cdf"
            if USE_QM_BIAS_CORRECTION else "none_raw_era5_threshold"
        )
        panel["qm_ref_mode"] = qm_diagnostics.get("qm_ref_mode", "")
        panel["qm_n_quantiles"] = qm_diagnostics.get("qm_n_quantiles", np.nan)
        panel["qm_overlap_days"] = qm_diagnostics.get("overlap_days", np.nan)
        panel["hw_threshold_raw_mean"] = float(hw_thr_u_raw.mean())
        panel["hw_threshold_qm_mean"] = (
            float(hw_thr_u.mean()) if USE_QM_BIAS_CORRECTION else np.nan
        )

        panel["dHNE"]      = panel["hne_d_U"]      - panel["hne_d_R"]
        panel["dTMIN"]     = panel["tmin_night_U"] - panel["tmin_night_R"]
        panel["dTMAX"]     = panel["tmax_day_U"]   - panel["tmax_day_R"]
        panel["AsymIndex"] = panel["dTMIN"] - panel["dTMAX"]

        panel["d_sleep_loss_min"] = (
            panel["sleep_loss_min_U"] - panel["sleep_loss_min_R"])

        panel["d_econ_loss_pct_short"] = (
            panel["econ_loss_pct_short_U"] - panel["econ_loss_pct_short_R"])
        panel["d_econ_loss_pct_long"] = (
            panel["econ_loss_pct_long_U"] - panel["econ_loss_pct_long_R"])
        panel["d_econ_loss_pct_hourly"] = (
            panel["econ_loss_pct_hourly_U"] - panel["econ_loss_pct_hourly_R"])
        panel["d_econ_loss_pct_present"] = (
            panel["econ_loss_pct_present_U"] - panel["econ_loss_pct_present_R"])

        panel["d_heat_loss_pct"] = (
            panel["heat_loss_pct_U"] - panel["heat_loss_pct_R"])
        panel["d_total_loss_pct"] = (
            panel["total_loss_pct_U"] - panel["total_loss_pct_R"])

        if wage_usd_per_hour > 0:
            panel["d_econ_loss_usd"] = (
                panel["econ_loss_usd_U"] - panel["econ_loss_usd_R"])
            panel["d_econ_loss_usd_sleep"] = (
                panel["econ_loss_usd_sleep_U"] - panel["econ_loss_usd_sleep_R"])
            panel["d_econ_loss_usd_heat"] = (
                panel["econ_loss_usd_heat_U"] - panel["econ_loss_usd_heat_R"])
            panel["d_total_loss_usd"] = (
                panel["total_loss_usd_U"] - panel["total_loss_usd_R"])
        else:
            for col in ["d_econ_loss_usd", "d_econ_loss_usd_sleep",
                        "d_econ_loss_usd_heat", "d_total_loss_usd"]:
                panel[col] = np.nan

        panel["month"]            = panel.index.month
        panel["year"]             = panel.index.year
        if "warm_season_flag" not in panel.columns:
            panel["warm_season_flag"] = pd.to_datetime(
                panel.index
            ).map(lambda d: is_warm_season_date_for_analysis(d, lat_u)).astype(int)
        panel["hne_method"]       = method
        panel["pair_id"]          = pair_id
        panel["lat_urban"]        = float(row["lat_urban"])
        panel["lon_urban"]        = float(row["lon_urban"])
        panel["lat_group"]        = row.get("lat_group", "Unknown")
        panel["kg_code"]          = row.get("kg_code", np.nan)
        panel["kg_group"]         = row.get("kg_group", np.nan)
        panel["climate_zone_main"]= row.get("climate_zone_main", np.nan)
        panel["output_type"]      = "paired"
        panel["period"]           = "annual"
        panel["n_valid_yrs_U"]    = len(yrs_u_all)
        panel["n_valid_yrs_R"]    = len(yrs_r_all)

        if "continent" in row:
            panel["continent"] = row["continent"]

        panel = panel.reset_index().rename(columns={"date_dt": "date"})
        annual_panels.append(panel)

        if has_warm:
            periods = build_warm_season_periods(
                panel, hw_flag_u, pair_id, row, method, "paired")
            for p_name, sub in periods.items():
                if sub is not None:
                    sub["n_valid_yrs_U_warm"] = len(yrs_u_warm)
                    sub["n_valid_yrs_R_warm"] = len(yrs_r_warm)
                    warm_period_panels.append(sub)

    return annual_panels, warm_period_panels

# ══════════════════════════════════════════════════════════════
# §15. HW 事件级统计
# ══════════════════════════════════════════════════════════════
def extract_hw_events_with_hne(panel):
    if "hw_flag" not in panel.columns:
        return pd.DataFrame()
    df        = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df        = df.sort_values("date").reset_index(drop=True)
    flag_vals = df["hw_flag"].fillna(0).astype(int).values
    events, i, n = [], 0, len(flag_vals)
    while i < n:
        if flag_vals[i] == 1:
            s = i
            while i < n and flag_vals[i] == 1:
                i += 1
            e   = i - 1
            seg = df.iloc[s: e + 1]
            ev  = {
                "event_start":   seg["date"].iloc[0].date(),
                "event_end":     seg["date"].iloc[-1].date(),
                "duration_days": e - s + 1,
                "year":          seg["date"].dt.year.iloc[0],
                "month_start":   seg["date"].dt.month.iloc[0],
            }
            for col in [
                "hne_d_U", "hne_d_R", "dHNE",
                "tmin_night_U", "tmin_night_R", "dTMIN",
                "tmax_day_U",  "dTMAX", "AsymIndex",
                "sleep_loss_min_U", "sleep_loss_min_R", "d_sleep_loss_min",
                "econ_loss_pct_short_U", "econ_loss_pct_short_R",
                "d_econ_loss_pct_short", "d_econ_loss_pct_long",
                "econ_loss_usd_sleep_U", "d_econ_loss_usd_sleep",
                "heat_loss_pct_U", "heat_loss_pct_R", "d_heat_loss_pct",
                "econ_loss_usd_heat_U",  "d_econ_loss_usd_heat",
                "total_loss_pct_U",  "d_total_loss_pct",
                "total_loss_usd_U",  "d_total_loss_usd",
            ]:
                if col in seg.columns:
                    ev[f"{col}_total"] = float(seg[col].sum())
                    ev[f"{col}_mean"]  = float(seg[col].mean())
                    ev[f"{col}_peak"]  = float(seg[col].max())
            events.append(ev)
        else:
            i += 1
    return pd.DataFrame(events)


# ══════════════════════════════════════════════════════════════
# §16. 稳健性统计
# ══════════════════════════════════════════════════════════════
def compute_robustness_stats_paired(panel_pooled, panel_loyo):
    if panel_pooled is None or panel_loyo is None:
        return None, {}
    key_cols = [
        "hne_d_U", "hne_d_R", "dHNE", "dTMIN", "hw_flag",
        "sleep_loss_min_U", "sleep_loss_min_R", "d_sleep_loss_min",
        "d_econ_loss_pct_short", "d_econ_loss_pct_long",
        "d_econ_loss_usd_sleep",
        "d_heat_loss_pct", "d_econ_loss_usd_heat",
        "d_total_loss_pct", "d_total_loss_usd",
    ]
    avail = list(set([c for c in key_cols
                      if c in panel_pooled.columns and c in panel_loyo.columns]))
    if not avail:
        return None, {}
    p = panel_pooled.set_index("date")[avail].copy()
    l = panel_loyo.set_index("date")[avail].copy()
    common = p.index.intersection(l.index)
    if len(common) < 10:
        return None, {}
    p, l = p.loc[common], l.loc[common]
    stats_dict = {}
    for col in avail:
        vp   = p[col].values.astype(float)
        vl   = l[col].values.astype(float)
        mask = ~np.isnan(vp) & ~np.isnan(vl)
        if mask.sum() < 5:
            continue
        vp, vl = vp[mask], vl[mask]
        diff   = vl - vp
        r, _   = stats.pearsonr(vp, vl) if len(vp) > 2 else (np.nan, np.nan)
        stats_dict[col] = {
            "n_common":             int(len(vp)),
            "mean_pooled":          float(np.mean(vp)),
            "mean_loyo":            float(np.mean(vl)),
            "bias_loyo_minus_pool": float(np.mean(diff)),
            "bias_pct":             float(np.mean(diff) /
                                         max(abs(np.mean(vp)), 1e-6) * 100),
            "rmse":                 float(np.sqrt(np.mean(diff**2))),
            "pearson_r":            float(r) if not np.isnan(r) else np.nan,
        }
    return None, stats_dict

def export_pair_period_sleep_lc_peak(
    all_paired_pooled,
    all_paired_warm,
    out_path,
    peak_source_col="d_econ_loss_pct_short"
):
    """
    为 integrated figure 输出 pair-period 级别的夜间 sleep-caused LC loss 峰值。

    定义：
      night_sleep_peak = max(abs(d_econ_loss_pct_short))
    按 pair_id × period_norm 聚合

    输出列：
      pair_id, period_norm, night_sleep_peak,
      climate_zone_main, kg_group, kg_code, lat_group
    """
    frames = []

    # annual
    if all_paired_pooled:
        df_ann = pd.concat(all_paired_pooled, ignore_index=True).copy()
        df_ann["period"] = "annual"
        frames.append(df_ann)

    # warm / HW / NHW
    if all_paired_warm:
        df_warm = pd.concat(all_paired_warm, ignore_index=True).copy()
        frames.append(df_warm)

    if not frames:
        print("  [EXPORT] No paired data for sleep LC peak")
        return None

    df = pd.concat(frames, ignore_index=True)

    if peak_source_col not in df.columns:
        print(f"  [EXPORT] Missing column: {peak_source_col}")
        return None

    keep_cols = ["pair_id", "period", peak_source_col]
    for c in ["climate_zone_main", "kg_group", "kg_code", "lat_group"]:
        if c in df.columns:
            keep_cols.append(c)

    tmp = df[keep_cols].copy()
    tmp["period_norm"] = tmp["period"].apply(norm_period_for_integrated)
    tmp["night_sleep_peak"] = tmp[peak_source_col].abs()

    agg_map = {"night_sleep_peak": "max"}
    for c in ["climate_zone_main", "kg_group", "kg_code", "lat_group"]:
        if c in tmp.columns:
            agg_map[c] = "first"

    out = (tmp.groupby(["pair_id", "period_norm"], observed=True)
             .agg(agg_map)
             .reset_index())

    out.to_csv(out_path, index=False)
    print(f"  [EXPORT] Saved integrated input: {out_path}")
    return out

# ══════════════════════════════════════════════════════════════
# §17. 多进程任务封装与 Main
# ══════════════════════════════════════════════════════════════
def process_single_pair(args):
    idx, row = args
    pair_id  = row["pair_id"]
    wage     = float(row.get("wage_usd_hour", ECON_WAGE_USD_DEFAULT))

    result = {
        "pair_id": pair_id,
        "indep_annual": [], "indep_warm": [],
        "paired_pooled": [], "paired_loyo": [],
        "paired_warm": [],
        "events_pooled": [], "events_loyo": [],
        "robust_rows": [],
        "error": None
    }
    try:
        indep_annual, indep_warm = build_independent_output(
            pair_id, row, wage_usd_per_hour=wage)
        result["indep_annual"].extend(indep_annual)
        result["indep_warm"].extend(indep_warm)

        paired_annual, paired_warm = build_paired_output(
            pair_id, row, wage_usd_per_hour=wage)
        if not paired_annual:
            result["error"] = "城乡全年数据不足或无有效配对数据"
            return result

        pooled_panel = None
        loyo_panel   = None
        for panel in paired_annual:
            method = panel["hne_method"].iloc[0]
            if method == "pooled":
                pooled_panel = panel
                result["paired_pooled"].append(panel)
            elif method == "loyo":
                loyo_panel = panel
                result["paired_loyo"].append(panel)
            ev_df = extract_hw_events_with_hne(panel)
            if len(ev_df) > 0:
                ev_df["pair_id"]    = pair_id
                ev_df["hne_method"] = method
                ev_df["lat_group"]  = row.get("lat_group", "Unknown")
                if method == "pooled":
                    result["events_pooled"].append(ev_df)
                else:
                    result["events_loyo"].append(ev_df)

        result["paired_warm"].extend(paired_warm)
        _, stats_d = compute_robustness_stats_paired(pooled_panel, loyo_panel)
        if stats_d:
            for var, s in stats_d.items():
                r_row = {"pair_id": pair_id,
                         "lat_group": row.get("lat_group", "Unknown"),
                         "variable": var}
                r_row.update(s)
                result["robust_rows"].append(r_row)

    except ERA5MissingOrInvalid as e:
        result["error"] = "ERA5_missing_or_invalid"
        result["error_detail"] = str(e)

    except Exception as e:
        result["error"] = f"代码崩溃: {str(e)}"
        result["error_detail"] = str(e)

    return result

# ══════════════════════════════════════════════════════════════
# ★ 新增独立模块：导出图 1、2、3 所需的真实 Sleep Loss 数据
# ══════════════════════════════════════════════════════════════
def bootstrap_ci(values, n_boot=500, alpha=0.05, random_seed=42):
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)

    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(random_seed)

    boots = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])

    return (
        float(np.mean(arr)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def _load_canonical_pair_groups(path=CANONICAL_GROUP_CSV):
    """
    Load the manuscript-level UHI/UCI group definition from the main
    analysis output. Do not redefine UHI/UCI inside HNE/Figure exports.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Canonical UHI/UCI group file not found: {path}"
        )

    g = pd.read_csv(path)
    required = {"pair_id", "group"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(
            f"Canonical UHI/UCI group file missing columns: {sorted(missing)}"
        )

    if "hw_method" in g.columns:
        g = g[g["hw_method"].astype(str).str.lower().eq("percentile")].copy()

    if "period" in g.columns:
        annual = g[g["period"].astype(str).str.lower().eq("annual")].copy()
        if len(annual) > 0:
            g = annual

    g = g[["pair_id", "group"]].copy()
    g["pair_id"] = g["pair_id"].astype(str)
    g["group"] = g["group"].astype(str).str.upper().str.strip()
    g = g[g["group"].isin(["UHI", "UCI"])].copy()

    if len(g) == 0:
        raise ValueError("Canonical group table contains no valid UHI/UCI rows.")

    return (
        g.groupby("pair_id", as_index=False)["group"]
         .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    )


def export_fig123_data_sleep(all_paired_warm, out_dir):
    """
    Export formal sleep-loss data for downstream figures.

    The Minor et al. sleep model is a nightly-Tmin model. Therefore this
    function exports only nightly/period-level sleep-loss values and does not
    apply the model independently to each hourly temperature.

    Outputs
    -------
    fig2_vulnerability_sleep_real.csv
        Mean urban and rural positive sleep-loss amounts by group and period.

    fig3_mechanism_sleep_real.csv
        Pair-period urban-minus-rural positive sleep-loss burden.

    Sign convention
    ---------------
    Upstream sleep_loss_min:
        negative = reduced sleep duration

    Positive sleep-loss amount:
        -sleep_loss_min

    Positive urban-rural burden:
        (-sleep_loss_min_U) - (-sleep_loss_min_R)
    """
    if not all_paired_warm:
        print(
            "  [Fig2-3] No paired warm-season data found for "
            "sleep loss. Skipping."
        )
        return

    df = pd.concat(all_paired_warm, ignore_index=True)

    if "pair_id" not in df.columns:
        raise ValueError(
            "export_fig123_data_sleep requires a pair_id column."
        )

    # Retain dTMAX only as a diagnostic.
    # It must not redefine the canonical UHI/UCI classification.
    if "dTMAX" in df.columns:
        pair_diag = (
            df.groupby("pair_id", as_index=False)["dTMAX"]
            .mean()
            .rename(columns={"dTMAX": "dTMAX_pair_mean"})
        )
    else:
        pair_diag = pd.DataFrame({
            "pair_id": df["pair_id"].astype(str).unique(),
            "dTMAX_pair_mean": np.nan,
        })

    canonical_group = _load_canonical_pair_groups()

    # Do not retain an old local or daily group definition.
    if "group" in df.columns:
        df = df.drop(columns=["group"])

    df["pair_id"] = df["pair_id"].astype(str)
    pair_diag["pair_id"] = pair_diag["pair_id"].astype(str)

    df = df.merge(
        canonical_group,
        on="pair_id",
        how="left",
    )

    df = df.merge(
        pair_diag,
        on="pair_id",
        how="left",
    )

    missing_group = df["group"].isna()

    if missing_group.any():
        missing_pairs = sorted(
            df.loc[missing_group, "pair_id"]
            .astype(str)
            .unique()
        )[:20]

        raise ValueError(
            "Missing canonical UHI/UCI group for "
            "export_fig123_data_sleep; "
            f"examples={missing_pairs}"
        )

    # Formal paired warm-season periods.
    sub = df[
        df["period"].isin(
            ["heatwave", "non_heatwave"]
        )
    ].copy()

    if sub.empty:
        print(
            "  [Fig2-3] No heatwave/non-heatwave sleep rows found. "
            "Skipping."
        )
        return

    fig_data_dir = os.path.join(
        out_dir,
        "integrated_fig_data",
    )
    os.makedirs(fig_data_dir, exist_ok=True)

    # ==========================================================
    # Figure 2: urban/rural positive nightly sleep-loss amounts
    # ==========================================================
    required_fig2 = {
        "group",
        "period",
        "sleep_loss_min_U",
        "sleep_loss_min_R",
    }

    missing_fig2 = sorted(
        required_fig2 - set(sub.columns)
    )

    if missing_fig2:
        raise ValueError(
            "Cannot export Figure 2 sleep data; "
            f"missing columns={missing_fig2}"
        )

    fig2_source = sub.copy()

    fig2_source["sleep_loss_min_U_pos"] = (
        -pd.to_numeric(
            fig2_source["sleep_loss_min_U"],
            errors="coerce",
        )
    )

    fig2_source["sleep_loss_min_R_pos"] = (
        -pd.to_numeric(
            fig2_source["sleep_loss_min_R"],
            errors="coerce",
        )
    )

    fig2 = (
        fig2_source
        .groupby(
            ["group", "period"],
            as_index=False,
            observed=True,
        )[
            [
                "sleep_loss_min_U_pos",
                "sleep_loss_min_R_pos",
            ]
        ]
        .mean()
    )

    fig2 = fig2.rename(columns={
        "sleep_loss_min_U_pos": "sleep_loss_min_U",
        "sleep_loss_min_R_pos": "sleep_loss_min_R",
    })

    fig2["period"] = fig2["period"].replace({
        "heatwave": "HW",
        "non_heatwave": "NHW",
    })

    fig2_path = os.path.join(
        fig_data_dir,
        "fig2_vulnerability_sleep_real.csv",
    )

    fig2.to_csv(
        fig2_path,
        index=False,
    )

    # ==========================================================
    # Figure 3: pair-level urban-minus-rural sleep-loss burden
    # ==========================================================
    required_fig3 = {
        "pair_id",
        "group",
        "period",
        "dTMIN",
        "d_sleep_loss_min",
    }

    missing_fig3 = sorted(
        required_fig3 - set(sub.columns)
    )

    if missing_fig3:
        raise ValueError(
            "Cannot export Figure 3 sleep data; "
            f"missing columns={missing_fig3}"
        )

    fig3 = sub[
        [
            "pair_id",
            "group",
            "period",
            "dTMIN",
            "d_sleep_loss_min",
        ]
    ].copy()

    fig3["d_sleep_loss_min"] = -pd.to_numeric(
        fig3["d_sleep_loss_min"],
        errors="coerce",
    )

    fig3["period"] = fig3["period"].replace({
        "heatwave": "HW",
        "non_heatwave": "NHW",
    })

    fig3_path = os.path.join(
        fig_data_dir,
        "fig3_mechanism_sleep_real.csv",
    )

    fig3.to_csv(
        fig3_path,
        index=False,
    )

    print(
        "  [Fig2-3] Formal nightly sleep-loss data exported to:"
    )
    print(f"    {fig2_path}")
    print(f"    {fig3_path}")
    print(
        "  [Fig1] Hourly sleep-loss export disabled because the "
        "Minor model is based on nightly Tmin."
    )
# ══════════════════════════════════════════════════════════════
# Figure 4 export module: sleep / nighttime exposure pathway
# ══════════════════════════════════════════════════════════════

def _fig4_period_norm_from_period(x):
    mp = {
        "heatwave": "HW",
        "non_heatwave": "NHW",
        "warm_season": "JJA",
        "annual": "annual",
        "HW": "HW",
        "NHW": "NHW",
        "JJA": "JJA",
    }
    return mp.get(str(x), str(x))


def _fig4_safe_nanmax(arr):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if valid.any():
        for i in range(arr.shape[0]):
            if valid[i].any():
                out[i] = np.nanmax(arr[i])
    return out


def _fig4_safe_nanmin(arr):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if valid.any():
        for i in range(arr.shape[0]):
            if valid[i].any():
                out[i] = np.nanmin(arr[i])
    return out


def _fig4_exposure_from_hourly_cols(df, suffix, hours, threshold):
    """
    Compute fixed-hour excess heat exposure from temp_hXX_U/R columns.

    Returns:
        exposure : sum(max(T - threshold, 0)) over selected hours
        n_valid  : number of valid hourly temperatures
    """
    cols = [f"temp_h{h:02d}_{suffix}" for h in hours]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        exposure = np.full(len(df), np.nan, dtype=float)
        n_valid = np.zeros(len(df), dtype=int)
        return exposure, n_valid

    arr = df[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    valid = np.isfinite(arr)

    excess = np.where(valid, np.maximum(arr - threshold, 0.0), np.nan)
    exposure = np.nansum(excess, axis=1)
    exposure[valid.sum(axis=1) == 0] = np.nan

    return exposure.astype(float), valid.sum(axis=1).astype(int)


def _load_fig4_canonical_pair_period_damp1(
    path=CANONICAL_GROUP_CSV,
):
    """
    Load the canonical period-specific dAmp1 from
    01_main_pair_period_metrics.py output.

    Canonical definition
    --------------------
    dAmp1 = Amp1_urban - Amp1_rural

    Amp1 is the first-harmonic amplitude of the two-harmonic reconstructed
    24-hour temperature cycle.

    Only percentile HW/NHW rows are retained. No Tmax-Tmin fallback is
    allowed in the HNE/sleep module.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Fig4 sleep] Canonical 01 output not found: {path}"
        )

    src = pd.read_csv(path)

    required = {
        "pair_id",
        "period",
        "dAmp1",
    }
    missing = required - set(src.columns)

    if missing:
        raise ValueError(
            "[Fig4 sleep] Canonical 01 output is missing columns: "
            f"{sorted(missing)}"
        )

    # Retain only the formal percentile HW definition.
    if "hw_method" in src.columns:
        src = src[
            src["hw_method"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("percentile")
        ].copy()

    src["pair_id"] = (
        src["pair_id"]
        .astype(str)
        .str.strip()
    )

    src["period_norm"] = (
        src["period"]
        .apply(_fig4_period_norm_from_period)
    )

    src = src[
        src["period_norm"].isin(
            ["HW", "NHW"]
        )
    ].copy()

    src["dAmp1"] = pd.to_numeric(
        src["dAmp1"],
        errors="coerce",
    )

    # The same pair-period must not contain conflicting canonical values.
    conflict_counts = (
        src.dropna(subset=["dAmp1"])
        .groupby(
            ["pair_id", "period_norm"],
            observed=True,
        )["dAmp1"]
        .nunique()
    )

    conflicts = conflict_counts[
        conflict_counts > 1
    ]

    if len(conflicts) > 0:
        examples = [
            f"{pair_id}/{period_norm}"
            for pair_id, period_norm in conflicts.index[:20]
        ]

        raise ValueError(
            "[Fig4 sleep] Conflicting canonical dAmp1 values in "
            "the 01 output. "
            f"Examples: {examples}"
        )

    def _first_finite(series):
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        return (
            float(values.iloc[0])
            if len(values)
            else np.nan
        )

    out = (
        src.groupby(
            ["pair_id", "period_norm"],
            as_index=False,
            observed=True,
        )["dAmp1"]
        .agg(_first_finite)
    )

    if out.empty:
        raise ValueError(
            "[Fig4 sleep] No percentile HW/NHW dAmp1 values "
            "were found in the canonical 01 output."
        )

    return out

def _fig4_assign_pair_group(df):
    """
    Assign pair-level UHI/UCI group using the canonical manuscript-level
    grouping from all_pair_period_metrics.csv.

    dTMAX_pair_mean is kept only as a diagnostic. It must not redefine
    UHI/UCI group inside Figure-4 HNE exports.
    """
    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)

    if "dTMAX" in out.columns:
        pair_diag = (
            out.groupby("pair_id", as_index=False)["dTMAX"]
            .mean()
            .rename(columns={"dTMAX": "dTMAX_pair_mean"})
        )
    else:
        pair_diag = pd.DataFrame({
            "pair_id": out["pair_id"].unique(),
            "dTMAX_pair_mean": np.nan,
        })

    canonical_group = _load_canonical_pair_groups()

    if "group" in out.columns:
        out = out.drop(columns=["group"])

    out = out.merge(canonical_group, on="pair_id", how="left")
    out = out.merge(pair_diag, on="pair_id", how="left")

    missing_group = out["group"].isna()
    if missing_group.any():
        missing_pairs = sorted(out.loc[missing_group, "pair_id"].unique())[:20]
        raise ValueError(
            "Missing canonical UHI/UCI group for Figure-4 HNE export; "
            f"examples={missing_pairs}"
        )

    return out

def _fig4_build_sleep_pair_period_table(
    all_paired_warm,
    primary_hne_method=FIG4_PRIMARY_HNE_METHOD,
    theta_night=FIG4_THETA_NIGHT_TEMP,
    theta_day=FIG4_THETA_DAY_TEMP,
):
    """
    Build the Figure 4 sleep pair-period table.

    Output level
    ------------
    One row per pair_id x period_norm, where period_norm is HW or NHW.

    Canonical mechanism variable
    ----------------------------
    dAmp1 = Amp1_urban - Amp1_rural

    dAmp1 is read directly from the percentile pair-period output of
    01_main_pair_period_metrics.py. It is not reconstructed from daily
    Tmax-Tmin ranges in this module.

    Other definitions
    -----------------
    night_heat_exposure:
        Fixed-hour dry-bulb exposure contrast:

        sum_night[max(T_urban - theta_night, 0)]
        - sum_night[max(T_rural - theta_night, 0)]

    night_hne_exposure:
        Original solar-night HNE contrast:

        hne_d_urban - hne_d_rural = dHNE

    sleep_heat_burden:
        Urban minus rural sleep-loss amount.

        Positive means the urban station has a larger sleep loss than
        the paired rural station.
    """
    if not all_paired_warm:
        raise ValueError(
            "[Fig4 sleep] all_paired_warm is empty."
        )

    df = pd.concat(
        all_paired_warm,
        ignore_index=True,
    ).copy()

    if "hne_method" in df.columns:
        df = df[
            df["hne_method"] == primary_hne_method
        ].copy()

    df = df[
        df["period"].isin(
            ["heatwave", "non_heatwave"]
        )
    ].copy()

    if len(df) == 0:
        raise ValueError(
            "[Fig4 sleep] No heatwave/non_heatwave rows found "
            f"for hne_method={primary_hne_method}."
        )

    if "pair_id" not in df.columns:
        raise ValueError(
            "[Fig4 sleep] Missing required column: pair_id"
        )

    df["pair_id"] = (
        df["pair_id"]
        .astype(str)
        .str.strip()
    )

    df["period_norm"] = (
        df["period"]
        .apply(_fig4_period_norm_from_period)
    )

    df = _fig4_assign_pair_group(df)

    # ──────────────────────────────────────────────────────────
    # Fixed-hour dry-bulb exposure: night and day
    # ──────────────────────────────────────────────────────────
    nhe_u, n_night_u = _fig4_exposure_from_hourly_cols(
        df,
        "U",
        FIG4_NIGHT_HOURS,
        theta_night,
    )

    nhe_r, n_night_r = _fig4_exposure_from_hourly_cols(
        df,
        "R",
        FIG4_NIGHT_HOURS,
        theta_night,
    )

    dhe_u, n_day_u = _fig4_exposure_from_hourly_cols(
        df,
        "U",
        FIG4_DAY_HOURS,
        theta_day,
    )

    dhe_r, n_day_r = _fig4_exposure_from_hourly_cols(
        df,
        "R",
        FIG4_DAY_HOURS,
        theta_day,
    )

    df["night_heat_exposure_urban"] = nhe_u
    df["night_heat_exposure_rural"] = nhe_r

    df["night_heat_exposure"] = (
        df["night_heat_exposure_urban"]
        - df["night_heat_exposure_rural"]
    )

    df["day_heat_exposure_urban"] = dhe_u
    df["day_heat_exposure_rural"] = dhe_r

    df["day_heat_exposure"] = (
        df["day_heat_exposure_urban"]
        - df["day_heat_exposure_rural"]
    )

    df["n_fig4_night_hours_U"] = n_night_u
    df["n_fig4_night_hours_R"] = n_night_r
    df["n_fig4_day_hours_U"] = n_day_u
    df["n_fig4_day_hours_R"] = n_day_r

    # ──────────────────────────────────────────────────────────
    # HNE exposure from the original solar-night model
    # ──────────────────────────────────────────────────────────
    if "hne_d_U" in df.columns:
        df["night_hne_exposure_urban"] = pd.to_numeric(
            df["hne_d_U"],
            errors="coerce",
        )
    else:
        df["night_hne_exposure_urban"] = np.nan

    if "hne_d_R" in df.columns:
        df["night_hne_exposure_rural"] = pd.to_numeric(
            df["hne_d_R"],
            errors="coerce",
        )
    else:
        df["night_hne_exposure_rural"] = np.nan

    if "dHNE" in df.columns:
        df["night_hne_exposure"] = pd.to_numeric(
            df["dHNE"],
            errors="coerce",
        )
    else:
        df["night_hne_exposure"] = (
            df["night_hne_exposure_urban"]
            - df["night_hne_exposure_rural"]
        )

    # ──────────────────────────────────────────────────────────
    # Sleep-loss direction cleanup
    # ──────────────────────────────────────────────────────────
    # Original sleep_loss_min:
    # negative = reduction in sleep duration.
    #
    # Converted sleep_loss_amount:
    # positive = minutes of sleep lost.
    if "sleep_loss_min_U" in df.columns:
        df["sleep_loss_amount_U"] = -pd.to_numeric(
            df["sleep_loss_min_U"],
            errors="coerce",
        )
    else:
        df["sleep_loss_amount_U"] = np.nan

    if "sleep_loss_min_R" in df.columns:
        df["sleep_loss_amount_R"] = -pd.to_numeric(
            df["sleep_loss_min_R"],
            errors="coerce",
        )
    else:
        df["sleep_loss_amount_R"] = np.nan

    if "d_sleep_loss_min" in df.columns:
        df["sleep_heat_burden"] = -pd.to_numeric(
            df["d_sleep_loss_min"],
            errors="coerce",
        )
    else:
        df["sleep_heat_burden"] = (
            df["sleep_loss_amount_U"]
            - df["sleep_loss_amount_R"]
        )

    # Positive sleep_heat_burden:
    # urban sleep-loss amount > rural sleep-loss amount.

    # ──────────────────────────────────────────────────────────
    # Optional sleep-related economic endpoints
    # ──────────────────────────────────────────────────────────
    if "d_econ_loss_pct_short" in df.columns:
        df["sleep_econ_loss_pct_short"] = pd.to_numeric(
            df["d_econ_loss_pct_short"],
            errors="coerce",
        )
    else:
        df["sleep_econ_loss_pct_short"] = np.nan

    if "d_econ_loss_usd_sleep" in df.columns:
        df["sleep_econ_loss_usd"] = pd.to_numeric(
            df["d_econ_loss_usd_sleep"],
            errors="coerce",
        )
    else:
        df["sleep_econ_loss_usd"] = np.nan

    # ──────────────────────────────────────────────────────────
    # Aggregate HNE/sleep daily variables to pair-period level
    # ──────────────────────────────────────────────────────────
    # dAmp1 is deliberately not included here.
    #
    # It is a period-level FFT metric supplied by analysis 01 and
    # is merged after this aggregation.
    meta_cols = [
        "group",
        "dTMAX_pair_mean",
        "kg_group",
        "kg_code",
        "climate_zone_main",
        "lat_group",
        "continent",
        "lat_urban",
        "lon_urban",
        "hne_method",
        "output_type",
    ]

    meta_cols = [
        c for c in meta_cols
        if c in df.columns
    ]

    value_cols = [
        "night_heat_exposure_urban",
        "night_heat_exposure_rural",
        "night_heat_exposure",
        "day_heat_exposure_urban",
        "day_heat_exposure_rural",
        "day_heat_exposure",
        "night_hne_exposure_urban",
        "night_hne_exposure_rural",
        "night_hne_exposure",
        "sleep_loss_min_U",
        "sleep_loss_min_R",
        "sleep_loss_amount_U",
        "sleep_loss_amount_R",
        "sleep_heat_burden",
        "sleep_econ_loss_pct_short",
        "sleep_econ_loss_usd",
        "dTMIN",
        "dTMAX",
        "AsymIndex",
        "tmin_night_U",
        "tmin_night_R",
        "tmax_day_U",
        "tmax_day_R",
        "n_fig4_night_hours_U",
        "n_fig4_night_hours_R",
        "n_fig4_day_hours_U",
        "n_fig4_day_hours_R",
    ]

    value_cols = [
        c for c in value_cols
        if c in df.columns
    ]

    agg_dict = {
        c: "mean"
        for c in value_cols
    }

    for c in meta_cols:
        agg_dict[c] = "first"

    if "date" in df.columns:
        agg_dict["date"] = "nunique"

    out = (
        df.groupby(
            ["pair_id", "period_norm"],
            as_index=False,
            dropna=False,
        )
        .agg(agg_dict)
    )

    if "date" in out.columns:
        out = out.rename(
            columns={
                "date": "n_days",
            }
        )

    # ──────────────────────────────────────────────────────────
    # Merge canonical dAmp1 from 01
    # ──────────────────────────────────────────────────────────
    canonical_amp = (
        _load_fig4_canonical_pair_period_damp1()
    )

    n_rows_before_merge = len(out)

    # Remove any accidental legacy/local field before authoritative merge.
    out = out.drop(
        columns=["dAmp1"],
        errors="ignore",
    )

    out = out.merge(
        canonical_amp,
        on=[
            "pair_id",
            "period_norm",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(out) != n_rows_before_merge:
        raise RuntimeError(
            "[Fig4 sleep] Canonical dAmp1 merge changed "
            "the number of pair-period rows."
        )

    n_missing_damp1 = int(
        out["dAmp1"].isna().sum()
    )

    if n_missing_damp1 > 0:
        print(
            "  [Fig4 sleep WARNING] "
            f"{n_missing_damp1} pair-period rows have no "
            "canonical dAmp1 in the 01 output. "
            "These values remain NaN; no Tmax-Tmin "
            "fallback is applied."
        )

    out["dAmp1_source"] = (
        "01_main_pair_period_metrics.py:"
        "urban_first_harmonic_amplitude_minus_rural"
    )

    out["theta_night_temp"] = theta_night
    out["theta_day_temp"] = theta_day

    out["fig4_night_hours"] = ",".join(
        str(h)
        for h in FIG4_NIGHT_HOURS
    )

    out["fig4_day_hours"] = ",".join(
        str(h)
        for h in FIG4_DAY_HOURS
    )

    out["primary_hne_method"] = (
        primary_hne_method
    )

    front_cols = [
        "pair_id",
        "period_norm",
        "group",
        "dAmp1",
        "dAmp1_source",
        "night_heat_exposure",
        "day_heat_exposure",
        "night_hne_exposure",
        "sleep_heat_burden",
        "sleep_loss_amount_U",
        "sleep_loss_amount_R",
        "dTMIN",
        "dTMAX",
        "AsymIndex",
        "n_days",
        "theta_night_temp",
        "theta_day_temp",
        "primary_hne_method",
    ]

    front_cols = [
        c for c in front_cols
        if c in out.columns
    ]

    other_cols = [
        c for c in out.columns
        if c not in front_cols
    ]

    return out[
        front_cols + other_cols
    ]

def _fig4_make_sleep_paired_diffs(pair_period_df):
    """
    Convert pair-period sleep table into HW − NHW paired differences.

    Sign conventions:
        ddAmp_sleep       = dAmp1_HW − dAmp1_NHW
        ampDamping_sleep  = −ddAmp_sleep

        dNightHeat_sleep  = night_heat_exposure_HW − night_heat_exposure_NHW
        dDayHeat_sleep    = day_heat_exposure_HW − day_heat_exposure_NHW
        dayRelief_sleep   = −dDayHeat_sleep

        dSleepBurden      = sleep_heat_burden_HW − sleep_heat_burden_NHW

    Positive dSleepBurden means:
        HW increases urban-relative sleep loss compared with NHW.
    """
    for c in ["pair_id", "period_norm"]:
        if c not in pair_period_df.columns:
            raise ValueError(f"[Fig4 sleep] Missing required column: {c}")

    hw = pair_period_df[pair_period_df["period_norm"] == "HW"].copy()
    nhw = pair_period_df[pair_period_df["period_norm"] == "NHW"].copy()

    if len(hw) == 0 or len(nhw) == 0:
        raise ValueError("[Fig4 sleep] HW or NHW rows missing.")

    merged = hw.merge(
        nhw,
        on="pair_id",
        how="inner",
        suffixes=("_HW", "_NHW")
    )

    if len(merged) == 0:
        raise ValueError("[Fig4 sleep] No matched HW × NHW pair rows.")

    out = pd.DataFrame()
    out["pair_id"] = merged["pair_id"]

    for c in [
        "group",
        "dTMAX_pair_mean",
        "dAmp1_source",
        "kg_group",
        "kg_code",
        "climate_zone_main",
        "lat_group",
        "continent",
        "lat_urban",
        "lon_urban",
    ]:
        c_hw = f"{c}_HW"
        if c_hw in merged.columns:
            out[c] = merged[c_hw]

    def diff(base_col):
        a = f"{base_col}_HW"
        b = f"{base_col}_NHW"
        if a not in merged.columns or b not in merged.columns:
            return np.full(len(merged), np.nan)
        return (
            pd.to_numeric(merged[a], errors="coerce").values.astype(float)
            - pd.to_numeric(merged[b], errors="coerce").values.astype(float)
        )

    # Mechanism.
    out["ddAmp_sleep"] = diff("dAmp1")
    out["ampDamping_sleep"] = -out["ddAmp_sleep"]

    # Exposure redistribution.
    out["dNightHeat_sleep"] = diff("night_heat_exposure")
    out["dDayHeat_sleep"] = diff("day_heat_exposure")
    out["dayRelief_sleep"] = -out["dDayHeat_sleep"]

    # Original solar-night HNE exposure.
    out["dNightHNE_sleep"] = diff("night_hne_exposure")

    # Sleep burden.
    out["dSleepBurden"] = diff("sleep_heat_burden")
    out["dSleepLossAmount_U"] = diff("sleep_loss_amount_U")
    out["dSleepLossAmount_R"] = diff("sleep_loss_amount_R")

    # Temperature diagnostics.
    out["dTMIN_sleep"] = diff("dTMIN")
    out["dTMAX_sleep"] = diff("dTMAX")
    out["dAsymIndex_sleep"] = diff("AsymIndex")

    # Optional economic endpoint.
    out["dSleepEconLossPct_short"] = diff("sleep_econ_loss_pct_short")
    out["dSleepEconLossUSD"] = diff("sleep_econ_loss_usd")

    # QC.
    out["valid_fig4_sleep"] = (
        np.isfinite(out["dNightHeat_sleep"])
        & np.isfinite(out["dSleepBurden"])
    )

    if "n_days_HW" in merged.columns:
        out["n_days_HW"] = merged["n_days_HW"]
    if "n_days_NHW" in merged.columns:
        out["n_days_NHW"] = merged["n_days_NHW"]

    # Amplitude-damping regime based on canonical 01 dAmp1.
    valid_amp = out["ampDamping_sleep"].dropna().values
    if len(valid_amp) >= 8:
        q33 = float(np.percentile(valid_amp, 33.3))
        q67 = float(np.percentile(valid_amp, 66.7))

        def _regime(v):
            if pd.isna(v):
                return "Weak change"
            if v > q67:
                return "Strong damping"
            if v < q33:
                return "Amplitude increase"
            return "Weak change"

        out["regime_sleep"] = out["ampDamping_sleep"].apply(_regime)
        out["ampDamping_q33_sleep"] = q33
        out["ampDamping_q67_sleep"] = q67
    else:
        out["regime_sleep"] = "Weak change"
        out["ampDamping_q33_sleep"] = np.nan
        out["ampDamping_q67_sleep"] = np.nan

    return out

def _fig4_write_sleep_schema(out_dir):
    schema_path = os.path.join(
        out_dir,
        "fig4_sleep_schema.txt",
    )

    lines = [
        "Figure 4 sleep data export",
        "=" * 72,
        "",
        "Files:",
        "  fig4_sleep_pair_period.csv",
        "  fig4_sleep_paired_diffs.csv",
        "",
        "pair-period level:",
        "  one row = pair_id x period_norm",
        "  period_norm = HW or NHW",
        "",
        "paired-difference level:",
        "  one row = pair_id",
        "  all paired differences are HW minus NHW unless stated otherwise",
        "",
        "Core definitions:",
        "",
        "  dAmp1 = Amp1_U - Amp1_R",
        "      Amp1 is the first-harmonic amplitude of the two-harmonic",
        "      reconstructed 24-hour temperature cycle.",
        "      Source: 01_main_pair_period_metrics.py percentile HW/NHW rows.",
        "      No daily Tmax-Tmin fallback is used.",
        "      unit: degC",
        "",
        "  night_heat_exposure = sum over fixed local night hours of",
        "      max(T_U - theta_night_temp, 0) - max(T_R - theta_night_temp, 0)",
        "      unit: degC h day-1",
        "",
        "  day_heat_exposure = sum over fixed local daytime hours of",
        "      max(T_U - theta_day_temp, 0) - max(T_R - theta_day_temp, 0)",
        "      unit: degC h day-1",
        "",
        "  night_hne_exposure = dHNE = hne_d_U - hne_d_R",
        "      original solar-night HNE exposure contrast from this script",
        "",
        "  sleep_loss_min_U/R:",
        "      original model output; negative means sleep is reduced",
        "",
        "  sleep_loss_amount_U/R = -sleep_loss_min_U/R",
        "      positive means minutes of sleep lost",
        "",
        "  sleep_heat_burden = sleep_loss_amount_U - sleep_loss_amount_R",
        "      positive means urban has larger sleep loss than rural",
        "",
        "  dSleepBurden = sleep_heat_burden_HW - sleep_heat_burden_NHW",
        "      positive means heatwave increases urban-relative sleep loss",
        "",
        "  ddAmp_sleep = dAmp1_HW - dAmp1_NHW",
        "  ampDamping_sleep = -ddAmp_sleep",
        "  dNightHeat_sleep = night_heat_exposure_HW - night_heat_exposure_NHW",
        "  dDayHeat_sleep = day_heat_exposure_HW - day_heat_exposure_NHW",
        "  dayRelief_sleep = -dDayHeat_sleep",
        "",
        "Settings:",
        f"  primary_hne_method = {FIG4_PRIMARY_HNE_METHOD}",
        f"  theta_night_temp = {FIG4_THETA_NIGHT_TEMP}",
        f"  theta_day_temp = {FIG4_THETA_DAY_TEMP}",
        f"  night hours = {FIG4_NIGHT_HOURS}",
        f"  day hours = {FIG4_DAY_HOURS}",
        "",
    ]

    with open(
        schema_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
        )

    return schema_path

def export_figure4_sleep_data(
    all_paired_warm,
    output_dir=FIG4_DATA_DIR,
    primary_hne_method=FIG4_PRIMARY_HNE_METHOD,
):
    """
    Independent export function for Figure 4 sleep pathway data.

    This function writes Figure 4-ready datasets to:
        <UNIFIED_ROOT>/shared/fig4_data

    It does not modify the existing Fig1-3 exports or economic-loss outputs.
    """
    ensure_dir(output_dir)

    print("\n" + "═" * 72)
    print("  [Figure 4] Exporting sleep / nighttime exposure pathway data")
    print(f"  Output directory: {output_dir}")
    print(f"  Primary HNE method: {primary_hne_method}")
    print("═" * 72)

    pair_period = _fig4_build_sleep_pair_period_table(
        all_paired_warm=all_paired_warm,
        primary_hne_method=primary_hne_method,
        theta_night=FIG4_THETA_NIGHT_TEMP,
        theta_day=FIG4_THETA_DAY_TEMP,
    )

    pair_path = os.path.join(output_dir, "fig4_sleep_pair_period.csv")
    pair_period.to_csv(pair_path, index=False)
    print(f"  Saved: {pair_path}")
    print(f"    rows = {len(pair_period)}")
    print(f"    HW rows = {(pair_period['period_norm'] == 'HW').sum()}")
    print(f"    NHW rows = {(pair_period['period_norm'] == 'NHW').sum()}")

    paired_diffs = _fig4_make_sleep_paired_diffs(pair_period)

    diff_path = os.path.join(output_dir, "fig4_sleep_paired_diffs.csv")
    paired_diffs.to_csv(diff_path, index=False)
    print(f"  Saved: {diff_path}")
    print(f"    paired rows = {len(paired_diffs)}")
    print(f"    valid_fig4_sleep = {paired_diffs['valid_fig4_sleep'].sum()}")

    schema_path = _fig4_write_sleep_schema(output_dir)
    print(f"  Saved: {schema_path}")

    # Console diagnostics.
    if "dNightHeat_sleep" in paired_diffs.columns:
        vals = paired_diffs["dNightHeat_sleep"].dropna()
        if len(vals) > 0:
            print(
                "  dNightHeat_sleep summary "
                f"(HW - NHW): mean={vals.mean():+.3f}, "
                f"median={vals.median():+.3f}, n={len(vals)}"
            )

    if "dSleepBurden" in paired_diffs.columns:
        vals = paired_diffs["dSleepBurden"].dropna()
        if len(vals) > 0:
            print(
                "  dSleepBurden summary "
                f"(HW - NHW, urban-relative extra sleep loss): "
                f"mean={vals.mean():+.3f} min/night, "
                f"median={vals.median():+.3f} min/night, n={len(vals)}"
            )

    if "ampDamping_sleep" in paired_diffs.columns:
        vals = paired_diffs["ampDamping_sleep"].dropna()
        if len(vals) > 0:
            print(
                "  ampDamping_sleep summary: "
                f"mean={vals.mean():+.3f} °C, "
                f"median={vals.median():+.3f} °C, n={len(vals)}"
            )

    print("  [Figure 4] Sleep export complete.\n")

    return {
        "pair_period_path": pair_path,
        "paired_diffs_path": diff_path,
        "schema_path": schema_path,
        "pair_period": pair_period,
        "paired_diffs": paired_diffs,
    }


def main():
    import time
    t_start = time.time()

    print(f"\n  [CONFIG] USE_ERA5_CLIMATOLOGY = {USE_ERA5_CLIMATOLOGY}")
    print(f"  [CONFIG] ERA5_STATION_DIR     = {ERA5_STATION_DIR}")

    # ── 目录结构 ──────────────────────────────────────────────
    dir_indep         = os.path.join(OUTPUT_DIR, "independent")
    dir_indep_annual  = os.path.join(dir_indep, "annual", "daily_panels")
    dir_indep_warm    = os.path.join(dir_indep, "warm_season")
    dir_paired        = os.path.join(OUTPUT_DIR, "paired")
    dir_paired_pooled = os.path.join(dir_paired, "method_pooled")
    dir_paired_loyo   = os.path.join(dir_paired, "method_loyo")
    dir_paired_warm_p = os.path.join(dir_paired_pooled, "warm_season_periods")
    dir_paired_warm_l = os.path.join(dir_paired_loyo,   "warm_season_periods")
    dir_events        = os.path.join(OUTPUT_DIR, "hw_events")
    dir_robust        = os.path.join(OUTPUT_DIR, "robustness")
    dir_econ          = os.path.join(OUTPUT_DIR, "economic_loss")

    for d in [dir_indep_annual, dir_indep_warm,
              os.path.join(dir_paired_pooled, "daily_panels"),
              os.path.join(dir_paired_loyo,   "daily_panels"),
              dir_paired_warm_p, dir_paired_warm_l,
              dir_events, dir_robust, dir_econ]:
        ensure_dir(d)

    pair_df = pd.read_csv(PAIR_CSV_PATH)
    if "lat_group" not in pair_df.columns:
        pair_df["lat_group"] = pair_df["lat_urban"].apply(lat_group)

    if os.path.exists(KG_TIF):
        print("  [KG] 正在提取 Köppen-Geiger 气候区（Beck et al. 2023）...")
        pair_df = add_climate_zone_columns(pair_df,
                                        lon_col="lon_urban",
                                        lat_col="lat_urban")
        print(f"  [KG] climate_zone_main 分布:\n"
            f"       {pair_df['climate_zone_main'].value_counts().to_dict()}")
    else:
        print(f"  [KG] ⚠ KG_TIF 不存在，跳过 Köppen 分区: {KG_TIF}")
        pair_df["kg_code"]           = np.nan
        pair_df["kg_group"]          = np.nan
        pair_df["climate_zone_main"] = np.nan
        
    # ── ★ 接入 05_station_features.csv：注入逐站工资和就业人口 ──
    if STATION_FEATURES_CSV and os.path.exists(STATION_FEATURES_CSV):
        sf = pd.read_csv(STATION_FEATURES_CSV, low_memory=False)
        sf_cols = ["pair_id"]
        for c in ["wage_usd_hour_final", "n_workers", "n_workers_source",
                  "country_iso3_urban", "wage_source", "wage_data_quality"]:
            if c in sf.columns:
                sf_cols.append(c)
        sf_merge = sf[sf_cols].drop_duplicates("pair_id")
        pair_df = pair_df.merge(sf_merge, on="pair_id", how="left")

        if "wage_usd_hour_final" in pair_df.columns:
            pair_df["wage_usd_hour"] = (
                pair_df["wage_usd_hour_final"].fillna(ECON_WAGE_USD_DEFAULT)
            )
            n_wage = pair_df["wage_usd_hour"].gt(0).sum()
            print(f"  [LABOUR] 接入 station_features：{n_wage}/{len(pair_df)} 对有工资数据")
            print(f"  [LABOUR] 工资范围: "
                  f"${pair_df['wage_usd_hour'].min():.1f}–"
                  f"${pair_df['wage_usd_hour'].max():.1f} /h")
        else:
            pair_df["wage_usd_hour"] = ECON_WAGE_USD_DEFAULT
            print("  [LABOUR] ⚠ station_features 无 wage_usd_hour_final 列，使用全局默认值")
    else:
        pair_df["wage_usd_hour"] = ECON_WAGE_USD_DEFAULT
        print(f"  [LABOUR] ⚠ 未找到 station_features（{STATION_FEATURES_CSV}），"
              f"工资列使用全局默认值 {ECON_WAGE_USD_DEFAULT}")

    all_indep_annual  = []
    all_indep_warm    = []
    all_paired_pooled = []
    all_paired_loyo   = []
    all_paired_warm   = []
    all_events_pooled = []
    all_events_loyo   = []
    robust_rows       = []
    error_records     = []
    processed = skipped = 0

    n_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"\n🚀 开始多进程加速处理，启用 {n_cores} 个 CPU 核心...\n")
    tasks = [(idx, row) for idx, row in pair_df.iterrows()]

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(process_single_pair, args): args for args in tasks}
        for future in as_completed(futures):
            idx, row = futures[future]
            pair_id = row["pair_id"]
            try:
                res = future.result()
                if res["error"]:
                    skipped += 1
                    error_records.append({
                        "pair_id": pair_id,
                        "fail_step": res["error"],
                        "missing_data": res.get("error_detail", res["error"])
                    })
                    continue
                processed += 1
                print(f"  [DONE] {pair_id}")

                all_indep_annual.extend(res["indep_annual"])
                all_indep_warm.extend(res["indep_warm"])
                for p in res["indep_annual"]:
                    p.to_csv(os.path.join(
                        dir_indep_annual,
                        f"{pair_id}_{p['station'].iloc[0]}_{p['hne_method'].iloc[0]}_annual.csv"
                    ), index=False)

                all_paired_pooled.extend(res["paired_pooled"])
                all_paired_loyo.extend(res["paired_loyo"])
                for p in res["paired_pooled"]:
                    p.to_csv(os.path.join(
                        dir_paired_pooled, "daily_panels", f"{pair_id}_annual.csv"
                    ), index=False)
                for p in res["paired_loyo"]:
                    p.to_csv(os.path.join(
                        dir_paired_loyo, "daily_panels", f"{pair_id}_annual.csv"
                    ), index=False)

                all_paired_warm.extend(res["paired_warm"])
                all_events_pooled.extend(res["events_pooled"])
                all_events_loyo.extend(res["events_loyo"])
                robust_rows.extend(res["robust_rows"])

            except Exception as exc:
                skipped += 1
                error_records.append({"pair_id": pair_id,
                                      "fail_step": "Crash",
                                      "missing_data": f"致命错误: {exc}"})

    print(f"\n{'='*70}\n合并输出...")

    if error_records:
        pd.DataFrame(error_records).to_csv(
            os.path.join(OUTPUT_DIR, "skipped_pairs_log.csv"), index=False)

    if all_indep_annual:
        df = pd.concat(all_indep_annual, ignore_index=True)
        df.to_csv(os.path.join(dir_indep, "all_stations_annual.csv"), index=False)
        print(f"  [indep/annual]  {len(df):,} rows")

    if all_indep_warm:
        pd.concat(all_indep_warm, ignore_index=True).to_csv(
            os.path.join(dir_indep_warm, "all_stations_warm_periods.csv"), index=False)

    for panels, out_dir, label in [
        (all_paired_pooled, dir_paired_pooled, "pooled"),
        (all_paired_loyo,   dir_paired_loyo,   "loyo"),
    ]:
        if panels:
            df = pd.concat(panels, ignore_index=True)
            df.to_csv(os.path.join(out_dir, "all_pairs_annual.csv"), index=False)
            print(f"  [paired/{label}/annual]  {len(df):,} rows")

            econ_cols = [c for c in df.columns if any(
                k in c for k in ["econ_loss", "sleep_loss", "heat_loss", "total_loss"])]
            if econ_cols:
                econ_summary = (df.groupby(["pair_id", "lat_group"])[econ_cols]
                                  .agg(["mean", "std"])
                                  .reset_index())
                econ_summary.to_csv(
                    os.path.join(dir_econ, f"econ_loss_summary_{label}.csv"),
                    index=False)
                print(f"  [econ/{label}] 经济损失汇总已保存 ({len(econ_summary)} 对)")

    if all_paired_warm:
        df_w = pd.concat(all_paired_warm, ignore_index=True)
        for method in ["pooled", "loyo"]:
            sub_m = df_w[df_w["hne_method"] == method]
            if sub_m.empty: continue
            out_d = dir_paired_warm_p if method == "pooled" else dir_paired_warm_l
            sub_m.to_csv(os.path.join(out_d, "all_pairs_warm_periods.csv"), index=False)

    for ev_list, label in [(all_events_pooled, "pooled"), (all_events_loyo, "loyo")]:
        if ev_list:
            pd.concat(ev_list, ignore_index=True).to_csv(
                os.path.join(dir_events, f"hw_events_{label}.csv"), index=False)

    if robust_rows:
        pd.DataFrame(robust_rows).to_csv(
            os.path.join(dir_robust, "robustness_summary.csv"), index=False)

    # ── ★ 给 integrated figure 输出夜间 sleep-caused LC peak ──
    export_pair_period_sleep_lc_peak(
        all_paired_pooled=all_paired_pooled,
        all_paired_warm=all_paired_warm,
        out_path=os.path.join(OUTPUT_DIR, "pair_period_sleep_lc_peak.csv"),
        peak_source_col="d_econ_loss_pct_short"
    )

    # ── ★ 劳动力总量加总（sleep loss / direct heat / total 三张表）──
    if STATION_FEATURES_CSV and os.path.exists(STATION_FEATURES_CSV):
        labour_df = pd.read_csv(STATION_FEATURES_CSV, low_memory=False)
        # 标准化 country_iso3 列名
        if "country_iso3_urban" in labour_df.columns:
            labour_df = labour_df.rename(columns={"country_iso3_urban": "country_iso3"})
        # 保留 pair_id → country_iso3 映射，用于注入面板
        pair_iso = (pair_df[["pair_id", "country_iso3_urban"]]
                    .rename(columns={"country_iso3_urban": "country_iso3"})
                    .drop_duplicates("pair_id")
                    if "country_iso3_urban" in pair_df.columns
                    else pd.DataFrame(columns=["pair_id", "country_iso3"]))

        print("\n  [ECON] ★ 劳动力总量加总（sleep loss + direct heat + total）...")

        for panels, label in [(all_paired_pooled, "pooled"),
                               (all_paired_loyo,   "loyo")]:
            if not panels:
                continue
            df_all = pd.concat(panels, ignore_index=True)

            # 注入国家代码（若面板无 country_iso3 则从 pair_df 合并）
            if "country_iso3" not in df_all.columns and len(pair_iso) > 0:
                df_all = df_all.merge(pair_iso, on="pair_id", how="left")

            # ── 三类损失对应的城市站列名 ──
            loss_map = {
                "sleep_loss":  "econ_loss_usd_sleep_U",
                "direct_heat": "econ_loss_usd_heat_U",
                "total":       "total_loss_usd_U",
            }

            loss_results = {}
            for loss_name, col_u in loss_map.items():
                if col_u not in df_all.columns or df_all[col_u].isna().all():
                    print(f"  [ECON/{label}/{loss_name}] 列 {col_u} 不存在或全为 NaN，跳过")
                    continue
                try:
                    total = aggregate_total_economic_loss(
                        df_all.rename(columns={col_u: "econ_loss_usd"}),
                        labour_df
                    )
                    out_path = os.path.join(
                        dir_econ,
                        f"total_annual_loss_{loss_name}_{label}.csv"
                    )
                    total.to_csv(out_path, index=False)
                    global_bn = total["total_annual_loss_bn_usd"].sum()
                    print(f"  [ECON/{label}/{loss_name}]  "
                          f"全球合计: {global_bn:.3f} 十亿 USD/年  "
                          f"({len(total)} 国) → {os.path.basename(out_path)}")
                    loss_results[loss_name] = total.assign(loss_type=loss_name)
                except Exception as e:
                    print(f"  [ECON/{label}/{loss_name}] 加总失败：{e}")

            # ── 三类损失拼合宽表 ────────────────────────────────
            if loss_results:
                combined = pd.concat(loss_results.values(), ignore_index=True)
                combined.to_csv(
                    os.path.join(dir_econ, f"total_annual_loss_ALL_{label}.csv"),
                    index=False
                )
                print(f"  [ECON/{label}] 三类损失合并宽表 → total_annual_loss_ALL_{label}.csv")

                # 打印摘要
                sep = "─" * 60
                print(f"\n  {sep}")
                print(f"  国家级总量经济损失摘要（{label}）")
                print(f"  {sep}")
                for ln, t in loss_results.items():
                    print(f"  {ln:<12}  全球合计: "
                          f"{t['total_annual_loss_bn_usd'].sum():.3f} 十亿 USD/年")
                print(f"  {sep}\n")
    else:
        print("\n  [ECON] ⚠ STATION_FEATURES_CSV 未配置，跳过劳动力总量加总")

    export_fig123_data_sleep(all_paired_warm, OUTPUT_DIR)

    export_figure4_sleep_data(
        all_paired_warm=all_paired_warm,
        output_dir=FIG4_DATA_DIR,
        primary_hne_method=FIG4_PRIMARY_HNE_METHOD,
    )

    elapsed = time.time() - t_start
    print(f"\n完成: {processed} 对处理成功, {skipped} 对跳过, "
          f"耗时 {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    main()
