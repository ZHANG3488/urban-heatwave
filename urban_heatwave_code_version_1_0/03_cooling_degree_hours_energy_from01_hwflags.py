#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_cdh_hdh_energy.py (v_sync - 多进程极速优化版 + KG 气候区整合)
══════════════════════════════════════════════════════════════════════════════
Urban–Rural CDH / HDH / COP / Cooling Energy  ——  Hourly ISD → Daily Panel
══════════════════════════════════════════════════════════════════════════════

计算链
──────
  ISD-Lite .gz
    → QC + 插值（向量化）
    → 逐小时指标
        CDH_h    = max(Ta_h − T_COOL, 0)              [°C·h]
        HDH_h    = max(T_HEAT − Ta_h, 0)              [°C·h]
        COP_h    = a0 + a1·Ta_h + a2·Ta_h²            [−]
        E_comm_h = UA_comm · CDH_h / COP_h  (08–18 h) [W·h]
        E_resi_h = UA_resi · CDH_h / COP_h  (18–08 h) [W·h]
    → 逐日聚合（有效小时 ≥ MIN_VALID_HOURS 才保留）
    → 动态分组判定：基于公共日期的 dTmax 计算 UHI / UCI
    → 动态热期标注：使用与 analysis_multiyear 相同的 Tmax P90 ±7天窗口算法
    → 城乡差值：dCDH, dHDH, dCOP, dE_comm, dE_resi, dE_total
    → 逐小时廓线：各分层 × 各时期平均廓线（写入 hourly_profiles.csv）
    → 统计汇总：period_summary.csv

输出文件
────────
  all_pairs_daily_panel.csv          —— 全样本逐日面板（主分析表，含 KG 字段）
  hourly_profiles.csv                —— 分层 × 时期 × 小时 廓线均值
  hourly_profiles_by_kggroup.csv     —— 气候区 × 时期 × 小时 廓线均值【新增】
  period_summary.csv                 —— 分层 × 时期 聚合统计 + t 检验
  period_summary_by_kggroup.csv      —— KG × 时期 聚合统计 + t 检验【新增】
  pair_period_summary.csv            —— pair × 时期 聚合值（含 KG 字段）【更新】
  daily/  {pair_id}_daily.csv        —— 逐对站明细
  skipped_pairs_log.csv              —— 剔除站点及原因追踪
══════════════════════════════════════════════════════════════════════════════

变更记录 (v_sync → v_kg)
────────────────────────
  [M1] main()         : pair_df 读入后即附加 KG 气候区字段并写入 hw_ref_mode
  [M2] process_single : meta 字典写入 kg_code / kg_group / climate_zone_main
  [M3] daily panel    : 确认 all_pairs_daily_panel.csv 携带上述 KG 字段
  [M4] build_summary  : 新增 period_summary_by_kggroup.csv；
                        pair_period_summary.csv 含 KG 字段
  [M5] process_all    : 新增 hourly_profiles_by_kggroup.csv
  [M6] 热浪函数       : 保留 P90 ±7天窗口 + 连续≥3天，与主分析完全一致
══════════════════════════════════════════════════════════════════════════════
"""

import os
import gzip
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

from config import (
    ERA5_STATION_DIR,
    ISD_BASE_DIR as ISD_DIR,
    KG_TIF,
    PAIR_CSV_PATH as PAIR_CSV,
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
KG_GROUP_ORDER = KG_GROUPS_ALL

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
# §0  路径配置  ← 修改为本地实际路径
# ══════════════════════════════════════════════════════════════
# File-system paths are centralized in config.py.
OUTPUT_DIR = UNIFIED_ROOT + "/analysis/cdh_energy"
YEARS      = list(range(2015, 2025))

# ══════════════════════════════════════════════════════════════
# Figure 4 data export path
# ══════════════════════════════════════════════════════════════
FIG4_DATA_DIR = UNIFIED_ROOT + "/shared/fig4_data"

FIG4_CDH_EXPORT_PERIODS = ["heatwave", "non_heatwave"]


# ══════════════════════════════════════════════════════════════
# §1  物理参数
# ══════════════════════════════════════════════════════════════

# ── 度时阈值 ──────────────────────────────────────────────────
T_COOL = 26.0          # 制冷设定温度 [°C]
T_HEAT = 18.0          # 供暖设定温度 [°C]

# ── COP 二次模型  [Sailor & Vasireddy 2006] ───────────────────
COP_A0, COP_A1, COP_A2 = 5.80, -0.088, 2.9e-4
COP_MIN = 1.0

# ── 建筑 UA [W K⁻¹]  [Kolokotroni et al. 2012] ───────────────
UA = {
    ("COMM", "HRHD"): 65.0,
    ("COMM", "LRLD"): 40.0,
    ("COMM", "OTHER"):50.0,
    ("RESI", "HRHD"): 28.0,
    ("RESI", "LRLD"): 16.0,
    ("RESI", "OTHER"):20.0,
}

# ── 楼宇参考面积 [m²]  用于将 W·h → W·h/m² ─────────────────
BLDG_AREA_M2 = {
    ("COMM", "HRHD"): 1000.0,
    ("COMM", "LRLD"):  800.0,
    ("COMM", "OTHER"): 900.0,
    ("RESI", "HRHD"):  120.0,
    ("RESI", "LRLD"):  100.0,
    ("RESI", "OTHER"): 110.0,
}

# ── 运营时段 ──────────────────────────────────────────────────
COMM_HOURS  = list(range(9, 22))                        # 08–18 h
RESI_HOURS  = list(range(18, 24)) + list(range(0, 8))  # 18–08 h
DAY_HOURS   = list(range(8, 20))                        # 08:00–19:59, 12 h
NIGHT_HOURS = list(range(20, 24)) + list(range(0, 8))  # 20:00–07:59, 12 h
H24         = list(range(24))

# ── 质控参数 ──────────────────────────────────────────────────
MAX_CONSEC_NAN      = 2    # 连续缺失上限（超过则整日作废）
MAX_INTERP_GAP      = 1    # 线性插值最大间隙 [h]
MIN_OBS_SUPPORT_HOURS = 6     # 原始观测支持小时数下限（兼容3-hourly）
MIN_USABLE_HOURS      = 18    # 重采样+小缺口插值后的可用小时数下限
MIN_VALID_DAYS_FRAC = 0.8 # 每年有效天数比例下限

# ── 热浪参数 (P90 ±7天窗口，连续≥3天，与 analysis_multiyear 完全同步) ──
# [M6] 以下三个参数定义与主分析保持严格一致，不可单独修改
HW_PERCENTILE  = 90        # 90th percentile（基于 Tmax）
HW_MIN_DAYS    = 3         # 连续超阈天数下限
HW_WINDOW_HALF = 7         # 动态阈值滑动窗口半宽 ±7天（共15天）
HW_REF_MODE    = f"Tmax_P{HW_PERCENTILE}_window{HW_WINDOW_HALF*2+1}d_consec{HW_MIN_DAYS}d"

# Hemisphere-aware warm-season definition, synced with analysis_multiyear.py
#   NH: JJA = Jun-Jul-Aug
#   SH: DJF = Dec-Jan-Feb
NH_WARM_MONTHS = [6, 7, 8]
SH_WARM_MONTHS = [12, 1, 2]
WARM_MONTHS    = NH_WARM_MONTHS  # backward-compatible default; do not use for pair-level filtering

# analysis_multiyear.py writes this file. Downstream scripts read it first.
ANALYSIS_HW_FLAGS_CSV = (
    UNIFIED_ROOT
    + "/analysis/main_multiyear/robustness_percentile/daily_heatwave_flags.csv"
)

CANONICAL_GROUP_CSV = (
    UNIFIED_ROOT
    + "/analysis/main_multiyear/robustness_percentile/all_pair_period_metrics.csv"
)


# ── LCZ → 密度  [Stewart & Oke 2012] ─────────────────────────
LCZ_HRHD = {1}
LCZ_LRLD = {6}

STRATA_ORDER = ["UHI_HRHD", "UHI_LRLD", "UCI_HRHD", "UCI_LRLD"]
PERIODS      = ["annual", "warm_season", "heatwave", "non_heatwave"]

# ── ERA5-MOD：ERA5 气候态热浪阈值开关及工具常量 ─────────────────────
USE_ERA5_CLIMATOLOGY = True   # ← 默认开启；改为 False 还原为 ISD 短期估计

USE_QM_BIAS_CORRECTION = True
USE_MONTHLY_MEAN_BIAS_CORRECTION = False

QM_N_QUANTILES = 1001
QM_MIN_OVERLAP_DAYS_PER_MONTH = 30
QM_MIN_OVERLAP_DAYS_ANNUAL = 100

HW_REF_MODE = (
    f"ERA5_QM_Tmax_P{HW_PERCENTILE}_window{HW_WINDOW_HALF*2+1}d_"
    f"consec{HW_MIN_DAYS}d_monthly_QM"
)



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
    """Load daily_heatwave_flags.csv generated by analysis_multiyear.py."""
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
    """Return the pair-level daily HW flags from analysis_multiyear.py, if present."""
    df = _load_analysis_hw_flags()
    if df is None or len(df) == 0:
        return None

    sub = df[df["pair_id"].astype(str) == str(pair_id)].copy()
    if sub.empty:
        return None

    sub = sub.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return sub


def align_analysis_hw_flags(analysis_hw_flags, date_index):
    """Align external daily HW flags to a target date index."""
    if analysis_hw_flags is None or len(analysis_hw_flags) == 0:
        return None

    idx = pd.to_datetime(pd.Index(date_index), errors="coerce").normalize()
    ext = analysis_hw_flags.copy()
    ext.index = pd.to_datetime(ext.index, errors="coerce").normalize()
    aligned = ext.reindex(idx)

    if aligned[["is_warm_season", "hw_flag_percentile_warm_season"]].dropna(how="all").empty:
        return None

    return aligned



_CANONICAL_PAIR_GROUPS_CACHE = None


def _load_canonical_pair_groups(path=CANONICAL_GROUP_CSV):
    """
    Load the manuscript-level UHI/UCI group definition from the main analysis.
    CDH/energy is downstream and must not redefine UHI/UCI locally.
    """
    global _CANONICAL_PAIR_GROUPS_CACHE
    if _CANONICAL_PAIR_GROUPS_CACHE is not None:
        return _CANONICAL_PAIR_GROUPS_CACHE

    if not os.path.exists(path):
        raise FileNotFoundError(f"Canonical UHI/UCI group file not found: {path}")

    g = pd.read_csv(path)
    required = {"pair_id", "group"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"Canonical group file missing columns: {sorted(missing)}")

    if "hw_method" in g.columns:
        g = g[g["hw_method"].astype(str).str.lower().eq("percentile")].copy()
    if "period" in g.columns:
        annual = g[g["period"].astype(str).str.lower().eq("annual")].copy()
        if len(annual) > 0:
            g = annual

    keep_cols = ["pair_id", "group"]
    for optional in [
        "uhi_definition", "uhi_classification_metric", "delta_tx_annual_synth",
        "delta_tmax_daily_ann_mean", "urban_tx_annual_synth", "rural_tx_annual_synth",
    ]:
        if optional in g.columns:
            keep_cols.append(optional)

    g = g[keep_cols].copy()
    g["pair_id"] = g["pair_id"].astype(str)
    g["group"] = g["group"].astype(str).str.upper().str.strip()
    g = g[g["group"].isin(["UHI", "UCI"])].copy()
    if g.empty:
        raise ValueError("Canonical group table contains no valid UHI/UCI rows.")

    def first_non_null(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan

    agg = {c: first_non_null for c in keep_cols if c != "pair_id"}
    _CANONICAL_PAIR_GROUPS_CACHE = g.groupby("pair_id", as_index=False).agg(agg)
    return _CANONICAL_PAIR_GROUPS_CACHE


def get_canonical_pair_group(pair_id):
    g = _load_canonical_pair_groups()
    sub = g[g["pair_id"].astype(str) == str(pair_id)]
    if sub.empty:
        raise KeyError(f"Missing canonical UHI/UCI group for pair_id={pair_id}")
    return sub.iloc[0].to_dict()


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

        hw_part = detect_heatwave(s_part, thr_part)
        out.loc[hw_part.index] = hw_part.astype(bool)

    return out.fillna(False).astype(bool)

# ERA5 站点 Tmax CSV 命名约定：{USAF}-{WBAN}_{station_type}_tmax.csv
# 必须包含 "date"（YYYY-MM-DD）和 "tmax"（℃）两列

MIN_LOYO_REF_YEARS = 5


def era5_station_path(usaf: str, wban: str, station_type: str) -> str:
    return os.path.join(
        ERA5_STATION_DIR,
        f"{str(usaf).strip()}_{str(wban).strip()}_{station_type}.csv"
    )


def load_era5_tmax_series(
        usaf: str,
        wban: str,
        station_type: str = "urban") -> "pd.Series | None":
    """
    读取 ERA5 逐日 Tmax 序列。

    文件名：
        {USAF}_{WBAN}_{urban/rural}.csv

    列名：
        date 必须存在；
        Tmax 优先使用 tmax_c，兼容 tmax。

    返回：
        pd.Series(index=date, values=tmax)；
        若文件缺失、列名不对或数据不足，返回 None。
    """
    fpath = era5_station_path(usaf, wban, station_type)

    if not os.path.exists(fpath):
        return None

    try:
        df = pd.read_csv(fpath)

        if "date" not in df.columns:
            return None

        if "tmax_c" in df.columns:
            tmax_col = "tmax_c"
        elif "tmax" in df.columns:
            tmax_col = "tmax"
        else:
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[tmax_col] = pd.to_numeric(df[tmax_col], errors="coerce")

        df = (
            df.dropna(subset=["date", tmax_col])
              .drop_duplicates(subset=["date"])
              .sort_values("date")
        )

        if df.empty:
            return None

        s = df.set_index("date")[tmax_col].astype(float).dropna()

        # 最低有效数据量：至少 MIN_LOYO_REF_YEARS × 30 天
        if len(s) < MIN_LOYO_REF_YEARS * 30:
            return None

        return s

    except Exception:
        return None


def compute_hw_doy_thr_era5(era5_tmax: pd.Series) -> dict:
    """
    基于 ERA5 长序列 Tmax，计算每个 DOY 的热浪温度阈值。
    使用 HW_PERCENTILE / HW_WINDOW_HALF。
    """
    _q = HW_PERCENTILE
    _half = HW_WINDOW_HALF

    ref = era5_tmax.copy()
    ref.index = pd.to_datetime(ref.index)

    ref_df = pd.DataFrame({
        "date": ref.index,
        "tmax": ref.values
    })

    ref_df["doy"] = ref_df["date"].apply(
        lambda x: (
            pd.Timestamp(2001, x.month, x.day).dayofyear
            if not (x.month == 2 and x.day == 29) else 59
        )
    )

    ref_df = ref_df.dropna(subset=["tmax"])

    doy_thr = {}

    for doy in range(1, 366):
        low = doy - _half
        high = doy + _half

        if low < 1:
            mask = (ref_df["doy"] >= 365 + low) | (ref_df["doy"] <= high)
        elif high > 365:
            mask = (ref_df["doy"] >= low) | (ref_df["doy"] <= high - 365)
        else:
            mask = (ref_df["doy"] >= low) & (ref_df["doy"] <= high)

        vals = ref_df.loc[mask, "tmax"].dropna().values.astype(float)
        doy_thr[doy] = float(np.nanpercentile(vals, _q)) if len(vals) > 0 else np.nan

    return doy_thr


def apply_doy_thr_to_index(date_index, doy_thr: dict) -> pd.Series:
    """
    将 DOY 阈值映射到目标日期索引。
    """
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

    corrected = np.interp(
        values_arr,
        sim_q_unique,
        obs_q_unique,
        left=obs_q_unique[0],
        right=obs_q_unique[-1],
    )

    return corrected


def apply_quantile_mapping_bias_correction(
        hw_thr_series_raw,
        isd_tmax,
        era5_tmax,
        min_overlap_days_per_month=30,
        min_overlap_days_annual=100,
        n_quantiles=1001):
    """
    Quantile Mapping correction for ERA5-derived heatwave thresholds.

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
                values=values,
                sim_train=sub["era5"],
                obs_train=sub["isd"],
                n_quantiles=n_quantiles,
            )
            diagnostics["monthly_methods"][m] = "monthly_QM"

        elif has_annual_qm:
            corrected_values = quantile_mapping_values(
                values=values,
                sim_train=overlap["era5"],
                obs_train=overlap["isd"],
                n_quantiles=n_quantiles,
            )
            diagnostics["monthly_methods"][m] = "annual_QM_fallback"

        else:
            corrected_values = values.copy()
            diagnostics["monthly_methods"][m] = "no_correction_insufficient_samples"

        corrected.loc[mask] = corrected_values

    diagnostics["isd_tmax_mean_overlap"] = float(overlap["isd"].mean())
    diagnostics["era5_tmax_mean_overlap"] = float(overlap["era5"].mean())

    return corrected, diagnostics

# ── ERA5-MOD 工具函数结束 ────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# §2  物理模型
# ══════════════════════════════════════════════════════════════

def cop(Ta: np.ndarray) -> np.ndarray:
    return np.maximum(COP_A0 + COP_A1 * np.asarray(Ta, float)
                      + COP_A2 * np.asarray(Ta, float) ** 2, COP_MIN)

def cdh_h(Ta: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(Ta, float) - T_COOL, 0.0)

def hdh_h(Ta: np.ndarray) -> np.ndarray:
    return np.maximum(T_HEAT - np.asarray(Ta, float), 0.0)

def energy_h(Ta: np.ndarray, ua: float) -> np.ndarray:
    return ua * cdh_h(Ta) / cop(Ta)

def get_ua(density: str, schedule: str) -> float:
    return UA.get((schedule, density), UA[(schedule, "OTHER")])


# ══════════════════════════════════════════════════════════════
# §3  工具函数
# ══════════════════════════════════════════════════════════════

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def isd_path(usaf, wban, year):
    return os.path.join(ISD_DIR, str(year), f"{usaf}-{wban}-{year}.gz")

def parse_pair_id(pid):
    parts = pid.split("__")
    u = parts[0].strip().split("_")
    r = parts[1].strip().split("_")
    return u[0], u[1], r[0], r[1]

def is_leap(yr):
    return (yr % 4 == 0) and (yr % 100 != 0 or yr % 400 == 0)

_LCZ_REMAP = {51: 1, 52: 2, 53: 3, 54: 4, 55: 5, 56: 6}

def lcz_to_density(val):
    try:
        v = int(float(val))
        v = _LCZ_REMAP.get(v, v)
        if v in LCZ_HRHD: return "HRHD"
        if v in LCZ_LRLD: return "LRLD"
    except (ValueError, TypeError):
        pass
    return "OTHER"


# ══════════════════════════════════════════════════════════════
# §4  ISD 读取 + 质控 + 本地化
# ══════════════════════════════════════════════════════════════

def read_isd(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            usecols=[0, 1, 2, 3, 4],
            names=["year", "month", "day", "hour", "temp_C"],
            na_values={"temp_C": -9999},
            engine="c",
            on_bad_lines="skip"
        )
        df["temp_C"] = df["temp_C"] / 10.0
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]], errors="coerce"
        )
        df = (
            df.dropna(subset=["datetime"])
              .drop(columns=["year", "month", "day", "hour"])
              .drop_duplicates(subset="datetime")
              .sort_values("datetime")
              .reset_index(drop=True)
        )
        if df.empty:
            return None
        return df[["datetime", "temp_C"]]
    except Exception:
        return None


def _qc_interp(s, full_i, max_consec, max_interp):
    s = s[~s.index.duplicated(keep="first")].reindex(full_i)
    if s.notna().all():
        return s.interpolate("linear", limit=max_interp, limit_area="inside")
    is_nan   = s.isna()
    group_id = (~is_nan).cumsum()
    run_len  = is_nan.groupby(group_id).transform("sum").where(is_nan, 0)
    date_col = pd.Series(full_i.date, index=full_i)
    bad = set(run_len.groupby(date_col).max()[lambda x: x > max_consec].index)
    if bad:
        s = s.copy()
        s.iloc[date_col.isin(bad).values] = np.nan
    return s.interpolate("linear", limit=max_interp, limit_area="inside")


def load_station(usaf, wban, lon):
    frames = []

    for yr in YEARS:
        raw = read_isd(isd_path(usaf, wban, yr))
        if raw is None:
            continue

        full_1h_i = pd.date_range(
            pd.Timestamp(yr, 1, 1, 0),
            pd.Timestamp(yr, 12, 31, 23),
            freq="1h"
        )

        df_i = raw.set_index("datetime").sort_index()

        # 统一到 hourly，不依赖原生时间步长
        hourly = df_i[["temp_C"]].resample("1h").mean()
        hourly = hourly.reindex(full_1h_i)

        # 只对小缺口插值
        hourly["temp_C"] = hourly["temp_C"].interpolate(
            method="time",
            limit=2,              # 可按需调成 1 或 2
            limit_area="inside"
        )

        # 原始观测支持小时数（不是插值）
        obs_hourly = (
            df_i[["temp_C"]]
            .resample("1h")
            .count()
            .reindex(full_1h_i)
            .fillna(0)
        )

        obs_support_count = obs_hourly["temp_C"].groupby(obs_hourly.index.date).transform(
            lambda s: (s > 0).sum()
        )

        # 插值后的可用小时数
        usable_count = hourly["temp_C"].groupby(hourly.index.date).transform(
            lambda s: s.notna().sum()
        )

        bad_days = (
            (obs_support_count < MIN_OBS_SUPPORT_HOURS) |
            (usable_count < MIN_USABLE_HOURS)
        )

        hourly.loc[bad_days, "temp_C"] = np.nan


        offset = pd.Timedelta(hours=lon / 15.0)
        ldt    = full_1h_i + offset

        out = pd.DataFrame({
            "datetime":    full_1h_i,
            "local_dt":    ldt,
            "temp_C":      hourly["temp_C"].values,
            "local_date":  ldt.date,
            "local_hour":  ldt.hour,
            "local_month": ldt.month,
            "year":        ldt.year,
        })

        n_days = 366 if is_leap(yr) else 365
        v_days = out.dropna(subset=["temp_C"])["local_date"].nunique()
        if v_days / n_days < MIN_VALID_DAYS_FRAC:
            continue

        frames.append(out)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════
# §5  逐小时指标计算
# ══════════════════════════════════════════════════════════════

def compute_hourly(df_loc, density):
    df = df_loc.copy()
    Ta = df["temp_C"].values
    h  = df["local_hour"].values
    ua_c = get_ua(density, "COMM")
    ua_r = get_ua(density, "RESI")
    df["CDH_h"]     = cdh_h(Ta)
    df["HDH_h"]     = hdh_h(Ta)
    df["COP_h"]     = cop(Ta)
    df["is_comm"]   = np.isin(h, COMM_HOURS)
    df["is_resi"]   = np.isin(h, RESI_HOURS)
    df["is_day"]    = np.isin(h, DAY_HOURS)
    df["is_night"]  = np.isin(h, NIGHT_HOURS)
    df["E_comm_h"]  = np.where(df["is_comm"], energy_h(Ta, ua_c), 0.0)
    df["E_resi_h"]  = np.where(df["is_resi"], energy_h(Ta, ua_r), 0.0)
    df["E_total_h"] = df["E_comm_h"] + df["E_resi_h"]
    df["ua_comm"]   = ua_c
    df["ua_resi"]   = ua_r
    return df


# ══════════════════════════════════════════════════════════════
# §6  逐日聚合
# ══════════════════════════════════════════════════════════════

def _group_sum(df, mask_col, value_col):
    return df[df[mask_col]].groupby("local_date")[value_col].sum()

def _group_mean(df, mask_col, value_col):
    return df[df[mask_col]].groupby("local_date")[value_col].mean()

def daily_aggregate(df_h):
    g       = df_h.groupby("local_date")
    n_valid = g["temp_C"].count()
    valid_dates = n_valid[n_valid >= MIN_USABLE_HOURS].index

    df_v    = df_h[df_h["local_date"].isin(valid_dates)]
    g_v     = df_v.groupby("local_date")

    agg = pd.DataFrame(index=valid_dates)
    agg["n_valid"]   = n_valid[valid_dates]
    agg["Tmean"]     = g_v["temp_C"].mean()
    agg["Tmax"]      = g_v["temp_C"].max()
    agg["Tmin"]      = g_v["temp_C"].min()

    agg["CDH_total"] = g_v["CDH_h"].sum()
    agg["CDH_comm"]  = _group_sum(df_v, "is_comm",  "CDH_h").reindex(valid_dates, fill_value=0)
    agg["CDH_resi"]  = _group_sum(df_v, "is_resi",  "CDH_h").reindex(valid_dates, fill_value=0)
    agg["CDH_day"]   = _group_sum(df_v, "is_day",   "CDH_h").reindex(valid_dates, fill_value=0)
    agg["CDH_night"] = _group_sum(df_v, "is_night", "CDH_h").reindex(valid_dates, fill_value=0)

    agg["HDH_total"] = g_v["HDH_h"].sum()
    agg["HDH_comm"]  = _group_sum(df_v, "is_comm",  "HDH_h").reindex(valid_dates, fill_value=0)
    agg["HDH_resi"]  = _group_sum(df_v, "is_resi",  "HDH_h").reindex(valid_dates, fill_value=0)
    agg["HDH_day"]   = _group_sum(df_v, "is_day",   "HDH_h").reindex(valid_dates, fill_value=0)
    agg["HDH_night"] = _group_sum(df_v, "is_night", "HDH_h").reindex(valid_dates, fill_value=0)

    agg["COP_mean"]  = g_v["COP_h"].mean()
    agg["COP_comm"]  = _group_mean(df_v, "is_comm", "COP_h").reindex(valid_dates)
    agg["COP_resi"]  = _group_mean(df_v, "is_resi", "COP_h").reindex(valid_dates)

    agg["E_comm"]    = g_v["E_comm_h"].sum()
    agg["E_resi"]    = g_v["E_resi_h"].sum()
    agg["E_total"]   = g_v["E_total_h"].sum()
    agg["E_day"]     = _group_sum(df_v, "is_day",   "E_total_h").reindex(valid_dates, fill_value=0)
    agg["E_night"]   = _group_sum(df_v, "is_night", "E_total_h").reindex(valid_dates, fill_value=0)

    agg["n_comm_h"]  = _group_sum(df_v, "is_comm",  "is_comm").reindex(valid_dates, fill_value=0)
    agg["n_resi_h"]  = _group_sum(df_v, "is_resi",  "is_resi").reindex(valid_dates, fill_value=0)
    agg["n_day_h"]   = _group_sum(df_v, "is_day",   "is_day").reindex(valid_dates, fill_value=0)
    agg["n_night_h"] = _group_sum(df_v, "is_night", "is_night").reindex(valid_dates, fill_value=0)

    agg = agg.reset_index().rename(columns={"index": "local_date"})
    agg["local_date"] = pd.to_datetime(agg["local_date"])
    agg["year"]  = agg["local_date"].dt.year
    agg["month"] = agg["local_date"].dt.month
    return agg


# ══════════════════════════════════════════════════════════════
# §7  热期识别 + 时期标注
#     [M6] 严格对齐 analysis_multiyear：
#          P90 动态阈值（±7天滑动窗口）+ 连续≥3天
# ══════════════════════════════════════════════════════════════

def compute_hw_threshold_from_tmax(tmax_series: pd.Series,
                                   q=HW_PERCENTILE,
                                   window_half_width=HW_WINDOW_HALF):
    """
    根据 Tmax 计算随 DOY 变化的 P{q} 动态阈值（±{window_half_width}天滑动窗口）。
    与 analysis_multiyear 完全一致。
    """
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

    return pd.Series(thresholds).sort_index(), pd.Series(valid_counts).sort_index()


def detect_heatwave(daily_tmax: pd.Series, threshold) -> pd.Series:
    """
    Detect heatwave events using the same run-length logic as analysis_multiyear.py:
    Tmax > threshold for >= HW_MIN_DAYS consecutive elements in the supplied series.
    The caller controls whether the supplied series is annual or warm-season only.
    """
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


def assign_periods(daily_df, hw_mask_series=None, lat=None, pair_id=None,
                   analysis_hw_flags=None):
    """
    Generate long-format period records using only the canonical pair-level
    HW/NHW flags exported by analysis_multiyear.py.

    Downstream HW/NHW recomputation is disabled. If the canonical flags are
    missing or cannot be aligned, raise an error so the pair is skipped rather
    than silently changing the method.
    """
    df = daily_df.copy()
    df["local_date"] = pd.to_datetime(df["local_date"]).dt.normalize()
    df["month"] = pd.to_numeric(
        df["month"], errors="coerce"
    ).astype("Int64")

    aligned = align_analysis_hw_flags(
        analysis_hw_flags, df["local_date"]
    )
    required_flag_cols = [
        "is_warm_season",
        "hw_flag_percentile_warm_season",
        "nhw_flag_percentile_warm_season",
    ]
    if (
        aligned is None
        or any(c not in aligned.columns for c in required_flag_cols)
        or aligned[required_flag_cols].dropna(how="any").empty
    ):
        raise ValueError(
            f"{pair_id}: canonical analysis_multiyear HW/NHW flags "
            "missing or mismatched; downstream fallback is disabled"
        )

    df["_date_key"] = df["local_date"]
    tmp = aligned[required_flag_cols].copy()
    tmp["_date_key"] = tmp.index
    df = df.merge(tmp, on="_date_key", how="left").drop(
        columns=["_date_key"]
    )

    valid_flags = df[required_flag_cols].notna().all(axis=1)
    df = df.loc[valid_flags].copy()
    if df.empty:
        raise ValueError(
            f"{pair_id}: no dates remain after canonical HW/NHW alignment"
        )

    df["warm_flag"] = (
        df["is_warm_season"].astype(int).astype(bool)
    )
    df["hw_flag"] = (
        df["hw_flag_percentile_warm_season"]
        .astype(int).astype(bool)
    )
    df["nhw_flag"] = (
        df["nhw_flag_percentile_warm_season"]
        .astype(int).astype(bool)
    )

    df["hemisphere"] = hemisphere_from_lat(lat)
    df["warm_season_label"] = warm_season_label_for_lat(lat)
    df["warm_season_months"] = warm_months_string(lat)
    df["hw_source"] = "analysis_multiyear_daily_heatwave_flags"

    parts = []
    for p in PERIODS:
        if p == "annual":
            sub = df.copy()
        elif p == "warm_season":
            sub = df[df["warm_flag"]].copy()
        elif p == "heatwave":
            sub = df[df["warm_flag"] & df["hw_flag"]].copy()
        elif p == "non_heatwave":
            sub = df[df["warm_flag"] & df["nhw_flag"]].copy()
        else:
            continue
        sub["period"] = p
        parts.append(sub)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# §8  城乡合并 + 差值
# ══════════════════════════════════════════════════════════════

def merge_pair(d_u, d_r, pair_id, meta):
    u = d_u.set_index("local_date").add_suffix("_u")
    r = d_r.set_index("local_date").add_suffix("_r")
    p = u.join(r, how="inner").reset_index()
    if len(p) == 0:
        return None

    for col in ["CDH_total","CDH_comm","CDH_resi","CDH_day","CDH_night",
                "HDH_total","HDH_comm","HDH_resi","HDH_day","HDH_night",
                "E_comm","E_resi","E_total","E_day","E_night",
                "COP_mean","COP_comm","COP_resi",
                "Tmean","Tmax","Tmin"]:
        if f"{col}_u" in p.columns and f"{col}_r" in p.columns:
            p[f"d{col}"] = p[f"{col}_u"] - p[f"{col}_r"]

    p["CDH_asym"] = p["dCDH_night"] - p["dCDH_day"]
    p["E_asym"]   = p["dE_night"]   - p["dE_day"]
    p["HDH_asym"] = p["dHDH_day"]   - p["dHDH_night"]

    p["pair_id"] = pair_id
    for k, v in meta.items():
        p[k] = v
    return p


# ══════════════════════════════════════════════════════════════
# §9  逐小时廓线聚合
# ══════════════════════════════════════════════════════════════

def hourly_profiles_for_period(h_u, h_r, period_dates):
    u = h_u[h_u["local_date"].isin(period_dates)].copy()
    r = h_r[h_r["local_date"].isin(period_dates)].copy()
    if len(u) == 0 or len(r) == 0:
        return None

    metrics = ["temp_C","CDH_h","HDH_h","COP_h","E_comm_h","E_resi_h","E_total_h"]
    pu = u.groupby("local_hour")[metrics].mean()
    pr = r.groupby("local_hour")[metrics].mean()

    rows = []
    for h in range(24):
        row = {"hour": h}
        for m in metrics:
            row[f"{m}_u"] = float(pu[m].get(h, np.nan))
            row[f"{m}_r"] = float(pr[m].get(h, np.nan))
            row[f"d{m}"]  = row[f"{m}_u"] - row[f"{m}_r"]
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════
# §10  统计汇总（含 KG 分层版本）
# ══════════════════════════════════════════════════════════════

def _pair_level_aggregate(df):
    """
    Step-1: aggregate pair x period x strata for CDH/HDH/energy outputs.

    Sleep-loss approximations are intentionally excluded. The canonical sleep
    model is implemented in compute_hne_panel.py using the Supplement method,
    so this CDH/HDH script must not emit an alternative slope-based sleep
    endpoint.
    """
    sum_cols = [c for c in df.columns if (
        c.startswith("dCDH")
        or c.startswith("dHDH")
        or c.startswith("dE_")
        or c in ("CDH_asym", "E_asym", "HDH_asym")
    ) and c in df.columns]
    mean_cols = [c for c in df.columns if (
        c.startswith("dCOP")
        or c.startswith("dTmean")
        or c.startswith("dTmax")
        or c.startswith("dTmin")
    ) and c in df.columns]

    kg_cols = [
        c for c in [
            "kg_code", "kg_group", "climate_zone_main"
        ] if c in df.columns
    ]

    pair_rows = []
    for (pid, period, strata), sub in df.groupby(
        ["pair_id", "period", "strata"], observed=True
    ):
        row = {
            "pair_id": pid,
            "period": period,
            "strata": strata,
            "n_days": sub["local_date"].nunique(),
        }

        for kc in kg_cols:
            vals = sub[kc].dropna().unique()
            row[kc] = vals[0] if len(vals) > 0 else np.nan

        for c in sum_cols:
            v = sub[c].dropna()
            row[f"{c}_sum"] = float(v.sum())
            row[f"{c}_mean"] = (
                float(v.mean()) if len(v) else np.nan
            )

        for c in mean_cols:
            v = sub[c].dropna()
            row[f"{c}_mean"] = (
                float(v.mean()) if len(v) else np.nan
            )

        pair_rows.append(row)

    return pd.DataFrame(pair_rows)


def _cross_pair_stats(df_pair, group_keys, stat_cols):
    """
    Step-2：对 df_pair 按 group_keys 分组，对 stat_cols 做跨对统计。
    """
    rows = []
    for keys, g in df_pair.groupby(group_keys, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row["n_pairs"]        = len(g)
        row["n_days_median"]  = float(g["n_days"].median())
        for c in stat_cols:
            v = g[c].dropna() if c in g.columns else pd.Series(dtype=float)
            if len(v) == 0:
                continue
            row[f"{c}_mean"]   = round(float(v.mean()),   4)
            row[f"{c}_median"] = round(float(v.median()), 4)
            row[f"{c}_se"]     = round(float(v.sem()),    4)
            row[f"{c}_p25"]    = round(float(v.quantile(0.25)), 4)
            row[f"{c}_p75"]    = round(float(v.quantile(0.75)), 4)
            if len(v) > 2:
                t, p = stats.ttest_1samp(v, 0)
                row[f"{c}_tstat"] = round(float(t), 3)
                row[f"{c}_pval"]  = round(float(p), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(df):
    """
    pair×period 两步聚合，消除冬季零值稀释。
    [M4] 新增 period_summary_by_kggroup.csv；pair_period_summary.csv 携带 KG 字段。
    """
    # ── Step-1：pair 级聚合 ───────────────────────────────────
    df_pair = _pair_level_aggregate(df)

    # 确定要做统计的列
    sum_cols  = [c for c in df.columns if (
        c.startswith("dCDH") or c.startswith("dHDH") or c.startswith("dE_") or
        c in ("CDH_asym", "E_asym", "HDH_asym")
    ) and c in df.columns]
    mean_cols = [c for c in df.columns if (
        c.startswith("dCOP") or c.startswith("dTmean") or
        c.startswith("dTmax") or c.startswith("dTmin")
    ) and c in df.columns]

    stat_cols = (
        [c + "_sum" for c in sum_cols]
        + [c + "_mean" for c in sum_cols + mean_cols]
    )
    stat_cols = [c for c in stat_cols if c in df_pair.columns]

    # ── Step-2a：strata × period 汇总 ────────────────────────
    summary = _cross_pair_stats(df_pair, ["period", "strata"], stat_cols)
    summary.to_csv(os.path.join(OUTPUT_DIR, "period_summary.csv"), index=False)
    print("  ✓  period_summary.csv  (strata × period，pair×period 两步聚合)")

    # ── Step-2b：climate_zone_main × period 汇总 [M4] ─────────
    if "climate_zone_main" in df_pair.columns:
        summary_kg = _cross_pair_stats(
            df_pair, ["period", "climate_zone_main"], stat_cols
        )
        summary_kg.to_csv(
            os.path.join(OUTPUT_DIR, "period_summary_by_kggroup.csv"), index=False
        )
        print("  ✓  period_summary_by_kggroup.csv  (KG × period，新增)")
    else:
        print("  ⚠  climate_zone_main 列缺失，跳过 period_summary_by_kggroup.csv")

    # ── [M4] pair 级汇总（含 KG 字段）供箱线图 ───────────────
    df_pair.to_csv(os.path.join(OUTPUT_DIR, "pair_period_summary.csv"), index=False)
    print("  ✓  pair_period_summary.csv  (含 kg_code / kg_group / climate_zone_main)")

    return summary

def export_cooling_trap_figure_csv(
        daily_panel: pd.DataFrame,
        hourly_profiles_pair: pd.DataFrame,
        output_dir: str,
        mechanism_csv: str | None = None):
    """
    Export CSV files for:
    Figure X | Night-time heat storage shifts urban energy demand into a cooling trap

    Outputs:
      1) panel_a_hourly_cdh_wave.csv
      2) panel_b_day_night_redistribution.csv
      3) panel_c_night_cooling_amplification.csv
      4) panel_d_mechanism_energy_coupling.csv, if mechanism_csv is provided
    """

    fig_dir = output_dir
    ensure_dir(fig_dir)

    df = daily_panel.copy()
    hp = hourly_profiles_pair.copy()

    # ─────────────────────────────────────────────
    # Panel a | Diurnal cooling demand wave
    # pair × period × hour profile
    # ─────────────────────────────────────────────
    panel_a_cols = [
        "pair_id", "group", "strata", "density",
        "period", "hour",
        "CDH_h_u", "CDH_h_r", "dCDH_h",
        "temp_C_u", "temp_C_r", "dtemp_C",
        "E_total_h_u", "E_total_h_r", "dE_total_h",
        "kg_code", "kg_group", "climate_zone_main"
    ]

    panel_a_cols = [c for c in panel_a_cols if c in hp.columns]

    panel_a = hp[panel_a_cols].copy()

    panel_a = panel_a.rename(columns={
        "CDH_h_u": "cdh_U",
        "CDH_h_r": "cdh_R",
        "dCDH_h": "d_cdh",
        "temp_C_u": "temp_U",
        "temp_C_r": "temp_R",
        "dtemp_C": "d_temp",
        "E_total_h_u": "energy_U",
        "E_total_h_r": "energy_R",
        "dE_total_h": "d_energy",
    })

    panel_a["is_night"] = panel_a["hour"].isin(NIGHT_HOURS)
    panel_a["is_day"] = panel_a["hour"].isin(DAY_HOURS)

    panel_a.to_csv(
        os.path.join(fig_dir, "panel_a_hourly_cdh_wave.csv"),
        index=False
    )

    # ─────────────────────────────────────────────
    # Panel b/c | pair-period summary
    # day vs night redistribution + night amplification
    # ─────────────────────────────────────────────
    group_cols = [
        "pair_id", "group", "strata", "density", "period",
        "kg_code", "kg_group", "climate_zone_main"
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    agg = (
        df.groupby(group_cols, observed=True)
          .agg(
              n_days=("local_date", "nunique"),

              day_cdh_U=("CDH_day_u", "sum"),
              day_cdh_R=("CDH_day_r", "sum"),
              night_cdh_U=("CDH_night_u", "sum"),
              night_cdh_R=("CDH_night_r", "sum"),

              total_cdh_U=("CDH_total_u", "sum"),
              total_cdh_R=("CDH_total_r", "sum"),

              d_day_cdh=("dCDH_day", "sum"),
              d_night_cdh=("dCDH_night", "sum"),
              d_total_cdh=("dCDH_total", "sum"),

              day_energy_U=("E_day_u", "sum"),
              day_energy_R=("E_day_r", "sum"),
              night_energy_U=("E_night_u", "sum"),
              night_energy_R=("E_night_r", "sum"),

              d_day_energy=("dE_day", "sum"),
              d_night_energy=("dE_night", "sum"),
              d_total_energy=("dE_total", "sum"),
          )
          .reset_index()
    )

    eps = 1e-9

    agg["night_share_U"] = agg["night_cdh_U"] / (agg["total_cdh_U"] + eps)
    agg["night_share_R"] = agg["night_cdh_R"] / (agg["total_cdh_R"] + eps)

    agg["urban_excess_night_cdh"] = agg["night_cdh_U"] - agg["night_cdh_R"]
    agg["urban_excess_day_cdh"] = agg["day_cdh_U"] - agg["day_cdh_R"]

    agg["night_amplification"] = agg["night_cdh_U"] / (agg["night_cdh_R"] + eps)
    agg["day_amplification"] = agg["day_cdh_U"] / (agg["day_cdh_R"] + eps)

    agg["excess_night_share"] = (
        agg["urban_excess_night_cdh"] /
        (agg["urban_excess_night_cdh"] + agg["urban_excess_day_cdh"] + eps)
    )

    agg["night_minus_day_excess_cdh"] = (
        agg["urban_excess_night_cdh"] - agg["urban_excess_day_cdh"]
    )

    agg.to_csv(
        os.path.join(fig_dir, "panel_b_day_night_redistribution.csv"),
        index=False
    )

    agg.to_csv(
        os.path.join(fig_dir, "panel_c_night_cooling_amplification.csv"),
        index=False
    )

    # ─────────────────────────────────────────────
    # Panel d | coupling with mechanism
    # mechanism_csv should contain:
    # pair_id, period, phase_lag_h and/or hysteresis_area
    # ─────────────────────────────────────────────
    if mechanism_csv is not None and os.path.exists(mechanism_csv):
        mech = pd.read_csv(mechanism_csv)

        needed = ["pair_id", "period"]
        if all(c in mech.columns for c in needed):
            panel_d = agg.merge(
                mech,
                on=["pair_id", "period"],
                how="left"
            )

            panel_d.to_csv(
                os.path.join(fig_dir, "panel_d_mechanism_energy_coupling.csv"),
                index=False
            )
        else:
            print("  ⚠ mechanism_csv 缺少 pair_id / period，跳过 panel d")
    else:
        print("  ⚠ 未提供 mechanism_csv，panel d coupling CSV 暂不输出")

    print(f"  ✓ Figure cooling-trap CSV exported to: {fig_dir}")


# ══════════════════════════════════════════════════════════════
# §11  多进程任务封装与主处理循环
# ══════════════════════════════════════════════════════════════

def process_single_pair(args):
    """供多进程调用的独立处理函数"""
    idx, row = args
    pid = str(row["pair_id"])

    result = {
        "pair_id":      pid,
        "panel":        None,
        "profile_rows": [],
        "strata_info":  "",
        "error":        None,
    }

    try:
        usaf_u, wban_u, usaf_r, wban_r = parse_pair_id(pid)
    except Exception as e:
        result["error"] = f"解析失败: {e}"
        return result

    lon_u = float(row["lon_urban"])
    lat_u = float(row["lat_urban"])
    lon_r = float(row["lon_rural"])
    lat_r = float(row["lat_rural"]) if "lat_rural" in row and pd.notna(row["lat_rural"]) else np.nan

    df_u_raw = load_station(usaf_u, wban_u, lon_u)
    df_r_raw = load_station(usaf_r, wban_r, lon_r)
    if df_u_raw is None or df_r_raw is None:
        result["error"] = "站点数据不足或全为缺测"
        return result

    lcz     = row.get("urban_lcz_corrected", row.get("urban_lcz_raw", np.nan))
    density = lcz_to_density(lcz)

    h_u = compute_hourly(df_u_raw, density)
    h_r = compute_hourly(df_r_raw, density)

    d_u = daily_aggregate(h_u)
    d_r = daily_aggregate(h_r)
    if len(d_u) == 0 or len(d_r) == 0:
        result["error"] = "有效日数不足"
        return result

    u_tmax = d_u.set_index("local_date")["Tmax"]
    r_tmax = d_r.set_index("local_date")["Tmax"]
    common_dates = u_tmax.index.intersection(r_tmax.index)

    # Diagnostic only: daily Tmax contrast. UHI/UCI grouping is canonical from main analysis.
    delta_tmax_daily_ann_mean = (
        float((u_tmax[common_dates] - r_tmax[common_dates]).mean())
        if len(common_dates) > 0 else np.nan
    )

    try:
        canonical_group_info = get_canonical_pair_group(pid)
    except Exception as exc:
        result["error"] = f"canonical_group_missing_or_invalid: {exc}"
        return result

    group = canonical_group_info["group"]
    strata = f"{group}_{density}"

    # Canonical HW/NHW dates and threshold diagnostics come only from
    # 01_main_pair_period_metrics.py. No downstream HW re-detection.
    analysis_hw_flags = get_analysis_hw_flags_for_pair(pid)
    _aligned_hw = align_analysis_hw_flags(analysis_hw_flags, u_tmax.index)
    required_flag_cols = [
        "is_warm_season",
        "hw_flag_percentile_warm_season",
        "nhw_flag_percentile_warm_season",
    ]
    if (
        _aligned_hw is None
        or any(c not in _aligned_hw.columns for c in required_flag_cols)
        or _aligned_hw[required_flag_cols].dropna(how="any").empty
    ):
        result["error"] = (
            "canonical 01_main_pair_period_metrics HW/NHW flags missing or "
            "mismatched; downstream fallback is disabled"
        )
        return result

    valid_hw = _aligned_hw[required_flag_cols].notna().all(axis=1)
    hw_mask = pd.Series(False, index=u_tmax.index)
    hw_mask.loc[valid_hw] = (
        _aligned_hw.loc[valid_hw, "hw_flag_percentile_warm_season"]
        .astype(int).astype(bool).values
    )

    # Preserve threshold/QM diagnostics by reading the values exported by 01.
    def _num_series_from_01(col):
        if col not in _aligned_hw.columns:
            return pd.Series(np.nan, index=u_tmax.index, dtype=float)
        return pd.to_numeric(_aligned_hw[col], errors="coerce").reindex(u_tmax.index)

    hw_thr_series_raw = _num_series_from_01("hw_threshold_raw")
    hw_thr_series = _num_series_from_01("hw_threshold_corrected")
    if hw_thr_series.notna().sum() == 0:
        hw_thr_series = hw_thr_series_raw.copy()

    def _first_nonempty(col, default=""):
        if col not in _aligned_hw.columns:
            return default
        vals = _aligned_hw[col].dropna().astype(str).str.strip()
        vals = vals[vals.ne("")]
        return vals.iloc[0] if len(vals) else default

    bias_correction_method_01 = _first_nonempty(
        "bias_correction_method", "from_01_main_pair_period_metrics"
    )
    hw_ref_mode_01 = _first_nonempty("hw_ref_mode", HW_REF_MODE)
    qm_diagnostics = {
        "qm_ref_mode": _first_nonempty("qm_ref_mode", ""),
        "qm_n_quantiles": QM_N_QUANTILES,
        "overlap_days": (
            float(pd.to_numeric(_aligned_hw["qm_overlap_days"], errors="coerce").dropna().iloc[0])
            if "qm_overlap_days" in _aligned_hw.columns
            and pd.to_numeric(_aligned_hw["qm_overlap_days"], errors="coerce").notna().any()
            else np.nan
        ),
    }

    try:
        d_u_long = assign_periods(
            d_u,
            hw_mask,
            lat=lat_u,
            pair_id=pid,
            analysis_hw_flags=analysis_hw_flags,
        )
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    # [M2] meta 字典写入 KG 字段
    meta = {
        "strata":             strata,
        "group":              group,
        "uhi_definition":     canonical_group_info.get("uhi_definition", "canonical_from_main_analysis"),
        "uhi_classification_metric": canonical_group_info.get("uhi_classification_metric", "delta_tx_annual_synth"),
        "delta_tx_annual_synth": canonical_group_info.get("delta_tx_annual_synth", np.nan),
        "delta_tmax_daily_ann_mean": delta_tmax_daily_ann_mean,
        "density":            density,
        "lat_urban":          float(row["lat_urban"]),
        "lon_urban":          lon_u,
        "continent":          row.get("continent", ""),
        "ua_comm":            get_ua(density, "COMM"),
        "ua_resi":            get_ua(density, "RESI"),
        "hw_ref_mode":        hw_ref_mode_01,        # from 01
        "bias_correction_method": bias_correction_method_01,
        "qm_ref_mode": qm_diagnostics.get("qm_ref_mode", ""),
        "qm_n_quantiles": qm_diagnostics.get("qm_n_quantiles", np.nan),
        "qm_overlap_days": qm_diagnostics.get("overlap_days", np.nan),
        "hw_threshold_raw_mean": float(hw_thr_series_raw.mean()),
        "hw_threshold_qm_mean": (
            float(hw_thr_series.mean()) if USE_QM_BIAS_CORRECTION else np.nan
        ),
        "exceed_days_raw": int(
            (u_tmax > hw_thr_series_raw.reindex(u_tmax.index)).fillna(False).sum()
        ),
        "exceed_days_qm": int(
            (u_tmax > hw_thr_series.reindex(u_tmax.index)).fillna(False).sum()
        ),

        # KG 气候区字段
        "kg_code":            row.get("kg_code",            np.nan),  # [M2]
        "kg_group":           row.get("kg_group",           np.nan),  # [M2]
        "climate_zone_main":  row.get("climate_zone_main",  np.nan),  # [M2]
        "lat_group":          row.get("lat_group",          np.nan),  # [M2]
        "hemisphere":         hemisphere_from_lat(lat_u),
        "warm_season_label":  warm_season_label_for_lat(lat_u),
        "warm_season_months": warm_months_string(lat_u),
    }

    panel_raw = merge_pair(d_u, d_r, pid, meta)
    if panel_raw is None:
        result["error"] = "城乡时间轴无交集"
        return result

    period_map = (
        d_u_long[["local_date", "period"]]
        .assign(local_date=lambda x: pd.to_datetime(x["local_date"]))
        .drop_duplicates()
    )
    panel_raw["local_date"] = pd.to_datetime(panel_raw["local_date"])
    panel = panel_raw.merge(period_map, on="local_date", how="left")
    panel["period"] = panel["period"].fillna("annual")

    daily_dir = os.path.join(OUTPUT_DIR, "daily")
    panel.to_csv(os.path.join(daily_dir, f"{pid}_daily.csv"), index=False)

    result["panel"] = panel

    # 廓线收集（各时期），附加 KG 信息供 process_all 分组
    date_map = (
        d_u_long.groupby("period")["local_date"].apply(set).to_dict()
    )
    profile_rows_local = []
    for period, date_set in date_map.items():
        date_set_dt = {pd.to_datetime(d).date() for d in date_set}
        prows = hourly_profiles_for_period(h_u, h_r, date_set_dt)
        if prows:
            for r2 in prows:
                r2.update({
                    "strata":            strata,
                    "density":           density,
                    "group":             group,
                    "period":            period,
                    "pair_id":           pid,
                    "kg_code":           row.get("kg_code",           np.nan),  # [M5]
                    "kg_group":          row.get("kg_group",          np.nan),  # [M5]
                    "climate_zone_main": row.get("climate_zone_main", np.nan),  # [M5]
                })
                profile_rows_local.append(r2)

    result["profile_rows"] = profile_rows_local

    n_d  = panel["local_date"].nunique()
    n_hw = panel[panel["period"] == "heatwave"]["local_date"].nunique()
    result["strata_info"] = (
        f"Strata={strata}  KG={row.get('kg_code', '?')}  有效天={n_d}  热浪天={n_hw}"
    )
    return result


def process_all(pair_df):
    daily_dir = os.path.join(OUTPUT_DIR, "daily")
    ensure_dir(daily_dir)

    figure_csv_dir = os.path.join(OUTPUT_DIR, "figure_cooling_trap", "csv")
    ensure_dir(figure_csv_dir)

    all_panels    = []
    profile_rows  = []
    error_records = []
    processed = skipped = 0
    total = len(pair_df)

    n_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"\n🚀 开始多进程加速处理，启用 {n_cores} 个 CPU 核心...\n")

    tasks = [(idx, row) for idx, row in pair_df.iterrows()]

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(process_single_pair, args): args for args in tasks}

        for future in as_completed(futures):
            idx, row = futures[future]
            pid = str(row["pair_id"])

            try:
                res = future.result()

                if res["error"]:
                    print(f"[{idx+1:>4d}/{total}] {pid} ⚠ {res['error']}")
                    skipped += 1
                    error_records.append({
                        "pair_id":   pid,
                        "fail_step": "Process",
                        "missing_data": res["error"],
                    })
                    continue

                if res["panel"] is not None:
                    all_panels.append(res["panel"])
                    profile_rows.extend(res["profile_rows"])
                    processed += 1
                    print(f"[{idx+1:>4d}/{total}] {pid} ✓ {res['strata_info']}")

            except Exception as exc:
                print(f"[{idx+1:>4d}/{total}] {pid} ⚠ 崩溃: {exc}")
                skipped += 1
                error_records.append({
                    "pair_id":   pid,
                    "fail_step": "Crash",
                    "missing_data": f"致命错误: {exc}",
                })

    print(f"\n完成: {processed} 对  跳过: {skipped} 对")

    if error_records:
        error_df = pd.DataFrame(error_records)
        error_df.to_csv(os.path.join(OUTPUT_DIR, "skipped_pairs_log.csv"), index=False)
        print(f"  ✓  skipped_pairs_log.csv (记录了 {len(error_df)} 个被剔除站点)")

    if not all_panels:
        return None, None

    # [M3] all_pairs_daily_panel.csv 已通过 meta 携带 KG 字段
    combined = pd.concat(all_panels, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "all_pairs_daily_panel.csv"), index=False)
    print(f"  ✓  all_pairs_daily_panel.csv  ({len(combined):,} 行，含 kg_code/kg_group/climate_zone_main)")

    # ── 廓线聚合 ─────────────────────────────────────────────
    profiles_df = pd.DataFrame(profile_rows)
    if len(profiles_df) > 0:
        # — pair-level hourly profiles for Figure X panel a/d —
        pair_hourly_path = os.path.join(
            figure_csv_dir,
            "pair_hourly_cdh_profiles.csv"
        )
        profiles_df.to_csv(pair_hourly_path, index=False)

        profiles_df.to_csv(
            os.path.join(figure_csv_dir, "pair_hourly_cdh_profiles.csv"),
            index=False
        )
        print(f"  ✓  figure_cooling_trap/csv/pair_hourly_cdh_profiles.csv  ({len(profiles_df):,} 行)")

        
        num_cols = [c for c in profiles_df.columns
                    if c not in ["strata","density","group","period","pair_id","hour",
                                 "kg_code","kg_group","climate_zone_main"]]

        # — strata 版廓线 ————————————————————————————————
        prof_agg = (
            profiles_df
            .groupby(["strata","density","group","period","hour"])[num_cols]
            .mean()
            .reset_index()
        )
        prof_agg["n_pairs"] = (
            profiles_df
            .groupby(["strata","density","group","period","hour"])["pair_id"]
            .nunique()
            .values
        )
        prof_agg.to_csv(os.path.join(OUTPUT_DIR, "hourly_profiles.csv"), index=False)
        print(f"  ✓  hourly_profiles.csv  ({len(prof_agg):,} 行)")

        # — KG 版廓线 [M5] ——————————————————————————————
        if "climate_zone_main" in profiles_df.columns:
            prof_kg = (
                profiles_df
                .dropna(subset=["climate_zone_main"])
                .groupby(["climate_zone_main","period","hour"])[num_cols]
                .mean()
                .reset_index()
            )
            prof_kg["n_pairs"] = (
                profiles_df
                .dropna(subset=["climate_zone_main"])
                .groupby(["climate_zone_main","period","hour"])["pair_id"]
                .nunique()
                .values
            )
            prof_kg.to_csv(
                os.path.join(OUTPUT_DIR, "hourly_profiles_by_kggroup.csv"), index=False
            )
            print(f"  ✓  hourly_profiles_by_kggroup.csv  ({len(prof_kg):,} 行，新增)")
        else:
            print("  ⚠  climate_zone_main 列缺失，跳过 hourly_profiles_by_kggroup.csv")
    else:
        prof_agg = pd.DataFrame()

    return combined, prof_agg


# ══════════════════════════════════════════════════════════════
# §12  控制台摘要
# ══════════════════════════════════════════════════════════════

def print_summary(df):
    strata_avail  = [s for s in STRATA_ORDER if s in df["strata"].unique()]
    periods_avail = [p for p in PERIODS if p in df["period"].unique()]

    w = 90
    print("\n" + "═"*w)
    print("  CDH / HDH / ENERGY  URBAN–RURAL SUMMARY")
    print("═"*w)
    hdr = (f"  {'Strata':12} {'Period':14} {'n_d':>5}"
           f" {'dCDH':>9} {'dCDH_c':>9} {'dCDH_r':>9}"
           f" {'dE_c':>8} {'dE_r':>8} {'dCOP':>8}")
    print(hdr)
    print("  " + "─"*(len(hdr)-2))
    for strat in strata_avail:
        for period in periods_avail:
            g = df[(df["strata"]==strat) & (df["period"]==period)]
            if len(g) == 0: continue
            nd = g["local_date"].nunique()
            def mv(col): return g[col].mean() if col in g else float("nan")
            print(f"  {strat:12} "
                  f"{'Ann' if period=='annual' else period[:12]:14} "
                  f"{nd:>5d}"
                  f" {mv('dCDH_total'):>+9.3f}"
                  f" {mv('dCDH_comm'):>+9.3f}"
                  f" {mv('dCDH_resi'):>+9.3f}"
                  f" {mv('dE_comm'):>+8.2f}"
                  f" {mv('dE_resi'):>+8.2f}"
                  f" {mv('dCOP_mean'):>+8.4f}")
    print("═"*w)

    if "heatwave" in df["period"].unique():
        print("\n  HEATWAVE AMPLIFICATION  (dCDH_total  ×  ratio vs warm_season)")
        base_p = "warm_season" if "warm_season" in df["period"].unique() else "annual"
        for strat in strata_avail:
            base = df[(df["period"]==base_p) & (df["strata"]==strat)]["dCDH_total"]
            hw   = df[(df["period"]=="heatwave") & (df["strata"]==strat)]["dCDH_total"]
            if len(base) > 0 and len(hw) > 0:
                amp = abs(hw.mean()) / (abs(base.mean()) + 1e-9)
                print(f"  {strat}: {base_p}={base.mean():+.3f}  HW={hw.mean():+.3f}  ×{amp:.2f}")
    print()

    # KG 分层摘要
    if "climate_zone_main" in df.columns:
        print("  CDH URBAN–RURAL BY KG CLIMATE ZONE  (dCDH_total 日均值)")
        print(f"  {'KG Zone':12} {'Period':14} {'n_pairs':>8} {'dCDH_mean':>12}")
        print("  " + "─"*52)
        for kg in KG_GROUP_ORDER:
            for period in periods_avail:
                g = df[(df["climate_zone_main"]==kg) & (df["period"]==period)]
                if len(g) == 0: continue
                n_p = g["pair_id"].nunique()
                print(f"  {kg:12} {period:14} {n_p:>8d} "
                      f"{g['dCDH_total'].mean():>+12.3f}")
        print()

def export_building_type_figure_csv(daily_panel, output_dir):
    """
    Export commercial / residential / operational cooling-demand data
    for Figure 4 panel c–d.

    This function only writes extra CSV files into:
        OUTPUT_DIR/figure_cooling_trap/csv/

    It does not modify the core pipeline or delete existing outputs.
    """

    fig_csv_dir = os.path.join(output_dir, "figure_cooling_trap", "csv")
    ensure_dir(fig_csv_dir)

    df = daily_panel.copy()

    group_cols = [
        "pair_id", "group", "strata", "density", "period",
        "kg_code", "kg_group", "climate_zone_main"
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    agg = (
        df.groupby(group_cols, observed=True)
          .agg(
              n_days=("local_date", "nunique"),

              comm_cdh_U=("CDH_comm_u", "sum"),
              comm_cdh_R=("CDH_comm_r", "sum"),
              resi_cdh_U=("CDH_resi_u", "sum"),
              resi_cdh_R=("CDH_resi_r", "sum"),

              comm_energy_U=("E_comm_u", "sum"),
              comm_energy_R=("E_comm_r", "sum"),
              resi_energy_U=("E_resi_u", "sum"),
              resi_energy_R=("E_resi_r", "sum"),

              day_energy_U=("E_day_u", "sum"),
              day_energy_R=("E_day_r", "sum"),
              night_energy_U=("E_night_u", "sum"),
              night_energy_R=("E_night_r", "sum"),

              total_energy_U=("E_total_u", "sum"),
              total_energy_R=("E_total_r", "sum"),
          )
          .reset_index()
    )

    eps = 1e-9

    agg["d_comm_cdh"] = agg["comm_cdh_U"] - agg["comm_cdh_R"]
    agg["d_resi_cdh"] = agg["resi_cdh_U"] - agg["resi_cdh_R"]

    agg["d_comm_energy"] = agg["comm_energy_U"] - agg["comm_energy_R"]
    agg["d_resi_energy"] = agg["resi_energy_U"] - agg["resi_energy_R"]

    agg["d_day_energy"] = agg["day_energy_U"] - agg["day_energy_R"]
    agg["d_night_energy"] = agg["night_energy_U"] - agg["night_energy_R"]
    agg["d_total_energy"] = agg["total_energy_U"] - agg["total_energy_R"]

    agg["comm_amplification"] = agg["comm_cdh_U"] / (agg["comm_cdh_R"] + eps)
    agg["resi_amplification"] = agg["resi_cdh_U"] / (agg["resi_cdh_R"] + eps)

    agg["resi_share_U"] = agg["resi_energy_U"] / (
        agg["comm_energy_U"] + agg["resi_energy_U"] + eps
    )
    agg["resi_share_R"] = agg["resi_energy_R"] / (
        agg["comm_energy_R"] + agg["resi_energy_R"] + eps
    )

    agg["comm_share_U"] = agg["comm_energy_U"] / (
        agg["comm_energy_U"] + agg["resi_energy_U"] + eps
    )
    agg["comm_share_R"] = agg["comm_energy_R"] / (
        agg["comm_energy_R"] + agg["resi_energy_R"] + eps
    )

    agg["urban_excess_resi_share"] = agg["d_resi_energy"] / (
        agg["d_comm_energy"] + agg["d_resi_energy"] + eps
    )

    agg["total_night_fraction_U"] = agg["night_energy_U"] / (
        agg["total_energy_U"] + eps
    )
    agg["total_night_fraction_R"] = agg["night_energy_R"] / (
        agg["total_energy_R"] + eps
    )

    agg["total_day_fraction_U"] = agg["day_energy_U"] / (
        agg["total_energy_U"] + eps
    )
    agg["total_day_fraction_R"] = agg["day_energy_R"] / (
        agg["total_energy_R"] + eps
    )

    agg["urban_excess_night_fraction"] = agg["d_night_energy"] / (
        agg["d_total_energy"] + eps
    )

    # Compatibility columns for downstream plotting
    agg["resi_share"] = agg["resi_share_U"]
    agg["comm_share"] = agg["comm_share_U"]
    agg["total_night_fraction"] = agg["total_night_fraction_U"]
    agg["total_day_fraction"] = agg["total_day_fraction_U"]

    out_path = os.path.join(
        fig_csv_dir,
        "building_type_commercial_residential_cooling.csv"
    )

    agg.to_csv(out_path, index=False)

    print(
        "  ✓ building_type_commercial_residential_cooling.csv "
        f"exported to: {fig_csv_dir}"
    )

# ══════════════════════════════════════════════════════════════
# Figure 4 export module: CDH / cooling-energy pathway data
# ══════════════════════════════════════════════════════════════

def _fig4_cdh_period_norm(x):
    """Normalize period names for Figure 4."""
    mp = {
        "heatwave": "HW",
        "non_heatwave": "NHW",
        "warm_season": "warm_season",
        "annual": "annual",
        "HW": "HW",
        "NHW": "NHW",
        "JJA": "warm_season",
    }
    return mp.get(str(x), str(x))


def _fig4_cdh_copy_numeric(df, new_col, old_col):
    """Create standardized numeric column if source column exists."""
    if old_col in df.columns:
        df[new_col] = pd.to_numeric(df[old_col], errors="coerce")
    else:
        df[new_col] = np.nan
    return df


def _fig4_cdh_build_pair_period_table(daily_panel):
    """
    Build Figure 4 CDH pair-period table.

    Input:
        daily_panel:
            combined all_pairs_daily_panel from this script.

    Output level:
        one row = pair_id × period_norm

    Key definitions:
        dAmp1:
            (Tmax_U - Tmin_U) - (Tmax_R - Tmin_R)
            = dTmax - dTmin

        night_heat_exposure:
            dCDH_night = CDH_night_U - CDH_night_R
            unit: degC h day-1

        day_heat_exposure:
            dCDH_day = CDH_day_U - CDH_day_R
            unit: degC h day-1

        night_cdh:
            same as dCDH_night, exported explicitly for energy endpoint.

        night_cooling_energy:
            dE_night = E_night_U - E_night_R
    """

    if daily_panel is None or len(daily_panel) == 0:
        raise ValueError("[Fig4 CDH] daily_panel is empty.")

    required_cols = ["pair_id", "period"]
    for c in required_cols:
        if c not in daily_panel.columns:
            raise ValueError(f"[Fig4 CDH] Missing required column: {c}")

    df = daily_panel.copy()
    df["period_norm"] = df["period"].apply(_fig4_cdh_period_norm)
    df = df[df["period_norm"].isin(["HW", "NHW"])].copy()

    if len(df) == 0:
        raise ValueError("[Fig4 CDH] No HW/NHW rows found in daily panel.")

    # ── Mechanism variable: dAmp1 ─────────────────────────────
    if "dAmp1" not in df.columns:
        if "dTmax" in df.columns and "dTmin" in df.columns:
            df["dAmp1"] = (
                pd.to_numeric(df["dTmax"], errors="coerce")
                - pd.to_numeric(df["dTmin"], errors="coerce")
            )
        elif all(c in df.columns for c in ["Tmax_u", "Tmin_u", "Tmax_r", "Tmin_r"]):
            df["dAmp1"] = (
                (pd.to_numeric(df["Tmax_u"], errors="coerce")
                 - pd.to_numeric(df["Tmin_u"], errors="coerce"))
                -
                (pd.to_numeric(df["Tmax_r"], errors="coerce")
                 - pd.to_numeric(df["Tmin_r"], errors="coerce"))
            )
        else:
            df["dAmp1"] = np.nan
            print(
                "  [Fig4 CDH WARNING] Cannot compute dAmp1. "
                "Missing dTmax/dTmin or Tmax/Tmin urban-rural columns."
            )

    # ── Standardized CDH / energy columns ─────────────────────
    # Urban-rural CDH contrasts.
    df = _fig4_cdh_copy_numeric(df, "night_heat_exposure", "dCDH_night")
    df = _fig4_cdh_copy_numeric(df, "day_heat_exposure", "dCDH_day")
    df = _fig4_cdh_copy_numeric(df, "total_heat_exposure", "dCDH_total")

    df = _fig4_cdh_copy_numeric(df, "night_cdh", "dCDH_night")
    df = _fig4_cdh_copy_numeric(df, "day_cdh", "dCDH_day")
    df = _fig4_cdh_copy_numeric(df, "total_cdh", "dCDH_total")

    # Absolute urban / rural CDH, useful for diagnostics.
    df = _fig4_cdh_copy_numeric(df, "night_cdh_urban", "CDH_night_u")
    df = _fig4_cdh_copy_numeric(df, "night_cdh_rural", "CDH_night_r")
    df = _fig4_cdh_copy_numeric(df, "day_cdh_urban", "CDH_day_u")
    df = _fig4_cdh_copy_numeric(df, "day_cdh_rural", "CDH_day_r")
    df = _fig4_cdh_copy_numeric(df, "total_cdh_urban", "CDH_total_u")
    df = _fig4_cdh_copy_numeric(df, "total_cdh_rural", "CDH_total_r")

    # Cooling energy contrasts.
    df = _fig4_cdh_copy_numeric(df, "night_cooling_energy", "dE_night")
    df = _fig4_cdh_copy_numeric(df, "day_cooling_energy", "dE_day")
    df = _fig4_cdh_copy_numeric(df, "total_cooling_energy", "dE_total")

    df = _fig4_cdh_copy_numeric(df, "commercial_cooling_energy", "dE_comm")
    df = _fig4_cdh_copy_numeric(df, "residential_cooling_energy", "dE_resi")

    df = _fig4_cdh_copy_numeric(df, "commercial_cdh", "dCDH_comm")
    df = _fig4_cdh_copy_numeric(df, "residential_cdh", "dCDH_resi")

    # Temperature diagnostics.
    df = _fig4_cdh_copy_numeric(df, "dTMAX_cdh_daily", "dTmax")
    df = _fig4_cdh_copy_numeric(df, "dTMIN_cdh_daily", "dTmin")
    df = _fig4_cdh_copy_numeric(df, "dTMEAN_cdh_daily", "dTmean")

    # Asymmetry diagnostics already defined in the CDH script.
    df = _fig4_cdh_copy_numeric(df, "CDH_asym_daily", "CDH_asym")
    df = _fig4_cdh_copy_numeric(df, "Energy_asym_daily", "E_asym")

    # ── Metadata ──────────────────────────────────────────────
    meta_cols = [
        "group", "strata", "density",
        "kg_code", "kg_group", "climate_zone_main",
        "lat_group", "continent",
        "lat_urban", "lon_urban",
        "ua_comm", "ua_resi",
        "hw_ref_mode", "bias_correction_method",
        "qm_ref_mode", "qm_n_quantiles", "qm_overlap_days",
        "hw_threshold_raw_mean", "hw_threshold_qm_mean",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]

    value_cols = [
        "dAmp1",

        "night_heat_exposure",
        "day_heat_exposure",
        "total_heat_exposure",

        "night_cdh",
        "day_cdh",
        "total_cdh",

        "night_cdh_urban",
        "night_cdh_rural",
        "day_cdh_urban",
        "day_cdh_rural",
        "total_cdh_urban",
        "total_cdh_rural",

        "night_cooling_energy",
        "day_cooling_energy",
        "total_cooling_energy",
        "commercial_cooling_energy",
        "residential_cooling_energy",

        "commercial_cdh",
        "residential_cdh",

        "dTMAX_cdh_daily",
        "dTMIN_cdh_daily",
        "dTMEAN_cdh_daily",
        "CDH_asym_daily",
        "Energy_asym_daily",
    ]
    value_cols = [c for c in value_cols if c in df.columns]

    # QC hourly counts if available.
    qc_cols = [
        "n_valid_u", "n_valid_r",
        "n_day_h_u", "n_day_h_r",
        "n_night_h_u", "n_night_h_r",
        "n_comm_h_u", "n_comm_h_r",
        "n_resi_h_u", "n_resi_h_r",
    ]
    qc_cols = [c for c in qc_cols if c in df.columns]
    value_cols += qc_cols

    agg_dict = {c: "mean" for c in value_cols}

    for c in meta_cols:
        agg_dict[c] = "first"

    if "local_date" in df.columns:
        agg_dict["local_date"] = "nunique"

    out = (
        df.groupby(["pair_id", "period_norm"], as_index=False, dropna=False)
        .agg(agg_dict)
    )

    if "local_date" in out.columns:
        out = out.rename(columns={"local_date": "n_days"})
    else:
        n_df = (
            df.groupby(["pair_id", "period_norm"], as_index=False)
            .size()
            .rename(columns={"size": "n_days"})
        )
        out = out.merge(n_df, on=["pair_id", "period_norm"], how="left")

    out["theta_cooling"] = T_COOL
    out["fig4_cdh_day_hours"] = ",".join([str(h) for h in DAY_HOURS])
    out["fig4_cdh_night_hours"] = ",".join([str(h) for h in NIGHT_HOURS])
    out["fig4_energy_comm_hours"] = ",".join([str(h) for h in COMM_HOURS])
    out["fig4_energy_resi_hours"] = ",".join([str(h) for h in RESI_HOURS])
    out["cdh_unit"] = "degC h day-1"
    out["energy_unit"] = "Wh day-1 equivalent, urban-rural contrast"

    front_cols = [
        "pair_id", "period_norm", "group", "strata", "density",
        "dAmp1",

        "night_heat_exposure",
        "day_heat_exposure",
        "total_heat_exposure",

        "night_cdh",
        "day_cdh",
        "total_cdh",

        "night_cooling_energy",
        "day_cooling_energy",
        "total_cooling_energy",

        "commercial_cooling_energy",
        "residential_cooling_energy",

        "n_days",
        "theta_cooling",
        "cdh_unit",
        "energy_unit",
    ]
    front_cols = [c for c in front_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in front_cols]
    out = out[front_cols + other_cols]

    return out


def _fig4_cdh_make_paired_diffs(pair_period_df):
    """
    Convert CDH pair-period table into HW − NHW paired differences.

    Sign conventions:
        ddAmp_cdh       = dAmp1_HW − dAmp1_NHW
        ampDamping_cdh  = −ddAmp_cdh

        dNightHeat_cdh  = night_heat_exposure_HW − night_heat_exposure_NHW
        dDayHeat_cdh    = day_heat_exposure_HW − day_heat_exposure_NHW
        dayRelief_cdh   = −dDayHeat_cdh

        dNightCDH       = night_cdh_HW − night_cdh_NHW
        dNightCoolingEnergy = night_cooling_energy_HW − night_cooling_energy_NHW
    """

    for c in ["pair_id", "period_norm"]:
        if c not in pair_period_df.columns:
            raise ValueError(f"[Fig4 CDH] Missing required column: {c}")

    hw = pair_period_df[pair_period_df["period_norm"] == "HW"].copy()
    nhw = pair_period_df[pair_period_df["period_norm"] == "NHW"].copy()

    if len(hw) == 0 or len(nhw) == 0:
        raise ValueError("[Fig4 CDH] HW or NHW rows missing.")

    merged = hw.merge(
        nhw,
        on="pair_id",
        how="inner",
        suffixes=("_HW", "_NHW")
    )

    if len(merged) == 0:
        raise ValueError("[Fig4 CDH] No matched HW × NHW pair rows.")

    out = pd.DataFrame()
    out["pair_id"] = merged["pair_id"]

    # Metadata from HW side.
    for c in [
        "group", "strata", "density",
        "kg_code", "kg_group", "climate_zone_main",
        "lat_group", "continent",
        "lat_urban", "lon_urban",
        "hw_ref_mode", "bias_correction_method",
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
    out["ddAmp_cdh"] = diff("dAmp1")
    out["ampDamping_cdh"] = -out["ddAmp_cdh"]

    # Exposure redistribution.
    out["dNightHeat_cdh"] = diff("night_heat_exposure")
    out["dDayHeat_cdh"] = diff("day_heat_exposure")
    out["dayRelief_cdh"] = -out["dDayHeat_cdh"]

    out["dTotalHeat_cdh"] = diff("total_heat_exposure")

    # CDH endpoint.
    out["dNightCDH"] = diff("night_cdh")
    out["dDayCDH"] = diff("day_cdh")
    out["dTotalCDH"] = diff("total_cdh")

    # Cooling energy endpoint.
    out["dNightCoolingEnergy"] = diff("night_cooling_energy")
    out["dDayCoolingEnergy"] = diff("day_cooling_energy")
    out["dTotalCoolingEnergy"] = diff("total_cooling_energy")

    out["dCommercialCoolingEnergy"] = diff("commercial_cooling_energy")
    out["dResidentialCoolingEnergy"] = diff("residential_cooling_energy")

    out["dCommercialCDH"] = diff("commercial_cdh")
    out["dResidentialCDH"] = diff("residential_cdh")

    # Diagnostics.
    out["dTMAX_cdh"] = diff("dTMAX_cdh_daily")
    out["dTMIN_cdh"] = diff("dTMIN_cdh_daily")
    out["dTMEAN_cdh"] = diff("dTMEAN_cdh_daily")

    out["dCDH_asym"] = diff("CDH_asym_daily")
    out["dEnergy_asym"] = diff("Energy_asym_daily")

    # Raw urban/rural CDH changes, useful for diagnostics.
    out["dNightCDH_urban"] = diff("night_cdh_urban")
    out["dNightCDH_rural"] = diff("night_cdh_rural")
    out["dDayCDH_urban"] = diff("day_cdh_urban")
    out["dDayCDH_rural"] = diff("day_cdh_rural")

    # QC.
    if "n_days_HW" in merged.columns:
        out["n_days_HW"] = merged["n_days_HW"]
    if "n_days_NHW" in merged.columns:
        out["n_days_NHW"] = merged["n_days_NHW"]

    out["valid_fig4_cdh"] = (
        np.isfinite(out["dNightCDH"])
        & np.isfinite(out["dNightCoolingEnergy"])
    )

    # Amplitude-damping regime, CDH-local version.
    valid_amp = out["ampDamping_cdh"].dropna().values
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

        out["regime_cdh"] = out["ampDamping_cdh"].apply(_regime)
        out["ampDamping_q33_cdh"] = q33
        out["ampDamping_q67_cdh"] = q67
    else:
        out["regime_cdh"] = "Weak change"
        out["ampDamping_q33_cdh"] = np.nan
        out["ampDamping_q67_cdh"] = np.nan

    return out


def _fig4_cdh_write_schema(out_dir):
    """Write Figure 4 CDH export schema."""
    schema_path = os.path.join(out_dir, "fig4_cdh_schema.txt")

    lines = [
        "Figure 4 CDH / cooling-energy data export",
        "=" * 72,
        "",
        "Files:",
        "  fig4_cdh_pair_period.csv",
        "  fig4_cdh_paired_diffs.csv",
        "",
        "pair-period level:",
        "  one row = pair_id x period_norm",
        "  period_norm = HW or NHW",
        "  metrics are averaged across days within each period",
        "",
        "paired-difference level:",
        "  one row = pair_id",
        "  all differences are HW minus NHW unless explicitly stated otherwise",
        "",
        "Core definitions:",
        "",
        "  dAmp1 = (Tmax_U - Tmin_U) - (Tmax_R - Tmin_R)",
        "        = dTmax - dTmin",
        "      unit: degC",
        "",
        "  CDH_h = max(Ta_h - T_COOL, 0)",
        f"      T_COOL = {T_COOL} degC",
        "",
        "  night_heat_exposure = dCDH_night",
        "      = CDH_night_U - CDH_night_R",
        "      unit: degC h day-1",
        "",
        "  day_heat_exposure = dCDH_day",
        "      = CDH_day_U - CDH_day_R",
        "      unit: degC h day-1",
        "",
        "  night_cooling_energy = dE_night",
        "      = E_night_U - E_night_R",
        "      unit: Wh day-1 equivalent, urban-rural contrast",
        "",
        "  ddAmp_cdh = dAmp1_HW - dAmp1_NHW",
        "  ampDamping_cdh = -ddAmp_cdh",
        "",
        "  dNightHeat_cdh = night_heat_exposure_HW - night_heat_exposure_NHW",
        "  dDayHeat_cdh = day_heat_exposure_HW - day_heat_exposure_NHW",
        "  dayRelief_cdh = -dDayHeat_cdh",
        "",
        "  dNightCDH = night_cdh_HW - night_cdh_NHW",
        "  dNightCoolingEnergy = night_cooling_energy_HW - night_cooling_energy_NHW",
        "",
        "Hours:",
        f"  DAY_HOURS = {DAY_HOURS}",
        f"  NIGHT_HOURS = {NIGHT_HOURS}",
        f"  COMM_HOURS = {COMM_HOURS}",
        f"  RESI_HOURS = {RESI_HOURS}",
        "",
        "Interpretation:",
        "  dNightCDH > 0 means HW increases urban-relative nighttime cooling degree hours.",
        "  dNightCoolingEnergy > 0 means HW increases urban-relative nighttime cooling energy demand.",
        "  dayRelief_cdh > 0 means HW reduces urban-relative daytime CDH burden.",
        "",
    ]

    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return schema_path


def export_figure4_cdh_data(
    daily_panel,
    output_dir=FIG4_DATA_DIR,
):
    """
    Independent export function for Figure 4 CDH / cooling-energy pathway data.

    This writes Figure 4-ready datasets to:
        <UNIFIED_ROOT>/shared/fig4_data

    It does not modify the existing CDH / cooling-trap outputs.
    """

    ensure_dir(output_dir)

    print("\n" + "═" * 72)
    print("  [Figure 4] Exporting CDH / cooling-energy pathway data")
    print(f"  Output directory: {output_dir}")
    print("═" * 72)

    pair_period = _fig4_cdh_build_pair_period_table(daily_panel)

    pair_path = os.path.join(output_dir, "fig4_cdh_pair_period.csv")
    pair_period.to_csv(pair_path, index=False)
    print(f"  Saved: {pair_path}")
    print(f"    rows = {len(pair_period)}")
    print(f"    HW rows = {(pair_period['period_norm'] == 'HW').sum()}")
    print(f"    NHW rows = {(pair_period['period_norm'] == 'NHW').sum()}")

    paired_diffs = _fig4_cdh_make_paired_diffs(pair_period)

    diff_path = os.path.join(output_dir, "fig4_cdh_paired_diffs.csv")
    paired_diffs.to_csv(diff_path, index=False)
    print(f"  Saved: {diff_path}")
    print(f"    paired rows = {len(paired_diffs)}")
    print(f"    valid_fig4_cdh = {paired_diffs['valid_fig4_cdh'].sum()}")

    schema_path = _fig4_cdh_write_schema(output_dir)
    print(f"  Saved: {schema_path}")

    # Console diagnostics.
    for col, label in [
        ("dNightCDH", "dNightCDH"),
        ("dNightCoolingEnergy", "dNightCoolingEnergy"),
        ("dDayCDH", "dDayCDH"),
        ("dayRelief_cdh", "dayRelief_cdh"),
    ]:
        if col in paired_diffs.columns:
            vals = paired_diffs[col].dropna()
            if len(vals) > 0:
                print(
                    f"  {label} summary "
                    f"(HW - NHW): mean={vals.mean():+.3f}, "
                    f"median={vals.median():+.3f}, n={len(vals)}"
                )

    print("  [Figure 4] CDH export complete.\n")

    return {
        "pair_period_path": pair_path,
        "paired_diffs_path": diff_path,
        "schema_path": schema_path,
        "pair_period": pair_period,
        "paired_diffs": paired_diffs,
    }

# ══════════════════════════════════════════════════════════════
# §13  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("="*70)
    print("  Compute: CDH / HDH / COP / Energy  (Raw Hourly ISD)")
    print(f"  T_COOL={T_COOL}°C  T_HEAT={T_HEAT}°C")
    print(f"  COP(26°C)={cop(np.array([26.]))[0]:.3f}  COP(35°C)={cop(np.array([35.]))[0]:.3f}")
    print(f"  UA (COMM): HRHD={UA[('COMM','HRHD')]}  LRLD={UA[('COMM','LRLD')]}  OTHER={UA[('COMM','OTHER')]} W K⁻¹")
    print(f"  UA (RESI): HRHD={UA[('RESI','HRHD')]}  LRLD={UA[('RESI','LRLD')]}  OTHER={UA[('RESI','OTHER')]} W K⁻¹")
    print(f"  HW定义: {HW_REF_MODE}")
    print(f"  ERA5 气候态: {'ENABLED' if USE_ERA5_CLIMATOLOGY else 'DISABLED (ISD fallback)'}"
          f"  [USE_ERA5_CLIMATOLOGY={USE_ERA5_CLIMATOLOGY}]")
    print(f"  Years: {YEARS[0]}–{YEARS[-1]}")
    print("="*70)

    ensure_dir(OUTPUT_DIR)

    # ─────────────────────────────────────────────
    # Sub-directories for clean outputs
    # ─────────────────────────────────────────────
    FIG_DIR = os.path.join(OUTPUT_DIR, "figure_cooling_trap")
    FIG_CSV_DIR = os.path.join(FIG_DIR, "csv")
    FIG_FIG_DIR = os.path.join(FIG_DIR, "figures")

    ensure_dir(FIG_DIR)
    ensure_dir(FIG_CSV_DIR)
    ensure_dir(FIG_FIG_DIR)

    if not os.path.exists(PAIR_CSV):
        print(f"\n⚠  PAIR_CSV 不存在: {PAIR_CSV}")
        return

    # ── 读入站对表 ────────────────────────────────────────────
    pair_df = pd.read_csv(PAIR_CSV)
    print(f"\n站对总数: {len(pair_df)}")

    # [M1] 附加 KG 气候区字段（基于城市站经纬度）
    print("  正在提取 Köppen–Geiger 气候区（城市站坐标）...")
    pair_df = add_climate_zone_columns(pair_df, lon_col="lon_urban", lat_col="lat_urban")

    # [M1] 写入热浪版本标记（与主分析对齐，方便溯源）
    pair_df["hw_ref_mode"] = HW_REF_MODE

    # 打印气候区分布
    if "climate_zone_main" in pair_df.columns:
        kg_dist = pair_df["climate_zone_main"].value_counts()
        print("  KG 气候区分布（按城市站）:")
        for kz, cnt in kg_dist.items():
            print(f"    {kz:<12} {cnt:>4d} 对")

    combined, profiles = process_all(pair_df)

    if combined is None:
        return

    export_building_type_figure_csv(
        daily_panel=combined,
        output_dir=OUTPUT_DIR
    )

    export_figure4_cdh_data(
        daily_panel=combined,
        output_dir=FIG4_DATA_DIR,
    )

    FIG_CSV_DIR = os.path.join(OUTPUT_DIR, "figure_cooling_trap", "csv")
    ensure_dir(FIG_CSV_DIR)

    pair_hourly_path = os.path.join(FIG_CSV_DIR, "pair_hourly_cdh_profiles.csv")

    if os.path.exists(pair_hourly_path):
        hourly_profiles_pair = pd.read_csv(pair_hourly_path)

        export_cooling_trap_figure_csv(
            daily_panel=combined,
            hourly_profiles_pair=hourly_profiles_pair,
            output_dir=FIG_CSV_DIR,
            mechanism_csv=None
        )
    else:
        print("  ⚠ pair_hourly_cdh_profiles.csv 缺失，跳过 Figure cooling-trap CSV 导出")


    pair_hourly_path = os.path.join(FIG_CSV_DIR, "pair_hourly_cdh_profiles.csv")

    if os.path.exists(pair_hourly_path):
        hourly_profiles_pair = pd.read_csv(pair_hourly_path)
    else:
        print("  ⚠ hourly_profiles_pair_raw.csv 缺失，跳过 Figure cooling-trap CSV 导出")
        hourly_profiles_pair = pd.DataFrame()

    mechanism_csv = None

    if len(hourly_profiles_pair) > 0:
        export_cooling_trap_figure_csv(
            daily_panel=combined,
            hourly_profiles_pair=hourly_profiles_pair,
            output_dir=FIG_CSV_DIR,
            mechanism_csv=mechanism_csv
        )


    summary = build_summary(combined)
    print_summary(combined)

    elapsed = time.time() - t_start
    print(f"\n✓ 输出目录: {OUTPUT_DIR} (总耗时: {elapsed/60:.1f} 分钟)")
    print("  all_pairs_daily_panel.csv          ← 含 kg_code / kg_group / climate_zone_main / hw_ref_mode")
    print("  hourly_profiles.csv")
    print("  hourly_profiles_by_kggroup.csv     ← 新增：KG × 时期 × 小时廓线")
    print("  period_summary.csv")
    print("  period_summary_by_kggroup.csv      ← 新增：KG × 时期聚合统计")
    print("  pair_period_summary.csv            ← 含 KG 字段")
    print("  daily/{pair_id}_daily.csv")
    print("  skipped_pairs_log.csv")


if __name__ == "__main__":
    main()

