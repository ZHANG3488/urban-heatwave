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

热浪 / UHI 热暴露风险 (Heat Exposure Risk Index, HERI)

heatwave_rr_isd.py  ── CPU 加速版
══════════════════════════════════════════════════════════════════════════════
热浪 / UHI 健康风险：基于 ISD 真实逐小时数据的城乡 Heat Exposure Risk Index 分析

加速策略（工作站多核）
──────────────────────
  [A] HW 阈值计算：DOY 查找表（365 次 numpy 循环）替代 N_days 次 pandas 循环
      原版: O(N_days) × pandas 布尔过滤 ≈ 3650 次/站
      新版: O(365)    × numpy 切片       ≈  365 次/站  → ~10× 加速

  [B] 热浪检测：numpy diff + 连续段扫描，替代 Python while 循环
      基准测试：0.065 ms/站（原版 ~0.8 ms/站）→ ~12× 加速

  [C] Bootstrap CI：全向量化 numpy（消除 Python for 循环）
      预分配 (N_boot, N_yrs, max_days) 三维数组 → nanmean 单次 ufunc
      原版: 1000 次 Python 循环/时期
      新版: 1 次 numpy ufunc       → ~20–50× 加速

  [D] 外层对站循环：joblib.Parallel（loky 后端）+ 动态 chunk
      自动使用全部 CPU 核心（默认 cpu_count - 1）

  [E] numba 可选加速（若工作站已安装 pip install numba）
      @njit(parallel=True, cache=True) 自动向量化 HW 内层循环和 bootstrap
      安装后无需修改代码，自动启用

  [F] DOY 预计算：lru_cache 缓存 (month,day)→DOY 映射（最多 366 个唯一值）

Köppen-Geiger 气候分区数据来源
──────────────────────────────
  Beck, H. E., McVicar, T. R., Vergopolan, N., Berg, A., Lutsko, N. J.,
  Dufour, A., Zeng, Z., Jiang, X., van Dijk, A. I. J. M., & Miralles, D. G.
  (2023). High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on
  constrained CMIP6 projections. Scientific Data, 10, 724.
  https://doi.org/10.1038/s41597-023-02549-6

  字段说明：
    kg_code        : 采样自 TIF 的 Köppen-Geiger 细分代码（如 "Cfb"）；
                     TIF 采样失败时为 NaN（不再默认填 "Cfb"）。
    kg_code_source : "tif"          → 成功从 KG_TIF 读取
                     "lat_fallback" → TIF 读取失败，回退至纬度估算
                     注意：lat_fallback 时 kg_code = NaN，kg_group 由
                     纬度粗估得到，不应视为真实 Köppen 分类。
    kg_group       : 首字母大类（A/B/C/D/E），用于查 β_heat；
                     来源于 kg_code（TIF 成功）或 _lat2kg（fallback）。

依赖：pip install pandas numpy scipy tqdm joblib rasterio
可选：pip install numba   # 进一步加速 ~3-5×
══════════════════════════════════════════════════════════════════════════════
"""

import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft as sp_fft
from scipy.stats import norm as sp_norm
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# ★ 配置
# ════════════════════════════════════════════════════════════════════════
PAIR_CSV   = str(_OPEN_SOURCE_PATHS.pair_csv)
ISD_DIR    = str(_OPEN_SOURCE_PATHS.isd_dir)
KG_TIF     = str(_OPEN_SOURCE_PATHS.kg_tif)
OUTPUT_DIR = Path(_OPEN_SOURCE_PATHS.mortality_output)

YEARS = list(range(2015, 2025))

MAX_CONSEC_NAN              = 2
MAX_INTERP_GAP              = 2   # 小缺口插值上限（1–2都可，3-hourly建议至少2）
MIN_OBS_SUPPORT_HOURS       = 6   # 原始观测支撑下限：兼容3-hourly
MIN_USABLE_HOURS            = 18  # 重采样+小缺口插值后，当日可用小时下限
FULL_YEAR_MIN_VALID_FRAC    = 0.80
# Backward-compatible nominal value only.
# The actual expected warm-season days are calculated by hemisphere and season year.
WARM_SEASON_DAYS_EXPECTED = 92
MIN_YEAR_VALID_FRAC       = 0.80

HW_PERCENTILE  = 90
HW_MIN_DAYS    = 3
HW_WINDOW_HALF = 7

# Hemisphere-aware warm season, synced with analysis_multiyear.py:
#   Northern Hemisphere: JJA = Jun-Jul-Aug
#   Southern Hemisphere: DJF = Dec-Jan-Feb
NH_WARM_MONTHS = [6, 7, 8]
SH_WARM_MONTHS = [12, 1, 2]
WARM_MONTHS    = NH_WARM_MONTHS  # backward-compatible default; do not use for pair-level filtering

# analysis_multiyear.py writes this file. This script reads it first.
ANALYSIS_HW_FLAGS_CSV = str(_OPEN_SOURCE_PATHS.daily_hw_flags_csv)

N_HARMONICS = 2
DAY_HOURS   = list(range(8, 20))                         # 08:00–19:59, 12 h
NIGHT_HOURS = list(range(20, 24)) + list(range(0, 8))   # 20:00–07:59, 12 h

TMM_PERCENTILE = 75
BOOT_N         = 1000
RANDOM_SEED    = 42
CI_LEVEL       = 0.95

# ── 并行配置 ──────────────────────────────────────────────────────────
N_WORKERS      = 0
JOBLIB_BACKEND = "loky"
JOBLIB_CHUNK   = 0

# ── ERA5-MOD：ERA5 气候态热浪阈值开关及工具常量 ─────────────────────
USE_ERA5_CLIMATOLOGY = True   # ← 默认开启；改为 False 还原为 ISD 短期估计

USE_QM_BIAS_CORRECTION = True
USE_MONTHLY_MEAN_BIAS_CORRECTION = False

QM_N_QUANTILES = 1001
QM_MIN_OVERLAP_DAYS_PER_MONTH = 30
QM_MIN_OVERLAP_DAYS_ANNUAL = 100


ERA5_STATION_DIR = str(_OPEN_SOURCE_PATHS.era5_station_dir)
# ERA5 站点 Tmax CSV 命名约定：{USAF}-{WBAN}_{station_type}_tmax.csv
# 必须包含 "date"（YYYY-MM-DD）和 "tmax"（℃）两列

MIN_LOYO_REF_YEARS = 5
# ERA5 序列最短有效年数（×30 天 = 最低观测量），低于此阈值回退 ISD

def load_era5_tmax_series(
        usaf: str,
        wban: str,
        station_type: str = "urban") -> "pd.Series | None":
    """
    读取站点对应 ERA5 逐日 Tmax 序列。

    文件名统一为：
        {USAF}_{WBAN}_{urban/rural}.csv

    列名优先级：
        1) tmax_c
        2) tmax

    Returns
    -------
    pd.Series(index=date, dtype=float)
    None 表示 ERA5 文件缺失、列名不对、日期解析失败或数据不足。
    """
    fpath = os.path.join(
        ERA5_STATION_DIR,
        f"{str(usaf).strip()}_{str(wban).strip()}_{station_type}.csv"
    )

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
        df = df.dropna(subset=["date"])

        if df.empty:
            return None

        s = pd.to_numeric(df[tmax_col], errors="coerce")
        out = pd.Series(s.values, index=df["date"]).dropna().sort_index()

        if len(out) < MIN_LOYO_REF_YEARS * 30:
            return None

        return out

    except Exception:
        return None

def compute_hw_doy_thr_era5(
        era5_tmax: pd.Series) -> dict:
    """
    基于 ERA5 长序列 Tmax，计算每个 DOY 的热浪温度阈值（P_q，滑动窗口）。

    本函数直接引用模块级常量 HW_PERCENTILE / HW_WINDOW_HALF，
    与 analysis_multiyear 及 compute_cdh_hdh_energy 中同名函数保持一致。

    Returns
    -------
    dict  {doy (1–365): threshold_°C}
    """
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
    QM bias correction for ERA5-derived HW thresholds.

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

def load_era5_tmean_series(
        usaf: str, wban: str,
        station_type: str = "urban") -> "pd.Series | None":
    """
    读取站点 ERA5 逐日 Tmean，专用于计算气候态 μ/σ（β 反推）。

    优先加载同目录下 tmin 文件，取 (Tmax+Tmin)/2；
    仅有 tmax 时以 Tmax 代替（σ 与 Tmean 相近，μ 偏高约 3–5°C，
    可接受用于气候带内相对比较，不推荐跨气候带绝对比较）。
    ERA5 通常覆盖 30 年参考期，比 ISD 10 年更稳定，优先使用。

    Returns
    -------
    pd.Series（index=date，dtype=float）或 None
    """
    def _load_col(col: str) -> "pd.Series | None":
        fpath = os.path.join(
            ERA5_STATION_DIR,
            f"{usaf.strip()}-{wban.strip()}_{station_type}_{col}.csv"
        )
        if not os.path.exists(fpath):
            return None
        try:
            df = pd.read_csv(fpath, parse_dates=["date"], index_col="date")
            if col not in df.columns:
                return None
            return pd.to_numeric(df[col], errors="coerce").dropna()
        except Exception:
            return None

    s_max = _load_col("tmax")
    s_min = _load_col("tmin")
    if s_max is None:
        return None
    if s_min is not None:
        idx = s_max.index.intersection(s_min.index)
        if len(idx) < MIN_LOYO_REF_YEARS * 30:
            return None
        return (s_max.loc[idx] + s_min.loc[idx]) / 2.0
    # Tmax only fallback
    if len(s_max) < MIN_LOYO_REF_YEARS * 30:
        return None
    return s_max

# ── ERA5-MOD 工具函数结束 ────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════
# §0  numba 可选加速
# ════════════════════════════════════════════════════════════════════════
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def _wrap(f): return f
        return _wrap
    prange = range


# ════════════════════════════════════════════════════════════════════════
# §1  Zhao 2021 Table S3 EDF → β_heat 严格反推
# ════════════════════════════════════════════════════════════════════════
_EDF_HEAT = {
    "A": ( 0.0054,  0.0000,  0.0117),
    "B": ( 0.0094,  0.0040,  0.0152),
    "C": ( 0.0093,  0.0042,  0.0146),
    "D": ( 0.0144,  0.0069,  0.0224),
    "E": ( 0.1524, -0.0561,  0.3705),
}
_CLIM_PARAMS = {
    "A": (27.0, 1.5), "B": (20.0, 4.0), "C": (14.0, 3.0),
    "D": ( 7.0, 4.5), "E": (-5.0, 5.0),
}
# Tobías et al. (2021) Environ Epidemiol 5:e169 – Figure 4
# 气候带分层 MMTP（最低死亡温度所在百分位），替代固定 TMM_PERCENTILE=75
# 回退顺序（MMTP 未知时）：_MMTP_BY_KOPPEN → TMM_PERCENTILE
_MMTP_BY_KOPPEN = {
    "A": 58.5,   # Tropical    (n=99,  I²=42.2%)
    "B": 68.0,   # Arid/Dry    (n=64,  I²=84.6%)
    "C": 79.5,   # Temperate   (n=379, I²=67.7%)
    "D": 75.4,   # Continental (n=112, I²=58.1%)
    "E": 41.4,   # Alpine/Polar(n=4,   I²=15.5%)
}

def _edf_to_beta(edf, mu_T, sigma_T, tmm_pct=TMM_PERCENTILE/100):
    if edf <= 0 or sigma_T <= 0: return 0.0
    tmm   = mu_T + sp_norm.ppf(tmm_pct) * sigma_T
    z0    = (tmm - mu_T) / sigma_T
    E_exc = sigma_T * sp_norm.pdf(z0) + (mu_T - tmm) * sp_norm.cdf(-z0)
    return edf / E_exc if E_exc > 1e-8 else 0.0

def _build_heat_beta():
    out = {}
    for z, (c, lo, hi) in _EDF_HEAT.items():
        mu, sig = _CLIM_PARAMS[z]
        tmm_pct = _MMTP_BY_KOPPEN.get(z, TMM_PERCENTILE) / 100  # Tobías 2021
        out[z] = (_edf_to_beta(c, mu, sig, tmm_pct),
                  _edf_to_beta(max(lo, 0.0), mu, sig, tmm_pct),
                  _edf_to_beta(hi, mu, sig, tmm_pct))
    return out

HEAT_BETA: dict = _build_heat_beta()

def build_beta_calibration_table() -> pd.DataFrame:
    rows = []
    for z, (c, lo, hi) in _EDF_HEAT.items():
        mu, sig = _CLIM_PARAMS[z]
        pct     = _MMTP_BY_KOPPEN.get(z, TMM_PERCENTILE) / 100  # Tobías 2021
        tmm     = mu + sp_norm.ppf(pct) * sig
        z0      = (tmm - mu) / sig
        E_exc   = sig * sp_norm.pdf(z0) + (mu - tmm) * sp_norm.cdf(-z0)
        b_c, b_lo, b_hi = HEAT_BETA[z]
        rows.append({"koppen_group": z,
                     "EDF_heat_central": c, "EDF_heat_lo95": lo, "EDF_heat_hi95": hi,
                     "mu_T_ERA5": mu, "sigma_T_ERA5": sig,
                     "TMM_degC": round(tmm, 3), "z0": round(z0, 4),
                     "E_excess_T": round(E_exc, 6),
                     "beta_central": round(b_c, 8),
                     "beta_lo95":    round(b_lo, 8),
                     "beta_hi95":    round(b_hi, 8),
                     "EDF_check":    round(b_c * E_exc, 6)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# §2  Köppen 编码
# ════════════════════════════════════════════════════════════════════════
KG_INT2CODE = {
    1:"Af",  2:"Am",  3:"Aw",
    4:"BWh", 5:"BWk", 6:"BSh", 7:"BSk",
    8:"Csa", 9:"Csb",10:"Csc",
    11:"Cwa",12:"Cwb",13:"Cwc",
    14:"Cfa",15:"Cfb",16:"Cfc",
    17:"Dsa",18:"Dsb",19:"Dsc",20:"Dsd",
    21:"Dwa",22:"Dwb",23:"Dwc",24:"Dwd",
    25:"Dfa",26:"Dfb",27:"Dfc",28:"Dfd",
    29:"ET", 30:"EF",
}

def kg_group(code) -> str:
    """
    Köppen 细分代码 → 大类首字母（A/B/C/D/E）。
    code 为 NaN 或无法识别时返回 "C"（作为保守默认，
    调用方应同时检查 kg_code_source 以判断是否为 fallback）。
    """
    if not isinstance(code, str) or not code.strip(): return "C"
    g = code.strip()[0].upper()
    return g if g in "ABCDE" else "C"

def _lat2kg(lat: float) -> str:
    """纬度粗估 Köppen 细分代码，仅在 TIF 读取失败时用于推算 kg_group。"""
    a = abs(lat)
    if a < 15:  return "Af"
    if a < 25:  return "BSh"
    if a < 35:  return "Cfa"
    if a < 50:  return "Cfb"
    if a < 66:  return "Dfb"
    return "ET"

def extract_koppen(lons, lats) -> tuple:
    """
    从 KG_TIF 采样 Köppen-Geiger 气候代码。

    返回值
    ------
    codes   : list，长度 == len(lons)
              TIF 采样成功 → str（如 "Cfb"）；
              采样失败（像元为 NaN 或整体 TIF 读取异常） → np.nan。
              注：不再以 "Cfb" 作为 fallback 填充，避免 fallback 被误当真值。
    sources : list[str]，长度 == len(lons)
              "tif"          → 成功从 Beck et al. (2023) KG TIF 读取
              "lat_fallback" → TIF 读取失败，回退至纬度粗估（kg_group 由
                               _lat2kg 独立计算，kg_code 保持 NaN）

    数据来源：Beck et al. (2023), Sci. Data, https://doi.org/10.1038/s41597-023-02549-6
    """
    try:
        import rasterio
        from rasterio.sample import sample_gen
        with rasterio.open(KG_TIF) as src:
            sampled = [v[0] for v in sample_gen(src, list(zip(lons, lats)))]

        codes, sources = [], []
        for v in sampled:
            try:
                fv = float(v)
                if np.isnan(fv):
                    codes.append(np.nan)
                    sources.append("lat_fallback")
                else:
                    code = KG_INT2CODE.get(int(fv), None)
                    if code is not None:
                        codes.append(code)
                        sources.append("tif")
                    else:
                        codes.append(np.nan)
                        sources.append("lat_fallback")
            except Exception:
                codes.append(np.nan)
                sources.append("lat_fallback")
        return codes, sources

    except Exception as e:
        print(f"  [!] Köppen TIF 读取失败：{e}，全部回退至纬度粗估")
        return [np.nan] * len(lats), ["lat_fallback"] * len(lats)


# ════════════════════════════════════════════════════════════════════════
# §3  工具函数
# ════════════════════════════════════════════════════════════════════════
def is_leap_year(yr): return (yr%4==0) and (yr%100!=0 or yr%400==0)


# ════════════════════════════════════════════════════════════════════════
# HW-sync helpers: hemisphere-aware warm season + analysis_multiyear flags
# ════════════════════════════════════════════════════════════════════════
_ANALYSIS_HW_FLAGS_CACHE = None


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
    """
    Load daily_heatwave_flags.csv generated by analysis_multiyear.py.

    Main columns expected:
      pair_id, date, is_warm_season,
      hw_flag_percentile_warm_season,
      nhw_flag_percentile_warm_season
    """
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
    """Return pair-level HW flags from analysis_multiyear.py, if available."""
    df = _load_analysis_hw_flags()
    if df is None or len(df) == 0:
        return None

    sub = df[df["pair_id"].astype(str) == str(pair_id)].copy()
    if sub.empty:
        return None

    sub = sub.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return sub


def align_analysis_hw_flags(analysis_hw_flags, date_index):
    """
    Align external daily HW flags to a target local_date index.

    The returned DataFrame uses the original date_index as its index, so it can
    be converted back to the caller's original date objects without type drift.
    """
    if analysis_hw_flags is None or len(analysis_hw_flags) == 0:
        return None

    idx_orig = pd.Index(date_index)
    idx_norm = pd.to_datetime(idx_orig, errors="coerce").normalize()

    ext = analysis_hw_flags.copy()
    ext.index = pd.to_datetime(ext.index, errors="coerce").normalize()

    aligned = ext.reindex(idx_norm)
    aligned.index = idx_orig

    required = ["is_warm_season", "hw_flag_percentile_warm_season"]
    if any(c not in aligned.columns for c in required):
        return None

    if aligned[required].dropna(how="all").empty:
        return None

    return aligned

def parse_pair_id(pair_id):
    parts = pair_id.split("__")
    if len(parts) != 2: raise ValueError(f"Cannot parse: {pair_id}")
    u = parts[0].strip().split("_"); r = parts[1].strip().split("_")
    if len(u) < 2 or len(r) < 2: raise ValueError(f"Cannot parse: {pair_id}")
    return u[0], u[1], r[0], r[1]

def assign_lat_group(lat):
    if pd.isna(lat): return "Unknown"
    if abs(lat) < 23.5: return "Tropical"
    if abs(lat) < 40:   return "Subtropical"
    if abs(lat) < 60:   return "Temperate"
    return "Polar"

def normalize_lcz(val):
    try:
        v = float(val)
        return int(v - 50) if 51 <= v <= 56 else np.nan
    except Exception: return np.nan

def lcz_compactness_class(lcz):
    if pd.isna(lcz): return np.nan
    return "compact" if int(lcz) in [1,2,3] else ("open" if int(lcz) in [4,5,6] else np.nan)

@lru_cache(maxsize=400)
def _md2doy(m: int, d: int) -> int:
    if m == 2 and d == 29: return 59
    return pd.Timestamp(2001, m, d).dayofyear

def _dates_to_doy_array(dates_index) -> np.ndarray:
    dti = pd.DatetimeIndex(pd.to_datetime(dates_index))
    return np.array([_md2doy(int(m), int(d))
                     for m, d in zip(dti.month, dti.day)], dtype=np.int32)


# ════════════════════════════════════════════════════════════════════════
# §4  ISD-Lite 读取
# ════════════════════════════════════════════════════════════════════════
def isd_path(usaf, wban, year):
    return os.path.join(ISD_DIR, str(year), f"{usaf}-{wban}-{year}.gz")

def read_isd_lite(filepath):
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
            on_bad_lines="skip",
        )
        df["temp_C"] = df["temp_C"] / 10.0
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]], errors="coerce"
        )
        df = (
            df.dropna(subset=["datetime"])
              .drop_duplicates(subset="datetime")
              .sort_values("datetime")
              .reset_index(drop=True)
        )
        if df.empty:
            return None
        return df[["datetime", "temp_C"]]
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════
# §5  质控 + 升频 + 本地时
# ════════════════════════════════════════════════════════════════════════
def clean_and_resample(df_raw, year):
    """
    统一重采样到 hourly：
    - 不依赖原生 ts
    - 保留所有年份
    - 只补小缺口
    - 用“原始观测支撑 + 可用小时数”判定每日是否有效
    """
    t_start = pd.Timestamp(year, 1, 1, 0)
    t_end   = pd.Timestamp(year, 12, 31, 23)
    full_1h = pd.date_range(t_start, t_end, freq="1h")

    df_i = df_raw.set_index("datetime").sort_index()

    # 1) 统一聚合到 hourly
    hourly = df_i[["temp_C"]].resample("1h").mean()

    # 2) 记录原始观测支撑
    obs_hourly = (
        df_i[["temp_C"]]
        .resample("1h")
        .count()
        .reindex(full_1h)
        .fillna(0)
        .rename(columns={"temp_C": "obs_count"})
    )

    # 3) 对齐完整全年 hourly 轴
    hourly = hourly.reindex(full_1h)

    # 4) 只对小缺口插值
    hourly["temp_C"] = hourly["temp_C"].interpolate(
        method="time",
        limit=MAX_INTERP_GAP,
        limit_area="inside"
    )

    # 5) 每日双重 QC：
    #    (a) 原始观测支撑小时数 >= MIN_OBS_SUPPORT_HOURS
    #    (b) 升频后可用小时数 >= MIN_USABLE_HOURS
    obs_support = obs_hourly["obs_count"].groupby(obs_hourly.index.date).transform(
        lambda s: (s > 0).sum()
    )
    usable_hours = hourly["temp_C"].groupby(hourly.index.date).transform(
        lambda s: s.notna().sum()
    )

    bad_day_mask = (
        (obs_support < MIN_OBS_SUPPORT_HOURS) |
        (usable_hours < MIN_USABLE_HOURS)
    )
    hourly.loc[bad_day_mask, "temp_C"] = np.nan

    result = pd.DataFrame({
        "datetime": full_1h,
        "temp_C": hourly["temp_C"].values,
        "obs_count": obs_hourly["obs_count"].values,
    })
    return result


def to_local_time(df, lon):
    offset           = pd.Timedelta(hours=lon / 15.0)
    out              = df.copy()
    out["local_dt"]  = out["datetime"] + offset
    out["local_hour"]= out["local_dt"].dt.hour + out["local_dt"].dt.minute / 60.0
    out["local_date"]= out["local_dt"].dt.date
    out["local_month"]= out["local_dt"].dt.month
    return out

def load_station_allyears(usaf, wban, lon, years=YEARS,
                           min_valid_frac=FULL_YEAR_MIN_VALID_FRAC,
                           verbose=False):
    frames, valid_yrs = [], []

    for yr in years:
        df_raw = read_isd_lite(isd_path(usaf, wban, yr))
        if df_raw is None:
            if verbose:
                print(f"    [{yr}] missing")
            continue

        df_clean = clean_and_resample(df_raw, yr)
        df_loc   = to_local_time(df_clean, lon)

        n_days = 366 if is_leap_year(yr) else 365
        valid_d = df_loc.dropna(subset=["temp_C"])["local_date"].nunique()

        if valid_d / n_days < min_valid_frac:
            if verbose:
                print(f"    [{yr}] {valid_d/n_days:.1%} → skip")
            continue

        frames.append(df_loc)
        valid_yrs.append(yr)

    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), valid_yrs


def slice_warm_season(df_all, lat, min_yr_frac=MIN_YEAR_VALID_FRAC):
    """
    Hemisphere-aware warm-season slicing.

    Pair-level analyses should pass the urban-station latitude so that urban
    and rural periods use the same warm-season calendar:
      NH: JJA
      SH: DJF, with Dec assigned to the following season year.
    """
    if df_all is None:
        return None, []

    df_w = df_all[
        df_all["local_date"].apply(
            lambda d: is_warm_season_date_for_analysis(d, lat)
        )
    ].copy()
    if df_w.empty:
        return None, []

    df_w["warm_season_year"] = df_w["local_date"].apply(
        lambda d: warm_season_year_from_date(d, lat)
    )

    valid_yrs = []
    for season_year, grp in df_w.groupby("warm_season_year"):
        expected_days = expected_warm_season_days(int(season_year), lat)
        valid_days = grp.dropna(subset=["temp_C"])["local_date"].nunique()
        if valid_days / expected_days >= min_yr_frac:
            valid_yrs.append(int(season_year))

    if not valid_yrs:
        return None, []

    return df_w[df_w["warm_season_year"].isin(valid_yrs)].copy(), valid_yrs


# ════════════════════════════════════════════════════════════════════════
# §6  逐日 Tmax / Tmean
# ════════════════════════════════════════════════════════════════════════
def compute_daily_minmax(df_loc):
    if df_loc is None or len(df_loc) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = df_loc.dropna(subset=["temp_C"]).groupby("local_date")["temp_C"]
    return g.min().dropna(), g.max().dropna()

def compute_daily_tmean(df_loc):
    if df_loc is None or len(df_loc) == 0: return pd.Series(dtype=float)
    return (df_loc.dropna(subset=["temp_C"])
                  .groupby("local_date")["temp_C"].mean().dropna())


# ════════════════════════════════════════════════════════════════════════
# §7  [A] 热浪阈值：DOY 查找表
# ════════════════════════════════════════════════════════════════════════
def compute_hw_threshold_vectorized(
        tmax_series: pd.Series,
        q: int   = HW_PERCENTILE,
        window_half: int = HW_WINDOW_HALF,
) -> tuple:
    dates    = pd.to_datetime(tmax_series.index)
    vals_arr = tmax_series.values.astype(float)
    valid    = ~np.isnan(vals_arr)

    doys_all = _dates_to_doy_array(dates)
    doys_v   = doys_all[valid]
    vals_v   = vals_arr[valid]

    doy_thr = np.full(367, np.nan)
    for doy in range(1, 366):
        lo, hi = doy - window_half, doy + window_half
        if lo < 1:
            mask = (doys_v >= 365 + lo) | (doys_v <= hi)
        elif hi > 365:
            mask = (doys_v >= lo) | (doys_v <= hi - 365)
        else:
            mask = (doys_v >= lo) & (doys_v <= hi)
        pool = vals_v[mask]
        if len(pool) > 0:
            doy_thr[doy] = np.nanpercentile(pool, q)

    threshold_vals = doy_thr[doys_all]
    valid_counts   = np.array([
        int(np.sum((doys_v >= max(1, d - window_half)) &
                   (doys_v <= min(365, d + window_half))))
        for d in doys_all
    ], dtype=np.int32)

    return (
        pd.Series(threshold_vals, index=tmax_series.index),
        pd.Series(valid_counts,   index=tmax_series.index),
    )


# ════════════════════════════════════════════════════════════════════════
# §8  [B] 热浪检测：numpy diff + 向量化连续段扫描
# ════════════════════════════════════════════════════════════════════════
@njit(cache=True) 
def _detect_hw_core(starts: np.ndarray, ends: np.ndarray,
                    min_days: int, n: int) -> np.ndarray:
    hw = np.zeros(n, dtype=np.bool_)
    for i in range(len(starts)):
        if ends[i] - starts[i] >= min_days:
            hw[starts[i]:ends[i]] = True
    return hw

def detect_heatwave(daily_tmax: pd.Series, threshold,
                    min_days: int = HW_MIN_DAYS) -> pd.Series:
    if np.isscalar(threshold):
        above = (daily_tmax.values > float(threshold))
    else:
        thr_s = pd.Series(threshold).reindex(daily_tmax.index)
        above = (daily_tmax.values > thr_s.fillna(-np.inf).values)

    above = above.astype(np.bool_)
    if not above.any():
        return pd.Series(np.zeros(len(above), dtype=bool),
                         index=daily_tmax.index)

    padded = np.empty(len(above) + 2, dtype=np.bool_)
    padded[0] = padded[-1] = False
    padded[1:-1] = above
    diff   = np.diff(padded.view(np.int8))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]

    hw = _detect_hw_core(starts, ends, min_days, len(above))
    return pd.Series(hw, index=daily_tmax.index)


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


# ════════════════════════════════════════════════════════════════════════
# §9  FFT 日循环重建
# ════════════════════════════════════════════════════════════════════════
def reconstruct_diurnal(mean_t, amps, phases, n_points=24):
    t   = np.arange(n_points)
    sig = np.full(n_points, mean_t, dtype=float)
    for k, (a, p) in enumerate(zip(amps, phases), start=1):
        sig += a * np.cos(k * 2 * np.pi * t / n_points + p)
    return sig

def compute_fft(df_loc, n_harm=N_HARMONICS, min_hour_bins=8):
    """
    Canonical FFT preprocessing used by the Supplement:
    - group observations into 24 local-hour bins;
    - require at least 8 valid hourly bins;
    - fill the remaining hourly bins by linear interpolation;
    - retain the 24 h and 12 h harmonics through N_HARMONICS=2.

    No per-hour minimum-day threshold is imposed here because the upstream
    daily/year QC already controls data availability.
    """
    if df_loc is None or len(df_loc) == 0:
        return None

    df = df_loc.dropna(subset=["temp_C"]).copy()
    if len(df) == 0:
        return None

    df["h_bin"] = df["local_hour"].round().astype(int) % 24

    hourly_mean = (
        df.groupby("h_bin")["temp_C"].mean().reindex(range(24))
    )
    hourly_n = (
        df.groupby("h_bin")["temp_C"]
        .count()
        .reindex(range(24), fill_value=0)
    )

    if hourly_mean.notna().sum() < min_hour_bins:
        return None

    hourly_mean = (
        hourly_mean
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    if hourly_mean.isna().any():
        return None

    vals = hourly_mean.values.astype(float)
    coef = sp_fft(vals) / len(vals)
    mean_t = float(np.real(coef[0]))

    amps, phases = [], []
    for k in range(1, n_harm + 1):
        amps.append(float(2 * np.abs(coef[k])))
        phases.append(float(np.angle(coef[k])))

    rec = reconstruct_diurnal(mean_t, amps, phases)

    return dict(
        mean=mean_t,
        amplitudes=amps,
        phases=phases,
        Tmax_fft=float(np.max(rec)),
        Tmin_fft=float(np.min(rec)),
        Tmax_hour=int(np.argmax(rec)),
        Tmin_hour=int(np.argmin(rec)),
        hourly_obs=vals,
        hourly_reconstructed=rec,
        hourly_n=hourly_n.values.astype(int),
        n_days=int(df["local_date"].nunique()),
    )

# ════════════════════════════════════════════════════════════════════════
# §10  [C] RR + 全向量化 Bootstrap CI
# ════════════════════════════════════════════════════════════════════════
def _build_year_padded(tmean_daily: pd.Series,
                       years: np.ndarray,
                       unique_yrs: np.ndarray) -> tuple:
    groups  = [tmean_daily.values[years == yr].astype(float)
               for yr in unique_yrs]
    max_len = max(len(g) for g in groups)
    n_yrs   = len(unique_yrs)
    padded  = np.full((n_yrs, max_len), np.nan, dtype=np.float64)
    for i, g in enumerate(groups):
        padded[i, :len(g)] = g
    return padded, max_len

def bootstrap_rr_ci(
        tmean_daily: pd.Series,
        tmm: float,
        beta: float,
        n_boot: int = BOOT_N,
        rng: np.random.Generator = None,
        alpha: float = 1 - CI_LEVEL,
) -> tuple:
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    tmean_daily = tmean_daily.dropna()
    if len(tmean_daily) == 0:
        return np.nan, np.nan, np.nan

    years      = np.array([pd.Timestamp(d).year for d in tmean_daily.index],
                           dtype=np.int32)
    unique_yrs = np.unique(years)
    n_yrs      = len(unique_yrs)
    vals       = tmean_daily.values.astype(np.float64)

    log_rr_pt = beta * np.maximum(vals - tmm, 0.0)
    point_rr  = float(np.exp(np.mean(log_rr_pt)))

    if n_yrs < 2:
        return point_rr, np.nan, np.nan

    padded, max_len = _build_year_padded(tmean_daily, years, unique_yrs)
    boot_idx = rng.integers(0, n_yrs, size=(n_boot, n_yrs), dtype=np.int32)
    sampled  = padded[boot_idx].reshape(n_boot, -1)
    log_rr_b = beta * np.maximum(sampled - tmm, 0.0)
    mean_log_rr = np.nanmean(log_rr_b, axis=1)
    boot_rrs    = np.exp(mean_log_rr)
    valid_boot  = boot_rrs[~np.isnan(boot_rrs)]
    if len(valid_boot) < 10:
        return point_rr, np.nan, np.nan

    return (
        point_rr,
        float(np.percentile(valid_boot, 100 * alpha / 2)),
        float(np.percentile(valid_boot, 100 * (1 - alpha / 2))),
    )


# ════════════════════════════════════════════════════════════════════════
# §11  单对站点完整分析（与加速模块集成）
# ════════════════════════════════════════════════════════════════════════
def process_single_pair(args) -> dict:
    idx, row = args
    pair_id  = str(row["pair_id"])
    rng      = np.random.default_rng(RANDOM_SEED + idx)
    result   = {"records": [], "diurnal": [], "threshold": {}, "error": None}

    try:
        usaf_u, wban_u, usaf_r, wban_r = parse_pair_id(pair_id)
    except ValueError as e:
        result["error"] = {"pair_id": pair_id, "step": "parse_pair_id", "reason": str(e)}
        return result

    lon_u = float(row["lon_urban"]); lat_u = float(row["lat_urban"])
    lon_r = float(row["lon_rural"]); lat_r = float(row["lat_rural"])

    df_u_all, yrs_u = load_station_allyears(usaf_u, wban_u, lon_u, verbose=False)
    df_r_all, yrs_r = load_station_allyears(usaf_r, wban_r, lon_r, verbose=False)
    if df_u_all is None or df_r_all is None:
        miss = (["urban_full_year"] if df_u_all is None else []) + \
               (["rural_full_year"] if df_r_all is None else [])
        result["error"] = {"pair_id": pair_id, "step": "load_station",
                           "reason": " & ".join(miss)}
        return result

    # Hemisphere-aware warm-season slicing.
    # Use urban latitude for the pair-level warm-season definition.
    df_u_warm, yrs_u_warm = slice_warm_season(df_u_all, lat_u)
    df_r_warm, yrs_r_warm = slice_warm_season(df_r_all, lat_u)
    has_warm = (df_u_warm is not None and df_r_warm is not None)

    _, u_tmax_all = compute_daily_minmax(df_u_all)
    _, r_tmax_all = compute_daily_minmax(df_r_all)
    u_tmean_all   = compute_daily_tmean(df_u_all)
    r_tmean_all   = compute_daily_tmean(df_r_all)

    if len(u_tmax_all.index.intersection(r_tmax_all.index)) == 0:
        result["error"] = {"pair_id": pair_id, "step": "common_dates",
                           "reason": "无城乡共同有效日期"}
        return result

    # [ERA5-MOD start]
    # 只使用 ERA5 长气候态计算热浪 DOY 阈值。
    # ERA5 文件缺失、列名不对或数据不足时，直接跳过该 pair。
    # 不再回退 ISD P90，也不调用 compute_hw_threshold_from_tmax()。
    _era5_tmax_u = load_era5_tmax_series(usaf_u, wban_u, "urban") \
        if USE_ERA5_CLIMATOLOGY else None

    if _era5_tmax_u is None:
        result["error"] = {
            "pair_id": pair_id,
            "step": "ERA5_missing_or_invalid",
            "reason": "urban ERA5 Tmax file missing, invalid columns, or insufficient data"
        }
        return result

    doy_thr_era5 = compute_hw_doy_thr_era5(_era5_tmax_u)
    hw_thr_s_raw = apply_doy_thr_to_index(u_tmax_all.index, doy_thr_era5)

    if hw_thr_s_raw.isna().all():
        result["error"] = {
            "pair_id": pair_id,
            "step": "ERA5_missing_or_invalid",
            "reason": "ERA5 DOY threshold all NaN"
        }
        return result

    if USE_QM_BIAS_CORRECTION:
        hw_thr_s, qm_diagnostics = apply_quantile_mapping_bias_correction(
            hw_thr_series_raw=hw_thr_s_raw,
            isd_tmax=u_tmax_all,
            era5_tmax=_era5_tmax_u,
            min_overlap_days_per_month=QM_MIN_OVERLAP_DAYS_PER_MONTH,
            min_overlap_days_annual=QM_MIN_OVERLAP_DAYS_ANNUAL,
            n_quantiles=QM_N_QUANTILES,
        )
    elif USE_MONTHLY_MEAN_BIAS_CORRECTION:
        raise NotImplementedError(
            "Monthly mean bias correction is not implemented in this script."
        )
    else:
        hw_thr_s = hw_thr_s_raw.copy()
        qm_diagnostics = {}

    if hw_thr_s.isna().all():
        result["error"] = {
            "pair_id": pair_id,
            "step": "ERA5_QM_threshold_invalid",
            "reason": "QM-corrected threshold all NaN"
        }
        return result




    hw_thr_mean = float(hw_thr_s.mean()) if hw_thr_s.notna().any() else np.nan
    if np.isnan(hw_thr_mean):
        result["error"] = {"pair_id": pair_id, "step": "hw_threshold",
                           "reason": "城市 Tmax 不足"}
        return result

    # Diagnostic all-year HW detection, kept for threshold diagnostics.
    # The actual HW/NHW period analysis below uses warm-season flags only.
    hw_mask_all   = detect_heatwave(u_tmax_all, hw_thr_s)
    hw_dates_all  = set(hw_mask_all[hw_mask_all].index.tolist())
    nhw_dates_all = set(hw_mask_all[~hw_mask_all].index.tolist())

    hw_intensity = (
        float((u_tmax_all[hw_mask_all] - hw_thr_s.reindex(u_tmax_all.index)[hw_mask_all]).mean())
        if hw_mask_all.any() else np.nan
    )

    if has_warm:
        warm_common = (set(df_u_warm["local_date"].unique()) &
                       set(df_r_warm["local_date"].unique()))
    else:
        warm_common = set()

    # Canonical rule: HW/NHW dates must come from analysis_multiyear.py.
    # If the exported pair-level flags are missing or mismatched, skip this
    # pair rather than redefining HW/NHW dates in this downstream script.
    analysis_hw_flags = get_analysis_hw_flags_for_pair(pair_id)
    aligned_hw_flags = align_analysis_hw_flags(
        analysis_hw_flags, u_tmax_all.index
    )

    required_flag_cols = [
        "is_warm_season",
        "hw_flag_percentile_warm_season",
        "nhw_flag_percentile_warm_season",
    ]
    if (
        aligned_hw_flags is None
        or any(c not in aligned_hw_flags.columns for c in required_flag_cols)
        or aligned_hw_flags[required_flag_cols].dropna(how="any").empty
    ):
        result["error"] = {
            "pair_id": pair_id,
            "step": "canonical_hw_flags",
            "reason": (
                "analysis_multiyear daily HW/NHW flags missing or "
                "mismatched; downstream fallback is disabled"
            ),
        }
        return result

    valid_flags = aligned_hw_flags[required_flag_cols].notna().all(axis=1)
    is_warm_ext = pd.Series(False, index=aligned_hw_flags.index)
    hw_ext = pd.Series(False, index=aligned_hw_flags.index)
    nhw_ext = pd.Series(False, index=aligned_hw_flags.index)

    is_warm_ext.loc[valid_flags] = (
        aligned_hw_flags.loc[valid_flags, "is_warm_season"]
        .astype(int).astype(bool).values
    )
    hw_ext.loc[valid_flags] = (
        aligned_hw_flags.loc[
            valid_flags, "hw_flag_percentile_warm_season"
        ].astype(int).astype(bool).values
    )
    nhw_ext.loc[valid_flags] = (
        aligned_hw_flags.loc[
            valid_flags, "nhw_flag_percentile_warm_season"
        ].astype(int).astype(bool).values
    )

    hw_warm = set(
        aligned_hw_flags.index[is_warm_ext & hw_ext]
    ) & warm_common
    nhw_warm = set(
        aligned_hw_flags.index[is_warm_ext & nhw_ext]
    ) & warm_common
    hw_source = "analysis_multiyear_daily_heatwave_flags"

    result["threshold"] = {
        "pair_id": pair_id,
        "hw_thr_mean": round(hw_thr_mean, 2),
        "bias_correction_method": qm_diagnostics.get(
            "bias_correction_method",
            "none_raw_era5_threshold"
        ),
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
        "hw_threshold_raw_mean": round(float(hw_thr_s_raw.mean()), 3),
        "hw_threshold_qm_mean": round(float(hw_thr_s.mean()), 3),
        "exceed_days_raw": int(
            (u_tmax_all > hw_thr_s_raw.reindex(u_tmax_all.index)).fillna(False).sum()
        ),
        "exceed_days_qm": int(
            (u_tmax_all > hw_thr_s.reindex(u_tmax_all.index)).fillna(False).sum()
        ),
        "final_hw_days": len(hw_dates_all),
        "hw_source": hw_source,
        "hemisphere": hemisphere_from_lat(lat_u),
        "warm_season_label": warm_season_label_for_lat(lat_u),
        "warm_season_months": warm_months_string(lat_u),

        "n_hw_days_annual": len(hw_dates_all), "n_nhw_days_annual": len(nhw_dates_all),
        "n_hw_days_warm": len(hw_warm), "n_nhw_days_warm": len(nhw_warm),
        "hw_intensity": round(hw_intensity, 3) if not np.isnan(hw_intensity) else np.nan,
        "valid_yrs_urban": ",".join(map(str, yrs_u)),
        "valid_yrs_rural": ",".join(map(str, yrs_r)),
        "n_valid_yrs_u": len(yrs_u), "n_valid_yrs_r": len(yrs_r),
    }

    # Canonical UHI/UCI classification:
    # mean annual urban-rural daily Tmax contrast on common valid dates.
    common_tmax_dates = u_tmax_all.index.intersection(r_tmax_all.index)
    if len(common_tmax_dates) == 0:
        result["error"] = {
            "pair_id": pair_id,
            "step": "canonical_uhi_uci_group",
            "reason": "no common annual daily Tmax dates",
        }
        return result

    delta_tmax_ann = float(
        (
            u_tmax_all.reindex(common_tmax_dates)
            - r_tmax_all.reindex(common_tmax_dates)
        ).mean()
    )
    group = "UHI" if delta_tmax_ann > 0 else "UCI"

    # Annual FFT is retained only for annual diurnal metrics.
    annual_fft_u = compute_fft(df_u_all)
    annual_fft_r = compute_fft(df_r_all)
    if annual_fft_u is None or annual_fft_r is None:
        result["error"] = {
            "pair_id": pair_id,
            "step": "annual_fft",
            "reason": "annual FFT 失败",
        }
        return result

    # ── Köppen 采样（修改：返回 code + source，fallback 时 kg_code = NaN）──
    kg_codes, kg_sources = extract_koppen([lon_u], [lat_u])
    kg_code        = kg_codes[0]          # str 或 np.nan
    kg_code_source = kg_sources[0]        # "tif" 或 "lat_fallback"

    # kg_group 推断：TIF 成功用 kg_code；fallback 用纬度粗估独立推算
    if isinstance(kg_code, str) and kg_code.strip():
        kg_grp = kg_group(kg_code)
    else:
        # kg_code 为 NaN → 纬度粗估（不覆写 kg_code，保持 NaN 以示透明）
        kg_grp = kg_group(_lat2kg(lat_u))

    # ── β：ERA5 30年 μ/σ + Zhao 2021 EDF + Tobías 2021 MMTP ──────────
    # μ/σ 回退优先级：ERA5 30年（最优）→ ISD 10年 → 气候带参考值 _CLIM_PARAMS
    # ERA5 30年参考期符合 WMO 气候正常值标准，比 ISD 10年更稳定
    _era5_tmean_u = load_era5_tmean_series(usaf_u, wban_u, "urban") \
        if USE_ERA5_CLIMATOLOGY else None
    if _era5_tmean_u is not None:
        mu_u   = float(_era5_tmean_u.mean())
        sig_u  = float(_era5_tmean_u.std(ddof=1))
        mu_src = "era5_30yr"
    elif len(u_tmean_all.dropna()) >= 30:
        mu_u   = float(u_tmean_all.dropna().mean())
        sig_u  = float(u_tmean_all.dropna().std(ddof=1))
        mu_src = "isd_10yr"
    else:
        mu_u, sig_u = _CLIM_PARAMS.get(kg_grp, _CLIM_PARAMS["C"])
        mu_src = "clim_fallback"
    _mmtp_pct  = _MMTP_BY_KOPPEN.get(kg_grp, TMM_PERCENTILE)  # Tobías 2021
    _tmm_pct_f = _mmtp_pct / 100
    edf_c, edf_lo, edf_hi = _EDF_HEAT.get(kg_grp, _EDF_HEAT["C"])
    β_c  = _edf_to_beta(edf_c,          mu_u, sig_u, _tmm_pct_f)
    β_lo = _edf_to_beta(max(edf_lo, 0), mu_u, sig_u, _tmm_pct_f)
    β_hi = _edf_to_beta(edf_hi,         mu_u, sig_u, _tmm_pct_f)

    # ── TMM：Tobías 2021 MMTP × 本地 ISD Tmean 分布 ──────────────────
    tmm_u = (float(np.percentile(u_tmean_all.dropna(), _mmtp_pct))
             if len(u_tmean_all.dropna()) > 0 else np.nan)
    tmm_r = (float(np.percentile(r_tmean_all.dropna(), _mmtp_pct))
             if len(r_tmean_all.dropna()) > 0 else np.nan)
    if np.isnan(tmm_u) or np.isnan(tmm_r):
        result["error"] = {"pair_id": pair_id, "step": "tmm", "reason": "TMM 计算失败"}
        return result

    lat_grp = row.get("lat_group", assign_lat_group(lat_u))
    lcz_raw = row.get("urban_lcz", np.nan)
    lcz_cor = normalize_lcz(lcz_raw)
    lcz_cls = lcz_compactness_class(lcz_cor)

    # ── base_meta（新增 kg_code_source 字段）────────────────────────────
    base_meta = dict(
        pair_id=pair_id, group=group,
        delta_tmax_ann=round(delta_tmax_ann, 3),
        kg_code=kg_code,              # str（TIF 成功）或 NaN（fallback）
        kg_code_source=kg_code_source,# "tif" 或 "lat_fallback"
        kg_group=kg_grp,              # 大类首字母，始终非空
        lat_group=lat_grp, lat_urban=lat_u, lon_urban=lon_u,
        lat_rural=lat_r, lon_rural=lon_r,
        urban_lcz=lcz_raw, urban_lcz_cor=lcz_cor, urban_lcz_class=lcz_cls,
        n_valid_yrs_u=len(yrs_u), n_valid_yrs_r=len(yrs_r),
        hw_thr_mean=round(hw_thr_mean, 2),
        bias_correction_method=qm_diagnostics.get(
            "bias_correction_method",
            "none_raw_era5_threshold"
        ),
        qm_ref_mode=qm_diagnostics.get("qm_ref_mode", ""),
        qm_n_quantiles=qm_diagnostics.get("qm_n_quantiles", np.nan),
        hw_threshold_raw_mean=round(float(hw_thr_s_raw.mean()), 3),
        hw_threshold_qm_mean=round(float(hw_thr_s.mean()), 3),
        hw_source=hw_source,
        hemisphere=hemisphere_from_lat(lat_u),
        warm_season_label=warm_season_label_for_lat(lat_u),
        warm_season_months=warm_months_string(lat_u),

        hw_intensity=round(hw_intensity, 3) if not np.isnan(hw_intensity) else np.nan,
        beta_heat=round(β_c, 8), beta_heat_lo95=round(β_lo, 8),
        beta_heat_hi95=round(β_hi, 8),
        edf_heat_src=round(_EDF_HEAT[kg_grp][0], 6),
        tmm_pct=_mmtp_pct,            # Tobías 2021 气候带分层 MMTP
        mu_local=round(mu_u, 4),      # β 计算实际使用的 μ
        sig_local=round(sig_u, 4),    # β 计算实际使用的 σ
        beta_mu_src=mu_src,           # "era5_30yr" | "isd_10yr" | "clim_fallback"
    )

    def _subset(df_loc, date_set):
        if df_loc is None or not date_set: return None
        return df_loc[df_loc["local_date"].isin(date_set)].copy()

    periods: dict = {"annual": (df_u_all, df_r_all, u_tmean_all, r_tmean_all)}
    if has_warm:
        u_tmw = compute_daily_tmean(df_u_warm)
        r_tmw = compute_daily_tmean(df_r_warm)
        # Keep the legacy output period name "JJA" for downstream compatibility.
        # For Southern Hemisphere pairs this represents DJF, documented by warm_season_label.
        periods["JJA"] = (df_u_warm, df_r_warm, u_tmw, r_tmw)
    if has_warm and len(hw_warm) >= HW_MIN_DAYS:
        u_hw = _subset(df_u_warm, hw_warm); r_hw = _subset(df_r_warm, hw_warm)
        u_tm_hw = u_tmean_all.reindex([d for d in hw_warm if d in u_tmean_all.index])
        r_tm_hw = r_tmean_all.reindex([d for d in hw_warm if d in r_tmean_all.index])
        if u_hw is not None and len(u_hw) > 0:
            periods["HW"] = (u_hw, r_hw, u_tm_hw, r_tm_hw)
    if has_warm and len(nhw_warm) > 0:
        u_nhw = _subset(df_u_warm, nhw_warm); r_nhw = _subset(df_r_warm, nhw_warm)
        u_tm_nhw = u_tmean_all.reindex([d for d in nhw_warm if d in u_tmean_all.index])
        r_tm_nhw = r_tmean_all.reindex([d for d in nhw_warm if d in r_tmean_all.index])
        if u_nhw is not None and len(u_nhw) > 0:
            periods["NHW"] = (u_nhw, r_nhw, u_tm_nhw, r_tm_nhw)

    for period_name, (sub_u, sub_r, tm_u, tm_r) in periods.items():
        if sub_u is None or sub_r is None or len(sub_u)==0 or len(sub_r)==0: continue
        fft_u = annual_fft_u if period_name == "annual" else compute_fft(sub_u)
        fft_r = annual_fft_r if period_name == "annual" else compute_fft(sub_r)
        if fft_u is None or fft_r is None: continue

        u_curve     = fft_u["hourly_reconstructed"]
        r_curve     = fft_r["hourly_reconstructed"]
        delta_curve = u_curve - r_curve
        dT_day      = float(np.mean(delta_curve[DAY_HOURS]))
        dT_night    = float(np.mean(delta_curve[NIGHT_HOURS]))

        rr_u_pt, rr_u_lo, rr_u_hi = bootstrap_rr_ci(tm_u, tmm_u, β_c, rng=rng)
        rr_r_pt, rr_r_lo, rr_r_hi = bootstrap_rr_ci(tm_r, tmm_r, β_c, rng=rng)
        af_u = ((rr_u_pt-1)/rr_u_pt if (not np.isnan(rr_u_pt) and rr_u_pt>1) else 0.0)
        af_r = ((rr_r_pt-1)/rr_r_pt if (not np.isnan(rr_r_pt) and rr_r_pt>1) else 0.0)
        dRR_abs  = (rr_u_pt - rr_r_pt if not(np.isnan(rr_u_pt) or np.isnan(rr_r_pt)) else np.nan)
        RR_ratio = (rr_u_pt / rr_r_pt if not(np.isnan(rr_u_pt) or np.isnan(rr_r_pt) or rr_r_pt==0) else np.nan)
        n_hw_col = (len(hw_warm)      if period_name=="HW"     else
                    len(nhw_warm)     if period_name=="NHW"    else
                    len(hw_dates_all) if period_name=="annual" else np.nan)

        rec = {
            **base_meta, "period": period_name,
            "urban_Tmean":    round(fft_u["mean"], 3),
            "rural_Tmean":    round(fft_r["mean"], 3),
            "dTmean":         round(fft_u["mean"]     - fft_r["mean"],     3),
            "urban_Tmax_fft": round(fft_u["Tmax_fft"], 3),
            "rural_Tmax_fft": round(fft_r["Tmax_fft"], 3),
            "dTmax_fft":      round(fft_u["Tmax_fft"] - fft_r["Tmax_fft"], 3),
            "urban_Tmin_fft": round(fft_u["Tmin_fft"], 3),
            "rural_Tmin_fft": round(fft_r["Tmin_fft"], 3),
            "dTmin_fft":      round(fft_u["Tmin_fft"] - fft_r["Tmin_fft"], 3),
            "urban_Amp1":     round(fft_u["amplitudes"][0], 4),
            "rural_Amp1":     round(fft_r["amplitudes"][0], 4),
            "dAmp1":          round(fft_u["amplitudes"][0]-fft_r["amplitudes"][0], 4),
            "urban_Amp2":     round(fft_u["amplitudes"][1], 4) if len(fft_u["amplitudes"])>1 else np.nan,
            "rural_Amp2":     round(fft_r["amplitudes"][1], 4) if len(fft_r["amplitudes"])>1 else np.nan,
            "dAmp2":          round(fft_u["amplitudes"][1]-fft_r["amplitudes"][1], 4) if len(fft_u["amplitudes"])>1 else np.nan,
            "urban_Phase1":   round(fft_u["phases"][0], 5),
            "rural_Phase1":   round(fft_r["phases"][0], 5),
            "urban_Tmax_hour":fft_u["Tmax_hour"],
            "rural_Tmax_hour":fft_r["Tmax_hour"],
            "dT_day":         round(dT_day,   3),
            "dT_night":       round(dT_night, 3),
            "n_days_urban":   fft_u["n_days"], "n_days_rural": fft_r["n_days"],
            "n_hw_days":      n_hw_col,
            "tmm_urban":      round(tmm_u, 3), "tmm_rural": round(tmm_r, 3),
            "rr_urban":       round(rr_u_pt, 6) if not np.isnan(rr_u_pt) else np.nan,
            "rr_urban_lo95":  round(rr_u_lo, 6) if not np.isnan(rr_u_lo) else np.nan,
            "rr_urban_hi95":  round(rr_u_hi, 6) if not np.isnan(rr_u_hi) else np.nan,
            "rr_rural":       round(rr_r_pt, 6) if not np.isnan(rr_r_pt) else np.nan,
            "rr_rural_lo95":  round(rr_r_lo, 6) if not np.isnan(rr_r_lo) else np.nan,
            "rr_rural_hi95":  round(rr_r_hi, 6) if not np.isnan(rr_r_hi) else np.nan,
            "af_urban":       round(af_u, 6), "af_rural": round(af_r, 6),
            "dRR_abs":        round(dRR_abs,  6) if not np.isnan(dRR_abs)  else np.nan,
            "RR_ratio":       round(RR_ratio, 6) if not np.isnan(RR_ratio) else np.nan,
            "dAF":            round(af_u - af_r, 6),
        }
        result["records"].append(rec)
        for h in range(24):
            result["diurnal"].append({
                "pair_id": pair_id, "period": period_name, "group": group,
                "hour_LST": h,
                "urban_T": round(float(u_curve[h]),     4),
                "rural_T": round(float(r_curve[h]),     4),
                "delta_T": round(float(delta_curve[h]), 4),
            })

    if not result["records"]:
        result["error"] = {"pair_id": pair_id, "step": "all_periods_fft",
                           "reason": "所有时期 FFT 均失败"}
    return result


# ════════════════════════════════════════════════════════════════════════
# §12  [D] 主流程：joblib.Parallel 多进程调度
# ════════════════════════════════════════════════════════════════════════
def main():
    import multiprocessing, time
    from joblib import Parallel, delayed

    n_cpu = multiprocessing.cpu_count()
    n_workers = (N_WORKERS if N_WORKERS > 0 else max(1, n_cpu - 1))
    chunk = (JOBLIB_CHUNK if JOBLIB_CHUNK > 0
             else max(1, 50 // n_workers))

    print("=" * 72)
    print("  heatwave_rr_isd.py  ── CPU 加速版  [Heat Exposure Risk Index]")
    print(f"  系统 CPU 核心: {n_cpu}  →  使用 {n_workers} 个 worker")
    print(f"  backend: {JOBLIB_BACKEND}  chunk: {chunk}")
    print(f"  加速模块: [A]DOY查找表 [B]numpy-diff-HW "
          f"[C]向量化Bootstrap [D]joblib并行")
    print(f"  numba: {'已安装，JIT加速启用' if HAS_NUMBA else '未安装，使用纯numpy（pip install numba 可进一步加速）'}")
    print(f"  Köppen: Beck et al. (2023) Sci. Data 10:724")
    print(f"  HW 检测: 优先读取 analysis_multiyear.py daily_heatwave_flags.csv；")
    print(f"           fallback=ERA5-QM 城市 Tmax P{HW_PERCENTILE} ±{HW_WINDOW_HALF}天 ≥{HW_MIN_DAYS}天，NH=JJA / SH=DJF")
    print(f"  ERA5 气候态: {'ENABLED' if USE_ERA5_CLIMATOLOGY else 'DISABLED (ISD fallback)'}"
          f"  [USE_ERA5_CLIMATOLOGY={USE_ERA5_CLIMATOLOGY}]")
    print(f"  UHI/UCI: annual FFT δTmax_fft > 0 → UHI")
    print(f"  β_heat : Zhao 2021 Table S3 EDF + 站点 ERA5/ISD μ/σ + Tobías 2021 MMTP")
    print(f"  MMTP   : Tobías et al. (2021) Environ Epidemiol 5:e169 气候带分层值")
    print("=" * 72)
    OUTPUT_DIR.mkdir(exist_ok=True)

    cal_df = build_beta_calibration_table()
    cal_df.to_csv(OUTPUT_DIR / "rr_06_beta_calibration.csv", index=False)
    print(f"\n  β 校准溯源（Zhao 2021 EDF + Tobías 2021 MMTP → β）：")
    print(cal_df[["koppen_group","EDF_heat_central","E_excess_T",
                  "beta_central","EDF_check"]].to_string(index=False))

    print(f"\n[1/4] 读取配对表：{PAIR_CSV}")
    pair_df = pd.read_csv(PAIR_CSV)
    pair_df["urban_lcz_corrected"] = pair_df["urban_lcz"].apply(normalize_lcz)
    pair_df["urban_lcz_class"]     = pair_df["urban_lcz_corrected"].apply(
        lcz_compactness_class)
    if "lat_group" not in pair_df.columns:
        pair_df["lat_group"] = pair_df["lat_urban"].apply(assign_lat_group)
    print(f"      {len(pair_df):,} 对站点 × {len(pair_df.columns)} 列")

    if HAS_NUMBA:
        print("  [numba] 预热 JIT 编译...")
        _detect_hw_core(np.array([0], dtype=np.int64),
                        np.array([3], dtype=np.int64), 3, 5)
        print("  [numba] 预热完成")

    tasks = [(int(idx), row.to_dict()) for idx, row in pair_df.iterrows()]
    print(f"\n[2/4] 逐对站处理（{len(tasks):,} 对）")

    t_start = time.perf_counter()

    if n_workers == 1:
        results = [process_single_pair(a)
                   for a in tqdm(tasks, desc="  pairs [serial]")]
    else:
        backend = JOBLIB_BACKEND
        try:
            results = Parallel(
                n_jobs     = n_workers,
                backend    = backend,
                batch_size = chunk,
                verbose    = 0,
            )(delayed(process_single_pair)(a)
              for a in tqdm(tasks, desc=f"  pairs [{backend}]"))
        except Exception as e:
            print(f"\n  [!] {backend} 并行失败（{e}），降级为串行")
            results = [process_single_pair(a)
                       for a in tqdm(tasks, desc="  pairs [serial-fallback]")]

    t_end = time.perf_counter()

    all_records, all_diurnal, all_thresh, all_errors = [], [], [], []
    for res in results:
        all_records.extend(res["records"])
        all_diurnal.extend(res["diurnal"])
        if res["threshold"]: all_thresh.append(res["threshold"])
        if res["error"]:     all_errors.append(res["error"])

    n_valid = len(set(r["pair_id"] for r in all_records))
    elapsed = t_end - t_start
    print(f"\n  完成：{n_valid:,} 对有效，{len(all_errors):,} 对跳过")
    print(f"  总耗时：{elapsed:.1f} s  ({elapsed/max(len(tasks),1):.2f} s/对)")

    # ── kg_code_source 统计（用于数据质量报告）──────────────────────
    if all_records:
        src_counts = pd.Series([r["kg_code_source"] for r in all_records]).value_counts()
        print(f"\n  Köppen 来源统计（rr_01 行计）：")
        for src, cnt in src_counts.items():
            print(f"    {src:<16}: {cnt:,} 行")
        mu_counts = pd.Series([r["beta_mu_src"] for r in all_records]).value_counts()
        print(f"\n  β μ/σ 来源统计：")
        for src, cnt in mu_counts.items():
            print(f"    {src:<16}: {cnt:,} 行")

    print(f"\n[3/4] 写出输出文件")
    rr_df      = pd.DataFrame(all_records)
    diurnal_df = pd.DataFrame(all_diurnal)
    thresh_df  = pd.DataFrame(all_thresh)
    error_df   = pd.DataFrame(all_errors)

    rr_df.to_csv(     OUTPUT_DIR / "rr_01_pair_period.csv",      index=False)
    diurnal_df.to_csv(OUTPUT_DIR / "rr_02_diurnal_profiles.csv", index=False)
    thresh_df.to_csv( OUTPUT_DIR / "rr_04_hw_threshold.csv",     index=False)

    error_df.to_csv(OUTPUT_DIR / "rr_05_skipped_pairs.csv", index=False)
    error_df.to_csv(OUTPUT_DIR / "skipped_pairs_log.csv", index=False)

    print(f"  [✓] rr_01  {len(rr_df):,} 行  "
          f"（字段含 kg_code / kg_code_source / kg_group / mu_local / beta_mu_src）")
    print(f"  [✓] rr_02  {len(diurnal_df):,} 行")

    if len(rr_df) > 0:
        print(f"\n[4/4] 分组汇总")
        def _agg(df, gc):
            rows = []
            for (gv, period), g in df.groupby([gc, "period"]):
                rows.append({
                    gc: gv, "period": period, "n_pairs": len(g),
                    "rr_urban_mean":   round(g["rr_urban"].mean(),   5),
                    "rr_urban_median": round(g["rr_urban"].median(), 5),
                    "rr_rural_mean":   round(g["rr_rural"].mean(),   5),
                    "dRR_mean":        round(g["dRR_abs"].mean(),    5),
                    "RR_ratio_mean":   round(g["RR_ratio"].mean(),   4),
                    "dTmax_fft_mean":  round(g["dTmax_fft"].mean(),  3),
                    "dTmin_fft_mean":  round(g["dTmin_fft"].mean(),  3),
                    "dAmp1_mean":      round(g["dAmp1"].mean(),      4),
                    "dT_day_mean":     round(g["dT_day"].mean(),     3),
                    "dT_night_mean":   round(g["dT_night"].mean(),   3),
                    "af_urban_mean":   round(g["af_urban"].mean(),   5),
                    "af_rural_mean":   round(g["af_rural"].mean(),   5),
                    "beta_heat_mean":  round(g["beta_heat"].mean(),  8),
                    "edf_heat_mean":   round(g["edf_heat_src"].mean(),6),
                    # kg_code_source 品质指标：tif 行占比
                    "frac_tif_source": round(
                        (g["kg_code_source"] == "tif").mean(), 4)
                        if "kg_code_source" in g.columns else np.nan,
                    "frac_era5_mu":    round(
                        (g["beta_mu_src"] == "era5_30yr").mean(), 4)
                        if "beta_mu_src" in g.columns else np.nan,
                    "ttest_dRR_p":     round(
                        stats.ttest_1samp(g["dRR_abs"].dropna(), 0).pvalue, 4)
                        if len(g["dRR_abs"].dropna()) >= 3 else np.nan,
                })
            return pd.DataFrame(rows)

        summ = pd.concat([
            _agg(rr_df[rr_df["kg_group"].notna()],           "kg_group").assign(group_type="koppen"),
            _agg(rr_df[rr_df["lat_group"]!="Unknown"],       "lat_group").assign(group_type="lat"),
            _agg(rr_df[rr_df["group"].isin(["UHI","UCI"])],  "group"   ).assign(group_type="uhi"),
        ], ignore_index=True)
        summ.to_csv(OUTPUT_DIR / "rr_03_group_summary.csv", index=False)
        print(f"  [✓] rr_03  {len(summ):,} 行  （含 frac_tif_source / frac_era5_mu 品质列）")

        print(f"\n{'─'*72}")
        hw = rr_df[rr_df["period"]=="HW"]
        if len(hw) > 0:
            for grp in ["UHI","UCI"]:
                sub = hw[hw["group"]==grp]
                if len(sub)==0: continue
                print(f"  {grp}（N={len(sub):,}）"
                      f"  RR_urban={sub['rr_urban'].mean():.5f}"
                      f"  RR_rural={sub['rr_rural'].mean():.5f}"
                      f"  ΔRR={sub['dRR_abs'].mean():.5f}"
                      f"  RR比={sub['RR_ratio'].mean():.4f}"
                      f"  δTmax={sub['dTmax_fft'].mean():.3f}°C")
        kg_hw = summ[(summ["group_type"]=="koppen")&(summ["period"]=="HW")]
        if len(kg_hw):
            print(f"\n  Köppen × HW：")
            print(kg_hw[["kg_group","n_pairs","rr_urban_mean","rr_rural_mean",
                          "dRR_mean","dTmax_fft_mean","beta_heat_mean",
                          "frac_tif_source"]].to_string(index=False))

    print(f"\n{'='*72}")
    print(f"  完成！{OUTPUT_DIR.resolve()}")
    for f, d in [("rr_01_pair_period.csv",     "对站×时期 HERI+FFT（含 mu_local / beta_mu_src）"),
                 ("rr_02_diurnal_profiles.csv", "24h 日循环廓线"),
                 ("rr_03_group_summary.csv",    "分组汇总（含 frac_tif_source / frac_era5_mu）"),
                 ("rr_04_hw_threshold.csv",     "HW 阈值诊断"),
                 ("rr_05_skipped_pairs.csv",    "跳过站点"),
                 ("rr_06_beta_calibration.csv", "β 校准溯源（Zhao 2021 + Tobías 2021）")]:
        print(f"  {f:<36} ← {d}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

############################################################
