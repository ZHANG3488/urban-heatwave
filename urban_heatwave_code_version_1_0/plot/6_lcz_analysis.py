#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uhi_uci_hw_jja_integrated_analysis_v4.py
═══════════════════════════════════════════════════════════════════════════════
UHI/UCI × Annual/HW/Warm-season 城乡综合影响集成分析 + 劳动/能源/睡眠经济损失

v4_updated 新增特性
───────────────────────────────────────────────────────────────────────────────
1. 适配 upstream v11 (sum_code_updated.txt):
   - 自动接收和处理蒙特卡洛不确定性列 (sleep_loss_ci_low/high)
   - 自动接收和传递 KG 气候分区分类与版本标示列 (kg_code, kg_group 等)
2. 新增核心结果总结文件:
   - 自动生成 `result_highlight.md`，从高维数据中提炼出各个群体 (UHI/UCI) 和
     各个时期 (HW/Warm season/Annual) 下的核心影响与经济损失摘要。

输出文件
───────────────────────────────────────────────────────────────────────────────
  integrated_output/
  ├── DIAGNOSTIC_REPORT.md
  ├── result_highlight.md        ← [新增] 核心结果提炼与摘要
  ├── pair_period_panel.csv
  ├── diff_summary_v4.csv
  ├── summary_table_full.csv
  ├── summary_table_display.csv
  ├── boxplots/                  ← 差值箱线图（含经济损失）
  ├── boxplots/absolute_values/  ← 城乡绝对值 + 百分比变化
  ├── diurnal/
  └── building_operations/

Method-freeze final patch
─────────────────────────
- Preserve pair_id/period string merge keys in the labour reader.
- Do not reconstruct warm season from fixed JJA months.
- Compute seasonal sleep totals only from formal paired HNE sleep-valid nights.
"""

import os, warnings, textwrap, time
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

from config import PAIR_CSV_PATH as PAIR_CSV, UNIFIED_ROOT


# ══════════════════════════════════════════════════════════════════
# §0  路径配置
# ══════════════════════════════════════════════════════════════════
CDH_DIR       = UNIFIED_ROOT + "/analysis/cdh_energy"
LABOUR_DIR    = UNIFIED_ROOT + "/analysis/labour"
HNE_DIR       = UNIFIED_ROOT + "/analysis/hne_econ/paired/method_pooled"
OUTPUT_DIR    = UNIFIED_ROOT + "/plot_data/integrated_v2"

CANONICAL_GROUP_CSV = (
    UNIFIED_ROOT + "/analysis/main_multiyear/"
    "robustness_percentile/all_pair_period_metrics.csv"
)

# The two HW/NHW comparison figures must start from exactly the same matched
# station-pair cohort used by plot_results_fig23.py:
#   1) hw_method == percentile;
#   2) both non_heatwave and heatwave rows are present for the same pair_id;
#   3) dTmean, dAmp1, dTx and dTn are complete in both periods.
REFERENCE_MATCH_REQUIRED_METRICS = ("dTmean", "dAmp1", "dTx", "dTn")

# ══════════════════════════════════════════════════════════════════
# §1  全局参数
# ══════════════════════════════════════════════════════════════════


_HRHD = {1};  LCZ_LRLD = {6}
T_COOL, T_HEAT = 26.0, 18.0
COP_A0, COP_A1, COP_A2 = 5.80, -0.088, 2.9e-4
UA = {("COMM","HRHD"):65., ("COMM","LRLD"):40., ("COMM","OTHER"):50.,
      ("RESI","HRHD"):28., ("RESI","LRLD"):16., ("RESI","OTHER"):20.}
COMM_HOURS  = list(range(9, 22))
RESI_HOURS  = list(range(18, 24)) + list(range(0, 8))
DAY_HOURS   = list(range(8, 20))                         # 08:00–19:59, 12 h
NIGHT_HOURS = list(range(20, 24)) + list(range(0, 8))   # 20:00–07:59, 12 h

# Minor et al. (2022), Table S37
SLEEP_B1 = -0.107
SLEEP_B2 = -0.618

SLEEP_K1 = -20.0
SLEEP_K2 = 10.0

# Piecewise segment slopes.
SLEEP_SLOPE_MINUS20_TO_10 = SLEEP_B1
SLEEP_SLOPE_ABOVE_10 = SLEEP_B2

PERIOD_NORM  = {
    "annual": "annual",
    "warm_season": "JJA",
    "JJA": "JJA",
    "heatwave": "HW",
    "HW": "HW",
    "non_heatwave": "NHW",
    "NHW": "NHW",
}
PERIOD_LABEL = {
    "annual": "Annual",
    "JJA": "Warm\nseason",
    "HW": "Heat\nwave",
    "NHW": "Non-HW",
}
PERIODS_PLOT = ["annual", "JJA", "HW"]

PERIOD_CLR = {"annual": "#2ca02c", "JJA": "#ff7f0e", "HW": "#d62728", "NHW": "#9467bd"}
PERIOD_LS  = {"annual": "--", "JJA": "-.", "HW": "-", "NHW": ":"}

COLOR_UHI = "#d62728";  COLOR_UCI = "#1f77b4"
GROUP_CLR  = {"UHI": COLOR_UHI, "UCI": COLOR_UCI}


# 经济损失颜色
COLOR_ECON_SLEEP = "#5b4fcf"
COLOR_ECON_HEAT  = "#e8521a"
COLOR_ECON_TOTAL = "#2ca02c"

# ─── [FIX-B] 经济损失列分类（供 pair_period_agg 使用） ───────────
ECON_PCT_PREFIXES = (
    "econ_loss_pct_short", "econ_loss_pct_long",
    "econ_loss_pct_hourly", "econ_loss_pct_present",
    "heat_loss_pct", "total_loss_pct",
    "d_econ_loss_pct_short", "d_econ_loss_pct_long",
    "d_econ_loss_pct_hourly", "d_econ_loss_pct_present",
    "d_heat_loss_pct", "d_total_loss_pct",
)
ECON_USD_PREFIXES = (
    "econ_loss_usd_sleep", "econ_loss_usd_heat",
    "econ_loss_usd", "total_loss_usd",
    "d_econ_loss_usd_sleep", "d_econ_loss_usd_heat",
    "d_econ_loss_usd", "d_total_loss_usd",
)

# ─── Boxplot 配置 ─────────────────────────────────────────────────
BOXPLOT_CFG = [
    # CDH / HDH — 期间累积
    ("dCDH_total_sum",   r"$\Sigma\Delta$CDH (°C·h/period)",
     "Total CDH — Period Sum",              None),
    ("dCDH_comm_sum",    r"$\Sigma\Delta$CDH$_{comm}$ (°C·h/period)",
     "Commercial CDH — Period Sum",         None),
    ("dHDH_total_sum",   r"$\Sigma\Delta$HDH (°C·h/period)",
     "Total HDH — Period Sum",              None),
    # CDH — 日均值
    ("dCDH_total_mean",  r"$\bar{\Delta}$CDH (°C·h/day)",
     "Total CDH — Daily Mean",              None),
    # Energy — 期间累积
    ("dE_comm_sum",      r"$\Sigma\Delta$E$_{comm}$ (W·h/period)",
     "Commercial Energy — Period Sum",      None),
    ("dE_resi_sum",      r"$\Sigma\Delta$E$_{resi}$ (W·h/period)",
     "Residential Energy — Period Sum",     None),
    ("dE_total_sum",     r"$\Sigma\Delta$E$_{total}$ (W·h/period)",
     "Total Energy — Period Sum",           None),
    # Energy — 日均值
    ("dE_total_mean",    r"$\bar{\Delta}$E$_{total}$ (W·h/day)",
     "Total Energy — Daily Mean",           None),
    # Labour Loss
    ("d_labour_loss_tx", r"$\Delta$Labour loss (%)",
     "Labour loss — Dunne Tx",        None),
    # 睡眠损失 — 日均值
    ("d_sleep_loss_min", r"$\Delta$Sleep Loss (min/night)",
     "Sleep Loss Daily Mean — Minor 2022",  None),
    # 睡眠损失 — 暖季累积
    ("d_sleep_season_min", r"$\Sigma\Delta$Sleep (min/period)",
     "Sleep Loss Period Total — formal HNE sleep-valid nights", ["JJA","HW"]),
    # 经济损失箱线图
    ("d_econ_loss_pct_short_mean",
     r"$\Delta$Econ$_{sleep,short}$ (%/day)",
     "Sleep-Loss Productivity Loss Δ — Gibson 2018 β_short=1.1%/h", None),
    ("d_econ_loss_pct_long_mean",
     r"$\Delta$Econ$_{sleep,long}$ (%/day)",
     "Sleep-Loss Productivity Loss Δ — Gibson 2018 β_long=5.0%/h",  None),
    ("d_heat_loss_pct_mean",
     r"$\Delta$Heat Labour Loss (%/day)",
     "Direct Heat Labour Loss Δ — Graff Zivin 2014 α=0.5%/°C",     None),
    ("d_total_loss_pct_mean",
     r"$\Delta$Total Econ Loss (%/day)",
     "Combined Loss Δ (Sleep + Heat) — Period Mean",                None),
    ("econ_loss_usd_sleep_U_sum",
     r"$\Sigma$Econ$_{sleep}$ (USD/period, Urban)",
     "Sleep-Loss USD Total Urban — Period Sum",                     None),
    ("econ_loss_usd_heat_U_sum",
     r"$\Sigma$Econ$_{heat}$ (USD/period, Urban)",
     "Heat Labour USD Total Urban — Period Sum",                    ["JJA","HW"]),
    ("d_total_loss_usd_sum",
     r"$\Sigma\Delta$Total Loss (USD/period)",
     "Combined USD Δ (Urban−Rural) — Period Sum",                   ["JJA","HW"]),
]

# ─── Main-result boxplot 配置（新增，不替换原有 B OXPLOT_CFG） ─────────────
KEY_RESULT_BOXPLOT_CFG = [
    (
        "d_total_loss_pct_mean",
        r"$\Delta$Total Economic Loss (%/day)",
        "Main Result 2 — Economic Impact",
        None
    ),
    (
        "dCDH_total_sum",
        r"$\Sigma\Delta$CDH (°C·h/period)",
        "Main Result 3 — Energy Impact",
        None
    ),
]


# ══════════════════════════════════════════════════════════════════
# §2  诊断日志类
# ══════════════════════════════════════════════════════════════════

class DiagnosticLog:
    def __init__(self):
        self.sections = []
        self._t_start = time.time()

    def _col_stat(self, series, name=""):
        v = series.dropna().values.astype(float)
        n_nan  = int(series.isna().sum())
        n_all  = len(series)
        n_zero = int((v == 0).sum())
        if len(v) == 0:
            return f"  {name:40s}: ALL NaN  (n_all={n_all})\n"
        neg = int((v < 0).sum()); pos = int((v > 0).sum())
        pct_zero = 100 * n_zero / len(v)
        pct_nan  = 100 * n_nan  / n_all
        return (f"  {name:40s}: "
                f"n={len(v)}({pct_nan:.0f}%NaN/{pct_zero:.0f}%zero)  "
                f"mean={v.mean():+.4f}  med={np.median(v):+.4f}  "
                f"[p25={np.percentile(v,25):+.4f}, p75={np.percentile(v,75):+.4f}]  "
                f"pos/neg={pos}/{neg}\n")

    def _df_shape(self, df, label=""):
        n_pairs = df["pair_id"].nunique() if "pair_id" in df.columns else "?"
        periods = df["period_norm"].unique().tolist() if "period_norm" in df.columns else "?"
        return f"  shape={df.shape}  n_pairs={n_pairs}  periods={periods}  [{label}]\n"

    def add(self, heading, body):
        self.sections.append((heading, body))
        print(f"\n{'─'*60}\n  [DIAG] {heading}\n{'─'*60}")
        print(body)

    def record_load(self, name, df, key_cols=None):
        if df is None:
            self.add(f"LOAD: {name}", "  ⚠ 文件未找到或为空\n"); return
        buf = self._df_shape(df, "loaded")
        for c in (key_cols or []):
            if c in df.columns: buf += self._col_stat(df[c], c)
        self.add(f"LOAD: {name}", buf)

    def record_agg(self, name, df_raw, df_agg, key_cols=None):
        buf = f"  原始日面板: {df_raw.shape}  → pair×period汇总: {df_agg.shape}\n"
        for c in (key_cols or []):
            if c in df_raw.columns:
                v = df_raw[c].dropna().values
                pct_zero = 100*(v==0).mean() if len(v) else 0
                buf += f"    日面板 {c:35s}: mean={v.mean():+.4f}  %zero={pct_zero:.1f}%\n"
        buf += "  汇总后（去零稀释后的正确量级）:\n"
        for c in (key_cols or []):
            for sfx in ["_sum", "_mean", ""]:
                cc = c+sfx if sfx else c
                if cc in df_agg.columns:
                    buf += self._col_stat(df_agg[cc], cc); break
        self.add(f"AGG: {name}", buf)

    def record_merge_step(self, name, df, key_cols=None):
        buf = self._df_shape(df, "after_merge")
        for c in (key_cols or []):
            if c in df.columns: buf += self._col_stat(df[c], c)
        self.add(f"MERGE: {name}", buf)

    def record_final(self, df_pair, df_summary):
        buf  = f"  pair×period 宽表: {df_pair.shape}\n"
        buf += f"  diff_summary 表 : {df_summary.shape}\n\n"
        buf += "  【关键差值变量（配对均值）】\n"
        key_cols = [
            "d_sleep_loss_min", "d_labour_loss_tx",
            "dCDH_total_sum", "dE_comm_sum",
            "d_econ_loss_pct_short_mean", "d_heat_loss_pct_mean",
            "d_total_loss_pct_mean", "econ_loss_usd_sleep_U_sum",
        ]
        for col in key_cols:
            if col not in df_pair.columns: continue
            buf += f"\n  [{col}]\n"
            for grp in ["UHI","UCI"]:
                for per in PERIODS_PLOT:
                    sub = df_pair[
                        (df_pair.get("group", pd.Series(dtype=str)) == grp) &
                        (df_pair.get("period_norm", pd.Series(dtype=str)) == per)
                    ][col].dropna()
                    if len(sub) == 0: continue
                    buf += (f"    {grp} {per:6s}: n={len(sub):5d}  "
                            f"mean={sub.mean():+.4f}  median={sub.median():+.4f}\n")
        self.add("FINAL: 宽表关键统计", buf)

    def write(self, out_path):
        elapsed = time.time() - self._t_start
        lines = [
            "# Integrated Analysis v4 — Diagnostic Report",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
            f"**总耗时**: {elapsed:.1f} s",
            "", "---", "",
            "## §0  v4 新增修复（经济损失接入）+ v11 升级",
            "",
            "| 修复 | 内容 |",
            "|------|------|",
            "| FIX-A | `load_hne_paired()` hne_diff 补全 ~20 个经济损失列 及 CI 边界、KG分区变量 |",
            "| FIX-B | `pair_period_agg()` 新增 econ_pct→mean / econ_usd→sum+mean 汇总 |",
            "| FIX-C | `BOXPLOT_CFG` 新增 7 条经济损失箱线图；`build_summary_table()` 补 econ 区块 |",
            "",
            "## §Z  预期量级对照",
            "",
            "| 变量 | UHI HW 期合理均值 | 参考文献 |",
            "|------|------------------|---------|",
            "| d_econ_loss_pct_short (HW) | +0.01 ~ +0.10 %/day | β=1.1%/h × |ΔSleep|/60 |",
            "| d_heat_loss_pct (HW)       | +0.5 ~ +5.0 %/day  | α=0.5 × (Tmax−29) |",
            "| d_total_loss_pct (HW)      | +0.5 ~ +5.1 %/day  | sleep + heat |",
            "| econ_loss_usd_sleep_U (JJA/HW) | +0.01 ~ +0.5 USD/day | pct × wage × 8h |",
            "| dCDH_total_sum (HW)        | +50 ~ +500 °C·h     | CDH累加 |",
            "| d_labour_loss_tx (HW)      | sign depends on U-R contrast | Dunne Tx loss |",
            "",
        ]
        for heading, body in self.sections:
            lines += [f"### {heading}", "", "```", body.rstrip(), "```", ""]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n  ✓ DIAGNOSTIC_REPORT.md → {out_path}")


# ══════════════════════════════════════════════════════════════════
# §3  工具函数
# ══════════════════════════════════════════════════════════════════
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def lcz_to_density(val):
    remap = {51:1,52:2,53:3,54:4,55:5,56:6}
    try:
        v = int(float(val)); v = remap.get(v, v)
        if v in LCZ_HRHD: return "HRHD"
        if v in LCZ_LRLD: return "LRLD"
    except: pass
    return "OTHER"

def norm_period(p): return PERIOD_NORM.get(str(p), str(p))

def enforce_labour_hw_nhw_complete_pairs(
    df,
    value_col="d_labour_loss_tx",
    pair_col="pair_id",
    period_col="period_norm",
    diag=None,
):
    """
    Require the labour metric to be finite in both HW and NHW
    for the same station pair.

    Rules
    -----
    1. The matching unit is pair_id.
    2. A pair is retained for labour only when value_col is finite
       in both HW and NHW.
    3. Rows are not deleted from the full multi-outcome table.
    4. For pairs failing the matched requirement, value_col is set
       to NaN in both HW and NHW rows.
    5. Annual, JJA and all non-labour variables are unchanged.
    """
    if df is None:
        return df

    out = df.copy()

    required = {
        pair_col,
        period_col,
        value_col,
    }

    missing = required - set(out.columns)

    if missing:
        raise ValueError(
            "Cannot enforce matched HW/NHW labour completeness; "
            f"missing columns: {sorted(missing)}"
        )

    out[pair_col] = (
        out[pair_col]
        .astype(str)
        .str.strip()
    )

    out[value_col] = pd.to_numeric(
        out[value_col],
        errors="coerce",
    )

    period_key = (
        out[period_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    is_target_period = period_key.isin(
        ["HW", "NHW"]
    )

    is_finite_labour = np.isfinite(
        out[value_col].to_numpy(dtype=float)
    )

    valid_pair_period = out.loc[
        is_target_period & is_finite_labour,
        [pair_col],
    ].copy()

    valid_pair_period["_period_key"] = period_key.loc[
        is_target_period & is_finite_labour
    ].to_numpy()

    valid_pair_period = (
        valid_pair_period
        .drop_duplicates(
            subset=[
                pair_col,
                "_period_key",
            ]
        )
    )

    complete_flag = (
        valid_pair_period
        .groupby(
            pair_col,
            observed=True,
        )["_period_key"]
        .apply(
            lambda values: {
                "HW",
                "NHW",
            }.issubset(set(values))
        )
    )

    complete_ids = set(
        complete_flag[
            complete_flag
        ].index.astype(str)
    )

    target_ids = set(
        out.loc[
            is_target_period,
            pair_col,
        ].astype(str)
    )

    excluded_ids = (
        target_ids - complete_ids
    )

    # Only mask the labour metric.
    # Sleep, CDH, temperature, labour and energy fields remain unchanged.
    invalid_labour_rows = (
        is_target_period
        & ~out[pair_col].isin(
            complete_ids
        )
    )

    out.loc[
        invalid_labour_rows,
        value_col,
    ] = np.nan

    message = (
        f"value_col={value_col}\n"
        f"HW/NHW candidate pairs={len(target_ids)}\n"
        f"pairs valid in both HW and NHW={len(complete_ids)}\n"
        f"pairs excluded from labour only={len(excluded_ids)}\n"
        f"masked labour rows={int(invalid_labour_rows.sum())}"
    )

    if diag is not None:
        diag.add(
            "Matched HW/NHW labour completeness",
            message,
        )
    else:
        print(
            "\n  [Matched HW/NHW labour completeness]"
        )
        print(
            "    "
            + message.replace(
                "\n",
                "\n    ",
            )
        )

    return out

def sig_stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def sleep_loss_min(tmin):
    """
    Minor et al. (2022), Table S37 nighttime-Tmin spline.

    Returns
    -------
    numpy.ndarray
        Sleep-duration change in min/night.

        Negative values indicate sleep reduction.
        The model is zero at and below -20 deg C under the
        adopted spline normalization.
    """
    t = np.asarray(tmin, dtype=float)

    segment1 = np.clip(
        t - SLEEP_K1,
        0.0,
        SLEEP_K2 - SLEEP_K1,
    )

    segment2 = np.maximum(
        t - SLEEP_K2,
        0.0,
    )

    return (
        SLEEP_B1 * segment1
        + SLEEP_B2 * segment2
    )

def _pair_summary(series, pfx=""):
    v = series.dropna().values.astype(float)
    if len(v) == 0:
        return {f"{pfx}mean":np.nan, f"{pfx}se":np.nan, f"{pfx}median":np.nan,
                f"{pfx}p25":np.nan, f"{pfx}p75":np.nan, f"{pfx}n":0, f"{pfx}sig":""}
    _, pv = stats.ttest_1samp(v, 0) if len(v) > 2 else (0, np.nan)
    return {f"{pfx}mean":   round(float(v.mean()), 5),
            f"{pfx}se":     round(float(v.std(ddof=1)/np.sqrt(len(v))), 5),
            f"{pfx}median": round(float(np.median(v)), 5),
            f"{pfx}p25":    round(float(np.percentile(v, 25)), 5),
            f"{pfx}p75":    round(float(np.percentile(v, 75)), 5),
            f"{pfx}n":      int(len(v)),
            f"{pfx}sig":    sig_stars(pv)}



def _filter_percentile_rows(df):
    """Filter an upstream table to hw_method=percentile when available."""
    if df is None:
        return df
    out = df.copy()
    if "hw_method" in out.columns:
        out = out[
            out["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()
    return out


def derive_dunne_tx_labour_loss(df_lc):
    """
    Attach the formal upstream Dunne labour-loss result evaluated at each
    station-specific work-hour Tx.

    The upstream labour script has already used common-time air-temperature
    and dew-point curves, the 80% T/Td availability rule, two harmonics for
    both variables, shaded WBGT, and the Dunne response model. This plotting
    workflow must not reselect Tx from the main temperature curve and must not
    reconstruct RH, wet-bulb temperature, WBGT, or dew point.
    """
    if df_lc is None or len(df_lc) == 0:
        return df_lc

    out = df_lc.copy()

    required = {
        "pair_id",
        "period",
        "tx_hour_urban_dunne",
        "tx_hour_rural_dunne",
        "wbgt_tx_urban_dunne",
        "wbgt_tx_rural_dunne",
        "dloss_dunne_peak_t_diff",
        "urban_wbgt_input_ok",
        "rural_wbgt_input_ok",
    }

    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(
            "Updated common-time labour fields are missing from "
            "labour_loss_full.csv: "
            f"{missing}"
        )

    # Preserve merge keys as strings; coerce only scientific numeric fields.
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["period"] = out["period"].astype(str).str.strip()

    numeric_cols = required - {"pair_id", "period"}
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid = (
        out["urban_wbgt_input_ok"].eq(1)
        & out["rural_wbgt_input_ok"].eq(1)
        & out["tx_hour_urban_dunne"].notna()
        & out["tx_hour_rural_dunne"].notna()
        & out["wbgt_tx_urban_dunne"].notna()
        & out["wbgt_tx_rural_dunne"].notna()
        & out["dloss_dunne_peak_t_diff"].notna()
    )

    bad_hours = valid & (
        ~out["tx_hour_urban_dunne"].between(8, 19, inclusive="both")
        | ~out["tx_hour_rural_dunne"].between(8, 19, inclusive="both")
    )
    if bad_hours.any():
        examples = out.loc[
            bad_hours,
            ["pair_id", "period", "tx_hour_urban_dunne", "tx_hour_rural_dunne"],
        ].head(20)
        raise ValueError(
            "Upstream Dunne Tx hour is outside the frozen work window "
            "08:00–19:59. Examples:\n"
            + examples.to_string(index=False)
        )

    def _loss_from_wbgt(values):
        w = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.clip(
            25.0 * np.maximum(w - 25.0, 0.0) ** (2.0 / 3.0),
            0.0,
            100.0,
        )

    loss_u = _loss_from_wbgt(out["wbgt_tx_urban_dunne"])
    loss_r = _loss_from_wbgt(out["wbgt_tx_rural_dunne"])
    recomputed_delta = loss_u - loss_r
    upstream_delta = out["dloss_dunne_peak_t_diff"].to_numpy(dtype=float)

    mismatch = valid.to_numpy() & ~np.isclose(
        recomputed_delta,
        upstream_delta,
        rtol=1e-8,
        atol=1e-10,
        equal_nan=False,
    )
    if mismatch.any():
        idx = np.flatnonzero(mismatch)[:20]
        examples = out.iloc[idx][["pair_id", "period"]].copy()
        examples["upstream_delta"] = upstream_delta[idx]
        examples["recomputed_delta"] = recomputed_delta[idx]
        raise ValueError(
            "Upstream dloss_dunne_peak_t_diff is inconsistent with the "
            "upstream station-specific Tx-hour WBGT fields. Examples:\n"
            + examples.to_string(index=False)
        )

    out["tx_hour_urban"] = out["tx_hour_urban_dunne"].where(valid)
    out["tx_hour_rural"] = out["tx_hour_rural_dunne"].where(valid)
    out["wbgt_urban_tx"] = out["wbgt_tx_urban_dunne"].where(valid)
    out["wbgt_rural_tx"] = out["wbgt_tx_rural_dunne"].where(valid)
    out["labour_loss_tx_u"] = pd.Series(loss_u, index=out.index).where(valid)
    out["labour_loss_tx_r"] = pd.Series(loss_r, index=out.index).where(valid)

    # The formal pair contrast is read from the upstream labour analysis.
    out["d_labour_loss_tx"] = out["dloss_dunne_peak_t_diff"].where(valid)
    out["labour_tx_source"] = (
        "02_labour_capacity_loss.py: common-time two-harmonic T/Td, "
        "station-specific work-hour Tx, shaded WBGT, Dunne loss"
    )

    return out



def _normalize_positive_sleep_loss(df, diag=None):
    """Convert sleep-duration-change direction to positive sleep-loss burden."""
    if df is None or "d_sleep_loss_min" not in df.columns:
        return df
    out = df.copy()
    out["d_sleep_loss_min"] = -pd.to_numeric(out["d_sleep_loss_min"], errors="coerce")
    for c in ["sleep_loss_min_U", "sleep_loss_min_R"]:
        if c in out.columns:
            out[c] = -pd.to_numeric(out[c], errors="coerce")
    if {"d_sleep_loss_ci_low", "d_sleep_loss_ci_high"}.issubset(out.columns):
        old_low = pd.to_numeric(out["d_sleep_loss_ci_low"], errors="coerce")
        old_high = pd.to_numeric(out["d_sleep_loss_ci_high"], errors="coerce")
        out["d_sleep_loss_ci_low"] = -old_high
        out["d_sleep_loss_ci_high"] = -old_low
    if diag is not None:
        diag.add(
            "Sleep-loss sign convention",
            "Converted sleep-duration change to positive sleep-loss burden: "
            "positive = urban sleep loss > rural sleep loss."
        )
    return out

# ══════════════════════════════════════════════════════════════════
# §4  数据加载
# ══════════════════════════════════════════════════════════════════

def load_cdh_panel(diag):
    p = os.path.join(CDH_DIR, "all_pairs_daily_panel.csv")
    if not os.path.exists(p): raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=["local_date"])
    df = _filter_percentile_rows(df)
    df["period_norm"] = df["period"].apply(norm_period)
    diag.record_load("CDH daily panel", df,
                     ["dCDH_total","dE_comm","dE_resi","dTmean","dTmin","dTmax"])
    return df

def load_cdh_hourly():
    p = os.path.join(CDH_DIR, "hourly_profiles.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    df["period_norm"] = df["period"].apply(norm_period)
    return df


def load_labour_full(diag):
    p = os.path.join(LABOUR_DIR, "labour_loss_full.csv")
    if not os.path.exists(p):
        diag.record_load("Labour (labour_loss_full.csv)", None)
        return None
    df = pd.read_csv(p)
    df = _filter_percentile_rows(df)
    df["period_norm"] = df["period"].apply(norm_period)
    df = derive_dunne_tx_labour_loss(df)
    df = apply_canonical_uhi_uci_group(df, diag=diag)
    diag.record_load(
        "Labour (Dunne Tx labour loss)", df,
        ["d_labour_loss_tx", "labour_loss_tx_u", "labour_loss_tx_r"],
    )
    return df


def load_hne_paired(diag):
    """
    加载 HNE 年度及暖季数据，包含物理量、经济损失三渠道及 MC 置信边界。
    """
    frames = []
    for rel in ["all_pairs_annual.csv",
                "warm_season_periods/all_pairs_warm_periods.csv"]:
        fp = os.path.join(HNE_DIR, rel)
        if os.path.exists(fp):
            tmp = pd.read_csv(fp)
            if "period" not in tmp.columns: tmp["period"] = "annual"
            tmp = _filter_percentile_rows(tmp)
            tmp["period_norm"] = tmp["period"].apply(norm_period)
            frames.append(tmp)
        else:
            diag.add(f"LOAD: HNE [{rel}]", f"  ⚠ 文件不存在: {fp}\n")

    if not frames:
        diag.add("LOAD: HNE (both files)", "  ⚠ 两个 HNE 文件均未找到\n")
        return None

    df = pd.concat(frames, ignore_index=True)

    diag.record_load("HNE (annual + warm_periods concat)", df,
                     ["d_sleep_loss_min", "dHNE", "dTMIN",
                      "econ_loss_pct_short_U", "heat_loss_pct_U",
                      "total_loss_pct_U", "econ_loss_usd_sleep_U",
                      "d_econ_loss_pct_short", "d_heat_loss_pct",
                      "d_total_loss_pct", "d_total_loss_usd",
                      "sleep_loss_ci_low_U"])
    diag.record_load("HNE (annual + warm_periods concat)", df,
                     ["d_sleep_loss_min", "dHNE", "temp_h00_U", "temp_h00_R"])
    return df


def load_pair_meta():
    df = pd.read_csv(PAIR_CSV)
    for c in ["urban_lcz_corrected","urban_lcz_raw","urban_lcz"]:
        if c in df.columns: df["lcz_urban"] = df[c]; break
    if "lcz_urban" not in df.columns: df["lcz_urban"] = np.nan
    df["density"] = df["lcz_urban"].apply(lcz_to_density)
    return df


def load_canonical_pair_groups(path=CANONICAL_GROUP_CSV):
    """Load strict annual percentile pair_id -> UHI/UCI classification."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Canonical UHI/UCI group file not found: {path}")
    g = pd.read_csv(path)
    required = {"pair_id", "period", "group"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"Canonical group file missing columns: {sorted(missing)}")
    g = _filter_percentile_rows(g)
    g = g[g["period"].astype(str).str.lower().str.strip().eq("annual")].copy()
    g["pair_id"] = g["pair_id"].astype(str)
    g["group"] = g["group"].astype(str).str.upper().str.strip()
    g = g[g["group"].isin(["UHI", "UCI"])].copy()
    if g.empty:
        raise ValueError("No annual percentile UHI/UCI rows in canonical source.")
    conflicts = g.groupby("pair_id", observed=True)["group"].nunique()
    conflict_ids = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflict_ids:
        raise ValueError(
            "Conflicting annual UHI/UCI classifications for "
            f"{len(conflict_ids)} pair(s); examples={conflict_ids[:20]}"
        )
    return g[["pair_id", "group"]].drop_duplicates("pair_id").reset_index(drop=True)




def apply_canonical_uhi_uci_group(df, diag=None):
    """Replace module-local labels with strict annual canonical groups."""
    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)
    canonical = load_canonical_pair_groups()
    if "group" in out.columns:
        out["group_module_original"] = out["group"]
        out = out.drop(columns=["group"])
    before = out["pair_id"].nunique()
    out = out.merge(canonical, on="pair_id", how="inner")
    after = out["pair_id"].nunique()
    if diag is not None:
        diag.add(
            "Canonical annual UHI/UCI group merge",
            f"pairs before={before}; retained annual-classified pairs={after}; "
            f"excluded without annual group={before-after}"
        )
    if out.empty:
        raise ValueError("No rows remain after strict annual UHI/UCI merge.")
    return out




def load_plot_results_fig23_matched_pair_ids(
        path=CANONICAL_GROUP_CSV,
        required_metrics=REFERENCE_MATCH_REQUIRED_METRICS):
    """Return strict annual-group percentile pairs complete in HW and NHW."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference input not found: {path}")
    ref = _filter_percentile_rows(pd.read_csv(path))
    required = {"pair_id", "period", *required_metrics}
    missing = required - set(ref.columns)
    if missing:
        raise ValueError(f"Reference cohort columns missing: {sorted(missing)}")
    ref["pair_id"] = ref["pair_id"].astype(str)
    ref["_period_key"] = ref["period"].astype(str).str.lower().str.strip()
    for c in required_metrics:
        ref[c] = pd.to_numeric(ref[c], errors="coerce")
    nhw = ref.loc[ref["_period_key"].eq("non_heatwave"), ["pair_id", *required_metrics]].copy()
    hw = ref.loc[ref["_period_key"].eq("heatwave"), ["pair_id", *required_metrics]].copy()
    nhw = nhw.rename(columns={c: f"{c}_nhw" for c in required_metrics})
    hw = hw.rename(columns={c: f"{c}_hw" for c in required_metrics})
    paired = nhw.merge(hw, on="pair_id", how="inner")
    complete = [f"{c}_nhw" for c in required_metrics] + [f"{c}_hw" for c in required_metrics]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=complete)
    canonical = load_canonical_pair_groups(path)
    paired = paired.merge(canonical, on="pair_id", how="inner")
    matched_ids = set(paired["pair_id"].astype(str))
    if not matched_ids:
        raise ValueError("Strict Fig23 matched cohort is empty.")
    group_counts = paired[["pair_id", "group"]].drop_duplicates()["group"].value_counts().to_dict()
    print("\n  [plot_results_fig23 reference cohort]")
    print(f"    matched complete pair IDs          : {len(matched_ids)}")
    print(f"    annual canonical group counts      : {group_counts}")
    print(f"    required reference metrics         : {list(required_metrics)}")
    return matched_ids



# ══════════════════════════════════════════════════════════════════
# §5  [FIX-1 + FIX-B] pair×period 汇总
# ══════════════════════════════════════════════════════════════════

_SLEEP_B1_ANNUAL  = -0.107

_WARM_PERIODS     = {"JJA", "HW", "NHW"}


def _is_econ_pct_col(c):
    return any(c.startswith(p) for p in ECON_PCT_PREFIXES)

def _is_econ_usd_col(c):
    return any(c.startswith(p) for p in ECON_USD_PREFIXES)


def pair_period_agg(df_cdh, diag):
    sum_cols  = [c for c in df_cdh.columns if (
        c.startswith("dCDH") or c.startswith("dHDH") or c.startswith("dE_"))]
    mean_cols = [c for c in df_cdh.columns if (
        c.startswith("dCOP") or c.startswith("dTmean") or
        c.startswith("dTmax") or c.startswith("dTmin"))]
    meta_pick = [c for c in ["group","strata","density","continent","period_norm"]
                 if c in df_cdh.columns]
    abs_u_cols = [c for c in df_cdh.columns
                  if c.endswith("_u") and not c.startswith("d") and
                  any(c.startswith(p) for p in
                      ["CDH","HDH","E_","COP","Tmax","Tmin","Tmean"])]
    abs_r_cols = [c.replace("_u","_r") for c in abs_u_cols
                  if c.replace("_u","_r") in df_cdh.columns]

    econ_pct_in_cdh = [c for c in df_cdh.columns if _is_econ_pct_col(c)]
    econ_usd_in_cdh = [c for c in df_cdh.columns if _is_econ_usd_col(c)]

    rows = []
    for (pid, pnorm), sub in df_cdh.groupby(["pair_id","period_norm"], observed=True):
        row = {"pair_id": pid, "period_norm": pnorm}

        for c in meta_pick:
            m = sub[c].dropna()
            row[c] = m.mode().iloc[0] if len(m) > 0 else np.nan

        n_days = len(sub)
        row["n_days"] = n_days

        # Period labels are inherited from the upstream hemisphere-aware
        # analysis. Do not reconstruct global warm season from calendar JJA.
        if pnorm in _WARM_PERIODS:
            row["n_days_warm"] = n_days
            row["n_days_cool"] = 0
        else:
            row["n_days_warm"] = np.nan
            row["n_days_cool"] = np.nan

        for c in sum_cols:
            if c not in sub.columns: continue
            v = sub[c].dropna()
            row[c+"_sum"]  = float(v.sum())
            row[c+"_mean"] = float(v.mean()) if len(v) else np.nan
            row[c+"_nz"]   = int((v.abs() > 0).sum())

        for c in mean_cols:
            if c not in sub.columns: continue
            row[c+"_mean"] = float(sub[c].dropna().mean())
            row[c]         = row[c+"_mean"]

        # Sleep outcomes are supplied only by the formal HNE/sleep daily
        # panel. The CDH panel must not generate either a nightly mean or a
        # seasonal total, and its day count must not be used as a sleep-night
        # denominator. These fields are populated after the HNE merge.
        row["d_sleep_loss_min"] = np.nan
        row["d_sleep_season_min"] = np.nan
        row["sleep_valid_nights"] = np.nan

        for c in econ_pct_in_cdh:
            if c not in sub.columns: continue
            row[c+"_mean"] = float(sub[c].dropna().mean())

        for c in econ_usd_in_cdh:
            if c not in sub.columns: continue
            v = sub[c].dropna()
            row[c+"_sum"]  = float(v.sum())
            row[c+"_mean"] = float(v.mean()) if len(v) else np.nan

        for c in abs_u_cols + abs_r_cols:
            if c in sub.columns:
                row[c+"_sum"]  = float(sub[c].sum())
                row[c+"_mean"] = float(sub[c].mean())

        rows.append(row)

    df_agg = pd.DataFrame(rows)
    diag.record_agg(
        "CDH daily→pair_period [v4: +econ_pct_mean, +econ_usd_sum]",
        df_raw=df_cdh, df_agg=df_agg,
        key_cols=["dCDH_total","dE_comm","dE_total",
                  "d_sleep_loss_min","d_sleep_season_min"]
    )
    return df_agg


# ══════════════════════════════════════════════════════════════════
# §6  合并模块
# ══════════════════════════════════════════════════════════════════

def merge_all_modules(df_cdh_agg, df_lc, df_hne, meta, diag):
    mc = [c for c in ["pair_id","density","continent","lat_urban","lon_urban"]
          if c in meta.columns]
    df = df_cdh_agg.merge(meta[mc], on="pair_id", how="left", suffixes=("","_m"))
    df["density"] = df.get("density", pd.Series("OTHER", index=df.index)).fillna("OTHER")

    df = apply_canonical_uhi_uci_group(df, diag=diag)
    if df_hne is not None:
        # Count formal paired sleep-valid nights directly from the HNE daily
        # panel. A valid night requires finite urban/rural nighttime Tmin and
        # finite urban/rural sleep-model outputs. This is the only denominator
        # allowed for d_sleep_season_min.
        df_hne = df_hne.copy()
        df_hne["pair_id"] = df_hne["pair_id"].astype(str).str.strip()
        df_hne["period_norm"] = df_hne["period_norm"].apply(norm_period)
        sleep_required = [
            "sleep_loss_min_U", "sleep_loss_min_R",
            "tmin_night_U", "tmin_night_R",
        ]
        sleep_valid_counts = pd.DataFrame(
            columns=["pair_id", "period_norm", "sleep_valid_nights"]
        )
        if all(c in df_hne.columns for c in sleep_required):
            valid_sleep = pd.Series(True, index=df_hne.index)
            for c in sleep_required:
                valid_sleep &= pd.to_numeric(df_hne[c], errors="coerce").notna()
            valid_rows = df_hne.loc[valid_sleep].copy()
            date_col = next((c for c in ["date", "local_date"] if c in valid_rows.columns), None)
            if date_col is not None:
                valid_rows[date_col] = pd.to_datetime(valid_rows[date_col], errors="coerce")
                valid_rows = valid_rows[valid_rows[date_col].notna()].copy()
                sleep_valid_counts = (
                    valid_rows.groupby(["pair_id", "period_norm"], observed=True)[date_col]
                    .nunique()
                    .rename("sleep_valid_nights")
                    .reset_index()
                )
            else:
                sleep_valid_counts = (
                    valid_rows.groupby(["pair_id", "period_norm"], observed=True)
                    .size()
                    .rename("sleep_valid_nights")
                    .reset_index()
                )

        hne_want = [
            # 物理量
            "d_sleep_loss_min","dHNE","dTMIN",
            "sleep_loss_min_U","sleep_loss_min_R",
            "tmin_night_U","tmin_night_R",
            "hne_d_U","hne_d_R",
            # 新增 v11 不确定性变量与分类变量
            "sleep_loss_ci_low_U", "sleep_loss_ci_high_U",
            "sleep_loss_ci_low_R", "sleep_loss_ci_high_R",
            "d_sleep_loss_ci_low", "d_sleep_loss_ci_high",
            "kg_code", "kg_group", "climate_zone_main",
            "hne_ref_mode", "hw_ref_mode",
            # 睡眠渠道经济损失
            "econ_loss_pct_short_U", "econ_loss_pct_short_R",
            "econ_loss_pct_long_U",  "econ_loss_pct_long_R",
            "econ_loss_pct_hourly_U","econ_loss_pct_hourly_R",
            "econ_loss_pct_present_U","econ_loss_pct_present_R",
            "econ_loss_usd_sleep_U", "econ_loss_usd_sleep_R",
            "d_econ_loss_pct_short","d_econ_loss_pct_long",
            "d_econ_loss_pct_hourly","d_econ_loss_pct_present",
            "d_econ_loss_usd_sleep",
            # 直接高温渠道
            "heat_loss_pct_U","heat_loss_pct_R",
            "econ_loss_usd_heat_U","econ_loss_usd_heat_R",
            "d_heat_loss_pct","d_econ_loss_usd_heat",
            # 合并总损失
            "total_loss_pct_U","total_loss_pct_R",
            "total_loss_usd_U","total_loss_usd_R",
            "d_total_loss_pct","d_total_loss_usd",
        ]
        for h in range(24):
            hne_want.append(f"temp_h{h:02d}_U")
            hne_want.append(f"temp_h{h:02d}_R")
        hne_diff = [c for c in hne_want if c in df_hne.columns]

        if hne_diff:
            hne_pct_cols = [c for c in hne_diff if _is_econ_pct_col(c) or
                            c.startswith("d_econ_loss_pct") or
                            c.startswith("d_heat_loss_pct") or
                            c.startswith("d_total_loss_pct")]
            hne_usd_cols = [c for c in hne_diff if _is_econ_usd_col(c) or
                            c.startswith("d_econ_loss_usd") or
                            c.startswith("d_econ_loss_usd_heat") or
                            c.startswith("d_total_loss_usd")]
            hne_cat_cols = [c for c in hne_diff if c in [
                "kg_code", "kg_group", "climate_zone_main", "hne_ref_mode", "hw_ref_mode"]]
            hne_mean_cols = [c for c in hne_diff
                             if c not in hne_pct_cols and c not in hne_usd_cols and c not in hne_cat_cols]

            agg_parts = []
            if hne_mean_cols:
                agg_parts.append(
                    df_hne.groupby(["pair_id","period_norm"], observed=True)
                    [hne_mean_cols].mean().reset_index()
                )
            if hne_cat_cols:
                agg_parts.append(
                    df_hne.groupby(["pair_id","period_norm"], observed=True)
                    [hne_cat_cols].first().reset_index()
                )
            if hne_pct_cols:
                tmp_pct = (df_hne.groupby(["pair_id","period_norm"], observed=True)
                           [hne_pct_cols].mean().reset_index())
                rename_pct = {c: c+"_mean" for c in hne_pct_cols
                              if c+"_mean" not in tmp_pct.columns}
                tmp_pct = tmp_pct.rename(columns=rename_pct)
                agg_parts.append(tmp_pct)
            if hne_usd_cols:
                tmp_sum = (df_hne.groupby(["pair_id","period_norm"], observed=True)
                           [hne_usd_cols].sum().reset_index())
                tmp_avg = (df_hne.groupby(["pair_id","period_norm"], observed=True)
                           [hne_usd_cols].mean().reset_index())
                rename_sum = {c: c+"_sum" for c in hne_usd_cols}
                rename_avg = {c: c+"_mean" for c in hne_usd_cols}
                tmp_sum = tmp_sum.rename(columns=rename_sum)
                tmp_avg = tmp_avg.rename(columns=rename_avg)
                agg_parts.append(tmp_sum)
                agg_parts.append(tmp_avg)

            if agg_parts:
                hne_agg = agg_parts[0]
                for part in agg_parts[1:]:
                    on_cols = ["pair_id","period_norm"]
                    hne_agg = hne_agg.merge(part, on=on_cols, how="outer")
                if len(sleep_valid_counts) > 0:
                    hne_agg = hne_agg.merge(
                        sleep_valid_counts,
                        on=["pair_id", "period_norm"],
                        how="left",
                        validate="one_to_one",
                    )
                df = df.merge(hne_agg, on=["pair_id","period_norm"],
                              how="left", suffixes=("","_hne"))
                
                all_combined_cols = (hne_mean_cols + hne_cat_cols +
                                     [c+"_mean" for c in hne_pct_cols] +
                                     [c+"_sum" for c in hne_usd_cols] +
                                     [c+"_mean" for c in hne_usd_cols] +
                                     (["sleep_valid_nights"] if "sleep_valid_nights" in hne_agg.columns else []))
                
                for c in all_combined_cols:
                    hc = c + "_hne"
                    if hc in df.columns:
                        df[c] = df[hc].combine_first(
                            df.get(c, pd.Series(np.nan, index=df.index)))

            key_diag = ["d_sleep_loss_min",
                        "d_econ_loss_pct_short_mean", "d_heat_loss_pct_mean",
                        "d_total_loss_pct_mean",
                        "econ_loss_usd_sleep_U_sum", "d_total_loss_usd_sum",
                        "sleep_loss_ci_low_U"]
            diag.record_merge_step(
                "HNE → pair×period (含三渠道经济损失及MC置信度)",
                df, key_cols=[c for c in key_diag if c in df.columns])

    # Formal sleep loss is read only from the HNE/sleep module.
    # Missing values remain NaN; no dTmin-based fallback is allowed here.
    if "d_sleep_loss_min" not in df.columns:
        df["d_sleep_loss_min"] = np.nan
    n_missing_sleep = int(df["d_sleep_loss_min"].isna().sum())
    diag.add(
        "Formal sleep-loss availability",
        "  Sleep loss is read only from the HNE/sleep module; "
        "no dTmin fallback is applied.\n"
        f"  missing pair-period rows: {n_missing_sleep}\n",
    )

    if df_lc is not None:
        lc_cols_raw = [c for c in df_lc.columns if (
            c.startswith("net_effect_") or
            c.startswith("dlc_") or
            c.startswith("dwbgt_") or
            c in {"d_labour_loss_tx", "labour_loss_tx_u", "labour_loss_tx_r", "tx_hour_urban", "tx_hour_rural"})]

        # Keep only numeric labour columns.
        # Exclude metadata columns such as:
        #   legacy_workhour_capacity_definition = "work_hour_mean_08_19"
        lc_cols = []
        lc_meta_cols = []

        for c in lc_cols_raw:
            vals = pd.to_numeric(df_lc[c], errors="coerce")
            if vals.notna().any():
                lc_cols.append(c)
            else:
                lc_meta_cols.append(c)

        if lc_meta_cols:
            diag.add(
                "Labour merge: skipped non-numeric metadata columns",
                "  These columns are not merged into numeric pair-period analysis:\n"
                + "\n".join(f"  - {c}" for c in lc_meta_cols[:50])
                + ("\n  ..." if len(lc_meta_cols) > 50 else "")
            )

        lc_sub = (df_lc[["pair_id","period_norm"]+lc_cols]
                  .drop_duplicates(["pair_id","period_norm"]))
        df = df.merge(lc_sub, on=["pair_id","period_norm"],
                      how="left", suffixes=("","_lc"))
        for c in lc_cols:
            lc = c+"_lc"
            if lc in df.columns:
                df[c] = df[lc].combine_first(
                    df.get(c, pd.Series(np.nan, index=df.index)))
        diag.record_merge_step("Labour → pair×period", df,
                               key_cols=["d_labour_loss_tx"])

    print(f"\n  Merge complete: {len(df):,} rows  pairs={df['pair_id'].nunique()}")
    df = _normalize_positive_sleep_loss(df, diag=diag)

    # Seasonal sleep total uses the formal HNE daily-panel denominator only.
    # It is not multiplied by CDH days or a fixed JJA day count.
    if "sleep_valid_nights" not in df.columns:
        df["sleep_valid_nights"] = np.nan
    df["sleep_valid_nights"] = pd.to_numeric(
        df["sleep_valid_nights"], errors="coerce"
    )
    warm_mask = df["period_norm"].isin(_WARM_PERIODS)
    valid_total = (
        warm_mask
        & pd.to_numeric(df["d_sleep_loss_min"], errors="coerce").notna()
        & df["sleep_valid_nights"].gt(0)
    )
    df["d_sleep_season_min"] = np.nan
    df.loc[valid_total, "d_sleep_season_min"] = (
        pd.to_numeric(df.loc[valid_total, "d_sleep_loss_min"], errors="coerce")
        * df.loc[valid_total, "sleep_valid_nights"]
    )
    diag.add(
        "Seasonal sleep-total denominator",
        "  d_sleep_season_min = positive urban-minus-rural nightly sleep "
        "loss × paired HNE sleep-valid nights.\n"
        "  sleep-valid requires finite U/R nighttime Tmin and finite U/R "
        "Minor-model sleep outputs.\n"
        f"  finite seasonal totals: {int(df['d_sleep_season_min'].notna().sum())}\n",
    )

    return df


# ══════════════════════════════════════════════════════════════════
# §6b  百分比变化计算
# ══════════════════════════════════════════════════════════════════

def compute_pct_changes(df):
    df = df.copy()

    cdh_pairs = [
        ("CDH_total_u_mean", "CDH_total_r_mean", "CDH_total_pct_chg"),
        ("CDH_comm_u_mean",  "CDH_comm_r_mean",  "CDH_comm_pct_chg"),
        ("HDH_total_u_mean", "HDH_total_r_mean", "HDH_total_pct_chg"),
        ("E_comm_u_mean",    "E_comm_r_mean",    "E_comm_pct_chg"),
        ("E_resi_u_mean",    "E_resi_r_mean",    "E_resi_pct_chg"),
        ("E_total_u_mean",   "E_total_r_mean",   "E_total_pct_chg"),
        ("Tmax_u_mean",      "Tmax_r_mean",      "Tmax_pct_chg"),
        ("Tmin_u_mean",      "Tmin_r_mean",      "Tmin_pct_chg"),
    ]

    for uc, rc, out_c in cdh_pairs:
        if uc in df.columns and rc in df.columns:
            denom = df[rc].abs().clip(lower=0.1)
            df[out_c] = (df[uc] - df[rc]) / denom * 100

    if (
        "sleep_loss_min_U" in df.columns
        and "sleep_loss_min_R" in df.columns
    ):
        denom = df["sleep_loss_min_R"].abs().clip(lower=0.01)
        df["sleep_loss_pct_chg"] = (
            (df["sleep_loss_min_U"] - df["sleep_loss_min_R"])
            / denom
            * 100
        )

    econ_abs_pairs = [
        (
            "econ_loss_usd_sleep_U",
            "econ_loss_usd_sleep_R",
            "econ_sleep_usd_pct_chg",
        ),
        (
            "econ_loss_usd_heat_U",
            "econ_loss_usd_heat_R",
            "econ_heat_usd_pct_chg",
        ),
        (
            "total_loss_usd_U",
            "total_loss_usd_R",
            "econ_total_usd_pct_chg",
        ),
        (
            "econ_loss_pct_short_U",
            "econ_loss_pct_short_R",
            "econ_sleep_pct_pct_chg",
        ),
        (
            "heat_loss_pct_U",
            "heat_loss_pct_R",
            "econ_heat_pct_pct_chg",
        ),
    ]

    for uc, rc, out_c in econ_abs_pairs:
        if uc in df.columns and rc in df.columns:
            denom = df[rc].abs().clip(lower=0.001)
            df[out_c] = (df[uc] - df[rc]) / denom * 100

    if "hne_d_U" in df.columns and "hne_d_R" in df.columns:
        denom = df["hne_d_R"].clip(lower=0.01)
        df["hne_pct_chg"] = (
            (df["hne_d_U"] - df["hne_d_R"])
            / denom
            * 100
        )

    cdh_sum_pairs = [
        (
            "CDH_total_u_sum",
            "CDH_total_r_sum",
            "CDH_total_sum_pct_chg",
        ),
        (
            "E_comm_u_sum",
            "E_comm_r_sum",
            "E_comm_sum_pct_chg",
        ),
        (
            "E_resi_u_sum",
            "E_resi_r_sum",
            "E_resi_sum_pct_chg",
        ),
    ]

    for uc, rc, out_c in cdh_sum_pairs:
        if uc in df.columns and rc in df.columns:
            denom = df[rc].abs().clip(lower=1.0)
            df[out_c] = (df[uc] - df[rc]) / denom * 100

    new_pct_cols = [
        c for c in df.columns
        if c.endswith("_pct_chg")
    ]

    print(
        f"  [pct_chg] 新增百分比变化列: {new_pct_cols}"
    )

    return df


# ══════════════════════════════════════════════════════════════════
# §7  统计汇总（配对级）
# ══════════════════════════════════════════════════════════════════

def build_diff_summary(df, diag):
    diff_cols = [
        c for c in df.columns
        if (
            c.endswith("_sum")
            or c.endswith("_mean")
            or c.startswith("net_effect_")
            or c in (
                "d_sleep_loss_min",
                "d_sleep_season_min",
                "dHNE",
                "dTMIN",
                "sleep_loss_min_U",
                "sleep_loss_min_R",
                "sleep_loss_ci_low_U",
                "sleep_loss_ci_high_U",
                "d_sleep_loss_ci_low",
                "d_sleep_loss_ci_high",
                "hne_d_U",
                "hne_d_R",
                "tmin_night_U",
                "tmin_night_R",
            )
            or c.startswith("dlc_")
            or c.startswith("dTmax_fft")
            or c.startswith("dTmin_fft")
            or c.startswith("dwbgt_")
            or c.endswith("_pct_chg")
        )
        and c not in (
            "pair_id",
            "period_norm",
            "group",
            "density",
            "strata",
        )
    ]

    numeric_diff_cols = []
    skipped_non_numeric = []

    for c in diff_cols:
        vals = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        if vals.notna().any():
            numeric_diff_cols.append(c)
        else:
            skipped_non_numeric.append(c)

    if skipped_non_numeric:
        diag.add(
            "build_diff_summary: skipped non-numeric columns",
            "  These columns matched diff-column name patterns "
            "but are not numeric:\n"
            + "\n".join(
                f"  - {c}"
                for c in skipped_non_numeric[:50]
            )
            + (
                "\n  ..."
                if len(skipped_non_numeric) > 50
                else ""
            ),
        )

    diff_cols = numeric_diff_cols

    rows = []

    for (grp, dens, per), sub in df.groupby(
        ["group", "density", "period_norm"],
        observed=True,
    ):
        row = {
            "group": grp,
            "density": dens,
            "period": per,
            "n_pairs": len(sub),
        }

        for c in diff_cols:
            row.update(
                _pair_summary(
                    sub[c],
                    pfx=f"{c}_",
                )
            )

        rows.append(row)

    df_sum = pd.DataFrame(rows)

    diag.record_final(
        df,
        df_sum,
    )

    return df_sum
# ══════════════════════════════════════════════════════════════════
# §7b  综合统计汇总表
# ══════════════════════════════════════════════════════════════════

LIT_REFS = {
    "temp":  ("Urban Tmax +0.3–2°C vs rural (Oke 1982 QJRMS 108:1); "
              "Night UHI > Day UHI (Li & Bou-Zeid 2013 ERL 8:034002)"),
    "cdh":   ("Urban CDH 10–25% higher per +1°C UHI (Sailor & Vasireddy 2006); "
              "HRHD cooling energy 20–40% higher (Kolokotroni et al. 2012 E&B 55:341)"),
    "lc_dunne": ("LC → 0 at WBGT > 32°C; heavy labor loss ~20% by 2100 RCP8.5 "
                 "(Dunne et al. 2013 Nat.CC 3:563)"),
    "lc_he":    ("Global mean workability loss 1.5%/yr; tropics 3–5%/yr "
                 "(He et al. 2022 One Earth 5:700)"),
    "sleep":    ("-0.618 min/night per °C Tmin (>10°C); "
                 "projected -58 h/person/yr by 2099 RCP8.5 "
                 "(Minor et al. 2022 One Earth 5:639, Table S37)"),
    "econ_sleep": ("ΔProductivity% = (β/60)×|ΔSleep|; "
                   "β_short=1.1%/h (Gibson & Shrader 2018 Rev.Econ.Stat. 100:838); "
                   "β_long=5.0%/h; β_hourly=4.2%/h (Costa-Font et al. 2024 JHE 93:102832)"),
    "econ_heat":  ("heat_loss_pct = 0.5×max(Tmax−29,0); "
                   "Graff Zivin & Neidell (2014) JPE 122:1391; "
                   "Seppanen et al. (2006) Indoor Air 16:28; Tref=29°C: Kjellstrom 2009"),
    "econ_total": ("total_loss = sleep_loss + heat_loss; "
                   "USD: (pct/100)×wage×8h; "
                   "Deschenes (2014) NBER WP18692; Kjellstrom et al. (2016) IJERPH 13:1326"),
}


def build_summary_table(df, out_dir):
    ensure_dir(out_dir)

    def _mean(sub, col):
        if col not in sub.columns: return np.nan
        return round(float(sub[col].dropna().mean()), 4)

    def _se(sub, col):
        if col not in sub.columns: return np.nan
        v = sub[col].dropna()
        return round(float(v.sem()), 4) if len(v) > 1 else np.nan

    rows = []
    for grp in ["UHI","UCI"]:
        for per in PERIODS_PLOT:
            sub = df[(df["group"]==grp) & (df["period_norm"]==per)]
            if len(sub) == 0: continue

            row = {
                "Group":   grp,
                "Period":  per,
                "n_pairs": len(sub),

                "Tmax_urban_°C":  _mean(sub,"Tmax_u_mean"),
                "Tmax_rural_°C":  _mean(sub,"Tmax_r_mean"),
                "dTmax_°C":       _mean(sub,"dTmax_mean"),
                "Tmin_urban_°C":  _mean(sub,"Tmin_u_mean"),
                "Tmin_rural_°C":  _mean(sub,"Tmin_r_mean"),
                "dTmin_°C":       _mean(sub,"dTmin_mean"),
                "Tmean_urban_°C": _mean(sub,"Tmean_u_mean"),
                "Tmean_rural_°C": _mean(sub,"Tmean_r_mean"),
                "dTmean_°C":      _mean(sub,"dTmean_mean"),

                "dCDH_total_°Ch":      _mean(sub,"dCDH_total_sum"),
                "dCDH_total_°Ch/day":  _mean(sub,"dCDH_total_mean"),
                "CDH_urban_°Ch":       _mean(sub,"CDH_total_u_sum"),
                "CDH_rural_°Ch":       _mean(sub,"CDH_total_r_sum"),
                "dCDH_pct_chg_%":      _mean(sub,"CDH_total_pct_chg"),

                "dE_comm_Wh":          _mean(sub,"dE_comm_sum"),
                "dE_comm_Wh/day":      _mean(sub,"dE_comm_mean"),
                "dE_resi_Wh":          _mean(sub,"dE_resi_sum"),
                "dE_resi_Wh/day":      _mean(sub,"dE_resi_mean"),
                "dE_total_Wh":         _mean(sub,"dE_total_sum"),
                "dE_total_Wh/day":     _mean(sub,"dE_total_mean"),

                "dLabourLoss_Tx_%": _mean(sub,"d_labour_loss_tx"),

                "sleep_urban_min/night": _mean(sub,"sleep_loss_min_U"),
                "sleep_rural_min/night": _mean(sub,"sleep_loss_min_R"),
                "d_sleep_min/night":     _mean(sub,"d_sleep_loss_min"),
                "d_sleep_season_min":    _mean(sub,"d_sleep_season_min"),
                "sleep_loss_pct_chg_%":  _mean(sub,"sleep_loss_pct_chg"),
                "d_sleep_SE":            _se(sub,  "d_sleep_loss_min"),

                "d_econ_sleep_pct_short_%/day":  _mean(sub,"d_econ_loss_pct_short_mean"),
                "d_econ_sleep_pct_long_%/day":   _mean(sub,"d_econ_loss_pct_long_mean"),
                "d_econ_sleep_SE":               _se(sub,  "d_econ_loss_pct_short_mean"),
                "econ_sleep_pct_urban_%/day":    _mean(sub,"econ_loss_pct_short_U"),
                "econ_sleep_pct_rural_%/day":    _mean(sub,"econ_loss_pct_short_R"),
                "econ_sleep_usd_urban_sum":      _mean(sub,"econ_loss_usd_sleep_U_sum"),
                "econ_sleep_usd_rural_sum":      _mean(sub,"econ_loss_usd_sleep_R_sum"),
                "d_econ_sleep_usd_sum":          _mean(sub,"d_econ_loss_usd_sleep_sum"),
                "econ_sleep_usd_pct_chg_%":      _mean(sub,"econ_sleep_usd_pct_chg"),

                "d_econ_heat_pct_%/day":         _mean(sub,"d_heat_loss_pct_mean"),
                "d_econ_heat_SE":                _se(sub,  "d_heat_loss_pct_mean"),
                "econ_heat_pct_urban_%/day":     _mean(sub,"heat_loss_pct_U"),
                "econ_heat_pct_rural_%/day":     _mean(sub,"heat_loss_pct_R"),
                "econ_heat_usd_urban_sum":       _mean(sub,"econ_loss_usd_heat_U_sum"),
                "econ_heat_usd_rural_sum":       _mean(sub,"econ_loss_usd_heat_R_sum"),
                "d_econ_heat_usd_sum":           _mean(sub,"d_econ_loss_usd_heat_sum"),
                "econ_heat_usd_pct_chg_%":       _mean(sub,"econ_heat_usd_pct_chg"),

                "d_total_loss_pct_%/day":        _mean(sub,"d_total_loss_pct_mean"),
                "d_total_loss_SE":               _se(sub,  "d_total_loss_pct_mean"),
                "total_loss_pct_urban_%/day":    _mean(sub,"total_loss_pct_U"),
                "total_loss_pct_rural_%/day":    _mean(sub,"total_loss_pct_R"),
                "total_loss_usd_urban_sum":      _mean(sub,"total_loss_usd_U_sum"),
                "total_loss_usd_rural_sum":      _mean(sub,"total_loss_usd_R_sum"),
                "d_total_loss_usd_sum":          _mean(sub,"d_total_loss_usd_sum"),
                "econ_total_usd_pct_chg_%":      _mean(sub,"econ_total_usd_pct_chg"),
            }
            rows.append(row)

    df_full = pd.DataFrame(rows)

    disp_rows = []
    for _, row in df_full.iterrows():
        grp, per = row["Group"], row["Period"]

        def f(key, dec=3):
            v = row.get(key, np.nan)
            return f"{v:+.{dec}f}" if pd.notna(v) else "—"
        def fu(key, dec=3):
            v = row.get(key, np.nan)
            return f"{v:.{dec}f}" if pd.notna(v) else "—"

        disp_rows.append({
            "Group":   grp,
            "Period":  per,
            "n_pairs": int(row["n_pairs"]),

            "Tmax (urban/rural/Δ) °C":
                f"{fu('Tmax_urban_°C',1)}/{fu('Tmax_rural_°C',1)}/{f('dTmax_°C',2)}",
            "Tmin (urban/rural/Δ) °C":
                f"{fu('Tmin_urban_°C',1)}/{fu('Tmin_rural_°C',1)}/{f('dTmin_°C',2)}",

            "dCDH_total (°C·h/period)": f("dCDH_total_°Ch",1),
            "dCDH_total (°C·h/valid day)": f("dCDH_total_°Ch/day",2),
            "CDH urban/rural (°C·h)":
                f"{fu('CDH_urban_°Ch',1)}/{fu('CDH_rural_°Ch',1)}",
            "CDH pct_chg (%)":          f("dCDH_pct_chg_%",1),

            "dE_comm (W·h/period)":     f("dE_comm_Wh",1),
            "dE_comm (W·h/valid day)":  f("dE_comm_Wh/day",2),
            "dE_total (W·h/period)":    f("dE_total_Wh",1),
            "dE_total (W·h/valid day)": f("dE_total_Wh/day",2),

            "ΔLabour loss at Tx (%)": f("dLabourLoss_Tx_%",3),

            "Sleep urban (min/night)": fu("sleep_urban_min/night",3),
            "Sleep rural (min/night)": fu("sleep_rural_min/night",3),
            "ΔSleep (min/night) ± SE":
                f"{f('d_sleep_min/night',3)} ± {fu('d_sleep_SE',3)}",
            "ΔSleep period total (min/period)": f("d_sleep_season_min",1),

            "Δ Econ_sleep_pct_short (%/day) ± SE":
                f"{f('d_econ_sleep_pct_short_%/day',4)} ± {fu('d_econ_sleep_SE',4)}",
            "Δ Econ_sleep_pct_long (%/day)":
                f("d_econ_sleep_pct_long_%/day",4),
            "Econ_sleep_pct urban/rural (%/day)":
                f"{fu('econ_sleep_pct_urban_%/day',4)}/{fu('econ_sleep_pct_rural_%/day',4)}",
            "Econ_sleep USD urban/rural (period sum)":
                f"{fu('econ_sleep_usd_urban_sum',2)}/{fu('econ_sleep_usd_rural_sum',2)}",
            "Δ Econ_sleep USD (period sum)": f("d_econ_sleep_usd_sum",2),
            "Econ_sleep USD pct_chg (%)":    f("econ_sleep_usd_pct_chg_%",1),

            "Δ Econ_heat_pct (%/day) ± SE":
                f"{f('d_econ_heat_pct_%/day',4)} ± {fu('d_econ_heat_SE',4)}",
            "Econ_heat_pct urban/rural (%/day)":
                f"{fu('econ_heat_pct_urban_%/day',4)}/{fu('econ_heat_pct_rural_%/day',4)}",
            "Econ_heat USD urban/rural (period sum)":
                f"{fu('econ_heat_usd_urban_sum',2)}/{fu('econ_heat_usd_rural_sum',2)}",
            "Δ Econ_heat USD (period sum)": f("d_econ_heat_usd_sum",2),
            "Econ_heat USD pct_chg (%)":    f("econ_heat_usd_pct_chg_%",1),

            "Δ Total_loss_pct (%/day) ± SE":
                f"{f('d_total_loss_pct_%/day',4)} ± {fu('d_total_loss_SE',4)}",
            "Total_loss_pct urban/rural (%/day)":
                f"{fu('total_loss_pct_urban_%/day',4)}/{fu('total_loss_pct_rural_%/day',4)}",
            "Total_loss USD urban/rural (period sum)":
                f"{fu('total_loss_usd_urban_sum',2)}/{fu('total_loss_usd_rural_sum',2)}",
            "Δ Total_loss USD (period sum)": f("d_total_loss_usd_sum",2),
            "Total_loss USD pct_chg (%)":    f("econ_total_usd_pct_chg_%",1),

            "Lit: Temperature":       LIT_REFS["temp"],
            "Lit: CDH/Energy":        LIT_REFS["cdh"],
            "Lit: LC Dunne 2013":     LIT_REFS["lc_dunne"],
            "Lit: LC He 2022":        LIT_REFS["lc_he"],
            "Lit: Sleep (Minor 2022)":LIT_REFS["sleep"],
            "Lit: Econ_sleep":        LIT_REFS["econ_sleep"],
            "Lit: Econ_heat":         LIT_REFS["econ_heat"],
            "Lit: Econ_total":        LIT_REFS["econ_total"],
        })

    df_disp = pd.DataFrame(disp_rows)
    full_csv = os.path.join(out_dir, "summary_table_full.csv")
    disp_csv = os.path.join(out_dir, "summary_table_display.csv")
    df_full.to_csv(full_csv, index=False)
    df_disp.to_csv(disp_csv, index=False)
    print(f"  ✓ summary_table_full.csv    ({len(df_full)} rows × {len(df_full.columns)} cols)")
    print(f"  ✓ summary_table_display.csv ({len(df_disp)} rows)")

    return df_disp


# ══════════════════════════════════════════════════════════════════
# §8  结果提炼摘要功能 (NEW)
# ══════════════════════════════════════════════════════════════════

def write_result_highlights(df, out_dir):
    lines = [
        "# Key Results Highlight",
        f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "---",
        ""
    ]

    available_grps = [g for g in ["UHI", "UCI"] if g in df["group"].dropna().unique()]
    periods_order = ["HW", "JJA", "annual"]

    def gm(sub, col, dec=3, signed=True):
        if col not in sub.columns:
            return "N/A"
        v = sub[col].dropna()
        if len(v) == 0:
            return "N/A"
        m = float(v.mean())
        return f"{m:+.{dec}f}" if signed else f"{m:.{dec}f}"

    for grp in available_grps:
        for per in periods_order:
            sub = df[(df["group"] == grp) & (df["period_norm"] == per)]
            if len(sub) == 0:
                continue

            lines.append(f"## {grp} Group - {per} Period")
            lines.append(f"- **Valid Station Pairs**: {sub['pair_id'].nunique()}")
            lines.append(f"- **Temperature (dTmax / dTmin)**: {gm(sub, 'dTmax_mean')} / {gm(sub, 'dTmin_mean')} °C")
            lines.append(f"- **CDH Difference (Sum)**: {gm(sub, 'dCDH_total_sum')} °C·h")
            lines.append(f"- **CDH Difference (Per valid day)**: {gm(sub, 'dCDH_total_mean')} °C·h/day")
            lines.append(f"- **Energy Difference (Per valid day)**: {gm(sub, 'dE_total_mean')} W·h/day")
            lines.append(f"- **Energy Difference (Sum)**: {gm(sub, 'dE_total_sum')} W·h")
            lines.append(f"- **Sleep Loss Difference**: {gm(sub, 'd_sleep_loss_min')} min/night")
            lines.append(f"- **Labour Loss (Dunne)**: {gm(sub, 'd_labour_loss_tx')} %")
            lines.append(f"- **Sleep Productivity Loss Pct Diff**: {gm(sub, 'd_econ_loss_pct_short_mean', dec=4)} %/day")
            lines.append(f"- **Heat Labour Loss Pct Diff**: {gm(sub, 'd_heat_loss_pct_mean', dec=4)} %/day")
            lines.append(f"- **Total Economic Loss Pct Diff**: {gm(sub, 'd_total_loss_pct_mean', dec=4)} %/day")
            lines.append(f"- **Total Economic Loss USD Diff**: {gm(sub, 'd_total_loss_usd_sum', dec=3)} USD/period")
            lines.append("")

    highlight_path = os.path.join(out_dir, "result_highlight.md")
    with open(highlight_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {os.path.basename(highlight_path)} generated successfully!")


# ══════════════════════════════════════════════════════════════════
# §9  箱线图
# ══════════════════════════════════════════════════════════════════

def _boxplot_one(df, col, ylabel, title, out_path,
                 periods=None, allowed_periods=None):
    if periods is None:
        periods = PERIODS_PLOT
    if allowed_periods is not None:
        periods = [p for p in periods if p in allowed_periods]
    if not periods:
        print(f"    [skip] {col}: no allowed periods"); return
    if col not in df.columns or df[col].isna().all():
        print(f"    [skip] {col}: no data"); return

    groups = ["UHI","UCI"]
    width, sep = 0.30, 0.10
    offsets = {g: (-sep/2 if g=="UHI" else sep/2) for g in groups}
    tick_x  = np.arange(len(periods), dtype=float)

    fig, ax = plt.subplots(figsize=(3.5*len(periods)+1, 5.5))
    for grp in groups:
        clr = GROUP_CLR[grp]
        for pi, per in enumerate(periods):
            sub = df[(df["period_norm"]==per)&(df["group"]==grp)][col].dropna()
            if len(sub) < 3: continue
            xp = tick_x[pi] + offsets[grp]
            ax.boxplot(sub.values, positions=[xp], widths=width,
                       patch_artist=True, showfliers=True, notch=False,
                       flierprops=dict(marker=".",ms=3,alpha=0.3,mfc=clr,mec=clr),
                       medianprops=dict(color="white",lw=2.2),
                       boxprops=dict(facecolor=clr,alpha=0.72,lw=1.1,edgecolor="k"),
                       whiskerprops=dict(color="k",lw=0.9,ls="--"),
                       capprops=dict(color="k",lw=1.0),
                       manage_ticks=False)
            ax.scatter(xp, sub.mean(), zorder=6, s=30, color="white",
                       edgecolors=clr, linewidths=1.5)
            _, pv = stats.ttest_1samp(sub.values, 0) if len(sub) > 2 else (0, 1.)
            st = sig_stars(pv)
            if st:
                ax.text(xp, float(np.percentile(sub.values, 97)), st,
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=clr)

    ax.axhline(0, color="k", lw=1.0, ls=":", alpha=0.5)
    for i, per in enumerate(periods):
        ax.axvspan(i-0.48, i+0.48, color=PERIOD_CLR.get(per,"lightgray"),
                   alpha=0.07, zorder=0)
    ax.set_xticks(tick_x)

    xlabels = []
    for per in periods:
        base_lbl = PERIOD_LABEL.get(per, per)
        if "n_days" in df.columns:
            nd_vals = df[df["period_norm"]==per]["n_days"].dropna()
            nd_med  = int(nd_vals.median()) if len(nd_vals) > 0 else "?"
            xlabels.append(f"{base_lbl}\n(med {nd_med} d)")
        else:
            xlabels.append(base_lbl)
    ax.set_xticklabels(xlabels, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.55, len(periods)-0.45)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"Urban − Rural  ·  {title}\n"
                 f"(UHI/UCI × Period,  n = pairs;  each point = 1 station pair)",
                 fontsize=11, fontweight="bold", pad=8)

    leg = [mpatches.Patch(fc=GROUP_CLR["UHI"],alpha=0.72,ec="k",label="UHI"),
           mpatches.Patch(fc=GROUP_CLR["UCI"],alpha=0.72,ec="k",label="UCI"),
           Line2D([0],[0],marker="o",ls="none",ms=7,mfc="white",
                  mec="gray",mew=1.5,label="Mean")]
    ax.legend(handles=leg, fontsize=9.5, loc="upper right", framealpha=0.88)

    n_lines = []
    for per in periods:
        ns = [f"{g}:n={int(df[(df['period_norm']==per)&(df['group']==g)][col].dropna().count())}"
              for g in groups]
        n_lines.append(f"{PERIOD_LABEL.get(per,per).replace(chr(10),' ')}: {', '.join(ns)}")
    ax.text(0.01, 0.03, "\n".join(n_lines), transform=ax.transAxes, fontsize=7.5,
            va="bottom", bbox=dict(boxstyle="round,pad=0.35",fc="wheat",alpha=0.5))

    ax.grid(axis="y", alpha=0.28, ls="--")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {os.path.basename(out_path)}")


def run_all_boxplots(df, bp_dir):
    print("\n[Boxplots] Generating difference boxplots (pair-level)...")
    ensure_dir(bp_dir)
    for item in BOXPLOT_CFG:
        col, ylabel, title = item[0], item[1], item[2]
        allowed = item[3] if len(item) >= 4 else None
        _boxplot_one(df, col, ylabel, title,
                     os.path.join(bp_dir, f"boxplot_{col}.png"),
                     allowed_periods=allowed)

def run_key_result_boxplots(df, bp_dir):
    """
    新增模块：
    额外输出 3 张主要结果箱线图（labour/economic/energy），不影响原有全部箱线图输出。
    
    输出目录：
        boxplots/key_results/
            ├── key_boxplot_d_labour_loss_tx.png
            ├── key_boxplot_d_total_loss_pct_mean.png
            └── key_boxplot_dCDH_total_sum.png
    """
    key_dir = os.path.join(bp_dir, "key_results")
    ensure_dir(key_dir)

    print("\n[Key Result Boxplots] Generating labour/economic/energy boxplots...")

    for col, ylabel, title, allowed in KEY_RESULT_BOXPLOT_CFG:
        if col not in df.columns:
            print(f"    [skip] {col}: column not found")
            continue
        if df[col].isna().all():
            print(f"    [skip] {col}: all NaN")
            continue

        out_png = os.path.join(key_dir, f"key_boxplot_{col}.png")
        _boxplot_one(
            df=df,
            col=col,
            ylabel=ylabel,
            title=title,
            out_path=out_png,
            allowed_periods=allowed
        )

    # 可选：再额外输出一个 1×3 拼图版，方便直接放 PPT
    print("\n[Key Result Boxplots] Generating combined 1x3 panel figure...")
    _plot_key_results_combined(df, os.path.join(key_dir, "key_results_combined.png"))

def _plot_key_results_combined(df, out_path):
    """Plot the three retained main outcomes: labour, economy and energy."""
    cfg = [
        ("d_labour_loss_tx", r"$\Delta$Labour loss (%)", "Labour loss"),
        ("d_total_loss_pct_mean", r"$\Delta$Total economic loss (%/day)", "Economic loss"),
        ("dCDH_total_sum", r"$\Sigma\Delta$CDH (°C·h/period)", "Cooling demand"),
    ]
    groups = ["UHI", "UCI"]
    periods = PERIODS_PLOT
    width, sep = 0.30, 0.10
    offsets = {g: (-sep / 2 if g == "UHI" else sep / 2) for g in groups}
    tick_x = np.arange(len(periods), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), squeeze=False)
    panel_letters = ["a", "b", "c"]

    for ax, (col, ylabel, title), letter in zip(axes.ravel(), cfg, panel_letters):
        if col not in df.columns or df[col].isna().all():
            ax.set_axis_off()
            continue
        for grp in groups:
            clr = GROUP_CLR[grp]
            for pi, per in enumerate(periods):
                vals = df.loc[(df["period_norm"] == per) & (df["group"] == grp), col].dropna()
                if len(vals) < 3:
                    continue
                xp = tick_x[pi] + offsets[grp]
                ax.boxplot(
                    vals.values, positions=[xp], widths=width, patch_artist=True,
                    showfliers=True, manage_ticks=False,
                    flierprops=dict(marker=".", ms=3, alpha=0.3, mfc=clr, mec=clr),
                    medianprops=dict(color="white", lw=2.0),
                    boxprops=dict(facecolor=clr, alpha=0.72, lw=1.0, edgecolor="k"),
                    whiskerprops=dict(color="k", lw=0.9, ls="--"),
                    capprops=dict(color="k", lw=1.0),
                )
                ax.scatter(xp, vals.mean(), zorder=6, s=28, color="white",
                           edgecolors=clr, linewidths=1.4)
        ax.axhline(0, color="k", lw=1.0, ls=":", alpha=0.5)
        ax.set_xticks(tick_x)
        ax.set_xticklabels([PERIOD_LABEL.get(p, p).replace("\n", " ") for p in periods])
        ax.set_xlim(-0.55, len(periods) - 0.45)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, ls="--")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", ha="left", va="bottom")

    fig.legend(handles=[
        mpatches.Patch(fc=GROUP_CLR["UHI"], alpha=0.72, ec="k", label="UHI"),
        mpatches.Patch(fc=GROUP_CLR["UCI"], alpha=0.72, ec="k", label="UCI"),
        Line2D([0], [0], marker="o", ls="none", ms=7, mfc="white",
               mec="gray", mew=1.4, label="Mean"),
    ], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {os.path.basename(out_path)}")


def run_abs_boxplots(df, bp_dir):
    abs_dir = os.path.join(bp_dir, "absolute_values")
    ensure_dir(abs_dir)
    print("\n[Abs Boxplots] Urban vs Rural absolute value boxplots...")

    ABS_CFG = [
        ("sleep_loss_min_U","sleep_loss_min_R",
         r"Sleep loss (min/night)", "Sleep Loss: Urban vs Rural"),
        ("hne_d_U","hne_d_R",
         r"HNE (°C·h/night)",      "Night Heat Exposure: Urban vs Rural"),
        ("Tmax_u_mean","Tmax_r_mean",
         r"$T_x$ (°C)",             "Daily Tmax: Urban vs Rural"),
        ("Tmin_u_mean","Tmin_r_mean",
         r"$T_n$ (°C)",             "Daily Tmin: Urban vs Rural"),
        ("CDH_total_u_mean","CDH_total_r_mean",
         r"CDH daily mean (°C·h)", "Daily CDH: Urban vs Rural"),
        ("econ_loss_pct_short_U","econ_loss_pct_short_R",
         r"Econ$_{sleep,short}$ (%/day)",
         "Sleep-Loss Productivity Loss: Urban vs Rural — Gibson 2018"),
        ("econ_loss_pct_long_U","econ_loss_pct_long_R",
         r"Econ$_{sleep,long}$ (%/day)",
         "Sleep-Loss Productivity Loss (Long): Urban vs Rural"),
        ("heat_loss_pct_U","heat_loss_pct_R",
         r"Econ$_{heat}$ (%/day)",
         "Heat Labour Loss: Urban vs Rural — Graff Zivin 2014"),
        ("total_loss_pct_U","total_loss_pct_R",
         r"Total Econ Loss (%/day)",
         "Combined Economic Loss: Urban vs Rural"),
        ("econ_loss_usd_sleep_U","econ_loss_usd_sleep_R",
         r"Econ$_{sleep}$ USD/day/worker",
         "Sleep-Loss USD: Urban vs Rural"),
        ("econ_loss_usd_heat_U","econ_loss_usd_heat_R",
         r"Econ$_{heat}$ USD/day/worker",
         "Heat Labour USD: Urban vs Rural"),
        ("total_loss_usd_U","total_loss_usd_R",
         r"Total Econ Loss USD/day/worker",
         "Total Economic Loss USD: Urban vs Rural"),
    ]

    for col_u, col_r, ylabel, title in ABS_CFG:
        if col_u not in df.columns or col_r not in df.columns: continue
        if df[col_u].isna().all() and df[col_r].isna().all(): continue

        groups  = ["UHI","UCI"]
        periods = PERIODS_PLOT
        n_per   = len(periods)
        width   = 0.25
        gap_grp = 0.08
        gap_ur  = 0.02
        tick_x  = np.arange(n_per, dtype=float)
        offsets = {
            ("UHI","u"): -gap_grp/2 - width - gap_ur/2,
            ("UHI","r"): -gap_grp/2 + gap_ur/2,
            ("UCI","u"):  gap_grp/2 + gap_ur/2,
            ("UCI","r"):  gap_grp/2 + width + gap_ur/2,
        }

        fig, ax = plt.subplots(figsize=(3.8*n_per+1, 5.5))
        for grp in groups:
            grp_clr = GROUP_CLR[grp]
            for side, col, clr in [
                ("u", col_u, grp_clr),
                ("r", col_r, "#aaaaaa"),
            ]:
                for pi, per in enumerate(periods):
                    sub = df[(df["period_norm"]==per)&(df["group"]==grp)][col].dropna()
                    if len(sub) < 3: continue
                    xp = tick_x[pi] + offsets[(grp, side)]
                    fc = grp_clr if side=="u" else "white"
                    ax.boxplot(
                        sub.values, positions=[xp], widths=width*0.85,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=grp_clr if side=="r" else "white", lw=2.0),
                        boxprops=dict(facecolor=fc, alpha=0.75 if side=="u" else 0.3,
                                      lw=1.1, edgecolor=grp_clr),
                        whiskerprops=dict(color=grp_clr, lw=0.9, ls="--"),
                        capprops=dict(color=grp_clr, lw=1.0),
                        manage_ticks=False)
                    ax.scatter(xp, sub.mean(), zorder=6, s=22,
                               color=grp_clr if side=="u" else "white",
                               edgecolors=grp_clr, linewidths=1.2)

        ax.set_xticks(tick_x)
        ax.set_xticklabels([PERIOD_LABEL.get(p, p) for p in periods],
                           fontsize=11, fontweight="bold")
        ax.set_xlim(-0.6, n_per-0.4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}\nSolid=Urban / Hollow=Rural  |  Red=UHI / Blue=UCI",
                     fontsize=11, fontweight="bold", pad=8)
        leg = [
            mpatches.Patch(fc=COLOR_UHI, alpha=0.75, ec="k", label="UHI Urban"),
            mpatches.Patch(fc="white", alpha=0.3, ec=COLOR_UHI, label="UHI Rural"),
            mpatches.Patch(fc=COLOR_UCI, alpha=0.75, ec="k", label="UCI Urban"),
            mpatches.Patch(fc="white", alpha=0.3, ec=COLOR_UCI, label="UCI Rural"),
        ]
        ax.legend(handles=leg, fontsize=9, loc="upper right", framealpha=0.88)
        ax.grid(axis="y", alpha=0.25, ls="--")
        ax.spines[["top","right"]].set_visible(False)
        safe = col_u.replace("/","_").replace(" ","_")
        fig.tight_layout()
        fig.savefig(os.path.join(abs_dir, f"abs_{safe}.png"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"    ✓ abs_{safe}.png")

    pct_cols = [(c,
                 f"{c.replace('_pct_chg','').replace('_',' ')} (%)\n(urban/rural−1)×100",
                 c.replace("_pct_chg","").replace("_"," "))
                for c in df.columns
                if c.endswith("_pct_chg") and not df[c].isna().all()]
    for col, ylabel, title in pct_cols:
        _boxplot_one(df, col, ylabel, f"% Change: {title}",
                     os.path.join(abs_dir, f"pct_{col}.png"))


# ══════════════════════════════════════════════════════════════════
# §10  日循环廓线
# ══════════════════════════════════════════════════════════════════

def _plot_diurnal(df_dc, val_col, ylabel, title, out_png,
                  group_col="group", period_col="period_norm",
                  hour_col="hour", shade_night=False):
    if df_dc is None or val_col not in df_dc.columns: return
    groups  = sorted(df_dc[group_col].dropna().unique()) if group_col in df_dc.columns else ["all"]
    periods = [p for p in PERIODS_PLOT
               if p in (df_dc[period_col].unique() if period_col in df_dc.columns else [])]
    if not periods:
        periods = list(df_dc[period_col].dropna().unique()
                       if period_col in df_dc.columns else ["all"])

    fig, axes = plt.subplots(1, max(len(groups),1), figsize=(5.5*max(len(groups),1), 4.8),
                             sharey=True, squeeze=False)
    for ai, grp in enumerate(groups):
        ax = axes[0, ai]
        sg = df_dc[df_dc[group_col]==grp] if group_col in df_dc.columns else df_dc
        for per in periods:
            sp = sg[sg[period_col]==per] if period_col in df_dc.columns else sg
            sp = sp.sort_values(hour_col)
            if len(sp) == 0: continue
            x  = sp[hour_col].values; y = sp[val_col].values
            se = sp[val_col+"_se"].values if (val_col+"_se") in sp.columns else np.zeros_like(y)
            clr = PERIOD_CLR.get(per,"gray"); ls = PERIOD_LS.get(per,"-")
            ax.plot(x, y, color=clr, lw=2.0, ls=ls,
                    label=PERIOD_LABEL.get(per, per).replace("\n"," "))
            ax.fill_between(x, y-1.96*se, y+1.96*se, color=clr, alpha=0.12)
        ax.axvspan(min(DAY_HOURS), max(DAY_HOURS)+1, color="gold", alpha=0.08, zorder=0)
        if shade_night:
            ax.axvspan(21, 24, color="navy", alpha=0.06, zorder=0)
            ax.axvspan(0, 7,  color="navy", alpha=0.06, zorder=0)
        ax.axhline(0, color="k", lw=0.9, ls=":", alpha=0.4)
        ax.set_title(f"{grp} Group", fontsize=11, fontweight="bold")
        ax.set_xlabel("Local Solar Time (h)", fontsize=10)
        if ai == 0: ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0,24,3)],
                           rotation=30, ha="right")
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
        ax.grid(alpha=0.22, ls="--"); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

def _plot_diurnal_urban_abs(df_dc, val_col, ylabel, title, out_png,
                            group_col="group", period_col="period_norm",
                            hour_col="hour", shade_night=False):
    """
    新增：
    仅绘制 Urban absolute diurnal line，
    按 group(UHI/UCI) 分面，按 period(annual/JJA/HW) 画线。
    """
    if df_dc is None or val_col not in df_dc.columns:
        return

    groups = ["UHI", "UCI"]
    periods = [p for p in PERIODS_PLOT
               if p in (df_dc[period_col].unique() if period_col in df_dc.columns else [])]
    if not periods:
        return

    fig, axes = plt.subplots(
        1, len(groups), figsize=(5.5 * len(groups), 4.8),
        sharey=True, squeeze=False
    )
    axes = axes[0]

    for ai, grp in enumerate(groups):
        ax = axes[ai]
        sg = df_dc[df_dc[group_col] == grp] if group_col in df_dc.columns else df_dc

        for per in periods:
            sp = sg[sg[period_col] == per] if period_col in df_dc.columns else sg
            sp = sp.sort_values(hour_col)
            if len(sp) == 0:
                continue

            x = sp[hour_col].values
            y = sp[val_col].values
            se = sp[val_col + "_se"].values if (val_col + "_se") in sp.columns else np.zeros_like(y)

            clr = PERIOD_CLR.get(per, "gray")
            ls = PERIOD_LS.get(per, "-")

            ax.plot(
                x, y, color=clr, lw=2.2, ls=ls,
                label=PERIOD_LABEL.get(per, per).replace("\n", " ")
            )
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=clr, alpha=0.12)

        ax.axvspan(min(DAY_HOURS), max(DAY_HOURS) + 1, color="gold", alpha=0.08, zorder=0)
        if shade_night:
            ax.axvspan(21, 24, color="navy", alpha=0.06, zorder=0)
            ax.axvspan(0, 7,  color="navy", alpha=0.06, zorder=0)

        ax.set_title(f"{grp} Group", fontsize=11, fontweight="bold")
        ax.set_xlabel("Local Solar Time (h)", fontsize=10)
        if ai == 0:
            ax.set_ylabel(ylabel, fontsize=10)

        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels(
            [f"{h:02d}:00" for h in range(0, 24, 3)],
            rotation=30, ha="right"
        )
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
        ax.grid(alpha=0.22, ls="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _agg_hourly(df, gc, vc, hc="hour"):
    rows = []
    for keys, sub in df.groupby(gc+[hc], observed=True):
        kd = dict(zip(gc+[hc], keys if isinstance(keys, tuple) else [keys]))
        for c in vc:
            if c in sub.columns:
                kd[c]       = sub[c].mean()
                kd[c+"_se"] = sub[c].sem()
        rows.append(kd)
    return pd.DataFrame(rows)

def build_cdh_diurnal(df_h, diur_dir):
    if df_h is None: return
    gc = [c for c in ["group","density","period_norm"] if c in df_h.columns]
    for px, lb, yl in [("CDH_h","CDH",r"CDH$_h$ (°C·h)"),
                       ("HDH_h","HDH",r"HDH$_h$ (°C·h)"),
                       ("E_comm_h","E_comm",r"E$_{comm,h}$ (W·h)"),
                       ("E_resi_h","E_resi",r"E$_{resi,h}$ (W·h)")]:
        dc = "d"+px; vc = [c for c in [px+"_u",px+"_r",dc] if c in df_h.columns]
        if not vc: continue
        ddc = _agg_hourly(df_h, gc, vc)
        ddc.to_csv(os.path.join(diur_dir, f"diurnal_{lb}.csv"), index=False)
        _plot_diurnal(ddc, dc, yl, f"Δ{lb} Diurnal",
                     os.path.join(diur_dir, f"diurnal_{lb}.png"))

        # 新增：仅绘制 Urban absolute（日循环），annual/JJA/HW × UHI/UCI
        if px == "E_comm_h" and (px + "_u") in ddc.columns:
            _plot_diurnal_urban_abs(
                ddc,
                px + "_u",
                r"Urban E$_{comm,h}$ (W·h)",
                "Commercial Energy Diurnal — Urban Absolute",
                os.path.join(diur_dir, "diurnal_E_comm_urban.png")
            )
        if px == "E_resi_h" and (px + "_u") in ddc.columns:
            _plot_diurnal_urban_abs(
                ddc,
                px + "_u",
                r"Urban E$_{resi,h}$ (W·h)",
                "Residential Energy Diurnal — Urban Absolute",
                os.path.join(diur_dir, "diurnal_E_resi_urban.png")
            )

        print(f"    ✓ diurnal_{lb}")


def build_lc_diurnal(df_lc, diur_dir):
    """Write hourly urban-rural Dunne labour-loss profiles."""
    if df_lc is None:
        return
    gc = [c for c in ["group", "period_norm"] if c in df_lc.columns]
    avail = [f"dloss_dunne_h{h:02d}" for h in range(24) if f"dloss_dunne_h{h:02d}" in df_lc.columns]
    if not avail:
        return
    rows = []
    for keys, sub in df_lc.groupby(gc, observed=True):
        kd = dict(zip(gc, keys if isinstance(keys, tuple) else [keys]))
        mat = sub[avail].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        for hi, ch in enumerate(avail):
            r = kd.copy(); r["hour"] = hi
            cv = mat[:, hi]; vv = cv[np.isfinite(cv)]
            r["dloss_dunne"] = float(np.nanmean(cv))
            r["dloss_dunne_se"] = float(np.nanstd(cv, ddof=1) / max(len(vv) ** 0.5, 1)) if len(vv) else np.nan
            rows.append(r)
    ddc = pd.DataFrame(rows)
    ddc.to_csv(os.path.join(diur_dir, "diurnal_labour_loss_dunne.csv"), index=False)
    _plot_diurnal(
        ddc, "dloss_dunne", r"$\Delta$Labour loss (%)",
        "Urban-rural Dunne labour-loss diurnal profile",
        os.path.join(diur_dir, "diurnal_labour_loss_dunne.png"),
    )
    print("    ✓ diurnal_labour_loss_dunne")


def build_sleep_diurnal(df_pair, diur_dir):
    """
    Do not generate an hourly sleep-loss curve.

    The Minor et al. model maps one nightly Tmin exposure to one nightly
    sleep-duration change. A nightly outcome must not be copied across
    individual nighttime hours or interpreted as an hourly response.

    Sleep results should instead be represented using nightly or
    pair-period summary values, such as boxplots.
    """
    print(
        "    - diurnal_sleep_loss skipped: "
        "the Minor et al. model is based on nightly Tmin and does not "
        "define an hourly sleep-loss response."
    )
    return

# ══════════════════════════════════════════════════════════════════
# §11  建筑运营文件
# ══════════════════════════════════════════════════════════════════

def build_operation_files(df_pair, density, ops_dir):
    sub_a = df_pair[(df_pair["density"]==density)&(df_pair["period_norm"]=="annual")]
    sub_h = df_pair[(df_pair["density"]==density)&(df_pair["period_norm"]=="HW")]
    sub_j = df_pair[(df_pair["density"]==density)&(df_pair["period_norm"]=="JJA")]
    sm = lambda c, s: round(float(s[c].mean()), 4) if c in s.columns and len(s)>0 else np.nan
    ua_c = UA.get(("COMM",density), UA[("COMM","OTHER")])
    ua_r = UA.get(("RESI",density), UA[("RESI","OTHER")])

    est = {
        "est_dCDH_comm_HW":         sm("dCDH_comm_sum",    sub_h),
        "est_dE_comm_HW":           sm("dE_comm_sum",       sub_h),
        "est_dE_resi_annual":       sm("dE_resi_sum",       sub_a),
        "est_d_sleep_JJA":          sm("d_sleep_loss_min",  sub_j),
        "est_d_econ_sleep_pct_HW":  sm("d_econ_loss_pct_short_mean", sub_h),
        "est_d_econ_heat_pct_HW":   sm("d_heat_loss_pct_mean",       sub_h),
        "est_d_total_loss_pct_HW":  sm("d_total_loss_pct_mean",      sub_h),
        "est_econ_sleep_usd_JJA":   sm("econ_loss_usd_sleep_U_sum",  sub_j),
        "est_econ_heat_usd_HW":     sm("econ_loss_usd_heat_U_sum",   sub_h),
    }

    def _comm():
        rows = []
        for h in range(24):
            if   8 <= h < 18: occ=1.00;hvc=1.00;hvh=0.00;lt=1.00;eq=1.00
            elif 6 <= h < 8:  occ=0.30;hvc=0.80;hvh=0.60;lt=0.50;eq=0.30
            elif 18<= h < 21: occ=0.20;hvc=0.60;hvh=0.40;lt=0.30;eq=0.20
            else:              occ=0.05;hvc=0.10;hvh=0.15;lt=0.05;eq=0.05
            rows.append({"hour":h,"occupancy":occ,"hvac_cool":hvc,
                         "hvac_heat":hvh,"lighting":lt,"equipment":eq})
        df = pd.DataFrame(rows)
        for k, v in {"type":"Commercial","density":density,"UA_W_per_K":ua_c,
                     "T_cool_C":T_COOL,"T_heat_C":T_HEAT,
                     "COP_A0":COP_A0,"COP_A1":COP_A1,"COP_A2":COP_A2,
                     "schedule_src":"ASHRAE 90.1-2022 App.G",**est}.items():
            df[k] = v
        return df

    def _resi():
        rows = []
        for h in range(24):
            if   18<= h < 24: occ=0.90;hvc=0.80;hvh=0.70;lt=0.75;eq=0.60
            elif  0<= h < 6:  occ=0.95;hvc=0.70;hvh=0.50;lt=0.10;eq=0.10
            elif  6<= h < 9:  occ=0.80;hvc=0.50;hvh=0.60;lt=0.60;eq=0.50
            else:              occ=0.35;hvc=0.30;hvh=0.30;lt=0.20;eq=0.25
            rows.append({"hour":h,"occupancy":occ,"hvac_cool":hvc,
                         "hvac_heat":hvh,"lighting":lt,"equipment":eq})
        df = pd.DataFrame(rows)
        for k, v in {"type":"Residential","density":density,"UA_W_per_K":ua_r,
                     "T_cool_C":T_COOL,"T_heat_C":T_HEAT,
                     "COP_A0":COP_A0,"COP_A1":COP_A1,"COP_A2":COP_A2,
                     "schedule_src":"ASHRAE 62.2-2022; DOE Prototype",
                     "note":"0-6h HVAC→sleep: Minor 2022; He 2022; Graff Zivin 2014",
                     **est}.items():
            df[k] = v
        return df

    _comm().to_csv(os.path.join(ops_dir, f"operation_commercial_{density}.csv"), index=False)
    _resi().to_csv(os.path.join(ops_dir, f"operation_residential_{density}.csv"), index=False)
    print(f"    ✓ operation_{density}")

def get_plot_results_fig23_exact_pair_ids(df_pair):
    """
    Return EXACTLY the station-pair IDs used by ``plot_results_fig23.py``.

    The reference cohort is defined only by the same rules as Fig23:
      1. ``hw_method == percentile`` when that column is available;
      2. the same ``pair_id`` has both NHW and HW records;
      3. ``dTmean``, ``dAmp1``, ``dTx`` and ``dTn`` are finite in both periods;
      4. the canonical group is UHI or UCI.

    IMPORTANT
    ---------
    Labour, sleep, hourly temperature, CDH and LCZ completeness are audited
    below, but they are NOT used to remove any additional pair. Therefore the
    returned ID set is identical to the Fig23 reference cohort.

    If the merged integrated table does not contain HW and NHW rows for every
    Fig23 pair, the function raises an error rather than silently producing a
    smaller cohort.
    """
    periods = ["HW", "NHW"]
    reference_ids = load_plot_results_fig23_matched_pair_ids()

    if df_pair is None or len(df_pair) == 0:
        raise ValueError(
            "Cannot apply the exact Fig23 plotting cohort: df_pair is empty."
        )

    required_base = {"pair_id", "period_norm"}
    missing_base = sorted(required_base - set(df_pair.columns))
    if missing_base:
        raise ValueError(
            "Cannot apply the exact Fig23 plotting cohort; missing columns: "
            f"{missing_base}"
        )

    pair_sub = df_pair.loc[
        df_pair["pair_id"].astype(str).isin(reference_ids)
        & df_pair["period_norm"].isin(periods)
    ].copy()
    pair_sub["pair_id"] = pair_sub["pair_id"].astype(str)

    # Verify that the integrated table actually contains BOTH requested periods
    # for every reference pair. Do not silently intersect away missing IDs.
    pair_period_present = (
        pair_sub.assign(_present=True)
        .groupby(["pair_id", "period_norm"], observed=True)["_present"]
        .any()
        .unstack("period_norm", fill_value=False)
        .reindex(columns=periods, fill_value=False)
    )
    integrated_both_ids = set(
        pair_period_present.index[
            pair_period_present[periods].all(axis=1)
        ].astype(str)
    )
    missing_from_integrated = reference_ids - integrated_both_ids
    if missing_from_integrated:
        preview = sorted(missing_from_integrated)[:20]
        raise ValueError(
            "The integrated pair-period table cannot reproduce the exact "
            "plot_results_fig23 cohort because some reference pairs do not "
            "have both HW and NHW rows after merging. "
            f"Missing pairs={len(missing_from_integrated)}; first IDs={preview}. "
            "Check the upstream CDH pair-period table and merge keys instead "
            "of dropping these stations downstream."
        )

    # Audit downstream-variable coverage WITHOUT using it as another filter.
    audit_metrics = [
        "d_labour_loss_tx",
        "d_sleep_loss_min",
        *[f"temp_h{h:02d}_U" for h in range(24)],
        *[f"temp_h{h:02d}_R" for h in range(24)],
    ]
    present_metrics = [c for c in audit_metrics if c in pair_sub.columns]
    missing_metric_columns = [c for c in audit_metrics if c not in pair_sub.columns]

    if present_metrics:
        audit = pair_sub[["pair_id", "period_norm"] + present_metrics].copy()
        for col in present_metrics:
            audit[col] = pd.to_numeric(audit[col], errors="coerce")
        audit["_all_available_metrics_finite"] = np.isfinite(
            audit[present_metrics].to_numpy(dtype=float)
        ).all(axis=1)
        complete_by_period = (
            audit.groupby(["pair_id", "period_norm"], observed=True)
            ["_all_available_metrics_finite"]
            .any()
            .unstack("period_norm", fill_value=False)
            .reindex(columns=periods, fill_value=False)
        )
        n_all_metrics_complete = int(
            complete_by_period[periods].all(axis=1).sum()
        )
    else:
        n_all_metrics_complete = 0

    lcz_1_6_ids = set()
    if "lcz_type" in pair_sub.columns:
        valid_lcz = {f"LCZ {i}" for i in range(1, 7)}
        lcz_1_6_ids = set(
            pair_sub.loc[
                pair_sub["lcz_type"].isin(valid_lcz), "pair_id"
            ].astype(str).unique()
        )

    group_counts = {}
    if "group" in pair_sub.columns:
        cohort_group = (
            pair_sub[["pair_id", "group"]]
            .dropna()
            .drop_duplicates("pair_id")
        )
        cohort_group["group"] = (
            cohort_group["group"].astype(str).str.upper().str.strip()
        )
        group_counts = cohort_group["group"].value_counts().to_dict()

    print("\n  [Exact plot_results_fig23 plotting cohort]")
    print(f"    Fig23 reference pair IDs            : {len(reference_ids)}")
    print(f"    retained pair IDs in integrated figs: {len(reference_ids)}")
    selected_ids = set(reference_ids)
    print(f"    exact ID equality                   : {selected_ids == reference_ids}")
    print(f"    group counts                        : {group_counts}")
    print(
        "    all downstream variables complete   : "
        f"{n_all_metrics_complete}  (audit only; NOT a filter)"
    )
    print(
        "    valid LCZ 1-6 metadata              : "
        f"{len(lcz_1_6_ids)}  (LCZ panel subset only; NOT a cohort filter)"
    )
    if missing_metric_columns:
        print(
            "    downstream metric columns absent    : "
            f"{missing_metric_columns}  (NOT used to remove pairs)"
        )

    return selected_ids

def build_pair_hourly_cdh_from_pair_table(df_pair):
    """
    Reconstruct pair-level hourly urban-rural cooling degree-hours from the
    same fixed Fig23 pair-period cohort used by labour, sleep and LCZ panels.

    For each pair, period and hour:
      CDH_U(h) = max(T_U(h) - T_COOL, 0)
      CDH_R(h) = max(T_R(h) - T_COOL, 0)
      dCDH_h   = CDH_U(h) - CDH_R(h)

    This avoids treating ``hourly_profiles.csv`` as a pair-level table. That
    file is already aggregated and therefore has no ``pair_id`` column.
    """
    base_required = {"pair_id", "period_norm", "group"}
    temp_required = [
        *[f"temp_h{h:02d}_U" for h in range(24)],
        *[f"temp_h{h:02d}_R" for h in range(24)],
    ]
    missing = sorted(
        (base_required | set(temp_required)) - set(df_pair.columns)
    )
    if missing:
        raise ValueError(
            "Cannot reconstruct pair-level hourly CDH; missing columns: "
            f"{missing}"
        )

    base = df_pair.copy()
    base["pair_id"] = base["pair_id"].astype(str)
    base["group"] = base["group"].astype(str).str.upper().str.strip()

    frames = []
    for h in range(24):
        u_col = f"temp_h{h:02d}_U"
        r_col = f"temp_h{h:02d}_R"
        temp_u = pd.to_numeric(base[u_col], errors="coerce")
        temp_r = pd.to_numeric(base[r_col], errors="coerce")

        frame = base[["pair_id", "period_norm", "group"]].copy()
        frame["hour"] = h
        frame["CDH_h_U"] = np.maximum(temp_u - T_COOL, 0.0)
        frame["CDH_h_R"] = np.maximum(temp_r - T_COOL, 0.0)
        frame["dCDH_h"] = frame["CDH_h_U"] - frame["CDH_h_R"]
        frames.append(frame)

    hourly = pd.concat(frames, ignore_index=True)
    hourly = hourly[
        hourly["period_norm"].isin(["HW", "NHW"])
        & hourly["group"].isin(["UHI", "UCI"])
        & np.isfinite(pd.to_numeric(hourly["dCDH_h"], errors="coerce"))
    ].copy()

    expected_rows = df_pair["pair_id"].nunique() * 2 * 24
    print("\n  [Pair-level hourly CDH reconstructed]")
    print(f"    threshold                           : {T_COOL:.1f} °C")
    print(f"    rows generated                      : {len(hourly)}")
    print(f"    expected rows (pairs x 2 x 24)      : {expected_rows}")

    return hourly

def plot_diurnal_hw_nhw_comparison(df_pair, df_cdh_h, df_lc, out_path):
    """
    HW/NHW integrated comparison, 2 rows x 3 columns:
      - column 1: hourly Delta-CDH curves (the only hourly impact metric);
      - column 2: pair-level labour-loss boxplots;
      - column 3: pair-level sleep-loss boxplots.

    Labour and sleep use exactly the same merged pair-period source columns as
    the standalone boxplots:
      - d_labour_loss_tx;
      - d_sleep_loss_min.

    ``df_lc`` is retained in the signature only for backward compatibility and
    is intentionally not used to reconstruct an hourly labour profile.
    """
    colors = {
        ("UHI", "HW"):  "#D95F02",
        ("UHI", "NHW"): "#FDB462",
        ("UCI", "HW"):  "#A50F15",
        ("UCI", "NHW"): "#FB9A99",
    }
    groups = ["UHI", "UCI"]
    periods = ["HW", "NHW"]

    def get_cdh_data():
        if df_cdh_h is None:
            return None
        required = {"group", "period_norm", "hour", "dCDH_h"}
        if not required.issubset(df_cdh_h.columns):
            return None
        tmp = df_cdh_h[df_cdh_h["period_norm"].isin(periods)].copy()
        tmp["hour"] = pd.to_numeric(tmp["hour"], errors="coerce")
        tmp["dCDH_h"] = pd.to_numeric(tmp["dCDH_h"], errors="coerce")
        tmp = tmp[np.isfinite(tmp["hour"]) & np.isfinite(tmp["dCDH_h"])].copy()
        return (
            tmp.groupby(["group", "period_norm", "hour"], observed=True)
            ["dCDH_h"]
            .mean()
            .reset_index()
        )

    data_cdh = get_cdh_data()
    boxplot_metrics = [
        (
            "d_labour_loss_tx",
            "Labour loss",
            r"$\Delta$Labour loss (%)",
        ),
        (
            "d_sleep_loss_min",
            "Sleep loss",
            r"$\Delta$Sleep loss (min/night)",
        ),
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12.2, 7.0),
        sharex=False,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(
        w_pad=0.04,
        h_pad=0.04,
        hspace=0.05,
        wspace=0.05,
    )

    def style_panel(ax):
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(0.9)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
            top=False,
            right=False,
            direction="out",
            width=0.9,
            length=3.5,
        )
        ax.grid(True, axis="y", ls="--", lw=0.35, alpha=0.22)

    def draw_pair_boxplot(ax, grp_name, col_name):
        """Use the same pair-period values and box definitions as _boxplot_one."""
        n_by_period = {}
        for pi, per in enumerate(periods):
            if col_name not in df_pair.columns:
                n_by_period[per] = 0
                continue

            mask = (
                (df_pair["group"] == grp_name)
                & (df_pair["period_norm"] == per)
            )
            sub = pd.to_numeric(
                df_pair.loc[mask, col_name], errors="coerce"
            )
            sub = sub[np.isfinite(sub)].dropna()
            n_by_period[per] = int(len(sub))

            if len(sub) < 3:
                continue

            clr = colors[(grp_name, per)]
            ax.boxplot(
                sub.values,
                positions=[pi],
                widths=0.56,
                patch_artist=True,
                showfliers=True,
                notch=False,
                flierprops=dict(
                    marker=".", ms=3, alpha=0.3, mfc=clr, mec=clr,
                ),
                medianprops=dict(color="white", lw=2.2),
                boxprops=dict(
                    facecolor=clr, alpha=0.72, lw=1.1, edgecolor="k",
                ),
                whiskerprops=dict(color="k", lw=0.9, ls="--"),
                capprops=dict(color="k", lw=1.0),
                manage_ticks=False,
            )
            ax.scatter(
                pi,
                sub.mean(),
                zorder=6,
                s=30,
                color="white",
                edgecolors=clr,
                linewidths=1.5,
            )

            _, p_value = stats.ttest_1samp(sub.values, 0)
            stars = sig_stars(p_value)
            if stars:
                ax.text(
                    pi,
                    float(np.percentile(sub.values, 97)),
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=clr,
                )

        ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.5)
        ax.set_xlim(-0.55, len(periods) - 0.45)
        ax.set_xticks(np.arange(len(periods), dtype=float))
        ax.set_xticklabels(
            [f"{per}\n($n$={n_by_period.get(per, 0)})" for per in periods],
            fontsize=10,
        )

    for row_grp_idx, grp_name in enumerate(groups):
        # Column 1: hourly Delta-CDH only.
        ax_cdh = axes[row_grp_idx, 0]
        if data_cdh is not None and not data_cdh.empty:
            for per in periods:
                sub = data_cdh[
                    (data_cdh["group"] == grp_name)
                    & (data_cdh["period_norm"] == per)
                ].sort_values("hour")
                if sub.empty:
                    continue
                ax_cdh.plot(
                    sub["hour"],
                    sub["dCDH_h"],
                    color=colors[(grp_name, per)],
                    ls="-" if per == "HW" else "--",
                    lw=2.2 if per == "HW" else 1.6,
                    alpha=1.0 if per == "HW" else 0.7,
                    label=per,
                )

        ax_cdh.axvspan(9, 17, color="gold", alpha=0.05, zorder=0)
        ax_cdh.axhline(0, color="k", lw=0.8, ls=":", alpha=0.5)
        ax_cdh.set_xlim(0, 23)
        ax_cdh.set_xticks([0, 6, 12, 18, 23])
        ax_cdh.set_ylabel(
            r"$\Delta$CDH ($^\circ$C$\cdot$h)",
            fontsize=11,
            fontweight="bold",
        )
        if row_grp_idx == 1:
            ax_cdh.set_xlabel("Local hour (h)", fontsize=11, fontweight="bold")
        else:
            ax_cdh.set_title(
                "Cooling degree-hours",
                fontsize=12,
                fontweight="bold",
                pad=6,
            )
        ax_cdh.text(
            -0.42,
            0.5,
            grp_name,
            transform=ax_cdh.transAxes,
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
        )
        style_panel(ax_cdh)

        # Columns 2--3: pair-level boxplots from the merged boxplot dataframe.
        for metric_offset, (col_name, panel_title, ylabel) in enumerate(
                boxplot_metrics, start=1):
            ax = axes[row_grp_idx, metric_offset]
            draw_pair_boxplot(ax, grp_name, col_name)
            ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
            if row_grp_idx == 1:
                ax.set_xlabel("Period", fontsize=11, fontweight="bold")
            else:
                ax.set_title(
                    panel_title,
                    fontsize=12,
                    fontweight="bold",
                    pad=6,
                )
            style_panel(ax)

            if row_grp_idx == 0 and metric_offset == 2:
                legend_handles = [
                    mpatches.Patch(
                        facecolor=colors[(grp_name, "HW")],
                        edgecolor="k",
                        alpha=0.72,
                        label="HW",
                    ),
                    mpatches.Patch(
                        facecolor=colors[(grp_name, "NHW")],
                        edgecolor="k",
                        alpha=0.72,
                        label="NHW",
                    ),
                    Line2D(
                        [0], [0],
                        marker="o",
                        linestyle="none",
                        markersize=6,
                        markerfacecolor="white",
                        markeredgecolor="gray",
                        markeredgewidth=1.3,
                        label="Mean",
                    ),
                ]
                ax.legend(
                    handles=legend_handles,
                    loc="upper right",
                    fontsize=9,
                    frameon=True,
                    framealpha=0.8,
                )

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f" [SUCCESS] Plot saved to {out_path}")


from scipy.ndimage import gaussian_filter1d
import seaborn as sns

def plot_lcz1_6_period_comparison(df_pair, df_lc, out_path):
    """
    LCZ 1--6 HW/NHW hourly comparison, reduced from 2 x 4 to 2 x 2.

    Retained without changing their calculations:
      - urban-rural air-temperature difference (Delta-Ta);
      - cooling degree-hours difference (Delta-CDH).

    Labour-loss and sleep-loss subplots are removed because those outcomes
    are not interpreted as 24-hour diurnal curves here. ``df_lc`` remains in
    the signature only to keep the existing main-call interface unchanged.
    """
    lcz_styles = {
        "LCZ 1": {"color": "#800000", "lw": 2.2, "ls": "-"},
        "LCZ 2": {"color": "#d73027", "lw": 2.0, "ls": "-"},
        "LCZ 3": {"color": "#f46d43", "lw": 1.6, "ls": "-"},
        "LCZ 4": {"color": "#a63603", "lw": 2.0, "ls": "--"},
        "LCZ 5": {"color": "#e6550d", "lw": 1.6, "ls": "--"},
        "LCZ 6": {"color": "#fdae61", "lw": 1.3, "ls": "--"},
    }
    periods = ["NHW", "HW"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.6, 6.3),
        sharex=True,
        sharey="col",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(
        w_pad=0.025,
        h_pad=0.025,
        hspace=0.025,
        wspace=0.025,
    )

    hours = np.arange(24)

    def smooth(y, sigma=0.8):
        y = np.asarray(y, dtype=float)
        y_pad = np.tile(y, 3)
        y_smooth = gaussian_filter1d(y_pad, sigma=sigma)
        return y_smooth[24:48]

    val_range_tracker = {0: [], 1: []}

    for row_idx, per in enumerate(periods):
        sub_df = df_pair[df_pair["period_norm"] == per]
        available_lczs = [
            f"LCZ {i}"
            for i in range(1, 7)
            if f"LCZ {i}" in sub_df["lcz_type"].unique()
        ]

        for lcz in available_lczs:
            style = lcz_styles[lcz]
            p_lcz = sub_df[sub_df["lcz_type"] == lcz]
            n_count = p_lcz["pair_id"].nunique()

            ta_raw = []
            cdh_raw = []
            for h in range(24):
                u_col = f"temp_h{h:02d}_U"
                r_col = f"temp_h{h:02d}_R"
                t_u = pd.to_numeric(p_lcz[u_col], errors="coerce").mean()
                t_r = pd.to_numeric(p_lcz[r_col], errors="coerce").mean()
                ta_raw.append(t_u - t_r)
                cdh_raw.append(
                    np.maximum(t_u - 26.0, 0.0)
                    - np.maximum(t_r - 26.0, 0.0)
                )

            ta_s = smooth(ta_raw)
            cdh_s = smooth(cdh_raw)
            label = f"{lcz} (n={n_count})"

            axes[row_idx, 0].plot(hours, ta_s, label=label, **style)
            axes[row_idx, 1].plot(hours, cdh_s, **style)
            val_range_tracker[0].extend(ta_s[np.isfinite(ta_s)])
            val_range_tracker[1].extend(cdh_s[np.isfinite(cdh_s)])

    col_titles = [
        r"$\Delta T_a$ ($^\circ$C)",
        r"$\Delta$CDH ($^\circ$C$\cdot$h)",
    ]

    for col in range(2):
        vals = val_range_tracker[col]
        if vals:
            v_min = float(np.min(vals))
            v_max = float(np.max(vals))
            pad = (v_max - v_min) * 0.15 if v_max != v_min else 1.0
            axes[0, col].set_ylim(min(-0.1, v_min - pad), v_max + pad)

        axes[0, col].set_title(
            col_titles[col],
            fontsize=13,
            fontweight="bold",
            pad=8,
        )

        for row in range(2):
            ax = axes[row, col]
            ax.set_xlim(0, 23)
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.grid(True, ls=":", alpha=0.3)
            ax.axhline(0, color="k", lw=0.6, alpha=0.3)
            ax.axvspan(6, 18, color="gray", alpha=0.04)
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color("black")
                ax.spines[side].set_linewidth(0.9)
            ax.tick_params(top=False, right=False, direction="out")

            if row == 1:
                ax.set_xlabel("Local Solar Time (h)", fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    "NON-HEATWAVE" if row == 0 else "HEATWAVE",
                    fontsize=12,
                    fontweight="bold",
                    color="navy",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f" [SUCCESS] LCZ 2x2 diurnal matrix saved to {out_path}")


def plot_lcz1_6_boxplot_comparison(df_master, out_path):
    """
    2x5 箱线图矩阵：对比 NHW 和 HW
    列：dTmax, dTmin, dCDH, Labour, Sleep
    """
    # 1. 数据准备
    valid_periods = ['NHW', 'HW']
    target_lczs = [f"LCZ {i}" for i in range(1, 7)]
    
    # 筛选数据
    df_plot = df_master[
        (df_master['period_norm'].isin(valid_periods)) & 
        (df_master['lcz_type'].isin(target_lczs))
    ].copy()

    # 2. 视觉与指标配置
    lcz_palette = {
        'LCZ 1': '#800000', 'LCZ 2': '#d73027', 'LCZ 3': '#f46d43',
        'LCZ 4': '#a63603', 'LCZ 5': '#e6550d', 'LCZ 6': '#fdae61'
    }
    
    # 定义 5 个指标及其列名
    metrics_map = [
        ('dTmax_mean',       r"$\Delta T_x$ ($^\circ$C)"),
        ('dTmin_mean',       r"$\Delta T_n$ ($^\circ$C)"),
        ('dCDH_total_mean',  r"$\Delta$CDH ($^\circ$C$\cdot$h/d)"),
        ('d_labour_loss_tx', r"$\Delta$Labour loss (%)"),
        ('d_sleep_loss_min', r"$\Delta$Sleep Loss (min/d)")
    ]

    # 3. 创建画布 (2行5列)
    fig, axes = plt.subplots(2, 5, figsize=(17, 8), sharex=True, sharey='col', constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03)

    # 4. 动态统一纵轴范围
    for col_idx, (col_name, _) in enumerate(metrics_map):
        if col_name in df_plot.columns:
            all_vals = df_plot[col_name].dropna()
            if not all_vals.empty:
                v_min, v_max = all_vals.min(), all_vals.max()
                # 增加 15% 缓冲
                padding = (v_max - v_min) * 0.15 if v_max != v_min else 1.0
                
                # 针对物理量指标，确保 0 线在视野内
                if col_name in ['dTmax_mean', 'dTmin_mean', 'dCDH_total_mean', 'd_sleep_loss_min']:
                    actual_min = min(-0.1, v_min - padding)
                else:
                    actual_min = v_min - padding
                
                axes[0, col_idx].set_ylim(actual_min, v_max + padding)

    # 5. 循环绘图
    for row_idx, per in enumerate(valid_periods):
        period_data = df_plot[df_plot['period_norm'] == per]
        
        for col_idx, (col_name, ylabel) in enumerate(metrics_map):
            ax = axes[row_idx, col_idx]
            
            if col_name not in period_data.columns or period_data[col_name].isna().all():
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                continue

            # 绘制箱线图
            sns.boxplot(
                data=period_data, x='lcz_type', y=col_name, order=target_lczs,
                palette=lcz_palette, ax=ax, fliersize=2, 
                linewidth=1.2, width=0.7, showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"5"}
            )
            
            # 辅助设计
            ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.3)
            ax.grid(axis='y', ls=':', alpha=0.5)
            
            # 标题与标签设置
            if row_idx == 0:
                ax.set_title(ylabel, fontsize=13, fontweight='bold', pad=10)
            
            if col_idx == 0:
                row_label = "NON-HEATWAVE" if per == 'NHW' else "HEATWAVE"
                ax.set_ylabel(row_label, fontsize=12, fontweight='bold', color='navy')
            else:
                ax.set_ylabel("")
            
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelsize=10, rotation=45)

            # 标注样本数 n
            for i, lcz in enumerate(target_lczs):
                n = period_data[period_data['lcz_type'] == lcz][col_name].count()
                if n > 0:
                    ax.text(i, ax.get_ylim()[1], f"n={n}", ha='center', va='bottom', fontsize=8, color='gray')

    # 6. 保存
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f" [SUCCESS] LCZ 2x5 Boxplot matrix saved to {out_path}")

def validate_sleep_model_reference(model_func):
    """
    Validate numerical equivalence with the frozen Minor spline.
    """
    temperatures = np.array(
        [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, np.nan],
        dtype=float,
    )

    expected = np.array(
        [
            0.0,
            0.0,
            -1.07,
            -2.14,
            -3.21,
            -9.39,
            -15.57,
            np.nan,
        ],
        dtype=float,
    )

    calculated = np.asarray(
        model_func(temperatures),
        dtype=float,
    )

    np.testing.assert_allclose(
        calculated[:-1],
        expected[:-1],
        rtol=0.0,
        atol=1e-12,
    )

    if not np.isnan(calculated[-1]):
        raise AssertionError(
            "NaN input must remain NaN."
        )

    print("Sleep model validation passed.")
    print(
        pd.DataFrame({
            "Tmin_C": temperatures,
            "calculated_sleep_change_min": calculated,
            "expected_sleep_change_min": expected,
        }).to_string(index=False)
    )
# ══════════════════════════════════════════════════════════════════
# §12  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0   = time.time()
    diag = DiagnosticLog()

    bp_dir   = os.path.join(OUTPUT_DIR, "boxplots")
    diur_dir = os.path.join(OUTPUT_DIR, "diurnal")
    ops_dir  = os.path.join(OUTPUT_DIR, "building_operations")
    comp_dir = os.path.join(OUTPUT_DIR, "hw_nhw_comparison") 
    for d in [OUTPUT_DIR, bp_dir, diur_dir, ops_dir, comp_dir]:
        ensure_dir(d)

    print("═"*72)
    print("  UHI/UCI Integrated Impact Analysis  v4_updated")
    print("  [FIX-A] load_hne_paired: hne_diff 补全三渠道经济损失列 + 置信度 + KG")
    print("  [FIX-B] pair_period_agg: econ_pct→mean / econ_usd→sum+mean")
    print("  [FIX-C] BOXPLOT_CFG + build_summary_table: 补经济损失区块")
    print("  [NEW]   自动生成结果大纲 (result_highlight.md)")
    print("═"*72)

    print("\n[1/10] Loading upstream module outputs...")
    df_cdh    = load_cdh_panel(diag)
    df_cdh_h  = load_cdh_hourly()
    df_lc     = load_labour_full(diag)
    df_hne    = load_hne_paired(diag)
    meta      = load_pair_meta()

    print("\n[2/10] CDH daily panel → pair×period aggregation...")
    df_cdh_agg = pair_period_agg(df_cdh, diag)

    print("\n[3/10] Merging all modules (pair×period).")
    df = merge_all_modules(
        df_cdh_agg,
        df_lc,
        df_hne,
        meta,
        diag,
    )

    # Labour-specific matched HW/NHW requirement.
    # Only d_labour_loss_tx is masked for incomplete pairs.
    # Other outcomes and Annual/JJA values remain unchanged.
    df = enforce_labour_hw_nhw_complete_pairs(
        df,
        value_col="d_labour_loss_tx",
        pair_col="pair_id",
        period_col="period_norm",
        diag=diag,
    )

    print(
        "\n[4/10] Computing percentage changes "
        "(urban/rural − 1) × 100 ."
    )
    df = compute_pct_changes(df)

    print("\n[5/10] Difference statistics summary (pair-level)...")
    df_sum = build_diff_summary(df, diag)
    df_sum.to_csv(os.path.join(OUTPUT_DIR, "diff_summary_v4.csv"), index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "pair_period_panel.csv"),   index=False)

    print("\n[6/10] Building comprehensive summary table (with econ loss)...")
    build_summary_table(df, OUTPUT_DIR)
    
    print("\n[7/10] Generating Key Results Highlight (result_highlight.md)...")
    write_result_highlights(df, OUTPUT_DIR)

    print("\n[8/11] Difference boxplots (pair-level, incl. econ loss)...")
    run_all_boxplots(df, bp_dir)

    print("\n[9/11] Main-result boxplots (labour / economy / energy)...")
    run_key_result_boxplots(df, bp_dir)

    print("\n[10/11] Absolute value + pct_change boxplots (incl. econ loss)...")
    run_abs_boxplots(df, bp_dir)

    print("\n[11/11] Generating Diurnal & Building operation files...")
    build_cdh_diurnal(df_cdh_h, diur_dir)
    build_lc_diurnal(df_lc,     diur_dir)
    build_sleep_diurnal(df,     diur_dir)

    print("\n[EXTRA] Preparing exact plot_results_fig23 cohort...")

    # Attach LCZ metadata before applying the fixed Fig23 pair-ID list.
    # LCZ availability is not used to reduce the master cohort.
    LCZ_SOURCE_PATH = (
        UNIFIED_ROOT + "/analysis/main_multiyear/"
        "robustness_percentile/all_pair_period_metrics.csv"
    )

    if os.path.exists(LCZ_SOURCE_PATH):
        lcz_source = pd.read_csv(LCZ_SOURCE_PATH)
        required_lcz_cols = {"pair_id", "urban_lcz_corrected"}
        missing_lcz_cols = required_lcz_cols - set(lcz_source.columns)
        if missing_lcz_cols:
            raise ValueError(
                "LCZ source is missing required columns: "
                f"{sorted(missing_lcz_cols)}"
            )

        # Keep the original LCZ metadata rule: take one static LCZ value per pair.
        # The percentile + matched HW/NHW restriction is applied separately by
        # load_plot_results_fig23_matched_pair_ids().
        df_lcz_meta = lcz_source[
            ["pair_id", "urban_lcz_corrected"]
        ].drop_duplicates(subset=["pair_id"]).copy()
        df_lcz_meta["pair_id"] = df_lcz_meta["pair_id"].astype(str)

        remap_lcz = {
            51: 1, 52: 2, 53: 3, 54: 4, 55: 5,
            56: 6, 57: 7, 58: 8, 59: 9, 60: 10,
        }

        def format_lcz(x):
            if pd.isna(x):
                return "Unknown"
            try:
                val = int(float(x))
            except (TypeError, ValueError):
                return "Unknown"
            val = remap_lcz.get(val, val)
            return f"LCZ {val}"

        df_lcz_meta["lcz_type"] = (
            df_lcz_meta["urban_lcz_corrected"].apply(format_lcz)
        )

        df["pair_id"] = df["pair_id"].astype(str)
        if "lcz_type" in df.columns:
            df = df.drop(columns=["lcz_type"])
        df = df.merge(
            df_lcz_meta[["pair_id", "lcz_type"]],
            on="pair_id",
            how="left",
        )

        if df_lc is not None:
            df_lc["pair_id"] = df_lc["pair_id"].astype(str)
            if "lcz_type" in df_lc.columns:
                df_lc = df_lc.drop(columns=["lcz_type"])
            df_lc = df_lc.merge(
                df_lcz_meta[["pair_id", "lcz_type"]],
                on="pair_id",
                how="left",
            )

        print(
            " [INFO] Successfully loaded LCZ metadata from the canonical "
            "percentile source. LCZ distribution:\n"
            f"{df['lcz_type'].value_counts(dropna=False)}"
        )
    else:
        raise FileNotFoundError(f"LCZ source file not found: {LCZ_SOURCE_PATH}")

    # Use EXACTLY the Fig23 percentile + matched HW/NHW cohort for the two
    # requested figures. Do not intersect it with labour, sleep, hourly
    # temperature or LCZ completeness. Earlier tables and plots remain
    # unchanged. Hourly CDH is reconstructed from available pair-level
    # temperatures within this fixed Fig23 ID set.
    fig23_exact_ids = get_plot_results_fig23_exact_pair_ids(df)
    valid_periods = ["HW", "NHW"]

    df_hw_nhw_plot = df[
        df["pair_id"].astype(str).isin(fig23_exact_ids)
        & df["period_norm"].isin(valid_periods)
    ].copy()
    df_hw_nhw_plot["pair_id"] = df_hw_nhw_plot["pair_id"].astype(str)

    # Hard assertion: each target period must contain exactly the Fig23 IDs.
    # Never silently shrink the plotting cohort.
    for per in valid_periods:
        period_ids = set(
            df_hw_nhw_plot.loc[
                df_hw_nhw_plot["period_norm"].eq(per), "pair_id"
            ].dropna().astype(str).unique()
        )
        if period_ids != fig23_exact_ids:
            missing_ids = sorted(fig23_exact_ids - period_ids)
            extra_ids = sorted(period_ids - fig23_exact_ids)
            raise ValueError(
                f"{per} rows are not identical to the Fig23 cohort: "
                f"missing={len(missing_ids)} {missing_ids[:20]}; "
                f"extra={len(extra_ids)} {extra_ids[:20]}"
            )

    print(
        f"  [ASSERT PASS] NHW and HW each contain exactly "
        f"{len(fig23_exact_ids)} Fig23 pair IDs."
    )

    df_cdh_h_hw_nhw_plot = build_pair_hourly_cdh_from_pair_table(
        df_hw_nhw_plot
    )

    df_lc_hw_nhw_plot = None

    if df_lc is not None:
        df_lc_hw_nhw_plot = df_lc[
            df_lc["pair_id"].astype(str).isin(
                fig23_exact_ids
            )
            & df_lc["period_norm"].isin(
                valid_periods
            )
        ].copy()

        df_lc_hw_nhw_plot = (
            enforce_labour_hw_nhw_complete_pairs(
                df_lc_hw_nhw_plot,
                value_col="d_labour_loss_tx",
                pair_col="pair_id",
                period_col="period_norm",
                diag=None,
            )
        )

    print("\n[EXTRA] Generating HW vs NHW Integrated Comparison...")
    comp_out = os.path.join(comp_dir, "integrated_diurnal_comparison.png")
    plot_diurnal_hw_nhw_comparison(
        df_hw_nhw_plot,
        df_cdh_h_hw_nhw_plot,
        df_lc_hw_nhw_plot,
        comp_out,
    )

    print("\n[EXTRA] Generating LCZ 1-6 Delta-Ta / Delta-CDH Comparison...")
    lcz_comparison_out = os.path.join(
        comp_dir,
        "lcz_1_6_hw_nhw_2x4_comparison.png",
    )
    plot_lcz1_6_period_comparison(
        df_hw_nhw_plot,
        df_lc_hw_nhw_plot,
        lcz_comparison_out,
    )

    # Keep the pre-existing LCZ pair-level 2x5 boxplot output unchanged.
    lcz_boxplot_out = os.path.join(
        comp_dir,
        "lcz_1_6_hw_nhw_2x4_boxplot.png",
    )
    plot_lcz1_6_boxplot_comparison(df, lcz_boxplot_out)

    for density in ["HRHD","LRLD"]:
        build_operation_files(df, density, ops_dir)
    with open(os.path.join(ops_dir, "README.md"), "w") as f:
        f.write(
            "# Building Operation Files\n\n"
            "Commercial schedule: ASHRAE 90.1-2022 App.G\n"
            "Residential schedule: ASHRAE 62.2-2022; DOE Prototype\n"
            "UA: Kolokotroni et al. 2012\n"
            "COP: Sailor & Vasireddy 2006\n"
            "Sleep-night HVAC: Minor 2022; He 2022\n"
            "Econ sleep-loss: Gibson & Shrader 2018; Costa-Font et al. 2024\n"
            "Econ direct heat: Graff Zivin & Neidell 2014; Seppanen et al. 2006\n"
            "USD conversion: Deschenes 2014; Kjellstrom et al. 2016\n"
        )

    diag_path = os.path.join(OUTPUT_DIR, "DIAGNOSTIC_REPORT.md")
    diag.write(diag_path)

    print(f"\n{'═'*72}")
    print(f"  ✓ Complete  elapsed {time.time()-t0:.1f} s")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── DIAGNOSTIC_REPORT.md")
    print(f"  ├── result_highlight.md            ← [NEW] Key results summary")
    print(f"  ├── pair_period_panel.csv          ← pair×period panel")
    print(f"  ├── diff_summary_v4.csv            ← all diff + abs + pct_chg stats")
    print(f"  ├── summary_table_full.csv         ← numeric summary (incl. econ loss)")
    print(f"  ├── summary_table_display.csv      ← formatted + lit refs (incl. econ)")
    print(f"  ├── boxplots/                      {len(BOXPLOT_CFG)} diff boxplots")
    print(f"  │   ├── key_results/               3 main-result boxplots + 1 combined panel")
    print(f"  ├── boxplots/absolute_values/      urban vs rural + pct_chg (incl. econ)")
    print(f"  ├── diurnal/                       diurnal profiles")
    print(f"  └── building_operations/           4 operation files (incl. econ est.)")
    print("═"*72)


if __name__ == "__main__":
    main()
