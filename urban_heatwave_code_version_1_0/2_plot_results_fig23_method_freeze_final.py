#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results_fig23.py
===============
读取 analysis_multiyear.py 生成的 all_pair_period_metrics.csv，
输出三张图及统计表格。

图1: 散点图 ΔAmp vs ΔT_mn (颜色=ΔT_x)
图2: UHI/UCI 三面板日循环（UHI曲线、UCI曲线、ΔT_a对比）
图3: LCZ 分类三面板（Compact / Open / 强度对比）
表1: 统计汇总表（含显著性星号）→ CSV + TXT
图6: 全球站点地图 + UHI/UCI 代表性日循环 inset
图7: [新增] 全球站点地图，红=曾出现热浪，蓝=未出现热浪（不区分城市农村）
图8: [新增] 热浪 vs 非热浪时期日循环对比（不区分城市农村），含 std 色带

Method-freeze final patch:
- Canonical UHI/UCI is always the upstream annual classification.
- NHW/HW sign-state transitions are explicitly labelled as period-specific
  dTx diagnostics and are not used to redefine canonical groups.
"""

# ====================== 1. 标准库导入 ======================
import os
import warnings
import math
# ====================== 2. 科学计算库 ======================
import numpy as np
import pandas as pd
from scipy import stats

# ====================== 3. Matplotlib 绘图库 ======================
import matplotlib
matplotlib.use("Agg")  # 无界面绘图模式
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as inset_axes

from config import (
    FIG23_OUTPUT_DIR,
    LEGACY_FIGURES_OUTPUT_DIR,
    UNIFIED_ROOT,
)

# ====================== 4. 地图可视化库 (Cartopy) ======================
HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except ImportError:
    HAS_CARTOPY = False
    print("警告：未安装 cartopy，地图绘图将跳过")

# ─────────────────────────────────────────────────────────────
# Config  ── 只需修改这两行与分析脚本一致
# ─────────────────────────────────────────────────────────────
# ── 改为 ──
INPUT_DIR = (
    UNIFIED_ROOT
    + "/analysis/main_multiyear/robustness_percentile"
)
OUTPUT_DIR = FIG23_OUTPUT_DIR

# Legacy dry-bulb exposure-proxy figure is not part of the formal impact workflow.
RUN_LEGACY_EXPOSURE_PROXY_FIGURE = False

# Period-specific sign states are diagnostics only. They describe the sign of
# dTx within NHW or HW and are not the canonical manuscript UHI/UCI group.
# The canonical group is loaded from the annual two-harmonic dTx classification:
# annual dTx >= 0 -> UHI; annual dTx < 0 -> UCI.
PERIOD_SPECIFIC_DTX_STATE_DEFINITION = (
    "diagnostic sign of period-specific dTx; positive if dTx>0, "
    "negative if dTx<0, zero/missing unclassified; not canonical annual UHI/UCI"
)
CANONICAL_GROUP_DEFINITION = (
    "annual two-harmonic dTx: UHI if dTx>=0, UCI if dTx<0"
)

# 字体全局设置
FONT_LABEL  = 16   # 轴标签
FONT_TICK   = 14   # 刻度
FONT_TITLE  = 17   # 面板标题
FONT_LEGEND = 14   # 图例
FONT_ANNOT  = 13   # 注释文字

# 颜色
COLOR_URBAN   = "#d62728"   # 红
COLOR_RURAL   = "#2ca02c"   # 绿
COLOR_UHI     = "#d62728"
COLOR_UCI     = "#1f77b4"
COLOR_COMPACT = "#d62728"
COLOR_OPEN    = "#1f77b4"
ALPHA_BAND    = 0.18

HOURS = np.arange(24)

plt.rcParams.update({
    "font.size":       FONT_TICK,
    "axes.labelsize":  FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "axes.titlesize":  FONT_TITLE,
    "legend.fontsize": FONT_LEGEND,
    "figure.dpi": 600,
    "savefig.dpi": 600,

    # 打开四周黑框
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,

    # 黑框样式
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
})


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def stars_from_p(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def display_period_label(period: str) -> str:
    mapping = {
        "annual": "Annual",
        "warm_season": "Warm season",
        "JJA": "Warm season",
        "non_heatwave": "NHW",
        "heatwave": "HW",
        "NHW": "NHW",
        "HW": "HW",
    }
    return mapping.get(str(period), str(period))


def get_diurnal_arrays(df_sub):
    """
    从包含 urban_diurnal_h00…h23 / rural_diurnal_h00…h23 的 DataFrame
    返回 (u_arr, r_arr)，shape = (n_pairs, 24)。
    """
    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]
    u_arr = df_sub[u_cols].values.astype(float)
    r_arr = df_sub[r_cols].values.astype(float)
    return u_arr, r_arr


def mean_std(arr):
    """Nanmean and nanstd across axis=0."""
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def add_stats_text(ax, n, r2=None, slope=None, p=None, loc="upper left"):
    """在坐标轴角落添加统计信息文本框。"""
    lines = [f"$n={n}$"]
    if slope  is not None: lines.append(f"slope$={slope:.3f}$")
    if r2     is not None: lines.append(f"$R^2={r2:.2f}$")
    if p      is not None:
        pstr = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        lines.append(f"$p={pstr}${stars_from_p(p)}")
    txt = "\n".join(lines)
    xpos = 0.03 if "left" in loc  else 0.97
    ypos = 0.97 if "upper" in loc else 0.06
    ha   = "left" if "left" in loc else "right"
    va   = "top"  if "upper" in loc else "bottom"
    ax.text(xpos, ypos, txt, transform=ax.transAxes,
            ha=ha, va=va, fontsize=FONT_ANNOT,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))

def add_black_frame(ax, lw=1.2):
    """给绘图区域加黑框（四周 spines），不是整张图片外框。"""
    for side in ["top", "right", "bottom", "left"]:
        if side in ax.spines:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(lw)


# ─────────────────────────────────────────────────────────────
# Figure 1: Scatter  ΔAmp vs ΔT_mean, color = ΔT_x
# ─────────────────────────────────────────────────────────────
def plot_figure1_scatter(annual_df, output_dir):
    """
    x: dTmean (ΔT_mean)  | y: dAmp1 (ΔAmp)  |  color: dTx (ΔT_x)
    """
    # 1. 将 "dTn" 修改为 "dTmean"
    df = annual_df[["dTmean", "dAmp1", "dTx"]].dropna().copy()
    if len(df) == 0:
        print("  [Fig1] No data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    vmin, vmax = -2, 4
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r

    # 2. X轴数据替换为 df["dTmean"]
    sc = ax.scatter(df["dTmean"], df["dAmp1"],
                    c=df["dTx"], cmap=cmap, norm=norm,
                    s=55, alpha=0.82, edgecolors="none", zorder=3)

    ax.axhline(0, color="gray", linestyle="--", lw=1.2, zorder=2)
    
    # 3. 标签更新为更加直观的 \Delta T_{mean}
    ax.set_xlabel(r"$\Delta T_{mean}$ (°C)", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$\Delta Amp$ (°C)",    fontsize=FONT_LABEL)
    ax.grid(True, lw=0.4, alpha=0.35, zorder=1)
    
    add_black_frame(ax)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.92)
    cbar.set_label(r"$\Delta T_x$ (°C)", fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    # 4. 线性回归计算替换为 dTmean
    lr = stats.linregress(df["dTmean"], df["dAmp1"])
    add_stats_text(ax, n=len(df), r2=lr.rvalue**2,
                   slope=lr.slope, p=lr.pvalue, loc="upper right")

    fig.tight_layout()
    # 5. 更新保存的文件名以防混淆
    fpath = os.path.join(output_dir, "plots", "Figure1_scatter_dAmp_dTmean_dTx.png")
    ensure_dir(os.path.dirname(fpath))
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────
# Figure 2: UHI vs UCI — three vertical panels
#   (a) UHI diurnal curves
#   (b) UCI diurnal curves
#   (c) ΔT_a = Urban-Rural across 24 h for both groups
# ─────────────────────────────────────────────────────────────
def plot_figure2_uhi_uci(annual_df, output_dir):
    uhi = annual_df[annual_df["group"] == "UHI"]
    uci = annual_df[annual_df["group"] == "UCI"]

    uhi_u, uhi_r = get_diurnal_arrays(uhi)
    uci_u, uci_r = get_diurnal_arrays(uci)

    uhi_u_m, uhi_u_s = mean_std(uhi_u)
    uhi_r_m, uhi_r_s = mean_std(uhi_r)
    uci_u_m, uci_u_s = mean_std(uci_u)
    uci_r_m, uci_r_s = mean_std(uci_r)

    uhi_delta = uhi_u_m - uhi_r_m
    uci_delta = uci_u_m - uci_r_m

    fig = plt.figure(figsize=(6.5, 13))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.42)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    def _diurnal_panel(ax, u_mean, u_std, r_mean, r_std,
                       n_u, n_r, label, tick_y=True):
        ax.plot(HOURS, u_mean, color=COLOR_URBAN, lw=2.2, label="Urban")
        ax.fill_between(HOURS, u_mean - u_std, u_mean + u_std,
                        color=COLOR_URBAN, alpha=ALPHA_BAND)
        ax.plot(HOURS, r_mean, color=COLOR_RURAL, lw=2.2, label="Rural")
        ax.fill_between(HOURS, r_mean - r_std, r_mean + r_std,
                        color=COLOR_RURAL, alpha=ALPHA_BAND)

        # 城高于乡 → 红色填充；乡高于城 → 蓝色填充
        ax.fill_between(HOURS, u_mean, r_mean,
                        where=(u_mean >= r_mean),
                        alpha=0.22, color=COLOR_URBAN, interpolate=True)
        ax.fill_between(HOURS, u_mean, r_mean,
                        where=(u_mean < r_mean),
                        alpha=0.22, color=COLOR_UCI, interpolate=True)

        ax.set_title(f"{label}  ($n_U={n_u},\\ n_R={n_r}$)",
                     loc="left", fontsize=FONT_TITLE, fontweight="bold")
        ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 4))
        ax.set_xlabel("Local Solar Time (h)", fontsize=FONT_LABEL)
        ax.set_ylabel("$T_a$ (°C)",            fontsize=FONT_LABEL)
        ax.legend(frameon=False, loc="upper left", fontsize=FONT_LEGEND)
        ax.grid(True, lw=0.4, alpha=0.35)
        add_black_frame(ax)

    _diurnal_panel(ax_a,
                   uhi_u_m, uhi_u_s, uhi_r_m, uhi_r_s,
                   n_u=len(uhi), n_r=len(uhi), label="(a)")

    _diurnal_panel(ax_b,
                   uci_u_m, uci_u_s, uci_r_m, uci_r_s,
                   n_u=len(uci), n_r=len(uci), label="(b)")

    # ── Panel (c): ΔT_a = U-R ──
    ax_c.plot(HOURS, uhi_delta, color=COLOR_UHI, lw=2.2,
              label=r"$\Delta T_a$ in UHI areas")
    ax_c.plot(HOURS, uci_delta, color=COLOR_UCI, lw=2.2,
              label=r"$\Delta T_a$ in UCI areas")
    ax_c.axhline(0, color="gray", linestyle="--", lw=1.2)
    ax_c.set_xlim(0, 23); ax_c.set_xticks(range(0, 24, 4))
    ax_c.set_xlabel("Local Solar Time (h)", fontsize=FONT_LABEL)
    ax_c.set_ylabel(r"$\Delta T_a$ (°C)",   fontsize=FONT_LABEL)
    ax_c.set_title("(c)", loc="left", fontsize=FONT_TITLE, fontweight="bold")
    ax_c.legend(frameon=False, loc="upper right", fontsize=FONT_LEGEND)
    ax_c.grid(True, lw=0.4, alpha=0.35)
    add_black_frame(ax_c)

    fig.savefig(
        os.path.join(output_dir, "plots", "Figure2_UHI_UCI_diurnal.png"),
        dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: Figure2_UHI_UCI_diurnal.png")

# ─────────────────────────────────────────────────────────────
# Figure 2b (NEW): Annual vs Heatwave UHI/UCI diurnal comparison
#   6 subplots, 3 rows × 2 cols
#   Left  column = annual
#   Right column = heatwave
#   NCC / Nature-like clean style
# ─────────────────────────────────────────────────────────────
def plot_figure2b_uhi_uci_annual_vs_heatwave(all_df, output_dir):
    """
    6-panel comparison:
      Left column  = annual
      Right column = heatwave

      Row 1: UHI diurnal curves
      Row 2: UCI diurnal curves
      Row 3: ΔT_a = Urban - Rural

    Notes:
    - Does NOT modify or replace existing Figure 2.
    - Adds a brand-new figure only.
    - All six panels are square.
    """

    if all_df is None or len(all_df) == 0:
        print("  [Fig2b] No data, skipping.")
        return

    annual_df = all_df[all_df["period"] == "annual"].copy()
    hw_df     = all_df[all_df["period"] == "heatwave"].copy()

    if len(annual_df) == 0:
        print("  [Fig2b] No annual data, skipping.")
        return
    if len(hw_df) == 0:
        print("  [Fig2b] No heatwave data, skipping.")
        return

    # ── split groups ─────────────────────────────────────────
    annual_uhi = annual_df[annual_df["group"] == "UHI"].copy()
    annual_uci = annual_df[annual_df["group"] == "UCI"].copy()
    hw_uhi     = hw_df[hw_df["group"] == "UHI"].copy()
    hw_uci     = hw_df[hw_df["group"] == "UCI"].copy()

    if (len(annual_uhi) == 0 and len(annual_uci) == 0 and
        len(hw_uhi) == 0 and len(hw_uci) == 0):
        print("  [Fig2b] No UHI/UCI rows found, skipping.")
        return

    # ── NCC / Nature-like styling (local only) ──────────────
    FONT_LABEL_LOCAL  = 12
    FONT_TICK_LOCAL   = 11
    FONT_TITLE_LOCAL  = 12
    FONT_LEGEND_LOCAL = 10
    FONT_ANNOT_LOCAL  = 10

    LINE_MAIN   = 1.8
    LINE_REF    = 0.9
    GRID_LW     = 0.35
    GRID_ALPHA  = 0.28
    BAND_ALPHA  = 0.16
    FILL_ALPHA  = 0.18

    COLOR_URBAN_LOCAL = COLOR_URBAN
    COLOR_RURAL_LOCAL = COLOR_RURAL
    COLOR_UHI_LOCAL   = COLOR_UHI
    COLOR_UCI_LOCAL   = COLOR_UCI

    def _safe_group_curves(df_sub):
        """Return u_mean, u_std, r_mean, r_std; if empty -> nan arrays."""
        if df_sub is None or len(df_sub) == 0:
            nan24 = np.full(24, np.nan)
            return nan24, nan24, nan24, nan24
        u_arr, r_arr = get_diurnal_arrays(df_sub)
        u_mean, u_std = mean_std(u_arr)
        r_mean, r_std = mean_std(r_arr)
        return u_mean, u_std, r_mean, r_std

    # annual
    a_uhi_u_m, a_uhi_u_s, a_uhi_r_m, a_uhi_r_s = _safe_group_curves(annual_uhi)
    a_uci_u_m, a_uci_u_s, a_uci_r_m, a_uci_r_s = _safe_group_curves(annual_uci)

    # heatwave
    h_uhi_u_m, h_uhi_u_s, h_uhi_r_m, h_uhi_r_s = _safe_group_curves(hw_uhi)
    h_uci_u_m, h_uci_u_s, h_uci_r_m, h_uci_r_s = _safe_group_curves(hw_uci)

    # deltas
    a_uhi_delta = a_uhi_u_m - a_uhi_r_m
    a_uci_delta = a_uci_u_m - a_uci_r_m
    h_uhi_delta = h_uhi_u_m - h_uhi_r_m
    h_uci_delta = h_uci_u_m - h_uci_r_m

    # ── unified y-limits so left/right formats are truly matched ─────────
    def _nanminmax(arr_list, pad_frac=0.06):
        vals = np.concatenate([x[np.isfinite(x)] for x in arr_list if np.any(np.isfinite(x))]) \
               if any(np.any(np.isfinite(x)) for x in arr_list) else np.array([0, 1], dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if np.isclose(vmin, vmax):
            vmin -= 1
            vmax += 1
        pad = (vmax - vmin) * pad_frac
        return vmin - pad, vmax + pad

    # same y-range for all absolute-temperature panels
    y_temp_min, y_temp_max = _nanminmax([
        a_uhi_u_m - a_uhi_u_s, a_uhi_u_m + a_uhi_u_s,
        a_uhi_r_m - a_uhi_r_s, a_uhi_r_m + a_uhi_r_s,
        a_uci_u_m - a_uci_u_s, a_uci_u_m + a_uci_u_s,
        a_uci_r_m - a_uci_r_s, a_uci_r_m + a_uci_r_s,
        h_uhi_u_m - h_uhi_u_s, h_uhi_u_m + h_uhi_u_s,
        h_uhi_r_m - h_uhi_r_s, h_uhi_r_m + h_uhi_r_s,
        h_uci_u_m - h_uci_u_s, h_uci_u_m + h_uci_u_s,
        h_uci_r_m - h_uci_r_s, h_uci_r_m + h_uci_r_s,
    ], pad_frac=0.05)

    # same y-range for both delta panels
    y_delta_min, y_delta_max = _nanminmax([
        a_uhi_delta, a_uci_delta, h_uhi_delta, h_uci_delta
    ], pad_frac=0.08)

    # keep zero visible and symmetric-looking if possible
    mabs = max(abs(y_delta_min), abs(y_delta_max))
    y_delta_min, y_delta_max = -mabs * 1.05, mabs * 1.05

    # ── canvas ───────────────────────────────────────────────
    # 3 rows × 2 cols, each axes square
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 14.8), dpi=600)
    plt.subplots_adjust(wspace=0.28, hspace=0.34)

    # column headers
    axes[0, 0].text(0.5, 1.14, "Annual period",
                    transform=axes[0, 0].transAxes,
                    ha="center", va="bottom",
                    fontsize=FONT_TITLE_LOCAL + 1, fontweight="bold")
    axes[0, 1].text(0.5, 1.14, "Heatwave period",
                    transform=axes[0, 1].transAxes,
                    ha="center", va="bottom",
                    fontsize=FONT_TITLE_LOCAL + 1, fontweight="bold")

    def _style_axes(ax):
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))
        ax.tick_params(axis="both", labelsize=FONT_TICK_LOCAL, width=0.8, length=3.5)
        ax.grid(True, lw=GRID_LW, alpha=GRID_ALPHA, color="#b0b0b0")
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.9)
            ax.spines[side].set_color("black")
        # square panel
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

    def _plot_diurnal(ax, u_mean, u_std, r_mean, r_std, n_pair, panel_label):
        ax.plot(HOURS, u_mean, color=COLOR_URBAN_LOCAL, lw=LINE_MAIN, label="Urban", zorder=3)
        ax.fill_between(HOURS, u_mean - u_std, u_mean + u_std,
                        color=COLOR_URBAN_LOCAL, alpha=BAND_ALPHA, zorder=2)

        ax.plot(HOURS, r_mean, color=COLOR_RURAL_LOCAL, lw=LINE_MAIN, label="Rural", zorder=3)
        ax.fill_between(HOURS, r_mean - r_std, r_mean + r_std,
                        color=COLOR_RURAL_LOCAL, alpha=BAND_ALPHA, zorder=2)

        ax.fill_between(HOURS, u_mean, r_mean,
                        where=(u_mean >= r_mean),
                        color=COLOR_URBAN_LOCAL, alpha=FILL_ALPHA,
                        interpolate=True, zorder=1)
        ax.fill_between(HOURS, u_mean, r_mean,
                        where=(u_mean < r_mean),
                        color=COLOR_UCI_LOCAL, alpha=FILL_ALPHA,
                        interpolate=True, zorder=1)

        ax.set_ylim(y_temp_min, y_temp_max)
        ax.set_xlabel("Local Solar Time (h)", fontsize=FONT_LABEL_LOCAL)
        ax.set_ylabel(r"$T_a$ (°C)", fontsize=FONT_LABEL_LOCAL)
        ax.set_title(f"{panel_label}  ($n={n_pair}$)",
                     loc="left", fontsize=FONT_TITLE_LOCAL, fontweight="bold", pad=5)

        leg = ax.legend(frameon=False, loc="upper left",
                        fontsize=FONT_LEGEND_LOCAL, handlelength=1.8)
        _style_axes(ax)

    def _plot_delta(ax, delta_uhi, delta_uci, n_uhi, n_uci, panel_label):
        ax.plot(HOURS, delta_uhi, color=COLOR_UHI_LOCAL, lw=LINE_MAIN,
                label=rf"UHI ($n={n_uhi}$)", zorder=3)
        ax.plot(HOURS, delta_uci, color=COLOR_UCI_LOCAL, lw=LINE_MAIN,
                label=rf"UCI ($n={n_uci}$)", zorder=3)
        ax.axhline(0, color="#666666", linestyle="--", lw=LINE_REF, zorder=1)

        ax.set_ylim(y_delta_min, y_delta_max)
        ax.set_xlabel("Local Solar Time (h)", fontsize=FONT_LABEL_LOCAL)
        ax.set_ylabel(r"$\Delta T_a$ (°C)", fontsize=FONT_LABEL_LOCAL)
        ax.set_title(f"{panel_label}",
                     loc="left", fontsize=FONT_TITLE_LOCAL, fontweight="bold", pad=5)

        ax.legend(frameon=False, loc="upper right",
                  fontsize=FONT_LEGEND_LOCAL, handlelength=1.8)
        _style_axes(ax)

    # Row 1: UHI
    _plot_diurnal(
        axes[0, 0],
        a_uhi_u_m, a_uhi_u_s, a_uhi_r_m, a_uhi_r_s,
        n_pair=len(annual_uhi), panel_label="(a) UHI"
    )
    _plot_diurnal(
        axes[0, 1],
        h_uhi_u_m, h_uhi_u_s, h_uhi_r_m, h_uhi_r_s,
        n_pair=len(hw_uhi), panel_label="(b) UHI"
    )

    # Row 2: UCI
    _plot_diurnal(
        axes[1, 0],
        a_uci_u_m, a_uci_u_s, a_uci_r_m, a_uci_r_s,
        n_pair=len(annual_uci), panel_label="(c) UCI"
    )
    _plot_diurnal(
        axes[1, 1],
        h_uci_u_m, h_uci_u_s, h_uci_r_m, h_uci_r_s,
        n_pair=len(hw_uci), panel_label="(d) UCI"
    )

    # Row 3: Delta
    _plot_delta(
        axes[2, 0],
        a_uhi_delta, a_uci_delta,
        n_uhi=len(annual_uhi), n_uci=len(annual_uci),
        panel_label="(e) Urban - Rural intensity"
    )
    _plot_delta(
        axes[2, 1],
        h_uhi_delta, h_uci_delta,
        n_uhi=len(hw_uhi), n_uci=len(hw_uci),
        panel_label="(f) Urban - Rural intensity"
    )

    fig.suptitle(
        "Figure 2b. Diurnal temperature characteristics in annual and heatwave periods",
        fontsize=FONT_TITLE_LOCAL + 2, fontweight="bold", y=0.995
    )

    ensure_dir(os.path.join(output_dir, "plots"))
    fpath = os.path.join(output_dir, "plots", "Figure2b_annual_vs_heatwave_UHI_UCI_diurnal.png")
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────
# Figure 3: LCZ 分类三面板
# ─────────────────────────────────────────────────────────────
def plot_figure3_lcz(annual_df, output_dir):
    compact = annual_df[annual_df["urban_lcz_class"] == "compact"]
    open_lz = annual_df[annual_df["urban_lcz_class"] == "open"]

    if len(compact) == 0 and len(open_lz) == 0:
        print("  [Fig3] No LCZ data, skipping.")
        return

    def lcz_delta(df_sub):
        if len(df_sub) == 0:
            return np.full(24, np.nan), np.full(24, np.nan)
        u, r = get_diurnal_arrays(df_sub)
        return mean_std(u), mean_std(r)

    (cp_u_m, cp_u_s), (cp_r_m, cp_r_s) = (
        lcz_delta(compact) if len(compact) else
        ((np.full(24,np.nan), np.full(24,np.nan)),
         (np.full(24,np.nan), np.full(24,np.nan)))
    )
    (op_u_m, op_u_s), (op_r_m, op_r_s) = (
        lcz_delta(open_lz) if len(open_lz) else
        ((np.full(24,np.nan), np.full(24,np.nan)),
         (np.full(24,np.nan), np.full(24,np.nan)))
    )

    # 需要分别算 compact/open 的均值（含 UHI+UCI 所有 pair）
    def _safe_mean_std(df_sub):
        if len(df_sub) == 0:
            return np.full(24, np.nan), np.full(24, np.nan)
        u_arr, r_arr = get_diurnal_arrays(df_sub)
        um, us = mean_std(u_arr)
        rm, rs = mean_std(r_arr)
        return um, us, rm, rs

    if len(compact) > 0:
        cp_u_m, cp_u_s, cp_r_m, cp_r_s = _safe_mean_std(compact)
    else:
        cp_u_m = cp_u_s = cp_r_m = cp_r_s = np.full(24, np.nan)

    if len(open_lz) > 0:
        op_u_m, op_u_s, op_r_m, op_r_s = _safe_mean_std(open_lz)
    else:
        op_u_m = op_u_s = op_r_m = op_r_s = np.full(24, np.nan)

    cp_delta = cp_u_m - cp_r_m
    op_delta = op_u_m - op_r_m

    fig = plt.figure(figsize=(6.5, 13))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.42)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    def _lcz_panel(ax, u_mean, u_std, r_mean, r_std, n, title):
        ax.plot(HOURS, u_mean, color=COLOR_URBAN, lw=2.2, label="Urban")
        ax.fill_between(HOURS, u_mean - u_std, u_mean + u_std,
                        color=COLOR_URBAN, alpha=ALPHA_BAND)
        ax.plot(HOURS, r_mean, color=COLOR_RURAL, lw=2.2, label="Rural")
        ax.fill_between(HOURS, r_mean - r_std, r_mean + r_std,
                        color=COLOR_RURAL, alpha=ALPHA_BAND)
        ax.fill_between(HOURS, u_mean, r_mean,
                        where=(u_mean >= r_mean),
                        alpha=0.22, color=COLOR_URBAN, interpolate=True)
        ax.fill_between(HOURS, u_mean, r_mean,
                        where=(u_mean < r_mean),
                        alpha=0.22, color=COLOR_UCI, interpolate=True)
        ax.set_title(f"{title}  ($n={n}$)",
                     loc="left", fontsize=FONT_TITLE, fontweight="bold")
        ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 4))
        ax.set_xlabel("Local Solar Time (h)", fontsize=FONT_LABEL)
        ax.set_ylabel("$T_a$ (°C)",           fontsize=FONT_LABEL)
        ax.legend(frameon=False, loc="upper left", fontsize=FONT_LEGEND)
        ax.grid(True, lw=0.4, alpha=0.35)
        add_black_frame(ax)

    _lcz_panel(ax_a, cp_u_m, cp_u_s, cp_r_m, cp_r_s,
               n=len(compact), title="(a) Compact Highrise")
    _lcz_panel(ax_b, op_u_m, op_u_s, op_r_m, op_r_s,
               n=len(open_lz), title="(b) Open Lowrise")

    # Panel (c): intensity comparison
    ax_c.plot(HOURS, cp_delta, color=COLOR_COMPACT, lw=2.2,
              label="Compact Highrise")
    ax_c.plot(HOURS, op_delta, color=COLOR_OPEN, lw=2.2,
              label="Open Lowrise")
    ax_c.axhline(0, color="gray", linestyle="--", lw=1.2)
    ax_c.set_xlim(0, 23); ax_c.set_xticks(range(0, 24, 4))
    ax_c.set_xlabel("Local Solar Time (h)", fontsize=FONT_LABEL)
    ax_c.set_ylabel(r"$\Delta T_a$ (°C)",   fontsize=FONT_LABEL)
    ax_c.set_title("(c) UHI/UCI Intensity Comparison",
                   loc="left", fontsize=FONT_TITLE, fontweight="bold")
    ax_c.legend(frameon=False, loc="upper right", fontsize=FONT_LEGEND)
    ax_c.grid(True, lw=0.4, alpha=0.35)
    add_black_frame(ax_c)

    fig.savefig(
        os.path.join(output_dir, "plots", "Figure3_LCZ_comparison.png"),
        dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: Figure3_LCZ_comparison.png")


# ─────────────────────────────────────────────────────────────
# Statistics Table (Table 1)
# ─────────────────────────────────────────────────────────────
def compute_stats_table(df_input, output_dir, suffix="Table1", title="annual"):

    """
    对 annual 期所有 UHI / UCI 对，计算：
      Urban mean±std | Rural mean±std | Δ(U-R) mean±std  [p-value via 1-sample t-test]
    输出 CSV + 格式化 TXT。
    """
    variables = [
        ("Tmn (°C)",      "urban_Tmean",    "rural_Tmean"),
        ("Amp_d1 (°C)",   "urban_Amp1",     "rural_Amp1"),
        ("Amp_d2 (°C)",   "urban_Amp2",     "rural_Amp2"),
        ("T_x (°C)",      "urban_Tmax_fft", "rural_Tmax_fft"),
        ("T_n (°C)",      "urban_Tmin_fft", "rural_Tmin_fft"),
    ]

    csv_rows = []
    txt_lines = []

    for group_name in ["UHI", "UCI"]:
        gdf = df_input[df_input["group"] == group_name].copy()
        n = len(gdf)
        txt_lines.append(f"\n{'─'*100}")
        txt_lines.append(f"  {group_name} Cluster  (n={n})")
        txt_lines.append(f"{'─'*100}")
        txt_lines.append(
            f"  {'Variable':18s}  {'Urban':18s}  {'Rural':18s}  {'Δ(U-R)':22s}  p-val  sig")
        txt_lines.append("  " + "-"*96)

        for label, u_col, r_col in variables:
            if u_col not in gdf.columns or r_col not in gdf.columns:
                txt_lines.append(f"  {label:18s}  {'N/A':18s}  {'N/A':18s}  {'N/A':22s}")
                continue

            u_vals = gdf[u_col].dropna()
            r_vals = gdf[r_col].dropna()
            common_idx = u_vals.index.intersection(r_vals.index)
            delta = gdf.loc[common_idx, u_col] - gdf.loc[common_idx, r_col]
            delta = delta.dropna()

            u_str = (f"{u_vals.mean():.2f} ± {u_vals.std():.2f}"
                     if len(u_vals) > 0 else "N/A")
            r_str = (f"{r_vals.mean():.2f} ± {r_vals.std():.2f}"
                     if len(r_vals) > 0 else "N/A")

            if len(delta) >= 3:
                t_stat, p_val = stats.ttest_1samp(delta, 0)
                d_str = f"{delta.mean():.2f} ± {delta.std():.2f}"
                sig   = stars_from_p(p_val)
                p_str = f"{p_val:.4f}"
            else:
                d_str = "N/A"
                p_val = np.nan
                sig   = ""
                p_str = "N/A"

            txt_lines.append(
                f"  {label:18s}  {u_str:18s}  {r_str:18s}  "
                f"{d_str:22s}  {p_str:7s}  {sig}")

            csv_rows.append({
                "group": group_name, "n": int(n),
                "variable": label,
                "urban_mean": float(u_vals.mean()) if len(u_vals) else np.nan,
                "urban_std":  float(u_vals.std())  if len(u_vals) else np.nan,
                "rural_mean": float(r_vals.mean()) if len(r_vals) else np.nan,
                "rural_std":  float(r_vals.std())  if len(r_vals) else np.nan,
                "delta_mean": float(delta.mean())  if len(delta) else np.nan,
                "delta_std":  float(delta.std())   if len(delta) else np.nan,
                "delta_p":    float(p_val),
                "significance": sig,
            })

    txt_lines.append("")

    # Print to console
    for line in txt_lines:
        print(line)

    # Save TXT
    txt_path = os.path.join(output_dir, f"{suffix}_statistics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"\n  Saved: {txt_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, f"{suffix}_statistics.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── 绘制可视化版表格图 ────────────────────────────────────
    plot_table_figure(csv_rows, output_dir, suffix=suffix, title=title)



def plot_table_figure(csv_rows, output_dir, suffix="Table1", title="annual"):
    """将统计表绘制为图像（便于论文插入）。"""
    if not csv_rows:
        return
    df = pd.DataFrame(csv_rows)

    groups     = ["UHI", "UCI"]
    var_order  = ["Tmn (°C)", "Amp_d1 (°C)", "Amp_d2 (°C)",
                  "T_x (°C)", "T_n (°C)"]
    col_labels = ["Variable", "Urban\nmean±std", "Rural\nmean±std", "Δ(U-R)\nmean±std", "sig"]

    nrows = len(var_order)
    ncols = len(col_labels)

    fig, axes = plt.subplots(1, 2, figsize=(18, 3.8))
    fig.suptitle(
    f"{suffix}. Diurnal Metrics: UHI vs UCI Clusters ({title} period)",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)

    for ax, group in zip(axes, groups):
        gdf = df[df["group"] == group].copy()
        n_grp = int(gdf["n"].iloc[0]) if len(gdf) > 0 else 0
        ax.set_title(f"{group} Cluster  (n={n_grp})",
                     fontsize=FONT_TITLE, fontweight="bold", pad=8)
        ax.axis("off")

        table_data = []
        for var in var_order:
            row_df = gdf[gdf["variable"] == var]
            if row_df.empty:
                table_data.append([var, "N/A", "N/A", "N/A", ""])
                continue
            r = row_df.iloc[0]
            u_str = f"{r['urban_mean']:.2f} ± {r['urban_std']:.2f}" if pd.notna(r["urban_mean"]) else "N/A"
            r_str = f"{r['rural_mean']:.2f} ± {r['rural_std']:.2f}" if pd.notna(r["rural_mean"]) else "N/A"
            d_str = f"{r['delta_mean']:.2f} ± {r['delta_std']:.2f}" if pd.notna(r["delta_mean"]) else "N/A"
            sig   = str(r["significance"]) if pd.notna(r["significance"]) else ""
            table_data.append([var, u_str, r_str, d_str, sig])

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(FONT_TICK)

        # Header styling
        for j in range(ncols):
            cell = tbl[(0, j)]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=FONT_TICK)

        # Alternating row colors
        for i in range(1, nrows + 1):
            fc = "#f0f4f8" if i % 2 == 0 else "white"
            for j in range(ncols):
                tbl[(i, j)].set_facecolor(fc)
                tbl[(i, j)].set_edgecolor("#cccccc")

        # Bold significant delta column
        for i, var in enumerate(var_order, start=1):
            row_df = gdf[gdf["variable"] == var]
            if row_df.empty: continue
            r = row_df.iloc[0]
            if pd.notna(r["delta_p"]) and r["delta_p"] < 0.05:
                tbl[(i, 3)].set_text_props(fontweight="bold", color="#c0392b")

    fig.tight_layout()
    fpath = os.path.join(output_dir, "plots", f"{suffix}_figure.png")
    ensure_dir(os.path.dirname(fpath))
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────
# Figure 4 (bonus): Mechanism scatter — BV/dAmp1/dTx/dTn
# ─────────────────────────────────────────────────────────────
def plot_figure4_mechanism(annual_df, output_dir):
    df = annual_df.copy()
    df["log10_BV"] = np.log10(df["urban_BV_m3"].replace(0, np.nan))
    df["color"]    = df["group"].map({"UHI": COLOR_UHI, "UCI": COLOR_UCI}).fillna("gray")

    panels = [
        ("log10_BV", "dAmp1",
         r"(a) $\log_{10}$(Urban BV) vs $\Delta Amp_1$",
         r"$\log_{10}$(Urban BV, m$^3$)", r"$\Delta Amp_1$ (°C)"),
        ("dAmp1", "dTx",
         r"(b) $\Delta Amp_1$ vs $\Delta T_x$",
         r"$\Delta Amp_1$ (°C)", r"$\Delta T_x$ (°C)"),
        ("dAmp1", "dTn",
         r"(c) $\Delta Amp_1$ vs $\Delta T_n$",
         r"$\Delta Amp_1$ (°C)", r"$\Delta T_n$ (°C)"),
        ("log10_BV", "dTx",
         r"(d) $\log_{10}$(Urban BV) vs $\Delta T_x$",
         r"$\log_{10}$(Urban BV, m$^3$)", r"$\Delta T_x$ (°C)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, (x, y, title, xlabel, ylabel) in zip(axes, panels):
        sub = df[[x, y, "color"]].dropna()
        if len(sub) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes); continue

        ax.scatter(sub[x], sub[y], c=sub["color"],
                   s=32, alpha=0.75, edgecolors="none")

        if len(sub) >= 3:
            lr  = stats.linregress(sub[x].values.astype(float),
                                   sub[y].values.astype(float))
            xx  = np.linspace(sub[x].min(), sub[x].max(), 200)
            ax.plot(xx, lr.intercept + lr.slope * xx, "k-", lw=1.5)
            add_stats_text(ax, n=len(sub), r2=lr.rvalue**2,
                           slope=lr.slope, p=lr.pvalue, loc="upper left")

        ax.set_title(title, loc="left", fontsize=FONT_TITLE)
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.grid(True, lw=0.4, alpha=0.35)
        add_black_frame(ax)

    handles = [
        plt.Line2D([0],[0], marker="o", ls="", color=COLOR_UHI, label="UHI", ms=9),
        plt.Line2D([0],[0], marker="o", ls="", color=COLOR_UCI, label="UCI", ms=9),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=FONT_LEGEND, loc="lower right")

    fig.tight_layout()
    fpath = os.path.join(output_dir, "plots", "Figure4_mechanism_closure.png")
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")

# ─────────────────────────────────────────────────────────────
# Figure 4b (NEW): Phase diagrams
#   (1) 2D phase space: ΔAmp vs ΔT_mn, colored by regime
#   (2) 3D phase space: ΔAmp vs ΔT_mn vs LCZ_corrected
# Nature-like clean style, minimal decoration
# ─────────────────────────────────────────────────────────────
def plot_phase_diagrams(annual_df, output_dir):
    df = annual_df.copy()

    needed = ["dTn", "dAmp1", "group", "urban_lcz_corrected"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  [Phase] Missing columns: {missing}, skipping.")
        return

    # keep numeric LCZ only for 3D phase diagram
    df["urban_lcz_corrected_num"] = pd.to_numeric(df["urban_lcz_corrected"], errors="coerce")

    # -------------------------
    # (a) 2D phase diagram
    # x = ΔT_mn (dTn)
    # y = ΔAmp (dAmp1)
    # -------------------------
    sub2d = df[["dTn", "dAmp1", "group"]].dropna().copy()
    if len(sub2d) > 0:
        fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=600)

        # Nature-like restrained palette
        color_map = {"UHI": "#c0392b", "UCI": "#2980b9"}
        for grp in ["UHI", "UCI"]:
            g = sub2d[sub2d["group"] == grp]
            if len(g) == 0:
                continue
            ax.scatter(
                g["dTn"], g["dAmp1"],
                s=24, alpha=0.78, linewidths=0,
                color=color_map[grp], label=grp, zorder=3
            )

        # reference lines
        ax.axhline(0, color="#7f8c8d", linestyle="--", lw=0.9, zorder=1)
        ax.axvline(0, color="#b0b0b0", linestyle=":", lw=0.8, zorder=1)

        # a simple regime-balance guideline: ΔAmp + ΔTmn = 0
        xlim = np.nanpercentile(sub2d["dTn"], [1, 99])
        xx = np.linspace(xlim[0], xlim[1], 300)
        yy = -xx
        ax.plot(xx, yy, color="black", lw=1.0, alpha=0.9, zorder=2)

        ax.text(
            0.98, 0.96, r"$\Delta Amp + \Delta T_{mn}=0$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_ANNOT-1, color="black"
        )

        ax.set_xlabel(r"$\Delta T_{mn}$ (°C)", fontsize=FONT_LABEL)
        ax.set_ylabel(r"$\Delta Amp$ (°C)", fontsize=FONT_LABEL)
        ax.set_title("Figure 4b(a). 2D phase diagram of urban thermal regimes",
                     loc="left", fontsize=FONT_TITLE, fontweight="bold", pad=6)

        ax.legend(frameon=False, loc="lower left", fontsize=FONT_LEGEND-1)
        ax.grid(True, lw=0.35, alpha=0.28, zorder=0)
        add_black_frame(ax)

        add_stats_text(ax, n=len(sub2d), loc="upper left")

        fpath = os.path.join(output_dir, "plots", "Figure4b_phase_diagram_2D.png")
        fig.tight_layout()
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fpath}")
    else:
        print("  [Phase 2D] No valid data, skipping.")

    # -------------------------
    # (b) 3D phase diagram
    # x = ΔT_mn (dTn)
    # y = ΔAmp (dAmp1)
    # z = urban_lcz_corrected
    # -------------------------
    sub3d = df[["dTn", "dAmp1", "urban_lcz_corrected_num", "group"]].dropna().copy()
    if len(sub3d) > 0:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(6.6, 5.6), dpi=600)
        ax = fig.add_subplot(111, projection="3d")

        color_map = {"UHI": "#c0392b", "UCI": "#2980b9"}
        marker_map = {"UHI": "o", "UCI": "^"}

        for grp in ["UHI", "UCI"]:
            g = sub3d[sub3d["group"] == grp]
            if len(g) == 0:
                continue
            ax.scatter(
                g["dTn"], g["dAmp1"], g["urban_lcz_corrected_num"],
                s=22, alpha=0.76, linewidths=0,
                color=color_map[grp], marker=marker_map[grp],
                depthshade=False, label=grp
            )

        ax.set_xlabel(r"$\Delta T_{mn}$ (°C)", labelpad=8, fontsize=FONT_LABEL-1)
        ax.set_ylabel(r"$\Delta Amp$ (°C)",    labelpad=8, fontsize=FONT_LABEL-1)
        ax.set_zlabel("Urban LCZ",             labelpad=6, fontsize=FONT_LABEL-1)
        ax.set_title("Figure 4b(b). 3D phase diagram with LCZ constraint",
                     loc="left", fontsize=FONT_TITLE, fontweight="bold", pad=8)

        # Nature-like clean panes
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0.0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0.0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0.0))

        ax.grid(True, alpha=0.18)
        add_black_frame(ax)
        ax.view_init(elev=22, azim=-58)

        # make z ticks integer-like if possible
        zmin = int(np.floor(sub3d["urban_lcz_corrected_num"].min()))
        zmax = int(np.ceil(sub3d["urban_lcz_corrected_num"].max()))
        if zmax - zmin <= 12:
            ax.set_zticks(list(range(zmin, zmax + 1)))

        ax.legend(frameon=False, loc="upper left", fontsize=FONT_LEGEND-2)

        fpath = os.path.join(output_dir, "plots", "Figure4b_phase_diagram_3D.png")
        fig.tight_layout()
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fpath}")
    else:
        print("  [Phase 3D] No valid numeric LCZ data, skipping.")


# ─────────────────────────────────────────────────────────────
# Figure 5 (bonus): Heatwave pressure test bar chart
# ─────────────────────────────────────────────────────────────
def plot_figure5_hw_pressure(all_df, output_dir):
    metrics = ["dTmean","dAmp1","dTx","dTn"]
    pretty  = {
        "dTmean": r"$\Delta T_a$",
        "dAmp1":  r"$\Delta Amp$",
        "dTx":    r"$\Delta T_x$",
        "dTn":    r"$\Delta T_n$",
    }
    hw_summary_path = os.path.join(output_dir, "hw_pressure_test_summary.csv")
    if not os.path.exists(hw_summary_path):
        # 尝试在上一级目录找（兼容旧路径）
        hw_summary_path = os.path.join(
            os.path.dirname(output_dir), "hw_pressure_test_summary.csv"
        )
    if not os.path.exists(hw_summary_path):
        print("  [Fig5] hw_pressure_test_summary.csv not found, skipping.")
        return

    summ = pd.read_csv(hw_summary_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics)); width = 0.30

    for offset, group, color in [(-width/2, "UHI", COLOR_UHI),
                                   ( width/2, "UCI", COLOR_UCI)]:
        ys, ylo, yhi = [], [], []
        for m in metrics:
            row = summ[(summ["group"]==group) & (summ["metric"]==m)]
            if row.empty:
                ys.append(np.nan); ylo.append(np.nan); yhi.append(np.nan)
                continue
            r = row.iloc[0]
            ys.append(r["mean_hw_minus_nhw"])
            ylo.append(r["mean_hw_minus_nhw"] - r["ci_low"])
            yhi.append(r["ci_high"] - r["mean_hw_minus_nhw"])

        ax.bar(x+offset, ys, width=width, color=color, alpha=0.85,
               label=group, edgecolor="black", lw=0.6)
        ax.errorbar(x+offset, ys, yerr=[ylo,yhi],
                    fmt="none", ecolor="black", capsize=3, lw=1.1)

    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[m] for m in metrics], fontsize=FONT_LABEL)
    ax.set_ylabel(r"HW$-$NHW difference (°C)", fontsize=FONT_LABEL)
    ax.set_title("Figure 5. Heatwave Pressure Test", fontsize=FONT_TITLE,
                 fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    ax.grid(True, axis="y", alpha=0.3, lw=0.7)
    add_black_frame(ax)
    ax.margins(x=0.06)
    fig.tight_layout(pad=0.8)

    fpath = os.path.join(output_dir, "plots", "Figure5_hw_pressure_test.png")
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")

# ─────────────────────────────────────────────────────────────
# Figure 6: 全球站点地图 + UHI/UCI 代表性日循环 inset
#   主图：全球地图，站点按 UHI/UCI 和 NDVI 四分位着色
#   Inset：左上=UHI代表配对日循环，右下=UCI代表配对日循环
#   风格：Nature 期刊
# ─────────────────────────────────────────────────────────────
def plot_figure_map_insets(annual_df, output_dir):
    if not HAS_CARTOPY:
        print("  [Fig6] cartopy not available, skipping.")
        return

    if annual_df is None or len(annual_df) == 0:
        print("  [Fig6] No data.")
        return

    # ── Nature 风格全局设置 ──────────────────────────────────
    NATURE_FONT = 7          # Nature 正文字号约7-8pt
    NATURE_LABEL = 8
    NATURE_TITLE = 8
    NATURE_TICK  = 7

    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":        NATURE_FONT,
        "axes.labelsize":   NATURE_LABEL,
        "xtick.labelsize":  NATURE_TICK,
        "ytick.labelsize":  NATURE_TICK,
        "axes.linewidth":   0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "legend.fontsize":   NATURE_TICK,
        "legend.frameon":    False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    # ── 数据准备 ─────────────────────────────────────────────
    df = annual_df.copy()

    # NDVI 四分位：用于区分植被覆盖程度
    if "urban_NDVI" in df.columns:
        df["ndvi_q"] = pd.qcut(
            df["urban_NDVI"].rank(method="first"),
            q=4, labels=["Q1\n(Low veg)", "Q2", "Q3", "Q4\n(High veg)"]
        )
    else:
        df["ndvi_q"] = "Unknown"

    uhi_df = df[df["group"] == "UHI"].copy()
    uci_df = df[df["group"] == "UCI"].copy()

    # ── 配色方案（Nature 常用配色） ──────────────────────────
    # UHI: 红色系，UCI: 蓝色系；深→浅 对应 NDVI Q1→Q4（植被越多越浅）
    UHI_COLORS = ["#a50f15", "#de2d26", "#fb6a4a", "#fcae91"]   # 深红→浅红
    UCI_COLORS = ["#084594", "#2171b5", "#6baed6", "#bdd7e7"]   # 深蓝→浅蓝
    NDVI_LEVELS = ["Q1\n(Low veg)", "Q2", "Q3", "Q4\n(High veg)"]

    # ── 画布布局 ─────────────────────────────────────────────
    # 单栏宽度约 89mm ≈ 3.5 inch；双栏约 183mm ≈ 7.2 inch
    fig = plt.figure(figsize=(7.2, 5.0), dpi=600)

    # 主地图轴
    proj = ccrs.Robinson(central_longitude=0)
    ax_map = fig.add_axes([0.02, 0.12, 0.96, 0.82], projection=proj)

    # ── 底图 ────────────────────────────────────────────────
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND,   facecolor="#f5f5f0", edgecolor="none", zorder=0)
    ax_map.add_feature(cfeature.OCEAN,  facecolor="#d6e8f5", edgecolor="none", zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.25, edgecolor="#888888", zorder=1)
    ax_map.add_feature(cfeature.BORDERS,   linewidth=0.18, edgecolor="#bbbbbb",
                       linestyle=":", zorder=1)
    ax_map.gridlines(linewidth=0.2, color="#cccccc", alpha=0.6,
                     xlocs=range(-180, 181, 60), ylocs=range(-60, 91, 30))

    # ── 绘制站点 ─────────────────────────────────────────────
    MARKER_SIZE = 10   # 单位 pt²，Nature 地图站点通常很小

    for qi, qlabel in enumerate(NDVI_LEVELS):
        # UHI
        sub_uhi = uhi_df[uhi_df["ndvi_q"] == qlabel]
        if len(sub_uhi) > 0:
            ax_map.scatter(
                sub_uhi["lon_urban"].values, sub_uhi["lat_urban"].values,
                transform=ccrs.PlateCarree(),
                s=MARKER_SIZE, c=UHI_COLORS[qi],
                marker="o", alpha=0.80, linewidths=0,
                zorder=3,
                label=f"UHI · NDVI {qlabel.split(chr(10))[0]}"
            )
        # UCI
        sub_uci = uci_df[uci_df["ndvi_q"] == qlabel]
        if len(sub_uci) > 0:
            ax_map.scatter(
                sub_uci["lon_urban"].values, sub_uci["lat_urban"].values,
                transform=ccrs.PlateCarree(),
                s=MARKER_SIZE, c=UCI_COLORS[qi],
                marker="^", alpha=0.80, linewidths=0,
                zorder=3,
                label=f"UCI · NDVI {qlabel.split(chr(10))[0]}"
            )

    # ── 图例 ─────────────────────────────────────────────────
    legend = ax_map.legend(
        loc="lower left", ncol=2,
        fontsize=5.5,
        markerscale=1.4,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        facecolor="white",
    )
    legend.get_frame().set_linewidth(0.4)

    ax_map.set_title("(a)  Global distribution of urban–rural station pairs",
                     loc="left", fontsize=NATURE_TITLE,
                     fontweight="bold", pad=4)

    # ── 统计注释 ─────────────────────────────────────────────
    n_uhi = len(uhi_df); n_uci = len(uci_df)
    ax_map.text(0.01, 0.97,
                f"$n_{{\\rm UHI}}={n_uhi}$,  $n_{{\\rm UCI}}={n_uci}$",
                transform=ax_map.transAxes,
                fontsize=6, va="top", ha="left",
                color="#333333")

    # ─────────────────────────────────────────────────────────
    # Inset 1：左上角 — UHI 代表配对日循环
    # ─────────────────────────────────────────────────────────
    # 选取 UHI 组 dTx 最大的 top-5 的均值作为代表曲线
    top_uhi = uhi_df.nlargest(min(5, len(uhi_df)), "dTx")
    u_cols  = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols  = [f"rural_diurnal_h{h:02d}" for h in range(24)]

    uhi_u_m = top_uhi[u_cols].values.mean(axis=0)
    uhi_r_m = top_uhi[r_cols].values.mean(axis=0)
    uhi_u_s = top_uhi[u_cols].values.std(axis=0)
    uhi_r_s = top_uhi[r_cols].values.std(axis=0)

    # Inset 轴位置：[left, bottom, width, height]（归一化坐标）
    ax_inset_uhi = fig.add_axes([0.08, 0.56, 0.22, 0.26])
    _draw_inset_diurnal(
        ax_inset_uhi, uhi_u_m, uhi_u_s, uhi_r_m, uhi_r_s,
        title="(b) UHI pairs", n=len(top_uhi),
        color_u="#d62728", color_r="#2ca02c",
        font_size=NATURE_FONT
    )

    # ─────────────────────────────────────────────────────────
    # Inset 2：右下角 — UCI 代表配对日循环
    # ─────────────────────────────────────────────────────────
    top_uci = uci_df.nsmallest(min(5, len(uci_df)), "dTx")   # dTx 最负（最强 UCI）

    uci_u_m = top_uci[u_cols].values.mean(axis=0)
    uci_r_m = top_uci[r_cols].values.mean(axis=0)
    uci_u_s = top_uci[u_cols].values.std(axis=0)
    uci_r_s = top_uci[r_cols].values.std(axis=0)

    ax_inset_uci = fig.add_axes([0.70, 0.14, 0.22, 0.26])
    _draw_inset_diurnal(
        ax_inset_uci, uci_u_m, uci_u_s, uci_r_m, uci_r_s,
        title="(c) UCI pairs", n=len(top_uci),
        color_u="#d62728", color_r="#2ca02c",
        font_size=NATURE_FONT
    )

    # ─────────────────────────────────────────────────────────
    # 从 Inset 到地图站点的指示箭头
    # ─────────────────────────────────────────────────────────
    # UHI 代表站点质心（用 top_uhi 的均值经纬度）
    uhi_lon_c = float(top_uhi["lon_urban"].mean())
    uhi_lat_c = float(top_uhi["lat_urban"].mean())
    _draw_map_arrow(fig, ax_map, ax_inset_uhi,
                    map_lonlat=(uhi_lon_c, uhi_lat_c),
                    inset_anchor=(0.5, 0.0),   # 从 inset 底部中央出发
                    proj=proj, color="#d62728")

    uci_lon_c = float(top_uci["lon_urban"].mean())
    uci_lat_c = float(top_uci["lat_urban"].mean())
    _draw_map_arrow(fig, ax_map, ax_inset_uci,
                    map_lonlat=(uci_lon_c, uci_lat_c),
                    inset_anchor=(0.5, 1.0),   # 从 inset 顶部中央出发
                    proj=proj, color="#1f77b4")

    # ── 保存 ─────────────────────────────────────────────────
    ensure_dir(os.path.join(output_dir, "plots"))
    fpath = os.path.join(output_dir, "plots", "Figure6_map_inset_diurnal.pdf")
    fig.savefig(fpath, dpi=600, bbox_inches="tight", format="pdf")
    # 同时输出 PNG 便于预览
    fig.savefig(fpath.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")

    # 恢复默认 rcParams（避免影响后续图）
    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────
# 辅助函数 1：绘制单个 inset 日循环面板（Nature 风格）
# ─────────────────────────────────────────────────────────────
def _draw_inset_diurnal(ax, u_mean, u_std, r_mean, r_std,
                         title, n, color_u, color_r, font_size=7):
    hours = np.arange(24)
    ALPHA_BAND = 0.18
    LW = 1.2

    # 曲线
    ax.plot(hours, u_mean, color=color_u, lw=LW, label="Urban")
    ax.fill_between(hours, u_mean - u_std, u_mean + u_std,
                    color=color_u, alpha=ALPHA_BAND)
    ax.plot(hours, r_mean, color=color_r, lw=LW, label="Rural")
    ax.fill_between(hours, r_mean - r_std, r_mean + r_std,
                    color=color_r, alpha=ALPHA_BAND)

    # 城乡差值填充
    ax.fill_between(hours, u_mean, r_mean,
                    where=(u_mean >= r_mean),
                    alpha=0.20, color=color_u, interpolate=True)
    ax.fill_between(hours, u_mean, r_mean,
                    where=(u_mean < r_mean),
                    alpha=0.20, color="#1f77b4", interpolate=True)

    # 轴样式
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["0", "6", "12", "18", "23"], fontsize=font_size - 1)
    ax.tick_params(axis="both", which="major", labelsize=font_size - 1,
                   length=2, width=0.5, pad=1)
    ax.set_xlabel("Local Solar Time (h)", fontsize=font_size, labelpad=2)
    ax.set_ylabel("$T_a$ (°C)",           fontsize=font_size, labelpad=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.grid(True, lw=0.3, alpha=0.35, color="#aaaaaa")
    add_black_frame(ax)

    # 标题 & 图例
    ax.set_title(f"{title}  ($n={n}$)",
                 loc="left", fontsize=font_size, fontweight="bold", pad=3)
    leg = ax.legend(fontsize=font_size - 1, loc="upper left",
                    handlelength=1.2, handletextpad=0.4,
                    borderpad=0.4, labelspacing=0.2)
    leg.get_frame().set_visible(False)

    # 白色半透明底（使 inset 在地图上清晰可读）
    ax.set_facecolor((1, 1, 1, 0.88))
    for spine in ax.spines.values():
        spine.set_edgecolor("#666666")


# ─────────────────────────────────────────────────────────────
# 辅助函数 2：从 inset 到地图站点质心画箭头
# ─────────────────────────────────────────────────────────────
def _draw_map_arrow(fig, ax_map, ax_inset, map_lonlat,
                    inset_anchor, proj, color="#555555"):
    """
    在 figure 坐标系中，从 inset 的某个锚点画一条折线指向地图上的站点。
    """
    from cartopy.crs import PlateCarree

    # 将地图经纬度转换为 display 坐标
    lon, lat = map_lonlat
    try:
        xy_map_display = ax_map.projection.transform_point(
            lon, lat, PlateCarree())
        xy_map_display = ax_map.transData.transform(xy_map_display)
        xy_map_fig    = fig.transFigure.inverted().transform(xy_map_display)
    except Exception:
        return   # 超出投影范围则跳过

    # inset 锚点（figure 坐标）
    ix, iy = inset_anchor
    inset_bbox = ax_inset.get_position()
    xi_fig = inset_bbox.x0 + ix * inset_bbox.width
    yi_fig = inset_bbox.y0 + iy * inset_bbox.height

    # 画线（figure 坐标系中的连线）
    line = plt.Line2D(
        [xi_fig, xy_map_fig[0]],
        [yi_fig, xy_map_fig[1]],
        transform=fig.transFigure,
        color=color, linewidth=0.7, linestyle="--",
        alpha=0.75, zorder=10,
        clip_on=False
    )
    fig.add_artist(line)

    # 终点小圆点（标记站点位置）
    dot = plt.Circle(xy_map_fig, radius=0.005,
                     transform=fig.transFigure,
                     color=color, zorder=11, clip_on=False)
    fig.add_artist(dot)


# ─────────────────────────────────────────────────────────────
# [新增] Figure 7: 全球站点热浪分布地图
#   所有站点（不区分城市/农村），
#   红色圆点 = 该配对周边曾出现热浪（数据集中存在 period=='heatwave' 记录）
#   蓝色圆点 = 未出现过热浪
# ─────────────────────────────────────────────────────────────
def plot_figure7_heatwave_map(all_df, output_dir):
    """
    [新增] 世界地图：全部站点（不区分城市农村），
    蓝色圆点 = 未出现热浪；
    红色圆点（深浅）= 热浪出现频次，颜色越深频次越高。
    """
    if all_df is None or len(all_df) == 0:
        print("  [Fig7] No data, skipping.")
        return

    df = all_df.copy()

    # ── 计算每个配对的热浪频次 ────────────────────────────────
    id_col = None
    for candidate in ["pair_id", "station_id", "city_id"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is not None:
        hw_counts = (
            df[df["period"] == "heatwave"]
            .groupby(id_col)
            .size()
            .rename("hw_count")
        )
        annual = (
            df[df["period"] == "annual"]
            .drop_duplicates(subset=[id_col])
            .copy()
        )
        annual = annual.join(hw_counts, on=id_col, how="left")
    else:
        df["_loc_key"] = (
            df["lat_urban"].astype(str) + "_" + df["lon_urban"].astype(str)
        )
        hw_counts = (
            df[df["period"] == "heatwave"]
            .groupby("_loc_key")
            .size()
            .rename("hw_count")
        )
        annual = (
            df[df["period"] == "annual"]
            .drop_duplicates(subset=["_loc_key"])
            .copy()
        )
        annual = annual.join(hw_counts, on="_loc_key", how="left")

    annual["hw_count"] = annual["hw_count"].fillna(0).astype(int)

    hw_yes = annual[annual["hw_count"] > 0].copy()
    hw_no  = annual[annual["hw_count"] == 0].copy()
    n_yes    = len(hw_yes)
    n_no     = len(hw_no)
    n_total  = len(annual)
    pct      = n_yes / n_total * 100 if n_total > 0 else 0.0

    COLOR_HW_NO  = "#1f77b4"
    MARKER_SIZE  = 18
    CMAP_HW      = plt.cm.Reds
    vmin_hw      = 1
    vmax_hw      = annual["hw_count"].max() if n_yes > 0 else 1
    norm_hw      = matplotlib.colors.Normalize(vmin=vmin_hw, vmax=vmax_hw)

    # ── 绘制地图 ──────────────────────────────────────────────
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(12, 6), dpi=600)
        proj = ccrs.Robinson(central_longitude=0)
        ax = fig.add_axes([0.02, 0.06, 0.88, 0.88], projection=proj)
        ax.set_global()
        ax.add_feature(cfeature.LAND,
                       facecolor="#f2f2ee", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN,
                       facecolor="#d4e8f4", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE,
                       linewidth=0.3, edgecolor="#606060", zorder=1)
        ax.add_feature(cfeature.BORDERS,
                       linewidth=0.2, edgecolor="#aaaaaa",
                       linestyle=":", zorder=1)
        ax.gridlines(linewidth=0.2, color="#cccccc", alpha=0.5,
                     xlocs=range(-180, 181, 60),
                     ylocs=range(-60, 91, 30))
        add_black_frame(ax)

        # 无热浪站点（蓝色）
        if len(hw_no) > 0:
            ax.scatter(
                hw_no["lon_urban"].values,
                hw_no["lat_urban"].values,
                transform=ccrs.PlateCarree(),
                s=MARKER_SIZE, c=COLOR_HW_NO,
                marker="o", alpha=0.65, linewidths=0,
                zorder=2,
                label=f"No heatwave  ($n={n_no}$)"
            )

        # 有热浪站点（红色深浅=频次）
        if len(hw_yes) > 0:
            sc = ax.scatter(
                hw_yes["lon_urban"].values,
                hw_yes["lat_urban"].values,
                transform=ccrs.PlateCarree(),
                s=MARKER_SIZE,
                c=hw_yes["hw_count"].values,
                cmap=CMAP_HW, norm=norm_hw,
                marker="o", alpha=0.85, linewidths=0,
                zorder=3
            )

        legend = ax.legend(
            loc="lower left",
            fontsize=FONT_LEGEND - 1,
            markerscale=1.6,
            frameon=True, framealpha=0.92,
            edgecolor="#cccccc", facecolor="white",
        )
        legend.get_frame().set_linewidth(0.5)

        # Colorbar（仅在有热浪站点时添加）
        if n_yes > 0:
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
            sm  = plt.cm.ScalarMappable(cmap=CMAP_HW, norm=norm_hw)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Heatwave frequency\n(number of events)",
                           fontsize=FONT_ANNOT - 1)
            cbar.ax.tick_params(labelsize=FONT_TICK - 2)
            if vmax_hw <= 10:
                cbar.set_ticks(range(int(vmin_hw), int(vmax_hw) + 1))

    else:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
        if len(hw_no) > 0:
            ax.scatter(hw_no["lon_urban"], hw_no["lat_urban"],
                       s=MARKER_SIZE, c=COLOR_HW_NO, alpha=0.65,
                       linewidths=0, zorder=2,
                       label=f"No heatwave  ($n={n_no}$)")
        if len(hw_yes) > 0:
            sc = ax.scatter(
                hw_yes["lon_urban"], hw_yes["lat_urban"],
                s=MARKER_SIZE,
                c=hw_yes["hw_count"].values,
                cmap=CMAP_HW, norm=norm_hw,
                alpha=0.85, linewidths=0, zorder=3
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
            cbar.set_label("Heatwave frequency\n(number of events)",
                           fontsize=FONT_ANNOT - 1)
            cbar.ax.tick_params(labelsize=FONT_TICK - 2)
        ax.set_xlabel("Longitude (°)", fontsize=FONT_LABEL)
        ax.set_ylabel("Latitude (°)",  fontsize=FONT_LABEL)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.grid(True, lw=0.3, alpha=0.35)
        add_black_frame(ax)
        ax.legend(frameon=False, fontsize=FONT_LEGEND - 1, loc="lower left")

    ax.set_title(
        f"Global distribution of all station pairs  —  "
        f"Heatwave frequency  ($N_{{\\rm total}}={n_total}$)",
        loc="left" if HAS_CARTOPY else "center",
        fontsize=FONT_TITLE, fontweight="bold", pad=6
    )
    ax.text(0.99, 0.97,
            f"Heatwave occurrence rate: {pct:.1f}%\n"
            f"(based on percentile-defined heatwave periods)",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=FONT_ANNOT - 1,
            color="#333333",
            bbox=dict(facecolor="white", alpha=0.88,
                      edgecolor="#cccccc", pad=4, lw=0.5))

    ensure_dir(os.path.join(output_dir, "plots"))
    fpath = os.path.join(output_dir, "plots", "Figure7_heatwave_station_map.png")
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")
    print(f"  [Fig7] Total pairs: {n_total}  |  With heatwave: {n_yes} ({pct:.1f}%)  "
          f"|  Without: {n_no}  |  Max freq: {int(annual['hw_count'].max())}")

def plot_figure8_hw_diurnal(all_df, output_dir):
    """
    [新增] 热浪 vs 非热浪时期日循环（不区分城市农村）。
    仅保留绝对温度对比面板，std 色带不加入图例，图幅 1:1。
    """
    if all_df is None or len(all_df) == 0:
        print("  [Fig8] No data, skipping.")
        return

    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]
    avail_u = [c for c in u_cols if c in all_df.columns]
    avail_r = [c for c in r_cols if c in all_df.columns]
    if len(avail_u) < 24 and len(avail_r) < 24:
        print("  [Fig8] Diurnal columns not found, skipping.")
        return

    def _extract_all_stations(df_period):
        parts = []
        if len(avail_u) == 24:
            parts.append(df_period[avail_u].values.astype(float))
        if len(avail_r) == 24:
            parts.append(df_period[avail_r].values.astype(float))
        if not parts:
            return np.full((0, 24), np.nan)
        combined = np.vstack(parts)
        valid_mask = ~np.all(np.isnan(combined), axis=1)
        return combined[valid_mask]

    hw_df  = all_df[all_df["period"] == "heatwave"]
    nhw_df = all_df[all_df["period"] == "non_heatwave"]

    if len(hw_df) == 0:
        print("  [Fig8] No heatwave period rows found, skipping.")
        return
    if len(nhw_df) == 0:
        print("  [Fig8] No non_heatwave period rows found, skipping.")
        return

    hw_mat  = _extract_all_stations(hw_df)
    nhw_mat = _extract_all_stations(nhw_df)

    if hw_mat.shape[0] == 0 or nhw_mat.shape[0] == 0:
        print("  [Fig8] Insufficient valid data after extraction, skipping.")
        return

    hw_mean,  hw_std  = np.nanmean(hw_mat,  axis=0), np.nanstd(hw_mat,  axis=0)
    nhw_mean, nhw_std = np.nanmean(nhw_mat, axis=0), np.nanstd(nhw_mat, axis=0)

    n_hw_pairs  = len(hw_df)
    n_nhw_pairs = len(nhw_df)

    COLOR_HW       = "#c0392b"
    COLOR_NHW      = "#2980b9"
    COLOR_HW_BAND  = "#e74c3c"
    COLOR_NHW_BAND = "#5dade2"
    ALPHA_STD      = 0.22
    LW_MAIN        = 2.4

    # ── 1:1 正方形画布，单面板 ─────────────────────────────────
    fig, ax_main = plt.subplots(figsize=(8, 8), dpi=600)

    # std 色带（不加入图例）
    ax_main.fill_between(
        HOURS, hw_mean - hw_std, hw_mean + hw_std,
        color=COLOR_HW_BAND, alpha=ALPHA_STD
    )
    ax_main.fill_between(
        HOURS, nhw_mean - nhw_std, nhw_mean + nhw_std,
        color=COLOR_NHW_BAND, alpha=ALPHA_STD
    )

    # 均值曲线（加入图例）
    ax_main.plot(
        HOURS, hw_mean, color=COLOR_HW, lw=LW_MAIN,
        label=f"Heatwave (n={n_hw_pairs} pairs)"
    )
    ax_main.plot(
        HOURS, nhw_mean, color=COLOR_NHW, lw=LW_MAIN,
        label=f"Non-heatwave (n={n_nhw_pairs} pairs)"
    )

    ax_main.set_xlim(0, 23)
    ax_main.set_xticks(range(0, 24, 3))
    ax_main.set_xticklabels(
        [f"{h:02d}:00" for h in range(0, 24, 3)],
        rotation=30, ha="right", fontsize=FONT_TICK - 1
    )
    ax_main.set_xlabel("Local Solar Time (h)", fontsize=FONT_LABEL)
    ax_main.set_ylabel("$T_a$ (°C)",           fontsize=FONT_LABEL)
    ax_main.set_title(
        "Diurnal cycle: Heatwave vs Non-heatwave periods\n"
        "(all station pairs combined, urban + rural)",
        loc="left", fontsize=FONT_TITLE, fontweight="bold", pad=6
    )
    ax_main.legend(
        frameon=True, framealpha=0.90,
        edgecolor="#cccccc", fontsize=FONT_LEGEND - 1,
        loc="upper left"
    )
    ax_main.grid(True, lw=0.4, alpha=0.35)

    peak_diff_h = int(np.nanargmax(np.abs(hw_mean - nhw_mean)))
    mean_diff   = float(np.nanmean(hw_mean - nhw_mean))
    print(f"  [Fig8] HW mean: {np.nanmean(hw_mean):+.2f} °C  "
          f"| NHW mean: {np.nanmean(nhw_mean):+.2f} °C  "
          f"| avg diff: {mean_diff:+.3f} °C  peak at h={peak_diff_h:02d}:00")

    fig.tight_layout()
    ensure_dir(os.path.join(output_dir, "plots"))
    fpath = os.path.join(
        output_dir, "plots", "Figure8_hw_nonhw_diurnal_all_stations.png"
    )
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")

# ─────────────────────────────────────────────────────────────
# [新增] Figure 8: 热浪 vs 非热浪时期日循环对比
#   不区分城市/农村，将所有站点（urban + rural）的温度合并后
#   分别计算热浪/非热浪时期的均值与标准差，
#   std 色带统一使用红色系（深红=热浪，浅红=非热浪）
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# [新增] Mechanism Composite (Four-Panel)
# 将原先独立的代码封装为不干扰全局的独立函数
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# [新增] Mechanism Composite (Four-Panel)
# ─────────────────────────────────────────────────────────────
# 注意：函数增加 all_df 传参
def plot_figure_mechanism_composite2(all_df, output_dir):
    import matplotlib.patches as mpatches
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
    }):
        colors = {'UHI': '#d62728', 'UCI': '#1f77b4', 'NHW_line': '#888888', 'HW_line': '#222222'}

        # ==========================================
        # 真实数据处理：提取与合并 NHW(annual) 和 HW 状态
        # ==========================================
        # 1. 自动寻找配对主键 (兼容你其他函数的逻辑)
        id_col = None
        for candidate in ["pair_id", "station_id", "city_id"]:
            if candidate in all_df.columns:
                id_col = candidate
                break
        
        # 2. 筛选热浪与非热浪数据
        nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
        hw_df = all_df[all_df["period"] == "heatwave"].copy()

        # 3. 提取特征并重命名，准备合并
        df_nhw = nhw_df[[id_col, "group", "dTmean", "dAmp1", "dTx"]].rename(
            columns={"dTmean": "dTmean_nhw", "dAmp1": "dAmp_nhw", "dTx": "dTx_nhw"}
        )
        df_hw = hw_df[[id_col, "dTmean", "dAmp1", "dTx"]].rename(
            columns={"dTmean": "dTmean_hw", "dAmp1": "dAmp_hw", "dTx": "dTx_hw"}
        )

        # 4. 基于配对 ID 合并，生成专属的 df_a 用于画图
        df_a = pd.merge(df_nhw, df_hw, on=id_col, how="inner").dropna()
        if len(df_a) == 0:
            print("  [Mechanism Composite] No valid paired data found for Panel a. Skipping.")
            return

        # 其他 Panel 临时写死的数据 (后续你按需替换)
        metrics = [r"$\Delta T_a$", r"$\Delta Amp$", r"$\Delta T_x$", r"$\Delta T_n$"]
        uhi_bars = [0.33, 0.10, 0.51, 0.26]
        uhi_err = [0.07, 0.06, 0.09, 0.10]
        uci_bars = [0.23, -0.28, 0.00, 0.58]
        uci_err = [0.12, 0.09, 0.13, 0.18]

        hours = np.arange(24)
        t_rural_nhw = 20 + 5 * np.sin(np.pi * (hours - 8) / 12)
        t_diff_nhw = 1.5 + 0.8 * np.cos(np.pi * (hours - 6) / 12)
        t_rural_hw = 26 + 6 * np.sin(np.pi * (hours - 8) / 12)
        t_diff_hw = 2.5 + 1.2 * np.cos(np.pi * (hours - 4) / 12) 

        # 3. 绘图逻辑
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.35)

        # ==========================================
        # Panel a: Thermodynamic State Migration (Dumbbell Plot)
        # ==========================================
        ax_a = fig.add_subplot(gs[0, 0])

        # 1. 动态获取 dTx 色带范围，保证 0 对齐中心
        vmin = min(df_a['dTx_nhw'].min(), df_a['dTx_hw'].min())
        vmax = max(df_a['dTx_nhw'].max(), df_a['dTx_hw'].max())
        vmin = min(vmin, -0.1) # 兜底防止全正数导致 norm 报错
        vmax = max(vmax,  0.1) 
        
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) 
        cmap = plt.cm.RdBu_r

        # 2. 自适应坐标系边界计算（排除极端离群值，聚焦 1% ~ 99% 的核心区域）
        all_x = np.concatenate([df_a['dTmean_nhw'].dropna(), df_a['dTmean_hw'].dropna()])
        all_y = np.concatenate([df_a['dAmp_nhw'].dropna(), df_a['dAmp_hw'].dropna()])
        x_min, x_max = np.nanpercentile(all_x, 1), np.nanpercentile(all_x, 99)
        y_min, y_max = np.nanpercentile(all_y, 1), np.nanpercentile(all_y, 99)
        
        # 留出 15% 的视觉边距
        x_margin, y_margin = (x_max - x_min)*0.15, (y_max - y_min)*0.15
        final_xlim = (x_min - x_margin, x_max + x_margin)
        final_ylim = (y_min - y_margin, y_max + y_margin)

        # 3. 绘制连线：调细线宽、调高透明度，避免“毛线球”效应
        for idx, row in df_a.iterrows():
            # 只绘制在可视范围内的线，避免无意义的渲染
            ax_a.plot([row['dTmean_nhw'], row['dTmean_hw']], 
                      [row['dAmp_nhw'], row['dAmp_hw']], 
                      color='#888888', alpha=0.15, lw=0.6, zorder=1)

        # 4. 绘制散点：缩小体积，避免互相遮盖
        sc_nhw = ax_a.scatter(df_a['dTmean_nhw'], df_a['dAmp_nhw'], 
                              c=df_a['dTx_nhw'], cmap=cmap, norm=norm, 
                              marker='o', s=15, alpha=0.8, edgecolors='none', zorder=2)
                              
        sc_hw  = ax_a.scatter(df_a['dTmean_hw'], df_a['dAmp_hw'], 
                              c=df_a['dTx_hw'], cmap=cmap, norm=norm, 
                              marker='^', s=25, alpha=0.9, edgecolors='black', linewidths=0.5, zorder=3)

        # 5. 绘制 dTx = 0 的基准线 (y = -x)
        x_vals = np.linspace(final_xlim[0] - 5, final_xlim[1] + 5, 100)
        y_vals = -x_vals
        # 将虚线变粗一点，确保在背景中清晰可见
        ax_a.plot(x_vals, y_vals, color='#222222', linestyle='--', lw=1.5, zorder=4)

        # 6. 坐标轴与背景装饰
        ax_a.axhline(0, color='#cccccc', linestyle='-', lw=1.0, zorder=0)
        ax_a.axvline(0, color='#cccccc', linestyle='-', lw=1.0, zorder=0)

        ax_a.set_xlim(final_xlim)
        ax_a.set_ylim(final_ylim)

        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")
        ax_a.set_title("a  Thermodynamic state shift", loc="left", fontweight="bold")

        # 7. 添加 Colorbar
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_a, width="4%", height="40%", loc='lower left', bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax_a.transAxes)
        cbar = fig.colorbar(sc_hw, cax=cax, orientation='vertical')
        cbar.set_label(r"$\Delta T_x$ (°C)", fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=8)

        # 8. 图例更新：直接把虚线含义写进图例
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='NHW state'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=7, label='HW state'),
            plt.Line2D([0], [0], color='#222222', linestyle='--', lw=1.5, label=r'$\Delta T_x = 0$')
        ]
        ax_a.legend(handles=handles, frameon=False, loc="upper right", fontsize=9)

        # Panel b
        ax_b = fig.add_subplot(gs[0, 1])
        x = np.arange(len(metrics))
        width = 0.35
        ax_b.bar(x - width/2, uhi_bars, width, yerr=uhi_err, label='UHI', color=colors['UHI'], alpha=0.85, capsize=4, edgecolor='black')
        ax_b.bar(x + width/2, uci_bars, width, yerr=uci_err, label='UCI', color=colors['UCI'], alpha=0.85, capsize=4, edgecolor='black')
        ax_b.axhline(0, color='black', lw=1.0)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(metrics)
        ax_b.set_ylabel("HW$-$NHW difference (°C)")
        ax_b.set_title("b  Heatwave Pressure Test", loc="left", fontweight="bold")
        ax_b.legend(frameon=False)

        # Panel c
        ax_c = fig.add_subplot(gs[1, 0])
        waterfall_x = ['NHW Base', '$\Delta T_{mean}$', '$\Delta Amp$ (Day)', '$\Delta Amp$ (Night)', 'HW $\Delta T_x$', 'HW $\Delta T_n$']
        starts = [0, 1.5, 2.3, 2.3, 0, 0]
        changes = [1.5, 0.8, -0.4, +0.4, 1.9, 2.7] 
        colors_wf = ['gray', '#d62728', '#1f77b4', '#d62728', 'black', 'black']
        for i in range(len(waterfall_x)):
            if waterfall_x[i].startswith('HW') or waterfall_x[i].startswith('NHW'):
                ax_c.bar(waterfall_x[i], changes[i], bottom=0, color=colors_wf[i], alpha=0.7, width=0.5)
            else:
                ax_c.bar(waterfall_x[i], changes[i], bottom=starts[i], color=colors_wf[i], alpha=0.8, width=0.5)
                ax_c.plot([i-0.5, i+0.5], [starts[i], starts[i]], color='black', lw=0.5, linestyle='--')
        ax_c.set_ylabel("Urban-Rural $\Delta T$ (°C)")
        ax_c.set_title("c  Kinematic decomposition of night-time amplification", loc="left", fontweight="bold")
        ax_c.tick_params(axis='x', rotation=30)

        # Panel d
        ax_d = fig.add_subplot(gs[1, 1])
        def draw_loop_with_arrows(ax, x, y, color, label):
            ax.plot(x, y, color=color, lw=2, label=label, alpha=0.8)
            for i in [6, 12, 18]: 
                ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            ax.scatter(x[0], y[0], color=color, marker='o', s=50, zorder=5)
            ax.text(x[0]-0.5, y[0]+0.1, "00:00", color=color, fontsize=9)

        draw_loop_with_arrows(ax_d, t_rural_nhw, t_diff_nhw, colors['NHW_line'], 'NHW period')
        draw_loop_with_arrows(ax_d, t_rural_hw, t_diff_hw, colors['HW_line'], 'Heatwave period')
        ax_d.set_xlabel("Rural Background Temperature (°C)")
        ax_d.set_ylabel("Urban$-$Rural $\Delta T$ (°C)")
        ax_d.set_title("d  Thermal hysteresis and phase shift", loc="left", fontweight="bold")
        ax_d.legend(frameon=False, loc='lower right')

        # 4. 统一的保存逻辑（适配你原脚本的目录结构）
        fpath = os.path.join(output_dir, "plots", "Figure_Mechanism_Composite.pdf")
        ensure_dir(os.path.dirname(fpath))
        fig.savefig(fpath, bbox_inches='tight', dpi=600)
        # 顺便存一张 png
        fig.savefig(fpath.replace(".pdf", ".png"), bbox_inches='tight', dpi=600)
        plt.close(fig)
        print(f"  Saved: {fpath}")

def plot_figure_mechanism_composite(all_df, output_dir):

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
    }):
        colors = {
            "UHI": "#d62728",
            "UCI": "#1f77b4",
            "NHW_line": "#888888",
            "HW_line": "#222222",
        }

        # -----------------------------
        # Data preparation
        # -----------------------------
        id_col = None
        for candidate in ["pair_id", "station_id", "city_id"]:
            if candidate in all_df.columns:
                id_col = candidate
                break

        if id_col is None:
            print("  [Mechanism Composite] No pair id column found. Skipping.")
            return

        nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
        hw_df = all_df[all_df["period"] == "heatwave"].copy()

        required_cols = [id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]
        for c in required_cols:
            if c not in all_df.columns:
                print(f"  [Mechanism Composite] Missing column: {c}. Skipping.")
                return

        nhw_plot = nhw_df[required_cols].dropna().copy()
        hw_plot = hw_df[required_cols].dropna().copy()

        if len(nhw_plot) == 0 or len(hw_plot) == 0:
            print("  [Mechanism Composite] NHW or HW data is empty. Skipping.")
            return

        # Unified color scale
        dtx_all = pd.concat([nhw_plot["dTx"], hw_plot["dTx"]], axis=0).dropna()
        vabs = np.nanpercentile(np.abs(dtx_all.values), 98)
        vabs = max(vabs, 0.1)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        cmap = plt.cm.RdBu_r

        # Unified x/y limits
        x_all = pd.concat([nhw_plot["dTmean"], hw_plot["dTmean"]]).dropna()
        y_all = pd.concat([nhw_plot["dAmp1"], hw_plot["dAmp1"]]).dropna()

        y_min, y_max = np.nanpercentile(y_all, [1, 99])
        y_margin = (y_max - y_min) * 0.15
        ylim = (y_min - y_margin, y_max + y_margin)

        x_min, x_max = np.nanpercentile(x_all, [1, 99])
        x_margin = (x_max - x_min) * 0.15
        xlim = (x_min - x_margin, x_max + x_margin)


        # 只显示 ΔTmean >= 0 的点
        # nhw_plot = nhw_plot[nhw_plot["dTmean"] >= 0].copy()
        # hw_plot = hw_plot[hw_plot["dTmean"] >= 0].copy()

        # -----------------------------
        # Helper: fit ΔTx = a * ΔTmean + b * ΔAmp + c
        # -----------------------------
        def fit_dtx_zero_line(df):
            sub = df[["dTmean", "dAmp1", "dTx"]].dropna().copy()
            if len(sub) < 5:
                return np.nan, np.nan, np.nan

            X = np.column_stack([
                sub["dTmean"].values,
                sub["dAmp1"].values,
                np.ones(len(sub)),
            ])
            y = sub["dTx"].values

            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b, c = coef

            if np.isclose(b, 0):
                return np.nan, np.nan, np.nan

            slope = -a / b
            intercept = -c / b
            return slope, intercept, len(sub)

        def draw_state_panel(ax, df, title, marker):
            slope, intercept, n_fit = fit_dtx_zero_line(df)

            sc = ax.scatter(
                df["dTmean"],
                df["dAmp1"],
                c=df["dTx"],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=42 if marker == "^" else 38,
                alpha=0.95,
                edgecolors="black" if marker == "^" else "#333333",
                linewidths=0.45 if marker == "^" else 0.25,
                zorder=3,
            )

            ax.axhline(0, color="#cccccc", lw=1.0, zorder=0)
            ax.axvline(0, color="#cccccc", lw=1.0, zorder=0)

            if np.isfinite(slope) and np.isfinite(intercept):
                xx = np.linspace(xlim[0], xlim[1], 200)
                yy = slope * xx + intercept
                ax.plot(
                    xx, yy,
                    color="#222222",
                    linestyle="--",
                    lw=1.5,
                    zorder=4,
                    label=rf"$\Delta T_x=0$, slope={slope:.2f}",
                )
                ax.legend(frameon=False, loc="upper right", fontsize=9)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
            ax.set_ylabel(r"$\Delta Amp$ (°C)")
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(True, lw=0.35, alpha=0.28)

            ax.text(
                0.03, 0.96,
                f"n = {len(df)}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
            )

        def stars_from_p(p):
            if pd.isna(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return "ns"

        def paired_hw_nhw_delta(metric):
            base = nhw_df[[id_col, "group", metric]].rename(
                columns={metric: f"{metric}_nhw"}
            )
            hw = hw_df[[id_col, metric]].rename(
                columns={metric: f"{metric}_hw"}
            )
            merged = pd.merge(base, hw, on=id_col, how="inner").dropna()
            merged[f"{metric}_diff"] = merged[f"{metric}_hw"] - merged[f"{metric}_nhw"]
            return merged

        def mean_ci(vals):
            vals = pd.Series(vals).dropna().values
            n = len(vals)
            if n < 2:
                return np.nan, np.nan, n
            mean = np.mean(vals)
            se = stats.sem(vals)
            ci = 1.96 * se  # 95% Confidence Interval
            return mean, ci, n

        def paired_p(vals):
            vals = pd.Series(vals).dropna().values
            if len(vals) < 3:
                return np.nan
            return stats.ttest_1samp(vals, 0).pvalue

        def get_diurnal_matrix(df, prefix):
            cols = [f"{prefix}_diurnal_h{h:02d}" for h in range(24)]
            if not all(c in df.columns for c in cols):
                return None
            return df[cols].values.astype(float)

        def loop_area(x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if len(x) < 4:
                return np.nan
            return 0.5 * np.abs(
                np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
            )

        def phase_lag_hours(x, y):
            """
            修改：利用取模逻辑保证迟滞时间严格为正数。
            反映物理意义：城市响应（y峰值）滞后于乡村背景（x峰值）的小时数。
            """
            if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                return np.nan
            h_x = int(np.nanargmax(x))
            h_y = int(np.nanargmax(y))
            # 严格的正数滞后 (确保因果性)
            lag = (h_y - h_x) % 24
            return lag

        # -----------------------------
        # Layout: first row 2 panels,
        # second row 3 panels
        # -----------------------------
        fig = plt.figure(figsize=(16.5, 10))
        gs = fig.add_gridspec(
            2, 6,
            wspace=0.75,
            hspace=0.50,
            height_ratios=[1.05, 1.0]
        )

        ax_a = fig.add_subplot(gs[0, 0:3])
        ax_b = fig.add_subplot(gs[0, 3:6])
        ax_c = fig.add_subplot(gs[1, 0:2])
        ax_d = fig.add_subplot(gs[1, 2:4])
        ax_e = fig.add_subplot(gs[1, 4:6])

        # Panel a: NHW
        draw_state_panel(
            ax_a,
            nhw_plot,
            "a  NHW thermodynamic state",
            marker="o",
        )

        # Panel b: HW
        sc_hw = ax_b.scatter(
            hw_plot["dTmean"],
            hw_plot["dAmp1"],
            c=hw_plot["dTx"],
            cmap=cmap,
            norm=norm,
            marker="^",
            s=34,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.45,
            zorder=3,
        )
        ax_b.clear()
        draw_state_panel(
            ax_b,
            hw_plot,
            "b  HW thermodynamic state",
            marker="^",
        )

        # Shared colorbar for a/b
        cax = inset_axes(
            ax_b,
            width="4%",
            height="55%",
            loc="lower right",
            bbox_to_anchor=(0.08, 0.05, 1, 1),
            bbox_transform=ax_b.transAxes,
            borderpad=0,
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(r"$\Delta T_x$ (°C)", fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=8)

        # ============================================================
        # Panel c: Heatwave-induced change
        # ============================================================
        metrics_list = [
            ("dTmean", r"$\Delta T_a$"),
            ("dAmp1",  r"$\Delta Amp$"),
            ("dTx",    r"$\Delta T_x$"),
            ("dTn",    r"$\Delta T_n$"),
        ]

        x = np.arange(len(metrics_list))
        width = 0.35

        summary_c = {}

        for group_name in ["UHI", "UCI"]:
            means, cis, ns, ps = [], [], [], []

            for metric, _ in metrics_list:
                merged = paired_hw_nhw_delta(metric)
                g = merged[merged["group"] == group_name]
                vals = g[f"{metric}_diff"]

                m, ci, n = mean_ci(vals)
                p = paired_p(vals)

                means.append(m)
                cis.append(ci)
                ns.append(n)
                ps.append(p)

            summary_c[group_name] = {
                "means": means,
                "cis": cis,
                "ns": ns,
                "ps": ps,
            }

        bars_uhi = ax_c.bar(
            x - width / 2,
            summary_c["UHI"]["means"],
            width,
            yerr=summary_c["UHI"]["cis"],
            label=f"UHI (n={min(summary_c['UHI']['ns'])})",
            color=colors["UHI"],
            alpha=0.85,
            capsize=4,
            edgecolor="black",
        )

        bars_uci = ax_c.bar(
            x + width / 2,
            summary_c["UCI"]["means"],
            width,
            yerr=summary_c["UCI"]["cis"],
            label=f"UCI (n={min(summary_c['UCI']['ns'])})",
            color=colors["UCI"],
            alpha=0.85,
            capsize=4,
            edgecolor="black",
        )

        ax_c.axhline(0, color="black", lw=1.0)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([label for _, label in metrics_list])
        ax_c.set_ylabel("HW$-$NHW difference (°C)")
        
        # 修改：增加误差棒说明和物理意义解释，降低认知负担
        ax_c.set_title("c  Heatwave-induced change", loc="left", fontweight="bold", fontsize=11)
        ax_c.text(0.02, 0.98, 
                  "Error bars: 95% CI\n(Positive values = amplified urban heat stress)", 
                  transform=ax_c.transAxes, ha="left", va="top", fontsize=8.5, color="#444444")
        
        # 调整legend位置以免遮挡新增的解释文字
        ax_c.legend(frameon=False, loc="lower right", fontsize=8.5)
        ax_c.grid(True, axis="y", lw=0.35, alpha=0.28)

        # significance labels
        ymin_c, ymax_c = ax_c.get_ylim()
        yrange_c = ymax_c - ymin_c

        for i, (_, label) in enumerate(metrics_list):
            for offset, group_name in [(-width / 2, "UHI"), (width / 2, "UCI")]:
                mean_val = summary_c[group_name]["means"][i]
                ci_val = summary_c[group_name]["cis"][i]
                p_val = summary_c[group_name]["ps"][i]

                if not np.isfinite(mean_val):
                    continue

                star = stars_from_p(p_val)
                y_pos = mean_val + np.sign(mean_val) * ci_val

                if mean_val >= 0:
                    y_text = y_pos + 0.04 * yrange_c
                    va = "bottom"
                else:
                    y_text = y_pos - 0.06 * yrange_c
                    va = "top"

                ax_c.text(
                    i + offset,
                    y_text,
                    star,
                    ha="center",
                    va=va,
                    fontsize=9,
                    fontweight="bold",
                )

        # ============================================================
        # Panel d: Data-derived decomposition
        # ============================================================
        merged_d = pd.merge(
            nhw_df[[id_col, "group", "dTmean", "dAmp1", "dTn"]].rename(
                columns={"dTmean": "dTmean_nhw", "dAmp1": "dAmp_nhw", "dTn": "dTn_nhw"}
            ),
            hw_df[[id_col, "dTmean", "dAmp1", "dTn"]].rename(
                columns={"dTmean": "dTmean_hw", "dAmp1": "dAmp_hw", "dTn": "dTn_hw"}
            ),
            on=id_col,
            how="inner",
        ).dropna()

        fit_df_nhw = merged_d[["dTmean_nhw", "dAmp_nhw", "dTn_nhw"]].rename(
            columns={"dTmean_nhw": "dTmean", "dAmp_nhw": "dAmp1", "dTn_nhw": "dTn"}
        )
        fit_df_hw = merged_d[["dTmean_hw", "dAmp_hw", "dTn_hw"]].rename(
            columns={"dTmean_hw": "dTmean", "dAmp_hw": "dAmp1", "dTn_hw": "dTn"}
        )

        fit_df = pd.concat([fit_df_nhw, fit_df_hw], axis=0).dropna()

        X = np.column_stack([
            np.ones(len(fit_df)),
            fit_df["dTmean"].values,
            fit_df["dAmp1"].values,
        ])
        y = fit_df["dTn"].values

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta0, beta_tmean, beta_amp = coef
        
        # 修改：计算模型的 R^2 并将其展示在图表中
        y_pred = X.dot(coef)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        merged_d["d_dTmean"] = merged_d["dTmean_hw"] - merged_d["dTmean_nhw"]
        merged_d["d_dAmp"] = merged_d["dAmp_hw"] - merged_d["dAmp_nhw"]
        merged_d["d_dTn"] = merged_d["dTn_hw"] - merged_d["dTn_nhw"]

        contrib_tmean = beta_tmean * merged_d["d_dTmean"].mean()
        contrib_amp = beta_amp * merged_d["d_dAmp"].mean()
        observed_dtn = merged_d["d_dTn"].mean()
        residual = observed_dtn - contrib_tmean - contrib_amp

        waterfall_labels = [
            "NHW\nbase",
            r"$\Delta T_{mean}$",
            r"$\Delta Amp$",
            "Residual",
            r"Observed" + "\n" + r"$\Delta T_n$",
        ]

        base_level = merged_d["dTn_nhw"].mean()

        values = [
            base_level,
            contrib_tmean,
            contrib_amp,
            residual,
            base_level + observed_dtn,
        ]

        running = base_level
        starts = [0, running]

        running += contrib_tmean
        starts.append(running)

        running += contrib_amp
        starts.append(running)

        running += residual
        starts.append(0)

        bar_colors = [
            "#aaaaaa",
            "#d62728" if contrib_tmean >= 0 else "#1f77b4",
            "#d62728" if contrib_amp >= 0 else "#1f77b4",
            "#777777",
            "#333333",
        ]

        for i, lab in enumerate(waterfall_labels):
            if i == 0 or i == 4:
                ax_d.bar(i, values[i], bottom=0, color=bar_colors[i], width=0.55, edgecolor="black", lw=0.4)
            else:
                change = values[i]
                bottom = starts[i]
                ax_d.bar(i, change, bottom=bottom, color=bar_colors[i], width=0.55, edgecolor="black", lw=0.4)

                ax_d.plot(
                    [i - 0.45, i + 0.45],
                    [bottom, bottom],
                    color="black",
                    lw=0.6,
                    linestyle="--",
                )

        ax_d.set_xticks(np.arange(len(waterfall_labels)))
        ax_d.set_xticklabels(waterfall_labels, rotation=25, ha="right")
        ax_d.set_ylabel(r"$\Delta T_n$ level / contribution (°C)")
        ax_d.set_title("d  Data-derived night-time amplification", loc="left", fontweight="bold", fontsize=11)
        ax_d.grid(True, axis="y", lw=0.35, alpha=0.28)

        # 修改：文本框中增加 R^2 的显示
        ax_d.text(
            0.03,
            0.96,
            rf"$\beta_{{mean}}={beta_tmean:.2f}$" + "\n" +
            rf"$\beta_{{amp}}={beta_amp:.2f}$" + "\n" +
            rf"$R^2={r_squared:.2f}$" + "\n" +
            f"n = {len(merged_d)}",
            transform=ax_d.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=2),
        )

        # ============================================================
        # Panel e: Thermal hysteresis with quantified area and phase lag
        # ============================================================
        u_nhw = get_diurnal_matrix(nhw_df, "urban")
        r_nhw = get_diurnal_matrix(nhw_df, "rural")
        u_hw = get_diurnal_matrix(hw_df, "urban")
        r_hw = get_diurnal_matrix(hw_df, "rural")

        # 修改：彻底移除原本的硬编码(synthetic fallback)，保护代码的纯洁性
        if u_nhw is None or r_nhw is None or u_hw is None or r_hw is None:
            print("  [Panel e] Diurnal columns not found. Skipping panel e generation.")
            ax_e.text(0.5, 0.5, "Diurnal data missing\nCannot plot thermal hysteresis", 
                      ha="center", va="center", transform=ax_e.transAxes, color="#888888")
            ax_e.axis('off')
        else:
            t_rural_nhw = np.nanmean(r_nhw, axis=0)
            t_diff_nhw = np.nanmean(u_nhw - r_nhw, axis=0)

            t_rural_hw = np.nanmean(r_hw, axis=0)
            t_diff_hw = np.nanmean(u_hw - r_hw, axis=0)

            area_nhw = loop_area(t_rural_nhw, t_diff_nhw)
            area_hw = loop_area(t_rural_hw, t_diff_hw)

            lag_nhw = phase_lag_hours(t_rural_nhw, t_diff_nhw)
            lag_hw = phase_lag_hours(t_rural_hw, t_diff_hw)

            def draw_loop_with_arrows(ax, x, y, color, label):
                ax.plot(x, y, color=color, lw=2.0, label=label, alpha=0.85)

                for i in [6, 12, 18]:
                    ax.annotate(
                        "",
                        xy=(x[(i + 1) % 24], y[(i + 1) % 24]),
                        xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
                    )

                for h in [0, 6, 12, 18]:
                    ax.scatter(x[h], y[h], color=color, s=28, zorder=5)
                    ax.text(
                        x[h],
                        y[h],
                        f"{h:02d}:00",
                        color=color,
                        fontsize=7.5,
                        ha="left",
                        va="bottom",
                    )

            draw_loop_with_arrows(
                ax_e,
                t_rural_nhw,
                t_diff_nhw,
                colors["NHW_line"],
                rf"NHW: area={area_nhw:.2f}, lag=+{lag_nhw:.0f} h",
            )

            draw_loop_with_arrows(
                ax_e,
                t_rural_hw,
                t_diff_hw,
                colors["HW_line"],
                rf"HW: area={area_hw:.2f}, lag=+{lag_hw:.0f} h",
            )

            ax_e.set_xlabel("Rural background temperature (°C)")
            ax_e.set_ylabel("Urban$-$Rural $\Delta T$ (°C)")
            ax_e.set_title("e  Thermal hysteresis and phase shift", loc="left", fontweight="bold", fontsize=11)
            ax_e.legend(frameon=False, loc="lower right", fontsize=8)
            ax_e.grid(True, lw=0.35, alpha=0.28)

            ax_e.text(
                0.03,
                0.96,
                rf"$\Delta$area = {area_hw - area_nhw:+.2f}" + "\n" +
                rf"$\Delta$lag = {lag_hw - lag_nhw:+.0f} h",
                transform=ax_e.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=2),
            )

        # -----------------------------
        # Save
        # -----------------------------
        fpath = os.path.join(output_dir, "plots", "Figure_Mechanism_Composite.pdf")
        ensure_dir(os.path.dirname(fpath))

        fig.savefig(fpath, bbox_inches="tight", dpi=600)
        fig.savefig(fpath.replace(".pdf", ".png"), bbox_inches="tight", dpi=600)
        plt.close(fig)

        print(f"  Saved: {fpath}")


# ══════════════════════════════════════════════════════════════
# ★ 新增独立模块：逐对导出热滞后面积 (Area) 与相移数据 (Phase Lag)
# ══════════════════════════════════════════════════════════════
def export_thermal_hysteresis_data(all_df, output_dir):
    import os
    import numpy as np
    import pandas as pd

    print("\n--- Exporting Thermal Hysteresis & Phase Lag Data ---")
    out_dir_data = os.path.join(output_dir, "integrated_fig_data")
    os.makedirs(out_dir_data, exist_ok=True)
    out_path = os.path.join(out_dir_data, "mechanism_hysteresis_per_pair.csv")

    df = all_df.copy()

    # 检查是否存在逐小时温度列
    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]

    if not all(c in df.columns for c in u_cols + r_cols):
        print("  [Warning] Diurnal columns missing in dataset, cannot compute hysteresis.")
        return

    def calc_row_metrics(row):
        u_arr = row[u_cols].values.astype(float)
        r_arr = row[r_cols].values.astype(float)
        
        if np.isnan(u_arr).all() or np.isnan(r_arr).all():
            return pd.Series({"hysteresis_area": np.nan, "phase_lag_h": np.nan})
            
        x = r_arr           # X轴：乡村背景温度
        y = u_arr - r_arr   # Y轴：城乡温差
        
        # 1. 计算热滞后闭环面积 (Shoelace formula)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        
        # 2. 计算相移滞后时间 (Phase Lag)
        h_x = int(np.nanargmax(x))
        h_y = int(np.nanargmax(y))
        lag = (h_y - h_x) % 24
        
        return pd.Series({"hysteresis_area": area, "phase_lag_h": lag})

    print("  Computing hysteresis area and phase lag for each pair...")
    res = df.apply(calc_row_metrics, axis=1)
    df["hysteresis_area"] = res["hysteresis_area"]
    df["phase_lag_h"]     = res["phase_lag_h"]

    # 提取关键机制变量，打包导出
    id_col = "pair_id" if "pair_id" in df.columns else ("station_id" if "station_id" in df.columns else "city_id")
    export_cols = [c for c in [id_col, "group", "period", "dTmean", "dAmp1", "dTx", "dTn", 
                               "hysteresis_area", "phase_lag_h"] if c in df.columns]

    df[export_cols].to_csv(out_path, index=False)
    print(f"  [Success] Saved detailed mechanism data to: {out_path}")

def plot_figure_hw_shift_arrows2(all_df, output_dir):
    """
    Nature-style two-panel HW-NHW migration figure.

    Panel a:
        Pair-level NHW -> HW arrows in mean-amplitude space.
        x = ΔTmean, y = ΔAmp.

    Panel b:
        Region-resolved quantification in the same ΔTmean–ΔAmp space.
        Final HW states are grouped into:
            Q1 :  +ΔTmean, +ΔAmp
            Q2 :  -ΔTmean, +ΔAmp
            Q3 :  -ΔTmean, -ΔAmp
            Q4a: +ΔTmean, -ΔAmp and 0 < -ΔAmp < ΔTmean
                 damped daytime UHI region
            Q4b: +ΔTmean, -ΔAmp and -ΔAmp > ΔTmean
                 daytime UCI region

        Q4 is split by:
            -ΔAmp = ΔTmean
        equivalently:
            ΔAmp = -ΔTmean

    Outputs:
        plots/Figure_HW_shift_arrows_quantified.png
        plots/Figure_HW_shift_arrows_quantified.pdf
        integrated_fig_data/Figure_HW_shift_pair_vectors.csv
        integrated_fig_data/Figure_HW_shift_region_summary.csv
        integrated_fig_data/Figure_HW_shift_regime_transition_summary.csv
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    # -----------------------------
    # 0. Basic settings / fallback helpers
    # -----------------------------
    COLOR_UHI_LOCAL = globals().get("COLOR_UHI", "#d62728")
    COLOR_UCI_LOCAL = globals().get("COLOR_UCI", "#1f77b4")

    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    # -----------------------------
    # 1. Find ID column
    # -----------------------------
    id_col = None
    for candidate in ["pair_id", "station_id", "city_id"]:
        if candidate in all_df.columns:
            id_col = candidate
            break

    if id_col is None:
        print("  [HW arrows] No pair id column found, skipping.")
        return

    needed = [id_col, "group", "period", "dTmean", "dAmp1", "dTx"]
    missing = [c for c in needed if c not in all_df.columns]
    if missing:
        print(f"  [HW arrows] Missing columns: {missing}, skipping.")
        return

    # -----------------------------
    # 2. Build matched NHW-HW table
    # -----------------------------
    nhw = all_df[all_df["period"] == "non_heatwave"][
        [id_col, "group", "dTmean", "dAmp1", "dTx"]
    ].rename(columns={
        "dTmean": "dTmean_nhw",
        "dAmp1":  "dAmp_nhw",
        "dTx":    "dTx_nhw",
    })

    hw = all_df[all_df["period"] == "heatwave"][
        [id_col, "dTmean", "dAmp1", "dTx"]
    ].rename(columns={
        "dTmean": "dTmean_hw",
        "dAmp1":  "dAmp_hw",
        "dTx":    "dTx_hw",
    })

    df = pd.merge(nhw, hw, on=id_col, how="inner").dropna(
        subset=[
            "dTmean_nhw", "dAmp_nhw", "dTx_nhw",
            "dTmean_hw", "dAmp_hw", "dTx_hw"
        ]
    ).copy()

    if len(df) == 0:
        print("  [HW arrows] No paired NHW-HW data, skipping.")
        return

    # -----------------------------
    # 3. Migration metrics
    # -----------------------------
    df["Rmean"] = df["dTmean_hw"] - df["dTmean_nhw"]
    df["Ramp"]  = df["dAmp_hw"]   - df["dAmp_nhw"]
    df["D"]     = np.sqrt(df["Rmean"] ** 2 + df["Ramp"] ** 2)
    df["theta_deg"] = np.degrees(np.arctan2(df["Ramp"], df["Rmean"]))

    # -----------------------------
    # 4. Region classification in ΔTmean–ΔAmp space
    #    Only split the fourth quadrant by -ΔAmp = ΔTmean
    # -----------------------------
    def _region_label_hw(x, y):
        """
        Final HW-state classification.

        x = ΔTmean
        y = ΔAmp

        Q4 is split by:
            y = -x
        """
        if x >= 0 and y >= 0:
            return "Q1 +dTmean,+dAmp"

        if x < 0 and y >= 0:
            return "Q2 -dTmean,+dAmp"

        if x < 0 and y < 0:
            return "Q3 -dTmean,-dAmp"

        # Q4: x >= 0 and y < 0
        # Above y=-x: 0 < -ΔAmp < ΔTmean
        # Below y=-x: -ΔAmp > ΔTmean
        if y >= -x:
            return "Q4a damped daytime UHI"
        else:
            return "Q4b daytime UCI"

    df["hw_region"] = [
        _region_label_hw(x, y) for x, y in zip(df["dTmean_hw"], df["dAmp_hw"])
    ]

    df["nhw_region"] = [
        _region_label_hw(x, y) for x, y in zip(df["dTmean_nhw"], df["dAmp_nhw"])
    ]

    df["region_transition"] = df["nhw_region"] + " -> " + df["hw_region"]
    df["region_switched"] = df["nhw_region"] != df["hw_region"]
    df["period_specific_state_definition"] = PERIOD_SPECIFIC_DTX_STATE_DEFINITION
    df["canonical_group_definition"] = CANONICAL_GROUP_DEFINITION

    # -----------------------------
    # 5. Tx-transition classes
    # -----------------------------
    df["tx_transition_4class"] = "other"

    df.loc[
        (df["dTx_nhw"] > 0) & (df["dTx_hw"] > 0),
        "tx_transition_4class"
    ] = "positive_dTx_to_positive_dTx"

    df.loc[
        (df["dTx_nhw"] > 0) & (df["dTx_hw"] < 0),
        "tx_transition_4class"
    ] = "positive_dTx_to_negative_dTx"

    df.loc[
        (df["dTx_nhw"] < 0) & (df["dTx_hw"] < 0),
        "tx_transition_4class"
    ] = "negative_dTx_to_negative_dTx"

    df.loc[
        (df["dTx_nhw"] < 0) & (df["dTx_hw"] > 0),
        "tx_transition_4class"
    ] = "negative_dTx_to_positive_dTx"

    # Strengthening / weakening categories
    df["uhi_strength_change"] = "not_uhi_origin"
    df.loc[
        (df["dTx_nhw"] > 0) & (df["dTx_hw"] > df["dTx_nhw"]),
        "uhi_strength_change"
    ] = "positive_dTx_strengthened"

    df.loc[
        (df["dTx_nhw"] > 0) &
        (df["dTx_hw"] > 0) &
        (df["dTx_hw"] < df["dTx_nhw"]),
        "uhi_strength_change"
    ] = "positive_dTx_weakened"

    df["uci_strength_change"] = "not_uci_origin"
    df.loc[
        (df["dTx_nhw"] < 0) & (df["dTx_hw"] < df["dTx_nhw"]),
        "uci_strength_change"
    ] = "negative_dTx_strengthened"

    df.loc[
        (df["dTx_nhw"] < 0) &
        (df["dTx_hw"] < 0) &
        (df["dTx_hw"] > df["dTx_nhw"]),
        "uci_strength_change"
    ] = "negative_dTx_weakened"

    # -----------------------------
    # 6. Export pair-level data
    # -----------------------------
    data_dir = os.path.join(output_dir, "integrated_fig_data")
    _ensure_dir(data_dir)

    pair_out = os.path.join(data_dir, "Figure_HW_shift_pair_vectors.csv")
    df.to_csv(pair_out, index=False)

    # -----------------------------
    # 7. Region summary
    # -----------------------------
    region_order = [
        "Q1 +dTmean,+dAmp",
        "Q2 -dTmean,+dAmp",
        "Q3 -dTmean,-dAmp",
        "Q4a damped daytime UHI",
        "Q4b daytime UCI",
    ]

    region_rows = []

    for region in region_order:
        g = df[df["hw_region"] == region].copy()

        if len(g) == 0:
            region_rows.append({
                "region": region,
                "n": 0,
                "mean_dTmean_nhw": np.nan,
                "mean_dAmp_nhw": np.nan,
                "mean_dTmean_hw": np.nan,
                "mean_dAmp_hw": np.nan,
                "mean_Rmean": np.nan,
                "mean_Ramp": np.nan,
                "theta_deg_from_mean_vector": np.nan,
                "median_distance_D": np.nan,
                "mean_distance_D": np.nan,
                "n_positive_dTx_to_positive_dTx": 0,
                "n_positive_dTx_to_negative_dTx": 0,
                "n_negative_dTx_to_negative_dTx": 0,
                "n_negative_dTx_to_positive_dTx": 0,
                "n_positive_dTx_strengthened": 0,
                "n_positive_dTx_weakened": 0,
                "n_negative_dTx_strengthened": 0,
                "n_negative_dTx_weakened": 0,
                "n_region_switch": 0,
                "n_other": 0,
            })
            continue

        mean_x0 = float(g["dTmean_nhw"].mean())
        mean_y0 = float(g["dAmp_nhw"].mean())
        mean_x1 = float(g["dTmean_hw"].mean())
        mean_y1 = float(g["dAmp_hw"].mean())

        mean_rmean = mean_x1 - mean_x0
        mean_ramp  = mean_y1 - mean_y0
        theta_mean = float(np.degrees(np.arctan2(mean_ramp, mean_rmean)))

        region_rows.append({
            "region": region,
            "n": int(len(g)),
            "mean_dTmean_nhw": mean_x0,
            "mean_dAmp_nhw": mean_y0,
            "mean_dTmean_hw": mean_x1,
            "mean_dAmp_hw": mean_y1,
            "mean_Rmean": mean_rmean,
            "mean_Ramp": mean_ramp,
            "theta_deg_from_mean_vector": theta_mean,
            "median_distance_D": float(g["D"].median()),
            "mean_distance_D": float(g["D"].mean()),
            "n_positive_dTx_to_positive_dTx": int((g["tx_transition_4class"] == "positive_dTx_to_positive_dTx").sum()),
            "n_positive_dTx_to_negative_dTx": int((g["tx_transition_4class"] == "positive_dTx_to_negative_dTx").sum()),
            "n_negative_dTx_to_negative_dTx": int((g["tx_transition_4class"] == "negative_dTx_to_negative_dTx").sum()),
            "n_negative_dTx_to_positive_dTx": int((g["tx_transition_4class"] == "negative_dTx_to_positive_dTx").sum()),
            "n_positive_dTx_strengthened": int((g["uhi_strength_change"] == "positive_dTx_strengthened").sum()),
            "n_positive_dTx_weakened": int((g["uhi_strength_change"] == "positive_dTx_weakened").sum()),
            "n_negative_dTx_strengthened": int((g["uci_strength_change"] == "negative_dTx_strengthened").sum()),
            "n_negative_dTx_weakened": int((g["uci_strength_change"] == "negative_dTx_weakened").sum()),
            "n_region_switch": int(g["region_switched"].sum()),
            "n_other": int((g["tx_transition_4class"] == "other").sum()),
        })

    region_summary = pd.DataFrame(region_rows)

    region_out = os.path.join(data_dir, "Figure_HW_shift_region_summary.csv")
    region_summary.to_csv(region_out, index=False)

    transition_summary = (
        df.groupby(["hw_region", "tx_transition_4class"], dropna=False)
          .size()
          .reset_index(name="n")
          .sort_values(["hw_region", "tx_transition_4class"])
    )

    trans_out = os.path.join(data_dir, "Figure_HW_shift_regime_transition_summary.csv")
    transition_summary.to_csv(trans_out, index=False)

    # -----------------------------
    # 8. Plot helpers
    # -----------------------------
    def _style_full_box(ax, lw=0.9):
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(lw)

        ax.tick_params(
            top=False,
            right=False,
            direction="out",
            width=0.8,
            length=3.5
        )

    def _add_panel_label(ax, label, subtitle, pad=9):
        ax.set_title(
            subtitle,
            loc="center",
            fontsize=10.0,
            fontweight="bold",
            pad=pad
        )

        # panel label shifted left
        ax.text(
            -0.080, 1.045, label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14.0,
            fontweight="bold"
        )

    def _robust_limits(vals, pct=(1, 99), pad_frac=0.16, min_pad=0.15):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            return (-1, 1)

        lo, hi = np.nanpercentile(vals, pct)

        if np.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5

        pad = max((hi - lo) * pad_frac, min_pad)
        return lo - pad, hi + pad

    color_map = {
        "UHI": COLOR_UHI_LOCAL,
        "UCI": COLOR_UCI_LOCAL,
    }

    # -----------------------------
    # 9. Figure
    # -----------------------------
    out_plot_dir = os.path.join(output_dir, "plots")
    _ensure_dir(out_plot_dir)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):

        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.35), dpi=600)
        plt.subplots_adjust(wspace=0.34)

        ax_a, ax_b = axes

        # ============================================================
        # Panel a: pair-level arrows
        # ============================================================
        for group_name, g in df.groupby("group"):
            color = color_map.get(group_name, "gray")

            ax_a.quiver(
                g["dTmean_nhw"],
                g["dAmp_nhw"],
                g["Rmean"],
                g["Ramp"],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=0.46,
                width=0.0038,
                headwidth=3.3,
                headlength=4.2,
                headaxislength=3.6,
                label=f"{group_name} (n={len(g)})",
                zorder=3
            )

            ax_a.scatter(
                g["dTmean_nhw"],
                g["dAmp_nhw"],
                s=13,
                color=color,
                alpha=0.34,
                edgecolors="none",
                zorder=2
            )

        all_x = np.concatenate([df["dTmean_nhw"].values, df["dTmean_hw"].values])
        all_y = np.concatenate([df["dAmp_nhw"].values, df["dAmp_hw"].values])

        xlim_a = _robust_limits(all_x, pct=(1, 99), pad_frac=0.16)
        ylim_a = _robust_limits(all_y, pct=(1, 99), pad_frac=0.16)

        ax_a.axhline(0, color="#999999", linestyle="--", lw=0.8, zorder=1)
        ax_a.axvline(0, color="#999999", linestyle="--", lw=0.8, zorder=1)

        ax_a.set_xlim(xlim_a)
        ax_a.set_ylim(ylim_a)

        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")

        ax_a.grid(True, lw=0.30, alpha=0.22, zorder=0)
        ax_a.legend(frameon=False, loc="best", handlelength=1.7)

        _style_full_box(ax_a, lw=0.9)

        _add_panel_label(
            ax_a, "a",
            "Pair-level HW–NHW state migration\nin mean–amplitude space"
        )

        # ============================================================
        # Panel b: region-resolved summary
        # ============================================================
        xlim_b = xlim_a
        ylim_b = ylim_a

        ax_b.set_xlim(xlim_b)
        ax_b.set_ylim(ylim_b)

        x_min, x_max = xlim_b
        y_min, y_max = ylim_b

        # Background shading
        if x_max > 0 and y_max > 0:
            ax_b.add_patch(
                Rectangle(
                    (0, 0), x_max, y_max,
                    facecolor="#f7d9d9",
                    edgecolor="none",
                    alpha=0.30,
                    zorder=0
                )
            )

        if x_min < 0 and y_max > 0:
            ax_b.add_patch(
                Rectangle(
                    (x_min, 0), -x_min, y_max,
                    facecolor="#ebebeb",
                    edgecolor="none",
                    alpha=0.24,
                    zorder=0
                )
            )

        if x_min < 0 and y_min < 0:
            ax_b.add_patch(
                Rectangle(
                    (x_min, y_min), -x_min, -y_min,
                    facecolor="#dce8f8",
                    edgecolor="none",
                    alpha=0.30,
                    zorder=0
                )
            )

        # Q4 background split by y = -x
        if x_max > 0 and y_min < 0:
            xq = np.linspace(0, x_max, 300)
            split_y = -xq
            split_y_clip = np.clip(split_y, y_min, 0)

            # Q4a: damped daytime UHI, above y=-x
            ax_b.fill_between(
                xq, split_y_clip, 0,
                color="#f4ddbd",
                alpha=0.36,
                linewidth=0,
                zorder=0
            )

            # Q4b: daytime UCI, below y=-x
            ax_b.fill_between(
                xq, y_min, split_y_clip,
                color="#e8cfa0",
                alpha=0.42,
                linewidth=0,
                zorder=0
            )

            # Separator line only inside Q4
            xe = min(x_max, -y_min)
            ax_b.plot(
                [0, xe], [0, -xe],
                color="#6b6b6b",
                lw=0.95,
                ls="--",
                alpha=0.90,
                zorder=1
            )

            ax_b.text(
                0.98, 0.06,
                r"$-\Delta Amp=\Delta T_{mean}$",
                transform=ax_b.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.7,
                color="#555555",
                path_effects=[
                    path_effects.withStroke(
                        linewidth=2.4,
                        foreground="white",
                        alpha=0.95
                    )
                ],
                zorder=8
            )

        ax_b.axhline(0, color="black", lw=0.85, zorder=1)
        ax_b.axvline(0, color="black", lw=0.85, zorder=1)

        # Background final HW points
        for group_name, g in df.groupby("group"):
            color = color_map.get(group_name, "gray")

            ax_b.scatter(
                g["dTmean_hw"],
                g["dAmp_hw"],
                s=15,
                color=color,
                alpha=0.32,
                edgecolors="none",
                zorder=2
            )

        # Mean arrows by region
        arrow_color = "#333333"

        for _, r in region_summary.iterrows():
            if r["n"] <= 0:
                continue

            if not (
                np.isfinite(r["mean_dTmean_nhw"]) and
                np.isfinite(r["mean_dAmp_nhw"]) and
                np.isfinite(r["mean_dTmean_hw"]) and
                np.isfinite(r["mean_dAmp_hw"])
            ):
                continue

            ax_b.annotate(
                "",
                xy=(r["mean_dTmean_hw"], r["mean_dAmp_hw"]),
                xytext=(r["mean_dTmean_nhw"], r["mean_dAmp_nhw"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=arrow_color,
                    lw=1.35,
                    mutation_scale=10,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=5
            )

            ax_b.scatter(
                r["mean_dTmean_nhw"],
                r["mean_dAmp_nhw"],
                s=24,
                facecolors="white",
                edgecolors=arrow_color,
                linewidth=0.8,
                zorder=6
            )

            ax_b.scatter(
                r["mean_dTmean_hw"],
                r["mean_dAmp_hw"],
                s=34,
                color=arrow_color,
                edgecolors="white",
                linewidth=0.6,
                zorder=7
            )

        halo = [
            path_effects.withStroke(
                linewidth=2.7,
                foreground="white",
                alpha=0.96
            )
        ]

        text_pos = {
            "Q1 +dTmean,+dAmp": (0.58, 0.81),
            "Q2 -dTmean,+dAmp": (0.06, 0.81),
            "Q3 -dTmean,-dAmp": (0.06, 0.17),
            "Q4a damped daytime UHI": (0.58, 0.43),
            "Q4b daytime UCI": (0.58, 0.17),
        }

        short_lab = {
            "Q1 +dTmean,+dAmp": "Q1",
            "Q2 -dTmean,+dAmp": "Q2",
            "Q3 -dTmean,-dAmp": "Q3",
            "Q4a damped daytime UHI": "Q4a",
            "Q4b daytime UCI": "Q4b",
        }

        for _, r in region_summary.iterrows():
            region = r["region"]
            x_txt, y_txt = text_pos[region]

            if r["n"] <= 0:
                txt = f"{short_lab[region]}\n" + r"$n=0$"
            else:
                txt = (
                    f"{short_lab[region]}\n"
                    + rf"$\theta={r['theta_deg_from_mean_vector']:.0f}^\circ$, "
                    + rf"$D_{{50}}={r['median_distance_D']:.2f}$°C" + "\n"
                    + f"UHI→UHI {int(r['n_positive_dTx_to_positive_dTx'])} | UHI→UCI {int(r['n_positive_dTx_to_negative_dTx'])}\n"
                    + f"UCI→UCI {int(r['n_negative_dTx_to_negative_dTx'])} | UCI→UHI {int(r['n_negative_dTx_to_positive_dTx'])}\n"
                    + f"UHI↑ {int(r['n_positive_dTx_strengthened'])} | UHI↓ {int(r['n_positive_dTx_weakened'])}\n"
                    + f"UCI↑ {int(r['n_negative_dTx_strengthened'])} | UCI↓ {int(r['n_negative_dTx_weakened'])}"
                )

            ax_b.text(
                x_txt,
                y_txt,
                txt,
                transform=ax_b.transAxes,
                ha="left",
                va="center",
                fontsize=7.2,
                color="#222222",
                linespacing=1.02,
                path_effects=halo,
                zorder=8
            )

        ax_b.text(
            0.50,
            0.03,
            r"Only Q4 is split by $-\Delta Amp=\Delta T_{mean}$.",
            transform=ax_b.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#555555",
            path_effects=halo,
            zorder=8
        )

        ax_b.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_b.set_ylabel(r"$\Delta Amp$ (°C)")

        ax_b.grid(True, lw=0.25, alpha=0.14, zorder=0)
        _style_full_box(ax_b, lw=0.9)

        _add_panel_label(
            ax_b, "b",
            "Region-resolved summary of\nheatwave-induced migration"
        )

        handles = [
            Line2D(
                [0], [0],
                marker="o",
                ls="",
                color=COLOR_UHI_LOCAL,
                label="UHI",
                markersize=5
            ),
            Line2D(
                [0], [0],
                marker="o",
                ls="",
                color=COLOR_UCI_LOCAL,
                label="UCI",
                markersize=5
            ),
        ]

        ax_b.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.50, -0.14),
            ncol=2,
            handletextpad=0.4,
            columnspacing=1.2
        )

        # -----------------------------
        # Save
        # -----------------------------
        f_png = os.path.join(out_plot_dir, "Figure_HW_shift_arrows_quantified.png")
        f_pdf = os.path.join(out_plot_dir, "Figure_HW_shift_arrows_quantified.pdf")

        fig.savefig(f_png, dpi=600, bbox_inches="tight")
        fig.savefig(f_pdf, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {f_png}")
    print(f"  Saved: {f_pdf}")
    print(f"  Saved pair vectors: {pair_out}")
    print(f"  Saved region summary: {region_out}")
    print(f"  Saved transition summary: {trans_out}")


def plot_figure_hw_shift_arrows(all_df, output_dir):
    """
    Nature-style HW-NHW migration figure.

    Panel a:
        Pair-level NHW -> HW arrows in mean-amplitude space.
        Black arrows = region-mean migration vectors.

    Panel b:
        Use exactly the same matched-pair sample as panel a.
        Separate NHW positive-dTx and negative-dTx diagnostic states.
        These states are not the canonical annual UHI/UCI classification.
        Categories:
            strengthened / reduced / transition
        Compare NHW -> HW only.

    Data source, paired HW/NHW merge, region definition, vector calculation,
    and panel-a logic remain unchanged.
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.lines import Line2D

    COLOR_UHI_LOCAL = globals().get("COLOR_UHI", "#d62728")
    COLOR_UCI_LOCAL = globals().get("COLOR_UCI", "#1f77b4")

    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    def _robust_limits(vals, pct=(1, 99), pad_frac=0.16, min_pad=0.15):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return (-1, 1)
        lo, hi = np.nanpercentile(vals, pct)
        if np.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        pad = max((hi - lo) * pad_frac, min_pad)
        return lo - pad, hi + pad

    def _bootstrap_vector_angle_ci(rmean_vals, ramp_vals, n_boot=1000, alpha=0.05, seed=42):
        x = np.asarray(rmean_vals, dtype=float)
        y = np.asarray(ramp_vals, dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        n = len(x)
        if n == 0:
            return np.nan, np.nan, np.nan

        point = float(np.degrees(np.arctan2(np.mean(y), np.mean(x))))

        if n == 1:
            return point, point, point

        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot, dtype=float)

        for i in range(n_boot):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            boot[i] = np.degrees(np.arctan2(np.mean(y[idx]), np.mean(x[idx])))

        return (
            point,
            float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)),
        )

    def _style_full_box(ax, lw=0.9):
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(lw)
            ax.spines[side].set_color("black")
        ax.tick_params(top=False, right=False, direction="out")

    def _panel_header(fig, ax, label, title, label_dx=-0.032):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + label_dx,
            bbox.y1 + 0.028,
            label,
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="bold"
        )
        fig.text(
            bbox.x0,
            bbox.y1 + 0.028,
            title,
            ha="left",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            linespacing=1.05
        )

    # -----------------------------
    # 1. Required columns
    # -----------------------------
    id_candidates = ["pair_id", "pair_key", "station_pair_id"]
    id_col = None
    for c in id_candidates:
        if c in all_df.columns:
            id_col = c
            break

    if id_col is None:
        print("  [HW shift] Missing pair identifier column, skipping.")
        return

    required_cols = [
        id_col, "group", "period",
        "dTmean", "dAmp1", "dTx"
    ]
    missing = [c for c in required_cols if c not in all_df.columns]
    if missing:
        print(f"  [HW shift] Missing columns: {missing}, skipping.")
        return

    # -----------------------------
    # 2. Paired HW/NHW source
    # -----------------------------
    nhw = all_df[all_df["period"] == "non_heatwave"][
        [id_col, "group", "dTmean", "dAmp1", "dTx"]
    ].rename(columns={
        "dTmean": "dTmean_nhw",
        "dAmp1":  "dAmp_nhw",
        "dTx":    "dTx_nhw",
    })

    hw = all_df[all_df["period"] == "heatwave"][
        [id_col, "dTmean", "dAmp1", "dTx"]
    ].rename(columns={
        "dTmean": "dTmean_hw",
        "dAmp1":  "dAmp_hw",
        "dTx":    "dTx_hw",
    })

    df = pd.merge(nhw, hw, on=id_col, how="inner").dropna(
        subset=[
            "dTmean_nhw", "dAmp_nhw", "dTx_nhw",
            "dTmean_hw",  "dAmp_hw",  "dTx_hw"
        ]
    ).copy()

    if len(df) == 0:
        print("  [HW shift] No paired HW/NHW rows, skipping.")
        return

    # -----------------------------
    # 3. Response vectors
    # -----------------------------
    df["Rmean"] = df["dTmean_hw"] - df["dTmean_nhw"]
    df["Ramp"]  = df["dAmp_hw"]   - df["dAmp_nhw"]
    df["Rtx"]   = df["dTx_hw"]    - df["dTx_nhw"]
    df["theta_deg"] = np.degrees(np.arctan2(df["Ramp"], df["Rmean"]))
    df["D"] = np.hypot(df["Rmean"], df["Ramp"])

    # -----------------------------
    # 4. Response-vector quadrant classification for panel a
    #    Consistent with Figure_Dynamics_Consistent_Final panel b
    # -----------------------------
    def _vector_quadrant(x, y):
        """
        Classify NHW -> HW migration direction by response vector:
          x = Rmean = dTmean_hw - dTmean_nhw
          y = Ramp  = dAmp_hw   - dAmp_nhw

        Q4 is split into:
          Q4a: +Rmean, -Ramp, mean-warming-dominant response
          Q4b: +Rmean, -Ramp, amplitude-damping-dominant response
        """
        if not (np.isfinite(x) and np.isfinite(y)):
            return np.nan
        if x >= 0 and y >= 0:
            return "Q1"
        elif x < 0 and y >= 0:
            return "Q2"
        elif x < 0 and y < 0:
            return "Q3"
        else:
            # Q4: x >= 0 and y < 0.
            # Boundary y = -x separates mean-dominant vs amplitude-dominant response.
            if y >= -x:
                return "Q4a"
            else:
                return "Q4b"

    df["vector_quadrant"] = [
        _vector_quadrant(x, y)
        for x, y in zip(df["Rmean"], df["Ramp"])
    ]

    # Keep this alias only to avoid changing downstream output names too much.
    # From here onward, "hw_region" actually means response-vector quadrant.
    df["hw_region"] = df["vector_quadrant"]

    region_order = ["Q1", "Q2", "Q3", "Q4a", "Q4b"]
    region_summary_rows = []

    for region in region_order:
        g = df[df["vector_quadrant"] == region].copy()

        if len(g) == 0:
            region_summary_rows.append({
                "region": region,
                "n": 0,
                "mean_dTmean_nhw": np.nan,
                "mean_dAmp_nhw": np.nan,
                "mean_dTmean_hw": np.nan,
                "mean_dAmp_hw": np.nan,
                "mean_Rmean": np.nan,
                "mean_Ramp": np.nan,
                "theta_deg_from_mean_vector": np.nan,
                "mean_vector_D": np.nan,
                "mean_distance_D": np.nan,
                "median_distance_D": np.nan,
            })
            continue

        mean_x0 = float(g["dTmean_nhw"].mean())
        mean_y0 = float(g["dAmp_nhw"].mean())
        mean_x1 = float(g["dTmean_hw"].mean())
        mean_y1 = float(g["dAmp_hw"].mean())

        mean_rmean = mean_x1 - mean_x0
        mean_ramp  = mean_y1 - mean_y0

        theta_mean = float(np.degrees(np.arctan2(mean_ramp, mean_rmean)))
        mean_vector_D = float(np.hypot(mean_rmean, mean_ramp))

        region_summary_rows.append({
            "region": region,
            "n": int(len(g)),
            "mean_dTmean_nhw": mean_x0,
            "mean_dAmp_nhw": mean_y0,
            "mean_dTmean_hw": mean_x1,
            "mean_dAmp_hw": mean_y1,
            "mean_Rmean": mean_rmean,
            "mean_Ramp": mean_ramp,
            "theta_deg_from_mean_vector": theta_mean,

            # D used for the mean arrow length in panel a.
            "mean_vector_D": mean_vector_D,

            # Pair-level migration-strength diagnostics.
            "mean_distance_D": float(g["D"].mean()),
            "median_distance_D": float(g["D"].median()),
        })

    region_summary = pd.DataFrame(region_summary_rows)

    # -----------------------------
    # 5. Angle summary diagnostic
    # -----------------------------
    angle_rows = []
    for group_name in ["UHI", "UCI"]:
        g = df[df["group"] == group_name].copy()
        theta_mean, theta_lo, theta_hi = _bootstrap_vector_angle_ci(
            g["Rmean"].values,
            g["Ramp"].values
        )
        angle_rows.append({
            "group": group_name,
            "n": int(len(g)),
            "theta_mean_vector_deg": theta_mean,
            "theta_ci_low": theta_lo,
            "theta_ci_high": theta_hi,
        })

    angle_summary = pd.DataFrame(angle_rows)

    # -----------------------------
    # 6. Panel b summary
    #    same sample as panel a, but split by NHW state
    # -----------------------------
    base_n_all_matched = int(len(df))

    mask_uhi_nhw = df["dTx_nhw"] > 0
    mask_uci_nhw = df["dTx_nhw"] < 0

    # positive-dTx state in NHW
    uhi_strengthened = mask_uhi_nhw & (df["dTx_hw"] > df["dTx_nhw"])
    uhi_reduced      = mask_uhi_nhw & (df["dTx_hw"] > 0) & (df["dTx_hw"] <= df["dTx_nhw"])
    uhi_transition   = mask_uhi_nhw & (df["dTx_hw"] < 0)

    # negative-dTx state in NHW
    uci_strengthened = mask_uci_nhw & (df["dTx_hw"] < df["dTx_nhw"])
    uci_reduced      = mask_uci_nhw & (df["dTx_hw"] < 0) & (df["dTx_hw"] >= df["dTx_nhw"])
    uci_transition   = mask_uci_nhw & (df["dTx_hw"] > 0)

    counted_mask = (
        uhi_strengthened | uhi_reduced | uhi_transition |
        uci_strengthened | uci_reduced | uci_transition
    )

    zero_or_unclassified_n = int(base_n_all_matched - counted_mask.sum())
    unclassified_mask = ~counted_mask

    n_uhi_origin = int(mask_uhi_nhw.sum())
    n_uci_origin = int(mask_uci_nhw.sum())

    transition_summary = pd.DataFrame([
        {
            "origin_state_nhw": "positive_dTx_state",
            "category": "strengthened",
            "base_n_all_matched_pairs": base_n_all_matched,
            "origin_n": n_uhi_origin,
            "n": int(uhi_strengthened.sum()),
            "fraction_within_origin": (
                float(uhi_strengthened.sum()) / n_uhi_origin if n_uhi_origin > 0 else np.nan
            ),
            "fraction_of_all_matched_pairs": (
                float(uhi_strengthened.sum()) / base_n_all_matched if base_n_all_matched > 0 else np.nan
            ),
            "plotted_in_panel_b": True,
        },
        {
            "origin_state_nhw": "positive_dTx_state",
            "category": "reduced",
            "base_n_all_matched_pairs": base_n_all_matched,
            "origin_n": n_uhi_origin,
            "n": int(uhi_reduced.sum()),
            "fraction_within_origin": (
                float(uhi_reduced.sum()) / n_uhi_origin if n_uhi_origin > 0 else np.nan
            ),
            "fraction_of_all_matched_pairs": (
                float(uhi_reduced.sum()) / base_n_all_matched if base_n_all_matched > 0 else np.nan
            ),
            "plotted_in_panel_b": True,
        },
        {
            "origin_state_nhw": "positive_dTx_state",
            "category": "transition",
            "base_n_all_matched_pairs": base_n_all_matched,
            "origin_n": n_uhi_origin,
            "n": int(uhi_transition.sum()),
            "fraction_within_origin": (
                float(uhi_transition.sum()) / n_uhi_origin if n_uhi_origin > 0 else np.nan
            ),
            "fraction_of_all_matched_pairs": (
                float(uhi_transition.sum()) / base_n_all_matched if base_n_all_matched > 0 else np.nan
            ),
            "plotted_in_panel_b": True,
        },
        {
            "origin_state_nhw": "negative_dTx_state",
            "category": "strengthened",
            "base_n_all_matched_pairs": base_n_all_matched,
            "origin_n": n_uci_origin,
            "n": int(uci_strengthened.sum()),
            "fraction_within_origin": (
                float(uci_strengthened.sum()) / n_uci_origin if n_uci_origin > 0 else np.nan
            ),
            "fraction_of_all_matched_pairs": (
                float(uci_strengthened.sum()) / base_n_all_matched if base_n_all_matched > 0 else np.nan
            ),
            "plotted_in_panel_b": True,
        },
        {
            "origin_state_nhw": "negative_dTx_state",
            "category": "reduced",
            "base_n_all_matched_pairs": base_n_all_matched,
            "origin_n": n_uci_origin,
            "n": int(uci_reduced.sum()),
            "fraction_within_origin": (
                float(uci_reduced.sum()) / n_uci_origin if n_uci_origin > 0 else np.nan
            ),
            "fraction_of_all_matched_pairs": (
                float(uci_reduced.sum()) / base_n_all_matched if base_n_all_matched > 0 else np.nan
            ),
            "plotted_in_panel_b": True,
        },
        {
            "origin_state_nhw": "negative_dTx_state",
            "category": "transition",
            "base_n_all_matched_pairs": base_n_all_matched,
            "origin_n": n_uci_origin,
            "n": int(uci_transition.sum()),
            "fraction_within_origin": (
                float(uci_transition.sum()) / n_uci_origin if n_uci_origin > 0 else np.nan
            ),
            "fraction_of_all_matched_pairs": (
                float(uci_transition.sum()) / base_n_all_matched if base_n_all_matched > 0 else np.nan
            ),
            "plotted_in_panel_b": True,
        },
        {
            "origin_state_nhw": "zero_or_unclassified",
            "category": "zero_or_unclassified",
            "base_n_all_matched_pairs": base_n_all_matched,
            "origin_n": np.nan,
            "n": zero_or_unclassified_n,
            "fraction_within_origin": np.nan,
            "fraction_of_all_matched_pairs": (
                float(zero_or_unclassified_n) / base_n_all_matched if base_n_all_matched > 0 else np.nan
            ),
            "plotted_in_panel_b": False,
        },
    ])

    # -----------------------------
    # 7. Save plotting data / diagnostics
    # -----------------------------
    out_data_dir = os.path.join(output_dir, "integrated_fig_data")
    _ensure_dir(out_data_dir)

    pair_out = os.path.join(out_data_dir, "Figure_HW_shift_pair_vectors.csv")
    region_out = os.path.join(out_data_dir, "Figure_HW_shift_region_summary.csv")
    angle_out = os.path.join(out_data_dir, "Figure_HW_shift_angle_summary.csv")
    trans_out = os.path.join(out_data_dir, "Figure_HW_shift_regime_transition_summary.csv")
    unclassified_out = os.path.join(out_data_dir, "Figure_HW_shift_unclassified_or_zero_sign_pairs.csv")

    df.to_csv(pair_out, index=False)
    region_summary.to_csv(region_out, index=False)
    angle_summary.to_csv(angle_out, index=False)
    transition_summary.to_csv(trans_out, index=False)
    df.loc[unclassified_mask].to_csv(unclassified_out, index=False)

    # -----------------------------
    # 8. Figure
    # -----------------------------
    out_plot_dir = os.path.join(output_dir, "plots")
    _ensure_dir(out_plot_dir)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.35), dpi=600)
        plt.subplots_adjust(wspace=0.34)
        ax_a, ax_b = axes

        # ============================================================
        # Panel a: pair-level arrows + regional mean displacement vectors
        # ============================================================

        region_color_map = {
            "Q1":  "#b2182b",   # +Rmean, +Ramp
            "Q2":  "#ef8a62",   # -Rmean, +Ramp
            "Q3":  "#8073ac",   # -Rmean, -Ramp
            "Q4a": "#fdb863",   # +Rmean, -Ramp; mean-warming-dominant
            "Q4b": "#2166ac",   # +Rmean, -Ramp; amplitude-damping-dominant
        }

        region_full_label = {
            "Q1":  r"Q1: $+R_{mean}, +R_{amp}$",
            "Q2":  r"Q2: $-R_{mean}, +R_{amp}$",
            "Q3":  r"Q3: $-R_{mean}, -R_{amp}$",
            "Q4a": r"Q4a: $+R_{mean}, -R_{amp}$, mean-dominant",
            "Q4b": r"Q4b: $+R_{mean}, -R_{amp}$, damping-dominant",
        }

        # Only Q labels are shown beside mean arrows.
        # n, theta and D are placed in the legend.
        panel_a_label_offsets = {
            "Q1":  (7, 7),
            "Q2":  (-16, 7),
            "Q3":  (-18, -14),
            "Q4a": (7, -12),
            "Q4b": (7, -20),
        }

        # ------------------------------------------------------------
        # Raw pair-level arrows as transparent base map
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # Raw pair-level arrows as transparent base map
        # Keep base arrows in original UHI/UCI red-blue colors.
        # Regional mean arrows below still use Q1-Q4b colors.
        # ------------------------------------------------------------
        base_group_color_map = {
            "UHI": COLOR_UHI_LOCAL,
            "UCI": COLOR_UCI_LOCAL,
        }

        for group_name in ["UHI", "UCI"]:
            g = df[df["group"] == group_name].copy()
            if len(g) == 0:
                continue

            base_color = base_group_color_map.get(group_name, "#777777")

            ax_a.quiver(
                g["dTmean_nhw"],
                g["dAmp_nhw"],
                g["Rmean"],
                g["Ramp"],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=base_color,
                alpha=0.40,
                width=0.0034,
                headwidth=3.0,
                headlength=4.0,
                headaxislength=3.5,
                zorder=3
            )

            ax_a.scatter(
                g["dTmean_nhw"],
                g["dAmp_nhw"],
                s=11,
                color=base_color,
                alpha=0.30,
                edgecolors="none",
                zorder=2
            )

        all_x = np.concatenate([df["dTmean_nhw"].values, df["dTmean_hw"].values])
        all_y = np.concatenate([df["dAmp_nhw"].values, df["dAmp_hw"].values])

        xlim_a = _robust_limits(all_x, pct=(1, 99), pad_frac=0.16)
        ylim_a = _robust_limits(all_y, pct=(1, 99), pad_frac=0.16)

        ax_a.axhline(0, color="#9a9a9a", linestyle="--", lw=0.75, zorder=1)
        ax_a.axvline(0, color="#9a9a9a", linestyle="--", lw=0.75, zorder=1)

        # Q4a / Q4b boundary: y = -x.
        # This only visualizes the existing classification boundary and does not change data.
        # Q4a / Q4b response-vector boundary: Ramp = -Rmean.
        # This line is only a visual guide for the Q4 split and does not change data.
        xx = np.linspace(xlim_a[0], xlim_a[1], 200)
        ax_a.plot(
            xx,
            -xx,
            color="#b0b0b0",
            linestyle=":",
            lw=0.75,
            alpha=0.75,
            zorder=1
        )

        ax_a.set_xlim(xlim_a)
        ax_a.set_ylim(ylim_a)

        panel_a_halo = [
            path_effects.withStroke(
                linewidth=2.8,
                foreground="white",
                alpha=0.96
            )
        ]

        legend_handles_a = []
        legend_labels_a = []

        # ------------------------------------------------------------
        # Regional mean displacement vectors
        # ------------------------------------------------------------
        for region in region_order:
            r_sub = region_summary[region_summary["region"] == region]
            if r_sub.empty:
                continue

            r = r_sub.iloc[0]
            if int(r["n"]) <= 0:
                continue

            if not (
                np.isfinite(r["mean_dTmean_nhw"]) and
                np.isfinite(r["mean_dAmp_nhw"]) and
                np.isfinite(r["mean_dTmean_hw"]) and
                np.isfinite(r["mean_dAmp_hw"]) and
                np.isfinite(r["mean_Rmean"]) and
                np.isfinite(r["mean_Ramp"])
            ):
                continue

            region_color = region_color_map.get(region, "#777777")

            theta_vec = float(r["theta_deg_from_mean_vector"])
            mean_vec_D = float(r["mean_vector_D"])

            # Mean vector arrow, same color as the corresponding raw arrows.
            ax_a.annotate(
                "",
                xy=(r["mean_dTmean_hw"], r["mean_dAmp_hw"]),
                xytext=(r["mean_dTmean_nhw"], r["mean_dAmp_nhw"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=region_color,
                    lw=2.05,
                    mutation_scale=12.5,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=7
            )

            # Mean NHW start point.
            ax_a.scatter(
                r["mean_dTmean_nhw"],
                r["mean_dAmp_nhw"],
                s=28,
                facecolors="white",
                edgecolors=region_color,
                linewidth=1.0,
                zorder=8
            )

            # Mean HW end point.
            ax_a.scatter(
                r["mean_dTmean_hw"],
                r["mean_dAmp_hw"],
                s=42,
                color=region_color,
                edgecolors="white",
                linewidth=0.8,
                zorder=9
            )

            # Only mark the Q label beside the mean arrow.
            dx_txt, dy_txt = panel_a_label_offsets.get(region, (7, 7))
            ha_txt = "right" if dx_txt < 0 else "left"
            va_txt = "top" if dy_txt < 0 else "bottom"

            ax_a.annotate(
                region,
                xy=(r["mean_dTmean_hw"], r["mean_dAmp_hw"]),
                xytext=(dx_txt, dy_txt),
                textcoords="offset points",
                ha=ha_txt,
                va=va_txt,
                fontsize=8.0,
                fontweight="bold",
                color=region_color,
                path_effects=panel_a_halo,
                zorder=10
            )

            # Put n, angle and distance into legend, not inside the panel.
            legend_handles_a.append(
                Line2D(
                    [0], [0],
                    color=region_color,
                    lw=2.2,
                    marker="o",
                    markersize=4.8,
                    markerfacecolor=region_color,
                    markeredgecolor="white"
                )
            )
            legend_labels_a.append(
                rf"{region_full_label.get(region, region)}; "
                rf"$n={int(r['n'])}$, "
                rf"$\theta={theta_vec:.0f}^\circ$, "
                rf"$D={mean_vec_D:.2f}$°C"
            )

        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")

        ax_a.grid(True, lw=0.30, alpha=0.22, zorder=0)

        ax_a.legend(
            legend_handles_a,
            legend_labels_a,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.50, -0.20),
            ncol=1,
            handlelength=1.8,
            handletextpad=0.45,
            borderaxespad=0.0,
            fontsize=7.2
        )

        _style_full_box(ax_a, lw=0.9)

        _panel_header(
            fig,
            ax_a,
            "a",
            "Pair-level HW–NHW state migration\nin mean–amplitude space"
        )

        # ============================================================
        # Panel b: same sample as panel a, split by NHW UHI/UCI
        # ============================================================
        panel_b_rows = [
            {
                "label": f"NHW +ΔTx state\n(n={n_uhi_origin})",
                "origin_state": "UHI",
                "base_n": n_uhi_origin,
                "segments": [
                    ("Strengthened", int(uhi_strengthened.sum()), "#b3000d"),
                    ("Reduced",      int(uhi_reduced.sum()),      "#fb6a4a"),
                    ("Transition",   int(uhi_transition.sum()),   "#4292c6"),
                ]
            },
            {
                "label": f"NHW −ΔTx state\n(n={n_uci_origin})",
                "origin_state": "UCI",
                "base_n": n_uci_origin,
                "segments": [
                    ("Strengthened", int(uci_strengthened.sum()), "#08519c"),
                    ("Reduced",      int(uci_reduced.sum()),      "#6baed6"),
                    ("Transition",   int(uci_transition.sum()),   "#ef3b2c"),
                ]
            },
        ]

        y_positions = [1.0, 0.0]
        ax_b.set_xlim(0, 100)
        ax_b.set_ylim(-0.55, 1.55)

        legend_handles = []
        legend_labels = []
        seen_legend = set()

        for row, y in zip(panel_b_rows, y_positions):
            left = 0.0
            row_n = int(row["base_n"])

            for seg_label, seg_n, seg_color in row["segments"]:
                if seg_n <= 0:
                    continue

                width = 0.0 if row_n <= 0 else 100.0 * seg_n / row_n

                ax_b.barh(
                    y,
                    width,
                    left=left,
                    height=0.36,
                    color=seg_color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3
                )

                leg_key = f"{row['origin_state']}_{seg_label}"
                if leg_key not in seen_legend:
                    legend_handles.append(Line2D([0], [0], color=seg_color, lw=6))
                    legend_labels.append(
                        f"{row['origin_state']} {seg_label} (n={seg_n})"
                    )
                    seen_legend.add(leg_key)

                left += width

        ax_b.set_yticks(y_positions)
        ax_b.set_yticklabels([row["label"] for row in panel_b_rows])
        ax_b.set_xlabel("Share within NHW-state subset (%)")
        ax_b.set_xticks([0, 25, 50, 75, 100])
        ax_b.grid(True, axis="x", lw=0.30, alpha=0.18, zorder=0)
        _style_full_box(ax_b, lw=0.9)


        ax_b.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.50, -0.18),
            ncol=3,
            handlelength=1.4,
            handletextpad=0.35,
            columnspacing=0.9
        )

        _panel_header(
            fig,
            ax_b,
            "b",
            "NHW→HW daytime Tx change\nby NHW UHI/UCI state"
        )

        f_png = os.path.join(out_plot_dir, "Figure_HW_shift_arrows_quantified.png")
        f_pdf = os.path.join(out_plot_dir, "Figure_HW_shift_arrows_quantified.pdf")

        fig.savefig(f_png, dpi=600, bbox_inches="tight")
        fig.savefig(f_pdf, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {f_png}")
    print(f"  Saved: {f_pdf}")
    print(f"  Saved pair vectors: {pair_out}")
    print(f"  Saved region summary: {region_out}")
    print(f"  Saved angle summary: {angle_out}")
    print(f"  Saved zero/unclassified sign pairs: {unclassified_out}")

# ─────────────────────────────────────────────────────────────
# [NEW] Combined Figure A: Mechanism core panels (a,b,c)
# ─────────────────────────────────────────────────────────────

def plot_combined_figure_main(all_df, output_dir):
    """
    Combined Figure A:
    vertical NCC-style layout containing mechanism composite panels a/b/c:
      (a) NHW thermodynamic state
      (b) HW thermodynamic state
      (c) Heatwave-induced change with 95% CI and significance

    Data-only: no hard-coded values.
    """
    ensure_dir(os.path.join(output_dir, "combined_plots"))

    id_col = None
    for c in ["pair_id", "station_id", "city_id"]:
        if c in all_df.columns:
            id_col = c
            break

    required = ["period", "group", "dTmean", "dAmp1", "dTx", "dTn"]
    missing = [c for c in required if c not in all_df.columns]
    if id_col is None:
        print("  [Combined A] Missing pair id column: pair_id/station_id/city_id. Skipping.")
        return
    if missing:
        print(f"  [Combined A] Missing columns: {missing}. Skipping.")
        return

    nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
    hw_df  = all_df[all_df["period"] == "heatwave"].copy()

    if len(nhw_df) == 0 or len(hw_df) == 0:
        print("  [Combined A] NHW or HW data is empty. Skipping.")
        return

    nhw_plot = nhw_df[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]].dropna()
    hw_plot  = hw_df[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]].dropna()

    if len(nhw_plot) == 0 or len(hw_plot) == 0:
        print("  [Combined A] Valid NHW/HW plotting data is empty after dropna. Skipping.")
        return

    def _fit_dtx_zero_line(df):
        sub = df[["dTmean", "dAmp1", "dTx"]].dropna()
        if len(sub) < 5:
            return np.nan, np.nan, len(sub)

        X = np.column_stack([
            sub["dTmean"].values,
            sub["dAmp1"].values,
            np.ones(len(sub)),
        ])
        y = sub["dTx"].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c0 = coef

        if np.isclose(b, 0):
            return np.nan, np.nan, len(sub)

        slope = -a / b
        intercept = -c0 / b
        return slope, intercept, len(sub)

    def _stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def _mean_ci(vals):
        vals = pd.Series(vals).dropna().values
        n = len(vals)
        if n < 2:
            return np.nan, np.nan, n
        return float(np.mean(vals)), float(1.96 * stats.sem(vals)), n

    def _paired_p(vals):
        vals = pd.Series(vals).dropna().values
        if len(vals) < 3:
            return np.nan
        return float(stats.ttest_1samp(vals, 0).pvalue)

    def _paired_delta(metric):
        base = nhw_df[[id_col, "group", metric]].rename(columns={metric: f"{metric}_nhw"})
        hw   = hw_df[[id_col, metric]].rename(columns={metric: f"{metric}_hw"})
        merged = pd.merge(base, hw, on=id_col, how="inner").dropna()
        merged[f"{metric}_diff"] = merged[f"{metric}_hw"] - merged[f"{metric}_nhw"]
        return merged

    dtx_all = pd.concat([nhw_plot["dTx"], hw_plot["dTx"]], axis=0).dropna()
    vabs = np.nanpercentile(np.abs(dtx_all.values), 98)
    vabs = max(vabs, 0.1)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    cmap = plt.cm.RdBu_r

    x_all = pd.concat([nhw_plot["dTmean"], hw_plot["dTmean"]]).dropna()
    y_all = pd.concat([nhw_plot["dAmp1"], hw_plot["dAmp1"]]).dropna()
    x_min, x_max = np.nanpercentile(x_all, [1, 99])
    y_min, y_max = np.nanpercentile(y_all, [1, 99])
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig = plt.figure(figsize=(6.2, 13.8), dpi=600)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.08, 1.08, 1.0], hspace=0.42)

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[2, 0])

        def _draw_state_panel(ax, df, title, marker):
            sc = ax.scatter(
                df["dTmean"], df["dAmp1"],
                c=df["dTx"], cmap=cmap, norm=norm,
                marker=marker,
                s=30 if marker == "o" else 34,
                alpha=0.88,
                edgecolors="#333333" if marker == "o" else "black",
                linewidths=0.25 if marker == "o" else 0.45,
                zorder=3,
            )

            ax.axhline(0, color="#c7c7c7", lw=0.8, zorder=0)
            ax.axvline(0, color="#c7c7c7", lw=0.8, zorder=0)

            slope, intercept, n_fit = _fit_dtx_zero_line(df)
            if np.isfinite(slope) and np.isfinite(intercept):
                xx = np.linspace(xlim[0], xlim[1], 240)
                yy = slope * xx + intercept
                ax.plot(
                    xx, yy,
                    color="black",
                    linestyle="--",
                    lw=1.2,
                    zorder=4,
                    label=rf"$\Delta T_x=0$, slope={slope:.2f}",
                )
                ax.legend(frameon=False, loc="upper right", handlelength=2.0)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
            ax.set_ylabel(r"$\Delta Amp$ (°C)")
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(True, lw=0.28, alpha=0.25)

            ax.text(
                0.03, 0.96, f"n = {len(df)}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2),
            )
            add_black_frame(ax, lw=0.8)
            return sc

        sc = _draw_state_panel(ax_a, nhw_plot, "a  NHW thermodynamic state", "o")
        _draw_state_panel(ax_b, hw_plot,  "b  HW thermodynamic state", "^")

        # Compact colorbar inside panel a, upper right
        cax = inset_axes(
            ax_a,
            width="4.5%",      
            height="65%",      
            loc="center right",
            bbox_to_anchor=(-0.08, 0, 1, 1),  
            bbox_transform=ax_a.transAxes,
            borderpad=0,
        )


        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(r"$\Delta T_x$ (°C)", fontsize=7, labelpad=2)
        cbar.ax.tick_params(labelsize=6, length=2)

        metrics_list = [
            ("dTmean", r"$\Delta T_a$"),
            ("dAmp1",  r"$\Delta Amp$"),
            ("dTx",    r"$\Delta T_x$"),
            ("dTn",    r"$\Delta T_n$"),
        ]

        x = np.arange(len(metrics_list))
        width = 0.35
        colors = {"UHI": COLOR_UHI, "UCI": COLOR_UCI}
        summary = {}

        for group_name in ["UHI", "UCI"]:
            means, cis, ns, ps = [], [], [], []
            for metric, _ in metrics_list:
                merged = _paired_delta(metric)
                g = merged[merged["group"] == group_name]
                vals = g[f"{metric}_diff"]
                m, ci, n = _mean_ci(vals)
                p = _paired_p(vals)
                means.append(m)
                cis.append(ci)
                ns.append(n)
                ps.append(p)
            summary[group_name] = {"means": means, "cis": cis, "ns": ns, "ps": ps}

        for offset, group_name in [(-width / 2, "UHI"), (width / 2, "UCI")]:
            n_min = min([n for n in summary[group_name]["ns"] if n is not None], default=0)
            ax_c.bar(
                x + offset,
                summary[group_name]["means"],
                width,
                yerr=summary[group_name]["cis"],
                label=f"{group_name} (n={n_min})",
                color=colors[group_name],
                alpha=0.86,
                capsize=3,
                edgecolor="black",
                linewidth=0.45,
            )

        ax_c.axhline(0, color="black", lw=0.8)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([label for _, label in metrics_list])
        ax_c.set_ylabel("HW$-$NHW difference (°C)")
        ax_c.set_title("c  Heatwave-induced change", loc="left", fontweight="bold")
        ax_c.text(
            0.02, 0.98,
            "Error bars: 95% CI\nStars: one-sample paired t-test vs 0",
            transform=ax_c.transAxes,
            ha="left", va="top",
            fontsize=7.5,
            color="#444444",
        )
        ax_c.legend(frameon=False, loc="lower right")
        ax_c.grid(True, axis="y", lw=0.28, alpha=0.25)
        add_black_frame(ax_c, lw=0.8)

        ymin, ymax = ax_c.get_ylim()
        yr = ymax - ymin
        for i, _ in enumerate(metrics_list):
            for offset, group_name in [(-width / 2, "UHI"), (width / 2, "UCI")]:
                mean_val = summary[group_name]["means"][i]
                ci_val = summary[group_name]["cis"][i]
                p_val = summary[group_name]["ps"][i]
                if not np.isfinite(mean_val):
                    continue
                ci_val = 0 if not np.isfinite(ci_val) else ci_val
                star = _stars(p_val)
                y_base = mean_val + np.sign(mean_val if mean_val != 0 else 1) * ci_val
                y_text = y_base + 0.04 * yr if mean_val >= 0 else y_base - 0.05 * yr
                va = "bottom" if mean_val >= 0 else "top"
                ax_c.text(
                    i + offset, y_text, star,
                    ha="center", va=va,
                    fontsize=8,
                    fontweight="bold",
                )

        fpath = os.path.join(output_dir, "combined_plots", "Figure_Main_Mechanism.png")
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        fig.savefig(fpath.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {fpath}")

# ─────────────────────────────────────────────────────────────
# [NEW] Combined Figure B: arrows + waterfall + hysteresis
# ─────────────────────────────────────────────────────────────

def plot_combined_figure_dynamics2(all_df, output_dir):
    """
    Combined Figure B:
      (a) HW shift arrows
      (b) data-derived mechanism decomposition for UHI/UCI × Tx/Tn
      (c) hysteresis loops for UHI/UCI × HW/NHW

    Data-only: no hard-coded values.
    """
    ensure_dir(os.path.join(output_dir, "combined_plots"))

    id_col = None
    for c in ["pair_id", "station_id", "city_id"]:
        if c in all_df.columns:
            id_col = c
            break

    required = ["period", "group", "dTmean", "dAmp1", "dTx", "dTn"]
    missing = [c for c in required if c not in all_df.columns]
    if id_col is None:
        print("  [Combined B] Missing pair id column: pair_id/station_id/city_id. Skipping.")
        return
    if missing:
        print(f"  [Combined B] Missing columns: {missing}. Skipping.")
        return

    nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
    hw_df  = all_df[all_df["period"] == "heatwave"].copy()

    if len(nhw_df) == 0 or len(hw_df) == 0:
        print("  [Combined B] NHW or HW data is empty. Skipping.")
        return

    paired = pd.merge(
        nhw_df[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
        hw_df[[id_col, "dTmean", "dAmp1", "dTx", "dTn"]],
        on=id_col,
        suffixes=("_nhw", "_hw"),
        how="inner",
    ).dropna()

    if len(paired) == 0:
        print("  [Combined B] No paired NHW-HW rows after merge/dropna. Skipping.")
        return

    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]
    has_diurnal = all(c in all_df.columns for c in u_cols + r_cols)
    if not has_diurnal:
        print("  [Combined B] Missing diurnal columns; panel c will show a missing-data note.")

    def _loop_area(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 4:
            return np.nan
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _phase_lag_hours(x, y):
        if np.all(np.isnan(x)) or np.all(np.isnan(y)):
            return np.nan
        return (int(np.nanargmax(y)) - int(np.nanargmax(x))) % 24

    def _fit_target_decomposition(group_name, target):
        """
        Fit target ~ dTmean + dAmp1 using NHW+HW rows within one group.
        Decompose mean HW-NHW target change into:
          beta_mean * ΔdTmean, beta_amp * ΔdAmp, residual
        """
        merged_g = paired[paired["group"] == group_name].copy()
        if len(merged_g) < 5:
            return None

        fit_nhw = merged_g[[f"dTmean_nhw", f"dAmp1_nhw", f"{target}_nhw"]].rename(
            columns={f"dTmean_nhw": "dTmean", f"dAmp1_nhw": "dAmp1", f"{target}_nhw": target}
        )
        fit_hw = merged_g[[f"dTmean_hw", f"dAmp1_hw", f"{target}_hw"]].rename(
            columns={f"dTmean_hw": "dTmean", f"dAmp1_hw": "dAmp1", f"{target}_hw": target}
        )
        fit_df = pd.concat([fit_nhw, fit_hw], axis=0).dropna()

        if len(fit_df) < 5:
            return None

        X = np.column_stack([
            np.ones(len(fit_df)),
            fit_df["dTmean"].values,
            fit_df["dAmp1"].values,
        ])
        y = fit_df[target].values

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta0, beta_mean, beta_amp = coef
        y_pred = X.dot(coef)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        d_mean = (merged_g["dTmean_hw"] - merged_g["dTmean_nhw"]).mean()
        d_amp  = (merged_g["dAmp1_hw"]  - merged_g["dAmp1_nhw"]).mean()
        obs    = (merged_g[f"{target}_hw"] - merged_g[f"{target}_nhw"]).mean()

        c_mean = beta_mean * d_mean
        c_amp  = beta_amp * d_amp
        resid  = obs - c_mean - c_amp

        return {
            "group": group_name,
            "target": target,
            "n": len(merged_g),
            "beta_mean": beta_mean,
            "beta_amp": beta_amp,
            "r2": r2,
            "mean_contrib": c_mean,
            "amp_contrib": c_amp,
            "residual": resid,
            "observed": obs,
        }

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig = plt.figure(figsize=(13.8, 5.2), dpi=600)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.0], wspace=0.42)

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])

        # -----------------------------
        # (a) Shift arrows
        # -----------------------------
        for group_name, color in [("UHI", COLOR_UHI), ("UCI", COLOR_UCI)]:
            g = paired[paired["group"] == group_name]
            if len(g) == 0:
                continue

            ax_a.quiver(
                g["dTmean_nhw"], g["dAmp1_nhw"],
                g["dTmean_hw"] - g["dTmean_nhw"],
                g["dAmp1_hw"]  - g["dAmp1_nhw"],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=0.3,
                width=0.004,
                headwidth=3.4,
                headlength=4.4,
                headaxislength=3.8,
                label=f"{group_name} (n={len(g)})",
            )
            ax_a.scatter(
                g["dTmean_nhw"], g["dAmp1_nhw"],
                s=12, color=color, alpha=0.35, edgecolors="none",
            )

        all_x = np.concatenate([paired["dTmean_nhw"].values, paired["dTmean_hw"].values])
        all_y = np.concatenate([paired["dAmp1_nhw"].values, paired["dAmp1_hw"].values])
        x_min, x_max = np.nanpercentile(all_x, [1, 99])
        y_min, y_max = np.nanpercentile(all_y, [1, 99])
        x_pad = (x_max - x_min) * 0.16
        y_pad = (y_max - y_min) * 0.16

        ax_a.axhline(0, color="#999999", linestyle="--", lw=0.8)
        ax_a.axvline(0, color="#999999", linestyle="--", lw=0.8)
        ax_a.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_a.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")
        ax_a.set_title("a  HW-induced state migration", loc="left", fontweight="bold")
        ax_a.legend(frameon=False, loc="best")
        ax_a.grid(True, lw=0.28, alpha=0.25)
        add_black_frame(ax_a, lw=0.8)

        # -----------------------------
        # (b) UHI/UCI Tx/Tn decomposition
        # -----------------------------
        rows = []
        for group_name in ["UHI", "UCI"]:
            for target in ["dTx", "dTn"]:
                res = _fit_target_decomposition(group_name, target)
                if res is None:
                    print(f"  [Combined B] Insufficient data for panel b: {group_name} {target}.")
                else:
                    rows.append(res)

        if len(rows) == 0:
            ax_b.text(
                0.5, 0.5,
                "Insufficient paired data\nfor Tx/Tn decomposition",
                ha="center", va="center",
                transform=ax_b.transAxes,
                color="#777777",
            )
            ax_b.axis("off")
        else:
            labels = []
            mean_vals, amp_vals, residual_vals, obs_vals = [], [], [], []
            for r in rows:
                labels.append(f"{r['group']}\n{r['target'].replace('dT', 'ΔT')}")
                mean_vals.append(r["mean_contrib"])
                amp_vals.append(r["amp_contrib"])
                residual_vals.append(r["residual"])
                obs_vals.append(r["observed"])

            x = np.arange(len(rows))
            width = 0.18

            ax_b.bar(x - 1.5 * width, mean_vals, width, color="#d62728",
                     alpha=0.86, edgecolor="black", linewidth=0.35,
                     label=r"$\Delta T_{mean}$ contribution")
            ax_b.bar(x - 0.5 * width, amp_vals, width, color="#1f77b4",
                     alpha=0.86, edgecolor="black", linewidth=0.35,
                     label=r"$\Delta Amp$ contribution")
            ax_b.bar(x + 0.5 * width, residual_vals, width, color="#7f7f7f",
                     alpha=0.82, edgecolor="black", linewidth=0.35,
                     label="Residual")
            ax_b.scatter(x + 1.5 * width, obs_vals, color="black",
                         s=28, marker="D", zorder=5, label="Observed HW$-$NHW")

            ax_b.axhline(0, color="black", lw=0.8)
            ax_b.set_xticks(x)
            ax_b.set_xticklabels(labels)
            ax_b.set_ylabel("Contribution to HW$-$NHW change (°C)")
            ax_b.set_title("b  Group-specific Tx/Tn mechanism decomposition",
                           loc="left", fontweight="bold")
            ax_b.legend(frameon=False, loc="best", fontsize=7)
            ax_b.grid(True, axis="y", lw=0.28, alpha=0.25)
            add_black_frame(ax_b, lw=0.8)

            txt = []
            for r in rows:
                target_label = r["target"].replace("dT", "ΔT")
                txt.append(f"{r['group']} {target_label}: R²={r['r2']:.2f}, n={r['n']}")
            ax_b.text(
                0.02, 0.98,
                "\n".join(txt),
                transform=ax_b.transAxes,
                ha="left", va="top",
                fontsize=6.8,
                bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=2),
            )

        # -----------------------------
        # (c) Hysteresis: UHI/UCI × NHW/HW
        # -----------------------------
        if not has_diurnal:
            ax_c.text(
                0.5, 0.5,
                "Diurnal columns missing\nCannot plot hysteresis",
                ha="center", va="center",
                transform=ax_c.transAxes,
                color="#777777",
            )
            ax_c.axis("off")
        else:
            style_map = {
                ("UHI", "non_heatwave"): (COLOR_UHI, "--", "UHI NHW"),
                ("UHI", "heatwave"):     (COLOR_UHI, "-",  "UHI HW"),
                ("UCI", "non_heatwave"): (COLOR_UCI, "--", "UCI NHW"),
                ("UCI", "heatwave"):     (COLOR_UCI, "-",  "UCI HW"),
            }

            for (group_name, period_name), (color, ls, label0) in style_map.items():
                sub = all_df[(all_df["group"] == group_name) & (all_df["period"] == period_name)]
                if len(sub) == 0:
                    print(f"  [Combined B] No data for panel c: {group_name}, {period_name}.")
                    continue

                u = sub[u_cols].values.astype(float)
                r = sub[r_cols].values.astype(float)

                x_curve = np.nanmean(r, axis=0)
                y_curve = np.nanmean(u - r, axis=0)

                area = _loop_area(x_curve, y_curve)
                lag = _phase_lag_hours(x_curve, y_curve)

                ax_c.plot(
                    x_curve, y_curve,
                    color=color,
                    linestyle=ls,
                    lw=1.8,
                    alpha=0.88,
                    label=f"{label0}: area={area:.2f}, lag={lag:.0f} h",
                )

                for h in [0, 6, 12, 18]:
                    ax_c.scatter(x_curve[h], y_curve[h], s=18, color=color, zorder=4)
                    ax_c.text(
                        x_curve[h], y_curve[h], f"{h:02d}",
                        fontsize=6.5, color=color,
                        ha="left", va="bottom",
                    )

            ax_c.set_xlabel("Rural background temperature (°C)")
            ax_c.set_ylabel("Urban$-$Rural $\Delta T$ (°C)")
            ax_c.set_title("c  Thermal hysteresis by group and period",
                           loc="left", fontweight="bold")
            ax_c.legend(frameon=False, loc="best", fontsize=6.7)
            ax_c.grid(True, lw=0.28, alpha=0.25)
            add_black_frame(ax_c, lw=0.8)

        fpath = os.path.join(output_dir, "combined_plots", "Figure_Dynamics.png")
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        fig.savefig(fpath.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {fpath}")

def plot_combined_figure_dynamics(all_df, output_dir):
    """
    NCC-style Figure 3 / Combined Figure B:
      (a) Heatwave-induced state migration with density reduction and physical constraint
      (b) ΔTn decomposition: ΔTmean, -ΔAmp, residual, observed
      (c) ΔTx–ΔTn compensation scatter with regression and 95% CI
      (d) Thermal hysteresis / phase lag for UHI/UCI × HW/NHW

    Data-only: no hard-coded scientific values.
    """
    ensure_dir(os.path.join(output_dir, "combined_plots"))

    id_col = None
    for c in ["pair_id", "station_id", "city_id"]:
        if c in all_df.columns:
            id_col = c
            break

    required = ["period", "group", "dTmean", "dAmp1", "dTx", "dTn"]
    missing = [c for c in required if c not in all_df.columns]

    if id_col is None:
        print("  [Combined B] Missing pair id column: pair_id/station_id/city_id. Skipping.")
        return

    if missing:
        print(f"  [Combined B] Missing columns: {missing}. Skipping.")
        return

    nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
    hw_df  = all_df[all_df["period"] == "heatwave"].copy()

    if len(nhw_df) == 0 or len(hw_df) == 0:
        print("  [Combined B] NHW or HW data is empty. Skipping.")
        return

    paired = pd.merge(
        nhw_df[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
        hw_df[[id_col, "dTmean", "dAmp1", "dTx", "dTn"]],
        on=id_col,
        suffixes=("_nhw", "_hw"),
        how="inner",
    ).dropna()

    if len(paired) == 0:
        print("  [Combined B] No paired NHW-HW rows after merge/dropna. Skipping.")
        return

    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]
    has_diurnal = all(c in all_df.columns for c in u_cols + r_cols)

    # ============================================================
    # 核心修改：高精度相位滞后计算 (FFT 基波法)
    # ============================================================
    def _get_precise_peak_time(signal):
        """利用前两个谐波重构信号并寻找高精度峰值时间"""
        vals = np.asarray(signal, dtype=float)
        if np.all(np.isnan(vals)): return np.nan
        
        # 填充NaN并去中心化
        vals = np.where(np.isfinite(vals), vals, np.nanmean(vals))
        sig_detrend = vals - np.mean(vals)
        
        # FFT 变换
        fft_vals = np.fft.fft(sig_detrend)
        
        # 在 0.01 小时精度的网格上进行重构 (2400个点)
        t_fine = np.linspace(0, 24, 2400, endpoint=False)
        
        # 重构信号 = 第一谐波 (k=1) + 第二谐波 (k=2)
        # 公式: A*cos(wt - phi) -> 在复数域即为 C[k]*exp(i*w*k*t) 的实部
        recon = np.zeros_like(t_fine)
        for k in [1, 2]:
            recon += np.real(fft_vals[k] * np.exp(1j * 2 * np.pi * k * t_fine / 24))
        
        # 返回重构曲线最大值对应的时间点
        return t_fine[np.argmax(recon)]

    def _phase_lag_hours(x_curve, y_curve):
        """
        计算 UHI 峰值相对于乡村背景温度峰值的滞后时间
        """
        t_peak_x = _get_precise_peak_time(x_curve) # 乡村背景达峰时间
        t_peak_y = _get_precise_peak_time(y_curve) # UHI 强度达峰时间
        if pd.isna(t_peak_x) or pd.isna(t_peak_y): return np.nan
        
        # 计算差值并取模，确保结果在 [0, 24) 范围内
        return (t_peak_y - t_peak_x) % 24

    if not has_diurnal:
        print("  [Combined B] Missing diurnal columns; panel d will show a missing-data note.")

    def _stars_from_p(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def _loop_area(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 4:
            return np.nan
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )

    def _fit_line_ci(x, y, x_grid):
        """
        Simple OLS fit y = a*x + b with 95% CI for mean prediction.
        Returns slope, intercept, p, y_hat, ci.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 5 or np.isclose(np.nanstd(x), 0):
            return np.nan, np.nan, np.nan, np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)

        lr = stats.linregress(x, y)
        slope, intercept, p_val = lr.slope, lr.intercept, lr.pvalue

        y_fit = intercept + slope * x_grid

        y_pred = intercept + slope * x
        resid = y - y_pred
        n = len(x)
        s_err = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
        x_mean = np.mean(x)
        ssx = np.sum((x - x_mean) ** 2)

        if ssx <= 0:
            ci = np.full_like(x_grid, np.nan)
        else:
            t_val = stats.t.ppf(0.975, df=max(n - 2, 1))
            ci = t_val * s_err * np.sqrt(1 / n + (x_grid - x_mean) ** 2 / ssx)

        return slope, intercept, p_val, y_fit, ci

    def _decompose_dTn(group_name):
        """
        Physics-based decomposition:
          ΔdTn ≈ ΔdTmean - ΔdAmp
        where Δ means HW - NHW.

        Components:
          contribution from ΔTmean = mean(HW-NHW dTmean)
          contribution from amplitude damping = -mean(HW-NHW dAmp1)
          residual = observed ΔdTn - contribution_mean - contribution_amp
        """
        g = paired[paired["group"] == group_name].copy()
        if len(g) < 3:
            return None

        dd_tmean = g["dTmean_hw"] - g["dTmean_nhw"]
        dd_amp   = g["dAmp1_hw"]  - g["dAmp1_nhw"]
        dd_tn    = g["dTn_hw"]    - g["dTn_nhw"]

        c_mean = float(np.nanmean(dd_tmean))
        c_amp  = float(np.nanmean(-dd_amp))
        obs    = float(np.nanmean(dd_tn))
        resid  = obs - c_mean - c_amp

        p_obs = stats.ttest_1samp(dd_tn.dropna(), 0).pvalue if len(dd_tn.dropna()) >= 3 else np.nan

        return {
            "group": group_name,
            "n": len(g),
            "c_mean": c_mean,
            "c_amp": c_amp,
            "resid": resid,
            "obs": obs,
            "p_obs": p_obs,
        }

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):

        fig = plt.figure(figsize=(12.8, 8.4), dpi=600)
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[1.0, 1.0],
            height_ratios=[1.0, 1.0],
            wspace=0.34,
            hspace=0.38
        )

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        # ============================================================
        # (a) State migration: reduced-density mean vectors + constraint
        # ============================================================
        all_x = np.concatenate([paired["dTmean_nhw"].values, paired["dTmean_hw"].values])
        all_y = np.concatenate([paired["dAmp1_nhw"].values, paired["dAmp1_hw"].values])

        x_min, x_max = np.nanpercentile(all_x, [1, 99])
        y_min, y_max = np.nanpercentile(all_y, [1, 99])
        x_pad = (x_max - x_min) * 0.16
        y_pad = (y_max - y_min) * 0.16

        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        def _binned_vectors(g, nx=10, ny=10, min_count=2):
            """
            Density + mean-flow version:
            one mean migration vector per occupied bin.
            """
            g = g.copy()
            g["dx"] = g["dTmean_hw"] - g["dTmean_nhw"]
            g["dy"] = g["dAmp1_hw"]  - g["dAmp1_nhw"]
            g["mag"] = np.sqrt(g["dx"] ** 2 + g["dy"] ** 2)

            g = g[
                np.isfinite(g["dTmean_nhw"]) &
                np.isfinite(g["dAmp1_nhw"]) &
                np.isfinite(g["dx"]) &
                np.isfinite(g["dy"])
            ].copy()

            if len(g) == 0:
                return pd.DataFrame()

            x_bins = np.linspace(xlim[0], xlim[1], nx + 1)
            y_bins = np.linspace(ylim[0], ylim[1], ny + 1)

            g["xb"] = pd.cut(g["dTmean_nhw"], bins=x_bins, labels=False, include_lowest=True)
            g["yb"] = pd.cut(g["dAmp1_nhw"],  bins=y_bins, labels=False, include_lowest=True)

            out = (
                g.dropna(subset=["xb", "yb"])
                 .groupby(["xb", "yb"], as_index=False)
                 .agg(
                     x0=("dTmean_nhw", "mean"),
                     y0=("dAmp1_nhw", "mean"),
                     dx=("dx", "mean"),
                     dy=("dy", "mean"),
                     n=("dx", "size"),
                     mag=("mag", "mean"),
                 )
            )

            out = out[out["n"] >= min_count].copy()
            return out

        # -----------------------------
        # density: NHW state distribution
        # -----------------------------
        dens_df = paired[["dTmean_nhw", "dAmp1_nhw"]].dropna().copy()

        hb = ax_a.hexbin(
            dens_df["dTmean_nhw"],
            dens_df["dAmp1_nhw"],
            gridsize=28,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            mincnt=1,
            cmap="Greys",
            alpha=0.42,
            linewidths=0,
            zorder=0,
        )

        # -----------------------------
        # mean flow: binned HW-NHW migration vectors
        # -----------------------------
        for group_name, color in [("UHI", COLOR_UHI), ("UCI", COLOR_UCI)]:
            g = paired[paired["group"] == group_name].copy()
            if len(g) == 0:
                continue

            vec = _binned_vectors(g, nx=11, ny=11, min_count=2)

            if len(vec) == 0:
                print(f"  [Combined B] Panel a: no binned mean-flow vectors for {group_name}.")
                continue

            ax_a.quiver(
                vec["x0"], vec["y0"],
                vec["dx"], vec["dy"],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=0.78,
                width=0.0042,
                headwidth=3.2,
                headlength=4.2,
                headaxislength=3.6,
                label=f"{group_name} mean flow",
                zorder=3,
            )

        # -----------------------------
        # constraint: one clean ΔTx = 0 isoline
        # ΔTx = ΔTmean + ΔAmp = 0  ->  ΔAmp = -ΔTmean
        # -----------------------------
        x_line = np.linspace(xlim[0], xlim[1], 200)
        ax_a.plot(
            x_line,
            -x_line,
            linestyle="--",
            color="black",
            lw=1.0,
            alpha=0.72,
            zorder=2,
            label=r"$\Delta T_x=0$ isoline",
        )

        ax_a.text(
            0.03,
            0.97,
            "Density: NHW state\nArrows: mean HW−NHW flow",
            transform=ax_a.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2),
        )

        # optional density colorbar; keep small and unobtrusive
        cax_den = inset_axes(
            ax_a,
            width="3%",
            height="38%",
            loc="lower left",
            bbox_to_anchor=(0.035, 0.08, 1, 1),
            bbox_transform=ax_a.transAxes,
            borderpad=0,
        )
        cbar_den = plt.colorbar(hb, cax=cax_den)
        cbar_den.set_label("Density", fontsize=6.5)
        cbar_den.ax.tick_params(labelsize=6)

        ax_a.axhline(0, color="#999999", linestyle=":", lw=0.8, zorder=1)
        ax_a.axvline(0, color="#999999", linestyle=":", lw=0.8, zorder=1)
        ax_a.set_xlim(xlim)
        ax_a.set_ylim(ylim)
        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")
        ax_a.set_title(
            "a  State migration follows a constrained pathway",
            loc="left",
            fontweight="bold",
        )
        ax_a.legend(frameon=False, loc="best", fontsize=6.7)
        ax_a.grid(True, lw=0.25, alpha=0.16)
        add_black_frame(ax_a, lw=0.8)


        # ============================================================
        # (b) ΔTn decomposition only
        # ============================================================
        decomp_rows = []
        for group_name in ["UHI", "UCI"]:
            res = _decompose_dTn(group_name)
            if res is None:
                print(f"  [Combined B] Panel b: insufficient data for {group_name} ΔTn decomposition.")
            else:
                decomp_rows.append(res)

        # ── 新增：导出 Panel B 数据到 TXT ──
        if len(decomp_rows) > 0:
            txt_lines = []
            txt_lines.append("Figure Dynamics - Panel B: ΔTn Decomposition Data (HW minus NHW)")
            txt_lines.append("="*85)
            txt_lines.append(f"{'Group':8s} | {'n':5s} | {'ΔTmean_cont':12s} | {'-ΔAmp_cont':12s} | {'Residual':10s} | {'Observed':10s} | {'p-value':9s}")
            txt_lines.append("-" * 85)
            for r in decomp_rows:
                txt_lines.append(
                    f"{r['group']:8s} | {r['n']:5d} | {r['c_mean']:12.4f} | {r['c_amp']:12.4f} | "
                    f"{r['resid']:10.4f} | {r['obs']:10.4f} | {r['p_obs']:9.4e}"
                )
            txt_lines.append("-" * 85)
            txt_lines.append("Note: ΔTn ≈ ΔTmean - ΔAmp + Residual. Contributions represent HW-NHW differences.")
            
            txt_path = os.path.join(output_dir, "combined_plots", "Figure_Dynamics_Panel_B_data.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines))
            print(f"  [Data Export] Saved Panel B decomposition data to: {txt_path}")

        if len(decomp_rows) == 0:
            ax_b.text(
                0.5, 0.5,
                "Insufficient paired data\nfor ΔTn decomposition",
                ha="center",
                va="center",
                transform=ax_b.transAxes,
                color="#777777",
            )
            ax_b.axis("off")
        else:
            x = np.arange(len(decomp_rows))
            width = 0.48

            vals_mean = np.array([r["c_mean"] for r in decomp_rows])
            vals_amp  = np.array([r["c_amp"]  for r in decomp_rows])
            vals_res  = np.array([r["resid"]  for r in decomp_rows])
            vals_obs  = np.array([r["obs"]    for r in decomp_rows])

            labels = [f"{r['group']}\n(n={r['n']})" for r in decomp_rows]

            bottom = np.zeros(len(decomp_rows))

            ax_b.bar(
                x,
                vals_mean,
                width=width,
                bottom=bottom,
                color="#b22222",
                alpha=0.88,
                edgecolor="black",
                linewidth=0.4,
                label=r"$\Delta T_{mean}$ contribution",
            )
            bottom = bottom + vals_mean

            ax_b.bar(
                x,
                vals_amp,
                width=width,
                bottom=bottom,
                color="#3a6ea5",
                alpha=0.88,
                edgecolor="black",
                linewidth=0.4,
                label=r"$-\Delta Amp$ contribution",
            )
            bottom = bottom + vals_amp

            ax_b.bar(
                x,
                vals_res,
                width=width,
                bottom=bottom,
                color="#8a8a8a",
                alpha=0.76,
                edgecolor="black",
                linewidth=0.4,
                label="Residual",
            )

            ax_b.scatter(
                x,
                vals_obs,
                s=34,
                marker="D",
                color="black",
                zorder=5,
                label=r"Observed $\Delta T_n$",
            )

            for i, r in enumerate(decomp_rows):
                star = _stars_from_p(r["p_obs"])
                y_txt = vals_obs[i] + 0.06 * (np.nanmax(vals_obs) - np.nanmin(vals_obs) + 1e-6)
                ax_b.text(
                    i,
                    y_txt,
                    star,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

            ax_b.axhline(0, color="black", lw=0.8)
            ax_b.set_xticks(x)
            ax_b.set_xticklabels(labels)
            ax_b.set_ylabel(r"HW$-$NHW change in $\Delta T_n$ (°C)")
            ax_b.set_title(
                "b  Nighttime warming is amplified by amplitude damping",
                loc="left",
                fontweight="bold",
            )
            ax_b.legend(frameon=False, loc="best", fontsize=7)
            ax_b.grid(True, axis="y", lw=0.25, alpha=0.20)
            add_black_frame(ax_b, lw=0.8)

            ax_b.text(
                0.02, 0.97,
                r"$\Delta T_n \approx \Delta T_{mean} - \Delta Amp$",
                transform=ax_b.transAxes,
                ha="left",
                va="top",
                fontsize=7.4,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2),
            )
        # ============================================================
        # (c) Background-removed Tx–Tn linkage controlled by ΔAmp
        # ============================================================
        comp_df = paired.copy()

        # Heatwave-induced changes: HW - NHW
        comp_df["ddTx"]    = comp_df["dTx_hw"]    - comp_df["dTx_nhw"]
        comp_df["ddTn"]    = comp_df["dTn_hw"]    - comp_df["dTn_nhw"]
        comp_df["ddTmean"] = comp_df["dTmean_hw"] - comp_df["dTmean_nhw"]
        comp_df["ddAmp"]   = comp_df["dAmp1_hw"]  - comp_df["dAmp1_nhw"]

        # Remove common background warming:
        #   ddTx_res ≈ +ddAmp
        #   ddTn_res ≈ -ddAmp
        comp_df["ddTx_res"] = comp_df["ddTx"] - comp_df["ddTmean"]
        comp_df["ddTn_res"] = comp_df["ddTn"] - comp_df["ddTmean"]

        plot_df = comp_df[[
            "group", "ddTx_res", "ddTn_res", "ddAmp"
        ]].dropna().copy()

        if len(plot_df) < 5:
            ax_c.text(
                0.5, 0.5,
                "Insufficient paired data\nfor ΔAmp-controlled linkage",
                ha="center", va="center",
                transform=ax_c.transAxes,
                color="#777777",
            )
            ax_c.axis("off")
        else:
            # Symmetric color scale for ΔAmp
            amp_abs = np.nanpercentile(np.abs(plot_df["ddAmp"].values), 98)
            amp_abs = max(amp_abs, 0.1)
            norm_amp = TwoSlopeNorm(vmin=-amp_abs, vcenter=0, vmax=amp_abs)
            cmap_amp = plt.cm.RdBu_r

            marker_map = {"UHI": "o", "UCI": "^"}

            for group_name in ["UHI", "UCI"]:
                g = plot_df[plot_df["group"] == group_name]
                if len(g) == 0:
                    continue

                ax_c.scatter(
                    g["ddTx_res"],
                    g["ddTn_res"],
                    c=g["ddAmp"],
                    cmap=cmap_amp,
                    norm=norm_amp,
                    s=22 if group_name == "UHI" else 28,
                    marker=marker_map[group_name],
                    alpha=0.72,
                    edgecolors="black",
                    linewidths=0.25,
                    label=f"{group_name} pairs",
                    zorder=2,
                )

            # Overall regression: this tests Tx–Tn linkage after removing common warming
            x_all = plot_df["ddTx_res"].values
            y_all = plot_df["ddTn_res"].values

            x_grid = np.linspace(
                np.nanpercentile(x_all, 2),
                np.nanpercentile(x_all, 98),
                160
            )

            slope, intercept, p_val, y_fit, ci = _fit_line_ci(
                x_all,
                y_all,
                x_grid
            )

            ax_c.plot(
                x_grid,
                y_fit,
                color="black",
                lw=1.6,
                label=rf"fit, slope={slope:.2f}{_stars_from_p(p_val)}",
                zorder=4,
            )

            ax_c.fill_between(
                x_grid,
                y_fit - ci,
                y_fit + ci,
                color="black",
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )

            # Theoretical compensation line: y = -x
            ax_c.plot(
                x_grid,
                -x_grid,
                linestyle="--",
                color="#555555",
                lw=1.0,
                alpha=0.75,
                label="theoretical slope = -1",
                zorder=3,
            )

            ax_c.axhline(0, color="#777777", linestyle="--", lw=0.8)
            ax_c.axvline(0, color="#777777", linestyle="--", lw=0.8)

            ax_c.set_xlabel(
                r"HW$-$NHW daytime response after removing $\Delta T_{mean}$ (°C)"
            )
            ax_c.set_ylabel(
                r"HW$-$NHW nighttime response after removing $\Delta T_{mean}$ (°C)"
            )

            ax_c.set_title(
                "c  Amplitude damping links daytime and nighttime responses",
                loc="left",
                fontweight="bold",
            )

            ax_c.text(
                0.03,
                0.97,
                r"Colour shows HW$-$NHW $\Delta Amp$;"
                "\ncommon warming removed",
                transform=ax_c.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2),
            )

            # Colorbar for ΔAmp
            cax = inset_axes(
                ax_c,
                width="3.2%",
                height="45%",
                loc="lower left",
                bbox_to_anchor=(0.04, 0.07, 1, 1),
                bbox_transform=ax_c.transAxes,
                borderpad=0,
            )
            sm = plt.cm.ScalarMappable(cmap=cmap_amp, norm=norm_amp)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label(r"HW$-$NHW $\Delta Amp$ (°C)", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            ax_c.legend(frameon=False, loc="best", fontsize=6.4)
            ax_c.grid(True, lw=0.25, alpha=0.20)
            add_black_frame(ax_c, lw=0.8)

        # ============================================================
        # (d) Hysteresis: UHI/UCI × NHW/HW
        # ============================================================
        if not has_diurnal:
            ax_d.text(
                0.5, 0.5,
                "Diurnal columns missing\nCannot plot hysteresis",
                ha="center",
                va="center",
                transform=ax_d.transAxes,
                color="#777777",
            )
            ax_d.axis("off")
        else:
            style_map = {
                ("UHI", "non_heatwave"): (COLOR_UHI, "--", "UHI NHW"),
                ("UHI", "heatwave"):     (COLOR_UHI, "-",  "UHI HW"),
                ("UCI", "non_heatwave"): (COLOR_UCI, "--", "UCI NHW"),
                ("UCI", "heatwave"):     (COLOR_UCI, "-",  "UCI HW"),
            }

            for (group_name, period_name), (color, ls, label0) in style_map.items():
                sub = all_df[(all_df["group"] == group_name) & (all_df["period"] == period_name)]
                if len(sub) == 0: continue
                
                u_mat = sub[u_cols].values.astype(float)
                r_mat = sub[r_cols].values.astype(float)
                
                # 计算该组的平均 24 小时曲线
                x_curve = np.nanmean(r_mat, axis=0)
                y_curve = np.nanmean(u_mat - r_mat, axis=0)
                
                # 计算物理量
                area = _loop_area(x_curve, y_curve)
                lag  = _phase_lag_hours(x_curve, y_curve) # 调用双谐波高精度算法
                
                ax_d.plot(
                    x_curve,
                    y_curve,
                    color=color,
                    linestyle=ls,
                    lw=1.55 if period_name == "heatwave" else 1.25,
                    alpha=0.90 if period_name == "heatwave" else 0.68,
                    label=f"{label0}: area={area:.2f}, lag={lag:.2f} h",
                    zorder=3 if period_name == "heatwave" else 2,
                )

                # time direction arrows
                for i_arrow in [5, 11, 17]:
                    ax_d.annotate(
                        "",
                        xy=(x_curve[(i_arrow + 1) % 24], y_curve[(i_arrow + 1) % 24]),
                        xytext=(x_curve[i_arrow], y_curve[i_arrow]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            lw=0.9,
                            alpha=0.85,
                        ),
                    )

                # time labels
                for h in [0, 6, 12, 18]:
                    ax_d.scatter(
                        x_curve[h],
                        y_curve[h],
                        s=17,
                        color=color,
                        zorder=4,
                    )
                    ax_d.text(
                        x_curve[h],
                        y_curve[h],
                        f"{h:02d}",
                        fontsize=6.4,
                        color=color,
                        ha="left",
                        va="bottom",
                    )

            ax_d.set_xlabel("Rural background temperature (°C)")
            ax_d.set_ylabel("Urban$-$Rural $\Delta T$ (°C)")
            ax_d.set_title(
                "d  Hysteresis indicates delayed heat release",
                loc="left",
                fontweight="bold",
            )
            ax_d.legend(frameon=False, loc="best", fontsize=6.4)
            ax_d.grid(True, lw=0.25, alpha=0.20)
            add_black_frame(ax_d, lw=0.8)

        fpath = os.path.join(output_dir, "combined_plots", "Figure_Dynamics.png")
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        fig.savefig(fpath.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {fpath}")

def plot_supplement_single_period_Tx_Tn_relationship(all_df, output_dir):
    """
    Supplementary figure:
    single-period ΔTx vs ΔTn relationships for NHW and HW periods.

    Purpose:
    This is not the main heatwave-induced mechanism test.
    It shows whether the ΔTx–ΔTn relationship is consistent within each period.
    """
    ensure_dir(os.path.join(output_dir, "combined_plots"))

    required = ["period", "group", "dTmean", "dTx", "dTn"]
    missing = [c for c in required if c not in all_df.columns]
    if missing:
        print(f"  [Supp Tx-Tn] Missing columns: {missing}. Skipping.")
        return

    df = all_df[all_df["period"].isin(["non_heatwave", "heatwave"])].copy()

    if len(df) == 0:
        print("  [Supp Tx-Tn] No NHW/HW data found. Skipping.")
        return

    # Remove common background within each period:
    #   Tx_res = dTx - dTmean
    #   Tn_res = dTn - dTmean
    df["Tx_res"] = df["dTx"] - df["dTmean"]
    df["Tn_res"] = df["dTn"] - df["dTmean"]

    def _stars_from_p_local(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def _fit_line_ci_local(x, y, x_grid):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 5 or np.isclose(np.nanstd(x), 0):
            return np.nan, np.nan, np.nan, np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)

        lr = stats.linregress(x, y)
        slope, intercept, p_val = lr.slope, lr.intercept, lr.pvalue
        y_fit = intercept + slope * x_grid

        y_pred = intercept + slope * x
        resid = y - y_pred
        n = len(x)
        s_err = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
        x_mean = np.mean(x)
        ssx = np.sum((x - x_mean) ** 2)

        if ssx <= 0:
            ci = np.full_like(x_grid, np.nan)
        else:
            t_val = stats.t.ppf(0.975, df=max(n - 2, 1))
            ci = t_val * s_err * np.sqrt(1 / n + (x_grid - x_mean) ** 2 / ssx)

        return slope, intercept, p_val, y_fit, ci

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=600)

        panel_info = [
            ("non_heatwave", "a  NHW single-period relationship"),
            ("heatwave",     "b  HW single-period relationship"),
        ]

        for ax, (period_name, title) in zip(axes, panel_info):
            sub = df[df["period"] == period_name].dropna(subset=["Tx_res", "Tn_res"]).copy()

            if len(sub) < 5:
                ax.text(
                    0.5, 0.5,
                    f"Insufficient data\nfor {period_name}",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    color="#777777",
                )
                ax.axis("off")
                continue

            for group_name, color in [("UHI", COLOR_UHI), ("UCI", COLOR_UCI)]:
                g = sub[sub["group"] == group_name]
                if len(g) == 0:
                    continue

                ax.scatter(
                    g["Tx_res"],
                    g["Tn_res"],
                    s=14,
                    color=color,
                    alpha=0.42,
                    edgecolors="none",
                    label=f"{group_name} pairs",
                    zorder=2,
                )

                if len(g) >= 5:
                    x_grid = np.linspace(
                        np.nanpercentile(g["Tx_res"], 2),
                        np.nanpercentile(g["Tx_res"], 98),
                        120
                    )
                    slope, intercept, p_val, y_fit, ci = _fit_line_ci_local(
                        g["Tx_res"].values,
                        g["Tn_res"].values,
                        x_grid
                    )

                    ax.plot(
                        x_grid,
                        y_fit,
                        color=color,
                        lw=1.4,
                        label=rf"{group_name} fit, slope={slope:.2f}{_stars_from_p_local(p_val)}",
                        zorder=3,
                    )
                    ax.fill_between(
                        x_grid,
                        y_fit - ci,
                        y_fit + ci,
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                        zorder=1,
                    )

            x_all = sub["Tx_res"].values
            x_ref = np.linspace(
                np.nanpercentile(x_all, 2),
                np.nanpercentile(x_all, 98),
                160
            )

            ax.plot(
                x_ref,
                -x_ref,
                linestyle="--",
                color="black",
                lw=0.9,
                alpha=0.60,
                label="theoretical slope = -1",
                zorder=1,
            )

            ax.axhline(0, color="#777777", linestyle="--", lw=0.8)
            ax.axvline(0, color="#777777", linestyle="--", lw=0.8)

            ax.set_xlabel(r"$\Delta T_x-\Delta T_{mean}$ (°C)")
            ax.set_ylabel(r"$\Delta T_n-\Delta T_{mean}$ (°C)")
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(True, lw=0.25, alpha=0.20)
            ax.legend(frameon=False, loc="best", fontsize=6.4)
            add_black_frame(ax, lw=0.8)

        fig.tight_layout()

        fpath = os.path.join(
            output_dir,
            "combined_plots",
            "Figure_Supp_single_period_Tx_Tn_relationship.png"
        )
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        fig.savefig(fpath.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {fpath}")

def plot_supplement_amp_vs_hysteresis(all_df, output_dir):
    """
    Supplementary figure:
    (a) Hysteresis loops stratified by NHW baseline ΔAmp regime
    (b) Regime-mean HW-NHW change in hysteresis area
    (c) Regime-mean HW-NHW change in phase lag

    Regimes are defined using NHW baseline dAmp1, not HW-NHW change.
    """
    ensure_dir(os.path.join(output_dir, "combined_plots"))

    # 1. 自动识别 pair_id 列
    id_col = None
    for c in ["pair_id", "station_id", "city_id"]:
        if c in all_df.columns:
            id_col = c
            break

    if id_col is None:
        print("  [Supp Amp vs Hysteresis] Missing pair id column. Skipping.")
        return

    # 2. 列检查
    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]
    required = ["period", "group", "dAmp1"] + u_cols + r_cols

    missing = [c for c in required if c not in all_df.columns]
    if missing:
        print("  [Warning] Supp Amp vs Hysteresis missing columns. Skipping.")
        return

    # 3. 数据合并
    nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
    hw_df  = all_df[all_df["period"] == "heatwave"].copy()

    if len(nhw_df) == 0 or len(hw_df) == 0:
        print("  [Supp Amp vs Hysteresis] NHW or HW data is empty. Skipping.")
        return

    cols_to_keep = [id_col, "group", "dAmp1"] + u_cols + r_cols
    paired = pd.merge(
        nhw_df[cols_to_keep],
        hw_df[cols_to_keep],
        on=[id_col, "group"],
        suffixes=("_nhw", "_hw"),
        how="inner"
    ).dropna(subset=["dAmp1_nhw", "dAmp1_hw"])

    if len(paired) < 5:
        print("  [Supp Amp vs Hysteresis] Not enough paired data. Skipping.")
        return

    # 4. 计算 HW-NHW ΔAmp，但分组使用 NHW baseline dAmp1
    paired["ddAmp"] = paired["dAmp1_hw"] - paired["dAmp1_nhw"]

    q1, q2 = np.nanpercentile(paired["dAmp1_nhw"], [33.3, 66.7])

    def _get_regime(v):
        if v <= q1:
            return "Low baseline ΔAmp"
        if v >= q2:
            return "High baseline ΔAmp"
        return "Intermediate baseline ΔAmp"

    paired["regime"] = paired["dAmp1_nhw"].apply(_get_regime)

    regime_order = [
        "Low baseline ΔAmp",
        "Intermediate baseline ΔAmp",
        "High baseline ΔAmp",
    ]

    regime_colors = {
        "Low baseline ΔAmp": "#2166ac",
        "Intermediate baseline ΔAmp": "#7f7f7f",
        "High baseline ΔAmp": "#b2182b",
    }

    # 5. 高鲁棒性物理量计算：First-harmonic phase + signed area
    def _get_first_harmonic_phase(arr):
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).all():
            return np.nan
        arr = arr - np.nanmean(arr)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        c = np.fft.fft(arr)[1]
        phi = -np.angle(c)
        peak_h = (phi * 24 / (2 * np.pi)) % 24
        return peak_h

    def _calc_row_metrics(u_arr, r_arr):
        u_arr = np.asarray(u_arr, dtype=float)
        r_arr = np.asarray(r_arr, dtype=float)

        if np.isnan(u_arr).all() or np.isnan(r_arr).all():
            return np.nan, np.nan

        x = r_arr
        y = u_arr - r_arr

        valid = np.isfinite(x) & np.isfinite(y)
        xv = x[valid]
        yv = y[valid]

        if len(xv) < 4:
            area = np.nan
        else:
            # Signed shoelace loop area
            area = 0.5 * np.sum(
                xv * np.roll(yv, -1) - np.roll(xv, -1) * yv
            )

        peak_x = _get_first_harmonic_phase(x)
        peak_y = _get_first_harmonic_phase(y)

        if pd.isna(peak_x) or pd.isna(peak_y):
            lag = np.nan
        else:
            lag = (peak_y - peak_x) % 24

        return area, lag

    # 6. 计算所有 pair 的 hysteresis metrics
    areas_nhw, lags_nhw, areas_hw, lags_hw = [], [], [], []

    for _, row in paired.iterrows():
        u_n = row[[f"{c}_nhw" for c in u_cols]].values.astype(float)
        r_n = row[[f"{c}_nhw" for c in r_cols]].values.astype(float)
        u_h = row[[f"{c}_hw" for c in u_cols]].values.astype(float)
        r_h = row[[f"{c}_hw" for c in r_cols]].values.astype(float)

        a_n, l_n = _calc_row_metrics(u_n, r_n)
        a_h, l_h = _calc_row_metrics(u_h, r_h)

        areas_nhw.append(a_n)
        lags_nhw.append(l_n)
        areas_hw.append(a_h)
        lags_hw.append(l_h)

    paired["area_nhw"] = areas_nhw
    paired["lag_nhw"]  = lags_nhw
    paired["area_hw"]  = areas_hw
    paired["lag_hw"]   = lags_hw

    paired["dArea"] = paired["area_hw"] - paired["area_nhw"]
    paired["dLag"]  = ((paired["lag_hw"] - paired["lag_nhw"] + 12) % 24) - 12

    # 统计辅助函数
    def _mean_ci_p(vals):
        vals = pd.Series(vals).dropna().values
        n = len(vals)
        if n < 3:
            return np.nan, np.nan, np.nan, n
        m = np.mean(vals)
        ci = 1.96 * stats.sem(vals)
        p = stats.ttest_1samp(vals, 0).pvalue
        return m, ci, p, n

    def _stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    # 7. 绘图
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig = plt.figure(figsize=(11, 7.5), dpi=600)
        gs = fig.add_gridspec(
            2, 6,
            height_ratios=[1.2, 1],
            hspace=0.45,
            wspace=0.8
        )

        # Top row
        ax_a1 = fig.add_subplot(gs[0, 0:2])
        ax_a2 = fig.add_subplot(gs[0, 2:4])
        ax_a3 = fig.add_subplot(gs[0, 4:6])
        axes_loops = [ax_a1, ax_a2, ax_a3]

        # Bottom row
        ax_b = fig.add_subplot(gs[1, 1:3])
        ax_c = fig.add_subplot(gs[1, 3:5])

        # --- Panel a: Hysteresis loops stratified by NHW baseline ΔAmp ---
        for ax, regime in zip(axes_loops, regime_order):
            g_df = paired[paired["regime"] == regime]
            if len(g_df) == 0:
                continue

            u_n = np.nanmean(
                g_df[[f"{c}_nhw" for c in u_cols]].values.astype(float),
                axis=0
            )
            r_n = np.nanmean(
                g_df[[f"{c}_nhw" for c in r_cols]].values.astype(float),
                axis=0
            )
            u_h = np.nanmean(
                g_df[[f"{c}_hw" for c in u_cols]].values.astype(float),
                axis=0
            )
            r_h = np.nanmean(
                g_df[[f"{c}_hw" for c in r_cols]].values.astype(float),
                axis=0
            )

            def _draw_loop(x, y, color, ls, label):
                ax.plot(
                    x, y,
                    color=color,
                    linestyle=ls,
                    lw=1.8,
                    alpha=0.85,
                    label=label
                )
                for h in [0, 6, 12, 18]:
                    ax.scatter(x[h], y[h], color=color, s=20, zorder=4)
                    ax.text(
                        x[h], y[h], f"{h:02d}",
                        color=color,
                        fontsize=6.5,
                        ha="left",
                        va="bottom"
                    )
                ax.annotate(
                    "",
                    xy=(x[13], y[13]),
                    xytext=(x[12], y[12]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.2
                    )
                )

            _draw_loop(r_n, u_n - r_n, "#888888", "--", "NHW")
            _draw_loop(r_h, u_h - r_h, regime_colors[regime], "-", "HW")

            ax.set_xlabel("Rural background $T_a$ (°C)")
            if ax == ax_a1:
                ax.set_ylabel("Urban−Rural $\Delta T_a$ (°C)")
                ax.set_title(f"a  {regime}", loc="left", fontweight="bold")
            else:
                ax.set_title(regime, loc="center", fontweight="bold")

            ax.text(
                0.03, 0.97,
                f"baseline ΔAmp tertile\nn={len(g_df)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2),
            )

            ax.grid(True, lw=0.3, alpha=0.3)
            ax.legend(frameon=False, loc="best", fontsize=7)
            add_black_frame(ax, lw=0.8)

        # --- Panel b & c ---
        targets = [
            (
                ax_b,
                "dArea",
                "b  Regime-mean change in hysteresis area",
                r"HW−NHW change in area (°C$^2$)"
            ),
            (
                ax_c,
                "dLag",
                "c  Regime-mean change in phase lag",
                "HW−NHW change in phase lag (h)"
            ),
        ]

        x_pos = np.arange(len(regime_order))
        width = 0.5

        for ax, col, title, ylabel in targets:
            means, cis, ps, ns = [], [], [], []

            for regime in regime_order:
                vals = paired[paired["regime"] == regime][col]
                m, ci, p, n = _mean_ci_p(vals)
                means.append(m)
                cis.append(ci)
                ps.append(p)
                ns.append(n)

            colors = [regime_colors[r] for r in regime_order]

            ax.bar(
                x_pos,
                means,
                width,
                yerr=cis,
                color=colors,
                alpha=0.85,
                edgecolor="black",
                capsize=4,
                linewidth=0.6
            )
            ax.axhline(0, color="black", lw=0.8)

            ymin, ymax = ax.get_ylim()
            yr = ymax - ymin

            for i, (m, ci, p) in enumerate(zip(means, cis, ps)):
                if not np.isfinite(m):
                    continue

                ci = 0 if not np.isfinite(ci) else ci
                star = _stars(p)

                y_text = m + np.sign(m if m != 0 else 1) * ci
                y_text = y_text + 0.05 * yr if m >= 0 else y_text - 0.08 * yr
                va = "bottom" if m >= 0 else "top"

                ax.text(
                    x_pos[i],
                    y_text,
                    star,
                    ha="center",
                    va=va,
                    fontsize=9,
                    fontweight="bold"
                )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [f"{r}\n(n={n})" for r, n in zip(regime_order, ns)]
            )
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(True, axis="y", lw=0.3, alpha=0.3)
            add_black_frame(ax, lw=0.8)

        fig.tight_layout()

        fpath = os.path.join(
            output_dir,
            "combined_plots",
            "Figure_Supp_Amp_vs_Hysteresis.png"
        )
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        fig.savefig(fpath.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved: {fpath}")


def plot_supplement_ncc_amp_controlled_linkage(all_df, output_dir):
    """
    Supplementary NCC-style mechanism validation:
      (a) Background-removed Tx–Tn linkage coloured by HW-induced ΔAmp
      (b) ΔAmp-stratified compensation slopes
      (c) closure residual: ε = Tx' + Tn'

    This directly tests whether daytime and nighttime responses are organized
    by amplitude damping rather than independent covariation.
    """
    ensure_dir(os.path.join(output_dir, "combined_plots"))

    id_col = None
    for c in ["pair_id", "station_id", "city_id"]:
        if c in all_df.columns:
            id_col = c
            break

    required = ["period", "group", "dTmean", "dAmp1", "dTx", "dTn"]
    missing = [c for c in required if c not in all_df.columns]

    if id_col is None:
        print("  [Supp NCC linkage] Missing pair id column. Skipping.")
        return
    if missing:
        print(f"  [Supp NCC linkage] Missing columns: {missing}. Skipping.")
        return

    nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
    hw_df  = all_df[all_df["period"] == "heatwave"].copy()

    if len(nhw_df) == 0 or len(hw_df) == 0:
        print("  [Supp NCC linkage] NHW or HW data missing. Skipping.")
        return

    paired = pd.merge(
        nhw_df[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
        hw_df[[id_col, "dTmean", "dAmp1", "dTx", "dTn"]],
        on=id_col,
        suffixes=("_nhw", "_hw"),
        how="inner",
    ).dropna()

    if len(paired) < 5:
        print("  [Supp NCC linkage] Insufficient paired data. Skipping.")
        return

    df = paired.copy()

    # HW-induced changes
    df["ddTx"]    = df["dTx_hw"]    - df["dTx_nhw"]
    df["ddTn"]    = df["dTn_hw"]    - df["dTn_nhw"]
    df["ddTmean"] = df["dTmean_hw"] - df["dTmean_nhw"]
    df["ddAmp"]   = df["dAmp1_hw"]  - df["dAmp1_nhw"]

    # Remove common warming
    df["Tx_res"] = df["ddTx"] - df["ddTmean"]
    df["Tn_res"] = df["ddTn"] - df["ddTmean"]

    # Closure residual: should be close to zero if Tx/Tn are opposite projections
    df["closure_eps"] = df["Tx_res"] + df["Tn_res"]

    df = df[["group", "Tx_res", "Tn_res", "ddAmp", "closure_eps"]].dropna().copy()

    if len(df) < 5:
        print("  [Supp NCC linkage] Insufficient valid residual data. Skipping.")
        return

    def _stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def _fit_ci(x, y, x_grid):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < 5 or np.isclose(np.nanstd(x), 0):
            return np.nan, np.nan, np.nan, np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)

        lr = stats.linregress(x, y)
        y_fit = lr.intercept + lr.slope * x_grid

        y_pred = lr.intercept + lr.slope * x
        resid = y - y_pred
        n = len(x)
        s_err = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
        x_mean = np.mean(x)
        ssx = np.sum((x - x_mean) ** 2)

        if ssx <= 0:
            ci = np.full_like(x_grid, np.nan)
        else:
            t_val = stats.t.ppf(0.975, df=max(n - 2, 1))
            ci = t_val * s_err * np.sqrt(1 / n + (x_grid - x_mean) ** 2 / ssx)

        return lr.slope, lr.intercept, lr.pvalue, y_fit, ci

    # ΔAmp groups: damping / neutral / amplification
    q1, q2 = np.nanpercentile(df["ddAmp"], [33.3, 66.7])

    def _amp_bin(v):
        if v <= q1:
            return "Strong damping"
        if v >= q2:
            return "Amplitude increase"
        return "Weak change"

    df["amp_bin"] = df["ddAmp"].apply(_amp_bin)

    bin_order = ["Strong damping", "Weak change", "Amplitude increase"]
    bin_colors = {
        "Strong damping": "#2166ac",
        "Weak change": "#7f7f7f",
        "Amplitude increase": "#b2182b",
    }

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=600)
        ax_a, ax_b, ax_c = axes

        # ------------------------------------------------------------
        # (a) Continuous ΔAmp-coloured linkage
        # ------------------------------------------------------------
        amp_abs = np.nanpercentile(np.abs(df["ddAmp"].values), 98)
        amp_abs = max(amp_abs, 0.1)
        norm_amp = TwoSlopeNorm(vmin=-amp_abs, vcenter=0, vmax=amp_abs)
        cmap_amp = plt.cm.RdBu_r

        marker_map = {"UHI": "o", "UCI": "^"}

        for group_name in ["UHI", "UCI"]:
            g = df[df["group"] == group_name]
            if len(g) == 0:
                continue

            sc = ax_a.scatter(
                g["Tx_res"], g["Tn_res"],
                c=g["ddAmp"],
                cmap=cmap_amp,
                norm=norm_amp,
                marker=marker_map[group_name],
                s=22 if group_name == "UHI" else 28,
                alpha=0.72,
                edgecolors="black",
                linewidths=0.25,
                label=f"{group_name}",
                zorder=2,
            )

        x_grid = np.linspace(
            np.nanpercentile(df["Tx_res"], 2),
            np.nanpercentile(df["Tx_res"], 98),
            160
        )
        slope, intercept, p_val, y_fit, ci = _fit_ci(
            df["Tx_res"].values,
            df["Tn_res"].values,
            x_grid
        )

        ax_a.plot(
            x_grid, y_fit,
            color="black", lw=1.5,
            label=rf"fit, slope={slope:.2f}{_stars(p_val)}",
            zorder=4,
        )
        ax_a.fill_between(
            x_grid, y_fit - ci, y_fit + ci,
            color="black", alpha=0.12, linewidth=0, zorder=1,
        )
        ax_a.plot(
            x_grid, -x_grid,
            "--", color="#555555", lw=1.0, alpha=0.75,
            label="theoretical slope = -1",
            zorder=3,
        )

        ax_a.axhline(0, color="#777777", ls="--", lw=0.8)
        ax_a.axvline(0, color="#777777", ls="--", lw=0.8)
        ax_a.set_xlabel(r"Daytime response after removing $\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"Nighttime response after removing $\Delta T_{mean}$ (°C)")
        ax_a.set_title("a  Tx–Tn linkage is organized by amplitude change",
                       loc="left", fontweight="bold")
        ax_a.legend(frameon=False, loc="best", fontsize=6.3)
        ax_a.grid(True, lw=0.25, alpha=0.20)
        add_black_frame(ax_a, lw=0.8)

        cbar = fig.colorbar(sc, ax=ax_a, pad=0.02, shrink=0.75)
        cbar.set_label(r"HW$-$NHW $\Delta Amp$ (°C)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # ------------------------------------------------------------
        # (b) ΔAmp-stratified slopes
        # ------------------------------------------------------------
        for bin_name in bin_order:
            g = df[df["amp_bin"] == bin_name]
            if len(g) < 5:
                print(f"  [Supp NCC linkage] Too few points in bin: {bin_name}")
                continue

            ax_b.scatter(
                g["Tx_res"], g["Tn_res"],
                s=16,
                color=bin_colors[bin_name],
                alpha=0.36,
                edgecolors="none",
                label=f"{bin_name} (n={len(g)})",
                zorder=2,
            )

            x_grid_b = np.linspace(
                np.nanpercentile(g["Tx_res"], 2),
                np.nanpercentile(g["Tx_res"], 98),
                120
            )
            slope_b, intercept_b, p_b, y_fit_b, ci_b = _fit_ci(
                g["Tx_res"].values,
                g["Tn_res"].values,
                x_grid_b
            )

            ax_b.plot(
                x_grid_b, y_fit_b,
                color=bin_colors[bin_name],
                lw=1.5,
                label=rf"{bin_name}: slope={slope_b:.2f}{_stars(p_b)}",
                zorder=3,
            )
            ax_b.fill_between(
                x_grid_b, y_fit_b - ci_b, y_fit_b + ci_b,
                color=bin_colors[bin_name],
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )

        x_all = np.linspace(
            np.nanpercentile(df["Tx_res"], 2),
            np.nanpercentile(df["Tx_res"], 98),
            160
        )
        ax_b.plot(
            x_all, -x_all,
            "--", color="#555555", lw=1.0, alpha=0.75,
            label="theoretical slope = -1",
        )

        ax_b.axhline(0, color="#777777", ls="--", lw=0.8)
        ax_b.axvline(0, color="#777777", ls="--", lw=0.8)
        ax_b.set_xlabel(r"Daytime response after removing $\Delta T_{mean}$ (°C)")
        ax_b.set_ylabel(r"Nighttime response after removing $\Delta T_{mean}$ (°C)")
        ax_b.set_title("b  Compensation persists across ΔAmp regimes",
                       loc="left", fontweight="bold")
        ax_b.legend(frameon=False, loc="best", fontsize=5.9)
        ax_b.grid(True, lw=0.25, alpha=0.20)
        add_black_frame(ax_b, lw=0.8)

        # ------------------------------------------------------------
        # (c) Closure residual ε = Tx' + Tn'
        # ------------------------------------------------------------
        eps_mean = df["closure_eps"].mean()
        eps_sem = stats.sem(df["closure_eps"].dropna()) if len(df["closure_eps"].dropna()) > 2 else np.nan
        p_eps = stats.ttest_1samp(df["closure_eps"].dropna(), 0).pvalue if len(df["closure_eps"].dropna()) >= 3 else np.nan

        for bin_name in bin_order:
            g = df[df["amp_bin"] == bin_name]
            if len(g) == 0:
                continue

            ax_c.scatter(
                g["ddAmp"],
                g["closure_eps"],
                s=18,
                color=bin_colors[bin_name],
                alpha=0.48,
                edgecolors="none",
                label=bin_name,
            )

        x_grid_c = np.linspace(
            np.nanpercentile(df["ddAmp"], 2),
            np.nanpercentile(df["ddAmp"], 98),
            120
        )
        slope_c, intercept_c, p_c, y_fit_c, ci_c = _fit_ci(
            df["ddAmp"].values,
            df["closure_eps"].values,
            x_grid_c
        )

        ax_c.plot(
            x_grid_c, y_fit_c,
            color="black", lw=1.4,
            label=rf"slope={slope_c:.2f}{_stars(p_c)}",
        )
        ax_c.fill_between(
            x_grid_c, y_fit_c - ci_c, y_fit_c + ci_c,
            color="black", alpha=0.12, linewidth=0,
        )

        ax_c.axhline(0, color="black", ls="--", lw=1.0)
        ax_c.axvline(0, color="#777777", ls="--", lw=0.8)

        ax_c.text(
            0.03, 0.97,
            rf"$\epsilon = T_x' + T_n'$" + "\n" +
            rf"mean={eps_mean:+.2f} °C, p={p_eps:.3f}",
            transform=ax_c.transAxes,
            ha="left", va="top",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2),
        )

        ax_c.set_xlabel(r"HW$-$NHW $\Delta Amp$ (°C)")
        ax_c.set_ylabel(r"Closure residual $\epsilon$ (°C)")
        ax_c.set_title("c  Small closure residual supports one-axis control",
                       loc="left", fontweight="bold")
        ax_c.legend(frameon=False, loc="best", fontsize=6.2)
        ax_c.grid(True, lw=0.25, alpha=0.20)
        add_black_frame(ax_c, lw=0.8)

        fig.tight_layout()

        fpath = os.path.join(
            output_dir,
            "combined_plots",
            "Figure_Supp_NCC_amp_controlled_linkage.png"
        )
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        fig.savefig(fpath.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {fpath}")

# ─────────────────────────────────────────────────────────────
# Legacy exploratory figure: dry-bulb exposure proxies (disabled by default)
# ─────────────────────────────────────────────────────────────
def plot_figure4_human_impacts(all_df, output_dir):
    """
    Legacy exploratory dry-bulb exposure-proxy figure.

    This is not the formal Figure 4 labour/sleep/CDH workflow and is disabled
    by default. It does not use dew point, WBGT or the Dunne labour model.

    Panels:
      (a) amplitude damping -> nighttime heat exposure
      (b) nighttime exposure -> sleep-loss proxy
      (c) daytime exposure -> labour-loss proxy
      (d) nighttime exposure -> cooling-demand proxy

    All indicators are computed from input diurnal temperature curves.
    If real sleep/labour/energy data are unavailable, these are exposure-based proxies.
    """
    ensure_dir(os.path.join(output_dir, "plots"))

    # 1. 自动识别 ID 列
    id_col = None
    for c in ["pair_id", "station_id", "city_id"]:
        if c in all_df.columns:
            id_col = c
            break

    if id_col is None:
        print("  [Fig4 Human] Missing pair id column. Skipping.")
        return

    # 2. 检查所需列
    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    required = ["period", "group", "dAmp1", "dTx", "dTn"] + u_cols
    missing = [c for c in required if c not in all_df.columns]
    if missing:
        print(f"  [Fig4 Human] Missing columns: {missing[:8]} ... Skipping.")
        return

    # 3. 提取并合并 NHW 和 HW 数据
    nhw = all_df[all_df["period"] == "non_heatwave"].copy()
    hw  = all_df[all_df["period"] == "heatwave"].copy()

    if len(nhw) == 0 or len(hw) == 0:
        print("  [Fig4 Human] NHW or HW data missing. Skipping.")
        return

    keep_cols = [id_col, "group", "dAmp1", "dTx", "dTn"] + u_cols
    paired = pd.merge(
        nhw[keep_cols],
        hw[keep_cols],
        on=[id_col, "group"],
        how="inner",
        suffixes=("_nhw", "_hw")
    ).dropna(subset=["dAmp1_nhw", "dAmp1_hw"])

    if len(paired) < 5:
        print("  [Fig4 Human] Not enough paired data. Skipping.")
        return

    # 4. 定义时间段并计算各项代理指标
    night_hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]     # 20:00–07:59, 12 h
    day_hours   = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 08:00–19:59, 12 h

    def _cols(period_suffix, hours):
        return [f"urban_diurnal_h{h:02d}_{period_suffix}" for h in hours]

    def _mean_temp(df, suffix, hours):
        return df[_cols(suffix, hours)].values.astype(float).mean(axis=1)

    def _degree_hours(df, suffix, hours, threshold):
        arr = df[_cols(suffix, hours)].values.astype(float)
        return np.maximum(arr - threshold, 0).sum(axis=1)

    paired["ddAmp"] = paired["dAmp1_hw"] - paired["dAmp1_nhw"]
    
    paired["nightT_nhw"] = _mean_temp(paired, "nhw", night_hours)
    paired["nightT_hw"]  = _mean_temp(paired, "hw",  night_hours)
    paired["dayT_nhw"]   = _mean_temp(paired, "nhw", day_hours)
    paired["dayT_hw"]    = _mean_temp(paired, "hw",  day_hours)

    paired["dNightT"] = paired["nightT_hw"] - paired["nightT_nhw"]
    paired["dDayT"]   = paired["dayT_hw"]   - paired["dayT_nhw"]

    # Exposure-based proxies
    paired["sleepExp_nhw"] = _degree_hours(paired, "nhw", night_hours, threshold=25.0)
    paired["sleepExp_hw"]  = _degree_hours(paired, "hw",  night_hours, threshold=25.0)
    paired["dSleepExp"]    = paired["sleepExp_hw"] - paired["sleepExp_nhw"]

    paired["labourExp_nhw"] = _degree_hours(paired, "nhw", day_hours, threshold=30.0)
    paired["labourExp_hw"]  = _degree_hours(paired, "hw",  day_hours, threshold=30.0)
    paired["dLabourLoss"]   = paired["labourExp_hw"] - paired["labourExp_nhw"]

    paired["cooling_nhw"] = _degree_hours(paired, "nhw", night_hours, threshold=26.0)
    paired["cooling_hw"]  = _degree_hours(paired, "hw",  night_hours, threshold=26.0)
    paired["dCooling"]    = paired["cooling_hw"] - paired["cooling_nhw"]

    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ddAmp", "dNightT", "dSleepExp", "dLabourLoss", "dCooling"]
    )

    if len(paired) < 5:
        print("  [Fig4 Human] Not enough valid computed exposure data. Skipping.")
        return

    # 5. 定义绘图辅助函数
    def _stars(p):
        if pd.isna(p): return ""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    def _fit_ci(x, y, x_grid):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < 5 or np.isclose(np.nanstd(x), 0):
            return np.nan, np.nan, np.nan, np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)

        lr = stats.linregress(x, y)
        y_fit = lr.intercept + lr.slope * x_grid
        y_pred = lr.intercept + lr.slope * x
        resid = y - y_pred
        n = len(x)
        s_err = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
        x_mean = np.mean(x)
        ssx = np.sum((x - x_mean) ** 2)

        if ssx <= 0:
            ci = np.full_like(x_grid, np.nan)
        else:
            t_val = stats.t.ppf(0.975, df=max(n - 2, 1))
            ci = t_val * s_err * np.sqrt(1 / n + (x_grid - x_mean) ** 2 / ssx)

        return lr.slope, lr.intercept, lr.pvalue, y_fit, ci

    def _scatter_fit(ax, x, y, xlabel, ylabel, title, note=None):
        for grp, color, marker in [("UHI", COLOR_UHI, "o"), ("UCI", COLOR_UCI, "^")]:
            g = paired[paired["group"] == grp]
            if len(g) == 0: continue
            ax.scatter(g[x], g[y], s=22, color=color, marker=marker,
                       alpha=0.55, edgecolors="none", label=grp)

        x_all, y_all = paired[x].values, paired[y].values
        x_grid = np.linspace(np.nanpercentile(x_all, 2), np.nanpercentile(x_all, 98), 160)
        slope, intercept, p, y_fit, ci = _fit_ci(x_all, y_all, x_grid)

        if np.isfinite(slope):
            ax.plot(x_grid, y_fit, color="black", lw=1.4,
                    label=rf"slope={slope:.2f}{_stars(p)}")
            ax.fill_between(x_grid, y_fit - ci, y_fit + ci, color="black", alpha=0.12, linewidth=0)

        ax.axhline(0, color="#777777", linestyle="--", lw=0.8)
        ax.axvline(0, color="#777777", linestyle="--", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, lw=0.28, alpha=0.25)

        if note is not None:
            ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.2, bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2))

        ax.legend(frameon=False, loc="best", fontsize=7)
        add_black_frame(ax, lw=0.8)

    # 6. 设置全局样式并绘图
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig = plt.figure(figsize=(11.5, 8.2), dpi=600)
        gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.38)

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        _scatter_fit(ax_a, "ddAmp", "dNightT",
                     r"HW−NHW change in $\Delta Amp$ (°C)", "HW−NHW change in nighttime temperature (°C)",
                     "a  Amplitude damping shifts exposure to nighttime", "Nighttime: 22:00–06:00")
        _scatter_fit(ax_b, "dNightT", "dSleepExp",
                     "HW−NHW change in nighttime temperature (°C)", "Change in sleep-loss heat exposure (°C·h)",
                     "b  Nighttime exposure increases sleep-loss risk", "Proxy: Σ max(Tnight − 25°C, 0)")
        _scatter_fit(ax_c, "dDayT", "dLabourLoss",
                     "HW−NHW change in daytime temperature (°C)", "Change in labour heat exposure (°C·h)",
                     "c  Daytime heat governs labour-capacity loss", "Proxy: Σ max(Tday − 30°C, 0)")
        _scatter_fit(ax_d, "dNightT", "dCooling",
                     "HW−NHW change in nighttime temperature (°C)", "Change in nighttime cooling demand (°C·h)",
                     "d  Nighttime heat increases cooling demand", "Proxy: Σ max(Tnight − 26°C, 0)")

        fig.tight_layout()
        custom_out_dir = LEGACY_FIGURES_OUTPUT_DIR
        os.makedirs(custom_out_dir, exist_ok=True)
        fpath = os.path.join(custom_out_dir, "Figure4_human_impacts.png")
        fig.savefig(fpath, dpi=600, bbox_inches="tight")
        fig.savefig(fpath.replace(".png", ".pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {fpath}")


def run_data_diagnostics(df, output_dir):
    """
    全链路数据流诊断报告：追踪 UHI/UCI 在各阶段的样本量变化
    """
    import os
    diag_path = os.path.join(output_dir, "data_flow_diagnostics.txt")
    
    with open(diag_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log("="*80)
        log("数据流追踪诊断报告 (Data Flow Diagnostics)")
        log("="*80)

        # Step 0: 原始载入
        log(f"0. 原始载入总量: {len(df)} 行")
        
        # 识别 ID 列
        id_col = None
        for c in ["pair_id", "station_id", "city_id"]:
            if c in df.columns:
                id_col = c
                break
        log(f"   识别到配对主键列: {id_col}")

        # Step 1: hw_method 过滤
        if "hw_method" in df.columns:
            df_percentile = df[df["hw_method"] == "percentile"]
            log(f"1. 过滤 hw_method=='percentile' 后: {len(df_percentile)} 行 (损失: {len(df) - len(df_percentile)})")
        else:
            df_percentile = df
            log("1. 未发现 'hw_method' 列，跳过过滤。")

        # Step 2: 周期与分组分布 (UHI/UCI x Period)
        log("\n2. 各周期(Period)与集群(Group)的初始分布:")
        dist = pd.crosstab(df_percentile['period'], df_percentile['group'], margins=True)
        log(dist.to_string())

        # Step 3: 检查 NHW 与 HW 的配对完整性 (用于机制分析图)
        nhw_ids = set(df_percentile[df_percentile["period"] == "non_heatwave"][id_col].unique())
        hw_ids = set(df_percentile[df_percentile["period"] == "heatwave"][id_col].unique())
        paired_ids = nhw_ids.intersection(hw_ids)

        log(f"\n3. 配对完整性检查 (用于需对比 NHW/HW 的图表):")
        log(f"   - 拥有 Non-Heatwave 记录的唯一配对数: {len(nhw_ids)}")
        log(f"   - 拥有 Heatwave 记录的唯一配对数: {len(hw_ids)}")
        log(f"   - 同时拥有两者的完整配对数 (Intersection): {len(paired_ids)}")
        log(f"   - 损失数 (只有单周期数据): {len(nhw_ids.union(hw_ids)) - len(paired_ids)}")

        # Step 4: 关键绘图变量的有效性 (NaN 检查)
        log("\n4. 关键变量缺失值分析 (NaN check - Annual Period):")
        annual_df = df_percentile[df_percentile["period"] == "annual"]
        critical_cols = ["dTmean", "dAmp1", "dTx", "dTn", "urban_diurnal_h00"]
        for col in critical_cols:
            if col in annual_df.columns:
                nan_count = annual_df[col].isna().sum()
                log(f"   - 列 [{col:15s}] 缺失数: {nan_count} / {len(annual_df)}")
            else:
                log(f"   - 列 [{col:15s}] 不存在!")

        # Step 5: Merge 后数据量的再次核对
        log("\n5. 机制分析图 (Figure Mechanism) 实际可用量预测:")
        df_paired = pd.merge(
            df_percentile[df_percentile["period"] == "non_heatwave"],
            df_percentile[df_percentile["period"] == "heatwave"],
            on=[id_col, "group"],
            suffixes=("_nhw", "_hw")
        )
        log(f"   - NHW 与 HW 合并(Merge)后行数: {len(df_paired)}")
        
        # 统计 Merge 后的 UHI/UCI 分布
        if len(df_paired) > 0:
            log(f"   - 其中 UHI: {len(df_paired[df_paired['group']=='UHI'])}")
            log(f"   - 其中 UCI: {len(df_paired[df_paired['group']=='UCI'])}")
            
            # 模拟 dropna 带来的损失
            df_final = df_paired.dropna(subset=["dTmean_nhw", "dTmean_hw", "dAmp1_nhw", "dAmp1_hw"])
            log(f"   - 经过关键变量 Dropna 后剩余行数: {len(df_final)}")
        
        log("="*80)
        log(f"报告已生成至: {diag_path}")

def find_missing_period_pairs(df, output_dir):
    """
    找出只有 HW 但没有 NHW，或者只有 NHW 但没有 HW 的异常 ID
    """
    # 1. 自动识别 ID 列
    id_col = None
    for c in ["pair_id", "station_id", "city_id"]:
        if c in df.columns:
            id_col = c
            break
    
    if id_col is None:
        print("无法识别 ID 列，请检查 CSV 文件表头。")
        return

    # 2. 提取不同周期的 ID 集合
    hw_ids = set(df[df["period"] == "heatwave"][id_col].unique())
    nhw_ids = set(df[df["period"] == "non_heatwave"][id_col].unique())

    # 3. 计算差异
    only_hw = hw_ids - nhw_ids
    only_nhw = nhw_ids - hw_ids

    print(f"\n" + "!"*40)
    print(f"数据配对一致性检查结果:")
    print(f" - 只有 HW 但缺失 NHW 的 ID 数量: {len(only_hw)}")
    print(f" - 只有 NHW 但缺失 HW 的 ID 数量: {len(only_nhw)}")
    print("!"*40 + "\n")

    # 4. 提取这些具体行的数据并保存
    if len(only_hw) > 0:
        error_hw_df = df[df[id_col].isin(only_hw)].copy()
        out_path = os.path.join(output_dir, "CRITICAL_MISSING_NHW_PAIRS.csv")
        error_hw_df.to_csv(out_path, index=False)
        print(f"已将 [只有热浪数据] 的站点详情保存至: {out_path}")
        print("这些站点可能导致在进行 HW vs NHW 对比分析时被自动剔除。")

    if len(only_nhw) > 0:
        error_nhw_df = df[df[id_col].isin(only_nhw)].copy()
        out_path_2 = os.path.join(output_dir, "STATIONS_WITH_NO_HEATWAVES.csv")
        error_nhw_df.to_csv(out_path_2, index=False)
        print(f"已将 [从未发生热浪] 的站点详情保存至: {out_path_2}")

    return list(only_hw) # 返回 ID 列表供进一步使用

def plot_figure_thermodynamic_regime_map(all_df, output_dir):
    """
    参考 Nature 风格的新增图：热力学机制分类图
    展示 ΔTmean 和 ΔAmp 的不同组合如何决定 UHI/UCI 的性质
    """
    import matplotlib.patches as mpatches

    # 1. 数据准备 (使用 annual 数据作为基准，或者 HW 数据)
    df = all_df[all_df["period"] == "annual"].copy()
    if len(df) == 0: return

    # 2. 画布设置
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "pdf.fonttype": 42
    }):
        fig, ax = plt.subplots(figsize=(8, 7), dpi=600)

        # 3. 确定坐标范围
        x_limit = np.nanpercentile(np.abs(df["dTmean"]), 99) + 0.5
        y_limit = np.nanpercentile(np.abs(df["dAmp1"]), 99) + 0.5
        
        # 4. 绘制理论背景颜色 (代表 ΔTx = ΔTmean + ΔAmp)
        # 这种“热图背景”常用于 Nature 解释两个变量的交互作用
        res = 100
        x = np.linspace(-x_limit, x_limit, res)
        y = np.linspace(-y_limit, y_limit, res)
        X, Y = np.meshgrid(x, y)
        Z = X + Y  # 这代表理论上的 ΔTx

        cmap = plt.cm.RdBu_r
        norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
        
        # 绘制等值线填充作为底色
        cp = ax.contourf(X, Y, Z, levels=50, cmap=cmap, norm=norm, alpha=0.2)
        
        # 5. 绘制关键分区线
        ax.axhline(0, color='black', lw=1.2, zorder=2) # ΔAmp = 0
        ax.axvline(0, color='black', lw=1.2, zorder=2) # ΔTmean = 0
        ax.plot(x, -x, color='black', linestyle='--', lw=1.5, alpha=0.8, zorder=2) # ΔTx = 0

        # 6. 叠加实际站点散点
        for grp, color, marker in [("UHI", "#d62728", "o"), ("UCI", "#1f77b4", "^")]:
            sub = df[df["group"] == grp]
            ax.scatter(sub["dTmean"], sub["dAmp1"], c=color, marker=marker, 
                       s=35, edgecolors='white', linewidths=0.5, alpha=0.9, 
                       label=f"Observed {grp}", zorder=4)

        # 7. 添加机制说明文字 (Nature 风格的标注)
        # Q1: 全面加剧
        ax.text(x_limit*0.5, y_limit*0.6, "Daytime-Dominant\nUHI Intensification", 
                ha='center', fontweight='bold', color='#7b0000', fontsize=9)
        # Q4: 夜间加剧 (振幅平减)
        ax.text(x_limit*0.5, -y_limit*0.6, "Nighttime-Dominant\nHeat Redistribution", 
                ha='center', fontweight='bold', color='#7b0000', fontsize=9)
        # 左侧: UCI 区域
        ax.text(-x_limit*0.5, -y_limit*0.2, "Urban Cool Island\n(UCI) Regimes", 
                ha='center', fontweight='bold', color='#003366', fontsize=9)

        # 8. 装饰
        ax.set_xlabel(r"Background Warming Impact ($\Delta T_{mean}$, °C)", fontsize=11)
        ax.set_ylabel(r"Diurnal Redistribution Impact ($\Delta Amp$, °C)", fontsize=11)
        ax.set_title("Thermodynamic Phase Space of Urban Heat Islands", loc='left', fontweight='bold', pad=15)
        
        # 添加公式标注
        ax.text(x_limit*0.6, y_limit*0.9, r"$\Delta T_x = \Delta T_{mean} + \Delta Amp$", 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(-y_limit, y_limit)
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        
        # 保存图片
        fpath = os.path.join(output_dir, "plots", "Figure_Thermodynamic_Regime_Map.png")
        plt.savefig(fpath, bbox_inches='tight', dpi=600)
        plt.close()
        print(f"  Saved: {fpath}")

def plot_figure_fft_schematic_mechanism(all_df, output_dir):
    """
    独立模块：基于 FFT 参数重构日循环情景
    展示 ΔTmean 和 ΔAmp 的组合如何导致 UHI/UCI 的产生
    """
    from matplotlib.patches import FancyArrowPatch
    import os

    # 1. 尝试从数据中提取一个典型的中纬度基准站 (Rural)
    # 筛选：中纬度(30-50N/S)，annual周期
    sample_df = all_df[
        (all_df["period"] == "annual") & 
        (all_df["lat_urban"].abs() > 30) & 
        (all_df["lat_urban"].abs() < 50)
    ].dropna(subset=["rural_Tmean", "rural_Amp1", "rural_Amp2"])

    if len(sample_df) > 0:
        # 随机取一个站点或取平均值
        ref_site = sample_df.iloc[0]
        m_ref  = ref_site["rural_Tmean"]
        a1_ref = ref_site["rural_Amp1"]
        p1_ref = ref_site["rural_phase1"] if "rural_phase1" in ref_site else 3.8
        a2_ref = ref_site["rural_Amp2"]
        p2_ref = ref_site["rural_phase2"] if "rural_phase2" in ref_site else 0.5
        site_name = ref_site["pair_id"] if "pair_id" in ref_site else "Mid-lat Site"
    else:
        # 如果没搜到，使用理想化中纬度参数
        m_ref, a1_ref, p1_ref, a2_ref, p2_ref = 15.0, 7.0, 3.8, 1.2, 0.5
        site_name = "Idealized Mid-lat"

    # --- 物理计算辅助函数 ---
    def reconstruct(t, m, a1, p1, a2, p2):
        w = 2 * np.pi / 24
        return m + a1 * np.cos(w * t - p1) + a2 * np.cos(2 * w * t - p2)

    def get_peaks(t_fine, vals):
        return np.max(vals), t_fine[np.argmax(vals)], np.min(vals), t_fine[np.argmin(vals)]

    # --- 绘图逻辑 ---
    t_fine = np.linspace(0, 24, 1000)
    
    # 设定变化量 (基于数据集中 UHI 组的平均变化量, 或使用典型值)
    # 模拟典型的城市化影响：均值升2度，振幅降2.5度
    dm_val, da_val = 2.0, -2.5

    scenarios = [
        (0, 0,        "(a) Rural Baseline", "Reference (Rural)"),
        (0, da_val,   "(b) Redistribution ($\Delta Amp < 0$)", "Amplitude Damping"),
        (dm_val, 0,   "(c) Offset ($\Delta T_{mean} > 0$)", "Background Warming"),
        (dm_val, da_val, "(d) Combined (Urban Case)", "Urbanization (UHI/UCI)")
    ]

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11, "pdf.fonttype": 42
    }):
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.12, hspace=0.2)

        C_REF, C_MOD, C_ARR = "#3498db", "#e67e22", "#2c3e50"

        for i, (dm, da, title, lab) in enumerate(scenarios):
            ax = axes.flatten()[i]
            
            # 计算基准和修改后的曲线
            y_ref = reconstruct(t_fine, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
            y_mod = reconstruct(t_fine, m_ref + dm, a1_ref + da, p1_ref, a2_ref, p2_ref)
            
            ax.plot(t_fine, y_ref, color=C_REF, lw=1.5, alpha=0.5, label="Rural" if i==0 else "")
            ax.plot(t_fine, y_mod, color=C_MOD, lw=2.2, label=lab)
            
            tx_r, hx_r, tn_r, hn_r = get_peaks(t_fine, y_ref)
            tx_m, hx_m, tn_m, hn_m = get_peaks(t_fine, y_mod)
            
            # 绘制指示箭头
            if i > 0:
                # ΔTmean 箭头 (红色)
                if dm != 0:
                    ax.annotate('', xy=(2, m_ref + dm), xytext=(2, m_ref),
                                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))
                    ax.text(2.3, m_ref + dm/2, r'$\Delta T_{mean}$', color='#d62728', fontweight='bold')

                # ΔAmp 箭头 (绿色)
                if da != 0:
                    # 在峰值处画压缩箭头
                    ax.annotate('', xy=(hx_r, tx_r + da + dm), xytext=(hx_r, tx_r + dm),
                                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))
                    ax.text(hx_r+0.5, tx_r + dm + da/2, r'$\Delta Amp$', color='#2ca02c')

                # 最终 Tx 和 Tn 的响应箭头 (深色)
                # Tx 位移
                arr_tx = FancyArrowPatch((hx_r, tx_r), (hx_m, tx_m), arrowstyle='->', 
                                         mutation_scale=12, color=C_ARR, lw=1, alpha=0.8)
                ax.add_patch(arr_tx)
                # Tn 位移
                arr_tn = FancyArrowPatch((hn_r, tn_r), (hn_m, tn_m), arrowstyle='->', 
                                         mutation_scale=12, color=C_ARR, lw=1, alpha=0.8)
                ax.add_patch(arr_tn)
                
                # 结果标注
                ax.text(hx_m, tx_m + 0.6, rf"$\Delta T_x={tx_m-tx_r:+.1f}$", ha='center', fontweight='bold', fontsize=8)
                ax.text(hn_m, tn_m - 1.4, rf"$\Delta T_n={tn_m-tn_r:+.1f}$", ha='center', fontweight='bold', fontsize=8)

            ax.set_title(title, loc='left', fontweight='bold')
            ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(True, lw=0.3, alpha=0.3)
            if i >= 2: ax.set_xlabel("Local Time (h)")
            if i % 2 == 0: ax.set_ylabel("Temperature (°C)")
            ax.legend(frameon=False, loc='upper left', fontsize=8)
            add_black_frame(ax)

        fig.suptitle(f"Thermodynamic Mechanism: {site_name} Parameters", fontsize=12, fontweight='bold', y=0.98)
        
        fpath = os.path.join(output_dir, "plots", "Figure_FFT_Schematic_Mechanism.png")
        ensure_dir(os.path.dirname(fpath))
        plt.savefig(fpath, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fpath}")

def plot_figure_fft_combined_mechanism_single(all_df, output_dir):
    """
    在一张图中集成所有场景：
    1. Rural 基准 (蓝色)
    2. 仅振幅平减 (绿色虚线)
    3. 仅背景升温 (红色虚线)
    4. 城市最终状态 (橙色实线)
    """
    from matplotlib.patches import FancyArrowPatch
    import os

    # --- 1. 参数提取 (同前，确保物理真实性) ---
    sample_df = all_df[
        (all_df["period"] == "annual") & (all_df["lat_urban"].abs() > 30)
    ].dropna(subset=["rural_Tmean", "rural_Amp1"])

    if len(sample_df) > 0:
        ref_site = sample_df.iloc[0]
        m_ref, a1_ref, a2_ref = ref_site["rural_Tmean"], ref_site["rural_Amp1"], ref_site["rural_Amp2"]
        p1_ref, p2_ref = 3.8, 0.5
    else:
        m_ref, a1_ref, a2_ref, p1_ref, p2_ref = 15.0, 7.0, 1.2, 3.8, 0.5

    def reconstruct(t, m, a1, p1, a2, p2):
        w = 2 * np.pi / 24
        return m + a1 * np.cos(w * t - p1) + a2 * np.cos(2 * w * t - p2)

    # --- 2. 场景计算 ---
    dm, da = 2.5, -2.8  # 设定显著的变化量以便观察
    t_fine = np.linspace(0, 24, 1000)
    
    y_rural    = reconstruct(t_fine, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
    y_amp_only = reconstruct(t_fine, m_ref, a1_ref + da, p1_ref, a2_ref, p2_ref)
    y_m_only   = reconstruct(t_fine, m_ref + dm, a1_ref, p1_ref, a2_ref, p2_ref)
    y_urban    = reconstruct(t_fine, m_ref + dm, a1_ref + da, p1_ref, a2_ref, p2_ref)

    # --- 3. 绘图 ---
    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 11, "pdf.fonttype": 42
    }):
        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=600)
        
        # 颜色与样式定义 (Nature 配色)
        C_RURAL = "#7f8c8d"  # 灰色 (基准)
        C_AMP   = "#27ae60"  # 绿色 (再分配)
        C_MEAN  = "#e74c3c"  # 红色 (偏移)
        C_URBAN = "#e67e22"  # 橙色 (最终)

        # 绘制曲线
        ax.plot(t_fine, y_rural, color=C_RURAL, lw=2.5, label="Rural (Baseline)", zorder=2)
        ax.plot(t_fine, y_amp_only, color=C_AMP, lw=1.5, ls="--", label="$\Delta Amp < 0$ only", zorder=3)
        ax.plot(t_fine, y_m_only, color=C_MEAN, lw=1.5, ls="--", label="$\Delta T_{mean} > 0$ only", zorder=3)
        ax.plot(t_fine, y_urban, color=C_URBAN, lw=3, label="Urban (Combined Case)", zorder=5)

        # --- 4. 添加物理分解箭头 (这是图的核心) ---
        # 我们在 14:00 (白天峰值附近) 和 04:00 (夜晚谷值附近) 标注
        h_day, h_night = 14.5, 4.0
        idx_d = np.argmin(np.abs(t_fine - h_day))
        idx_n = np.argmin(np.abs(t_fine - h_night))

        # A. 白天分解 (Tx 处)
        # 从 Rural 到 Mean 上升
        ax.annotate('', xy=(h_day, y_m_only[idx_d]), xytext=(h_day, y_rural[idx_d]),
                    arrowprops=dict(arrowstyle='->', color=C_MEAN, lw=1.5))
        # 从 Mean 下降到 Urban (因为 Amp 变小)
        ax.annotate('', xy=(h_day, y_urban[idx_d]), xytext=(h_day, y_m_only[idx_d]),
                    arrowprops=dict(arrowstyle='->', color=C_AMP, lw=1.5))
        ax.text(h_day+0.3, (y_rural[idx_d] + y_m_only[idx_d])/2, r"$\Delta T_{mean}$", color=C_MEAN, va='center')
        ax.text(h_day+0.3, (y_m_only[idx_d] + y_urban[idx_d])/2, r"$\Delta Amp$", color=C_AMP, va='center')
        
        # B. 夜间分解 (Tn 处)
        # 两个效应都在上升，所以重叠
        ax.annotate('', xy=(h_night, y_m_only[idx_n]), xytext=(h_night, y_rural[idx_n]),
                    arrowprops=dict(arrowstyle='->', color=C_MEAN, lw=1.5))
        ax.annotate('', xy=(h_night, y_urban[idx_n]), xytext=(h_night, y_m_only[idx_n]),
                    arrowprops=dict(arrowstyle='->', color=C_AMP, lw=1.5))
        ax.text(h_night-0.5, (y_rural[idx_n] + y_m_only[idx_n])/2, r"$\Delta T_{mean}$", color=C_MEAN, ha='right')
        ax.text(h_night-0.5, (y_m_only[idx_n] + y_urban[idx_n])/2, r"$-\Delta Amp$", color=C_AMP, ha='right')

        # --- 5. 标注最终结果 ---
        # 寻找最终的峰值变化
        dtx = np.max(y_urban) - np.max(y_rural)
        dtn = np.min(y_urban) - np.min(y_rural)
        
        # 用文本框标注
        ax.text(14.5, np.max(y_urban)+1, f"Daytime Response:\n$\Delta T_x = {dtx:+.1f}$°C", 
                color=C_URBAN, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.text(4.0, np.min(y_urban)-2, f"Nighttime Response:\n$\Delta T_n = {dtn:+.1f}$°C", 
                color=C_URBAN, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 装饰
        ax.set_xlabel("Local Solar Time (h)", fontsize=12)
        ax.set_ylabel("$T_a$ (°C)", fontsize=12)
        ax.set_title("Decomposition of Urban Thermal Forcing", loc='left', fontweight='bold', fontsize=13)
        ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(True, lw=0.4, alpha=0.3, ls=':')
        ax.legend(frameon=False, loc='lower right', fontsize=9)
        
        # 添加物理总结公式
        ax.text(0.5, 0.95, r"$\Delta T_a(t) = \Delta T_{mean} + \Delta Amp \cdot \cos(\omega t - \phi)$", 
                transform=ax.transAxes, fontsize=11, fontstyle='italic', color='#2c3e50')

        add_black_frame(ax)
        
        # 保存
        fpath = os.path.join(output_dir, "plots", "Figure_FFT_Combined_Mechanism_Single.png")
        plt.savefig(fpath, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fpath}")

def plot_figure_fft_clean_mechanism(all_df, output_dir):
    """
    Nature 风格整洁版机制图：
    1. 仅保留 Rural 和 Urban 两条主线，突出对比。
    2. 使用分步箭头 (Warming -> Redistribution) 展示逻辑。
    3. 优化文字布局，防止遮挡。
    """
    
    # --- 1. 参数设定 (使用更稳健的典型中纬度参数) ---
    m_ref, a1_ref, a2_ref, p1_ref, p2_ref = 15.0, 6.0, 1.0, 3.8, 0.5
    dm, da = 2.4, -2.8  # 模拟城市化：均值上升，振幅下降

    def reconstruct(t, m, a1, p1, a2, p2):
        w = 2 * np.pi / 24
        return m + a1 * np.cos(w * t - p1) + a2 * np.cos(2 * w * t - p2)

    t_fine = np.linspace(0, 24, 1000)
    y_rural = reconstruct(t_fine, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
    y_urban = reconstruct(t_fine, m_ref + dm, a1_ref + da, p1_ref, a2_ref, p2_ref)
    
    # 过程中间点 (用于画箭头)
    y_intermediate = reconstruct(t_fine, m_ref + dm, a1_ref, p1_ref, a2_ref, p2_ref)

    # --- 2. 绘图设置 ---
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    
    # 颜色与样式
    C_RURAL = "#5a6364"  # 深灰色
    C_URBAN = "#e67e22"  # 橙色
    C_WARM  = "#d62728"  # 红色 (Warming)
    C_REDIST = "#2ca02c" # 绿色 (Redistribution)
    
    # 绘制主曲线
    ax.plot(t_fine, y_rural, color=C_RURAL, lw=2, label="Rural (Baseline)", zorder=2)
    ax.plot(t_fine, y_urban, color=C_URBAN, lw=3.5, label="Urban (Combined Outcome)", zorder=5)

    # 文字描边对象 (确保清晰度)
    halo = [path_effects.withStroke(linewidth=3, foreground='white', alpha=0.8)]

    # --- 3. 逻辑分解：夜间 (Nighttime, 4:00) ---
    h_n = 4.0
    idx_n = np.argmin(np.abs(t_fine - h_n))
    
    # 第一步：Background Warming (dm)
    ax.annotate('', xy=(h_n, y_intermediate[idx_n]), xytext=(h_n, y_rural[idx_n]),
                arrowprops=dict(arrowstyle='->', color=C_WARM, lw=1.5, mutation_scale=10))
    # 第二步：Redistribution (-da)
    ax.annotate('', xy=(h_n, y_urban[idx_n]), xytext=(h_n, y_intermediate[idx_n]),
                arrowprops=dict(arrowstyle='->', color=C_REDIST, lw=1.5, mutation_scale=10))
    
    # 标注夜间逻辑
    ax.text(h_n - 0.6, (y_rural[idx_n] + y_intermediate[idx_n])/2, r"$\Delta T_{mean}$", 
            color=C_WARM, ha='right', va='center', fontweight='bold', path_effects=halo)
    ax.text(h_n - 0.6, (y_intermediate[idx_n] + y_urban[idx_n])/2, r"$-\Delta Amp$", 
            color=C_REDIST, ha='right', va='center', fontweight='bold', path_effects=halo)

    # --- 4. 逻辑分解：白天 (Daytime, 14:00) ---
    h_d = 14.2
    idx_d = np.argmin(np.abs(t_fine - h_d))
    
    # 第一步：Background Warming (dm)
    ax.annotate('', xy=(h_d, y_intermediate[idx_d]), xytext=(h_d, y_rural[idx_d]),
                arrowprops=dict(arrowstyle='->', color=C_WARM, lw=1.5, mutation_scale=10))
    # 第二步：Redistribution (da)
    ax.annotate('', xy=(h_d, y_urban[idx_d]), xytext=(h_d, y_intermediate[idx_d]),
                arrowprops=dict(arrowstyle='->', color=C_REDIST, lw=1.5, mutation_scale=10))
    
    # 标注白天逻辑
    ax.text(h_d + 0.6, (y_rural[idx_d] + y_intermediate[idx_d])/2, r"$\Delta T_{mean}$", 
            color=C_WARM, ha='left', va='center', fontweight='bold', path_effects=halo)
    ax.text(h_d + 0.6, (y_intermediate[idx_d] + y_urban[idx_d])/2, r"$\Delta Amp$", 
            color=C_REDIST, ha='left', va='center', fontweight='bold', path_effects=halo)

    # --- 5. 最终响应总结 (放在顶部和底部，避开核心区) ---
    dtx = np.max(y_urban) - np.max(y_rural)
    dtn = np.min(y_urban) - np.min(y_rural)
    
    # 顶部：白天
    ax.text(h_d, ax.get_ylim()[1]*0.95, f"Daytime Response: $\Delta T_x = {dtx:+.1f}$°C", 
            color=C_URBAN, ha='center', fontsize=10, fontweight='bold', path_effects=halo)
    # 底部：夜间
    ax.text(h_n+1, ax.get_ylim()[0]*1.05, f"Nighttime Response: $\Delta T_n = {dtn:+.1f}$°C", 
            color=C_URBAN, ha='left', fontsize=10, fontweight='bold', path_effects=halo)

    # --- 6. 物理公式与装饰 ---
    # 将公式放在左上角留白处
    ax.text(0.05, 0.92, r"$\Delta T_a(t) = \Delta T_{mean} + \Delta Amp \cdot \cos(\omega t - \phi)$", 
            transform=ax.transAxes, fontsize=11, color="#2c3e50", alpha=0.8)

    ax.set_xlabel("Local Solar Time (h)", fontsize=11)
    ax.set_ylabel(" $T_a$ (°C)", fontsize=11)
    ax.set_title("Thermodynamic Decomposition of Urban Heat Stress", loc='left', fontweight='bold', pad=15)
    
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.grid(True, lw=0.4, alpha=0.2, ls='--')
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    
    # 移除多余的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 保存图片
    fpath = os.path.join(output_dir, "plots", "Figure_FFT_Clean_Mechanism.png")
    plt.savefig(fpath, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

def plot_figure_fft_four_lines_clean(all_df, output_dir):
    """
    Nature 风格机制图 (颜色与线型定制版)：
    1. Rural: 灰色虚线
    2. Urban: 红色实线 (Combined)
    3. ΔTmean only: 橙色实线
    4. ΔAmp only: 红色虚线
    """
    # 1. 物理参数设定 (确保 ΔTx < 0)
    m_ref, a1_ref, a2_ref, p1_ref, p2_ref = 12.0, 7.0, 1.0, 3.8, 0.5
    dm, da = 2.0, -3.0  # ΔTmean=+2.0, ΔAmp=-3.0 -> ΔTx = -1.0

    def reconstruct(t, m, a1, p1, a2, p2):
        w = 2 * np.pi / 24
        return m + a1 * np.cos(w * t - p1) + a2 * np.cos(2 * w * t - p2)

    t_fine = np.linspace(0, 24, 1000)
    y_rural    = reconstruct(t_fine, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
    y_amp_only = reconstruct(t_fine, m_ref, a1_ref + da, p1_ref, a2_ref, p2_ref)
    y_m_only   = reconstruct(t_fine, m_ref + dm, a1_ref, p1_ref, a2_ref, p2_ref)
    y_urban    = reconstruct(t_fine, m_ref + dm, a1_ref + da, p1_ref, a2_ref, p2_ref)

    # 2. 绘图设置
    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    
    # 颜色定义
    C_RURAL = "#7f8c8d"   # 灰色
    C_MEAN  = "#e67e22"   # 橙色
    C_AMP   = "#e74c3c"   # 红色
    C_URBAN = "#e74c3c"   # 红色 (最终结果也用红色，但用实线加粗)
    
    halo = [path_effects.withStroke(linewidth=3, foreground='white', alpha=0.9)]

    # 3. 绘制四条线 (按要求修改线型和颜色)
    # Rural: 灰色虚线
    ax.plot(t_fine, y_rural,    color=C_RURAL, lw=2.0, ls="--", label="Rural (Baseline)", zorder=2)
    # ΔTmean only: 橙色实线
    ax.plot(t_fine, y_m_only,   color=C_MEAN,  lw=2.0, ls="-",  label="$\Delta T_{mean} > 0$ only", zorder=3)
    # ΔAmp only: 红色虚线
    ax.plot(t_fine, y_amp_only, color=C_AMP,   lw=1.5, ls="--", label="$\Delta Amp < 0$ only", zorder=1)
    # Urban (Combined): 红色实线 (加粗以示区分)
    ax.plot(t_fine, y_urban,    color=C_URBAN, lw=4.5, ls="-",
            label=r"Urban ($\Delta Amp < 0$ & $\Delta T_{mean} > 0$)", zorder=5)

    # 4. Tmean 水平虚线
    ax.axhline(m_ref, color=C_RURAL, lw=1.2, ls=":", alpha=0.6, zorder=0)
    ax.axhline(m_ref + dm, color=C_MEAN, lw=1.2, ls=":", alpha=0.6, zorder=0)
    
    # 5. 标注 ΔTmean 变化 (红色/橙色箭头表示升温)
    h_m = 1.0
    ax.annotate('', xy=(h_m, m_ref + dm), xytext=(h_m, m_ref),
                arrowprops=dict(arrowstyle='<->', color=C_MEAN, lw=2, mutation_scale=15))
    ax.text(h_m + 0.3, m_ref + dm/2, r"$\Delta T_{mean}$", color=C_MEAN, 
            va='center', fontweight='bold', fontsize=12, path_effects=halo)

    # 6. 分解标注
    h_n, h_d = 4.0, 14.5
    idx_n = np.argmin(np.abs(t_fine - h_n))
    idx_d = np.argmin(np.abs(t_fine - h_d))

    # 夜间: 均值上升 + 振幅平减(谷值抬升) = 双重增温
    ax.annotate('', xy=(h_n, y_urban[idx_n]), xytext=(h_n, y_m_only[idx_n]),
                arrowprops=dict(arrowstyle='->', color=C_AMP, lw=1.5))
    ax.text(h_n-0.6, (y_m_only[idx_n]+y_urban[idx_n])/2, r"$-\Delta Amp$", 
            color=C_AMP, ha='right', va='center', fontweight='bold', path_effects=halo)

    # 白天: 均值上升 + 振幅平减(峰值下降) = 相互抵消
    ax.annotate('', xy=(h_d, y_urban[idx_d]), xytext=(h_d, y_m_only[idx_d]),
                arrowprops=dict(arrowstyle='->', color=C_AMP, lw=1.5))
    ax.text(h_d+0.6, (y_m_only[idx_d]+y_urban[idx_d])/2, r"$\Delta Amp$", 
            color=C_AMP, ha='left', va='center', fontweight='bold', path_effects=halo)

    # 7. 响应总结 (确保不遮挡且不出界)
    dtx = np.max(y_urban) - np.max(y_rural)
    dtn = np.min(y_urban) - np.min(y_rural)
    
    ax.text(14.5, np.max(y_m_only)+1.5, f"Daytime Response: $\Delta T_x = {dtx:+.1f}$°C", 
            color=C_URBAN, ha='center', va='bottom', fontweight='bold', fontsize=11, path_effects=halo)
    
    ax.text(6.0, np.min(y_rural)-2.0, f"Nighttime Response: $\Delta T_n = {dtn:+.1f}$°C", 
            color=C_URBAN, ha='center', va='top', fontweight='bold', fontsize=11, path_effects=halo)

    # 8. 修饰
    ax.set_xlabel("Local Solar Time (h)", fontsize=12)
    ax.set_ylabel(" $T_a$ (°C)", fontsize=12)
    ax.set_title("Thermodynamic Decomposition of Urban Heat Stress", loc='left', fontweight='bold', fontsize=16, pad=20)
    
    ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(np.min(y_rural)-4, np.max(y_m_only)+4.5) # 调整高度防止出界
    
    ax.grid(True, lw=0.4, alpha=0.1, ls=':')
    ax.legend(frameon=False, loc='lower right', fontsize=10)
    
    # 顶部公式
    ax.text(0.95, 0.96, r"$\Delta T_a(t) = \Delta T_{mean} + \Delta Amp \cdot \cos(\omega t - \phi)$", 
            transform=ax.transAxes, fontsize=11, color="#34495e", ha='right', alpha=0.7)

    # 黑边框
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(1.2)

    # 9. 保存
    fpath = os.path.join(output_dir, "plots", "Figure_FFT_Mechanism_ColorsFixed.png")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    plt.savefig(fpath, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

def plot_figure_combined_mechanism_2x22(all_df, output_dir):
    """
    Nature 风格 2x2 组合机制图 (精细标注版)：
    (a) NHW State Space | (b) HW State Space 
    (c) FFT Schematic (标注优化) | (d) ΔHW Statistics
    """

    id_col = next((c for c in ["pair_id", "station_id", "city_id"] if c in all_df.columns), None)
    nhw_plot = all_df[all_df["period"] == "non_heatwave"].dropna(subset=["dTmean", "dAmp1", "dTx"]).copy()
    hw_plot  = all_df[all_df["period"] == "heatwave"].dropna(subset=["dTmean", "dAmp1", "dTx"]).copy()

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 20,      # 代码2是20
        "axes.labelsize": 26, "axes.titlesize": 30,       # 代码2是26/30
        "xtick.labelsize": 24, "ytick.labelsize": 24,     # 代码2是24
        "legend.fontsize": 22, "axes.linewidth": 2.5,     # 代码2是22/2.5
        "pdf.fonttype": 42
    }):
        fig, axes = plt.subplots(2, 2, figsize=(26, 24), dpi=600) # 与代码2对齐
        plt.subplots_adjust(wspace=0.35, hspace=0.45)           # 与代码2对齐

        plt.subplots_adjust(wspace=0.35, hspace=0.45)
        halo = [path_effects.withStroke(linewidth=5, foreground='white', alpha=0.9)]

        # --- (a) & (b) 保持原样 ---
        dtx_all = pd.concat([nhw_plot["dTx"], hw_plot["dTx"]])
        vabs = max(np.nanpercentile(np.abs(dtx_all), 98), 0.1)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        cmap = plt.cm.RdBu_r
        x_all, y_all = pd.concat([nhw_plot["dTmean"], hw_plot["dTmean"]]), pd.concat([nhw_plot["dAmp1"], hw_plot["dAmp1"]])
        xlim, ylim = (np.nanpercentile(x_all, 1)-0.5, np.nanpercentile(x_all, 99)+0.5), (np.nanpercentile(y_all, 1)-0.5, np.nanpercentile(y_all, 99)+0.5)

        for df_p, ax, title, marker in [(nhw_plot, axes[0, 0], "a  Non-Heatwave (NHW)", "o"), (hw_plot, axes[0, 1], "b  Heatwave (HW)", "^")]:
            sc = ax.scatter(df_p["dTmean"], df_p["dAmp1"], c=df_p["dTx"], cmap=cmap, norm=norm, marker=marker, s=100, edgecolors='black', linewidths=0.8, alpha=0.9, zorder=3)
            if len(df_p) > 2:
                X = np.column_stack([df_p["dTmean"], df_p["dAmp1"], np.ones(len(df_p))]); y = df_p["dTx"].values
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                ax.plot(np.linspace(xlim[0], xlim[1], 100), (-coef[0]/coef[1])*np.linspace(xlim[0], xlim[1], 100) + (-coef[2]/coef[1]), 'k--', lw=3, label='$\Delta T_x = 0$')
            ax.axhline(0, color='#cccccc', lw=1.5); ax.axvline(0, color='#cccccc', lw=1.5)
            ax.set_title(title, loc='left', fontweight='bold', pad=20); ax.set_xlabel(r"$\Delta T_{mean}$ (°C)"); ax.set_ylabel(r"$\Delta Amp$ (°C)")
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.legend(frameon=False, loc='upper right')
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
            plt.colorbar(sc, cax=cax).set_label(r"Daytime $\Delta T_x$ (°C)", fontsize=20)

        # ─────────────────────────────────────────────────────────────
        # 子图 (c): FFT Mechanism (标注重排)
        # ─────────────────────────────────────────────────────────────
        ax_c = axes[1, 0]
        C_RURAL, C_MEAN, C_AMP, C_URBAN = "#7f8c8d", "#e67e22", "#e74c3c", "#e74c3c"
        m_ref, a1_ref, a2_ref, p1_ref, p2_ref = 12.0, 7.0, 1.0, 3.8, 0.5
        dm, da = 2.0, -3.0 
        t_f = np.linspace(0, 24, 1000)
        def recon(t, m, a, p, a2, p2): 
            w = 2 * np.pi / 24
            return m + a*np.cos(w*t - p) + a2*np.cos(2*w*t - p2)
        
        y_rural = recon(t_f, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
        y_m_only = recon(t_f, m_ref + dm, a1_ref, p1_ref, a2_ref, p2_ref)
        y_amp_only = recon(t_f, m_ref, a1_ref + da, p1_ref, a2_ref, p2_ref)
        y_urban = recon(t_f, m_ref + dm, a1_ref + da, p1_ref, a2_ref, p2_ref)

        # 线条绘制
        ax_c.axhline(m_ref, color=C_RURAL, lw=1.5, ls=":", alpha=0.5, zorder=0)
        ax_c.axhline(m_ref + dm, color=C_MEAN, lw=1.5, ls=":", alpha=0.5, zorder=0)
        ax_c.plot(t_f, y_rural,    color=C_RURAL, lw=3.0, ls="--", label="Rural (Baseline)", zorder=2)
        ax_c.plot(t_f, y_m_only,   color=C_MEAN,  lw=3.0, ls="-",  label="$\Delta T_{mean} > 0$", zorder=3)
        ax_c.plot(t_f, y_amp_only, color=C_AMP,   lw=2.0, ls="--", label="$\Delta Amp < 0$", zorder=1)
        ax_c.plot(t_f, y_urban,    color=C_URBAN, lw=6.0, ls="-",  label="Urban ($\Delta Amp < 0$ and $\Delta T_{mean} > 0$)", zorder=5)

        # --- 标注 1: ΔTmean (左侧) ---
        ax_c.annotate('', xy=(0.8, m_ref + dm), xytext=(0.8, m_ref), arrowprops=dict(arrowstyle='<->', color=C_MEAN, lw=3))
        ax_c.text(1.0, m_ref + dm/2, r"$\Delta T_{mean}$", color=C_MEAN, fontweight='bold', fontsize=18, va='center', path_effects=halo)
        
        # --- 标注 2: ΔAmp (夜间波谷) ---
        h_n = 4.0; idx_n = np.argmin(np.abs(t_f - h_n))
        ax_c.annotate(
            '',
            xy=(h_n, y_amp_only[idx_n]),      # 指向橙/红色虚线：ΔAmp only
            xytext=(h_n, y_rural[idx_n]),     # 起点：灰色虚线 Rural
            arrowprops=dict(
                arrowstyle='->',
                color=C_AMP,
                lw=2.5
            )
        )

        ax_c.text(
            h_n + 0.5,
            (y_rural[idx_n] + y_amp_only[idx_n]) / 2,
            r"$-\Delta Amp$",
            color=C_AMP,
            ha='left',
            fontweight='bold',
            fontsize=16,
            path_effects=halo
        )        
        # --- 标注 3: ΔTx (白天波峰) ---
        idx_pu = np.argmax(y_urban); t_pu = t_f[idx_pu]; y_pu = y_urban[idx_pu]; y_pr = y_rural[idx_pu]
        ax_c.annotate('', xy=(t_pu, y_pu), xytext=(t_pu, y_pr), arrowprops=dict(arrowstyle='<->', color='black', lw=2.5, zorder=6))
        ax_c.text(t_pu + 2.1, (y_pu + y_pr)/2, f"\n$\Delta T_x = {(y_pu-y_pr):+.1f}$°C", color=C_URBAN, fontweight='bold', fontsize=18, ha='left', va='center', path_effects=halo)

        # --- 标注 4: 公式 (右上角) ---
        ax_c.text(0.95, 0.95, r"$\Delta T_a(t) = \Delta T_{mean} + \Delta Amp \cdot \cos(\omega t - \phi)$", transform=ax_c.transAxes, fontsize=17, color="#34495e", ha='right', fontweight='bold', alpha=0.8, path_effects=halo)

        ax_c.set_title("c  Theoretical Mechanism", loc='left', fontweight='bold', pad=20)
        ax_c.set_xlabel("Local Solar Time (h)"); ax_c.set_ylabel(" $T_a$ (°C)")
        ax_c.set_xlim(0, 24); ax_c.set_xticks([0, 6, 12, 18, 24]); ax_c.set_ylim(np.min(y_rural)-2, np.max(y_m_only)+4)
        ax_c.legend(frameon=False, loc='lower right', fontsize=14)

        # --- (d) ΔHW Statistics (保持不变) ---
        ax_d = axes[1, 1]
        metrics = [("dTmean", r"$\Delta T_a$"), ("dAmp1", r"$\Delta Amp$"), ("dTx", r"$\Delta T_x$"), ("dTn", r"$\Delta T_n$")]
        merged = pd.merge(nhw_plot[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]], hw_plot[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]], on=[id_col, "group"], suffixes=('_nhw', '_hw'))
        if not merged.empty:
            x_pos = np.arange(len(metrics)); width = 0.35
            for offset, group, color in [(-width/2, "UHI", "#d62728"), (width/2, "UCI", "#1f77b4")]:
                g_data = merged[merged["group"] == group]
                if g_data.empty: continue
                diffs = [(g_data[f"{m}_hw"] - g_data[f"{m}_nhw"]).dropna() for m, _ in metrics]
                means, sems = [d.mean() for d in diffs], [d.sem() for d in diffs]
                pvals = [stats.ttest_1samp(d, 0).pvalue if len(d)>1 else 1.0 for d in diffs]
                ax_d.bar(x_pos + offset, means, width, yerr=sems, color=color, alpha=0.85, edgecolor='black', capsize=6, label=f"{group} (n={len(g_data)})")
                # for i, p in enumerate(pvals):
                    # star = stars_from_p(p)
                    # ax_d.text(i + offset, means[i] + (sems[i] if means[i]>0 else -sems[i]*3.5), star, ha='center', fontweight='bold', fontsize=24)
            ax_d.axhline(0, color='black', lw=2.0); ax_d.set_xticks(x_pos); ax_d.set_xticklabels([l for _, l in metrics])
            ax_d.set_ylabel("HW$-$NHW urban–rural response (°C)"); ax_d.set_title("d  Heatwave-induced Changes", loc='left', fontweight='bold', pad=20); ax_d.legend(frameon=False, loc='lower right')

        for ax in axes.flatten():
            for side in ["top", "right", "bottom", "left"]: ax.spines[side].set_linewidth(2.0)

    fpath = os.path.join(output_dir, "plots", "Figure_Combined_Mechanism_2x2_Final_V3.png")
    fig.savefig(fpath, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved Final Version 3: {fpath}")

def plot_combined_figure_dynamics_consistent2(all_df, output_dir):
    """
    Nature 风格 2x2 动力学分析图:
    (a) 坐标轴固定 [-2, 4], 原始向量背景 + 网格均值趋势 + Hexbin 密度
    (b) ΔTn 机制分解: ΔTmean, -ΔAmp, residual (严格对齐原函数)
    (c) 能量补偿散点图 (残差回归 + 95% CI)
    (d) 热滞回环 (严格复刻 FFT 高精度相位滞后算法)
    """

    # --- 内部辅助函数 (严格复刻原函数逻辑) ---
    def _get_precise_peak_time(signal):
        vals = np.asarray(signal, dtype=float)
        if np.all(np.isnan(vals)): return np.nan
        vals = np.where(np.isfinite(vals), vals, np.nanmean(vals))
        sig_detrend = vals - np.mean(vals)
        fft_vals = np.fft.fft(sig_detrend)
        t_fine = np.linspace(0, 24, 2400, endpoint=False)
        recon = np.zeros_like(t_fine)
        for k in [1, 2]: 
            recon += np.real(fft_vals[k] * np.exp(1j * 2 * np.pi * k * t_fine / 24))
        return t_fine[np.argmax(recon)]

    def _phase_lag_hours(x_curve, y_curve):
        t_peak_x = _get_precise_peak_time(x_curve)
        t_peak_y = _get_precise_peak_time(y_curve)
        if pd.isna(t_peak_x) or pd.isna(t_peak_y): return np.nan
        return (t_peak_y - t_peak_x) % 24

    def _loop_area(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 4: return np.nan
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _fit_line_ci(x, y, x_grid):
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5: return [np.nan]*5
        lr = stats.linregress(x, y)
        y_fit = lr.intercept + lr.slope * x_grid
        y_pred = lr.intercept + lr.slope * x
        n = len(x)
        s_err = np.sqrt(np.sum((y - y_pred)**2) / (n - 2))
        t_val = stats.t.ppf(0.975, df=n-2)
        ci = t_val * s_err * np.sqrt(1/n + (x_grid - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        return lr.slope, lr.intercept, lr.pvalue, y_fit, ci

    # --- 数据准备 ---
    id_col = next((c for c in ["pair_id", "station_id", "city_id"] if c in all_df.columns), None)
    nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
    hw_df  = all_df[all_df["period"] == "heatwave"].copy()
    paired = pd.merge(nhw_df[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
                      hw_df[[id_col, "dTmean", "dAmp1", "dTx", "dTn"]],
                      on=id_col, suffixes=("_nhw", "_hw"), how="inner").dropna()

    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]
    
    # --- 样式设置 (同步 Combined Mechanism 图的风格) ---
    COLOR_UHI, COLOR_UCI = "#d62728", "#1f77b4"
    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 20,
        "axes.labelsize": 26, "axes.titlesize": 30,
        "xtick.labelsize": 24, "ytick.labelsize": 24,
        "legend.fontsize": 22, "axes.linewidth": 2.5, "pdf.fonttype": 42
    }):
        fig, axes = plt.subplots(2, 2, figsize=(26, 24), dpi=600)
        plt.subplots_adjust(wspace=0.35, hspace=0.45)

        # ============================================================
        # (a) State migration: 严格复刻原函数逻辑 + 新增原始向量底图 (无 Hexbin)
        # ============================================================
        ax_a = axes[0, 0]
        
        # 1. 计算坐标范围 (严格复刻原函数百分位计算)
        all_x = np.concatenate([paired["dTmean_nhw"].values, paired["dTmean_hw"].values])
        all_y = np.concatenate([paired["dAmp1_nhw"].values, paired["dAmp1_hw"].values])
        x_min, x_max = np.nanpercentile(all_x, [1, 99])
        y_min, y_max = np.nanpercentile(all_y, [1, 99])
        x_pad = (x_max - x_min) * 0.16
        y_pad = (y_max - y_min) * 0.16
        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        # 2. 定义网格向量辅助函数 (严格复刻原函数)
        def _binned_vectors_local(g, nx=11, ny=11, min_count=2):
            g = g.copy()
            g["dx"] = g["dTmean_hw"] - g["dTmean_nhw"]
            g["dy"] = g["dAmp1_hw"]  - g["dAmp1_nhw"]
            g = g[np.isfinite(g["dTmean_nhw"]) & np.isfinite(g["dAmp1_nhw"]) & 
                  np.isfinite(g["dx"]) & np.isfinite(g["dy"])].copy()
            if len(g) == 0: return pd.DataFrame()
            x_bins = np.linspace(xlim[0], xlim[1], nx + 1)
            y_bins = np.linspace(ylim[0], ylim[1], ny + 1)
            g["xb"] = pd.cut(g["dTmean_nhw"], bins=x_bins, labels=False, include_lowest=True)
            g["yb"] = pd.cut(g["dAmp1_nhw"],  bins=y_bins, labels=False, include_lowest=True)
            out = g.dropna(subset=["xb", "yb"]).groupby(["xb", "yb"], as_index=False).agg(
                     x0=("dTmean_nhw", "mean"), y0=("dAmp1_nhw", "mean"),
                     dx=("dx", "mean"), dy=("dy", "mean"), n=("dx", "size"))
            return out[out["n"] >= min_count].copy()

        # 3. 【新增】绘制原始微小向量底图 (替代 Hexbin)
        for grp_name, color in [("UHI", COLOR_UHI), ("UCI", COLOR_UCI)]:
            g_raw = paired[paired["group"] == grp_name]
            ax_a.quiver(
                g_raw["dTmean_nhw"], g_raw["dAmp1_nhw"],
                g_raw["dTmean_hw"] - g_raw["dTmean_nhw"], g_raw["dAmp1_hw"] - g_raw["dAmp1_nhw"],
                angles="xy", scale_units="xy", scale=1,
                color=color, alpha=0.12, width=0.0012, zorder=1
            )

        # 4. 绘制网格均值向量 (严格复刻原函数参数: width, headwidth, headlength等)
        for group_name, color in [("UHI", COLOR_UHI), ("UCI", COLOR_UCI)]:
            g = paired[paired["group"] == group_name].copy()
            if len(g) == 0: continue
            vec = _binned_vectors_local(g, nx=11, ny=11, min_count=2)
            if len(vec) == 0: continue

            ax_a.quiver(
                vec["x0"], vec["y0"], vec["dx"], vec["dy"],
                angles="xy", scale_units="xy", scale=1,
                color=color, alpha=0.78,
                width=0.0042,      # 严格对齐原图粗细
                headwidth=3.2,     # 严格对齐箭头形状
                headlength=4.2,
                headaxislength=3.6,
                label=f"{group_name} mean flow",
                zorder=3,
            )

        # 5. 绘制约束线 (严格复刻原函数参数: lw=1.0, alpha=0.72)
        x_line_range = np.linspace(xlim[0], xlim[1], 200)
        ax_a.plot(x_line_range, -x_line_range, linestyle="--", color="black", 
                 lw=1.5, alpha=0.72, zorder=2, label=r"$\Delta T_x=0$ isoline")

        # 6. 文本说明与轴设置 (严格复刻原函数文字与样式)
        ax_a.text(0.03, 0.97, "Background: Original vectors\nArrows: mean HW−NHW flow",
                 transform=ax_a.transAxes, ha="left", va="top", fontsize=16,
                 bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2))

        ax_a.axhline(0, color="#999999", linestyle=":", lw=0.8, zorder=1)
        ax_a.axvline(0, color="#999999", linestyle=":", lw=0.8, zorder=1)
        ax_a.set_xlim(xlim); ax_a.set_ylim(ylim)
        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")
        ax_a.set_title("a  State migration trajectory", loc="left", fontweight="bold", pad=20)
        ax_a.legend(frameon=False, loc="best", fontsize=15)
        ax_a.grid(True, lw=0.25, alpha=0.16)

        # ============================================================
        # (b) Mechanism decomposition: 四柱同轴, 红蓝配色, 带花纹
        # ============================================================
        ax_b = axes[0, 1]
        
        # 1. 定义物理分解算法 (严格保持一致)
        def get_decomp_final(grp, target):
            g = paired[paired["group"] == grp]
            dm = (g["dTmean_hw"] - g["dTmean_nhw"]).mean()
            da = (g["dAmp1_hw"] - g["dAmp1_nhw"]).mean()
            total = (g[f"{target}_hw"] - g[f"{target}_nhw"]).mean()
            
            # 物理核心：Tx受正向Amp影响，Tn受负向Amp影响
            c_mean = dm
            c_amp  = da if target == "dTx" else -da
            c_res  = total - (c_mean + c_amp)
            return c_mean, c_amp, c_res, total

        # 2. 绘图配置
        cases = [("UHI","dTx"), ("UHI","dTn"), ("UCI","dTx"), ("UCI","dTn")]
        x_pos = np.arange(len(cases))
        bw = 0.52  # 细柱体宽度

        for i, (grp, tgt) in enumerate(cases):
            cm, ca, cr, ct = get_decomp_final(grp, tgt)
            
            # 颜色设置：UHI 统一红色系，UCI 统一蓝色系
            if grp == "UHI":
                base_color = "#d62728"  # 深红
                light_color = "#ff9999" # 浅红
            else:
                base_color = "#1f77b4"  # 深蓝
                light_color = "#a0cbe8" # 浅蓝

            # A. 绘制 ΔTmean 贡献 (深色, 实心)
            ax_b.bar(i, cm, bw, color=base_color, edgecolor='black', lw=1.8, 
                     label=r"$\Delta T_{mean}$ contribution" if i==0 else "")
            
            # B. 绘制 ±ΔAmp 影响 (浅色, 带斜纹)
            ax_b.bar(i, ca, bw, bottom=cm, color=light_color, hatch='////', edgecolor='black', lw=1.8,
                     label=r"$\pm\Delta Amp$ effect" if i==0 else "")
            
            # C. 绘制 Residual/Shape (灰色, 半透明)
            ax_b.bar(i, cr, bw, bottom=cm+ca, color='#aaaaaa', alpha=0.6, edgecolor='black', lw=1.8,
                     label="Shape/Residual" if i==0 else "")
            
            # D. 绘制 Observed Change (黑色菱形)
            ax_b.scatter(i, ct, color='black', marker='D', s=180, zorder=5, 
                         label="Observed Total" if i==0 else "")

        # 3. 装饰与图例 (针对 26x24 大画幅优化)
        ax_b.axhline(0, color='black', lw=2)
        ax_b.set_xticks(x_pos)
        # 横轴标签：UHI R ΔTx / UHI R ΔTn / UCI R ΔTx / UCI R ΔTn
        ax_b.set_xticklabels(["UHI R\n$\Delta T_x$", "UHI R\n$\Delta T_n$", "UCI R\n$\Delta T_x$", "UCI R\n$\Delta T_n$"])
        # 纵轴标签：RΔTa
        ax_b.set_ylabel("$R_a$")
        # 子图b 纵坐标范围（自己改数字：最小值, 最大值）
        ax_b.set_ylim(-0.1, 0.7) 
        ax_b.set_title("b  Mechanism decomposition", loc='left', fontweight='bold', pad=20)
        
        # 调整图例显示，确保能看清花纹
        ax_b.legend(frameon=False, loc='upper left', ncol=2, fontsize=16, handleheight=2, handlelength=3)
        ax_b.grid(True, axis='y', lw=0.3, alpha=0.2)

        # ─────────────────────────────────────────────────────────────
        # (c) Compensation Scatter: 残差回归 + 95% CI
        # ─────────────────────────────────────────────────────────────
        ax_c = axes[1, 0]
        p_df = paired.copy()
        p_df["dx_res"] = (p_df["dTx_hw"]-p_df["dTx_nhw"]) - (p_df["dTmean_hw"]-p_df["dTmean_nhw"])
        p_df["dn_res"] = (p_df["dTn_hw"]-p_df["dTn_nhw"]) - (p_df["dTmean_hw"]-p_df["dTmean_nhw"])
        p_df["d_amp"] = p_df["dAmp1_hw"] - p_df["dAmp1_nhw"]
        
        sc = ax_c.scatter(p_df["dx_res"], p_df["dn_res"], c=p_df["d_amp"], cmap='RdBu_r', 
                          norm=TwoSlopeNorm(vcenter=0), s=130, alpha=0.7, edgecolors='black', lw=0.8)
        
        x_grid = np.linspace(p_df["dx_res"].min(), p_df["dx_res"].max(), 100)
        slope, inter, p_v, y_fit, ci = _fit_line_ci(p_df["dx_res"].values, p_df["dn_res"].values, x_grid)
        ax_c.plot(x_grid, y_fit, 'k-', lw=4, label=f'Fit (slope={slope:.2f})')
        ax_c.fill_between(x_grid, y_fit-ci, y_fit+ci, color='black', alpha=0.15)
        ax_c.plot(x_grid, -x_grid, 'k--', lw=2, alpha=0.5, label='Theoretical (-1)')
        
        ax_c.set_xlabel(r"$R_x$ (°C)")
        ax_c.set_ylabel(r"$R_n$ (°C)")
        ax_c.set_title("c  Response compensation linkage", loc='left', fontweight='bold', pad=20)
        ax_c.legend(frameon=False)
        cax = inset_axes(ax_c, width="3%", height="40%", loc="lower left", bbox_to_anchor=(0.05, 0.08, 1, 1), bbox_transform=ax_c.transAxes)
        plt.colorbar(sc, cax=cax).set_label(r"$R_{amp}$ (°C)")

        # ─────────────────────────────────────────────────────────────
        # (d) Thermal hysteresis: 严格复刻高精度算法
        # ─────────────────────────────────────────────────────────────
        ax_d = axes[1, 1]
        style_map = {("UHI", "non_heatwave"): (COLOR_UHI, "--", "UHI NHW"),
                     ("UHI", "heatwave"):     (COLOR_UHI, "-",  "UHI HW"),
                     ("UCI", "non_heatwave"): (COLOR_UCI, "--", "UCI NHW"),
                     ("UCI", "heatwave"):     (COLOR_UCI, "-",  "UCI HW")}

        for (grp, prd), (color, ls, lab) in style_map.items():
            sub = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)]
            if sub.empty: continue
            x_curve = np.nanmean(sub[r_cols].values, axis=0)
            y_curve = np.nanmean(sub[u_cols].values - sub[r_cols].values, axis=0)
            
            area = _loop_area(x_curve, y_curve)
            lag = _phase_lag_hours(x_curve, y_curve)
            
            ax_d.plot(x_curve, y_curve, color=color, ls=ls, lw=4, alpha=0.8, label=f"{lab}: area={area:.2f}, lag={lag:.2f}h")
            for i in [5, 11, 17]: # 方向箭头
                ax_d.annotate("", xy=(x_curve[(i+1)%24], y_curve[(i+1)%24]), xytext=(x_curve[i], y_curve[i]),
                              arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
            for h in [0, 6, 12, 18]: # 关键时刻打点
                ax_d.scatter(x_curve[h], y_curve[h], s=70, color=color, zorder=5)
                ax_d.text(x_curve[h], y_curve[h], f"{h:02d}", fontsize=14, color=color, fontweight='bold', ha='left', va='bottom')

        ax_d.set_xlabel(r"$T_{a,r}$ (°C)")
        ax_d.set_ylabel(r"$\Delta T_a$ (°C)")
        # 子图d 横纵坐标范围（自己改数字）
        ax_d.set_ylim(-1.5, 2.3)    # 纵轴：城市强度范围
        ax_d.set_title("d  Thermal hysteresis loops", loc='left', fontweight='bold', pad=20)
        ax_d.legend(frameon=False, loc='lower right', fontsize=15)

        for ax in axes.flatten():
            for side in ["top", "right", "bottom", "left"]: ax.spines[side].set_linewidth(2.5)

    fpath = os.path.join(output_dir, "plots", "Figure_Dynamics_Consistent_Final.png")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    fig.savefig(fpath, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved Final Consistent Dynamics Figure: {fpath}")



def _bootstrap_mean_ci(values, n_boot=1000, ci=95, rng=None):
    """
    Station-pair bootstrap confidence interval for the mean.

    Resampling unit: station pair.
    This returns the empirical mean and percentile bootstrap CI.
    """
    vals = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
    if len(vals) == 0:
        return np.nan, np.nan, np.nan

    mean_val = float(np.mean(vals))
    if len(vals) == 1:
        return mean_val, mean_val, mean_val

    if rng is None:
        rng = np.random.default_rng(20260529)

    boot_means = np.empty(n_boot, dtype=float)
    chunk_size = 1000
    for start in range(0, n_boot, chunk_size):
        end = min(start + chunk_size, n_boot)
        idx = rng.integers(0, len(vals), size=(end - start, len(vals)))
        boot_means[start:end] = vals[idx].mean(axis=1)

    alpha = (100 - ci) / 2.0
    ci_low, ci_high = np.percentile(boot_means, [alpha, 100 - alpha])
    return mean_val, float(ci_low), float(ci_high)

def _bootstrap_paired_ratio_pct(
        numerator,
        denominator,
        n_boot=1000,
        ci=95,
        rng=None,
        denominator_eps=1e-12,
):
    """
    Station-pair bootstrap CI for a ratio of group means:

        percentage = 100 × mean(numerator) / mean(denominator)

    numerator and denominator must come from the same matched station pairs.
    """
    tmp = pd.DataFrame({
        "numerator": pd.to_numeric(numerator, errors="coerce"),
        "denominator": pd.to_numeric(denominator, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) == 0:
        return np.nan, np.nan, np.nan

    num = tmp["numerator"].to_numpy(dtype=float)
    den = tmp["denominator"].to_numpy(dtype=float)

    mean_num = float(np.mean(num))
    mean_den = float(np.mean(den))

    if abs(mean_den) <= denominator_eps:
        return np.nan, np.nan, np.nan

    ratio_pct = 100.0 * mean_num / mean_den

    if len(tmp) == 1:
        return ratio_pct, ratio_pct, ratio_pct

    if rng is None:
        rng = np.random.default_rng(20260529)

    boot_ratio = np.full(n_boot, np.nan, dtype=float)

    chunk_size = 1000
    for start in range(0, n_boot, chunk_size):
        end = min(start + chunk_size, n_boot)

        # 同一索引同时重采样 Ramp 和 RTn，保持 station-pair 配对结构
        idx = rng.integers(
            0,
            len(tmp),
            size=(end - start, len(tmp)),
        )

        boot_num = num[idx].mean(axis=1)
        boot_den = den[idx].mean(axis=1)

        valid = np.abs(boot_den) > denominator_eps

        chunk_ratio = np.full(end - start, np.nan, dtype=float)
        chunk_ratio[valid] = (
            100.0 * boot_num[valid] / boot_den[valid]
        )

        boot_ratio[start:end] = chunk_ratio

    boot_ratio = boot_ratio[np.isfinite(boot_ratio)]

    if len(boot_ratio) == 0:
        return ratio_pct, np.nan, np.nan

    alpha = (100.0 - ci) / 2.0
    ci_low, ci_high = np.percentile(
        boot_ratio,
        [alpha, 100.0 - alpha],
    )

    return ratio_pct, float(ci_low), float(ci_high)

def plot_supplement_pair_level_hw_nhw_distribution(all_df, output_dir):
    """
    Supplementary module:
    Pair-level distribution of HW-NHW urban-rural responses.

    This SI figure complements Figure_Combined_Mechanism_2x2_Nature panel d.
    The main figure uses station-pair bootstrap 95% confidence intervals for
    the group mean. This SI module reports the full pair-level distribution,
    including SD and IQR, to show cross-city heterogeneity.
    """
    id_col = next((c for c in ["pair_id", "station_id", "city_id"] if c in all_df.columns), None)
    if id_col is None:
        raise ValueError("No valid ID column found. Expected one of: pair_id, station_id, city_id.")

    required_cols = ["period", "group", "dTmean", "dAmp1", "dTx", "dTn"]
    missing = [c for c in required_cols if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    nhw = (
        all_df[all_df["period"] == "non_heatwave"]
        [[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]]
        .copy()
    )
    hw = (
        all_df[all_df["period"] == "heatwave"]
        [[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]]
        .copy()
    )

    merged = pd.merge(
        nhw, hw,
        on=[id_col, "group"],
        suffixes=("_nhw", "_hw")
    )

    if merged.empty:
        print("  [SI pair-level distribution] No paired NHW-HW data, skipping.")
        return

    metric_specs = [
        ("dTmean", r"$R_{mean}$", "Rmean"),
        ("dAmp1",  r"$R_{amp}$",  "Ramp"),
        ("dTx",    r"$R_x$",      "Rx"),
        ("dTn",    r"$R_n$",      "Rn"),
    ]

    long_rows = []
    summary_rows = []
    rng_summary = np.random.default_rng(20260529)

    for raw_metric, latex_label, plain_label in metric_specs:
        diff_col = f"{raw_metric}_response"
        merged[diff_col] = merged[f"{raw_metric}_hw"] - merged[f"{raw_metric}_nhw"]

        for group in ["UHI", "UCI"]:
            vals = (
                merged.loc[merged["group"] == group, [id_col, diff_col]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[diff_col])
                .copy()
            )
            vals[diff_col] = vals[diff_col].astype(float)

            for _, row in vals.iterrows():
                long_rows.append({
                    id_col: row[id_col],
                    "group": group,
                    "metric": plain_label,
                    "metric_label": latex_label,
                    "response": float(row[diff_col]),
                })

            arr = vals[diff_col].values
            if len(arr) == 0:
                summary_rows.append({
                    "group": group,
                    "metric": plain_label,
                    "n": 0,
                    "mean": np.nan,
                    "bootstrap_ci_low": np.nan,
                    "bootstrap_ci_high": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "iqr": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                })
                continue

            mean_val, ci_low, ci_high = _bootstrap_mean_ci(arr, rng=rng_summary)
            p25 = float(np.percentile(arr, 25))
            p75 = float(np.percentile(arr, 75))
            summary_rows.append({
                "group": group,
                "metric": plain_label,
                "n": int(len(arr)),
                "mean": mean_val,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
                "median": float(np.median(arr)),
                "p25": p25,
                "p75": p75,
                "iqr": p75 - p25,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            })

    # ============================================================
    # Additional summary:
    # Ramp / RTn percentage with station-pair bootstrap 95% CI
    #
    # Ramp = dAmp1_HW - dAmp1_NHW
    # RTn  = dTn_HW   - dTn_NHW = Rn
    #
    # percentage = 100 × mean(Ramp) / mean(RTn)
    # ============================================================
    rng_ratio = np.random.default_rng(20260529)

    for group in ["UHI", "UCI"]:
        ratio_data = (
            merged.loc[
                merged["group"] == group,
                [id_col, "dAmp1_response", "dTn_response"],
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["dAmp1_response", "dTn_response"])
            .copy()
        )

        if len(ratio_data) == 0:
            summary_rows.append({
                "group": group,
                "metric": "Ramp_over_RTn_pct",
                "n": 0,
                "mean": np.nan,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "std": np.nan,
                "median": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "iqr": np.nan,
                "min": np.nan,
                "max": np.nan,
            })
            continue

        ratio_pct, ratio_ci_low, ratio_ci_high = (
            _bootstrap_paired_ratio_pct(
                numerator=-ratio_data["dAmp1_response"],
                denominator=ratio_data["dTn_response"],
                n_boot=1000,
                ci=95,
                rng=rng_ratio,
            )
        )

        summary_rows.append({
            "group": group,
            "metric": "Ramp_over_RTn_pct",
            "n": int(len(ratio_data)),
            "mean": ratio_pct,
            "bootstrap_ci_low": ratio_ci_low,
            "bootstrap_ci_high": ratio_ci_high,

            # 这是组均值之比，不是 pair-level ratio 分布，
            # 因此下面这些分布统计不适用
            "std": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        })

    dist_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)

    out_dir = os.path.join(output_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(
        out_dir,
        "Supplement_pair_level_HW_NHW_distribution_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True, dpi=600)

        group_order = ["UHI", "UCI"]
        colors = {
            "UHI": "#b2182b",
            "UCI": "#2166ac",
        }
        rng_plot = np.random.default_rng(20260529)

        for ax, (_, latex_label, plain_label) in zip(axes, metric_specs):
            data_for_violin = []
            positions = []

            for i, group in enumerate(group_order, start=1):
                vals = (
                    dist_df[
                        (dist_df["group"] == group) &
                        (dist_df["metric"] == plain_label)
                    ]["response"]
                    .dropna()
                    .values
                )
                data_for_violin.append(vals)
                positions.append(i)

            parts = ax.violinplot(
                data_for_violin,
                positions=positions,
                widths=0.70,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body, group in zip(parts["bodies"], group_order):
                body.set_facecolor(colors[group])
                body.set_edgecolor("black")
                body.set_alpha(0.28)
                body.set_linewidth(0.6)

            bp = ax.boxplot(
                data_for_violin,
                positions=positions,
                widths=0.32,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.0),
                boxprops=dict(linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
            )

            for patch, group in zip(bp["boxes"], group_order):
                patch.set_facecolor("white")
                patch.set_edgecolor(colors[group])
                patch.set_alpha(0.95)

            for i, group in enumerate(group_order, start=1):
                vals = (
                    dist_df[
                        (dist_df["group"] == group) &
                        (dist_df["metric"] == plain_label)
                    ]["response"]
                    .dropna()
                    .values
                )
                if len(vals) == 0:
                    continue
                jitter = rng_plot.normal(0, 0.035, size=len(vals))
                ax.scatter(
                    np.full(len(vals), i) + jitter,
                    vals,
                    s=5,
                    color=colors[group],
                    alpha=0.28,
                    linewidths=0,
                    zorder=3,
                )

            ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.set_xticks(positions)
            ax.set_xticklabels(group_order)
            ax.set_title(latex_label, fontweight="bold", pad=4)
            ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.35)

            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(0.7)
                ax.spines[side].set_color("black")

        axes[0].set_ylabel("Pair-level HW$-$NHW response (°C)")

        handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=colors["UHI"], markeredgecolor="none",
                   markersize=5, label="UHI"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=colors["UCI"], markeredgecolor="none",
                   markersize=5, label="UCI"),
        ]
        axes[-1].legend(handles=handles, frameon=False, loc="upper right")

        fig.tight_layout(w_pad=0.8)

        png_path = os.path.join(out_dir, "Supplement_pair_level_HW_NHW_distribution.png")
        pdf_path = os.path.join(out_dir, "Supplement_pair_level_HW_NHW_distribution.pdf")

        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved SI pair-level distribution: {png_path}")
    print(f"  Saved SI pair-level distribution PDF: {pdf_path}")
    print(f"  Saved SI summary table: {summary_path}")

def plot_figure_combined_mechanism_2x2(all_df, output_dir):
    """
    Nature-style 2x2 combined mechanism figure (v22-aligned).

    a  Mean–amplitude framework for daytime UHI/UCI transition
    b  Mean–amplitude phase space during NHW periods
    c  Mean–amplitude phase space during HW periods
    d  Heatwave-induced mean, amplitude, daytime and nighttime responses
    """

    id_col = next((c for c in ["pair_id", "station_id", "city_id"] if c in all_df.columns), None)
    if id_col is None:
        raise ValueError("No valid ID column found. Expected one of: pair_id, station_id, city_id.")

    required_cols = ["period", "group", "dTmean", "dAmp1", "dTx", "dTn"]
    missing = [c for c in required_cols if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    nhw_plot = (
        all_df[all_df["period"] == "non_heatwave"]
        .dropna(subset=["dTmean", "dAmp1", "dTx"]).copy()
    )
    hw_plot = (
        all_df[all_df["period"] == "heatwave"]
        .dropna(subset=["dTmean", "dAmp1", "dTx"]).copy()
    )
    if nhw_plot.empty or hw_plot.empty:
        raise ValueError("NHW or HW data is empty after dropping NaNs.")

    SUBTITLE_FONTSIZE = 24
    LABEL_FONTSIZE = 32
    AX_LABEL_FONTSIZE = 26
    TICK_FONTSIZE = 24
    LEGEND_FONTSIZE = 22
    ANNOT_FONTSIZE = 20

    FRAME_LW = 2.5
    TICK_W = 1.6
    TICK_LEN = 6

    PANEL_WSPACE = 0.32
    PANEL_HSPACE = 0.40

    def add_panel_label(ax, label, subtitle, pad=18):
        ax.set_title(
            subtitle,
            loc="center",
            fontweight="bold",
            pad=pad,
            fontsize=SUBTITLE_FONTSIZE
        )
        ax.text(
            0.0, 1.04, label,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontweight="bold",
            fontsize=LABEL_FONTSIZE
        )

    def style_full_box(ax, lw=FRAME_LW):
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(lw)

        ax.tick_params(
            axis="both",
            which="major",
            top=False,
            right=False,
            direction="out",
            width=TICK_W,
            length=TICK_LEN,
            labelsize=TICK_FONTSIZE
        )

    def integer_cb_ticks(vabs_val):
        lo = math.ceil(-vabs_val)
        hi = math.floor(vabs_val)
        return np.arange(lo, hi + 1) if hi >= lo else None

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 20,
        "axes.labelsize": 26,
        "axes.titlesize": SUBTITLE_FONTSIZE,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "axes.linewidth": 2.5,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }):

        fig, axes = plt.subplots(2, 2, figsize=(26, 24), dpi=600)
        fig.subplots_adjust(
            left=0.075,
            right=0.94,
            bottom=0.07,
            top=0.92,
            wspace=PANEL_WSPACE,
            hspace=PANEL_HSPACE
        )

        ax_a = axes[0, 0]
        ax_b = axes[0, 1]
        ax_c = axes[1, 0]
        ax_d = axes[1, 1]

        halo = [path_effects.withStroke(linewidth=5, foreground="white", alpha=0.9)]

        dtx_all = pd.concat([nhw_plot["dTx"], hw_plot["dTx"]], ignore_index=True)
        x_all = pd.concat([nhw_plot["dTmean"], hw_plot["dTmean"]], ignore_index=True)
        y_all = pd.concat([nhw_plot["dAmp1"], hw_plot["dAmp1"]], ignore_index=True)

        vabs = max(np.nanpercentile(np.abs(dtx_all), 98), 0.1)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        cmap = plt.cm.RdBu_r
        cb_ticks = integer_cb_ticks(vabs)

        xlim = (np.nanpercentile(x_all, 1) - 0.5,
                np.nanpercentile(x_all, 99) + 0.5)
        ylim = (np.nanpercentile(y_all, 1) - 0.5,
                np.nanpercentile(y_all, 99) + 0.5)

        # =================================================================
        # Panels b & c
        # =================================================================
        phase_panels = [
            (nhw_plot, ax_b, "b",
             "Mean–amplitude state space\nduring NHW periods", "o"),
            (hw_plot, ax_c, "c",
             "Mean–amplitude state space\nduring HW periods", "^")
        ]

        for df_p, ax, lbl, subtitle, marker in phase_panels:
            sc = ax.scatter(
                df_p["dTmean"], df_p["dAmp1"],
                c=df_p["dTx"], cmap=cmap, norm=norm,
                marker=marker, s=100,
                edgecolors="black", linewidths=0.8,
                alpha=0.9, zorder=3
            )

            if len(df_p) > 2:
                X = np.column_stack([
                    df_p["dTmean"].values,
                    df_p["dAmp1"].values,
                    np.ones(len(df_p))
                ])
                y = df_p["dTx"].values
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                if np.isfinite(coef[1]) and abs(coef[1]) > 1e-8:
                    xx = np.linspace(xlim[0], xlim[1], 200)
                    yy = (-coef[0] / coef[1]) * xx + (-coef[2] / coef[1])
                    ax.plot(xx, yy, color="black", linestyle="--", lw=3,
                            label=r"$\Delta T_x = 0$", zorder=4)

            ax.axhline(0, color="#d0d0d0", lw=1.5, zorder=1)
            ax.axvline(0, color="#d0d0d0", lw=1.5, zorder=1)
            ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
            ax.set_ylabel(r"$\Delta Amp$ (°C)")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            leg = ax.legend(frameon=False, loc="upper right",
                            handlelength=1.8, borderaxespad=0.4)
            if leg is not None:
                for line in leg.get_lines():
                    line.set_linewidth(2.5)

            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
            cb = plt.colorbar(sc, cax=cax)
            cb.set_label(r"Daytime $\Delta T_x$ (°C)", fontsize=22)
            cb.ax.tick_params(labelsize=18, width=1.4, length=5)
            if cb_ticks is not None and len(cb_ticks) >= 2:
                cb.set_ticks(cb_ticks)

            add_panel_label(ax, lbl, subtitle)

        # =================================================================
        # Panel a
        # =================================================================
        m_ref, a1_ref, a2_ref, p1_ref, p2_ref = 12.0, 7.0, 1.0, 3.8, 0.5
        w = 2 * np.pi / 24
        t_f = np.linspace(0, 24, 1000)

        def recon(t, m, a, p, a2, p2):
            return m + a * np.cos(w * t - p) + a2 * np.cos(2 * w * t - p2)

        y_rural_ref = recon(t_f, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
        idx_tx = np.argmax(y_rural_ref)
        t_tx = t_f[idx_tx]
        Kx = np.cos(w * t_tx - p1_ref)

        x_grid = np.linspace(xlim[0], xlim[1], 320)
        y_grid = np.linspace(ylim[0], ylim[1], 320)
        XX, YY = np.meshgrid(x_grid, y_grid)
        ZZ = XX + Kx * YY
        ZZ_disp = np.clip(ZZ, -vabs, vabs)
        levels = np.linspace(-vabs, vabs, 25)

        cf = ax_a.contourf(
            XX, YY, ZZ_disp, levels=levels,
            cmap=cmap, norm=norm, alpha=0.5, zorder=0
        )

        # ΔTx = 0 虚线加粗，并在标签位置断开
        x_label_frac = 0.60
        x_label_data = xlim[0] + x_label_frac * (xlim[1] - xlim[0])
        line_gap_half = 0.62

        if np.isfinite(Kx) and abs(Kx) > 1e-8:
            x_left_end = max(xlim[0], x_label_data - line_gap_half)
            x_right_start = min(xlim[1], x_label_data + line_gap_half)

            if x_left_end > xlim[0]:
                xx1 = np.linspace(xlim[0], x_left_end, 200)
                yy1 = -xx1 / Kx
                ax_a.plot(xx1, yy1, color="black", linestyle="--",
                          lw=4.8, zorder=3)

            if x_right_start < xlim[1]:
                xx2 = np.linspace(x_right_start, xlim[1], 200)
                yy2 = -xx2 / Kx
                ax_a.plot(xx2, yy2, color="black", linestyle="--",
                          lw=4.8, zorder=3)

        ax_a.axhline(0, color="#bdbdbd", lw=1.5, zorder=2)
        ax_a.axvline(0, color="#bdbdbd", lw=1.5, zorder=2)

        ax_a.text(0.50, 0.97, r"$\Delta T_x > 0$ — Daytime UHI",
                  transform=ax_a.transAxes, ha="center", va="top",
                  fontsize=19, fontweight="bold", color="#b2182b",
                  path_effects=halo, zorder=10)
        ax_a.text(0.50, 0.03, r"$\Delta T_x < 0$ — Daytime UCI",
                  transform=ax_a.transAxes, ha="center", va="bottom",
                  fontsize=19, fontweight="bold", color="#2166ac",
                  path_effects=halo, zorder=10)

        ax_a.scatter(nhw_plot["dTmean"], nhw_plot["dAmp1"],
                     s=30, facecolors="none", edgecolors="black",
                     linewidths=0.7, alpha=0.10, zorder=4)

        def add_fft_cycle_inset(parent_ax, bbox, dm, da, title, title_color,
                                title_fs=12.0, curve_lw=2.2,
                                patch_alpha=0.86,
                                title_inside=False):
            iax = inset_axes(
                parent_ax,
                width="100%", height="100%",
                bbox_to_anchor=bbox,
                bbox_transform=parent_ax.transAxes,
                loc="center",
                borderpad=0
            )

            y_rural = recon(t_f, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
            y_urban = recon(t_f, m_ref + dm, a1_ref + da, p1_ref, a2_ref, p2_ref)

            iax.fill_between(
                t_f, y_rural, y_urban,
                where=(y_urban >= y_rural),
                color="#e89090", alpha=0.42,
                interpolate=True, linewidth=0, zorder=1
            )
            iax.fill_between(
                t_f, y_rural, y_urban,
                where=(y_urban < y_rural),
                color="#7eb6d9", alpha=0.42,
                interpolate=True, linewidth=0, zorder=1
            )

            iax.plot(t_f, y_rural, color="#4d4d4d", lw=1.45, ls="--", zorder=2)
            iax.plot(t_f, y_urban, color=title_color, lw=curve_lw, zorder=3)

            yall = np.concatenate([y_rural, y_urban])
            iax.set_xlim(0, 24)
            iax.set_ylim(np.nanmin(yall) - 1.1, np.nanmax(yall) + 1.1)

            iax.set_xticks([])
            iax.set_yticks([])
            iax.set_xlabel("")
            iax.set_ylabel("")

            if title_inside:
                iax.text(
                    0.06, 0.94, title,
                    transform=iax.transAxes,
                    ha="left", va="top",
                    fontsize=title_fs,
                    fontweight="bold",
                    color=title_color,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=2),
                    zorder=10
                )
            else:
                iax.text(
                    0.5, 1.055, title,
                    transform=iax.transAxes,
                    ha="center", va="bottom",
                    fontsize=title_fs,
                    fontweight="bold",
                    color=title_color
                )

            for side in ["top", "right", "bottom", "left"]:
                iax.spines[side].set_visible(True)
                iax.spines[side].set_linewidth(0.8)
                iax.spines[side].set_color("#4d4d4d")

            iax.patch.set_facecolor("white")
            iax.patch.set_alpha(patch_alpha)
            return iax

        inset_w, inset_h = 0.235, 0.175
        top_bottom = 0.705
        bottom_bottom = 0.105
        left_left = 0.060
        right_left = 0.705

        add_fft_cycle_inset(
            ax_a, bbox=(right_left, top_bottom, inset_w, inset_h),
            dm=+1.0, da=+2.0,
            title=r"$+\Delta T_{mean}$, $+\Delta Amp$",
            title_color="#b2182b",
            patch_alpha=0.86,
        )
        add_fft_cycle_inset(
            ax_a, bbox=(left_left, bottom_bottom, inset_w, inset_h),
            dm=-1.0, da=-2.0,
            title=r"$-\Delta T_{mean}$, $-\Delta Amp$",
            title_color="#2166ac",
            patch_alpha=0.86,
        )
        add_fft_cycle_inset(
            ax_a, bbox=(right_left, bottom_bottom, inset_w, inset_h),
            dm=+1.55, da=-1.15,
            title=r"$+\Delta T_{mean}$, $-\Delta Amp$",
            title_color="#d95f0e",
            patch_alpha=0.86,
        )

        # =================================================================
        # Emphasized mechanism arrows for panel a
        # 三个箭头是主要结果：加粗、加大，并调整标签位置
        # =================================================================
        ARR_COL = "#2c3e50"
        LBL_FS = 14.5
        arrow_kwargs = dict(
            arrowstyle="->",
            lw=2.7,
            color=ARR_COL,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=19,
        )

        center_xy = (0.405, 0.485)

        # ΔAmp：向下
        amp_end_xy = (0.405, 0.345)

        # ΔTmean：向右
        tmean_end_xy = (0.555, 0.485)

        # Combined forcing：与 ΔTx = 0 虚线平行，方向向右下
        # 这里保持起点相同，只调整终点，使斜率接近 ΔTx=0 虚线
        comb_end_xy = (0.555, 0.315)

        # ΔAmp decreases
        ax_a.annotate(
            "", xy=amp_end_xy, xytext=center_xy,
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=arrow_kwargs,
            zorder=8
        )
        ax_a.text(
            0.382, 0.405, r"$\Delta Amp\downarrow$",
            transform=ax_a.transAxes,
            ha="right", va="center",
            fontsize=LBL_FS,
            fontweight="bold",
            color=ARR_COL,
            path_effects=halo,
            zorder=9
        )

        # ΔTmean increases — label above the horizontal arrow
        ax_a.annotate(
            "", xy=tmean_end_xy, xytext=center_xy,
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=arrow_kwargs,
            zorder=8
        )
        ax_a.text(
            0.480, 0.525, r"$\Delta T_{mean}\uparrow$",
            transform=ax_a.transAxes,
            ha="center", va="bottom",
            fontsize=LBL_FS,
            fontweight="bold",
            color=ARR_COL,
            path_effects=halo,
            zorder=9
        )

        # Combined forcing — arrow parallel to ΔTx = 0, label moved upward
        ax_a.annotate(
            "", xy=comb_end_xy, xytext=center_xy,
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=arrow_kwargs,
            zorder=8
        )
        ax_a.text(
            0.535, 0.395, "Combined forcing",
            transform=ax_a.transAxes,
            ha="left", va="center",
            fontsize=LBL_FS,
            fontweight="bold",
            color=ARR_COL,
            path_effects=halo,
            zorder=9
        )

        # 只修复底图被 autoscale 缩小的问题：恢复 panel a 原始坐标范围
        ax_a.set_xlim(xlim)
        ax_a.set_ylim(ylim)
        ax_a.set_autoscale_on(False)

        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")

        # ΔTx = 0 标签加粗，并覆盖虚线
        if np.isfinite(Kx) and abs(Kx) > 1e-8:
            y_label_data = -x_label_data / Kx

            fig.canvas.draw()
            p1 = ax_a.transData.transform((x_label_data - 0.5, -(x_label_data - 0.5) / Kx))
            p2 = ax_a.transData.transform((x_label_data + 0.5, -(x_label_data + 0.5) / Kx))
            label_rot = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

            ax_a.text(
                x_label_data, y_label_data, r"$\Delta T_x=0$",
                rotation=label_rot, rotation_mode="anchor",
                ha="center", va="center",
                fontsize=15,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=0.35, alpha=1.0),
                path_effects=halo,
                zorder=10
            )

        cax = make_axes_locatable(ax_a).append_axes("right", size="5%", pad=0.15)
        cb = plt.colorbar(cf, cax=cax)
        cb.set_label(r"Daytime $\Delta T_x$ (°C)", fontsize=22)
        cb.ax.tick_params(labelsize=18, width=1.4, length=5)
        if cb_ticks is not None and len(cb_ticks) >= 2:
            cb.set_ticks(cb_ticks)

        add_panel_label(
            ax_a, "a",
            "Mean–amplitude framework for\ndaytime UHI/UCI transition"
        )

        # =================================================================
        # Panel d
        # =================================================================
        metrics = [
            ("dTmean", r"$R_{mean}$"),
            ("dAmp1",  r"$R_{amp}$"),
            ("dTx",    r"$R_x$"),
            ("dTn",    r"$R_n$")
        ]
        merged = pd.merge(
            nhw_plot[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
            hw_plot[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
            on=[id_col, "group"], suffixes=("_nhw", "_hw")
        )

        if not merged.empty:
            x_pos = np.arange(len(metrics))
            bar_w = 0.35

            group_specs = [
                (-bar_w / 2, "UHI", "#b2182b"),
                ( bar_w / 2, "UCI", "#2166ac")
            ]
            for offset, group, color in group_specs:
                g_data = merged[merged["group"] == group]
                if g_data.empty:
                    continue
                diffs = [(g_data[f"{m}_hw"] - g_data[f"{m}_nhw"]).dropna()
                         for m, _ in metrics]
                rng_boot = np.random.default_rng(20260529)
                boot_stats = [_bootstrap_mean_ci(d, n_boot=1000, ci=95, rng=rng_boot)
                              for d in diffs]
                means = np.array([s[0] for s in boot_stats], dtype=float)
                ci_low = np.array([s[1] for s in boot_stats], dtype=float)
                ci_high = np.array([s[2] for s in boot_stats], dtype=float)
                yerr = np.vstack([means - ci_low, ci_high - means])
                ax_d.bar(x_pos + offset, means, bar_w,
                         yerr=yerr, color=color, alpha=0.88,
                         edgecolor="black", linewidth=1.0,
                         capsize=6,
                         error_kw=dict(lw=1.6, capthick=1.6),
                         label=f"{group} (n={len(g_data)})",
                         zorder=3)

            ax_d.axhline(0, color="black", lw=2.0, zorder=2)
            ax_d.set_xticks(x_pos)
            ax_d.set_xticklabels([label for _, label in metrics])
            ax_d.set_ylabel("HW$-$NHW urban–rural response (°C)")
            ax_d.legend(frameon=False, loc="best",
                        handlelength=1.4, borderaxespad=0.4)
            ax_d.yaxis.grid(True, color="#e6e6e6", lw=1.0, zorder=0)
            ax_d.set_axisbelow(True)
        else:
            ax_d.text(0.5, 0.5, "No paired NHW–HW data available",
                      transform=ax_d.transAxes,
                      ha="center", va="center", fontsize=22)

        add_panel_label(
            ax_d, "d",
            "Heatwave-induced\nmean, amplitude, daytime\nand nighttime responses"
        )

        for ax in axes.flatten():
            style_full_box(ax, lw=2.5)

        out_dir = os.path.join(output_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)
        fpath_png = os.path.join(out_dir, "Figure_Combined_Mechanism_2x2_Nature.png")
        fpath_pdf = os.path.join(out_dir, "Figure_Combined_Mechanism_2x2_Nature.pdf")
        fig.savefig(fpath_png, dpi=600, bbox_inches="tight")
        fig.savefig(fpath_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved Nature-style figure: {fpath_png}")
        print(f"  Saved editable vector figure: {fpath_pdf}")


def plot_figure_combined_mechanism_2x2_fixed(all_df, output_dir):
    """
    Nature-style 2x2 combined mechanism figure (v22-aligned).

    a  Mean–amplitude framework for daytime UHI/UCI transition
    b  Mean–amplitude phase space during NHW periods
    c  Mean–amplitude phase space during HW periods
    d  Heatwave-induced mean, amplitude, daytime and nighttime responses
    """

    id_col = next((c for c in ["pair_id", "station_id", "city_id"] if c in all_df.columns), None)
    if id_col is None:
        raise ValueError("No valid ID column found. Expected one of: pair_id, station_id, city_id.")

    required_cols = ["period", "group", "dTmean", "dAmp1", "dTx", "dTn"]
    missing = [c for c in required_cols if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    nhw_plot = (
        all_df[all_df["period"] == "non_heatwave"]
        .dropna(subset=["dTmean", "dAmp1", "dTx"]).copy()
    )
    hw_plot = (
        all_df[all_df["period"] == "heatwave"]
        .dropna(subset=["dTmean", "dAmp1", "dTx"]).copy()
    )
    if nhw_plot.empty or hw_plot.empty:
        raise ValueError("NHW or HW data is empty after dropping NaNs.")

    SUBTITLE_FONTSIZE = 24
    LABEL_FONTSIZE = 32

    def add_panel_label(ax, label, subtitle, pad=18):
        ax.set_title(subtitle, loc="center", fontweight="bold",
                     pad=pad, fontsize=SUBTITLE_FONTSIZE)
        ax.text(0.0, 1.04, label,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontweight="bold", fontsize=LABEL_FONTSIZE)

    def style_full_box(ax, lw=2.5):
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(lw)
        ax.tick_params(top=False, right=False,
                       direction="out", width=1.6, length=6)

    def integer_cb_ticks(vabs_val):
        lo = math.ceil(-vabs_val)
        hi = math.floor(vabs_val)
        return np.arange(lo, hi + 1) if hi >= lo else None

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 20,
        "axes.labelsize": 26,
        "axes.titlesize": SUBTITLE_FONTSIZE,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "axes.linewidth": 2.5,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }):

        fig, axes = plt.subplots(2, 2, figsize=(26, 24), dpi=600)
        plt.subplots_adjust(wspace=0.35, hspace=0.45)

        ax_a = axes[0, 0]
        ax_b = axes[0, 1]
        ax_c = axes[1, 0]
        ax_d = axes[1, 1]

        halo = [path_effects.withStroke(linewidth=5, foreground="white", alpha=0.9)]

        dtx_all = pd.concat([nhw_plot["dTx"], hw_plot["dTx"]], ignore_index=True)
        x_all = pd.concat([nhw_plot["dTmean"], hw_plot["dTmean"]], ignore_index=True)
        y_all = pd.concat([nhw_plot["dAmp1"], hw_plot["dAmp1"]], ignore_index=True)

        vabs = max(np.nanpercentile(np.abs(dtx_all), 98), 0.1)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        cmap = plt.cm.RdBu_r
        cb_ticks = integer_cb_ticks(vabs)

        xlim = (np.nanpercentile(x_all, 1) - 0.5,
                np.nanpercentile(x_all, 99) + 0.5)
        ylim = (np.nanpercentile(y_all, 1) - 0.5,
                np.nanpercentile(y_all, 99) + 0.5)

        # =================================================================
        # Panels b & c
        # =================================================================
        phase_panels = [
            (nhw_plot, ax_b, "b",
             "Mean–amplitude state space\nduring NHW periods", "o"),
            (hw_plot, ax_c, "c",
             "Mean–amplitude state space\nduring HW periods", "^")
        ]

        for df_p, ax, lbl, subtitle, marker in phase_panels:
            sc = ax.scatter(
                df_p["dTmean"], df_p["dAmp1"],
                c=df_p["dTx"], cmap=cmap, norm=norm,
                marker=marker, s=100,
                edgecolors="black", linewidths=0.8,
                alpha=0.9, zorder=3
            )

            if len(df_p) > 2:
                X = np.column_stack([
                    df_p["dTmean"].values,
                    df_p["dAmp1"].values,
                    np.ones(len(df_p))
                ])
                y = df_p["dTx"].values
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                if np.isfinite(coef[1]) and abs(coef[1]) > 1e-8:
                    xx = np.linspace(xlim[0], xlim[1], 200)
                    yy = (-coef[0] / coef[1]) * xx + (-coef[2] / coef[1])
                    ax.plot(xx, yy, color="black", linestyle="--", lw=3,
                            label=r"$\Delta T_x = 0$", zorder=4)

            ax.axhline(0, color="#d0d0d0", lw=1.5, zorder=1)
            ax.axvline(0, color="#d0d0d0", lw=1.5, zorder=1)
            ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
            ax.set_ylabel(r"$\Delta Amp$ (°C)")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            leg = ax.legend(frameon=False, loc="upper right",
                            handlelength=1.8, borderaxespad=0.4)
            if leg is not None:
                for line in leg.get_lines():
                    line.set_linewidth(2.5)

            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
            cb = plt.colorbar(sc, cax=cax)
            cb.set_label(r"Daytime $\Delta T_x$ (°C)", fontsize=22)
            cb.ax.tick_params(labelsize=18, width=1.4, length=5)
            if cb_ticks is not None and len(cb_ticks) >= 2:
                cb.set_ticks(cb_ticks)

            add_panel_label(ax, lbl, subtitle)

        # =================================================================
        # Panel a  —— layout-refined, Nature-like, NO data logic changes
        # =================================================================
        m_ref, a1_ref, a2_ref, p1_ref, p2_ref = 12.0, 7.0, 1.0, 3.8, 0.5
        w = 2 * np.pi / 24
        t_f = np.linspace(0, 24, 1000)

        def recon(t, m, a, p, a2, p2):
            return m + a * np.cos(w * t - p) + a2 * np.cos(2 * w * t - p2)

        y_rural_ref = recon(t_f, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
        idx_tx = np.argmax(y_rural_ref)
        t_tx = t_f[idx_tx]
        Kx = np.cos(w * t_tx - p1_ref)

        # -----------------------------
        # background field (UNCHANGED logic)
        # -----------------------------
        x_grid = np.linspace(xlim[0], xlim[1], 320)
        y_grid = np.linspace(ylim[0], ylim[1], 320)
        XX, YY = np.meshgrid(x_grid, y_grid)
        ZZ = XX + Kx * YY
        ZZ_disp = np.clip(ZZ, -vabs, vabs)
        levels = np.linspace(-vabs, vabs, 25)

        cf = ax_a.contourf(
            XX, YY, ZZ_disp, levels=levels,
            cmap=cmap, norm=norm, alpha=0.50, zorder=0
        )

        # -----------------------------
        # reference lines
        # -----------------------------
        ax_a.axhline(0, color="#cfcfcf", lw=1.3, zorder=1)
        ax_a.axvline(0, color="#cfcfcf", lw=1.3, zorder=1)

        # -----------------------------
        # scatter cloud (UNCHANGED data)
        # -----------------------------
        ax_a.scatter(
            nhw_plot["dTmean"], nhw_plot["dAmp1"],
            s=22, facecolors="none", edgecolors="#9a9a9a",
            linewidths=0.65, alpha=0.38, zorder=4
        )

        # -----------------------------
        # line label position + line gap
        # -----------------------------
        halo_small = [path_effects.withStroke(linewidth=4.0, foreground="white", alpha=0.95)]

        x_label_frac = 0.22
        x_label_data = xlim[0] + x_label_frac * (xlim[1] - xlim[0])
        line_gap_half = 0.52

        if np.isfinite(Kx) and abs(Kx) > 1e-8:
            x_left_end = max(xlim[0], x_label_data - line_gap_half)
            x_right_start = min(xlim[1], x_label_data + line_gap_half)

            if x_left_end > xlim[0]:
                xx1 = np.linspace(xlim[0], x_left_end, 200)
                yy1 = -xx1 / Kx
                ax_a.plot(
                    xx1, yy1, color="black", linestyle="--",
                    lw=3.0, dashes=(6, 4), zorder=3
                )

            if x_right_start < xlim[1]:
                xx2 = np.linspace(x_right_start, xlim[1], 200)
                yy2 = -xx2 / Kx
                ax_a.plot(
                    xx2, yy2, color="black", linestyle="--",
                    lw=3.0, dashes=(6, 4), zorder=3
                )

        # -----------------------------
        # inset helper
        # -----------------------------
        def add_fft_cycle_inset(
            parent_ax, bbox, dm, da, corner_label, curve_color,
            patch_alpha=0.90
        ):
            iax = inset_axes(
                parent_ax,
                width="100%", height="100%",
                bbox_to_anchor=bbox,
                bbox_transform=parent_ax.transAxes,
                loc="center",
                borderpad=0
            )

            y_rural = recon(t_f, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
            y_urban = recon(t_f, m_ref + dm, a1_ref + da, p1_ref, a2_ref, p2_ref)

            iax.fill_between(
                t_f, y_rural, y_urban,
                where=(y_urban >= y_rural),
                color="#e99696", alpha=0.34,
                interpolate=True, linewidth=0, zorder=1
            )
            iax.fill_between(
                t_f, y_rural, y_urban,
                where=(y_urban < y_rural),
                color="#8ec1e0", alpha=0.34,
                interpolate=True, linewidth=0, zorder=1
            )

            iax.plot(t_f, y_rural, color="#666666", lw=1.4, ls=(0, (3, 3)), zorder=2)
            iax.plot(t_f, y_urban, color=curve_color, lw=2.0, zorder=3)

            yall = np.concatenate([y_rural, y_urban])
            # Keep the diurnal cycle in 0–24 h.
            # Labels are placed inside the inset with short leader lines.
            ymin_in = np.nanmin(yall) - 1.0
            ymax_in = np.nanmax(yall) + 1.0

            iax.set_xlim(0, 24)
            iax.set_ylim(ymin_in, ymax_in)

            iax.set_xticks([])
            iax.set_yticks([])
            iax.set_xlabel("")
            iax.set_ylabel("")

            # Direct internal curve labels with short leader lines.
            # No right-side blank space; labels remain inside the 0–24 h inset.
            label_pe = [
                path_effects.withStroke(
                    linewidth=2.2,
                    foreground="white",
                    alpha=0.95
                )
            ]

            # Use slightly different anchor positions for A/B/C to avoid curve overlap.
            if corner_label == "A":
                t_anchor = 18.8
                upper_text_xy = (0.78, 0.80)
                lower_text_xy = (0.78, 0.62)
            elif corner_label == "B":
                t_anchor = 18.2
                upper_text_xy = (0.76, 0.78)
                lower_text_xy = (0.76, 0.58)
            else:  # C
                t_anchor = 17.8
                upper_text_xy = (0.72, 0.78)
                lower_text_xy = (0.72, 0.58)

            yr0 = np.interp(t_anchor, t_f, y_rural)
            yu0 = np.interp(t_anchor, t_f, y_urban)

            # Put the label of the upper curve above, and the lower curve below.
            if yu0 >= yr0:
                urban_text_xy = upper_text_xy
                rural_text_xy = lower_text_xy
            else:
                urban_text_xy = lower_text_xy
                rural_text_xy = upper_text_xy

            iax.annotate(
                r"$T_{a,\mathrm{u}}$",
                xy=(t_anchor, yu0),
                xycoords="data",
                xytext=urban_text_xy,
                textcoords="axes fraction",
                ha="left",
                va="center",
                fontsize=7.6,
                fontweight="bold",
                color=curve_color,
                path_effects=label_pe,
                arrowprops=dict(
                    arrowstyle="-",
                    color=curve_color,
                    lw=0.9,
                    shrinkA=1.5,
                    shrinkB=1.5,
                    connectionstyle="arc3,rad=0.0"
                ),
                zorder=6,
                clip_on=True
            )

            iax.annotate(
                r"$T_{a,\mathrm{r}}$",
                xy=(t_anchor, yr0),
                xycoords="data",
                xytext=rural_text_xy,
                textcoords="axes fraction",
                ha="left",
                va="center",
                fontsize=7.6,
                fontweight="bold",
                color="#666666",
                path_effects=label_pe,
                arrowprops=dict(
                    arrowstyle="-",
                    color="#666666",
                    lw=0.9,
                    linestyle=(0, (3, 3)),
                    shrinkA=1.5,
                    shrinkB=1.5,
                    connectionstyle="arc3,rad=0.0"
                ),
                zorder=6,
                clip_on=True
            )

            for side in ["top", "right", "bottom", "left"]:
                iax.spines[side].set_visible(True)
                iax.spines[side].set_linewidth(0.8)
                iax.spines[side].set_color("#8c8c8c")

            iax.patch.set_facecolor("white")
            iax.patch.set_alpha(patch_alpha)
            return iax

        # -----------------------------
        # inset layout (docked to edges)
        # -----------------------------
        bbox_A = (0.69, 0.63, 0.25, 0.19)
        bbox_B = (0.74, 0.20, 0.23, 0.18)
        bbox_C = (0.06, 0.08, 0.22, 0.18)

        add_fft_cycle_inset(
            ax_a, bbox=bbox_A,
            # Match the A anchor point in phase space: (ΔTmean, ΔAmp) = (3, 1)
            dm=+3.0, da=+1.0,
            corner_label="A",
            curve_color="#d7191c"
        )
        add_fft_cycle_inset(
            ax_a, bbox=bbox_B,
            # Match the B anchor point in phase space: (ΔTmean, ΔAmp) = (3, -2)
            dm=+3.0, da=-2.0,
            corner_label="B",
            curve_color="#f16913"
        )
        add_fft_cycle_inset(
            ax_a, bbox=bbox_C,
            # Keep C unchanged
            dm=+0.9, da=-2.2,
            corner_label="C",
            curve_color="#2257d5"
        )

        # -----------------------------
        # label anchor positions
        # -----------------------------
        xA, yA1, yA2 = 0.76, 0.94, 0.90

        xB_letter = 0.70
        xB_center = 0.81
        yB1, yB2 = 0.47, 0.405

        xC, yC1, yC2, yC3 = 0.31, 0.17, 0.13, 0.09

        # -----------------------------
        # titles / subtitles outside insets
        # Rule:
        #   1) A/B/C letter is placed outside the upper-left corner of each inset
        #   2) label text is centered relative to the inset box
        #   3) long labels are split into two lines
        # -----------------------------

        def add_inset_outside_label(
            ax, bbox, letter, title, formula1,
            color_title, color_formula,
            formula2=None,
            title_pad=0.075,
            formula1_pad=0.035,
            formula2_pad=0.010,
            letter_dx=0.012,
            letter_dy=0.012,
            title_fs=14.0,
            formula_fs=12.5,
        ):
            """
            bbox is the same axes-fraction bbox used by inset_axes:
            bbox = (left, bottom, width, height)
            """
            left, bottom, width, height = bbox
            top = bottom + height
            cx = left + width / 2.0

            # A/B/C outside upper-left of inset
            ax.text(
                left - letter_dx, top + letter_dy, letter,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=15.5,
                fontweight="bold",
                color=color_title,
                path_effects=halo_small,
                zorder=10
            )

            # Main label centered relative to inset
            ax.text(
                cx, top + title_pad, title,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=color_title,
                linespacing=0.90,
                path_effects=halo_small,
                zorder=10
            )

            # Formula line 1 centered relative to inset
            ax.text(
                cx, top + formula1_pad, formula1,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=formula_fs,
                fontweight="bold",
                color=color_formula,
                path_effects=halo_small,
                zorder=10
            )

            # Optional formula line 2
            if formula2 is not None:
                ax.text(
                    cx, top + formula2_pad, formula2,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=formula_fs,
                    fontweight="bold",
                    color=color_formula,
                    path_effects=halo_small,
                    zorder=10
                )


        # A label: amplified diurnal UHI
        add_inset_outside_label(
            ax=ax_a,
            bbox=bbox_A,
            letter="A",
            title="Amplified\ndiurnal UHI",
            formula1=r"$-\Delta Amp < 0 < \Delta T_{mean}$",
            formula2=r"$\Delta T_x > \Delta T_n$",
            color_title="#7f0000",
            color_formula="#a50f15",
            title_pad=0.080,
            formula1_pad=0.040,
            formula2_pad=0.012,
            title_fs=13.4,
            formula_fs=11.6,
        )

        # B label: damped daytime UHI
        add_inset_outside_label(
            ax=ax_a,
            bbox=bbox_B,
            letter="B",
            title="Damped daytime\nUHI",
            formula1=r"$0 < -\Delta Amp < \Delta T_{mean}$",
            formula2=r"$\Delta T_n > \Delta T_x > 0$",
            color_title="#b35806",
            color_formula="#d95f0e",
            title_pad=0.086,
            formula1_pad=0.037,
            formula2_pad=0.012,
            title_fs=13.0,
            formula_fs=11.2,
        )

        # C label: daytime UCI
        add_inset_outside_label(
            ax=ax_a,
            bbox=bbox_C,
            letter="C",
            title="Daytime UCI",
            formula1=r"$0 < \Delta T_{mean} < -\Delta Amp$",
            formula2=r"$\Delta T_x < 0,\ \Delta T_n > 0$",
            color_title="#08306b",
            color_formula="#225ea8",
            title_pad=0.082,
            formula1_pad=0.040,
            formula2_pad=0.014,
            title_fs=13.2,
            formula_fs=11.2,
        )

        # -----------------------------
        # anchor points and connectors
        # A/B are fixed conceptual coordinates; C keeps the current data-based selection.
        # -----------------------------
        xr = xlim[1] - xlim[0]
        yr = ylim[1] - ylim[0]

        # A and B are fixed at the requested phase-space coordinates.
        # These coordinates are in data units: (ΔTmean, ΔAmp).
        pt_A = (3.0, 1.0)
        pt_B = (3.0, -2.0)

        # Keep C unchanged: select a representative actual point in the daytime-UCI region.
        def _inside_bbox_data_mask(df_sub, bbox, pad=0.015):
            """
            Convert inset bbox in axes-fraction coordinates to data coordinates,
            then return mask for points covered by this inset region.
            """
            left, bottom, width, height = bbox

            x0 = xlim[0] + (left - pad) * xr
            x1 = xlim[0] + (left + width + pad) * xr
            y0 = ylim[0] + (bottom - pad) * yr
            y1 = ylim[0] + (bottom + height + pad) * yr

            return (
                (df_sub["dTmean"] >= min(x0, x1)) &
                (df_sub["dTmean"] <= max(x0, x1)) &
                (df_sub["dAmp1"]  >= min(y0, y1)) &
                (df_sub["dAmp1"]  <= max(y0, y1))
            )

        def _choose_actual_anchor(df_source, target_xy, condition_func=None,
                                  avoid_bboxes=None, prefer_transition=False):
            """
            Pick one real observed point from df_source using actual dTmean and dAmp1.

            target_xy:
                conceptual target position, only used to choose the nearest real point.
            condition_func:
                restricts the candidate region, e.g. UHI, offset, UCI.
            avoid_bboxes:
                excludes points hidden by inset boxes.
            prefer_transition:
                if True, also prefers points close to ΔTx = 0.
            """
            sub = df_source[["dTmean", "dAmp1", "dTx"]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna().copy()

            if condition_func is not None:
                sub = sub[condition_func(sub)].copy()

            # Exclude points covered by inset boxes so the anchor is visible
            if avoid_bboxes is not None and len(sub) > 0:
                keep = np.ones(len(sub), dtype=bool)
                for bb in avoid_bboxes:
                    keep &= ~_inside_bbox_data_mask(sub, bb).values
                sub_visible = sub.loc[keep].copy()

                # If filtering is too strict, fall back to the original candidate set
                if len(sub_visible) > 0:
                    sub = sub_visible

            # Final fallback: use all visible points if regional condition returns none
            if len(sub) == 0:
                sub = df_source[["dTmean", "dAmp1", "dTx"]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna().copy()

                if avoid_bboxes is not None and len(sub) > 0:
                    keep = np.ones(len(sub), dtype=bool)
                    for bb in avoid_bboxes:
                        keep &= ~_inside_bbox_data_mask(sub, bb).values
                    if keep.sum() > 0:
                        sub = sub.loc[keep].copy()

            tx, ty = target_xy

            # Normalized distance in phase space
            score = ((sub["dTmean"] - tx) / max(xr, 1e-9)) ** 2 + \
                    ((sub["dAmp1"]  - ty) / max(yr, 1e-9)) ** 2

            if prefer_transition and "dTx" in sub.columns:
                score = score + 0.35 * (sub["dTx"] / max(vabs, 1e-9)) ** 2

            idx = score.idxmin()
            return float(sub.loc[idx, "dTmean"]), float(sub.loc[idx, "dAmp1"])

        avoid_bboxes = [bbox_A, bbox_B, bbox_C]

        pt_C = _choose_actual_anchor(
            nhw_plot,
            target_xy=(0.9, -2.2),
            condition_func=lambda d: (
                (d["dTmean"] > 0) &
                (d["dAmp1"]  < 0) &
                (d["dTx"]    < 0)
            ),
            avoid_bboxes=avoid_bboxes,
            prefer_transition=False
        )

        # Plot anchor points
        ax_a.scatter(*pt_A, s=135, color="#b2182b",
                     edgecolor="white", linewidth=1.0, zorder=8)
        ax_a.scatter(*pt_B, s=120, color="#f16913",
                     edgecolor="white", linewidth=1.0, zorder=8)
        ax_a.scatter(*pt_C, s=135, color="#2257d5",
                     edgecolor="white", linewidth=1.0, zorder=8)

        # -----------------------------
        # connectors: start from inset edge, end at anchor point
        # -----------------------------
        def connect_point_to_inset(pt_data, pt_axesfrac, lw=1.15, color="#7a7a7a"):
            ax_a.annotate(
                "",
                xy=pt_data, xycoords="data",
                xytext=pt_axesfrac, textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=lw,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=7
            )

        # A/B connectors start from the left edge of each inset and point to fixed coordinates.
        connect_point_to_inset(
            pt_A,
            (bbox_A[0], bbox_A[1] + 0.52 * bbox_A[3])
        )

        connect_point_to_inset(
            pt_B,
            (bbox_B[0], bbox_B[1] + 0.52 * bbox_B[3])
        )

        # C remains unchanged: connector starts from the right edge of the C inset.
        connect_point_to_inset(
            pt_C,
            (bbox_C[0] + bbox_C[2], bbox_C[1] + 0.58 * bbox_C[3])
        )
        # -----------------------------
        # mechanism arrows (compact)
        # -----------------------------
        ARR_COL = "#173f73"
        arrow_kwargs = dict(
            arrowstyle="->",
            lw=2.2,
            color=ARR_COL,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=17,
        )

        center_xy = (0.41, 0.49)
        amp_end_xy = (0.41, 0.35)
        tmean_end_xy = (0.57, 0.49)
        comb_end_xy = (0.57, 0.33)

        ax_a.annotate(
            "", xy=amp_end_xy, xytext=center_xy,
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=arrow_kwargs, zorder=8
        )
        ax_a.text(
            0.385, 0.405, r"$-\Delta Amp$",
            transform=ax_a.transAxes,
            ha="right", va="center",
            fontsize=14.0, fontweight="bold",
            color=ARR_COL, path_effects=halo_small, zorder=9
        )

        ax_a.annotate(
            "", xy=tmean_end_xy, xytext=center_xy,
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=arrow_kwargs, zorder=8
        )
        ax_a.text(
            0.49, 0.525, r"$+\Delta T_{mean}$",
            transform=ax_a.transAxes,
            ha="center", va="bottom",
            fontsize=14.0, fontweight="bold",
            color=ARR_COL, path_effects=halo_small, zorder=9
        )

        ax_a.annotate(
            "", xy=comb_end_xy, xytext=center_xy,
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=arrow_kwargs, zorder=8
        )
        ax_a.text(
            0.545, 0.405, "Combined\nforcing",
            transform=ax_a.transAxes,
            ha="left", va="center",
            fontsize=13.2, fontweight="bold",
            color=ARR_COL, path_effects=halo_small, zorder=9
        )

        # -----------------------------
        # axes settings
        # -----------------------------
        ax_a.set_xlim(xlim)
        ax_a.set_ylim(ylim)
        ax_a.set_autoscale_on(False)

        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")

        # -----------------------------
        # strong diagonal-line label
        # -----------------------------
        if np.isfinite(Kx) and abs(Kx) > 1e-8:
            y_label_data = -x_label_data / Kx

            fig.canvas.draw()
            p1 = ax_a.transData.transform((x_label_data - 0.5, -(x_label_data - 0.5) / Kx))
            p2 = ax_a.transData.transform((x_label_data + 0.5, -(x_label_data + 0.5) / Kx))
            label_rot = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

            ax_a.text(
                x_label_data, y_label_data,
                r"$-\Delta Amp = \Delta T_{mean}$" + "\n" + r"$\Delta T_x \approx 0$",
                rotation=label_rot,
                rotation_mode="anchor",
                ha="center", va="center",
                fontsize=15.0,
                fontweight="bold",
                color="#222222",
                linespacing=0.95,
                path_effects=halo_small,
                zorder=10
            )

        # -----------------------------
        # colorbar
        # -----------------------------
        cax = make_axes_locatable(ax_a).append_axes("right", size="5%", pad=0.15)
        cb = plt.colorbar(cf, cax=cax)
        cb.set_label(r"Daytime $\Delta T_x$ (°C)", fontsize=22)
        cb.ax.tick_params(labelsize=18, width=1.4, length=5)
        if cb_ticks is not None and len(cb_ticks) >= 2:
            cb.set_ticks(cb_ticks)

        add_panel_label(
            ax_a, "a",
            "Mean–amplitude framework for\ndaytime UHI/UCI transition"
        )

        # =================================================================
        # Panel d
        # =================================================================
        metrics = [
            ("dTmean", r"$R_{mean}$"),
            ("dAmp1",  r"$R_{amp}$"),
            ("dTx",    r"$R_x$"),
            ("dTn",    r"$R_n$")
        ]
        merged = pd.merge(
            nhw_plot[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
            hw_plot[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
            on=[id_col, "group"], suffixes=("_nhw", "_hw")
        )

        if not merged.empty:
            x_pos = np.arange(len(metrics))
            bar_w = 0.35

            group_specs = [
                (-bar_w / 2, "UHI", "#b2182b"),
                ( bar_w / 2, "UCI", "#2166ac")
            ]
            for offset, group, color in group_specs:
                g_data = merged[merged["group"] == group]
                if g_data.empty:
                    continue
                diffs = [(g_data[f"{m}_hw"] - g_data[f"{m}_nhw"]).dropna()
                         for m, _ in metrics]
                rng_boot = np.random.default_rng(20260529)
                boot_stats = [_bootstrap_mean_ci(d, n_boot=1000, ci=95, rng=rng_boot)
                              for d in diffs]
                means = np.array([s[0] for s in boot_stats], dtype=float)
                ci_low = np.array([s[1] for s in boot_stats], dtype=float)
                ci_high = np.array([s[2] for s in boot_stats], dtype=float)
                yerr = np.vstack([means - ci_low, ci_high - means])
                ax_d.bar(x_pos + offset, means, bar_w,
                         yerr=yerr, color=color, alpha=0.88,
                         edgecolor="black", linewidth=1.0,
                         capsize=6,
                         error_kw=dict(lw=1.6, capthick=1.6),
                         label=f"{group} (n={len(g_data)})",
                         zorder=3)

            ax_d.axhline(0, color="black", lw=2.0, zorder=2)
            ax_d.set_xticks(x_pos)
            ax_d.set_xticklabels([label for _, label in metrics])
            ax_d.set_ylabel("HW$-$NHW urban–rural response (°C)")
            ax_d.legend(frameon=False, loc="best",
                        handlelength=1.4, borderaxespad=0.4)
            ax_d.yaxis.grid(True, color="#e6e6e6", lw=1.0, zorder=0)
            ax_d.set_axisbelow(True)
        else:
            ax_d.text(0.5, 0.5, "No paired NHW–HW data available",
                      transform=ax_d.transAxes,
                      ha="center", va="center", fontsize=22)

        add_panel_label(
            ax_d, "d",
            "Heatwave-induced\nmean, amplitude, daytime\nand nighttime responses"
        )

        for ax in axes.flatten():
            style_full_box(ax, lw=2.5)

        out_dir = os.path.join(output_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)
        fpath_png = os.path.join(out_dir, "Figure_Combined_Mechanism_2x2_Nature.png")
        fpath_pdf = os.path.join(out_dir, "Figure_Combined_Mechanism_2x2_Nature.pdf")
        fig.savefig(fpath_png, dpi=600, bbox_inches="tight")
        fig.savefig(fpath_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved Nature-style figure: {fpath_png}")
        print(f"  Saved editable vector figure: {fpath_pdf}")


def plot_combined_figure_dynamics_consistent(all_df, output_dir):
    """
    Nature-style 2x2 dynamics figure (format aligned with the mechanism figure):

    a  State migration trajectory
    b  Mechanism decomposition
    c  Response compensation linkage
    d  Thermal hysteresis loops
    """

    # ---------------------------------------------------------------------
    # Shared format helpers (identical to mechanism figure)
    # ---------------------------------------------------------------------
    SUBTITLE_FONTSIZE = 24
    LABEL_FONTSIZE = 32

    def add_panel_label(ax, label, subtitle, pad=18):
        ax.set_title(subtitle, loc="center", fontweight="bold",
                     pad=pad, fontsize=SUBTITLE_FONTSIZE)
        ax.text(0.0, 1.04, label,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontweight="bold", fontsize=LABEL_FONTSIZE)

    def style_full_box(ax, lw=2.5):
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(lw)
        ax.tick_params(top=False, right=False,
                       direction="out", width=1.6, length=6,
                       labelsize=24)

    # ---------------------------------------------------------------------
    # Internal helpers (UNCHANGED — exact logic preserved)
    # ---------------------------------------------------------------------
    def _get_precise_peak_time(signal):
        vals = np.asarray(signal, dtype=float)
        if np.all(np.isnan(vals)): return np.nan
        vals = np.where(np.isfinite(vals), vals, np.nanmean(vals))
        sig_detrend = vals - np.mean(vals)
        fft_vals = np.fft.fft(sig_detrend)
        t_fine = np.linspace(0, 24, 2400, endpoint=False)
        recon = np.zeros_like(t_fine)
        for k in [1, 2]:
            recon += np.real(fft_vals[k] * np.exp(1j * 2 * np.pi * k * t_fine / 24))
        return t_fine[np.argmax(recon)]

    def _phase_lag_hours(x_curve, y_curve):
        t_peak_x = _get_precise_peak_time(x_curve)
        t_peak_y = _get_precise_peak_time(y_curve)
        if pd.isna(t_peak_x) or pd.isna(t_peak_y): return np.nan
        return (t_peak_y - t_peak_x) % 24

    def _loop_area(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 4: return np.nan
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _fit_line_ci(x, y, x_grid):
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5: return [np.nan]*5
        lr = stats.linregress(x, y)
        y_fit = lr.intercept + lr.slope * x_grid
        y_pred = lr.intercept + lr.slope * x
        n = len(x)
        s_err = np.sqrt(np.sum((y - y_pred)**2) / (n - 2))
        t_val = stats.t.ppf(0.975, df=n-2)
        ci = t_val * s_err * np.sqrt(
            1/n + (x_grid - np.mean(x))**2 / np.sum((x - np.mean(x))**2)
        )
        return lr.slope, lr.intercept, lr.pvalue, y_fit, ci

    # ---------------------------------------------------------------------
    # Data preparation (UNCHANGED)
    # ---------------------------------------------------------------------
    id_col = next((c for c in ["pair_id", "station_id", "city_id"]
                   if c in all_df.columns), None)
    nhw_df = all_df[all_df["period"] == "non_heatwave"].copy()
    hw_df  = all_df[all_df["period"] == "heatwave"].copy()
    paired = pd.merge(
        nhw_df[[id_col, "group", "dTmean", "dAmp1", "dTx", "dTn"]],
        hw_df[[id_col, "dTmean", "dAmp1", "dTx", "dTn"]],
        on=id_col, suffixes=("_nhw", "_hw"), how="inner"
    ).dropna()

    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]

    COLOR_UHI, COLOR_UCI = "#d62728", "#1f77b4"

    def _bootstrap_vector_angle_ci(rmean_vals, ramp_vals, n_boot=1000, alpha=0.05, seed=42):
        """
        Group migration angle from the mean migration vector:
            theta = atan2(mean(Ramp), mean(Rmean))
        Kept consistent with Figure_HW_shift_arrows_quantified panel b.
        """
        x = np.asarray(rmean_vals, dtype=float)
        y = np.asarray(ramp_vals, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        n = len(x)
        if n == 0:
            return np.nan, np.nan, np.nan

        point = float(np.degrees(np.arctan2(np.mean(y), np.mean(x))))

        if n == 1:
            return point, point, point

        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot, dtype=float)

        for i in range(n_boot):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            boot[i] = np.degrees(np.arctan2(np.mean(y[idx]), np.mean(x[idx])))

        return (
            point,
            float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)),
        )

    def _build_hw_shift_angle_table_for_panel_c():
        """
        Build the same NHW -> HW migration-angle table used by
        Figure_HW_shift_arrows_quantified panel b.
        This only prepares plotting data; it does not change any scientific definition.
        """
        needed = [id_col, "group", "period", "dTmean", "dAmp1", "dTx"]
        missing = [c for c in needed if c not in all_df.columns]
        if missing:
            print(f"  [Dynamics panel c] Missing columns for HW-shift angle panel: {missing}")
            return pd.DataFrame()

        nhw_angle = all_df[all_df["period"] == "non_heatwave"][
            [id_col, "group", "dTmean", "dAmp1", "dTx"]
        ].rename(columns={
            "dTmean": "dTmean_nhw",
            "dAmp1":  "dAmp_nhw",
            "dTx":    "dTx_nhw",
        })

        hw_angle = all_df[all_df["period"] == "heatwave"][
            [id_col, "dTmean", "dAmp1", "dTx"]
        ].rename(columns={
            "dTmean": "dTmean_hw",
            "dAmp1":  "dAmp_hw",
            "dTx":    "dTx_hw",
        })

        angle_df = pd.merge(nhw_angle, hw_angle, on=id_col, how="inner").dropna(
            subset=[
                "dTmean_nhw", "dAmp_nhw", "dTx_nhw",
                "dTmean_hw", "dAmp_hw", "dTx_hw"
            ]
        ).copy()

        if len(angle_df) == 0:
            return angle_df

        angle_df["Rmean"] = angle_df["dTmean_hw"] - angle_df["dTmean_nhw"]
        angle_df["Ramp"]  = angle_df["dAmp_hw"]   - angle_df["dAmp_nhw"]
        angle_df["theta_deg"] = np.degrees(
            np.arctan2(angle_df["Ramp"], angle_df["Rmean"])
        )

        return angle_df

    def _draw_hw_shift_angle_panel(ax, angle_df):
        """
        Draw Figure_HW_shift_arrows_quantified angle panel inside
        Figure_Dynamics_Consistent_Final panel b.

        Plot-only changes:
        - Keep overall UHI/UCI mean-vector angle marker.
        - Use symbols for quadrant-specific mean-vector angles.
        - Put all numeric angle labels at the vertical middle height
        of the corresponding regime band.
        - Put marker labels directly above the numeric angle labels.
        - Do not draw angle error bars / CI bars.
        - Keep x positions unchanged.
        - Do not change data source, grouping, angle calculation,
        or quadrant definition.
        """
        color_map = {
            "UHI": COLOR_UHI,
            "UCI": COLOR_UCI,
        }

        # Plot-only font alignment: match the 2x2 Nature-style figure scale.
        # No data source, filtering, grouping, or angle algorithm is changed.
        FONT_ANGLE = 22
        FONT_LEGEND_LOCAL = 20

        halo = [
            path_effects.withStroke(
                linewidth=3.0,
                foreground="white",
                alpha=0.96
            )
        ]

        if angle_df is None or len(angle_df) == 0:
            ax.text(
                0.5, 0.5,
                "Insufficient paired HW–NHW data\nfor migration-angle panel",
                ha="center", va="center",
                transform=ax.transAxes,
                color="#777777",
                fontsize=FONT_ANGLE
            )
            ax.axis("off")
            return

        def _vector_quadrant(x, y):
            if not (np.isfinite(x) and np.isfinite(y)):
                return np.nan
            if x >= 0 and y >= 0:
                return "Q1"
            elif x < 0 and y >= 0:
                return "Q2"
            elif x < 0 and y < 0:
                return "Q3"
            else:
                return "Q4"

        angle_df = angle_df.copy()
        angle_df["vector_quadrant"] = [
            _vector_quadrant(x, y)
            for x, y in zip(angle_df["Rmean"], angle_df["Ramp"])
        ]

        def _mean_vector_angle(tbl):
            if tbl is None or len(tbl) == 0:
                return np.nan, 0

            x = tbl["Rmean"].values.astype(float)
            y = tbl["Ramp"].values.astype(float)
            valid = np.isfinite(x) & np.isfinite(y)

            if valid.sum() == 0:
                return np.nan, 0

            theta = float(np.degrees(np.arctan2(np.mean(y[valid]), np.mean(x[valid]))))
            return theta, int(valid.sum())

        rng = np.random.default_rng(42)

        # 带状中心
        group_y = {
            "UHI": 1.0,
            "UCI": 0.0,
        }

        # 角度数字位置：对应 regime 色带的纵向中线
        label_y = {
            "UHI": 1.0,
            "UCI": 0.0,
        }

        # marker 放在 Qx/Mean 标签和角度数字的纵向中间
        marker_y = {
            "UHI": label_y["UHI"] + 0.08,
            "UCI": label_y["UCI"] + 0.08,
        }

        # Qx/Mean 标签放在 marker 上方
        tag_y = {
            "UHI": label_y["UHI"] + 0.16,
            "UCI": label_y["UCI"] + 0.16,
        }

        theta_vals = angle_df["theta_deg"].values
        theta_finite = theta_vals[np.isfinite(theta_vals)]

        if len(theta_finite) == 0:
            theta_xlim = (-90, 90)
        else:
            t_lo = min(float(np.nanpercentile(theta_finite, 2)), -45.0, 0.0) - 10.0
            t_hi = max(float(np.nanpercentile(theta_finite, 98)), -45.0, 0.0) + 10.0
            theta_xlim = (t_lo, t_hi)

        def _clip_text_x(x):
            pad = 0.04 * (theta_xlim[1] - theta_xlim[0])
            return float(np.clip(x, theta_xlim[0] + pad, theta_xlim[1] - pad))

        quadrant_targets = {
            "UHI": ["Q1", "Q4"],
            "UCI": ["Q3", "Q4"],
        }

        quadrant_marker_map = {
            "Q1": "o",
            "Q3": "^",
            "Q4": "D",
        }

        for group_name in ["UHI", "UCI"]:
            g = angle_df[angle_df["group"] == group_name].copy()
            vals = g["theta_deg"].values.astype(float)
            vals = vals[np.isfinite(vals)]
            y0 = group_y[group_name]
            y_text = label_y[group_name]

            # pair-level dots
            if len(vals) > 0:
                jitter = rng.uniform(0.05, 0.14, size=len(vals))
                ax.scatter(
                    vals,
                    np.full(len(vals), y0) + jitter,
                    s=42,
                    color=color_map[group_name],
                    alpha=0.16,
                    edgecolors="none",
                    zorder=2
                )
                # ─────────────────────────────────────────────────────────────
                # Nature-style collision-aware annotation for panel b
                # Plot-only change:
                # - Keep true marker x positions unchanged.
                # - Spread text labels slightly when angles are close.
                # - Use subtle leader lines from text to true marker.
                # - Put UHI labels above the red band and UCI labels below the blue band.
                # ─────────────────────────────────────────────────────────────
                LABEL_FS = max(FONT_ANGLE - 2, 18)
                ANGLE_FS = max(FONT_ANGLE - 3, 17)

                TEXT_COLOR = "#333333"
                LEADER_COLOR = "#8a8a8a"

                # text rows: separate UHI and UCI to avoid band-level crowding
                tag_row_y = {
                    "UHI": 1.34,
                    "UCI": -0.17,
                }
                angle_row_y = {
                    "UHI": 1.20,
                    "UCI": -0.31,
                }

                def _spread_text_x(items, min_sep=32, pad=16):
                    """
                    Spread text x positions only for annotation.
                    The real marker x positions are not changed.
                    """
                    if len(items) <= 1:
                        for it in items:
                            it["text_x"] = _clip_text_x(it["theta"])
                        return items

                    lo, hi = theta_xlim
                    lo += pad
                    hi -= pad

                    placed = sorted(items, key=lambda d: d["theta"])

                    for it in placed:
                        it["text_x"] = float(np.clip(_clip_text_x(it["theta"]), lo, hi))

                    # left-to-right spreading
                    for i in range(1, len(placed)):
                        if placed[i]["text_x"] - placed[i - 1]["text_x"] < min_sep:
                            placed[i]["text_x"] = placed[i - 1]["text_x"] + min_sep

                    # pull back if right boundary exceeded
                    overflow = placed[-1]["text_x"] - hi
                    if overflow > 0:
                        for it in placed:
                            it["text_x"] -= overflow

                    # right-to-left correction
                    for i in range(len(placed) - 2, -1, -1):
                        if placed[i + 1]["text_x"] - placed[i]["text_x"] < min_sep:
                            placed[i]["text_x"] = placed[i + 1]["text_x"] - min_sep

                    # final boundary correction
                    underflow = lo - placed[0]["text_x"]
                    if underflow > 0:
                        for it in placed:
                            it["text_x"] += underflow

                    for it in placed:
                        it["text_x"] = float(np.clip(it["text_x"], lo, hi))

                    return placed

                def _draw_stacked_angle_label(ax, theta, text_x, tag, angle_text, group_name):
                    """Draw tag + angle with subtle leader line."""
                    y_marker = marker_y[group_name]
                    y_tag = tag_row_y[group_name]
                    y_angle = angle_row_y[group_name]

                    # leader line only when text is shifted away from the real marker
                    if abs(text_x - theta) > 3:
                        ax.plot(
                            [theta, text_x],
                            [y_marker, y_angle + 0.035],
                            color=LEADER_COLOR,
                            lw=0.75,
                            alpha=0.75,
                            zorder=6,
                            solid_capstyle="round"
                        )

                    ax.text(
                        text_x,
                        y_tag,
                        tag,
                        ha="center",
                        va="center",
                        fontsize=LABEL_FS,
                        color=TEXT_COLOR,
                        path_effects=halo,
                        zorder=9
                    )

                    ax.text(
                        text_x,
                        y_angle,
                        angle_text,
                        ha="center",
                        va="center",
                        fontsize=ANGLE_FS,
                        color=TEXT_COLOR,
                        path_effects=halo,
                        zorder=9
                    )

                for group_name in ["UHI", "UCI"]:
                    g = angle_df[angle_df["group"] == group_name].copy()
                    if len(g) == 0:
                        continue

                    # raw paired vectors as pale background points
                    y0 = group_y[group_name]
                    jitter = rng.normal(0, 0.035, size=len(g))

                    ax.scatter(
                        g["theta_deg"].values,
                        y0 + jitter,
                        s=42,
                        color=color_map[group_name],
                        alpha=0.14,
                        edgecolors="none",
                        zorder=2
                    )

                    label_items = []

                    # overall mean angle
                    theta_mean, theta_lo, theta_hi = _bootstrap_vector_angle_ci(
                        g["Rmean"].values,
                        g["Ramp"].values
                    )

                    if np.isfinite(theta_mean):
                        ax.scatter(
                            theta_mean,
                            marker_y[group_name],
                            s=180,
                            color=color_map[group_name],
                            edgecolors="white",
                            linewidth=1.5,
                            zorder=7
                        )

                        label_items.append({
                            "theta": float(theta_mean),
                            "tag": "Mean",
                            "angle_text": rf"{theta_mean:.0f}°",
                        })

                    # quadrant-specific mean angles
                    y_quad = marker_y[group_name]

                    for q in quadrant_targets[group_name]:
                        qg = g[g["vector_quadrant"] == q].copy()
                        theta_q, n_q = _mean_vector_angle(qg)

                        if not np.isfinite(theta_q) or n_q <= 0:
                            continue

                        marker_q = quadrant_marker_map.get(q, "o")

                        ax.scatter(
                            theta_q,
                            y_quad,
                            s=64,
                            marker=marker_q,
                            color=color_map[group_name],
                            edgecolors="white",
                            linewidth=1.0,
                            zorder=8
                        )

                        label_items.append({
                            "theta": float(theta_q),
                            "tag": f"{q}",
                            "angle_text": rf"{theta_q:.0f}°",
                        })

                    # Spread only text positions, not marker positions
                    label_items = _spread_text_x(
                        label_items,
                        min_sep=34 if group_name == "UHI" else 32,
                        pad=16
                    )

                    for it in label_items:
                        _draw_stacked_angle_label(
                            ax=ax,
                            theta=it["theta"],
                            text_x=it["text_x"],
                            tag=it["tag"],
                            angle_text=it["angle_text"],
                            group_name=group_name
                        )

        ax.axvline(-45, color="#555555", lw=2.2, ls=":", zorder=1)

        ax.text(
            -45,
            0.58,
            r"$\Delta T_x \approx 0$" + "\n" + "boundary",
            ha="center",
            va="center",
            fontsize=FONT_ANGLE,
            color="#4d4d4d",
            linespacing=0.95,
            path_effects=halo,
            zorder=9
        )

        ax.text(
            -45,
            0.42,
            r"$-45^\circ$",
            ha="center",
            va="center",
            fontsize=FONT_ANGLE,
            color="#4d4d4d",
            path_effects=halo,
            zorder=9
        )

        ax.set_xlim(theta_xlim)
        ax.set_ylim(-0.38, 2.00)

        ax.set_yticks([1.0, 0.0])
        ax.set_yticklabels([
            f"UHI\n(n={(angle_df['group'] == 'UHI').sum()})",
            f"UCI\n(n={(angle_df['group'] == 'UCI').sum()})"
        ])

        ax.set_xlabel(r"Migration angle, $\theta$ (°)", fontsize=26)
        ax.tick_params(axis="both", labelsize=24, width=1.6, length=6)
        ax.grid(True, axis="x", lw=0.35, alpha=0.18, zorder=0)

        # ax.text(
        #     0.04,
        #     0.03,
        #     r"$\theta=\mathrm{atan2}(R_{amp},R_{mean})$",
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="bottom",
        #     fontsize=FONT_ANGLE,
        #     color="#555555",
        #     path_effects=halo,
        #     zorder=6
        # )

        legend_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=COLOR_UHI,
                markeredgecolor="#8f1d1d",
                markeredgewidth=1.2,
                markersize=9,
                label="UHI mean angle"
            ),
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=COLOR_UCI,
                markeredgecolor="#145080",
                markeredgewidth=1.2,
                markersize=9,
                label="UCI mean angle"
            ),
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=7,
                label="Q1: mean-state amplification"
            ),
            Line2D(
                [0], [0],
                marker="^",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=7,
                label="Q3: negative mean-state response with damping"
            ),
            Line2D(
                [0], [0],
                marker="D",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=6.5,
                label="Q4: warming with damping"
            ),
        ]

        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=FONT_LEGEND_LOCAL,
            handletextpad=0.45,
            labelspacing=0.25,
            borderaxespad=0.0
        )
        
    def _draw_original_compensation_panel(ax, add_numbered_label=True):
        """
        Draw the original Dynamics panel c.
        If add_numbered_label=False, export it as a standalone unnumbered figure.
        """
        p_df = paired.copy()
        p_df["dx_res"] = (
            (p_df["dTx_hw"] - p_df["dTx_nhw"])
            - (p_df["dTmean_hw"] - p_df["dTmean_nhw"])
        )
        p_df["dn_res"] = (
            (p_df["dTn_hw"] - p_df["dTn_nhw"])
            - (p_df["dTmean_hw"] - p_df["dTmean_nhw"])
        )
        p_df["d_amp"] = p_df["dAmp1_hw"] - p_df["dAmp1_nhw"]

        sc = ax.scatter(
            p_df["dx_res"], p_df["dn_res"],
            c=p_df["d_amp"],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0),
            s=130,
            alpha=0.7,
            edgecolors="black",
            lw=0.8
        )

        x_grid = np.linspace(p_df["dx_res"].min(), p_df["dx_res"].max(), 100)
        slope, inter, p_v, y_fit, ci = _fit_line_ci(
            p_df["dx_res"].values,
            p_df["dn_res"].values,
            x_grid
        )

        ax.plot(x_grid, y_fit, "k-", lw=4, label=f"Fit (slope={slope:.2f})")
        ax.fill_between(x_grid, y_fit - ci, y_fit + ci, color="black", alpha=0.15)
        ax.plot(x_grid, -x_grid, "k--", lw=2, alpha=0.5, label="Theoretical (-1)")

        ax.set_xlabel(r"$R_x$ (°C)")
        ax.set_ylabel(r"$R_n$ (°C)")
        ax.legend(frameon=False)

        cax = inset_axes(
            ax,
            width="3%",
            height="40%",
            loc="lower left",
            bbox_to_anchor=(0.05, 0.08, 1, 1),
            bbox_transform=ax.transAxes
        )
        plt.colorbar(sc, cax=cax).set_label(r"$R_{amp}$ (°C)")

        if add_numbered_label:
            add_panel_label(
                ax, "c",
                "Daytime damping is coupled\nto nighttime amplification"
            )
        else:
            ax.set_title(
                "Daytime damping is coupled\nto nighttime amplification",
                loc="center",
                fontweight="bold",
                pad=18,
                fontsize=SUBTITLE_FONTSIZE
            )

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 20,
        "axes.labelsize": 26,
        "axes.titlesize": SUBTITLE_FONTSIZE,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "axes.linewidth": 2.5,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }):
        fig, axes = plt.subplots(2, 2, figsize=(26, 24), dpi=600)
        plt.subplots_adjust(wspace=0.35, hspace=0.45)

        # ============================================================
        # (a) State migration — UNCHANGED logic
        # ============================================================
        ax_a = axes[0, 0]

        all_x = np.concatenate([paired["dTmean_nhw"].values, paired["dTmean_hw"].values])
        all_y = np.concatenate([paired["dAmp1_nhw"].values, paired["dAmp1_hw"].values])
        x_min, x_max = np.nanpercentile(all_x, [1, 99])
        y_min, y_max = np.nanpercentile(all_y, [1, 99])
        x_pad = (x_max - x_min) * 0.16
        y_pad = (y_max - y_min) * 0.16
        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        def _binned_vectors_local(g, nx=11, ny=11, min_count=2):
            g = g.copy()
            g["dx"] = g["dTmean_hw"] - g["dTmean_nhw"]
            g["dy"] = g["dAmp1_hw"]  - g["dAmp1_nhw"]
            g = g[np.isfinite(g["dTmean_nhw"]) & np.isfinite(g["dAmp1_nhw"]) &
                  np.isfinite(g["dx"]) & np.isfinite(g["dy"])].copy()
            if len(g) == 0: return pd.DataFrame()
            x_bins = np.linspace(xlim[0], xlim[1], nx + 1)
            y_bins = np.linspace(ylim[0], ylim[1], ny + 1)
            g["xb"] = pd.cut(g["dTmean_nhw"], bins=x_bins, labels=False, include_lowest=True)
            g["yb"] = pd.cut(g["dAmp1_nhw"],  bins=y_bins, labels=False, include_lowest=True)
            out = g.dropna(subset=["xb", "yb"]).groupby(["xb", "yb"], as_index=False).agg(
                     x0=("dTmean_nhw", "mean"), y0=("dAmp1_nhw", "mean"),
                     dx=("dx", "mean"), dy=("dy", "mean"), n=("dx", "size"))
            return out[out["n"] >= min_count].copy()

        for grp_name, color in [("UHI", COLOR_UHI), ("UCI", COLOR_UCI)]:
            g_raw = paired[paired["group"] == grp_name]
            ax_a.quiver(
                g_raw["dTmean_nhw"], g_raw["dAmp1_nhw"],
                g_raw["dTmean_hw"] - g_raw["dTmean_nhw"],
                g_raw["dAmp1_hw"] - g_raw["dAmp1_nhw"],
                angles="xy", scale_units="xy", scale=1,
                color=color, alpha=0.12, width=0.0012, zorder=1
            )

        for group_name, color in [("UHI", COLOR_UHI), ("UCI", COLOR_UCI)]:
            g = paired[paired["group"] == group_name].copy()
            if len(g) == 0: continue
            vec = _binned_vectors_local(g, nx=11, ny=11, min_count=2)
            if len(vec) == 0: continue
            ax_a.quiver(
                vec["x0"], vec["y0"], vec["dx"], vec["dy"],
                angles="xy", scale_units="xy", scale=1,
                color=color, alpha=0.78,
                width=0.0042,
                headwidth=3.2,
                headlength=4.2,
                headaxislength=3.6,
                label=f"{group_name} mean flow",
                zorder=3,
            )

        x_line_range = np.linspace(xlim[0], xlim[1], 200)
        ax_a.plot(x_line_range, -x_line_range, linestyle="--", color="black",
                  lw=1.5, alpha=0.72, zorder=2, label=r"$\Delta T_x=0$ isoline")

        ax_a.text(0.03, 0.97,
                  "Background: Original vectors\nArrows: mean HW−NHW flow",
                  transform=ax_a.transAxes, ha="left", va="top", fontsize=20,
                  bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2))

        ax_a.axhline(0, color="#999999", linestyle=":", lw=0.8, zorder=1)
        ax_a.axvline(0, color="#999999", linestyle=":", lw=0.8, zorder=1)
        ax_a.set_xlim(xlim); ax_a.set_ylim(ylim)
        ax_a.set_xlabel(r"$\Delta T_{mean}$ (°C)")
        ax_a.set_ylabel(r"$\Delta Amp$ (°C)")
        ax_a.legend(frameon=False, loc="best", fontsize=22)
        ax_a.grid(True, lw=0.25, alpha=0.16)

        add_panel_label(
            ax_a, "a",
            "Heatwave-induced state migration\nin mean–amplitude space"
        )

        # ============================================================
        # (b) Migration angle panel
        #     Move original panel c content to the old panel-b position.
        # ============================================================
        ax_b = axes[0, 1]
        angle_df_for_b = _build_hw_shift_angle_table_for_panel_c()
        _draw_hw_shift_angle_panel(ax_b, angle_df_for_b)

        add_panel_label(
            ax_b, "b",
            "Migration direction\nby UHI/UCI regime"
        )

        # ============================================================
        # (c) Mechanism decomposition
        #     Move original panel b content to the old panel-c position.
        # ============================================================
        ax_c = axes[1, 0]

        def get_decomp_final(grp, target):
            g = paired[paired["group"] == grp]
            dm = (g["dTmean_hw"] - g["dTmean_nhw"]).mean()
            da = (g["dAmp1_hw"]  - g["dAmp1_nhw"]).mean()
            total = (g[f"{target}_hw"] - g[f"{target}_nhw"]).mean()
            c_mean = dm
            c_amp  = da if target == "dTx" else -da
            c_res  = total - (c_mean + c_amp)
            return c_mean, c_amp, c_res, total

        cases = [("UHI","dTx"), ("UHI","dTn"), ("UCI","dTx"), ("UCI","dTn")]
        x_pos = np.arange(len(cases))
        bw = 0.52

        for i, (grp, tgt) in enumerate(cases):
            cm, ca, cr, ct = get_decomp_final(grp, tgt)
            if grp == "UHI":
                base_color = "#d62728"
                light_color = "#ff9999"
            else:
                base_color = "#1f77b4"
                light_color = "#a0cbe8"

            ax_c.bar(i, cm, bw, color=base_color,
                     edgecolor='black', lw=1.8,
                     label=r"$\Delta T_{mean}$ contribution" if i == 0 else "")
            ax_c.bar(i, ca, bw, bottom=cm, color=light_color,
                     hatch='////', edgecolor='black', lw=1.8,
                     label=r"$\pm\Delta Amp$ effect" if i == 0 else "")
            ax_c.bar(i, cr, bw, bottom=cm + ca, color='#aaaaaa',
                     alpha=0.6, edgecolor='black', lw=1.8,
                     label="Shape/Residual" if i == 0 else "")
            ax_c.scatter(i, ct, color='black', marker='D', s=180, zorder=5,
                         label="Observed Total" if i == 0 else "")

        ax_c.axhline(0, color='black', lw=2)
        ax_c.set_xticks(x_pos)
        ax_c.set_xticklabels([
            r"UHI $R_x$",
            r"UHI $R_n$",
            r"UCI $R_x$",
            r"UCI $R_n$"
        ])
        ax_c.set_ylabel(r"$R_a$")
        ax_c.set_ylim(-0.2, 0.8)
        ax_c.legend(frameon=False, loc='upper left', ncol=2,
                    fontsize=22, handleheight=2, handlelength=3)
        ax_c.grid(True, axis='y', lw=0.3, alpha=0.2)

        add_panel_label(
            ax_c, "c",
            "Mean and amplitude contributions\nto nocturnal amplification"
        )

        # ============================================================
        # Original Dynamics panel c exported as standalone figure
        # No panel label / no numbering.
        # ============================================================
        standalone_dir = os.path.join(output_dir, "plots")
        os.makedirs(standalone_dir, exist_ok=True)

        fig_c_single, ax_c_single = plt.subplots(figsize=(12, 10), dpi=600)
        _draw_original_compensation_panel(
            ax_c_single,
            add_numbered_label=False
        )
        style_full_box(ax_c_single, lw=2.5)

        fig_c_single.savefig(
            os.path.join(
                standalone_dir,
                "Figure_Dynamics_Consistent_original_panel_c.png"
            ),
            dpi=600,
            bbox_inches="tight"
        )
        fig_c_single.savefig(
            os.path.join(
                standalone_dir,
                "Figure_Dynamics_Consistent_original_panel_c.pdf"
            ),
            bbox_inches="tight"
        )
        plt.close(fig_c_single)

        # ============================================================
        # Original Dynamics panel c exported as standalone figure
        # No panel label / no numbering.
        # ============================================================
        standalone_dir = os.path.join(output_dir, "plots")
        os.makedirs(standalone_dir, exist_ok=True)

        fig_c_single, ax_c_single = plt.subplots(figsize=(12, 10), dpi=600)
        _draw_original_compensation_panel(
            ax_c_single,
            add_numbered_label=False
        )
        style_full_box(ax_c_single, lw=2.5)

        fig_c_single.savefig(
            os.path.join(
                standalone_dir,
                "Figure_Dynamics_Consistent_original_panel_c.png"
            ),
            dpi=600,
            bbox_inches="tight"
        )
        fig_c_single.savefig(
            os.path.join(
                standalone_dir,
                "Figure_Dynamics_Consistent_original_panel_c.pdf"
            ),
            bbox_inches="tight"
        )
        plt.close(fig_c_single)
        # ============================================================
        # (d) Thermal hysteresis — UNCHANGED logic
        # ============================================================
        ax_d = axes[1, 1]
        style_map = {
            ("UHI", "non_heatwave"): (COLOR_UHI, "--", "UHI NHW"),
            ("UHI", "heatwave"):     (COLOR_UHI, "-",  "UHI HW"),
            ("UCI", "non_heatwave"): (COLOR_UCI, "--", "UCI NHW"),
            ("UCI", "heatwave"):     (COLOR_UCI, "-",  "UCI HW"),
        }

        for (grp, prd), (color, ls, lab) in style_map.items():
            sub = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)]
            if sub.empty: continue
            x_curve = np.nanmean(sub[r_cols].values, axis=0)
            y_curve = np.nanmean(sub[u_cols].values - sub[r_cols].values, axis=0)

            area = _loop_area(x_curve, y_curve)
            lag = _phase_lag_hours(x_curve, y_curve)

            ax_d.plot(x_curve, y_curve, color=color, ls=ls, lw=4, alpha=0.8,
                      label=f"{lab}: area={area:.2f}, lag={lag:.2f}h")
            for i in [5, 11, 17]:
                ax_d.annotate("", xy=(x_curve[(i+1) % 24], y_curve[(i+1) % 24]),
                              xytext=(x_curve[i], y_curve[i]),
                              arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
            for h in [0, 6, 12, 18]:
                ax_d.scatter(x_curve[h], y_curve[h], s=70, color=color, zorder=5)
                ax_d.text(x_curve[h], y_curve[h], f"{h:02d}", fontsize=20,
                          color=color, fontweight='bold', ha='left', va='bottom')

        ax_d.set_xlabel(r"$T_{a,r}$ (°C)")
        ax_d.set_ylabel(r"$\Delta T_a$ (°C)")
        ax_d.set_ylim(-2.0, 2.3)
        ax_d.legend(frameon=False, loc='lower right', fontsize=22)

        add_panel_label(
            ax_d, "d",
            "Hysteresis between rural background\ntemperature and urban–rural contrast"
        )

        # ============================================================
        # Final styling (4 spines + no top/right ticks)
        # ============================================================
        for ax in axes.flatten():
            style_full_box(ax, lw=2.5)

    out_dir = os.path.join(output_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, "Figure_Dynamics_Consistent_Final.png")
    fig.savefig(fpath, dpi=600, bbox_inches='tight')


def plot_fig3paneld_hysteresis_variants(all_df, output_dir,
                                        max_raw_per_group_period=80,
                                        random_seed=20260530):
    """
    Standalone Fig. 3 panel d variants.
    Does NOT modify the original plot_combined_figure_dynamics_consistent().

    Outputs:
      1) Fig3paneld_raw_background_loops.png/pdf
         Mean loops with pair-level raw loops in the background.

      2) Fig3paneld_uncertainty_envelope_loops.png/pdf
         Mean loops with hourly covariance ellipses showing loop variability.

    x-axis: rural Ta
    y-axis: ΔTa = urban Ta - rural Ta
    """

    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    # -----------------------------
    # Required columns
    # -----------------------------
    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]

    missing = [c for c in (u_cols + r_cols + ["group", "period"]) if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Fig3paneld variants: {missing[:10]} ...")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_precise_peak_time(signal):
        vals = np.asarray(signal, dtype=float)
        if np.all(np.isnan(vals)):
            return np.nan
        vals = np.where(np.isfinite(vals), vals, np.nanmean(vals))
        sig_detrend = vals - np.mean(vals)
        fft_vals = np.fft.fft(sig_detrend)
        t_fine = np.linspace(0, 24, 2400, endpoint=False)
        recon = np.zeros_like(t_fine)
        for k in [1, 2]:
            recon += np.real(fft_vals[k] * np.exp(1j * 2 * np.pi * k * t_fine / 24))
        return t_fine[np.argmax(recon)]

    def _phase_lag_hours(x_curve, y_curve):
        t_peak_x = _get_precise_peak_time(x_curve)
        t_peak_y = _get_precise_peak_time(y_curve)
        if pd.isna(t_peak_x) or pd.isna(t_peak_y):
            return np.nan
        return (t_peak_y - t_peak_x) % 24

    def _loop_area(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 4:
            return np.nan
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )

    def _curves_from_subset(sub):
        """
        Return pair-level x/y arrays and mean curves.
        x = rural Ta
        y = urban Ta - rural Ta
        """
        x_arr = sub[r_cols].to_numpy(dtype=float)
        u_arr = sub[u_cols].to_numpy(dtype=float)
        r_arr = sub[r_cols].to_numpy(dtype=float)
        y_arr = u_arr - r_arr

        x_mean = np.nanmean(x_arr, axis=0)
        y_mean = np.nanmean(y_arr, axis=0)

        return x_arr, y_arr, x_mean, y_mean

    def _style_ax(ax):
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.2)

        ax.tick_params(direction="out", width=1.0, length=4, top=False, right=False)
        ax.grid(True, lw=0.35, alpha=0.22)
        ax.axhline(0, color="#666666", lw=0.8, ls="--", alpha=0.65, zorder=0)

        ax.set_xlabel(r"$T_{a,r}$ (°C)")
        ax.set_ylabel(r"$\Delta T_a$ (°C)")

    def _add_direction_arrows(ax, x, y, color, lw=1.4, alpha=0.85):
        """
        Add a few small arrows along the mean loop.
        """
        for h in [5, 11, 17]:
            ax.annotate(
                "",
                xy=(x[(h + 1) % 24], y[(h + 1) % 24]),
                xytext=(x[h], y[h]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=5
            )

    def _add_hour_labels(ax, x, y, color):
        """
        Label only key hours to avoid clutter.
        """
        for h in [0, 6, 12, 18]:
            ax.scatter(x[h], y[h], s=28, color=color, zorder=6)
            ax.text(
                x[h], y[h], f"{h:02d}",
                fontsize=7.5,
                color=color,
                fontweight="bold",
                ha="left",
                va="bottom",
                zorder=7
            )

    def _add_cov_ellipse(ax, x_vals, y_vals, color,
                         n_std=1.0, alpha=0.13, lw=0.7):
        """
        Draw covariance ellipse for one hour across station pairs.
        Represents ~1 standard-deviation spread in 2D x-y loop space.
        """
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]

        if len(x_vals) < 5:
            return

        cov = np.cov(x_vals, y_vals)
        if not np.all(np.isfinite(cov)):
            return

        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0)

        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(vals)

        ell = Ellipse(
            xy=(np.nanmean(x_vals), np.nanmean(y_vals)),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            lw=lw,
            zorder=1
        )
        ax.add_patch(ell)

    # -----------------------------
    # Plot settings
    # -----------------------------
    COLOR_UHI = "#d62728"
    COLOR_UCI = "#1f77b4"

    style_specs = {
        ("UHI", "non_heatwave"): dict(color=COLOR_UHI, ls="--", label="UHI NHW", alpha_raw=0.1),
        ("UHI", "heatwave"):     dict(color=COLOR_UHI, ls="-",  label="UHI HW",  alpha_raw=0.2),
        ("UCI", "non_heatwave"): dict(color=COLOR_UCI, ls="--", label="UCI NHW", alpha_raw=0.1),
        ("UCI", "heatwave"):     dict(color=COLOR_UCI, ls="-",  label="UCI HW",  alpha_raw=0.2),
    }

    rng = np.random.default_rng(random_seed)

    # Precompute global axis limits from valid loops
    all_x_vals = []
    all_y_vals = []

    for (grp, prd), spec in style_specs.items():
        sub = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)].copy()
        if sub.empty:
            continue
        x_arr, y_arr, _, _ = _curves_from_subset(sub)
        all_x_vals.append(x_arr[np.isfinite(x_arr)])
        all_y_vals.append(y_arr[np.isfinite(y_arr)])

    if not all_x_vals or not all_y_vals:
        raise ValueError("No valid loop data found for Fig3paneld variants.")

    all_x_vals = np.concatenate(all_x_vals)
    all_y_vals = np.concatenate(all_y_vals)

    x0, x1 = np.nanpercentile(all_x_vals, [1, 99])
    y0, y1 = np.nanpercentile(all_y_vals, [1, 99])
    xpad = 0.08 * (x1 - x0)
    ypad = 0.16 * (y1 - y0)

    xlim_loop = (x0 - xpad, x1 + xpad)
    ylim_loop = (y0 - ypad, y1 + ypad)

    out_dir = os.path.join(output_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # Figure 1: raw background loops + mean loops
    # ============================================================
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }):

        fig, ax = plt.subplots(figsize=(3.6, 3.1), dpi=600)

        for (grp, prd), spec in style_specs.items():
            sub = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)].copy()
            if sub.empty:
                continue

            if len(sub) > max_raw_per_group_period:
                sub = sub.sample(n=max_raw_per_group_period, random_state=random_seed)

            x_arr, y_arr, x_mean, y_mean = _curves_from_subset(sub)

            # pair-level raw loops in the background
            for i in range(x_arr.shape[0]):
                ax.plot(
                    x_arr[i, :], y_arr[i, :],
                    color=spec["color"],
                    ls=spec["ls"],
                    lw=0.45,
                    alpha=spec["alpha_raw"],
                    zorder=1
                )

        # mean loops on top, calculated from all data not just sampled rows
        for (grp, prd), spec in style_specs.items():
            sub_full = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)].copy()
            if sub_full.empty:
                continue

            x_arr, y_arr, x_mean, y_mean = _curves_from_subset(sub_full)

            area = _loop_area(x_mean, y_mean)
            lag = _phase_lag_hours(x_mean, y_mean)

            ax.plot(
                x_mean, y_mean,
                color=spec["color"],
                ls=spec["ls"],
                lw=1.9,
                alpha=0.95,
                zorder=4,
                label=f'{spec["label"]}: A={area:.2f}, lag={lag:.1f}h'
            )

            _add_direction_arrows(ax, x_mean, y_mean, spec["color"], lw=1.1, alpha=0.9)
            _add_hour_labels(ax, x_mean, y_mean, spec["color"])

        _style_ax(ax)
        ax.set_xlim(xlim_loop)
        ax.set_ylim(ylim_loop)
        ax.set_title(
            "Hysteresis loops with pair-level background",
            loc="left",
            fontweight="bold",
            pad=5
        )

        ax.legend(frameon=False, loc="best", fontsize=5.8, handlelength=2.2)

        f_png = os.path.join(out_dir, "Fig3paneld_raw_background_loops.png")
        f_pdf = os.path.join(out_dir, "Fig3paneld_raw_background_loops.pdf")
        fig.savefig(f_png, dpi=600, bbox_inches="tight")
        fig.savefig(f_pdf, bbox_inches="tight")
        plt.close(fig)

    # ============================================================
    # Figure 2: uncertainty envelope / covariance ellipses
    # ============================================================
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }):

        fig, ax = plt.subplots(figsize=(3.6, 3.1), dpi=600)

        for (grp, prd), spec in style_specs.items():
            sub = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)].copy()
            if sub.empty:
                continue

            x_arr, y_arr, x_mean, y_mean = _curves_from_subset(sub)

            # Draw 1σ covariance ellipses every 3 hours.
            # This is cleaner than drawing all raw loops and better shows loop spread.
            for h in range(0, 24, 3):
                _add_cov_ellipse(
                    ax,
                    x_arr[:, h],
                    y_arr[:, h],
                    color=spec["color"],
                    n_std=1.0,
                    alpha=0.11 if prd == "heatwave" else 0.07,
                    lw=0.5
                )

            area = _loop_area(x_mean, y_mean)
            lag = _phase_lag_hours(x_mean, y_mean)

            ax.plot(
                x_mean, y_mean,
                color=spec["color"],
                ls=spec["ls"],
                lw=2.0,
                alpha=0.96,
                zorder=4,
                label=f'{spec["label"]}: A={area:.2f}, lag={lag:.1f}h'
            )

            _add_direction_arrows(ax, x_mean, y_mean, spec["color"], lw=1.1, alpha=0.9)
            _add_hour_labels(ax, x_mean, y_mean, spec["color"])

        _style_ax(ax)
        ax.set_xlim(xlim_loop)
        ax.set_ylim(ylim_loop)
        ax.set_title(
            "Hysteresis-loop variability across station pairs",
            loc="left",
            fontweight="bold",
            pad=5
        )

        # Custom legend for uncertainty
        handles = [
            Line2D([0], [0], color=COLOR_UHI, lw=2.0, ls="-", label="UHI HW"),
            Line2D([0], [0], color=COLOR_UHI, lw=2.0, ls="--", label="UHI NHW"),
            Line2D([0], [0], color=COLOR_UCI, lw=2.0, ls="-", label="UCI HW"),
            Line2D([0], [0], color=COLOR_UCI, lw=2.0, ls="--", label="UCI NHW"),
            Line2D([0], [0], color="black", lw=0, marker="o",
                   markerfacecolor="lightgray", markeredgecolor="none",
                   alpha=0.45, label="1σ hourly spread"),
        ]
        ax.legend(handles=handles, frameon=False, loc="best", fontsize=6.2, handlelength=2.2)

        f_png = os.path.join(out_dir, "Fig3paneld_uncertainty_envelopes.png")
        f_pdf = os.path.join(out_dir, "Fig3paneld_uncertainty_envelopes.pdf")
        fig.savefig(f_png, dpi=600, bbox_inches="tight")
        fig.savefig(f_pdf, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {os.path.join(out_dir, 'Fig3paneld_raw_background_loops.png')}")
    print(f"  Saved: {os.path.join(out_dir, 'Fig3paneld_uncertainty_envelopes.png')}")

def plot_fig3d_supplement(all_df, output_dir,
                          max_raw_per_group_period=80,
                          random_seed=20260530):
    """
    Supplementary figure for Fig. 3d hysteresis loops.

    Panel a:
        Mean hysteresis loops with pair-level raw loops in the background.

    Panel b:
        Mean hysteresis loops with hourly covariance ellipses
        (drawn only at 00, 06, 12, 18) to show loop spread.

    Output:
        output_dir/plots/fig3d_supplement.png
        output_dir/plots/fig3d_supplement.pdf
    """

    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    # ---------------------------------------------------------------------
    # Shared format helpers (aligned with your other Nature-style figures)
    # ---------------------------------------------------------------------
    SUBTITLE_FONTSIZE = 24
    LABEL_FONTSIZE = 32

    def add_panel_label(ax, label, subtitle, pad=18):
        ax.set_title(subtitle, loc="center", fontweight="bold",
                     pad=pad, fontsize=SUBTITLE_FONTSIZE)
        ax.text(0.0, 1.04, label,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontweight="bold", fontsize=LABEL_FONTSIZE)

    def style_full_box(ax, lw=2.5):
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(lw)
        ax.tick_params(top=False, right=False,
                       direction="out", width=1.6, length=6)

    # ---------------------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------------------
    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]

    required_cols = ["group", "period"] + u_cols + r_cols
    missing = [c for c in required_cols if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for fig3d supplement: {missing[:10]} ...")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _get_precise_peak_time(signal):
        vals = np.asarray(signal, dtype=float)
        if np.all(np.isnan(vals)):
            return np.nan
        vals = np.where(np.isfinite(vals), vals, np.nanmean(vals))
        sig_detrend = vals - np.mean(vals)
        fft_vals = np.fft.fft(sig_detrend)
        t_fine = np.linspace(0, 24, 2400, endpoint=False)
        recon = np.zeros_like(t_fine)
        for k in [1, 2]:
            recon += np.real(fft_vals[k] * np.exp(1j * 2 * np.pi * k * t_fine / 24))
        return t_fine[np.argmax(recon)]

    def _phase_lag_hours(x_curve, y_curve):
        t_peak_x = _get_precise_peak_time(x_curve)
        t_peak_y = _get_precise_peak_time(y_curve)
        if pd.isna(t_peak_x) or pd.isna(t_peak_y):
            return np.nan
        return (t_peak_y - t_peak_x) % 24

    def _loop_area(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 4:
            return np.nan
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )

    def _curves_from_subset(sub):
        """
        Return pair-level curves and mean curves.

        x = rural Ta
        y = urban Ta - rural Ta
        """
        x_arr = sub[r_cols].to_numpy(dtype=float)
        u_arr = sub[u_cols].to_numpy(dtype=float)
        r_arr = sub[r_cols].to_numpy(dtype=float)
        y_arr = u_arr - r_arr

        x_mean = np.nanmean(x_arr, axis=0)
        y_mean = np.nanmean(y_arr, axis=0)

        return x_arr, y_arr, x_mean, y_mean

    def _style_loop_axes(ax):
        ax.axhline(0, color="#999999", linestyle=":", lw=0.8, zorder=0)
        ax.set_xlabel(r"$T_{a,r}$ (°C)")
        ax.set_ylabel(r"$\Delta T_a$ (°C)")
        ax.grid(True, lw=0.25, alpha=0.16)
        ax.set_axisbelow(True)

    def _add_direction_arrows(ax, x, y, color, lw=2.0, alpha=0.9):
        """
        Add arrows to show loop direction.
        """
        for i in [5, 11, 17]:
            ax.annotate(
                "",
                xy=(x[(i + 1) % 24], y[(i + 1) % 24]),
                xytext=(x[i], y[i]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=5
            )

    def _add_hour_labels(ax, x, y, color):
        """
        Label only 00, 06, 12, 18 to reduce clutter.
        """
        for h in [0, 6, 12, 18]:
            ax.scatter(x[h], y[h], s=60, color=color, zorder=6)
            ax.text(
                x[h], y[h], f"{h:02d}",
                fontsize=13,
                color=color,
                fontweight="bold",
                ha="left",
                va="bottom",
                zorder=7
            )

    def _add_cov_ellipse(ax, x_vals, y_vals, color,
                         n_std=1.0, alpha=0.10, lw=0.8):
        """
        Draw covariance ellipse at one hour across station pairs.
        """
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]

        if len(x_vals) < 5:
            return

        cov = np.cov(x_vals, y_vals)
        if not np.all(np.isfinite(cov)):
            return

        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(vals)

        ell = Ellipse(
            xy=(np.nanmean(x_vals), np.nanmean(y_vals)),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            lw=lw,
            alpha=alpha,
            zorder=1
        )
        ax.add_patch(ell)

    # ---------------------------------------------------------------------
    # Style map
    # ---------------------------------------------------------------------
    COLOR_UHI = "#d62728"
    COLOR_UCI = "#1f77b4"

    style_specs = {
        ("UHI", "non_heatwave"): dict(color=COLOR_UHI, ls="--", label="UHI NHW", alpha_raw=0.2),
        ("UHI", "heatwave"):     dict(color=COLOR_UHI, ls="-",  label="UHI HW",  alpha_raw=0.3),
        ("UCI", "non_heatwave"): dict(color=COLOR_UCI, ls="--", label="UCI NHW", alpha_raw=0.2),
        ("UCI", "heatwave"):     dict(color=COLOR_UCI, ls="-",  label="UCI HW",  alpha_raw=0.3),
    }

    # ---------------------------------------------------------------------
    # Global axis limits
    # ---------------------------------------------------------------------
    all_x_vals = []
    all_y_vals = []

    for (grp, prd), spec in style_specs.items():
        sub = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)].copy()
        if sub.empty:
            continue
        x_arr, y_arr, _, _ = _curves_from_subset(sub)
        all_x_vals.append(x_arr[np.isfinite(x_arr)])
        all_y_vals.append(y_arr[np.isfinite(y_arr)])

    if not all_x_vals or not all_y_vals:
        raise ValueError("No valid hysteresis-loop data found.")

    all_x_vals = np.concatenate(all_x_vals)
    all_y_vals = np.concatenate(all_y_vals)

    x0, x1 = np.nanpercentile(all_x_vals, [1, 99])
    y0, y1 = np.nanpercentile(all_y_vals, [1, 99])

    xpad = 0.08 * (x1 - x0)
    ypad = 0.16 * (y1 - y0)

    xlim_loop = (x0 - xpad, x1 + xpad)
    ylim_loop = (y0 - ypad, y1 + ypad)

    rng = np.random.default_rng(random_seed)

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 20,
        "axes.labelsize": 26,
        "axes.titlesize": SUBTITLE_FONTSIZE,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 18,
        "axes.linewidth": 2.5,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }):

        fig, axes = plt.subplots(1, 2, figsize=(26, 12), dpi=600)
        plt.subplots_adjust(wspace=0.35, hspace=0.0)

        ax_a = axes[0]
        ax_b = axes[1]

        # ============================================================
        # Panel a: raw background loops + mean loops
        # ============================================================
        for (grp, prd), spec in style_specs.items():
            sub = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)].copy()
            if sub.empty:
                continue

            # Background raw loops: sample if too many
            if len(sub) > max_raw_per_group_period:
                sub_raw = sub.sample(n=max_raw_per_group_period, random_state=random_seed)
            else:
                sub_raw = sub.copy()

            x_arr_raw, y_arr_raw, _, _ = _curves_from_subset(sub_raw)

            for i in range(x_arr_raw.shape[0]):
                ax_a.plot(
                    x_arr_raw[i, :], y_arr_raw[i, :],
                    color=spec["color"],
                    ls=spec["ls"],
                    lw=0.55,
                    alpha=spec["alpha_raw"],
                    zorder=1
                )

            # Mean loop from full data
            x_arr, y_arr, x_mean, y_mean = _curves_from_subset(sub)
            area = _loop_area(x_mean, y_mean)
            lag = _phase_lag_hours(x_mean, y_mean)

            ax_a.plot(
                x_mean, y_mean,
                color=spec["color"],
                ls=spec["ls"],
                lw=3.2,
                alpha=0.95,
                zorder=4,
                label=f'{spec["label"]}: area={area:.2f}, lag={lag:.2f}h'
            )

            _add_direction_arrows(ax_a, x_mean, y_mean, spec["color"], lw=2.0, alpha=0.9)
            _add_hour_labels(ax_a, x_mean, y_mean, spec["color"])

        _style_loop_axes(ax_a)
        ax_a.set_xlim(xlim_loop)
        ax_a.set_ylim(ylim_loop)
        ax_a.legend(frameon=False, loc="lower right", fontsize=15)

        add_panel_label(
            ax_a, "a",
            "Hysteresis loops with pair-level\nbackground trajectories"
        )

        # ============================================================
        # Panel b: uncertainty ellipses + mean loops
        # ============================================================
        for (grp, prd), spec in style_specs.items():
            sub = all_df[(all_df["group"] == grp) & (all_df["period"] == prd)].copy()
            if sub.empty:
                continue

            x_arr, y_arr, x_mean, y_mean = _curves_from_subset(sub)

            # Draw ellipses only at midnight, morning, noon, evening
            for h in [0, 6, 12, 18]:
                _add_cov_ellipse(
                    ax_b,
                    x_arr[:, h],
                    y_arr[:, h],
                    color=spec["color"],
                    n_std=1.0,
                    alpha=0.10 if prd == "heatwave" else 0.07,
                    lw=0.7
                )

            area = _loop_area(x_mean, y_mean)
            lag = _phase_lag_hours(x_mean, y_mean)

            ax_b.plot(
                x_mean, y_mean,
                color=spec["color"],
                ls=spec["ls"],
                lw=3.2,
                alpha=0.96,
                zorder=4,
                label=f'{spec["label"]}: area={area:.2f}, lag={lag:.2f}h'
            )

            _add_direction_arrows(ax_b, x_mean, y_mean, spec["color"], lw=2.0, alpha=0.9)
            _add_hour_labels(ax_b, x_mean, y_mean, spec["color"])

        _style_loop_axes(ax_b)
        ax_b.set_xlim(xlim_loop)
        ax_b.set_ylim(ylim_loop)

        # Custom legend for spread
        handles = [
            Line2D([0], [0], color=COLOR_UHI, lw=3.2, ls="-", label="UHI HW"),
            Line2D([0], [0], color=COLOR_UHI, lw=3.2, ls="--", label="UHI NHW"),
            Line2D([0], [0], color=COLOR_UCI, lw=3.2, ls="-", label="UCI HW"),
            Line2D([0], [0], color=COLOR_UCI, lw=3.2, ls="--", label="UCI NHW"),
            Line2D([0], [0], color="black", lw=0, marker="o",
                   markerfacecolor="lightgray", markeredgecolor="none",
                   alpha=0.45, label="Hourly spread ellipse"),
        ]
        ax_b.legend(handles=handles, frameon=False, loc="lower right", fontsize=15)

        add_panel_label(
            ax_b, "b",
            "Hysteresis-loop variability\nat 00, 06, 12 and 18 h"
        )

        # Final styling
        for ax in axes.flatten():
            style_full_box(ax, lw=2.5)

        out_dir = os.path.join(output_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)

        fpath_png = os.path.join(out_dir, "fig3d_supplement.png")
        fpath_pdf = os.path.join(out_dir, "fig3d_supplement.pdf")

        fig.savefig(fpath_png, dpi=600, bbox_inches="tight")
        fig.savefig(fpath_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved supplementary figure: {fpath_png}")
        print(f"  Saved supplementary figure: {fpath_pdf}")

def apply_nature_style():
    """设置学术期刊风格的绘图参数"""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def plot_reference_style_4panel_maps_v2(all_df, output_dir):
    """
    严格对齐 reference (supplement_period_specific_uhi_maps) 的 4-panel 地图。
    
    布局规范:
      - figsize: (7.2, 4.55), dpi=600
      - GridSpec: 2x2, hspace=0.28, wspace=0.08
      - 边距: left=0.045, right=0.995, top=0.90, bottom=0.085
      - 投影: Robinson(central_longitude=0)
      - 底色: 白海洋 + 灰陆地(#f7f7f7)
      - 配色: RdYlBu_r
      - 色标: 每图下方独立小色标,居中对齐
      - 线宽: 0.55-0.60 统一
      - 字号: 子图编号 8.5pt, 标题 6.8pt, 色标 6.5pt, 刻度 6.0pt
    
    数据保留: NHW/HW × ΔTmean/ΔAmp 四面板,所有数据点统一样式
    """
    if not HAS_CARTOPY:
        print("  [4Panel Map] Cartopy not installed, skipping.")
        return

    # ── 数据配置 ──
    periods = [("non_heatwave", "Non-heatwave"), ("heatwave", "Heatwave")]
    metrics = [("dTmean", r"$\Delta T_{mean}$"), ("dAmp1", r"$\Delta Amp$")]
    
    # 每个指标独立计算色轴范围(98百分位,至少2.0)
    vmax_dict = {
        m_col: max(2.0, np.ceil(np.nanpercentile(np.abs(all_df[m_col].dropna()), 98)))
        for m_col, _ in metrics
    }
    threshold_dict = {
        m_col: np.nanpercentile(np.abs(all_df[m_col].dropna()), 95)
        for m_col, _ in metrics
    }

    # ── 投影与画布(严格对齐 reference) ──
    PLOT_PROJECTION = ccrs.Robinson(central_longitude=0)
    DATA_CRS = ccrs.PlateCarree()
    MAP_CMAP = plt.cm.RdYlBu_r

    fig = plt.figure(figsize=(7.2, 4.55), dpi=600)
    gs = GridSpec(
        2, 2, figure=fig,
        hspace=0.28,
        wspace=0.08,
        left=0.045, right=0.995, top=0.90, bottom=0.085
    )

    # ── 四子图配置 ──
    specs = [
        (0, 0, "dTmean", "non_heatwave", "Non-heatwave " + r"$\Delta T_{mean}$",
         r"$\Delta T_{mean,\mathrm{NHW}}$ (°C)", "a"),
        (0, 1, "dTmean", "heatwave", "Heatwave " + r"$\Delta T_{mean}$",
         r"$\Delta T_{mean,\mathrm{HW}}$ (°C)", "b"),
        (1, 0, "dAmp1", "non_heatwave", "Non-heatwave " + r"$\Delta Amp$",
         r"$\Delta Amp_{\mathrm{NHW}}$ (°C)", "c"),
        (1, 1, "dAmp1", "heatwave", "Heatwave " + r"$\Delta Amp$",
         r"$\Delta Amp_{\mathrm{HW}}$ (°C)", "d"),
    ]

    for row, col, m_col, p_key, title, cbar_label, letter in specs:
        ax = fig.add_subplot(gs[row, col], projection=PLOT_PROJECTION)

        # ── 底图(严格对齐 reference) ──
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#f7f7f7', zorder=0)
        ax.add_feature(
            cfeature.COASTLINE,
            linewidth=0.25, edgecolor="#505050", zorder=1
        )
        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.15, edgecolor="#aaaaaa", linestyle=":", zorder=1
        )

        # 边框样式(0.60 线宽,黑色 geo spine)
        for s in ['left', 'right', 'top', 'bottom']:
            if s in ax.spines:
                ax.spines[s].set_visible(False)
        if 'geo' in ax.spines:
            ax.spines['geo'].set_visible(True)
            ax.spines['geo'].set_linewidth(0.60)
            ax.spines['geo'].set_edgecolor("black")

        # ── 数据散点(统一样式,无 Top 5% 区分) ──
        data = all_df[all_df["period"] == p_key].dropna(
            subset=["lon_urban", "lat_urban", m_col]
        ).copy()

        if len(data) == 0:
            sc = None
        else:
            vlimit = vmax_dict[m_col]
            threshold = threshold_dict[m_col]

            # outlier: 当前指标绝对值 >= 全样本该指标 95% 分位数
            is_outlier = np.abs(data[m_col]) >= threshold
            is_normal = ~is_outlier

            # 普通点：较小、无黑边，避免和 outlier 混淆
            sc = ax.scatter(
                data.loc[is_normal, "lon_urban"].values,
                data.loc[is_normal, "lat_urban"].values,
                c=data.loc[is_normal, m_col].values,
                cmap=MAP_CMAP,
                vmin=-vlimit, vmax=vlimit,
                s=8, marker="o",
                linewidths=0.0, edgecolors="none",
                alpha=0.78,
                transform=DATA_CRS,
                zorder=3
            )

            # outlier 点：更大、黑色描边、置于上层
            ax.scatter(
                data.loc[is_outlier, "lon_urban"].values,
                data.loc[is_outlier, "lat_urban"].values,
                c=data.loc[is_outlier, m_col].values,
                cmap=MAP_CMAP,
                vmin=-vlimit, vmax=vlimit,
                s=14, marker="o",
                linewidths=0.40, edgecolors="black",
                alpha=1.0,
                transform=DATA_CRS,
                zorder=4
            )

        # ── 标题与编号(严格对齐 reference) ──
        ax.set_title(title, fontsize=6.8, fontweight="bold", pad=4.0)
        ax.text(
            -0.04, 1.04, letter,
            transform=ax.transAxes,
            fontsize=8.5, fontweight="bold",
            ha="left", va="bottom",
            clip_on=False
        )

        # ── Colorbar(每图下方独立小色标,严格对齐 reference) ──
        if sc is not None:
            cax = inset_axes(
                ax,
                width="20%", height="3.5%",
                loc="lower center",
                bbox_to_anchor=(0.0, -0.06, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=0
            )
            cbar = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
            cbar.set_label(cbar_label, fontsize=6.5, labelpad=0.5)
            cbar.ax.tick_params(labelsize=6.0, length=1.2, pad=0.5)
            cbar.outline.set_linewidth(0.5)
            for s in cax.spines.values():
                s.set_visible(False)
    # ── Outlier 图例 ──
    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker="o", color="w",
            label="Standard pair",
            markerfacecolor="#cccccc",
            markeredgecolor="none",
            markersize=4,
            alpha=0.78
        ),
        plt.Line2D(
            [0], [0],
            marker="o", color="w",
            label="Top 5% response",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=5
        )
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=6.0
    )

    # ── 保存(对齐 reference 的 SAVEFIG_KW) ──
    ensure_dir(os.path.join(output_dir, "plots"))
    SAVEFIG_KW = dict(dpi=600, bbox_inches="tight", pad_inches=0.025)
    fpath_png = os.path.join(output_dir, "plots", "Figure_4Panel_Impact_Maps_v3.png")
    fpath_pdf = os.path.join(output_dir, "plots", "Figure_4Panel_Impact_Maps_v3.pdf")
    fig.savefig(fpath_png, **SAVEFIG_KW)
    fig.savefig(fpath_pdf, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved reference-style 4-panel map: {fpath_png}")


# ─────────────────────────────────────────────────────────────

def load_canonical_annual_groups_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return strict pair_id -> annual UHI/UCI classification."""
    required = {"pair_id", "period", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Canonical annual group columns missing: {sorted(missing)}")

    g = df.copy()
    if "hw_method" in g.columns:
        g = g[
            g["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()
    g = g[g["period"].astype(str).str.lower().str.strip().eq("annual")].copy()
    g["pair_id"] = g["pair_id"].astype(str)
    g["annual_group"] = g["group"].astype(str).str.upper().str.strip()
    g = g[g["annual_group"].isin(["UHI", "UCI"])].copy()
    if g.empty:
        raise ValueError("No annual percentile UHI/UCI rows are available.")

    conflicts = g.groupby("pair_id", observed=True)["annual_group"].nunique()
    conflict_ids = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflict_ids:
        raise ValueError(
            "Conflicting annual UHI/UCI groups for "
            f"{len(conflict_ids)} pair(s); examples={conflict_ids[:20]}"
        )
    return g[["pair_id", "annual_group"]].drop_duplicates("pair_id")


def apply_canonical_annual_group(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite period-specific group labels with strict annual labels."""
    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)
    lookup = load_canonical_annual_groups_from_df(out)
    if "group" in out.columns:
        out["group_period_original"] = out["group"]
        out = out.drop(columns=["group"])
    out = out.merge(lookup, on="pair_id", how="inner")
    out = out.rename(columns={"annual_group": "group"})
    return out

# Matched HW–NHW cohort helpers  (added: matched-cohort safe plotting)
# ─────────────────────────────────────────────────────────────
def infer_pair_id_col(df):
    """Infer the pair/station identifier column used across plotting functions."""
    for c in ["pair_id", "station_id", "city_id", "pair_key", "station_pair_id"]:
        if c in df.columns:
            return c
    return None



def build_matched_hw_nhw_cohort(all_df, output_dir=None,
                                required_metrics=("dTmean", "dAmp1", "dTx", "dTn")):
    """Build the strict percentile + annual-group + matched HW/NHW cohort."""
    if all_df is None or len(all_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "period" not in all_df.columns:
        raise ValueError("Column 'period' is required to build matched HW/NHW cohort.")

    df = all_df.copy()
    if "hw_method" in df.columns:
        df = df[
            df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()

    id_col = infer_pair_id_col(df)
    if id_col is None:
        raise ValueError("Cannot find a pair identifier column.")
    if id_col != "pair_id":
        df = df.rename(columns={id_col: "pair_id"})
        id_col = "pair_id"

    # Re-apply strict annual group labels even if the caller already did so.
    annual_lookup = load_canonical_annual_groups_from_df(df)
    if "group" in df.columns:
        df["group_period_original"] = df["group"]
        df = df.drop(columns=["group"])
    df["pair_id"] = df["pair_id"].astype(str)
    df = df.merge(annual_lookup, on="pair_id", how="inner")
    df = df.rename(columns={"annual_group": "group"})

    missing = [c for c in ["pair_id", "period", *required_metrics] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required matched-cohort columns: {missing}")

    for c in required_metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nhw = df.loc[
        df["period"].astype(str).str.lower().eq("non_heatwave"),
        ["pair_id", *required_metrics],
    ].copy()
    hw = df.loc[
        df["period"].astype(str).str.lower().eq("heatwave"),
        ["pair_id", *required_metrics],
    ].copy()
    nhw = nhw.rename(columns={m: f"{m}_nhw" for m in required_metrics})
    hw = hw.rename(columns={m: f"{m}_hw" for m in required_metrics})

    paired = nhw.merge(hw, on="pair_id", how="inner")
    complete = [f"{m}_nhw" for m in required_metrics] + [f"{m}_hw" for m in required_metrics]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=complete)
    paired = paired.merge(annual_lookup, on="pair_id", how="inner")
    paired = paired.rename(columns={"annual_group": "group"})

    if paired.empty:
        print("  [Matched cohort] No complete annual-group HW/NHW pairs.")
        return df.iloc[0:0].copy(), paired, pd.DataFrame()

    for m in required_metrics:
        response_name = {
            "dTmean": "Rmean", "dAmp1": "Ramp", "dTx": "Rx", "dTn": "Rn",
        }.get(m, f"R_{m}")
        paired[response_name] = paired[f"{m}_hw"] - paired[f"{m}_nhw"]

    keep_ids = set(paired["pair_id"].astype(str))
    matched_all = df[df["pair_id"].astype(str).isin(keep_ids)].copy()

    audit_rows = []
    for period in ["annual", "warm_season", "non_heatwave", "heatwave"]:
        sub = matched_all[matched_all["period"].astype(str).str.lower().eq(period)]
        audit_rows.append({
            "cohort": "annual_group_percentile_matched_hw_nhw_required_metrics",
            "period": period,
            "n_rows": int(len(sub)),
            "n_pairs": int(sub["pair_id"].nunique()),
            "n_UHI": int((sub["group"] == "UHI").sum()),
            "n_UCI": int((sub["group"] == "UCI").sum()),
        })
    cohort_audit = pd.DataFrame(audit_rows)

    if output_dir is not None:
        out_data_dir = os.path.join(output_dir, "integrated_fig_data")
        ensure_dir(out_data_dir)
        paired.to_csv(os.path.join(out_data_dir, "matched_hw_nhw_response_cohort.csv"), index=False)
        cohort_audit.to_csv(os.path.join(out_data_dir, "matched_hw_nhw_cohort_audit.csv"), index=False)
        matched_all[["pair_id", "period", "group"]].drop_duplicates().to_csv(
            os.path.join(out_data_dir, "matched_hw_nhw_long_ids_by_period.csv"), index=False
        )
        annual_lookup.to_csv(
            os.path.join(out_data_dir, "canonical_annual_uhi_uci_groups.csv"), index=False
        )

    print(
        f"  [Matched cohort] matched valid pairs={len(paired)}; "
        f"matched long rows={len(matched_all)}"
    )
    return matched_all, paired, cohort_audit



def plot_si_annual_nhw_hw_state_transition(all_df, output_dir):
    """
    Additional SI figure/table: annual → NHW → HW period-specific dTx-sign states.

    The canonical manuscript group remains the upstream annual UHI/UCI label
    (annual dTx >= 0 -> UHI; annual dTx < 0 -> UCI). The states plotted here
    are separate diagnostics based on the sign of period-specific dTx. Positive
    and negative states use strict >0 and <0; exact zero or missing values are
    reported as unclassified and are not converted into canonical groups.
    """
    if all_df is None or len(all_df) == 0:
        print("  [SI annual-NHW-HW] No data, skipping.")
        return

    id_col = infer_pair_id_col(all_df)
    if id_col is None:
        print("  [SI annual-NHW-HW] No pair id column found, skipping.")
        return

    needed = [id_col, "period", "group", "dTx", "dTmean", "dAmp1", "dTn"]
    missing = [c for c in needed if c not in all_df.columns]
    if missing:
        print(f"  [SI annual-NHW-HW] Missing columns: {missing}, skipping.")
        return

    def _period_block(period, suffix):
        cols = [id_col, "group", "dTx", "dTmean", "dAmp1", "dTn"]
        sub = all_df[all_df["period"] == period][cols].copy()
        rename = {
            "group": f"group_{suffix}",
            "dTx": f"dTx_{suffix}",
            "dTmean": f"dTmean_{suffix}",
            "dAmp1": f"dAmp_{suffix}",
            "dTn": f"dTn_{suffix}",
        }
        return sub.rename(columns=rename)

    annual = _period_block("annual", "annual")
    nhw = _period_block("non_heatwave", "nhw")
    hw = _period_block("heatwave", "hw")

    wide = annual.merge(nhw, on=id_col, how="inner").merge(hw, on=id_col, how="inner")
    wide = wide.dropna(subset=["dTx_annual", "dTx_nhw", "dTx_hw"]).copy()

    if len(wide) == 0:
        print("  [SI annual-NHW-HW] No complete annual/NHW/HW matched rows, skipping.")
        return

    # Preserve upstream annual baseline group as the main group label.
    wide["group"] = wide["group_annual"]
    wide = wide[wide["group"].isin(["UHI", "UCI"])].copy()

    def _state(v):
        if not np.isfinite(v):
            return "zero_or_unclassified"
        if v > 0:
            return "positive-dTx state"
        if v < 0:
            return "negative-dTx state"
        return "zero_or_unclassified"

    for suffix in ["annual", "nhw", "hw"]:
        wide[f"state_{suffix}_by_dTx"] = wide[f"dTx_{suffix}"].apply(_state)

    wide["transition_annual_to_nhw"] = wide["state_annual_by_dTx"] + " -> " + wide["state_nhw_by_dTx"]
    wide["transition_nhw_to_hw"] = wide["state_nhw_by_dTx"] + " -> " + wide["state_hw_by_dTx"]
    wide["transition_annual_to_hw"] = wide["state_annual_by_dTx"] + " -> " + wide["state_hw_by_dTx"]

    wide["Rmean"] = wide["dTmean_hw"] - wide["dTmean_nhw"]
    wide["Ramp"] = wide["dAmp_hw"] - wide["dAmp_nhw"]
    wide["Rx"] = wide["dTx_hw"] - wide["dTx_nhw"]
    wide["Rn"] = wide["dTn_hw"] - wide["dTn_nhw"]
    wide["period_specific_state_definition"] = PERIOD_SPECIFIC_DTX_STATE_DEFINITION
    wide["canonical_group_definition"] = CANONICAL_GROUP_DEFINITION

    out_data_dir = os.path.join(output_dir, "integrated_fig_data")
    ensure_dir(out_data_dir)

    pair_table = os.path.join(out_data_dir, "SI_annual_nhw_hw_pair_transition_table.csv")
    wide.to_csv(pair_table, index=False)

    base_n = int(len(wide))
    four_classes = [
        "positive-dTx state -> positive-dTx state",
        "positive-dTx state -> negative-dTx state",
        "negative-dTx state -> negative-dTx state",
        "negative-dTx state -> positive-dTx state",
    ]

    summary_rows = []
    transition_specs = [
        ("annual_to_nhw", "transition_annual_to_nhw"),
        ("nhw_to_hw", "transition_nhw_to_hw"),
        ("annual_to_hw", "transition_annual_to_hw"),
    ]

    for transition_name, col in transition_specs:
        for cls in four_classes:
            n = int((wide[col] == cls).sum())
            summary_rows.append({
                "transition_type": transition_name,
                "transition_class": cls,
                "n": n,
                "base_n_all_matched_pairs": base_n,
                "fraction_of_base_n": n / base_n if base_n else np.nan,
                "shown_in_four_class_panel": True,
            })
        n_unclassified = int((~wide[col].isin(four_classes)).sum())
        summary_rows.append({
            "transition_type": transition_name,
            "transition_class": "zero_or_unclassified_not_plotted_as_fifth_class",
            "n": n_unclassified,
            "base_n_all_matched_pairs": base_n,
            "fraction_of_base_n": n_unclassified / base_n if base_n else np.nan,
            "shown_in_four_class_panel": False,
        })

    summary = pd.DataFrame(summary_rows)
    summary_table = os.path.join(out_data_dir, "SI_annual_nhw_hw_transition_summary.csv")
    summary.to_csv(summary_table, index=False)

    # -----------------------------
    # SI figure: keep Nature-style clean format, red/blue palette.
    # -----------------------------
    out_plot_dir = os.path.join(output_dir, "plots")
    ensure_dir(out_plot_dir)

    period_order = ["annual", "nhw", "hw"]
    period_labels = ["Annual", "NHW", "HW"]
    state_colors = {
        "positive-dTx state": COLOR_UHI,
        "negative-dTx state": COLOR_UCI,
    }

    comp_rows = []
    for suffix, label in zip(period_order, period_labels):
        state_col = f"state_{suffix}_by_dTx"
        for state in ["positive-dTx state", "negative-dTx state"]:
            n = int((wide[state_col] == state).sum())
            comp_rows.append({"period": label, "state": state, "n": n, "frac": n / base_n if base_n else np.nan})

    comp = pd.DataFrame(comp_rows)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9.2,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), dpi=600)
        ax_a, ax_b = axes
        plt.subplots_adjust(left=0.080, right=0.985, bottom=0.24, top=0.82, wspace=0.36)

        # Panel a: state composition across annual/NHW/HW using same base n.
        x = np.arange(len(period_labels))
        bottom = np.zeros(len(period_labels), dtype=float)
        for state in ["positive-dTx state", "negative-dTx state"]:
            vals = []
            counts = []
            for label in period_labels:
                row = comp[(comp["period"] == label) & (comp["state"] == state)]
                n = int(row["n"].iloc[0]) if len(row) else 0
                counts.append(n)
                vals.append(100.0 * n / base_n if base_n else 0.0)
            ax_a.bar(x, vals, bottom=bottom, color=state_colors[state],
                     edgecolor="white", linewidth=0.7, label=state, zorder=3)
            for xi, v, btm, n in zip(x, vals, bottom, counts):
                if v >= 5:
                    ax_a.text(xi, btm + v / 2, f"{n}", ha="center", va="center",
                              fontsize=6.7, color="white", zorder=4)
            bottom += np.asarray(vals)

        ax_a.set_xticks(x)
        ax_a.set_xticklabels(period_labels)
        ax_a.set_ylim(0, 100)
        ax_a.set_ylabel("Share of matched pairs (%)")
        ax_a.grid(True, axis="y", lw=0.30, alpha=0.18, zorder=0)
        ax_a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        _style = globals().get("add_black_frame", None)
        if callable(_style):
            _style(ax_a, lw=0.9)
        else:
            for side in ["top", "right", "bottom", "left"]:
                ax_a.spines[side].set_visible(True)
                ax_a.spines[side].set_linewidth(0.9)

        # Panel b: four sign-defined transitions only; denominator remains base_n.
        trans_display = [
            ("annual_to_nhw", "Annual→NHW"),
            ("nhw_to_hw", "NHW→HW"),
            ("annual_to_hw", "Annual→HW"),
        ]
        class_labels_short = {
            "positive-dTx state -> positive-dTx state": "+ΔTx→+ΔTx",
            "positive-dTx state -> negative-dTx state": "+ΔTx→−ΔTx",
            "negative-dTx state -> negative-dTx state": "−ΔTx→−ΔTx",
            "negative-dTx state -> positive-dTx state": "−ΔTx→+ΔTx",
        }
        class_colors = {
            "positive-dTx state -> positive-dTx state": "#b3000d",
            "positive-dTx state -> negative-dTx state": "#4292c6",
            "negative-dTx state -> negative-dTx state": "#08519c",
            "negative-dTx state -> positive-dTx state": "#ef3b2c",
        }

        y = np.arange(len(trans_display))[::-1]
        legend_handles = []
        legend_labels = []
        for yy, (transition_name, label) in zip(y, trans_display):
            left = 0.0
            for cls in four_classes:
                row = summary[(summary["transition_type"] == transition_name) & (summary["transition_class"] == cls)]
                n = int(row["n"].iloc[0]) if len(row) else 0
                width = 100.0 * n / base_n if base_n else 0.0
                color = class_colors[cls]
                ax_b.barh(yy, width, left=left, height=0.52, color=color,
                          edgecolor="white", linewidth=0.7, zorder=3)
                if n > 0 and width >= 5.0:
                    ax_b.text(left + width / 2.0, yy, f"{n}", ha="center", va="center",
                              fontsize=6.2, color="white", zorder=4)
                if class_labels_short[cls] not in legend_labels:
                    legend_handles.append(Line2D([0], [0], color=color, lw=6))
                    legend_labels.append(class_labels_short[cls])
                left += width

        ax_b.set_yticks(y)
        ax_b.set_yticklabels([label for _, label in trans_display])
        ax_b.set_xlim(0, 100)
        ax_b.set_xticks([0, 25, 50, 75, 100])
        ax_b.set_xlabel("Share of all matched pairs (%)")
        ax_b.grid(True, axis="x", lw=0.30, alpha=0.18, zorder=0)
        ax_b.legend(legend_handles, legend_labels, frameon=False,
                    loc="upper center", bbox_to_anchor=(0.50, -0.16),
                    ncol=2, handlelength=1.4, handletextpad=0.35, columnspacing=0.9)
        if callable(_style):
            _style(ax_b, lw=0.9)
        else:
            for side in ["top", "right", "bottom", "left"]:
                ax_b.spines[side].set_visible(True)
                ax_b.spines[side].set_linewidth(0.9)

        # Panel labels and concise denominator note.
        for ax, lab, title in [
            (ax_a, "a", "Annual–NHW–HW state composition"),
            (ax_b, "b", "Four sign-defined state transitions"),
        ]:
            bbox = ax.get_position()
            fig.text(bbox.x0 - 0.032, bbox.y1 + 0.028, lab,
                     ha="left", va="bottom", fontsize=13, fontweight="bold")
            fig.text(bbox.x0, bbox.y1 + 0.028, title,
                     ha="left", va="bottom", fontsize=10.0, fontweight="bold")

        unclassified_total = int(summary.loc[~summary["shown_in_four_class_panel"], "n"].sum())
        fig.text(0.080, 0.045,
                 f"Same matched HW–NHW base cohort used throughout: n={base_n}. "
                 "Bars show period-specific dTx-sign states, not the canonical annual UHI/UCI group. "
                 "Zero/missing sign cases are reported in CSV, not plotted as a fifth class.",
                 ha="left", va="bottom", fontsize=6.8, color="#555555")

        f_png = os.path.join(out_plot_dir, "Figure_SI_Annual_NHW_HW_State_Transition.png")
        f_pdf = os.path.join(out_plot_dir, "Figure_SI_Annual_NHW_HW_State_Transition.pdf")
        fig.savefig(f_png, dpi=600, bbox_inches="tight")
        fig.savefig(f_pdf, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved SI annual-NHW-HW transition figure: {f_png}")
    print(f"  Saved SI annual-NHW-HW pair table: {pair_table}")
    print(f"  Saved SI annual-NHW-HW summary table: {summary_table}")

def get_matched_nhw_hw_tables(all_df):
    """
    Return matched non-heatwave and heatwave tables.

    Matching rule:
    - Keep only pairs/stations that have BOTH non_heatwave and heatwave rows.
    - Do not modify other periods or other figure inputs.
    """

    if all_df is None or len(all_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), set()

    if "period" not in all_df.columns:
        raise ValueError("Column 'period' is required to build matched NHW/HW tables.")

    # Use the same ID priority as other HW-NHW paired analyses.
    id_col = None
    for c in ["pair_id", "station_id", "city_id"]:
        if c in all_df.columns:
            id_col = c
            break

    if id_col is None:
        raise ValueError("Cannot find an ID column among pair_id, station_id, or city_id.")

    nhw_ids = set(
        all_df.loc[all_df["period"] == "non_heatwave", id_col]
        .dropna()
        .astype(str)
        .unique()
    )
    hw_ids = set(
        all_df.loc[all_df["period"] == "heatwave", id_col]
        .dropna()
        .astype(str)
        .unique()
    )

    keep_ids = nhw_ids & hw_ids

    matched = all_df[all_df[id_col].astype(str).isin(keep_ids)].copy()

    nhw_matched = matched[matched["period"] == "non_heatwave"].copy()
    hw_matched = matched[matched["period"] == "heatwave"].copy()

    print(
        f"  [Table matched NHW/HW] id_col={id_col}; "
        f"matched IDs={len(keep_ids)}; "
        f"NHW rows={len(nhw_matched)}; HW rows={len(hw_matched)}"
    )

    return nhw_matched, hw_matched, keep_ids
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  plot_results.py  — UHI/UCI Multi-Year Result Visualization")
    print("=" * 72)

    metrics_path = os.path.join(INPUT_DIR, "all_pair_period_metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"ERROR: {metrics_path} not found.")
        print("Please run analysis_multiyear.py first.")
        return

    print(f"\nLoading: {metrics_path}")
    raw_df = pd.read_csv(metrics_path)
    all_df = raw_df.copy()

    # ─────────────────────────────────────────────────────────────
    # ★ 插入此处：运行诊断模块
    # ─────────────────────────────────────────────────────────────
    ensure_dir(OUTPUT_DIR) # 确保输出目录存在
    run_data_diagnostics(all_df, OUTPUT_DIR)

    missing_ids = find_missing_period_pairs(all_df, OUTPUT_DIR)

    # ─────────────────────────────────────────────────────────────
    # ★ 最小修改：强制要求 HW 和 NHW 必须成对存在
    # ─────────────────────────────────────────────────────────────
    # Step 1: restrict to the percentile method, then apply strict annual groups.
    if "hw_method" in all_df.columns:
        all_df = all_df[
            all_df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()
        print(f"  Filtered to hw_method=percentile: {len(all_df)} rows")

    canonical_all_df = apply_canonical_annual_group(all_df)
    annual_df = canonical_all_df[
        canonical_all_df["period"].astype(str).str.lower().eq("annual")
    ].copy()

    # Step 2: explicit matched cohort for all HW/NHW response/mechanism figures.
    matched_all_df, matched_response_df, matched_audit_df = build_matched_hw_nhw_cohort(
        canonical_all_df, OUTPUT_DIR
    )
    if matched_all_df.empty:
        raise ValueError("The strict matched HW/NHW cohort is empty.")
    all_df = matched_all_df.copy()

    print(f"  Canonical all-period rows: {len(canonical_all_df)}")
    print(f"  Strict matched plotting rows: {len(all_df)}")

    export_thermal_hysteresis_data(all_df, OUTPUT_DIR)

    # annual_df was created before the HW/NHW matched-cohort restriction.
    n_uhi = (annual_df["group"]=="UHI").sum()
    n_uci = (annual_df["group"]=="UCI").sum()
    print(f"  Annual pairs: UHI={n_uhi}, UCI={n_uci}")

    ensure_dir(os.path.join(OUTPUT_DIR, "plots"))

    print("\n--- Figure 1: Scatter ΔAmp vs ΔT_mn ---")
    plot_figure1_scatter(annual_df, OUTPUT_DIR)

    print("\n--- Figure 2: UHI/UCI diurnal curves ---")
    plot_figure2_uhi_uci(annual_df, OUTPUT_DIR)

    print("\n--- Figure 2b: [NEW] Annual vs Heatwave UHI/UCI diurnal comparison ---")
    plot_figure2b_uhi_uci_annual_vs_heatwave(all_df, OUTPUT_DIR)

    print("\n--- Figure 3: LCZ comparison ---")
    plot_figure3_lcz(annual_df, OUTPUT_DIR)

    # print("\n--- Table 1: Statistics table ---")
    # compute_stats_table(annual_df, OUTPUT_DIR)

    print("\n--- [新增] Figure: Mechanism Composite ---")
    plot_figure_mechanism_composite(all_df, OUTPUT_DIR) 

    print("\n--- Figure: Heatwave shift arrows ---")
    plot_figure_hw_shift_arrows(all_df, OUTPUT_DIR)

    print("\n--- [New SI] Annual-NHW-HW state transition audit ---")
    plot_si_annual_nhw_hw_state_transition(all_df, OUTPUT_DIR)

    print("\n--- [New] Figure: Thermodynamic Regime Map ---")
    plot_figure_thermodynamic_regime_map(all_df, OUTPUT_DIR)

    # # ─────────────────────────────────────────────
    # print("\n--- Table 2: Heatwave Statistics table ---")
    # hw_df = all_df[all_df["period"] == "heatwave"].copy()

    # if len(hw_df) > 0:
    #     compute_stats_table(
    #         hw_df,
    #         OUTPUT_DIR,
    #         suffix="Table2",
    #         title="heatwave"
    #     )
    # else:
    #     print("  [Table2] No heatwave data, skipping.")
    # ─────────────────────────────────────────────
    # Matched NHW/HW tables only.
    # Other figures still use annual_df as before.
    # ─────────────────────────────────────────────
    nhw_matched_df, hw_matched_df, matched_nhw_hw_ids = get_matched_nhw_hw_tables(all_df)

    print("\n--- Table 1: Non-heatwave Statistics table, matched with heatwave ---")
    if len(nhw_matched_df) > 0:
        compute_stats_table(
            nhw_matched_df,
            OUTPUT_DIR,
            suffix="Table1",
            title="non_heatwave"
        )
    else:
        print("  [Table1] No matched non_heatwave data, skipping.")

    print("\n--- Table 2: Heatwave Statistics table, matched with non-heatwave ---")
    if len(hw_matched_df) > 0:
        compute_stats_table(
            hw_matched_df,
            OUTPUT_DIR,
            suffix="Table2",
            title="heatwave"
        )
    else:
        print("  [Table2] No matched heatwave data, skipping.")

    print("\n--- Figure 4: Mechanism closure ---")
    plot_figure4_mechanism(annual_df, OUTPUT_DIR)

    print("\n--- Figure 4b: [NEW] Phase diagrams (2D + 3D) ---")
    plot_phase_diagrams(annual_df, OUTPUT_DIR)

    print("\n--- Figure 5: Heatwave pressure test ---")
    plot_figure5_hw_pressure(all_df, OUTPUT_DIR)

    print("\n--- Figure 6: Global map + inset diurnal cycles ---")
    plot_figure_map_insets(annual_df, OUTPUT_DIR)

    print("\n--- Figure 7: [NEW] Heatwave station distribution map ---")
    plot_figure7_heatwave_map(all_df, OUTPUT_DIR)

    print("\n--- Figure 8: [NEW] Heatwave vs Non-heatwave diurnal cycle ---")
    plot_figure8_hw_diurnal(all_df, OUTPUT_DIR)

    print("\n--- Combined Figure A ---")
    plot_combined_figure_main(all_df, OUTPUT_DIR)

    print("\n--- Combined Figure B ---")
    plot_combined_figure_dynamics(all_df, OUTPUT_DIR)
    plot_combined_figure_dynamics_consistent(all_df, OUTPUT_DIR)

    plot_fig3paneld_hysteresis_variants(
        all_df,
        OUTPUT_DIR,
        max_raw_per_group_period=40
    )

    plot_fig3d_supplement(all_df, OUTPUT_DIR)

    plot_supplement_single_period_Tx_Tn_relationship(all_df, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/plots/")
    print("Done.\n")

    plot_supplement_ncc_amp_controlled_linkage(all_df, OUTPUT_DIR)

    print("\n--- [NEW] Supplement: Amp vs Hysteresis ---")
    plot_supplement_amp_vs_hysteresis(all_df, OUTPUT_DIR)

    if RUN_LEGACY_EXPOSURE_PROXY_FIGURE:
        print("\n--- Legacy exploratory dry-bulb exposure proxies ---")
        plot_figure4_human_impacts(all_df, OUTPUT_DIR)

    print("\n--- [New] Figure: FFT Schematic Mechanism ---")
    plot_figure_fft_four_lines_clean(all_df, OUTPUT_DIR)

    print("\n--- [New] Figure: Combined Mechanism 2x2 ---")
    # plot_figure_combined_mechanism_2x2(all_df, OUTPUT_DIR)

    plot_figure_combined_mechanism_2x2_fixed(all_df, OUTPUT_DIR)

    print("\n--- [New SI] Pair-level HW-NHW distribution ---")
    plot_supplement_pair_level_hw_nhw_distribution(all_df, OUTPUT_DIR)

    plot_reference_style_4panel_maps_v2(all_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
