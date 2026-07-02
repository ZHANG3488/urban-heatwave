#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labour_loss_final.py  (v3 — +KG climate zone integration)
==========================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import UNIFIED_ROOT

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. Paths & global parameters
# ─────────────────────────────────────────────────────────────
# [v3→v5] 路径更新：v3 → v5
INPUT_CSV = (
    UNIFIED_ROOT + "/analysis/main_multiyear/"
    "robustness_percentile/all_pair_period_metrics.csv"
)
OUTPUT_DIR = UNIFIED_ROOT + "/analysis/labour"

# Time windows
DAY_HOURS   = list(range(8, 20))                         # 08:00–19:59, 12 h
NIGHT_HOURS = list(range(20, 24)) + list(range(0, 8))   # 20:00–07:59, 12 h
WORK_HOURS  = list(range(8, 20))                         # 08:00–19:59, 12 h (now same as DAY_HOURS)

# Must match 01_main_pair_period_metrics.py.
DEWPOINT_MIN_VALID_FRAC = 0.80

# ─────────────────────────────────────────────────────────────
# Figure 4 data export path
# ─────────────────────────────────────────────────────────────
FIG4_DATA_DIR = UNIFIED_ROOT + "/shared/fig4_data"

# Figure 4 exposure settings
# Main Fig. 4 uses exposure-based variables rather than raw temperature means.
FIG4_DAY_EXPOSURE_HOURS = list(range(8, 20))    # 08:00–19:59, for daytime heat exposure
FIG4_WORK_EXPOSURE_HOURS = WORK_HOURS           # 08:00–19:00, for labour-relevant exposure

FIG4_THETA_DAY_TEMP = 30.0      # degC, dry-bulb daytime heat exposure threshold
FIG4_THETA_WBGT = 25.0          # degC WBGT, Dunne labour-loss onset threshold

FIG4_PRIMARY_LABOUR_METHOD = "dunne"


BOOT_N      = 1000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# He et al. (2022) ERF parameters (w1, w2)
HE_PARAMS = {
    "low":    (24.64, 22.72),
    "medium": (32.98, 17.81),
    "high":   (30.94, 16.64),
}

COLORS = {"UHI": "#d62728", "UCI": "#1f77b4"}

PERIOD_STYLE = {
    "annual":       "-",
    "warm_season":  "-.",
    "heatwave":     "--",
    "non_heatwave": ":",
}
PERIOD_LABEL = {
    "annual":       "Annual",
    "warm_season":  "Warm season (NH JJA / SH DJF)",
    "heatwave":     "Heatwave",
    "non_heatwave": "Non-heatwave",
}
PERIOD_COLOR = {
    "annual":       "#2ca02c",
    "warm_season":  "#ff7f0e",
    "heatwave":     "#d62728",
    "non_heatwave": "#9467bd",
}

METHOD_DESC = {
    "dunne":  "Dunne (2013) Heavy Labour",
    "high":   "He (2022) High Intensity (Construction)",
    "medium": "He (2022) Medium Intensity (Manufacturing/Transport)",
    "low":    "He (2022) Low Intensity (Service)",
}

# [v3→v5] KG 主分组映射（与 analysis_multiyear.py 保持一致）
KG_GROUP_MAP = {
    "A": "Tropical",
    "B": "Arid",
    "C": "Temperate",
    "D": "Cold",
    "E": "Polar",
}
KG_GROUPS_ALL = ["A", "B", "C", "D", "E"]


# ─────────────────────────────────────────────────────────────
# 1. WBGT calculation
# ─────────────────────────────────────────────────────────────
def rh_from_t_td(T_C, Td_C):
    es = lambda t: 6.112 * np.exp(17.67 * t / (t + 243.5))
    return np.clip(100.0 * es(Td_C) / es(T_C), 1.0, 100.0)


def wbt_stull(T_C, RH_pct):
    T, RH = np.asarray(T_C, float), np.asarray(RH_pct, float)
    wbt = (
        T * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )
    return np.minimum(wbt, T)

def shade_wbgt(T_C, Td_C):
    """
    Calculate shaded WBGT only where both air temperature and
    dew-point temperature are available.

    No fixed-relative-humidity fallback is applied.
    """
    T = np.asarray(T_C, dtype=float)
    Td = np.asarray(Td_C, dtype=float)

    wbgt = np.full_like(T, np.nan, dtype=float)

    valid = np.isfinite(T) & np.isfinite(Td)

    if valid.any():
        # Enforce the physical constraint Td <= Ta.
        Td_use = np.minimum(Td[valid], T[valid])

        rh = rh_from_t_td(
            T[valid],
            Td_use,
        )

        wbgt[valid] = (
            0.7 * wbt_stull(T[valid], rh)
            + 0.3 * T[valid]
        )

    return wbgt
# ─────────────────────────────────────────────────────────────
# 2. Labour capacity functions
# ─────────────────────────────────────────────────────────────
def lc_dunne(wbgt):
    wbgt = np.asarray(wbgt, float)
    return np.clip(100.0 - 25.0 * np.maximum(0.0, wbgt - 25.0) ** (2.0 / 3.0),
                   0.0, 100.0)


def workability_he(wbgt, intensity="high"):
    w1, w2 = HE_PARAMS[intensity]
    wbgt   = np.asarray(wbgt, float)
    return np.clip(0.1 + 0.9 / (1.0 + (wbgt / w1) ** w2), 0.1, 1.0) * 100.0


# ─────────────────────────────────────────────────────────────
# 3. Extract curves from a row
# ─────────────────────────────────────────────────────────────
def _get_curve(row, prefix, suffix_tpl):
    return np.array(
        [row.get(f"{prefix}{suffix_tpl.format(h=h)}", np.nan) for h in range(24)],
        dtype=float
    )

def extract_T(row, role):
    """Temperature FFT built from the same timestamps as dew point."""
    return _get_curve(
        row,
        f"{role}_wbgt_temp_fft",
        "_h{h:02d}",
    )

def extract_Td(row, role):
    """Two-harmonic dew-point FFT from the same timestamps as T."""
    return _get_curve(
        row,
        f"{role}_dew_fft",
        "_h{h:02d}",
    )

def _hour_mean(curve, hours):
    vals = curve[np.array(hours)]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

# ─────────────────────────────────────────────────────────────
# 4. Core per-row calculation
# ─────────────────────────────────────────────────────────────
def compute_row_lc(row):
    T_u = extract_T(row, "urban")
    T_r = extract_T(row, "rural")
    Td_u = extract_Td(row, "urban")
    Td_r = extract_Td(row, "rural")

    n_valid_temp_u = int(np.isfinite(T_u).sum())
    n_valid_temp_r = int(np.isfinite(T_r).sum())
    n_valid_dew_u = int(np.isfinite(Td_u).sum())
    n_valid_dew_r = int(np.isfinite(Td_r).sum())

    common_frac_u = pd.to_numeric(
        row.get("temp_dew_common_time_frac_urban", np.nan),
        errors="coerce",
    )
    common_frac_r = pd.to_numeric(
        row.get("temp_dew_common_time_frac_rural", np.nan),
        errors="coerce",
    )
    common_fft_u = pd.to_numeric(
        row.get("temp_dew_common_fft_available_urban", 0),
        errors="coerce",
    )
    common_fft_r = pd.to_numeric(
        row.get("temp_dew_common_fft_available_rural", 0),
        errors="coerce",
    )

    urban_wbgt_input_ok = (
        np.isfinite(common_frac_u)
        and common_frac_u >= DEWPOINT_MIN_VALID_FRAC
        and common_fft_u == 1
        and n_valid_temp_u == 24
        and n_valid_dew_u == 24
    )

    rural_wbgt_input_ok = (
        np.isfinite(common_frac_r)
        and common_frac_r >= DEWPOINT_MIN_VALID_FRAC
        and common_fft_r == 1
        and n_valid_temp_r == 24
        and n_valid_dew_r == 24
    )

    has_dew_final = urban_wbgt_input_ok and rural_wbgt_input_ok

    wbgt_u = (
        shade_wbgt(T_u, Td_u)
        if urban_wbgt_input_ok
        else np.full(24, np.nan, dtype=float)
    )
    wbgt_r = (
        shade_wbgt(T_r, Td_r)
        if rural_wbgt_input_ok
        else np.full(24, np.nan, dtype=float)
    )

    out = {
        # Pair-level compatibility fields.
        "has_dewpoint": int(has_dew_final),
        "has_dewpoint_final": int(has_dew_final),

        # Station-specific common-time WBGT input status.
        "urban_wbgt_input_ok": int(urban_wbgt_input_ok),
        "rural_wbgt_input_ok": int(rural_wbgt_input_ok),
        "n_valid_temp_urban_final": n_valid_temp_u,
        "n_valid_temp_rural_final": n_valid_temp_r,
        "n_valid_dew_urban_final": n_valid_dew_u,
        "n_valid_dew_rural_final": n_valid_dew_r,
        "temp_dew_common_time_frac_urban_final": (
            float(common_frac_u) if np.isfinite(common_frac_u) else np.nan
        ),
        "temp_dew_common_time_frac_rural_final": (
            float(common_frac_r) if np.isfinite(common_frac_r) else np.nan
        ),
        "temp_dew_common_fft_used": int(has_dew_final),
        "temp_dew_min_valid_frac_required": DEWPOINT_MIN_VALID_FRAC,
    }

    for h in range(24):
        out[f"wbgt_urban_h{h:02d}"] = float(wbgt_u[h]) if np.isfinite(wbgt_u[h]) else np.nan
        out[f"wbgt_rural_h{h:02d}"] = float(wbgt_r[h]) if np.isfinite(wbgt_r[h]) else np.nan

    out["wbgt_urban_day_mean"]   = _hour_mean(wbgt_u, DAY_HOURS)
    out["wbgt_rural_day_mean"]   = _hour_mean(wbgt_r, DAY_HOURS)
    out["wbgt_urban_night_mean"] = _hour_mean(wbgt_u, NIGHT_HOURS)
    out["wbgt_rural_night_mean"] = _hour_mean(wbgt_r, NIGHT_HOURS)
    out["dwbgt_day_mean"]   = (
        out["wbgt_urban_day_mean"] - out["wbgt_rural_day_mean"]
        if pd.notna(out["wbgt_urban_day_mean"]) and pd.notna(out["wbgt_rural_day_mean"])
        else np.nan
    )
    out["dwbgt_night_mean"] = (
        out["wbgt_urban_night_mean"] - out["wbgt_rural_night_mean"]
        if pd.notna(out["wbgt_urban_night_mean"]) and pd.notna(out["wbgt_rural_night_mean"])
        else np.nan
    )

    methods = {
        "dunne":  lc_dunne(wbgt_u)               - lc_dunne(wbgt_r),
        "high":   workability_he(wbgt_u, "high")  - workability_he(wbgt_r, "high"),
        "medium": workability_he(wbgt_u, "medium")- workability_he(wbgt_r, "medium"),
        "low":    workability_he(wbgt_u, "low")   - workability_he(wbgt_r, "low"),
    }

    for method, dlc in methods.items():
        for h in range(24):
            out[f"dlc_{method}_h{h:02d}"] = float(dlc[h]) if np.isfinite(dlc[h]) else np.nan

        d_mean = _hour_mean(dlc, DAY_HOURS)
        n_mean = _hour_mean(dlc, NIGHT_HOURS)
        w_mean = _hour_mean(dlc, WORK_HOURS)
        a_mean = float(np.nanmean(dlc))

        out[f"dlc_{method}_day_mean"]   = d_mean
        out[f"dlc_{method}_night_mean"] = n_mean
        out[f"dlc_{method}_work_mean"]  = w_mean
        out[f"dlc_{method}_24h_mean"]   = a_mean

        day_vals   = dlc[np.array(DAY_HOURS)]
        night_vals = dlc[np.array(NIGHT_HOURS)]
        work_vals  = dlc[np.array(WORK_HOURS)]

        # 旧定义保留：capacity difference 的绝对峰值
        out[f"dlc_{method}_day_peak"] = (
            float(np.nanmax(np.abs(day_vals)))
            if np.isfinite(day_vals).any() else np.nan
        )
        out[f"dlc_{method}_work_peak"] = (
            float(np.nanmax(np.abs(work_vals)))
            if np.isfinite(work_vals).any() else np.nan
        )

        # ── NEW: labour loss difference = -(capacity difference) ──
        # dloss > 0  表示 urban loss 比 rural 更大
        # dloss < 0  表示 urban loss 比 rural 更小
        dloss = -dlc
        day_loss_vals   = dloss[np.array(DAY_HOURS)]
        night_loss_vals = dloss[np.array(NIGHT_HOURS)]
        work_loss_vals  = dloss[np.array(WORK_HOURS)]

        out[f"dloss_{method}_day_mean"] = (
            float(np.nanmean(day_loss_vals))
            if np.isfinite(day_loss_vals).any() else np.nan
        )
        out[f"dloss_{method}_night_mean"] = (
            float(np.nanmean(night_loss_vals))
            if np.isfinite(night_loss_vals).any() else np.nan
        )
        out[f"dloss_{method}_work_mean"] = (
            float(np.nanmean(work_loss_vals))
            if np.isfinite(work_loss_vals).any() else np.nan
        )
        out[f"dloss_{method}_24h_mean"] = (
            float(np.nanmean(dloss))
            if np.isfinite(dloss).any() else np.nan
        )

        # integrated figure 推荐使用这个：
        # 白天时段内“城市相对乡村的最大额外 labour loss”
        out[f"dloss_{method}_day_peak"] = (
            float(np.nanmax(day_loss_vals))
            if np.isfinite(day_loss_vals).any() else np.nan
        )
        out[f"dloss_{method}_work_peak"] = (
            float(np.nanmax(work_loss_vals))
            if np.isfinite(work_loss_vals).any() else np.nan
        )

        # 如果以后你还想看“最大差值幅度（不分方向）”，也一起留着
        out[f"dloss_{method}_day_peak_abs"] = (
            float(np.nanmax(np.abs(day_loss_vals)))
            if np.isfinite(day_loss_vals).any() else np.nan
        )
        out[f"dloss_{method}_work_peak_abs"] = (
            float(np.nanmax(np.abs(work_loss_vals)))
            if np.isfinite(work_loss_vals).any() else np.nan
        )

        # Use work-hour mean for the legacy net_effect field.
        # Do not add string metadata columns here, because downstream summary
        # treats net_effect_* as numeric.
        net = w_mean if pd.notna(w_mean) else np.nan
        out[f"net_effect_{method}"] = net

        if pd.notna(d_mean) and pd.notna(n_mean):
            if   d_mean > 0  and n_mean >= 0: cls = "both_benefit"
            elif d_mean > 0  and n_mean <  0: cls = "day_save_night_cost"
            elif d_mean <= 0 and n_mean <  0: cls = "both_cost"
            else:                              cls = "day_cost_night_save"
        else:
            cls = "unknown"
        out[f"asymmetry_class_{method}"] = cls

        out[f"asymmetry_index_{method}"] = (
            (d_mean - abs(n_mean))
            if pd.notna(d_mean) and pd.notna(n_mean) else np.nan
        )
        out[f"day_night_ratio_{method}"] = (
            d_mean / abs(n_mean)
            if pd.notna(d_mean) and pd.notna(n_mean) and abs(n_mean) > 1e-6 else np.nan
        )

        # ------------------------------------------------------------
        # Station-specific Tx-hour labour loss from shaded WBGT
        # ------------------------------------------------------------
        work_idx = np.asarray(WORK_HOURS, dtype=int)

        valid_u_t = np.isfinite(T_u[work_idx])
        valid_r_t = np.isfinite(T_r[work_idx])

        if valid_u_t.any() and valid_r_t.any():
            u_tx_hour = int(
                work_idx[
                    np.nanargmax(T_u[work_idx])
                ]
            )

            r_tx_hour = int(
                work_idx[
                    np.nanargmax(T_r[work_idx])
                ]
            )

            u_wbgt_tx = wbgt_u[u_tx_hour]
            r_wbgt_tx = wbgt_r[r_tx_hour]

            out[f"tx_hour_urban_{method}"] = u_tx_hour
            out[f"tx_hour_rural_{method}"] = r_tx_hour
            out[f"wbgt_tx_urban_{method}"] = (
                float(u_wbgt_tx)
                if np.isfinite(u_wbgt_tx)
                else np.nan
            )
            out[f"wbgt_tx_rural_{method}"] = (
                float(r_wbgt_tx)
                if np.isfinite(r_wbgt_tx)
                else np.nan
            )

            if np.isfinite(u_wbgt_tx) and np.isfinite(r_wbgt_tx):
                if method == "dunne":
                    u_peak_loss = (
                        100.0
                        - float(lc_dunne(u_wbgt_tx))
                    )
                    r_peak_loss = (
                        100.0
                        - float(lc_dunne(r_wbgt_tx))
                    )

                else:
                    u_peak_loss = (
                        100.0
                        - float(
                            workability_he(
                                u_wbgt_tx,
                                intensity=method,
                            )
                        )
                    )
                    r_peak_loss = (
                        100.0
                        - float(
                            workability_he(
                                r_wbgt_tx,
                                intensity=method,
                            )
                        )
                    )

                out[
                    f"dloss_{method}_peak_t_diff"
                ] = float(
                    u_peak_loss - r_peak_loss
                )

            else:
                out[
                    f"dloss_{method}_peak_t_diff"
                ] = np.nan

        else:
            out[
                f"dloss_{method}_peak_t_diff"
            ] = np.nan
    return out
# ─────────────────────────────────────────────────────────────
# 5. Statistics
# ─────────────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=BOOT_N, alpha=0.05):
    arr = pd.Series(values).dropna().values.astype(float)
    if len(arr) == 0: return np.nan, np.nan, np.nan
    if len(arr) == 1: return arr[0], arr[0], arr[0]
    boots = [
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ]
    return float(np.mean(arr)), float(np.quantile(boots, alpha/2)), float(np.quantile(boots, 1-alpha/2))


def sig_stars(p):
    if pd.isna(p):  return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"


def build_summary(df, groupby, metrics):
    rows = []
    for keys, g in df.groupby(groupby, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(groupby, keys)); row["n"] = len(g)
        for m in metrics:
            if m not in g.columns: continue
            mn, lo, hi = bootstrap_ci(g[m])
            row[f"{m}_mean"]  = mn
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
            arr = g[m].dropna().values.astype(float)
            if len(arr) >= 5:
                try:
                    _, pval = stats.wilcoxon(arr)
                    row[f"{m}_p"]   = float(pval)
                    row[f"{m}_sig"] = sig_stars(pval)
                except Exception:
                    row[f"{m}_p"] = np.nan; row[f"{m}_sig"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 6. Plot utilities
# ─────────────────────────────────────────────────────────────
def _get_diurnal_matrix(df, col_prefix):
    cols  = [f"{col_prefix}_h{h:02d}" for h in range(24)]
    avail = [c for c in cols if c in df.columns]
    if not avail: return None
    return df[avail].values.astype(float)


def _plot_mean_ci(ax, matrix, color, ls="-", label="", n_boot=500):
    mean  = np.nanmean(matrix, axis=0)
    ci_lo = np.zeros(24); ci_hi = np.zeros(24)
    for h in range(24):
        _, lo, hi = bootstrap_ci(matrix[:, h], n_boot=n_boot)
        ci_lo[h] = lo; ci_hi[h] = hi
    x = np.arange(24)
    ax.plot(x, mean, color=color, ls=ls, lw=2.0, label=label)
    ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.12)
    return mean


def _add_time_shading(ax):
    ax.axvspan(min(DAY_HOURS), max(DAY_HOURS), alpha=0.08, color="gold",  zorder=0)
    ax.axvspan(21, 24,         alpha=0.06, color="navy", zorder=0)
    ax.axvspan(0,  max(h for h in NIGHT_HOURS if h < 7) + 1,
               alpha=0.06, color="navy", zorder=0)


def _set_hour_xticks(ax):
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in range(0, 24, 3)], rotation=30, ha="right"
    )


# ─────────────────────────────────────────────────────────────
# 7. Figure 1: dLC diurnal cycle (four periods)
# ─────────────────────────────────────────────────────────────
def plot_fig1_diurnal_dlc(df, method, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, group in zip(axes, ["UHI", "UCI"]):
        sub = df[df["group"] == group]
        for period in ["annual", "warm_season", "heatwave", "non_heatwave"]:
            sp  = sub[sub["period"] == period]
            mat = _get_diurnal_matrix(sp, f"dlc_{method}")
            if mat is None or len(sp) < 5: continue
            _plot_mean_ci(ax, mat, COLORS[group],
                          ls=PERIOD_STYLE[period],
                          label=f"{PERIOD_LABEL[period]} (n={len(sp)})")
        _add_time_shading(ax)
        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.6)
        ax.set_title(f"{group} Group", fontsize=13, fontweight="bold")
        ax.set_xlabel("Local Time (hour)", fontsize=11)
        ax.set_ylabel(r"$\Delta$LC = Urban $-$ Rural (%)", fontsize=11)
        _set_hour_xticks(ax)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
        ax.text(0.97, 0.95, "Urban advantage\n(less heat stress)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color="green", style="italic")
        ax.text(0.97, 0.05, "Rural advantage\n(urban hotter)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="red", style="italic")
    fig.suptitle(
        f"Urban-Rural Labour Capacity Difference: Diurnal Cycle\n"
        f"[{METHOD_DESC.get(method, method)}]  "
        "Gold=Daytime(09-16h)  Navy=Nighttime(21-06h)",
        fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 8. Figure 2: Day-Night scatter (four quadrants)
# ─────────────────────────────────────────────────────────────
def plot_fig2_scatter(df, method, out_path):
    annual = df[df["period"] == "annual"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, group in zip(axes, ["UHI", "UCI"]):
        sub   = annual[annual["group"] == group]
        x_col = f"dlc_{method}_day_mean"
        y_col = f"dlc_{method}_night_mean"
        valid = sub[[x_col, y_col]].dropna()
        if len(valid) < 3: ax.set_visible(False); continue
        ax.scatter(valid[x_col], valid[y_col],
                   c=COLORS[group], alpha=0.45, s=20, edgecolors="none")
        mx, xlo, xhi = bootstrap_ci(valid[x_col])
        my, ylo, yhi = bootstrap_ci(valid[y_col])
        ax.errorbar(mx, my,
                    xerr=[[mx-xlo], [xhi-mx]], yerr=[[my-ylo], [yhi-my]],
                    fmt="*", color="black", ms=14, capsize=5, zorder=10,
                    label=f"Mean ({mx:+.2f}, {my:+.2f})")
        ax.axvline(0, color="gray", lw=1.0, ls="--")
        ax.axhline(0, color="gray", lw=1.0, ls="--")
        for (tx, ty), (txt, fc) in {
            (0.82, 0.88): ("Day save\nNight save", "lightgreen"),
            (0.82, 0.12): ("Day save\nNight cost", "lightyellow"),
            (0.18, 0.88): ("Day cost\nNight save", "lightblue"),
            (0.18, 0.12): ("Both cost",             "mistyrose"),
        }.items():
            ax.text(tx, ty, txt, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, alpha=0.7, lw=0))
        ax.set_title(f"{group} Group  (Annual, n={len(valid)})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Daytime $\Delta$LC (%)", fontsize=11)
        ax.set_ylabel(r"Nighttime $\Delta$LC (%)", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Urban-Rural Labour Capacity: Daytime vs Nighttime Asymmetry (Annual)\n"
        f"[{METHOD_DESC.get(method, method)}]  "
        "Positive = urban advantage;  Negative = urban disadvantage",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 9. Figure 3: Net effect violin
# ─────────────────────────────────────────────────────────────
def plot_fig3_net_violin(df, method, out_path):
    net_col = f"net_effect_{method}"
    if net_col not in df.columns: return
    data_list, colors_list, labels = [], [], []
    for period in ["annual", "warm_season", "heatwave", "non_heatwave"]:
        for group in ["UHI", "UCI"]:
            sub = df[(df["period"] == period) & (df["group"] == group)][net_col].dropna()
            if len(sub) < 3: continue
            data_list.append(sub.values); colors_list.append(COLORS[group])
            labels.append(f"{group}\n{PERIOD_LABEL[period]}")
    if not data_list: return
    fig, ax = plt.subplots(figsize=(max(8, len(data_list) * 1.4), 5))
    parts = ax.violinplot(data_list, showmeans=True, showmedians=True,
                          showextrema=False, widths=0.7)
    for pc, color in zip(parts["bodies"], colors_list):
        pc.set_facecolor(color); pc.set_alpha(0.60)
    parts["cmeans"].set_color("black"); parts["cmedians"].set_color("white")
    ax.axhline(0, color="black", ls="--", lw=1.2, label="No difference")
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"Net $\Delta$LC (%)", fontsize=11)
    ax.set_title(
        f"Net Labour Capacity Effect Distribution\n"
        f"[{METHOD_DESC.get(method, method)}]\n"
        r"Net = (Day $\Delta$LC $\times$ 8h + Night $\Delta$LC $\times$ 10h) / 24h",
        fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 10. Figure 4: WBGT diurnal cycle
# ─────────────────────────────────────────────────────────────
def plot_fig4_wbgt_diurnal(df, out_path):
    annual = df[df["period"] == "annual"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, group in zip(axes, ["UHI", "UCI"]):
        sub = annual[annual["group"] == group]
        if len(sub) < 3: continue
        for prefix, label, color, ls in [
            ("wbgt_urban", f"Urban WBGT (n={len(sub)})", COLORS[group], "-"),
            ("wbgt_rural", "Rural WBGT",                 "gray",        "--"),
        ]:
            mat = _get_diurnal_matrix(sub, prefix)
            if mat is None: continue
            _plot_mean_ci(ax, mat, color, ls=ls, label=label, n_boot=500)
        for thr, color, lbl in [
            (25.0, "orange",     "25 degC: Heavy labour onset"),
            (27.9, "darkorange", "27.9 degC: Heavy labour 50%"),
            (30.0, "red",        "30 degC: Light+25% heavy mix"),
        ]:
            ax.axhline(thr, color=color, ls=":", lw=1.5, alpha=0.85, label=lbl)
        ax.axvspan(min(DAY_HOURS), max(DAY_HOURS), alpha=0.07, color="gold", zorder=0)
        ax.set_title(f"{group} Group", fontsize=13, fontweight="bold")
        ax.set_xlabel("Local Time (hour)", fontsize=11)
        ax.set_ylabel("WBGT (degC)", fontsize=11)
        _set_hour_xticks(ax); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.suptitle(
        "Urban vs Rural WBGT Diurnal Cycle (Annual Mean +/- 95% CI)\n"
        "Gold = Daytime (09-16h);  Dotted lines = Dunne (2013) labour thresholds",
        fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 11. Figure 5: Method comparison
# ─────────────────────────────────────────────────────────────
def plot_fig5_method_compare(df, out_path):
    annual   = df[df["period"] == "annual"]
    methods  = ["dunne", "high", "medium", "low"]
    m_labels = {
        "dunne":  "Dunne 2013\nHeavy",
        "high":   "He 2022\nHigh",
        "medium": "He 2022\nMedium",
        "low":    "He 2022\nLow",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(methods)); width = 0.55
    for ax, group in zip(axes, ["UHI", "UCI"]):
        sub = annual[annual["group"] == group]
        means, ci_los, ci_his = [], [], []
        for m in methods:
            col = f"net_effect_{m}"
            mn, lo, hi = bootstrap_ci(sub[col]) if col in sub.columns else (np.nan,)*3
            means.append(mn); ci_los.append(lo); ci_his.append(hi)
        means  = np.array(means, float)
        ci_los = np.array(ci_los, float); ci_his = np.array(ci_his, float)
        err_lo = np.where(np.isfinite(means - ci_los), means - ci_los, 0)
        err_hi = np.where(np.isfinite(ci_his - means), ci_his - means, 0)
        bars = ax.bar(x, means, width, color=COLORS[group], alpha=0.72,
                      yerr=[err_lo, err_hi], capsize=6, error_kw={"lw": 1.5})
        for rect, mn in zip(bars, means):
            if np.isfinite(mn):
                ax.text(rect.get_x() + rect.get_width() / 2, mn, f"{mn:+.2f}",
                        ha="center", va="bottom" if mn >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
        ax.set_title(f"{group} Group (n={len(sub)})", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"Net $\Delta$LC (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m_labels[m] for m in methods], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Net Effect Comparison Across Methods (Annual, Urban - Rural)\n"
        "Positive = urban net advantage;  Negative = urban net disadvantage",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 12. Figure 6: Three-period asymmetry comparison
# ─────────────────────────────────────────────────────────────
def plot_fig6_period_asymmetry(df, method, out_path):
    """2 rows (UHI/UCI) x 3 cols (diurnal curve / bar chart / AI violin)."""
    target_periods = ["annual", "warm_season", "heatwave"]
    ai_col  = f"asymmetry_index_{method}"
    day_col = f"dlc_{method}_day_mean"
    ngt_col = f"dlc_{method}_night_mean"
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.subplots_adjust(hspace=0.40, wspace=0.32)

    for row_idx, group in enumerate(["UHI", "UCI"]):
        sub_g = df[df["group"] == group]; color = COLORS[group]

        # Col 0: diurnal curves
        ax0 = axes[row_idx, 0]
        for period in target_periods:
            sp  = sub_g[sub_g["period"] == period]
            mat = _get_diurnal_matrix(sp, f"dlc_{method}")
            if mat is None or len(sp) < 5: continue
            _plot_mean_ci(ax0, mat, PERIOD_COLOR[period],
                          ls=PERIOD_STYLE[period],
                          label=f"{PERIOD_LABEL[period]} (n={len(sp)})", n_boot=500)
        ax0.axhline(0, color="black", lw=1.2, ls="--", alpha=0.6)
        _add_time_shading(ax0)
        ax0.set_title(f"{group} -- dLC Diurnal Curves", fontsize=11, fontweight="bold")
        ax0.set_xlabel("Local Hour", fontsize=10)
        ax0.set_ylabel(r"$\Delta$LC (%)", fontsize=10)
        _set_hour_xticks(ax0); ax0.legend(fontsize=8, loc="upper left"); ax0.grid(alpha=0.3)
        ax0.text(0.97, 0.95, "Urban advantage", transform=ax0.transAxes,
                 ha="right", va="top", fontsize=7, color="green", style="italic")
        ax0.text(0.97, 0.05, "Rural advantage", transform=ax0.transAxes,
                 ha="right", va="bottom", fontsize=7, color="red", style="italic")

        # Col 1: paired bar chart
        ax1 = axes[row_idx, 1]
        x = np.arange(len(target_periods)); width = 0.35
        day_m, day_elo, day_ehi = [], [], []
        ngt_m, ngt_elo, ngt_ehi = [], [], []
        for period in target_periods:
            sp = sub_g[sub_g["period"] == period]
            dm, dlo, dhi = (
                bootstrap_ci(sp[day_col])
                if day_col in sp.columns and len(sp) >= 3 else (np.nan,)*3
            )
            nm, nlo, nhi = (
                bootstrap_ci(sp[ngt_col])
                if ngt_col in sp.columns and len(sp) >= 3 else (np.nan,)*3
            )
            day_m.append(dm if pd.notna(dm) else 0)
            day_elo.append((dm - dlo) if pd.notna(dm) and pd.notna(dlo) else 0)
            day_ehi.append((dhi - dm) if pd.notna(dm) and pd.notna(dhi) else 0)
            ngt_m.append(nm if pd.notna(nm) else 0)
            ngt_elo.append((nm - nlo) if pd.notna(nm) and pd.notna(nlo) else 0)
            ngt_ehi.append((nhi - nm) if pd.notna(nm) and pd.notna(nhi) else 0)
        b1 = ax1.bar(x - width / 2, day_m, width, color=color, alpha=0.80,
                     label="Daytime dLC", yerr=[day_elo, day_ehi], capsize=4,
                     error_kw={"lw": 1.2})
        b2 = ax1.bar(x + width / 2, ngt_m, width, color=color, alpha=0.35,
                     label="Nighttime dLC", yerr=[ngt_elo, ngt_ehi], capsize=4,
                     error_kw={"lw": 1.2}, hatch="//")
        for rect, val in list(zip(b1, day_m)) + list(zip(b2, ngt_m)):
            if abs(val) > 0:
                ax1.text(rect.get_x() + rect.get_width() / 2,
                         val + (0.02 if val >= 0 else -0.02), f"{val:+.2f}",
                         ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5)
        ax1.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax1.set_title(f"{group} -- Day vs Night dLC", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([PERIOD_LABEL[p] for p in target_periods], fontsize=9)
        ax1.set_ylabel(r"$\Delta$LC (%)", fontsize=10)
        ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

        # Col 2: AI violin
        ax2 = axes[row_idx, 2]
        data_list, color_list, xtick_labels = [], [], []
        for period in target_periods:
            sp   = sub_g[sub_g["period"] == period]
            vals = (
                sp[ai_col].dropna().values.astype(float)
                if ai_col in sp.columns else np.array([])
            )
            if len(vals) < 3: continue
            data_list.append(vals); color_list.append(PERIOD_COLOR[period])
            xtick_labels.append(f"{PERIOD_LABEL[period]}\nn={len(vals)}")
        if data_list:
            parts = ax2.violinplot(data_list, showmeans=True, showmedians=True,
                                   showextrema=False, widths=0.65)
            for pc, c in zip(parts["bodies"], color_list):
                pc.set_facecolor(c); pc.set_alpha(0.65)
            parts["cmeans"].set_color("black"); parts["cmedians"].set_color("white")
            for i, (vals, _) in enumerate(zip(data_list, color_list), start=1):
                mn = np.mean(vals)
                ax2.text(i, mn, f"{mn:+.2f}", ha="center",
                         va="bottom" if mn >= 0 else "top",
                         fontsize=8, fontweight="bold", color="black")
        ax2.axhline(0, color="black", ls="--", lw=1.2)
        ax2.set_title(
            f"{group} -- Asymmetry Index (AI)\n"
            r"AI = $\Delta$LC$_\mathrm{day}$ $-$ |$\Delta$LC$_\mathrm{night}$|",
            fontsize=11, fontweight="bold")
        if xtick_labels:
            ax2.set_xticks(range(1, len(xtick_labels) + 1))
            ax2.set_xticklabels(xtick_labels, fontsize=8.5)
        ax2.set_ylabel("AI (%)", fontsize=10); ax2.grid(axis="y", alpha=0.3)
        ax2.text(0.97, 0.97, "AI > 0:\nDay protection\n> Night cost",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=7.5, color="green", style="italic")
        ax2.text(0.97, 0.03, "AI < 0:\nNight cost\n> Day protection",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=7.5, color="red", style="italic")

    fig.suptitle(
        f"Diurnal Asymmetry: Annual vs Summer/JJA vs Heatwave\n"
        f"[{METHOD_DESC.get(method, method)}]  "
        "Gold=Daytime(09-16h)  Navy=Nighttime(21-06h)",
        fontsize=12, y=1.005)
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 13. Figure 7: Period x Hour heatmap
# ─────────────────────────────────────────────────────────────
def plot_fig7_asymmetry_heatmap(df, method, out_path):
    target_periods = ["annual", "warm_season", "heatwave"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    fig.subplots_adjust(wspace=0.25)
    for ax, group in zip(axes, ["UHI", "UCI"]):
        sub_g = df[df["group"] == group]
        matrix = []; ylabels = []
        for period in target_periods:
            sp  = sub_g[sub_g["period"] == period]
            mat = _get_diurnal_matrix(sp, f"dlc_{method}")
            if mat is None or len(sp) < 3: continue
            matrix.append(np.nanmean(mat, axis=0))
            ylabels.append(f"{PERIOD_LABEL[period]} (n={len(sp)})")
        if not matrix: ax.set_visible(False); continue
        matrix_np = np.array(matrix)
        vmax = max(np.nanmax(np.abs(matrix_np)), 0.01)
        im = ax.imshow(matrix_np, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[-0.5, 23.5, len(matrix_np) - 0.5, -0.5],
                       interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, label=r"Mean $\Delta$LC (%)", shrink=0.85)
        cbar.ax.tick_params(labelsize=8)
        for h in DAY_HOURS:
            ax.axvline(h - 0.5, color="gold", lw=0.8, alpha=0.7)
        ax.axvline(max(DAY_HOURS) + 0.5, color="gold", lw=0.8, alpha=0.7)
        ax.axvline(20.5, color="navy", lw=0.8, alpha=0.7)
        ax.axvline(6.5,  color="navy", lw=0.8, alpha=0.7)
        ax.set_title(f"{group} Group -- dLC Heatmap", fontsize=11, fontweight="bold")
        ax.set_xlabel("Local Hour", fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)], fontsize=9)
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=9)
        for i, row_vals in enumerate(matrix_np):
            for j, val in enumerate(row_vals):
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(val) > 0.4 * vmax else "black")
    fig.suptitle(
        f"dLC Heatmap: Period x Hour  [{METHOD_DESC.get(method, method)}]\n"
        "Red = Urban advantage (dLC > 0);  Blue = Rural advantage (dLC < 0)",
        fontsize=11, y=1.02)
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# [v3→v5] Figure 8 (NEW): KG 主分组 × dLC 日间/夜间均值条形图
# ─────────────────────────────────────────────────────────────
def plot_fig8_kg_bar(df, method, out_path):
    """
    [v3→v5 新增]
    按 KG 主分组（A/B/C/D/E）绘制 UHI/UCI 各组的
    Daytime dLC 与 Nighttime dLC 均值+CI 条形图（annual period）。
    """
    if "kg_group" not in df.columns:
        print(f"  [fig8] kg_group column not found, skipped.")
        return

    annual  = df[df["period"] == "annual"].copy()
    day_col = f"dlc_{method}_day_mean"
    ngt_col = f"dlc_{method}_night_mean"
    kg_order = [k for k in KG_GROUPS_ALL
                if k in annual["kg_group"].dropna().unique()]

    if not kg_order:
        print(f"  [fig8] No KG groups found in annual data, skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x     = np.arange(len(kg_order))
    width = 0.35

    for ax, group in zip(axes, ["UHI", "UCI"]):
        sub   = annual[annual["group"] == group]
        color = COLORS[group]
        day_m, day_elo, day_ehi = [], [], []
        ngt_m, ngt_elo, ngt_ehi = [], [], []

        for kg in kg_order:
            sub_kg = sub[sub["kg_group"] == kg]
            dm, dlo, dhi = (
                bootstrap_ci(sub_kg[day_col])
                if day_col in sub_kg.columns and len(sub_kg) >= 3
                else (np.nan,)*3
            )
            nm, nlo, nhi = (
                bootstrap_ci(sub_kg[ngt_col])
                if ngt_col in sub_kg.columns and len(sub_kg) >= 3
                else (np.nan,)*3
            )
            day_m.append(dm if pd.notna(dm) else 0)
            day_elo.append((dm - dlo) if pd.notna(dm) and pd.notna(dlo) else 0)
            day_ehi.append((dhi - dm) if pd.notna(dm) and pd.notna(dhi) else 0)
            ngt_m.append(nm if pd.notna(nm) else 0)
            ngt_elo.append((nm - nlo) if pd.notna(nm) and pd.notna(nlo) else 0)
            ngt_ehi.append((nhi - nm) if pd.notna(nm) and pd.notna(nhi) else 0)

        b1 = ax.bar(x - width / 2, day_m, width, color=color, alpha=0.80,
                    label="Daytime dLC",
                    yerr=[day_elo, day_ehi], capsize=4, error_kw={"lw": 1.2})
        b2 = ax.bar(x + width / 2, ngt_m, width, color=color, alpha=0.35,
                    label="Nighttime dLC",
                    yerr=[ngt_elo, ngt_ehi], capsize=4,
                    error_kw={"lw": 1.2}, hatch="//")
        for rect, val in list(zip(b1, day_m)) + list(zip(b2, ngt_m)):
            if abs(val) > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        val + (0.02 if val >= 0 else -0.02), f"{val:+.2f}",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5)

        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_title(f"{group} Group -- KG Climate Zones (Annual, n={len(sub)})",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{k}\n({KG_GROUP_MAP.get(k, k)})\nn={len(sub[sub['kg_group']==k])}"
             for k in kg_order],
            fontsize=8.5
        )
        ax.set_ylabel(r"$\Delta$LC (%)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Labour Capacity Difference by KG Climate Zone (Annual)\n"
        f"[{METHOD_DESC.get(method, method)}]  "
        "Positive = urban advantage;  Negative = urban disadvantage",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 14. Three-period statistical tests
# ─────────────────────────────────────────────────────────────
def save_period_comparison_stats(df, out_path):
    target_periods = ["annual", "warm_season", "heatwave"]
    pairs_to_test  = [
        ("annual", "warm_season"),
        ("annual", "heatwave"),
        ("warm_season", "heatwave"),
    ]
    metrics = []
    for m in ["dunne", "high", "medium", "low"]:
        metrics += [
            f"dlc_{m}_day_mean", f"dlc_{m}_night_mean",
            f"net_effect_{m}", f"asymmetry_index_{m}", f"day_night_ratio_{m}",
        ]
    metrics += ["dwbgt_day_mean", "dwbgt_night_mean"]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for group in ["UHI", "UCI"]:
        sub_g = df[df["group"] == group]
        for metric in metrics:
            gd   = [sub_g[sub_g["period"] == p][metric].dropna().values.astype(float)
                    for p in target_periods]
            gd_v = [g for g in gd if len(g) >= 3]
            if len(gd_v) >= 2:
                try:    kw_stat, kw_p = stats.kruskal(*gd_v)
                except: kw_stat, kw_p = np.nan, np.nan
            else:
                kw_stat, kw_p = np.nan, np.nan
            for p1, p2 in pairs_to_test:
                a = sub_g[sub_g["period"] == p1][metric].dropna().values.astype(float)
                b = sub_g[sub_g["period"] == p2][metric].dropna().values.astype(float)
                mn_a, lo_a, hi_a = bootstrap_ci(a) if len(a) >= 3 else (np.nan,)*3
                mn_b, lo_b, hi_b = bootstrap_ci(b) if len(b) >= 3 else (np.nan,)*3
                if len(a) >= 5 and len(b) >= 5:
                    try:
                        _, mwu_p  = stats.mannwhitneyu(a, b, alternative="two-sided")
                        mwu_p_adj = min(float(mwu_p) * 3, 1.0)
                    except:
                        mwu_p = mwu_p_adj = np.nan
                else:
                    mwu_p = mwu_p_adj = np.nan
                rows.append({
                    "group": group, "metric": metric,
                    "period_A": p1, "period_B": p2,
                    "n_A": len(a), "mean_A": mn_a, "ci_lo_A": lo_a, "ci_hi_A": hi_a,
                    "n_B": len(b), "mean_B": mn_b, "ci_lo_B": lo_b, "ci_hi_B": hi_b,
                    "mean_diff_AminusB": (
                        (mn_a - mn_b) if pd.notna(mn_a) and pd.notna(mn_b) else np.nan
                    ),
                    "kw_stat": kw_stat, "kw_p": kw_p, "kw_sig": sig_stars(kw_p),
                    "mwu_p": mwu_p, "mwu_p_bonferroni": mwu_p_adj,
                    "mwu_sig": sig_stars(mwu_p_adj),
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 15. Results summary for paper writing
# ─────────────────────────────────────────────────────────────
def save_results_summary(df, output_dir):
    out_path = os.path.join(output_dir, "results_summary.txt")
    lines = []

    def h1(t): lines.extend(["\n" + "=" * 72, f"  {t}", "=" * 72])
    def h2(t): lines.append(f"\n-- {t} " + "-" * max(1, 60 - len(t)))
    def note(t): lines.append(f"  [NOTE] {t}")
    def blank(): lines.append("")

    lines.extend([
        "=" * 72,
        "  LABOUR CAPACITY LOSS: DIURNAL ASYMMETRY RESULTS SUMMARY",
        f"  Input : {INPUT_CSV}",
        "  Methods: Dunne (2013) / He (2022) high / medium / low",
        "  Periods: Annual | Summer/JJA | Heatwave",
        # [v3→v5]
        "  Climate zone: Koeppen-Geiger (primary) + lat_group (compat.)",
        "=" * 72,
    ])

    n_pairs = df["pair_id"].nunique() if "pair_id" in df.columns else "N/A"
    n_uhi   = (df[df["group"] == "UHI"]["pair_id"].nunique()
               if "pair_id" in df.columns else len(df[df["group"] == "UHI"]))
    n_uci   = (df[df["group"] == "UCI"]["pair_id"].nunique()
               if "pair_id" in df.columns else len(df[df["group"] == "UCI"]))
    lines += [
        f"\n  Total city-rural pairs : {n_pairs}",
        f"  UHI group              : {n_uhi}",
        f"  UCI group              : {n_uci}",
    ]

    # ── [v3→v5] KG 分布快报 ──────────────────────────────────
    annual_pct = df[(df["period"] == "annual") & (df["hw_method"] == "percentile")] \
        if "hw_method" in df.columns else df[df["period"] == "annual"]
    if "kg_group" in annual_pct.columns and len(annual_pct) > 0:
        lines.append("\n  KG group distribution (annual x percentile):")
        kg_dist = annual_pct["kg_group"].value_counts(dropna=False)
        for grp, cnt in kg_dist.items():
            label = KG_GROUP_MAP.get(str(grp), str(grp))
            lines.append(f"    {grp} ({label}): {cnt}")

    mdl = {
        "dunne":  "Dunne (2013) Heavy Labour (350-500 kcal/h)",
        "high":   "He (2022) High Intensity -- Construction",
        "medium": "He (2022) Medium Intensity -- Manufacturing/Transport",
        "low":    "He (2022) Low Intensity -- Service",
    }

    # Part 1
    h1("PART 1 | WBGT BACKGROUND")
    note("WBGT_shade = 0.7*WBT + 0.3*Tdry  (Stull 2011 wet-bulb approx).")
    note("Dunne thresholds: 25 degC (onset), 27.9 degC (50%), 30 degC (light mix).")
    blank()
    for group in ["UHI", "UCI"]:
        ann = df[(df["group"] == group) & (df["period"] == "annual")]
        if len(ann) == 0: continue
        h2(f"{group} Group  (Annual, n={len(ann)})")
        for role, lbl in [("urban", "Urban"), ("rural", "Rural")]:
            dm_c = f"wbgt_{role}_day_mean"; nm_c = f"wbgt_{role}_night_mean"
            if dm_c not in ann.columns: continue
            dm, dlo, dhi = bootstrap_ci(ann[dm_c]); nm, nlo, nhi = bootstrap_ci(ann[nm_c])
            lines.append(f"  {lbl:6s}  Day WBGT  : {dm:+.2f} degC  [{dlo:+.2f}, {dhi:+.2f}]")
            lines.append(f"  {lbl:6s}  Night WBGT: {nm:+.2f} degC  [{nlo:+.2f}, {nhi:+.2f}]")
        for period in ["annual", "warm_season", "heatwave"]:
            sub = df[(df["group"] == group) & (df["period"] == period)]
            if len(sub) == 0: continue
            dm, dlo, dhi = bootstrap_ci(sub.get("dwbgt_day_mean",   pd.Series(dtype=float)))
            nm, nlo, nhi = bootstrap_ci(sub.get("dwbgt_night_mean", pd.Series(dtype=float)))
            pl = PERIOD_LABEL.get(period, period)
            lines.append(f"  dWBGT_day  [{pl:<12s}]: {dm:+.3f} degC  [{dlo:+.3f}, {dhi:+.3f}]")
            lines.append(f"  dWBGT_night[{pl:<12s}]: {nm:+.3f} degC  [{nlo:+.3f}, {nhi:+.3f}]")
        pct25 = (
            (ann["wbgt_urban_day_mean"] > 25).mean()
            if "wbgt_urban_day_mean" in ann.columns else np.nan
        )
        note(f"{group}: {pct25:.1%} of pairs have urban daytime WBGT > 25 degC (annual average).")

    # Part 2
    h1("PART 2 | LABOUR CAPACITY DIFFERENCE (dLC) -- ANNUAL BASELINE")
    note("dLC = LC_urban - LC_rural.  +: urban advantage.  -: rural advantage.")
    blank()
    for method in ["dunne", "high", "medium", "low"]:
        h2(mdl[method])
        for group in ["UHI", "UCI"]:
            ann = df[(df["group"] == group) & (df["period"] == "annual")]
            if len(ann) == 0: continue
            dm, dlo, dhi = bootstrap_ci(ann.get(f"dlc_{method}_day_mean",   pd.Series(dtype=float)))
            nm, nlo, nhi = bootstrap_ci(ann.get(f"dlc_{method}_night_mean", pd.Series(dtype=float)))
            wm, wlo, whi = bootstrap_ci(ann.get(f"dlc_{method}_work_mean",  pd.Series(dtype=float)))
            ne, nelo, nehi = bootstrap_ci(ann.get(f"net_effect_{method}",   pd.Series(dtype=float)))
            lines += [
                f"\n  [{group}]  n={len(ann)}",
                f"    Daytime  dLC (09-16h)      : {dm:+.3f}%  [{dlo:+.3f}, {dhi:+.3f}]",
                f"    Nighttime dLC (21-06h)     : {nm:+.3f}%  [{nlo:+.3f}, {nhi:+.3f}]",
                f"    Working-hr dLC (08-20h)    : {wm:+.3f}%  [{wlo:+.3f}, {whi:+.3f}]",
                f"    Net (8h*day + 10h*night)/24: {ne:+.3f}%  [{nelo:+.3f}, {nehi:+.3f}]",
            ]
            for arr, tag in [
                (ann.get(f"dlc_{method}_day_mean",   pd.Series(dtype=float)).dropna().values, "day"),
                (ann.get(f"dlc_{method}_night_mean", pd.Series(dtype=float)).dropna().values, "night"),
            ]:
                if len(arr) >= 5:
                    try:
                        _, pv = stats.wilcoxon(arr)
                        lines.append(f"    Wilcoxon vs 0 ({tag}): p={pv:.4f} {sig_stars(pv)}")
                    except: pass

    # Part 3
    h1("PART 3 | ASYMMETRY INDEX (AI) & QUADRANT CLASSIFICATION")
    note("AI = dLC_day - |dLC_night|.  >0: day protection dominates.  <0: night cost dominates.")
    note("'day_save_night_cost' is the typical asymmetric UHI pattern.")
    blank()
    for method in ["dunne", "high", "medium", "low"]:
        h2(mdl[method])
        for group in ["UHI", "UCI"]:
            for period in ["annual", "warm_season", "heatwave"]:
                sub = df[(df["group"] == group) & (df["period"] == period)]
                if len(sub) == 0: continue
                aim, ailo, aihi = bootstrap_ci(
                    sub.get(f"asymmetry_index_{method}", pd.Series(dtype=float))
                )
                ratm = sub.get(f"day_night_ratio_{method}", pd.Series(dtype=float)).median()
                pl = PERIOD_LABEL.get(period, period)
                lines.append(
                    f"  [{group}] {pl:<14s}  AI={aim:+.3f}% [{ailo:+.3f},{aihi:+.3f}]  "
                    f"day/night ratio (median)={ratm:.2f}"
                )
        blank()
        if method == "dunne" and "asymmetry_class_dunne" in df.columns:
            h2("Quadrant frequency  (Dunne)")
            for period in ["annual", "warm_season", "heatwave"]:
                sub_p = df[df["period"] == period]
                pl    = PERIOD_LABEL.get(period, period)
                lines.append(f"  Period: {pl}")
                for group in ["UHI", "UCI"]:
                    sub = sub_p[sub_p["group"] == group]
                    if len(sub) == 0: continue
                    cnt = sub["asymmetry_class_dunne"].value_counts(); total = cnt.sum()
                    lines.append(f"    {group}  (n={total})")
                    for cls, n in cnt.items():
                        lines.append(f"      {cls:<30s}: {n:>4d}  ({n/total*100:.1f}%)")

    # Part 4
    h1("PART 4 | THREE-PERIOD COMPARISON  (Annual vs Summer vs Heatwave)")
    note("Mann-Whitney U with Bonferroni correction (alpha/3).")
    blank()
    stats_path = os.path.join(output_dir, "period_comparison_stats.csv")
    if os.path.exists(stats_path):
        stat_df = pd.read_csv(stats_path)
        focus   = [
            "dlc_dunne_day_mean", "dlc_dunne_night_mean",
            "net_effect_dunne", "asymmetry_index_dunne",
        ]
        for group in ["UHI", "UCI"]:
            h2(f"{group} Group -- key metrics")
            sub_s = stat_df[stat_df["group"] == group]
            for metric in focus:
                sub_m = sub_s[sub_s["metric"] == metric]
                if len(sub_m) == 0: continue
                lines.append(f"\n  Metric: {metric}")
                for _, row in sub_m.iterrows():
                    pA_l = PERIOD_LABEL.get(row["period_A"], row["period_A"])
                    pB_l = PERIOD_LABEL.get(row["period_B"], row["period_B"])
                    lines.append(
                        f"    {pA_l:<14s} vs {pB_l:<14s}: "
                        f"A={row.get('mean_A', np.nan):+.3f}  "
                        f"B={row.get('mean_B', np.nan):+.3f}  "
                        f"diff={row.get('mean_diff_AminusB', np.nan):+.3f}  "
                        f"p_adj={row.get('mwu_p_bonferroni', np.nan):.4f} "
                        f"{row.get('mwu_sig', '')}  "
                        f"KW={row.get('kw_sig', '')}"
                    )
    else:
        note("period_comparison_stats.csv not found -- run save_period_comparison_stats() first.")

    blank()
    h2("Three-period quick-reference table  (Dunne heavy labour)")
    lines.append(
        f"  {'Group':<6s} {'Period':<14s} {'dLC_day':>9s} "
        f"{'dLC_night':>10s} {'Net':>8s} {'AI':>8s}"
    )
    lines.append("  " + "-" * 58)
    for group in ["UHI", "UCI"]:
        for period in ["annual", "warm_season", "heatwave"]:
            sub = df[(df["group"] == group) & (df["period"] == period)]
            if len(sub) == 0: continue
            dm  = sub.get("dlc_dunne_day_mean",  pd.Series(dtype=float)).mean()
            nm  = sub.get("dlc_dunne_night_mean", pd.Series(dtype=float)).mean()
            net = sub.get("net_effect_dunne",      pd.Series(dtype=float)).mean()
            ai  = sub.get("asymmetry_index_dunne", pd.Series(dtype=float)).mean()
            pl  = PERIOD_LABEL.get(period, period)
            lines.append(
                f"  {group:<6s} {pl:<14s} {dm:>+9.3f} {nm:>+10.3f} "
                f"{net:>+8.3f} {ai:>+8.3f}"
            )
        lines.append("  " + "-" * 58)

    # Part 5
    h1("PART 5 | ROBUSTNESS: METHOD COMPARISON")
    note("Consistent sign across all methods = robust finding.")
    blank()
    h2("Net effect (annual) -- all methods")
    lines.append(
        f"  {'Group':<6s} {'Method':<12s} {'Net mean':>10s} "
        f"{'CI_lo':>8s} {'CI_hi':>8s} {'Sig vs 0':>10s}"
    )
    lines.append("  " + "-" * 56)
    for group in ["UHI", "UCI"]:
        ann = df[(df["group"] == group) & (df["period"] == "annual")]
        if len(ann) == 0: continue
        for method in ["dunne", "high", "medium", "low"]:
            col = f"net_effect_{method}"
            if col not in ann.columns: continue
            mn, lo, hi = bootstrap_ci(ann[col]); arr = ann[col].dropna().values; sig = ""
            if len(arr) >= 5:
                try:    _, pv = stats.wilcoxon(arr); sig = sig_stars(pv)
                except: pass
            lines.append(
                f"  {group:<6s} {method:<12s} {mn:>+10.3f} "
                f"{lo:>+8.3f} {hi:>+8.3f} {sig:>10s}"
            )
        lines.append("  " + "-" * 56)

    # ── [v3→v5] Part 8: KG 分层快速参考表 ─────────────────────
    h1("PART 8 | KG CLIMATE ZONE STRATIFICATION  [v3 NEW]")
    note("Primary grouping: Koeppen-Geiger (A=Tropical, B=Arid, C=Temperate, D=Cold, E=Polar).")
    note("lat_group is retained as compatibility field only.")
    blank()

    annual_df = df[df["period"] == "annual"].copy()
    if "kg_group" in annual_df.columns and len(annual_df) > 0:
        for method in ["dunne"]:          # 仅输出 Dunne 节省篇幅，其余见 CSV
            h2(f"KG x Group  (Annual, {mdl[method]})")
            lines.append(
                f"  {'KG':<4s} {'Climate':<12s} {'Group':<6s} "
                f"{'n':>5s} {'dLC_day':>9s} {'dLC_night':>10s} "
                f"{'Net':>8s} {'AI':>8s}"
            )
            lines.append("  " + "-" * 64)
            kg_present = sorted(
                annual_df["kg_group"].dropna().unique(),
                key=lambda k: KG_GROUPS_ALL.index(k) if k in KG_GROUPS_ALL else 99
            )
            for kg in kg_present:
                climate = KG_GROUP_MAP.get(str(kg), kg)
                for group in ["UHI", "UCI"]:
                    sub_kg = annual_df[
                        (annual_df["kg_group"] == kg) & (annual_df["group"] == group)
                    ]
                    if len(sub_kg) < 3:
                        continue
                    dm  = sub_kg.get(f"dlc_{method}_day_mean",  pd.Series(dtype=float)).mean()
                    nm  = sub_kg.get(f"dlc_{method}_night_mean", pd.Series(dtype=float)).mean()
                    net = sub_kg.get(f"net_effect_{method}",     pd.Series(dtype=float)).mean()
                    ai  = sub_kg.get(f"asymmetry_index_{method}", pd.Series(dtype=float)).mean()
                    lines.append(
                        f"  {kg:<4s} {climate:<12s} {group:<6s} "
                        f"{len(sub_kg):>5d} {dm:>+9.3f} {nm:>+10.3f} "
                        f"{net:>+8.3f} {ai:>+8.3f}"
                    )
            lines.append("  " + "-" * 64)
            note("Full per-kg_code detail: summary_kggroup.csv / summary_kgcode.csv")
    else:
        note("kg_group column not found -- KG stratification skipped.")

    # Part 6
    h1("PART 6 | INTERPRETATION GUIDE FOR PAPER WRITING")
    blank()
    interp = [
        ("Finding 1 -- WBGT background",
         "Urban and rural areas show distinct WBGT diurnal profiles. During daytime, "
         "urban WBGT is [higher/lower] than rural by dWBGT_day (Part 1), reflecting "
         "[UCI cooling / UHI warming]. At night, urban WBGT consistently exceeds rural "
         "due to nocturnal heat island retention."),
        ("Finding 2 -- Labour capacity difference",
         "dLC is positive in daytime for UCI cities and negative for UHI cities, showing "
         "that urban form determines the direction of daytime heat advantage. At night, "
         "dLC is negative for both groups, confirming that nocturnal UHI universally "
         "impairs worker recovery."),
        ("Finding 3 -- Asymmetry Index",
         "AI = dLC_day - |dLC_night| quantifies whether daytime protection or nighttime "
         "cost dominates. Positive AI (UCI cities) = net urban advantage; negative AI "
         "(UHI cities) = net urban disadvantage. AI magnitude increases from Annual "
         "through Summer to Heatwave, suggesting extreme heat non-linearly amplifies "
         "the asymmetry."),
        ("Finding 4 -- Heatwave amplification",
         "Three-period comparison (Part 4) tests whether heatwave conditions significantly "
         "amplify diurnal asymmetry beyond the summer baseline. Significant Mann-Whitney U "
         "(Bonferroni-corrected) between heatwave and annual confirms heatwave-specific "
         "amplification. The direction differs between UHI and UCI groups."),
        ("Finding 5 -- Robustness",
         "Consistent net effect sign across Dunne (2013) and He et al. (2022) models "
         "(Part 5) confirms the asymmetry pattern is not a model artefact. Dunne detects "
         "mainly heatwave signals (threshold at 25 degC WBGT); He models detect summer "
         "background signals at lower temperatures."),
        # [v3→v5]
        ("Finding 6 -- KG climate zone heterogeneity",
         "Part 8 shows that the sign and magnitude of dLC_day, dLC_night, and AI vary "
         "across KG groups. Arid (B) cities tend to show the largest nighttime heat penalty "
         "while Cold (D) cities show the smallest. Temperate (C) cities mirror the global "
         "mean pattern most closely. See summary_kggroup.csv for full stratified statistics."),
    ]
    for title, text in interp:
        lines.append(f"  >> {title}")
        words = text.split(); buf = "     "; max_w = 68
        for w in words:
            if len(buf) + len(w) + 1 > max_w: lines.append(buf); buf = "     " + w + " "
            else: buf += w + " "
        if buf.strip(): lines.append(buf)
        blank()

    # Part 7
    h1("PART 7 | SUGGESTED PAPER SENTENCES (auto-filled)")
    blank(); lines.append("  Replace bracketed text with significance values from Parts 1-5.\n")

    ann_uhi = df[(df["group"] == "UHI") & (df["period"] == "annual")]
    ann_uci = df[(df["group"] == "UCI") & (df["period"] == "annual")]
    hw_uhi  = df[(df["group"] == "UHI") & (df["period"] == "heatwave")]
    hw_uci  = df[(df["group"] == "UCI") & (df["period"] == "heatwave")]

    def fmt(series):
        m, lo, hi = bootstrap_ci(series)
        return "[N/A]" if pd.isna(m) else f"{m:+.2f}% (95%CI: {lo:+.2f}, {hi:+.2f})"

    sentences = []
    if len(ann_uhi) > 0 and "dlc_dunne_day_mean" in df.columns:
        sentences.append(
            f"S1 (UHI annual): Under annual mean conditions, UHI-type cities showed a "
            f"daytime labour capacity difference of {fmt(ann_uhi['dlc_dunne_day_mean'])} "
            f"and a nighttime difference of {fmt(ann_uhi['dlc_dunne_night_mean'])} relative "
            f"to paired rural stations (Dunne heavy labour model)."
        )
    if len(ann_uci) > 0 and "dlc_dunne_day_mean" in df.columns:
        sentences.append(
            f"S2 (UCI annual): UCI-type cities showed a daytime difference of "
            f"{fmt(ann_uci['dlc_dunne_day_mean'])} and a nighttime difference of "
            f"{fmt(ann_uci['dlc_dunne_night_mean'])}, indicating daytime urban cooling "
            f"partially offsets but does not eliminate the nocturnal heat island penalty "
            f"on worker recovery."
        )
    if len(ann_uhi) > 0 and "asymmetry_index_dunne" in df.columns:
        sentences.append(
            f"S3 (AI annual): The Asymmetry Index for UHI cities under annual conditions "
            f"was {fmt(ann_uhi['asymmetry_index_dunne'])}, indicating that [nighttime cost / "
            f"daytime protection] dominated the net urban labour capacity effect."
        )
    if len(hw_uhi) > 0 and "asymmetry_index_dunne" in df.columns:
        sentences.append(
            f"S4 (AI heatwave): During heatwave periods, the Asymmetry Index was "
            f"{fmt(hw_uhi['asymmetry_index_dunne'])} for UHI cities and "
            f"{fmt(hw_uci['asymmetry_index_dunne']) if len(hw_uci) > 0 else '[N/A]'} for UCI "
            f"cities, demonstrating that extreme heat non-linearly amplifies the diurnal "
            f"asymmetry of urban labour capacity loss."
        )
    if len(ann_uhi) > 0 and "net_effect_dunne" in df.columns:
        sentences.append(
            f"S5 (Net effect): The hour-weighted net effect was "
            f"{fmt(ann_uhi['net_effect_dunne'])} for UHI cities and "
            f"{fmt(ann_uci['net_effect_dunne']) if len(ann_uci) > 0 else '[N/A]'} for UCI "
            f"cities (annual), with [significant/non-significant] amplification during "
            f"heatwave periods (Mann-Whitney U, Bonferroni-corrected p [value])."
        )

    for s in sentences:
        words = s.split(); buf = "  "; max_w = 70
        for w in words:
            if len(buf) + len(w) + 1 > max_w: lines.append(buf); buf = "  " + w + " "
            else: buf += w + " "
        if buf.strip(): lines.append(buf)
        blank()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")
    return out_path
# ─────────────────────────────────────────────────────────────
# 15b. Export compact table for integrated figure
# ─────────────────────────────────────────────────────────────

def save_integrated_figure_input(df, output_dir):
    """
    导出给 integrated figure 使用的轻量 labour 表。

    最小修改原则：
    1. 不改数学模型，不重算 LC / WBGT。
    2. 不改数据来源，只使用当前 df 中已经存在的 dlc_dunne_* / dloss_dunne_* 列。
    3. 同时导出 capacity 方向和 loss 方向，避免下游符号混淆。

    Main capacity definition:
        d_labour_capacity_pct = dlc_dunne = LC_urban - LC_rural

    Optional loss definition:
        d_labour_loss_pct = -d_labour_capacity_pct
    """
    keep_periods = ["heatwave", "non_heatwave"]
    sub = df[df["period"].isin(keep_periods)].copy()

    if len(sub) == 0:
        print("    [integrated labour input] no heatwave/non_heatwave rows; skipped.")
        return None

    # 基础 metadata
    keep_cols = [
        "pair_id",
        "period",
        "group",
        "kg_group",
        "kg_code",
        "lat_group",
        "continent",
        "climate_zone",
        "density",
        "hw_method",
        "n_days",
    ]

    # 保留原有 labour capacity / loss summary 列
    for m in ["dunne", "high", "medium", "low"]:
        keep_cols += [
            f"dlc_{m}_day_mean",
            f"dlc_{m}_night_mean",
            f"dlc_{m}_work_mean",
            f"dlc_{m}_24h_mean",
            f"dlc_{m}_day_peak",
            f"dlc_{m}_work_peak",
            f"dloss_{m}_day_mean",
            f"dloss_{m}_night_mean",
            f"dloss_{m}_work_mean",
            f"dloss_{m}_24h_mean",
            f"dloss_{m}_day_peak",
            f"dloss_{m}_work_peak",
            f"dloss_{m}_day_peak_abs",
            f"dloss_{m}_work_peak_abs",
            f"dloss_{m}_peak_t_diff",
            f"net_effect_{m}",
            f"asymmetry_index_{m}",
            f"day_night_ratio_{m}",
        ]

        # 关键：导出完整 hourly dLC，保证 Plot fig4 可与 fig1_diurnal_labour_real.csv 同源
        for h in range(24):
            keep_cols.append(f"dlc_{m}_h{h:02d}")

    # WBGT 和温度诊断列保留
    keep_cols += [
        "dwbgt_day_mean",
        "dwbgt_night_mean",
        "wbgt_urban_day_mean",
        "wbgt_rural_day_mean",
        "wbgt_urban_night_mean",
        "wbgt_rural_night_mean",
    ]

    keep_cols = [c for c in keep_cols if c in sub.columns]
    out = sub[keep_cols].copy()

    out["period_norm"] = out["period"].map({
        "heatwave": "HW",
        "non_heatwave": "NHW",
    })

    # 统一给 Figure 4 使用的主字段：capacity 方向
    primary_method = "dunne"

    out["d_labour_capacity_pct_work_mean"] = out.get(
        f"dlc_{primary_method}_work_mean",
        np.nan
    )
    out["d_labour_capacity_pct_day_mean"] = out.get(
        f"dlc_{primary_method}_day_mean",
        np.nan
    )
    out["d_labour_capacity_pct_night_mean"] = out.get(
        f"dlc_{primary_method}_night_mean",
        np.nan
    )
    out["d_labour_capacity_pct_24h_mean"] = out.get(
        f"dlc_{primary_method}_24h_mean",
        np.nan
    )

    # loss 方向作为并行列，不作为 capacity 主图默认列
    out["d_labour_loss_pct_work_mean"] = out.get(
        f"dloss_{primary_method}_work_mean",
        -out["d_labour_capacity_pct_work_mean"]
    )
    out["d_labour_loss_pct_day_mean"] = out.get(
        f"dloss_{primary_method}_day_mean",
        -out["d_labour_capacity_pct_day_mean"]
    )
    out["d_labour_loss_pct_night_mean"] = out.get(
        f"dloss_{primary_method}_night_mean",
        -out["d_labour_capacity_pct_night_mean"]
    )
    out["d_labour_loss_pct_24h_mean"] = out.get(
        f"dloss_{primary_method}_24h_mean",
        -out["d_labour_capacity_pct_24h_mean"]
    )

    # signed peak from the same hourly dLC curve
    work_cols = [f"dlc_{primary_method}_h{h:02d}" for h in WORK_HOURS]
    work_cols = [c for c in work_cols if c in out.columns]

    if work_cols:
        work_arr = out[work_cols].to_numpy(dtype=float)
        out["d_labour_capacity_pct_work_peak_signed"] = np.nanmax(work_arr, axis=1)
        out["d_labour_capacity_pct_work_trough_signed"] = np.nanmin(work_arr, axis=1)
        out["d_labour_capacity_pct_work_peak_abs"] = np.nanmax(np.abs(work_arr), axis=1)

        loss_arr = -work_arr
        out["d_labour_loss_pct_work_peak_signed"] = np.nanmax(loss_arr, axis=1)
        out["d_labour_loss_pct_work_trough_signed"] = np.nanmin(loss_arr, axis=1)
        out["d_labour_loss_pct_work_peak_abs"] = np.nanmax(np.abs(loss_arr), axis=1)
    else:
        out["d_labour_capacity_pct_work_peak_signed"] = np.nan
        out["d_labour_capacity_pct_work_trough_signed"] = np.nan
        out["d_labour_capacity_pct_work_peak_abs"] = np.nan
        out["d_labour_loss_pct_work_peak_signed"] = np.nan
        out["d_labour_loss_pct_work_trough_signed"] = np.nan
        out["d_labour_loss_pct_work_peak_abs"] = np.nan

    out["labour_primary_method"] = primary_method
    out["labour_capacity_sign"] = "positive = urban labour capacity > rural labour capacity"
    out["labour_loss_sign"] = "positive = urban labour loss > rural labour loss"
    out["labour_source_columns"] = "dlc_dunne_h00-h23 and dlc_dunne_* summary columns"

    out_path = os.path.join(output_dir, "integrated_figure_labour_input.csv")
    out.to_csv(out_path, index=False)

    print("    Saved: integrated_figure_labour_input.csv")
    print("    Labour capacity source: dlc_dunne_h00-h23 / dlc_dunne_*")
    print("    Labour loss source: -dlc_dunne or existing dloss_dunne_*")
    return out_path

# ══════════════════════════════════════════════════════════════
# Figure 4 export module: labour pathway data
# ══════════════════════════════════════════════════════════════

def _fig4_first_existing_col(df, candidates):
    """Return the first existing column from a candidate list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fig4_curve_from_row(row, col_template):
    """Extract a 24-hour curve from row using a template like 'wbgt_urban_h{h:02d}'."""
    return np.array(
        [row.get(col_template.format(h=h), np.nan) for h in range(24)],
        dtype=float
    )


def _fig4_excess_sum(curve, hours, threshold):
    """
    Sum positive exceedance over selected hours.

    Because each row in all_pair_period_metrics is already a mean diurnal cycle,
    summing hourly exceedance gives degree-hours per representative day.
    """
    idx = np.array(hours, dtype=int)
    vals = np.asarray(curve, dtype=float)[idx]
    valid = np.isfinite(vals)

    if valid.sum() == 0:
        return np.nan, 0

    excess = np.maximum(vals[valid] - threshold, 0.0)
    return float(np.sum(excess)), int(valid.sum())


def _fig4_build_labour_pair_period_table(
    df,
    primary_method=FIG4_PRIMARY_LABOUR_METHOD,
    theta_day_temp=FIG4_THETA_DAY_TEMP,
    theta_wbgt=FIG4_THETA_WBGT,
):
    """
    Build Figure 4 labour pair-period table.

    Minimal scientific change:
    - Does not recompute LC or WBGT.
    - Uses existing dlc_{method}_h00-h23, dlc_{method}_* and dloss_{method}_* columns.
    - Adds labour-capacity direction columns parallel to existing labour-loss columns.

    Main Figure 4 capacity endpoint:
        labour_capacity_pct = dlc_{method}_work_mean
        positive = urban labour capacity > rural labour capacity

    Parallel loss endpoint:
        labour_loss_pct = dloss_{method}_work_mean = -labour_capacity_pct
        positive = urban labour loss > rural labour loss
    """

    if "pair_id" not in df.columns:
        raise ValueError("[Fig4 labour] Missing required column: pair_id")

    work = df.copy()

    if "period_norm" not in work.columns:
        work["period_norm"] = work["period"].map({
            "heatwave": "HW",
            "non_heatwave": "NHW",
            "HW": "HW",
            "NHW": "NHW",
        })

    work = work[work["period_norm"].isin(["HW", "NHW"])].copy()

    if len(work) == 0:
        raise ValueError("[Fig4 labour] No HW/NHW rows found.")

    meta_cols = [
        "period",
        "group",
        "kg_group",
        "kg_code",
        "lat_group",
        "continent",
        "climate_zone",
        "density",
        "hw_method",
        "n_days",
    ]
    meta_cols = [c for c in meta_cols if c in work.columns]

    records = []

    for _, row in work.iterrows():
        # Existing temperature curves from upstream all_pair_period_metrics.
        t_u = _fig4_curve_from_row(row, "urban_diurnal_h{h:02d}")
        t_r = _fig4_curve_from_row(row, "rural_diurnal_h{h:02d}")

        day_u, n_day_u = _fig4_excess_sum(t_u, FIG4_DAY_EXPOSURE_HOURS, theta_day_temp)
        day_r, n_day_r = _fig4_excess_sum(t_r, FIG4_DAY_EXPOSURE_HOURS, theta_day_temp)

        work_u, n_work_u = _fig4_excess_sum(t_u, FIG4_WORK_EXPOSURE_HOURS, theta_day_temp)
        work_r, n_work_r = _fig4_excess_sum(t_r, FIG4_WORK_EXPOSURE_HOURS, theta_day_temp)

        # Existing WBGT curves from compute_row_lc(); no WBGT recomputation here.
        wbgt_u = _fig4_curve_from_row(row, "wbgt_urban_h{h:02d}")
        wbgt_r = _fig4_curve_from_row(row, "wbgt_rural_h{h:02d}")

        wbgt_work_u, n_wbgt_work_u = _fig4_excess_sum(
            wbgt_u, FIG4_WORK_EXPOSURE_HOURS, theta_wbgt
        )
        wbgt_work_r, n_wbgt_work_r = _fig4_excess_sum(
            wbgt_r, FIG4_WORK_EXPOSURE_HOURS, theta_wbgt
        )

        wbgt_day_u, n_wbgt_day_u = _fig4_excess_sum(
            wbgt_u, FIG4_DAY_EXPOSURE_HOURS, theta_wbgt
        )
        wbgt_day_r, n_wbgt_day_r = _fig4_excess_sum(
            wbgt_r, FIG4_DAY_EXPOSURE_HOURS, theta_wbgt
        )

        # Main source: existing hourly labour-capacity difference.
        dlc_curve = np.array(
            [row.get(f"dlc_{primary_method}_h{h:02d}", np.nan) for h in range(24)],
            dtype=float
        )
        dloss_curve = -dlc_curve

        def _mean_from_curve(curve, hours):
            vals = curve[np.array(hours, dtype=int)]
            return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

        def _max_from_curve(curve, hours):
            vals = curve[np.array(hours, dtype=int)]
            return float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan

        def _min_from_curve(curve, hours):
            vals = curve[np.array(hours, dtype=int)]
            return float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan

        def _maxabs_from_curve(curve, hours):
            vals = curve[np.array(hours, dtype=int)]
            return float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else np.nan

        # Prefer already-computed summary columns; fallback to same hourly dLC curve.
        capacity_work_mean = row.get(
            f"dlc_{primary_method}_work_mean",
            _mean_from_curve(dlc_curve, WORK_HOURS)
        )
        capacity_day_mean = row.get(
            f"dlc_{primary_method}_day_mean",
            _mean_from_curve(dlc_curve, DAY_HOURS)
        )
        capacity_night_mean = row.get(
            f"dlc_{primary_method}_night_mean",
            _mean_from_curve(dlc_curve, NIGHT_HOURS)
        )
        capacity_24h_mean = row.get(
            f"dlc_{primary_method}_24h_mean",
            float(np.nanmean(dlc_curve)) if np.isfinite(dlc_curve).any() else np.nan
        )

        loss_work_mean = row.get(
            f"dloss_{primary_method}_work_mean",
            -capacity_work_mean if pd.notna(capacity_work_mean) else np.nan
        )
        loss_day_mean = row.get(
            f"dloss_{primary_method}_day_mean",
            -capacity_day_mean if pd.notna(capacity_day_mean) else np.nan
        )
        loss_night_mean = row.get(
            f"dloss_{primary_method}_night_mean",
            -capacity_night_mean if pd.notna(capacity_night_mean) else np.nan
        )
        loss_24h_mean = row.get(
            f"dloss_{primary_method}_24h_mean",
            -capacity_24h_mean if pd.notna(capacity_24h_mean) else np.nan
        )

        rec = {
            "pair_id": row.get("pair_id"),
            "period_norm": row.get("period_norm"),

            # Mechanism / temperature fields kept unchanged.
            "dAmp1": row.get("dAmp1", np.nan),
            "dTmean": row.get("dTmean", np.nan),
            "dTx": row.get("dTx", np.nan),
            "dTn": row.get("dTn", np.nan),

            # Dry-bulb exposure.
            "day_heat_exposure": (
                day_u - day_r
                if np.isfinite(day_u) and np.isfinite(day_r) else np.nan
            ),
            "work_heat_exposure": (
                work_u - work_r
                if np.isfinite(work_u) and np.isfinite(work_r) else np.nan
            ),

            # WBGT exposure.
            "labour_wbgt_work_exposure": (
                wbgt_work_u - wbgt_work_r
                if np.isfinite(wbgt_work_u) and np.isfinite(wbgt_work_r) else np.nan
            ),
            "labour_wbgt_day_exposure": (
                wbgt_day_u - wbgt_day_r
                if np.isfinite(wbgt_day_u) and np.isfinite(wbgt_day_r) else np.nan
            ),

            # Main labour-capacity endpoint: same source as fig1_diurnal_labour_real.csv.
            "labour_capacity_pct": capacity_work_mean,
            "labour_capacity_day_pct": capacity_day_mean,
            "labour_capacity_night_pct": capacity_night_mean,
            "labour_capacity_24h_pct": capacity_24h_mean,

            # Signed peaks from the same hourly dLC curve.
            "labour_capacity_work_peak_signed": _max_from_curve(dlc_curve, WORK_HOURS),
            "labour_capacity_work_trough_signed": _min_from_curve(dlc_curve, WORK_HOURS),
            "labour_capacity_work_peak_abs": _maxabs_from_curve(dlc_curve, WORK_HOURS),

            # Parallel labour-loss endpoint for backward compatibility.
            "labour_loss_pct": loss_work_mean,
            "labour_loss_day_pct": loss_day_mean,
            "labour_loss_night_pct": loss_night_mean,
            "labour_loss_24h_pct": loss_24h_mean,
            "labour_loss_peak_pct": _max_from_curve(dloss_curve, WORK_HOURS),
            "labour_loss_trough_pct": _min_from_curve(dloss_curve, WORK_HOURS),
            "labour_loss_peak_abs_pct": _maxabs_from_curve(dloss_curve, WORK_HOURS),

            # Old diagnostic retained, but not the main endpoint.
            "labour_loss_tmax_peak_to_peak_pct": row.get(
                f"dloss_{primary_method}_peak_t_diff",
                np.nan
            ),

            # Diagnostics for valid-hour availability.
            "n_day_temp_hours_urban": n_day_u,
            "n_day_temp_hours_rural": n_day_r,
            "n_work_temp_hours_urban": n_work_u,
            "n_work_temp_hours_rural": n_work_r,
            "n_work_wbgt_hours_urban": n_wbgt_work_u,
            "n_work_wbgt_hours_rural": n_wbgt_work_r,

            # Settings.
            "theta_day_temp": theta_day_temp,
            "theta_wbgt": theta_wbgt,
            "primary_labour_method": primary_method,
            "labour_capacity_source": f"dlc_{primary_method}_h00-h23",
            "labour_capacity_sign": "positive = urban labour capacity > rural labour capacity",
            "labour_loss_sign": "positive = urban labour loss > rural labour loss",
        }

        for c in meta_cols:
            rec[c] = row.get(c, np.nan)

        # Export all methods as optional robustness columns.
        for m in ["dunne", "high", "medium", "low"]:
            rec[f"labour_loss_{m}_work_mean"] = row.get(f"dloss_{m}_work_mean", np.nan)
            rec[f"labour_loss_{m}_day_mean"] = row.get(f"dloss_{m}_day_mean", np.nan)
            rec[f"labour_loss_{m}_work_peak"] = row.get(f"dloss_{m}_work_peak", np.nan)
            rec[f"labour_capacity_{m}_work_mean"] = row.get(f"dlc_{m}_work_mean", np.nan)
            rec[f"labour_capacity_{m}_day_mean"] = row.get(f"dlc_{m}_day_mean", np.nan)
            rec[f"labour_capacity_{m}_night_mean"] = row.get(f"dlc_{m}_night_mean", np.nan)

            for h in range(24):
                rec[f"labour_capacity_{m}_h{h:02d}"] = row.get(f"dlc_{m}_h{h:02d}", np.nan)

        records.append(rec)

    out = pd.DataFrame(records)

    # Aggregate in case upstream contains duplicated pair_id × period_norm rows.
    group_keys = ["pair_id", "period_norm"]
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    first_cols = [
        c for c in out.columns
        if c not in numeric_cols and c not in group_keys
    ]

    agg_dict = {c: "mean" for c in numeric_cols}
    for c in first_cols:
        agg_dict[c] = "first"

    out_pp = (
        out.groupby(group_keys, as_index=False, dropna=False)
        .agg(agg_dict)
    )

    front_cols = [
        "pair_id", "period_norm", "period", "group",
        "dAmp1", "dTmean", "dTx", "dTn",
        "day_heat_exposure", "work_heat_exposure",
        "labour_wbgt_work_exposure", "labour_wbgt_day_exposure",

        # Capacity first.
        "labour_capacity_pct",
        "labour_capacity_day_pct",
        "labour_capacity_night_pct",
        "labour_capacity_24h_pct",
        "labour_capacity_work_peak_signed",
        "labour_capacity_work_trough_signed",
        "labour_capacity_work_peak_abs",

        # Loss retained after capacity.
        "labour_loss_pct",
        "labour_loss_day_pct",
        "labour_loss_night_pct",
        "labour_loss_24h_pct",
        "labour_loss_peak_pct",
        "labour_loss_trough_pct",
        "labour_loss_peak_abs_pct",
        "labour_loss_tmax_peak_to_peak_pct",

        "theta_day_temp", "theta_wbgt", "primary_labour_method",
        "labour_capacity_source",
        "labour_capacity_sign",
        "labour_loss_sign",
    ]
    front_cols = [c for c in front_cols if c in out_pp.columns]
    other_cols = [c for c in out_pp.columns if c not in front_cols]
    out_pp = out_pp[front_cols + other_cols]

    return out_pp

def _fig4_make_labour_paired_diffs(pair_period_df):
    """
    Convert pair-period labour table into HW − NHW paired differences.

    Main capacity response:
        dLabourCapacity = labour_capacity_pct_HW - labour_capacity_pct_NHW

    Main loss response:
        dLabourLoss = labour_loss_pct_HW - labour_loss_pct_NHW

    Sign:
        dLabourCapacity > 0:
            HW increases urban-rural labour-capacity difference.

        dLabourLoss > 0:
            HW increases urban-relative labour loss.
    """

    required = ["pair_id", "period_norm"]
    for c in required:
        if c not in pair_period_df.columns:
            raise ValueError(f"[Fig4 labour] Missing required column in pair-period table: {c}")

    hw = pair_period_df[pair_period_df["period_norm"] == "HW"].copy()
    nhw = pair_period_df[pair_period_df["period_norm"] == "NHW"].copy()

    if len(hw) == 0 or len(nhw) == 0:
        raise ValueError("[Fig4 labour] HW or NHW rows missing. Cannot compute paired differences.")

    merged = hw.merge(
        nhw,
        on="pair_id",
        how="inner",
        suffixes=("_HW", "_NHW")
    )

    if len(merged) == 0:
        raise ValueError("[Fig4 labour] No matched HW × NHW pair_id rows.")

    out = pd.DataFrame()
    out["pair_id"] = merged["pair_id"]

    for c in ["group", "kg_group", "kg_code", "lat_group", "continent", "climate_zone", "density"]:
        c_hw = f"{c}_HW"
        if c_hw in merged.columns:
            out[c] = merged[c_hw]

    def diff(base_col):
        a = f"{base_col}_HW"
        b = f"{base_col}_NHW"
        if a not in merged.columns or b not in merged.columns:
            return np.full(len(merged), np.nan)
        return (
            pd.to_numeric(merged[a], errors="coerce").to_numpy(float)
            - pd.to_numeric(merged[b], errors="coerce").to_numpy(float)
        )

    def copy_hw_nhw(base_col, out_base=None):
        out_base = out_base or base_col
        a = f"{base_col}_HW"
        b = f"{base_col}_NHW"

        if b in merged.columns:
            out[f"{out_base}_NHW"] = pd.to_numeric(merged[b], errors="coerce")
        else:
            out[f"{out_base}_NHW"] = np.nan

        if a in merged.columns:
            out[f"{out_base}_HW"] = pd.to_numeric(merged[a], errors="coerce")
        else:
            out[f"{out_base}_HW"] = np.nan

    # Mechanism.
    out["ddAmp_labour"] = diff("dAmp1")
    out["ampDamping_labour"] = -out["ddAmp_labour"]

    # Dry-bulb exposure differences.
    out["dDayHeat_labour"] = diff("day_heat_exposure")
    out["dayRelief_labour"] = -out["dDayHeat_labour"]

    out["dWorkHeat_labour"] = diff("work_heat_exposure")
    out["workRelief_labour"] = -out["dWorkHeat_labour"]

    # WBGT exposure differences.
    out["dLabourWBGTWorkExposure"] = diff("labour_wbgt_work_exposure")
    out["dLabourWBGTDayExposure"] = diff("labour_wbgt_day_exposure")

    # Copy HW/NHW values for reporting.
    for base_col in [
        "labour_capacity_pct",
        "labour_capacity_day_pct",
        "labour_capacity_night_pct",
        "labour_capacity_24h_pct",
        "labour_capacity_work_peak_signed",
        "labour_capacity_work_trough_signed",
        "labour_capacity_work_peak_abs",
        "labour_loss_pct",
        "labour_loss_day_pct",
        "labour_loss_night_pct",
        "labour_loss_24h_pct",
        "labour_loss_peak_pct",
        "labour_loss_trough_pct",
        "labour_loss_peak_abs_pct",
        "labour_loss_tmax_peak_to_peak_pct",
    ]:
        copy_hw_nhw(base_col)

    # Capacity response: HW - NHW
    out["dLabourCapacity"] = diff("labour_capacity_pct")
    out["R_labour_capacity_work_mean"] = out["dLabourCapacity"]
    out["R_labour_capacity_day_mean"] = diff("labour_capacity_day_pct")
    out["R_labour_capacity_night_mean"] = diff("labour_capacity_night_pct")
    out["R_labour_capacity_24h_mean"] = diff("labour_capacity_24h_pct")
    out["R_labour_capacity_work_peak_signed"] = diff("labour_capacity_work_peak_signed")
    out["R_labour_capacity_work_trough_signed"] = diff("labour_capacity_work_trough_signed")
    out["R_labour_capacity_work_peak_abs"] = diff("labour_capacity_work_peak_abs")

    # Loss response: HW - NHW
    out["dLabourLoss"] = diff("labour_loss_pct")
    out["R_labour_loss_work_mean"] = out["dLabourLoss"]
    out["R_labour_loss_day_mean"] = diff("labour_loss_day_pct")
    out["R_labour_loss_night_mean"] = diff("labour_loss_night_pct")
    out["R_labour_loss_24h_mean"] = diff("labour_loss_24h_pct")
    out["R_labour_loss_work_peak_signed"] = diff("labour_loss_peak_pct")
    out["R_labour_loss_work_peak_abs"] = diff("labour_loss_peak_abs_pct")

    # Old Tmax peak-to-peak response retained as diagnostic only.
    out["R_labour_loss_tmax_peak_to_peak"] = diff("labour_loss_tmax_peak_to_peak_pct")

    # Consistency diagnostics.
    out["capacity_loss_sign_check"] = (
        out["R_labour_loss_work_mean"] + out["R_labour_capacity_work_mean"]
    )

    out["valid_fig4_labour_capacity"] = np.isfinite(out["R_labour_capacity_work_mean"])
    out["valid_fig4_labour_loss"] = np.isfinite(out["R_labour_loss_work_mean"])

    # Backward-compatible name used by older print statements.
    out["valid_fig4_labour"] = out["valid_fig4_labour_capacity"]

    out["labour_capacity_sign"] = "positive = HW increases urban-rural labour-capacity difference"
    out["labour_loss_sign"] = "positive = HW increases urban-relative labour loss"
    out["labour_source_columns"] = "dlc_dunne_h00-h23 / dlc_dunne_*; dloss columns retained only as loss-direction mirror"

    if "dDayHeat_labour" in out.columns and "dLabourCapacity" in out.columns:
        ok = out[["dDayHeat_labour", "dLabourCapacity"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ok) >= 5:
            slope, intercept, r, p, se = stats.linregress(
                ok["dDayHeat_labour"],
                ok["dLabourCapacity"]
            )
            out["capacity_vs_dayheat_slope_all"] = slope
            out["capacity_vs_dayheat_r_all"] = r
            out["capacity_vs_dayheat_p_all"] = p
        else:
            out["capacity_vs_dayheat_slope_all"] = np.nan
            out["capacity_vs_dayheat_r_all"] = np.nan
            out["capacity_vs_dayheat_p_all"] = np.nan

    return out

def _fig4_make_labour_hw_change_summary(paired_diffs):
    """
    Group-level summary for Figure 4 labour HW response.

    This is a reporting table only.
    It does not change any model, source data, or pair-level values.
    """
    rows = []

    metrics = [
        (
            "labour_capacity_work_mean",
            "labour_capacity_pct",
            "R_labour_capacity_work_mean",
            "% labour capacity",
            "positive means HW increases urban-rural labour-capacity difference",
        ),
        (
            "labour_capacity_day_mean",
            "labour_capacity_day_pct",
            "R_labour_capacity_day_mean",
            "% labour capacity",
            "positive means HW increases daytime urban-rural labour-capacity difference",
        ),
        (
            "labour_capacity_night_mean",
            "labour_capacity_night_pct",
            "R_labour_capacity_night_mean",
            "% labour capacity",
            "positive means HW increases nighttime urban-rural labour-capacity difference",
        ),
        (
            "labour_capacity_work_peak_signed",
            "labour_capacity_work_peak_signed",
            "R_labour_capacity_work_peak_signed",
            "% labour capacity",
            "positive means HW increases the signed work-hour peak of urban-rural labour capacity",
        ),
        (
            "labour_loss_work_mean",
            "labour_loss_pct",
            "R_labour_loss_work_mean",
            "% labour loss",
            "positive means HW increases urban-relative labour loss",
        ),
    ]

    if paired_diffs is None or len(paired_diffs) == 0:
        return pd.DataFrame(rows)

    group_col = "group" if "group" in paired_diffs.columns else None
    groups = (
        sorted(paired_diffs[group_col].dropna().unique())
        if group_col is not None
        else ["ALL"]
    )

    for group in groups:
        if group_col is not None:
            sub = paired_diffs[paired_diffs[group_col] == group].copy()
        else:
            sub = paired_diffs.copy()

        for metric_name, base_col, response_col, unit, sign_def in metrics:
            nhw_col = f"{base_col}_NHW"
            hw_col = f"{base_col}_HW"

            if response_col not in sub.columns:
                continue

            vals = pd.to_numeric(sub[response_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

            if len(vals) == 0:
                mean_val, ci_lo, ci_hi = np.nan, np.nan, np.nan
            else:
                mean_val, ci_lo, ci_hi = bootstrap_ci(vals)

            rows.append({
                "group": group,
                "metric": metric_name,
                "n_pairs": int(len(vals)),
                "NHW_mean": (
                    float(pd.to_numeric(sub[nhw_col], errors="coerce").mean())
                    if nhw_col in sub.columns else np.nan
                ),
                "HW_mean": (
                    float(pd.to_numeric(sub[hw_col], errors="coerce").mean())
                    if hw_col in sub.columns else np.nan
                ),
                "HW_minus_NHW_mean": mean_val,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "unit": unit,
                "response_col": response_col,
                "source_col_NHW": nhw_col,
                "source_col_HW": hw_col,
                "sign_definition": sign_def,
                "source_note": "All labour-capacity metrics derive from dlc_dunne_h00-h23 or dlc_dunne_* columns.",
            })

    return pd.DataFrame(rows)

def _fig4_write_labour_schema(out_dir):
    """Write a human-readable schema for Figure 4 labour exports."""
    schema_path = os.path.join(out_dir, "fig4_labour_schema.txt")

    lines = [
        "Figure 4 labour data export",
        "=" * 72,
        "",
        "Files:",
        "  fig4_labour_pair_period.csv",
        "  fig4_labour_paired_diffs.csv",
        "  fig4_labour_hw_change_summary.csv",
        "",
        "pair-period level:",
        "  one row = pair_id x period_norm",
        "  period_norm = HW or NHW",
        "",
        "paired-difference level:",
        "  one row = pair_id",
        "  all R_* / d* variables are HW minus NHW unless explicitly stated otherwise",
        "",
        "Core definitions:",
        "  day_heat_exposure = sum over 10:00-17:00 of",
        "      max(T_urban - theta_day_temp, 0) - max(T_rural - theta_day_temp, 0)",
        "      unit: degC h day-1",
        "",
        "  work_heat_exposure = sum over 08:00-19:00 of",
        "      max(T_urban - theta_day_temp, 0) - max(T_rural - theta_day_temp, 0)",
        "      unit: degC h day-1",
        "",
        "  labour_wbgt_work_exposure = sum over 08:00-19:00 of",
        "      max(WBGT_urban - theta_wbgt, 0) - max(WBGT_rural - theta_wbgt, 0)",
        "      unit: degC WBGT h day-1",
        "",
        "  labour_capacity_pct = dlc_dunne_work_mean",
        "      source: dlc_dunne_h00-h23 and dlc_dunne_* columns",
        "      positive means urban labour capacity > rural labour capacity",
        "",
        "  labour_loss_pct = dloss_dunne_work_mean = -labour_capacity_pct",
        "      positive means urban labour loss > rural labour loss",
        "",
        "  dLabourCapacity = labour_capacity_pct_HW - labour_capacity_pct_NHW",
        "      positive means heatwave increases the urban-rural labour-capacity difference",
        "",
        "  dLabourLoss = labour_loss_pct_HW - labour_loss_pct_NHW",
        "      positive means heatwave increases urban-relative labour loss",
        "",
        "  signed work peak:",
        "      labour_capacity_work_peak_signed = max(dlc_dunne_h08-h19)",
        "      labour_capacity_work_trough_signed = min(dlc_dunne_h08-h19)",
        "      labour_capacity_work_peak_abs = max(abs(dlc_dunne_h08-h19))",
        "",
        "Important note:",
        "  dloss_dunne_peak_t_diff is retained only as an older Tmax peak-to-peak diagnostic.",
        "  It is not used as the main Figure 4 labour-capacity endpoint.",
        "",
        "Settings:",
        f"  theta_day_temp = {FIG4_THETA_DAY_TEMP}",
        f"  theta_wbgt = {FIG4_THETA_WBGT}",
        f"  primary_labour_method = {FIG4_PRIMARY_LABOUR_METHOD}",
        f"  day exposure hours = {FIG4_DAY_EXPOSURE_HOURS}",
        f"  work exposure hours = {FIG4_WORK_EXPOSURE_HOURS}",
        "",
    ]

    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return schema_path

def export_figure4_labour_data(
    df,
    output_dir=FIG4_DATA_DIR,
    primary_method=FIG4_PRIMARY_LABOUR_METHOD,
):
    """
    Independent export function for Figure 4 labour pathway data.

    Minimal modification:
    - Existing pair-period and paired-diff files are still written.
    - Adds capacity-direction fields based on dlc_dunne_h00-h23.
    - Adds a group-level HW-change summary for downstream Plot fig4.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "═" * 72)
    print("  [Figure 4] Exporting labour pathway data")
    print(f"  Output directory: {output_dir}")
    print("  Main labour capacity source: dlc_dunne_h00-h23 / dlc_dunne_*")
    print("  Labour loss retained as sign-reversed mirror")
    print("═" * 72)

    pair_period = _fig4_build_labour_pair_period_table(
        df,
        primary_method=primary_method,
        theta_day_temp=FIG4_THETA_DAY_TEMP,
        theta_wbgt=FIG4_THETA_WBGT,
    )

    pair_path = os.path.join(output_dir, "fig4_labour_pair_period.csv")
    pair_period.to_csv(pair_path, index=False)
    print(f"  Saved: {pair_path}")
    print(f"    rows = {len(pair_period)}")
    print(f"    HW rows = {(pair_period['period_norm'] == 'HW').sum()}")
    print(f"    NHW rows = {(pair_period['period_norm'] == 'NHW').sum()}")

    paired_diffs = _fig4_make_labour_paired_diffs(pair_period)

    diff_path = os.path.join(output_dir, "fig4_labour_paired_diffs.csv")
    paired_diffs.to_csv(diff_path, index=False)
    print(f"  Saved: {diff_path}")
    print(f"    paired rows = {len(paired_diffs)}")
    print(f"    valid_fig4_labour_capacity = {paired_diffs['valid_fig4_labour_capacity'].sum()}")
    print(f"    valid_fig4_labour_loss = {paired_diffs['valid_fig4_labour_loss'].sum()}")

    summary_df = _fig4_make_labour_hw_change_summary(paired_diffs)
    summary_path = os.path.join(output_dir, "fig4_labour_hw_change_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    print(f"    summary rows = {len(summary_df)}")

    schema_path = _fig4_write_labour_schema(output_dir)
    print(f"  Saved: {schema_path}")

    # Console diagnostics.
    if "dLabourCapacity" in paired_diffs.columns:
        vals = paired_diffs["dLabourCapacity"].dropna()
        if len(vals) > 0:
            print(
                "  dLabourCapacity summary "
                "(HW - NHW; positive = larger urban-rural labour capacity difference): "
                f"mean={vals.mean():+.3f}%, "
                f"median={vals.median():+.3f}%, "
                f"n={len(vals)}"
            )

    if "dLabourLoss" in paired_diffs.columns:
        vals = paired_diffs["dLabourLoss"].dropna()
        if len(vals) > 0:
            print(
                "  dLabourLoss summary "
                "(HW - NHW; positive = larger urban-relative extra loss): "
                f"mean={vals.mean():+.3f}%, "
                f"median={vals.median():+.3f}%, "
                f"n={len(vals)}"
            )

    if "capacity_loss_sign_check" in paired_diffs.columns:
        chk = paired_diffs["capacity_loss_sign_check"].dropna()
        if len(chk) > 0:
            print(
                "  capacity/loss sign check "
                "(R_loss + R_capacity should be near 0): "
                f"max_abs={np.nanmax(np.abs(chk)):.6g}, "
                f"mean={np.nanmean(chk):+.6g}"
            )

    if "dLabourWBGTWorkExposure" in paired_diffs.columns:
        vals = paired_diffs["dLabourWBGTWorkExposure"].dropna()
        if len(vals) > 0:
            print(
                "  dLabourWBGTWorkExposure summary "
                "(HW - NHW): "
                f"mean={vals.mean():+.3f}, "
                f"median={vals.median():+.3f}, "
                f"n={len(vals)}"
            )

    print("  [Figure 4] Labour export complete.\n")

    return {
        "pair_period_path": pair_path,
        "paired_diffs_path": diff_path,
        "summary_path": summary_path,
        "schema_path": schema_path,
        "pair_period": pair_period,
        "paired_diffs": paired_diffs,
        "summary": summary_df,
    }

# ══════════════════════════════════════════════════════════════
# ★ 新增独立模块：导出图 1、2、3 所需的真实 Labour Loss 数据
# ══════════════════════════════════════════════════════════════
def export_fig123_data_labour(df, out_dir):

    fig_data_dir = os.path.join(out_dir, "integrated_fig_data")
    os.makedirs(fig_data_dir, exist_ok=True)

    sub = df[df["period"].isin(["heatwave", "non_heatwave"])].copy()

    # === 图 1：日循环真实数据：Delta Labour Capacity + CI ===
    # 与 plot_fig1_diurnal_dlc() 保持一致：
    # dLC = LC_urban - LC_rural
    # dLC > 0 表示城市劳动能力更高；dLC < 0 表示城市劳动能力更低
    fig1_rows = []

    for (grp, prd), g in sub.groupby(["group", "period"]):
        for h in range(24):
            col = f"dlc_dunne_h{h:02d}"
            if col not in g.columns:
                continue

            vals = g[col].dropna().astype(float).values
            if len(vals) == 0:
                continue

            mean_val = float(np.mean(vals))
            se_val = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan

            # 和 plot_fig1_diurnal_dlc() 一样用 bootstrap CI
            _, ci_lo, ci_hi = bootstrap_ci(vals, n_boot=500)

            fig1_rows.append({
                "group": grp,
                "period": "HW" if prd == "heatwave" else "NHW",
                "hour": h,
                "n": int(len(vals)),
                "d_labour_capacity_pct_mean": mean_val,
                "d_labour_capacity_pct_se": se_val,
                "d_labour_capacity_pct_ci_lo": ci_lo,
                "d_labour_capacity_pct_ci_hi": ci_hi,
            })

    pd.DataFrame(fig1_rows).to_csv(
        os.path.join(fig_data_dir, "fig1_diurnal_labour_real.csv"),
        index=False
    )

    # === 图 2：脆弱性差距真实数据 (日间绝对值提取) ===
    # 原始脚本里没有存储绝对值的 mean，所以我们在这里提取逐小时 WBGT 从头算一遍绝对 Loss
    DAY_HOURS = list(range(8, 20))
    fig2_rows = []
    for (grp, prd), g in sub.groupby(["group", "period"]):
        u_loss_list = []
        r_loss_list = []
        for _, row in g.iterrows():
            u_wbgt = [row.get(f"wbgt_urban_h{h:02d}", np.nan) for h in DAY_HOURS]
            r_wbgt = [row.get(f"wbgt_rural_h{h:02d}", np.nan) for h in DAY_HOURS]
            
            # 使用本文件顶部的 lc_dunne 函数
            u_lc = np.nanmean(lc_dunne(u_wbgt))
            r_lc = np.nanmean(lc_dunne(r_wbgt))
            
            # 损失 = 100 - 劳动能力
            u_loss_list.append(100.0 - u_lc)
            r_loss_list.append(100.0 - r_lc)

        fig2_rows.append({
            "group": grp, "period": "HW" if prd == "heatwave" else "NHW",
            "labour_loss_pct_U": np.nanmean(u_loss_list),
            "labour_loss_pct_R": np.nanmean(r_loss_list)
        })
    pd.DataFrame(fig2_rows).to_csv(os.path.join(fig_data_dir, "fig2_vulnerability_labour_real.csv"), index=False)

    # === 图 3：机制散点真实数据 (配对差值) ===
    # dloss = -dLC
    # dloss > 0 表示 Urban 比 Rural 劳动能力损失更多
    fig3_cols = [
        "pair_id",
        "group",
        "period",
        "dwbgt_day_mean",
        "dwbgt_night_mean",
        "dloss_dunne_day_mean",
        "dloss_dunne_day_peak",
    ]

    fig3 = sub[[c for c in fig3_cols if c in sub.columns]].copy()

    fig3 = fig3.rename(columns={
        "dloss_dunne_day_mean": "d_labour_loss_day_mean",
        "dloss_dunne_day_peak": "d_labour_loss_day_peak",
    })

    fig3["period"] = fig3["period"].replace({
        "heatwave": "HW",
        "non_heatwave": "NHW"
    })

    fig3.to_csv(
        os.path.join(fig_data_dir, "fig3_mechanism_labour_real.csv"),
        index=False
    )


    print(f"  [Fig1-3] Labour loss real data exported to: {fig_data_dir}")


# ─────────────────────────────────────────────────────────────
# 16. Main
# ─────────────────────────────────────────────────────────────
def main():
    sep = "=" * 72
    print(sep)
    # [v3→v5]
    print("  Labour Loss Diurnal Asymmetry Analysis  v3 (+KG climate zone)")
    print("  Methods: Dunne (2013), He (2022)")
    print("  Periods: Annual | Summer/JJA | Heatwave | Non-heatwave")
    print("  Climate: Koeppen-Geiger (primary) + lat_group (compat.)")
    print(sep)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # [1] Load
    print(f"\n[1] Loading: {INPUT_CSV}")

    df_raw = pd.read_csv(INPUT_CSV)

    required_common_time_cols = [
        *[f"urban_wbgt_temp_fft_h{h:02d}" for h in range(24)],
        *[f"rural_wbgt_temp_fft_h{h:02d}" for h in range(24)],
        *[f"urban_dew_fft_h{h:02d}" for h in range(24)],
        *[f"rural_dew_fft_h{h:02d}" for h in range(24)],
        "temp_dew_common_time_frac_urban",
        "temp_dew_common_time_frac_rural",
        "temp_dew_common_fft_available_urban",
        "temp_dew_common_fft_available_rural",
    ]

    missing_common_time_cols = [
        col for col in required_common_time_cols
        if col not in df_raw.columns
    ]

    if missing_common_time_cols:
        raise ValueError(
            "The labour script requires common-time T/Td two-harmonic "
            "FFT fields from 01_main_pair_period_metrics.py. "
            f"Missing columns: {missing_common_time_cols[:20]}"
        )

    df_raw["pair_id"] = df_raw["pair_id"].astype(str)

    print(f"    Shape   : {df_raw.shape}")
    print(f"    Periods : {df_raw['period'].value_counts().to_dict()}")
    print(f"    Groups  : {df_raw['group'].value_counts().to_dict()}")

    # [v3→v5] KG 分布统计
    if "kg_group" in df_raw.columns:
        annual_raw = df_raw[df_raw["period"] == "annual"]
        print("    KG group distribution (annual):")
        for kg, cnt in annual_raw["kg_group"].value_counts(dropna=False).items():
            label = KG_GROUP_MAP.get(str(kg), str(kg))
            print(f"      {kg} ({label}): {cnt}")

    # [2] Compute WBGT + LC
    print("\n[2] Computing WBGT + Labour Capacity ...")
    lc_records = [compute_row_lc(row) for _, row in df_raw.iterrows()]
    df = pd.concat([df_raw.reset_index(drop=True), pd.DataFrame(lc_records)], axis=1)

    if "has_dewpoint" in df.columns:
        cov = df["has_dewpoint"].mean()

        print(
            "    Rows with common-time T/Td coverage >=80% and successful "
            "two-harmonic FFT at both stations: "
            f"{cov:.1%}"
        )

    print("    Period counts after LC calc:")
    for p in ["annual", "warm_season", "heatwave", "non_heatwave"]:
        print(f"      {p:<15s}: {(df['period'] == p).sum()}")

    df.to_csv(os.path.join(OUTPUT_DIR, "labour_loss_full.csv"), index=False)
    print("    Saved: labour_loss_full.csv")
    save_integrated_figure_input(df, OUTPUT_DIR)
    
    export_fig123_data_labour(df, OUTPUT_DIR)

    export_figure4_labour_data(
        df,
        output_dir=FIG4_DATA_DIR,
        primary_method=FIG4_PRIMARY_LABOUR_METHOD,
    )
    
    # [3] Summary tables
    print("\n[3] Building summary tables ...")
    key_metrics = []
    for m in ["dunne", "high", "medium", "low"]:
        key_metrics += [
            f"dlc_{m}_day_mean", f"dlc_{m}_night_mean",
            f"dlc_{m}_work_mean", f"net_effect_{m}",
            f"asymmetry_index_{m}", f"day_night_ratio_{m}",
        ]
    key_metrics += ["dwbgt_day_mean", "dwbgt_night_mean"]
    key_metrics = [m for m in key_metrics if m in df.columns]

    build_summary(df, ["group", "period"], key_metrics).to_csv(
        os.path.join(OUTPUT_DIR, "summary_group_period.csv"), index=False)
    print("    Saved: summary_group_period.csv")

    if "lat_group" in df.columns:
        build_summary(df, ["group", "lat_group", "period"], key_metrics).to_csv(
            os.path.join(OUTPUT_DIR, "summary_latband.csv"), index=False)
        print("    Saved: summary_latband.csv  [兼容性保留]")

    if "continent" in df.columns:
        build_summary(
            df[df["period"].isin(["annual", "warm_season", "heatwave"])],
            ["group", "continent", "period"], key_metrics
        ).to_csv(os.path.join(OUTPUT_DIR, "summary_continent.csv"), index=False)
        print("    Saved: summary_continent.csv")

    # ── [v3→v5] KG 主分组汇总 ──────────────────────────────────
    if "kg_group" in df.columns:
        annual_only = df[df["period"] == "annual"].copy()
        if len(annual_only) > 0:
            build_summary(
                annual_only, ["group", "kg_group"], key_metrics
            ).to_csv(os.path.join(OUTPUT_DIR, "summary_kggroup.csv"), index=False)
            print("    Saved: summary_kggroup.csv  [v3 NEW -- KG 主分组 A/B/C/D/E]")

    if "kg_code" in df.columns:
        annual_only = df[df["period"] == "annual"].copy()
        if len(annual_only) > 0:
            # 仅输出样本量 >= 5 的细码
            kg_code_counts = annual_only["kg_code"].value_counts()
            valid_codes    = kg_code_counts[kg_code_counts >= 5].index.tolist()
            annual_valid   = annual_only[annual_only["kg_code"].isin(valid_codes)]
            if len(annual_valid) > 0:
                build_summary(
                    annual_valid, ["group", "kg_code"], key_metrics
                ).to_csv(os.path.join(OUTPUT_DIR, "summary_kgcode.csv"), index=False)
                print("    Saved: summary_kgcode.csv  [v3 NEW -- KG 细码 (n>=5)]")

    # ── [v3→v5] KG × period 三时期汇总 ────────────────────────
    if "kg_group" in df.columns:
        sub_3p = df[df["period"].isin(["annual", "warm_season", "heatwave"])].copy()
        if len(sub_3p) > 0:
            build_summary(
                sub_3p, ["group", "kg_group", "period"], key_metrics
            ).to_csv(os.path.join(OUTPUT_DIR, "summary_kggroup_period.csv"), index=False)
            print("    Saved: summary_kggroup_period.csv  [v3 NEW -- KG x period]")

    if "asymmetry_class_dunne" in df.columns:
        for period in ["annual", "warm_season", "heatwave"]:
            sub_p = df[df["period"] == period]
            if len(sub_p) == 0: continue
            asym = sub_p.groupby(
                ["group", "asymmetry_class_dunne"]
            ).size().reset_index(name="n")
            asym["pct"]    = asym["n"] / asym.groupby("group")["n"].transform("sum") * 100
            asym["period"] = period
            asym.to_csv(
                os.path.join(OUTPUT_DIR, f"asymmetry_classification_{period}.csv"),
                index=False
            )
            print(f"    Saved: asymmetry_classification_{period}.csv")

    mwu_rows = []
    for group in ["UHI", "UCI"]:
        for m in ["dunne", "high", "medium", "low"]:
            col = f"net_effect_{m}"
            if col not in df.columns: continue
            hw  = df[(df["period"] == "heatwave")     & (df["group"] == group)][col].dropna()
            nhw = df[(df["period"] == "non_heatwave") & (df["group"] == group)][col].dropna()
            ws  = df[(df["period"] == "warm_season")  & (df["group"] == group)][col].dropna()
            for A, B, tag in [
                (hw, nhw, "hw_vs_nhw"),
                (hw, ws,  "hw_vs_warm_season"),
            ]:
                if len(A) >= 5 and len(B) >= 5:
                    _, pval = stats.mannwhitneyu(A, B, alternative="two-sided")
                    mwu_rows.append({
                        "group": group, "method": m, "comparison": tag,
                        "n_A": len(A), "mean_A": A.mean(),
                        "n_B": len(B), "mean_B": B.mean(),
                        "mean_diff": A.mean() - B.mean(),
                        "p_mwu": float(pval), "sig": sig_stars(pval),
                    })
    if mwu_rows:
        pd.DataFrame(mwu_rows).to_csv(
            os.path.join(OUTPUT_DIR, "hw_nonhw_mwu_test.csv"), index=False)
        print("    Saved: hw_nonhw_mwu_test.csv")

    save_period_comparison_stats(df, os.path.join(OUTPUT_DIR, "period_comparison_stats.csv"))

    # [4] Figures
    print("\n[4] Generating figures ...")
    for method in ["dunne", "high", "medium", "low"]:
        plot_fig1_diurnal_dlc(
            df, method, os.path.join(fig_dir, f"fig1_diurnal_dlc_{method}.png"))
        plot_fig2_scatter(
            df, method, os.path.join(fig_dir, f"fig2_scatter_daynight_{method}.png"))
        plot_fig3_net_violin(
            df, method, os.path.join(fig_dir, f"fig3_violin_net_{method}.png"))
        plot_fig6_period_asymmetry(
            df, method, os.path.join(fig_dir, f"fig6_period_asymmetry_{method}.png"))
        plot_fig7_asymmetry_heatmap(
            df, method, os.path.join(fig_dir, f"fig7_asymmetry_heatmap_{method}.png"))
        # [v3→v5] 新增 Fig8
        plot_fig8_kg_bar(
            df, method, os.path.join(fig_dir, f"fig8_kg_bar_{method}.png"))

    plot_fig4_wbgt_diurnal(df,   os.path.join(fig_dir, "fig4_wbgt_diurnal.png"))
    plot_fig5_method_compare(df, os.path.join(fig_dir, "fig5_method_compare.png"))

    # [5] Results summary
    print("\n[5] Generating results summary ...")
    save_results_summary(df, OUTPUT_DIR)

    # [6] Console summary
    print(f"\n{'─'*72}")
    print("  Quick Summary  (Dunne 2013 Heavy Labour)")
    print(f"{'─'*72}")
    for period in ["annual", "warm_season", "heatwave"]:
        sub_p = df[df["period"] == period]
        if len(sub_p) == 0: continue
        print(f"\n  -- Period: {PERIOD_LABEL.get(period, period)} --")
        for group in ["UHI", "UCI"]:
            sub = sub_p[sub_p["group"] == group]
            if len(sub) == 0: continue
            dm,  dlo, dhi  = bootstrap_ci(sub.get("dlc_dunne_day_mean",      pd.Series(dtype=float)))
            nm,  nlo, nhi  = bootstrap_ci(sub.get("dlc_dunne_night_mean",    pd.Series(dtype=float)))
            ne,  *_        = bootstrap_ci(sub.get("net_effect_dunne",        pd.Series(dtype=float)))
            ai,  ailo, aihi = bootstrap_ci(sub.get("asymmetry_index_dunne", pd.Series(dtype=float)))
            wm,  wlo, whi  = bootstrap_ci(sub.get("dwbgt_day_mean",          pd.Series(dtype=float)))
            print(f"\n  [{group}]  n={len(sub)}")
            print(f"    Daytime  dLC       : {dm:+.3f}%  [{dlo:+.3f}, {dhi:+.3f}]")
            print(f"    Nighttime dLC      : {nm:+.3f}%  [{nlo:+.3f}, {nhi:+.3f}]")
            print(f"    Net effect         : {ne:+.3f}%")
            print(f"    Asymmetry Index(AI): {ai:+.3f}%  [{ailo:+.3f}, {aihi:+.3f}]")
            print(f"    Daytime  dWBGT     : {wm:+.3f} degC  [{wlo:+.3f}, {whi:+.3f}]")
            if "asymmetry_class_dunne" in sub.columns:
                cnt = sub["asymmetry_class_dunne"].value_counts().to_dict()
                total = sum(cnt.values())
                print("    Asymmetry class:")
                for cls, n in sorted(cnt.items(), key=lambda x: -x[1]):
                    print(f"      {cls:<28s}: {n:>4d}  ({n/total*100:.1f}%)")

    # [v3→v5] KG 快报
    if "kg_group" in df.columns:
        print(f"\n{'─'*72}")
        print("  KG Climate Zone Quick Summary  (Annual, Dunne, dLC_day mean)")
        print(f"{'─'*72}")
        ann = df[df["period"] == "annual"]
        kg_present = sorted(
            ann["kg_group"].dropna().unique(),
            key=lambda k: KG_GROUPS_ALL.index(k) if k in KG_GROUPS_ALL else 99
        )
        for kg in kg_present:
            climate = KG_GROUP_MAP.get(str(kg), kg)
            for group in ["UHI", "UCI"]:
                sub_kg = ann[(ann["kg_group"] == kg) & (ann["group"] == group)]
                if len(sub_kg) < 3: continue
                dm, dlo, dhi = bootstrap_ci(sub_kg.get(
                    "dlc_dunne_day_mean", pd.Series(dtype=float)))
                nm, nlo, nhi = bootstrap_ci(sub_kg.get(
                    "dlc_dunne_night_mean", pd.Series(dtype=float)))
                print(
                    f"  {kg} ({climate:<10s}) [{group}] n={len(sub_kg):>4d}  "
                    f"dLC_day={dm:+.3f}% [{dlo:+.3f},{dhi:+.3f}]  "
                    f"dLC_night={nm:+.3f}% [{nlo:+.3f},{nhi:+.3f}]"
                )

    print(f"\n  All outputs --> {OUTPUT_DIR}")
    # [v3→v5] 更新目录树
    print(
        f"\n  {OUTPUT_DIR}/\n"
        "  |-- labour_loss_full.csv\n"
        "  |-- summary_group_period.csv\n"
        "  |-- summary_latband.csv          (兼容性保留)\n"
        "  |-- summary_continent.csv\n"
        "  |-- summary_kggroup.csv          [v3 NEW] KG 主分组 A/B/C/D/E x group\n"
        "  |-- summary_kgcode.csv           [v3 NEW] KG 细码 (n>=5) x group\n"
        "  |-- summary_kggroup_period.csv   [v3 NEW] KG x period x group\n"
        "  |-- asymmetry_classification_{period}.csv\n"
        "  |-- hw_nonhw_mwu_test.csv\n"
        "  |-- period_comparison_stats.csv\n"
        "  |-- results_summary.txt          (含 Part 8: KG 分层)\n"
        "  |-- figures/\n"
        "      |-- fig1_diurnal_dlc_{method}.png\n"
        "      |-- fig2_scatter_daynight_{method}.png\n"
        "      |-- fig3_violin_net_{method}.png\n"
        "      |-- fig4_wbgt_diurnal.png\n"
        "      |-- fig5_method_compare.png\n"
        "      |-- fig6_period_asymmetry_{method}.png\n"
        "      |-- fig7_asymmetry_heatmap_{method}.png\n"
        "      |-- fig8_kg_bar_{method}.png [v3 NEW] KG 分组条形图\n"
    )
    print("  Done.\n")


if __name__ == "__main__":
    main()
