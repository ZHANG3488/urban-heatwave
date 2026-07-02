#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sensitivity_analysis_v7_nature.py

Standalone Nature-style plotting script for outputs from sensitivity_analysis_v7.py.

Default input:
  <UNIFIED_ROOT>/analysis/sensitivity

Default output:
  <input>/figures_nature

The script does NOT recompute heatwaves, ERA5-ISD correction, FFT, or UHI/UCI.
It only reads the CSV files already produced by the sensitivity-analysis script.
All figure labels are in English.
"""

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import (
    SENSITIVITY_FIGURE_OUTPUT_DIR,
    SENSITIVITY_OUTPUT_DIR,
)

warnings.filterwarnings("ignore")

INPUT_DEFAULT = SENSITIVITY_OUTPUT_DIR

MM = 1 / 25.4
W_SINGLE = 89 * MM
W_DOUBLE = 183 * MM

DQ_ORDER = [0.70, 0.80, 0.90]
HW_ORDER = [
    "percentile_P85",
    "percentile_P90",
    "percentile_P95",
    "absolute_35C",
]
HW_LABEL = {
    "percentile_P85": "P85",
    "percentile_P90": "P90",
    "percentile_P95": "P95",
    "absolute_35C": "35 °C",
}
GROUP_ORDER = ["UHI", "UCI"]
PERIOD_LABEL = {
    "annual": "Annual",
    "warm_season": "Warm season",
    "heatwave": "HW",
    "non_heatwave": "NHW",
}

COLOR_UHI = "#d62728"
COLOR_UCI = "#1f77b4"

COL = {
    "UHI": COLOR_UHI,
    "UCI": COLOR_UCI,

    # Urban/Rural curves follow the same red/blue convention.
    "Urban": COLOR_UHI,
    "Rural": COLOR_UCI,

    # Other scenario dimensions use line style differences.
    "HW": COLOR_UHI,
    "NHW": COLOR_UCI,
    "P85": "#4d4d4d",
    "P90": "#4d4d4d",
    "P95": "#4d4d4d",
    "35 °C": "#4d4d4d",
    "70%": "#4d4d4d",
    "80%": "#4d4d4d",
    "90%": "#4d4d4d",
    "zero": "#333333",
}

LINESTYLE_DQ = {
    "70%": (0, (1.2, 1.2)),
    "80%": "-",
    "90%": "--",
}

LINESTYLE_HW = {
    "P85": (0, (1.2, 1.2)),
    "P90": "-",
    "P95": "--",
    "35 °C": "-.",
}

LINESTYLE_YEAR_WINDOW = {
    1: (0, (1.2, 1.2)),
    3: "--",
    5: "-.",
    10: "-",
}

METRIC_LABEL = {
    "dAmp1": r"$\Delta$Amp$_1$ (°C)",
    "dT_day_mean": r"Daytime $\Delta T$ (°C)",
    "dT_night_mean": r"Night-time $\Delta T$ (°C)",
    "dTmean": r"$\Delta T_{\mathrm{mean}}$ (°C)",
    "dTx": r"$\Delta T_{\max,\mathrm{FFT}}$ (°C)",
    "dTn": r"$\Delta T_{\min,\mathrm{FFT}}$ (°C)",
    "phase1_peak_hour_diff": "Phase-1 peak-hour shift (h)",
    "dPhase1_hours": r"$\Delta$ phase$_1$ (h)",
    "delta_tropical_night_freq": r"$\Delta$ tropical-night frequency",
    "delta_hotnight_excess": r"$\Delta$ hot-night excess (°C)",
}

SCEN_COLS = [
    "scenario_id",
    "sensitivity_axis",

    "analysis_years",
    "isd_n_years",
    "isd_years_label",
    "isd_start_year",
    "isd_end_year",

    "data_quality_min_year_valid_frac",
    "hw_def",
    "hw_method",
    "hw_percentile",
    "hw_abs_mode",
    "hw_abs_threshold",
]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.unicode_minus": False,
    })


def read_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def read_optional(path):
    if not path.exists():
        print(f"[Warning] Optional file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def clean(df):
    for c in ["data_quality_min_year_valid_frac", "warm_min_year_valid_frac",
              "full_year_min_valid_frac", "hw_percentile", "hw_abs_threshold",
              "isd_n_years", "isd_start_year", "isd_end_year",
              "hour", "n_hw_days_warm", "n_nhw_days_warm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def hw_lab(x):
    return HW_LABEL.get(str(x), str(x))


def dq_lab(x):
    return f"{int(round(float(x) * 100))}%" if pd.notna(x) else "NA"


def hw_sort(x):
    try:
        return HW_ORDER.index(str(x))
    except Exception:
        return 999


def scenario_sort(df):
    out = df.copy()
    out["_dq"] = out["data_quality_min_year_valid_frac"].round(3) if "data_quality_min_year_valid_frac" in out else 999
    out["_hw"] = out["hw_def"].map(hw_sort) if "hw_def" in out else 999
    out = out.sort_values(["_dq", "_hw", "scenario_id"] if "scenario_id" in out else ["_dq", "_hw"])
    return out.drop(columns=["_dq", "_hw"], errors="ignore")


def sub_q(df, q):
    return df[np.isclose(df["data_quality_min_year_valid_frac"], q, atol=1e-6)].copy()


def sub_hw(df, h):
    return df[df["hw_def"].astype(str) == h].copy()


def sub_period(df, p):
    return df[df["period"].astype(str) == p].copy()


def sub_axis(df, axis_name):
    if df.empty:
        return df.copy()
    if "sensitivity_axis" not in df.columns:
        if axis_name == "data_quality_hw_definition":
            return df.copy()
        return df.iloc[0:0].copy()
    axis = df["sensitivity_axis"].fillna("data_quality_hw_definition").astype(str)
    return df[axis == axis_name].copy()


def sub_year_window(df):
    return sub_axis(df, "isd_year_window")

def despine(ax):
    """
    Nature/Supplement style boxed axis:
    keep top and right spines as a black frame, but do not show ticks there.
    """
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.6)

    ax.tick_params(
        direction="out",
        top=False,
        right=False
    )
    ax.grid(False)


def zero(ax):
    ax.axhline(0, color=COL["zero"], lw=0.6, ls="--", zorder=0)


def panel_labels(axes):
    for ax, lab in zip(np.ravel(axes), list("abcdefghijklmnopqrstuvwxyz")):
        if ax.get_visible():
            ax.text(-0.15, 1.08, lab, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="bold")


def save(fig, out, stem, dpi):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        p = out / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        else:
            fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def boot(values, n=1000, seed=42):
    a = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(a) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(a) == 1:
        return float(a[0]), float(a[0]), float(a[0]), 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    b = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(a)


def summary(df, keys, value, nboot=1000, seed=42):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        if not isinstance(k, tuple):
            k = (k,)
        m, lo, hi, n = boot(g[value], n=nboot, seed=seed)
        vals = pd.to_numeric(g[value], errors="coerce").dropna()
        rows.append({**dict(zip(keys, k)), "metric": value, "n": n,
                     "mean": m, "ci_low": lo, "ci_high": hi,
                     "median": vals.median() if len(vals) else np.nan,
                     "q25": vals.quantile(0.25) if len(vals) else np.nan,
                     "q75": vals.quantile(0.75) if len(vals) else np.nan,
                     "se": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan})
    return pd.DataFrame(rows)


def curve_summary(df, keys, vals):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        if not isinstance(k, tuple):
            k = (k,)
        row = dict(zip(keys, k))
        row["n_pairs"] = g["pair_id"].nunique() if "pair_id" in g else len(g)
        for v in vals:
            x = pd.to_numeric(g[v], errors="coerce").dropna()
            row[f"{v}_mean"] = x.mean() if len(x) else np.nan
            row[f"{v}_se"] = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
            row[f"{v}_q25"] = x.quantile(0.25) if len(x) else np.nan
            row[f"{v}_q75"] = x.quantile(0.75) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def uhi_uci_distribution(all_df):
    annual = sub_period(all_df, "annual")
    keep = [c for c in SCEN_COLS + ["pair_id", "group"] if c in annual.columns]
    annual = annual[keep].drop_duplicates()
    keys = [c for c in SCEN_COLS if c in annual.columns]
    rows = []
    for k, g in annual.groupby(keys, dropna=False):
        if not isinstance(k, tuple):
            k = (k,)
        base = dict(zip(keys, k))
        total = g["pair_id"].nunique()
        for grp in GROUP_ORDER:
            n = g.loc[g["group"] == grp, "pair_id"].nunique()
            rows.append({**base, "group": grp, "n_pairs": int(n),
                         "total_pairs": int(total),
                         "proportion": n / total if total else np.nan})
    return scenario_sort(pd.DataFrame(rows))


def pairwise_hw_nhw_diff(all_df, metrics):
    metrics = [m for m in metrics if m in all_df.columns]
    sub = all_df[all_df["period"].isin(["heatwave", "non_heatwave"])].copy()
    keys = [c for c in SCEN_COLS + ["pair_id", "group", "continent", "kg_group"]
            if c in sub.columns]
    rows = []
    for k, g in sub.groupby(keys, dropna=False):
        if not isinstance(k, tuple):
            k = (k,)
        hw = g[g["period"] == "heatwave"]
        nhw = g[g["period"] == "non_heatwave"]
        if hw.empty or nhw.empty:
            continue
        row = dict(zip(keys, k))
        for m in metrics:
            hv = pd.to_numeric(hw[m], errors="coerce").dropna()
            nv = pd.to_numeric(nhw[m], errors="coerce").dropna()
            hv = float(hv.iloc[0]) if len(hv) else np.nan
            nv = float(nv.iloc[0]) if len(nv) else np.nan
            row[f"{m}_heatwave"] = hv
            row[f"{m}_non_heatwave"] = nv
            row[f"{m}_hw_minus_nhw"] = hv - nv if pd.notna(hv) and pd.notna(nv) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fig01_distribution(all_df, out, pdata, dpi):
    dist = uhi_uci_distribution(all_df)
    if dist.empty:
        return
    dist.to_csv(pdata / "fig01_uhi_uci_distribution_data.csv", index=False)

    scen = dist[[c for c in ["scenario_id", "data_quality_min_year_valid_frac", "hw_def"] if c in dist]].drop_duplicates()
    scen = scenario_sort(scen).reset_index(drop=True)
    scen["x"] = np.arange(len(scen))
    scen["label"] = scen.apply(lambda r: f"{hw_lab(r.hw_def)}\n{dq_lab(r.data_quality_min_year_valid_frac)}", axis=1)
    d = dist.merge(scen[["scenario_id", "x", "label"]], on="scenario_id", how="left")

    fig, ax = plt.subplots(1, 2, figsize=(W_DOUBLE, 82 * MM), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.35, 1.0]})
    p = d.pivot_table(index=["x", "label"], columns="group", values="n_pairs", aggfunc="sum").fillna(0).reset_index().sort_values("x")
    bottom = np.zeros(len(p))
    for grp in GROUP_ORDER:
        vals = p[grp].values if grp in p.columns else np.zeros(len(p))
        ax[0].bar(p["x"], vals, bottom=bottom, color=COL[grp], edgecolor="white",
                  lw=0.4, width=0.82, label=grp)
        bottom += vals
    ax[0].set_xticks(p["x"])
    ax[0].set_xticklabels(p["label"], rotation=45, ha="right")
    ax[0].set_ylabel("Retained pairs (n)")
    ax[0].set_title("Retained UHI/UCI samples")
    ax[0].legend(frameon=False, ncol=2, loc="best")
    despine(ax[0])

    uci = dist[dist["group"] == "UCI"].copy()
    for h in HW_ORDER:
        g = uci[uci["hw_def"].astype(str) == h].sort_values("data_quality_min_year_valid_frac")
        if g.empty:
            continue
        lab = hw_lab(h)
        ax[1].plot(
            g["data_quality_min_year_valid_frac"] * 100,
            g["proportion"] * 100,
            marker="o",
            color=COLOR_UCI,
            linestyle=LINESTYLE_HW.get(lab, "-"),
            label=lab.replace("\n", " ")
        )
    ax[1].set_xlabel("Minimum warm-season valid-year fraction (%)")
    ax[1].set_ylabel("UCI proportion (%)")
    ax[1].set_title("UCI share across scenarios")
    ax[1].set_xticks([70, 80, 90])
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[1].legend(frameon=False, loc="best")
    despine(ax[1])
    panel_labels(ax)
    save(fig, out, "fig01_uhi_uci_distribution", dpi)


def fig02_data_integrity_metrics(all_df, out, pdata, baseline_hw, dpi, nboot):
    df = sub_hw(sub_period(all_df, "annual"), baseline_hw)
    metrics = [m for m in ["dAmp1", "dT_day_mean", "dT_night_mean", "phase1_peak_hour_diff"] if m in df.columns]
    if not metrics:
        return
    sm = pd.concat([summary(df, ["data_quality_min_year_valid_frac", "group"], m, nboot, 11)
                    for m in metrics], ignore_index=True)
    sm.to_csv(pdata / "fig02_data_integrity_metric_summary.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(W_DOUBLE, 110 * MM), constrained_layout=True)
    ax = np.ravel(ax)
    for i, m in enumerate(metrics):
        a = ax[i]
        for grp in GROUP_ORDER:
            g = sm[(sm["metric"] == m) & (sm["group"] == grp)].sort_values("data_quality_min_year_valid_frac")
            if g.empty:
                continue
            x, y = g["data_quality_min_year_valid_frac"] * 100, g["mean"]
            a.errorbar(x, y, yerr=[y - g["ci_low"], g["ci_high"] - y],
                       marker="o", capsize=2, color=COL[grp], label=grp)
        zero(a)
        a.set_xlabel("Minimum warm-season valid-year fraction (%)")
        a.set_ylabel(METRIC_LABEL.get(m, m))
        a.set_title(METRIC_LABEL.get(m, m).replace(" (°C)", ""))
        a.set_xticks([70, 80, 90])
        despine(a)
    for j in range(len(metrics), len(ax)):
        ax[j].axis("off")
    ax[0].legend(frameon=False, ncol=2, loc="best")
    panel_labels(ax[:len(metrics)])
    save(fig, out, "fig02_data_integrity_annual_metrics", dpi)


def fig03_diurnal_shape(diurnal, out, pdata, baseline_hw, dpi):
    df = sub_hw(sub_period(diurnal, "annual"), baseline_hw)
    if df.empty:
        return
    cs = curve_summary(df, ["data_quality_min_year_valid_frac", "group", "hour"], ["delta_temp"])
    cs.to_csv(pdata / "fig03_data_integrity_diurnal_shape_data.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(W_DOUBLE, 76 * MM), sharex=True, constrained_layout=True)
    for a, grp in zip(ax, GROUP_ORDER):
        for q in DQ_ORDER:
            g = cs[(cs["group"] == grp) & np.isclose(cs["data_quality_min_year_valid_frac"], q)].sort_values("hour")
            if g.empty:
                continue
            lab = dq_lab(q)
            x = g["hour"].values
            y = g["delta_temp_mean"].values
            se = g["delta_temp_se"].fillna(0).values
            group_color = COL[grp]
            a.plot(
                x, y,
                marker="o",
                ms=2.2,
                color=group_color,
                linestyle=LINESTYLE_DQ.get(lab, "-"),
                label=lab
            )
            a.fill_between(x, y - se, y + se, color=group_color, alpha=0.10, lw=0)
        zero(a)
        a.set_title(f"{grp} group")
        a.set_xlabel("Local hour")
        a.set_ylabel("Urban - rural temperature (°C)")
        a.set_xticks([0, 6, 12, 18, 23])
        despine(a)
    ax[0].legend(title="Data integrity", frameon=False, loc="best")
    panel_labels(ax)
    save(fig, out, "fig03_data_integrity_diurnal_shape", dpi)


def fig04_hw_def_metrics(all_df, out, pdata, baseline_q, dpi, nboot):
    metrics = ["dT_night_mean", "dT_day_mean", "dAmp1", "delta_tropical_night_freq"]
    diff = sub_q(pairwise_hw_nhw_diff(all_df, metrics), baseline_q)
    if diff.empty:
        return
    vals = [f"{m}_hw_minus_nhw" for m in metrics if f"{m}_hw_minus_nhw" in diff.columns]
    sm = pd.concat([summary(diff, ["hw_def", "group"], v, nboot, 22) for v in vals], ignore_index=True)
    sm.to_csv(pdata / "fig04_hw_definition_hw_nhw_changes_data.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(W_DOUBLE, 112 * MM), constrained_layout=True)
    ax = np.ravel(ax)
    x = np.arange(len(HW_ORDER))
    w = 0.34
    for a, v in zip(ax, vals):
        base_m = v.replace("_hw_minus_nhw", "")
        for grp, off in [("UHI", -w / 2), ("UCI", w / 2)]:
            y, lo, hi = [], [], []
            g0 = sm[(sm["metric"] == v) & (sm["group"] == grp)]
            for h in HW_ORDER:
                r = g0[g0["hw_def"].astype(str) == h]
                if r.empty:
                    y.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                else:
                    y.append(float(r["mean"].iloc[0])); lo.append(float(r["ci_low"].iloc[0])); hi.append(float(r["ci_high"].iloc[0]))
            y, lo, hi = np.array(y), np.array(lo), np.array(hi)
            a.bar(x + off, y, width=w, color=COL[grp], edgecolor="white", lw=0.4, label=grp)
            a.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="none",
                       ecolor="#333333", elinewidth=0.6, capsize=1.8)
        zero(a)
        a.set_xticks(x)
        a.set_xticklabels([HW_LABEL[h].replace("\n", " ") for h in HW_ORDER], rotation=30, ha="right")
        a.set_ylabel("HW - NHW")
        a.set_title(METRIC_LABEL.get(base_m, base_m))
        despine(a)
    ax[0].legend(frameon=False, ncol=2, loc="best")
    panel_labels(ax)
    save(fig, out, "fig04_hw_definition_hw_nhw_changes", dpi)


def fig05_hw_nhw_urban_rural(diurnal, out, pdata, baseline_q, baseline_hw, dpi):
    df = sub_hw(sub_q(diurnal, baseline_q), baseline_hw)
    df = df[df["period"].isin(["heatwave", "non_heatwave"])]
    if df.empty:
        return
    cs = curve_summary(df, ["period", "group", "hour"], ["urban_temp", "rural_temp", "delta_temp"])
    cs.to_csv(pdata / "fig05_hw_nhw_urban_rural_diurnal_data.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(W_DOUBLE, 116 * MM), sharex=True, constrained_layout=True)
    for i, grp in enumerate(GROUP_ORDER):
        for j, per in enumerate(["heatwave", "non_heatwave"]):
            a = ax[i, j]
            g = cs[(cs["group"] == grp) & (cs["period"] == per)].sort_values("hour")
            if g.empty:
                a.text(0.5, 0.5, "No data", transform=a.transAxes, ha="center", va="center")
                continue
            for site, lab in [("urban_temp", "Urban"), ("rural_temp", "Rural")]:
                x = g["hour"].values
                y = g[f"{site}_mean"].values
                se = g[f"{site}_se"].fillna(0).values
                a.plot(x, y, marker="o", ms=2.2, color=COL[lab], label=lab)
                a.fill_between(x, y - se, y + se, color=COL[lab], alpha=0.13, lw=0)
            a.set_title(f"{grp} group — {PERIOD_LABEL[per]}")
            a.set_xlabel("Local hour")
            a.set_ylabel("Temperature (°C)")
            a.set_xticks([0, 6, 12, 18, 23])
            despine(a)
    ax[0, 0].legend(frameon=False, ncol=2, loc="best")
    panel_labels(ax)
    save(fig, out, "fig05_hw_nhw_urban_rural_diurnal", dpi)


def fig06_hw_def_delta_diurnal(diurnal, out, pdata, baseline_q, dpi):
    df = sub_q(diurnal, baseline_q)
    df = df[df["period"] == "heatwave"]
    if df.empty:
        return
    cs = curve_summary(df, ["hw_def", "group", "hour"], ["delta_temp"])
    cs.to_csv(pdata / "fig06_hw_definition_delta_diurnal_data.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(W_DOUBLE, 76 * MM), sharex=True, constrained_layout=True)
    for a, grp in zip(ax, GROUP_ORDER):
        for h in HW_ORDER:
            g = cs[(cs["group"] == grp) & (cs["hw_def"].astype(str) == h)].sort_values("hour")
            if g.empty:
                continue
            lab = hw_lab(h)
            x = g["hour"].values
            y = g["delta_temp_mean"].values
            se = g["delta_temp_se"].fillna(0).values
            group_color = COL[grp]
            a.plot(
                x, y,
                marker="o",
                ms=2.0,
                color=group_color,
                linestyle=LINESTYLE_HW.get(lab, "-"),
                label=lab.replace("\n", " ")
            )
            a.fill_between(x, y - se, y + se, color=group_color, alpha=0.08, lw=0)
        zero(a)
        a.set_title(f"{grp} group — HW")
        a.set_xlabel("Local hour")
        a.set_ylabel("Urban - rural temperature (°C)")
        a.set_xticks([0, 6, 12, 18, 23])
        despine(a)
    ax[0].legend(frameon=False, loc="best")
    panel_labels(ax)
    save(fig, out, "fig06_hw_definition_delta_diurnal", dpi)


def fig07_isd_year_window_metrics(all_df, out, pdata, dpi, nboot):
    df = sub_year_window(sub_period(all_df, "annual"))
    if df.empty:
        print("[Skip] fig07: no ISD year-window data.")
        return

    metrics = [
        m for m in [
            "dAmp1",
            "dT_day_mean",
            "dT_night_mean",
            "phase1_peak_hour_diff",
        ]
        if m in df.columns
    ]
    if not metrics:
        print("[Skip] fig07: no requested year-window metrics.")
        return

    sm = pd.concat(
        [
            summary(df, ["isd_n_years", "isd_years_label", "group"], m, nboot, 77)
            for m in metrics
        ],
        ignore_index=True
    )
    sm.to_csv(pdata / "fig07_isd_year_window_metric_summary.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(W_DOUBLE, 110 * MM), constrained_layout=True)
    ax = np.ravel(ax)

    for i, m in enumerate(metrics):
        a = ax[i]
        for grp in GROUP_ORDER:
            g = sm[(sm["metric"] == m) & (sm["group"] == grp)].sort_values("isd_n_years")
            if g.empty:
                continue

            x = g["isd_n_years"].astype(float).values
            y = g["mean"].astype(float).values
            lo = g["ci_low"].astype(float).values
            hi = g["ci_high"].astype(float).values

            a.errorbar(
                x,
                y,
                yerr=[y - lo, hi - y],
                marker="o",
                capsize=2,
                color=COL[grp],
                label=grp,
            )

        zero(a)
        a.set_xlabel("Number of ISD years")
        a.set_ylabel(METRIC_LABEL.get(m, m))
        a.set_title(METRIC_LABEL.get(m, m).replace(" (°C)", ""))
        a.set_xticks([1, 3, 5, 10])
        despine(a)

    for j in range(len(metrics), len(ax)):
        ax[j].axis("off")

    ax[0].legend(frameon=False, ncol=2, loc="best")
    panel_labels(ax[:len(metrics)])
    save(fig, out, "fig07_isd_year_window_annual_metrics", dpi)


def fig08_isd_year_window_diurnal_shape(diurnal, out, pdata, dpi):
    df = sub_year_window(sub_period(diurnal, "annual"))
    if df.empty:
        print("[Skip] fig08: no ISD year-window diurnal data.")
        return

    cs = curve_summary(
        df,
        ["isd_n_years", "isd_years_label", "group", "hour"],
        ["delta_temp"]
    )
    cs.to_csv(pdata / "fig08_isd_year_window_diurnal_shape_data.csv", index=False)

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(W_DOUBLE, 76 * MM),
        sharex=True,
        constrained_layout=True
    )

    for a, grp in zip(ax, GROUP_ORDER):
        for n_years in [1, 3, 5, 10]:
            g = cs[
                (cs["group"] == grp)
                & (cs["isd_n_years"].astype(float) == float(n_years))
            ].sort_values("hour")

            if g.empty:
                continue

            x = g["hour"].values
            y = g["delta_temp_mean"].values
            se = g["delta_temp_se"].fillna(0).values

            a.plot(
                x,
                y,
                marker="o",
                ms=2.1,
                color=COL[grp],
                linestyle=LINESTYLE_YEAR_WINDOW.get(n_years, "-"),
                label=f"{n_years} yr",
            )
            a.fill_between(x, y - se, y + se, color=COL[grp], alpha=0.08, lw=0)

        zero(a)
        a.set_title(f"{grp} group")
        a.set_xlabel("Local hour")
        a.set_ylabel("Urban - rural temperature (°C)")
        a.set_xticks([0, 6, 12, 18, 23])
        despine(a)

    ax[0].legend(title="ISD window", frameon=False, loc="best")
    panel_labels(ax)
    save(fig, out, "fig08_isd_year_window_diurnal_shape", dpi)

def fig06_08_combined_diurnal(diurnal_main, diurnal_yearwin, out, pdata, baseline_q, dpi):
    """
    Combined figure:
    Top row: original Fig. 6, HW-definition sensitivity of delta diurnal curves.
    Bottom row: original Fig. 8, ISD year-window sensitivity of delta diurnal curves.

    The original plotting styles, labels, colors, legends, and panel formats are retained.
    """
    df6 = sub_q(diurnal_main, baseline_q)
    df6 = df6[df6["period"] == "heatwave"] if not df6.empty else df6

    df8 = sub_year_window(sub_period(diurnal_yearwin, "annual"))

    if df6.empty and df8.empty:
        print("[Skip] fig06_08: no HW-definition or ISD year-window diurnal data.")
        return

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(W_DOUBLE, 152 * MM),
        sharex=True,
        constrained_layout=True
    )

    # ------------------------------------------------------------------
    # Top row: original Fig. 6
    # ------------------------------------------------------------------
    if not df6.empty:
        cs6 = curve_summary(df6, ["hw_def", "group", "hour"], ["delta_temp"])
        cs6.to_csv(pdata / "fig06_hw_definition_delta_diurnal_data.csv", index=False)

        for a, grp in zip(ax[0, :], GROUP_ORDER):
            for h in HW_ORDER:
                g = cs6[
                    (cs6["group"] == grp)
                    & (cs6["hw_def"].astype(str) == h)
                ].sort_values("hour")

                if g.empty:
                    continue

                lab = hw_lab(h)
                x = g["hour"].values
                y = g["delta_temp_mean"].values
                se = g["delta_temp_se"].fillna(0).values
                group_color = COL[grp]

                a.plot(
                    x,
                    y,
                    marker="o",
                    ms=2.0,
                    color=group_color,
                    linestyle=LINESTYLE_HW.get(lab, "-"),
                    label=lab.replace("\n", " ")
                )
                a.fill_between(x, y - se, y + se, color=group_color, alpha=0.08, lw=0)

            zero(a)
            a.set_title(f"{grp} group — HW")
            a.set_xlabel("Local hour")
            a.set_ylabel("Urban - rural temperature (°C)")
            a.set_xticks([0, 6, 12, 18, 23])
            despine(a)

        ax[0, 0].legend(frameon=False, loc="best")

    else:
        for a, grp in zip(ax[0, :], GROUP_ORDER):
            a.text(0.5, 0.5, "No data", transform=a.transAxes,
                   ha="center", va="center")
            a.set_title(f"{grp} group — HW")
            a.set_xlabel("Local hour")
            a.set_ylabel("Urban - rural temperature (°C)")
            a.set_xticks([0, 6, 12, 18, 23])
            despine(a)

    # ------------------------------------------------------------------
    # Bottom row: original Fig. 8
    # ------------------------------------------------------------------
    if not df8.empty:
        cs8 = curve_summary(
            df8,
            ["isd_n_years", "group", "hour"],
            ["delta_temp"]
        )
        cs8.to_csv(pdata / "fig08_isd_year_window_diurnal_shape_data.csv", index=False)

        for a, grp in zip(ax[1, :], GROUP_ORDER):
            for n_years in [1, 3, 5, 10]:
                g = cs8[
                    (cs8["group"] == grp)
                    & (cs8["isd_n_years"].astype(float) == float(n_years))
                ].sort_values("hour")

                if g.empty:
                    continue

                x = g["hour"].values
                y = g["delta_temp_mean"].values
                se = g["delta_temp_se"].fillna(0).values

                a.plot(
                    x,
                    y,
                    marker="o",
                    ms=2.1,
                    color=COL[grp],
                    linestyle=LINESTYLE_YEAR_WINDOW.get(n_years, "-"),
                    label=f"{n_years} yr",
                )
                a.fill_between(x, y - se, y + se, color=COL[grp], alpha=0.08, lw=0)

            zero(a)
            a.set_title(f"{grp} group")
            a.set_xlabel("Local hour")
            a.set_ylabel("Urban - rural temperature (°C)")
            a.set_xticks([0, 6, 12, 18, 23])
            despine(a)

        ax[1, 0].legend(title="ISD window", frameon=False, loc="best")

    else:
        for a, grp in zip(ax[1, :], GROUP_ORDER):
            a.text(0.5, 0.5, "No data", transform=a.transAxes,
                   ha="center", va="center")
            a.set_title(f"{grp} group")
            a.set_xlabel("Local hour")
            a.set_ylabel("Urban - rural temperature (°C)")
            a.set_xticks([0, 6, 12, 18, 23])
            despine(a)

    panel_labels(ax)
    save(fig, out, "fig06_08_combined_diurnal", dpi)

def figS01_heatmaps(all_df, out, pdata, dpi):
    dist = uhi_uci_distribution(all_df)
    if dist.empty:
        return
    total = (dist.drop_duplicates(["scenario_id", "data_quality_min_year_valid_frac", "hw_def", "total_pairs"])
             .pivot_table(index="data_quality_min_year_valid_frac", columns="hw_def", values="total_pairs", aggfunc="first")
             .reindex(index=DQ_ORDER, columns=HW_ORDER))
    uci = (dist[dist["group"] == "UCI"]
           .pivot_table(index="data_quality_min_year_valid_frac", columns="hw_def", values="proportion", aggfunc="first")
           .reindex(index=DQ_ORDER, columns=HW_ORDER) * 100)
    pd.concat({"total_pairs": total, "uci_percent": uci}).to_csv(pdata / "figS01_sample_retention_heatmap_data.csv")

    fig, ax = plt.subplots(1, 2, figsize=(W_DOUBLE, 82 * MM), constrained_layout=True)
    for a, tab, title, fmt, lab in [(ax[0], total, "Retained pairs", "{:.0f}", "Pairs (n)"),
                                    (ax[1], uci, "UCI proportion", "{:.1f}", "UCI (%)")]:
        arr = tab.values.astype(float)
        im = a.imshow(arr, cmap="Greys", aspect="auto")
        a.set_title(title)
        a.set_xticks(np.arange(len(HW_ORDER)))
        a.set_xticklabels([HW_LABEL[h].replace("\n", " ") for h in HW_ORDER], rotation=30, ha="right")
        a.set_yticks(np.arange(len(DQ_ORDER)))
        a.set_yticklabels([dq_lab(q) for q in DQ_ORDER])
        a.set_xlabel("Heatwave definition")
        a.set_ylabel("Data integrity")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    a.text(j, i, fmt.format(arr[i, j]), ha="center", va="center", fontsize=6)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        cb.ax.set_ylabel(lab)
    panel_labels(ax)
    save(fig, out, "figS01_sample_retention_heatmaps", dpi)


def figS02_thresholds(thr, out, pdata, baseline_q, dpi):
    if thr.empty:
        return
    df = sub_q(thr, baseline_q)
    df = df[df["hw_def"].astype(str).isin(HW_ORDER)].copy()
    cols = [c for c in ["hw_threshold_mean", "n_hw_days_warm", "hw_intensity"] if c in df.columns]
    if df.empty or not cols:
        return
    df.to_csv(pdata / "figS02_threshold_diagnostics_data.csv", index=False)

    fig, ax = plt.subplots(1, len(cols), figsize=(W_DOUBLE, 76 * MM), constrained_layout=True)
    ax = np.array([ax]) if len(cols) == 1 else np.ravel(ax)
    for a, c in zip(ax, cols):
        data = [pd.to_numeric(df[df["hw_def"].astype(str) == h][c], errors="coerce").dropna().values for h in HW_ORDER]
        bp = a.boxplot(data, positions=np.arange(len(HW_ORDER)), widths=0.55, patch_artist=True,
                       showfliers=False, medianprops={"color": "#111111", "lw": 0.8})
        for patch, h in zip(bp["boxes"], HW_ORDER):
            patch.set_facecolor("#4d4d4d")
            patch.set_alpha(0.45)
        rng = np.random.default_rng(123)
        for i, vals in enumerate(data):
            if len(vals):
                a.scatter(rng.normal(i, 0.035, len(vals)), vals, s=3.5, color="#333333", alpha=0.16, lw=0)
        a.set_xticks(np.arange(len(HW_ORDER)))
        a.set_xticklabels([HW_LABEL[h].replace("\n", " ") for h in HW_ORDER], rotation=30, ha="right")
        ylabel = {"hw_threshold_mean": "Threshold (°C)", "n_hw_days_warm": "HW days in warm season (n)",
                  "hw_intensity": "HW intensity (°C)"}.get(c, c)
        a.set_ylabel(ylabel)
        a.set_title(c.replace("_", " "))
        despine(a)
    panel_labels(ax)
    save(fig, out, "figS02_threshold_diagnostics", dpi)


def figS03_damp1_boxes(all_df, out, pdata, baseline_hw, dpi):
    df = sub_hw(sub_period(all_df, "annual"), baseline_hw)
    if df.empty or "dAmp1" not in df.columns:
        return
    df[[c for c in ["data_quality_min_year_valid_frac", "group", "pair_id", "dAmp1",
                    "dT_day_mean", "dT_night_mean", "phase1_peak_hour_diff"] if c in df.columns]].to_csv(
        pdata / "figS03_pair_level_distributions_data.csv", index=False)

    metrics = [m for m in ["dAmp1", "dT_day_mean", "dT_night_mean"] if m in df.columns]
    fig, ax = plt.subplots(1, len(metrics), figsize=(W_DOUBLE, 76 * MM), constrained_layout=True)
    ax = np.array([ax]) if len(metrics) == 1 else np.ravel(ax)
    for a, m in zip(ax, metrics):
        data, pos, colors, labels = [], [], [], []
        p = 0
        for q in DQ_ORDER:
            for grp in GROUP_ORDER:
                vals = pd.to_numeric(df[np.isclose(df["data_quality_min_year_valid_frac"], q) & (df["group"] == grp)][m], errors="coerce").dropna().values
                data.append(vals); pos.append(p); colors.append(COL[grp]); labels.append(f"{dq_lab(q)}\n{grp}")
                p += 1
            p += 0.45
        bp = a.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
                       medianprops={"color": "#111111", "lw": 0.8})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.70)
        zero(a)
        a.set_xticks(pos); a.set_xticklabels(labels)
        a.set_title(METRIC_LABEL.get(m, m))
        a.set_ylabel(METRIC_LABEL.get(m, m))
        despine(a)
    panel_labels(ax)
    save(fig, out, "figS03_pair_level_metric_distributions", dpi)


def figS04_skipped(skipped, out, pdata, dpi):
    if skipped.empty or "fail_step" not in skipped.columns:
        return
    sm = skipped.groupby("fail_step", dropna=False).size().sort_values(ascending=True).tail(10)
    sm.reset_index(name="n_pairs").to_csv(pdata / "figS04_skipped_pair_diagnostics_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(W_SINGLE, 82 * MM), constrained_layout=True)
    ax.barh(np.arange(len(sm)), sm.values, color="#777777", edgecolor="white", lw=0.4)
    ax.set_yticks(np.arange(len(sm)))
    ax.set_yticklabels(sm.index)
    ax.set_xlabel("Dropped pair-scenario records (n)")
    ax.set_title("Main exclusion reasons")
    despine(ax)
    save(fig, out, "figS04_skipped_pair_diagnostics", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=INPUT_DEFAULT)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--baseline-quality", type=float, default=0.80)
    ap.add_argument("--baseline-hw-def", default="percentile_P90")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    style()
    inp = Path(args.input_dir)
    out = (
        Path(args.out_dir)
        if args.out_dir
        else Path(SENSITIVITY_FIGURE_OUTPUT_DIR)
    )
    pdata = out / "plot_data"
    out.mkdir(parents=True, exist_ok=True)
    pdata.mkdir(parents=True, exist_ok=True)

    all_df = clean(read_required(inp / "all_sensitivity_pair_period_metrics.csv"))
    diurnal = clean(read_required(inp / "pair_diurnal_curves_long.csv"))
    thr = clean(read_optional(inp / "station_threshold_summary_by_scenario.csv"))
    skipped = clean(read_optional(inp / "skipped_pairs_log_by_scenario.csv"))

    # Exclude unsupported or deprecated heatwave definitions from all plots.
    valid_hw_defs = set(HW_ORDER)
    if "hw_def" in all_df.columns:
        all_df = all_df[all_df["hw_def"].astype(str).isin(valid_hw_defs)].copy()
    if "hw_def" in diurnal.columns:
        diurnal = diurnal[diurnal["hw_def"].astype(str).isin(valid_hw_defs)].copy()
    if not thr.empty and "hw_def" in thr.columns:
        thr = thr[thr["hw_def"].astype(str).isin(valid_hw_defs)].copy()
    if not skipped.empty and "hw_def" in skipped.columns:
        skipped = skipped[skipped["hw_def"].astype(str).isin(valid_hw_defs)].copy()

    # Keep original sensitivity branch and ISD year-window branch separate.
    main_all_df = sub_axis(all_df, "data_quality_hw_definition")
    main_diurnal = sub_axis(diurnal, "data_quality_hw_definition")
    main_thr = sub_axis(thr, "data_quality_hw_definition") if not thr.empty else thr
    main_skipped = sub_axis(skipped, "data_quality_hw_definition") if not skipped.empty else skipped

    yearwin_all_df = sub_axis(all_df, "isd_year_window")
    yearwin_diurnal = sub_axis(diurnal, "isd_year_window")

    print("=" * 78)
    print("Nature-style sensitivity-analysis figures")
    print(f"Input : {inp}")
    print(f"Output: {out}")
    print(f"Rows  : metrics={len(all_df):,}, diurnal={len(diurnal):,}, thresholds={len(thr):,}")
    print(f"Main branch rows      : metrics={len(main_all_df):,}, diurnal={len(main_diurnal):,}")
    print(f"ISD year-window rows  : metrics={len(yearwin_all_df):,}, diurnal={len(yearwin_diurnal):,}")
    print("=" * 78)

    # Original branch: data integrity × heatwave definition.
    fig01_distribution(main_all_df, out, pdata, args.dpi)
    fig02_data_integrity_metrics(main_all_df, out, pdata, args.baseline_hw_def, args.dpi, args.n_boot)
    fig03_diurnal_shape(main_diurnal, out, pdata, args.baseline_hw_def, args.dpi)
    fig04_hw_def_metrics(main_all_df, out, pdata, args.baseline_quality, args.dpi, args.n_boot)
    fig05_hw_nhw_urban_rural(main_diurnal, out, pdata, args.baseline_quality, args.baseline_hw_def, args.dpi)
    # fig06_hw_def_delta_diurnal(main_diurnal, out, pdata, args.baseline_quality, args.dpi)

    fig06_08_combined_diurnal(
        main_diurnal,
        yearwin_diurnal,
        out,
        pdata,
        args.baseline_quality,
        args.dpi
    )

    figS01_heatmaps(main_all_df, out, pdata, args.dpi)
    figS02_thresholds(main_thr, out, pdata, args.baseline_quality, args.dpi)
    figS03_damp1_boxes(main_all_df, out, pdata, args.baseline_hw_def, args.dpi)
    figS04_skipped(main_skipped, out, pdata, args.dpi)

    # Additional lightweight branch: ISD 1/3/5/10-year window sensitivity.
    fig07_isd_year_window_metrics(yearwin_all_df, out, pdata, args.dpi, args.n_boot)
    # fig08_isd_year_window_diurnal_shape(yearwin_diurnal, out, pdata, args.dpi)

    with open(out / "README_figures.txt", "w", encoding="utf-8") as f:
        f.write(
            "Generated figures for sensitivity_analysis_v7.\n\n"
            "Main figures:\n"
            "fig01_uhi_uci_distribution: UHI/UCI sample distribution and UCI share.\n"
            "fig02_data_integrity_annual_metrics: annual dAmp1/day/night/phase stability.\n"
            "fig03_data_integrity_diurnal_shape: annual urban-rural diurnal-shape stability.\n"
            "fig04_hw_definition_hw_nhw_changes: HW-NHW metric changes by heatwave definition.\n"
            "fig05_hw_nhw_urban_rural_diurnal: HW/NHW urban-rural mean diurnal curves.\n"
            "fig06_hw_definition_delta_diurnal: HW delta curves under different heatwave definitions.\n"
            "fig07_isd_year_window_annual_metrics: annual metrics under 1/3/5/10-year ISD windows.\n"
            "fig08_isd_year_window_diurnal_shape: annual diurnal-shape stability under 1/3/5/10-year ISD windows.\n\n"
            "Supplementary figures:\n"
            "figS01_sample_retention_heatmaps, figS02_threshold_diagnostics, "
            "figS03_pair_level_metric_distributions, figS04_skipped_pair_diagnostics.\n\n"
            "All figures are exported as PDF, SVG, and PNG. Plotting data are saved in plot_data/.\n"
        )

    print("Done.")
    print(f"Figures: {out}")
    print(f"Plot data: {pdata}")


if __name__ == "__main__":
    main()
