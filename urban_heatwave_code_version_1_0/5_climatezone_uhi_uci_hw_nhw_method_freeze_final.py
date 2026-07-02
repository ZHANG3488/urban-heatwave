#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
climatezone_uhi_uci_hw_nhw_standalone.py
========================================================================
Standalone derivative of integrated_climatezone_figure_v4.py.\n\nPatch 2026-05-18:\n- Fix R-map merge error caused by MultiIndex columns after pivot_table.\n- Add diagnostic check for whether NHW equals warm season minus HW days.\n- Add diagnostic check confirming whether CDH boxplot is degree-hours per day.\n
Purpose
-------
1) Output one Nature-style figure containing:
   - Four climate-zone diurnal panels (A-D)
   - Three boxplot panels: Sleep loss, Labour loss, CDH

2) Use only two periods:
   - HW  = heatwave
   - NHW = non_heatwave

3) Plot only stations that have valid data in BOTH HW and NHW.

4) Use UHI/UCI grouping from the 01 main pair-period metrics output:
   - annual percentile UHI/UCI classification from all_pair_period_metrics.csv

5) Output a txt file reporting, for each climate zone, how many UHI and UCI
   stations are plotted.

6) Output one large 3-panel map figure showing the spatial distribution of
   the matched HW-minus-NHW urban-rural response, denoted as R:
       R_X = (X_urban - X_rural)_HW - (X_urban - X_rural)_NHW
   for:
       - Sleep loss
       - Labour loss
       - CDH

Important
---------
The data sources and metric construction follow the original v4 script.
This script changes only filtering, grouping, and figure layout.

Latest plotting update:
- Larger gaps between panels and boxplots.
- Added a separate boxplot legend.
- R maps are vertically stacked and enlarged.
- Every R map now plots (urban-rural)_HW - (urban-rural)_NHW directly.
- Only pairs finite in both HW and NHW are retained for each outcome.
"""

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats  # kept for compatibility with original workflow

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
warnings.filterwarnings("ignore")

from config import UNIFIED_ROOT

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    LONGITUDE_FORMATTER = None
    LATITUDE_FORMATTER = None
    print("Warning: cartopy not installed; maps will use plain lon-lat scatter.")


# ============================================================================
# §0 Paths: same data sources as climate figure v4
# ============================================================================
# CDH / HDH / energy output from compute_cdh_hdh_energy.py
CDH_DIR = UNIFIED_ROOT + "/analysis/cdh_energy"

LABOUR_DIR    = UNIFIED_ROOT + "/analysis/labour"

HNE_DIR       = UNIFIED_ROOT + "/analysis/hne_econ/paired/method_pooled"

FFT_BASE_DIR  = UNIFIED_ROOT + "/analysis/main_multiyear/robustness_percentile"
# 核心数据源：v6 的所有指标总表
V6_METRICS_CSV = os.path.join(FFT_BASE_DIR, "all_pair_period_metrics.csv")

OUTPUT_DIR    = UNIFIED_ROOT + "/plot_data/climatezone"


# ============================================================================
# §1 Globals
# ============================================================================
PERIOD_NORM = {
    "annual": "annual",
    "warm_season": "JJA",
    "JJA": "JJA",
    "heatwave": "HW",
    "HW": "HW",
    "non_heatwave": "NHW",
    "NHW": "NHW"
}
PERIOD_LABEL = {
    "HW": "Heatwave",
    "NHW": "Non-heatwave",
}
PLOT_PERIODS = ["NHW", "HW"]

CLIMATE_ORDER = ["A", "B", "C", "D"]
CLIMATE_NAME = {
    "A": "Tropical",
    "B": "Arid",
    "C": "Temperate",
    "D": "Cold",
}

THERMAL_ORDER = ["UCI", "UHI"]
DIURNAL_COLOR = {
    "urban": "#b2182b",  # 城市 - 红色
    "rural": "#2166ac",  # 郊区 - 蓝色
}

BOX_COLOR = {
    "UHI": "#FFB800",  # 黄色/金橘色
    "UCI": "#A04000",  # 砖红色
}

GROUP_LABEL = {
    "UHI": "UHI",
    "UCI": "UCI",
}
PERIOD_LINESTYLE = {
    "HW": "-",
    "NHW": "--",
}
PERIOD_MARKER = {
    "HW": "o",
    "NHW": "s",
}
PERIOD_ALPHA = {
    "HW": 0.95,
    "NHW": 0.58,
}

# Boxplot layout: four boxes within each climate zone are deliberately separated
# to avoid overlap at high point density.
# Boxplot layout: wider separation to avoid overlap.
# Within each climate group:
#   UCI-NHW | UCI-HW | UHI-NHW | UHI-HW
BOX_WIDTH = 0.105

BOX_OFFSETS = {
    ("UCI", "NHW"): -0.54,
    ("UCI", "HW"):  -0.30,
    ("UHI", "NHW"): +0.30,
    ("UHI", "HW"):  +0.54,
}

# Larger distance between climate-zone groups.
BOX_GROUP_GAP = 2.05

# Jitter kept narrow so points do not visually merge neighbouring boxes.
BOX_JITTER_SD = 0.006


FONT_FAMILY = "DejaVu Sans"
FONT_BASE = 7.0
HOURS = np.arange(24)

MAP_CMAP = plt.cm.RdYlBu_r
MAP_LAND_COLOUR  = "#e2ddd4"
MAP_OCEAN_COLOUR = "#cfe0ec"
MAP_LAKE_COLOUR  = "#cfe0ec"
MAP_BORDER_COLOUR = "#b0a898"


# ============================================================================
# §2 Utilities
# ============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def norm_period(x):
    return PERIOD_NORM.get(str(x), str(x))

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
    # Sleep, CDH, temperature, mortality and all other fields remain unchanged.
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

def station_id_from_pair(pair_id, side):
    return f"{pair_id}_{side}"

def climate4_from_any(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if len(s) > 0 and s[0] in {"A", "B", "C", "D"}:
        return s[0]
    return np.nan

def climate4_from_row(row):
    for c in ["kg_group", "climate_zone_main", "climate_zone", "kg_code"]:
        if c in row.index and pd.notna(row[c]):
            out = climate4_from_any(row[c])
            if pd.notna(out):
                return out
    return np.nan

def sem_safe(v):
    v = pd.Series(v).dropna()
    if len(v) <= 1:
        return np.nan
    return float(v.std(ddof=1) / np.sqrt(len(v)))

def apply_nature_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_BASE,
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
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
    })

def add_panel_label(ax, letter, x=-0.12, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold",
            ha="left", va="bottom", clip_on=False)

def style_axis(ax):
    ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.22)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_edgecolor("black")

def savefig_both(fig, out_dir, basename):
    ensure_dir(out_dir)
    png_path = os.path.join(out_dir, f"{basename}.png")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    print(f"  ✓ saved: {png_path}")
    print(f"  ✓ saved: {pdf_path}")
    return png_path, pdf_path


class DiagnosticLog:
    def __init__(self):
        self.lines = []
        self.t0 = time.time()

    def add(self, title, text):
        msg = f"\n## {title}\n\n```\n{text.rstrip()}\n```\n"
        self.lines.append(msg)
        print(f"\n[DIAG] {title}\n{text}")

    def write(self, out_path):
        elapsed = time.time() - self.t0
        content = [
            "# UHI/UCI HW-NHW climate-zone figure diagnostic",
            f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Elapsed: {elapsed:.1f} s",
            ""
        ]
        content.extend(self.lines)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        print(f"  ✓ diagnostic written: {out_path}")



def load_canonical_annual_groups(path=V6_METRICS_CSV, diag=None):
    """Load strict annual percentile UHI/UCI pair classifications."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    g = pd.read_csv(path)
    required = {"pair_id", "period", "group"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"Canonical annual group columns missing: {sorted(missing)}")
    if "hw_method" in g.columns:
        g = g[g["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    g = g[g["period"].astype(str).str.lower().str.strip().eq("annual")].copy()
    g["pair_id"] = g["pair_id"].astype(str)
    g["thermal_group"] = g["group"].astype(str).str.upper().str.strip()
    g = g[g["thermal_group"].isin(THERMAL_ORDER)].copy()
    if g.empty:
        raise ValueError("No annual percentile UHI/UCI groups are available.")
    conflicts = g.groupby("pair_id", observed=True)["thermal_group"].nunique()
    conflict_ids = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflict_ids:
        raise ValueError(
            "Conflicting annual UHI/UCI groups for "
            f"{len(conflict_ids)} pair(s); examples={conflict_ids[:20]}"
        )
    out = g[["pair_id", "thermal_group"]].drop_duplicates("pair_id")
    if diag is not None:
        diag.add(
            "Canonical annual UHI/UCI groups",
            f"pairs={len(out)}\ncounts:\n{out['thermal_group'].value_counts().to_string()}"
        )
    return out


def load_reference_matched_pair_ids(path=V6_METRICS_CSV, diag=None):
    """Strict percentile HW/NHW cohort complete in dTmean/dAmp1/dTx/dTn."""
    df = pd.read_csv(path)
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    required_metrics = ["dTmean", "dAmp1", "dTx", "dTn"]
    missing = {"pair_id", "period", *required_metrics} - set(df.columns)
    if missing:
        raise ValueError(f"Matched-cohort columns missing: {sorted(missing)}")
    df["pair_id"] = df["pair_id"].astype(str)
    for c in required_metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    p = df["period"].astype(str).str.lower().str.strip()
    nhw = df.loc[p.eq("non_heatwave"), ["pair_id", *required_metrics]].copy()
    hw = df.loc[p.eq("heatwave"), ["pair_id", *required_metrics]].copy()
    nhw = nhw.rename(columns={c: f"{c}_nhw" for c in required_metrics})
    hw = hw.rename(columns={c: f"{c}_hw" for c in required_metrics})
    paired = nhw.merge(hw, on="pair_id", how="inner")
    complete = [f"{c}_nhw" for c in required_metrics] + [f"{c}_hw" for c in required_metrics]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=complete)
    annual = load_canonical_annual_groups(path)
    paired = paired.merge(annual, on="pair_id", how="inner")
    ids = set(paired["pair_id"].astype(str))
    if not ids:
        raise ValueError("Strict matched HW/NHW cohort is empty.")
    if diag is not None:
        diag.add(
            "Reference matched HW/NHW cohort",
            f"n_pairs={len(ids)}\nannual groups:\n{paired['thermal_group'].value_counts().to_string()}"
        )
    return ids


def _dunne_capacity(w):
    if not np.isfinite(w):
        return np.nan
    return float(np.clip(100.0 - 25.0 * max(float(w) - 25.0, 0.0) ** (2.0 / 3.0), 0.0, 100.0))

# ============================================================================
# §3 Load original data sources
# ============================================================================

def load_hw_diurnal_csv(diag):
    """Load v6 pair-period curves with strict annual UHI/UCI labels."""
    if not os.path.exists(V6_METRICS_CSV):
        raise FileNotFoundError(f"Missing v6 metrics: {V6_METRICS_CSV}")
    df = pd.read_csv(V6_METRICS_CSV)
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    df["pair_id"] = df["pair_id"].astype(str)
    df["period_norm"] = df["period"].apply(norm_period)
    df["climate4"] = df["kg_group"].apply(climate4_from_any)
    annual = load_canonical_annual_groups(V6_METRICS_CSV, diag=diag)
    if "group" in df.columns:
        df["group_period_original"] = df["group"]
        df = df.drop(columns=["group"])
    df = df.merge(annual, on="pair_id", how="inner")
    df = df.rename(columns={"thermal_group": "group"})
    df = df[df["climate4"].isin(CLIMATE_ORDER)].copy()
    diag.add(
        "Load v6 Diurnal Data",
        f"path={V6_METRICS_CSV}\nshape={df.shape}\n"
        f"periods={sorted(df['period_norm'].dropna().unique().tolist())}\n"
        f"annual-group counts:\n{df[['pair_id','group']].drop_duplicates()['group'].value_counts().to_string()}"
    )
    return df



def load_uhi_uci_pair_groups(diag):
    """Backward-compatible wrapper for strict annual group loading."""
    return load_canonical_annual_groups(V6_METRICS_CSV, diag=diag)



def load_cdh_panel(diag):
    p = os.path.join(CDH_DIR, "all_pairs_daily_panel.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=["local_date"])
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    df["period_norm"] = df["period"].apply(norm_period)
    diag.add("Load CDH daily panel", f"path={p}\nshape={df.shape}")
    return df


def load_labour_full(diag):
    """
    Load the formal common-time labour output.

    No legacy fallback is allowed. Dew point, RH, wet-bulb temperature, WBGT,
    and station-specific work-hour Tx have already been handled upstream by
    01_main_pair_period_metrics.py and 02_labour_capacity_loss.py.
    """
    p = os.path.join(LABOUR_DIR, "labour_loss_full.csv")

    if not os.path.exists(p):
        diag.add("Load labour", f"missing required file:\n{p}")
        return None

    df = pd.read_csv(p)
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()

    if "period_norm" in df.columns:
        df["period_norm"] = df["period_norm"].apply(norm_period)
    elif "period" in df.columns:
        df["period_norm"] = df["period"].apply(norm_period)
    else:
        diag.add(
            "Load labour",
            f"path={p}\nmissing period / period_norm column",
        )
        return None

    if "pair_id" not in df.columns:
        raise ValueError("labour_loss_full.csv is missing pair_id")

    df["pair_id"] = df["pair_id"].astype(str).str.strip()

    required = {
        "tx_hour_urban_dunne",
        "tx_hour_rural_dunne",
        "wbgt_tx_urban_dunne",
        "wbgt_tx_rural_dunne",
        "dloss_dunne_peak_t_diff",
        "urban_wbgt_input_ok",
        "rural_wbgt_input_ok",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Formal common-time labour fields are missing from "
            f"{p}: {missing}"
        )

    diag.add(
        "Load labour",
        f"path={p}\n"
        "source_kind=labour_loss_full_common_time\n"
        f"shape={df.shape}\n"
        f"periods={sorted(df['period_norm'].dropna().unique().tolist())}\n"
        "dew_point_read_downstream=no; formal upstream Tx/WBGT fields required",
    )

    return df


def load_hne_paired(diag):
    frames = []
    for rel in ["all_pairs_annual.csv",
                "warm_season_periods/all_pairs_warm_periods.csv"]:
        fp = os.path.join(HNE_DIR, rel)
        if os.path.exists(fp):
            tmp = pd.read_csv(fp)
            if "period" not in tmp.columns:
                tmp["period"] = "annual"
            if "hw_method" in tmp.columns:
                tmp = tmp[tmp["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
            tmp["period_norm"] = tmp["period"].apply(norm_period)
            frames.append(tmp)

    if not frames:
        diag.add("Load HNE", "No HNE files found")
        return None

    df = pd.concat(frames, ignore_index=True)
    diag.add("Load HNE", f"shape={df.shape}\ncols={list(df.columns[:25])}")
    return df


# ============================================================================
# §4 Build station-period panel: same construction logic as v4
# ============================================================================
def build_station_daily_from_cdh(df_cdh, diag):
    rows = []
    for _, r in df_cdh.iterrows():
        base = {
            "pair_id": str(r.get("pair_id")),
            "period_norm": r.get("period_norm"),
            "local_date": r.get("local_date"),
        }
        for side, sfx in [("urban", "_u"), ("rural", "_r")]:
            row = base.copy()
            row["station_side"] = side
            row["station_id"] = station_id_from_pair(row["pair_id"], side)
            mapping = {
                "CDH_total": f"CDH_total{sfx}",
                "HDH_total": f"HDH_total{sfx}",
                "E_total":   f"E_total{sfx}",
                "Tmax":      f"Tmax{sfx}",
                "Tmin":      f"Tmin{sfx}",
                "Tmean":     f"Tmean{sfx}",
            }
            for out_c, in_c in mapping.items():
                row[out_c] = r[in_c] if in_c in df_cdh.columns else np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    diag.add("Build station daily from CDH",
             f"shape={out.shape}\nstation_n={out['station_id'].nunique()}")
    return out


def aggregate_station_period_from_daily(df_station_daily, diag):
    agg_map = {
        "CDH_total": "mean",
        "HDH_total": "mean",
        "E_total":   "mean",
        "Tmax":      "mean",
        "Tmin":      "mean",
        "Tmean":     "mean",
    }
    g = (df_station_daily
         .groupby(["station_id", "pair_id", "station_side", "period_norm"], observed=True)
         .agg(agg_map)
         .reset_index())
    g = g.rename(columns={
        "CDH_total": "CDH_total_mean",
        "HDH_total": "HDH_total_mean",
        "E_total":   "E_total_mean",
        "Tmax":      "Tmax_mean",
        "Tmin":      "Tmin_mean",
        "Tmean":     "Tmean_mean",
    })
    diag.add("Aggregate station-period from daily", f"shape={g.shape}")
    return g


def stationize_pair_period(df_pair, metric_specs):
    if df_pair is None or len(df_pair) == 0:
        return pd.DataFrame()

    base_cols = [c for c in ["pair_id", "period_norm"] if c in df_pair.columns]
    if not base_cols:
        return pd.DataFrame()

    df_pair = df_pair.copy()
    df_pair["pair_id"] = df_pair["pair_id"].astype(str)

    urban = df_pair[base_cols].copy()
    urban["station_side"] = "urban"
    urban["station_id"] = urban["pair_id"].astype(str) + "_urban"
    for out_name, side_map in metric_specs.items():
        candidates = side_map.get("U", [])
        found = next((c for c in candidates if c in df_pair.columns), None)
        urban[out_name] = df_pair[found].values if found is not None else np.nan

    rural = df_pair[base_cols].copy()
    rural["station_side"] = "rural"
    rural["station_id"] = rural["pair_id"].astype(str) + "_rural"
    for out_name, side_map in metric_specs.items():
        candidates = side_map.get("R", [])
        found = next((c for c in candidates if c in df_pair.columns), None)
        rural[out_name] = df_pair[found].values if found is not None else np.nan

    return pd.concat([urban, rural], ignore_index=True)


def build_station_period_from_hne(df_hne, diag):
    if df_hne is None or len(df_hne) == 0:
        return pd.DataFrame()

    df_hne = df_hne[df_hne["period_norm"].isin(PLOT_PERIODS)].copy()

    metric_specs = {
        "sleep_loss_min_mean": {
            "U": ["sleep_loss_min_U"],
            "R": ["sleep_loss_min_R"],
        },
        "total_loss_usd_mean": {
            "U": ["total_loss_usd_U"],
            "R": ["total_loss_usd_R"],
        },
        "econ_loss_usd_sleep_mean": {
            "U": ["econ_loss_usd_sleep_U"],
            "R": ["econ_loss_usd_sleep_R"],
        },
        "econ_loss_usd_heat_mean": {
            "U": ["econ_loss_usd_heat_U"],
            "R": ["econ_loss_usd_heat_R"],
        },
        "total_loss_pct_mean": {
            "U": ["total_loss_pct_U"],
            "R": ["total_loss_pct_R"],
        },
        "econ_loss_pct_short_mean": {
            "U": ["econ_loss_pct_short_U"],
            "R": ["econ_loss_pct_short_R"],
        },
    }

    needed_cols = ["pair_id", "period_norm",
                   "sleep_loss_min_U",      "sleep_loss_min_R",
                   "total_loss_usd_U",      "total_loss_usd_R",
                   "econ_loss_usd_sleep_U", "econ_loss_usd_sleep_R",
                   "econ_loss_usd_heat_U",  "econ_loss_usd_heat_R",
                   "total_loss_pct_U",      "total_loss_pct_R",
                   "econ_loss_pct_short_U", "econ_loss_pct_short_R"]
    needed_cols = [c for c in dict.fromkeys(needed_cols) if c in df_hne.columns]
    df_hne = df_hne[needed_cols].copy()

    active_specs = {}
    for k, v in metric_specs.items():
        all_cands = v.get("U", []) + v.get("R", [])
        if any(c in df_hne.columns for c in all_cands):
            active_specs[k] = v

    df = stationize_pair_period(df_hne, active_specs)
    if len(df) == 0:
        return df

    # ------------------------------------------------------------
    # Sleep-loss sign convention fix
    # ------------------------------------------------------------
    # Upstream sleep_loss_min_U/R is actually ΔSleep:
    #   negative value = sleep duration decreases = sleep loss occurs
    #
    # Convert it to positive sleep-loss amount:
    #   positive value = more minutes of sleep lost
    #
    # After this conversion:
    #   urban - rural > 0 means urban has more sleep loss than rural
    # ------------------------------------------------------------
    if "sleep_loss_min_mean" in df.columns:
        df["sleep_loss_min_mean"] = -pd.to_numeric(
            df["sleep_loss_min_mean"],
            errors="coerce"
        )

        diag.add(
            "Sleep-loss sign convention",
            "Converted upstream sleep_loss_min from ΔSleep to positive sleep-loss amount.\n"
            "Upstream definition: negative = sleep reduction.\n"
            "Downstream plotting definition: positive = more sleep loss.\n"
            "Therefore urban-rural contrast > 0 now means urban has more sleep loss than rural."
        )

    df = (
        df.groupby(["station_id", "pair_id", "station_side", "period_norm"], observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )


    diag.add("Build station-period from HNE", f"shape={df.shape}")
    return df




def build_station_period_from_labour(df_lc, diag):
    """
    Carry the formal upstream Dunne loss at each station-specific work-hour Tx.

    The plotting workflow does not read dew point, reconstruct WBGT, or select
    Tx from the main temperature curve.
    """
    if df_lc is None or len(df_lc) == 0:
        return pd.DataFrame()

    work = df_lc[df_lc["period_norm"].isin(PLOT_PERIODS)].copy()
    required = {
        "pair_id",
        "tx_hour_urban_dunne",
        "tx_hour_rural_dunne",
        "wbgt_tx_urban_dunne",
        "wbgt_tx_rural_dunne",
        "dloss_dunne_peak_t_diff",
        "urban_wbgt_input_ok",
        "rural_wbgt_input_ok",
    }
    missing = sorted(required - set(work.columns))
    if missing:
        raise ValueError(
            "Cannot load formal common-time Dunne Tx labour result; "
            f"missing columns: {missing}"
        )

    work["pair_id"] = work["pair_id"].astype(str).str.strip()
    for col in required - {"pair_id"}:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    valid = (
        work["urban_wbgt_input_ok"].eq(1)
        & work["rural_wbgt_input_ok"].eq(1)
        & work["tx_hour_urban_dunne"].notna()
        & work["tx_hour_rural_dunne"].notna()
        & work["wbgt_tx_urban_dunne"].notna()
        & work["wbgt_tx_rural_dunne"].notna()
        & work["dloss_dunne_peak_t_diff"].notna()
    )

    bad_hours = valid & (
        ~work["tx_hour_urban_dunne"].between(8, 19, inclusive="both")
        | ~work["tx_hour_rural_dunne"].between(8, 19, inclusive="both")
    )
    if bad_hours.any():
        examples = work.loc[
            bad_hours,
            ["pair_id", "period_norm", "tx_hour_urban_dunne", "tx_hour_rural_dunne"],
        ].head(20)
        raise ValueError(
            "Upstream Tx hour is outside 08:00–19:59. Examples:\n"
            + examples.to_string(index=False)
        )

    def _loss(values):
        w = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.clip(
            25.0 * np.maximum(w - 25.0, 0.0) ** (2.0 / 3.0),
            0.0,
            100.0,
        )

    loss_u = _loss(work["wbgt_tx_urban_dunne"])
    loss_r = _loss(work["wbgt_tx_rural_dunne"])
    check_delta = loss_u - loss_r
    upstream_delta = work["dloss_dunne_peak_t_diff"].to_numpy(dtype=float)

    mismatch = valid.to_numpy() & ~np.isclose(
        check_delta,
        upstream_delta,
        rtol=1e-8,
        atol=1e-10,
        equal_nan=False,
    )
    if mismatch.any():
        idx = np.flatnonzero(mismatch)[:20]
        examples = work.iloc[idx][["pair_id", "period_norm"]].copy()
        examples["upstream_delta"] = upstream_delta[idx]
        examples["recomputed_delta"] = check_delta[idx]
        raise ValueError(
            "Formal upstream labour difference is inconsistent with upstream "
            "Tx-hour WBGT. Examples:\n" + examples.to_string(index=False)
        )

    work["tx_hour_urban"] = work["tx_hour_urban_dunne"].where(valid)
    work["tx_hour_rural"] = work["tx_hour_rural_dunne"].where(valid)
    work["wbgt_urban_tx"] = work["wbgt_tx_urban_dunne"].where(valid)
    work["wbgt_rural_tx"] = work["wbgt_tx_rural_dunne"].where(valid)
    work["labour_loss_tx_u"] = pd.Series(loss_u, index=work.index).where(valid)
    work["labour_loss_tx_r"] = pd.Series(loss_r, index=work.index).where(valid)
    work["d_labour_loss_tx"] = work["dloss_dunne_peak_t_diff"].where(valid)

    tmp = (
        work[
            [
                "pair_id",
                "period_norm",
                "d_labour_loss_tx",
                "tx_hour_urban",
                "tx_hour_rural",
                "wbgt_urban_tx",
                "wbgt_rural_tx",
                "labour_loss_tx_u",
                "labour_loss_tx_r",
            ]
        ]
        .dropna(subset=["d_labour_loss_tx"])
        .groupby(["pair_id", "period_norm"], observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )

    # Duplicate only for the legacy station-panel merge; impact plots
    # deduplicate at pair level.
    urban = tmp.assign(
        station_side="urban",
        station_id=tmp["pair_id"] + "_urban",
    )
    rural = tmp.assign(
        station_side="rural",
        station_id=tmp["pair_id"] + "_rural",
    )
    out = pd.concat([urban, rural], ignore_index=True)
    out["labour_loss_is_pair_delta"] = True
    out["labour_loss_metric_source"] = (
        "02 labour output: common-time two-harmonic T/Td, "
        "station-specific work-hour Tx, shaded WBGT, Dunne loss"
    )
    out["labour_loss_metric_interpretation"] = (
        "Loss_urban(Tx_urban)-Loss_rural(Tx_rural)"
    )

    diag.add(
        "Build station-period from labour",
        f"shape={out.shape}\nselected_metric=d_labour_loss_tx\n"
        "sign=positive means urban labour loss > rural labour loss\n"
        f"summary:\n{tmp['d_labour_loss_tx'].describe().to_string()}",
    )
    return out



def merge_station_modules(df_phys, df_hne_st, df_lab_st, diag):
    frames = [x for x in [df_phys, df_hne_st, df_lab_st]
              if x is not None and len(x) > 0]
    if not frames:
        return pd.DataFrame()

    out = frames[0].copy()
    for add in frames[1:]:
        out = out.merge(
            add,
            on=["station_id", "pair_id", "station_side", "period_norm"],
            how="outer",
            suffixes=("", "_dup")
        )
        dup_cols = [c for c in out.columns if c.endswith("_dup")]
        for dc in dup_cols:
            base = dc[:-4]
            if base in out.columns:
                out[base] = out[base].combine_first(out[dc])
            else:
                out[base] = out[dc]
        out = out.drop(columns=dup_cols)

    diag.add("Merge station modules",
             f"shape={out.shape}\nstation_n={out['station_id'].nunique()}")
    return out


def build_station_climate_map_from_hw_diurnal(df_v6):
    """
    针对 v6 数据源的修复版本：
    从 pair-level 的宽表拆分出 station-level 的坐标映射表
    """
    # 基础列
    common_cols = ["pair_id", "period_norm", "climate4", "kg_group"]
    
    # 1. 提取城市站点
    u = df_v6[common_cols + ["lon_urban", "lat_urban"]].copy()
    u["station_side"] = "urban"
    u["station_id"] = u["pair_id"].astype(str) + "_urban"
    u = u.rename(columns={"lon_urban": "lon_station", "lat_urban": "lat_station"})
    
    # 2. 提取农村站点
    r = df_v6[common_cols + ["lon_rural", "lat_rural"]].copy()
    r["station_side"] = "rural"
    r["station_id"] = r["pair_id"].astype(str) + "_rural"
    r = r.rename(columns={"lon_rural": "lon_station", "lat_rural": "lat_station"})
    
    # 合并
    dedup = pd.concat([u, r], ignore_index=True)
    
    # 去重
    dedup = dedup.drop_duplicates(["pair_id", "station_side", "station_id", "period_norm"])
    
    # 补全 climate_zone_main 列名以匹配下游
    dedup["climate_zone_main"] = dedup["climate4"].map(CLIMATE_NAME)
    
    return dedup


def merge_climate_group_to_station_panel(df_station, df_hw_diurnal, df_pair_group, diag):
    clim_map = build_station_climate_map_from_hw_diurnal(df_hw_diurnal)

    out = df_station.merge(
        clim_map,
        on=["station_id", "pair_id", "station_side", "period_norm"],
        how="left"
    )

    # Pair-level fallback fill
    pair_clim = (clim_map[["pair_id", "climate4", "kg_group", "climate_zone_main"]]
                 .dropna(subset=["climate4"])
                 .drop_duplicates("pair_id"))
    out = out.merge(pair_clim, on="pair_id", how="left", suffixes=("", "_pairfill"))
    for c in ["climate4", "kg_group", "climate_zone_main"]:
        cf = c + "_pairfill"
        if cf in out.columns:
            out[c] = out[c].combine_first(out[cf])
            out = out.drop(columns=[cf])

    out["pair_id"] = out["pair_id"].astype(str)
    out = out.merge(df_pair_group, on="pair_id", how="left")
    out = out[out["climate4"].isin(CLIMATE_ORDER)].copy()
    out = out[out["thermal_group"].isin(THERMAL_ORDER)].copy()

    diag.add(
        "Merge climate + UHI/UCI groups into station panel",
        f"shape={out.shape}\n"
        f"thermal counts:\n{out['thermal_group'].value_counts(dropna=False).to_string()}"
    )
    return out


# ============================================================================
# §5 Complete-case filters
# ============================================================================
def keep_stations_with_both_periods(df, value_col=None, station_col="station_id"):
    """
    Keep only stations with valid HW and NHW rows.

    If value_col is supplied, the station must have non-NaN value_col in both
    periods. Otherwise, only the period rows are required.
    """
    sub = df[df["period_norm"].isin(PLOT_PERIODS)].copy()
    if value_col is not None:
        sub = sub[sub[value_col].notna()].copy()

    ok = (sub.groupby(station_col, observed=True)["period_norm"]
            .nunique()
            .loc[lambda s: s == 2]
            .index)
    return sub[sub[station_col].isin(ok)].copy()


def keep_pairs_with_complete_urban_rural_hw_nhw(df, value_col):
    """
    For map R calculation, keep only pairs with urban and rural values in both
    HW and NHW.

    Important pandas detail:
    pivot_table creates a DataFrame with MultiIndex columns because columns are
    ["period_norm", "station_side"].  Do not return pvt[["R_value"]] directly,
    because that keeps two-level columns and later causes:
        MergeError: Not allowed to merge between different levels.
    Instead, compute R as a Series and build a new flat two-column DataFrame.
    """
    sub = df[df["period_norm"].isin(PLOT_PERIODS)].copy()
    sub = sub[sub["station_side"].isin(["urban", "rural"])].copy()
    sub = sub[sub[value_col].notna()].copy()

    pvt = sub.pivot_table(
        index="pair_id",
        columns=["period_norm", "station_side"],
        values=value_col,
        aggfunc="mean"
    )

    required = [("HW", "urban"), ("HW", "rural"),
                ("NHW", "urban"), ("NHW", "rural")]
    for c in required:
        if c not in pvt.columns:
            return pd.DataFrame(columns=["pair_id", "R_value"])

    pvt = pvt.dropna(subset=required)

    R_series = (
        (pvt[("HW", "urban")] - pvt[("HW", "rural")]) -
        (pvt[("NHW", "urban")] - pvt[("NHW", "rural")])
    )

    out = R_series.rename("R_value").reset_index()
    out.columns = ["pair_id", "R_value"]  # force one-level columns
    out["pair_id"] = out["pair_id"].astype(str)
    out["R_value"] = pd.to_numeric(out["R_value"], errors="coerce")
    return out


# ============================================================================
# §6 Diurnal table and summaries
# ============================================================================
def build_diurnal_long_hw_nhw_uhi_uci(df_v6, df_pair_group, diag):
    """
    针对 v6 数据源的修复版本：
    1. 处理宽格式列 (urban_diurnal_hXX 和 rural_diurnal_hXX)
    2. 手动创建 station_id 列，以修复 KeyError
    """
    # 筛选 period
    df = df_v6[df_v6["period_norm"].isin(PLOT_PERIODS)].copy()
    
    # 转换成长格式
    long_rows = []
    
    # 识别小时列
    for h in range(24):
        u_col = f"urban_diurnal_h{h:02d}"
        r_col = f"rural_diurnal_h{h:02d}"
        
        # 提取 Urban 数据
        # 注意：v6 中的列名是 'group'，我们需要重命名为 'thermal_group'
        u_df = df[["pair_id", "period_norm", "climate4", "group", u_col]].copy()
        u_df.columns = ["pair_id", "period_norm", "climate4", "thermal_group", "Ta"]
        u_df["station_side"] = "urban"
        u_df["station_id"] = u_df["pair_id"].astype(str) + "_urban"  # <--- 修复点：添加 station_id
        u_df["hour"] = h
        long_rows.append(u_df)
        
        # 提取 Rural 数据
        r_df = df[["pair_id", "period_norm", "climate4", "group", r_col]].copy()
        r_df.columns = ["pair_id", "period_norm", "climate4", "thermal_group", "Ta"]
        r_df["station_side"] = "rural"
        r_df["station_id"] = r_df["pair_id"].astype(str) + "_rural"  # <--- 修复点：添加 station_id
        r_df["hour"] = h
        long_rows.append(r_df)

    out = pd.concat(long_rows, ignore_index=True)
    out = out.dropna(subset=["Ta"]).copy()

    # 确保 thermal_group 符合 THERMAL_ORDER (UCI/UHI)
    out = out[out["thermal_group"].isin(THERMAL_ORDER)].copy()

    diag.add(
        "Build Diurnal Long from v6",
        f"Converted shape={out.shape}\n"
        f"n_pairs={out['pair_id'].nunique()}\n"
        f"columns={out.columns.tolist()}" # 诊断列名
    )
    return out


def summarize_diurnal(df_diurnal_long):
    # 关键：按 station_side 分组，而不是按 thermal_group
    g = (df_diurnal_long
         .groupby(["climate4", "station_side", "period_norm", "hour"], observed=True)["Ta"]
         .agg(Ta_mean="mean", Ta_std="std", n="count")
         .reset_index())
    g["Ta_sem"] = g["Ta_std"] / np.sqrt(g["n"].clip(lower=1))
    return g


def build_pair_impact_panel(df_station_clim, matched_ids, diag):
    """Build one pair-period table of U-R sleep loss, labour loss and CDH."""
    base = df_station_clim[df_station_clim["period_norm"].isin(PLOT_PERIODS)].copy()
    base["pair_id"] = base["pair_id"].astype(str)
    base = base[base["pair_id"].isin(set(map(str, matched_ids)))].copy()

    keys = ["pair_id", "period_norm", "climate4", "thermal_group"]
    pieces = []
    for source_col, out_col in [
        ("sleep_loss_min_mean", "d_sleep_loss"),
        ("CDH_total_mean", "d_cdh_total"),
    ]:
        if source_col not in base.columns:
            continue
        p = base.pivot_table(index=keys, columns="station_side", values=source_col, aggfunc="mean").reset_index()
        if {"urban", "rural"}.issubset(p.columns):
            p[out_col] = pd.to_numeric(p["urban"], errors="coerce") - pd.to_numeric(p["rural"], errors="coerce")
            pieces.append(p[keys + [out_col]])

    if "d_labour_loss_tx" in base.columns:
        p = (
            base[keys + ["d_labour_loss_tx"]]
            .dropna(subset=["d_labour_loss_tx"])
            .groupby(keys, observed=True)["d_labour_loss_tx"].mean().reset_index()
        )
        pieces.append(p)

    if not pieces:
        return pd.DataFrame(columns=keys)
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=keys, how="outer")
    diag.add(
        "Build pair-level impact panel",
        f"shape={out.shape}\nmatched pairs={out['pair_id'].nunique()}\ncolumns={list(out.columns)}"
    )
    return out


def keep_pairs_with_both_periods(df, value_col):
    sub = df[df["period_norm"].isin(PLOT_PERIODS) & df[value_col].notna()].copy()
    ok = sub.groupby("pair_id", observed=True)["period_norm"].nunique()
    ids = set(ok[ok == 2].index.astype(str))
    return sub[sub["pair_id"].astype(str).isin(ids)].copy()

# ============================================================================
# §7 Plot: diurnal + boxplots
# ============================================================================
def draw_diurnal_panel(ax, df_sum, zone):
    """
    保持原有逻辑，但确保 DIURNAL_COLOR 对应正确
    """
    sub = df_sum[df_sum["climate4"] == zone].copy()

    for side in ["urban", "rural"]:
        for per in PLOT_PERIODS:
            sp = sub[(sub["station_side"] == side) & 
                     (sub["period_norm"] == per)].sort_values("hour")
            if len(sp) == 0: continue

            x = sp["hour"].values
            y = sp["Ta_mean"].values
            se = sp["Ta_sem"].fillna(0).values

            # 城市=红色, 农村=蓝色
            color = DIURNAL_COLOR[side]
            
            # 线型: HW=实线, NHW=虚线
            ls = PERIOD_LINESTYLE[per] 

            ax.plot(x, y, color=color, linestyle=ls, 
                    lw=1.2, alpha=0.9, zorder=3)
            ax.fill_between(x, y-se, y+se, color=color, 
                            alpha=0.1, linewidth=0, zorder=2)

    ax.set_title(f"Zone {zone}: {CLIMATE_NAME.get(zone)}", fontsize=7.5, fontweight="bold")
    ax.set_xticks([0, 6, 12, 18, 23])
    style_axis(ax)


def draw_metric_boxplot(ax, df_pair, value_col, ylabel, title, diag):
    df = keep_pairs_with_both_periods(df_pair, value_col)
    df = df[df["climate4"].isin(CLIMATE_ORDER) & df["thermal_group"].isin(THERMAL_ORDER)].copy()
    rng = np.random.default_rng(2026)
    base_x = np.arange(len(CLIMATE_ORDER), dtype=float) * BOX_GROUP_GAP
    for i, z in enumerate(CLIMATE_ORDER):
        for grp in THERMAL_ORDER:
            for per in PLOT_PERIODS:
                vals = df[(df["climate4"] == z) & (df["thermal_group"] == grp) & (df["period_norm"] == per)][value_col].dropna().astype(float)
                if len(vals) < 3:
                    continue
                pos = base_x[i] + BOX_OFFSETS[(grp, per)]
                c = BOX_COLOR[grp]
                ax.boxplot([vals.values], positions=[pos], widths=BOX_WIDTH, patch_artist=True,
                           showfliers=False,
                           boxprops=dict(facecolor=c, edgecolor="black", linewidth=0.55, alpha=PERIOD_ALPHA[per]),
                           medianprops=dict(color="white", linewidth=0.95),
                           whiskerprops=dict(color="black", linewidth=0.50),
                           capprops=dict(color="black", linewidth=0.50), manage_ticks=False)
                ax.scatter(pos + rng.normal(0, BOX_JITTER_SD, size=len(vals)), vals.values,
                           s=1.5, color="black", alpha=0.060, linewidths=0, zorder=2)
                ax.scatter([pos], [vals.mean()], s=11, marker=PERIOD_MARKER[per],
                           color="white", edgecolor="black", linewidth=0.45, zorder=6)
    ax.set_xticks(base_x)
    ax.set_xticklabels([CLIMATE_NAME[z] for z in CLIMATE_ORDER])
    ax.set_xlim(base_x[0] - 0.82, base_x[-1] + 0.82)
    ax.set_ylabel(ylabel, labelpad=0.6)
    ax.set_title(title, fontsize=7.2, fontweight="bold", pad=4)
    style_axis(ax)



def make_diurnal_box_figure(df_diurnal_sum, df_pair_impact, out_dir, diag):
    apply_nature_style()
    fig = plt.figure(figsize=(8.2, 6.0), dpi=600)
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.00, 1.18],
                  hspace=0.66, wspace=1.25, left=0.070, right=0.990, top=0.875, bottom=0.175)
    diurnal_axes = [fig.add_subplot(gs[0, i*3:(i+1)*3]) for i in range(4)]
    for ax, z, letter in zip(diurnal_axes, CLIMATE_ORDER, ["a", "b", "c", "d"]):
        draw_diurnal_panel(ax, df_diurnal_sum, z)
        if z == "A": ax.set_ylabel(r"$T_a$ (°C)")
        add_panel_label(ax, letter, x=-0.18, y=1.06)
    ax_sleep = fig.add_subplot(gs[1, 0:4]); ax_lab = fig.add_subplot(gs[1, 4:8]); ax_cdh = fig.add_subplot(gs[1, 8:12])
    draw_metric_boxplot(ax_sleep, df_pair_impact, "d_sleep_loss", r"$\Delta$Sleep loss (min/night)", "Sleep loss", diag)
    add_panel_label(ax_sleep, "e", x=-0.14, y=1.06)
    draw_metric_boxplot(ax_lab, df_pair_impact, "d_labour_loss_tx", r"$\Delta$Labour loss (%)", "Labour loss", diag)
    add_panel_label(ax_lab, "f", x=-0.14, y=1.06)
    draw_metric_boxplot(ax_cdh, df_pair_impact, "d_cdh_total", r"$\Delta$CDH (°C·h/day)", "CDH", diag)
    add_panel_label(ax_cdh, "g", x=-0.14, y=1.06)
    line_handles = [
        Line2D([0],[0], color=DIURNAL_COLOR["urban"], lw=2.0, label="Urban"),
        Line2D([0],[0], color=DIURNAL_COLOR["rural"], lw=2.0, label="Rural"),
        Line2D([0],[0], color="black", lw=1.2, ls="--", label="Non-heatwave"),
        Line2D([0],[0], color="black", lw=1.2, ls="-", label="Heatwave"),
    ]
    fig.legend(handles=line_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.52,0.955), ncol=4, columnspacing=1.30)
    box_handles = [
        Patch(facecolor=BOX_COLOR["UCI"], edgecolor="black", alpha=PERIOD_ALPHA["NHW"], label="UCI Non-heatwave"),
        Patch(facecolor=BOX_COLOR["UCI"], edgecolor="black", alpha=PERIOD_ALPHA["HW"], label="UCI Heatwave"),
        Patch(facecolor=BOX_COLOR["UHI"], edgecolor="black", alpha=PERIOD_ALPHA["NHW"], label="UHI Non-heatwave"),
        Patch(facecolor=BOX_COLOR["UHI"], edgecolor="black", alpha=PERIOD_ALPHA["HW"], label="UHI Heatwave"),
    ]
    fig.legend(handles=box_handles, frameon=False, loc="lower center", bbox_to_anchor=(0.52,0.035), ncol=4, columnspacing=1.20)
    return savefig_both(fig, out_dir, "climatezone_uhi_uci_diurnal_box_hw_nhw")


# ============================================================================
# §8 Map R metrics
# ============================================================================
def build_pair_location_table(df_v6, df_pair_group):
    """
    针对 v6 数据源的修复版本：
    获取每对站点的绘图坐标（优先使用 lon_urban/lat_urban）
    """
    df = df_v6.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    
    # 提取核心位置列
    loc = df[["pair_id", "lon_urban", "lat_urban", "climate4"]].copy()
    loc = loc.rename(columns={"lon_urban": "lon_plot", "lat_urban": "lat_plot"})
    
    # 去重
    loc = loc.drop_duplicates("pair_id")

    # 关联分组信息
    loc = loc.merge(df_pair_group, on="pair_id", how="left")
    return loc



def build_R_map_metric(df_pair_impact, metric_col, metric_name, diag):
    """
    Build the frozen HW-minus-NHW response map metric:

        R_X = (X_urban - X_rural)_HW - (X_urban - X_rural)_NHW

    The input metric_col must already be a pair-period urban-minus-rural
    contrast. A pair is retained only when both HW and NHW values are finite.
    Missing one-sided responses remain NaN and are excluded from the map.
    """
    required = {"pair_id", "period_norm", metric_col}
    missing = required - set(df_pair_impact.columns)
    if missing:
        raise ValueError(
            f"Cannot build {metric_name} R map; missing columns: {sorted(missing)}"
        )

    sub = df_pair_impact.loc[
        df_pair_impact["period_norm"].isin(["HW", "NHW"]),
        ["pair_id", "period_norm", metric_col],
    ].copy()
    sub["pair_id"] = sub["pair_id"].astype(str).str.strip()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan)

    wide = (
        sub.groupby(["pair_id", "period_norm"], observed=True)[metric_col]
        .mean()
        .unstack("period_norm")
    )
    for period in ["NHW", "HW"]:
        if period not in wide.columns:
            wide[period] = np.nan

    wide = wide.dropna(subset=["NHW", "HW"]).copy()
    wide["R_value"] = wide["HW"] - wide["NHW"]
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(subset=["R_value"])

    out = (
        wide.rename(columns={"NHW": "delta_nhw", "HW": "delta_hw"})
        .reset_index()
    )
    out["metric"] = metric_name
    out["response_definition"] = (
        "(urban-rural)_HW - (urban-rural)_NHW"
    )

    diag.add(
        f"Build R map metric: {metric_name}",
        f"metric_col={metric_col}\n"
        "definition=(urban-rural)_HW - (urban-rural)_NHW\n"
        f"complete matched pairs={len(out)}\n"
        + (out["R_value"].describe().to_string() if len(out) else "empty"),
    )
    return out[[
        "pair_id", "delta_nhw", "delta_hw", "R_value",
        "metric", "response_definition",
    ]]



def setup_map_axis(ax, show_y_labels=True):
    """
    严格对齐 reference (supplement_period_specific_uhi_maps) 的地图底图样式。
    
    规范:
      - 投影: Robinson(central_longitude=0) (在调用处设定)
      - 海洋: white
      - 陆地: #f7f7f7 (浅灰)
      - 海岸线: #505050, lw=0.25
      - 国界: #aaaaaa, lw=0.15, 虚线 ':'
      - 地图外框 geo spine: black, lw=0.60
      - 四周矩形 spine 不可见
      - 网格线: #cccccc, lw=0.20, alpha=0.5
      - 网格标签字号: 5.0 pt
    """
    if HAS_CARTOPY:
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

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=0.20, color="#cccccc", alpha=0.5, linestyle="-"
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = show_y_labels
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": 5.0}
        gl.ylabel_style = {"size": 5.0}

        # 隐藏矩形四边 spine，只保留 geo spine
        for s in ['left', 'right', 'top', 'bottom']:
            if s in ax.spines:
                ax.spines[s].set_visible(False)
        if 'geo' in ax.spines:
            ax.spines['geo'].set_visible(True)
            ax.spines['geo'].set_linewidth(0.60)
            ax.spines['geo'].set_edgecolor("black")
    else:
        ax.set_facecolor("white")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.tick_params(labelsize=5.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.60)
            spine.set_edgecolor("black")

def draw_R_map_panel(
        ax,
        df_map,
        title,
        cmap,
        vmin,
        vmax,
        tick_fmt="%.2f",
        show_y_labels=True):
    """Draw one HW-minus-NHW response map using the R_value field."""
    setup_map_axis(ax, show_y_labels=show_y_labels)

    vals = df_map["R_value"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        ax.set_title(title + " (no complete HW/NHW data)", fontsize=7.4,
                     fontweight="bold", pad=4)
        return None

    scatter_kw = dict(
        c=df_map["R_value"].values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=11.0,
        marker="o",
        edgecolors="black",
        linewidths=0.25,
        alpha=0.90,
        zorder=3,
    )
    if HAS_CARTOPY:
        scatter_kw["transform"] = ccrs.PlateCarree()

    sc = ax.scatter(
        df_map["lon_plot"].values,
        df_map["lat_plot"].values,
        **scatter_kw,
    )
    ax.set_title(title, fontsize=7.6, fontweight="bold", pad=4)
    return sc


def make_R_map_figure(
        df_pair_impact,
        df_hw_diurnal,
        df_pair_group,
        out_dir,
        diag):
    """
    Draw a 3-row x 2-column HW-minus-NHW response-map matrix.

    Scientific definition
    ---------------------
    For every outcome:

        R_X
        =
        (X_urban - X_rural)_HW
        -
        (X_urban - X_rural)_NHW

    Layout
    ------
                  UHI                 UCI
        row 1:    Sleep loss          Sleep loss
        row 2:    Labour loss         Labour loss
        row 3:    CDH                 CDH

    Scientific data are unchanged relative to the existing 3-panel version.

    Specifically:
    - build_R_map_metric() is unchanged;
    - the same pair-level R_value is used;
    - both HW and NHW must remain finite;
    - no missing value is converted to zero;
    - the same colour limits are shared by UHI and UCI for each outcome;
    - only the graphical arrangement is changed.
    """
    # ------------------------------------------------------------
    # 1. Pair locations and canonical annual UHI/UCI groups
    # ------------------------------------------------------------
    loc = build_pair_location_table(
        df_hw_diurnal,
        df_pair_group,
    )

    loc["pair_id"] = (
        loc["pair_id"]
        .astype(str)
        .str.strip()
    )

    if "thermal_group" not in loc.columns:
        raise ValueError(
            "Six-panel R-map layout requires the canonical "
            "'thermal_group' column in the pair-location table."
        )

    loc["thermal_group"] = (
        loc["thermal_group"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # ------------------------------------------------------------
    # 2. Keep exactly the same scientific metrics as the
    #    current 3-panel version
    # ------------------------------------------------------------
    metric_specs = [
        {
            "col": "d_sleep_loss",
            "name": "Sleep loss",
            "cbar_label": r"$R_{sleep}$ (min/night)",
            "tick_fmt": "%.2f",
            "min_abs_lim": None,
        },
        {
            "col": "d_labour_loss_tx",
            "name": "Labour loss",
            "cbar_label": r"$R_{labour}$ (%)",
            "tick_fmt": "%.2f",
            "min_abs_lim": 0.001,
        },
        {
            "col": "d_cdh_total",
            "name": "CDH",
            "cbar_label": r"$R_{CDH}$ (°C·h/day)",
            "tick_fmt": "%.2f",
            "min_abs_lim": None,
        },
    ]

    # ------------------------------------------------------------
    # 3. Build exactly the same R values as before
    # ------------------------------------------------------------
    map_data = {}

    for spec in metric_specs:
        response = build_R_map_metric(
            df_pair_impact=df_pair_impact,
            metric_col=spec["col"],
            metric_name=spec["name"],
            diag=diag,
        )

        m = response.merge(
            loc,
            on="pair_id",
            how="left",
            validate="one_to_one",
        )

        m = m.dropna(
            subset=[
                "lon_plot",
                "lat_plot",
                "R_value",
            ]
        ).copy()

        m = m[
            m["thermal_group"].isin(
                ["UHI", "UCI"]
            )
        ].copy()

        map_data[spec["col"]] = m

        # --------------------------------------------------------
        # Use all UHI + UCI values together to calculate one
        # common colour range for this outcome.
        #
        # This is identical to the existing 3-panel calculation.
        # --------------------------------------------------------
        vals = (
            m["R_value"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(vals):
            lim = float(
                np.nanpercentile(
                    np.abs(
                        vals.to_numpy(dtype=float)
                    ),
                    98,
                )
            )

            if not np.isfinite(lim) or lim <= 0:
                lim = float(
                    np.nanmax(
                        np.abs(
                            vals.to_numpy(dtype=float)
                        )
                    )
                )
        else:
            lim = 1.0

        if spec["min_abs_lim"] is not None:
            lim = max(
                lim,
                float(spec["min_abs_lim"]),
            )

        if not np.isfinite(lim) or lim <= 0:
            lim = 1.0

        spec["vmin"] = -lim
        spec["vmax"] = +lim

    # ------------------------------------------------------------
    # 4. Plot configuration
    # ------------------------------------------------------------
    apply_nature_style()

    plot_projection = (
        ccrs.Robinson(central_longitude=0)
        if HAS_CARTOPY
        else None
    )

    local_cmap = plt.cm.RdYlBu_r

    # UHI on the left and UCI on the right.
    group_order = ["UHI", "UCI"]

    fig = plt.figure(
        figsize=(11.2, 8.2),
        dpi=600,
    )

    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.26,
        wspace=0.055,
        left=0.045,
        right=0.925,
        top=0.945,
        bottom=0.070,
    )

    letters = [
        ["a", "b"],
        ["c", "d"],
        ["e", "f"],
    ]

    # Retain one available scatter object per metric row.
    # It is used to make one shared colour bar for UHI and UCI.
    row_axes = []
    row_scatters = []

    # ------------------------------------------------------------
    # 5. Draw 6 map panels
    # ------------------------------------------------------------
    for row_idx, spec in enumerate(metric_specs):
        row_axes_current = []
        row_scatter = None

        all_metric_data = map_data[spec["col"]]

        for col_idx, group_name in enumerate(group_order):
            if HAS_CARTOPY:
                ax = fig.add_subplot(
                    gs[row_idx, col_idx],
                    projection=plot_projection,
                )
            else:
                ax = fig.add_subplot(
                    gs[row_idx, col_idx]
                )

            group_data = all_metric_data[
                all_metric_data["thermal_group"]
                .eq(group_name)
            ].copy()

            sc = draw_R_map_panel(
                ax=ax,
                df_map=group_data,
                title=(
                    f"{spec['name']} response — "
                    f"{group_name}"
                ),
                cmap=local_cmap,
                vmin=spec["vmin"],
                vmax=spec["vmax"],
                tick_fmt=spec["tick_fmt"],

                # Latitude labels only on the left column.
                show_y_labels=(col_idx == 0),
            )

            if row_scatter is None and sc is not None:
                row_scatter = sc

            ax.text(
                -0.045,
                1.035,
                letters[row_idx][col_idx],
                transform=ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )

            row_axes_current.append(ax)

        row_axes.append(row_axes_current)
        row_scatters.append(row_scatter)

    # ------------------------------------------------------------
    # 6. One shared colour bar per metric row
    #
    # UHI and UCI use exactly the same scale within each row.
    # ------------------------------------------------------------
    fig.canvas.draw()

    for row_idx, spec in enumerate(metric_specs):
        sc = row_scatters[row_idx]

        if sc is None:
            continue

        right_ax = row_axes[row_idx][1]
        bbox = right_ax.get_position()

        cax = fig.add_axes([
            bbox.x1 + 0.010,
            bbox.y0,
            0.012,
            bbox.height,
        ])

        cbar = fig.colorbar(
            sc,
            cax=cax,
            orientation="vertical",
            extend="both",
        )

        cbar.set_label(
            spec["cbar_label"],
            fontsize=6.5,
            labelpad=2.5,
        )

        cbar.ax.tick_params(
            labelsize=6.0,
            length=1.5,
            pad=1,
        )

        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor("black")

        cbar.ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        cbar.ax.yaxis.set_major_formatter(
            FormatStrFormatter(
                spec["tick_fmt"]
            )
        )

    # ------------------------------------------------------------
    # 7. Common definition note
    # ------------------------------------------------------------
    fig.text(
        0.045,
        0.020,
        (
            r"$R_X=(X_{urban}-X_{rural})_{HW}"
            r"-(X_{urban}-X_{rural})_{NHW}$; "
            "positive values indicate a larger "
            "urban-relative burden during HW."
        ),
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#444444",
    )

    # Keep the same output basename so downstream file paths
    # do not need to change.
    return savefig_both(
        fig,
        out_dir,
        "climatezone_R_maps_HW_minus_NHW",
    )

def make_hw_nhw_period_specific_map_figure(
        df_pair_impact,
        df_hw_diurnal,
        df_pair_group,
        out_dir,
        diag):
    """
    Reproduce the old six-panel period-specific impact map.

    Scientific quantity
    -------------------
    Each panel shows the urban-rural contrast within one period:

        Delta X_period
        =
        X_urban,period - X_rural,period

    This is different from the separate R-map figure:

        R_X
        =
        Delta X_HW - Delta X_NHW

    Layout
    ------
                              Non-heatwave        Heatwave
        row 1: Sleep loss          a                  b
        row 2: Labour loss         c                  d
        row 3: CDH                 e                  f

    Data rules
    ----------
    - Uses the existing pair-period contrasts in df_pair_impact.
    - Does not recalculate sleep, labour or CDH.
    - Uses all geographical regions together.
    - Does not split or filter by UHI/UCI.
    - For each outcome, a pair must have finite values in both
      NHW and HW.
    - Missing values are never converted to zero.
    - NHW and HW share the same colour scale within each outcome.
    """

    # ------------------------------------------------------------
    # 1. Pair locations
    # ------------------------------------------------------------
    loc = build_pair_location_table(
        df_hw_diurnal,
        df_pair_group,
    )

    loc["pair_id"] = (
        loc["pair_id"]
        .astype(str)
        .str.strip()
    )

    # Only location fields are required for this all-region figure.
    loc = (
        loc[
            [
                "pair_id",
                "lon_plot",
                "lat_plot",
            ]
        ]
        .drop_duplicates("pair_id")
        .copy()
    )

    # ------------------------------------------------------------
    # 2. Existing pair-period urban-rural contrasts
    # ------------------------------------------------------------
    metric_specs = [
        {
            "col": "d_sleep_loss",
            "name": "Sleep loss",
            "cbar_label": (
                r"$\Delta$Sleep loss "
                r"(urban$-$rural, min/night)"
            ),
            "tick_fmt": "%.2f",
            "min_abs_lim": None,
        },
        {
            "col": "d_labour_loss_tx",
            "name": "Labour loss",
            "cbar_label": (
                r"$\Delta$Labour loss "
                r"(urban$-$rural, %)"
            ),
            "tick_fmt": "%.2f",
            "min_abs_lim": 0.001,
        },
        {
            "col": "d_cdh_total",
            "name": "CDH",
            "cbar_label": (
                r"$\Delta$CDH "
                r"(urban$-$rural, °C·h/day)"
            ),
            "tick_fmt": "%.2f",
            "min_abs_lim": None,
        },
    ]

    periods = [
        ("NHW", "Non-heatwave"),
        ("HW", "Heatwave"),
    ]

    # ------------------------------------------------------------
    # 3. Build period-specific map data
    #
    # Require outcome-specific complete NHW/HW pairs.
    # Sleep, labour and CDH are matched independently.
    # ------------------------------------------------------------
    map_data = {}
    diagnostic_lines = []

    for spec in metric_specs:
        metric_col = spec["col"]

        required = {
            "pair_id",
            "period_norm",
            metric_col,
        }

        missing = required - set(df_pair_impact.columns)

        if missing:
            raise ValueError(
                f"Cannot draw {spec['name']} period maps; "
                f"missing columns: {sorted(missing)}"
            )

        work = df_pair_impact[
            [
                "pair_id",
                "period_norm",
                metric_col,
            ]
        ].copy()

        work["pair_id"] = (
            work["pair_id"]
            .astype(str)
            .str.strip()
        )

        work["period_norm"] = (
            work["period_norm"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        work[metric_col] = pd.to_numeric(
            work[metric_col],
            errors="coerce",
        )

        work = work[
            work["period_norm"].isin(
                ["NHW", "HW"]
            )
        ].copy()

        work = work[
            np.isfinite(
                work[metric_col].to_numpy(dtype=float)
            )
        ].copy()

        # Do not silently average duplicate pair-period rows.
        duplicate_mask = work.duplicated(
            subset=[
                "pair_id",
                "period_norm",
            ],
            keep=False,
        )

        if duplicate_mask.any():
            examples = (
                work.loc[
                    duplicate_mask,
                    [
                        "pair_id",
                        "period_norm",
                        metric_col,
                    ],
                ]
                .head(20)
            )

            raise ValueError(
                f"{spec['name']}: duplicate pair-period records "
                "were found. Refusing to aggregate them silently.\n"
                + examples.to_string(index=False)
            )

        # Require the same pair to have a finite metric in both periods.
        complete_flag = (
            work
            .groupby(
                "pair_id",
                observed=True,
            )["period_norm"]
            .apply(
                lambda values: {
                    "NHW",
                    "HW",
                }.issubset(set(values))
            )
        )

        complete_ids = set(
            complete_flag[
                complete_flag
            ].index.astype(str)
        )

        work = work[
            work["pair_id"].isin(
                complete_ids
            )
        ].copy()

        values_for_range = []

        for period_norm, period_label in periods:
            period_data = work[
                work["period_norm"].eq(
                    period_norm
                )
            ].copy()

            period_data = period_data.rename(
                columns={
                    metric_col: "delta_value",
                }
            )

            period_data = period_data[
                [
                    "pair_id",
                    "delta_value",
                ]
            ].copy()

            period_data = period_data.merge(
                loc,
                on="pair_id",
                how="left",
                validate="one_to_one",
            )

            period_data = period_data.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            period_data = period_data.dropna(
                subset=[
                    "lon_plot",
                    "lat_plot",
                    "delta_value",
                ]
            ).copy()

            map_data[
                (
                    metric_col,
                    period_norm,
                )
            ] = period_data

            if len(period_data):
                values_for_range.append(
                    period_data[
                        "delta_value"
                    ].to_numpy(dtype=float)
                )

        # --------------------------------------------------------
        # One symmetric colour range shared by NHW and HW.
        # Same rule as the old six-panel figure.
        # --------------------------------------------------------
        if values_for_range:
            all_values = np.concatenate(
                values_for_range
            )

            lim = float(
                np.nanpercentile(
                    np.abs(all_values),
                    98,
                )
            )

            if not np.isfinite(lim) or lim <= 0:
                lim = float(
                    np.nanmax(
                        np.abs(all_values)
                    )
                )
        else:
            lim = 1.0

        if spec["min_abs_lim"] is not None:
            lim = max(
                lim,
                float(spec["min_abs_lim"]),
            )

        if not np.isfinite(lim) or lim <= 0:
            lim = 1.0

        spec["vmin"] = -lim
        spec["vmax"] = +lim

        n_nhw = len(
            map_data[
                (
                    metric_col,
                    "NHW",
                )
            ]
        )

        n_hw = len(
            map_data[
                (
                    metric_col,
                    "HW",
                )
            ]
        )

        diagnostic_lines.append(
            f"{spec['name']}: "
            f"complete HW/NHW pairs={len(complete_ids)}, "
            f"mapped NHW={n_nhw}, mapped HW={n_hw}, "
            f"colour range=[{-lim:+.6f}, {lim:+.6f}]"
        )

    diag.add(
        "Six-panel period-specific impact maps",
        "\n".join(diagnostic_lines),
    )

    # ------------------------------------------------------------
    # 4. Plot configuration copied from the old six-panel map
    # ------------------------------------------------------------
    apply_nature_style()

    plot_projection = (
        ccrs.Robinson(
            central_longitude=0
        )
        if HAS_CARTOPY
        else None
    )

    local_cmap = plt.cm.RdYlBu_r

    fig = plt.figure(
        figsize=(7.2, 6.825),
        dpi=600,
    )

    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.28,
        wspace=0.08,
        left=0.045,
        right=0.885,
        top=0.920,
        bottom=0.057,
    )

    letters = [
        ["a", "b"],
        ["c", "d"],
        ["e", "f"],
    ]

    row_right_axes = {}
    row_scatters = {}

    # ------------------------------------------------------------
    # 5. Draw the six panels
    # ------------------------------------------------------------
    for row_idx, spec in enumerate(metric_specs):
        row_scatters[row_idx] = None

        for col_idx, (
                period_norm,
                period_label) in enumerate(periods):

            if HAS_CARTOPY:
                ax = fig.add_subplot(
                    gs[row_idx, col_idx],
                    projection=plot_projection,
                )
            else:
                ax = fig.add_subplot(
                    gs[row_idx, col_idx]
                )

            setup_map_axis(
                ax,
                show_y_labels=(
                    col_idx == 0
                ),
            )

            panel_data = map_data[
                (
                    spec["col"],
                    period_norm,
                )
            ]

            if len(panel_data) == 0:
                ax.set_title(
                    (
                        f"{spec['name']} — "
                        f"{period_label} "
                        "(no complete data)"
                    ),
                    fontsize=6.8,
                    fontweight="bold",
                    pad=4.0,
                )

                sc = None

            else:
                scatter_kw = dict(
                    c=panel_data[
                        "delta_value"
                    ].to_numpy(dtype=float),
                    cmap=local_cmap,
                    vmin=spec["vmin"],
                    vmax=spec["vmax"],

                    # Old-map marker format.
                    s=8.0,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.15,
                    alpha=0.85,
                    zorder=3,
                )

                if HAS_CARTOPY:
                    scatter_kw["transform"] = (
                        ccrs.PlateCarree()
                    )

                sc = ax.scatter(
                    panel_data[
                        "lon_plot"
                    ].to_numpy(dtype=float),
                    panel_data[
                        "lat_plot"
                    ].to_numpy(dtype=float),
                    **scatter_kw,
                )

                ax.set_title(
                    (
                        f"{spec['name']} — "
                        f"{period_label}"
                    ),
                    fontsize=6.8,
                    fontweight="bold",
                    pad=4.0,
                )

            if (
                row_scatters[row_idx] is None
                and sc is not None
            ):
                row_scatters[row_idx] = sc

            ax.text(
                -0.04,
                1.04,
                letters[row_idx][col_idx],
                transform=ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )

            if col_idx == 1:
                row_right_axes[row_idx] = ax

    # ------------------------------------------------------------
    # 6. One shared vertical colour bar per metric row
    # ------------------------------------------------------------
    fig.canvas.draw()

    for row_idx, spec in enumerate(metric_specs):
        sc = row_scatters.get(row_idx)
        right_ax = row_right_axes.get(row_idx)

        if sc is None or right_ax is None:
            continue

        bbox = right_ax.get_position()

        cax = fig.add_axes(
            [
                bbox.x1 + 0.010,
                bbox.y0,
                0.012,
                bbox.height,
            ]
        )

        cbar = fig.colorbar(
            sc,
            cax=cax,
            orientation="vertical",
            extend="both",
        )

        cbar.set_label(
            spec["cbar_label"],
            fontsize=6.5,
            labelpad=2.5,
        )

        cbar.ax.tick_params(
            labelsize=6.0,
            length=1.5,
            width=0.5,
            pad=1.0,
        )

        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor("black")

        cbar.ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=5
            )
        )

        cbar.ax.yaxis.set_major_formatter(
            FormatStrFormatter(
                spec["tick_fmt"]
            )
        )

    # ------------------------------------------------------------
    # 7. Figure title
    # ------------------------------------------------------------
    fig.suptitle(
        (
            "Spatial distribution of urban–rural contrasts "
            "in heat-related impacts during\n"
            "non-heatwave and heatwave periods"
        ),
        fontsize=8.3,
        fontweight="bold",
        y=0.985,
    )

    # Keep the old output basename, but write it to the current
    # method-freeze OUTPUT_DIR.
    output_paths = savefig_both(
        fig,
        out_dir,
        "climatezone_delta_maps_aligned_professional",
    )

    plt.close(fig)

    return output_paths
# ============================================================================
# §9 Counts txt
# ============================================================================


def count_complete_stations_by_metric(df_pair_impact, metric_col):
    df = keep_pairs_with_both_periods(df_pair_impact, metric_col)
    tab = (
        df.drop_duplicates(["pair_id", "climate4", "thermal_group"])
        .groupby(["climate4", "thermal_group"], observed=True)
        .agg(n_pairs=("pair_id", "nunique")).reset_index()
    )
    tab["n_stations"] = tab["n_pairs"]
    return tab[["climate4", "thermal_group", "n_stations", "n_pairs"]]


def count_complete_diurnal_stations(df_diurnal_long):
    tab = (df_diurnal_long.drop_duplicates(["station_id", "pair_id", "climate4", "thermal_group"])
             .groupby(["climate4", "thermal_group"], observed=True)
             .agg(n_stations=("station_id", "nunique"),
                  n_pairs=("pair_id", "nunique"))
             .reset_index())
    return tab


def format_count_table(tab):
    if tab is None or len(tab) == 0:
        return "No complete data.\n"

    idx = pd.MultiIndex.from_product([CLIMATE_ORDER, THERMAL_ORDER],
                                     names=["climate4", "thermal_group"])
    tab = tab.set_index(["climate4", "thermal_group"]).reindex(idx).fillna(0).reset_index()
    tab["climate_name"] = tab["climate4"].map(CLIMATE_NAME)
    tab["n_stations"] = tab["n_stations"].astype(int)
    tab["n_pairs"] = tab["n_pairs"].astype(int)

    lines = []
    lines.append("climate4\tclimate_name\tgroup\tn_stations\tn_pairs")
    for _, r in tab.iterrows():
        lines.append(f"{r['climate4']}\t{r['climate_name']}\t{r['thermal_group']}\t"
                     f"{int(r['n_stations'])}\t{int(r['n_pairs'])}")
    return "\n".join(lines) + "\n"



def write_counts_txt(df_diurnal_long, df_pair_impact, out_dir, diag):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "climatezone_uhi_uci_counts_hw_nhw.txt")
    sections = []
    diurnal_counts = count_complete_diurnal_stations(df_diurnal_long)
    sections.append("## Plotted diurnal stations: strict reference pairs with valid HW and NHW\n")
    sections.append(format_count_table(diurnal_counts))
    for metric_col, title in [
        ("d_sleep_loss", "Sleep-loss boxplot pairs"),
        ("d_labour_loss_tx", "Dunne Tx labour-loss boxplot pairs"),
        ("d_cdh_total", "CDH boxplot pairs"),
    ]:
        tab = count_complete_stations_by_metric(df_pair_impact, metric_col)
        sections.append(f"\n## {title}: valid HW and NHW for {metric_col}\n")
        sections.append(format_count_table(tab))
    content = "\n".join([
        "UHI/UCI plotted counts by climate zone",
        "======================================", "",
        "Definitions:",
        "- UHI/UCI group is the strict annual percentile canonical group.",
        "- HW/NHW analyses start from the Fig23 matched cohort complete in dTmean, dAmp1, dTx and dTn.",
        "- All three impact boxplots are pair-level urban-rural differences.",
        "- Labour is positive Dunne labour loss at each station-specific FFT Tx hour.", "",
        *sections,
    ])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    diag.add("Write UHI/UCI count txt", out_path)
    print(f"  ✓ counts written: {out_path}")




# ============================================================================
# §9b Check NHW definition: whether NHW equals warm season minus HW
# ============================================================================
def check_nhw_is_jja_minus_hw(df_cdh, df_hw_diurnal, out_dir, diag):
    """
    Diagnose whether NHW equals the explicit upstream warm-season dates minus
    HW dates. The internal label ``JJA`` is retained only as a backward-
    compatible alias for ``warm_season``; no calendar-month [6,7,8] fallback
    is permitted because Southern Hemisphere warm seasons are DJF.
    """
    ensure_dir(out_dir)

    daily_rows = []
    if df_cdh is not None and len(df_cdh) and "local_date" in df_cdh.columns:
        d = df_cdh.copy()
        d["pair_id"] = d["pair_id"].astype(str).str.strip()
        if "period" in d.columns:
            d["period_norm"] = d["period"].apply(norm_period)
        else:
            d["period_norm"] = d["period_norm"].apply(norm_period)
        d["local_date"] = pd.to_datetime(d["local_date"], errors="coerce").dt.normalize()
        d = d[d["local_date"].notna()].copy()

        for pair_id, g in d.groupby("pair_id", observed=True):
            warm_dates = set(g.loc[g["period_norm"] == "JJA", "local_date"])
            hw_dates = set(g.loc[g["period_norm"] == "HW", "local_date"])
            nhw_dates = set(g.loc[g["period_norm"] == "NHW", "local_date"])
            if not warm_dates:
                daily_rows.append({
                    "pair_id": pair_id,
                    "status": "UNVERIFIED_NO_EXPLICIT_WARM_SEASON_ROWS",
                    "ok_nhw_equals_warm_minus_hw": False,
                    "n_warm_dates": 0,
                    "n_hw_dates": len(hw_dates),
                    "n_nhw_dates": len(nhw_dates),
                })
                continue

            expected_nhw = warm_dates - hw_dates
            missing = expected_nhw - nhw_dates
            extra = nhw_dates - expected_nhw
            overlap = hw_dates & nhw_dates
            outside = nhw_dates - warm_dates
            ok = not (missing or extra or overlap or outside)
            daily_rows.append({
                "pair_id": pair_id,
                "status": "PASS" if ok else "FAIL",
                "ok_nhw_equals_warm_minus_hw": ok,
                "n_warm_dates": len(warm_dates),
                "n_hw_dates": len(hw_dates),
                "n_nhw_dates": len(nhw_dates),
                "n_expected_nhw_dates": len(expected_nhw),
                "n_missing_from_nhw": len(missing),
                "n_extra_in_nhw": len(extra),
                "n_overlap_hw_nhw": len(overlap),
                "n_nhw_outside_warm": len(outside),
            })

    daily_summary = pd.DataFrame(daily_rows)
    daily_csv = os.path.join(out_dir, "check_nhw_definition_daily_pair_summary.csv")
    if len(daily_summary):
        daily_summary.to_csv(daily_csv, index=False)

    ndays_rows = []
    if (df_hw_diurnal is not None and len(df_hw_diurnal) and
            all(c in df_hw_diurnal.columns for c in
                ["pair_id", "station_type", "period_norm", "n_days"])):
        h = df_hw_diurnal.copy()
        h["pair_id"] = h["pair_id"].astype(str).str.strip()
        h["station_side"] = h["station_type"].astype(str)
        h["period_norm"] = h["period_norm"].apply(norm_period)
        h["n_days"] = pd.to_numeric(h["n_days"], errors="coerce")
        pvt = h.pivot_table(
            index=["pair_id", "station_side"], columns="period_norm",
            values="n_days", aggfunc="sum",
        ).reset_index()
        for c in ["JJA", "HW", "NHW"]:
            if c not in pvt.columns:
                pvt[c] = np.nan
        complete = pvt[["JJA", "HW", "NHW"]].notna().all(axis=1)
        pvt["n_days_expected_warm_from_hw_plus_nhw"] = pvt["HW"] + pvt["NHW"]
        pvt["n_days_diff_warm_minus_hw_plus_nhw"] = (
            pvt["JJA"] - pvt["n_days_expected_warm_from_hw_plus_nhw"]
        )
        pvt["ok_n_days_warm_equals_hw_plus_nhw"] = complete & np.isclose(
            pvt["n_days_diff_warm_minus_hw_plus_nhw"], 0.0, atol=1e-6,
        )
        ndays_rows = pvt

    ndays_summary = pd.DataFrame(ndays_rows)
    ndays_csv = os.path.join(out_dir, "check_nhw_definition_station_ndays_summary.csv")
    if len(ndays_summary):
        ndays_summary.to_csv(ndays_csv, index=False)

    txt_path = os.path.join(out_dir, "check_nhw_definition_warm_season_minus_hw.txt")
    lines = [
        "Check whether NHW equals explicit warm season minus HW",
        "====================================================",
        "",
        "Internal period label JJA is a backward-compatible alias for warm_season.",
        "No calendar-month JJA fallback is used; SH warm seasons remain DJF.",
        "",
    ]
    if len(daily_summary):
        lines += [
            f"Daily pairs checked: {len(daily_summary)}",
            f"PASS: {int((daily_summary['status'] == 'PASS').sum())}",
            f"FAIL: {int((daily_summary['status'] == 'FAIL').sum())}",
            "UNVERIFIED_NO_EXPLICIT_WARM_SEASON_ROWS: "
            f"{int((daily_summary['status'] == 'UNVERIFIED_NO_EXPLICIT_WARM_SEASON_ROWS').sum())}",
            f"Daily detail CSV: {daily_csv}",
            "",
        ]
    else:
        lines += ["Daily set check unavailable.", ""]
    if len(ndays_summary):
        lines += [
            f"Pair×station count rows: {len(ndays_summary)}",
            "Rows passing warm season = HW + NHW: "
            f"{int(ndays_summary['ok_n_days_warm_equals_hw_plus_nhw'].sum())}",
            f"Count detail CSV: {ndays_csv}",
        ]
    else:
        lines.append("n_days count check unavailable.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    diag.add("Check NHW definition", f"txt={txt_path}\ndaily_csv={daily_csv}\nndays_csv={ndays_csv}")
    print(f"  ✓ NHW definition check written: {txt_path}")
    return txt_path




# ============================================================================
# §9c Check CDH boxplot unit
# ============================================================================
def check_cdh_boxplot_is_degree_hours_per_day(df_cdh, df_station_clim, out_dir, diag):
    """
    Diagnostic check for the CDH boxplot unit.

    The plotted CDH boxplot uses:
        value_col = "CDH_total_mean"

    In this script, CDH_total_mean is produced by:
        1) reading all_pairs_daily_panel.csv;
        2) stationising daily columns CDH_total_u / CDH_total_r;
        3) grouping by station_id × period_norm;
        4) taking the arithmetic mean across daily rows.

    Therefore, if CDH_total_u / CDH_total_r are daily accumulated degree-hours,
    the plotted boxplot unit is:
        °C·h/day

    This diagnostic recomputes CDH_total_mean directly from the raw daily panel
    and compares it with the values in df_station_clim.
    """
    ensure_dir(out_dir)

    txt_path = os.path.join(out_dir, "check_cdh_boxplot_unit.txt")
    csv_path = os.path.join(out_dir, "check_cdh_boxplot_unit_station_period.csv")

    lines = []
    lines.append("Check CDH boxplot unit")
    lines.append("======================")
    lines.append("")
    lines.append("Conclusion to verify:")
    lines.append("  The CDH boxplot is plotted as mean daily CDH, i.e. °C·h/day.")
    lines.append("")
    lines.append("Reason:")
    lines.append("  CDH_total_mean is computed as the mean of daily CDH_total_u/CDH_total_r")
    lines.append("  across local_date within each station × period.")
    lines.append("")

    if df_cdh is None or len(df_cdh) == 0:
        lines.append("Status: FAILED TO CHECK because df_cdh is empty.")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        diag.add("Check CDH boxplot unit", f"failed: empty df_cdh\n{txt_path}")
        return txt_path

    required = ["pair_id", "period", "local_date", "CDH_total_u", "CDH_total_r"]
    missing = [c for c in required if c not in df_cdh.columns]
    if missing:
        lines.append("Status: FAILED TO CHECK because required columns are missing:")
        lines.append("  " + ", ".join(missing))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        diag.add("Check CDH boxplot unit", f"failed: missing columns {missing}\n{txt_path}")
        return txt_path

    d = df_cdh.copy()
    d["pair_id"] = d["pair_id"].astype(str)
    d["period_norm"] = d["period"].apply(norm_period)
    d["local_date"] = pd.to_datetime(d["local_date"], errors="coerce").dt.normalize()
    d = d[d["period_norm"].isin(PLOT_PERIODS)].copy()

    rows = []
    for _, r in d.iterrows():
        for side, col in [("urban", "CDH_total_u"), ("rural", "CDH_total_r")]:
            rows.append({
                "pair_id": r["pair_id"],
                "station_side": side,
                "station_id": f"{r['pair_id']}_{side}",
                "period_norm": r["period_norm"],
                "local_date": r["local_date"],
                "CDH_total_daily": r[col],
            })

    daily = pd.DataFrame(rows)
    daily["CDH_total_daily"] = pd.to_numeric(daily["CDH_total_daily"], errors="coerce")

    recomputed = (
        daily.dropna(subset=["CDH_total_daily"])
             .groupby(["station_id", "pair_id", "station_side", "period_norm"], observed=True)
             .agg(
                 n_days=("local_date", "nunique"),
                 cdh_mean_recomputed=("CDH_total_daily", "mean"),
                 cdh_sum_over_period=("CDH_total_daily", "sum"),
                 cdh_median_daily=("CDH_total_daily", "median")
             )
             .reset_index()
    )

    plotted = df_station_clim[
        ["station_id", "pair_id", "station_side", "period_norm", "climate4", "thermal_group", "CDH_total_mean"]
    ].copy()
    plotted["pair_id"] = plotted["pair_id"].astype(str)
    plotted = plotted[plotted["period_norm"].isin(PLOT_PERIODS)].copy()
    plotted["CDH_total_mean"] = pd.to_numeric(plotted["CDH_total_mean"], errors="coerce")

    chk = plotted.merge(
        recomputed,
        on=["station_id", "pair_id", "station_side", "period_norm"],
        how="left"
    )
    chk["difference_plotted_minus_recomputed"] = chk["CDH_total_mean"] - chk["cdh_mean_recomputed"]
    chk["abs_difference"] = chk["difference_plotted_minus_recomputed"].abs()
    chk["is_mean_daily_match"] = np.isclose(
        chk["CDH_total_mean"].fillna(np.inf),
        chk["cdh_mean_recomputed"].fillna(-np.inf),
        rtol=1e-10,
        atol=1e-10
    )
    chk["period_total_if_needed"] = chk["CDH_total_mean"] * chk["n_days"]

    chk.to_csv(csv_path, index=False)

    n_rows = len(chk)
    n_matched = int(chk["is_mean_daily_match"].sum())
    n_missing_recomputed = int(chk["cdh_mean_recomputed"].isna().sum())
    max_abs_diff = chk["abs_difference"].max()

    lines.append("Computation chain used by the current boxplot:")
    lines.append("  all_pairs_daily_panel.csv")
    lines.append("    -> CDH_total_u / CDH_total_r per local_date")
    lines.append("    -> station-period aggregation by mean")
    lines.append("    -> plotted value_col='CDH_total_mean'")
    lines.append("")
    lines.append("Therefore:")
    lines.append("  CDH_total_mean = average daily CDH over the selected period.")
    lines.append("  Boxplot unit = °C·h/day, not °C·h per full HW/NHW period.")
    lines.append("")
    lines.append("Verification against raw daily panel:")
    lines.append(f"  Station-period rows checked: {n_rows}")
    lines.append(f"  Rows exactly matching recomputed daily mean: {n_matched}")
    lines.append(f"  Rows missing recomputed daily mean: {n_missing_recomputed}")
    lines.append(f"  Maximum absolute difference: {max_abs_diff}")
    lines.append(f"  Detailed CSV: {csv_path}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  If the source daily columns CDH_total_u/CDH_total_r are daily accumulated")
    lines.append("  degree-hours, then the plotted CDH boxplot is correctly labelled as")
    lines.append("  CDH (°C·h/day).")
    lines.append("")
    lines.append("  If you want period-total CDH instead, use:")
    lines.append("      CDH_total_mean × n_days")
    lines.append("  The diagnostic CSV includes this as 'period_total_if_needed'.")
    lines.append("")

    if n_rows > 0 and n_matched == n_rows:
        lines.append("Status: PASS — plotted CDH_total_mean equals the recomputed mean daily CDH.")
    elif n_rows > 0 and n_matched + n_missing_recomputed == n_rows:
        lines.append("Status: PARTIAL — all non-missing recomputed rows match, but some rows lack raw daily CDH.")
    else:
        lines.append("Status: CHECK DIFFERENCES — some plotted values differ from recomputed daily means.")
        bad = chk[~chk["is_mean_daily_match"] & chk["cdh_mean_recomputed"].notna()].copy()
        if len(bad) > 0:
            lines.append("")
            lines.append("First 20 differing rows:")
            cols = [
                "station_id", "period_norm", "climate4", "thermal_group",
                "CDH_total_mean", "cdh_mean_recomputed",
                "difference_plotted_minus_recomputed", "n_days"
            ]
            lines.append(bad[cols].head(20).to_string(index=False))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    diag.add(
        "Check CDH boxplot unit",
        f"txt={txt_path}\n"
        f"csv={csv_path}\n"
        f"rows={n_rows}, matched={n_matched}, missing_recomputed={n_missing_recomputed}"
    )
    print(f"  ✓ CDH unit check written: {txt_path}")
    return txt_path



# ============================================================================
# §10 Main
# ============================================================================
def main():
    t0 = time.time()
    diag = DiagnosticLog()
    ensure_dir(OUTPUT_DIR)

    print("=" * 88)
    print("Climate-zone UHI/UCI HW-NHW standalone plotting script")
    print("  · four climate-zone diurnal panels")
    print("  · sleep loss / labour loss / CDH boxplots")
    print("  · only HW and NHW")
    print("  · only stations with both HW and NHW data")
    print("  · UHI red, UCI blue")
    print("  · 3-panel R maps: sleep loss, labour loss, CDH")
    print("=" * 88)

    print("\n[1/8] Loading source data...")
    df_hw_diurnal = load_hw_diurnal_csv(diag)
    df_pair_group = load_canonical_annual_groups(V6_METRICS_CSV, diag=diag)
    matched_ids = load_reference_matched_pair_ids(V6_METRICS_CSV, diag=diag)
    df_hw_diurnal = df_hw_diurnal[df_hw_diurnal["pair_id"].astype(str).isin(matched_ids)].copy()

    df_cdh = load_cdh_panel(diag)
    df_lc = load_labour_full(diag)
    df_hne = load_hne_paired(diag)

    print("\n[Check] Verifying whether NHW is warm season excluding HW days...")
    # check_nhw_is_jja_minus_hw(df_cdh, df_hw_diurnal, OUTPUT_DIR, diag)

    print("\n[2/8] Building station-period metric panel...")
    df_station_daily = build_station_daily_from_cdh(df_cdh, diag)
    df_phys = aggregate_station_period_from_daily(df_station_daily, diag)
    df_hne_st = build_station_period_from_hne(df_hne, diag)
    df_lab_st = build_station_period_from_labour(df_lc, diag)
    df_station = merge_station_modules(df_phys, df_hne_st, df_lab_st, diag)

    print("\n[3/8] Merging climate labels and UHI/UCI group labels...")
    df_station_clim = merge_climate_group_to_station_panel(
        df_station, df_hw_diurnal, df_pair_group, diag
    )
    df_station_clim = df_station_clim[df_station_clim["period_norm"].isin(PLOT_PERIODS)].copy()

    df_pair_impact = build_pair_impact_panel(
        df_station_clim,
        matched_ids,
        diag,
    )

    # Labour-specific matched HW/NHW requirement.
    # This affects all downstream labour boxplots, maps and count tables.
    df_pair_impact = enforce_labour_hw_nhw_complete_pairs(
        df_pair_impact,
        value_col="d_labour_loss_tx",
        pair_col="pair_id",
        period_col="period_norm",
        diag=diag,
    )

    pair_impact_out = os.path.join(
        OUTPUT_DIR,
        "pair_period_impact_panel_uhi_uci_hw_nhw.csv",
    )
    df_pair_impact.to_csv(pair_impact_out, index=False)
    print(f"  ✓ pair impact panel written: {pair_impact_out}")

    station_panel_out = os.path.join(OUTPUT_DIR, "station_period_panel_uhi_uci_hw_nhw.csv")
    df_station_clim.to_csv(station_panel_out, index=False)
    print(f"  ✓ station panel written: {station_panel_out}")

    print("\n[Check] Verifying CDH boxplot unit...")
    check_cdh_boxplot_is_degree_hours_per_day(df_cdh, df_station_clim, OUTPUT_DIR, diag)

    print("\n[4/8] Building complete-case diurnal summary...")
    df_diurnal_long = build_diurnal_long_hw_nhw_uhi_uci(df_hw_diurnal, df_pair_group, diag)
    df_diurnal_summary = summarize_diurnal(df_diurnal_long)

    diurnal_long_out = os.path.join(OUTPUT_DIR, "diurnal_long_uhi_uci_hw_nhw_complete.csv")
    df_diurnal_long.to_csv(diurnal_long_out, index=False)
    print(f"  ✓ diurnal long table written: {diurnal_long_out}")

    print("\n[5/8] Writing UHI/UCI station counts...")
    write_counts_txt(df_diurnal_long, df_pair_impact, OUTPUT_DIR, diag)

    print("\n[6/8] Plotting diurnal + boxplot figure...")
    make_diurnal_box_figure(df_diurnal_summary, df_pair_impact, OUTPUT_DIR, diag)

    print("\n[7/8] Plotting 3-panel R map figure...")
    make_R_map_figure(df_pair_impact, df_hw_diurnal, df_pair_group, OUTPUT_DIR, diag)

    print("\n[7b/8] Plotting 6-panel NHW/HW impact maps...")
    make_hw_nhw_period_specific_map_figure(
        df_pair_impact,
        df_hw_diurnal,
        df_pair_group,
        OUTPUT_DIR,
        diag,
    )

    print("\n[8/8] Writing diagnostic log...")
    diag.write(os.path.join(OUTPUT_DIR, "diagnostic_climatezone_uhi_uci_hw_nhw.md"))

    print("\n" + "=" * 88)
    print(f"Done. Elapsed: {time.time() - t0:.1f} s")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Files:")
    print("  climatezone_uhi_uci_diurnal_box_hw_nhw.png")
    print("  climatezone_uhi_uci_diurnal_box_hw_nhw.pdf")
    print("  climatezone_R_maps_sleep_labour loss_CDH_hw_nhw.png")
    print("  climatezone_R_maps_sleep_labour loss_CDH_hw_nhw.pdf")
    print("  climatezone_uhi_uci_counts_hw_nhw.txt")
    print("  check_nhw_definition_jja_minus_hw.txt")
    print("  check_nhw_definition_daily_pair_summary.csv")
    print("  check_nhw_definition_station_ndays_summary.csv")
    print("  check_cdh_boxplot_unit.txt")
    print("  check_cdh_boxplot_unit_station_period.csv")
    print("  station_period_panel_uhi_uci_hw_nhw.csv")
    print("  diurnal_long_uhi_uci_hw_nhw_complete.csv")
    print("  diagnostic_climatezone_uhi_uci_hw_nhw.md")
    print("=" * 88)


if __name__ == "__main__":
    main()

