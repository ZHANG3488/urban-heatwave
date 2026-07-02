#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_result_fig1.py
========================================================================
Updates from v3:
- Map land/ocean colours adjusted to match reference (grey land, light-blue ocean)
- Climate zone labels in b-e panels moved ABOVE the subplot boundary
- GridSpec extended to 5 rows (height_ratios=[2.90, 1.02, 1.06, 1.06, 1.06])
- Row 3 (h, i)  : CDH and HDH boxplots (each spans 2 symmetric columns)
- Row 4 (j, k)  : Sleep-loss and Economic-loss boxplots (symmetric 2+2 cols)
- All non-map rows use symmetric 2+2 column spans → left/right alignment
- build_station_period_from_hne now also stationises total_loss_pct
- Figure height scaled to 11.0 to accommodate extra row

[01-main unified input patch]
- Diurnal profiles and map metrics are constructed directly from
  01_main_pair_period_metrics.py output (all_pair_period_metrics.csv).
- The pair-period table is reshaped in memory to the station-period interface
  expected by the existing plotting functions; no HW or FFT metric is recomputed.
- Canonical UHI/UCI labels remain the annual two-harmonic FFT dTx groups from 01.
- The retired mortality module is not loaded or merged.

[Method-freeze final patch]
- Preserve pair_id and period as strings when reading formal labour outputs.
- Keep unmatched HW/NHW scientific responses as NaN (never zero-fill).
- Remove the fixed 92-day warm-season fallback; use upstream NH-JJA/SH-DJF
  day counts only.
"""

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as mpl_inset_axes
import matplotlib.path as mpath

warnings.filterwarnings("ignore")

from config import UNIFIED_ROOT

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed, map will use plain lon-lat scatter.")


# ============================================================================
# §0 Paths
# ============================================================================
CDH_DIR       = UNIFIED_ROOT + "/analysis/cdh_energy"
LABOUR_DIR    = UNIFIED_ROOT + "/analysis/labour"
HNE_DIR       = UNIFIED_ROOT + "/analysis/hne_econ/paired/method_pooled"

ECON_HNE_DIR = UNIFIED_ROOT + "/analysis/hne_econ"
PAIR_SLEEP_PEAK_CSV = os.path.join(ECON_HNE_DIR, "pair_period_sleep_lc_peak.csv")

# 01_main_pair_period_metrics.py is the single source of pair-period curves
# and canonical annual UHI/UCI groups.
FFT_BASE_DIR  = UNIFIED_ROOT + "/analysis/main_multiyear/robustness_percentile"


OUTPUT_DIR    = UNIFIED_ROOT + "/plot_data/fig1_n"

ANALYSIS_MULTIYEAR_CSV = UNIFIED_ROOT + "/analysis/main_multiyear/robustness_percentile/all_pair_period_metrics.csv"
# ============================================================================
# §1 Globals
# ============================================================================

PLOT_PROJECTION = ccrs.Robinson(central_longitude=0) if HAS_CARTOPY else None
DATA_CRS = ccrs.PlateCarree() if HAS_CARTOPY else None


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
    "annual": "Annual",
    "JJA": "Warm season",
    "HW": "HW",
    "NHW": "NHW"
}

CLIMATE_ORDER = ["A", "B", "C", "D"]
CLIMATE_NAME = {
    "A": "Tropical",
    "B": "Arid",
    "C": "Temperate",
    "D": "Cold",
}

TOP_PERIODS = ["NHW", "HW"]
BOX_PERIODS = ["NHW", "HW"]
HOURS = np.arange(24)

FONT_FAMILY = "DejaVu Sans"
FONT_BASE = 7

CLR_PERIOD = {
    "HW":  "#b2182b",
    "NHW": "#2166ac",
    "JJA": "#f4a582",
}
CLR_BOX = {
    "HW":  "#b2182b",
    "NHW": "#2166ac",
    "JJA": "#8f8f8f",
}
ZONE_EDGE = {
    "A": "#8c2d04",
    "B": "#cc4c02",
    "C": "#225ea8",
    "D": "#41b6c4",
}
MAP_CMAP = plt.cm.RdYlBu_r

# ── Map colours matching reference (light-grey land, pale-blue ocean) ──────
MAP_LAND_COLOUR  = "#e2ddd4"   # light tan-grey, like NaturalEarth
MAP_OCEAN_COLOUR = "#cfe0ec"   # pale blue, similar to reference
MAP_LAKE_COLOUR  = "#cfe0ec"
MAP_BORDER_COLOUR = "#b0a898"

FIG_W = 7.2
FIG_DPI = 600

FIG_H_COMPOSITE = 9.15
FIG_H_PERIOD_MAPS = 4.55
FIG_H_KG_STATS = 5.05

SAVEFIG_KW = dict(dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.025)


# ============================================================================
# §2 Utilities
# ============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def norm_period(x):
    return PERIOD_NORM.get(str(x), str(x))

def sem_safe(v):
    v = pd.Series(v).dropna()
    if len(v) <= 1:
        return np.nan
    return float(v.std(ddof=1) / np.sqrt(len(v)))

def station_id_from_pair(pair_id, side):
    return f"{pair_id}_{side}"

def climate4_from_row(row):
    for c in ["kg_group", "climate_zone_main", "kg_code"]:
        if c in row.index and pd.notna(row[c]):
            s = str(row[c]).strip().upper()
            if len(s) > 0 and s[0] in {"A", "B", "C", "D"}:
                return s[0]
    return np.nan

def format_ylabel(label, is_delta=False):
    if is_delta:
        return rf"$\Delta$ {label}"
    return label

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
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
    })

def add_panel_label(ax, letter, x=-0.10, y=1.04):
    """Add bold panel letter. For labels inside axes (x,y in [0,1]), pass
    both coordinates in axes fraction. For labels outside, pass negative x."""
    use_transform = ax.transAxes
    ax.text(x, y, letter, transform=use_transform,
            fontsize=8.5, fontweight="bold",
            ha="left", va="bottom",
            clip_on=False)

def style_small_axis(ax):
    # 检查这个 axis 是否是带有地理投影的地图
    if hasattr(ax, 'projection'):

        for spine in ax.spines.values():
            spine.set_visible(False)
            
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        if 'geo' in ax.spines:
            ax.spines['geo'].set_visible(True)
            ax.spines['geo'].set_linewidth(0.65)
            ax.spines['geo'].set_edgecolor("black")
            
    else:
        # --- 情况 B：它是普通的曲线图、柱状图（保持原样） ---
        ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.22)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_edgecolor("black")

def add_shared_horizontal_cbar(fig, mappable, slot_ax, label,
                               width_frac=0.40, height=0.010):
    fig.canvas.draw()
    pos = slot_ax.get_position()

    cbar_width = pos.width * width_frac
    cbar_height = min(height, pos.height * 0.34)

    cbar_x = pos.x0 + (pos.width - cbar_width) / 2.0
    cbar_y = pos.y0 + (pos.height - cbar_height) / 2.0

    cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=5.8, labelpad=1.0)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=4.8, length=1.5, pad=0.8)
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(0.55)
    cax.set_facecolor("none")
    return cbar


class DiagnosticLog:
    def __init__(self):
        self.lines = []
        self.t0 = time.time()

    def add(self, title, text):
        block = f"\n## {title}\n\n```\n{text.rstrip()}\n```\n"
        self.lines.append(block)
        print(f"\n[DIAG] {title}\n{text}")

    def write(self, out_path):
        elapsed = time.time() - self.t0
        content = [
            "# Integrated climate-zone figure diagnostic (v4)",
            f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Elapsed: {elapsed:.1f} s",
            ""
        ]
        content.extend(self.lines)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        print(f"  ✓ diagnostic written: {out_path}")



def load_canonical_annual_groups(path=ANALYSIS_MULTIYEAR_CSV, diag=None):
    """Load strict annual percentile UHI/UCI pair classifications."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    g = pd.read_csv(path, low_memory=False)
    required = {"pair_id", "period", "group"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"Canonical annual group columns missing: {sorted(missing)}")
    if "hw_method" in g.columns:
        g = g[g["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    g = g[g["period"].astype(str).str.lower().str.strip().eq("annual")].copy()
    g["pair_id"] = g["pair_id"].astype(str)
    g["group"] = g["group"].astype(str).str.upper().str.strip()
    g = g[g["group"].isin(["UHI", "UCI"])].copy()
    if g.empty:
        raise ValueError("No annual percentile UHI/UCI rows are available.")
    conflicts = g.groupby("pair_id", observed=True)["group"].nunique()
    conflict_ids = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflict_ids:
        raise ValueError(
            "Conflicting annual UHI/UCI classifications for "
            f"{len(conflict_ids)} pair(s); examples={conflict_ids[:20]}"
        )
    out = g[["pair_id", "group"]].drop_duplicates("pair_id")
    if diag is not None:
        diag.add(
            "Canonical annual UHI/UCI groups",
            f"pairs={len(out)}\ncounts:\n{out['group'].value_counts().to_string()}"
        )
    return out


def apply_canonical_annual_group(df, diag=None):
    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)
    lookup = load_canonical_annual_groups(diag=diag)
    if "group" in out.columns:
        out["group_period_original"] = out["group"]
        out = out.drop(columns=["group"])
    return out.merge(lookup, on="pair_id", how="inner")


def load_reference_matched_pair_ids(path=ANALYSIS_MULTIYEAR_CSV, diag=None):
    """Strict percentile HW/NHW cohort complete in dTmean/dAmp1/dTx/dTn."""
    df = pd.read_csv(path, low_memory=False)
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    metrics = ["dTmean", "dAmp1", "dTx", "dTn"]
    missing = {"pair_id", "period", *metrics} - set(df.columns)
    if missing:
        raise ValueError(f"Matched cohort columns missing: {sorted(missing)}")
    df["pair_id"] = df["pair_id"].astype(str)
    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    p = df["period"].astype(str).str.lower().str.strip()
    nhw = df.loc[p.eq("non_heatwave"), ["pair_id", *metrics]].copy()
    hw = df.loc[p.eq("heatwave"), ["pair_id", *metrics]].copy()
    nhw = nhw.rename(columns={c: f"{c}_nhw" for c in metrics})
    hw = hw.rename(columns={c: f"{c}_hw" for c in metrics})
    paired = nhw.merge(hw, on="pair_id", how="inner")
    complete = [f"{c}_nhw" for c in metrics] + [f"{c}_hw" for c in metrics]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=complete)
    paired = paired.merge(load_canonical_annual_groups(path), on="pair_id", how="inner")
    ids = set(paired["pair_id"].astype(str))
    if not ids:
        raise ValueError("Strict matched HW/NHW cohort is empty.")
    if diag is not None:
        diag.add(
            "Reference matched HW/NHW cohort",
            f"pairs={len(ids)}\nannual group counts:\n{paired['group'].value_counts().to_string()}"
        )
    return ids


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

    # Preserve identifier columns as strings. Only scientific numeric fields
    # are coerced to numeric; converting pair_id/period would destroy the
    # producer-consumer merge keys.
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


# ============================================================================
# §3 Load data   (unchanged from v3)
# ============================================================================


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
    p = os.path.join(LABOUR_DIR, "labour_loss_full.csv")
    if not os.path.exists(p):
        diag.add("Load labour", f"missing: {p}")
        return None
    df = pd.read_csv(p)
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    df["period_norm"] = df["period"].apply(norm_period)
    df = derive_dunne_tx_labour_loss(df)
    df = apply_canonical_annual_group(df, diag=diag)
    diag.add(
        "Load labour",
        f"path={p}\nshape={df.shape}\nmetric=d_labour_loss_tx\n"
        "sign=positive means urban labour loss > rural labour loss"
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
    diag.add("Load HNE", f"shape={df.shape}\ncols={list(df.columns[:20])}")
    return df


# ============================================================================
# §3b  [v4-allstation] Load upstream heatwave diurnal reconstruction
# ============================================================================

def load_hw_diurnal_csv(diag):
    """
    Construct the legacy station-period plotting interface directly from
    01_main_pair_period_metrics.py output.

    This is a pure reshape:
      pair_id × period with urban_* / rural_* columns
        -> pair_id × station_type × period with h00...h23 columns.

    HW/NHW dates, FFT reconstruction, period-specific Tx/Tn and canonical
    annual UHI/UCI labels are not recalculated.
    """
    p = ANALYSIS_MULTIYEAR_CSV
    if not os.path.exists(p):
        raise FileNotFoundError(f"01 main-analysis output not found: {p}")

    df = pd.read_csv(p, low_memory=False)

    if "hw_method" in df.columns:
        df = df[
            df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()

    required = {
        "pair_id", "period", "group",
        "kg_group", "lon_urban", "lat_urban", "lon_rural", "lat_rural",
        "urban_Tmax_fft", "rural_Tmax_fft",
        "urban_Tmin_fft", "rural_Tmin_fft",
        *(f"urban_diurnal_h{h:02d}" for h in range(24)),
        *(f"rural_diurnal_h{h:02d}" for h in range(24)),
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "01 main-analysis interface is incomplete for Figure 1; "
            f"missing columns: {missing}"
        )

    df["pair_id"] = df["pair_id"].astype(str)
    df = apply_canonical_annual_group(df, diag=diag)

    matched_ids = load_reference_matched_pair_ids(path=p, diag=diag)
    df = df[df["pair_id"].isin(matched_ids)].copy()

    station_frames = []
    for side in ("urban", "rural"):
        out = pd.DataFrame(index=df.index)
        out["pair_id"] = df["pair_id"].values
        out["station_type"] = side
        out["period"] = df["period"].values
        out["period_norm"] = df["period"].apply(norm_period).values
        out["group"] = df["group"].values

        if "hw_method" in df.columns:
            out["hw_method"] = df["hw_method"].values

        out["kg_group"] = df["kg_group"].values
        out["kg_code"] = (
            df["kg_code"].values if "kg_code" in df.columns else np.nan
        )
        out["climate_zone"] = (
            df["climate_zone_main"].values
            if "climate_zone_main" in df.columns else np.nan
        )
        out["climate4"] = out["kg_group"].apply(
            lambda x: (
                str(x).strip()[0].upper()
                if pd.notna(x)
                and str(x).strip()
                and str(x).strip()[0].upper() in CLIMATE_ORDER
                else np.nan
            )
        )

        out["lon"] = pd.to_numeric(df[f"lon_{side}"], errors="coerce").values
        out["lat"] = pd.to_numeric(df[f"lat_{side}"], errors="coerce").values
        out["Tmax_fft"] = pd.to_numeric(
            df[f"{side}_Tmax_fft"], errors="coerce"
        ).values
        out["Tmin_fft"] = pd.to_numeric(
            df[f"{side}_Tmin_fft"], errors="coerce"
        ).values
        out["Tmean"] = (
            pd.to_numeric(df[f"{side}_Tmean"], errors="coerce").values
            if f"{side}_Tmean" in df.columns else np.nan
        )
        out["n_days"] = (
            pd.to_numeric(df[f"{side}_ndays"], errors="coerce").values
            if f"{side}_ndays" in df.columns else np.nan
        )

        all_year_col = f"n_valid_years_{side}_all"
        warm_year_col = f"n_valid_years_{side}_warm"
        if all_year_col in df.columns and warm_year_col in df.columns:
            annual_mask = df["period"].astype(str).str.lower().eq("annual")
            out["n_valid_years"] = np.where(
                annual_mask,
                pd.to_numeric(df[all_year_col], errors="coerce"),
                pd.to_numeric(df[warm_year_col], errors="coerce"),
            )
        else:
            out["n_valid_years"] = np.nan

        for h in range(24):
            out[f"h{h:02d}"] = pd.to_numeric(
                df[f"{side}_diurnal_h{h:02d}"], errors="coerce"
            ).values

        station_frames.append(out.reset_index(drop=True))

    station_df = pd.concat(station_frames, ignore_index=True)

    diag.add(
        "Build station-period diurnal interface from 01 main analysis",
        f"path={p}\nshape={station_df.shape}\n"
        f"matched_pairs={station_df['pair_id'].nunique()}\n"
        f"station-period source=all_pair_period_metrics.csv\n"
        f"periods={sorted(station_df['period_norm'].dropna().unique().tolist())}"
    )
    return station_df



def load_fig9_source(diag):
    """Load percentile HW/NHW rows, attach annual groups, and use matched IDs."""
    p = ANALYSIS_MULTIYEAR_CSV
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    df = apply_canonical_annual_group(df, diag=diag)
    ids = load_reference_matched_pair_ids(path=p, diag=diag)
    df = df[
        df["pair_id"].astype(str).isin(ids)
        & df["period"].astype(str).str.lower().isin(["heatwave", "non_heatwave"])
    ].copy()
    diag.add(
        "Load Figure9 source",
        f"path={p}\nshape={df.shape}\nmatched_pairs={df['pair_id'].nunique()}\n"
        f"groups={sorted(df['group'].dropna().unique().tolist())}"
    )
    return df



# ============================================================================
# §4 Build station-period panel   (unchanged from v3)
# ============================================================================


def build_pair_period_labour_loss_tx(df_lc, diag):
    """Extract pair-period Dunne Tx labour loss, positive in loss direction."""
    if df_lc is None or len(df_lc) == 0:
        return pd.DataFrame()
    work = df_lc[df_lc["period_norm"].isin({"HW", "NHW"})].copy()
    if "d_labour_loss_tx" not in work.columns:
        raise ValueError("d_labour_loss_tx is missing from labour input.")
    out = (
        work[["pair_id", "period_norm", "d_labour_loss_tx"]]
        .dropna(subset=["d_labour_loss_tx"])
        .groupby(["pair_id", "period_norm"], observed=True)["d_labour_loss_tx"]
        .mean().reset_index()
    )
    diag.add(
        "Build pair-period labour loss at Tx",
        f"shape={out.shape}\nsource=d_labour_loss_tx"
    )
    return out



def build_station_daily_from_cdh(df_cdh, diag):
    rows = []
    for _, r in df_cdh.iterrows():
        base = {
            "pair_id": r.get("pair_id"),
            "period_norm": r.get("period_norm"),
            "local_date": r.get("local_date"),
        }
        for side, sfx in [("urban", "_u"), ("rural", "_r")]:
            row = base.copy()
            row["station_side"] = side
            row["station_id"] = station_id_from_pair(r.get("pair_id"), side)
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


# ── CHANGED: also stationise economic-loss columns ───────────────────────────
def build_station_period_from_hne(df_hne, diag):
    if df_hne is None or len(df_hne) == 0:
        return pd.DataFrame()

    keep_periods = {"HW", "NHW", "JJA"}
    if "period_norm" in df_hne.columns:
        df_hne = df_hne[df_hne["period_norm"].isin(keep_periods)].copy()

    # Expanded metric specs – sleep loss + econ loss (USD primary, pct fallback)
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
        # pct columns kept as fallback
        "total_loss_pct_mean": {
            "U": ["total_loss_pct_U"],
            "R": ["total_loss_pct_R"],
        },
        "econ_loss_pct_short_mean": {
            "U": ["econ_loss_pct_short_U"],
            "R": ["econ_loss_pct_short_R"],
        },
    }

    # Keep only columns that actually exist
    needed_cols = ["pair_id", "period_norm",
                   "sleep_loss_min_U",      "sleep_loss_min_R",
                   "total_loss_usd_U",      "total_loss_usd_R",
                   "econ_loss_usd_sleep_U", "econ_loss_usd_sleep_R",
                   "econ_loss_usd_heat_U",  "econ_loss_usd_heat_R",
                   "total_loss_pct_U",      "total_loss_pct_R",
                   "econ_loss_pct_short_U", "econ_loss_pct_short_R"]
    needed_cols = [c for c in dict.fromkeys(needed_cols) if c in df_hne.columns]
    df_hne = df_hne[needed_cols].copy()

    diag.add(
        "HNE pre-filter before stationize (v4)",
        f"shape={df_hne.shape}\nperiods={sorted(df_hne['period_norm'].dropna().unique().tolist())}\n"
        f"cols={list(df_hne.columns)}"
    )

    # Filter metric_specs to only those with at least one available column
    active_specs = {}
    for k, v in metric_specs.items():
        all_cands = v.get("U", []) + v.get("R", [])
        if any(c in df_hne.columns for c in all_cands):
            active_specs[k] = v

    df = stationize_pair_period(df_hne, active_specs)
    if len(df) == 0:
        return df

    df = (
        df.groupby(["station_id", "pair_id", "station_side", "period_norm"], observed=True)
          .mean(numeric_only=True)
          .reset_index()
    )
    if "sleep_loss_min_mean" in df.columns:
        df["sleep_loss_min_mean"] = -df["sleep_loss_min_mean"]
        
    diag.add("Build station-period from HNE (v4)", f"shape={df.shape}")
    return df



def build_station_period_from_labour(df_lc, diag):
    """Legacy merge wrapper carrying pair-level Dunne Tx labour loss."""
    if df_lc is None or len(df_lc) == 0:
        return pd.DataFrame()
    tmp = df_lc[df_lc["period_norm"].isin({"HW", "NHW", "JJA"})][
        ["pair_id", "period_norm", "d_labour_loss_tx"]
    ].dropna(subset=["d_labour_loss_tx"]).drop_duplicates(["pair_id", "period_norm"])
    urban = tmp.assign(station_side="urban", station_id=tmp["pair_id"].astype(str)+"_urban")
    rural = tmp.assign(station_side="rural", station_id=tmp["pair_id"].astype(str)+"_rural")
    out = pd.concat([urban, rural], ignore_index=True)
    out["labour_loss_is_pair_delta"] = True
    diag.add("Build station-period from labour", f"shape={out.shape}\nsource=d_labour_loss_tx")
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
def build_pair_period_sleep_delta(df_station_clim, diag):
    """
    从 station-level sleep_loss_min_mean 构建 pair-period 级别城乡差值：
      delta_sleep_min = urban - rural   (min per night)
    输出：
      pair_id, period_norm, climate4, delta_sleep_min
    """
    need_cols = ["pair_id", "period_norm", "station_side", "sleep_loss_min_mean", "climate4"]
    need_cols = [c for c in need_cols if c in df_station_clim.columns]
    if "sleep_loss_min_mean" not in need_cols:
        diag.add("Build pair sleep delta", "sleep_loss_min_mean not found")
        return pd.DataFrame(columns=["pair_id", "period_norm", "climate4", "delta_sleep_min"])

    tmp = df_station_clim[need_cols].copy()
    tmp = tmp[tmp["period_norm"].isin(BOX_PERIODS)].copy()
    tmp = tmp.dropna(subset=["pair_id", "period_norm", "station_side", "sleep_loss_min_mean"])

    piv = (tmp.pivot_table(
        index=["pair_id", "period_norm", "climate4"],
        columns="station_side",
        values="sleep_loss_min_mean",
        aggfunc="mean"
    ).reset_index())

    if "urban" not in piv.columns or "rural" not in piv.columns:
        diag.add("Build pair sleep delta", "urban or rural sleep column missing after pivot")
        return pd.DataFrame(columns=["pair_id", "period_norm", "climate4", "delta_sleep_min"])

    piv["delta_sleep_min"] = piv["urban"] - piv["rural"]

    out = piv[["pair_id", "period_norm", "climate4", "delta_sleep_min"]].copy()
    diag.add("Build pair sleep delta", f"shape={out.shape}")
    return out


# ============================================================================
# §5 Merge climate labels   (unchanged)
# ============================================================================


def merge_climate_to_pair_panel(df_pair_metrics, df_hw_diurnal, diag):
    pair_clim = (df_hw_diurnal[["pair_id", "climate4", "kg_group", "climate_zone"]]
                 .dropna(subset=["climate4"])
                 .drop_duplicates("pair_id")
                 .rename(columns={"climate_zone": "climate_zone_main"}))

    out = df_pair_metrics.merge(pair_clim, on="pair_id", how="left")
    # out = out[out["climate4"].isin(CLIMATE_ORDER)].copy()

    diag.add("Merge climate to pair panel", f"shape={out.shape}")
    return out


def build_pair_daynight_panel(df_lc, df_station_clim, df_hw_diurnal, diag):
    df_labour = build_pair_period_labour_loss_tx(df_lc, diag)
    df_sleep = build_pair_period_sleep_delta(df_station_clim, diag)
    out = df_labour.merge(
        df_sleep[["pair_id", "period_norm", "delta_sleep_min"]],
        on=["pair_id", "period_norm"], how="outer"
    )
    out = merge_climate_to_pair_panel(out, df_hw_diurnal, diag)
    diag.add("Build pair impact panel", f"shape={out.shape}")
    return out


# ============================================================================
# §5b  [v4-allstation] Climate-label helpers using upstream HW diurnal CSV
# ============================================================================
def build_station_climate_map_from_hw_diurnal(df_hw_diurnal):
    """
    Build a station→(climate4, location) mapping from the upstream
    01 all_pair_period_metrics.csv.

    station_id is constructed as  "{pair_id}_{station_type}"  to match
    the convention used by build_station_daily_from_cdh().
    """
    rows = []
    # One row per (pair_id, station_type, period_norm) — already unique in
    # the 01 main-analysis CSV, but drop_duplicates just in case.
    dedup = df_hw_diurnal.drop_duplicates(
        ["pair_id", "station_type", "period_norm"]
    )
    for _, r in dedup.iterrows():
        side = str(r.get("station_type", "")).strip()
        station_id = f"{r['pair_id']}_{side}"
        rows.append({
            "pair_id":           r["pair_id"],
            "period_norm":       r["period_norm"],
            "station_side":      side,
            "station_id":        station_id,
            "climate4":          r.get("climate4", np.nan),
            "kg_group":          r.get("kg_group", np.nan),
            "climate_zone_main": r.get("climate_zone", np.nan),
            "lon_station":       r.get("lon", np.nan),
            "lat_station":       r.get("lat", np.nan),
        })
    return pd.DataFrame(rows)


def merge_climate_to_station_panel_hw(df_station, df_hw_diurnal, diag):
    """
    Variant of merge_climate_to_station_panel() that sources climate labels
    from the upstream HW diurnal CSV rather than the paired-FFT results.
    """
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

    # out = out[out["climate4"].isin(CLIMATE_ORDER)].copy()
    diag.add(
        "Merge climate into station panel (HW diurnal, all-station)",
        f"shape={out.shape}"
    )
    return out


def build_diurnal_long_from_hw_diurnal(df_hw_diurnal, diag):
    """
    [v4-allstation] Build hour-level long table from the upstream
    01 all_pair_period_metrics.csv.  Urban and rural stations are
    combined without distinction.
    """
    hour_cols = [f"h{h:02d}" for h in range(24)]
    # Only keep columns that actually exist in the file
    hour_cols = [c for c in hour_cols if c in df_hw_diurnal.columns]

    rows = []
    for _, r in df_hw_diurnal.iterrows():
        if pd.isna(r.get("climate4")):
            continue
        side = str(r.get("station_type", ""))
        station_id = f"{r.get('pair_id')}_{side}"
        for h, c in enumerate(hour_cols):
            rows.append({
                "pair_id":      r.get("pair_id"),
                "station_id":   station_id,
                "station_side": side,
                "period_norm":  r.get("period_norm"),
                "climate4":     r.get("climate4"),
                "hour":         h,
                "Ta":           r.get(c, np.nan)
            })

    out = pd.DataFrame(rows)
    diag.add(
        "Build diurnal long from HW diurnal CSV (all-station)",
        f"shape={out.shape}\n"
        f"n_stations={out['station_id'].nunique()}\n"
        f"n_pairs={out['pair_id'].nunique()}"
    )
    return out


def summarize_diurnal(df_diurnal_long):
    g = (df_diurnal_long
         .groupby(["climate4", "period_norm", "hour"], observed=True)["Ta"]
         .agg(["mean", "std", "count"])
         .reset_index())
    g = g.rename(columns={"mean": "Ta_mean", "std": "Ta_std", "count": "n"})
    return g

def _extract_fig9_group_period(df, group_name, period_name):
    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]

    sub = df[(df["group"] == group_name) & (df["period"] == period_name)].copy()
    if len(sub) == 0:
        nan24 = np.full(24, np.nan)
        return {
            "n": 0,
            "u_mean": nan24.copy(),
            "u_std": nan24.copy(),
            "r_mean": nan24.copy(),
            "r_std": nan24.copy(),
            "delta": nan24.copy(),
        }

    u_arr = sub[u_cols].values.astype(float)
    r_arr = sub[r_cols].values.astype(float)

    u_mean = np.nanmean(u_arr, axis=0)
    u_std  = np.nanstd(u_arr, axis=0)
    r_mean = np.nanmean(r_arr, axis=0)
    r_std  = np.nanstd(r_arr, axis=0)
    delta  = u_mean - r_mean

    return {
        "n": len(sub),
        "u_mean": u_mean,
        "u_std": u_std,
        "r_mean": r_mean,
        "r_std": r_std,
        "delta": delta,
    }


def build_fig9_summary(df_fig9, diag):
    out = {}
    for g in ["UHI", "UCI"]:
        out[(g, "HW")]  = _extract_fig9_group_period(df_fig9, g, "heatwave")
        out[(g, "NHW")] = _extract_fig9_group_period(df_fig9, g, "non_heatwave")

    msg = []
    for g in ["UHI", "UCI"]:
        msg.append(
            f"{g}: HW n={out[(g,'HW')]['n']}, NHW n={out[(g,'NHW')]['n']}"
        )
    diag.add("Build Figure9 summary", "\n".join(msg))
    return out

# ============================================================================
# §7 Counts   (unchanged; accepts df_hw_diurnal in place of df_fft)
# ============================================================================
def write_counts(df_station, df_fft_or_hw, out_dir):
    pair_counts = (df_fft_or_hw.drop_duplicates(["pair_id", "climate4"])
                   .dropna(subset=["climate4"])
                   .groupby("climate4", observed=True)
                   .agg(n_pairs=("pair_id", "nunique"))
                   .reset_index()
                   .sort_values("climate4"))
    pair_counts.to_csv(os.path.join(out_dir, "climate_zone_counts_pairs.csv"), index=False)

    station_counts = (df_station.drop_duplicates(["station_id", "climate4"])
                      .dropna(subset=["climate4"])
                      .groupby("climate4", observed=True)
                      .agg(n_stations=("station_id", "nunique"))
                      .reset_index()
                      .sort_values("climate4"))
    station_counts.to_csv(os.path.join(out_dir, "climate_zone_counts_stations.csv"), index=False)

    zone_period_counts = (df_station.drop_duplicates(["station_id", "period_norm", "climate4"])
                          .dropna(subset=["climate4"])
                          .groupby(["climate4", "period_norm"], observed=True)
                          .agg(n_stations=("station_id", "nunique"))
                          .reset_index()
                          .sort_values(["climate4", "period_norm"]))
    zone_period_counts.to_csv(os.path.join(out_dir, "climate_zone_period_counts.csv"), index=False)

    print("\nPair counts by climate zone")
    print(pair_counts.to_string(index=False))
    return pair_counts, station_counts, zone_period_counts


# ============================================================================
# §8 Manual fixed map boxes   (unchanged)
# ============================================================================
def compute_fixed_boxes(df_hw_diurnal):
    """
    根据每个 climate zone 内站点最密集的区域自动生成框。
    框大小允许不同，颜色仍由 ZONE_EDGE 控制。
    优先使用 urban 站点经纬度；若缺失则退回该 pair 的其他站点。
    """
    need_cols = ["pair_id", "station_type", "lon", "lat", "climate4"]
    tmp = df_hw_diurnal[need_cols].copy()
    tmp = tmp.dropna(subset=["pair_id", "lon", "lat", "climate4"])
    tmp = tmp[tmp["climate4"].isin(CLIMATE_ORDER)].copy()

    # 优先 urban
    tmp["_order"] = tmp["station_type"].map({"urban": 0, "rural": 1}).fillna(9)
    tmp = tmp.sort_values(["pair_id", "_order"])
    tmp = tmp.drop_duplicates("pair_id")

    boxes = {}

    for z in CLIMATE_ORDER:
        sub = tmp[tmp["climate4"] == z].copy()
        if len(sub) == 0:
            continue

        lon = sub["lon"].to_numpy(dtype=float)
        lat = sub["lat"].to_numpy(dtype=float)

        # 样本很少时直接用全体分位数
        if len(sub) < 8:
            lon0, lon1 = np.nanpercentile(lon, [10, 90])
            lat0, lat1 = np.nanpercentile(lat, [10, 90])
        else:
            # 先找最密集的二维直方图区块
            nbx = int(np.clip(np.sqrt(len(sub)) * 1.2, 4, 8))
            nby = int(np.clip(np.sqrt(len(sub)) * 1.0, 4, 8))

            H, xedges, yedges = np.histogram2d(lon, lat, bins=[nbx, nby])
            idx = np.unravel_index(np.argmax(H), H.shape)

            x0, x1 = xedges[idx[0]], xedges[idx[0] + 1]
            y0, y1 = yedges[idx[1]], yedges[idx[1] + 1]

            # 取最密集格及其邻近区域内的点
            core = sub[
                (sub["lon"] >= x0) & (sub["lon"] <= x1) &
                (sub["lat"] >= y0) & (sub["lat"] <= y1)
            ].copy()

            if len(core) < max(4, int(len(sub) * 0.25)):
                # 如果核心格太小，则按离核心中心距离取最近的 50%
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                sub["_dist"] = np.sqrt(((sub["lon"] - cx) / 10.0) ** 2 +
                                       ((sub["lat"] - cy) / 6.0) ** 2)
                core = sub.nsmallest(max(4, int(len(sub) * 0.50)), "_dist").copy()

            lon0, lon1 = np.nanpercentile(core["lon"], [5, 95])
            lat0, lat1 = np.nanpercentile(core["lat"], [5, 95])

        # 每个区单独 padding，允许框大小不一样，也稍微放大一点
        pad_lon = max(4.5, 0.22 * (lon1 - lon0 + 1e-6))
        pad_lat = max(3.5, 0.22 * (lat1 - lat0 + 1e-6))

        lon0 -= pad_lon
        lon1 += pad_lon
        lat0 -= pad_lat
        lat1 += pad_lat

        # 保证最小尺寸，避免太瘦
        min_w = {"A": 20, "B": 18, "C": 20, "D": 22}.get(z, 18)
        min_h = {"A": 14, "B": 12, "C": 12, "D": 14}.get(z, 12)

        cx = 0.5 * (lon0 + lon1)
        cy = 0.5 * (lat0 + lat1)
        if (lon1 - lon0) < min_w:
            lon0, lon1 = cx - min_w / 2, cx + min_w / 2
        if (lat1 - lat0) < min_h:
            lat0, lat1 = cy - min_h / 2, cy + min_h / 2

        lon0 = max(-180, lon0)
        lon1 = min(180, lon1)
        lat0 = max(-60, lat0)
        lat1 = min(85, lat1)

        boxes[z] = {"lon0": lon0, "lon1": lon1, "lat0": lat0, "lat1": lat1}

    return boxes

def build_hw_nhw_tmax_map_metric_from_hw_diurnal(df_hw_diurnal, diag):
    """
    Compute U-R Tmax delta (UHI) and its HW-NHW difference.
    """
    df = df_hw_diurnal[
        ["pair_id", "station_type", "period_norm", "lon", "lat", "Tmax_fft", "climate4"]
    ].copy()

    # Location for map scatter (prefer urban)
    df_sorted = df.sort_values(
        "station_type",
        key=lambda s: s.map({"urban": 0, "rural": 1}).fillna(2)
    )
    loc = (df_sorted
           .drop_duplicates("pair_id")[["pair_id", "lon", "lat", "climate4"]]
           .rename(columns={"lon": "lon_urban", "lat": "lat_urban"}))

    # Pivot to get Urban and Rural Tmax
    piv = df.pivot_table(
        index=["pair_id", "period_norm"], 
        columns="station_type", 
        values="Tmax_fft"
    ).reset_index()
    
    if "urban" in piv.columns and "rural" in piv.columns:
        piv["uhi_tmax"] = piv["urban"] - piv["rural"]
    else:
        piv["uhi_tmax"] = np.nan

    hw_uhi = piv[piv["period_norm"] == "HW"][["pair_id", "uhi_tmax"]].rename(columns={"uhi_tmax": "uhi_HW"})
    nhw_uhi = piv[piv["period_norm"] == "NHW"][["pair_id", "uhi_tmax"]].rename(columns={"uhi_tmax": "uhi_NHW"})

    # Merge back to locations
    merged = (loc
              .merge(hw_uhi, on="pair_id", how="left")
              .merge(nhw_uhi, on="pair_id", how="left"))

    # Delta UHI (HW - NHW). A response is defined only when both
    # matched-period operands are finite; missing scientific values remain NaN.
    merged["delta_uhi_hw_nhw"] = merged["uhi_HW"] - merged["uhi_NHW"]

    diag.add("Build map metric (U-R)", f"shape={merged.shape}")
    return merged


def build_nhw_group_intensity_from_fig9(df_fig9, diag):
    """
    Build pair-level NHW UHI/UCI class and absolute intensity.

    group:
      UHI / UCI from Figure9 source

    nhw_abs_intensity:
      abs(mean urban-rural diurnal Ta contrast during NHW)
    """
    u_cols = [f"urban_diurnal_h{h:02d}" for h in range(24)]
    r_cols = [f"rural_diurnal_h{h:02d}" for h in range(24)]

    need_cols = ["pair_id", "group", "period"] + u_cols + r_cols
    need_cols = [c for c in need_cols if c in df_fig9.columns]

    sub = df_fig9[
        (df_fig9["period"] == "non_heatwave") &
        (df_fig9["group"].isin(["UHI", "UCI"]))
    ][need_cols].copy()

    rows = []
    for _, r in sub.iterrows():
        u = r[u_cols].to_numpy(dtype=float)
        rr = r[r_cols].to_numpy(dtype=float)
        delta = np.nanmean(u - rr)

        rows.append({
            "pair_id": r["pair_id"],
            "group": r["group"],
            "nhw_abs_intensity": abs(delta),
        })

    out = pd.DataFrame(rows)
    out = (
        out.groupby(["pair_id", "group"], observed=True)["nhw_abs_intensity"]
           .mean()
           .reset_index()
    )

    diag.add(
        "Build NHW UHI/UCI map intensity",
        f"shape={out.shape}\n"
        f"group counts:\n{out['group'].value_counts(dropna=False).to_string()}"
    )
    return out


def build_nhw_group_intensity_from_hw_diurnal(df_hw_diurnal, diag):
    """Build NHW intensity while retaining the strict annual group label."""
    need = ["pair_id", "station_type", "period_norm", "Tmax_fft", "group"]
    tmp = df_hw_diurnal[[c for c in need if c in df_hw_diurnal.columns]].copy()
    tmp = tmp[tmp["period_norm"] == "NHW"].dropna(subset=["pair_id", "station_type"])
    piv = tmp.pivot_table(index="pair_id", columns="station_type", values="Tmax_fft", aggfunc="mean").reset_index()
    if not {"urban", "rural"}.issubset(piv.columns):
        raise ValueError("urban or rural station_type missing in df_hw_diurnal")
    group = df_hw_diurnal[["pair_id", "group"]].drop_duplicates("pair_id")
    piv["nhw_delta"] = piv["urban"] - piv["rural"]
    piv["nhw_abs_intensity"] = piv["nhw_delta"].abs()
    out = piv.merge(group, on="pair_id", how="inner")
    out = out[["pair_id", "group", "nhw_abs_intensity", "nhw_delta"]]
    diag.add(
        "Build NHW intensity with annual groups",
        f"shape={out.shape}\ncounts:\n{out['group'].value_counts().to_string()}"
    )
    return out


def write_pair_loss_trace_txt(df_hw_diurnal, df_fig9, out_dir, diag):
    """
    Minimal diagnostic output to trace why original 342 pairs become
    324 UHI/UCI pairs in later statistics.

    Output:
      pair_loss_trace_342_to_324.txt
    """
    ensure_dir(out_dir)

    out_txt = os.path.join(out_dir, "pair_loss_trace_342_to_324.txt")

    # 1) 原始 heatwave all-station pair
    all_pairs = set(df_hw_diurnal["pair_id"].dropna().astype(str).unique())

    # 2) Figure9 source 中的 pair
    fig9_pairs = set(df_fig9["pair_id"].dropna().astype(str).unique())

    # 3) 只保留 heatwave / non_heatwave 后的 pair
    period_pairs = set(
        df_fig9.loc[
            df_fig9["period"].isin(["heatwave", "non_heatwave"]),
            "pair_id"
        ].dropna().astype(str).unique()
    )

    # 4) 只保留 UHI / UCI 后的 pair
    group_pairs = set(
        df_fig9.loc[
            df_fig9["group"].isin(["UHI", "UCI"]),
            "pair_id"
        ].dropna().astype(str).unique()
    )

    # 5) 最终用于 UHI/UCI + HW/NHW 统计的 pair
    final_pairs = set(
        df_fig9.loc[
            df_fig9["group"].isin(["UHI", "UCI"]) &
            df_fig9["period"].isin(["heatwave", "non_heatwave"]),
            "pair_id"
        ].dropna().astype(str).unique()
    )

    missing_from_final = sorted(all_pairs - final_pairs)

    # 分阶段丢失
    lost_not_in_fig9 = sorted(all_pairs - fig9_pairs)
    lost_by_period = sorted(fig9_pairs - period_pairs)
    lost_by_group = sorted(period_pairs - final_pairs)

    # 气候区信息
    clim_cols = [c for c in ["pair_id", "climate4", "kg_group", "climate_zone"] if c in df_hw_diurnal.columns]
    if clim_cols:
        pair_clim = (
            df_hw_diurnal[clim_cols]
            .drop_duplicates()
            .copy()
        )
        pair_clim["pair_id"] = pair_clim["pair_id"].astype(str)
        missing_clim = pair_clim[pair_clim["pair_id"].isin(missing_from_final)].copy()
    else:
        missing_clim = pd.DataFrame()

    # 缺失 pair 在 df_fig9 里的原始 group / period 情况
    fig9_missing = df_fig9[df_fig9["pair_id"].astype(str).isin(missing_from_final)].copy()
    if len(fig9_missing):
        gp_cols = [c for c in ["pair_id", "group", "period"] if c in fig9_missing.columns]
        fig9_detail = (
            fig9_missing
            .groupby(gp_cols, dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(gp_cols)
        )
    else:
        fig9_detail = pd.DataFrame()

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Pair loss trace: 342 to 324\n\n")

        f.write("## Pair counts by filtering step\n")
        f.write(f"all_pairs from df_hw_diurnal: {len(all_pairs)}\n")
        f.write(f"fig9_pairs from df_fig9: {len(fig9_pairs)}\n")
        f.write(f"period_pairs heatwave/non_heatwave: {len(period_pairs)}\n")
        f.write(f"group_pairs UHI/UCI: {len(group_pairs)}\n")
        f.write(f"final_pairs UHI/UCI + heatwave/non_heatwave: {len(final_pairs)}\n")
        f.write(f"missing_from_final: {len(missing_from_final)}\n\n")

        f.write("## Lost not in df_fig9\n")
        f.write(f"count={len(lost_not_in_fig9)}\n")
        f.write("\n".join(lost_not_in_fig9) + "\n\n")

        f.write("## Lost by period filter\n")
        f.write(f"count={len(lost_by_period)}\n")
        f.write("\n".join(lost_by_period) + "\n\n")

        f.write("## Lost by group filter after period filter\n")
        f.write(f"count={len(lost_by_group)}\n")
        f.write("\n".join(lost_by_group) + "\n\n")

        f.write("## All missing pairs from final\n")
        f.write(f"count={len(missing_from_final)}\n")
        f.write("\n".join(missing_from_final) + "\n\n")

        f.write("## Missing pairs climate information\n")
        if len(missing_clim):
            f.write(missing_clim.sort_values("pair_id").to_string(index=False))
        else:
            f.write("No climate information found or climate columns missing.")
        f.write("\n\n")

        f.write("## Missing pairs records in df_fig9: pair_id / group / period\n")
        if len(fig9_detail):
            f.write(fig9_detail.to_string(index=False))
        else:
            f.write("No records found in df_fig9 for missing pairs.")
        f.write("\n")

    diag.add(
        "Write pair loss trace txt",
        f"path={out_txt}\n"
        f"all_pairs={len(all_pairs)}\n"
        f"final_pairs={len(final_pairs)}\n"
        f"missing={len(missing_from_final)}"
    )


def build_latitudinal_hw_nhw_single_station(df_hw_diurnal, diag, metric="Tmax_fft"):
    """
    Station-level HW − NHW temperature anomaly by latitude.

    This is NOT urban-rural contrast.
    It computes, for each individual station:
        delta_tx = metric(HW) - metric(NHW)

    metric:
      - Tmax_fft for daytime temperature anomaly
      - Tmin_fft for nighttime temperature anomaly
    """
    need = ["pair_id", "station_type", "period_norm", metric, "lat", "climate4"]
    tmp = df_hw_diurnal[need].copy()

    tmp = tmp[tmp["period_norm"].isin(["HW", "NHW"])]
    tmp = tmp.dropna(subset=["pair_id", "station_type", "period_norm", metric, "lat"])

    tmp["station_id"] = (
        tmp["pair_id"].astype(str) + "_" + tmp["station_type"].astype(str)
    )

    piv = tmp.pivot_table(
        index=["station_id", "pair_id", "station_type", "climate4"],
        columns="period_norm",
        values=metric,
        aggfunc="mean"
    ).reset_index()

    lat_map = (
        tmp.groupby("station_id", observed=True)["lat"]
        .mean()
        .reset_index()
        .rename(columns={"lat": "lat_ref"})
    )

    piv = piv.merge(lat_map, on="station_id", how="left")

    if "HW" not in piv.columns or "NHW" not in piv.columns:
        diag.add(
            f"Build latitudinal HW-NHW single-station {metric}",
            "Missing HW or NHW column after pivot"
        )
        return pd.DataFrame(
            columns=["station_id", "pair_id", "station_type", "climate4",
                     "lat_ref", "period_norm", "delta_tx"]
        )

    piv["delta_tx"] = piv["HW"] - piv["NHW"]
    piv["period_norm"] = "HW"

    out = piv[
        ["station_id", "pair_id", "station_type", "climate4",
         "lat_ref", "period_norm", "delta_tx"]
    ].dropna(subset=["lat_ref", "delta_tx"]).copy()

    diag.add(
        f"Build latitudinal HW-NHW single-station {metric}",
        f"shape={out.shape}\n"
        f"stations={out['station_id'].nunique()}\n"
        f"mean={out['delta_tx'].mean():.3f}, "
        f"min={out['delta_tx'].min():.3f}, "
        f"max={out['delta_tx'].max():.3f}"
    )

    return out


def summarize_latitudinal_profile(df_lat, lat_bin_width=5):
    """
    将逐 pair 的 delta_tx 按纬度分箱，计算每个纬度带的均值和不确定性。
    返回：
      period_norm, lat_mid, delta_mean, delta_sem, n
    """
    if df_lat is None or len(df_lat) == 0:
        return pd.DataFrame(columns=["period_norm", "lat_mid", "delta_mean", "delta_sem", "n"])

    bins = np.arange(-60, 90, lat_bin_width)
    df = df_lat.copy()
    df["lat_bin"] = pd.cut(df["lat_ref"], bins=bins, include_lowest=True)

    prof = (
        df.groupby(["period_norm", "lat_bin"], observed=True)["delta_tx"]
        .agg(
            delta_mean="mean",
            delta_sem=lambda x: sem_safe(x),
            n="count"
        )
        .reset_index()
    )

    prof["lat_mid"] = prof["lat_bin"].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    prof = prof.dropna(subset=["lat_mid", "delta_mean"]).sort_values(["period_norm", "lat_mid"])

    return prof

def draw_latitudinal_profile(ax, df_prof, period, xlim=None, show_ylabel=False):
    """
    纵向 line profile:
      y = latitude
      x = delta_tx
    """
    sub = df_prof[df_prof["period_norm"] == period].copy()
    ax.set_facecolor("none")

    if len(sub) == 0:
        ax.set_visible(False)
        return

    sub = sub.sort_values("lat_mid")
    x = sub["delta_mean"].to_numpy(dtype=float)
    y = sub["lat_mid"].to_numpy(dtype=float)
    s = sub["delta_sem"].fillna(0).to_numpy(dtype=float)

    if xlim is None:
        vmax = np.nanpercentile(np.abs(x), 95) if len(x) else 1.0
        vmax = max(vmax, 0.5)
        xlim = (-vmax, vmax)

    # 置信带
    ax.fill_betweenx(
        y, x - s, x + s,
        color="#bdbdbd", alpha=0.30, linewidth=0, zorder=1
    )

    # 主线
    line_color = CLR_PERIOD.get(period, "black")
    ax.plot(x, y, color=line_color, lw=1.3, zorder=2)

    # x=0 参考线
    ax.axvline(0, color="black", lw=0.55, ls=":", zorder=0)

    ax.set_xlim(*xlim)
    ax.set_ylim(-60, 85)
    ax.set_title(period, fontsize=5.8, fontweight="bold", pad=1.5)
    ax.set_xlabel(r"$\Delta T_a$ (°C)", fontsize=5.2, labelpad=1.5)

    if show_ylabel:
        ax.set_ylabel("Latitude", fontsize=5.4)
        ax.set_yticks([-40, -20, 0, 20, 40, 60, 80])
        ax.set_yticklabels(["40°S", "20°S", "0°", "20°N", "40°N", "60°N", "80°N"])
        ax.tick_params(axis="y", labelsize=5.0, length=2)
    else:
        ax.set_yticks([])
        ax.set_ylabel("")

    ax.tick_params(axis="x", labelsize=5.0, length=2, pad=1)

    # --- 针对 NCC 审美优化的边框处理 ---
    if hasattr(ax, 'projection') and ax.projection is not None:
        # 情况 A：这是 Robinson 投影地图
        # 1. 隐藏多余的矩形外框（left, right, top, bottom）
        for s in ['left', 'right', 'top', 'bottom']:
            ax.spines[s].set_visible(False)
        
        # 2. 只开启地理轮廓线（geo），这会沿着 Robinson 的弧形边缘画线
        if 'geo' in ax.spines:
            ax.spines['geo'].set_visible(True)
            ax.spines['geo'].set_linewidth(0.55)
            ax.spines['geo'].set_edgecolor("black")
    else:
        # 情况 B：这是侧边的纬度剖面图（Latitudinal Profile）或其他普通图表
        # 保持完整的矩形框架，以界定坐标空间
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_edgecolor("black")


    ax.grid(axis="y", linestyle="--", linewidth=0.25, alpha=0.18)
    ax.grid(axis="x", linestyle="--", linewidth=0.25, alpha=0.18)

def draw_map_with_latitudinal_profiles(fig, gs_cell, df_hw_diurnal, df_fig9,
                                       diag, panel_letter="a"):
    """
    画地图 + 右侧两个纵向纬向 line profile：
    左 = JJA
    右 = HW

    v11 patch:
      - remove climate-zone boxes
      - station marker shape = NHW-period UHI / UCI class
      - marker size = absolute NHW UHI/UCI intensity
    """
    map_metric = build_hw_nhw_tmax_map_metric_from_hw_diurnal(df_hw_diurnal, diag)

    # Use df_hw_diurnal itself to classify all available pairs.
    # This keeps all 328 pairs instead of the 324 pairs available in df_fig9.
    nhw_group = build_nhw_group_intensity_from_hw_diurnal(df_hw_diurnal, diag)

    map_metric = map_metric.merge(nhw_group, on="pair_id", how="left")
    map_metric = map_metric[map_metric["group"].isin(["UHI", "UCI"])].copy()

    df_day_lat = build_latitudinal_hw_nhw_single_station(
        df_hw_diurnal, diag, metric="Tmax_fft"
    )
    df_night_lat = build_latitudinal_hw_nhw_single_station(
        df_hw_diurnal, diag, metric="Tmin_fft"
    )

    df_day_prof = summarize_latitudinal_profile(df_day_lat, lat_bin_width=5)
    df_night_prof = summarize_latitudinal_profile(df_night_lat, lat_bin_width=5)

    df_prof = pd.concat([df_day_prof, df_night_prof], ignore_index=True)


    color_vals = map_metric["delta_uhi_hw_nhw"].dropna()
    vmin = np.nanpercentile(color_vals, 2) if len(color_vals) else -2
    vmax = np.nanpercentile(color_vals, 98) if len(color_vals) else 4

    size_vals = map_metric["nhw_abs_intensity"].dropna()
    if len(size_vals):
        smin = np.nanpercentile(size_vals, 5)
        smax = np.nanpercentile(size_vals, 95)
        if np.isclose(smin, smax):
            smin, smax = 0, max(float(smax), 1.0)
    else:
        smin, smax = 0, 1

    def scale_marker_size(v):
        v = np.asarray(v, dtype=float)
        out = 10 + 45 * (v - smin) / (smax - smin + 1e-9)
        return np.clip(out, 10, 55)

    prof_vals = df_prof["delta_mean"].dropna()
    if len(prof_vals):
        prof_v = np.nanpercentile(np.abs(prof_vals), 95)
        prof_v = max(prof_v, 0.5)
    else:
        prof_v = 1.0
    prof_xlim = (-prof_v, prof_v)

    subgs = gs_cell.subgridspec(
        1, 3,
        width_ratios=[19.0, 1.15, 1.15],
        wspace=0.06
    )

    # --- map axis ---
    if HAS_CARTOPY:
        ax_map = fig.add_subplot(subgs[0, 0], projection=PLOT_PROJECTION)
        ax_map.patch.set_linewidth(0.8)
        for spine in ax_map.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")

        ax_map.set_extent([-180, 180, -60, 85], crs=DATA_CRS)
        ax_map.set_facecolor("white")
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.30,
                           edgecolor="#606060", zorder=1)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.15,
                           edgecolor="#aaaaaa", linestyle=":", zorder=1)

        gl = ax_map.gridlines(
            crs=DATA_CRS, draw_labels=True,
            linewidth=0.20, color="#cccccc", alpha=0.5, linestyle="-"
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 5.5}
        gl.ylabel_style = {"size": 5.5}

        sc = None
        marker_map = {"UHI": "^", "UCI": "v"}

        for grp in ["UHI", "UCI"]:
            sub = map_metric[map_metric["group"] == grp].copy()
            if len(sub) == 0:
                continue

            sc = ax_map.scatter(
                sub["lon_urban"].values,
                sub["lat_urban"].values,
                transform=DATA_CRS,
                c=sub["delta_uhi_hw_nhw"].values,
                cmap=MAP_CMAP,
                vmin=vmin,
                vmax=vmax,
                s=scale_marker_size(sub["nhw_abs_intensity"].values),
                marker=marker_map[grp],
                linewidths=0.25,
                edgecolors="black",
                alpha=0.88,
                zorder=3,
                label=grp
            )

    else:
        ax_map = fig.add_subplot(subgs[0, 0])
        ax_map.patch.set_linewidth(0.8)
        for spine in ax_map.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")

        sc = None
        marker_map = {"UHI": "^", "UCI": "v"}

        for grp in ["UHI", "UCI"]:
            sub = map_metric[map_metric["group"] == grp].copy()
            if len(sub) == 0:
                continue

            sc = ax_map.scatter(
                sub["lon_urban"].values,
                sub["lat_urban"].values,
                c=sub["delta_tx_hw_nhw"].values,
                cmap=MAP_CMAP,
                vmin=vmin,
                vmax=vmax,
                s=scale_marker_size(sub["nhw_abs_intensity"].values),
                marker=marker_map[grp],
                linewidths=0.25,
                edgecolors="black",
                alpha=0.88,
                label=grp
            )

        ax_map.set_facecolor("white")
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-60, 85)
        ax_map.set_xlabel("Longitude", fontsize=5.5)
        ax_map.set_ylabel("Latitude", fontsize=5.5)
        ax_map.grid(True, linestyle="--", linewidth=0.25, alpha=0.2)

    # colorbar
    # colorbar
    if sc is not None:
        try:
            cax = mpl_inset_axes(
                ax_map,
                width="20%", height="3.0%",
                loc="lower center",          # <--- 移到下中，避开左下的直方图
                borderpad=1.5                # <--- 移除原有的 bbox_to_anchor，让其自然居中贴底
            )
            cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
            cbar.set_label(r"$\Delta$UHI (HW$-$NHW) (°C)",   # <--- 更新标签
                           fontsize=5.6, labelpad=1.5)
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.tick_params(labelsize=5.0, length=1.8, pad=1)
            cbar.outline.set_visible(False)
            # --- Colorbar 视觉统一化处理 ---

            # 1. 隐藏 cax 容器的四条边（这是对的，保持不动）
            for spine in cax.spines.values():
                spine.set_visible(False)
            cax.set_facecolor("none")

            # 2. 【核心补充】精修 Colorbar 自身的轮廓线
            # cbar 是你定义的 colorbar 对象（例如 cbar = fig.colorbar(...)）
            cbar.outline.set_visible(True)           # 确保色块周围有一圈细线，这样浅色才不会跟背景混掉
            cbar.outline.set_linewidth(0.55)        # 关键！必须与地图的 0.55 线宽完全一致
            cbar.outline.set_edgecolor("black")     # 颜色统一为纯黑

            # 3. 刻度微调（让它看起来更精致）
            cbar.ax.tick_params(labelsize=5.0, length=1.8, width=0.55, pad=1)

        except Exception:
            # 兼容性 Fallback
            cbar = fig.colorbar(sc, ax=ax_map, orientation="horizontal",
                                pad=0.08, shrink=0.50)
            cbar.set_label(r"$\Delta$UHI (HW$-$NHW) (°C)", fontsize=6)
            cbar.ax.tick_params(labelsize=5.8)

    # ===== [ADD] bimodality inset =====
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # distribution of U-R ΔTmax (使用 NHW 期间的基线城乡温差)
    dTa = map_metric["uhi_NHW"].dropna().values

    # 将第一个参数由 ax 改为 ax_map
    ax_in = mpl_inset_axes(ax_map, width="18%", height="18%", loc="lower left", borderpad=1.8)
    ax_in.hist(dTa, bins=30, density=True, color="gray", alpha=0.7)
    ax_in.axvline(0, color="k", linestyle="--", lw=1)

    ax_in.set_title("Bimodality", fontsize=6)
    ax_in.set_xlabel(r"U-R $\Delta T_{max}$", fontsize=4.5)  # 添加 x 轴提示
    ax_in.set_xticks([])
    ax_in.set_yticks([])

    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=4.8, label="UHI"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=4.8, label="UCI"),
    ]
    ax_map.legend(handles=handles, frameon=False, loc="upper left",  # <--- 移到左上
                  fontsize=5.6)  # <--- 删除了 title 参数


    add_panel_label(ax_map, panel_letter, x=-0.03, y=1.02)

    # --- right-side latitudinal profiles ---
    ax_jja = fig.add_subplot(subgs[0, 1])
    ax_hw = fig.add_subplot(subgs[0, 2], sharey=ax_jja)

    # ===== [REPLACE] diurnal ΔTa by latitude =====

    draw_latitudinal_profile(
        ax_jja, df_day_prof, "HW",
        xlim=prof_xlim, show_ylabel=True
    )
    ax_jja.set_title("Daytime ΔTa", fontsize=5.8, fontweight="bold")

    draw_latitudinal_profile(
        ax_hw, df_night_prof, "HW",
        xlim=prof_xlim, show_ylabel=False
    )
    ax_hw.set_title("Nighttime ΔTa", fontsize=5.8, fontweight="bold")

    return ax_map, ax_jja, ax_hw

def build_hw_nhw_delta_uhi_daynight_from_hw_diurnal(df_hw_diurnal, diag):
    """
    Pair-level heatwave amplification of urban-rural thermal contrast.

    Daytime:
      ΔUHI_Tmax = (Tmax_U - Tmax_R)_HW - (Tmax_U - Tmax_R)_NHW

    Nighttime:
      ΔUHI_Tmin = (Tmin_U - Tmin_R)_HW - (Tmin_U - Tmin_R)_NHW
    """
    df = df_hw_diurnal[
        ["pair_id", "station_type", "period_norm", "lon", "lat",
         "Tmax_fft", "Tmin_fft", "climate4"]
    ].copy()

    df = df[df["period_norm"].isin(["HW", "NHW"])].copy()

    loc = (
        df.sort_values(
            "station_type",
            key=lambda s: s.map({"urban": 0, "rural": 1}).fillna(2)
        )
        .drop_duplicates("pair_id")
        [["pair_id", "lon", "lat", "climate4"]]
        .rename(columns={"lon": "lon_ref", "lat": "lat_ref"})
    )

    piv_tx = df.pivot_table(
        index=["pair_id", "period_norm"],
        columns="station_type",
        values="Tmax_fft",
        aggfunc="mean"
    ).reset_index()

    piv_tn = df.pivot_table(
        index=["pair_id", "period_norm"],
        columns="station_type",
        values="Tmin_fft",
        aggfunc="mean"
    ).reset_index()

    for piv, out_col in [
        (piv_tx, "uhi_tmax"),
        (piv_tn, "uhi_tmin"),
    ]:
        if "urban" in piv.columns and "rural" in piv.columns:
            piv[out_col] = piv["urban"] - piv["rural"]
        else:
            piv[out_col] = np.nan

    tx_hw = piv_tx[piv_tx["period_norm"] == "HW"][["pair_id", "uhi_tmax"]].rename(
        columns={"uhi_tmax": "uhi_tmax_HW"}
    )
    tx_nhw = piv_tx[piv_tx["period_norm"] == "NHW"][["pair_id", "uhi_tmax"]].rename(
        columns={"uhi_tmax": "uhi_tmax_NHW"}
    )

    tn_hw = piv_tn[piv_tn["period_norm"] == "HW"][["pair_id", "uhi_tmin"]].rename(
        columns={"uhi_tmin": "uhi_tmin_HW"}
    )
    tn_nhw = piv_tn[piv_tn["period_norm"] == "NHW"][["pair_id", "uhi_tmin"]].rename(
        columns={"uhi_tmin": "uhi_tmin_NHW"}
    )

    out = (
        loc
        .merge(tx_hw, on="pair_id", how="left")
        .merge(tx_nhw, on="pair_id", how="left")
        .merge(tn_hw, on="pair_id", how="left")
        .merge(tn_nhw, on="pair_id", how="left")
    )

    out["delta_uhi_daytime"] = out["uhi_tmax_HW"] - out["uhi_tmax_NHW"]
    out["delta_uhi_nighttime"] = out["uhi_tmin_HW"] - out["uhi_tmin_NHW"]

    # Do not convert a one-sided or missing HW/NHW response to zero.
    # Each response remains NaN unless both matched-period contrasts exist.

    diag.add(
        "Build day/night ΔUHI map metrics",
        f"shape={out.shape}\n"
        f"day mean={out['delta_uhi_daytime'].mean():.3f}\n"
        f"night mean={out['delta_uhi_nighttime'].mean():.3f}"
    )

    return out

def write_rt_regime_summary_txt(map_metric, df_fig9, out_dir, diag):
    """
    Output summary statistics for RTx / RTn by UHI/UCI regime.

    Definitions:
      RTx = delta_uhi_daytime   = (Tmax_U - Tmax_R)_HW - (Tmax_U - Tmax_R)_NHW
      RTn = delta_uhi_nighttime = (Tmin_U - Tmin_R)_HW - (Tmin_U - Tmin_R)_NHW

    Regime-aware enhancement:
      UHI enhanced if RT > 0, weakened if RT < 0
      UCI enhanced if RT < 0, weakened if RT > 0
    """
    ensure_dir(out_dir)

    df_group = (
        df_fig9[["pair_id", "group"]]
        .dropna(subset=["pair_id", "group"])
        .drop_duplicates("pair_id")
        .copy()
    )

    df = map_metric.merge(df_group, on="pair_id", how="left")
    df = df[df["group"].isin(["UHI", "UCI"])].copy()

    def _count_regime_change(sub, col, group):
        v = sub[col].replace([np.inf, -np.inf], np.nan).dropna()

        if group == "UHI":
            enhanced = int((v > 0).sum())
            weakened = int((v < 0).sum())
        elif group == "UCI":
            enhanced = int((v < 0).sum())
            weakened = int((v > 0).sum())
        else:
            enhanced = weakened = 0

        unchanged = int((v == 0).sum())
        return enhanced, weakened, unchanged, float(v.mean()) if len(v) else np.nan

    lines = []
    lines.append("RTx / RTn regime summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Definitions:")
    lines.append("RTx = (Tmax_U - Tmax_R)_HW - (Tmax_U - Tmax_R)_NHW")
    lines.append("RTn = (Tmin_U - Tmin_R)_HW - (Tmin_U - Tmin_R)_NHW")
    lines.append("")
    lines.append("Enhancement rule:")
    lines.append("UHI: enhanced if RT > 0, weakened if RT < 0")
    lines.append("UCI: enhanced if RT < 0, weakened if RT > 0")
    lines.append("")

    for group in ["UHI", "UCI"]:
        sub = df[df["group"] == group].copy()
        n_pair = sub["pair_id"].nunique()

        tx_enh, tx_weak, tx_zero, tx_mean = _count_regime_change(
            sub, "delta_uhi_daytime", group
        )
        tn_enh, tn_weak, tn_zero, tn_mean = _count_regime_change(
            sub, "delta_uhi_nighttime", group
        )

        lines.append(f"{group} pairs: {n_pair}")
        lines.append(
            f"  RTx: enhanced={tx_enh}, weakened={tx_weak}, unchanged={tx_zero}, "
            f"mean={tx_mean:.4f} °C"
        )
        lines.append(
            f"  RTn: enhanced={tn_enh}, weakened={tn_weak}, unchanged={tn_zero}, "
            f"mean={tn_mean:.4f} °C"
        )
        lines.append("")

    lines.append("Overall mean changes:")
    lines.append(
        f"  RTx mean = {df['delta_uhi_daytime'].replace([np.inf, -np.inf], np.nan).mean():.4f} °C"
    )
    lines.append(
        f"  RTn mean = {df['delta_uhi_nighttime'].replace([np.inf, -np.inf], np.nan).mean():.4f} °C"
    )

    out_path = os.path.join(out_dir, "rtx_rtn_regime_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    diag.add("RTx / RTn regime summary written", out_path)
    print(f"  ✓ saved: {out_path}")


def get_lat_profile_stats(df, value_col, bin_width=5):
    """计算纬度分箱统计量"""
    valid = df.dropna(subset=['lat_ref', value_col]).copy()
    if len(valid) == 0: return None
    bins = np.arange(-60, 90, bin_width)
    valid['lat_bin'] = pd.cut(valid['lat_ref'], bins=bins)
    prof = valid.groupby('lat_bin', observed=True)[value_col].agg(['mean', 'sem']).reset_index()
    prof['lat_mid'] = prof['lat_bin'].apply(lambda x: x.mid)
    return prof.dropna(subset=['mean'])

def draw_side_latitudinal_profile(ax, prof_data, color, xlabel):
    """在地图右侧绘制垂直对齐的剖面线图"""
    if prof_data is None:
        ax.axis('off')
        return
    
    y = prof_data['lat_mid'].values
    x = prof_data['mean'].values
    err = prof_data['sem'].fillna(0).values

    # 填充误差带 (SEM)
    ax.fill_betweenx(y, x - err, x + err, color=color, alpha=0.15, lw=0)
    # 绘制主趋势线
    ax.plot(x, y, color=color, lw=1.3)
    
    # 辅助线和样式
    ax.axvline(0, color='black', lw=0.6, ls='--', alpha=0.4)
    ax.set_ylim(-60, 85) # 必须与地图纬度严格一致
    ax.set_xlabel(xlabel, fontsize=5.5)
    ax.tick_params(axis='both', labelsize=5, length=2, pad=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, ls=':', lw=0.4, alpha=0.3)

def make_composite_maps_de_figure(df_hw_diurnal, df_fig9, out_dir, diag):
    """
    Composite figure (Top-Tier Journal Layout):
    - Absolute vertical separation to prevent overlapping.
    - Flat, wide aspect ratio for the diurnal profile.
    - Left-aligned edges.
    """
    apply_nature_style()

    write_pair_loss_trace_txt(df_hw_diurnal, df_fig9, out_dir, diag)
    fig9_sum = build_fig9_summary(df_fig9, diag)
    map_metric = build_hw_nhw_delta_uhi_daynight_from_hw_diurnal(df_hw_diurnal, diag)
    write_rt_regime_summary_txt(map_metric, df_fig9, out_dir, diag)
    # 画布略微加高，给三行图表留出充足的呼吸空间
    fig = plt.figure(figsize=(7.5, 9.8), dpi=600)

    gs = GridSpec(
        4, 2, figure=fig,
        width_ratios=[1.6, 1],
        # row 0: map a
        # row 1: shared colorbar
        # row 2: map b
        # row 3: panel c + legend
        height_ratios=[1.00, 0.12, 1.00, 0.62],
        hspace=0.10,
        wspace=0.10,
        left=0.06,
        right=0.98,
        top=0.94,
        bottom=0.06
    )

    if HAS_CARTOPY:
        ax_a = fig.add_subplot(gs[0, :], projection=PLOT_PROJECTION)
        ax_b = fig.add_subplot(gs[2, :], projection=PLOT_PROJECTION)
    else:
        ax_a = fig.add_subplot(gs[0, :])
        ax_b = fig.add_subplot(gs[2, :])

    # Dedicated empty slot for the shared colorbar
    ax_cbar_slot = fig.add_subplot(gs[1, :])
    ax_cbar_slot.axis("off")


    # Shared colour scale for daytime and nighttime maps
    _shared_vals = pd.concat(
        [
            map_metric["delta_uhi_daytime"],
            map_metric["delta_uhi_nighttime"],
        ],
        ignore_index=True
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(_shared_vals):
        _shared_vmax = np.nanpercentile(np.abs(_shared_vals), 98)
        _shared_vmax = max(_shared_vmax, 0.5)
    else:
        _shared_vmax = 1.0

    _shared_vmin = -_shared_vmax

    # Compact variable names.
    # Matplotlib mathtext does not support \scriptstyle, so use a robust form.
    _rtx = r"$R\Delta T_{\mathrm{x}}$"
    _rtn = r"$R\Delta T_{\mathrm{n}}$"

    sc_a = draw_uniform_delta_uhi_map(
        fig, ax_a, map_metric,
        value_col="delta_uhi_daytime",
        dist_col="delta_uhi_daytime",
        title=f"Daytime amplification ({_rtx}) during heatwave",
        cbar_label="",
        inset_xlabel=r"$R\Delta T_{\mathrm{x}}$",
        panel_letter="a",
        vmin=_shared_vmin,
        vmax=_shared_vmax,
        add_colorbar=False
    )


    sc_b = draw_uniform_delta_uhi_map(
        fig, ax_b, map_metric,
        value_col="delta_uhi_nighttime",
        dist_col="delta_uhi_nighttime",
        title=f"Nighttime amplification ({_rtn}) during heatwave",
        cbar_label="",
        inset_xlabel=r"$R\Delta T_{\mathrm{n}}$",
        panel_letter="b",
        vmin=_shared_vmin,
        vmax=_shared_vmax,
        add_colorbar=False
    )


    # -------------------------------------------------------------------------
    # Shared colorbar: centered in a dedicated row between panel a and panel b
    # -------------------------------------------------------------------------
    fig.canvas.draw()
    pos_cb = ax_cbar_slot.get_position()

    cbar_width = pos_cb.width * 0.28
    cbar_height = 0.010

    cbar_x = pos_cb.x0 + (pos_cb.width - cbar_width) / 2.0
    cbar_y = pos_cb.y0 + (pos_cb.height - cbar_height) / 2.0

    cax_shared = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])

    cbar = fig.colorbar(sc_b, cax=cax_shared, orientation="horizontal")

    # Keep one clear label only; avoid separate title + unit label crowding.
    cbar.set_label(
        r"$R\Delta T_{\mathrm{x}}$ (a), "
        r"$R\Delta T_{\mathrm{n}}$ (b) (°C)",
        fontsize=5.5,
        labelpad=1.0
    )

    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=4.8, length=1.6, pad=1)
    cbar.outline.set_visible(False)
    cax_shared.set_facecolor("none")


    # --- Row 2 Left: 扁长的曲线图 ---
    ax_c = fig.add_subplot(gs[3, 0])
    handles = draw_combined_fig9_delta_panel(ax_c, fig9_sum)

    ax_c.set_ylim(-1.1, 2.2)
    ax_c.set_yticks([-1, 0, 1, 2])

    title_y_c = 1.06
    ax_c.set_title(
        "Heatwave modulation of diurnal profiles",
        fontsize=8.5,
        fontweight="bold",
        y=title_y_c,
        pad=0.0
    )
    add_panel_label(ax_c, "c", x=-0.08, y=title_y_c)

    # --- Row 2 Right: 独立的图例区 ---
    ax_leg = fig.add_subplot(gs[3, 1])
    ax_leg.axis("off")  
    
    ax_leg.legend(
        handles=handles, loc="center", frameon=False,
        fontsize=7.5, labelspacing=1.0, handlelength=1.6,
        title="Thermal Regime & Period",
        title_fontproperties={'weight': 'bold', 'size': 8.5}
    )


    # =========================================================================
    # 【安全对齐修正：只改X轴，绝对不碰Y轴】
    fig.canvas.draw()
    
    pos_map = ax_b.get_position()
    pos_c = ax_c.get_position()
    pos_leg = ax_leg.get_position()
    
    # c图的宽度设定为地图宽度的 60%
    new_width_c = pos_map.width * 0.60
    
    # [核心修复]：只修改 x0 和 width。
    # 绝对保留 pos_c.y0 和 pos_c.height（由 GridSpec 安全分配，绝不上突）
    ax_c.set_position([pos_map.x0, pos_c.y0, new_width_c, pos_c.height])
    
    # 同步把图例移过来
    leg_x0 = pos_map.x0 + new_width_c + 0.05
    leg_width = pos_map.width - new_width_c - 0.05
    ax_leg.set_position([leg_x0, pos_leg.y0, leg_width, pos_leg.height])
    # =========================================================================

    ensure_dir(out_dir)
    png_path = os.path.join(out_dir, "composite_delta_uhi_daynight_de.png")
    pdf_path = os.path.join(out_dir, "composite_delta_uhi_daynight_de.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    diag.add("Composite day/night ΔUHI + combined profile figure saved", f"{png_path}\n{pdf_path}")
    print(f"  ✓ saved: {png_path}")
    print(f"  ✓ saved: {pdf_path}")

def make_composite_maps_de_figure_v2(df_hw_diurnal, df_fig9, out_dir, diag, fig_name="composite_delta_uhi_final"):
    """
    优化版：a/b 布局更紧凑，Colorbar 严格对齐地图中心。
    """
    apply_nature_style()

    write_pair_loss_trace_txt(df_hw_diurnal, df_fig9, out_dir, diag)
    fig9_sum = build_fig9_summary(df_fig9, diag)
    map_metric = build_hw_nhw_delta_uhi_daynight_from_hw_diurnal(df_hw_diurnal, diag)
    
    prof_day = get_lat_profile_stats(map_metric, "delta_uhi_daytime")
    prof_night = get_lat_profile_stats(map_metric, "delta_uhi_nighttime")

    # 保持画布比例
    fig = plt.figure(figsize=(8.0, 10.0), dpi=600) 

    # 1. 紧凑化高度分配：减少中间空隙 (height_ratios 第二项调小)
    # 2. 减小 hspace 使得 a 和 b 靠得更近
    gs = GridSpec(
        4, 1, figure=fig,
        height_ratios=[1.00, 0.05, 1.00, 0.85], 
        hspace=0.12, 
        left=0.08, right=0.92, top=0.94, bottom=0.06
    )

    _shared_vals = pd.concat([map_metric["delta_uhi_daytime"], map_metric["delta_uhi_nighttime"]]).replace([np.inf, -np.inf], np.nan).dropna()
    _shared_vmax = np.nanpercentile(np.abs(_shared_vals), 98) if len(_shared_vals) else 2.5
    _shared_vmin = -_shared_vmax
    _rtx, _rtn = r"$R\Delta T_{\mathrm{x}}$", r"$R\Delta T_{\mathrm{n}}$"

    # --- Panel a ---
    sub_gs_a = gs[0].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
    ax_a_map = fig.add_subplot(sub_gs_a[0], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    sc_a = draw_uniform_delta_uhi_map(
        fig, ax_a_map, map_metric, "delta_uhi_daytime", "delta_uhi_daytime",
        title=f"Daytime amplification ({_rtx})", cbar_label="", inset_xlabel=_rtx,
        panel_letter="a", vmin=_shared_vmin, vmax=_shared_vmax, add_colorbar=False
    )
    ax_a_prof = fig.add_subplot(sub_gs_a[1])
    draw_side_latitudinal_profile(ax_a_prof, prof_day, "#d73027", r"$\Delta$T (°C)")
    ax_a_prof.set_yticklabels([])

    # --- Panel b ---
    sub_gs_b = gs[2].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
    ax_b_map = fig.add_subplot(sub_gs_b[0], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    sc_b = draw_uniform_delta_uhi_map(
        fig, ax_b_map, map_metric, "delta_uhi_nighttime", "delta_uhi_nighttime",
        title=f"Nighttime amplification ({_rtn})", cbar_label="", inset_xlabel=_rtn,
        panel_letter="b", vmin=_shared_vmin, vmax=_shared_vmax, add_colorbar=False
    )
    ax_b_prof = fig.add_subplot(sub_gs_b[1])
    draw_side_latitudinal_profile(ax_b_prof, prof_night, "#4575b4", r"$\Delta$T (°C)")
    ax_b_prof.set_yticklabels([])

    # --- 统一对齐高度 (必须在 draw 之后) ---
    fig.canvas.draw() 
    for ax_map, ax_prof in [(ax_a_map, ax_a_prof), (ax_b_map, ax_b_prof)]:
        pos_map = ax_map.get_position()
        pos_prof = ax_prof.get_position()
        ax_prof.set_position([pos_prof.x0, pos_map.y0, pos_prof.width, pos_map.height])

    # --- Colorbar 放置：对齐地图中心 ---
    ax_cbar_slot = fig.add_subplot(gs[1])
    ax_cbar_slot.axis("off")
    
    # 核心逻辑：获取地图轴的横向位置
    p_map = ax_b_map.get_position() 
    cb_w = p_map.width * 0.4  # Colorbar 宽度设为地图宽度的 40%
    cb_h = 0.012
    # 起点 x = 地图起点 + (地图宽度 - 颜色条宽度)/2
    cb_x = p_map.x0 + (p_map.width - cb_w) / 2
    cb_y = ax_cbar_slot.get_position().y0 + 0.01 # 微调垂直高度
    
    cax_shared = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    cbar = fig.colorbar(sc_b, cax=cax_shared, orientation="horizontal")
    cbar.set_label(r"HW$-$NHW urban–rural response, $R_x$ / $R_n$ (°C)", fontsize=6, labelpad=1)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=5, length=1.5, pad=1)

    # --- Panel c ---
    ax_c = fig.add_subplot(gs[3])
    handles = draw_combined_fig9_delta_panel(ax_c, fig9_sum)
    add_panel_label(ax_c, "c", x=-0.08, y=1.06)

    # 修整 Panel c 和 Legend 的位置
    fig.canvas.draw()
    p_map_final = ax_b_map.get_position()
    p_c = ax_c.get_position()
    # c 图宽度与上方地图对齐
    ax_c.set_position([p_map_final.x0, p_c.y0, p_map_final.width, p_c.height])
    
    # 放置图例：紧跟在地图和剖面图的总宽度之后
    ax_leg = fig.add_axes([ax_b_prof.get_position().x1 - 0.12, p_c.y0, 0.15, p_c.height])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center left", frameon=False, fontsize=7, 
                  title="Regime & Period", title_fontproperties={'weight':'bold', 'size':7.5})

    # 保存
    ensure_dir(out_dir)
    fn = os.path.join(out_dir, fig_name)
    fig.savefig(fn + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(fn + ".pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    diag.add("Composite map with aligned profiles saved", fn)

def make_composite_maps_de_figure_v2_pro(df_hw_diurnal, df_fig9, out_dir, diag, fig_name="composite_delta_uhi_final2"):
    """
    优化版：a/b 布局更紧凑，Colorbar 严格对齐地图中心。
    【已修正】垂直剖面恢复为符合 final 版本的单线汇总热岛强幅图（即基于与地图变量一致的真实数据）。
    """
    apply_nature_style()

    write_pair_loss_trace_txt(df_hw_diurnal, df_fig9, out_dir, diag)
    fig9_sum = build_fig9_summary(df_fig9, diag)
    map_metric = build_hw_nhw_delta_uhi_daynight_from_hw_diurnal(df_hw_diurnal, diag)
    
    prof_day = get_lat_profile_stats(map_metric, "delta_uhi_daytime")
    prof_night = get_lat_profile_stats(map_metric, "delta_uhi_nighttime")

    # 保持画布比例
    fig = plt.figure(figsize=(8.0, 10.0), dpi=600) 

    # 1. 紧凑化高度分配：减少中间空隙 (height_ratios 第二项调小)
    # 2. 减小 hspace 使得 a 和 b 靠得更近
    gs = GridSpec(
        4, 1, figure=fig,
        height_ratios=[1.00, 0.05, 1.00, 0.85], 
        hspace=0.12, 
        left=0.08, right=0.92, top=0.94, bottom=0.06
    )

    _shared_vals = pd.concat([map_metric["delta_uhi_daytime"], map_metric["delta_uhi_nighttime"]]).replace([np.inf, -np.inf], np.nan).dropna()
    _shared_vmax = np.nanpercentile(np.abs(_shared_vals), 98) if len(_shared_vals) else 2.5
    _shared_vmin = -_shared_vmax
    _rtx, _rtn = r"$R\Delta T_{\mathrm{x}}$", r"$R\Delta T_{\mathrm{n}}$"

    # --- Panel a ---
    sub_gs_a = gs[0].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
    ax_a_map = fig.add_subplot(sub_gs_a[0], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    sc_a = draw_uniform_delta_uhi_map(
        fig, ax_a_map, map_metric, "delta_uhi_daytime", "delta_uhi_daytime",
        title=f"Daytime amplification ({_rtx})", cbar_label="", inset_xlabel=_rtx,
        panel_letter="a", vmin=_shared_vmin, vmax=_shared_vmax, add_colorbar=False
    )
    ax_a_prof = fig.add_subplot(sub_gs_a[1])
    # 【更换回标准单曲线函数以保持与 final 的剖面线结构严格吻合】
    draw_side_latitudinal_profile(ax_a_prof, prof_day, "#d73027", r"$\Delta$T (°C)")
    ax_a_prof.set_yticklabels([])

    # --- Panel b ---
    sub_gs_b = gs[2].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
    ax_b_map = fig.add_subplot(sub_gs_b[0], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    sc_b = draw_uniform_delta_uhi_map(
        fig, ax_b_map, map_metric, "delta_uhi_nighttime", "delta_uhi_nighttime",
        title=f"Nighttime amplification ({_rtn})", cbar_label="", inset_xlabel=_rtn,
        panel_letter="c", vmin=_shared_vmin, vmax=_shared_vmax, add_colorbar=False
    )
    ax_b_prof = fig.add_subplot(sub_gs_b[1])
    # 【更换回标准单曲线函数以保持与 final 的剖面线结构严格吻合】
    draw_side_latitudinal_profile(ax_b_prof, prof_night, "#4575b4", r"$\Delta$T (°C)")
    ax_b_prof.set_yticklabels([])

    # --- 统一对齐高度 (必须在 draw 之后) ---
    fig.canvas.draw() 
    for ax_map, ax_prof in [(ax_a_map, ax_a_prof), (ax_b_map, ax_b_prof)]:
        pos_map = ax_map.get_position()
        pos_prof = ax_prof.get_position()
        ax_prof.set_position([pos_prof.x0, pos_map.y0, pos_prof.width, pos_map.height])

    # --- Added Profile Labels ---
    add_panel_label(ax_a_prof, "b", x=-0.20, y=1.06)
    add_panel_label(ax_b_prof, "d", x=-0.20, y=1.06)

    # --- Colorbar 放置：对齐地图中心 ---
    ax_cbar_slot = fig.add_subplot(gs[1])
    ax_cbar_slot.axis("off")
    
    # 核心逻辑：获取地图轴的横向位置
    p_map = ax_b_map.get_position() 
    cb_w = p_map.width * 0.4  # Colorbar 宽度设为地图宽度的 40%
    cb_h = 0.012
    # 起点 x = 地图起点 + (地图宽度 - 颜色条宽度)/2
    cb_x = p_map.x0 + (p_map.width - cb_w) / 2
    cb_y = ax_cbar_slot.get_position().y0 + 0.01 # 微调垂直高度
    
    cax_shared = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    cbar = fig.colorbar(sc_b, cax=cax_shared, orientation="horizontal")
    cbar.set_label(r"HW$-$NHW urban–rural response, $R_x$ / $R_n$ (°C)", fontsize=6, labelpad=1)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=5, length=1.5, pad=1)

    # --- Panel c ---
    ax_c = fig.add_subplot(gs[3])
    handles = draw_combined_fig9_delta_panel(ax_c, fig9_sum)
    add_panel_label(ax_c, "e", x=-0.08, y=1.06)

    # 修整 Panel c 和 Legend 的位置
    fig.canvas.draw()
    p_map_final = ax_b_map.get_position()
    p_c = ax_c.get_position()
    # c 图宽度与上方地图对齐
    ax_c.set_position([p_map_final.x0, p_c.y0, p_map_final.width, p_c.height])
    
    # 放置图例：紧跟在地图和剖面图的总宽度之后
    ax_leg = fig.add_axes([ax_b_prof.get_position().x1 - 0.12, p_c.y0, 0.15, p_c.height])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center left", frameon=False, fontsize=7, 
                  title="Regime & Period", title_fontproperties={'weight':'bold', 'size':7.5})

    # 保存
    ensure_dir(out_dir)
    fn = os.path.join(out_dir, fig_name)
    fig.savefig(fn + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(fn + ".pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    diag.add("Composite map with aligned profiles saved", fn)

def make_composite_maps_de_figure_v3(df_hw_diurnal, df_fig9, out_dir, diag, fig_name="composite_delta_uhi_final"):
    """
    优化版：a/b 布局更紧凑，Colorbar 严格对齐地图中心。
    【修改点】右侧纬度垂直剖面图改为站点的 HW-NHW 绝对温差（Delta Tx, Delta Tn）。
    """
    apply_nature_style()

    write_pair_loss_trace_txt(df_hw_diurnal, df_fig9, out_dir, diag)
    fig9_sum = build_fig9_summary(df_fig9, diag)
    
    # 1. 地图所需的数据：保持为 RTx 和 RTn (delta_uhi_daytime/nighttime)
    map_metric = build_hw_nhw_delta_uhi_daynight_from_hw_diurnal(df_hw_diurnal, diag)
    
    # 2. 【修改核心】垂直剖面所需的数据：改为读取站点级别的 HW - NHW (Delta Tx/Tn)
    df_day_lat = build_latitudinal_hw_nhw_single_station(df_hw_diurnal, diag, metric="Tmax_fft")
    df_night_lat = build_latitudinal_hw_nhw_single_station(df_hw_diurnal, diag, metric="Tmin_fft")

    # 获取分箱统计（注意：build_latitudinal_hw_nhw_single_station 的输出列名固定为 "delta_tx"）
    prof_day = get_lat_profile_stats(df_day_lat, "delta_tx")
    prof_night = get_lat_profile_stats(df_night_lat, "delta_tx") 

    # 保持画布比例
    fig = plt.figure(figsize=(8.0, 10.0), dpi=600) 

    # 1. 紧凑化高度分配：减少中间空隙 (height_ratios 第二项调小)
    # 2. 减小 hspace 使得 a 和 b 靠得更近
    gs = GridSpec(
        4, 1, figure=fig,
        height_ratios=[1.00, 0.05, 1.00, 0.85], 
        hspace=0.12, 
        left=0.08, right=0.92, top=0.94, bottom=0.06
    )

    _shared_vals = pd.concat([map_metric["delta_uhi_daytime"], map_metric["delta_uhi_nighttime"]]).replace([np.inf, -np.inf], np.nan).dropna()
    _shared_vmax = np.nanpercentile(np.abs(_shared_vals), 98) if len(_shared_vals) else 2.5
    _shared_vmin = -_shared_vmax
    _rtx, _rtn = r"$R\Delta T_{\mathrm{x}}$", r"$R\Delta T_{\mathrm{n}}$"

    # --- Panel a ---
    sub_gs_a = gs[0].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
    ax_a_map = fig.add_subplot(sub_gs_a[0], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    sc_a = draw_uniform_delta_uhi_map(
        fig, ax_a_map, map_metric, "delta_uhi_daytime", "delta_uhi_daytime",
        title=f"Daytime amplification ({_rtx})", cbar_label="", inset_xlabel=_rtx,
        panel_letter="a", vmin=_shared_vmin, vmax=_shared_vmax, add_colorbar=False
    )
    ax_a_prof = fig.add_subplot(sub_gs_a[1])
    # 将 X 轴标签更新为 $\Delta$Tx，反映实际绘制变量
    draw_side_latitudinal_profile(ax_a_prof, prof_day, "#d73027", r"$\Delta$Tx (°C)")
    ax_a_prof.set_yticklabels([])

    # --- Panel b ---
    sub_gs_b = gs[2].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
    ax_b_map = fig.add_subplot(sub_gs_b[0], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    sc_b = draw_uniform_delta_uhi_map(
        fig, ax_b_map, map_metric, "delta_uhi_nighttime", "delta_uhi_nighttime",
        title=f"Nighttime amplification ({_rtn})", cbar_label="", inset_xlabel=_rtn,
        panel_letter="b", vmin=_shared_vmin, vmax=_shared_vmax, add_colorbar=False
    )
    ax_b_prof = fig.add_subplot(sub_gs_b[1])
    # 将 X 轴标签更新为 $\Delta$Tn，反映实际绘制变量
    draw_side_latitudinal_profile(ax_b_prof, prof_night, "#4575b4", r"$\Delta$Tn (°C)")
    ax_b_prof.set_yticklabels([])

    # --- 统一对齐高度 (必须在 draw 之后) ---
    fig.canvas.draw() 
    for ax_map, ax_prof in [(ax_a_map, ax_a_prof), (ax_b_map, ax_b_prof)]:
        pos_map = ax_map.get_position()
        pos_prof = ax_prof.get_position()
        ax_prof.set_position([pos_prof.x0, pos_map.y0, pos_prof.width, pos_map.height])

    # --- Colorbar 放置：对齐地图中心 ---
    ax_cbar_slot = fig.add_subplot(gs[1])
    ax_cbar_slot.axis("off")
    
    # 核心逻辑：获取地图轴的横向位置
    p_map = ax_b_map.get_position() 
    cb_w = p_map.width * 0.4  # Colorbar 宽度设为地图宽度的 40%
    cb_h = 0.012
    # 起点 x = 地图起点 + (地图宽度 - 颜色条宽度)/2
    cb_x = p_map.x0 + (p_map.width - cb_w) / 2
    cb_y = ax_cbar_slot.get_position().y0 + 0.01 # 微调垂直高度
    
    cax_shared = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    cbar = fig.colorbar(sc_b, cax=cax_shared, orientation="horizontal")
    cbar.set_label(r"HW$-$NHW urban–rural response, $R_x$ / $R_n$ (°C)", fontsize=6, labelpad=1)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=5, length=1.5, pad=1)

    # --- Panel c ---
    ax_c = fig.add_subplot(gs[3])
    handles = draw_combined_fig9_delta_panel(ax_c, fig9_sum)
    add_panel_label(ax_c, "c", x=-0.08, y=1.06)

    # 修整 Panel c 和 Legend 的位置
    fig.canvas.draw()
    p_map_final = ax_b_map.get_position()
    p_c = ax_c.get_position()
    # c 图宽度与上方地图对齐
    ax_c.set_position([p_map_final.x0, p_c.y0, p_map_final.width, p_c.height])
    
    # 放置图例：紧跟在地图和剖面图的总宽度之后
    ax_leg = fig.add_axes([ax_b_prof.get_position().x1 - 0.12, p_c.y0, 0.15, p_c.height])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center left", frameon=False, fontsize=7, 
                  title="Regime & Period", title_fontproperties={'weight':'bold', 'size':7.5})

    # 保存
    ensure_dir(out_dir)
    fn = os.path.join(out_dir, fig_name)
    fig.savefig(fn + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(fn + ".pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    diag.add("Composite map with aligned profiles saved", fn)

def _draw_urban_rural_vertical_profile(ax, df_lat, color, xlabel, show_legend=False):
    """
    辅助函数：在侧边栏绘制 Urban (实线) vs Rural (细线) 的纬度对比
    """
    df_lat = df_lat.copy()
    df_lat["_station_type_norm"] = df_lat["station_type"].astype(str).str.lower()

    # 线型配置：Urban 较粗，Rural 较细
    specs = [
        ("urban", "Urban", 1.55),
        ("rural", "Rural", 0.85),
    ]

    has_data = False
    for station_type, label, lw in specs:
        # 调用已有的分箱统计函数
        prof = get_lat_profile_stats(
            df_lat[df_lat["_station_type_norm"] == station_type],
            "delta_tx"
        )
        if prof is None or len(prof) == 0:
            continue

        has_data = True
        y = prof["lat_mid"].values
        x = prof["mean"].values
        err = prof["sem"].fillna(0).values

        # 填充 SEM 阴影
        ax.fill_betweenx(y, x - err, x + err, color=color, alpha=0.10, lw=0)
        # 绘制主线
        ax.plot(x, y, color=color, lw=lw, ls="-", label=label)

    if not has_data:
        ax.axis("off")
        return

    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_ylim(-60, 85)
    ax.set_xlabel(xlabel, fontsize=5.5)
    ax.tick_params(axis="both", labelsize=5, length=2, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls=":", lw=0.4, alpha=0.3)

    if show_legend:
        ax.legend(frameon=False, loc="upper right", fontsize=4.8, handlelength=1.4)

def draw_uniform_delta_uhi_map(
    fig, ax, map_metric, value_col, dist_col,
    title, cbar_label, inset_xlabel, panel_letter,
    vmin=None, vmax=None, add_colorbar=True
):
    """
    绘制 Robinson 投影地图，并优化左下角直方图 Inset 的坐标轴显示。
    """
    import cartopy.mpl.geoaxes
    valid = map_metric.dropna(subset=["lon_ref", "lat_ref", value_col]).copy()
    if vmin is None or vmax is None:
        vals = valid[value_col].replace([np.inf, -np.inf], np.nan).dropna()
        _vmax = max(np.nanpercentile(np.abs(vals), 98) if len(vals) else 1.0, 0.5)
        vmin, vmax = -_vmax, _vmax

    is_geo_ax = isinstance(ax, cartopy.mpl.geoaxes.GeoAxes)

    if is_geo_ax:
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#f7f7f7', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.30, edgecolor="#606060", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.15, edgecolor="#aaaaaa", linestyle=":", zorder=1)

        gl = ax.gridlines(crs=DATA_CRS, draw_labels=True, linewidth=0.20, color="#cccccc", alpha=0.5)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 5.0}

        sc = ax.scatter(
            valid["lon_ref"].values, valid["lat_ref"].values,
            transform=DATA_CRS, c=valid[value_col].values,
            cmap=MAP_CMAP, vmin=vmin, vmax=vmax,
            s=12, marker="o", linewidths=0.25, edgecolors="black", alpha=0.88, zorder=3
        )

        for s in ['left', 'right', 'top', 'bottom']:
            ax.spines[s].set_visible(False)
        if 'geo' in ax.spines:
            ax.spines['geo'].set_visible(True)
            ax.spines['geo'].set_linewidth(0.60)
            ax.spines['geo'].set_edgecolor("black")
    else:
        sc = ax.scatter(valid["lon_ref"].values, valid["lat_ref"].values,
                        c=valid[value_col].values, cmap=MAP_CMAP, vmin=vmin, vmax=vmax,
                        s=12, marker="o", edgecolors="black", alpha=0.88)

    # Subtitle 保持在地图上方固定位置
    ax.set_title(title, fontsize=7.2, fontweight="bold", y=1.04)
    add_panel_label(ax, panel_letter, x=-0.04, y=1.04)

    if add_colorbar:
        cax = mpl_inset_axes(ax, width="25%", height="4.5%", loc="lower center", borderpad=1.5)
        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.outline.set_linewidth(0.60)

    # --- [修改部分] 为 Inset Hist 添加坐标轴 ---
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as mpl_inset_axes_local
    ax_in = mpl_inset_axes_local(ax, width="18%", height="18%", loc="lower left", borderpad=2.2)
    ax_in.hist(valid[dist_col].dropna(), bins=25, density=True, color="gray", alpha=0.7)
    
    # 添加 X 轴标签和刻度
    ax_in.set_xlabel(inset_xlabel, fontsize=5.8, labelpad=1)
    ax_in.tick_params(axis='x', labelsize=4.8, length=1.2, pad=0.8)
    ax_in.set_yticks([]) # Y 轴密度通常不显示，保持简洁
    for s in ax_in.spines.values():
        s.set_linewidth(0.4)
    
    return sc

def make_composite_maps_de_figure_v42(df_hw_diurnal, df_fig9, out_dir, diag, fig_name="composite_delta_uhi_final_ver"):
    """
    主图绘制：优化 a/b 子图布局，防止标题、色标相互遮挡，并添加 Inset 坐标。
    """
    apply_nature_style()
    fig9_sum = build_fig9_summary(df_fig9, diag)
    map_metric = build_hw_nhw_delta_uhi_daynight_from_hw_diurnal(df_hw_diurnal, diag)
    df_day_lat = build_latitudinal_hw_nhw_single_station(df_hw_diurnal, diag, metric="Tmax_fft")
    df_night_lat = build_latitudinal_hw_nhw_single_station(df_hw_diurnal, diag, metric="Tmin_fft")

    fig = plt.figure(figsize=(7.2, 9.15), dpi=600)
    
    # [布局优化] 增加 hspace (0.32)，减小 top (0.93)
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.0, 0.12, 1.0, 0.75], 
                  hspace=0.32, left=0.07, right=0.98, top=0.93, bottom=0.06)

    _shared_vals = pd.concat([map_metric["delta_uhi_daytime"], map_metric["delta_uhi_nighttime"]]).dropna()
    _vmax = max(np.nanpercentile(np.abs(_shared_vals), 98) if len(_shared_vals) else 2.5, 0.5)
    _rtx, _rtn = r"$RT_{\mathrm{x}}$", r"$RT_{\mathrm{n}}$"

    # --- Panel a ---
    sub_gs_a = gs[0].subgridspec(1, 2, width_ratios=[5.2, 1], wspace=0.04)
    ax_a_map = fig.add_subplot(sub_gs_a[0], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    sc_a = draw_uniform_delta_uhi_map(fig, ax_a_map, map_metric, "delta_uhi_daytime", "delta_uhi_daytime",
                                     f"Daytime amplification ({_rtx})", "", _rtx, "a", -_vmax, _vmax, False)
    ax_a_prof = fig.add_subplot(sub_gs_a[1])
    _draw_urban_rural_vertical_profile(ax_a_prof, df_day_lat, "#FF7F00", rf"{_rtx} (°C)", True)
    ax_a_prof.xaxis.label.set_size(8.5)
    ax_a_prof.tick_params(labelsize=7.5)
    add_panel_label(ax_a_prof, "b", x=-0.18, y=1.02)

    # --- Panel b ---
    sub_gs_b = gs[2].subgridspec(1, 2, width_ratios=[5.2, 1], wspace=0.04)
    ax_b_map = fig.add_subplot(sub_gs_b[0], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    sc_b = draw_uniform_delta_uhi_map(fig, ax_b_map, map_metric, "delta_uhi_nighttime", "delta_uhi_nighttime",
                                     f"Nighttime amplification ({_rtn})", "", _rtn, "c", -_vmax, _vmax, False)
    ax_b_prof = fig.add_subplot(sub_gs_b[1])
    _draw_urban_rural_vertical_profile(ax_b_prof, df_night_lat, "#A04000", rf"{_rtn} (°C)", False)
    ax_b_prof.xaxis.label.set_size(8.5)
    ax_b_prof.tick_params(labelsize=7.5)
    add_panel_label(ax_b_prof, "d", x=-0.18, y=1.02)

    # --- Colorbar Slot (位于 a 和 b 之间) ---
    ax_cbar_slot = fig.add_subplot(gs[1]); ax_cbar_slot.axis("off")
    fig.canvas.draw()
    p_slot = ax_cbar_slot.get_position()
    p_map = ax_b_map.get_position()
    p_prof = ax_b_prof.get_position()
    
    cb_w, cb_h = p_map.width * 0.45, 0.012
    # 将色标稍微抬高，确保不遮挡下方 b 的标题
    cax = fig.add_axes([p_map.x0 + (p_map.width - cb_w)/2, p_slot.y0 + (p_slot.height - cb_h)/2 + 0.005, cb_w, cb_h])
    cbar = fig.colorbar(sc_b, cax=cax, orientation="horizontal")
    cbar.set_label(r"HW$-$NHW urban–rural response, $R_x$ / $R_n$ (°C)", fontsize=8.0, labelpad=1.0)
    cbar.ax.tick_params(labelsize=7.0)

    # --- Panel c (维持与地图 b 宽度对齐，图例在右) ---
    ax_c = fig.add_subplot(gs[3])
    handles = draw_combined_fig9_delta_panel(ax_c, fig9_sum)
    add_panel_label(ax_c, "e", x=-0.06, y=1.02)
    ax_c.xaxis.label.set_size(9.0)
    ax_c.yaxis.label.set_size(9.0)
    ax_c.tick_params(labelsize=7.5)
    
    # 宽度对齐地图轴 b
    ax_c.set_position([p_map.x0, ax_c.get_position().y0, p_map.width, ax_c.get_position().height])

    # 图例放置在 c 右侧空白区
    ax_leg = fig.add_axes([p_prof.x0, ax_c.get_position().y0, p_prof.width, ax_c.get_position().height])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center left", frameon=False, fontsize=8.0, 
                  title="Regime & Period", title_fontproperties={'weight':'bold', 'size':8.5})

    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, fig_name + ".png"), **SAVEFIG_KW)
    plt.close(fig)


def make_composite_maps_de_figure_v4(df_hw_diurnal, df_fig9, out_dir, diag, fig_name="composite_delta_uhi_final_fianl_ver"):
    """
    主图绘制：优化 a/b 子图布局，防止标题、色标相互遮挡，并添加 Inset 坐标。
    """
    apply_nature_style()
    fig9_sum = build_fig9_summary(df_fig9, diag)
    map_metric = build_hw_nhw_delta_uhi_daynight_from_hw_diurnal(df_hw_diurnal, diag)
    df_day_lat = build_latitudinal_hw_nhw_single_station(df_hw_diurnal, diag, metric="Tmax_fft")
    df_night_lat = build_latitudinal_hw_nhw_single_station(df_hw_diurnal, diag, metric="Tmin_fft")

    fig = plt.figure(figsize=(7.2, 9.15), dpi=600)
    
    gs = GridSpec(
        4, 1,
        figure=fig,
        height_ratios=[1.0, 0.12, 1.0, 0.75],
        hspace=0.32,
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.06
    )

    _shared_vals = pd.concat([
        map_metric["delta_uhi_daytime"],
        map_metric["delta_uhi_nighttime"]
    ]).dropna()

    _vmax = max(
        np.nanpercentile(np.abs(_shared_vals), 98) if len(_shared_vals) else 2.5,
        0.5
    )

    _rtx, _rtn = r"$RT_{\mathrm{x}}$", r"$RT_{\mathrm{n}}$"
    _rx, _rn = r"$R_{\mathrm{x}}$", r"$R_{\mathrm{n}}$"

    TITLE_FS = 8.8
    PANEL_FS = 9.2
    TITLE_PAD = 7

    def _set_center_subtitle(ax, title):
        ax.set_title(
            title,
            loc="center",
            fontsize=TITLE_FS,
            fontweight="normal",
            pad=TITLE_PAD
        )

    def _add_outer_panel_label(ax, label, x=-0.08, y=1.035):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=PANEL_FS,
            fontweight="bold",
            clip_on=False
        )

    # --- Panel a ---
    sub_gs_a = gs[0].subgridspec(1, 2, width_ratios=[5.2, 1], wspace=0.04)

    ax_a_map = fig.add_subplot(
        sub_gs_a[0],
        projection=PLOT_PROJECTION if HAS_CARTOPY else None
    )

    sc_a = draw_uniform_delta_uhi_map(
        fig,
        ax_a_map,
        map_metric,
        "delta_uhi_daytime",
        "delta_uhi_daytime",
        "",
        "",
        _rx,
        "",
        -_vmax,
        _vmax,
        False
    )

    _set_center_subtitle(
        ax_a_map,
        rf"Daytime HW–NHW urban–rural response ({_rx})"
    )
    _add_outer_panel_label(ax_a_map, "a", x=-0.035, y=1.035)

    ax_a_prof = fig.add_subplot(sub_gs_a[1])

    _draw_urban_rural_vertical_profile(
        ax_a_prof,
        df_day_lat,
        "#FF7F00",
        rf"{_rtx} (°C)",
        True
    )

    ax_a_prof.xaxis.label.set_size(8.5)
    ax_a_prof.tick_params(labelsize=7.5)

    ax_a_prof.set_ylim(-90, 90)
    ax_a_prof.set_ylabel("Latitude (°)", fontsize=8.5, labelpad=3)
    ax_a_prof.yaxis.set_label_position("left")
    ax_a_prof.yaxis.tick_left()
    ax_a_prof.set_yticks([-90, -60, -30, 0, 30, 60, 90])

    ax_a_prof.legend(
        loc="lower right",
        frameon=False,
        fontsize=7.5,
        handlelength=1.4,
        handletextpad=0.4,
        borderaxespad=0.2
    )

    _set_center_subtitle(
        ax_a_prof,
        rf"Station daytime HW–NHW" "\n"
        rf"temperature response, {_rtx}"
    )
    _add_outer_panel_label(ax_a_prof, "b", x=-0.4, y=1.035)

    # --- Panel c ---
    sub_gs_b = gs[2].subgridspec(1, 2, width_ratios=[5.2, 1], wspace=0.04)

    ax_b_map = fig.add_subplot(
        sub_gs_b[0],
        projection=PLOT_PROJECTION if HAS_CARTOPY else None
    )

    sc_b = draw_uniform_delta_uhi_map(
        fig,
        ax_b_map,
        map_metric,
        "delta_uhi_nighttime",
        "delta_uhi_nighttime",
        "",
        "",
        _rn,
        "",
        -_vmax,
        _vmax,
        False
    )

    _set_center_subtitle(
        ax_b_map,
        rf"Nighttime HW–NHW urban–rural response ({_rn})"
    )
    _add_outer_panel_label(ax_b_map, "c", x=-0.035, y=1.035)

    ax_b_prof = fig.add_subplot(sub_gs_b[1])

    # 关键修复：恢复 d 图的数据绘制
    _draw_urban_rural_vertical_profile(
        ax_b_prof,
        df_night_lat,
        "#A04000",
        rf"{_rtn} (°C)",
        False
    )

    ax_b_prof.xaxis.label.set_size(8.5)
    ax_b_prof.tick_params(labelsize=7.5)

    ax_b_prof.set_ylim(-90, 90)
    ax_b_prof.set_ylabel("Latitude (°)", fontsize=8.5, labelpad=3)
    ax_b_prof.yaxis.set_label_position("left")
    ax_b_prof.yaxis.tick_left()
    ax_b_prof.set_yticks([-90, -60, -30, 0, 30, 60, 90])

    ax_b_prof.legend(
        loc="lower right",
        frameon=False,
        fontsize=7.5,
        handlelength=1.4,
        handletextpad=0.4,
        borderaxespad=0.2
    )

    _set_center_subtitle(
        ax_b_prof,
        rf"Station nighttime HW–NHW" "\n"
        rf"temperature response, {_rtn}"
    )
    _add_outer_panel_label(ax_b_prof, "d", x=-0.4, y=1.035)

    # --- Colorbar Slot ---
    ax_cbar_slot = fig.add_subplot(gs[1])
    ax_cbar_slot.axis("off")

    fig.canvas.draw()

    # Align profile panels b/d vertically with their corresponding maps a/c
    p_a_map = ax_a_map.get_position()
    p_a_prof = ax_a_prof.get_position()

    ax_a_prof.set_position([
        p_a_prof.x0,
        p_a_map.y0,
        p_a_prof.width,
        p_a_map.height
    ])

    p_b_map = ax_b_map.get_position()
    p_b_prof = ax_b_prof.get_position()

    ax_b_prof.set_position([
        p_b_prof.x0,
        p_b_map.y0,
        p_b_prof.width,
        p_b_map.height
    ])

    fig.canvas.draw()

    p_slot = ax_cbar_slot.get_position()
    p_map = ax_b_map.get_position()
    p_prof = ax_b_prof.get_position()

    cb_w, cb_h = p_map.width * 0.45, 0.012

    cax = fig.add_axes([
        p_map.x0 + (p_map.width - cb_w) / 2,
        p_slot.y0 + (p_slot.height - cb_h) / 2 + 0.005,
        cb_w,
        cb_h
    ])

    cbar = fig.colorbar(sc_b, cax=cax, orientation="horizontal")
    cbar.set_label(
        r"HW$-$NHW urban–rural response, $R_x$ / $R_n$ (°C)",
        fontsize=8.0,
        labelpad=1.0
    )
    cbar.ax.tick_params(labelsize=7.0)

    # --- Panel e ---
    ax_c = fig.add_subplot(gs[3])

    handles = draw_combined_fig9_delta_panel(ax_c, fig9_sum)

    ax_c.set_ylim(-2.5, 2.5)

    _set_center_subtitle(
        ax_c,
        r"Diurnal urban–rural thermal contrasts ($\Delta T$) under HW and NHW"
    )
    _add_outer_panel_label(ax_c, "e", x=-0.035, y=1.035)

    ax_c.xaxis.label.set_size(9.0)
    ax_c.yaxis.label.set_size(9.0)
    ax_c.tick_params(labelsize=7.5)
    
    # 宽度对齐地图轴 c
    ax_c.set_position([
        p_map.x0,
        ax_c.get_position().y0,
        p_map.width,
        ax_c.get_position().height
    ])

    # 图例放置在 e 右侧空白区
    ax_leg = fig.add_axes([
        p_prof.x0,
        ax_c.get_position().y0,
        p_prof.width,
        ax_c.get_position().height
    ])
    ax_leg.axis("off")

    ax_leg.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=8.0,
        title="Regime & Period",
        title_fontproperties={"weight": "bold", "size": 8.5}
    )

    # Main title removed; panel labels and subtitles are retained.

    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, fig_name + ".png"), **SAVEFIG_KW)
    plt.close(fig)


# ============================================================================
# §10 Plot helpers
# ============================================================================

# ── CHANGED: climate-zone label moved ABOVE the subplot boundary ─────────────
def draw_diurnal_panel(ax, df_diurnal_summary, zone):
    """
    NCC-style diurnal profile: clean, consistent, and well-labeled.
    """
    sub = df_diurnal_summary[
        (df_diurnal_summary["climate4"] == zone) &
        (df_diurnal_summary["period_norm"].isin(TOP_PERIODS))
    ].copy()

    # 1. 基础背景设置
    ax.set_facecolor("none") # 保持透明背景，增加呼吸感

    # 2. 绘制曲线
    for p in TOP_PERIODS:
        sp = sub[sub["period_norm"] == p].sort_values("hour")
        if len(sp) == 0: continue
        
        x = sp["hour"].values
        y = sp["Ta_mean"].values
        s = sp["Ta_std"].fillna(0).values

        # HW (热浪期) 稍微加粗，强调核心数据
        line_width = 1.3 if p == "HW" else 0.9
        line_alpha = 1.0 if p == "HW" else 0.8
        
        ax.plot(x, y, color=CLR_PERIOD[p], lw=line_width, label=p, zorder=3, alpha=line_alpha)
        # 阴影透明度要低，避免遮挡网格线
        ax.fill_between(x, y - s, y + s, color=CLR_PERIOD[p], alpha=0.07, zorder=2, linewidth=0)

    # 3. 坐标轴范围与刻度
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    # 只有最左边的图显示 ylabel，或者统一精简
    ax.set_xlabel("Local Solar Time (h)", fontsize=6.0)
    
    # 4. 【核心修改】调用统一的轴样式
    # 确保 style_small_axis(ax) 内部已经统一为 0.60 的线宽
    style_small_axis(ax)
    
    # 5. 精修刻度线（Nature 风格：刻度线短而细）
    ax.tick_params(axis='both', which='major', labelsize=5.5, length=2, width=0.60, pad=1)

    # 6. 【核心修改】气候区标签：加宽呼吸感
    # 在 NCC 这种多 Panel 图中，顶部标签的颜色应该与地图中的区域颜色对应
    ax.text(
        0.50, 1.08, # 稍微调高一点，避免和轴线太挤
        CLIMATE_NAME.get(zone, zone),
        transform=ax.transAxes,
        fontsize=7.0, fontweight="bold", # 稍微加大字体
        ha="center", va="bottom",
        color=ZONE_EDGE.get(zone, "black"),
        clip_on=False
    )

    # 7. 添加日夜背景（可选，但强烈推荐用于日循环研究）
    # 浅灰色背景表示夜晚，让读者一眼看出温度波动的相位
    # ax.axvspan(0, 6, color='gray', alpha=0.03, zorder=1)
    # ax.axvspan(18, 23, color='gray', alpha=0.03, zorder=1)

def draw_fig9_abs_panel(ax, s_hw, s_nhw, group_label, show_legend=False, nhw_color=None, hw_color=None):
    # 修改颜色赋值逻辑，优先使用传入的同色系颜色
    c_hw  = hw_color if hw_color else CLR_PERIOD["HW"]
    c_nhw = nhw_color if nhw_color else CLR_PERIOD["NHW"]

    ax.plot(HOURS, s_nhw["u_mean"], color=c_nhw, lw=1.5, ls="-",  label="Urban / NHW")
    ax.plot(HOURS, s_nhw["r_mean"], color=c_nhw, lw=1.2, ls="--", label="Rural / NHW")
    ax.fill_between(HOURS,
                    s_nhw["u_mean"] - s_nhw["u_std"],
                    s_nhw["u_mean"] + s_nhw["u_std"],
                    color=c_nhw, alpha=0.08)
    ax.fill_between(HOURS,
                    s_nhw["r_mean"] - s_nhw["r_std"],
                    s_nhw["r_mean"] + s_nhw["r_std"],
                    color=c_nhw, alpha=0.05)

    ax.plot(HOURS, s_hw["u_mean"], color=c_hw, lw=1.8, ls="-",  label="Urban / HW")
    ax.plot(HOURS, s_hw["r_mean"], color=c_hw, lw=1.4, ls="--", label="Rural / HW")
    ax.fill_between(HOURS,
                    s_hw["u_mean"] - s_hw["u_std"],
                    s_hw["u_mean"] + s_hw["u_std"],
                    color=c_hw, alpha=0.10)
    ax.fill_between(HOURS,
                    s_hw["r_mean"] - s_hw["r_std"],
                    s_hw["r_mean"] + s_hw["r_std"],
                    color=c_hw, alpha=0.06)

    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Local Solar Time (h)", labelpad=5)
    ax.set_ylabel(r"$T_a$ (°C)")
    ax.set_title(f"{group_label}: urban and rural absolute temperature",
                 fontsize=6.5, fontweight="bold", pad=1.5)
    style_small_axis(ax)

    if show_legend:
        ax.legend(frameon=False, loc="upper left", fontsize=5.6, ncol=2)

def draw_fig9_delta_panel(ax, s_hw, s_nhw, group_label, show_legend=False, nhw_color=None, hw_color=None):
    # 修改参数和赋值逻辑，确保 NHW 不再硬编码为蓝色
    c_hw  = hw_color if hw_color else CLR_PERIOD["HW"]
    c_nhw = nhw_color if nhw_color else CLR_PERIOD["NHW"]

    ax.plot(HOURS, s_nhw["delta"], color=c_nhw, lw=1.4, ls="--", label="NHW")
    ax.fill_between(HOURS, 0, s_nhw["delta"], color=c_nhw, alpha=0.07)

    ax.plot(HOURS, s_hw["delta"], color=c_hw, lw=1.9, ls="-", label="HW")
    ax.fill_between(HOURS, 0, s_hw["delta"], color=c_hw, alpha=0.10)

    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Local Solar Time (h)", labelpad=5)
    ax.set_ylabel(r"$\Delta T_a$ (Urban-Rural, °C)")
    ax.set_title(f"{group_label}: urban-rural thermal contrast",
                 fontsize=6.5, fontweight="bold", pad=1.5)
    style_small_axis(ax)

    if show_legend:
        ax.legend(frameon=False, loc="upper left", fontsize=5.8)

def draw_combined_fig9_delta_panel(ax, fig9_sum):
    """
    [精调] 合并 UHI 和 UCI 曲线。
    """
    # --- 统一配色 ---
    c_uhi = "#FF7F00"  # UHI: 亮橙色 (与地图和 profile 一致)
    c_uci = "#A04000"  # UCI: 砖红色 (与地图和 profile 一致)

    s_uhi_hw  = fig9_sum[("UHI", "HW")]
    s_uhi_nhw = fig9_sum[("UHI", "NHW")]
    s_uci_hw  = fig9_sum[("UCI", "HW")]
    s_uci_nhw = fig9_sum[("UCI", "NHW")]

    # 绘制 UHI 曲线
    ax.plot(HOURS, s_uhi_nhw["delta"], color=c_uhi, lw=1.2, ls="--", zorder=3)
    ax.fill_between(HOURS, 0, s_uhi_nhw["delta"], color=c_uhi, alpha=0.04, zorder=1)
    ax.plot(HOURS, s_uhi_hw["delta"], color=c_uhi, lw=2.0, ls="-", zorder=4)

    # 绘制 UCI 曲线
    ax.plot(HOURS, s_uci_nhw["delta"], color=c_uci, lw=1.2, ls="--", zorder=3)
    ax.fill_between(HOURS, 0, s_uci_hw["delta"], color=c_uci, alpha=0.04, zorder=1)
    ax.plot(HOURS, s_uci_hw["delta"], color=c_uci, lw=2.0, ls="-", zorder=4)

    ax.axhline(0, color="black", lw=0.6, ls=":", zorder=5)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Local Solar Time (h)", labelpad=4, fontsize=8)
    ax.set_ylabel(r"$\Delta T_a$ (Urban$-$Rural, °C)", fontsize=8, labelpad=6)
    
    # --- 修改背景遮罩颜色为浅绿/浅紫 ---
    c_day_bg = "#A6DBA0"   # 浅绿色
    c_night_bg = "#C2A5CF" # 浅紫色
    
    ax.axvspan(6, 18, color=c_day_bg, alpha=0.12, zorder=0)     # 白天：绿
    ax.axvspan(18, 24, color=c_night_bg, alpha=0.12, zorder=0)  # 晚上：紫
    ax.axvspan(0, 6, color=c_night_bg, alpha=0.12, zorder=0)

    style_small_axis(ax)

    handles = [
        # HW 使用实心线条，NHW 使用虚线或较淡的颜色
        Line2D([0], [0], color=c_uhi, lw=2.0, ls="-",  label="UHI (HW, Solid)"),
        Line2D([0], [0], color=c_uhi, lw=1.2, ls="--", label="UHI (NHW, Dash)"),
        Line2D([0], [0], color=c_uci, lw=2.0, ls="-",  label="UCI (HW, Solid)"),
        Line2D([0], [0], color=c_uci, lw=1.2, ls="--", label="UCI (NHW, Dash)"),
    ]
    return handles

def grouped_boxplot_compact(ax, df, value_col, ylabel, title,
                             show_legend=False, zero_line=False, is_delta=False):
    width   = 0.16
    offsets = {"HW": -width, "NHW": 0.0, "JJA": +width}
    xs      = np.arange(len(CLIMATE_ORDER), dtype=float)
    rng     = np.random.default_rng(42)

    for p in BOX_PERIODS:
        color = CLR_BOX[p]
        for i, z in enumerate(CLIMATE_ORDER):
            sub = df[(df["climate4"] == z) &
                     (df["period_norm"] == p)][value_col].dropna()
            if len(sub) < 3:
                continue

            pos = xs[i] + offsets[p]
            ax.boxplot(
                [sub.values],
                positions=[pos],
                widths=0.11,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, edgecolor="black",
                              linewidth=0.45, alpha=0.88),
                medianprops=dict(color="white", linewidth=0.80),
                whiskerprops=dict(color="black", linewidth=0.45),
                capprops=dict(color="black", linewidth=0.45),
                manage_ticks=False
            )

            jitter_x = pos + rng.normal(0, 0.009, size=len(sub))
            ax.scatter(jitter_x, sub.values,
                       s=1.8, color="black", alpha=0.08, linewidths=0, zorder=2)
            ax.scatter([pos], [sub.mean()],
                       s=7, color="white", edgecolor="black", linewidth=0.35, zorder=5)

    if zero_line:
        ax.axhline(0, color="black", linestyle=":", linewidth=0.50)

    ax.set_xticks(xs)
    ax.set_xticklabels([CLIMATE_NAME[z] for z in CLIMATE_ORDER], rotation=0)
    ax.set_ylabel(format_ylabel(ylabel, is_delta=is_delta))
    style_small_axis(ax)
    ax.set_title(title, fontsize=6.5, fontweight="bold", pad=2)

    if show_legend:
        handles = [Line2D([0], [0], color=CLR_BOX[p], lw=2.2,
                          label=PERIOD_LABEL.get(p, p)) for p in BOX_PERIODS]
        ax.legend(handles=handles, frameon=False,
                  loc="upper right", fontsize=5.5)

def grouped_boxplot_uhi_uci_by_climate(ax, df, value_col, ylabel, title,
                                       show_legend=False, zero_line=True):
    """
    Supplementary 用：
    按 climate breakdown 画差值箱线图
    每个 climate 下 4 个箱线：
      UHI-HW, UHI-NHW, UCI-HW, UCI-NHW
    """
    width = 0.14
    offsets = {
        ("UHI", "HW"):  -1.5 * width,
        ("UHI", "NHW"): -0.5 * width,
        ("UCI", "HW"):  +0.5 * width,
        ("UCI", "NHW"): +1.5 * width,
    }

    colors = {
        ("UHI", "HW"):  "#b2182b",
        ("UHI", "NHW"): "#ef8a62",
        ("UCI", "HW"):  "#2166ac",
        ("UCI", "NHW"): "#67a9cf",
    }

    xs = np.arange(len(CLIMATE_ORDER), dtype=float)
    rng = np.random.default_rng(42)

    for i, z in enumerate(CLIMATE_ORDER):
        sub_z = df[df["climate4"] == z].copy()

        for key, off in offsets.items():
            grp, per = key
            sub = sub_z[
                (sub_z["group"] == grp) &
                (sub_z["period_norm"] == per)
            ][value_col].dropna()

            if len(sub) < 3:
                continue

            pos = xs[i] + off
            col = colors[key]

            ax.boxplot(
                [sub.values],
                positions=[pos],
                widths=0.11,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=col, edgecolor="black", linewidth=0.45, alpha=0.92),
                medianprops=dict(color="white", linewidth=0.8),
                whiskerprops=dict(color="black", linewidth=0.45),
                capprops=dict(color="black", linewidth=0.45),
                manage_ticks=False
            )

            jitter_x = pos + rng.normal(0, 0.01, size=len(sub))
            ax.scatter(jitter_x, sub.values, s=2.0, color="black", alpha=0.08, linewidths=0, zorder=2)
            ax.scatter([pos], [sub.mean()],
                       s=7, color="white", edgecolor="black", linewidth=0.35, zorder=5)

    if zero_line:
        ax.axhline(0, color="black", linestyle=":", linewidth=0.55)

    ax.set_xticks(xs)
    ax.set_xticklabels([CLIMATE_NAME[z] for z in CLIMATE_ORDER], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=6.5, fontweight="bold", pad=1.5)
    style_small_axis(ax)

    if show_legend:
        handles = [
            Line2D([0], [0], color=colors[("UHI", "HW")],  lw=2.2, label="UHI-HW"),
            Line2D([0], [0], color=colors[("UHI", "NHW")], lw=2.2, label="UHI-NHW"),
            Line2D([0], [0], color=colors[("UCI", "HW")],  lw=2.2, label="UCI-HW"),
            Line2D([0], [0], color=colors[("UCI", "NHW")], lw=2.2, label="UCI-NHW"),
        ]
        ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=5.3)

def grouped_boxplot_hw_nhw_overlap(
    ax,
    df,
    value_col,
    ylabel,
    title,
    show_legend=False,
    zero_line=False,
    hw_first=True,
    overlap=0.055,
    box_width=0.18,
    jitter_sd=0.010,
    alpha_hw=0.78,
    alpha_nhw=0.72,
    ylim=None
):
    """
    按 climate group 画 NHW / HW 两个轻微交叠的箱线图。
    用于 LC 和 Sleep 两个 pair-level 指标。

    Parameters
    ----------
    overlap : float
        两个箱线图中心的偏移量。越小重叠越明显。
    box_width : float
        每个箱体宽度。建议略大于 overlap，这样能形成轻微交叠。
    hw_first : bool
        True  -> 先画 HW 再画 NHW（NHW 压在上面）
        False -> 先画 NHW 再画 HW（HW 压在上面）
    """
    xs = np.arange(len(CLIMATE_ORDER), dtype=float)
    rng = np.random.default_rng(42)

    # 让 NHW / HW 稍微交叠
    pos_map = {
        "NHW": xs + overlap / 2.0,
        "HW":  xs - overlap / 2.0,
    }

    draw_order = ["HW", "NHW"] if hw_first else ["NHW", "HW"]

    color_map = {
        "HW": CLR_BOX["HW"],
        "NHW": CLR_BOX["NHW"],
    }
    alpha_map = {
        "HW": alpha_hw,
        "NHW": alpha_nhw,
    }
    z_map = {
        "HW": 3,
        "NHW": 4,
    } if hw_first else {
        "NHW": 3,
        "HW": 4,
    }

    for p in draw_order:
        color = color_map[p]
        alpha = alpha_map[p]

        for i, z in enumerate(CLIMATE_ORDER):
            sub = df[
                (df["climate4"] == z) &
                (df["period_norm"] == p)
            ][value_col].dropna()

            if len(sub) < 3:
                continue

            pos = pos_map[p][i]

            ax.boxplot(
                [sub.values],
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops=dict(
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.50,
                    alpha=alpha
                ),
                medianprops=dict(color="white", linewidth=0.90),
                whiskerprops=dict(color="black", linewidth=0.50),
                capprops=dict(color="black", linewidth=0.50),
                zorder=z_map[p]
            )

            jitter_x = pos + rng.normal(0, jitter_sd, size=len(sub))
            ax.scatter(
                jitter_x, sub.values,
                s=4.0, color=color, alpha=0.18, linewidths=0,
                zorder=z_map[p] - 0.2
            )

            ax.scatter(
                [pos], [sub.mean()],
                s=16, color="white", edgecolor=color, linewidth=0.8,
                zorder=z_map[p] + 1
            )

    if zero_line:
        ax.axhline(0, color="black", linestyle=":", linewidth=0.55)

    ax.set_xticks(xs)
    ax.set_xticklabels([CLIMATE_NAME[z] for z in CLIMATE_ORDER], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=6.8, fontweight="bold", y=1.01, pad=0.5)
    style_small_axis(ax)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_legend:
        handles = [
            Line2D([0], [0], color=CLR_BOX["NHW"], lw=3.0, alpha=alpha_nhw, label="NHW"),
            Line2D([0], [0], color=CLR_BOX["HW"],  lw=3.0, alpha=alpha_hw,  label="HW"),
        ]
        ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=5.6)

def grouped_boxplot_period_uhi_uci(
    ax,
    df,
    value_col,
    ylabel,
    title,
    show_legend=False,
    zero_line=False,
    ylim=None
):
    """
    横轴:
      NHW, HW, ΔHW−NHW

    每个时期下两个箱体:
      UHI / UCI

    ΔHW−NHW:
      pair-level paired difference, HW minus NHW
    """
    periods = ["NHW", "HW", "ΔHW−NHW"]
    groups = ["UHI", "UCI"]

    x_centers = np.arange(len(periods), dtype=float)
    width = 0.24
    offsets = {"UHI": -0.14, "UCI": 0.14}
    colors = {"UHI": "#b2182b", "UCI": "#2166ac"}

    rng = np.random.default_rng(42)

    # paired HW − NHW delta
    delta_df = pd.DataFrame()
    if all(c in df.columns for c in ["pair_id", "group", "period_norm", value_col]):
        tmp = df[
            df["period_norm"].isin(["HW", "NHW"]) &
            df["group"].isin(groups)
        ][["pair_id", "group", "period_norm", value_col]].dropna().copy()

        tmp = (
            tmp.groupby(["pair_id", "group", "period_norm"], observed=True)[value_col]
               .mean()
               .reset_index()
        )

        piv = tmp.pivot_table(
            index=["pair_id", "group"],
            columns="period_norm",
            values=value_col,
            aggfunc="mean"
        ).reset_index()

        if "HW" in piv.columns and "NHW" in piv.columns:
            piv["delta_hw_nhw"] = piv["HW"] - piv["NHW"]
            delta_df = piv[["pair_id", "group", "delta_hw_nhw"]].dropna().copy()

    for i, p in enumerate(periods):
        for g in groups:

            if p == "ΔHW−NHW":
                if len(delta_df) == 0:
                    continue
                sub = delta_df[
                    delta_df["group"] == g
                ]["delta_hw_nhw"].dropna()
            else:
                sub = df[
                    (df["period_norm"] == p) &
                    (df["group"] == g)
                ][value_col].dropna()

            if len(sub) < 3:
                continue

            pos = x_centers[i] + offsets[g]

            ax.boxplot(
                [sub.values],
                positions=[pos],
                widths=width * 0.85,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops=dict(
                    facecolor=colors[g],
                    edgecolor="black",
                    linewidth=0.50,
                    alpha=0.85
                ),
                medianprops=dict(color="white", linewidth=0.90),
                whiskerprops=dict(color="black", linewidth=0.50),
                capprops=dict(color="black", linewidth=0.50),
            )

            jitter_x = pos + rng.normal(0, 0.018, size=len(sub))
            ax.scatter(
                jitter_x, sub.values,
                s=4.0, color="black", alpha=0.10,
                linewidths=0, zorder=2
            )

            ax.scatter(
                [pos], [sub.mean()],
                s=16, color="white", edgecolor="black",
                linewidth=0.6, zorder=5
            )

    if zero_line:
        ax.axhline(0, color="black", linestyle=":", linewidth=0.55)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(periods)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=6.8, fontweight="bold", pad=2.0)
    style_small_axis(ax)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_legend:
        handles = [
            Line2D([0], [0], color=colors["UHI"], lw=3.0, label="UHI"),
            Line2D([0], [0], color=colors["UCI"], lw=3.0, label="UCI"),
        ]
        ax.legend(handles=handles, frameon=False,
                  loc="upper right", fontsize=5.6)

# ============================================================================
# [NEW MODULE] Latitudinal Profile Analysis (Map + Side Panels)
# ============================================================================

def get_latitudinal_summary(df_map, value_col, bin_size=5):
    """将逐点数据按纬度带聚合，计算均值和标准误"""
    df = df_map.dropna(subset=['lat_ref', value_col]).copy()
    bins = np.arange(-60, 90, bin_size)
    df['lat_bin'] = pd.cut(df['lat_ref'], bins=bins)
    
    summary = df.groupby('lat_bin')[value_col].agg(['mean', 'sem', 'count']).reset_index()
    summary['lat_mid'] = summary['lat_bin'].apply(lambda x: x.mid)
    return summary.dropna(subset=['mean'])

def draw_side_profile(ax, df_summary, color, label, xlim=None):
    """绘制竖直方向的纬度剖面图 (Y轴为纬度)"""
    y = df_summary['lat_mid'].values
    x = df_summary['mean'].values
    err = df_summary['sem'].fillna(0).values
    
    # 绘制置信区间 (SEM)
    ax.fill_betweenx(y, x - err, x + err, color=color, alpha=0.2, lw=0)
    # 绘制主趋势线
    ax.plot(x, y, color=color, lw=1.5, label=label)
    
    # 辅助线
    ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylim(-60, 85)
    if xlim: ax.set_xlim(xlim)
    
    ax.tick_params(axis='both', labelsize=6)
    ax.grid(True, ls=':', lw=0.5, alpha=0.3)
    # 移除上方和右方边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def make_latitudinal_analysis_figure(df_hw_diurnal, out_dir, diag):
    """生成：左侧地图 + 右侧纬度剖面对齐图"""
    apply_nature_style()
    map_metric = build_hw_nhw_delta_uhi_daynight_from_hw_diurnal(df_hw_diurnal, diag)
    
    # 准备剖面数据
    prof_day = get_latitudinal_summary(map_metric, 'delta_uhi_daytime')
    prof_night = get_latitudinal_summary(map_metric, 'delta_uhi_nighttime')
    
    fig = plt.figure(figsize=(8.5, 4.5), dpi=600)
    # 分配比例：地图占 70%，两个剖面图各占 15%
    gs = GridSpec(1, 3, width_ratios=[4.5, 1, 1], wspace=0.12, 
                  left=0.08, right=0.95, top=0.88, bottom=0.15)

    # 1. 左侧地图 (Panel a)
    if HAS_CARTOPY:
        ax_map = fig.add_subplot(gs[0], projection=PLOT_PROJECTION)
        ax_map.set_extent([-180, 180, -60, 85], crs=DATA_CRS)
        ax_map.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="#707070")
        ax_map.add_feature(cfeature.BORDERS, lw=0.2, edgecolor="#cccccc", ls=':')
    else:
        ax_map = fig.add_subplot(gs[0])
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-60, 85)

    sc = ax_map.scatter(
        map_metric["lon_ref"], map_metric["lat_ref"],
        c=map_metric["delta_uhi_nighttime"], # 以夜间增幅为主色调
        cmap=MAP_CMAP, vmin=-1.5, vmax=3.0, s=12, 
        edgecolors='black', linewidths=0.2, alpha=0.85
    )
    ax_map.set_title("Heatwave Amplification of UHI", fontsize=8, fontweight='bold', loc='left')
    add_panel_label(ax_map, "a", x=-0.02, y=1.02)

    # 2. 中间剖面 (Panel b: RTx)
    ax_prof1 = fig.add_subplot(gs[1])
    draw_side_profile(ax_prof1, prof_day, "#d73027", r"$R\Delta T_{\mathrm{x}}$", xlim=(-0.5, 2.5))
    ax_prof1.set_xlabel("Day (°C)", fontsize=7)
    ax_prof1.set_yticklabels([]) # 隐藏中间的刻度文字
    ax_prof1.set_title("Daytime", fontsize=7)
    add_panel_label(ax_prof1, "b", x=-0.05, y=1.02)

    # 3. 右侧剖面 (Panel c: RTn)
    ax_prof2 = fig.add_subplot(gs[2])
    draw_side_profile(ax_prof2, prof_night, "#4575b4", r"$R\Delta T_{\mathrm{n}}$", xlim=(-0.5, 2.5))
    ax_prof2.set_xlabel("Night (°C)", fontsize=7)
    ax_prof2.yaxis.tick_right() # 刻度放右边
    ax_prof2.set_ylabel("Latitude", fontsize=7)
    ax_prof2.yaxis.set_label_position("right")
    ax_prof2.set_title("Nighttime", fontsize=7)
    add_panel_label(ax_prof2, "c", x=-0.05, y=1.02)

    # 颜色条
    cax = fig.add_axes([0.12, 0.22, 0.15, 0.02])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.set_label(r"$\Delta$UHI (°C)", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    # 保存
    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, "figure_latitudinal_analysis.png"), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, "figure_latitudinal_analysis.pdf"), bbox_inches='tight')
    plt.close(fig)
    diag.add("Latitudinal analysis figure saved", "figure_latitudinal_analysis.png")


# ============================================================================
# §11 Main figure  ── RESTRUCTURED to 5 rows ──
# ============================================================================

def make_integrated_figure(df_hw_diurnal, df_diurnal_summary,
                           df_station_clim, df_pair_daynight, df_fig9, out_dir, diag):
    """
    Main figure:
      a : map + latitudinal profiles
      b : UHI HW/NHW absolute diurnal
      c : UCI HW/NHW absolute diurnal
      d : UHI HW/NHW delta Ta
      e : UCI HW/NHW delta Ta
      f : LC boxplot by climate group (NHW vs HW, slight overlap)
      g : Sleep-loss boxplot by climate group (NHW vs HW, slight overlap)
    """
    apply_nature_style()

    fig9_sum = build_fig9_summary(df_fig9, diag)

    # pair-level LC / sleep 数据，并 merge UHI/UCI group
    df_pair_box = df_pair_daynight[
        df_pair_daynight["period_norm"].isin(BOX_PERIODS) &
        df_pair_daynight["climate4"].isin(CLIMATE_ORDER)
    ].copy()

    df_group = (
        df_fig9[["pair_id", "group"]]
        .dropna(subset=["pair_id", "group"])
        .drop_duplicates()
        .copy()
    )

    df_pair_box = df_pair_box.merge(df_group, on="pair_id", how="left")
    df_pair_box = df_pair_box[df_pair_box["group"].isin(["UHI", "UCI"])].copy()

    fig = plt.figure(figsize=(7.2, 9.2), dpi=600)
    gs = GridSpec(
        4, 2, figure=fig,
        height_ratios=[2.25, 1.15, 1.15, 1.08],
        hspace=0.32, wspace=0.24,
        left=0.07, right=0.985, top=0.968, bottom=0.06
    )

    draw_map_with_latitudinal_profiles(
        fig, gs[0, :], df_hw_diurnal, df_fig9, diag, panel_letter="a"
    )

    # b
    ax_b = fig.add_subplot(gs[1, 0])
    draw_fig9_abs_panel(
        ax_b,
        fig9_sum[("UHI", "HW")],
        fig9_sum[("UHI", "NHW")],
        "UHI",
        show_legend=True
    )
    ax_b.set_title("Night-warming regime", fontsize=6.5, fontweight="bold")
    add_panel_label(ax_b, "b", x=-0.10, y=1.02)

    # c
    ax_c = fig.add_subplot(gs[1, 1])
    draw_fig9_abs_panel(
        ax_c,
        fig9_sum[("UCI", "HW")],
        fig9_sum[("UCI", "NHW")],
        "UCI",
        show_legend=False
    )
    ax_c.set_title("Day-cooling / night-warming regime", fontsize=6.5, fontweight="bold")
    add_panel_label(ax_c, "c", x=-0.10, y=1.02)

    # d
    ax_d = fig.add_subplot(gs[2, 0])
    draw_fig9_delta_panel(
        ax_d,
        fig9_sum[("UHI", "HW")],
        fig9_sum[("UHI", "NHW")],
        "UHI",
        show_legend=True
    )
    ax_d.set_title("Persistent urban warming", fontsize=6.5, fontweight="bold")
    ax_d.set_ylim(-1, 2)
    ax_d.set_yticks([-1, 0, 1, 2])
    add_panel_label(ax_d, "d", x=-0.10, y=1.02)

    # e
    ax_e = fig.add_subplot(gs[2, 1])
    draw_fig9_delta_panel(
        ax_e,
        fig9_sum[("UCI", "HW")],
        fig9_sum[("UCI", "NHW")],
        "UCI",
        show_legend=False
    )
    ax_e.set_title("Daytime cooling but nighttime warming", fontsize=6.5, fontweight="bold")
    ax_e.set_ylim(-1, 2)
    ax_e.set_yticks([-1, 0, 1, 2])
    add_panel_label(ax_e, "e", x=-0.10, y=1.02)

    # ===== [ADD] day/night shading =====
    for ax in [ax_d, ax_e]:
        ax.axvspan(6, 18, color="orange", alpha=0.06, zorder=0)
        ax.axvspan(18, 24, color="blue", alpha=0.05, zorder=0)
        ax.axvspan(0, 6, color="blue", alpha=0.05, zorder=0)

    # f : Labour-loss difference，横轴 HW/NHW，箱体 UHI/UCI
    ax_f = fig.add_subplot(gs[3, 0])
    _lc_col = "d_labour_loss_tx"
    if _lc_col not in df_pair_box.columns or df_pair_box[_lc_col].isna().all():
        diag.add("Plot LC (main)", f"Column {_lc_col} not found or all-NaN; skipping.")
        ax_f.set_visible(False)
    else:
        grouped_boxplot_period_uhi_uci(
            ax_f,
            df_pair_box,
            _lc_col,
            r"$\Delta$Labour loss at Tx (%)",
            "Labour-loss difference",
            show_legend=True,
            zero_line=True
        )
        add_panel_label(ax_f, "f", x=-0.08, y=1.02)

    # g : Sleep loss，横轴 HW/NHW，箱体 UHI/UCI
    ax_g = fig.add_subplot(gs[3, 1])
    _sleep_col = "delta_sleep_min"

    if _sleep_col not in df_pair_box.columns or df_pair_box[_sleep_col].isna().all():
        diag.add(
            "Plot sleep loss (main)",
            f"Column {_sleep_col} not found or all-NaN; skipping."
        )
        ax_g.set_visible(False)

    else:
        grouped_boxplot_period_uhi_uci(
            ax_g,
            df_pair_box,
            _sleep_col,
            r"$\Delta$ Sleep loss (min/night)",
            "Sleep loss",
            show_legend=False,
            zero_line=True
        )
        add_panel_label(ax_g, "g", x=-0.08, y=1.02)


    ensure_dir(out_dir)
    png_path = os.path.join(out_dir, "integrated_climatezone_figure_main_v11.png")
    pdf_path = os.path.join(out_dir, "integrated_climatezone_figure_main_v11.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    diag.add("Main figure saved", f"{png_path}\n{pdf_path}")
    print(f"  ✓ saved: {png_path}")
    print(f"  ✓ saved: {pdf_path}")

def make_supplementary_climate_breakdown_figure(df_station_clim, df_pair_daynight,
                                                df_fig9, out_dir, diag):
    """
    Supplementary:
      a : Labour-loss difference by climate × (UHI/UCI × HW/NHW)
      b : Sleep loss difference by climate × (UHI/UCI × HW/NHW)
    只画差值，不画城乡绝对值。
    """
    apply_nature_style()

    # pair-level Labour-loss difference
    df_pair_box = df_pair_daynight[
        df_pair_daynight["period_norm"].isin(BOX_PERIODS) &
        df_pair_daynight["climate4"].isin(CLIMATE_ORDER)
    ].copy()

    # pair-level sleep difference
    df_pair_sleep_delta = build_pair_period_sleep_delta(df_station_clim, diag)
    df_pair_sleep_delta = df_pair_sleep_delta[
        df_pair_sleep_delta["period_norm"].isin(BOX_PERIODS) &
        df_pair_sleep_delta["climate4"].isin(CLIMATE_ORDER)
    ].copy()

    # UHI / UCI group
    df_group = (
        df_fig9[["pair_id", "group"]]
        .dropna(subset=["pair_id", "group"])
        .drop_duplicates()
        .copy()
    )

    df_pair_box = df_pair_box.merge(df_group, on="pair_id", how="left")
    df_pair_sleep_delta = df_pair_sleep_delta.merge(df_group, on="pair_id", how="left")

    fig = plt.figure(figsize=(7.2, 4.1), dpi=600)
    gs = GridSpec(
        1, 2, figure=fig,
        wspace=0.28,
        left=0.08, right=0.985, top=0.90, bottom=0.15
    )

    # a
    ax_a = fig.add_subplot(gs[0, 0])
    if "d_labour_loss_tx" not in df_pair_box.columns or df_pair_box["d_labour_loss_tx"].isna().all():
        diag.add("Plot supp LC", "Column d_labour_loss_tx not found or all-NaN; skipping.")
        ax_a.set_visible(False)
    else:
        grouped_boxplot_uhi_uci_by_climate(
            ax_a,
            df_pair_box,
            "d_labour_loss_tx",
            r"$\Delta$ LC (%)",
            "Labour loss at Tx",
            show_legend=True,
            zero_line=True
        )
        add_panel_label(ax_a, "a", x=-0.10, y=1.03)

    # b
    ax_b = fig.add_subplot(gs[0, 1])
    if "delta_sleep_min" not in df_pair_sleep_delta.columns or df_pair_sleep_delta["delta_sleep_min"].isna().all():
        diag.add("Plot supp sleep", "Column delta_sleep_min not found or all-NaN; skipping.")
        ax_b.set_visible(False)
    else:
        grouped_boxplot_uhi_uci_by_climate(
            ax_b,
            df_pair_sleep_delta,
            "delta_sleep_min",
            r"$\Delta$ Sleep loss (min per night)",
            "Sleep loss",
            show_legend=False,
            zero_line=True
        )
        add_panel_label(ax_b, "b", x=-0.10, y=1.03)

    ensure_dir(out_dir)
    png_path = os.path.join(out_dir, "supplementary_climate_breakdown_v10.png")
    pdf_path = os.path.join(out_dir, "supplementary_climate_breakdown_v10.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    diag.add("Supplementary figure saved", f"{png_path}\n{pdf_path}")
    print(f"  ✓ saved: {png_path}")
    print(f"  ✓ saved: {pdf_path}")

# ============================================================================
# §11b  [total version] Build n_days table & compute annual-total metrics
# ============================================================================
def build_n_days_table(df_hw_diurnal, diag):
    """
    从上游 01 all_pair_period_metrics.csv 提取每站点每时期的
    年均有效天数（n_days / n_valid_years）。

    Returns
    -------
    DataFrame with columns:
        pair_id, station_side, period_norm, n_days_per_year
    """
    need = ["pair_id", "station_type", "period_norm", "n_days", "n_valid_years"]
    need = [c for c in need if c in df_hw_diurnal.columns]
    df = df_hw_diurnal[need].copy()
    df = df.rename(columns={"station_type": "station_side"})
    df["n_valid_years"] = pd.to_numeric(df["n_valid_years"], errors="coerce").clip(lower=1)
    df["n_days_per_year"] = (
        pd.to_numeric(df["n_days"], errors="coerce") / df["n_valid_years"]
    )
    # No fixed 92-day fallback is permitted in a hemispherically aware
    # workflow. Warm-season day counts must come from the upstream main
    # analysis (NH JJA / SH DJF); missing counts remain NaN.

    df = df.drop_duplicates(["pair_id", "station_side", "period_norm"])

    diag.add(
        "Build n_days table (total version)",
        f"shape={df.shape}\n"
        "mean n_days_per_year by period:\n"
        + df.groupby("period_norm")["n_days_per_year"].mean().round(1).to_string()
    )
    return df[["pair_id", "station_side", "period_norm", "n_days_per_year"]]


def compute_total_metrics(df_station_clim, df_n_days, diag):
    """
    将 df_station_clim 中的逐日均值乘以 n_days_per_year，
    生成新列（后缀 _annual），供总量版图使用。
    原有列不变，返回新 DataFrame。
    """
    df = df_station_clim.merge(
        df_n_days,
        on=["pair_id", "station_side", "period_norm"],
        how="left"
    )
    matched = df["n_days_per_year"].notna().sum()
    diag.add(
        "Compute total metrics",
        f"rows matched with n_days: {matched}/{len(df)}  ({matched/len(df):.1%})"
    )

    per_day_to_annual = {
        "CDH_total_mean":           "CDH_total_annual",
        "HDH_total_mean":           "HDH_total_annual",
        "sleep_loss_min_mean":      "sleep_loss_min_annual",
        "total_loss_usd_mean":      "total_loss_usd_annual",
        "econ_loss_usd_sleep_mean": "econ_loss_usd_sleep_annual",
        "econ_loss_usd_heat_mean":  "econ_loss_usd_heat_annual",
        "d_labour_loss_tx":   "d_labour_loss_tx_annual",
    }
    for src, dst in per_day_to_annual.items():
        if src in df.columns:
            df[dst] = df[src] * df["n_days_per_year"]

    return df


# ============================================================================
# §11c  [total version] Annual-total figure
# ============================================================================
def make_integrated_figure_total(df_hw_diurnal, df_diurnal_summary,
                                  df_station_total, out_dir, diag):
    """
    与 make_integrated_figure 完全相同的版式，但 f-k 各面板的纵坐标
    改为乘以对应时期天数后的年均总量：
      g  – d_labour_loss_tx_annual   (%·day / yr)
      h  – CDH_total_annual           (°C·h / yr)
      i  – HDH_total_annual           (°C·h / yr)
      j  – sleep_loss_min_annual      (min / yr)
      k  – total_loss_usd_annual 等   (USD / person / yr)
    Mortality RR (f) 无需累积，保持原值。
    """
    apply_nature_style()

    df_station_box = df_station_total[
        df_station_total["period_norm"].isin(BOX_PERIODS) &
        df_station_total["climate4"].isin(CLIMATE_ORDER)
    ].copy()

    map_metric = build_hw_nhw_tmax_map_metric_from_hw_diurnal(df_hw_diurnal, diag)
    boxes = compute_fixed_boxes(df_hw_diurnal)

    color_vals = map_metric["delta_tx_hw_nhw"].dropna()
    vmin = np.nanpercentile(color_vals, 2)  if len(color_vals) else -2
    vmax = np.nanpercentile(color_vals, 98) if len(color_vals) else  4

    fig = plt.figure(figsize=(7.2, 9.8), dpi=600)
    gs = GridSpec(
        5, 4, figure=fig,
        height_ratios=[3.20, 1.10, 1.28, 1.28, 1.28],
        width_ratios=[1, 1, 1, 1],
        hspace=0.52, wspace=0.34,
        left=0.07, right=0.98, top=0.97, bottom=0.03
    )

    # ── Row 0: panel a – global map (identical to original) ──────────────────
    if HAS_CARTOPY:
        ax_map = fig.add_subplot(gs[0, :], projection=PLOT_PROJECTION)
        ax_map.patch.set_linewidth(0.8)   # 确保地图边框粗细一致
        for spine in ax_map.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")
        ax_map.set_extent([-180, 180, -60, 85], crs=DATA_CRS)
        ax_map.set_facecolor("white")
        ax_map.add_feature(cfeature.COASTLINE,
                           linewidth=0.30, edgecolor="#606060", zorder=1)
        ax_map.add_feature(cfeature.BORDERS,
                           linewidth=0.15, edgecolor="#aaaaaa", linestyle=":", zorder=1)
        gl = ax_map.gridlines(
            crs=DATA_CRS, draw_labels=True,
            linewidth=0.20, color="#cccccc", alpha=0.5, linestyle="-"
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 5.5}
        gl.ylabel_style = {"size": 5.5}
        sc = ax_map.scatter(
            map_metric["lon_urban"].values,
            map_metric["lat_urban"].values,
            transform=DATA_CRS,
            c=map_metric["delta_tx_hw_nhw"].values,
            cmap=MAP_CMAP, vmin=vmin, vmax=vmax,
            s=5, linewidths=0, alpha=0.88, zorder=3
        )
        for z, box in boxes.items():
            rect = Rectangle(
                (box["lon0"], box["lat0"]),
                box["lon1"] - box["lon0"],
                box["lat1"] - box["lat0"],
                transform=DATA_CRS,
                fill=False, linewidth=0.85, edgecolor=ZONE_EDGE[z], zorder=4
            )
            ax_map.add_patch(rect)
            ax_map.text(
                (box["lon0"] + box["lon1"]) / 2, box["lat0"] - 1.8,
                CLIMATE_NAME.get(z, z),
                transform=DATA_CRS,
                fontsize=5.5, fontweight="bold",
                color=ZONE_EDGE[z], ha="center", va="top"
            )
    else:
        ax_map = fig.add_subplot(gs[0, :])
        ax_map.patch.set_linewidth(0.8)   # 确保地图边框粗细一致
        for spine in ax_map.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")
        sc = ax_map.scatter(
            map_metric["lon_urban"].values,
            map_metric["lat_urban"].values,
            c=map_metric["delta_tx_hw_nhw"].values,
            cmap=MAP_CMAP, vmin=vmin, vmax=vmax,
            s=5, linewidths=0, alpha=0.88
        )
        ax_map.set_facecolor("white")
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-60, 85)
        ax_map.set_xlabel("Longitude", fontsize=5.5)
        ax_map.set_ylabel("Latitude", fontsize=5.5)
        ax_map.grid(True, linestyle="--", linewidth=0.25, alpha=0.2)
        for z, box in boxes.items():
            rect = Rectangle(
                (box["lon0"], box["lat0"]),
                box["lon1"] - box["lon0"],
                box["lat1"] - box["lat0"],
                fill=False, linewidth=0.85, edgecolor=ZONE_EDGE[z]
            )
            ax_map.add_patch(rect)
            ax_map.text(
                (box["lon0"] + box["lon1"]) / 2, box["lat0"] - 1.8,
                CLIMATE_NAME.get(z, z),
                fontsize=5.5, fontweight="bold", color=ZONE_EDGE[z],
                ha="center", va="top"
            )

    try:
        cax = mpl_inset_axes(
            ax_map,
            width="22%", height="4%",
            loc="lower left",
            bbox_to_anchor=(0.03, 0.05, 1, 1),
            bbox_transform=ax_map.transAxes,
            borderpad=0
        )
        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.set_label(r"$\Delta$Tmax, HW$-$NHW (°C)", fontsize=5.8, labelpad=2)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(labelsize=5.2, length=2, pad=1,
                            which="both", bottom=True, top=False)
        cax.set_facecolor((1, 1, 1, 0.55))
    except Exception:
        cbar = fig.colorbar(sc, ax=ax_map, orientation="vertical",
                            pad=0.008, shrink=0.50)
        cbar.set_label(r"$\Delta$Tmax HW$-$NHW (°C)", fontsize=6)
        cbar.ax.tick_params(labelsize=5.8)

    add_panel_label(ax_map, "a", x=-0.03, y=1.02)

    # ── Row 1: panels b-e – diurnal profiles (identical to original) ─────────
    inset_axes_list = [fig.add_subplot(gs[1, i]) for i in range(4)]
    for ax, z, letter in zip(inset_axes_list, CLIMATE_ORDER, ["b", "c", "d", "e"]):
        draw_diurnal_panel(ax, df_diurnal_summary, z)
        if z == "A":
            ax.set_ylabel("Air temperature (°C)")
        else:
            ax.set_ylabel("")
        add_panel_label(ax, letter, x=-0.14, y=1.06)
        if z == "D":
            handles = [Line2D([0], [0], color=CLR_PERIOD[p], lw=1.5, label=p)
                       for p in TOP_PERIODS]
            ax.legend(handles=handles, frameon=False, loc="upper right")

    # ── Row 2: panels f, g ────────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 0:2])
    ax_g = fig.add_subplot(gs[2, 2:4])

    # f – Mortality RR: dimensionless ratio, no accumulation needed
    grouped_boxplot_compact(
        ax_f, df_station_box, "rr_mean",
        "Heat mortality RR", "Mortality",
        show_legend=False, zero_line=False, is_delta=False
    )
    add_panel_label(ax_f, "f", x=-0.08, y=1.04)

    # g – Labour loss per year (%·day / yr)
    grouped_boxplot_compact(
        ax_g, df_station_box, "d_labour_loss_tx_annual",
        "Labour loss (%·day/yr)", "Labour loss per year",
        show_legend=False, zero_line=False, is_delta=False
    )
    add_panel_label(ax_g, "g", x=-0.08, y=1.04)

    # ── Row 3: panels h, i – CDH / HDH annual total ───────────────────────────
    ax_h = fig.add_subplot(gs[3, 0:2])
    ax_i = fig.add_subplot(gs[3, 2:4])

    grouped_boxplot_compact(
        ax_h, df_station_box, "CDH_total_annual",
        "CDH (°C·h/yr)", "CDH",
        show_legend=False, zero_line=False, is_delta=False
    )
    add_panel_label(ax_h, "h", x=-0.08, y=1.04)

    grouped_boxplot_compact(
        ax_i, df_station_box, "HDH_total_annual",
        "HDH (°C·h/yr)", "HDH",
        show_legend=False, zero_line=False, is_delta=False
    )
    add_panel_label(ax_i, "i", x=-0.08, y=1.04)

    # ── Row 4: panels j, k – Sleep loss / Economic loss annual total ──────────
    ax_j = fig.add_subplot(gs[4, 0:2])
    ax_k = fig.add_subplot(gs[4, 2:4])

    # j – Sleep loss annual
    _sleep_col_ann = "sleep_loss_min_annual"
    if _sleep_col_ann not in df_station_box.columns or \
            df_station_box[_sleep_col_ann].isna().all():
        diag.add("Plot sleep loss (total)", f"Column {_sleep_col_ann} empty; skipping.")
        ax_j.set_visible(False)
    else:
        grouped_boxplot_compact(
            ax_j, df_station_box, _sleep_col_ann,
            "Sleep loss (min/yr)", "Sleep loss",
            show_legend=False, zero_line=False, is_delta=False
        )
    add_panel_label(ax_j, "j", x=-0.08, y=1.04)

    # k – Economic loss annual (prefer USD total; fallback chain)
    _econ_col_ann = None
    for candidate in ["total_loss_usd_annual",
                      "econ_loss_usd_sleep_annual",
                      "econ_loss_usd_heat_annual"]:
        if (candidate in df_station_box.columns
                and not df_station_box[candidate].isna().all()):
            _econ_col_ann = candidate
            break

    if _econ_col_ann is None:
        diag.add("Plot econ loss (total)", "No annual USD column found; skipping.")
        ax_k.set_visible(False)
    else:
        grouped_boxplot_compact(
            ax_k, df_station_box, _econ_col_ann,
            "USD per person per year", "Economic loss",
            show_legend=True, zero_line=False, is_delta=False
        )
        diag.add("Plot econ loss (total)", f"Using column: {_econ_col_ann}")
    add_panel_label(ax_k, "k", x=-0.08, y=1.04)

    # ── Save ─────────────────────────────────────────────────────────────────
    ensure_dir(out_dir)
    png_path = os.path.join(out_dir, "integrated_climatezone_figure_v4_total.png")
    pdf_path = os.path.join(out_dir, "integrated_climatezone_figure_v4_total.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    diag.add("Total figure saved", f"{png_path}\n{pdf_path}")
    print(f"  ✓ saved: {png_path}")
    print(f"  ✓ saved: {pdf_path}")

def build_pair_hw_nhw_temperature_delta(df_hw_diurnal, diag):
    """
    Pair-level temperature forcing:
      daytime  = mean station Tmax_fft(HW) - mean station Tmax_fft(NHW)
      nighttime = mean station Tmin_fft(HW) - mean station Tmin_fft(NHW)

    Urban and rural stations are averaged within each pair-period.
    """
    tmp = df_hw_diurnal[[
        "pair_id", "period_norm", "Tmax_fft", "Tmin_fft"
    ]].copy()

    tmp = tmp[tmp["period_norm"].isin(["HW", "NHW"])]
    tmp = tmp.dropna(subset=["pair_id", "period_norm"])

    pair_period = (
        tmp.groupby(["pair_id", "period_norm"], observed=True)
        [["Tmax_fft", "Tmin_fft"]]
        .mean()
        .reset_index()
    )

    piv = pair_period.pivot_table(
        index="pair_id",
        columns="period_norm",
        values=["Tmax_fft", "Tmin_fft"],
        aggfunc="mean"
    )

    out = pd.DataFrame(index=piv.index)

    if ("Tmax_fft", "HW") in piv.columns and ("Tmax_fft", "NHW") in piv.columns:
        out["delta_tmax_hw_nhw"] = piv[("Tmax_fft", "HW")] - piv[("Tmax_fft", "NHW")]

    if ("Tmin_fft", "HW") in piv.columns and ("Tmin_fft", "NHW") in piv.columns:
        out["delta_tmin_hw_nhw"] = piv[("Tmin_fft", "HW")] - piv[("Tmin_fft", "NHW")]

    out = out.reset_index()

    diag.add(
        "Build pair HW-NHW temperature delta",
        f"shape={out.shape}\n"
        f"cols={list(out.columns)}"
    )

    return out


def build_pair_hw_nhw_impact_delta(df_pair_daynight, diag):
    """
    Pair-level impact response:
      labour response = d_labour_loss_tx(HW) - d_labour_loss_tx(NHW)
      sleep response  = delta_sleep_min(HW) - delta_sleep_min(NHW)
    """
    tmp = df_pair_daynight[
        df_pair_daynight["period_norm"].isin(["HW", "NHW"])
    ].copy()

    tmp = tmp.dropna(subset=["pair_id", "period_norm"])

    piv = tmp.pivot_table(
        index="pair_id",
        columns="period_norm",
        values=["d_labour_loss_tx", "delta_sleep_min"],
        aggfunc="mean"
    )

    out = pd.DataFrame(index=piv.index)

    if ("d_labour_loss_tx", "HW") in piv.columns and ("d_labour_loss_tx", "NHW") in piv.columns:
        out["delta_labour_hw_nhw"] = (
            piv[("d_labour_loss_tx", "HW")] - piv[("d_labour_loss_tx", "NHW")]
        )

    if ("delta_sleep_min", "HW") in piv.columns and ("delta_sleep_min", "NHW") in piv.columns:
        out["delta_sleep_hw_nhw"] = (
            piv[("delta_sleep_min", "HW")] - piv[("delta_sleep_min", "NHW")]
        )

    out = out.reset_index()

    diag.add(
        "Build pair HW-NHW impact delta",
        f"shape={out.shape}\n"
        f"cols={list(out.columns)}"
    )

    return out


def _scatter_with_fit(ax, df, x_col, y_col, color, title, xlabel, ylabel):
    valid = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) < 3:
        ax.set_visible(False)
        return

    x = valid[x_col].to_numpy(dtype=float)
    y = valid[y_col].to_numpy(dtype=float)

    ax.scatter(x, y, s=8, alpha=0.22, linewidths=0)

    if np.nanstd(x) > 0 and np.nanstd(y) > 0:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, m * xx + b, color=color, lw=1.4)

        r = np.corrcoef(x, y)[0, 1]
        r2 = r ** 2
        ax.text(
            0.04, 0.96,
            f"n={len(valid)}\nR²={r2:.2f}\nslope={m:.2f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=6.2
        )

    ax.axhline(0, color="black", lw=0.55, ls=":")
    ax.axvline(0, color="black", lw=0.55, ls=":")
    ax.set_title(title, fontsize=7.0, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_small_axis(ax)


def make_mechanism_scatter(df_hw_diurnal, df_pair_daynight, out_dir, diag):
    """
    Mechanism scatter:
      Daytime forcing:   pair-level Tmax(HW − NHW)
      Labour response:   pair-level labour loss(HW − NHW)

      Nighttime forcing: pair-level Tmin(HW − NHW)
      Sleep response:    pair-level sleep loss(HW − NHW)
    """
    df_temp = build_pair_hw_nhw_temperature_delta(df_hw_diurnal, diag)
    df_imp = build_pair_hw_nhw_impact_delta(df_pair_daynight, diag)

    df_mech = df_temp.merge(df_imp, on="pair_id", how="inner")

    diag.add(
        "Build mechanism scatter dataframe",
        f"shape={df_mech.shape}\n"
        f"cols={list(df_mech.columns)}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), dpi=600)

    _scatter_with_fit(
        axes[0],
        df_mech,
        "delta_tmax_hw_nhw",
        "delta_labour_hw_nhw",
        color="red",
        title="Daytime warming → labour loss",
        xlabel=r"$\Delta$Tmax, HW−NHW (°C)",
        ylabel=r"$\Delta$ labour loss, HW−NHW (%)"
    )

    _scatter_with_fit(
        axes[1],
        df_mech,
        "delta_tmin_hw_nhw",
        "delta_sleep_hw_nhw",
        color="blue",
        title="Nighttime warming → sleep loss",
        xlabel=r"$\Delta$Tmin, HW−NHW (°C)",
        ylabel=r"$\Delta$ sleep loss, HW−NHW (min/night)"
    )

    plt.tight_layout()

    out = os.path.join(out_dir, "mechanism_scatter_hw_nhw.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    diag.add("Mechanism scatter saved", out)

def build_period_uhi_daynight_maps_from_hw_diurnal(df_hw_diurnal, diag):
    """
    Supplement map metrics:
      NHW daytime   : (Tmax_U - Tmax_R)_NHW
      HW daytime    : (Tmax_U - Tmax_R)_HW
      NHW nighttime : (Tmin_U - Tmin_R)_NHW
      HW nighttime  : (Tmin_U - Tmin_R)_HW
    """
    df = df_hw_diurnal[
        ["pair_id", "station_type", "period_norm", "lon", "lat",
         "Tmax_fft", "Tmin_fft", "climate4"]
    ].copy()

    df = df[df["period_norm"].isin(["NHW", "HW"])].copy()

    loc = (
        df.sort_values(
            "station_type",
            key=lambda s: s.map({"urban": 0, "rural": 1}).fillna(2)
        )
        .drop_duplicates("pair_id")
        [["pair_id", "lon", "lat", "climate4"]]
        .rename(columns={"lon": "lon_ref", "lat": "lat_ref"})
    )

    tx = df.pivot_table(
        index=["pair_id", "period_norm"],
        columns="station_type",
        values="Tmax_fft",
        aggfunc="mean"
    ).reset_index()

    tn = df.pivot_table(
        index=["pair_id", "period_norm"],
        columns="station_type",
        values="Tmin_fft",
        aggfunc="mean"
    ).reset_index()

    if "urban" in tx.columns and "rural" in tx.columns:
        tx["uhi_tmax"] = tx["urban"] - tx["rural"]
    else:
        tx["uhi_tmax"] = np.nan

    if "urban" in tn.columns and "rural" in tn.columns:
        tn["uhi_tmin"] = tn["urban"] - tn["rural"]
    else:
        tn["uhi_tmin"] = np.nan

    out = loc.copy()

    for period in ["NHW", "HW"]:
        out = out.merge(
            tx[tx["period_norm"] == period][["pair_id", "uhi_tmax"]]
              .rename(columns={"uhi_tmax": f"uhi_tmax_{period}"}),
            on="pair_id",
            how="left"
        )
        out = out.merge(
            tn[tn["period_norm"] == period][["pair_id", "uhi_tmin"]]
              .rename(columns={"uhi_tmin": f"uhi_tmin_{period}"}),
            on="pair_id",
            how="left"
        )

    diag.add(
        "Build supplement period-specific UHI maps",
        f"shape={out.shape}\ncols={list(out.columns)}"
    )

    return out

def draw_period_uhi_map(ax, fig, df_map, value_col, title, cbar_label, panel_letter):
    valid = df_map.dropna(subset=["lon_ref", "lat_ref", value_col]).copy()
    _vmax = max(np.nanpercentile(np.abs(valid[value_col].dropna()), 98), 0.5)
    
    if HAS_CARTOPY:
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#f7f7f7', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, edgecolor="#505050", zorder=1)
        sc = ax.scatter(valid["lon_ref"].values, valid["lat_ref"].values, transform=DATA_CRS,
                        c=valid[value_col].values, cmap=MAP_CMAP, vmin=-_vmax, vmax=_vmax,
                        s=8, marker="o", linewidths=0.15, edgecolors="black", alpha=0.85, zorder=3)
        if 'geo' in ax.spines: ax.spines['geo'].set_linewidth(0.55)

    ax.set_title(title, fontsize=6.8, fontweight="bold", pad=4.0)
    add_panel_label(ax, panel_letter, x=-0.04, y=1.04)

    cax = mpl_inset_axes(ax, width="20%", height="3.5%", loc="lower center", 
                         bbox_to_anchor=(0.0, -0.06, 1.0, 1.0), 
                         bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=6.5, labelpad=0.5) # 字体调大
    cbar.ax.tick_params(labelsize=6.0, length=1.2, pad=0.5)
    for s in cax.spines.values(): s.set_visible(False)
    return sc


def make_supplement_period_uhi_maps(df_hw_diurnal, out_dir, diag):
    apply_nature_style()
    df_map = build_period_uhi_daynight_maps_from_hw_diurnal(df_hw_diurnal, diag)
    fig = plt.figure(figsize=(7.2, 4.5), dpi=600)
    gs = GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.05, 
                  left=0.04, right=0.96, top=0.92, bottom=0.06)
    
    specs = [("uhi_tmax_NHW", "Non-heatwave daytime contrast", r"$\Delta T_{x,\mathrm{NHW}}$ (°C)", "a"),
             ("uhi_tmax_HW", "Heatwave daytime contrast", r"$\Delta T_{x,\mathrm{HW}}$ (°C)", "b"),
             ("uhi_tmin_NHW", "Non-heatwave nighttime contrast", r"$\Delta T_{n,\mathrm{NHW}}$ (°C)", "c"),
             ("uhi_tmin_HW", "Heatwave nighttime contrast", r"$\Delta T_{n,\mathrm{HW}}$ (°C)", "d")]

    for i, spec in enumerate(specs):
        ax = fig.add_subplot(gs[i//2, i%2], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
        draw_period_uhi_map(ax, fig, df_map, *spec)
        
    fig.savefig(os.path.join(out_dir, "supplement_period_specific_uhi_maps.png"), **SAVEFIG_KW)
    plt.close(fig)


def make_supplement_period_uhi_maps(df_hw_diurnal, out_dir, diag):
    """
    附图：采用 FIG_H_PERIOD_MAPS 高度，增大 hspace
    """
    apply_nature_style()
    df_map = build_period_uhi_daynight_maps_from_hw_diurnal(df_hw_diurnal, diag)

    fig = plt.figure(figsize=(FIG_W, FIG_H_PERIOD_MAPS), dpi=FIG_DPI)

    gs = GridSpec(
        2, 2, figure=fig,
        hspace=0.28,  # 留给色标的空间
        wspace=0.08,
        left=0.045, right=0.995, top=0.90, bottom=0.085
    )

    specs = [
        ("uhi_tmax_NHW", "Non-heatwave daytime contrast", r"$\Delta T_{x,\mathrm{NHW}}$ (°C)", "a"),
        ("uhi_tmax_HW", "Heatwave daytime contrast", r"$\Delta T_{x,\mathrm{HW}}$ (°C)", "b"),
        ("uhi_tmin_NHW", "Non-heatwave nighttime contrast", r"$\Delta T_{n,\mathrm{NHW}}$ (°C)", "c"),
        ("uhi_tmin_HW", "Heatwave nighttime contrast", r"$\Delta T_{n,\mathrm{HW}}$ (°C)", "d"),
    ]

    for i, (value_col, title, cbar_label, letter) in enumerate(specs):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection=PLOT_PROJECTION if HAS_CARTOPY else None)
        draw_period_uhi_map(ax, fig, df_map, value_col, title, cbar_label, letter)

    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, "supplement_period_specific_uhi_maps.png"), **SAVEFIG_KW)
    fig.savefig(os.path.join(out_dir, "supplement_period_specific_uhi_maps.pdf"), **SAVEFIG_KW)
    plt.close(fig)

def make_supplement_integrated_bc(df_fig9, out_dir, diag):
    """
    Supplement file 2:
      original integrated figure b and c only:
        b : UHI absolute urban/rural diurnal temperature
        c : UCI absolute urban/rural diurnal temperature
    """
    apply_nature_style()

    fig9_sum = build_fig9_summary(df_fig9, diag)

    fig = plt.figure(figsize=(7.2, 2.75), dpi=600)

    gs = GridSpec(
        1, 2, figure=fig,
        wspace=0.24,
        left=0.075,
        right=0.985,
        top=0.80,
        bottom=0.20
    )

    # --- 统一配色方案 ---
    CLR_UHI_HW = "#FF7F00"   # UHI 热浪：亮橙色
    CLR_UHI_NHW = "#FFCC80"  # UHI 非热浪：浅橙色
    CLR_UCI_HW = "#A04000"   # UCI 热浪：砖红色
    CLR_UCI_NHW = "#D98880"  # UCI 非热浪：浅砖红色 (与深色系统一)

    ax_b = fig.add_subplot(gs[0, 0])

    draw_fig9_abs_panel(
        ax_b,
        fig9_sum[("UHI", "HW")],
        fig9_sum[("UHI", "NHW")],
        "UHI",
        show_legend=True,
        hw_color=CLR_UHI_HW,     # 橙色系-深
        nhw_color=CLR_UHI_NHW    # 橙色系-浅
    )
    ax_b.set_title("UHI regime", fontsize=6.5, fontweight="bold")
    add_panel_label(ax_b, "a", x=-0.10, y=1.02)

    ax_c = fig.add_subplot(gs[0, 1])
    draw_fig9_abs_panel(
        ax_c,
        fig9_sum[("UCI", "HW")],
        fig9_sum[("UCI", "NHW")],
        "UCI",
        show_legend=False,
        hw_color=CLR_UCI_HW,     # 砖红色系-深
        nhw_color=CLR_UCI_NHW    # 砖红色系-浅
    )
    ax_c.set_title("UCI regime", fontsize=6.5, fontweight="bold")
    add_panel_label(ax_c, "b", x=-0.10, y=1.02)
    
    ensure_dir(out_dir)
    png_path = os.path.join(out_dir, "supplement_integrated_panels_bc.png")
    pdf_path = os.path.join(out_dir, "supplement_integrated_panels_bc.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    diag.add("Supplement integrated b-c panels saved", f"{png_path}\n{pdf_path}")
    print(f"  ✓ saved: {png_path}")
    print(f"  ✓ saved: {pdf_path}")

def make_supplement_station_kg_stats(df_hw_diurnal, out_dir, diag):
    apply_nature_style()
    pair_df = df_hw_diurnal.drop_duplicates("pair_id").dropna(subset=["climate4"])
    fig = plt.figure(figsize=(7.2, 4.2), dpi=600)
    
    ax_map = fig.add_subplot(1, 1, 1, projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    if HAS_CARTOPY:
        ax_map.set_global()
        ax_map.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
        ax_map.add_feature(cfeature.LAND, facecolor='#f7f7f7', zorder=0)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#444444")
        for z in CLIMATE_ORDER:
            sub = pair_df[pair_df["climate4"] == z]
            ax_map.scatter(sub["lon"], sub["lat"], transform=DATA_CRS, color=ZONE_EDGE[z],
                           s=6.5, label=CLIMATE_NAME[z], edgecolors="black", linewidths=0.18, zorder=3)

    ax_map.set_title("Global distribution of urban-rural station pairs", fontsize=7.5, fontweight="bold")
    add_panel_label(ax_map, "a", x=0.03, y=0.94)

    ax_map.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.0, title="Climate zone", 
                  title_fontsize=7.5, frameon=True)

    # --- 柱状图 Inset (不透明 + 50步长) ---
    ax_bar = mpl_inset_axes(ax_map, width="25%", height="23%", loc="lower left", 
                            bbox_to_anchor=(0.06, 0.12, 1, 1), 
                            bbox_transform=ax_map.transAxes, borderpad=0)
    
    # 完全遮挡地图线
    ax_bar.patch.set_facecolor('white')
    ax_bar.patch.set_alpha(1.0)
    ax_bar.set_zorder(10)

    counts = pair_df.groupby("climate4").size()
    x_short = ["Trop", "Arid", "Temp", "Cold"]
    y_vals = [counts.get(z, 0) for z in CLIMATE_ORDER]
    ax_bar.bar(x_short, y_vals, color=[ZONE_EDGE[z] for z in CLIMATE_ORDER], alpha=1.0, 
               edgecolor="black", linewidth=0.4, width=0.6, zorder=11)
    
    # 设置 50 步长坐标
    import matplotlib.ticker as ticker
    ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(50))
    
    ax_bar.set_ylabel("Pairs", fontsize=7.0)
    ax_bar.tick_params(axis="both", labelsize=6.5, length=1.5, pad=1)
    add_panel_label(ax_bar, "b", x=0.0, y=1.05)

    ensure_dir(out_dir); fig.savefig(os.path.join(out_dir, "supplement_station_kg_distribution.png"), **SAVEFIG_KW); plt.close(fig)


def make_supplement_uhi_uci_pdf(df_hw_diurnal, out_dir, diag):
    """
    采用指定的深蓝-珊瑚配色方案：
    - NHW (Non-heatwave): 深蓝色 (#084594), 虚线
    - HW (Heatwave): 珊瑚色 (#f03b20), 实线
    """
    from scipy.stats import gaussian_kde
    apply_nature_style()
    
    # 提取数据 (保持原逻辑)
    df_map = build_period_uhi_daynight_maps_from_hw_diurnal(df_hw_diurnal, diag)
    
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=600)
    plt.subplots_adjust(wspace=0.3, bottom=0.2, top=0.85, left=0.1, right=0.95)

    metrics = [
        ("Daytime", "uhi_tmax_NHW", "uhi_tmax_HW"),
        ("Nighttime", "uhi_tmin_NHW", "uhi_tmin_HW")
    ]
    
    letters = ["a", "b"]

    # --- 颜色指定 ---
    color_nhw = "#084594" # 深蓝色
    color_hw = "#f03b20"  # 珊瑚色

    for i, (label, col_nhw, col_hw) in enumerate(metrics):
        ax = axes[i]
        
        # 配置绘图参数: (时期, 列名, 颜色, 线型, 线宽)
        plot_configs = [
            ("NHW", col_nhw, color_nhw, "--", 1.3),
            ("HW",  col_hw,  color_hw,  "-",  2.2)
        ]

        for period, col, color, ls, lw in plot_configs:
            data = df_map[col].dropna()
            if len(data) < 2: 
                continue
            
            # 计算 KDE (保持原逻辑)
            x_grid = np.linspace(data.min() - 1, data.max() + 1, 200)
            kde = gaussian_kde(data)
            pdf_vals = kde(x_grid)
            
            # 1. 绘制 PDF 曲线 (线型 ls 和 线宽 lw 保持区分度)
            ax.plot(x_grid, pdf_vals, color=color, ls=ls, lw=lw, label=period, zorder=3)
            
            # 2. 填充半透明阴影
            ax.fill_between(x_grid, 0, pdf_vals, color=color, alpha=0.12, zorder=2)
            
            # 3. 绘制均值标注虚线 (线型与主曲线匹配)
            ax.axvline(data.mean(), color=color, ls=ls, lw=0.9, alpha=0.8, zorder=4)

        # 辅助设置 (保持原逻辑)
        ax.axvline(0, color='black', lw=0.6, ls='-', alpha=0.3, zorder=1)
        ax.set_title(f"{label} urban-rural contrast", fontsize=8, fontweight='bold')

        # 坐标轴标签
        xlabel = r"$\Delta T_{x}$ (U-R, °C)" if i == 0 else r"$\Delta T_{n}$ (U-R, °C)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        
        style_small_axis(ax)
        add_panel_label(ax, letters[i], x=-0.12, y=1.05)
        
        if i == 0: 
            ax.legend(frameon=False, fontsize=6, loc="upper right")

    # 保存文件
    ensure_dir(out_dir)
    save_path = os.path.join(out_dir, "supplement_uhi_uci_pdf_navy_coral.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    diag.add("Supplement PDF figure (Navy & Coral) saved", save_path)


# ============================================================================
# §12 Main
# ============================================================================
def main():
    t0   = time.time()
    diag = DiagnosticLog()

    ensure_dir(OUTPUT_DIR)

    print("=" * 80)
    print("Integrated climate-zone figure v4  [all-station patch]")
    print("  · Figure inputs unified to 01 main-analysis pair-period outputs")
    print("  · map & diurnal profiles reshaped from all_pair_period_metrics.csv")
    print("  · all boxplots pool urban + rural stations (no distinction)")
    print("  · pairs with no heatwave → delta_Tmax = 0 on map")
    print("  · subplot titles: no '(urban)' qualifier")
    print("=" * 80)

    print("\n[1/7] Loading datasets...")
    df_cdh  = load_cdh_panel(diag)
    df_lc   = load_labour_full(diag)
    df_hne  = load_hne_paired(diag)
    
    df_fig9 = load_fig9_source(diag)
    # Build station-period interface from 01 main-analysis output.
    df_hw_diurnal = load_hw_diurnal_csv(diag)
    matched_ids = load_reference_matched_pair_ids(diag=diag)
    for _name, _df in [("cdh", df_cdh), ("labour", df_lc), ("hne", df_hne)]:
        if _df is not None and "pair_id" in _df.columns:
            _df["pair_id"] = _df["pair_id"].astype(str)
    df_cdh = df_cdh[df_cdh["pair_id"].astype(str).isin(matched_ids)].copy()
    if df_lc is not None: df_lc = df_lc[df_lc["pair_id"].astype(str).isin(matched_ids)].copy()
    if df_hne is not None: df_hne = df_hne[df_hne["pair_id"].astype(str).isin(matched_ids)].copy()

    print("\n[2/7] Building station daily panel...")
    df_station_daily = build_station_daily_from_cdh(df_cdh, diag)

    print("\n[3/7] Aggregating physical metrics...")
    df_phys = aggregate_station_period_from_daily(df_station_daily, diag)

    print("\n[4/7] Stationising impact modules (HNE now includes econ loss)...")
    df_hne_st  = build_station_period_from_hne(df_hne, diag)
    df_lab_st  = build_station_period_from_labour(df_lc, diag)

    print("\n[5/7] Merging station-period modules...")
    df_station = merge_station_modules(df_phys, df_hne_st, df_lab_st, diag)

    print("\n[6/7] Merging climate zone labels + building diurnal summary...")
    # Climate labels are carried from 01 main-analysis output.
    df_station_clim = merge_climate_to_station_panel_hw(
        df_station, df_hw_diurnal, diag
    )
    df_station_clim.to_csv(
        os.path.join(OUTPUT_DIR, "station_period_panel_with_climate_v4.csv"),
        index=False)

    print("\n[6b] Building pair-level day/night loss panel...")
    df_pair_daynight = build_pair_daynight_panel(
        df_lc, df_station_clim, df_hw_diurnal, diag
    )

    df_pair_daynight.to_csv(
        os.path.join(OUTPUT_DIR, "pair_daynight_sleepmin_panel.csv"),
        index=False
    )

    # Build diurnal long table from the reshaped 01 output.
    df_diurnal_long    = build_diurnal_long_from_hw_diurnal(df_hw_diurnal, diag)
    df_diurnal_summary = summarize_diurnal(df_diurnal_long)

    print("\n[Counts]")
    # Pass df_hw_diurnal as the pair-level reference (has pair_id + climate4)
    write_counts(df_station_clim, df_hw_diurnal, OUTPUT_DIR)

    print("\n[7/7] Plotting split figures...")
    make_integrated_figure(
        df_hw_diurnal,
        df_diurnal_summary,
        df_station_clim,
        df_pair_daynight,
        df_fig9,
        OUTPUT_DIR,
        diag
    )

    make_composite_maps_de_figure(
        df_hw_diurnal,
        df_fig9,
        OUTPUT_DIR,
        diag
    )

    make_composite_maps_de_figure_v2(
        df_hw_diurnal,
        df_fig9,
        OUTPUT_DIR,
        diag
    )

    make_composite_maps_de_figure_v2_pro(
        df_hw_diurnal,
        df_fig9,
        OUTPUT_DIR,
        diag
    )

    make_composite_maps_de_figure_v4(
        df_hw_diurnal,
        df_fig9,
        OUTPUT_DIR,
        diag
    )

    make_supplementary_climate_breakdown_figure(
        df_station_clim,
        df_pair_daynight,
        df_fig9,
        OUTPUT_DIR,
        diag
    )


    make_mechanism_scatter(
        df_hw_diurnal,
        df_pair_daynight,
        OUTPUT_DIR,
        diag
    )

    make_supplement_period_uhi_maps(
        df_hw_diurnal,
        OUTPUT_DIR,
        diag
    )

    make_supplement_integrated_bc(
        df_fig9,
        OUTPUT_DIR,
        diag
    )

        # [新增调用] 生成地图+纬度剖面对齐图
    print("\n[New] Plotting latitudinal analysis figure...")
    make_latitudinal_analysis_figure(df_hw_diurnal, OUTPUT_DIR, diag)

    # [新增调用]
    print("\n[Supplementary] Plotting KG stats and UHI PDFs...")
    make_supplement_station_kg_stats(df_hw_diurnal, OUTPUT_DIR, diag)
    make_supplement_uhi_uci_pdf(df_hw_diurnal, OUTPUT_DIR, diag)


    diag.write(os.path.join(OUTPUT_DIR, "diagnostic_integrated_climatezone_v4.md"))

    print("\n" + "=" * 80)
    print(f"Done. Elapsed: {time.time() - t0:.1f} s")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Files:")
    print("  figure_map_gj_compact.png")
    print("  figure_map_gj_compact.pdf")
    print("  figure_diurnal_abcd.png")
    print("  figure_diurnal_abcd.pdf")
    print("  figure_fhik_abcd.png")
    print("  figure_fhik_abcd.pdf")
    print("  station_period_panel_with_climate_v4.csv")
    print("  diagnostic_integrated_climatezone_v4.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
