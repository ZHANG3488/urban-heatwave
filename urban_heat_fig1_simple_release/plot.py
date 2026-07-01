#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Figure 1 plotting workflow.

This module intentionally generates only the following five PNG files:

1. composite_delta_uhi_final_fianl_ver.png
2. supplement_integrated_panels_bc.png
3. supplement_period_specific_uhi_maps.png
4. supplement_station_kg_distribution.png
5. supplement_uhi_uci_pdf_navy_coral.png

Only two upstream datasets are read:

- analysis/main_multiyear/robustness_percentile/all_pair_period_metrics.csv
- analysis/heatwave_flags/station_diurnal_reconstructed.csv

No CDH, labour, mortality, HNE, economic-loss, station-panel, boxplot,
mechanism-regression, or unrelated supplementary calculations are executed.
The scientific definitions used by the retained figures are unchanged.
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as mpl_inset_axes

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception as exc:  # pragma: no cover - depends on compiled dependency
    raise RuntimeError(
        "Cartopy is required for the five-map Figure 1 workflow. "
        "Install the supplied conda environment before plotting."
    ) from exc

warnings.filterwarnings("ignore")
HAS_CARTOPY = True



# -----------------------------------------------------------------------------
# Single-file configuration
# -----------------------------------------------------------------------------
def _export_config_to_environment() -> dict:
    """Read config.yaml beside this script.

    Only paths are configured here; retained plotting calculations are unchanged.
    """
    import yaml

    config_path = Path(__file__).resolve().with_name("config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing configuration file: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    paths = config.get("paths", {})
    unified_root = paths.get("unified_root")
    if not unified_root:
        raise ValueError("config.yaml: paths.unified_root is required")

    output_dir = paths.get(
        "figure1_output_dir",
        str(Path(unified_root) / "plot_data" / "fig1_n"),
    )
    os.environ["UHR_UNIFIED_ROOT"] = str(unified_root)
    os.environ["UHR_FIG1_OUTPUT_DIR"] = str(output_dir)
    return config


CONFIG = _export_config_to_environment()

UNIFIED_ROOT = os.environ.get(
    "UHR_UNIFIED_ROOT",
    "/home/yuxizhang/code/gzz/final_code/impact/final/code_era/Final_code/"
    "latest_full_analysis_plot_code/unified_outputs",
)
OUTPUT_DIR = os.environ.get(
    "UHR_FIG1_OUTPUT_DIR",
    os.path.join(UNIFIED_ROOT, "plot_data", "fig1_n"),
)
HW_DETECTION_DIR = os.path.join(UNIFIED_ROOT, "analysis", "heatwave_flags")
ANALYSIS_MULTIYEAR_CSV = os.path.join(
    UNIFIED_ROOT,
    "analysis",
    "main_multiyear",
    "robustness_percentile",
    "all_pair_period_metrics.csv",
)

PLOT_PROJECTION = ccrs.Robinson(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()

PERIOD_NORM = {
    "annual": "annual",
    "warm_season": "JJA",
    "JJA": "JJA",
    "heatwave": "HW",
    "HW": "HW",
    "non_heatwave": "NHW",
    "NHW": "NHW",
}
CLIMATE_ORDER = ["A", "B", "C", "D"]
CLIMATE_NAME = {
    "A": "Tropical",
    "B": "Arid",
    "C": "Temperate",
    "D": "Cold",
}
HOURS = np.arange(24)
FONT_FAMILY = "DejaVu Sans"
FONT_BASE = 7
CLR_PERIOD = {
    "HW": "#b2182b",
    "NHW": "#2166ac",
    "JJA": "#f4a582",
}
ZONE_EDGE = {
    "A": "#8c2d04",
    "B": "#cc4c02",
    "C": "#225ea8",
    "D": "#41b6c4",
}
MAP_CMAP = plt.cm.RdYlBu_r
FIG_W = 7.2
FIG_DPI = 600
FIG_H_PERIOD_MAPS = 4.55
SAVEFIG_KW = dict(dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.025)


_NATURAL_EARTH_AVAILABLE: bool | None = None


def natural_earth_available() -> bool:
    """Return whether Cartopy Natural Earth basemap files are locally usable.

    Cartopy downloads these files lazily. On an offline Linux machine the
    scientific station layers can still be reproduced; only coastlines, land
    fill and borders are omitted.
    """
    global _NATURAL_EARTH_AVAILABLE
    if _NATURAL_EARTH_AVAILABLE is not None:
        return _NATURAL_EARTH_AVAILABLE
    try:
        from cartopy.io import shapereader
        shapereader.natural_earth(
            resolution="110m", category="physical", name="land"
        )
        shapereader.natural_earth(
            resolution="110m", category="physical", name="coastline"
        )
        shapereader.natural_earth(
            resolution="110m", category="cultural",
            name="admin_0_boundary_lines_land"
        )
        _NATURAL_EARTH_AVAILABLE = True
    except Exception as exc:
        _NATURAL_EARTH_AVAILABLE = False
        print(
            "Warning: Cartopy Natural Earth basemap files are unavailable; "
            "continuing with projection outlines, graticules and station data only. "
            f"Details: {exc}"
        )
    return _NATURAL_EARTH_AVAILABLE

TARGET_OUTPUTS = (
    "composite_delta_uhi_final_fianl_ver.png",
    "supplement_integrated_panels_bc.png",
    "supplement_period_specific_uhi_maps.png",
    "supplement_station_kg_distribution.png",
    "supplement_uhi_uci_pdf_navy_coral.png",
)


def validate_minimal_plot_inputs() -> None:
    """Fail early with a concise description of missing input files/columns."""
    station_path = Path(HW_DETECTION_DIR) / "station_diurnal_reconstructed.csv"
    metrics_path = Path(ANALYSIS_MULTIYEAR_CSV)
    missing_files = [str(p) for p in (station_path, metrics_path) if not p.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing minimal Figure 1 input file(s):\n  - "
            + "\n  - ".join(missing_files)
        )

    metrics_cols = set(pd.read_csv(metrics_path, nrows=0).columns)
    station_cols = set(pd.read_csv(station_path, nrows=0).columns)

    required_metrics = {
        "pair_id", "period", "group", "dTmean", "dAmp1", "dTx", "dTn",
        *(f"urban_diurnal_h{h:02d}" for h in range(24)),
        *(f"rural_diurnal_h{h:02d}" for h in range(24)),
    }
    required_station = {
        "pair_id", "period", "station_type", "kg_group", "climate_zone",
        "lon", "lat", "Tmax_fft", "Tmin_fft",
    }

    missing_metrics = sorted(required_metrics - metrics_cols)
    missing_station = sorted(required_station - station_cols)
    if missing_metrics or missing_station:
        messages = []
        if missing_metrics:
            messages.append(
                "all_pair_period_metrics.csv missing columns: "
                + ", ".join(missing_metrics)
            )
        if missing_station:
            messages.append(
                "station_diurnal_reconstructed.csv missing columns: "
                + ", ".join(missing_station)
            )
        raise ValueError("\n".join(messages))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def norm_period(x):
    return PERIOD_NORM.get(str(x), str(x))


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


def load_hw_diurnal_csv(diag):
    """Load station diurnal reconstruction and attach strict annual groups."""
    p = os.path.join(HW_DETECTION_DIR, "station_diurnal_reconstructed.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Upstream diurnal CSV not found: {p}")
    df = pd.read_csv(p)
    if "hw_method" in df.columns:
        df = df[df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")].copy()
    df["pair_id"] = df["pair_id"].astype(str)
    df["period_norm"] = df["period"].apply(norm_period)
    df["climate4"] = df["kg_group"].apply(
        lambda x: str(x).strip()[0].upper()
        if pd.notna(x) and str(x).strip() and str(x).strip()[0].upper() in CLIMATE_ORDER
        else np.nan
    )
    df = apply_canonical_annual_group(df, diag=diag)
    ids = load_reference_matched_pair_ids(diag=diag)
    df = df[df["pair_id"].isin(ids)].copy()
    diag.add(
        "Load HW diurnal CSV",
        f"path={p}\nshape={df.shape}\nmatched_pairs={df['pair_id'].nunique()}\n"
        f"periods={sorted(df['period_norm'].dropna().unique().tolist())}"
    )
    return df


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

    out[["delta_uhi_daytime", "delta_uhi_nighttime"]] = (
        out[["delta_uhi_daytime", "delta_uhi_nighttime"]].fillna(0)
    )

    diag.add(
        "Build day/night ΔUHI map metrics",
        f"shape={out.shape}\n"
        f"day mean={out['delta_uhi_daytime'].mean():.3f}\n"
        f"night mean={out['delta_uhi_nighttime'].mean():.3f}"
    )

    return out


def get_lat_profile_stats(df, value_col, bin_width=5):
    """计算纬度分箱统计量"""
    valid = df.dropna(subset=['lat_ref', value_col]).copy()
    if len(valid) == 0: return None
    bins = np.arange(-60, 90, bin_width)
    valid['lat_bin'] = pd.cut(valid['lat_ref'], bins=bins)
    prof = valid.groupby('lat_bin', observed=True)[value_col].agg(['mean', 'sem']).reset_index()
    prof['lat_mid'] = prof['lat_bin'].apply(lambda x: x.mid)
    return prof.dropna(subset=['mean'])


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
        ax.set_facecolor('white')
        if natural_earth_available():
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
        ax.set_facecolor('white')
        if natural_earth_available():
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
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    diag.add("Supplement integrated b-c panels saved", png_path)
    print(f"  ✓ saved: {png_path}")


def make_supplement_station_kg_stats(df_hw_diurnal, out_dir, diag):
    apply_nature_style()
    pair_df = df_hw_diurnal.drop_duplicates("pair_id").dropna(subset=["climate4"])
    fig = plt.figure(figsize=(7.2, 4.2), dpi=600)
    
    ax_map = fig.add_subplot(1, 1, 1, projection=PLOT_PROJECTION if HAS_CARTOPY else None)
    if HAS_CARTOPY:
        ax_map.set_global()
        ax_map.set_facecolor('white')
        if natural_earth_available():
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




def main() -> None:
    """Load only the two required tables and generate exactly five PNG files."""
    started = time.time()
    diag = DiagnosticLog()
    validate_minimal_plot_inputs()
    ensure_dir(OUTPUT_DIR)

    print("=" * 78)
    print("Minimal Figure 1 plotting workflow")
    print("Inputs: all_pair_period_metrics.csv + station_diurnal_reconstructed.csv")
    print("Outputs: exactly five requested PNG files")
    print("=" * 78)

    df_fig9 = load_fig9_source(diag)
    df_hw_diurnal = load_hw_diurnal_csv(diag)

    make_composite_maps_de_figure_v4(
        df_hw_diurnal, df_fig9, OUTPUT_DIR, diag
    )
    make_supplement_integrated_bc(
        df_fig9, OUTPUT_DIR, diag
    )
    make_supplement_period_uhi_maps(
        df_hw_diurnal, OUTPUT_DIR, diag
    )
    make_supplement_station_kg_stats(
        df_hw_diurnal, OUTPUT_DIR, diag
    )
    make_supplement_uhi_uci_pdf(
        df_hw_diurnal, OUTPUT_DIR, diag
    )

    missing = [
        name for name in TARGET_OUTPUTS
        if not (Path(OUTPUT_DIR) / name).is_file()
        or (Path(OUTPUT_DIR) / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "The plotting workflow finished but these outputs are missing/empty: "
            + ", ".join(missing)
        )

    print("\nGenerated files:")
    for name in TARGET_OUTPUTS:
        print(f"  ✓ {Path(OUTPUT_DIR) / name}")
    print(f"Elapsed: {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
