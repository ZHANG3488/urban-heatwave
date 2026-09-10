#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot-only Figure 1: global day-night heatwave reorganization.

Formal mode reads exactly the same upstream table used by the legacy Figure 1:
``all_pair_period_metrics.csv``.  It does not read a prepared map table and it
does not fit a model, resample a confidence interval, or change the cohort.

Panels
------
a  Global night-minus-day heatwave response, R_n - R_x.
b  UHI/UCI heatwave-response diurnal curves, R(t) = HW - NHW.
c  Pair-level daytime and night-time responses, R_x versus R_n; large labelled
   circles are the UHI and UCI group means.

The script writes a new timestamped directory containing a fixed-canvas
600-dpi RGB JPG, a small WebP preview, a manifest and ``_SUCCESS.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

_MPL_CACHE = Path(tempfile.gettempdir()) / "figure1_global_daynight_mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False


SCRIPT_VERSION = "plot_figure1_global_daynight"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_CSV = (
    PROJECT_ROOT
    / "outputs/analysis/main_multiyear/robustness_percentile/"
    "all_pair_period_metrics.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/figures/figure1_global_daynight"
)

MAIN_FILENAME = "figure1_global_daynight.jpg"
PREVIEW_FILENAME = "figure1_global_daynight_preview.webp"
MANIFEST_FILENAME = "figure1_global_daynight_plot_manifest.json"

MM_TO_IN = 1.0 / 25.4
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 180.0
FORMAL_DPI = 600

# Manuscript-wide Nature Geoscience figure style.  Keep these constants aligned
# across Figures 1--4; panel hierarchy comes from layout and mark emphasis, not
# from changing typography between panels.
BASE_FONT_SIZE = 6.3
SUBTITLE_SIZE = 7.8
PANEL_LABEL_SIZE = 9.5
AXIS_LABEL_SIZE = 7.2
TICK_SIZE = 6.4
LEGEND_SIZE = 6.0
ANNOT_SIZE = 6.0
COLORBAR_LABEL_SIZE = 6.8
COLORBAR_TICK_SIZE = 6.2

FRAME_LW = 0.75
DATA_LW = 0.9
REFERENCE_LW = 0.7
GRID_LW = 0.4
MARKER_EDGE_LW = 0.35
LOWER_ROW_GAP_FRACTION = 0.10

EXPECTED_TOTAL = 317
EXPECTED_GROUP_COUNTS = {"UHI": 200, "UCI": 117}
NUMERIC_TOLERANCE = 1.0e-8

# UHI/UCI retain the manuscript-wide red/blue semantics.  The map colour is
# reserved exclusively for the continuous R_n - R_x response.
DAY_ACCENT_COLOR = "#D58A0A"
NIGHT_ACCENT_COLOR = "#A63D32"
DAY_EXTREME_COLOR = "#914F00"
NIGHT_EXTREME_COLOR = "#681F20"
WARM_NEUTRAL_COLOR = "#E8DDCA"
NEGATIVE_RESPONSE_COLOR = "#706A64"
WEAK_NEGATIVE_RESPONSE_COLOR = "#B8AFA5"
UHI_CURVE_COLOR = "#BB3445"
UCI_CURVE_COLOR = "#3C79B6"
ZERO_COLOR = WARM_NEUTRAL_COLOR
UHI_EDGE = "#8C4A32"
UCI_EDGE = "#4D4D4D"
GRID_COLOR = "#D8D2CB"
DAY_BACKGROUND = "#E8C783"
NIGHT_BACKGROUND = "#D8A9A1"

# Two contiguous sign-gradients with no white centre: negative values ramp
# from light to deep purple, positive values from light to deep green, and
# the two light ends meet directly at zero.
ASYMMETRY_CMAP = LinearSegmentedColormap.from_list(
    "fig1_asymmetry_prgn",
    [(0.0, "#40004B"), (0.5, "#C2A5CF"), (0.5, "#A6DBA0"), (1.0, "#00441B")],
    N=256,
)

PERIOD_MAP = {
    "annual": "annual",
    "warm_season": "JJA",
    "jja": "JJA",
    "heatwave": "HW",
    "hw": "HW",
    "non_heatwave": "NHW",
    "nhw": "NHW",
}
HOURS = np.arange(24)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def norm_period(value: object) -> str:
    key = str(value).strip().lower()
    return PERIOD_MAP.get(key, str(value).strip())


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": SUBTITLE_SIZE,
            "axes.titleweight": "semibold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": FRAME_LW,
            "lines.linewidth": DATA_LW,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_formal_metrics(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Formal Figure 1 source not found: {path}")
    data = pd.read_csv(path, low_memory=False)
    required = {
        "pair_id",
        "period",
        "group",
        "lat_urban",
        "lon_urban",
        "lat_rural",
        "lon_rural",
        "urban_Tmax_fft",
        "rural_Tmax_fft",
        "urban_Tmin_fft",
        "rural_Tmin_fft",
        "dTmean",
        "dAmp1",
        "dTx",
        "dTn",
        *(f"urban_diurnal_h{hour:02d}" for hour in range(24)),
        *(f"rural_diurnal_h{hour:02d}" for hour in range(24)),
    }
    require_columns(data, required, str(path))
    return data


def freeze_matched_cohort(
    metrics: pd.DataFrame,
    enforce_formal_counts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Apply the legacy Figure 1 cohort and construct exact R_x/R_n maps."""
    data = metrics.copy()
    if "hw_method" in data.columns:
        data = data.loc[
            data["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()
    data["pair_id"] = data["pair_id"].astype(str)
    data["period_norm"] = data["period"].map(norm_period)

    duplicate = data.duplicated(["pair_id", "period_norm"], keep=False)
    if duplicate.any():
        examples = data.loc[duplicate, ["pair_id", "period_norm"]].head(20)
        raise ValueError(
            "Formal table has duplicate pair-period rows; examples: "
            + examples.to_dict("records").__repr__()
        )

    annual = data.loc[
        data["period_norm"].eq("annual"), ["pair_id", "group"]
    ].copy()
    annual["group"] = annual["group"].astype(str).str.upper().str.strip()
    annual = annual.loc[annual["group"].isin(["UHI", "UCI"])]
    conflicts = annual.groupby("pair_id", observed=True)["group"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Conflicting annual UHI/UCI classifications.")
    canonical = annual.drop_duplicates("pair_id")

    numeric = [
        "dTmean",
        "dAmp1",
        "dTx",
        "dTn",
        "urban_Tmax_fft",
        "rural_Tmax_fft",
        "urban_Tmin_fft",
        "rural_Tmin_fft",
        "lat_urban",
        "lon_urban",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    period_columns = [
        "pair_id",
        "dTmean",
        "dAmp1",
        "dTx",
        "dTn",
        "urban_Tmax_fft",
        "rural_Tmax_fft",
        "urban_Tmin_fft",
        "rural_Tmin_fft",
        "lat_urban",
        "lon_urban",
    ]
    nhw = data.loc[data["period_norm"].eq("NHW"), period_columns].copy()
    hw = data.loc[data["period_norm"].eq("HW"), period_columns].copy()
    nhw = nhw.rename(columns={c: f"{c}_nhw" for c in period_columns if c != "pair_id"})
    hw = hw.rename(columns={c: f"{c}_hw" for c in period_columns if c != "pair_id"})
    paired = nhw.merge(hw, on="pair_id", how="inner", validate="one_to_one")
    complete = [
        f"{metric}_{period}"
        for metric in ("dTmean", "dAmp1", "dTx", "dTn")
        for period in ("nhw", "hw")
    ]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=complete)
    paired = paired.merge(canonical, on="pair_id", how="inner", validate="one_to_one")

    paired["R_x"] = (
        paired["urban_Tmax_fft_hw"]
        - paired["rural_Tmax_fft_hw"]
        - paired["urban_Tmax_fft_nhw"]
        + paired["rural_Tmax_fft_nhw"]
    )
    paired["R_n"] = (
        paired["urban_Tmin_fft_hw"]
        - paired["rural_Tmin_fft_hw"]
        - paired["urban_Tmin_fft_nhw"]
        + paired["rural_Tmin_fft_nhw"]
    )
    paired["R_x_direct"] = paired["dTx_hw"] - paired["dTx_nhw"]
    paired["R_n_direct"] = paired["dTn_hw"] - paired["dTn_nhw"]
    paired["R_mean"] = paired["dTmean_hw"] - paired["dTmean_nhw"]
    paired["R_Amp"] = paired["dAmp1_hw"] - paired["dAmp1_nhw"]
    paired["asymmetry"] = paired["R_n"] - paired["R_x"]
    paired["lat"] = paired[["lat_urban_nhw", "lat_urban_hw"]].mean(axis=1)
    paired["lon"] = paired[["lon_urban_nhw", "lon_urban_hw"]].mean(axis=1)

    rx_diff = np.abs(paired["R_x"] - paired["R_x_direct"])
    rn_diff = np.abs(paired["R_n"] - paired["R_n_direct"])
    max_rx_diff = float(rx_diff.max(skipna=True))
    max_rn_diff = float(rn_diff.max(skipna=True))
    if max_rx_diff > NUMERIC_TOLERANCE or max_rn_diff > NUMERIC_TOLERANCE:
        raise ValueError(
            "Legacy Tmax/Tmin construction disagrees with formal dTx/dTn fields: "
            f"max |R_x difference|={max_rx_diff:.3g}; "
            f"max |R_n difference|={max_rn_diff:.3g}."
        )

    ids = set(paired["pair_id"])
    cohort = data.loc[data["pair_id"].isin(ids)].copy()
    cohort = cohort.drop(columns=["group"], errors="ignore").merge(
        canonical, on="pair_id", how="inner", validate="many_to_one"
    )

    counts = paired["group"].value_counts().to_dict()
    if enforce_formal_counts:
        if len(paired) != EXPECTED_TOTAL or counts != EXPECTED_GROUP_COUNTS:
            raise ValueError(
                "Formal Figure 1 cohort mismatch: "
                f"observed total={len(paired)}, counts={counts}; expected "
                f"total={EXPECTED_TOTAL}, counts={EXPECTED_GROUP_COUNTS}."
            )
    audit = {
        "n_pairs": int(len(paired)),
        "n_uhi": int(counts.get("UHI", 0)),
        "n_uci": int(counts.get("UCI", 0)),
        "max_abs_rx_identity_difference": max_rx_diff,
        "max_abs_rn_identity_difference": max_rn_diff,
    }
    return cohort, paired, audit


def summarize_diurnal_response(cohort: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    urban_cols = [f"urban_diurnal_h{hour:02d}" for hour in range(24)]
    rural_cols = [f"rural_diurnal_h{hour:02d}" for hour in range(24)]
    output: dict[str, dict[str, np.ndarray]] = {}
    for group in ("UHI", "UCI"):
        nhw = cohort.loc[cohort["group"].eq(group) & cohort["period_norm"].eq("NHW")]
        hw = cohort.loc[cohort["group"].eq(group) & cohort["period_norm"].eq("HW")]
        if nhw.empty or hw.empty:
            raise ValueError(f"No {group} NHW/HW rows for the diurnal panel.")
        nhw = nhw.set_index("pair_id")
        hw = hw.set_index("pair_id")
        ids = sorted(set(nhw.index) & set(hw.index))
        u_nhw = nhw.loc[ids, urban_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        r_nhw = nhw.loc[ids, rural_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        u_hw = hw.loc[ids, urban_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        r_hw = hw.loc[ids, rural_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        # Pair-level HW minus NHW contrast curves, one per matched pair.
        pair_curves = (u_hw - r_hw) - (u_nhw - r_nhw)
        mean = np.nanmean(pair_curves, axis=0)
        se = np.nanstd(pair_curves, axis=0, ddof=1) / np.sqrt(len(ids))
        output[group] = {
            "mean": mean,
            "ci_low": mean - 1.96 * se,
            "ci_high": mean + 1.96 * se,
            "n_pairs": len(ids),
        }
    return output


def panel_heading(
    ax: plt.Axes,
    label: str,
    subtitle: str,
    subtitle_color: str = "#111111",
    label_x: float = -0.04,
    label_y: float = 1.12,
) -> None:
    # The panel letter sits immediately to the left of a left-aligned
    # subtitle, sharing its baseline.
    ax.set_title(
        subtitle,
        loc="center",
        fontsize=SUBTITLE_SIZE,
        fontweight="bold",
        pad=8.0,
        color=subtitle_color,
    )
    panel_title = ax.set_title(
        label,
        loc="left",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        pad=8.0,
    )
    panel_title.set_x(label_x)
    panel_title.set_ha("right")
    panel_title.set_clip_on(False)


def format_axis(ax: plt.Axes, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LW)
        spine.set_color("#222222")
    if grid:
        ax.grid(
            True, color=GRID_COLOR, linestyle=(0, (1.3, 2.2)),
            linewidth=GRID_LW,
        )
        ax.set_axisbelow(True)


def align_lower_row_to_map(
    fig: plt.Figure,
    map_ax: plt.Axes,
    left_ax: plt.Axes,
    right_ax: plt.Axes,
) -> dict[str, float]:
    """Align the lower-row frames to the map's rendered left/right edges."""
    # GeoAxes applies its projection aspect after layout.  A draw is therefore
    # required before the active map box can be used as the alignment anchor.
    fig.canvas.draw()
    map_box = map_ax.get_position()
    lower_box = left_ax.get_position()
    gap = LOWER_ROW_GAP_FRACTION * map_box.width
    panel_width = 0.5 * (map_box.width - gap)
    if panel_width <= 0:
        raise ValueError("Map-aligned lower-row panel width is not positive.")

    left_ax.set_position(
        [map_box.x0, lower_box.y0, panel_width, lower_box.height]
    )
    right_ax.set_position(
        [map_box.x1 - panel_width, lower_box.y0, panel_width, lower_box.height]
    )
    left_box = left_ax.get_position()
    right_box = right_ax.get_position()
    tolerance = 1.0e-10
    if not (
        abs(left_box.x0 - map_box.x0) <= tolerance
        and abs(right_box.x1 - map_box.x1) <= tolerance
        and abs(left_box.width - right_box.width) <= tolerance
        and abs(left_box.height - right_box.height) <= tolerance
    ):
        raise RuntimeError("Lower-row/map edge alignment invariant failed.")
    return {
        "map_left": float(map_box.x0),
        "map_right": float(map_box.x1),
        "lower_left": float(left_box.x0),
        "lower_right": float(right_box.x1),
        "panel_width": float(panel_width),
        "panel_height": float(lower_box.height),
        "gap": float(gap),
    }


def set_map_base(ax: plt.Axes, use_cartopy: bool) -> object | None:
    if use_cartopy:
        ax.set_global()
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white", edgecolor="none")
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F7F7F7", edgecolor="none")
        ax.add_feature(
            cfeature.COASTLINE.with_scale("110m"),
            linewidth=MARKER_EDGE_LW, edgecolor="#606060",
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            linewidth=0.14,
            edgecolor="#AAAAAA",
            linestyle=":",
        )
        ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.2,
            color="#CCCCCC", alpha=0.55,
        )
        if "geo" in ax.spines:
            ax.spines["geo"].set_linewidth(FRAME_LW)
            ax.spines["geo"].set_edgecolor("#222222")
        return ccrs.PlateCarree()
    ax.set_facecolor("#F7F7F7")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_box_aspect(0.50)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    format_axis(ax, grid=True)
    # The fallback is a geometry/QC surrogate for the Robinson map, not a
    # scientific longitude--latitude panel; suppress labels to match the
    # cartographic presentation used in the formal figure.
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    return None


def draw_map(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_column: str,
    norm: TwoSlopeNorm,
    label: str,
    subtitle: str,
    use_cartopy: bool,
    show_group_legend: bool,
    cmap: LinearSegmentedColormap,
    subtitle_color: str = "#111111",
) -> plt.cm.ScalarMappable:
    work = data[["pair_id", "lon", "lat", value_column]].copy()
    for column in ("lon", "lat", value_column):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["lon", "lat", value_column])
    transform = set_map_base(ax, use_cartopy)
    kwargs = {"transform": transform} if transform is not None else {}
    ax.scatter(
        work["lon"], work["lat"], c=work[value_column],
        cmap=cmap, norm=norm, s=22, marker="o",
        edgecolor="#665F59", linewidth=MARKER_EDGE_LW, alpha=0.94,
        rasterized=True, zorder=4, **kwargs,
    )
    panel_heading(
        ax, label, subtitle, subtitle_color=subtitle_color,
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    return scalar


def draw_diurnal_response(ax: plt.Axes, summary: dict[str, dict[str, np.ndarray]]) -> None:
    # The diurnal cycle is periodic: repeat the hour-0 value at hour 24 so the
    # curves close over the full 0-24 h axis.
    plot_hours = np.append(HOURS, 24)
    for group, color in (("UHI", UHI_CURVE_COLOR), ("UCI", UCI_CURVE_COLOR)):
        entry = summary[group]
        # Pale 95% CI band marks the group curve as an average across pairs.
        ax.fill_between(
            plot_hours,
            np.append(entry["ci_low"], entry["ci_low"][0]),
            np.append(entry["ci_high"], entry["ci_high"][0]),
            color=color, alpha=0.14, linewidth=0, zorder=3,
        )
        ax.plot(
            plot_hours, np.append(entry["mean"], entry["mean"][0]),
            color=color, linestyle="-",
            linewidth=DATA_LW, label=group, zorder=4,
        )
    ax.axvspan(0, 6, color=NIGHT_BACKGROUND, alpha=0.10, linewidth=0)
    ax.axvspan(6, 18, color=DAY_BACKGROUND, alpha=0.16, linewidth=0)
    ax.axvspan(18, 24, color=NIGHT_BACKGROUND, alpha=0.10, linewidth=0)
    ax.axhline(
        0, color="#333333", linewidth=REFERENCE_LW, linestyle=":",
    )
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("Local solar time (h)")
    ax.set_ylabel(r"Heatwave response, $R(t)$ (°C)")
    ax.legend(
        loc="upper right", ncol=1, frameon=False, columnspacing=0.9,
        handlelength=1.6, handletextpad=0.35,
    )
    panel_heading(ax, "b", "Mean diurnal heatwave response")
    format_axis(ax, grid=True)


def draw_daynight_scatter(ax: plt.Axes, paired: pd.DataFrame) -> None:
    """Show pair responses and label the two group-mean states explicitly."""
    values = np.concatenate(
        [paired["R_x"].to_numpy(float), paired["R_n"].to_numpy(float)]
    )
    values = values[np.isfinite(values)]
    low, high = float(np.min(values)), float(np.max(values))
    pad = 0.10 * max(high - low, 1.0)
    low, high = low - pad, high + pad

    ax.axhline(0.0, color="#8F8983", linewidth=REFERENCE_LW, zorder=1)
    ax.axvline(0.0, color="#8F8983", linewidth=REFERENCE_LW, zorder=1)
    ax.plot(
        [low, high], [low, high], color="#5F5A55", linestyle="--",
        linewidth=REFERENCE_LW, alpha=0.85, zorder=1,
    )
    ax.text(
        0.56, 0.56, r"$R_n=R_x$", transform=ax.transAxes,
        ha="center", va="bottom", rotation=45, fontsize=ANNOT_SIZE,
        color="#5F5A55",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.6),
        zorder=6,
    )
    mean_offsets = {"UHI": (14, -11), "UCI": (-14, 12)}
    mean_align = {"UHI": "left", "UCI": "right"}
    for group, color in (("UHI", UHI_CURVE_COLOR), ("UCI", UCI_CURVE_COLOR)):
        subset = paired.loc[paired["group"].eq(group)]
        ax.scatter(
            subset["R_x"], subset["R_n"], s=13, marker="o",
            color=color, alpha=0.10, edgecolors="none", rasterized=True,
            label=f"{group} (n={len(subset)})", zorder=2,
        )
        mean_x = float(subset["R_x"].mean())
        mean_n = float(subset["R_n"].mean())
        se_x = 1.96 * float(subset["R_x"].std(ddof=1) / np.sqrt(len(subset)))
        se_n = 1.96 * float(subset["R_n"].std(ddof=1) / np.sqrt(len(subset)))
        ax.errorbar(
            [mean_x], [mean_n],
            xerr=[[se_x], [se_x]], yerr=[[se_n], [se_n]],
            fmt="none", ecolor=color, elinewidth=0.5, capsize=1.5,
            capthick=0.5, zorder=4,
        )
        ax.scatter(
            [mean_x], [mean_n],
            s=7, marker="o", facecolor=color, edgecolor="none",
            zorder=6,
        )
        ax.annotate(
            f"{group} mean",
            xy=(mean_x, mean_n),
            xytext=mean_offsets[group],
            textcoords="offset points",
            ha=mean_align[group], va="center",
            fontsize=ANNOT_SIZE, fontweight="bold", color=color,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=0.5, alpha=0.9,
                mutation_scale=6, shrinkA=5, shrinkB=8,
            ),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
            zorder=5,
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    # Use the complete subplot box so panels b and c have identical physical
    # dimensions.  The identical x/y limits preserve the quantitative scale;
    # the labelled R_n=R_x line defines equality without relying on a 45° cue.
    ax.set_aspect("auto")
    ax.set_xlabel(r"Daytime response, $R_x$ (°C)")
    ax.set_ylabel(r"Night-time response, $R_n$ (°C)")
    ax.legend(
        loc="upper left", frameon=True, facecolor="white", framealpha=0.8,
        edgecolor="none", handletextpad=0.35, labelspacing=0.25,
        borderaxespad=0.35, markerscale=0.9,
    )
    ax.text(
        0.03, 0.68, "Night stronger", transform=ax.transAxes,
        ha="left", va="center", fontsize=ANNOT_SIZE, color="#5F5A55",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.5),
        zorder=6,
    )
    ax.text(
        0.97, 0.44, "Day stronger", transform=ax.transAxes,
        ha="right", va="center", fontsize=ANNOT_SIZE, color="#5F5A55",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.5),
        zorder=6,
    )
    panel_heading(ax, "c", "Day–night response space")
    format_axis(ax, grid=True)


def symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot construct a colour scale from empty values.")
    limit = max(float(np.max(np.abs(finite))), 1.0e-6)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def save_rgb_jpg(fig: plt.Figure, path: Path) -> dict[str, object]:
    temporary = path.with_name(f".{path.stem}.temporary.jpg")
    try:
        fig.savefig(
            temporary, format="jpg", dpi=FORMAL_DPI, facecolor="white",
            edgecolor="white", transparent=False,
            pil_kwargs={"quality": 96, "subsampling": 0, "optimize": True},
        )
        with Image.open(temporary) as image:
            expected_width = round(FIGURE_WIDTH_MM * MM_TO_IN * FORMAL_DPI)
            expected_height = round(FIGURE_HEIGHT_MM * MM_TO_IN * FORMAL_DPI)
            dpi = image.info.get("dpi", (0, 0))
            if image.mode != "RGB":
                raise RuntimeError(f"Expected RGB JPG, received {image.mode}.")
            if abs(image.width - expected_width) > 2 or abs(image.height - expected_height) > 2:
                raise RuntimeError(
                    f"Fixed canvas mismatch: {(image.width, image.height)} versus "
                    f"{(expected_width, expected_height)}."
                )
            if min(dpi) < 590:
                raise RuntimeError(f"DPI metadata below 600: {dpi}")
            info = {
                "mode": image.mode,
                "pixels": [image.width, image.height],
                "dpi": [float(dpi[0]), float(dpi[1])],
            }
        os.replace(temporary, path)
        return info
    finally:
        temporary.unlink(missing_ok=True)


def save_preview(jpg_path: Path, preview_path: Path) -> None:
    with Image.open(jpg_path) as image:
        preview = image.copy()
        preview.thumbnail((1800, 1400), Image.Resampling.LANCZOS)
        preview.save(preview_path, format="WEBP", quality=82, method=6)


def create_output_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"run_{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = root / f"run_{stamp}_v{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def render(
    metrics: pd.DataFrame,
    output_dir: Path,
    source_path: Path | None,
    use_cartopy: bool,
    enforce_formal_counts: bool,
) -> tuple[Path, Path]:
    apply_nature_style()
    cohort, paired, audit = freeze_matched_cohort(metrics, enforce_formal_counts)
    diurnal = summarize_diurnal_response(cohort)

    asym_norm = symmetric_norm(paired["asymmetry"].to_numpy(float))

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
        dpi=FORMAL_DPI,
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2, 2,
        height_ratios=[1.20, 0.80],
        width_ratios=[1.0, 1.0],
    )
    projection = ccrs.Robinson(central_longitude=0) if use_cartopy else None
    ax_a = fig.add_subplot(grid[0, :], projection=projection)
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(
        left=0.080, right=0.965, bottom=0.070, top=0.925,
        wspace=0.24, hspace=0.32,
    )

    asym_scalar = draw_map(
        ax_a, paired, "asymmetry", asym_norm, "a",
        "Global day–night response asymmetry", use_cartopy, False,
        cmap=ASYMMETRY_CMAP,
    )
    draw_diurnal_response(ax_b, diurnal)
    draw_daynight_scatter(ax_c, paired)
    layout_audit = align_lower_row_to_map(fig, ax_a, ax_b, ax_c)

    # Place the asymmetry colour bar in the inter-row margin rather than
    # inside the map.  Its position follows the fixed Panel-a axis box, so it
    # remains separated from both the map and the Panel-c subtitle.
    box_a = ax_a.get_position()
    cax_a = fig.add_axes(
        [
            box_a.x0 + 0.35 * box_a.width,
            box_a.y0 - 0.028,
            0.30 * box_a.width,
            0.016,
        ]
    )
    cb_a = fig.colorbar(asym_scalar, cax=cax_a, orientation="horizontal", extend="both")
    cb_a.locator = MaxNLocator(nbins=5)
    cb_a.update_ticks()
    cb_a.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE, length=2.0, width=0.6, pad=1.0,
    )
    cb_a.outline.set_linewidth(0.6)
    # The unit label sits to the left of the bar so it never competes with
    # the tick numbers below.
    cax_a.text(
        -0.12, 0.5, r"$R_n-R_x$ (°C)",
        transform=cax_a.transAxes, ha="right", va="center",
        fontsize=COLORBAR_LABEL_SIZE, clip_on=False,
    )
    cax_a.text(
        -0.04, -1.8, "Stronger daytime response",
        transform=cax_a.transAxes, ha="right", va="center",
        fontsize=ANNOT_SIZE, color="#5F5A55", clip_on=False,
    )
    cax_a.text(
        1.04, -1.8, "Stronger night-time response",
        transform=cax_a.transAxes, ha="left", va="center",
        fontsize=ANNOT_SIZE, color="#5F5A55", clip_on=False,
    )

    main_path = output_dir / MAIN_FILENAME
    preview_path = output_dir / PREVIEW_FILENAME
    image_info = save_rgb_jpg(fig, main_path)
    plt.close(fig)
    save_preview(main_path, preview_path)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "formal_source": str(source_path.resolve()) if source_path else "SELF_TEST",
        "formal_source_sha256": sha256_file(source_path.resolve()) if source_path else None,
        "cohort": audit,
        "main_image": MAIN_FILENAME,
        "main_image_sha256": sha256_file(main_path),
        "preview": PREVIEW_FILENAME,
        "preview_sha256": sha256_file(preview_path),
        "image_info": image_info,
        "fixed_canvas_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "layout_alignment": layout_audit,
        "lower_row_aligned_to_map_edges": True,
        "lower_panels_equal_size": True,
        "bbox_inches_tight_used": False,
        "plot_only": True,
        "source_definition": "legacy Figure 1 all_pair_period_metrics.csv only",
        "panel_b_definition": "group mean of pair-matched HW minus NHW diurnal contrasts",
        "panel_b_shading": "pair-level 95% CI (1.96 x SE across pairs)",
        "panel_c_error_bars": "pair-level 95% CI (1.96 x SE across pairs)",
        "panel_c_definition": "all 317 pair-level R_x versus R_n responses",
        "panel_c_large_markers": "UHI and UCI arithmetic group means",
        "regression_fitted_in_plot": False,
        "bootstrap_run_in_plot": False,
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    write_json(
        output_dir / "_SUCCESS.json",
        {
            "status": "SUCCESS",
            "script_version": SCRIPT_VERSION,
            "main_image": MAIN_FILENAME,
            "main_image_sha256": sha256_file(main_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return main_path, preview_path


def synthetic_metrics(seed: int = 20260818) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(36):
        pair_id = f"P{index:03d}"
        group = "UHI" if index < 23 else "UCI"
        lat = rng.uniform(-55, 75)
        lon = rng.uniform(-175, 175)
        base_mean = rng.normal(0.8 if group == "UHI" else -0.5, 0.35)
        base_amp = rng.normal(1.0, 0.35)
        for period in ("annual", "non_heatwave", "heatwave"):
            is_hw = period == "heatwave"
            mean = base_mean + (rng.normal(0.25, 0.15) if is_hw else 0.0)
            amp = base_amp + (rng.normal(-0.18, 0.12) if is_hw else 0.0)
            rural_curve = 23 + 5 * np.cos(2 * np.pi * (HOURS - 15) / 24)
            contrast = mean + amp * np.cos(2 * np.pi * (HOURS - 15) / 24)
            urban_curve = rural_curve + contrast
            dtx = float(np.max(urban_curve) - np.max(rural_curve))
            dtn = float(np.min(urban_curve) - np.min(rural_curve))
            row: dict[str, object] = {
                "pair_id": pair_id,
                "period": period,
                "group": group,
                "hw_method": "percentile",
                "lat_urban": lat,
                "lon_urban": lon,
                "lat_rural": lat + 0.1,
                "lon_rural": lon + 0.2,
                "urban_Tmax_fft": float(np.max(urban_curve)),
                "rural_Tmax_fft": float(np.max(rural_curve)),
                "urban_Tmin_fft": float(np.min(urban_curve)),
                "rural_Tmin_fft": float(np.min(rural_curve)),
                "dTmean": mean,
                "dAmp1": amp,
                "dTx": dtx,
                "dTn": dtn,
            }
            for hour in range(24):
                row[f"urban_diurnal_h{hour:02d}"] = urban_curve[hour]
                row[f"rural_diurnal_h{hour:02d}"] = rural_curve[hour]
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--plain-map", action="store_true",
        help="Use a plain lon-lat map instead of Cartopy (QA fallback only).",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Render deterministic synthetic inputs without reading formal data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = create_output_dir(args.output_root)
    if args.self_test:
        metrics = synthetic_metrics()
        source = None
        use_cartopy = False
        enforce_counts = False
    else:
        source = args.metrics_csv.expanduser().resolve()
        source_hash_before = sha256_file(source)
        metrics = read_formal_metrics(source)
        use_cartopy = HAS_CARTOPY and not args.plain_map
        if not HAS_CARTOPY and not args.plain_map:
            raise RuntimeError(
                "Cartopy is unavailable. Install it for the formal map or use "
                "--plain-map only for diagnostic QA."
            )
        enforce_counts = True
    main_path, preview_path = render(
        metrics, output_dir, source, use_cartopy, enforce_counts
    )
    if not args.self_test and source_hash_before != sha256_file(source):
        raise RuntimeError("READ_ONLY_INPUT_MODIFIED_DURING_PLOTTING")
    print(f"SUCCESS: {main_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
