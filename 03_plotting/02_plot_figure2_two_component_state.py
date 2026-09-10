#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot-only Figure 2: dominant two-component state representation.

The formal figure contains three panels:

a  Mean-amplitude geometry with three labelled archetype anchors.
b  Four UHI/UCI group-mean NHW/HW states and their enlarged transitions.
c  Frozen UHI/UCI response summary shown as a horizontal dot-whisker plot.

The detailed former panels a and b are retained together as an SI candidate.

No regression line or confidence interval is estimated here.  Panel c requires
an upstream formal summary table with columns ``annual_group``, ``component``,
``estimate``, ``ci_low``, ``ci_high`` and ``n_pairs``.
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

_MPL_CACHE = Path(tempfile.gettempdir()) / "figure2_two_component_mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib import patheffects
from PIL import Image


SCRIPT_VERSION = "plot_figure2_two_component_state"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_CSV = (
    PROJECT_ROOT
    / "outputs/analysis/main_multiyear/robustness_percentile/"
    "all_pair_period_metrics.csv"
)
DEFAULT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "source_data/"
    "Supplement_pair_level_HW_NHW_distribution_summary.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/figures/figure2_two_component_state"
)

MAIN_FILENAME = "figure2_two_component_state.jpg"
PREVIEW_FILENAME = "figure2_two_component_state_preview.webp"
SI_FILENAME = "figure2_detailed_geometry_pair_states_si.jpg"
SI_PREVIEW_FILENAME = "figure2_detailed_geometry_pair_states_si_preview.webp"
GROUP_MEANS_FILENAME = "figure2_panel_b_group_state_means.csv"
MANIFEST_FILENAME = "figure2_two_component_plot_manifest.json"

MM_TO_IN = 1.0 / 25.4
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 180.0
FORMAL_DPI = 600
EXPECTED_TOTAL = 317
EXPECTED_COUNTS = {"UHI": 200, "UCI": 117}

COLOR_UHI = "#BB3445"
COLOR_UCI = "#3C79B6"
COLOR_ZERO = "#4A4540"
COLOR_GRID = "#D8D2CB"
COLOR_NHW = "#FFFFFF"
STATE_CMAP = LinearSegmentedColormap.from_list(
    "fig2_state_projection", [COLOR_UCI, "#F5F1EA", COLOR_UHI], N=256
)

PERIOD_MAP = {
    "annual": "annual",
    "non_heatwave": "NHW",
    "nhw": "NHW",
    "heatwave": "HW",
    "hw": "HW",
}
COMPONENT_ORDER = ("R_mean", "R_Amp", "R_x", "R_n")
COMPONENT_LABELS = {
    "R_mean": r"$R_{\mathrm{mean}}$",
    "R_Amp": r"$R_{\mathrm{amp}}$",
    "R_x": r"$R_x$",
    "R_n": r"$R_n$",
}


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


def require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def norm_period(value: object) -> str:
    key = str(value).strip().lower()
    return PERIOD_MAP.get(key, str(value).strip())


def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 6.3,
            "axes.titlesize": 7.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_heading(
    ax: plt.Axes,
    label: str,
    subtitle: str,
    label_x: float = -0.04,
    label_y: float = 1.14,
) -> None:
    # The panel letter sits immediately to the left of the centred subtitle,
    # sharing its baseline (different title locations keep both titles).
    ax.set_title(
        subtitle, loc="center", pad=8.0, color="black", linespacing=1.02,
        fontsize=7.8, fontweight="bold",
    )
    panel_title = ax.set_title(
        label,
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        pad=8.0,
    )
    panel_title.set_x(label_x)
    panel_title.set_ha("right")
    panel_title.set_clip_on(False)


def format_axis(ax: plt.Axes, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    if grid:
        ax.grid(True, color=COLOR_GRID, linestyle=(0, (1.3, 2.2)), linewidth=0.45)
        ax.set_axisbelow(True)


def read_inputs(metrics_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = metrics_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Formal state source not found: {metrics_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Formal Figure 2 group summary is required; plotting-stage bootstrap "
            f"is forbidden. Missing: {summary_path}"
        )
    metrics = pd.read_csv(metrics_path, low_memory=False)
    summary = pd.read_csv(summary_path, low_memory=False)
    require_columns(
        metrics,
        ["pair_id", "period", "group", "dTmean", "dAmp1", "dTx", "dTn"],
        str(metrics_path),
    )
    canonical_summary = {
        "annual_group", "component", "estimate", "ci_low", "ci_high", "n_pairs"
    }
    frozen_output_summary = {
        "group", "metric", "mean", "bootstrap_ci_low", "bootstrap_ci_high", "n"
    }
    if canonical_summary.issubset(summary.columns):
        summary = summary[list(canonical_summary)].copy()
    elif frozen_output_summary.issubset(summary.columns):
        component_map = {
            "Rmean": "R_mean", "Ramp": "R_Amp", "Rx": "R_x", "Rn": "R_n"
        }
        summary = (
            summary.loc[summary["metric"].astype(str).isin(component_map)]
            .assign(component=lambda frame: frame["metric"].astype(str).map(component_map))
            .rename(
                columns={
                    "group": "annual_group", "mean": "estimate",
                    "bootstrap_ci_low": "ci_low",
                    "bootstrap_ci_high": "ci_high", "n": "n_pairs",
                }
            )[
                ["annual_group", "component", "estimate", "ci_low", "ci_high", "n_pairs"]
            ]
            .copy()
        )
    else:
        raise ValueError(
            f"Frozen Figure 2 summary has an unsupported schema: {summary_path}"
        )
    return metrics, summary


def freeze_state_tables(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    enforce_formal_counts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    data = metrics.copy()
    if "hw_method" in data.columns:
        data = data.loc[
            data["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()
    data["pair_id"] = data["pair_id"].astype(str)
    data["period_norm"] = data["period"].map(norm_period)
    if data.duplicated(["pair_id", "period_norm"]).any():
        raise ValueError("Duplicate pair-period rows in the formal state source.")

    annual = data.loc[data["period_norm"].eq("annual"), ["pair_id", "group"]].copy()
    annual["annual_group"] = annual["group"].astype(str).str.upper().str.strip()
    annual = annual.loc[annual["annual_group"].isin(["UHI", "UCI"]), ["pair_id", "annual_group"]]
    if annual.groupby("pair_id")["annual_group"].nunique().gt(1).any():
        raise ValueError("Conflicting annual UHI/UCI labels.")
    annual = annual.drop_duplicates("pair_id")

    for column in ("dTmean", "dAmp1", "dTx", "dTn"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    needed = ["pair_id", "dTmean", "dAmp1", "dTx", "dTn"]
    nhw = data.loc[data["period_norm"].eq("NHW"), needed].copy()
    hw = data.loc[data["period_norm"].eq("HW"), needed].copy()
    nhw = nhw.rename(columns={c: f"{c}_NHW" for c in needed if c != "pair_id"})
    hw = hw.rename(columns={c: f"{c}_HW" for c in needed if c != "pair_id"})
    paired = nhw.merge(hw, on="pair_id", validate="one_to_one")
    complete = [f"{name}_{period}" for name in ("dTmean", "dAmp1", "dTx", "dTn") for period in ("NHW", "HW")]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=complete)
    paired = paired.merge(annual, on="pair_id", validate="one_to_one")
    counts = paired["annual_group"].value_counts().to_dict()
    if enforce_formal_counts:
        if len(paired) != EXPECTED_TOTAL or counts != EXPECTED_COUNTS:
            raise ValueError(
                f"Figure 2 cohort mismatch: total={len(paired)}, counts={counts}; "
                f"expected total={EXPECTED_TOTAL}, counts={EXPECTED_COUNTS}."
            )

    states = pd.concat(
        [
            paired[["pair_id", "annual_group", "dTmean_NHW", "dAmp1_NHW", "dTx_NHW"]]
            .rename(columns={"dTmean_NHW": "dTmean", "dAmp1_NHW": "dAmp1", "dTx_NHW": "dTx"})
            .assign(period="NHW"),
            paired[["pair_id", "annual_group", "dTmean_HW", "dAmp1_HW", "dTx_HW"]]
            .rename(columns={"dTmean_HW": "dTmean", "dAmp1_HW": "dAmp1", "dTx_HW": "dTx"})
            .assign(period="HW"),
        ],
        ignore_index=True,
    )

    frozen = summary.copy()
    frozen["annual_group"] = frozen["annual_group"].astype(str).str.upper().str.strip()
    frozen["component"] = frozen["component"].astype(str).str.strip()
    frozen = frozen.loc[
        frozen["annual_group"].isin(["UHI", "UCI"])
        & frozen["component"].isin(COMPONENT_ORDER)
    ].copy()
    duplicates = frozen.duplicated(["annual_group", "component"])
    if duplicates.any():
        raise ValueError("Duplicate group-component rows in the formal Figure 2 summary.")
    expected_keys = {(g, c) for g in ("UHI", "UCI") for c in COMPONENT_ORDER}
    observed_keys = set(zip(frozen["annual_group"], frozen["component"]))
    if observed_keys != expected_keys:
        raise ValueError(
            f"Formal Figure 2 summary key mismatch: missing={sorted(expected_keys-observed_keys)}, "
            f"unexpected={sorted(observed_keys-expected_keys)}"
        )
    expected_n = {"UHI": counts.get("UHI", 0), "UCI": counts.get("UCI", 0)}
    for row in frozen.itertuples(index=False):
        if int(row.n_pairs) != int(expected_n[str(row.annual_group)]):
            raise ValueError(
                f"Summary n mismatch for {row.annual_group}/{row.component}: "
                f"{row.n_pairs} versus {expected_n[str(row.annual_group)]}."
            )
    audit = {
        "n_pairs": int(len(paired)),
        "n_uhi": int(counts.get("UHI", 0)),
        "n_uci": int(counts.get("UCI", 0)),
    }
    return states, frozen, audit


def limits_for_states(states: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x = states["dTmean"].to_numpy(float)
    y = states["dAmp1"].to_numpy(float)
    # Use the complete finite range: valid extremes must remain visible.
    x_low, x_high = float(np.nanmin(x)), float(np.nanmax(x))
    y_low, y_high = float(np.nanmin(y)), float(np.nanmax(y))
    x_low = min(float(x_low), -0.5)
    x_high = max(float(x_high), 3.4)
    y_low = min(float(y_low), -2.6)
    y_high = max(float(y_high), 1.6)
    x_pad = 0.06 * (x_high - x_low)
    y_pad = 0.06 * (y_high - y_low)
    return (x_low - x_pad, x_high + x_pad), (y_low - y_pad, y_high + y_pad)


def limits_for_group_states(
    group_states: pd.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Zoom around the four group means while retaining zero as a reference."""
    def padded(values: np.ndarray, minimum_span: float) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("Cannot set group-state limits from empty values.")
        low = min(float(finite.min()), 0.0)
        high = max(float(finite.max()), 0.0)
        centre = 0.5 * (low + high)
        span = max(high - low, minimum_span)
        low = centre - 0.5 * span
        high = centre + 0.5 * span
        pad = 0.18 * span
        return low - pad, high + pad

    return (
        padded(group_states["dTmean"].to_numpy(float), 1.35),
        padded(group_states["dAmp1"].to_numpy(float), 1.80),
    )


def add_cycle_inset(
    parent: plt.Axes,
    bounds: tuple[float, float, float, float],
    delta_mean: float,
    delta_amp: float,
    color: str,
    label: str,
    description: str | None = None,
) -> None:
    """Draw the original full diurnal inset without changing its data logic."""
    inset = parent.inset_axes(bounds, zorder=6)
    hours = np.linspace(0, 24, 1000)
    omega = 2 * np.pi / 24
    mean_ref, amp1_ref, amp2_ref = 12.0, 7.0, 1.0
    phase1_ref, phase2_ref = 3.8, 0.5

    def reconstruct(mean: float, amp1: float) -> np.ndarray:
        return (
            mean
            + amp1 * np.cos(omega * hours - phase1_ref)
            + amp2_ref * np.cos(2 * omega * hours - phase2_ref)
        )

    rural = reconstruct(mean_ref, amp1_ref)
    urban = reconstruct(mean_ref + delta_mean, amp1_ref + delta_amp)
    inset.fill_between(
        hours, rural, urban, where=urban >= rural,
        color="#E99696", alpha=0.34, interpolate=True, linewidth=0, zorder=1,
    )
    inset.fill_between(
        hours, rural, urban, where=urban < rural,
        color="#8EC1E0", alpha=0.34, interpolate=True, linewidth=0, zorder=1,
    )
    inset.plot(hours, rural, color="#666666", linestyle=(0, (3, 3)), linewidth=0.85, zorder=2)
    inset.plot(hours, urban, color=color, linewidth=1.20, zorder=3)
    inset.set_xlim(0, 24)
    inset.set_ylim(float(np.nanmin([rural, urban])) - 1.0, float(np.nanmax([rural, urban])) + 1.0)
    inset.set_xticks([])
    inset.set_yticks([])
    label_effect = [patheffects.withStroke(linewidth=1.6, foreground="white", alpha=0.95)]
    anchor_hour = {"A": 18.8, "B": 18.2, "C": 17.8}[label]
    rural_anchor = float(np.interp(anchor_hour, hours, rural))
    urban_anchor = float(np.interp(anchor_hour, hours, urban))
    urban_y = 0.93 if urban_anchor >= rural_anchor else 0.07
    rural_y = 0.07 if urban_anchor >= rural_anchor else 0.93
    inset.annotate(
        r"$T_{a,\mathrm{u}}$", xy=(anchor_hour, urban_anchor), xycoords="data",
        xytext=(0.80, urban_y), textcoords="axes fraction",
        fontsize=5.9, fontweight="bold", color=color, ha="left", va="center",
        path_effects=label_effect,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55},
        clip_on=True, zorder=5,
    )
    inset.annotate(
        r"$T_{a,\mathrm{r}}$", xy=(anchor_hour, rural_anchor), xycoords="data",
        xytext=(0.80, rural_y), textcoords="axes fraction",
        fontsize=5.9, fontweight="bold", color="#666666", ha="left", va="center",
        path_effects=label_effect,
        arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.55,
                    "linestyle": (0, (3, 3))},
        clip_on=True, zorder=5,
    )
    if description:
        inset.text(
            0.035, 0.965, description, transform=inset.transAxes,
            ha="left", va="top", fontsize=5.1, fontweight="bold",
            color=color, linespacing=0.94,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.45),
            zorder=7,
        )
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.91)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#8A847E")


def draw_geometry_panel(
    ax: plt.Axes,
    states: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    colorbar_ax: plt.Axes,
    *,
    show_cloud: bool = True,
    show_mechanism_arrows: bool = True,
    inset_descriptions: bool = False,
    external_inset_hosts: Mapping[str, plt.Axes] | None = None,
    show_panel_heading: bool = True,
    panel_subtitle: str = "Mean–amplitude\nresponse geometry",
) -> None:
    omega = 2 * np.pi / 24
    hours = np.linspace(0, 24, 1000)
    rural_reference = (
        12.0
        + 7.0 * np.cos(omega * hours - 3.8)
        + 1.0 * np.cos(2 * omega * hours - 0.5)
    )
    time_tx = float(hours[int(np.nanargmax(rural_reference))])
    kx = float(np.cos(omega * time_tx - 3.8))

    xx = np.linspace(xlim[0], xlim[1], 260)
    yy = np.linspace(ylim[0], ylim[1], 260)
    grid_x, grid_y = np.meshgrid(xx, yy)
    projection = grid_x + kx * grid_y
    finite_dtx = states["dTx"].to_numpy(float)
    limit = max(float(np.nanmax(np.abs(finite_dtx))), 1.0e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    # Explicit symmetric contour levels must match the symmetric norm.  Using
    # an integer number of automatic levels left the unused negative tail of
    # the colour bar unfilled when the plotted grid had an asymmetric range.
    contour_levels = np.linspace(-limit, limit, 31)
    contours = ax.contourf(
        grid_x, grid_y, np.clip(projection, -limit, limit), levels=contour_levels,
        cmap=STATE_CMAP, norm=norm, alpha=0.50, zorder=0,
    )
    ax.axhline(0, color="#CFCFCF", linewidth=0.65, zorder=1)
    ax.axvline(0, color="#CFCFCF", linewidth=0.65, zorder=1)
    nhw = states.loc[states["period"].eq("NHW")]
    if show_cloud:
        ax.scatter(
            nhw["dTmean"], nhw["dAmp1"], s=10, facecolors="none",
            edgecolors="#77716C", linewidths=0.45, alpha=0.35,
            rasterized=True, zorder=2,
        )

    # Preserve the former panel-a line gap and its full two-line definition.
    # Put the boundary label in the lower-right low-density portion of the
    # complete (untruncated) state range instead of over the main point cloud.
    x_label = min(4.2, xlim[1] - 0.45, -0.78 * ylim[0] * kx)
    x_label = max(x_label, xlim[0] + 0.36 * (xlim[1] - xlim[0]))
    gap = 0.62
    for start, stop in ((xlim[0], x_label - gap), (x_label + gap, xlim[1])):
        if stop > start:
            xline = np.linspace(start, stop, 200)
            ax.plot(
                xline, -xline / kx, color=COLOR_ZERO, linestyle=(0, (6, 4)),
                linewidth=1.15, zorder=3,
            )
    y_label = -x_label / kx
    boundary_label = (
        r"$\Delta T_x=0$"
        if inset_descriptions
        else r"$-\Delta Amp=\Delta T_{mean}$" + "\n" + r"$\Delta T_x\approx0$"
    )
    ax.text(
        x_label, y_label,
        boundary_label,
        ha="center", va="center", rotation=-34, rotation_mode="anchor",
        fontsize=6.4, fontweight="bold",
        color="#222222",
        path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white", alpha=0.95)],
        zorder=10,
    )

    # The main layout places all three explanatory archetypes in a
    # dedicated column beside the state plane.  The legacy in-axis placement
    # remains available for the unchanged SI composition.
    if external_inset_hosts is None:
        if inset_descriptions:
            bbox_a = (0.69, 0.60, 0.28, 0.31)
            bbox_b = (0.69, 0.055, 0.28, 0.31)
            bbox_c = (0.030, 0.050, 0.28, 0.32)
        else:
            bbox_a = (0.68, 0.65, 0.27, 0.20)
            bbox_b = (0.70, 0.13, 0.27, 0.20)
            bbox_c = (0.040, 0.050, 0.26, 0.20)
    inset_labels = {
        "A": (
            "A  Amplified diurnal UHI\n"
            + r"$-\Delta Amp<0<\Delta T_{mean}$"
            + "\n"
            + r"$\Delta T_x>\Delta T_n$"
        ),
        "B": (
            "B  Damped daytime UHI\n"
            + r"$0<-\Delta Amp<\Delta T_{mean}$"
            + "\n"
            + r"$\Delta T_n>\Delta T_x>0$"
        ),
        "C": (
            "C  Daytime UCI / night-time UHI\n"
            + r"$0<\Delta T_{mean}<-\Delta Amp$"
            + "\n"
            + r"$\Delta T_x<0,\ \Delta T_n>0$"
        ),
    }
    if external_inset_hosts is None:
        add_cycle_inset(
            ax, bbox_a, 3.0, 1.0, COLOR_UHI, "A",
            inset_labels["A"] if inset_descriptions else None,
        )
        add_cycle_inset(
            ax, bbox_b, 3.0, -2.0, "#C56A22", "B",
            inset_labels["B"] if inset_descriptions else None,
        )
        add_cycle_inset(
            ax, bbox_c, 0.9, -2.2, COLOR_UCI, "C",
            inset_labels["C"] if inset_descriptions else None,
        )
    else:
        missing_hosts = sorted(set(("A", "B", "C")).difference(external_inset_hosts))
        if missing_hosts:
            raise ValueError(f"Missing external archetype axes: {missing_hosts}")
        archetype_specs = {
            "A": (3.0, 1.0, COLOR_UHI),
            "B": (3.0, -2.0, "#C56A22"),
            "C": (0.9, -2.2, COLOR_UCI),
        }
        # Two-line captions sit above each curve panel, outside the inset, so
        # the diurnal curves stay fully unobstructed.
        external_labels = {
            "A": (
                "A  Amplified diurnal UHI",
                r"$-\Delta Amp<0<\Delta T_{mean};\ \Delta T_x>\Delta T_n$",
            ),
            "B": (
                "B  Damped daytime UHI",
                r"$0<-\Delta Amp<\Delta T_{mean};\ \Delta T_n>\Delta T_x>0$",
            ),
            "C": (
                "C  Daytime UCI / night-time UHI",
                r"$0<\Delta T_{mean}<-\Delta Amp;\ \Delta T_x<0,\ \Delta T_n>0$",
            ),
        }
        for key in ("A", "B", "C"):
            host = external_inset_hosts[key]
            host.set_axis_off()
            delta_mean, delta_amp, color = archetype_specs[key]
            host.text(
                0.02, 0.985, "\n".join(external_labels[key]),
                transform=host.transAxes, ha="left", va="top",
                fontsize=5.4, fontweight="bold", color=color, linespacing=1.3,
                clip_on=False, zorder=8,
            )
            add_cycle_inset(
                host, (0.005, 0.015, 0.990, 0.67),
                delta_mean, delta_amp, color, key, None,
            )

    halo = [patheffects.withStroke(linewidth=2.0, foreground="white", alpha=0.95)]

    def add_inset_letter(
        bounds: tuple[float, float, float, float],
        letter: str,
        color: str,
    ) -> None:
        left, bottom, width, height = bounds
        top = bottom + height
        ax.text(
            left + 0.015, top - 0.015, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.6, fontweight="bold",
            color=color, path_effects=halo, zorder=10,
        )

    if external_inset_hosts is None and not inset_descriptions:
        add_inset_letter(bbox_a, "A", "#7F0000")
        add_inset_letter(bbox_b, "B", "#B35806")
        add_inset_letter(bbox_c, "C", "#08306B")

    # Preserve the three conceptual/observed anchors and their inset connectors.
    point_a = (3.0, 1.0)
    point_b = (3.0, -2.0)
    c_candidates = nhw.loc[
        nhw["dTmean"].gt(0) & nhw["dAmp1"].lt(0) & nhw["dTx"].lt(0)
    ].copy()
    if c_candidates.empty:
        c_candidates = nhw.copy()
    c_score = (
        ((c_candidates["dTmean"] - 0.9) / max(xlim[1] - xlim[0], 1e-9)) ** 2
        + ((c_candidates["dAmp1"] + 2.2) / max(ylim[1] - ylim[0], 1e-9)) ** 2
    )
    c_row = c_candidates.loc[c_score.idxmin()]
    point_c = (float(c_row["dTmean"]), float(c_row["dAmp1"]))
    for point, color, size in (
        (point_a, COLOR_UHI, 32),
        (point_b, "#F16913", 30),
        (point_c, COLOR_UCI, 32),
    ):
        ax.scatter(*point, s=size, color=color, edgecolor="white", linewidth=0.65, zorder=8)

    if external_inset_hosts is None:
        for point, start, curvature in (
            (point_a, (bbox_a[0], bbox_a[1] + 0.52 * bbox_a[3]), 0.08),
            (point_b, (bbox_b[0], bbox_b[1] + 0.52 * bbox_b[3]), -0.12),
            (point_c, (bbox_c[0] + bbox_c[2], bbox_c[1] + 0.58 * bbox_c[3]), 0.08),
        ):
            ax.annotate(
                "", xy=point, xycoords="data", xytext=start, textcoords="axes fraction",
                arrowprops={
                    "arrowstyle": "-", "color": "#77716C", "lw": 0.65,
                    "connectionstyle": f"arc3,rad={curvature}",
                },
                zorder=7,
            )
    else:
        # Cross-axis connectors make the archetype column visibly subordinate
        # to, and derived from, the state plane without occupying data space.
        for key, point in (("A", point_a), ("B", point_b), ("C", point_c)):
            connector = ConnectionPatch(
                xyA=point,
                coordsA=ax.transData,
                xyB=(0.0, 0.50),
                coordsB=external_inset_hosts[key].transAxes,
                arrowstyle="-",
                linewidth=0.60,
                color="#77716C",
                alpha=0.90,
                clip_on=False,
                zorder=5,
            )
            ax.figure.add_artist(connector)

    if show_mechanism_arrows:
        arrow_color = "#173F73"
        arrow_style = {
            "arrowstyle": "->", "lw": 1.15, "color": arrow_color,
            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8,
        }
        center = (0.41, 0.49)
        for endpoint in ((0.41, 0.35), (0.57, 0.49), (0.57, 0.33)):
            ax.annotate(
                "", xy=endpoint, xytext=center, xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=arrow_style, zorder=8,
            )
        ax.text(
            0.385, 0.405, r"$-\Delta Amp$", transform=ax.transAxes,
            ha="right", va="center", fontsize=6.5, fontweight="bold",
            color=arrow_color, path_effects=halo, zorder=9,
        )
        ax.text(
            0.49, 0.525, r"$+\Delta T_{mean}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            color=arrow_color, path_effects=halo, zorder=9,
        )
        ax.text(
            0.545, 0.405, "Combined\nforcing", transform=ax.transAxes,
            ha="left", va="center", fontsize=6.2, fontweight="bold",
            color=arrow_color, path_effects=halo, zorder=9,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_autoscale_on(False)
    ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
    ax.set_ylabel(r"$\Delta Amp$ (°C)")
    colorbar = ax.figure.colorbar(
        contours, cax=colorbar_ax, orientation="horizontal"
    )
    # Five symmetric labels preserve the formal colour range without the
    # overlap produced by the former dense automatic locator.
    colorbar_ticks = np.linspace(-limit, limit, 5)

    def format_colorbar_tick(value: float) -> str:
        if abs(value) < 5.0e-8:
            return "0"
        magnitude = abs(value)
        if abs(magnitude - round(magnitude)) < 0.05:
            label = f"{magnitude:.0f}"
        else:
            label = f"{magnitude:.1f}"
        return f"−{label}" if value < 0 else label

    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([format_colorbar_tick(value) for value in colorbar_ticks])
    colorbar.set_label(r"Daytime $\Delta T_x$ (°C)", fontsize=7.3, labelpad=4.0)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(
        axis="x", labelsize=6.5, width=0.65, length=2.5, pad=1.0
    )
    if show_panel_heading:
        panel_heading(
            ax, "a", panel_subtitle,
            label_x=-0.035 if inset_descriptions else -0.14,
            label_y=1.11 if inset_descriptions else 1.20,
        )
    format_axis(ax, grid=False)


def draw_geometry_notes(ax: plt.Axes) -> None:
    """Keep all former A/B/C definitions outside the square data region."""
    ax.set_axis_off()
    entries = (
        (
            0.48, "#7F0000", "A  Amplified diurnal UHI",
            r"$-\Delta Amp<0<\Delta T_{mean};\quad \Delta T_x>\Delta T_n$",
        ),
        (
            0.25, "#B35806", "B  Damped daytime UHI",
            r"$0<-\Delta Amp<\Delta T_{mean};\quad \Delta T_n>\Delta T_x>0$",
        ),
        (
            0.02, "#08306B", "C  Daytime UCI",
            r"$0<\Delta T_{mean}<-\Delta Amp;\quad \Delta T_x<0,\ \Delta T_n>0$",
        ),
    )
    for y, color, title, formula in entries:
        ax.text(
            0.02, y + 0.085, title, transform=ax.transAxes,
            ha="left", va="center", fontsize=6.4, fontweight="bold",
            color=color, clip_on=False,
        )
        ax.text(
            0.02, y - 0.015, formula, transform=ax.transAxes,
            ha="left", va="center", fontsize=5.7, fontweight="bold",
            color=color, clip_on=False,
        )


def draw_combined_states(
    ax: plt.Axes,
    states: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    audit: Mapping[str, int],
) -> None:
    # Larger hollow NHW rings remain visible around the smaller filled HW
    # triangles even when the two observations occupy nearly the same state.
    for period, marker, filled, size, linewidth, zorder in (
        ("NHW", "o", False, 30, 0.85, 3),
        ("HW", "^", True, 19, 0.38, 4),
    ):
        for group, color in (("UHI", COLOR_UHI), ("UCI", COLOR_UCI)):
            subset = states.loc[
                states["period"].eq(period) & states["annual_group"].eq(group)
            ]
            ax.scatter(
                subset["dTmean"], subset["dAmp1"], s=size, marker=marker,
                facecolor=color if filled else COLOR_NHW,
                edgecolor="#333333" if filled else color,
                linewidth=linewidth, alpha=0.72,
                rasterized=True, zorder=zorder,
            )
    line = np.linspace(xlim[0], xlim[1], 300)
    ax.plot(line, -line, color=COLOR_ZERO, linestyle="--", linewidth=0.9)
    ax.axhline(0, color="#A9A39D", linewidth=0.65)
    ax.axvline(0, color="#A9A39D", linewidth=0.65)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
    ax.set_ylabel(r"$\Delta Amp$ (°C)")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=COLOR_UHI, markeredgewidth=0.85,
               markersize=5.1, label="NHW UHI"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor=COLOR_UHI,
               markeredgecolor="#333333", markeredgewidth=0.38,
               markersize=4.7, label="HW UHI"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=COLOR_UCI, markeredgewidth=0.85,
               markersize=5.1, label="NHW UCI"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor=COLOR_UCI,
               markeredgecolor="#333333", markeredgewidth=0.38,
               markersize=4.7, label="HW UCI"),
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=2, frameon=False,
        title=f"n={audit['n_pairs']} (UHI {audit['n_uhi']}; UCI {audit['n_uci']})",
        title_fontsize=6.8, columnspacing=0.85, handletextpad=0.35,
        labelspacing=0.30, borderaxespad=0.0,
    )
    panel_heading(ax, "b", "NHW and HW states")
    format_axis(ax, grid=True)


def summarize_group_states(states: pd.DataFrame) -> pd.DataFrame:
    """Return the four descriptive group-period means used in main panel b."""
    summary = (
        states.groupby(["annual_group", "period"], observed=True, as_index=False)
        .agg(
            dTmean=("dTmean", "mean"),
            dAmp1=("dAmp1", "mean"),
            n_pairs=("pair_id", "nunique"),
            dTmean_se=("dTmean", lambda s: float(np.nanstd(s, ddof=1)) / np.sqrt(len(s))),
            dAmp1_se=("dAmp1", lambda s: float(np.nanstd(s, ddof=1)) / np.sqrt(len(s))),
        )
    )
    expected = {(g, p) for g in ("UHI", "UCI") for p in ("NHW", "HW")}
    observed = set(zip(summary["annual_group"], summary["period"]))
    if observed != expected:
        raise ValueError(
            f"Four-state summary mismatch: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    return summary


def draw_simple_geometry(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Draw only the identities, response directions and three exact sectors."""
    xline = np.linspace(xlim[0], xlim[1], 400)
    ax.axhline(0.0, color="#AAA39C", linewidth=0.75, zorder=1)
    ax.axvline(0.0, color="#C8C2BC", linewidth=0.65, zorder=1)
    ax.plot(
        xline, -xline, color=COLOR_ZERO, linestyle=(0, (6, 4)),
        linewidth=1.15, zorder=2,
    )
    ax.text(
        0.055, 0.955,
        r"$\Delta T_x\approx\Delta T_{mean}+\Delta A_1$" + "\n"
        + r"$\Delta T_n\approx\Delta T_{mean}-\Delta A_1$",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2),
        zorder=8,
    )
    label_x = min(max(0.60 * xlim[1], xlim[0]), xlim[1])
    label_y = -label_x
    ax.text(
        label_x, label_y, r"$\Delta T_x=0$", rotation=-38,
        rotation_mode="anchor", ha="center", va="bottom", fontsize=6.8,
        fontweight="bold", color=COLOR_ZERO,
        path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")],
        zorder=7,
    )

    origin = (0.38, 0.50)
    mean_end = (0.67, 0.50)
    amp_end = (0.38, 0.24)
    arrow_style = {
        "arrowstyle": "-|>", "lw": 1.55, "color": "#244D70",
        "mutation_scale": 11.0,
    }
    ax.annotate(
        "", xy=mean_end, xytext=origin, xycoords="axes fraction",
        textcoords="axes fraction", arrowprops=arrow_style, zorder=5,
    )
    ax.annotate(
        "", xy=amp_end, xytext=origin, xycoords="axes fraction",
        textcoords="axes fraction", arrowprops=arrow_style, zorder=5,
    )
    ax.text(
        0.53, 0.555, "Mean ↑\n(day ↑, night ↑)",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=6.3, fontweight="bold", color="#244D70", zorder=6,
    )
    ax.text(
        0.41, 0.30, "Amplitude ↓\n(day ↓, night ↑)",
        transform=ax.transAxes, ha="left", va="center",
        fontsize=6.3, fontweight="bold", color="#244D70", zorder=6,
    )

    sector_style = dict(
        fontsize=6.3, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.8),
    )
    ax.text(
        0.76, 0.72, "Daytime preferentially\namplified",
        transform=ax.transAxes, ha="center", va="center",
        color="#9B2F3D", **sector_style,
    )
    ax.text(
        0.78, 0.34, "Daytime damped;\nboth remain UHI",
        transform=ax.transAxes, ha="center", va="center",
        color="#A65E18", **sector_style,
    )
    ax.text(
        0.19, 0.09, "Daytime UCI /\nnight-time UHI",
        transform=ax.transAxes, ha="center", va="center",
        color="#285F98", **sector_style,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
    ax.set_ylabel(r"$\Delta Amp$ (°C)")
    panel_heading(ax, "a", "Mean–amplitude geometry")
    format_axis(ax, grid=False)


def draw_geometry_main(
    ax: plt.Axes,
    states: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Main panel a: state plane with contour, three labelled archetype
    anchors and two coordinate-direction arrows along the zero lines."""
    colorbar_ax = ax.inset_axes([0.16, -0.245, 0.68, 0.026])
    omega = 2 * np.pi / 24
    hours = np.linspace(0, 24, 1000)
    rural_reference = (
        12.0
        + 7.0 * np.cos(omega * hours - 3.8)
        + 1.0 * np.cos(2 * omega * hours - 0.5)
    )
    time_tx = float(hours[int(np.nanargmax(rural_reference))])
    kx = float(np.cos(omega * time_tx - 3.8))

    xx = np.linspace(xlim[0], xlim[1], 260)
    yy = np.linspace(ylim[0], ylim[1], 260)
    grid_x, grid_y = np.meshgrid(xx, yy)
    projection = grid_x + kx * grid_y
    finite_dtx = states["dTx"].to_numpy(float)
    limit = max(float(np.nanmax(np.abs(finite_dtx))), 1.0e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    contour_levels = np.linspace(-limit, limit, 31)
    contours = ax.contourf(
        grid_x, grid_y, np.clip(projection, -limit, limit), levels=contour_levels,
        cmap=STATE_CMAP, norm=norm, alpha=0.50, zorder=0,
    )

    # The two coordinate zero lines carry the two interpretable state directions.
    ax.axhline(0, color="#B3ADA6", linestyle="--", linewidth=0.75, zorder=1)
    ax.axvline(0, color="#B3ADA6", linestyle="--", linewidth=0.75, zorder=1)

    ax.text(
        0.04, 0.96,
        r"$\Delta T_x\approx\Delta T_{mean}+\Delta Amp$" + "\n"
        + r"$\Delta T_n\approx\Delta T_{mean}-\Delta Amp$",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
        color="#252525", linespacing=1.18,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8),
        zorder=8,
    )

    # Delta T_x = 0 boundary with a label gap, as in the former main panel.
    x_label = min(4.2, xlim[1] - 0.45, -0.78 * ylim[0] * kx)
    x_label = max(x_label, xlim[0] + 0.36 * (xlim[1] - xlim[0]))
    gap = 0.62
    for start, stop in ((xlim[0], x_label - gap), (x_label + gap, xlim[1])):
        if stop > start:
            xline = np.linspace(start, stop, 200)
            ax.plot(
                xline, -xline / kx, color=COLOR_ZERO, linestyle=(0, (6, 4)),
                linewidth=1.15, zorder=3,
            )
    y_label = -x_label / kx
    ax.text(
        x_label, y_label,
        r"$\Delta T_{mean}=-\Delta Amp$" + "\n" + r"$\Delta T_x=0$",
        ha="center", va="center", rotation=-34, rotation_mode="anchor",
        fontsize=6.1, fontweight="bold", color="#222222", linespacing=1.00,
        path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white", alpha=0.95)],
        zorder=10,
    )

    # Direction arrows along the zero lines (deep grey).
    dir_arrow = {
        "arrowstyle": "-|>", "lw": 1.55, "color": "#4A4A4A",
        "mutation_scale": 11.0,
    }
    ax.annotate(
        "", xy=(0.55 * xlim[1], 0.0), xytext=(0.0, 0.0),
        arrowprops=dir_arrow, zorder=5,
    )
    ax.text(
        0.58, 0.535,
        r"$\Delta T_{mean}\ \rightarrow$" + "\n" + r"$T_x\uparrow,\ T_n\uparrow$",
        transform=ax.transAxes, fontsize=6.2, fontweight="bold",
        color="#4A4A4A", linespacing=1.05,
        ha="left", va="bottom", zorder=6,
    )
    ax.annotate(
        "", xy=(0.0, 0.58 * ylim[0]), xytext=(0.0, 0.0),
        arrowprops=dir_arrow, zorder=5,
    )
    ax.text(
        0.02, 0.31,
        r"$\Delta Amp\downarrow$" + "\n" + r"$T_x\downarrow,\ T_n\uparrow$",
        transform=ax.transAxes, fontsize=6.2, fontweight="bold",
        color="#4A4A4A", linespacing=1.05,
        ha="left", va="center", zorder=6,
    )

    # Three archetype anchors with descriptive labels (no insets; the SI
    # figure retains the detailed diurnal reconstructions).
    nhw = states.loc[states["period"].eq("NHW")]
    point_a = (3.0, 1.0)
    point_b = (3.0, -2.0)
    c_candidates = nhw.loc[
        nhw["dTmean"].gt(0) & nhw["dAmp1"].lt(0) & nhw["dTx"].lt(0)
    ].copy()
    if c_candidates.empty:
        c_candidates = nhw.copy()
    c_score = (
        ((c_candidates["dTmean"] - 0.9) / max(xlim[1] - xlim[0], 1e-9)) ** 2
        + ((c_candidates["dAmp1"] + 2.2) / max(ylim[1] - ylim[0], 1e-9)) ** 2
    )
    c_row = c_candidates.loc[c_score.idxmin()]
    point_c = (float(c_row["dTmean"]), float(c_row["dAmp1"]))
    for point, color, size, label_text, text_position, ha, va in (
        (point_a, COLOR_UHI, 32, "amplified day and night", (0.67, 0.79), "center", "center"),
        (point_b, "#C56A22", 30, "damped daytime UHI", (0.69, 0.28), "center", "center"),
        (point_c, COLOR_UCI, 32, "daytime UCI /\nnight-time UHI", (0.16, 0.12), "center", "center"),
    ):
        ax.scatter(
            *point, s=size, color=color, edgecolor="white", linewidth=0.65,
            zorder=8,
        )
        ax.annotate(
            label_text, xy=point, xycoords="data", xytext=text_position,
            textcoords=ax.transAxes, ha=ha, va=va, fontsize=6.0,
            fontweight="bold", color=color, linespacing=1.05,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.55),
            arrowprops={
                "arrowstyle": "-", "color": color, "lw": 0.65,
                "alpha": 0.72, "shrinkA": 2.0, "shrinkB": 3.0,
            },
            zorder=9,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_autoscale_on(False)
    ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
    ax.set_ylabel(r"$\Delta Amp$ (°C)")
    colorbar = ax.figure.colorbar(contours, cax=colorbar_ax, orientation="horizontal")
    colorbar_ticks = np.linspace(-limit, limit, 5)

    def format_colorbar_tick(value: float) -> str:
        if abs(value) < 5.0e-8:
            return "0"
        magnitude = abs(value)
        if abs(magnitude - round(magnitude)) < 0.05:
            label = f"{magnitude:.0f}"
        else:
            label = f"{magnitude:.1f}"
        return f"−{label}" if value < 0 else label

    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([format_colorbar_tick(value) for value in colorbar_ticks])
    colorbar.set_label(r"Daytime $\Delta T_x$ (°C)", fontsize=6.7, labelpad=2.0)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(axis="x", labelsize=6.5, width=0.65, length=2.5, pad=1.0)
    panel_heading(ax, "a", "Mean–amplitude response geometry")
    format_axis(ax, grid=False)


def draw_four_group_states(
    ax: plt.Axes,
    group_states: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Draw four means using colour for group and fill for period."""
    xline = np.linspace(xlim[0], xlim[1], 300)
    ax.plot(xline, -xline, color=COLOR_ZERO, linestyle="--", linewidth=0.9)
    ax.axhline(0.0, color="#A9A39D", linewidth=0.65)
    ax.axvline(0.0, color="#A9A39D", linewidth=0.65)

    colors = {"UHI": COLOR_UHI, "UCI": COLOR_UCI}
    label_offsets = {
        ("UHI", "NHW"): (-7, 8), ("UHI", "HW"): (7, 8),
        ("UCI", "NHW"): (-7, -11), ("UCI", "HW"): (7, -11),
    }
    for group in ("UHI", "UCI"):
        subset = group_states.loc[group_states["annual_group"].eq(group)].set_index("period")
        nhw = subset.loc["NHW"]
        hw = subset.loc["HW"]
        ax.annotate(
            "", xy=(float(hw.dTmean), float(hw.dAmp1)),
            xytext=(float(nhw.dTmean), float(nhw.dAmp1)),
            arrowprops={
                "arrowstyle": "-|>", "lw": 1.45, "color": colors[group],
                "mutation_scale": 8.5, "alpha": 0.92,
                "shrinkA": 4.0, "shrinkB": 4.0,
            },
            zorder=3,
        )
        for period in ("NHW", "HW"):
            row = subset.loc[period]
            filled = period == "HW"
            # Pair-level 95% CI error bars (1.96 x SE), matching Figure 1.
            ax.errorbar(
                [float(row.dTmean)], [float(row.dAmp1)],
                xerr=[[1.96 * float(row.dTmean_se)]],
                yerr=[[1.96 * float(row.dAmp1_se)]],
                fmt="none", ecolor=colors[group], elinewidth=0.9,
                capsize=2.2, zorder=3.5,
            )
            ax.scatter(
                [float(row.dTmean)], [float(row.dAmp1)], s=34, marker="o",
                facecolor=colors[group] if filled else "white",
                edgecolor=colors[group], linewidth=1.05, zorder=4,
            )
            dx, dy = label_offsets[(group, period)]
            ax.annotate(
                f"{group} {period}", xy=(float(row.dTmean), float(row.dAmp1)),
                xytext=(dx, dy), textcoords="offset points",
                ha="right" if dx < 0 else "left",
                va="bottom" if dy > 0 else "top",
                fontsize=6.3, fontweight="bold", color=colors[group], zorder=5,
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\Delta T_{mean}$ (°C)")
    ax.set_ylabel(r"$\Delta Amp$ (°C)")
    ax.text(
        0.96, 0.06, r"$\Delta T_x=0$",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.1, color=COLOR_ZERO,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.5),
    )
    panel_heading(ax, "b", "Group-mean state shifts")
    format_axis(ax, grid=True)


def draw_group_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Plot frozen group responses and bootstrap intervals as dot-whiskers."""
    positions = np.arange(len(COMPONENT_ORDER), dtype=float)
    offsets = {"UHI": -0.115, "UCI": 0.115}
    all_low: list[float] = []
    all_high: list[float] = []
    for group, color in (
        ("UHI", COLOR_UHI),
        ("UCI", COLOR_UCI),
    ):
        subset = summary.loc[summary["annual_group"].eq(group)].set_index("component")
        estimates = np.array([float(subset.loc[name, "estimate"]) for name in COMPONENT_ORDER])
        low = np.array([float(subset.loc[name, "ci_low"]) for name in COMPONENT_ORDER])
        high = np.array([float(subset.loc[name, "ci_high"]) for name in COMPONENT_ORDER])
        all_low.extend(low.tolist())
        all_high.extend(high.tolist())
        ax.errorbar(
            estimates, positions + offsets[group],
            xerr=np.vstack([estimates - low, high - estimates]),
            fmt="o", markersize=4.7, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.55,
            color=color, ecolor=color, elinewidth=1.05, capsize=2.7,
            capthick=1.0,
            label=f"{group} (n={int(subset['n_pairs'].iloc[0])})", zorder=3,
        )
    ax.axvline(0, color="#222222", linewidth=0.85, zorder=2)
    ax.set_yticks(positions, [COMPONENT_LABELS[name] for name in COMPONENT_ORDER])
    ax.set_ylim(-0.55, len(COMPONENT_ORDER) - 0.45)
    ax.invert_yaxis()
    low_limit = min(min(all_low), 0.0)
    high_limit = max(max(all_high), 0.0)
    span = max(high_limit - low_limit, 0.25)
    ax.set_xlim(low_limit - 0.12 * span, high_limit + 0.27 * span)
    ax.set_xlabel("HW−NHW urban–rural response (°C)")
    ax.legend(
        loc="lower right", ncol=1, frameon=False,
        columnspacing=0.9, handlelength=1.3, handletextpad=0.35,
        labelspacing=0.35, borderaxespad=0.45,
    )
    panel_heading(
        ax, "c",
        "Heatwave-response components",
        label_x=-0.045,
        label_y=1.12,
    )
    format_axis(ax, grid=False)
    ax.grid(True, axis="x", color=COLOR_GRID, linestyle=(0, (1.3, 2.2)), linewidth=0.45)
    ax.set_axisbelow(True)


def save_rgb_jpg(fig: plt.Figure, path: Path) -> dict[str, object]:
    temporary = path.with_name(f".{path.stem}.temporary.jpg")
    try:
        fig.savefig(
            temporary, format="jpg", dpi=FORMAL_DPI, facecolor="white", edgecolor="white",
            pil_kwargs={"quality": 96, "subsampling": 0, "optimize": True},
        )
        with Image.open(temporary) as image:
            expected = (
                round(FIGURE_WIDTH_MM * MM_TO_IN * FORMAL_DPI),
                round(FIGURE_HEIGHT_MM * MM_TO_IN * FORMAL_DPI),
            )
            dpi = image.info.get("dpi", (0, 0))
            if image.mode != "RGB" or abs(image.width - expected[0]) > 2 or abs(image.height - expected[1]) > 2:
                raise RuntimeError(
                    f"Invalid formal image: mode={image.mode}, pixels={(image.width, image.height)}, expected={expected}"
                )
            if min(dpi) < 590:
                raise RuntimeError(f"DPI metadata below 600: {dpi}")
            info = {"mode": image.mode, "pixels": [image.width, image.height], "dpi": list(dpi)}
        os.replace(temporary, path)
        return info
    finally:
        temporary.unlink(missing_ok=True)


def save_preview(jpg_path: Path, webp_path: Path) -> None:
    with Image.open(jpg_path) as image:
        preview = image.copy()
        preview.thumbnail((1800, 1400), Image.Resampling.LANCZOS)
        preview.save(webp_path, format="WEBP", quality=82, method=6)


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
    summary: pd.DataFrame,
    output_dir: Path,
    metrics_path: Path | None,
    summary_path: Path | None,
    enforce_formal_counts: bool,
) -> tuple[Path, Path, Path, Path]:
    apply_nature_style()
    states, frozen_summary, audit = freeze_state_tables(
        metrics, summary, enforce_formal_counts
    )
    group_states = summarize_group_states(states)
    xlim, ylim = limits_for_states(states)
    group_xlim, group_ylim = limits_for_group_states(group_states)
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
        dpi=FORMAL_DPI,
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 0.544], width_ratios=[1.0, 1.0]
    )
    # Row 1: two square panels (a and b) side by side.  Row 2: panel c spans
    # the full width as a wide dot-whisker summary.
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(
        left=0.080,
        right=0.965,
        bottom=0.070,
        top=0.925,
        wspace=0.24,
        hspace=0.34,
    )
    # Panels a and b are exact squares: the row-1 cells are taller than wide,
    # so each square is centred vertically in its cell.
    cell_a = ax_a.get_position()
    cell_b = ax_b.get_position()
    square_side = min(cell_a.width, cell_a.height)
    ax_a.set_position(
        [cell_a.x0, cell_a.y0 + 0.5 * (cell_a.height - square_side),
         square_side, square_side]
    )
    ax_b.set_position(
        [cell_b.x0, cell_b.y0 + 0.5 * (cell_b.height - square_side),
         square_side, square_side]
    )
    draw_geometry_main(ax_a, states, xlim, ylim)
    draw_four_group_states(ax_b, group_states, group_xlim, group_ylim)
    draw_group_summary(ax_c, frozen_summary)
    fig.canvas.draw()
    box_a = ax_a.get_position()
    box_b = ax_b.get_position()
    box_c = ax_c.get_position()
    layout_tolerance = 1.0e-9
    # Panels a and b are equal squares, centred in their row-1 cells; panel c
    # spans the full plotting width below them.
    if not (
        abs(box_a.width - box_b.width) <= layout_tolerance
        and abs(box_a.height - box_b.height) <= layout_tolerance
        and abs(box_a.height - box_a.width) <= layout_tolerance
    ):
        raise RuntimeError("Figure 2 main-panel alignment invariant failed.")
    layout_audit = {
        "top_left": float(box_a.x0),
        "top_right": float(box_b.x1),
        "panel_a_square_side": float(box_a.width),
        "panel_a_box_aspect": float(box_a.height / box_a.width),
        "panel_b_box_aspect": float(box_b.height / box_b.width),
        "panel_c_left": float(box_c.x0),
        "panel_c_right": float(box_c.x1),
        "panel_c_width": float(box_c.width),
        "panel_c_height": float(box_c.height),
    }

    main_path = output_dir / MAIN_FILENAME
    preview_path = output_dir / PREVIEW_FILENAME
    image_info = save_rgb_jpg(fig, main_path)
    plt.close(fig)
    save_preview(main_path, preview_path)

    # Preserve the former detailed geometry and complete pair cloud as an SI
    # candidate rather than discarding those explanatory layers.
    si_fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
        dpi=FORMAL_DPI,
        facecolor="white",
    )
    si_grid = si_fig.add_gridspec(2, 2, height_ratios=[1.0, 0.34])
    si_a = si_fig.add_subplot(si_grid[0, 0])
    si_b = si_fig.add_subplot(si_grid[0, 1])
    si_notes = si_fig.add_subplot(si_grid[1, 0])
    si_notes.set_axis_off()
    si_fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.055, top=0.815,
        wspace=0.27, hspace=0.32,
    )
    for axis in (si_a, si_b):
        axis.set_box_aspect(1)
    si_colorbar_ax = si_a.inset_axes([0.09, -0.305, 0.82, 0.025])
    draw_geometry_panel(si_a, states, xlim, ylim, si_colorbar_ax)
    draw_geometry_notes(si_notes)
    draw_combined_states(si_b, states, xlim, ylim, audit)
    si_path = output_dir / SI_FILENAME
    si_preview_path = output_dir / SI_PREVIEW_FILENAME
    si_image_info = save_rgb_jpg(si_fig, si_path)
    plt.close(si_fig)
    save_preview(si_path, si_preview_path)

    group_means_path = output_dir / GROUP_MEANS_FILENAME
    group_states.to_csv(group_means_path, index=False, float_format="%.10g")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "metrics_source": str(metrics_path.resolve()) if metrics_path else "SELF_TEST",
        "metrics_source_sha256": sha256_file(metrics_path.resolve()) if metrics_path else None,
        "summary_source": str(summary_path.resolve()) if summary_path else "SELF_TEST",
        "summary_source_sha256": sha256_file(summary_path.resolve()) if summary_path else None,
        "cohort": audit,
        "main_image": MAIN_FILENAME,
        "main_image_sha256": sha256_file(main_path),
        "preview": PREVIEW_FILENAME,
        "preview_sha256": sha256_file(preview_path),
        "si_image": SI_FILENAME,
        "si_image_sha256": sha256_file(si_path),
        "si_preview": SI_PREVIEW_FILENAME,
        "si_preview_sha256": sha256_file(si_preview_path),
        "group_state_means": GROUP_MEANS_FILENAME,
        "group_state_means_sha256": sha256_file(group_means_path),
        "image_info": image_info,
        "si_image_info": si_image_info,
        "fixed_canvas_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "main_layout": (
            "row 1: square panels a and b side by side; row 2: full-width "
            "dot-whisker panel c"
        ),
        "layout_alignment": layout_audit,
        "panel_a_pair_cloud_shown": False,
        "panel_a_mechanism_arrows_shown": False,
        "panel_a_archetype_insets_in_main": False,
        "panel_a_archetype_labels": "amplified day and night / damped daytime UHI / daytime UCI night-time UHI",
        "panel_a_direction_arrows": "along the two coordinate zero lines",
        "panel_a_state_plane_box_aspect": 1.0,
        "panel_b_error_bars": "pair-level 95% CI (1.96 x SE)",
        "panel_c_style": "horizontal dot-whiskers with frozen bootstrap 95% CIs",
        "panel_a_boundary_label": "Delta T_mean = -Delta Amp; Delta T_x = 0",
        "panel_b_boundary_label": "Delta T_x = 0",
        "panel_b_zoom_limits": {
            "x": [float(group_xlim[0]), float(group_xlim[1])],
            "y": [float(group_ylim[0]), float(group_ylim[1])],
        },
        "bbox_inches_tight_used": False,
        "plot_only": True,
        "regression_fitted_in_plot": False,
        "bootstrap_run_in_plot": False,
        "descriptive_aggregation_only": True,
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
            "si_image": SI_FILENAME,
            "si_image_sha256": sha256_file(si_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return main_path, preview_path, si_path, si_preview_path


def synthetic_inputs(seed: int = 20260818) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    n_uhi, n_uci = 23, 13
    for index in range(n_uhi + n_uci):
        pair = f"P{index:03d}"
        group = "UHI" if index < n_uhi else "UCI"
        base_mean = rng.normal(0.9 if group == "UHI" else -0.6, 0.45)
        base_amp = rng.normal(1.0, 0.4)
        for period in ("annual", "non_heatwave", "heatwave"):
            hw = period == "heatwave"
            mean = base_mean + (rng.normal(0.28, 0.18) if hw else 0.0)
            amp = base_amp + (rng.normal(-0.22, 0.16) if hw else 0.0)
            rows.append(
                {
                    "pair_id": pair,
                    "period": period,
                    "group": group,
                    "hw_method": "percentile",
                    "dTmean": mean,
                    "dAmp1": amp,
                    "dTx": mean + amp,
                    "dTn": mean - amp,
                }
            )
    summary_rows = []
    for group, n in (("UHI", n_uhi), ("UCI", n_uci)):
        for component, estimate in zip(COMPONENT_ORDER, (0.3, -0.2, 0.1, 0.5)):
            sign = 1 if group == "UHI" else -1
            value = sign * estimate
            summary_rows.append(
                {
                    "annual_group": group,
                    "component": component,
                    "estimate": value,
                    "ci_low": value - 0.12,
                    "ci_high": value + 0.12,
                    "n_pairs": n,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = create_output_dir(args.output_root)
    if args.self_test:
        metrics, summary = synthetic_inputs()
        metrics_path = None
        summary_path = None
        enforce_counts = False
    else:
        metrics_path = args.metrics_csv.expanduser().resolve()
        summary_path = args.summary_csv.expanduser().resolve()
        hashes_before = {
            "metrics": sha256_file(metrics_path),
            "summary": sha256_file(summary_path),
        }
        metrics, summary = read_inputs(metrics_path, summary_path)
        enforce_counts = True
    main_path, preview_path, si_path, si_preview_path = render(
        metrics, summary, output_dir, metrics_path, summary_path, enforce_counts
    )
    if not args.self_test:
        hashes_after = {
            "metrics": sha256_file(metrics_path),
            "summary": sha256_file(summary_path),
        }
        if hashes_before != hashes_after:
            raise RuntimeError("READ_ONLY_INPUT_MODIFIED_DURING_PLOTTING")
    print(f"SUCCESS: {main_path}")
    print(f"Preview: {preview_path}")
    print(f"SI candidate: {si_path}")
    print(f"SI preview: {si_preview_path}")


if __name__ == "__main__":
    main()
