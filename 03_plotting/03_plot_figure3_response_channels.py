#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot-only Figure 3: response channels and timing structure.

The main 1 x 2 figure contains (a) pair-level responses in the
R_mean--R_Amp plane, frozen UHI/UCI group vectors and sign-only constraints
from a linear thermal-balance model, and (b) the frozen pooled FWL timing
association with UHI/UCI colour context.  The event-centred dynamic panel is
deliberately excluded pending its separate upstream audit.  Thermal-balance
sensitivity is retained as a standalone SI candidate.

This script performs no model fitting, bootstrap resampling, threshold
selection, event matching, or sample filtering.  With no arguments it uses
the actual project paths and can be run directly as

    python 03_plotting/03_plot_figure3_response_channels.py

Every formal image is a fixed-canvas 600-dpi RGB JPEG.  Small WebP copies are
created only for visual QA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

_MPL_CACHE = Path(tempfile.gettempdir()) / "figure3_response_channels_mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Wedge
from PIL import Image


SCRIPT_VERSION = "plot_figure3_response_channels"
ANALYSIS_VERSION = "figure3_dynamic_mechanism_analysis"

MAIN_FILENAME = "figure3_response_channels.jpg"
MAIN_PREVIEW_FILENAME = "figure3_response_channels_preview.webp"
THEORY_FILENAME = "figure3_linear_thermal_balance_sensitivity.jpg"
THEORY_PREVIEW_FILENAME = (
    "figure3_linear_thermal_balance_sensitivity_preview.webp"
)
SI_FULL_FILENAME = "figure3_full_range_response_plane_si.jpg"
SI_FULL_PREVIEW_FILENAME = "figure3_full_range_response_plane_si_preview.webp"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = (
    PROJECT_ROOT / "outputs/analysis/figure3_dynamic_mechanism"
)
DEFAULT_PLOT_ROOT = (
    PROJECT_ROOT / "outputs/figures/figure3_response_channels"
)

# Exact principal red/blue colours sampled from the formal Figure 2 artwork.
COLOR_UHI = "#BB3445"
COLOR_UCI = "#3C79B6"
COLOR_THEORY = "#453A33"
COLOR_REMOVAL = "#C37A10"
COLOR_STORAGE = "#B33A2B"
COLOR_TIMING = "#6E4B36"
COLOR_FIT = "#88422F"
COLOR_BAND = "#E9D5B8"
COLOR_POINTS = "#8B837C"
COLOR_GRID = "#D8D2CB"
COLOR_ZERO = "#403B37"

# Figure 2 uses a visibly larger type system at the same manuscript width.
# These values increase the complete Figure 3 hierarchy together instead of
# enlarging isolated labels and disturbing the visual balance.
SUBTITLE_SIZE = 7.8
PANEL_LABEL_SIZE = 9.5
AXIS_LABEL_SIZE = 7.2
TICK_SIZE = 6.4
LEGEND_SIZE = 6.0
ANNOT_SIZE = 6.2
FRAME_LW = 0.75

MAIN_CANVAS_IN = (180.0 / 25.4, 96.0 / 25.4)
THEORY_CANVAS_IN = (180.0 / 25.4, 100.0 / 25.4)

REQUIRED_FILES = (
    "fig3_panel_a_pair_state.csv",
    "fig3_panel_a_binned_flow.csv",
    "fig3_panel_b_group_summary.csv",
    "fig3_panel_b_jacobian.csv",
    "fig3_panel_b_group_angle_contrast.csv",
    "fig3_panel_c_fwl_pair.csv",
    "fig3_panel_c_fwl_curve.csv",
    "fig3_panel_c_fwl_bins.csv",
    "fig3_panel_c_timing_coefficient.csv",
    "fig3_standalone_finite_frequency_sensitivity.csv",
    "fig3_dynamic_analysis_manifest.json",
    "_SUCCESS.json",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Mapping) -> None:
    path.write_text(
        json.dumps(dict(obj), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Plot output directory is not empty: {path}. Use a new directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_analysis_bundle(
    analysis_dir: Path, allow_test_bundle: bool
) -> Tuple[Dict, Dict]:
    missing = [name for name in REQUIRED_FILES if not (analysis_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Figure 3 analysis bundle; missing: {missing}")
    success = load_json(analysis_dir / "_SUCCESS.json")
    manifest_path = analysis_dir / "fig3_dynamic_analysis_manifest.json"
    manifest = load_json(manifest_path)
    if success.get("status") != "SUCCESS":
        raise ValueError("Analysis completion marker is not SUCCESS.")
    if manifest.get("script_version") != ANALYSIS_VERSION:
        raise ValueError(
            f"Unexpected analysis version {manifest.get('script_version')!r}; "
            f"expected {ANALYSIS_VERSION!r}."
        )
    run_mode = str(manifest.get("run_mode", "")).upper()
    if run_mode != "FORMAL" and not allow_test_bundle:
        raise ValueError(
            "Refusing a non-formal bundle. Use --allow-test-bundle only for QA."
        )
    if success.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Manifest SHA-256 does not match _SUCCESS.json.")
    for filename, expected_hash in manifest.get("output_sha256", {}).items():
        path = analysis_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-listed output is missing: {filename}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for analysis output: {filename}")
    return manifest, success


def read_tables(analysis_dir: Path) -> Dict[str, pd.DataFrame]:
    mapping = {
        "pair": "fig3_panel_a_pair_state.csv",
        "flow": "fig3_panel_a_binned_flow.csv",
        "group": "fig3_panel_b_group_summary.csv",
        "jacobian": "fig3_panel_b_jacobian.csv",
        "angle_contrast": "fig3_panel_b_group_angle_contrast.csv",
        "fwl_pair": "fig3_panel_c_fwl_pair.csv",
        "fwl_curve": "fig3_panel_c_fwl_curve.csv",
        "fwl_bins": "fig3_panel_c_fwl_bins.csv",
        "timing_coefficient": "fig3_panel_c_timing_coefficient.csv",
        "pi": "fig3_standalone_finite_frequency_sensitivity.csv",
    }
    return {key: pd.read_csv(analysis_dir / name) for key, name in mapping.items()}


def validate_formal_results(tables: Mapping[str, pd.DataFrame], allow_test_bundle: bool) -> Dict:
    """Freeze the formal 317-pair state and FWL quantities before plotting."""
    pair = tables["pair"]
    required_pair = {
        "annual_group", "delta_Tmean_NHW", "delta_Amp_NHW",
        "delta_Tmean_HW", "delta_Amp_HW", "R_mean", "R_Amp",
    }
    if not required_pair.issubset(pair.columns):
        raise ValueError(f"Pair table missing: {sorted(required_pair-set(pair.columns))}")
    counts = pair["annual_group"].astype(str).value_counts().to_dict()
    coefficient = tables["timing_coefficient"]
    primary = coefficient.loc[coefficient["term"].astype(str).eq("amplitude_damping")]
    if len(primary) != 1:
        raise ValueError("Expected exactly one frozen amplitude_damping coefficient.")
    row = primary.iloc[0]
    group_summary = tables["group"]
    required_group = {
        "annual_group", "R_mean", "R_Amp", "theta_deg",
        "theta_ci_low", "theta_ci_high",
    }
    if not required_group.issubset(group_summary.columns):
        raise ValueError("Group response summary is incomplete.")
    if not allow_test_bundle:
        if len(pair) != 317 or counts != {"UHI": 200, "UCI": 117}:
            raise ValueError(
                f"Formal Figure 3 cohort mismatch: n={len(pair)}, counts={counts}."
            )
        if int(row["n_pairs"]) != 317 or int(row["n_blocks"]) != 76:
            raise ValueError(
                "Frozen FWL count mismatch: "
                f"n={row['n_pairs']}, blocks={row['n_blocks']}."
            )
    return {
        "n_pairs": int(len(pair)),
        "n_uhi": int(counts.get("UHI", 0)),
        "n_uci": int(counts.get("UCI", 0)),
        "fwl_n_pairs": int(row["n_pairs"]),
        "fwl_n_blocks": int(row["n_blocks"]),
        "beta_A": float(row["estimate"]),
        "beta_A_ci": [float(row["ci_low"]), float(row["ci_high"])],
    }


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(FRAME_LW)
    ax.tick_params(
        axis="both", which="major", direction="out", top=False, right=False,
        width=0.75, length=3.0, labelsize=TICK_SIZE, colors="black",
    )
    if grid:
        ax.grid(True, color=COLOR_GRID, linewidth=0.58, alpha=0.43, zorder=0)
    else:
        ax.grid(False)


def add_panel_heading(
    ax: plt.Axes,
    label: str,
    subtitle: str,
    pad: float = 8.0,
) -> None:
    # The panel letter sits immediately to the left of the centred subtitle,
    # sharing its baseline (different title locations keep both titles).
    ax.set_title(
        subtitle,
        loc="center",
        fontsize=SUBTITLE_SIZE,
        fontweight="bold",
        pad=pad,
        linespacing=1.02,
    )

    panel_title = ax.set_title(
        label,
        loc="left",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        pad=pad,
    )
    panel_title.set_x(-0.04)
    panel_title.set_ha("right")
    panel_title.set_clip_on(False)


def group_color(group: str) -> str:
    return COLOR_UHI if str(group) == "UHI" else COLOR_UCI


def finite_limits(values: Sequence[np.ndarray], pad_fraction: float = 0.08) -> Tuple[float, float]:
    array = np.concatenate([np.asarray(value, dtype=float).ravel() for value in values])
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return -1.0, 1.0
    low, high = float(array.min()), float(array.max())
    if low == high:
        return low - 0.5, high + 0.5
    padding = (high - low) * pad_fraction
    return low - padding, high + padding


def draw_panel_a(ax: plt.Axes, pair: pd.DataFrame, flow: pd.DataFrame) -> None:
    required = {
        "annual_group", "delta_Tmean_NHW", "delta_Amp_NHW",
        "delta_Tmean_HW", "delta_Amp_HW", "R_mean", "R_Amp",
    }
    missing = required.difference(pair.columns)
    if missing:
        raise ValueError(f"Panel a pair table is missing: {sorted(missing)}")
    required_flow = {"annual_group", "x0", "y0", "dx", "dy"}
    if not required_flow.issubset(flow.columns):
        raise ValueError("Panel a frozen grid-flow table is incomplete.")

    # Equal-visibility, all-pair paths remain a faint observational substrate.
    for group in ("UHI", "UCI"):
        subset = pair[pair["annual_group"].eq(group)]
        ax.quiver(
            subset["delta_Tmean_NHW"], subset["delta_Amp_NHW"],
            subset["R_mean"], subset["R_Amp"],
            angles="xy", scale_units="xy", scale=1, color=group_color(group),
            alpha=0.105, width=0.00145, headwidth=3.6, headlength=4.8,
            headaxislength=4.1, zorder=1, rasterized=True,
        )

    # Fixed 10 x 10 start-state grid means are the foreground summary.
    for group in ("UHI", "UCI"):
        subset = flow[flow["annual_group"].eq(group)]
        color = group_color(group)
        for row in subset.itertuples(index=False):
            arrow = FancyArrowPatch(
                (float(row.x0), float(row.y0)),
                (float(row.x0 + row.dx), float(row.y0 + row.dy)),
                arrowstyle="-|>", mutation_scale=14.0, linewidth=2.35,
                color=color, alpha=0.96,
                path_effects=[
                    path_effects.withStroke(linewidth=4.25, foreground="white", alpha=0.90)
                ],
                zorder=5,
            )
            ax.add_patch(arrow)

    x_limits = finite_limits(
        [pair["delta_Tmean_NHW"], pair["delta_Tmean_HW"]], pad_fraction=0.06
    )
    y_limits = finite_limits(
        [pair["delta_Amp_NHW"], pair["delta_Amp_HW"]], pad_fraction=0.07
    )
    xline = np.linspace(x_limits[0], x_limits[1], 300)
    ax.plot(xline, -xline, color=COLOR_ZERO, linestyle="--", linewidth=1.05,
            alpha=0.70, zorder=2)
    ax.text(
        0.02, 0.03, r"$\Delta T_{mean}+\Delta A_1=0$", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=ANNOT_SIZE - 0.3, color=COLOR_ZERO,
        path_effects=[path_effects.withStroke(linewidth=3.0, foreground="white")],
        zorder=7,
    )
    ax.axhline(0.0, color="#AAA39C", linewidth=0.78, zorder=1)
    ax.axvline(0.0, color="#AAA39C", linewidth=0.78, zorder=1)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(r"Mean contrast, $\Delta T_{mean}$ (°C)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(
        r"First-harmonic amplitude contrast, $\Delta A_1$ (°C)",
        fontsize=AXIS_LABEL_SIZE,
    )
    add_panel_heading(ax, "a", "NHW-to-HW state migration")
    style_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_UHI, linewidth=2.3, label="UHI regime"),
            Line2D([0], [0], color=COLOR_UCI, linewidth=2.3, label="UCI regime"),
        ],
        frameon=False, loc="upper left", fontsize=LEGEND_SIZE,
        handlelength=1.8, borderaxespad=0.55,
    )


def draw_response_vector_key(
    parent: plt.Axes,
    group_summary: pd.DataFrame,
    jacobian: pd.DataFrame,
) -> None:
    """Draw response displacements in a separate origin, never on state origins."""
    required = {
        "annual_group", "R_mean", "R_Amp", "theta_deg",
        "theta_ci_low", "theta_ci_high",
    }
    if not required.issubset(group_summary.columns):
        raise ValueError("Response-vector key lacks frozen group directions.")
    expected_channels = {
        "persistent_mean_loading", "relative_amplitude_damping",
        "weaker_effective_removal",
    }
    if not expected_channels.issubset(set(jacobian["channel"].astype(str))):
        raise ValueError("Jacobian sign-direction table is incomplete.")

    inset = parent.inset_axes([0.565, 0.055, 0.405, 0.405])
    values = np.concatenate(
        [
            group_summary["R_mean"].to_numpy(float),
            group_summary["R_Amp"].to_numpy(float),
        ]
    )
    extent = max(float(np.nanmax(np.abs(values))) * 1.75, 0.55)
    inset.axhline(0, color="#A9A29B", linewidth=0.65)
    inset.axvline(0, color="#A9A29B", linewidth=0.65)

    # Frozen observed group response directions.
    for row in group_summary.itertuples(index=False):
        group = str(row.annual_group)
        color = group_color(group)
        radius = min(max(1.25 * float(getattr(row, "rho_group_mean_vector", 0.0)),
                         0.16 * extent), 0.55 * extent)
        inset.add_patch(
            Wedge(
                (0, 0), radius, float(row.theta_ci_low), float(row.theta_ci_high),
                width=0.66 * radius, facecolor=color, edgecolor="none",
                alpha=0.14, zorder=1,
            )
        )
        inset.add_patch(
            FancyArrowPatch(
                (0, 0), (float(row.R_mean), float(row.R_Amp)),
                arrowstyle="-|>", mutation_scale=11.5, linewidth=2.0,
                color=color,
                path_effects=[path_effects.withStroke(linewidth=3.2, foreground="white")],
                zorder=5,
            )
        )
        inset.text(
            float(row.R_mean), float(row.R_Amp),
            f"  {group} {float(row.theta_deg):.0f}°\n"
            f"  [{float(row.theta_ci_low):.0f}, {float(row.theta_ci_high):.0f}]",
            color=color, fontsize=ANNOT_SIZE - 2.1, fontweight="bold",
            ha="left", va="center", zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.5),
        )

    # Jacobian directions are sign keys only; their lengths are arbitrary.
    length = 0.43 * extent
    theory_style = dict(
        arrowstyle="-|>", linewidth=1.15, linestyle="--", mutation_scale=8.5
    )
    directions = [
        ((length, 0.0), COLOR_REMOVAL, r"$F_0\uparrow$", (0.64, 0.07), "left", "bottom"),
        ((0.0, -length), "#955038", r"$F_1\downarrow$ / $C\uparrow$", (0.06, -0.69), "left", "top"),
        ((0.33 * extent, 0.29 * extent), COLOR_THEORY, r"$\lambda\downarrow$", (0.50, 0.52), "left", "bottom"),
    ]
    for endpoint, color, text, label_fraction, ha, va in directions:
        inset.annotate(
            "", xy=endpoint, xytext=(0, 0),
            arrowprops={**theory_style, "color": color}, zorder=3,
        )
        inset.text(
            label_fraction[0] * extent, label_fraction[1] * extent, text,
            color=color, fontsize=ANNOT_SIZE - 2.5, ha=ha, va=va,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.45),
            zorder=6,
        )

    inset.set_xlim(-extent, extent)
    inset.set_ylim(-extent, extent)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xlabel(r"$R_{mean}$", fontsize=ANNOT_SIZE - 1.5, labelpad=0.5)
    inset.set_ylabel(r"$R_{Amp}$", fontsize=ANNOT_SIZE - 1.5, labelpad=0.5)
    inset.set_title(
        "Response-vector key", fontsize=ANNOT_SIZE - 0.8,
        fontweight="bold", pad=3.0,
    )
    inset.tick_params(labelsize=ANNOT_SIZE - 3.0, length=2.5, width=0.7, pad=1.0)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#554F49")
    inset.text(
        0.98, 0.02, "Jacobian: direction only\nlength not empirical",
        transform=inset.transAxes, ha="right", va="bottom",
        fontsize=ANNOT_SIZE - 3.0, color=COLOR_THEORY,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.5),
    )


def draw_response_direction_key(
    parent: plt.Axes,
    group_summary: pd.DataFrame,
    jacobian: pd.DataFrame,
) -> None:
    """Compact direction key with separated observed and theoretical origins."""
    required = {
        "annual_group", "R_mean", "R_Amp", "theta_deg",
        "theta_ci_low", "theta_ci_high",
    }
    if not required.issubset(group_summary.columns):
        raise ValueError("Response-direction key lacks frozen group directions.")
    expected_channels = {
        "persistent_mean_loading", "relative_amplitude_damping",
        "weaker_effective_removal",
    }
    if not expected_channels.issubset(set(jacobian["channel"].astype(str))):
        raise ValueError("Jacobian sign-direction table is incomplete.")

    inset = parent.inset_axes([0.535, 0.055, 0.445, 0.445])
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#554F49")
    inset.set_title(
        "Response-direction key", fontsize=ANNOT_SIZE - 0.9,
        fontweight="bold", pad=3.0,
    )
    inset.plot([0.50, 0.50], [0.14, 0.88], color="#D8D2CB", linewidth=0.65)
    inset.text(0.25, 0.92, "Observed", ha="center", va="center",
               fontsize=ANNOT_SIZE - 2.4, fontweight="bold")
    inset.text(0.75, 0.92, "Jacobian signs", ha="center", va="center",
               fontsize=ANNOT_SIZE - 2.4, fontweight="bold")

    origin = np.array([0.20, 0.47])
    radii = np.hypot(
        group_summary["R_mean"].to_numpy(float),
        group_summary["R_Amp"].to_numpy(float),
    )
    scale = 0.29 / max(float(np.nanmax(radii)), 1.0e-9)
    label_positions = {"UHI": (0.04, 0.79), "UCI": (0.04, 0.19)}
    for row in group_summary.itertuples(index=False):
        group = str(row.annual_group)
        color = group_color(group)
        vector = scale * np.array([float(row.R_mean), float(row.R_Amp)])
        endpoint = origin + vector
        radius = max(float(np.hypot(*vector)), 0.08)
        inset.add_patch(
            Wedge(
                tuple(origin), radius, float(row.theta_ci_low), float(row.theta_ci_high),
                width=0.60 * radius, transform=inset.transAxes,
                facecolor=color, edgecolor="none", alpha=0.14, zorder=1,
            )
        )
        inset.add_patch(
            FancyArrowPatch(
                tuple(origin), tuple(endpoint), transform=inset.transAxes,
                arrowstyle="-|>", mutation_scale=10.0, linewidth=1.9,
                color=color,
                path_effects=[path_effects.withStroke(linewidth=3.0, foreground="white")],
                zorder=4,
            )
        )
        label_xy = label_positions[group]
        inset.annotate(
            f"{group} {float(row.theta_deg):.0f}°\n"
            f"[{float(row.theta_ci_low):.0f}, {float(row.theta_ci_high):.0f}]",
            xy=tuple(endpoint), xycoords=inset.transAxes,
            xytext=label_xy, textcoords=inset.transAxes,
            ha="left", va="center", color=color,
            fontsize=ANNOT_SIZE - 2.4, fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.35),
            zorder=6,
        )
    inset.plot(origin[0], origin[1], marker="o", markersize=2.2,
               color=COLOR_ZERO, transform=inset.transAxes, zorder=7)
    inset.text(0.45, 0.13, r"$R_{mean}\;\rightarrow$", transform=inset.transAxes,
               ha="right", va="center", fontsize=ANNOT_SIZE - 2.8)
    inset.text(0.015, 0.50, r"$R_{Amp}\;\uparrow$", transform=inset.transAxes,
               ha="left", va="center", rotation=90, fontsize=ANNOT_SIZE - 2.8)

    theory_origin = np.array([0.68, 0.47])
    theory_specs = (
        (np.array([0.91, 0.47]), COLOR_REMOVAL, r"$F_0\uparrow$", (0.93, 0.43), "right"),
        (np.array([0.68, 0.22]), "#955038", r"$F_1\downarrow$ / $C\uparrow$", (0.68, 0.16), "center"),
        (np.array([0.87, 0.68]), COLOR_THEORY, r"$\lambda\downarrow$", (0.90, 0.72), "right"),
    )
    for endpoint, color, label, label_xy, horizontal_alignment in theory_specs:
        inset.annotate(
            "", xy=tuple(endpoint), xytext=tuple(theory_origin),
            xycoords=inset.transAxes, textcoords=inset.transAxes,
            arrowprops={
                "arrowstyle": "-|>", "color": color, "lw": 1.0,
                "linestyle": "--", "mutation_scale": 8.0,
            },
            zorder=3,
        )
        inset.text(
            label_xy[0], label_xy[1], label, transform=inset.transAxes,
            ha=horizontal_alignment, va="center", color=color,
            fontsize=ANNOT_SIZE - 2.6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.25),
            zorder=5,
        )
    inset.plot(theory_origin[0], theory_origin[1], marker="o", markersize=2.2,
               color=COLOR_ZERO, transform=inset.transAxes, zorder=7)



def draw_panel_a_merged(
    ax: plt.Axes,
    pair: pd.DataFrame,
    flow: pd.DataFrame,
    group_summary: pd.DataFrame,
    jacobian: pd.DataFrame,
) -> None:
    draw_panel_a(ax, pair, flow)
    ax.set_title(
        "State migrations and response directions",
        loc="center", fontsize=SUBTITLE_SIZE, fontweight="bold", pad=14,
    )
    draw_response_direction_key(ax, group_summary, jacobian)


def draw_response_plane(
    ax: plt.Axes,
    pair: pd.DataFrame,
    group_summary: pd.DataFrame,
    jacobian: pd.DataFrame,
    zoom_limits: float | None = None,
    panel_subtitle: str = "Observed response directions\nand thermal constraints",
) -> None:
    required = {"annual_group", "R_mean", "R_Amp"}
    if not required.issubset(pair.columns):
        raise ValueError("Panel a pair-response table is incomplete.")
    if not {"annual_group", "R_mean", "R_Amp", "theta_deg", "theta_ci_low", "theta_ci_high"}.issubset(
        group_summary.columns
    ):
        raise ValueError("Panel a group summary is incomplete.")
    expected_channels = {
        "persistent_mean_loading", "relative_amplitude_damping",
        "weaker_effective_removal",
    }
    if not expected_channels.issubset(set(jacobian["channel"].astype(str))):
        raise ValueError("Panel a thermal-balance sign-direction table is incomplete.")

    if zoom_limits is None:
        values = np.concatenate(
            [pair["R_mean"].to_numpy(float), pair["R_Amp"].to_numpy(float)]
        )
        values = values[np.isfinite(values)]
        extent = max(float(np.max(np.abs(values))) * 1.14, 0.5)
    else:
        # Robust zoomed response plane: the viewport concentrates on the
        # central response range so the empirical group-mean arrows remain
        # legible.  No computation or cohort rule changes; points outside the
        # viewport are simply not shown in the main panel.
        extent = float(zoom_limits)
    ax.axhline(0.0, color="#A49D96", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#A49D96", linewidth=0.8, zorder=1)
    for group in ("UHI", "UCI"):
        subset = pair[pair["annual_group"].eq(group)]
        ax.scatter(
            subset["R_mean"], subset["R_Amp"], s=10, marker="o",
            color=group_color(group), alpha=0.12, edgecolors="none",
            rasterized=True, zorder=2,
        )

    # Theory constraints: longer than the observed UHI/UCI arrows, drawn in a
    # lighter grey so they stay subordinate, with every label placed in front
    # of its arrowhead.
    theory_color = "#9A9A9A"
    arrow_style = dict(arrowstyle="-|>", linewidth=1.6, linestyle="--", mutation_scale=11)
    ax.annotate("", xy=(0.55 * extent, 0.0), xytext=(0, 0),
                arrowprops={**arrow_style, "color": theory_color}, zorder=3)
    ax.text(
        0.55 * extent, -0.02 * extent, "persistent loading\n" + r"($F_0\uparrow$)",
        color=theory_color, ha="center", va="top",
        fontsize=ANNOT_SIZE - 0.8, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.6),
        zorder=9,
    )
    ax.annotate("", xy=(0.0, -0.55 * extent), xytext=(0, 0),
                arrowprops={**arrow_style, "color": theory_color}, zorder=3)
    ax.text(
        -0.06 * extent, -0.66 * extent,
        "amplitude damping\n" + r"($F_1\downarrow$ or $C\uparrow$)",
        color=theory_color, ha="center", va="top",
        fontsize=ANNOT_SIZE - 0.8, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.6),
        zorder=9,
    )
    ax.annotate("", xy=(0.44 * extent, 0.39 * extent), xytext=(0, 0),
                arrowprops={**arrow_style, "color": theory_color}, zorder=3)
    ax.text(
        0.44 * extent, 0.46 * extent, "weaker removal\n" + r"($\lambda\downarrow$)",
        color=theory_color, ha="center", va="bottom",
        fontsize=ANNOT_SIZE - 0.8, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.6),
        zorder=9,
    )
    ax.text(
        0.025, 0.975, "Linear thermal balance: direction only",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=ANNOT_SIZE - 0.2, fontweight="bold", color=COLOR_THEORY,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.0),
        zorder=8,
    )

    for row in group_summary.itertuples(index=False):
        group = str(row.annual_group)
        color = group_color(group)
        radius = min(max(1.45 * float(row.rho_group_mean_vector), 0.14 * extent),
                     0.48 * extent)
        ax.add_patch(
            Wedge(
                (0, 0), radius, float(row.theta_ci_low), float(row.theta_ci_high),
                width=0.68 * radius, facecolor=color, edgecolor="none",
                alpha=0.14, zorder=1.5,
            )
        )
        # Observed group-mean arrows keep their true empirical lengths; the
        # zoomed viewport, not any length scaling, makes them legible.
        arrow = FancyArrowPatch(
            (0, 0), (float(row.R_mean), float(row.R_Amp)), arrowstyle="-|>",
            mutation_scale=12, linewidth=1.6, color=color,
            path_effects=[path_effects.withStroke(linewidth=2.6, foreground="white")],
            zorder=6,
        )
        ax.add_patch(arrow)
        ax.scatter(
            [float(row.R_mean)], [float(row.R_Amp)], s=42,
            marker="o", facecolor=color, edgecolor="white", linewidth=0.9,
            zorder=7,
        )
        # Angle labels sit directly beside the empirical arrows (no leader
        # lines), with extent-relative offsets so the zoomed main panel and
        # the full-range SI view both keep the labels clear of the arrowheads.
        if group == "UHI":
            # Swapped with the F0 theory label: the UHI angle annotation now
            # sits above the UHI arrowhead where the F0 label used to be.
            label_xy = (0.55 * extent, 0.13 * extent)
            label_ha, label_va = "center", "bottom"
        else:
            label_xy = (
                float(row.R_mean) + 0.10 * extent,
                float(row.R_Amp) - 0.18 * extent,
            )
            label_ha, label_va = "left", "top"
        ax.text(
            *label_xy,
            f"{group}: {float(row.theta_deg):.0f}°\n"
            f"[{float(row.theta_ci_low):.0f}°, {float(row.theta_ci_high):.0f}°]",
            ha=label_ha, va=label_va, color=color,
            fontsize=ANNOT_SIZE, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.2),
            zorder=8,
        )

    box = ax.get_position()
    figure_width, figure_height = ax.figure.get_size_inches()
    physical_ratio = (box.width * figure_width) / (box.height * figure_height)
    y_extent = max(extent, extent / physical_ratio)
    x_extent = physical_ratio * y_extent
    ax.set_xlim(-x_extent, x_extent)
    ax.set_ylim(-y_extent, y_extent)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.98, 0.02,
        "All observations were retained in the calculations;\n"
        "the display range was truncated for clarity.",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=ANNOT_SIZE - 2.0, color="#403B37",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8),
        zorder=9,
    )
    ax.set_xlabel(r"Mean response, $R_{mean}$ (°C)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"Amplitude response, $R_{Amp}$ (°C)", fontsize=AXIS_LABEL_SIZE)
    add_panel_heading(ax, "a", panel_subtitle)
    style_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color=COLOR_UHI,
                   markersize=5.8, label="UHI pairs"),
            Line2D([0], [0], marker="o", linestyle="none", color=COLOR_UCI,
                   markersize=5.8, label="UCI pairs"),
            Line2D([0], [0], color=COLOR_THEORY, linestyle="--", linewidth=1.4,
                   label="Thermal-balance constraint"),
        ],
        frameon=False, loc="upper left", bbox_to_anchor=(0.015, 0.865),
        fontsize=LEGEND_SIZE - 0.4, handlelength=1.7, labelspacing=0.32,
    )


def draw_panel_c(
    ax: plt.Axes,
    pair: pd.DataFrame,
    curve: pd.DataFrame,
    bins: pd.DataFrame,
    coefficient: pd.DataFrame,
    panel_label: str = "c",
    group_lookup: pd.DataFrame | None = None,
) -> None:
    pair_required = {"fwl_damping_residual", "fwl_timing_residual_h"}
    curve_required = {"x_fwl_damping", "fit_timing_h", "ci_low", "ci_high"}
    bin_required = {
        "x_mean", "x_ci_low", "x_ci_high", "y_mean", "y_ci_low", "y_ci_high"
    }
    if not pair_required.issubset(pair.columns):
        raise ValueError("Panel c FWL pair table is incomplete.")
    if not curve_required.issubset(curve.columns):
        raise ValueError("Panel c FWL curve table is incomplete.")
    if not bin_required.issubset(bins.columns):
        raise ValueError("Panel c FWL bin table is incomplete.")
    primary = coefficient[coefficient["term"].astype(str).eq("amplitude_damping")]
    if len(primary) != 1:
        raise ValueError("Panel c requires one frozen amplitude_damping coefficient.")
    row = primary.iloc[0]

    plot_pair = pair.copy()
    if "annual_group" not in plot_pair.columns and group_lookup is not None:
        if "pair_id" in plot_pair.columns and {"pair_id", "annual_group"}.issubset(group_lookup.columns):
            lookup = group_lookup[["pair_id", "annual_group"]].drop_duplicates("pair_id")
            plot_pair = plot_pair.merge(lookup, on="pair_id", how="left", validate="one_to_one")
    if "annual_group" not in plot_pair.columns:
        raise ValueError(
            "FWL pair table needs annual_group, or pair_id for a verified join, "
            "to apply the requested UHI/UCI colour encoding."
        )
    plot_pair["annual_group"] = (
        plot_pair["annual_group"].astype("string").str.upper().str.strip()
    )
    invalid_groups = set(plot_pair["annual_group"].dropna().astype(str)) - {"UHI", "UCI"}
    if invalid_groups or plot_pair["annual_group"].isna().any():
        raise ValueError(f"Invalid or missing FWL group labels: {sorted(invalid_groups)}")
    for group in ("UHI", "UCI"):
        subset = plot_pair.loc[plot_pair["annual_group"].astype(str).eq(group)]
        ax.scatter(
            subset["fwl_damping_residual"], subset["fwl_timing_residual_h"],
            s=17, marker="o", color=group_color(group), alpha=0.16,
            edgecolors="none", rasterized=True, zorder=1,
        )
    ordered = curve.sort_values("x_fwl_damping")
    ax.fill_between(
        ordered["x_fwl_damping"].to_numpy(float),
        ordered["ci_low"].to_numpy(float),
        ordered["ci_high"].to_numpy(float),
        color=COLOR_BAND, alpha=0.72, linewidth=0, zorder=2,
    )
    ax.plot(
        ordered["x_fwl_damping"], ordered["fit_timing_h"],
        color=COLOR_FIT, linewidth=1.2, zorder=3,
    )
    xerr = np.vstack(
        [bins["x_mean"] - bins["x_ci_low"], bins["x_ci_high"] - bins["x_mean"]]
    ).clip(min=0)
    yerr = np.vstack(
        [bins["y_mean"] - bins["y_ci_low"], bins["y_ci_high"] - bins["y_mean"]]
    ).clip(min=0)
    ax.errorbar(
        bins["x_mean"], bins["y_mean"], xerr=xerr, yerr=yerr, fmt="o",
        markersize=6.0, markerfacecolor="white", markeredgecolor=COLOR_THEORY,
        markeredgewidth=1.35, ecolor=COLOR_THEORY, elinewidth=1.15,
        capsize=2.7, zorder=5,
    )
    ax.axhline(0.0, color=COLOR_ZERO, linestyle="--", linewidth=0.85, alpha=0.75)
    ax.axvline(0.0, color=COLOR_ZERO, linestyle=":", linewidth=0.85, alpha=0.70)
    # Truncated display range: the pair cloud's 1-99 percentiles define the
    # axes; every observation remains in the calculations.
    xvals = plot_pair["fwl_damping_residual"].to_numpy(float)
    yvals = plot_pair["fwl_timing_residual_h"].to_numpy(float)
    x_lo, x_hi = np.nanquantile(xvals, [0.01, 0.99])
    y_lo, y_hi = np.nanquantile(yvals, [0.01, 0.99])
    x_pad = 0.05 * max(x_hi - x_lo, 1.0e-6)
    y_pad = 0.05 * max(y_hi - y_lo, 1.0e-6)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.text(
        0.035, 0.955,
        rf"$\beta_A={float(row['estimate']):.2f}$ h SD$^{{-1}}$" + "\n"
        + rf"95% CI [{float(row['ci_low']):.2f}, {float(row['ci_high']):.2f}]" + "\n"
        + rf"$n={int(row['n_pairs'])}$; blocks={int(row['n_blocks'])}",
        transform=ax.transAxes, ha="left", va="top", fontsize=ANNOT_SIZE,
        color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=1.6),
        zorder=7,
    )
    ax.set_xlabel(r"Amplitude damping, $-z(R_{Amp})$ (adjusted)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("HW−NHW timing change\n(adjusted, h)", fontsize=AXIS_LABEL_SIZE)
    add_panel_heading(ax, panel_label, "Timing change beyond\nmean response")
    style_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_FIT, linewidth=1.2,
                   label="Fitted partial association\n(mean response removed)"),
            Line2D([0], [0], marker="o", linestyle="none", color=COLOR_THEORY,
                   markerfacecolor="white", markersize=6.0,
                   label="Adjusted quintile means"),
            Line2D([0], [0], marker="o", linestyle="none", color=COLOR_UHI,
                   markersize=5.3, label="UHI pairs"),
            Line2D([0], [0], marker="o", linestyle="none", color=COLOR_UCI,
                   markersize=5.3, label="UCI pairs"),
        ],
        frameon=False, loc="lower right", fontsize=LEGEND_SIZE - 0.5,
        handlelength=1.8, borderaxespad=0.6, ncol=2,
    )


def _draw_dynamic_component(
    ax: plt.Axes,
    summary: pd.DataFrame,
    component: str,
    ylabel: str,
    show_xlabels: bool,
) -> None:
    component_data = summary[summary["component"].astype(str).eq(component)].copy()
    if component_data.empty:
        raise ValueError(f"Panel d lacks dynamic summary for {component}.")
    for group in ("UHI", "UCI"):
        subset = component_data[component_data["annual_group"].eq(group)].sort_values(
            "phase_order"
        )
        if len(subset) != 7:
            raise ValueError(f"Panel d requires seven {component} phases for {group}.")
        x = subset["phase_order"].to_numpy(float)
        y = subset["estimate"].to_numpy(float)
        error = np.vstack(
            [y - subset["ci_low"].to_numpy(float),
             subset["ci_high"].to_numpy(float) - y]
        ).clip(min=0)
        color = group_color(group)
        ax.plot(x, y, color=color, linewidth=2.0, zorder=3)
        ax.errorbar(
            x, y, yerr=error, fmt="o", markersize=5.9,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8,
            ecolor=color, elinewidth=1.15, capsize=2.6, zorder=4,
        )
    ax.axhline(0.0, color=COLOR_ZERO, linestyle="--", linewidth=0.85, alpha=0.75)
    ax.axvline(1.5, color="#8D847C", linestyle=":", linewidth=0.95, alpha=0.85)
    ax.axvline(3.5, color="#8D847C", linestyle=":", linewidth=0.95, alpha=0.85)
    ax.set_xlim(-0.35, 6.35)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE - 0.4)
    ax.set_xticks(np.arange(7))
    if show_xlabels:
        ax.set_xticklabels(
            ["−2 d", "−1 d", "Onset", "Final\nHW day", "+1 d", "+2 d", "+3 d"],
            fontsize=TICK_SIZE - 0.5,
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    style_axis(ax)


def draw_panel_d(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    required = {
        "annual_group", "component", "phase", "phase_order", "estimate",
        "ci_low", "ci_high", "n_pairs",
    }
    if not required.issubset(summary.columns):
        raise ValueError("Panel d event-state summary is incomplete.")
    _draw_dynamic_component(ax_top, summary, "R_mean", r"$R_{mean}$ (°C)", False)
    _draw_dynamic_component(ax_bottom, summary, "R_Amp", r"$R_{Amp}$ (°C)", True)
    ax_bottom.set_xlabel("Time relative to heatwave", fontsize=AXIS_LABEL_SIZE)
    ax_top.set_title(
        "Event-centred state dynamics", loc="center", fontsize=SUBTITLE_SIZE,
        fontweight="bold", pad=14,
    )
    ax_top.text(
        -0.118, 1.038, "d", transform=ax_top.transAxes, ha="left", va="bottom",
        fontsize=PANEL_LABEL_SIZE, fontweight="bold", clip_on=False,
    )
    ax_top.text(
        1.5, 0.97, "HW onset", transform=ax_top.get_xaxis_transform(),
        ha="center", va="top", fontsize=ANNOT_SIZE - 0.2, color=COLOR_ZERO,
    )
    ax_top.text(
        3.5, 0.97, "HW end", transform=ax_top.get_xaxis_transform(),
        ha="center", va="top", fontsize=ANNOT_SIZE - 0.2, color=COLOR_ZERO,
    )
    ax_top.legend(
        handles=[
            Line2D([0], [0], color=COLOR_UHI, marker="o", markersize=5.8,
                   linewidth=2.0, label="UHI regime"),
            Line2D([0], [0], color=COLOR_UCI, marker="o", markersize=5.8,
                   linewidth=2.0, label="UCI regime"),
        ],
        frameon=False, loc="upper left", fontsize=LEGEND_SIZE,
        handlelength=1.8, borderaxespad=0.55,
    )


def draw_standalone_sensitivity(ax: plt.Axes, pi_table: pd.DataFrame) -> None:
    required = {
        "Pi_1", "removal_weight_w_lambda", "storage_weight_w_C", "timing_weight_w_phi"
    }
    if not required.issubset(pi_table.columns):
        raise ValueError("Standalone thermal-balance table is incomplete.")
    table = (
        pi_table[list(required)].replace([np.inf, -np.inf], np.nan).dropna()
        .sort_values("Pi_1")
    )
    if table.empty or (table["Pi_1"] <= 0).any():
        raise ValueError("Standalone sensitivity requires finite positive Pi values.")
    ax.axvspan(table["Pi_1"].min(), 1.0, color="#F3E5C8", alpha=0.25, zorder=0)
    ax.axvspan(1.0, table["Pi_1"].max(), color="#EBD5CB", alpha=0.22, zorder=0)
    ax.plot(table["Pi_1"], table["removal_weight_w_lambda"],
            color=COLOR_REMOVAL, linewidth=2.8, zorder=4)
    ax.plot(table["Pi_1"], table["storage_weight_w_C"],
            color=COLOR_STORAGE, linewidth=2.8, zorder=4)
    ax.plot(table["Pi_1"], table["timing_weight_w_phi"],
            color=COLOR_TIMING, linewidth=2.5, linestyle="--", zorder=4)
    ax.axvline(1.0, color=COLOR_ZERO, linestyle=":", linewidth=1.25, zorder=2)
    tau_cross = 24.0 / (2.0 * np.pi)
    ax.scatter([1.0], [0.5], s=46, facecolor="white", edgecolor=COLOR_ZERO,
               linewidth=1.1, zorder=6)
    ax.annotate(
        rf"$\Pi=1$" + "\n" + rf"$\tau={tau_cross:.2f}$ h",
        xy=(1.0, 0.5), xytext=(12, 14), textcoords="offset points",
        ha="left", va="bottom", fontsize=ANNOT_SIZE, color=COLOR_ZERO,
        path_effects=[path_effects.withStroke(linewidth=3, foreground="white")], zorder=7,
    )
    ax.text(0.027, 0.94, r"Removal $w_{\lambda}$", transform=ax.transAxes,
            ha="left", va="top", color="#87570B", fontsize=ANNOT_SIZE + 0.4,
            fontweight="bold")
    ax.text(0.973, 0.94, r"Storage $w_C$", transform=ax.transAxes,
            ha="right", va="top", color="#873529", fontsize=ANNOT_SIZE + 0.4,
            fontweight="bold")
    ax.text(0.30, 0.39, r"Timing $w_{\phi}$", transform=ax.transAxes,
            ha="center", va="bottom", color=COLOR_TIMING, fontsize=ANNOT_SIZE)
    ax.set_xscale("log")
    ax.set_xlim(float(table["Pi_1"].min()), float(table["Pi_1"].max()))
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel(r"Thermal-memory number, $\Pi=\omega C/\lambda$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Dimensionless sensitivity weight", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("Linear thermal-balance sensitivity", loc="center",
                 fontsize=SUBTITLE_SIZE, fontweight="bold", pad=14)
    style_axis(ax)


def save_jpg(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path, dpi=600, format="jpg", facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0, "optimize": True},
    )


def save_preview_webp(jpg_path: Path, webp_path: Path) -> None:
    with Image.open(jpg_path) as image:
        preview = image.convert("RGB")
        preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        preview.save(webp_path, format="WEBP", quality=72, method=6)


def verify_formal_image(path: Path, expected_inches: Tuple[float, float]) -> Dict:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0, 0))
        if image.mode != "RGB":
            raise ValueError(f"Formal image is not RGB: {image.mode}")
        if min(dpi) < 599:
            raise ValueError(f"Formal image DPI metadata is below 600: {dpi}")
        expected_pixels = tuple(round(value * 600) for value in expected_inches)
        if abs(image.width - expected_pixels[0]) > 2 or abs(image.height - expected_pixels[1]) > 2:
            raise ValueError(
                f"Fixed canvas mismatch for {path.name}: "
                f"{(image.width, image.height)} versus {expected_pixels}."
            )
        return {
            "mode": image.mode,
            "width_px": int(image.width),
            "height_px": int(image.height),
            "dpi": [float(dpi[0]), float(dpi[1])],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Figure 3 using Figure-2-matched typography and the retained "
            "linear thermal-balance sensitivity figure."
        )
    )
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument(
        "--outdir", default=None,
        help="New empty output directory; default is a timestamped project directory.",
    )
    parser.add_argument(
        "--allow-test-bundle", action="store_true",
        help="Allow a TEST bundle for visual QA only.",
    )
    return parser.parse_args()


def resolve_output_dir(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    return (DEFAULT_PLOT_ROOT / stamp).resolve()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    input_hashes_before = {
        name: sha256_file(analysis_dir / name) for name in REQUIRED_FILES
        if (analysis_dir / name).is_file()
    }
    manifest, success = validate_analysis_bundle(analysis_dir, args.allow_test_bundle)
    tables = read_tables(analysis_dir)
    formal_audit = validate_formal_results(tables, args.allow_test_bundle)
    outdir = resolve_output_dir(args.outdir)
    prepare_output_dir(outdir)

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": TICK_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": SUBTITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": FRAME_LW,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=MAIN_CANVAS_IN, dpi=600, facecolor="white")
        grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        fig.subplots_adjust(
            left=0.080, right=0.965, bottom=0.140, top=0.880,
            wspace=0.27,
        )
        draw_response_plane(
            ax_a, tables["pair"], tables["group"], tables["jacobian"],
            zoom_limits=0.9,
        )
        draw_panel_c(
            ax_b, tables["fwl_pair"], tables["fwl_curve"], tables["fwl_bins"],
            tables["timing_coefficient"], panel_label="b",
            group_lookup=tables["pair"],
        )
        main_path = outdir / MAIN_FILENAME
        save_jpg(fig, main_path)
        plt.close(fig)

        theory_fig = plt.figure(figsize=THEORY_CANVAS_IN, dpi=600, facecolor="white")
        theory_ax = theory_fig.add_subplot(111)
        theory_fig.subplots_adjust(left=0.225, right=0.985, bottom=0.095, top=0.90)
        draw_standalone_sensitivity(theory_ax, tables["pi"])
        theory_path = outdir / THEORY_FILENAME
        save_jpg(theory_fig, theory_path)
        plt.close(theory_fig)

        # SI candidate: the full-range pair-level response distribution that
        # the zoomed main panel does not show.
        si_full_fig = plt.figure(
            figsize=(180.0 / 25.4, 120.0 / 25.4), dpi=600, facecolor="white"
        )
        si_full_ax = si_full_fig.add_subplot(111)
        si_full_fig.subplots_adjust(
            left=0.225, right=0.985, bottom=0.095, top=0.90,
        )
        draw_response_plane(
            si_full_ax, tables["pair"], tables["group"], tables["jacobian"],
            panel_subtitle="Full-range pair-level responses",
        )
        si_full_path = outdir / SI_FULL_FILENAME
        save_jpg(si_full_fig, si_full_path)
        plt.close(si_full_fig)

    main_preview = outdir / MAIN_PREVIEW_FILENAME
    theory_preview = outdir / THEORY_PREVIEW_FILENAME
    si_full_preview = outdir / SI_FULL_PREVIEW_FILENAME
    save_preview_webp(main_path, main_preview)
    save_preview_webp(theory_path, theory_preview)
    save_preview_webp(si_full_path, si_full_preview)
    input_hashes_after = {
        name: sha256_file(analysis_dir / name) for name in REQUIRED_FILES
        if (analysis_dir / name).is_file()
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("READ_ONLY_INPUT_MODIFIED_DURING_PLOTTING")
    image_info = {
        MAIN_FILENAME: verify_formal_image(main_path, MAIN_CANVAS_IN),
        THEORY_FILENAME: verify_formal_image(theory_path, THEORY_CANVAS_IN),
    }
    plot_manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_directory": str(analysis_dir),
        "analysis_manifest_sha256": sha256_file(
            analysis_dir / "fig3_dynamic_analysis_manifest.json"
        ),
        "analysis_run_mode": manifest.get("run_mode"),
        "formal_input_sha256": input_hashes_before,
        "analysis_state_pairs": success.get("state_pairs"),
        "formal_result_audit": formal_audit,
        "event_centred_panel_in_main": False,
        "thermal_model_name": "linear thermal-balance model",
        "fwl_group_specific_slopes_fitted": False,
        "fwl_point_group_encoding": "UHI/UCI colours; pooled formal fit retained",
        "main_image": MAIN_FILENAME,
        "main_image_sha256": sha256_file(main_path),
        "main_preview": MAIN_PREVIEW_FILENAME,
        "main_preview_sha256": sha256_file(main_preview),
        "standalone_sensitivity_image": THEORY_FILENAME,
        "standalone_sensitivity_image_sha256": sha256_file(theory_path),
        "standalone_sensitivity_preview": THEORY_PREVIEW_FILENAME,
        "standalone_sensitivity_preview_sha256": sha256_file(theory_preview),
        "si_full_range_image": SI_FULL_FILENAME,
        "si_full_range_image_sha256": sha256_file(si_full_path),
        "si_full_range_preview": SI_FULL_PREVIEW_FILENAME,
        "si_full_range_preview_sha256": sha256_file(si_full_preview),
        "panel_a_zoom": "robust zoomed response plane, extent 0.9 C",
        "panel_a_observed_arrows_true_length": True,
        "display_truncation": (
            "all observations retained in the calculations; display ranges "
            "truncated for clarity (panel a: extent 0.9 C; panel b: 1-99 "
            "percentile of pair-level FWL values)"
        ),
        "image_info": image_info,
        "main_fixed_canvas_inches": list(MAIN_CANVAS_IN),
        "standalone_fixed_canvas_inches": list(THEORY_CANVAS_IN),
        "bbox_inches_tight_used": False,
        "plot_only": True,
    }
    manifest_path = outdir / "fig3_dynamic_plot_manifest.json"
    write_json(manifest_path, plot_manifest)
    write_json(
        outdir / "_SUCCESS.json",
        {
            "status": "SUCCESS",
            "script_version": SCRIPT_VERSION,
            "main_image": MAIN_FILENAME,
            "main_image_sha256": sha256_file(main_path),
            "standalone_sensitivity_image": THEORY_FILENAME,
            "standalone_sensitivity_image_sha256": sha256_file(theory_path),
            "si_full_range_image": SI_FULL_FILENAME,
            "si_full_range_image_sha256": sha256_file(si_full_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(f"SUCCESS: {main_path}")
    print(f"Retained standalone sensitivity figure: {theory_path}")
    print(f"QA previews: {main_preview}; {theory_preview}")


if __name__ == "__main__":
    main()
