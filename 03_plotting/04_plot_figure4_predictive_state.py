#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot-only Figure 4: irreducible two-component predictive information.

This script is plotting-only. It reads frozen source-data tables written by
``02_figure_analysis/05_analyze_figure4_prediction.py``; it does not refit a model,
recompute cross-validation, rerun a bootstrap, or alter an input file.

A formal run creates a three-panel Nature Geoscience main figure (a-c) and a
standalone SI threshold panel.  Panel c annotates only metrics already present
in the formal performance table; it never fits a calibration slope or computes
RMSE from the displayed points.  The SI panel uses the frozen attenuation
estimate and confidence interval, avoiding plotting-stage bootstrap summaries.
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

import numpy as np
import pandas as pd


_MPL_CACHE = Path(tempfile.gettempdir()) / "mpl_predictive_figure4"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image


SCRIPT_VERSION = "plot_figure4_predictive_state"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = (
    PROJECT_ROOT / "outputs/analysis/figure4_predictive_state"
)
DEFAULT_TABLES_DIR = DEFAULT_ANALYSIS_DIR / "tables"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/figures/figure4_predictive_state"
)

FORMAL_DPI = 600
OUTPUT_NAMES = {
    "final": "figure4_predictive_state.jpg",
    "final_preview": "figure4_predictive_state_preview.webp",
    "si": "figure4_residual_thresholds_si.jpg",
    "si_preview": "figure4_residual_thresholds_si_preview.webp",
}

TABLE_FILES = {
    "a": "figure4a_joint_performance_source.csv",
    "b": "figure4b_incremental_value_source.csv",
    "c": "figure4c_oof_night_predictions_source.csv",
    "d": "figure4d_attenuation_threshold_source.csv",
    "performance": "primary_model_performance_complete.csv",
    "robustness": "robustness_decision_matrix.csv",
    "si_points": "figure4si_residual_points_source.csv",
    "si_influence": "figure4si_residual_influence_source.csv",
}

MM_TO_INCH = 1.0 / 25.4
FIGURE_WIDTH_MM = 180.0
MAIN_HEIGHT_MM = 150.0
LEGACY_HEIGHT_MM = 126.0
SI_HEIGHT_MM = 78.0

# Model-comparison palette (ink-ochre), independent of the red/blue UHI/UCI
# semantics in Figures 1--3.  Green is excluded to avoid ecological/vegetation
# connotations.  M2 is the one-dimensional benchmark (ochre), M3 the core
# result (deep ink blue), M4 the equivalent day-night coordinate (taupe), and
# M0/M1 recede in grey/charcoal.
# Teal-berry-slate model palette: deliberately disjoint from the colour
# families used in Figures 1-3 (red, blue, purple, green, orange, browns).
# M2 is the one-dimensional benchmark (teal), M3 the core result (berry),
# M4 the equivalent day-night coordinate (slate), M0/M1 stay neutral grey.
MODEL_COLORS = {
    "M0": "#B8B8B8",
    "M1": "#555555",
    "M2": "#2A8F8F",
    "M3": "#9E2A63",
    "M4": "#5F6E7E",
}
MODEL_LABELS = {
    "M0": "Climatology",
    "M1": "Daytime scalar",
    "M2": "Training-optimized rank-one",
    "M3": "Two-component state",
    "M4": "Day–night coordinate",
}
MODEL_ORDER = ("M0", "M1", "M2", "M3", "M4")
PANEL_A_LABELS = {
    "M0": "M0\nClimatology",
    "M1": "M1\nDaytime scalar",
    "M2": "M2\nTraining-optimized\nrank-one",
    "M3": "M3\nMean–amplitude state",
    "M4": "M4\nDay–night pair",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the final and comparison Figure 4 layouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Render deterministic synthetic tables in a temporary directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def file_inventory(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
        rows.append(
            {
                "path": str(resolved),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(resolved),
            }
        )
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def read_formal_tables(
    tables_dir: Path,
) -> tuple[dict[str, pd.DataFrame], list[Path], pd.DataFrame]:
    tables_dir = tables_dir.expanduser().resolve()
    analysis_dir = tables_dir.parent
    if not tables_dir.is_dir():
        raise FileNotFoundError(f"Analysis tables directory not found: {tables_dir}")
    complete = analysis_dir / "ANALYSIS_COMPLETE"
    manifest_path = analysis_dir / "run_manifest.json"
    if not complete.is_file() or complete.read_text(encoding="utf-8").strip() != "COMPLETE":
        raise RuntimeError(f"Analysis completion marker is missing or invalid: {complete}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Analysis manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or not manifest.get(
        "plotting_separated", False
    ):
        raise RuntimeError("Analysis manifest is not a complete plotting-separated run.")

    expected_hashes = manifest.get("output_sha256", {})
    inputs = [complete, manifest_path]
    result: dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        path = tables_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required source table not found: {path}")
        relative = f"tables/{filename}"
        expected = expected_hashes.get(relative)
        if not expected:
            raise RuntimeError(f"Manifest has no SHA-256 for {relative}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch for read-only source table: {path}")
        dtype = {"pair_id": str} if key in {"c", "si_points"} else None
        result[key] = pd.read_csv(path, dtype=dtype, low_memory=False)
        inputs.append(path)

    validate_tables(result)
    inventory = file_inventory(inputs)
    return result, inputs, inventory


def validate_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    require_columns(
        tables["a"],
        ["model", "model_label", "estimate", "ci_low", "ci_high"],
        "joint performance source",
    )
    require_columns(
        tables["b"],
        ["metric", "outcome_label", "estimate", "ci_low", "ci_high"],
        "incremental skill source",
    )
    require_columns(
        tables["c"],
        ["pair_id", "model", "Rn_obs", "Rn_pred", "highlight"],
        "OOF night prediction source",
    )
    require_columns(
        tables["d"],
        [
            "threshold_label",
            "min_events",
            "primary_cohort",
            "n_pairs",
            "slope_M1",
            "slope_M3",
            "attenuation_estimate",
            "attenuation_ci_low",
            "attenuation_ci_high",
            "m3_minus_m2_joint_q2",
            "m3_minus_m2_joint_q2_ci_low",
            "m3_minus_m2_joint_q2_ci_high",
        ],
        "event-threshold source",
    )
    require_columns(
        tables["performance"],
        ["task", "model", "metric", "estimate", "ci_low", "ci_high"],
        "complete model-performance source",
    )
    require_columns(
        tables["robustness"],
        ["scenario", "estimate", "ci_low", "ci_high", "source"],
        "robustness source",
    )
    require_columns(
        tables["si_points"],
        [
            "pair_id",
            "model",
            "D_E",
            "Rn_obs",
            "Rn_pred",
            "night_residual",
            "highlight",
        ],
        "residual-point source",
    )
    require_columns(
        tables["si_influence"],
        [
            "removal_type",
            "removed",
            "slope_M1",
            "intercept_M1",
            "slope_M3",
            "intercept_M3",
            "attenuation",
        ],
        "residual-influence source",
    )

    if set(tables["a"]["model"].astype(str)) != set(MODEL_ORDER):
        raise RuntimeError("Joint performance source must contain M0--M4 exactly.")
    if set(tables["b"]["metric"].astype(str)) != {"joint_q2", "Rx_q2", "Rn_q2"}:
        raise RuntimeError("Incremental source must contain joint, daytime and night-time Q2.")
    if set(tables["c"]["model"].astype(str)) != {"M2", "M3"}:
        raise RuntimeError("OOF source must compare M2 and M3 exactly.")
    oof = tables["c"].copy()
    if oof.duplicated(["pair_id", "model"]).any():
        raise RuntimeError("OOF source has duplicate pair/model rows.")
    pair_sets = {
        model: set(oof.loc[oof["model"].astype(str).eq(model), "pair_id"].astype(str))
        for model in ("M2", "M3")
    }
    if pair_sets["M2"] != pair_sets["M3"]:
        raise RuntimeError("M2 and M3 OOF predictions must use the identical qualified pairs.")
    if not {"M1", "M3"}.issubset(set(tables["si_points"]["model"].astype(str))):
        raise RuntimeError("Residual source must contain M1 and M3.")
    primary = tables["d"].loc[tables["d"]["primary_cohort"].astype(bool)]
    if len(primary) != 1:
        raise RuntimeError("Threshold source must identify exactly one primary cohort.")
    if int(primary.iloc[0]["n_pairs"]) != len(pair_sets["M3"]):
        raise RuntimeError(
            "Primary threshold count does not match the qualified OOF prediction sample."
        )


def create_timestamped_output(base: Path) -> Path:
    base = base.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base / f"run_{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = base / f"run_{stamp}_v{suffix}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 6.3,
            "axes.titlesize": 7.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#222222", pad=2.2)
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color="#D8D8D8",
            linewidth=0.45,
            linestyle=(0, (1.4, 2.2)),
            alpha=0.75,
            zorder=0,
        )


def panel_heading(ax: plt.Axes, label: str, subtitle: str, label_x: float = -0.04) -> None:
    ax.set_title(
        subtitle, loc="center", pad=8.0, fontsize=7.8,
        fontweight="bold", linespacing=1.02,
    )
    panel_title = ax.set_title(
        label,
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        color="#111111",
        pad=8.0,
    )
    panel_title.set_x(label_x)
    panel_title.set_ha("right")
    panel_title.set_clip_on(False)


def error_array(row: pd.Series, value: str = "estimate", low: str = "ci_low", high: str = "ci_high") -> np.ndarray:
    estimate = float(row[value])
    return np.array(
        [[max(0.0, estimate - float(row[low]))], [max(0.0, float(row[high]) - estimate)]]
    )


def threshold_labels(data: pd.DataFrame) -> list[str]:
    labels = []
    for row in data.itertuples(index=False):
        prefix = (
            "Baseline"
            if bool(row.primary_cohort)
            else f"≥{int(row.min_events)} events"
        )
        labels.append(f"{prefix}\n(n={int(row.n_pairs)})")
    return labels

def plot_joint_performance(
    ax: plt.Axes,
    data: pd.DataFrame,
    panel: str,
    subtitle: str,
    compact_labels: bool = False,
) -> None:
    y_positions = np.arange(len(MODEL_ORDER))[::-1]
    for y, model in zip(y_positions, MODEL_ORDER):
        row = data.loc[data["model"].astype(str).eq(model)].iloc[0]
        ax.errorbar(
            float(row["estimate"]),
            y,
            xerr=error_array(row),
            fmt="o",
            ms=4.6,
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.05,
            capsize=2.2,
            capthick=0.9,
            zorder=3,
        )
    ax.axvline(0, color="#5A5A5A", linestyle="--", linewidth=0.85, zorder=1)
    tick_labels = (
        [MODEL_LABELS[m] for m in MODEL_ORDER]
        if compact_labels
        else [PANEL_A_LABELS[m] for m in MODEL_ORDER]
    )
    ax.set_yticks(y_positions, tick_labels)
    for tick_label in ax.get_yticklabels():
        tick_label.set_linespacing(0.92)
    ax.set_ylim(-0.55, len(MODEL_ORDER) - 0.45)
    ax.set_xlabel(r"Joint predictive skill ($Q^2$)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    panel_heading(ax, panel, subtitle)
    format_axis(ax, "x")


def plot_incremental_skill(
    ax: plt.Axes,
    data: pd.DataFrame,
    panel: str,
    subtitle: str,
) -> None:
    order = ("joint_q2", "Rx_q2", "Rn_q2")
    labels = {
        "joint_q2": "Joint\nresponse",
        "Rx_q2": "Daytime\nresponse",
        "Rn_q2": "Night-time\nresponse",
    }
    y_positions = np.arange(len(order))[::-1]
    for y, metric in zip(y_positions, order):
        row = data.loc[data["metric"].astype(str).eq(metric)].iloc[0]
        ax.errorbar(
            float(row["estimate"]),
            y,
            xerr=error_array(row),
            fmt="D",
            ms=4.5,
            markerfacecolor=MODEL_COLORS["M3"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            color=MODEL_COLORS["M3"],
            ecolor=MODEL_COLORS["M3"],
            elinewidth=1.1,
            capsize=2.2,
            zorder=3,
        )
    ax.axvline(0, color="#5A5A5A", linestyle="--", linewidth=0.85, zorder=1)
    ax.set_yticks(y_positions, [labels[m] for m in order])
    for tick_label in ax.get_yticklabels():
        tick_label.set_linespacing(0.92)
    ax.set_ylim(-0.55, len(order) - 0.45)
    ax.set_xlabel(r"Mean–amplitude minus rank-one ($\Delta Q^2$)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    panel_heading(ax, panel, subtitle)
    format_axis(ax, "x")


def square_limits(x: np.ndarray, y: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    values = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    low, high = float(values.min()), float(values.max())
    width = max(high - low, 1.0)
    return low - pad_fraction * width, high + pad_fraction * width


def formal_night_metric_text(performance: pd.DataFrame, model: str) -> str:
    """Return only upstream-reported metrics; never estimate from OOF points."""
    subset = performance.loc[performance["model"].astype(str).eq(model)].copy()
    if "task" in subset.columns and subset["task"].astype(str).eq("vector").any():
        subset = subset.loc[subset["task"].astype(str).eq("vector")].copy()
    metric_lookup = {
        str(row.metric).strip().lower(): float(row.estimate)
        for row in subset.itertuples(index=False)
        if np.isfinite(float(row.estimate))
    }

    def first(candidates: tuple[str, ...]) -> float | None:
        for candidate in candidates:
            if candidate.lower() in metric_lookup:
                return metric_lookup[candidate.lower()]
        return None

    q2 = first(("Rn_q2", "night_q2", "night-time_q2"))
    if q2 is None:
        raise RuntimeError(f"Formal night-time Q2 is missing for {model}.")
    rmse = first(("Rn_rmse", "night_rmse", "night-time_rmse", "rmse_rn"))
    slope = first(
        (
            "Rn_calibration_slope",
            "night_calibration_slope",
            "night-time_calibration_slope",
            "calibration_slope_rn",
        )
    )
    lines = [rf"$Q^2={q2:.2f}$"]
    if rmse is not None:
        lines.append(f"RMSE={rmse:.2f} °C")
    if slope is not None:
        lines.append(f"slope={slope:.2f}")
    return "  |  ".join(lines)


def plot_oof_final(
    ax: plt.Axes,
    data: pd.DataFrame,
    panel: str,
    subtitle: str,
    performance: pd.DataFrame | None = None,
) -> None:
    """Plot M2 and M3 OOF predictions in two internal matched facets.

    The two facets use identical numerical limits and ticks. No observations
    are removed, shifted or jittered, and no additional model is fitted.
    """

    # The outer axis defines one evidence unit. Its two matched facets are
    # deliberately grouped under one panel label, subtitle and pair of shared
    # axis labels.
    ax.set_axis_off()
    panel_heading(ax, panel, subtitle, label_x=-0.045)

    model_styles = {
        "M2": {
            "title": "M2  Training-optimized rank-one",
            "marker": "o",
            "color": MODEL_COLORS["M2"],
            "alpha": 0.30,
            "size": 9,
        },
        "M3": {
            "title": "M3  Two-component",
            "marker": "D",
            "color": MODEL_COLORS["M3"],
            "alpha": 0.32,
            "size": 9,
        },
    }

    # Pooled limits ensure an exact visual comparison between M2 and M3.
    pooled_observed = pd.to_numeric(
        data["Rn_obs"], errors="coerce"
    ).to_numpy(float)

    pooled_predicted = pd.to_numeric(
        data["Rn_pred"], errors="coerce"
    ).to_numpy(float)

    low, high = square_limits(
        pooled_observed,
        pooled_predicted,
        pad_fraction=0.06,
    )

    # Match the physical size and horizontal column boundaries of each
    # internal facet to panels a and b. The shared heading and shared axis
    # labels keep M2 and M3 grouped as one paired comparison in panel c.
    facet_positions = {
        "M2": [0.000, 0.045, 0.427, 0.90],
        "M3": [0.573, 0.045, 0.427, 0.90],
    }

    inner_axes: dict[str, plt.Axes] = {}

    for model in ("M2", "M3"):
        style = model_styles[model]
        subset = data.loc[
            data["model"].astype(str).eq(model)
        ].copy()

        facet = ax.inset_axes(facet_positions[model])
        inner_axes[model] = facet

        facet.scatter(
            subset["Rn_obs"],
            subset["Rn_pred"],
            s=style["size"],
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="none",
            alpha=style["alpha"],
            rasterized=True,
            zorder=2,
        )

        # Identical 1:1 reference line in both facets.
        facet.plot(
            [low, high],
            [low, high],
            color="#5A5A5A",
            linestyle="--",
            linewidth=0.85,
            zorder=1,
        )

        facet.set_xlim(low, high)
        facet.set_ylim(low, high)

        # Identical numerical limits and equal physical aspect preserve the
        # direct observed-versus-predicted comparison in both facets.
        facet.set_aspect("equal", adjustable="box")

        metric_line = (
            formal_night_metric_text(performance, model)
            if performance is not None else ""
        )
        # Method and frozen performance metrics are internal facet headers,
        # not independent panel subtitles.
        facet.text(
            0.035,
            0.965,
            style["title"] + ("\n" + metric_line if metric_line else ""),
            transform=facet.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="semibold",
            color=style["color"],
            linespacing=1.18,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2),
            zorder=5,
        )

        facet.xaxis.set_major_locator(MaxNLocator(4))
        facet.yaxis.set_major_locator(MaxNLocator(4))

        format_axis(facet)

    # Only the left facet displays y-axis tick labels.
    inner_axes["M3"].tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
    )

    # Shared axis labels belong to the complete outer panel.
    ax.text(
        0.50,
        -0.105,
        "Observed night-time response (°C)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#222222",
        clip_on=False,
    )

    ax.text(
        -0.070,
        0.48,
        "Predicted night-time response (°C)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.2,
        color="#222222",
        clip_on=False,
    )

def plot_oof_pre_adjustment(
    ax: plt.Axes,
    data: pd.DataFrame,
    panel: str,
    subtitle: str,
) -> None:
    highlighted = data.loc[data["highlight"].astype(bool)]
    if highlighted.empty:
        plot_oof_final(ax, data, panel, subtitle)
        return
    highlight_pair = str(highlighted["pair_id"].astype(str).iloc[0])
    central = data.loc[~data["pair_id"].astype(str).eq(highlight_pair)].copy()
    styles = {
        "M2": ("o", MODEL_COLORS["M2"], 0.38, 14),
        "M3": ("D", MODEL_COLORS["M3"], 0.34, 15),
    }
    for model in ("M2", "M3"):
        subset = central.loc[central["model"].astype(str).eq(model)]
        marker, color, alpha, size = styles[model]
        ax.scatter(
            subset["Rn_obs"], subset["Rn_pred"], s=size, marker=marker,
            facecolor=color, edgecolor="none", alpha=alpha, rasterized=True,
            label=MODEL_LABELS[model], zorder=2,
        )
    low, high = square_limits(central["Rn_obs"].to_numpy(float), central["Rn_pred"].to_numpy(float), 0.07)
    ax.plot([low, high], [low, high], color="#5A5A5A", linestyle="--", linewidth=0.9)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed night-time response (°C)")
    ax.set_ylabel("Predicted night-time response (°C)")
    ax.legend(loc="upper left", handletextpad=0.35, labelspacing=0.3)

    extreme = data.loc[data["pair_id"].astype(str).eq(highlight_pair)]
    m3_extreme = extreme.loc[extreme["model"].astype(str).eq("M3")].iloc[0]
    boundary_x = float(np.clip(float(m3_extreme["Rn_obs"]), low, high))
    boundary_y = low + 0.015 * (high - low)
    ax.scatter(
        boundary_x, boundary_y, s=27, marker="v", facecolor=MODEL_COLORS["M3"],
        edgecolor="white", linewidth=0.45, zorder=5, clip_on=False,
    )
    ax.annotate(
        "Largest OOF residual\nshown at full range",
        xy=(boundary_x, boundary_y), xytext=(0.04, 0.07), textcoords="axes fraction",
        fontsize=5.6, color="#4A4A4A", ha="left", va="bottom",
        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
    )
    inset = ax.inset_axes([0.61, 0.08, 0.35, 0.35])
    for model in ("M2", "M3"):
        subset = data.loc[data["model"].astype(str).eq(model)]
        marker, color, alpha, _ = styles[model]
        inset.scatter(
            subset["Rn_obs"], subset["Rn_pred"], s=5.5, marker=marker,
            facecolor=color, edgecolor="none", alpha=max(alpha, 0.45), rasterized=True,
        )
    full_low, full_high = square_limits(data["Rn_obs"].to_numpy(float), data["Rn_pred"].to_numpy(float), 0.05)
    inset.plot([full_low, full_high], [full_low, full_high], color="#666666", linestyle="--", linewidth=0.55)
    inset.set_xlim(full_low, full_high)
    inset.set_ylim(full_low, full_high)
    inset.set_aspect("equal", adjustable="box")
    inset.set_title("Full range", fontsize=5.6, pad=1.5)
    inset.tick_params(labelsize=4.7, length=1.8, width=0.5, pad=1.0)
    inset.xaxis.set_major_locator(MaxNLocator(3))
    inset.yaxis.set_major_locator(MaxNLocator(3))
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)
    panel_heading(ax, panel, subtitle)
    format_axis(ax)


def plot_attenuation_thresholds(
    ax: plt.Axes,
    data: pd.DataFrame,
    panel: str,
    subtitle: str,
) -> None:
    work = data.sort_values("min_events").reset_index(drop=True)
    y_positions = np.arange(len(work))[::-1]
    for y, (_, row) in zip(y_positions, work.iterrows()):
        ax.errorbar(
            float(row["attenuation_estimate"]), y,
            xerr=error_array(row, "attenuation_estimate", "attenuation_ci_low", "attenuation_ci_high"),
            fmt="o", ms=5.0,
            markerfacecolor=MODEL_COLORS["M3"] if bool(row["primary_cohort"]) else "white",
            markeredgecolor=MODEL_COLORS["M3"], markeredgewidth=1.0,
            ecolor=MODEL_COLORS["M3"], elinewidth=1.05, capsize=2.2, zorder=3,
        )
    ax.axvline(0, color="#5A5A5A", linestyle="--", linewidth=0.85, zorder=1)
    ax.set_yticks(y_positions, threshold_labels(work))
    ax.set_ylim(-0.55, len(work) - 0.45)
    ax.set_xlabel("Attenuation of residual slope (M1 − M3)")
    panel_heading(ax, panel, subtitle)
    format_axis(ax, "x")


def plot_component_performance(
    ax: plt.Axes,
    performance: pd.DataFrame,
    panel: str,
    subtitle: str,
) -> None:
    work = performance.loc[
        performance["task"].astype(str).eq("vector")
        & performance["model"].isin(["M1", "M2", "M3", "M4"])
        & performance["metric"].isin(["Rx_q2", "Rn_q2"])
    ].copy()
    outcomes = [("Rx_q2", "Daytime"), ("Rn_q2", "Night-time")]
    x_centres = np.arange(2, dtype=float)
    offsets = {"M1": -0.18, "M2": -0.06, "M3": 0.06, "M4": 0.18}
    for model in ("M1", "M2", "M3", "M4"):
        for x, (metric, _) in zip(x_centres, outcomes):
            subset = work.loc[work["model"].eq(model) & work["metric"].eq(metric)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            ax.errorbar(
                x + offsets[model], float(row["estimate"]), yerr=error_array(row),
                fmt="o", ms=3.9, color=MODEL_COLORS[model], ecolor=MODEL_COLORS[model],
                elinewidth=0.95, capsize=1.9, zorder=3,
            )
    ax.axhline(0, color="#5A5A5A", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_centres, [label for _, label in outcomes])
    ax.set_ylabel(r"Predictive skill ($Q^2$)")
    handles = [
        Line2D([0], [0], marker="o", color=MODEL_COLORS[m], lw=0, ms=4.0, label=m)
        for m in ("M1", "M2", "M3", "M4")
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2, handletextpad=0.3, columnspacing=0.8)
    panel_heading(ax, panel, subtitle)
    format_axis(ax, "y")


def plot_residual_scatter(
    ax: plt.Axes,
    points: pd.DataFrame,
    influence: pd.DataFrame,
    panel: str,
    subtitle: str,
    show_highlight: bool = False,
    label_x: float = -0.16,
) -> None:
    base = influence.loc[influence["removal_type"].astype(str).eq("none")].iloc[0]
    for model, marker, alpha in (("M1", "o", 0.27), ("M3", "D", 0.24)):
        subset = points.loc[points["model"].astype(str).eq(model)]
        # All observations remain visible. ``show_highlight`` controls only an
        # optional outline drawn on top and never changes the plotted cohort.
        regular = subset
        ax.scatter(
            regular["D_E"], regular["night_residual"], s=10, marker=marker,
            color=MODEL_COLORS[model], edgecolor="none", alpha=alpha,
            rasterized=True, label=MODEL_LABELS[model], zorder=2,
        )
        if show_highlight:
            extreme = subset.loc[subset["highlight"].astype(bool)]
            ax.scatter(
                extreme["D_E"], extreme["night_residual"], s=30, marker=marker,
                facecolor="white", edgecolor=MODEL_COLORS[model], linewidth=1.0, zorder=5,
            )
        x = np.array([points["D_E"].min(), points["D_E"].max()], float)
        slope = float(base[f"slope_{model}"])
        intercept = float(base[f"intercept_{model}"])
        ax.plot(x, intercept + slope * x, color=MODEL_COLORS[model], lw=1.1)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.75)
    ax.axvline(0, color="#B0B0B0", linestyle=(0, (1.5, 2.0)), linewidth=0.6)
    ax.set_xlabel("Early orthogonal state coordinate (°C)")
    ax.set_ylabel("Night-time prediction residual (°C)")
    ax.legend(loc="upper right", handletextpad=0.35, labelspacing=0.3)
    panel_heading(ax, panel, subtitle, label_x=label_x)
    format_axis(ax)


def plot_threshold_gain(
    ax: plt.Axes,
    summary: pd.DataFrame,
    panel: str,
    subtitle: str,
) -> None:
    work = summary.sort_values("min_events").reset_index(drop=True)
    y_positions = np.arange(len(work))[::-1]
    for y, (_, row) in zip(y_positions, work.iterrows()):
        ax.errorbar(
            float(row["m3_minus_m2_joint_q2"]), y,
            xerr=error_array(
                row, "m3_minus_m2_joint_q2",
                "m3_minus_m2_joint_q2_ci_low", "m3_minus_m2_joint_q2_ci_high",
            ),
            fmt="D", ms=4.4,
            markerfacecolor=MODEL_COLORS["M3"] if bool(row["primary_cohort"]) else "white",
            markeredgecolor=MODEL_COLORS["M3"], markeredgewidth=0.9,
            color=MODEL_COLORS["M3"], ecolor=MODEL_COLORS["M3"],
            elinewidth=1.05, capsize=2.1, zorder=3,
        )
    ax.axvline(0, color="#5A5A5A", linestyle="--", linewidth=0.85)
    ax.set_yticks(y_positions, threshold_labels(work))
    ax.set_ylim(-0.55, len(work) - 0.45)
    ax.set_xlabel(r"Mean–amplitude minus rank-one ($\Delta Q^2$)")
    panel_heading(ax, panel, subtitle)
    format_axis(ax, "x")


def plot_influence(
    ax: plt.Axes,
    influence: pd.DataFrame,
    panel: str,
    subtitle: str,
) -> None:
    base = influence.loc[influence["removal_type"].astype(str).eq("none")].iloc[0]
    pair = influence.loc[influence["removal_type"].astype(str).eq("leave_one_pair_out")]
    block = influence.loc[influence["removal_type"].astype(str).eq("leave_one_block_out")]
    ax.scatter(
        np.zeros(len(pair)), pair["slope_M3"], s=12, color=MODEL_COLORS["M3"],
        alpha=0.35, edgecolor="none", rasterized=True,
    )
    ax.scatter(
        np.ones(len(block)), block["slope_M3"], s=19, marker="D",
        facecolor="white", edgecolor=MODEL_COLORS["M3"], linewidth=0.75,
        alpha=0.8, rasterized=True,
    )
    ax.axhline(float(base["slope_M3"]), color=MODEL_COLORS["M3"], lw=1.0)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.75)
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1], ["Leave one pair out", "Leave one block out"])
    ax.set_ylabel("M3 residual slope")
    panel_heading(ax, panel, subtitle)
    format_axis(ax, "y")


def plot_legacy_schematic(ax: plt.Axes, panel: str = "a") -> None:
    ax.set_axis_off()
    y = 0.72
    xs = [0.15, 0.50, 0.85]
    labels = ["2015", "2019/20", "2024"]
    colors = [MODEL_COLORS["M1"], MODEL_COLORS["M3"], MODEL_COLORS["M4"]]
    ax.plot([xs[0], xs[-1]], [y, y], transform=ax.transAxes, color="#333333", lw=1.0)
    for x, label, color in zip(xs, labels, colors):
        ax.scatter([x], [y], transform=ax.transAxes, s=34, color=color, zorder=3)
        ax.text(x, y + 0.12, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0)
    ax.text(
        0.22, 0.49, "Estimate early\nresponse state", transform=ax.transAxes,
        ha="center", va="center", fontsize=5.7,
    )
    ax.text(
        0.75, 0.49, "Predict late day and\nnight responses", transform=ax.transAxes,
        ha="center", va="center", fontsize=5.7,
    )
    ax.annotate(
        "", xy=(0.66, 0.34), xytext=(0.32, 0.34), xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.85, "color": "#333333"},
    )
    ax.text(
        0.50, 0.18,
        "Training-optimized rank-one versus two-component state\nRepeated spatial-block cross-validation",
        transform=ax.transAxes, ha="center", va="center", fontsize=5.5,
    )
    panel_heading(ax, panel, "Early-to-late predictive design", label_x=-0.08)


def plot_legacy_oof(ax: plt.Axes, points: pd.DataFrame, panel: str, subtitle: str) -> None:
    for model, marker, alpha in (("M1", "o", 0.30), ("M3", "^", 0.30)):
        subset = points.loc[points["model"].astype(str).eq(model)]
        ax.scatter(
            subset["Rn_obs"], subset["Rn_pred"], s=11, marker=marker,
            color=MODEL_COLORS[model], edgecolor="none", alpha=alpha,
            rasterized=True, label=model,
        )
    low, high = square_limits(points["Rn_obs"].to_numpy(float), points["Rn_pred"].to_numpy(float), 0.06)
    ax.plot([low, high], [low, high], color="#5A5A5A", linestyle="--", linewidth=0.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed night-time response (°C)")
    ax.set_ylabel("Predicted night-time response (°C)")
    ax.legend(loc="upper left", handletextpad=0.3, labelspacing=0.25)
    panel_heading(ax, panel, subtitle)
    format_axis(ax)


def friendly_scenario(value: str) -> str:
    text = str(value)
    replacements = {
        "primary": "Primary",
        "reverse_L_to_E": "Reverse",
        "static_covariates_training_fold_only": "Static",
        "inverse_variance_training_only": "Precision weighted",
    }
    if text in replacements:
        return replacements[text]
    if text.startswith("alternative_cut_"):
        return f"Cut {text.rsplit('_', 1)[-1]}"
    if text.startswith("late_magnitude_ge_"):
        return f"Response magnitude ≥{float(text.rsplit('_', 1)[-1]):.2f}"
    if text.startswith("LOCO_"):
        return "LOCO " + text[5:].replace("_", " ")
    return text.replace("_", " ")


def plot_robustness_forest(
    ax: plt.Axes,
    robustness: pd.DataFrame,
    panel: str,
    subtitle: str,
) -> None:
    work = robustness.dropna(subset=["estimate"]).copy()
    preferred = [
        "primary",
        "alternative_cut_2018",
        "alternative_cut_2020",
        "reverse_L_to_E",
        "static_covariates_training_fold_only",
        "inverse_variance_training_only",
    ]
    selected = work.loc[work["scenario"].astype(str).isin(preferred)].copy()
    if len(selected) >= 4:
        selected["preferred_order"] = selected["scenario"].astype(str).map(
            {name: index for index, name in enumerate(preferred)}
        )
        work = selected.sort_values("preferred_order")
    else:
        work["priority"] = np.where(
            work["scenario"].astype(str).eq("primary"),
            0,
            np.where(work["scenario"].astype(str).str.startswith("LOCO_"), 1, 2),
        )
        work = work.sort_values(["priority", "scenario"])
    work["has_ci"] = work[["ci_low", "ci_high"]].notna().all(axis=1)
    work = work.head(6).reset_index(drop=True)
    y_positions = np.arange(len(work))[::-1]
    for y, (_, row) in zip(y_positions, work.iterrows()):
        if bool(row["has_ci"]):
            ax.errorbar(
                float(row["estimate"]), y, xerr=error_array(row), fmt="o", ms=3.8,
                color=MODEL_COLORS["M3"], ecolor=MODEL_COLORS["M3"],
                elinewidth=0.9, capsize=1.8, zorder=3,
            )
        else:
            ax.scatter(float(row["estimate"]), y, s=17, facecolor="white", edgecolor=MODEL_COLORS["M3"], linewidth=0.85, zorder=3)
    ax.axvline(0, color="#5A5A5A", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_positions, [friendly_scenario(v) for v in work["scenario"]])
    ax.set_ylim(-0.55, len(work) - 0.45)
    ax.set_xlabel(r"Vector gain, M3 $-$ M2 ($\Delta Q^2$)")
    panel_heading(ax, panel, subtitle)
    format_axis(ax, "x")


def save_rgb_jpg(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.temporary.jpg")
    expected_pixels = tuple(round(value * dpi) for value in fig.get_size_inches())
    try:
        fig.savefig(
            temporary, format="jpg", dpi=dpi, facecolor="white", edgecolor="white",
            pil_kwargs={"quality": 96, "subsampling": 0, "optimize": True},
        )
        with Image.open(temporary) as image:
            if image.mode != "RGB":
                raise RuntimeError(f"JPG output is not RGB: {image.mode}")
            if abs(image.width - expected_pixels[0]) > 2 or abs(image.height - expected_pixels[1]) > 2:
                raise RuntimeError(
                    f"Fixed canvas mismatch: {(image.width, image.height)} versus "
                    f"{expected_pixels}."
                )
            observed_dpi = image.info.get("dpi", (0, 0))
            if dpi == FORMAL_DPI and min(observed_dpi) < 590:
                raise RuntimeError(f"JPG DPI metadata is below 600 dpi: {observed_dpi}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_final(tables: Mapping[str, pd.DataFrame], path: Path, dpi: int) -> None:
    apply_nature_style()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, MAIN_HEIGHT_MM * MM_TO_INCH),
        facecolor="white",
    )
    grid = fig.add_gridspec(2, 2, height_ratios=[0.80, 1.20])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    plot_joint_performance(
        ax_a, tables["a"], "a", "Joint predictive skill"
    )
    plot_incremental_skill(
        ax_b, tables["b"], "b", "Incremental two-component gain"
    )
    plot_oof_final(
        ax_c, tables["c"], "c", "Out-of-fold night-time predictions",
        performance=tables["performance"],
    )
    fig.subplots_adjust(
        left=0.160,
        right=0.965,
        bottom=0.080,
        top=0.920,
        wspace=0.30,
        hspace=0.38,
    )
    save_rgb_jpg(fig, path, dpi)
    plt.close(fig)


def render_si_thresholds(
    tables: Mapping[str, pd.DataFrame], path: Path, dpi: int
) -> None:
    """Render the frozen attenuation summary without recomputing slope CIs."""
    apply_nature_style()
    fig, ax = plt.subplots(
        1, 1, figsize=(92.0 * MM_TO_INCH, SI_HEIGHT_MM * MM_TO_INCH),
        facecolor="white",
    )
    plot_attenuation_thresholds(
        ax, tables["d"], "a", "Residual attenuation by event support"
    )
    fig.subplots_adjust(left=0.225, right=0.985, bottom=0.095, top=0.90)
    save_rgb_jpg(fig, path, dpi)
    plt.close(fig)


def render_pre_adjustment(tables: Mapping[str, pd.DataFrame], path: Path, dpi: int) -> None:
    apply_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, MAIN_HEIGHT_MM * MM_TO_INCH))
    plot_joint_performance(axes[0, 0], tables["a"], "a", "Joint predictive skill\nacross state representations")
    plot_incremental_skill(axes[0, 1], tables["b"], "b", "Two-component gain in\npredictive skill")
    plot_oof_pre_adjustment(axes[1, 0], tables["c"], "c", "Out-of-fold night-time\nresponse predictions")
    plot_attenuation_thresholds(axes[1, 1], tables["d"], "d", "Residual-slope attenuation\nacross event thresholds")
    fig.subplots_adjust(left=0.225, right=0.985, bottom=0.095, top=0.90, wspace=0.42, hspace=0.55)
    save_rgb_jpg(fig, path, dpi)
    plt.close(fig)


def render_legacy(tables: Mapping[str, pd.DataFrame], path: Path, dpi: int) -> None:
    apply_nature_style()
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, LEGACY_HEIGHT_MM * MM_TO_INCH))
    plot_legacy_schematic(axes[0, 0], "a")
    plot_joint_performance(
        axes[0, 1], tables["a"], "b", "Joint held-out\npredictive skill",
        compact_labels=True,
    )
    plot_component_performance(axes[0, 2], tables["performance"], "c", "Daytime and night-time\npredictive skill")
    plot_legacy_oof(axes[1, 0], tables["si_points"], "d", "Observed and predicted\nnight-time responses")
    plot_residual_scatter(
        axes[1, 1], tables["si_points"], tables["si_influence"], "e",
        "Orthogonal-state dependence\nof prediction residuals",
        show_highlight=False, label_x=-0.22,
    )
    plot_robustness_forest(axes[1, 2], tables["robustness"], "f", "Predictive gain across\nvalidation settings")
    fig.subplots_adjust(left=0.075, right=0.988, bottom=0.105, top=0.91, wspace=0.50, hspace=0.60)
    save_rgb_jpg(fig, path, dpi)
    plt.close(fig)


def render_si(tables: Mapping[str, pd.DataFrame], path: Path, dpi: int) -> None:
    apply_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, SI_HEIGHT_MM * MM_TO_INCH))
    plot_component_performance(axes[0, 0], tables["performance"], "a", "Component-specific\npredictive skill")
    plot_residual_scatter(axes[0, 1], tables["si_points"], tables["si_influence"], "b", "Orthogonal-state dependence\nof night-time residuals", show_highlight=False)
    plot_threshold_gain(axes[1, 0], tables["d"], "c", "Two-component gain across\nevent-support thresholds")
    plot_influence(axes[1, 1], tables["si_influence"], "d", "Influence of individual pairs\nand spatial blocks")
    fig.subplots_adjust(left=0.225, right=0.985, bottom=0.095, top=0.90, wspace=0.42, hspace=0.55)
    save_rgb_jpg(fig, path, dpi)
    plt.close(fig)


def synthetic_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20260812)
    panel_a = pd.DataFrame(
        {
            "model": list(MODEL_ORDER),
            "model_label": [MODEL_LABELS[m] for m in MODEL_ORDER],
            "estimate": [-0.02, 0.30, 0.27, 0.53, 0.52],
            "ci_low": [-0.06, 0.15, 0.13, 0.40, 0.40],
            "ci_high": [0.03, 0.42, 0.37, 0.62, 0.61],
        }
    )
    panel_b = pd.DataFrame(
        {
            "metric": ["joint_q2", "Rx_q2", "Rn_q2"],
            "outcome_label": ["Joint response", "Daytime response", "Night-time response"],
            "estimate": [0.26, 0.24, 0.29],
            "ci_low": [0.18, 0.15, 0.14],
            "ci_high": [0.35, 0.34, 0.40],
        }
    )
    pairs = [f"P{i:03d}" for i in range(272)]
    obs = rng.normal(0.40, 0.90, len(pairs))
    rows_c = []
    for model, noise in (("M2", 0.70), ("M3", 0.42)):
        pred = 0.12 + 0.72 * obs + rng.normal(0, noise, len(obs))
        pred[0] = -2.2 if model == "M3" else -1.7
        for index, pair in enumerate(pairs):
            rows_c.append(
                {
                    "pair_id": pair, "model": model, "Rn_obs": obs[index],
                    "Rn_pred": pred[index], "highlight": index == 0,
                    "n_hw_events_E": 3 if index == 0 else 7, "n_hw_events_L": 8,
                }
            )
    panel_c = pd.DataFrame(rows_c)
    threshold_values = [2, 3, 5, 7, 10]
    n_pairs = [272, 240, 179, 116, 45]
    slope_m1 = [0.70, 0.72, 0.77, 0.81, 0.75]
    slope_m3 = [0.05, 0.06, 0.02, -0.02, 0.04]
    panel_d = pd.DataFrame(
        {
            "threshold_label": ["Baseline (>=2)", ">=3", ">=5", ">=7", ">=10"],
            "min_events": threshold_values,
            "primary_cohort": [True, False, False, False, False],
            "n_pairs": n_pairs,
            "slope_M1": slope_m1,
            "slope_M3": slope_m3,
            "attenuation_estimate": np.subtract(slope_m1, slope_m3),
            "attenuation_ci_low": [0.57, 0.58, 0.70, 0.76, 0.56],
            "attenuation_ci_high": [0.71, 0.77, 0.85, 0.91, 0.89],
            "m3_minus_m2_joint_q2": [0.26, 0.28, 0.31, 0.30, 0.29],
            "m3_minus_m2_joint_q2_ci_low": [0.18, 0.19, 0.21, 0.18, 0.09],
            "m3_minus_m2_joint_q2_ci_high": [0.35, 0.37, 0.40, 0.42, 0.47],
        }
    )
    performance_rows = []
    component_values = {
        "M1": (0.53, -0.02), "M2": (0.40, 0.12),
        "M3": (0.52, 0.33), "M4": (0.53, 0.23),
    }
    for model, (day, night) in component_values.items():
        for metric, estimate in (("Rx_q2", day), ("Rn_q2", night)):
            performance_rows.append(
                {
                    "task": "vector", "model": model, "metric": metric,
                    "estimate": estimate, "ci_low": estimate - 0.14,
                    "ci_high": estimate + 0.14,
                }
            )
    performance = pd.DataFrame(performance_rows)
    robustness = pd.DataFrame(
        {
            "scenario": [
                "primary", "LOCO_Africa", "LOCO_Asia", "LOCO_Europe",
                "alternative_cut_2018", "alternative_cut_2020",
                "reverse_L_to_E", "static_covariates_training_fold_only",
            ],
            "estimate": [0.26, 0.24, 0.28, 0.22, 0.25, 0.27, 0.18, 0.25],
            "ci_low": [0.18, 0.10, 0.15, 0.08, np.nan, np.nan, np.nan, np.nan],
            "ci_high": [0.35, 0.39, 0.41, 0.34, np.nan, np.nan, np.nan, np.nan],
            "source": [
                "paired_block_bootstrap", "leave_one_continent_out",
                "leave_one_continent_out", "leave_one_continent_out",
                "one_factor_sensitivity", "one_factor_sensitivity",
                "one_factor_sensitivity", "one_factor_sensitivity",
            ],
        }
    )
    d_early = rng.normal(0, 1.35, len(pairs))
    residual_rows = []
    influence_rows = []
    for model, slope in (("M1", 0.70), ("M3", 0.05)):
        residual = slope * d_early + rng.normal(0, 0.48, len(pairs))
        pred = obs - residual
        for index, pair in enumerate(pairs):
            residual_rows.append(
                {
                    "pair_id": pair, "spatial_block": f"B{index // 6:02d}",
                    "model": model, "D_E": d_early[index], "Rn_obs": obs[index],
                    "Rn_pred": pred[index], "night_residual": residual[index],
                    "highlight": index == 0,
                }
            )
    si_points = pd.DataFrame(residual_rows)
    influence_rows.append(
        {
            "removal_type": "none", "removed": "NONE", "slope_M1": 0.70,
            "intercept_M1": 0.0, "slope_M3": 0.05, "intercept_M3": 0.0,
            "attenuation": 0.65,
        }
    )
    for index, pair in enumerate(pairs):
        influence_rows.append(
            {
                "removal_type": "leave_one_pair_out", "removed": pair,
                "slope_M1": 0.70 + rng.normal(0, 0.012), "intercept_M1": 0.0,
                "slope_M3": 0.05 + rng.normal(0, 0.012), "intercept_M3": 0.0,
                "attenuation": 0.65 + rng.normal(0, 0.018),
            }
        )
    for index in range(20):
        influence_rows.append(
            {
                "removal_type": "leave_one_block_out", "removed": f"B{index:02d}",
                "slope_M1": 0.70 + rng.normal(0, 0.025), "intercept_M1": 0.0,
                "slope_M3": 0.05 + rng.normal(0, 0.025), "intercept_M3": 0.0,
                "attenuation": 0.65 + rng.normal(0, 0.035),
            }
        )
    result = {
        "a": panel_a, "b": panel_b, "c": panel_c, "d": panel_d,
        "performance": performance,
        "robustness": robustness, "si_points": si_points,
        "si_influence": pd.DataFrame(influence_rows),
    }
    validate_tables(result)
    return result


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="figure4_self_test_") as temp:
        output = Path(temp)
        tables = synthetic_tables()
        paths = {
            "final": output / OUTPUT_NAMES["final"],
            "si": output / OUTPUT_NAMES["si"],
        }
        render_final(tables, paths["final"], 150)
        render_si_thresholds(tables, paths["si"], 150)
        for path in paths.values():
            if not path.is_file() or path.stat().st_size < 10_000:
                raise AssertionError(f"Self-test figure was not created correctly: {path}")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise AssertionError(f"Invalid self-test JPG: {path}")
    print("SELF_TEST_PASS")


def save_preview(jpg_path: Path, webp_path: Path) -> None:
    with Image.open(jpg_path) as image:
        preview = image.convert("RGB")
        preview.thumbnail((1800, 1500), Image.Resampling.LANCZOS)
        preview.save(webp_path, format="WEBP", quality=82, method=6)


def write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    tables, input_paths, inventory_pre = read_formal_tables(args.tables_dir)
    output = create_timestamped_output(args.output_dir)
    paths = {key: output / name for key, name in OUTPUT_NAMES.items()}
    render_final(tables, paths["final"], FORMAL_DPI)
    render_si_thresholds(tables, paths["si"], FORMAL_DPI)
    save_preview(paths["final"], paths["final_preview"])
    save_preview(paths["si"], paths["si_preview"])

    inventory_post = file_inventory(input_paths)
    if not inventory_pre.equals(inventory_post):
        raise RuntimeError("READ_ONLY_INPUT_MODIFIED_DURING_PLOTTING")
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "formal_tables_directory": str(Path(args.tables_dir).expanduser().resolve()),
        "formal_input_sha256": {
            str(row.path): str(row.sha256) for row in inventory_pre.itertuples(index=False)
        },
        "main_image": OUTPUT_NAMES["final"],
        "main_image_sha256": sha256_file(paths["final"]),
        "main_preview": OUTPUT_NAMES["final_preview"],
        "main_preview_sha256": sha256_file(paths["final_preview"]),
        "si_image": OUTPUT_NAMES["si"],
        "si_image_sha256": sha256_file(paths["si"]),
        "si_preview": OUTPUT_NAMES["si_preview"],
        "si_preview_sha256": sha256_file(paths["si_preview"]),
        "main_panels": ["a", "b", "c"],
        "main_fixed_canvas_mm": [FIGURE_WIDTH_MM, MAIN_HEIGHT_MM],
        "main_layout": "two compact comparison panels above two matched square OOF facets",
        "plot_only": True,
        "bootstrap_run_in_plot": False,
        "calibration_fitted_in_plot": False,
        "bbox_inches_tight_used": False,
    }
    manifest_path = output / "figure4_predictive_state_plot_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        output / "_SUCCESS.json",
        {
            "status": "SUCCESS",
            "script_version": manifest["script_version"],
            "main_image": OUTPUT_NAMES["final"],
            "main_image_sha256": sha256_file(paths["final"]),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(f"SUCCESS: {paths['final']}")
    print(f"SI candidate: {paths['si']}")
    print(f"Previews: {paths['final_preview']}; {paths['si_preview']}")


if __name__ == "__main__":
    main()
