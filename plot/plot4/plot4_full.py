#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Figure 4 from the prepared UHI/UCI CSV outputs.

This script performs no statistical modelling. It only visualizes CSV outputs
from the updated preparation script.

Panel-b/panel-c interpretation
-------------------------------
Annual rows define the canonical UHI/UCI classification upstream.

Panel b uses signed urban-minus-rural +1 degree marginal sensitivity
contrasts for all three outcomes:

    Delta S = S_urban - S_rural

Positive values indicate greater urban marginal sensitivity.
Negative values indicate greater rural marginal sensitivity.

    sleep: Minor nighttime-Tmin sleep-loss model
    building CDH: 24-hour CDH=max(T-26,0) model
    labour loss: Dunne model at station-specific FFT Tx-hour shaded WBGT

Panel c shows urban-minus-rural burden components:
    sleep: day=0, night=total
    building CDH: day+night=total
    labour loss: day=total, night=0 structural zero / not applicable

Positive panel-c values indicate a larger urban-relative burden.
"""

from __future__ import annotations


# -----------------------------------------------------------------------------
# Open-source path configuration (scientific calculations below are unchanged)
# -----------------------------------------------------------------------------
from pathlib import Path as _OpenSourcePath
import sys as _open_source_sys
_OPEN_SOURCE_REPO_ROOT = _OpenSourcePath(__file__).resolve().parents[2]
if str(_OPEN_SOURCE_REPO_ROOT) not in _open_source_sys.path:
    _open_source_sys.path.insert(0, str(_OPEN_SOURCE_REPO_ROOT))
from project_paths import PATHS as _OPEN_SOURCE_PATHS
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    import geopandas as gpd

    WORLD_GEODF = None
    HAS_GEOPANDAS_WORLD = False

    try:
        WORLD_GEODF = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        HAS_GEOPANDAS_WORLD = True
    except Exception:
        try:
            import geodatasets
            WORLD_GEODF = gpd.read_file(geodatasets.get_path("naturalearth.land"))
            HAS_GEOPANDAS_WORLD = True
        except Exception:
            WORLD_GEODF = None
            HAS_GEOPANDAS_WORLD = False
except Exception:
    gpd = None
    WORLD_GEODF = None
    HAS_GEOPANDAS_WORLD = False


UNIFIED_ROOT = Path(_OPEN_SOURCE_PATHS.unified_root)
PROJECT_ROOT = UNIFIED_ROOT

DEFAULT_DATA_DIR = UNIFIED_ROOT / "plot_data/ncc_diurnal"
DEFAULT_OUT_DIR = str(UNIFIED_ROOT / "plot_data/fig4_new")

FILES: Dict[str, str] = {
    "a": "main_panel_a_hw_asymmetry_map.csv",
    "a_meta": "main_panel_a_hw_asymmetry_map_metadata.json",
    "b": "main_panel_b_asymmetry_contrast.csv",
    "c": "main_panel_c_additive_decomposition.csv",
    "meta": "main_model_outputs_metadata.json",
    "supp": "supp_daynight_delta_boxplot_data.csv",
    "supp_meta": "supp_daynight_delta_boxplot_data_metadata.json",
}

UHI_UCI_GROUPS = ("UHI", "UCI")

COLORS = {
    "NHW": "#7A7A7A",
    "HW": "#262626",
    "total": "#222222",
    "axis": "#222222",
}

# Keep the group palette consistent with the temperature figures:
# UHI = red family; UCI = blue family.
GROUP_COLORS = {
    "UHI": "#D73027",
    "UCI": "#2166AC",
}

# Panel-c day/night components retain the UHI/UCI group hue.
# Day uses the lighter shade and night the darker shade.
GROUP_COMPONENT_COLORS = {
    "UHI": {
        "day": "#F4A582",
        "night": "#B2182B",
    },
    "UCI": {
        "day": "#92C5DE",
        "night": "#2166AC",
    },
}

# Neutral legend swatches explain the light/dark component convention.
COMPONENT_LEGEND_COLORS = {
    "day": "#D9D9D9",
    "night": "#737373",
}

# Panel-a red-blue diverging palette.
# Negative values use the UCI blue family and positive values use the UHI red
# family. The transition at zero is deliberately abrupt: there is no white,
# grey, or other neutral midpoint in either the map or the colour bar.
#
# Values still retain magnitude information through shade intensity within
# each sign:
#   strong negative -> dark blue; near-zero negative -> light blue
#   near-zero positive -> light red; strong positive -> dark red
MAP_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_red_no_neutral_midpoint",
    [
        (0.000000, "#053061"),
        (0.300000, "#2166AC"),
        (0.499999, "#92C5DE"),
        (0.500000, "#92C5DE"),
        (0.500001, "#F4A582"),
        (0.700000, "#D73027"),
        (1.000000, "#67001F"),
    ],
    N=256,
)

GROUP_MARKERS = {
    "UHI": "s",             # square
    "UCI": "^",             # triangle
}

# Enlarged map symbols for legibility at manuscript size.
MAP_POINT_SIZE = {
    "UHI": 38,
    "UCI": 44,
}

PANEL_POINT_SIZE = {
    "UHI": 34,
    "UCI": 40,
}

PHASE_MARKER_FACE = {
    "NHW": "white",
    "HW": COLORS["HW"],
}

MAP_EXTENT = [-180, 180, -60, 80]
MAP_DATA_ASPECT = (
    (MAP_EXTENT[1] - MAP_EXTENT[0]) /
    (MAP_EXTENT[3] - MAP_EXTENT[2])
)

OUTCOME_LABEL = {
    "sleep": "Sleep loss",
    "building_energy": "Building CDH",
    "residential_energy": "Residential energy",
    "commercial_energy": "Commercial energy",

    # Main updated labour outcome

    # Backward-compatible legacy outcome
    "labour_loss": "Labour loss",
}


VAR_LABEL = {
    "Sleep air temperature": "Sleep air\ntemperature",
    "Cooling air temperature": "Cooling air\ntemperature",
    "Labour heat stress": "Labour heat\nstress exposure",
}

REQUIRED_COLUMNS: Dict[str, Sequence[str]] = {
    "a": ("lat", "lon", "uhi_uci_group", "asym_hw_minus_nhw"),
    "b": (
        "uhi_uci_group",
        "outcome",
        "period_phase",
        "panel_b_estimate",
        "panel_b_ci_low",
        "panel_b_ci_high",
    ),
    "c": (
        "uhi_uci_group",
        "outcome",
        "period_phase",
        "component",
        "estimate",
        "ci_low",
        "ci_high",
    ),
    "supp": (
        "uhi_uci_group",
        "variable",
        "period_phase",
        "exposure_period",
        "delta_value",
    ),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Noto Sans", "Helvetica"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 600,
    "savefig.dpi": 600,
})

PANEL_NOTE_BBOX = dict(
    facecolor="white",
    edgecolor="none",
    alpha=0.82,
    pad=2,
)


def add_panel_note(
    ax,
    text: str,
    xy: Tuple[float, float] = (0.02, 0.03),
    ha: str = "left",
    va: str = "bottom",
    fontsize: float = 6.4,
) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="0.35",
        bbox=PANEL_NOTE_BBOX,
        zorder=20,
    )


def add_world_outline_fallback(ax) -> None:
    if HAS_GEOPANDAS_WORLD and WORLD_GEODF is not None:
        try:
            WORLD_GEODF.boundary.plot(
                ax=ax,
                linewidth=0.35,
                color="0.62",
                zorder=1,
            )
        except Exception:
            pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    i = 2
    while True:
        p = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not p.exists():
            return p
        i += 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()


def file_mtime_utc(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  ! missing: {path}", file=sys.stderr)
        return None

    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  ! read failed for {path}: {e}", file=sys.stderr)
        return None


def load_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "_status": "missing",
            "_path": str(path),
        }

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "_status": "read_or_parse_failed",
            "_path": str(path),
            "_error": repr(e),
        }


def dataframe_audit(
    key: str,
    path: Path,
    df: Optional[pd.DataFrame],
    required_columns: Sequence[str] = (),
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "key": key,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime_utc": file_mtime_utc(path),
        "sha256": sha256_file(path),
        "required_columns": list(required_columns),
    }

    if df is None:
        rec.update({
            "readable": False,
            "n_rows": None,
            "n_columns": None,
            "columns": [],
            "missing_required_columns": list(required_columns),
        })
    else:
        cols = list(df.columns)
        rec.update({
            "readable": True,
            "n_rows": int(len(df)),
            "n_columns": int(len(cols)),
            "columns": cols,
            "missing_required_columns": [c for c in required_columns if c not in set(cols)],
        })

    return rec


def json_file_audit(key: str, path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "key": key,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime_utc": file_mtime_utc(path),
        "sha256": sha256_file(path),
        "content": meta,
    }


def output_file_audit(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime_utc": file_mtime_utc(path),
        "sha256": sha256_file(path),
    }


def write_provenance_manifest(
    out_dir: Path,
    manifest_basename: str,
    data_dir: Path,
    args: argparse.Namespace,
    csv_audits: Sequence[Dict[str, Any]],
    json_audits: Sequence[Dict[str, Any]],
    output_paths: Sequence[Path],
) -> Path:
    manifest = {
        "manifest_type": "ncc_diurnal_uhi_uci_plot_provenance",
        "created_at_utc": utc_now_iso(),
        "script": {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "has_cartopy": HAS_CARTOPY,
            "has_geopandas_world": HAS_GEOPANDAS_WORLD,
        },
        "arguments": vars(args),
        "input_data_dir": str(data_dir),
        "input_csv_files": list(csv_audits),
        "input_metadata_json_files": list(json_audits),
        "generated_outputs": [output_file_audit(p) for p in output_paths],
        "plotting_notes": [
            "This script performs no statistical modelling.",
            "All plotted empirical values are read from the listed CSV files.",
            "The map uses asym_hw_minus_nhw and marker shape for UHI/UCI.",
            "Map markers are enlarged and use a non-white-centred green-grey-purple diverging palette.",
            "Panel b uses signed urban-minus-rural marginal sensitivity contrasts for sleep loss, building CDH and labour loss; panel c contains urban-minus-rural burden values.",
            "Panels b/c read ci_low and ci_high from upstream bootstrap outputs.",
            "Error bars in panels b/c are station-pair cluster bootstrap percentile 95% CIs when generated by the updated upstream script.",
            "Sleep decomposition rows are beta × U-R air-temperature exposure delta.",
            "Building energy rows are direct U-R CDH day/night differences from CDH_day_u/r and CDH_night_u/r.",
            "Energy does not use E_comm_u, E_resi_u, residential_energy, or commercial_energy.",
            "Labour-loss rows use the Dunne model at station-specific FFT Tx-hour shaded WBGT.",
            "Panel c labour values are validated as day=total and night=0 structural zero; the plotting script does not overwrite them.",
            "Labour-loss sign: positive means urban labour loss > rural labour loss.",
            "Panel b reads outcome-specific marginal sensitivities from panel_b_estimate and panel_b_ci_*: all three outcomes use S_urban minus S_rural.",
            "The HW amplification text file reports signed HW-minus-NHW changes for signed outcomes; relative percentages are diagnostic only when baseline sign is stable and non-zero.",
            "Rows with raw total approximately zero are marked as percentage undefined.",
        ],
    }

    path = safe_output_path(out_dir / f"{manifest_basename}.json")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")

    return path


def filt(data: Optional[pd.DataFrame], group: str) -> Optional[pd.DataFrame]:
    if data is None:
        return None

    if "uhi_uci_group" not in data.columns:
        return data.iloc[0:0].copy()

    return data[data["uhi_uci_group"].astype(str).str.upper() == group].copy()

def _normalise_outcome_names(df: Optional[pd.DataFrame], table_name: str = "") -> Optional[pd.DataFrame]:
    """
    Normalize outcome names without changing their scientific meaning.

    Minimal rule:
    - labour_capacity remains labour_capacity.
    - labour_loss remains labour_loss.
    - Do NOT map labour_capacity to labour_loss.
    """
    if df is None or "outcome" not in df.columns:
        return df

    out = df.copy()
    out["outcome_original"] = out["outcome"].astype(str)
    out["outcome"] = out["outcome"].astype(str).str.strip()

    aliases = {
        # Updated main labour outcome
        "labor_capacity": "labour_capacity",
        "labour_capacity": "labour_capacity",
        "work_capacity": "labour_capacity",
        "labor_work_capacity": "labour_capacity",
        "labour_work_capacity": "labour_capacity",

        # Legacy loss-direction outcome
        "labor_loss": "labour_loss",
        "labour_loss": "labour_loss",
        "labor_capacity_loss": "labour_loss",
        "labour_capacity_loss": "labour_loss",

        # Energy aliases retained
        "building_cdh": "building_energy",
        "building_energy_cdh": "building_energy",
        "cooling_degree_hours": "building_energy",
        "cdh_energy": "building_energy",
    }

    out["outcome"] = out["outcome"].replace(aliases)

    if "unit" in out.columns:
        cap_rows = out["outcome"].eq("labour_capacity")
        if cap_rows.any():
            unit_text = " ".join(out.loc[cap_rows, "unit"].astype(str).unique()).lower()
            if "capacity" not in unit_text and "lc" not in unit_text:
                print(
                    f"[warn] {table_name}: labour_capacity rows detected, "
                    f"but unit does not clearly indicate capacity: {unit_text}",
                    file=sys.stderr,
                )

        loss_rows = out["outcome"].eq("labour_loss")
        if loss_rows.any():
            unit_text = " ".join(out.loc[loss_rows, "unit"].astype(str).unique()).lower()
            if "loss" not in unit_text:
                print(
                    f"[warn] {table_name}: labour_loss rows detected, "
                    f"but unit does not clearly indicate loss: {unit_text}",
                    file=sys.stderr,
                )

        energy_rows = out["outcome"].eq("building_energy")
        if energy_rows.any():
            unit_text = " ".join(out.loc[energy_rows, "unit"].astype(str).unique()).lower()
            if "cdh" not in unit_text and "degree-hour" not in unit_text:
                print(
                    f"[warn] {table_name}: building_energy rows detected, "
                    f"but unit does not clearly indicate CDH/degree-hours: {unit_text}",
                    file=sys.stderr,
                )

    return out


def _drop_labour_capacity_if_loss_present(
    df: Optional[pd.DataFrame],
    table_name: str = "",
) -> Optional[pd.DataFrame]:
    """Require burden-oriented labour_loss rows in all main Figure 4 tables."""
    if df is None or "outcome" not in df.columns:
        return df

    out = df.copy()
    outcomes = set(out["outcome"].astype(str).str.strip())
    if "labour_capacity" in outcomes:
        raise ValueError(
            f"{table_name}: labour_capacity rows are not permitted in the "
            "unified main Figure 4. Regenerate inputs with Dunne labour_loss."
        )
    return out


def _apply_panelc_labour_day_only_for_plot(
    df: Optional[pd.DataFrame],
    table_name: str = "",
) -> Optional[pd.DataFrame]:
    """
    Validate the upstream Dunne Tx-based labour decomposition without changing values.

    Required convention for labour_loss:
        day   = dLoss_Tx
        night = 0, structural zero / not applicable
        total = dLoss_Tx

    Positive values mean urban labour loss > rural labour loss.

    """
    if df is None:
        return df

    required = {
        "uhi_uci_group",
        "outcome",
        "period_phase",
        "component",
        "estimate",
    }
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    out["outcome"] = out["outcome"].astype(str).str.strip()
    out["component"] = out["component"].astype(str).str.lower().str.strip()

    labour = out[
        out["outcome"].isin(["labour_loss", "labor_loss"])
    ].copy()
    if labour.empty:
        return out

    night = pd.to_numeric(
        labour.loc[labour["component"].eq("night"), "estimate"],
        errors="coerce",
    ).dropna()
    if len(night) and not np.allclose(night.to_numpy(float), 0.0, atol=1e-10):
        raise ValueError(
            f"{table_name}: Tx-based Dunne labour-loss night component must be "
            "a structural zero upstream."
        )

    keys = ["uhi_uci_group", "outcome", "period_phase"]
    for key, sub in labour.groupby(keys, observed=True, dropna=False):
        day = pd.to_numeric(
            sub.loc[sub["component"].eq("day"), "estimate"],
            errors="coerce",
        ).dropna()
        total = pd.to_numeric(
            sub.loc[sub["component"].eq("total"), "estimate"],
            errors="coerce",
        ).dropna()

        if len(day) and len(total):
            if not np.isclose(float(day.iloc[0]), float(total.iloc[0]), atol=1e-10):
                raise ValueError(
                    f"{table_name}: Tx-based labour-loss total must equal day "
                    f"for {key}."
                )

    return out

def outcome_display_label(outcome: str) -> str:
    labels = {
        **OUTCOME_LABEL,
        "building_energy": "Building CDH",
            "labour_loss": "Labour loss",
    }
    key = str(outcome).strip()
    return labels.get(key, key.replace("_", " ").title())


def plot_outcome_order(
    data: Optional[pd.DataFrame],
) -> List[str]:
    """Strict burden-oriented outcome order for Figure 4."""
    if data is None or "outcome" not in data.columns:
        return []
    present = set(data["outcome"].astype(str).str.strip())
    if "labour_capacity" in present:
        raise ValueError(
            "Figure 4 requires labour_loss; labour_capacity is not allowed."
        )
    return [o for o in ["sleep", "building_energy", "labour_loss"] if o in present]



def plot_panel_b_outcome_order(
    data: Optional[pd.DataFrame],
) -> List[str]:
    """
    Outcome order for panel-b model-implied marginal sensitivities.
    """
    if data is None or "outcome" not in data.columns:
        return []

    present = set(
        data["outcome"].astype(str).str.strip()
    )

    order = [
        "sleep",
        "building_energy",
        "labour_loss",
    ]

    return [o for o in order if o in present]


def _panel_b_numeric_columns(
    data: pd.DataFrame,
) -> Tuple[str, str, str]:
    """
    Return panel-b estimate/CI columns.

    The updated preparation script writes generic outcome-specific fields.
    Legacy night-minus-day columns remain a read-only fallback.
    """
    preferred = (
        "panel_b_estimate",
        "panel_b_ci_low",
        "panel_b_ci_high",
    )
    if set(preferred).issubset(data.columns):
        return preferred

    legacy = (
        "beta_contrast_night_minus_day",
        "ci_low",
        "ci_high",
    )
    return legacy


def variable_display_label(var: str) -> str:
    labels = {
        **VAR_LABEL,
        "Building cooling degree-hours": "Building cooling\ndegree-hours",
    }
    key = str(var).strip()
    return labels.get(key, key)

def harmonize_model_outcomes(df: Optional[pd.DataFrame], table_name: str = "") -> Optional[pd.DataFrame]:
    """
    Harmonize outcome names for plotting.

    Important:
    - Do not map labour_capacity to labour_loss.
    - This script only visualizes prepared CSVs; it should not change
      the sign meaning of upstream labour outputs.
    """
    if df is None or "outcome" not in df.columns:
        return df

    out = _normalise_outcome_names(df, table_name=table_name)

    legacy_energy = out["outcome"].isin(["residential_energy", "commercial_energy"])
    if legacy_energy.any():
        print(
            f"[warn] {table_name}: legacy residential/commercial energy rows detected. "
            "They are not remapped to building_energy because the current definition "
            "should use direct CDH day/night differences.",
            file=sys.stderr,
        )

    return out


def get_common_map_norm2(
    data: pd.DataFrame,
    val_col: str = "asym_hw_minus_nhw",
) -> Tuple[TwoSlopeNorm, float]:
    if data is None or len(data) == 0 or val_col not in data.columns:
        vmax = 1.0
    else:
        vals = pd.to_numeric(data[val_col], errors="coerce").dropna().values
        vmax = max(float(np.nanpercentile(np.abs(vals), 95)), 1e-3)

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    return norm, vmax

def get_common_map_norm(
    data: pd.DataFrame,
    val_col: str = "asym_hw_minus_nhw",
) -> Tuple[TwoSlopeNorm, float]:
    # Difference map: centered at zero rather than at a ratio of 1.
    if data is None or len(data) == 0 or val_col not in data.columns:
        return TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), 1.0

    vals = pd.to_numeric(data[val_col], errors="coerce").dropna().values
    vmax = max(float(np.nanpercentile(np.abs(vals), 95)), 1e-3)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    return norm, vmax


def draw_map(ax, data: pd.DataFrame, group: str, label: str = "a") -> None:
    data = filt(data, group)
    val = "asym_hw_minus_nhw"

    if data is None or len(data) == 0 or not {"lat", "lon", val}.issubset(data.columns):
        ax.text(
            0.5,
            0.5,
            f"{group} map unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777",
        )
        ax.set_axis_off()
        return

    data = data.copy()

    for c in ["lat", "lon", val]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=["lat", "lon", val])

    if len(data) == 0:
        ax.text(
            0.5,
            0.5,
            f"{group} map unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777",
        )
        ax.set_axis_off()
        return

    vals = data[val].values
    vmax = max(float(np.nanpercentile(np.abs(vals), 95)), 1e-3)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = MAP_DIVERGING_CMAP

    if HAS_CARTOPY:
        fig = ax.figure
        pos = ax.get_position()
        ax.remove()
        ax = fig.add_axes(pos, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())

        ax.add_feature(
            cfeature.OCEAN.with_scale("110m"),
            facecolor="white",
            edgecolor="none",
        )
        ax.add_feature(
            cfeature.LAND.with_scale("110m"),
            facecolor="#f5f5f5",
            edgecolor="none",
        )
        ax.coastlines(linewidth=0.4, color="#777")

        sc = ax.scatter(
            data["lon"],
            data["lat"],
            c=vals,
            cmap=cmap,
            norm=norm,
            s=12,
            marker="s",
            edgecolor="white",
            linewidth=0.25,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
    else:
        ax.set_facecolor("#fafafa")
        add_world_outline_fallback(ax)

        sc = ax.scatter(
            data["lon"],
            data["lat"],
            c=vals,
            cmap=cmap,
            norm=norm,
            s=14,
            marker="s",
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )

        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 80)
        ax.set_xticks([-120, -60, 0, 60, 120])
        ax.set_yticks([-30, 0, 30, 60])
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")

    pos = ax.get_position()
    cax = ax.figure.add_axes([
        pos.x0 + 0.10 * pos.width,
        pos.y0 - 0.026,
        0.70 * pos.width,
        0.015,
    ])

    cb = plt.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label(
        r"HW amplification of urban-rural diurnal asymmetry "
        r"[$(\Delta_{\mathrm{night}}-\Delta_{\mathrm{day}})_{\mathrm{HW}}"
        r"-(\Delta_{\mathrm{night}}-\Delta_{\mathrm{day}})_{\mathrm{NHW}}$]",
        fontsize=7.2,
    )

    ax.text(
        0.02,
        0.04,
        f"n = {len(data):,} pairs\n"
        f"{np.mean(vals > 0) * 100:.0f}% > 0\n"
        "Variable: U-R delta asymmetry",
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="bottom",
        bbox=PANEL_NOTE_BBOX,
        zorder=10,
    )

    ax.set_title(
        f"{label}  {group}: heatwave amplification of diurnal asymmetry",
        loc="left",
        fontweight="bold",
        fontsize=9,
        pad=4,
    )

PANEL_LABEL_FONTSIZE = 8.5
PANEL_TITLE_FONTSIZE = 10
PANEL_LABEL_X = -0.04
PANEL_LABEL_Y = 1.04

def draw_map_combined(ax, data: pd.DataFrame, label: str = "a") -> None:
    """
    Reference-style Robinson projection map for HW-amplified diurnal difference.

    布局更新: 图例和统计文本紧凑垂直堆叠在左下角 (Antarctica 左侧空白区)。
    
    其余规范保持不变 (投影、底色、颜色映射、边框、字号等)。
    """
    req = {"lat", "lon", "uhi_uci_group", "asym_hw_minus_nhw"}
    if data is None or len(data) == 0 or not req.issubset(data.columns):
        ax.text(
            0.5, 0.5, "Data Missing",
            transform=ax.transAxes,
            ha="center", va="center",
            color="0.45", fontsize=7.5,
        )
        ax.set_axis_off()
        return

    # ── 数据预处理 (保持不变) ──
    data = data.copy()
    data["uhi_uci_group"] = data["uhi_uci_group"].astype(str).str.upper()
    data["asym_hw_minus_nhw"] = pd.to_numeric(data["asym_hw_minus_nhw"], errors="coerce")
    data = data.dropna(subset=["asym_hw_minus_nhw", "lat", "lon"])
    data = data[data["uhi_uci_group"].isin(["UHI", "UCI"])]

    if len(data) == 0:
        ax.text(
            0.5, 0.5, "No difference data",
            transform=ax.transAxes,
            ha="center", va="center",
            color="0.45", fontsize=7.5,
        )
        ax.set_axis_off()
        return

    # Difference values: keep the upstream difference directly, no ratio/log transform.
    data["asym_diff"] = data["asym_hw_minus_nhw"]

    # Symmetric colour scale around zero for differences.
    v_limit = max(float(np.nanpercentile(np.abs(data["asym_diff"]), 95)), 1e-3)
    norm = TwoSlopeNorm(vmin=-v_limit, vcenter=0.0, vmax=v_limit)
    cmap = MAP_DIVERGING_CMAP

    # ── Reference-style Robinson 投影底图 ──
    if HAS_CARTOPY:
        fig = ax.figure
        pos = ax.get_position()
        ax.remove()
        ax = fig.add_axes(pos, projection=ccrs.Robinson(central_longitude=0))
        ax.set_global()

        ax.add_feature(
            cfeature.OCEAN.with_scale("110m"),
            facecolor="white", edgecolor="none", zorder=0,
        )
        ax.add_feature(
            cfeature.LAND.with_scale("110m"),
            facecolor="#f7f7f7", edgecolor="none", zorder=0,
        )
        ax.add_feature(
            cfeature.COASTLINE.with_scale("110m"),
            linewidth=0.25, edgecolor="#505050", zorder=1,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            linewidth=0.15, edgecolor="#aaaaaa", linestyle=":", zorder=1,
        )
        ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            linewidth=0.20, color="#cccccc", alpha=0.5, linestyle="-",
        )

        for s in ["left", "right", "top", "bottom"]:
            if s in ax.spines:
                ax.spines[s].set_visible(False)
        if "geo" in ax.spines:
            ax.spines["geo"].set_visible(True)
            ax.spines["geo"].set_linewidth(0.60)
            ax.spines["geo"].set_edgecolor("black")

        sc = None
        for grp, mkr, zd in [("UCI", "^", 3), ("UHI", "s", 4)]:
            sub = data[data["uhi_uci_group"] == grp]
            if len(sub) == 0:
                continue
            sc = ax.scatter(
                sub["lon"], sub["lat"],
                c=sub["asym_diff"],
                cmap=cmap, norm=norm,
                s=MAP_POINT_SIZE[grp], marker=mkr,
                edgecolor="#202020",
                linewidth=0.35,
                alpha=1.0,
                transform=ccrs.PlateCarree(),
                zorder=zd,
            )
    else:
        ax.set_facecolor("white")
        add_world_outline_fallback(ax)
        sc = None
        for grp, mkr, zd in [("UCI", "^", 3), ("UHI", "s", 4)]:
            sub = data[data["uhi_uci_group"] == grp]
            if len(sub) == 0:
                continue
            sc = ax.scatter(
                sub["lon"], sub["lat"],
                c=sub["asym_diff"],
                cmap=cmap, norm=norm,
                s=MAP_POINT_SIZE[grp], marker=mkr,
                edgecolor="#202020",
                linewidth=0.35,
                alpha=1.0,
                zorder=zd,
            )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 80)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.60)
            spine.set_edgecolor("black")

    # ── 计算统计量 ──
    n_uhi = int((data["uhi_uci_group"] == "UHI").sum())
    n_uci = int((data["uhi_uci_group"] == "UCI").sum())
    p_uhi_night = (
        float(np.mean(data.loc[data["uhi_uci_group"] == "UHI", "asym_diff"] > 0) * 100)
        if n_uhi else np.nan
    )
    p_uci_night = (
        float(np.mean(data.loc[data["uhi_uci_group"] == "UCI", "asym_diff"] > 0) * 100)
        if n_uci else np.nan
    )

    # ── 左下角紧凑布局: 图例 + 统计文本垂直堆叠 ──
    LEGEND_FONT = 8.5
    LEGEND_X = 0.15        # 统一左对齐
    STATS_Y = 0.040           # 统计文本基线
    LEGEND_Y = 0.05  # 图例紧贴统计文本上方,间距 ~0.06 axes 高度

    # 1. 图例 (UHI/UCI 标记) - 紧凑无框 
    shape_handles = [
        Line2D(
            [0], [0],
            marker="s", color="none",
            markerfacecolor="0.40",
            markeredgecolor="black",
            markeredgewidth=0.4,
            linestyle="none",
            label="UHI total (square)",
            markersize=9.5,
        ),
        Line2D(
            [0], [0],
            marker="^", color="none",
            markerfacecolor="0.40",
            markeredgecolor="black",
            markeredgewidth=0.4,
            linestyle="none",
            label="UCI total (triangle)",
            markersize=9.5,
        ),
    ]
    leg = ax.legend(
        handles=shape_handles,
        frameon=False,        # 开启图例边框/背景
        # facecolor="white",   # 设置纯白色底色 ✅
        # framealpha=1,        # 完全不透明（防止底色透图）        
        fontsize=LEGEND_FONT,
        loc="lower left",
        bbox_to_anchor=(LEGEND_X, LEGEND_Y),
        borderaxespad=0.0,
        handletextpad=0.35,
        labelspacing=0.20,                   # 行距压缩
        handlelength=1.0,                    # 标记到文字距离压缩
        borderpad=0.2,       # 轻微内边距，文字不贴边
    )

    # # 2. 统计文本 - 紧贴图例下方,同一左边界
    # ax.text(
    #     LEGEND_X, STATS_Y,
    #     f"UHI: n = {n_uhi:,}, {p_uhi_night:.0f}% night-dom.\n"
    #     f"UCI: n = {n_uci:,}, {p_uci_night:.0f}% night-dom.",
    #     transform=ax.transAxes,
    #     fontsize=LEGEND_FONT,
    #     ha="left", va="bottom",
    #     color="0.15",
    #     linespacing=1.15,                    # 行距与图例一致
    #     zorder=10,
    # )
    # 2. 生成统计文本并保存为 TXT 文件（同目录）
    stats_content = (
        f"UHI: n = {n_uhi:,}, {p_uhi_night:.0f}% > 0.\n"
        f"UCI: n = {n_uci:,}, {p_uci_night:.0f}% > 0."
    )

    # 保存到当前目录，文件名为 map_statistics.txt
    with open("map_statistics.txt", "w", encoding="utf-8") as f:
        f.write(stats_content)
    # ── 标题 + panel 编号 (保持不变) ──

    ax.set_title(
        "Heatwave-amplified day–night asymmetry",
        loc="center",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        pad=4.0,
    )

    # ── Colorbar (保持不变) ──
    if sc is not None:
        ax.figure.canvas.draw()
        pos = ax.get_position()

        cbar_width = 0.45 * pos.width
        cbar_left = pos.x0 + (pos.width - cbar_width) / 2.0
        cbar_height = 0.016
        cbar_pad = 0.030

        cax = ax.figure.add_axes([
            cbar_left,
            pos.y0 - cbar_pad,
            cbar_width,
            cbar_height,
        ])

        cb = plt.colorbar(
            sc, cax=cax,
            orientation="horizontal",
            extend="both",
        )

        ticks = np.linspace(-v_limit, v_limit, 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.2g}" for t in ticks])

        cb.ax.tick_params(labelsize=6.0, length=1.5, pad=1.0, width=0.55)
        cb.outline.set_linewidth(0.55)
        cb.outline.set_edgecolor("black")
        cb.set_label(
            r"$R_n - R_x$  (night$-$day difference, °C)",
            fontsize=6.5,
            labelpad=2.5,
        )

def render_combined_main_figure(
    a: Optional[pd.DataFrame],
    b: Optional[pd.DataFrame],
    c: Optional[pd.DataFrame],
    out_dir: Path,
    basename: str,
) -> List[Path]:
    """
    Combined main figure:
      Row 1: panel a (Robinson world map, reference-style)
      Row 2: panel b (sensitivity) | panel c (decomposition: UHI | UCI)

    Panels b and c visualize upstream urban-minus-rural (U-R) values.
    This plotting script does not recompute or reverse those differences.

    Layout规范:
      - figsize: (7.35, 7.45) 略微加高,容纳 Robinson 投影 + 下方 colorbar
      - panel a: 单独 add_axes, 宽度铺满 + Robinson aspect 决定高度
      - 下半行 b/c: GridSpec, 与 reference 风格字号一致
      - panel a 与下半行之间留出 colorbar + 间距空间
    """
    fig = plt.figure(figsize=(8.8, 9))
    PANEL_LABEL_FONTSIZE = 10
    PANEL_TITLE_FONTSIZE = 10
    PANEL_LABEL_X = -0.04
    PANEL_LABEL_Y = 1.04

    LEFT = 0.075         # 推荐 (略收一点,给 y 轴更大字号留位置)
    RIGHT = 0.975        # 推荐
    TOP = 0.955          # 不动
    BOTTOM = 0.130       # 推荐 (底部图例字号变大,但画布也大了,反而可以稍收)

    # panel a 单独定位 (覆盖到 b 左侧标签区域)
    MAP_LEFT = 0.015
    MAP_RIGHT = RIGHT

    # Robinson 投影的近似 aspect:
    # 数据范围 360° × 140° = 2.57 (PlateCarree)
    # Robinson 实际 aspect ≈ 2.04 (压缩极地)
    # 这里用一个稳健的视觉比例
    ROBINSON_VISUAL_ASPECT = 2.04

    fig_w, fig_h = fig.get_size_inches()
    map_width_frac = MAP_RIGHT - MAP_LEFT
    map_width_in = fig_w * map_width_frac
    map_height_in = map_width_in / ROBINSON_VISUAL_ASPECT
    map_height_frac = map_height_in / fig_h

    map_y1 = TOP
    map_y0 = map_y1 - map_height_frac

    # 给 panel a 的 colorbar (在地图下方) + 间距留空间
    lower_top = map_y0 - 0.110

    # panel a
    axa = fig.add_axes([MAP_LEFT, map_y0, map_width_frac, map_height_frac])

    # 下半行 panel b + c
    lower = fig.add_gridspec(
        1, 2,
        left=LEFT,
        right=RIGHT,
        bottom=BOTTOM,
        top=lower_top,
        width_ratios=[0.95, 1.25],
        wspace=0.40,
    )

    axb = fig.add_subplot(lower[0, 0])

    c_grid = lower[0, 1].subgridspec(1, 2, wspace=0.18)
    axc_uhi = fig.add_subplot(c_grid[0, 0])
    axc_uci = fig.add_subplot(c_grid[0, 1], sharey=axc_uhi)

    # 绘制各 panel
    draw_map_combined(axa, a, "a")
    draw_sensitivity_combined(axb, b, "b")

    fig.canvas.draw()

    pos_a = axa.get_position()
    pos_b = axb.get_position()

    LABEL_X = pos_b.x0 - 0.035
    LABEL_PAD_Y = 0.006

    fig.text(
        LABEL_X, pos_a.y1 + LABEL_PAD_Y,
        "a",
        ha="left", va="bottom",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )

    fig.text(
        LABEL_X, pos_b.y1 + LABEL_PAD_Y,
        "b",
        ha="left", va="bottom",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )
    fig.text(
        0.5 * (pos_b.x0 + pos_b.x1),
        pos_b.y1 + LABEL_PAD_Y,
        "Outcome-specific marginal sensitivities\nof modelled heat burdens",
        ha="center", va="bottom",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
    )
    if c is None or len(c) == 0:
        axc_uhi.text(
            0.5, 0.5, "Contribution unavailable",
            ha="center", va="center",
            transform=axc_uhi.transAxes,
            color="0.45",
        )
        axc_uhi.set_axis_off()
        axc_uci.set_axis_off()
    else:
        c_work = c.copy()
        c_work["uhi_uci_group"] = c_work["uhi_uci_group"].astype(str).str.upper()
        c_work["period_phase"] = c_work["period_phase"].astype(str).str.upper()

        # Use the same outcome normalization as the text export.
        # Labour loss is the intended burden-oriented outcome.
        c_work = _normalise_outcome_names(
            c_work,
            table_name="main_panel_c_additive_decomposition_for_panel_c",
        )

        # If both capacity and loss rows exist, retain labour loss only.
        c_work = _drop_labour_capacity_if_loss_present(
            c_work,
            table_name="main_panel_c_additive_decomposition_for_panel_c",
        )

        # Panel-c visual convention only:
        # labour is displayed as daytime-only contribution.
        # This must be applied BEFORE scale is calculated.
        c_work = _apply_panelc_labour_day_only_for_plot(
            c_work,
            table_name="main_panel_c_additive_decomposition_for_panel_c",
        )

        present = plot_outcome_order(c_work)

        if not present:
            axc_uhi.text(
                0.5, 0.5, "No plotted outcomes available",
                ha="center", va="center",
                transform=axc_uhi.transAxes,
                color="0.45",
            )
            axc_uhi.set_axis_off()
            axc_uci.set_axis_off()
        else:
            scale = _decomp_scale_by_outcome(c_work, present)

            _draw_decomp_column(
                axc_uhi, c_work, "UHI", present, scale,
                show_ylabels=True, title="UHI",
            )
            _draw_decomp_column(
                axc_uci, c_work, "UCI", present, scale,
                show_ylabels=False, title="UCI",
            )

            axc_uhi.set_xlabel("Urban–rural scaled contribution", fontsize=7.3)
            axc_uci.set_xlabel("Urban–rural scaled contribution", fontsize=7.3)

            pos_l = axc_uhi.get_position()
            pos_r = axc_uci.get_position()
            title_y = max(pos_l.y1, pos_r.y1) + 0.018
            title_x = 0.5 * (pos_l.x0 + pos_r.x1)

            fig.text(
                pos_l.x0 - 0.035,
                title_y,
                "c",
                ha="left", va="bottom",
                fontsize=PANEL_LABEL_FONTSIZE,
                fontweight="bold",
            )


            fig.text(
                title_x,
                title_y,
                "Outcome-relevant urban–rural contributions\nto total modelled burden",
                ha="center", va="bottom",
                fontsize=10,
                fontweight="bold",
            )


    # 底部统一图例
    legend_handles = [
        Line2D(
            [0], [0], marker="s", color="none",
            markerfacecolor=GROUP_COLORS["UHI"],
            markeredgecolor="white",
            label="UHI (square)",
            linestyle="none", markersize=5.6,
        ),
        Line2D(
            [0], [0], marker="^", color="none",
            markerfacecolor=GROUP_COLORS["UCI"],
            markeredgecolor="white",
            label="UCI (triangle)",
            linestyle="none", markersize=6.0,
        ),
        Line2D(
            [0], [0], marker="o", color=COLORS["NHW"],
            markerfacecolor="white",
            markeredgecolor=COLORS["NHW"],
            label="NHW", linestyle="none", markersize=5.0,
        ),
        Line2D(
            [0], [0], marker="o", color=COLORS["HW"],
            markerfacecolor=COLORS["HW"],
            markeredgecolor=COLORS["HW"],
            label="HW", linestyle="none", markersize=5.0,
        ),
        Patch(
            facecolor=COMPONENT_LEGEND_COLORS["day"],
            edgecolor="none",
            label="Day (lighter shade)",
        ),
        Patch(
            facecolor=COMPONENT_LEGEND_COLORS["night"],
            edgecolor="none",
            label="Night (darker shade)",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=6.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.038),
        ncol=6,
        columnspacing=0.78,
        handletextpad=0.38,
        borderaxespad=0.0,
    )

    png = safe_output_path(out_dir / f"{basename}.png")
    pdf = safe_output_path(out_dir / f"{basename}.pdf")

    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[save] {png}")
    print(f"[save] {pdf}")

    return [png, pdf]


def draw_sensitivity(ax, data: pd.DataFrame, group: str, label: str = "b") -> None:
    data = filt(data, group)

    base_req = {
        "outcome",
        "period_phase",
    }

    if data is None or len(data) == 0 or not base_req.issubset(data.columns):
        ax.text(
            0.5,
            0.5,
            f"{group} sensitivity unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777",
        )
        ax.set_axis_off()
        return

    est_col, lo_col, hi_col = _panel_b_numeric_columns(data)
    if not {est_col, lo_col, hi_col}.issubset(data.columns):
        ax.text(
            0.5,
            0.5,
            f"{group} sensitivity unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777",
        )
        ax.set_axis_off()
        return

    present = plot_panel_b_outcome_order(data)

    ybase = np.arange(len(present))[::-1]
    off = {"NHW": -0.13, "HW": 0.13}

    for i, oc in enumerate(present):
        sub_oc = data[data["outcome"] == oc].copy()

        for c in [est_col, lo_col, hi_col]:
            sub_oc[c] = pd.to_numeric(sub_oc[c], errors="coerce")

        scale_vals = pd.concat([
            sub_oc[est_col].abs(),
            sub_oc[lo_col].abs(),
            sub_oc[hi_col].abs(),
        ])

        scale = max(
            float(np.nanmax(scale_vals.values))
            if np.isfinite(scale_vals.values).any()
            else 1.0,
            1e-12,
        )

        for ph in ["NHW", "HW"]:
            sub = sub_oc[sub_oc["period_phase"] == ph]

            if len(sub) == 0:
                continue

            r = sub.iloc[0]
            yy = ybase[i] + off[ph]

            x = float(r[est_col]) / scale
            lo = float(r[lo_col]) / scale
            hi = float(r[hi_col]) / scale

            ax.plot([lo, hi], [yy, yy], color=COLORS[ph], lw=1.15, zorder=3)
            ax.scatter(
                x,
                yy,
                color=COLORS[ph],
                s=30 if ph == "HW" else 24,
                marker="o" if ph == "HW" else "s",
                edgecolor="white",
                lw=0.5,
                label=ph if i == 0 else None,
                zorder=4,
            )

        ax.axhline(ybase[i] - 0.5, color="0.94", lw=0.6, zorder=0)

    ax.axvline(0, color="0.45", lw=0.7, ls=":")
    ax.set_xlim(-1.18, 1.18)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["−1", "−1/2", "0", "+1/2", "+1"])

    ax.set_yticks(ybase)
    ax.set_yticklabels([outcome_display_label(o) for o in present], fontsize=8)

    ax.set_xlabel(
        "Scaled urban−rural marginal sensitivity contrast\n"
        "negative: rural more sensitive; positive: urban more sensitive",
        fontsize=7.5,
    )

    ax.set_title(
        f"{label}  {group}: urban−rural marginal sensitivity contrast",
        loc="left",
        fontweight="bold",
        fontsize=9,
        pad=4,
    )

    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_sensitivity_combined(ax, data: pd.DataFrame, label: str = "b") -> None:
    base_req = {
        "uhi_uci_group",
        "outcome",
        "period_phase",
    }

    if data is None or len(data) == 0 or not base_req.issubset(data.columns):
        ax.text(
            0.5, 0.5, "Sensitivity unavailable",
            ha="center", va="center",
            transform=ax.transAxes,
            color="0.45",
        )
        ax.set_axis_off()
        return

    data = data.copy()
    data["uhi_uci_group"] = data["uhi_uci_group"].astype(str).str.upper()
    data["period_phase"] = data["period_phase"].astype(str).str.upper()

    est_col, lo_col, hi_col = _panel_b_numeric_columns(data)
    if not {est_col, lo_col, hi_col}.issubset(data.columns):
        ax.text(
            0.5, 0.5, "Sensitivity unavailable",
            ha="center", va="center",
            transform=ax.transAxes,
            color="0.45",
        )
        ax.set_axis_off()
        return

    present = plot_panel_b_outcome_order(data)
    if not present:
        ax.text(
            0.5, 0.5, "No plotted outcomes available",
            ha="center", va="center",
            transform=ax.transAxes,
            color="0.45",
        )
        ax.set_axis_off()
        return

    ybase = np.arange(len(present))[::-1]

    offsets = {
        ("UHI", "NHW"): -0.24,
        ("UHI", "HW"): -0.08,
        ("UCI", "NHW"): 0.08,
        ("UCI", "HW"): 0.24,
    }

    for i, oc in enumerate(present):
        sub_oc = data[data["outcome"] == oc].copy()

        for c in [est_col, lo_col, hi_col]:
            sub_oc[c] = pd.to_numeric(sub_oc[c], errors="coerce")

        scale_vals = pd.concat([
            sub_oc[est_col].abs(),
            sub_oc[lo_col].abs(),
            sub_oc[hi_col].abs(),
        ])

        scale = max(
            float(np.nanmax(scale_vals.values))
            if np.isfinite(scale_vals.values).any()
            else 1.0,
            1e-12,
        )

        for group in ["UHI", "UCI"]:
            group_color = GROUP_COLORS[group]

            for ph in ["NHW", "HW"]:
                sub = sub_oc[
                    (sub_oc["uhi_uci_group"] == group)
                    & (sub_oc["period_phase"] == ph)
                ]
                if len(sub) == 0:
                    continue

                r = sub.iloc[0]
                yy = ybase[i] + offsets[(group, ph)]
                x = float(r[est_col]) / scale
                lo = float(r[lo_col]) / scale
                hi = float(r[hi_col]) / scale

                ax.plot(
                    [lo, hi],
                    [yy, yy],
                    color=group_color,
                    lw=1.15,
                    solid_capstyle="round",
                    zorder=3,
                )

                m_face = group_color if ph == "HW" else "white"

                ax.scatter(
                    x,
                    yy,
                    marker=GROUP_MARKERS[group],
                    s=PANEL_POINT_SIZE[group],
                    facecolor=m_face,
                    edgecolor=group_color,
                    linewidth=1.0,
                    zorder=4,
                )

        ax.axhline(ybase[i] - 0.5, color="0.93", lw=0.55, zorder=0)

    ax.axvline(0, color="0.30", lw=0.8, ls=":", zorder=1)

    # All three outcomes use signed urban-minus-rural contrasts, so the shared
    # display range must include negatives.
    ax.set_xlim(-1.18, 1.18)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["−1", "−1/2", "0", "+1/2", "+1"])

    ax.text(
        0.01,
        0.98,
        "Rural more sensitive",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="0.42",
    )

    ax.text(
        0.99,
        0.98,
        "Urban more sensitive",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color="0.42",
    )

    ax.set_yticks(ybase)
    ax.set_yticklabels([outcome_display_label(o) for o in present], fontsize=8)

    ax.set_xlabel(
        "Scaled urban−rural marginal sensitivity contrast\n"
        "negative: rural more sensitive; positive: urban more sensitive",
        fontsize=7.5,
    )

    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=3, width=0.7)



def draw_decomp(ax, data: pd.DataFrame, group: str, label: str = "c") -> None:
    data = filt(data, group)

    req = {
        "outcome",
        "period_phase",
        "component",
        "estimate",
        "ci_low",
        "ci_high",
    }

    if data is None or len(data) == 0 or not req.issubset(data.columns):
        ax.text(
            0.5,
            0.5,
            f"{group} contribution unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777",
        )
        ax.set_axis_off()
        return

    present = plot_panel_b_outcome_order(data)

    ypos: Dict[Tuple[str, str], float] = {}
    y = 0.0

    for oc in present[::-1]:
        for ph in ["NHW", "HW"]:
            ypos[(oc, ph)] = y
            y += 1.0
        y += 0.45

    scale: Dict[str, float] = {}

    for oc in present:
        s = data[data["outcome"] == oc].copy()

        for c in ["estimate", "ci_low", "ci_high"]:
            s[c] = pd.to_numeric(s[c], errors="coerce")

        vals = pd.concat([
            s["estimate"].abs(),
            s["ci_low"].abs(),
            s["ci_high"].abs(),
        ])

        scale[oc] = max(
            float(np.nanmax(vals.values))
            if np.isfinite(vals.values).any()
            else 1.0,
            1e-12,
        )

    for oc in present:
        for ph in ["NHW", "HW"]:
            yy = ypos[(oc, ph)]

            sub = data[(data["outcome"] == oc) & (data["period_phase"] == ph)].copy()

            if len(sub) == 0:
                continue

            for c in ["estimate", "ci_low", "ci_high"]:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")

            lp, ln = 0.0, 0.0

            for comp in ["day", "night"]:
                r = sub[sub["component"] == comp]
                if len(r) == 0:
                    continue

                v = float(r["estimate"].iloc[0]) / scale[oc]

                if v >= 0:
                    ax.barh(yy, v, left=lp, height=0.34, color=GROUP_COMPONENT_COLORS[group][comp], edgecolor="white", lw=0.4, zorder=3)
                    lp += v
                else:
                    ax.barh(yy, v, left=ln, height=0.34, color=GROUP_COMPONENT_COLORS[group][comp], edgecolor="white", lw=0.4, zorder=3)
                    ln += v

            r = sub[sub["component"] == "total"]

            if len(r):
                rr = r.iloc[0]
                est = float(rr["estimate"]) / scale[oc]
                lo = float(rr["ci_low"]) / scale[oc]
                hi = float(rr["ci_high"]) / scale[oc]

                ax.plot([lo, hi], [yy, yy], color=GROUP_COLORS[group], lw=1.0, zorder=5)
                ax.scatter(est, yy, marker=GROUP_MARKERS[group],
                           facecolor=GROUP_COLORS[group], edgecolor=GROUP_COLORS[group],
                           s=22, zorder=6)

            ax.text(-1.16, yy, ph, ha="right", va="center", fontsize=7, color=COLORS[ph])

    ax.axvline(0, color="0.45", lw=0.7, ls=":")

    ax.set_yticks([np.mean([ypos[(oc, "NHW")], ypos[(oc, "HW")]]) for oc in present])
    ax.set_yticklabels([outcome_display_label(o) for o in present], fontsize=8)

    ax.set_xlim(-1.18, 1.18)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["-1", "-1/2", "0", "+1/2", "+1"])

    ax.set_xlabel(
        "Signed contribution to urban-rural impact difference"
        "\nsleep: beta × U-R temperature delta; energy: direct U-R CDH; labour: Dunne Tx-based U-R labour loss",
        fontsize=7.2,
    )

    ax.set_title(f"{label}  {group}: day and night contributions", loc="left", fontweight="bold", fontsize=9, pad=4)

    ax.legend(
        handles=[
            Patch(facecolor=GROUP_COMPONENT_COLORS[group]["day"], label="Day (lighter shade)"),
            Patch(facecolor=GROUP_COMPONENT_COLORS[group]["night"], label="Night (darker shade)"),
            Line2D([0], [0], marker=GROUP_MARKERS[group], color=GROUP_COLORS[group],
                   markerfacecolor=GROUP_COLORS[group], markeredgecolor=GROUP_COLORS[group],
                   label="Total", ls="none", markersize=4.5),
        ],
        frameon=False,
        fontsize=7,
        loc="lower right",
        ncol=2,
    )

    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

def draw_decomp_combined(ax, data: pd.DataFrame, label: str = "c") -> None:
    req = {
        "uhi_uci_group",
        "outcome",
        "period_phase",
        "component",
        "estimate",
        "ci_low",
        "ci_high",
    }

    if data is None or len(data) == 0 or not req.issubset(data.columns):
        ax.text(
            0.5, 0.5, "Contribution unavailable",
            ha="center", va="center",
            transform=ax.transAxes,
            color="0.45",
        )
        ax.set_axis_off()
        return

    data = data.copy()
    data["uhi_uci_group"] = data["uhi_uci_group"].astype(str).str.upper()
    data["period_phase"] = data["period_phase"].astype(str).str.upper()

    data = _normalise_outcome_names(
        data,
        table_name="draw_decomp_combined",
    )

    data = _drop_labour_capacity_if_loss_present(
        data,
        table_name="draw_decomp_combined",
    )

    data = _apply_panelc_labour_day_only_for_plot(
        data,
        table_name="draw_decomp_combined",
    )

    present = plot_panel_b_outcome_order(data)

    ypos: Dict[Tuple[str, str, str], float] = {}
    y = 0.0

    for oc in present[::-1]:
        for group in ["UHI", "UCI"]:
            for ph in ["NHW", "HW"]:
                ypos[(oc, group, ph)] = y
                y += 0.68
        y += 0.42

    scale: Dict[str, float] = {}

    for oc in present:
        s = data[data["outcome"] == oc].copy()

        for c in ["estimate", "ci_low", "ci_high"]:
            s[c] = pd.to_numeric(s[c], errors="coerce")

        vals = pd.concat([
            s["estimate"].abs(),
            s["ci_low"].abs(),
            s["ci_high"].abs(),
        ])
        scale[oc] = max(
            float(np.nanmax(vals.values))
            if np.isfinite(vals.values).any()
            else 1.0,
            1e-12,
        )

    label_trans = blended_transform_factory(ax.transAxes, ax.transData)

    for oc in present:
        for group in ["UHI", "UCI"]:
            for ph in ["NHW", "HW"]:
                yy = ypos[(oc, group, ph)]

                sub = data[
                    (data["outcome"] == oc)
                    & (data["uhi_uci_group"] == group)
                    & (data["period_phase"] == ph)
                ].copy()

                if len(sub) == 0:
                    continue

                for c in ["estimate", "ci_low", "ci_high"]:
                    sub[c] = pd.to_numeric(sub[c], errors="coerce")

                lp, ln = 0.0, 0.0

                for comp in ["day", "night"]:
                    r = sub[sub["component"] == comp]
                    if len(r) == 0:
                        continue

                    v = float(r["estimate"].iloc[0]) / scale[oc]

                    if v >= 0:
                        ax.barh(
                            yy,
                            v,
                            left=lp,
                            height=0.28,
                            color=GROUP_COMPONENT_COLORS[group][comp],
                            edgecolor="white",
                            lw=0.35,
                            zorder=3,
                        )
                        lp += v
                    else:
                        ax.barh(
                            yy,
                            v,
                            left=ln,
                            height=0.28,
                            color=GROUP_COMPONENT_COLORS[group][comp],
                            edgecolor="white",
                            lw=0.35,
                            zorder=3,
                        )
                        ln += v

                r = sub[sub["component"] == "total"]
                if len(r):
                    rr = r.iloc[0]
                    est = float(rr["estimate"]) / scale[oc]
                    lo = float(rr["ci_low"]) / scale[oc]
                    hi = float(rr["ci_high"]) / scale[oc]

                    ax.plot(
                        [lo, hi],
                        [yy, yy],
                        color=COLORS["total"],
                        lw=1.0,
                        solid_capstyle="round",
                        zorder=5,
                    )
                    ax.scatter(est, yy, color=COLORS["total"], s=18, zorder=6)

                row_color = COLORS["HW"] if ph == "HW" else "0.45"
                ax.text(
                    0.016,
                    yy,
                    f"{group}-{ph}",
                    transform=label_trans,
                    ha="left",
                    va="center",
                    fontsize=5.9,
                    color=row_color,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.35),
                    zorder=8,
                )

    ax.axvline(0, color="0.40", lw=0.7, ls=":")

    ax.set_yticks([
        np.mean([
            ypos[(oc, "UHI", "NHW")],
            ypos[(oc, "UHI", "HW")],
            ypos[(oc, "UCI", "NHW")],
            ypos[(oc, "UCI", "HW")],
        ])
        for oc in present
    ])
    ax.set_yticklabels([outcome_display_label(o) for o in present], fontsize=8)

    ax.set_xlim(-1.18, 1.18)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["-1", "-1/2", "0", "+1/2", "+1"])

    ax.set_xlabel("Signed contribution, scaled within outcome", fontsize=7.5)

    ax.set_title(
        f"{label}  Day and night contributions",
        loc="left",
        fontweight="bold",
        fontsize=9.5,
        pad=4,
    )

    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=3, width=0.7)

def _is_labour_outcome_for_plot(outcome: str) -> bool:
    """
    Plot-level rule only.

    Labour in panel c is shown as daytime-only contribution:
        day = original day estimate
        night = 0
        total = day

    This does not modify upstream CSV values.
    """
    key = str(outcome).strip().lower()
    return key in {
        "labour_capacity",
        "labor_capacity",
        "labour_loss",
        "labor_loss",
    }

def _decomp_scale_by_outcome(data: pd.DataFrame, present: Sequence[str]) -> Dict[str, float]:
    """
    Use one scale per outcome across both UHI/UCI and HW/NHW.
    This keeps the two columns in panel c directly comparable within each outcome.
    """
    scale: Dict[str, float] = {}

    for oc in present:
        s = data[data["outcome"] == oc].copy()

        for c in ["estimate", "ci_low", "ci_high"]:
            if c in s.columns:
                s[c] = pd.to_numeric(s[c], errors="coerce")

        vals = pd.concat([
            s["estimate"].abs(),
            s["ci_low"].abs(),
            s["ci_high"].abs(),
        ])

        scale[oc] = max(
            float(np.nanmax(vals.values))
            if np.isfinite(vals.values).any()
            else 1.0,
            1e-12,
        )

    return scale


def _draw_decomp_column(
    ax,
    data: pd.DataFrame,
    group: str,
    present: Sequence[str],
    scale: Dict[str, float],
    *,
    show_ylabels: bool,
    title: str,
) -> None:
    """
    Draw one column of panel c.

    Rows    = outcomes
    Column  = UHI or UCI
    Within each row: NHW and HW decomposition bars.

    Plot-level labour rule:
        For labour_capacity / labour_loss only:
            day   = original day-hour component
            night = 0
            total = day

    This is only a panel-c plotting convention and does not modify the
    upstream CSV or upstream labour calculation.
    """
    sub_group = data[data["uhi_uci_group"] == group].copy()
    ybase = np.arange(len(present))[::-1]

    phase_offsets = {
        "NHW": -0.15,
        "HW": 0.15,
    }

    def _get_component_row(sub_df: pd.DataFrame, comp: str):
        r = sub_df[sub_df["component"] == comp]
        if len(r) == 0:
            return None
        return r.iloc[0]

    def _safe_num(row, col, default=np.nan):
        if row is None or col not in row.index:
            return default
        return pd.to_numeric(row[col], errors="coerce")

    for i, oc in enumerate(present):
        base_y = ybase[i]
        is_labour = _is_labour_outcome_for_plot(oc)

        for ph in ["NHW", "HW"]:
            yy = base_y + phase_offsets[ph]

            sub = sub_group[
                (sub_group["outcome"] == oc)
                & (sub_group["period_phase"] == ph)
            ].copy()

            if len(sub) == 0:
                continue

            for cnum in ["estimate", "ci_low", "ci_high"]:
                sub[cnum] = pd.to_numeric(sub[cnum], errors="coerce")

            r_day = _get_component_row(sub, "day")
            r_night = _get_component_row(sub, "night")
            r_total = _get_component_row(sub, "total")

            left_pos = 0.0
            left_neg = 0.0

            for comp in ["day", "night"]:
                if is_labour and comp == "night":
                    # Labour has no night-hour contribution in this figure.
                    # Keep it as 0 conceptually, but do not draw a zero-width bar.
                    continue

                r = r_day if comp == "day" else r_night
                if r is None:
                    continue

                v_raw = _safe_num(r, "estimate")
                if not np.isfinite(v_raw):
                    continue

                v = float(v_raw) / scale[oc]
                if not np.isfinite(v):
                    continue

                if v >= 0:
                    ax.barh(
                        yy,
                        v,
                        left=left_pos,
                        height=0.23,
                        color=GROUP_COMPONENT_COLORS[group][comp],
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=3,
                    )
                    left_pos += v
                else:
                    ax.barh(
                        yy,
                        v,
                        left=left_neg,
                        height=0.23,
                        color=GROUP_COMPONENT_COLORS[group][comp],
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=3,
                    )
                    left_neg += v

            if is_labour:
                est_raw = _safe_num(r_day, "estimate")
                lo_raw = _safe_num(r_day, "ci_low")
                hi_raw = _safe_num(r_day, "ci_high")
            else:
                if r_total is None:
                    day_raw = _safe_num(r_day, "estimate", 0.0)
                    night_raw = _safe_num(r_night, "estimate", 0.0)
                    est_raw = day_raw + night_raw
                    lo_raw = np.nan
                    hi_raw = np.nan
                else:
                    est_raw = _safe_num(r_total, "estimate")
                    lo_raw = _safe_num(r_total, "ci_low")
                    hi_raw = _safe_num(r_total, "ci_high")

            if np.isfinite(est_raw):
                est = float(est_raw) / scale[oc]
                lo = float(lo_raw) / scale[oc] if np.isfinite(lo_raw) else np.nan
                hi = float(hi_raw) / scale[oc] if np.isfinite(hi_raw) else np.nan

                marker = GROUP_MARKERS[group]

                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot(
                        [lo, hi],
                        [yy, yy],
                        color=GROUP_COLORS[group],
                        lw=1.05,
                        solid_capstyle="round",
                        zorder=5,
                    )

                ax.scatter(
                    est,
                    yy,
                    marker=marker,
                    s=PANEL_POINT_SIZE[group],
                    facecolor=(GROUP_COLORS[group] if ph == "HW" else "white"),
                    edgecolor=GROUP_COLORS[group],
                    linewidth=0.90,
                    zorder=6,
                )

            ax.text(
                -1.15,
                yy,
                ph,
                ha="right",
                va="center",
                fontsize=6.2,
                color=(GROUP_COLORS[group] if ph == "HW" else COLORS["NHW"]),
            )

        ax.axhline(base_y - 0.5, color="0.93", lw=0.55, zorder=0)

    ax.axvline(0, color="0.40", lw=0.7, ls=":")
    ax.set_xlim(-1.18, 1.18)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["-1", "-1/2", "0", "+1/2", "+1"])

    ax.set_yticks(ybase)

    if show_ylabels:
        ax.set_yticklabels([outcome_display_label(o) for o in present], fontsize=8)
    else:
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=8.2, fontweight="bold", pad=3)
    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=3, width=0.7)

def make_decomp_percent_data(data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Convert raw day/night/total decomposition estimates into signed percentages.

    Percent definition:
        day_percent   = 100 * raw_day   / raw_total
        night_percent = 100 * raw_night / raw_total
        total_percent = 100 * raw_total / raw_total = 100

    No absolute denominator is used.

    If raw_total is zero or nearly zero, percentage is undefined.
    These rows are retained and flagged as percent_defined=False.
    """
    req = {
        "uhi_uci_group",
        "outcome",
        "period_phase",
        "component",
        "estimate",
        "ci_low",
        "ci_high",
    }

    if data is None or len(data) == 0 or not req.issubset(data.columns):
        return pd.DataFrame()

    work = data.copy()
    work["uhi_uci_group"] = work["uhi_uci_group"].astype(str).str.upper()
    work["outcome"] = work["outcome"].astype(str).str.strip()

    aliases = {
        "labour_capacity": "labour_capacity",
        "labor_capacity": "labour_capacity",
        "work_capacity": "labour_capacity",
        "labour_work_capacity": "labour_capacity",
        "labor_work_capacity": "labour_capacity",
        "labour_capacity_loss": "labour_loss",
        "labor_capacity_loss": "labour_loss",
        "labour_loss": "labour_loss",
        "labor_loss": "labour_loss",
    }
    work["outcome"] = work["outcome"].replace(aliases)

    for c in ["estimate", "ci_low", "ci_high"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    present = plot_outcome_order(work)
    rows: List[Dict[str, Any]] = []

    for oc in present:
        for group in ["UHI", "UCI"]:
            for ph in ["NHW", "HW"]:
                sub = work[
                    (work["outcome"] == oc)
                    & (work["uhi_uci_group"] == group)
                    & (work["period_phase"] == ph)
                ].copy()


                def get_component(comp: str) -> Optional[pd.Series]:
                    r = sub[sub["component"] == comp]
                    if len(r) == 0:
                        return None
                    return r.iloc[0]

                r_day = get_component("day")
                r_night = get_component("night")
                r_total = get_component("total")

                if r_day is None or r_night is None:
                    continue

                day_est = float(r_day["estimate"])
                night_est = float(r_night["estimate"])

                if r_total is not None and np.isfinite(float(r_total["estimate"])):
                    total_est = float(r_total["estimate"])
                else:
                    total_est = day_est + night_est

                denom = total_est
                percent_defined = bool(np.isfinite(denom) and abs(denom) >= 1e-12)

                for comp in ["day", "night", "total"]:
                    r = get_component(comp)
                    if r is None:
                        continue

                    est = float(r["estimate"])
                    lo = float(r["ci_low"])
                    hi = float(r["ci_high"])

                    if percent_defined:
                        estimate_pct = 100.0 * est / denom
                        ci_low_pct = 100.0 * lo / denom
                        ci_high_pct = 100.0 * hi / denom

                        if np.isfinite(ci_low_pct) and np.isfinite(ci_high_pct):
                            ci_low_pct, ci_high_pct = sorted([ci_low_pct, ci_high_pct])
                    else:
                        estimate_pct = np.nan
                        ci_low_pct = np.nan
                        ci_high_pct = np.nan

                    rows.append({
                        "outcome": oc,
                        "uhi_uci_group": group,
                        "period_phase": ph,
                        "component": comp,
                        "estimate_pct": estimate_pct,
                        "ci_low_pct": ci_low_pct,
                        "ci_high_pct": ci_high_pct,
                        "raw_estimate": est,
                        "raw_ci_low": lo,
                        "raw_ci_high": hi,
                        "raw_day": day_est,
                        "raw_night": night_est,
                        "raw_total": total_est,
                        "percent_denominator": denom,
                        "percent_defined": percent_defined,
                    })

    return pd.DataFrame(rows)


def draw_decomp_combined_percent(ax, data: pd.DataFrame, label: str = "d") -> None:
    data_pct = make_decomp_percent_data(data)

    req = {
        "uhi_uci_group",
        "outcome",
        "period_phase",
        "component",
        "estimate_pct",
        "ci_low_pct",
        "ci_high_pct",
        "percent_defined",
    }

    if data_pct is None or len(data_pct) == 0 or not req.issubset(data_pct.columns):
        ax.text(
            0.5,
            0.5,
            "Percentage contribution unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777",
        )
        ax.set_axis_off()
        return

    present = plot_panel_b_outcome_order(data)

    ypos: Dict[Tuple[str, str, str], float] = {}
    y = 0.0

    for oc in present[::-1]:
        for group in ["UHI", "UCI"]:
            for ph in ["NHW", "HW"]:
                ypos[(oc, group, ph)] = y
                y += 0.72
        y += 0.42

    label_trans = blended_transform_factory(ax.transAxes, ax.transData)
    all_x: List[float] = []

    for oc in present:
        for group in ["UHI", "UCI"]:
            for ph in ["NHW", "HW"]:
                yy = ypos[(oc, group, ph)]

                sub = data_pct[
                    (data_pct["outcome"] == oc)
                    & (data_pct["uhi_uci_group"] == group)
                    & (data_pct["period_phase"] == ph)
                ].copy()

                if len(sub) == 0:
                    continue

                percent_defined = bool(sub["percent_defined"].iloc[0])

                ax.text(
                    0.012,
                    yy,
                    f"{group}-{ph}",
                    transform=label_trans,
                    ha="left",
                    va="center",
                    fontsize=6.2,
                    color=COLORS[ph],
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.5),
                    zorder=8,
                )

                if not percent_defined:
                    ax.scatter(0, yy, marker="x", s=26, color="0.25", linewidths=1.0, zorder=6)
                    ax.text(4, yy, "net total ≈ 0", ha="left", va="center", fontsize=6.1, color="0.35", zorder=6)
                    continue

                lp, ln = 0.0, 0.0

                for comp in ["day", "night"]:
                    r = sub[sub["component"] == comp]
                    if len(r) == 0:
                        continue

                    v = float(r["estimate_pct"].iloc[0])
                    if not np.isfinite(v):
                        continue

                    all_x.append(v)

                    if v >= 0:
                        ax.barh(yy, v, left=lp, height=0.30, color=GROUP_COMPONENT_COLORS[group][comp], edgecolor="white", lw=0.4, zorder=3)
                        lp += v
                    else:
                        ax.barh(yy, v, left=ln, height=0.30, color=GROUP_COMPONENT_COLORS[group][comp], edgecolor="white", lw=0.4, zorder=3)
                        ln += v

                r = sub[sub["component"] == "total"]

                if len(r):
                    rr = r.iloc[0]
                    est = float(rr["estimate_pct"])
                    lo = float(rr["ci_low_pct"])
                    hi = float(rr["ci_high_pct"])

                    if np.isfinite(est):
                        all_x.append(est)
                        ax.scatter(est, yy, color=COLORS["total"], s=18, zorder=6)

                    if np.isfinite(lo) and np.isfinite(hi):
                        all_x.extend([lo, hi])
                        ax.plot([lo, hi], [yy, yy], color=COLORS["total"], lw=1.0, zorder=5)

    ax.axvline(0, color="0.45", lw=0.7, ls=":")
    ax.axvline(100, color="0.70", lw=0.6, ls="--")

    ax.set_yticks([
        np.mean([ypos[(oc, "UHI", "NHW")], ypos[(oc, "UHI", "HW")], ypos[(oc, "UCI", "NHW")], ypos[(oc, "UCI", "HW")]])
        for oc in present
    ])
    ax.set_yticklabels([outcome_display_label(o) for o in present], fontsize=8)

    finite_x = np.array([x for x in all_x if np.isfinite(x)], dtype=float)

    if len(finite_x):
        xmax = max(float(np.nanpercentile(np.abs(finite_x), 95)), 120.0)
    else:
        xmax = 120.0

    xmax = min(xmax, 500.0)
    xmax = float(np.ceil(1.08 * xmax / 50.0) * 50.0)

    ax.set_xlim(-xmax, xmax)
    xticks = np.linspace(-xmax, xmax, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.0f}%" for x in xticks])

    ax.set_xlabel(
        "Signed contribution as percentage of net day-night total"
        "\ncomponent percentage = 100 × raw component / raw total; no absolute denominator"
        "\nlabour: Dunne Tx-based U-R labour loss; positive = urban > rural",
        fontsize=7.2,
    )

    ax.set_title(
        f"{label}  Percentage day and night contributions\nusing signed raw totals",
        loc="left",
        fontweight="bold",
        fontsize=9,
        pad=4,
    )

    ax.legend(
        handles=[
            Patch(facecolor=COMPONENT_LEGEND_COLORS["day"], label="Day (lighter shade)"),
            Patch(facecolor=COMPONENT_LEGEND_COLORS["night"], label="Night (darker shade)"),
            Line2D([0], [0], marker="o", color=COLORS["total"], label="Total", ls="none", markersize=4),
            Line2D([0], [0], marker="x", color="0.25", label="Net total ≈ 0", ls="none", markersize=4.5),
        ],
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def render_main(
    group: str,
    a: Optional[pd.DataFrame],
    b: Optional[pd.DataFrame],
    c: Optional[pd.DataFrame],
    out_dir: Path,
    basename: str,
) -> List[Path]:
    fig = plt.figure(figsize=(7.2, 6.2))

    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.16, 1.0],
        width_ratios=[1, 1],
        hspace=0.50,
        wspace=0.34,
        left=0.075,
        right=0.965,
        top=0.955,
        bottom=0.090,
    )

    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])

    draw_map(axa, a, group, "a")
    draw_sensitivity(axb, b, group, "b")
    draw_decomp(axc, c, group, "c")

    png = safe_output_path(out_dir / f"{basename}_{group.lower()}.png")
    pdf = safe_output_path(out_dir / f"{basename}_{group.lower()}.pdf")

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[save] {png}")
    print(f"[save] {pdf}")

    return [png, pdf]


def render_combined_decomp_percent_figure(
    c: Optional[pd.DataFrame],
    out_dir: Path,
    basename: str,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    draw_decomp_combined_percent(ax, c, "d")

    fig.subplots_adjust(
        left=0.135,
        right=0.965,
        top=0.900,
        bottom=0.245,
    )

    png = safe_output_path(out_dir / f"{basename}.png")
    pdf = safe_output_path(out_dir / f"{basename}.pdf")

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[save] {png}")
    print(f"[save] {pdf}")

    return [png, pdf]

def render_supp(
    data: Optional[pd.DataFrame],
    out_dir: Path,
    basename: str,
) -> List[Path]:
    if data is None or len(data) == 0:
        print("[supp] skipped: no data")
        return []

    req = {
        "uhi_uci_group",
        "variable",
        "period_phase",
        "exposure_period",
        "delta_value",
    }

    if not req.issubset(data.columns):
        print(f"[supp] skipped: missing {sorted(req - set(data.columns))}")
        return []

    data = data.copy()
    data["delta_value"] = pd.to_numeric(data["delta_value"], errors="coerce")
    data["uhi_uci_group"] = data["uhi_uci_group"].astype(str).str.upper()
    data = data.dropna(subset=["variable", "period_phase", "exposure_period", "delta_value"])

    vars_ = [
        v for v in ["Sleep air temperature", "Building cooling degree-hours"]
        if v in set(data["variable"])
    ]

    if not vars_:
        print("[supp] skipped: no variables")
        return []

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)

    for ax, group in zip(np.atleast_1d(axes), UHI_UCI_GROUPS):
        suball = data[data["uhi_uci_group"] == group]

        positions: List[float] = []
        values: List[np.ndarray] = []
        colors: List[str] = []
        centers: List[float] = []
        labels: List[str] = []

        for i, var in enumerate(vars_):
            center = i + 1
            centers.append(center)
            labels.append(variable_display_label(var))

            offsets = {
                ("NHW", "day"): -0.30,
                ("NHW", "night"): -0.10,
                ("HW", "day"): 0.10,
                ("HW", "night"): 0.30,
            }

            for ph in ["NHW", "HW"]:
                for ep in ["day", "night"]:
                    vals = suball[
                        (suball["variable"] == var)
                        & (suball["period_phase"] == ph)
                        & (suball["exposure_period"] == ep)
                    ]["delta_value"].dropna().values

                    positions.append(center + offsets[(ph, ep)])
                    values.append(vals if len(vals) else np.array([np.nan]))
                    colors.append(GROUP_COMPONENT_COLORS[group][ep])

        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.16,
            patch_artist=True,
            showfliers=False,
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("0.25")
            patch.set_linewidth(0.7)

        for key in ["medians", "whiskers", "caps"]:
            for artist in bp[key]:
                artist.set_color("0.25")
                artist.set_linewidth(0.8)

        ax.axhline(0, color="0.45", lw=0.7, ls=":")
        ax.set_xlim(0.45, len(vars_) + 0.55)
        ax.set_ylabel("Urban - rural\ncomponent difference\npositive: urban > rural")
        ax.set_title(
            f"{group}: empirical day/night component differences",
            loc="left",
            fontweight="bold",
            fontsize=9,
            pad=4,
        )

        add_panel_note(
            ax,
            "Energy variable: U-R CDH delta, not E_comm/E_resi",
            xy=(0.98, 0.04),
            ha="right",
            fontsize=6.2,
        )

        ymin, ymax = ax.get_ylim()
        yy = ymin - 0.08 * max(ymax - ymin, 1e-6)

        for center in centers:
            ax.text(center - 0.20, yy, "NHW", ha="center", va="top", fontsize=6.5, color=COLORS["NHW"])
            ax.text(center + 0.20, yy, "HW", ha="center", va="top", fontsize=6.5, color=COLORS["HW"])

        ax.grid(axis="y", color="0.92", lw=0.5)
        ax.set_axisbelow(True)

    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels(labels, fontsize=8)

    axes[0].legend(
        handles=[
            Patch(facecolor=COMPONENT_LEGEND_COLORS["day"], alpha=0.85, label="Day (lighter shade)"),
            Patch(facecolor=COMPONENT_LEGEND_COLORS["night"], alpha=0.85, label="Night (darker shade)"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.965,
        top=0.93,
        bottom=0.12,
        hspace=0.34,
    )

    png = safe_output_path(out_dir / f"{basename}.png")
    pdf = safe_output_path(out_dir / f"{basename}.pdf")

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[save] {png}")
    print(f"[save] {pdf}")

    return [png, pdf]

def export_uhi_uci_detailed_summary_txt2(b_data: pd.DataFrame, d_data: pd.DataFrame, supp_data: pd.DataFrame, out_dir: Path, basename: str):
    """
    更新后的详细统计导出函数：
    1. PART 1: 展示物理暴露量 (Urban-Rural Delta)。
    2. PART 2: 展示边际敏感性 (Betas)。
    3. PART 3 (新增): 展示模拟的变化总量 (Simulated Impacts: Day, Night, Total)。
    """
    out_path = safe_output_path(out_dir / f"{basename}_comprehensive_stats.txt")
    lines = []
    lines.append("="*130)
    lines.append("COMPREHENSIVE STATISTICS: PHYSICAL EXPOSURE, SENSITIVITY (BETA), AND MODELLED IMPACT TOTALS")
    lines.append("="*130)
    lines.append(f"Exported on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\nDEFINITIONS & METHODOLOGY:")
    lines.append("- Beta: For Sleep, it's the marginal change per 1°C UHI. For Labour/Energy, it's the observed difference.")
    lines.append("- Simulated Impact: The final estimated change in outcome (Sleep min, CDH, or % Labour Loss).")
    lines.append("                   (Impact = Beta * Delta_Exposure for regression-based outcomes).")
    lines.append("- Labour logic: Day = Total = Tx-based ΔLabour loss; Night = 0 structural zero; positive means urban labour loss > rural labour loss.")
    lines.append("- Sign Logic: Positive (+) values always mean Urban > Rural (Urbanization increases the loss/burden).")
    lines.append("\n")

    # --- Part 1: Observed Physical Exposure (from supp_data) ---
    lines.append("-" * 50)
    lines.append("PART 1: OBSERVED PHYSICAL EXPOSURE (Urban - Rural Δ)")
    lines.append("-" * 50)
    
    if supp_data is not None:
        group_col = 'uhi_uci_group' if 'uhi_uci_group' in supp_data.columns else 'group'
        groups = sorted(supp_data[group_col].unique())
        for grp in groups:
            lines.append(f"\n[Group: {grp}]")
            df_g = supp_data[supp_data[group_col] == grp]
            for var in df_g['variable'].unique():
                lines.append(f"  Variable: {var}")
                lines.append(f"  {'Phase':8s} | {'Period':8s} | {'Mean Δ':>12s} | {'STD Δ':>12s} | {'(n)':>6s}")
                lines.append("  " + "-"*65)
                df_v = df_g[df_g['variable'] == var]
                for ph in ["NHW", "HW"]:
                    for ep in ["day", "night"]:
                        sub = df_v[(df_v['period_phase'] == ph) & (df_v['exposure_period'] == ep)]
                        if not sub.empty:
                            m, s, n = sub['delta_value'].mean(), sub['delta_value'].std(), len(sub)
                            lines.append(f"  {ph:8s} | {ep:8s} | {m:12.4f} | {s:12.4f} | {n:6d}")

    # --- Part 2: TWFE Betas / Direct Observed Diff (from b_data) ---
    lines.append("\n" + "-" * 50)
    lines.append("PART 2: SENSITIVITY OR DIRECT CONTRAST (Beta Coefficients)")
    lines.append("-" * 50)
    if b_data is not None:
        group_col = 'uhi_uci_group' if 'uhi_uci_group' in b_data.columns else 'group'
        for grp in sorted(b_data[group_col].unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = b_data[b_data[group_col] == grp]
            for oc in df_g['outcome'].unique():
                oc_label = outcome_display_label(oc)
                lines.append(f"  Outcome: {oc_label}")
                lines.append(f"  {'Phase':8s} | {'Beta Day':>12s} | {'Beta Night':>12s} | {'Unit'}")
                lines.append("  " + "-"*65)
                df_oc = df_g[df_g['outcome'] == oc]
                for ph in ["NHW", "HW"]:
                    sub = df_oc[df_oc['period_phase'] == ph]
                    if not sub.empty:
                        row = sub.iloc[0]
                        lines.append(f"  {ph:8s} | {row.get('beta_day',0.0):12.4f} | {row.get('beta_night',0.0):12.4f} | {row.get('unit','')}")

    # --- Part 3: Modelled Impact Totals (Simulated Change) ---
    lines.append("\n" + "-" * 110)
    lines.append("PART 3: SIMULATED IMPACT TOTALS (Day, Night, and Diurnal Total Changes)")
    lines.append("-" * 110)
    if d_data is not None:
        group_col = 'uhi_uci_group' if 'uhi_uci_group' in d_data.columns else 'group'
        for grp in sorted(d_data[group_col].unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = d_data[d_data[group_col] == grp]
            for oc in df_g['outcome'].unique():
                oc_label = outcome_display_label(oc)
                unit = df_g[df_g['outcome'] == oc]['unit'].iloc[0] if 'unit' in df_g.columns else ""
                
                lines.append(f"  Outcome: {oc_label} (Unit: {unit})")
                lines.append(f"  {'Phase':8s} | {'Day Impact':>12s} | {'Night Impact':>12s} | {'Total Impact':>12s} | {'95% CI (Total)'}")
                lines.append("  " + "-"*105)
                
                df_oc = df_g[df_g['outcome'] == oc]
                for ph in ["NHW", "HW"]:
                    # 提取该 Phase 下的 day, night, total 三行数据
                    sub_ph = df_oc[df_oc['period_phase'] == ph]
                    if not sub_ph.empty:
                        d_val = sub_ph[sub_ph['component'] == 'day']['estimate'].values
                        n_val = sub_ph[sub_ph['component'] == 'night']['estimate'].values
                        t_row = sub_ph[sub_ph['component'] == 'total']
                        
                        dv = d_val[0] if len(d_val) else 0.0
                        nv = n_val[0] if len(n_val) else 0.0
                        
                        if not t_row.empty:
                            tv = t_row['estimate'].iloc[0]
                            ci = f"[{t_row['ci_low'].iloc[0]:8.3f}, {t_row['ci_high'].iloc[0]:8.3f}]"
                        else:
                            tv, ci = dv + nv, "N/A"
                        
                        lines.append(f"  {ph:8s} | {dv:12.4f} | {nv:12.4f} | {tv:12.4f} | {ci}")
                lines.append("")

    # 写入文件
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[export] Detailed comprehensive statistics saved to: {out_path}")

def export_hw_amplification_stats_txt2(d_data: pd.DataFrame, supp_data: pd.DataFrame, out_dir: Path, basename: str):
    """
    新增导出函数：计算 HW 相对于 NHW 的增长比例 (Heatwave Amplification).
    公式: Growth = (HW_value - NHW_value) / NHW_value * 100%
    """
    out_path = safe_output_path(out_dir / f"{basename}_hw_amplification_growth.txt")
    lines = []
    lines.append("="*120)
    lines.append("HEATWAVE AMPLIFICATION: GROWTH OF EXPOSURE AND IMPACTS (HW vs. NHW)")
    lines.append("="*120)
    lines.append(f"Exported on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\nFORMULA: Growth (%) = (HW - NHW) / |NHW| * 100")
    lines.append("Interpretation: +50% means the value is 1.5x larger during heatwaves compared to neutral periods.")
    lines.append("\n")

    # --- Part 1: Physical Exposure Growth ---
    lines.append("-" * 60)
    lines.append("PART 1: PHYSICAL EXPOSURE AMPLIFICATION (Urban-Rural Δ)")
    lines.append("-" * 60)
    
    if supp_data is not None:
        group_col = 'uhi_uci_group' if 'uhi_uci_group' in supp_data.columns else 'group'
        for grp in sorted(supp_data[group_col].unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = supp_data[supp_data[group_col] == grp]
            
            lines.append(f"  {'Variable':35s} | {'Period':8s} | {'NHW Mean':>10s} | {'HW Mean':>10s} | {'Growth (%)'}")
            lines.append("  " + "-"*85)
            
            for var in df_g['variable'].unique():
                df_v = df_g[df_g['variable'] == var]
                for ep in ["day", "night"]:
                    val_nhw = df_v[(df_v['period_phase'] == 'NHW') & (df_v['exposure_period'] == ep)]['delta_value'].mean()
                    val_hw = df_v[(df_v['period_phase'] == 'HW') & (df_v['exposure_period'] == ep)]['delta_value'].mean()
                    
                    if np.isfinite(val_nhw) and np.isfinite(val_hw):
                        growth = (val_hw - val_nhw) / abs(val_nhw) * 100 if val_nhw != 0 else np.nan
                        lines.append(f"  {var[:35]:35s} | {ep:8s} | {val_nhw:10.3f} | {val_hw:10.3f} | {growth:10.1f}%")
            lines.append("")

    # --- Part 2: Modelled Impact Growth ---
    lines.append("\n" + "-" * 60)
    lines.append("PART 2: MODELLED IMPACT AMPLIFICATION (Simulated Totals)")
    lines.append("-" * 60)
    
    if d_data is not None:
        group_col = 'uhi_uci_group' if 'uhi_uci_group' in d_data.columns else 'group'
        for grp in sorted(d_data[group_col].unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = d_data[d_data[group_col] == grp]
            
            lines.append(f"  {'Outcome':25s} | {'Comp':8s} | {'NHW Impact':>12s} | {'HW Impact':>12s} | {'Growth (%)'}")
            lines.append("  " + "-"*85)
            
            for oc in df_g['outcome'].unique():
                df_oc = df_g[df_g['outcome'] == oc]
                oc_label = outcome_display_label(oc)
                
                # 计算 Day, Night, 和 Total 的增长
                for comp in ["day", "night", "total"]:
                    v_nhw_s = df_oc[(df_oc['period_phase'] == 'NHW') & (df_oc['component'] == comp)]['estimate'].values
                    v_hw_s = df_oc[(df_oc['period_phase'] == 'HW') & (df_oc['component'] == comp)]['estimate'].values
                    
                    if len(v_nhw_s) > 0 and len(v_hw_s) > 0:
                        v_nhw, v_hw = v_nhw_s[0], v_hw_s[0]
                        
                        # 针对 Labour Night = 0 的处理
                        if abs(v_nhw) < 1e-6 and abs(v_hw) < 1e-6:
                            growth_str = "0.0% (N/A)"
                        elif abs(v_nhw) < 1e-6:
                            growth_str = "Inf (New)"
                        else:
                            growth = (v_hw - v_nhw) / abs(v_nhw) * 100
                            growth_str = f"{growth:10.1f}%"
                        
                        lines.append(f"  {oc_label[:25]:25s} | {comp:8s} | {v_nhw:12.4f} | {v_hw:12.4f} | {growth_str}")
                lines.append("  " + "."*85) # 分隔符

    # 写入文件
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[export] HW amplification stats saved to: {out_path}")

def export_uhi_uci_detailed_summary_txt(
    b_data: pd.DataFrame,
    d_data: pd.DataFrame,
    supp_data: pd.DataFrame,
    out_dir: Path,
    basename: str,
):
    out_path = safe_output_path(out_dir / f"{basename}_comprehensive_stats.txt")
    lines = []
    lines.append("=" * 130)
    lines.append("COMPREHENSIVE STATISTICS: PHYSICAL EXPOSURE, DIRECT CONTRASTS, AND IMPACT TOTALS")
    lines.append("=" * 130)
    lines.append(f"Exported on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Labour note:")
    lines.append("  labour_loss uses the Dunne model at station-specific FFT Tx-hour shaded WBGT.")
    lines.append("  Positive labour_loss means urban labour loss > rural labour loss in panel c.")
    lines.append("  Panel-b labour values are signed urban-minus-rural marginal sensitivities: S_urban - S_rural.")
    lines.append("")

    b_data = _normalise_outcome_names(b_data, "main_panel_b_asymmetry_contrast")
    d_data = _normalise_outcome_names(d_data, "main_panel_c_additive_decomposition")

    # ------------------------------------------------------------------
    # Part 1: physical exposure deltas
    # ------------------------------------------------------------------
    lines.append("-" * 60)
    lines.append("PART 1: PHYSICAL EXPOSURE DELTAS")
    lines.append("-" * 60)

    if supp_data is not None and len(supp_data) > 0:
        group_col = "uhi_uci_group" if "uhi_uci_group" in supp_data.columns else "group"

        for grp in sorted(supp_data[group_col].dropna().unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = supp_data[supp_data[group_col] == grp]

            if "variable" not in df_g.columns:
                continue

            for var in df_g["variable"].dropna().unique():
                df_v = df_g[df_g["variable"] == var]
                lines.append(f"  Variable: {variable_display_label(var)}")
                lines.append(f"  {'Phase':8s} | {'Exposure':8s} | {'Mean':>12s} | {'n':>6s}")
                lines.append("  " + "-" * 50)

                for ph in ["NHW", "HW"]:
                    for ep in ["day", "night"]:
                        sub = df_v[
                            (df_v["period_phase"] == ph)
                            & (df_v["exposure_period"] == ep)
                        ]
                        if sub.empty:
                            continue

                        vals = pd.to_numeric(sub["delta_value"], errors="coerce")
                        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

                        if len(vals) == 0:
                            continue

                        lines.append(
                            f"  {ph:8s} | {ep:8s} | {vals.mean():12.4f} | {len(vals):6d}"
                        )
                lines.append("")

    # ------------------------------------------------------------------
    # Part 2: panel-b urban-minus-rural marginal sensitivity contrasts
    # ------------------------------------------------------------------
    lines.append("\n" + "-" * 86)
    lines.append("PART 2: URBAN-MINUS-RURAL MARGINAL SENSITIVITY CONTRASTS")
    lines.append("-" * 86)

    if b_data is not None and len(b_data) > 0:
        group_col = "uhi_uci_group" if "uhi_uci_group" in b_data.columns else "group"
        est_col, lo_col, hi_col = _panel_b_numeric_columns(b_data)

        for grp in sorted(b_data[group_col].dropna().unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = b_data[b_data[group_col] == grp]

            for oc in plot_panel_b_outcome_order(df_g):
                df_oc = df_g[df_g["outcome"] == oc]
                if df_oc.empty:
                    continue

                oc_label = outcome_display_label(oc)
                unit_col = "panel_b_unit" if "panel_b_unit" in df_oc.columns else "unit"
                unit = df_oc[unit_col].iloc[0] if unit_col in df_oc.columns else ""
                metric = (
                    df_oc["panel_b_metric"].iloc[0]
                    if "panel_b_metric" in df_oc.columns
                    else "urban_minus_rural_marginal_sensitivity"
                )

                lines.append(f"  Outcome: {oc_label} (Unit: {unit})")
                lines.append(f"  Metric: {metric}")
                lines.append(
                    f"  {'Phase':8s} | {'U-R sensitivity contrast':>22s} | {'95% CI'}"
                )
                lines.append("  " + "-" * 72)

                for ph in ["NHW", "HW"]:
                    sub = df_oc[df_oc["period_phase"] == ph]
                    if sub.empty:
                        continue

                    row = sub.iloc[0]
                    estimate = pd.to_numeric(pd.Series([row.get(est_col)]), errors="coerce").iloc[0]
                    ci_low = pd.to_numeric(pd.Series([row.get(lo_col)]), errors="coerce").iloc[0]
                    ci_high = pd.to_numeric(pd.Series([row.get(hi_col)]), errors="coerce").iloc[0]

                    ci = (
                        f"[{ci_low:8.3f}, {ci_high:8.3f}]"
                        if np.isfinite(ci_low) and np.isfinite(ci_high)
                        else "N/A"
                    )
                    est_text = f"{estimate:22.4f}" if np.isfinite(estimate) else f"{'N/A':>22s}"
                    lines.append(f"  {ph:8s} | {est_text} | {ci}")
                lines.append("")

    # ------------------------------------------------------------------
    # Part 3: panel c day/night/total impact components
    # ------------------------------------------------------------------
    lines.append("\n" + "-" * 110)
    lines.append("PART 3: DAY, NIGHT, AND TOTAL IMPACT COMPONENTS")
    lines.append("-" * 110)

    if d_data is not None and len(d_data) > 0:
        group_col = "uhi_uci_group" if "uhi_uci_group" in d_data.columns else "group"

        for grp in sorted(d_data[group_col].dropna().unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = d_data[d_data[group_col] == grp]

            for oc in plot_outcome_order(df_g):
                df_oc = df_g[df_g["outcome"] == oc]
                if df_oc.empty:
                    continue

                oc_label = outcome_display_label(oc)
                unit = df_oc["unit"].iloc[0] if "unit" in df_oc.columns else ""

                lines.append(f"  Outcome: {oc_label} (Unit: {unit})")
                lines.append(
                    f"  {'Phase':8s} | {'Day':>12s} | {'Night':>12s} | "
                    f"{'Total':>12s} | {'95% CI (Total)'}"
                )
                lines.append("  " + "-" * 105)

                for ph in ["NHW", "HW"]:
                    sub_ph = df_oc[df_oc["period_phase"] == ph]
                    if sub_ph.empty:
                        continue

                    d_val = sub_ph[sub_ph["component"] == "day"]["estimate"].values
                    n_val = sub_ph[sub_ph["component"] == "night"]["estimate"].values
                    t_row = sub_ph[sub_ph["component"] == "total"]

                    dv = float(d_val[0]) if len(d_val) else 0.0
                    nv = float(n_val[0]) if len(n_val) else 0.0

                    if not t_row.empty:
                        tv = float(t_row["estimate"].iloc[0])
                        lo = t_row["ci_low"].iloc[0]
                        hi = t_row["ci_high"].iloc[0]
                        ci = (
                            f"[{lo:8.3f}, {hi:8.3f}]"
                            if np.isfinite(lo) and np.isfinite(hi)
                            else "N/A"
                        )
                    else:
                        tv, ci = dv + nv, "N/A"

                    lines.append(
                        f"  {ph:8s} | {dv:12.4f} | {nv:12.4f} | "
                        f"{tv:12.4f} | {ci}"
                    )
                lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[export] Detailed comprehensive statistics saved to: {out_path}")

def export_hw_amplification_stats_txt(
    d_data: pd.DataFrame,
    supp_data: pd.DataFrame,
    out_dir: Path,
    basename: str,
):
    """
    Export HW-minus-NHW changes.

    Important update:
    - For signed urban-rural deltas, the primary quantity is signed change:
          HW - NHW
    - Relative growth (%) is only reported as a diagnostic when the NHW baseline
      is not near zero and HW/NHW have the same sign.
    - labour_loss is the intended burden-direction labour outcome.
    """
    out_path = safe_output_path(out_dir / f"{basename}_hw_amplification_growth.txt")
    lines = []

    lines.append("=" * 120)
    lines.append("HEATWAVE RESPONSE: SIGNED HW−NHW CHANGE OF EXPOSURE AND IMPACTS")
    lines.append("=" * 120)
    lines.append(f"Exported on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Primary formula:")
    lines.append("  Signed change = HW value − NHW value")
    lines.append("")
    lines.append("Relative change is diagnostic only:")
    lines.append("  Relative change (%) is reported only when |NHW| is not near zero and HW/NHW have the same sign.")
    lines.append("")
    lines.append("Labour-loss convention:")
    lines.append("  Panel c: positive labour_loss = urban labour loss > rural labour loss.")
    lines.append("  Panel b labour: Dunne Tx-based signed urban-minus-rural marginal sensitivity, S_urban - S_rural.")
    lines.append("")

    d_data = _normalise_outcome_names(d_data, "main_panel_c_additive_decomposition")

    def _relative_change_string(v_nhw, v_hw):
        if not (np.isfinite(v_nhw) and np.isfinite(v_hw)):
            return "N/A"
        if abs(v_nhw) < 1e-7:
            return "N/A (NHW≈0)"
        if np.sign(v_nhw) != np.sign(v_hw) and abs(v_hw) > 1e-7:
            return "not reported (sign change)"
        return f"{(v_hw - v_nhw) / abs(v_nhw) * 100:10.1f}%"

    def _fmt(x):
        return f"{x:12.4f}" if np.isfinite(x) else f"{'N/A':>12s}"

    # ------------------------------------------------------------------
    # Part 1: physical exposure changes from supplementary data
    # ------------------------------------------------------------------
    lines.append("-" * 70)
    lines.append("PART 1: PHYSICAL EXPOSURE RESPONSE (Urban-rural delta; HW−NHW)")
    lines.append("-" * 70)

    if supp_data is not None and len(supp_data) > 0:
        group_col = "uhi_uci_group" if "uhi_uci_group" in supp_data.columns else "group"

        for grp in sorted(supp_data[group_col].dropna().unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = supp_data[supp_data[group_col] == grp]

            if "variable" not in df_g.columns:
                continue

            lines.append(
                f"  {'Variable':35s} | {'Exposure':8s} | "
                f"{'NHW':>12s} | {'HW':>12s} | {'HW-NHW':>12s} | {'Relative'}"
            )
            lines.append("  " + "-" * 105)

            for var in df_g["variable"].dropna().unique():
                df_v = df_g[df_g["variable"] == var]

                for ep in ["day", "night"]:
                    val_nhw = pd.to_numeric(
                        df_v[
                            (df_v["period_phase"] == "NHW")
                            & (df_v["exposure_period"] == ep)
                        ]["delta_value"],
                        errors="coerce",
                    ).mean()

                    val_hw = pd.to_numeric(
                        df_v[
                            (df_v["period_phase"] == "HW")
                            & (df_v["exposure_period"] == ep)
                        ]["delta_value"],
                        errors="coerce",
                    ).mean()

                    if not (np.isfinite(val_nhw) and np.isfinite(val_hw)):
                        continue

                    change = val_hw - val_nhw
                    rel = _relative_change_string(val_nhw, val_hw)

                    lines.append(
                        f"  {variable_display_label(var)[:35]:35s} | {ep:8s} | "
                        f"{_fmt(val_nhw)} | {_fmt(val_hw)} | {_fmt(change)} | {rel}"
                    )
            lines.append("")

    # ------------------------------------------------------------------
    # Part 2: modelled/direct impact changes
    # ------------------------------------------------------------------
    lines.append("\n" + "-" * 80)
    lines.append("PART 2: IMPACT RESPONSE (Day/night/total components; HW−NHW)")
    lines.append("-" * 80)

    if d_data is not None and len(d_data) > 0:
        group_col = "uhi_uci_group" if "uhi_uci_group" in d_data.columns else "group"

        for grp in sorted(d_data[group_col].dropna().unique()):
            lines.append(f"\n[Group: {grp}]")
            df_g = d_data[d_data[group_col] == grp]

            for oc in plot_outcome_order(df_g):
                df_oc = df_g[df_g["outcome"] == oc]
                if df_oc.empty:
                    continue

                oc_label = outcome_display_label(oc)
                unit = df_oc["unit"].iloc[0] if "unit" in df_oc.columns else ""

                lines.append(f"  Outcome: {oc_label} (Unit: {unit})")
                lines.append(
                    f"  {'Comp':8s} | {'NHW':>12s} | {'HW':>12s} | "
                    f"{'HW-NHW':>12s} | {'Relative'}"
                )
                lines.append("  " + "-" * 85)

                def get_val(phase, comp):
                    rows = df_oc[
                        (df_oc["period_phase"] == phase)
                        & (df_oc["component"] == comp)
                    ]

                    if not rows.empty:
                        return float(pd.to_numeric(rows["estimate"], errors="coerce").iloc[0])

                    if comp == "total":
                        d = get_val(phase, "day")
                        n = get_val(phase, "night")
                        if np.isfinite(d) and np.isfinite(n):
                            return d + n

                    return np.nan

                for comp in ["day", "night", "total"]:
                    v_nhw = get_val("NHW", comp)
                    v_hw = get_val("HW", comp)

                    if not (np.isfinite(v_nhw) and np.isfinite(v_hw)):
                        continue

                    change = v_hw - v_nhw
                    rel = _relative_change_string(v_nhw, v_hw)

                    lines.append(
                        f"  {comp:8s} | {_fmt(v_nhw)} | {_fmt(v_hw)} | "
                        f"{_fmt(change)} | {rel}"
                    )

                # Explicit note for signed labour-capacity values.
                if oc == "labour_capacity":
                    lines.append(
                        "  Note: signed HW−NHW change is the main quantity for Dunne labour loss; "
                        "relative change may be misleading when NHW/HW values are negative."
                    )

                lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[export] HW response stats saved to: {out_path}")

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Plot NCC-style UHI/UCI diurnal asymmetry figures from prepared CSV files. "
            "The script performs no statistical modelling and writes a provenance manifest."
        )
    )

    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))

    ap.add_argument(
        "--basename",
        type=str,
        default="main_fig_ncc_diurnal_hw_asymmetry",
        help="Base name for original UHI/UCI separate main figures.",
    )
    ap.add_argument(
        "--combined-basename",
        type=str,
        default="main_fig_ncc_diurnal_hw_asymmetry_combined",
        help="Base name for the combined main figure.",
    )
    ap.add_argument(
        "--percent-basename",
        type=str,
        default="main_fig_ncc_diurnal_hw_asymmetry_combined_percent_decomposition",
        help="Base name for the percentage contribution version of panel c.",
    )
    ap.add_argument(
        "--supp-basename",
        type=str,
        default="supp_fig_daynight_delta_boxplot_uhi_uci",
        help="Base name for supplementary boxplot figure.",
    )
    ap.add_argument(
        "--manifest-basename",
        type=str,
        default="provenance_manifest_ncc_diurnal_uhi_uci",
        help="Base name for provenance manifest JSON.",
    )

    ap.add_argument("--skip-main", action="store_true")
    ap.add_argument("--skip-combined-main", action="store_true")
    ap.add_argument("--skip-percent", action="store_true")
    ap.add_argument("--skip-supp", action="store_true")
    ap.add_argument("--skip-provenance", action="store_true")

    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] data_dir = {data_dir}")
    print(f"[info] out_dir  = {out_dir}")
    print(f"[info] cartopy  = {HAS_CARTOPY}")
    print(f"[info] geopandas_world = {HAS_GEOPANDAS_WORLD}")

    path_a = data_dir / FILES["a"]
    path_b = data_dir / FILES["b"]
    path_c = data_dir / FILES["c"]
    path_supp = data_dir / FILES["supp"]

    a = load_csv(path_a)
    b = load_csv(path_b)
    c = load_csv(path_c)
    supp = load_csv(path_supp)

    b = harmonize_model_outcomes(b, "main_panel_b_asymmetry_contrast")
    c = harmonize_model_outcomes(c, "main_panel_c_additive_decomposition")

    csv_audits = [
        dataframe_audit("a", path_a, a, REQUIRED_COLUMNS["a"]),
        dataframe_audit("b", path_b, b, REQUIRED_COLUMNS["b"]),
        dataframe_audit("c", path_c, c, REQUIRED_COLUMNS["c"]),
        dataframe_audit("supp", path_supp, supp, REQUIRED_COLUMNS["supp"]),
    ]

    for audit in csv_audits:
        if audit["missing_required_columns"]:
            print(
                f"[warn] {audit['key']} missing required columns: "
                f"{audit['missing_required_columns']}",
                file=sys.stderr,
            )
        else:
            print(
                f"[audit] {audit['key']}: rows={audit['n_rows']}, "
                f"cols={audit['n_columns']}, sha256={audit['sha256']}"
            )

    meta_paths = {
        "a_meta": data_dir / FILES["a_meta"],
        "meta": data_dir / FILES["meta"],
        "supp_meta": data_dir / FILES["supp_meta"],
    }

    loaded_meta = {key: load_meta(path) for key, path in meta_paths.items()}
    json_audits = [json_file_audit(key, meta_paths[key], loaded_meta[key]) for key in meta_paths]

    output_paths: List[Path] = []

    if not args.skip_main:
        for group in UHI_UCI_GROUPS:
            output_paths.extend(render_main(group, a, b, c, out_dir, args.basename))

    if not args.skip_combined_main:
        output_paths.extend(render_combined_main_figure(a, b, c, out_dir, args.combined_basename))

    if not args.skip_percent:
        output_paths.extend(render_combined_decomp_percent_figure(c, out_dir, args.percent_basename))

    if not args.skip_supp:
        output_paths.extend(render_supp(supp, out_dir, args.supp_basename))

    if not args.skip_provenance:
        manifest_path = write_provenance_manifest(
            out_dir=out_dir,
            manifest_basename=args.manifest_basename,
            data_dir=data_dir,
            args=args,
            csv_audits=csv_audits,
            json_audits=json_audits,
            output_paths=output_paths,
        )
        output_paths.append(manifest_path)

    # 导出详细统计（含 Beta 和总量）
    export_uhi_uci_detailed_summary_txt(b, c, supp, out_dir, args.combined_basename)

    # 导出热浪增长比例（同比增长统计）
    export_hw_amplification_stats_txt(c, supp, out_dir, args.combined_basename)

    print("[done] generated outputs:")
    for p in output_paths:
        print(f"  - {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


############################################################
