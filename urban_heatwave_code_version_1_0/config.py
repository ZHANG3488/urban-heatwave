#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central path configuration for Urban Heatwave Code Version 1.0.

Edit this file before running the workflow, or override individual paths with
the environment variables documented below. Scientific parameters and model
definitions remain inside the original analysis scripts and are not configured
here.

Version 1.0 centralizes file-system paths only. Later versions may update or
extend the workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

__version__ = "1.0"

PathLike = Union[str, os.PathLike]


def _path_from_env(name: str, default: PathLike) -> str:
    """Return an absolute, user-expanded path from an environment variable."""
    raw = os.environ.get(name)
    path = Path(raw).expanduser() if raw else Path(default).expanduser()
    return str(path.resolve())


# Repository directory containing this config.py file.
REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Main roots
# ---------------------------------------------------------------------------
# Default input layout:
#   <repository>/data/
#
# Default output layout:
#   <repository>/outputs/unified_outputs/
#
# Either edit the defaults below or set:
#   UHI_DATA_ROOT
#   UHI_UNIFIED_ROOT
DATA_ROOT = Path(
    _path_from_env("UHI_DATA_ROOT", REPO_ROOT / "data")
)
UNIFIED_ROOT = _path_from_env(
    "UHI_UNIFIED_ROOT",
    REPO_ROOT / "outputs" / "unified_outputs",
)

# ---------------------------------------------------------------------------
# Required input data
# ---------------------------------------------------------------------------
PAIR_CSV_PATH = _path_from_env(
    "UHI_PAIR_CSV",
    DATA_ROOT / "annual_ml_dataset_2024_new.csv",
)

STATION_META_PATH = _path_from_env(
    "UHI_STATION_META_CSV",
    DATA_ROOT / "station_metadata" / "Extract_Final_Integrated_2023.csv",
)

ISD_BASE_DIR = _path_from_env(
    "UHI_ISD_BASE_DIR",
    DATA_ROOT / "isd_lite",
)

CONTINENT_SHP = _path_from_env(
    "UHI_CONTINENT_SHP",
    DATA_ROOT
    / "station_metadata"
    / "World_Continents_-8107292174417139505"
    / "World_Continents.shp",
)

BV_CSV_PATH = _path_from_env(
    "UHI_BUILDING_VOLUME_CSV",
    DATA_ROOT / "station_metadata" / "stations_building_volume_100m.csv",
)

KG_TIF = _path_from_env(
    "UHI_KOPPEN_GEIGER_TIF",
    DATA_ROOT / "koppen_geiger" / "koppen_geiger_0p00833333.tif",
)

ERA5_STATION_DIR = _path_from_env(
    "UHI_ERA5_STATION_DIR",
    DATA_ROOT / "era5_station_daily",
)

STATION_FEATURES_CSV = _path_from_env(
    "UHI_STATION_FEATURES_CSV",
    DATA_ROOT / "socioeconomic" / "05_station_features.csv",
)

# ---------------------------------------------------------------------------
# Workflow output locations
# ---------------------------------------------------------------------------
# These retain the directory structure expected by the original scripts.
MAIN_ANALYSIS_OUTPUT_DIR = str(
    Path(UNIFIED_ROOT) / "analysis" / "main_multiyear"
)
SENSITIVITY_OUTPUT_DIR = str(
    Path(UNIFIED_ROOT) / "analysis" / "sensitivity"
)
SENSITIVITY_FIGURE_OUTPUT_DIR = str(
    Path(UNIFIED_ROOT) / "plot_data" / "sensitivity_figures"
)
FIG23_OUTPUT_DIR = str(
    Path(UNIFIED_ROOT) / "plot_data" / "fig23_fixed_new"
)
LEGACY_FIGURES_OUTPUT_DIR = str(
    Path(UNIFIED_ROOT) / "figures"
)


def configured_paths() -> dict[str, str]:
    """Return the active Version 1.0 path configuration."""
    return {
        "REPO_ROOT": str(REPO_ROOT),
        "DATA_ROOT": str(DATA_ROOT),
        "UNIFIED_ROOT": UNIFIED_ROOT,
        "PAIR_CSV_PATH": PAIR_CSV_PATH,
        "STATION_META_PATH": STATION_META_PATH,
        "ISD_BASE_DIR": ISD_BASE_DIR,
        "CONTINENT_SHP": CONTINENT_SHP,
        "BV_CSV_PATH": BV_CSV_PATH,
        "KG_TIF": KG_TIF,
        "ERA5_STATION_DIR": ERA5_STATION_DIR,
        "STATION_FEATURES_CSV": STATION_FEATURES_CSV,
        "MAIN_ANALYSIS_OUTPUT_DIR": MAIN_ANALYSIS_OUTPUT_DIR,
        "SENSITIVITY_OUTPUT_DIR": SENSITIVITY_OUTPUT_DIR,
        "SENSITIVITY_FIGURE_OUTPUT_DIR": SENSITIVITY_FIGURE_OUTPUT_DIR,
        "FIG23_OUTPUT_DIR": FIG23_OUTPUT_DIR,
        "LEGACY_FIGURES_OUTPUT_DIR": LEGACY_FIGURES_OUTPUT_DIR,
    }


if __name__ == "__main__":
    print(f"Urban Heatwave Code configuration — Version {__version__}")
    for key, value in configured_paths().items():
        print(f"{key}={value}")
