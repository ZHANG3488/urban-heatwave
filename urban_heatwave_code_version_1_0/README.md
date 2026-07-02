# Urban Heatwave Analysis and Figure Workflow

**Version 1.0**

This repository contains the analysis, impact-model, sensitivity-analysis, data-preparation, and plotting scripts used in the urban–rural heatwave workflow developed by **Zhang et al.**

## Version status

Version 1.0 is the initial public code release. The workflow will continue to be reviewed and updated, and later versions may contain additional documentation, interface improvements, or code revisions.

For questions, suspected problems, or reproducibility issues, please open a GitHub issue or contact the authors, **Zhang et al.**

## Scope of the Version 1.0 packaging update

In this release-preparation step:

- hard-coded local file-system paths were removed from the scripts;
- input and output paths were centralized in `config.py`;
- repository-relative default paths and optional environment-variable overrides were added;
- the scientific algorithms, model formulas, thresholds, statistical procedures, filtering rules, and plotting calculations in the supplied scripts were otherwise left unchanged.

Version 1.0 should therefore be treated as the first shareable workflow release rather than the final permanent code archive.

## Repository contents

### Configuration

- `config.py`  
  Central location for all machine-specific input paths and the unified output root.

### Core analysis

1. `01_main_pair_period_metrics.py`  
   Main multi-year pair-period temperature analysis, heatwave classification, harmonic reconstruction, and canonical UHI/UCI output.

2. `02_labour_capacity_loss.py`  
   Labour-capacity and labour-loss analysis using the upstream pair-period results.

3. `03_cooling_degree_hours.py`  
   Cooling-degree-hour, heating-degree-hour, COP, and building-energy analysis using the upstream heatwave flags.

4. `04_sleep_hne_panel.py`  
   Hot-night, sleep, and associated economic-loss workflow using the upstream heatwave flags.

5. `05_sensitivity_analysis.py`  
   Data-quality, heatwave-definition, and observation-window sensitivity analysis.

### Integrated and climate-zone analyses

- `5_cliamtezone.py`
- `6_lcz_analysis.py`

### Figure preparation and plotting

- `1_plot_result_fig1.py`
- `2_plot_results_fig23.py`
- `3_prepare_fig4.py`
- `4_plot_fig4.py`
- `8_plot_sensitivity_analysis.py`

## Data availability

Large observational, reanalysis, raster, shapefile, and socioeconomic input data are not included in this code package. Users must obtain the required datasets separately and configure their locations in `config.py`.

The default repository-relative input layout is:

```text
urban_heatwave_code_version_1_0/
├── config.py
├── data/
│   ├── annual_ml_dataset_2024_new.csv
│   ├── isd_lite/
│   ├── era5_station_daily/
│   ├── koppen_geiger/
│   │   └── koppen_geiger_0p00833333.tif
│   ├── socioeconomic/
│   │   └── 05_station_features.csv
│   └── station_metadata/
│       ├── Extract_Final_Integrated_2023.csv
│       ├── stations_building_volume_100m.csv
│       └── World_Continents_-8107292174417139505/
│           └── World_Continents.shp
└── outputs/
    └── unified_outputs/
```

Associated shapefile sidecar files such as `.dbf`, `.shx`, `.prj`, and `.cpg` must remain beside the `.shp` file.

The files do not have to follow this directory layout. Any absolute or relative locations can be specified in `config.py`.

## Configuration

Open `config.py` and update the path variables before running the workflow. The most important settings are:

- `UNIFIED_ROOT`
- `PAIR_CSV_PATH`
- `STATION_META_PATH`
- `ISD_BASE_DIR`
- `CONTINENT_SHP`
- `BV_CSV_PATH`
- `KG_TIF`
- `ERA5_STATION_DIR`
- `STATION_FEATURES_CSV`

To print the active configuration:

```bash
python config.py
```

### Environment-variable overrides

Instead of editing `config.py`, paths can be overridden with environment variables:

| Environment variable | Purpose |
|---|---|
| `UHI_DATA_ROOT` | Root directory containing input data |
| `UHI_UNIFIED_ROOT` | Root directory for all generated outputs |
| `UHI_PAIR_CSV` | Urban–rural station-pair CSV |
| `UHI_STATION_META_CSV` | Station metadata CSV |
| `UHI_ISD_BASE_DIR` | ISD-Lite data directory |
| `UHI_CONTINENT_SHP` | Continent shapefile |
| `UHI_BUILDING_VOLUME_CSV` | Building-volume CSV |
| `UHI_KOPPEN_GEIGER_TIF` | Köppen–Geiger raster |
| `UHI_ERA5_STATION_DIR` | ERA5 station-level daily Tmax directory |
| `UHI_STATION_FEATURES_CSV` | Socioeconomic station-feature CSV |

Example:

```bash
export UHI_DATA_ROOT="/path/to/input_data"
export UHI_UNIFIED_ROOT="/path/to/output/unified_outputs"
python config.py
```

Keep `config.py` in the same directory as the scripts.

## Python environment

The scripts use Python 3 and may require the following packages, depending on the script being run:

```text
numpy
pandas
scipy
matplotlib
geopandas
shapely
rasterio
statsmodels
cartopy
seaborn
joblib
geodatasets
```

Some geospatial packages may require system libraries. A Conda environment is generally the easiest way to install `geopandas`, `rasterio`, and `cartopy`.

## Recommended execution order

Run commands from the repository directory.

### 1. Inspect the configuration

```bash
python config.py
```

### 2. Run the main analysis

```bash
python 01_main_pair_period_metrics_common_time_dew2harm_fixed.py
```

### 3. Run the impact analyses

```bash
python 02_labour_capacity_loss_common_time_dew2harm_fixed.py
python 03_cooling_degree_hours_energy_from01_hwflags.py
python 04_sleep_hne_panel_from01_hwflags.py
```

### 4. Run the sensitivity analysis

```bash
python 05_sensitivity_analysis.py
python 8_plot_sensitivity_analysis_v7_nature.py
```

### 5. Run the main and supplementary plotting workflows

```bash
python 1_plot_result_fig1_method_freeze_final.py
python 2_plot_results_fig23_method_freeze_final.py
python 5_climatezone_uhi_uci_hw_nhw_method_freeze_final.py
python 6_uhi_uci_hw_warmseason_integrated_analysis_v4_method_freeze_final.py
```

### 6. Prepare and plot Figure 4

```bash
python 3_prepare_ncc_diurnal_paired_hw_summary_method_freeze_final.py
python 4_plot_fig4_paired_hw_summary_method_freeze_final.py
```

The preparation and plotting scripts also support command-line options. Display them with:

```bash
python 3_prepare_ncc_diurnal_paired_hw_summary_method_freeze_final.py --help
python 4_plot_fig4_paired_hw_summary_method_freeze_final.py --help
python 8_plot_sensitivity_analysis_v7_nature.py --help
```

## Output structure

By default, generated files are written under:

```text
outputs/unified_outputs/
├── analysis/
│   ├── main_multiyear/
│   ├── labour/
│   ├── cdh_energy/
│   ├── hne_econ/
│   └── sensitivity/
├── plot_data/
├── shared/
└── figures/
```

Several downstream scripts expect the upstream output names and directory structure to remain unchanged. Move or rename intermediate files only after updating all corresponding interfaces.

## Reproducibility notes

- Run the upstream analysis before downstream impact and plotting scripts.
- Do not mix outputs from different code versions without recording their provenance.
- Preserve `pair_id`, `period`, `hw_method`, and canonical UHI/UCI fields when moving intermediate data.
- Keep the configured `UNIFIED_ROOT` consistent across the full workflow.
- Version-control the scripts and configuration template, but do not commit large raw datasets, generated outputs, credentials, or private data.

## Authors and contact

Code and workflow: **Zhang et al.**

For questions or issues, please use the GitHub issue tracker or contact the authors through the contact information associated with the manuscript or repository.

## Release note

**Version 1.0** is the initial release. Further updates are planned.
