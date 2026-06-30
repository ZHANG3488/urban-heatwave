# Reproducible Figure Workflows for Global Urban–Rural Heatwave Analysis

## Overview

This repository contains portable workflows for reproducing four main figures from processed urban–rural heatwave analysis tables. Large or restricted source datasets are not included. The scientific calculations were extracted from the supplied production scripts; path handling and workflow entry points were reorganized for public use.

## Repository structure

```text
workflows/                 Command-line entry points
src/urban_heat_figures/    Extracted scientific and plotting code
docs/                      Dependencies, methods, schemas, validation
inputs/                    Instructions only; research data are not committed
outputs/                   Generated files, ignored by Git
tests/                     Lightweight static and schema tests
```

## Figures

| Figure | Output | Workflow | Main input |
|---|---|---|---|
| Figure 1 | `composite_delta_uhi_final_fianl_ver.png/.pdf` | `run_fig1.py` | processed-data root |
| Figure 2 | `Figure_Combined_Mechanism_2x2_Nature.png/.pdf` | `run_fig23.py` | `all_pair_period_metrics.csv` |
| Figure 3 | `Figure_Dynamics_Consistent_Final.png/.pdf` | `run_fig23.py` | `all_pair_period_metrics.csv` |
| Figure 4 | `main_fig_ncc_diurnal_hw_asymmetry_combined.png/.pdf` | `run_fig4.py` | processed-data root or prepared CSVs |

The spelling `fianl` is retained in the Figure 1 output name to match the archived production result.

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate urban-heat-figures
```

### pip

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The Conda and pip environments have not been proven bit-for-bit equivalent.

## Input data

Place or reference the processed files described in `inputs/README.md` and `docs/input_data_dictionary.md`. Do not commit large, restricted, or third-party datasets without confirming redistribution rights. Urban–rural values use urban minus rural; HW responses use HW minus NHW.

## Reproduction commands

```bash
python workflows/run_fig1.py \
  --data-root /path/to/unified_processed_outputs \
  --output-dir outputs/fig1
```

```bash
python workflows/run_fig23.py \
  --metrics-csv /path/to/all_pair_period_metrics.csv \
  --output-dir outputs/fig23
```

```bash
python workflows/run_fig4.py \
  --data-root /path/to/unified_processed_outputs \
  --output-dir outputs/fig4
```

Plot Figure 4 from previously prepared CSVs:

```bash
python workflows/run_fig4.py \
  --plot-only \
  --prepared-data-dir /path/to/prepared_figure4_csvs \
  --output-dir outputs/fig4
```

Run all figures:

```bash
bash workflows/run_all.sh \
  --data-root /path/to/unified_processed_outputs \
  --output-dir outputs
```

## Methods summary

Canonical UHI/UCI classes come from annual percentile-method rows. HW/NHW mechanism figures use complete matched station pairs. Upstream temperature cycles use the established 24 h and 12 h harmonic reconstruction. Figure 4 uses a 1,000-replicate station-pair cluster bootstrap by default. Sleep, CDH, and labour retain the original Minor, 26 °C CDH, and Dunne model definitions. Figure 4 panel b uses signed urban-minus-rural marginal-sensitivity contrasts for all three outcomes; panel c retains the original outcome-specific day/night decomposition.

## Outputs

The workflows write PNG files at 600 dpi where specified by the production code and PDF files for the mechanism, dynamics, and Figure 4 outputs.

## Reproducibility notes

- Cartopy, GeoPandas, Matplotlib, and font versions can produce small layout differences.
- PDF metadata can change binary hashes without changing content.
- Bootstrap defaults: 1,000 replicates; seed 20260529 for Figure 4.
- Compare sample counts and numerical summaries, not only image hashes.

## Testing

```bash
python -m py_compile workflows/*.py src/urban_heat_figures/*.py
python -m pytest -q
```

## Citation

Citation information will be added upon publication.

## License

A software license has not yet been selected. Add an OSI-approved license before public release.

## Contact

Add the corresponding author name and institutional email before public release.
