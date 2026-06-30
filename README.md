# Global urban–rural heatwave analysis and figure workflows

This repository preserves the full production analysis scripts and separates analysis from plotting.
The scientific calculations inside the seven analysis scripts have not been reduced to the needs of the
currently released figures, so their outputs can support additional figures in later releases.

## Repository structure

```text
analysis/                 Full analysis scripts, kept in the original seven-step division
plot/plot1/               Full Figure 1 plotting script
plot/plot23/              Full Figure 2/3 plotting script
plot/plot4/               Figure 4 data preparation and plotting scripts
```

## Important design choice

Analysis and plotting are deliberately separate:

1. `workflow/run_analysis.py` runs all seven complete analysis scripts in their original order.
2. `workflow/run_plot1.py` runs the complete Plot 1 script.
3. `workflow/run_plot23.py` runs the complete Plot 2/3 script.
4. `workflow/run_plot4.py` first prepares Figure 4 data and then plots Figure 4.

The plotting scripts retain their additional plotting functions. They are not reduced to only four PNG files.

## Installation

```bash
conda env create -f environment.yml
conda activate urban-heatwave-workflow
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

Geospatial dependencies are generally easier to install with Conda.

## Configure local data paths

```bash
cp config.example.env .env
```

Edit `.env`. Private absolute paths remain local and must not be committed.

## Run the complete analysis

```bash
python -u workflow/run_analysis.py
```

Run selected analysis steps only:

```bash
python -u workflow/run_analysis.py --only 01,05,06
```

The seven steps are:

1. Main multiyear temperature/dewpoint analysis and canonical annual UHI/UCI grouping
2. Dunne labour-loss analysis
3. Station-level heatwave detection
4. Heatwave relative-risk analysis
5. CDH/HDH and building-energy analysis
6. Hot-night, sleep-loss and economic analysis
7. Sensitivity analysis

## Run plots separately

```bash
python -u workflow/run_plot1.py
python -u workflow/run_plot23.py
python -u workflow/run_plot4.py
```

Plot Figure 4 from previously prepared CSV files:

```bash
python -u workflow/run_plot4.py --plot-only
```

Run everything:

```bash
python -u workflow/run_all.py
```

Equivalent shell runners are available in `workflow/*.sh`.

## Main archived figure outputs

- `plot_data/fig1_n/composite_delta_uhi_final_fianl_ver.png`
- `plot_data/fig23_fixed_new/plots/Figure_Combined_Mechanism_2x2_Nature.png`
- `plot_data/fig23_fixed_new/plots/Figure_Dynamics_Consistent_Final.png`
- `plot_data/fig4_new/main_fig_ncc_diurnal_hw_asymmetry_combined.png`

The historical `fianl` spelling is retained for archive compatibility.

## Scientific definitions retained

- Canonical UHI/UCI classification uses annual urban–rural temperature contrast.
- Heatwaves use the aligned Tmax percentile definition and the original ERA5-to-ISD correction pathway.
- Northern and Southern Hemisphere warm seasons retain the original JJA/DJF boundary handling.
- Diurnal temperature metrics retain the 24 h and 12 h FFT/harmonic reconstruction.
- Station-pair bootstrap uses 1,000 replicates where specified by the production scripts.
- Figure 4 panel b uses `S_urban - S_rural` for sleep loss, building CDH, and labour loss.
- Figure 4 panel c retains outcome-specific day/night decomposition.

## Data and licensing

Large or restricted datasets are not included. Users must obtain data under the applicable source licences.
Keep the existing `LICENSE` file in your GitHub repository; this package intentionally does not replace it.
Citation metadata should be added after the manuscript record is finalised.
