# Urban heatwave analysis and plotting release

This release contains the final analysis and selected plotting workflows used by
the manuscript. All paths are supplied by command-line arguments or through the
`UNIFIED_ROOT` environment variable. The public workflow does not contain
personal absolute paths.

## Directory layout

```text
urban_heatwave_release/
├── analysis/
│   ├── 0_prepare_inputs.py
│   ├── 1_analysis_multiyear.py
│   ├── 2_compute_labour_capacity.py
│   ├── 3_compute_cdh.py
│   ├── 4_compute_sleep_loss.py
│   ├── 5_sensitivity_analysis.py
│   ├── 6_validate_analysis_outputs.py
│   ├── 7_run_analysis.py
│   ├── analysis_common.py
│   ├── environment.yml
│   ├── requirements.txt
│   └── README.md
└── plot/
    ├── 1_plot_result_fig1_selected_outputs.py
    ├── 2_plot_results_fig23_fixed2_selected_outputs.py
    ├── 3_prepare_fig4_direct_impacts.py
    ├── 4_plot_fig4_v2_selected_outputs.py
    ├── 6_integrated_hw_nhw_only.py
    ├── 8_plot_sensitivity_selected_outputs.py
    └── README.md
```

## Fixed mathematical conventions

### Temperature and heatwave analysis

The main analysis uses local solar time, hemisphere-aware warm seasons, an
ERA5-to-ISD empirical quantile mapping, a seasonally varying percentile
threshold in a ±7-day day-of-year window, at least three consecutive exceedance
days, and a four-harmonic reconstruction of the diurnal cycle. Missing hourly
bins used by the FFT workflow are linearly interpolated in both the main and
sensitivity analyses.

### Labour capacity: calculated from Tx

Labour capacity is calculated from the period-specific FFT maximum temperature
`Tx` produced by script 1:

1. `Tx` is read from `urban_Tmax_fft` and `rural_Tmax_fft`.
2. The station-specific Tx hour is located on the reconstructed 24-hour curve.
3. Dewpoint at the same Tx hour is used to estimate relative humidity and
   shaded wet-bulb globe temperature.
4. The Dunne et al. function is applied separately to the urban and rural
   stations:

```text
LC = 100 - 25 * max(0, WBGTshade - 25)^(2/3)
```

The reported capacity difference is `LC_urban(Tx) - LC_rural(Tx)`. The labour
night component is zero. Positive values indicate greater urban labour
capacity; the corresponding labour-loss burden is the negative of this value.

### Sleep loss: calculated from Tn

Sleep loss is calculated from the daily nighttime minimum temperature `Tn`, not
from a nighttime mean or an hourly accumulation. Solar-night observations after
midnight are assigned to the preceding evening, and at least six valid nighttime
observations are required. The Minor et al. Table S37 spline is applied
separately to the urban and rural daily Tn values:

```text
sleep_duration_change(Tn)
  = -0.107 * max(Tn + 20, 0)
    -0.618 * max(Tn - 10, 0)
```

`d_sleep_loss_min` is the urban-minus-rural sleep-duration change. The plotting
workflow uses `d_sleep_loss_burden_min = -d_sleep_loss_min`, so positive values
indicate greater urban sleep-loss burden.

### Cooling degree-hours

CDH is calculated for each station and hour before any aggregation:

```text
CDH_h = max(T_h - 26, 0)
Delta_CDH_h = CDH_h,urban - CDH_h,rural
```

The integrated plots read the official pair-level hourly CDH output. They do
not apply the nonlinear threshold to an already averaged temperature curve.

### Cohorts and uncertainty

HW/NHW comparisons use matched station pairs. Multi-outcome figures use one
complete-case cohort with all required variables available in both periods.
Missing HW/NHW responses are never replaced by zero. Percentile bootstrap
intervals use the station pair as the resampling unit, 1,000 replicates, and
seed `20260529`.

## Validation status

All Python entry points pass syntax compilation. The analysis workflow and all
selected plotting workflows were run on synthetic data. The complete Fig23
control flow was tested with `FIG23_DPI=120` to limit sandbox rendering cost;
the public default remains 600 dpi. The automated validator confirmed:

- labour Tx values exactly match script 1 FFT Tx;
- labour capacity and labour-loss signs are internally consistent;
- sleep loss uses daily nighttime minimum Tn and at least six nighttime values;
- CDH is computed hourly before urban-rural differencing;
- HW, NHW, and warm-season dates inherit script 1 flags;
- the selected plotting directories contain only the requested outputs.

This software validation does not replace a complete rerun on the final
manuscript dataset. Before public archiving, compare the real-data pair cohort,
summary statistics, confidence intervals, and figure inputs with the frozen
submission results.
