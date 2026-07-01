# Global urban–rural heatwave analysis: Figure 1 workflow

This repository currently provides the analysis and plotting workflow used to reproduce **Figure 1 and a selected set of supplementary figures** from the global urban–rural heatwave study.

The repository is under active development. Code for the remaining main figures, supplementary figures, and additional analysis modules will be added in later releases.

## Current release scope

The current version includes:

- the multiyear urban–rural temperature and dew-point analysis used by Figure 1;
- canonical annual UHI/UCI classification;
- heatwave and non-heatwave temperature metrics;
- diurnal temperature reconstruction;
- the Figure 1 plotting workflow;
- selected supplementary plots directly associated with Figure 1.

The current version does **not yet include the complete production workflow** for all figures in the manuscript.

In particular, the following components are planned for future releases:

- remaining supplementary figures;
- Figure 2 and Figure 3 analysis and plotting workflows;
- Figure 4 socioeconomic-burden workflow;
- additional sensitivity, robustness, and attribution analyses;
- complete end-to-end orchestration of all production scripts.

## Repository structure

```text
config.yaml       Local paths and runtime settings
analysis.py       Multiyear analysis used to prepare Figure 1 inputs
plot.py           Plotting script for Figure 1 and selected supplements
environment.yml   Conda environment
README.md         Usage and release documentation
LICENSE           Software licence
