#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import argparse
from urban_heat_figures.fig1 import generate_figure1
from urban_heat_figures.fig23 import generate_figures_2_and_3
from urban_heat_figures.fig4_prepare import prepare_figure4
from urban_heat_figures.fig4_plot import plot_figure4


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce all four target figures.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260529)
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    metrics = root / "analysis/main_multiyear/robustness_percentile/all_pair_period_metrics.csv"

    print("[1/4] Figure 1")
    generate_figure1(root, out / "fig1")
    print("[2/4] Figures 2 and 3")
    generate_figures_2_and_3(metrics, out / "fig23")
    print("[3/4] Figure 4 preparation")
    prepared = out / "fig4/prepared"
    prepare_figure4(root, prepared, args.n_boot, args.bootstrap_seed)
    print("[4/4] Figure 4 plotting")
    plot_figure4(prepared, out / "fig4")
    print("[done] All target figures generated successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
