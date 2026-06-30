#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import argparse
from urban_heat_figures.fig23 import generate_figures_2_and_3


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce Figures 2 and 3.")
    ap.add_argument("--metrics-csv", required=True, help="Path to all_pair_period_metrics.csv")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/fig23"))
    args = ap.parse_args()
    outputs = generate_figures_2_and_3(args.metrics_csv, args.output_dir)
    print("[done] Figures 2 and 3")
    for path in outputs:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
