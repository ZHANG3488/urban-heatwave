#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import argparse
from urban_heat_figures.fig4_prepare import prepare_figure4
from urban_heat_figures.fig4_plot import plot_figure4


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare and reproduce Figure 4.")
    ap.add_argument("--data-root", help="Root matching the documented unified processed-data layout.")
    ap.add_argument("--prepared-data-dir", help="Existing prepared Figure 4 CSV directory.")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/fig4"))
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260529)
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    prepared = Path(args.prepared_data_dir).expanduser().resolve() if args.prepared_data_dir else out / "prepared"

    if args.plot_only:
        if not args.prepared_data_dir:
            ap.error("--plot-only requires --prepared-data-dir")
    else:
        if not args.data_root:
            ap.error("--data-root is required unless --plot-only is used")
        print("[1/2] Preparing Figure 4 data")
        prepare_figure4(args.data_root, prepared, args.n_boot, args.bootstrap_seed)

    print("[2/2] Plotting Figure 4")
    outputs = plot_figure4(prepared, out)
    print("[done] Figure 4")
    for path in outputs:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
