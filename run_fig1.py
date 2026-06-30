#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import argparse
from urban_heat_figures.fig1 import generate_figure1


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce Figure 1 from processed data.")
    ap.add_argument("--data-root", required=True, help="Root matching the documented unified processed-data layout.")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/fig1"))
    args = ap.parse_args()
    outputs = generate_figure1(args.data_root, args.output_dir)
    print("[done] Figure 1")
    for path in outputs:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
