#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SI mean-amplitude states and illustrative diurnal cycles.

Standalone extraction of the active plot_figure_combined_mechanism_2x2_fixed
function. Upper panels retain original b/c data selection, percentile display
limits, colour normalization and least-squares zero-boundary algorithm.
Lower A/B/C are prescribed harmonic illustrations, NOT observed city curves.
Canonical/matched-cohort helper functions are retained.
No old script or output is overwritten. All outputs use a new UTC run directory.

Run: python 03_plotting/05_plot_supplementary_mean_amplitude_states.py
QA only: python 03_plotting/05_plot_supplementary_mean_amplitude_states.py --self-test --output-root ./qa_si
Dependencies: numpy, pandas, matplotlib, Pillow.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image

VERSION = "plot_supplementary_mean_amplitude_states"
INPUT_DIR = (
    "./"
    "outputs/analysis/main_multiyear/"
    "robustness_percentile"
)
DEFAULT_SOURCE = Path(INPUT_DIR) / "all_pair_period_metrics.csv"
DEFAULT_OUTPUT_ROOT = Path(
    "./"
    "outputs/figures/"
    "si_mean_amplitude_states_cycles"
)
WIDTH_MM, HEIGHT_MM, DPI = 180.0, 150.0, 600
STEM = "figure_si_mean_amplitude_states_cycles"
SCENARIOS = [
    ("A", 3.0, 1.0, "#d7191c", "Amplified diurnal UHI",
     r"$-\Delta Amp<0<\Delta T_{\mathrm{mean}}$",
     r"$\Delta T_x>\Delta T_n$"),
    ("B", 3.0, -2.0, "#f16913", "Damped daytime UHI",
     r"$0<-\Delta Amp<\Delta T_{\mathrm{mean}}$",
     r"$\Delta T_n>\Delta T_x>0$"),
    ("C", 0.9, -2.2, "#2257d5", "Daytime UCI /\nnight-time UHI",
     r"$0<\Delta T_{\mathrm{mean}}<-\Delta Amp$",
     r"$\Delta T_x<0,\ \Delta T_n>0$"),
]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# The following four helpers are copied verbatim from the supplied script.
def load_canonical_annual_groups_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return strict pair_id -> annual UHI/UCI classification."""
    required = {"pair_id", "period", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Canonical annual group columns missing: {sorted(missing)}")

    g = df.copy()
    if "hw_method" in g.columns:
        g = g[
            g["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()
    g = g[g["period"].astype(str).str.lower().str.strip().eq("annual")].copy()
    g["pair_id"] = g["pair_id"].astype(str)
    g["annual_group"] = g["group"].astype(str).str.upper().str.strip()
    g = g[g["annual_group"].isin(["UHI", "UCI"])].copy()
    if g.empty:
        raise ValueError("No annual percentile UHI/UCI rows are available.")

    conflicts = g.groupby("pair_id", observed=True)["annual_group"].nunique()
    conflict_ids = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflict_ids:
        raise ValueError(
            "Conflicting annual UHI/UCI groups for "
            f"{len(conflict_ids)} pair(s); examples={conflict_ids[:20]}"
        )
    return g[["pair_id", "annual_group"]].drop_duplicates("pair_id")


def apply_canonical_annual_group(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite period-specific group labels with strict annual labels."""
    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)
    lookup = load_canonical_annual_groups_from_df(out)
    if "group" in out.columns:
        out["group_period_original"] = out["group"]
        out = out.drop(columns=["group"])
    out = out.merge(lookup, on="pair_id", how="inner")
    out = out.rename(columns={"annual_group": "group"})
    return out

# Matched HW–NHW cohort helpers  (added: matched-cohort safe plotting)
# ─────────────────────────────────────────────────────────────
def infer_pair_id_col(df):
    """Infer the pair/station identifier column used across plotting functions."""
    for c in ["pair_id", "station_id", "city_id", "pair_key", "station_pair_id"]:
        if c in df.columns:
            return c
    return None



def build_matched_hw_nhw_cohort(all_df, output_dir=None,
                                required_metrics=("dTmean", "dAmp1", "dTx", "dTn")):
    """Build the strict percentile + annual-group + matched HW/NHW cohort."""
    if all_df is None or len(all_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "period" not in all_df.columns:
        raise ValueError("Column 'period' is required to build matched HW/NHW cohort.")

    df = all_df.copy()
    if "hw_method" in df.columns:
        df = df[
            df["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()

    id_col = infer_pair_id_col(df)
    if id_col is None:
        raise ValueError("Cannot find a pair identifier column.")
    if id_col != "pair_id":
        df = df.rename(columns={id_col: "pair_id"})
        id_col = "pair_id"

    # Re-apply strict annual group labels even if the caller already did so.
    annual_lookup = load_canonical_annual_groups_from_df(df)
    if "group" in df.columns:
        df["group_period_original"] = df["group"]
        df = df.drop(columns=["group"])
    df["pair_id"] = df["pair_id"].astype(str)
    df = df.merge(annual_lookup, on="pair_id", how="inner")
    df = df.rename(columns={"annual_group": "group"})

    missing = [c for c in ["pair_id", "period", *required_metrics] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required matched-cohort columns: {missing}")

    for c in required_metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nhw = df.loc[
        df["period"].astype(str).str.lower().eq("non_heatwave"),
        ["pair_id", *required_metrics],
    ].copy()
    hw = df.loc[
        df["period"].astype(str).str.lower().eq("heatwave"),
        ["pair_id", *required_metrics],
    ].copy()
    nhw = nhw.rename(columns={m: f"{m}_nhw" for m in required_metrics})
    hw = hw.rename(columns={m: f"{m}_hw" for m in required_metrics})

    paired = nhw.merge(hw, on="pair_id", how="inner")
    complete = [f"{m}_nhw" for m in required_metrics] + [f"{m}_hw" for m in required_metrics]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=complete)
    paired = paired.merge(annual_lookup, on="pair_id", how="inner")
    paired = paired.rename(columns={"annual_group": "group"})

    if paired.empty:
        print("  [Matched cohort] No complete annual-group HW/NHW pairs.")
        return df.iloc[0:0].copy(), paired, pd.DataFrame()

    for m in required_metrics:
        response_name = {
            "dTmean": "Rmean", "dAmp1": "Ramp", "dTx": "Rx", "dTn": "Rn",
        }.get(m, f"R_{m}")
        paired[response_name] = paired[f"{m}_hw"] - paired[f"{m}_nhw"]

    keep_ids = set(paired["pair_id"].astype(str))
    matched_all = df[df["pair_id"].astype(str).isin(keep_ids)].copy()

    audit_rows = []
    for period in ["annual", "warm_season", "non_heatwave", "heatwave"]:
        sub = matched_all[matched_all["period"].astype(str).str.lower().eq(period)]
        audit_rows.append({
            "cohort": "annual_group_percentile_matched_hw_nhw_required_metrics",
            "period": period,
            "n_rows": int(len(sub)),
            "n_pairs": int(sub["pair_id"].nunique()),
            "n_UHI": int((sub["group"] == "UHI").sum()),
            "n_UCI": int((sub["group"] == "UCI").sum()),
        })
    cohort_audit = pd.DataFrame(audit_rows)

    if output_dir is not None:
        out_data_dir = os.path.join(output_dir, "integrated_fig_data")
        ensure_dir(out_data_dir)
        paired.to_csv(os.path.join(out_data_dir, "matched_hw_nhw_response_cohort.csv"), index=False)
        cohort_audit.to_csv(os.path.join(out_data_dir, "matched_hw_nhw_cohort_audit.csv"), index=False)
        matched_all[["pair_id", "period", "group"]].drop_duplicates().to_csv(
            os.path.join(out_data_dir, "matched_hw_nhw_long_ids_by_period.csv"), index=False
        )
        annual_lookup.to_csv(
            os.path.join(out_data_dir, "canonical_annual_uhi_uci_groups.csv"), index=False
        )

    print(
        f"  [Matched cohort] matched valid pairs={len(paired)}; "
        f"matched long rows={len(matched_all)}"
    )
    return matched_all, paired, cohort_audit




def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 6.3,
        "axes.labelsize": 7.2, "axes.titlesize": 7.8,
        "axes.titleweight": "bold", "axes.linewidth": 0.75,
        "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
        "xtick.major.width": 0.75, "ytick.major.width": 0.75,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "legend.fontsize": 6.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_states(raw):
    # Exact main() preparation order from the uploaded script.
    work = raw.copy()
    if "hw_method" in work.columns:
        work = work[
            work["hw_method"].astype(str).str.lower().str.strip().eq("percentile")
        ].copy()
    canonical = apply_canonical_annual_group(work)
    matched, paired, cohort_audit = build_matched_hw_nhw_cohort(canonical, None)
    if matched.empty:
        raise ValueError("The strict matched HW/NHW cohort is empty.")
    nhw = matched[matched["period"] == "non_heatwave"].dropna(
        subset=["dTmean", "dAmp1", "dTx"]).copy()
    hw = matched[matched["period"] == "heatwave"].dropna(
        subset=["dTmean", "dAmp1", "dTx"]).copy()
    if nhw.empty or hw.empty:
        raise ValueError("NHW or HW data is empty after dropping NaNs.")
    return nhw, hw, paired, cohort_audit


def state_geometry(nhw, hw):
    # Original display limits and common 98th-percentile colour saturation.
    dtx_all = pd.concat([nhw["dTx"], hw["dTx"]], ignore_index=True)
    x_all = pd.concat([nhw["dTmean"], hw["dTmean"]], ignore_index=True)
    y_all = pd.concat([nhw["dAmp1"], hw["dAmp1"]], ignore_index=True)
    vabs = max(np.nanpercentile(np.abs(dtx_all), 98), 0.1)
    xlim = (np.nanpercentile(x_all, 1) - 0.5,
            np.nanpercentile(x_all, 99) + 0.5)
    ylim = (np.nanpercentile(y_all, 1) - 0.5,
            np.nanpercentile(y_all, 99) + 0.5)
    return xlim, ylim, vabs


def boundary_coefficients(frame):
    if len(frame) <= 2:
        return None
    X = np.column_stack([frame["dTmean"].values, frame["dAmp1"].values,
                         np.ones(len(frame))])
    coef, *_ = np.linalg.lstsq(X, frame["dTx"].values, rcond=None)
    return coef


def illustration_curves():
    # Exact reference values, harmonics and scenario parameters of original a.
    m_ref, a1_ref, a2_ref, p1_ref, p2_ref = 12.0, 7.0, 1.0, 3.8, 0.5
    w = 2 * np.pi / 24
    t = np.linspace(0, 24, 1000)
    def recon(t, m, a, p, a2, p2):
        return m + a * np.cos(w * t - p) + a2 * np.cos(2 * w * t - p2)
    rural = recon(t, m_ref, a1_ref, p1_ref, a2_ref, p2_ref)
    urban = [recon(t, m_ref + s[1], a1_ref + s[2], p1_ref, a2_ref, p2_ref)
             for s in SCENARIOS]
    return t, rural, urban


def heading(ax, letter, title):
    ax.set_title(title, loc="center", fontsize=7.8, fontweight="bold",
                 pad=8, linespacing=1.02)
    label = ax.set_title(letter, loc="left", fontsize=9.5,
                         fontweight="bold", pad=8)
    label.set_x(-0.04)
    label.set_ha("right")
    label.set_clip_on(False)


def render(nhw, hw, output, qa=False):
    apply_style()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
                     facecolor="white")
    # 64.5 x 64.5 mm square axes; data-unit aspect is intentionally not forced.
    side_mm = 64.5
    ax_a = fig.add_axes([0.090, 0.480, side_mm / WIDTH_MM, side_mm / HEIGHT_MM])
    ax_b = fig.add_axes([0.605, 0.480, side_mm / WIDTH_MM, side_mm / HEIGHT_MM])
    xlim, ylim, vabs = state_geometry(nhw, hw)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    fit_audit = {}
    for frame, ax, letter, period, marker in [
        (nhw, ax_a, "a", "NHW", "o"), (hw, ax_b, "b", "HW", "^")
    ]:
        sc = ax.scatter(frame["dTmean"], frame["dAmp1"], c=frame["dTx"],
                        cmap="RdBu_r", norm=norm, marker=marker, s=11,
                        edgecolors="black", linewidths=0.30, alpha=0.9,
                        zorder=3)
        coef = boundary_coefficients(frame)
        drawn = coef is not None and np.isfinite(coef[1]) and abs(coef[1]) > 1e-8
        if drawn:
            xx = np.linspace(xlim[0], xlim[1], 200)
            yy = (-coef[0] / coef[1]) * xx + (-coef[2] / coef[1])
            ax.plot(xx, yy, color="#444444", linestyle="--", linewidth=0.8,
                    label=r"$\widehat{\Delta T_x}=0$", zorder=4)
            ax.legend(frameon=False, loc="upper right", handlelength=1.8)
        ax.axhline(0, color="#d0d0d0", linewidth=0.55, zorder=1)
        ax.axvline(0, color="#d0d0d0", linewidth=0.55, zorder=1)
        ax.set(xlim=xlim, ylim=ylim,
               xlabel=r"$\Delta T_{\mathrm{mean}}$ (°C)",
               ylabel=r"$\Delta Amp$ (°C)")
        ax.tick_params(top=False, right=False, direction="out")
        heading(ax, letter, "Mean–amplitude states\nunder " + period)
        outside = ((frame["dTmean"] < xlim[0]) | (frame["dTmean"] > xlim[1]) |
                   (frame["dAmp1"] < ylim[0]) | (frame["dAmp1"] > ylim[1]))
        fit_audit[period] = {
            "coefficients_mean_amp_intercept": None if coef is None else coef.tolist(),
            "boundary_drawn": bool(drawn), "n_rows": len(frame),
            "n_pairs": frame["pair_id"].nunique(),
            "points_outside_original_display_limits": int(outside.sum()),
        }

    # Shared scale lies below both axis titles, without borrowing axis width.
    cax = fig.add_axes([0.335, 0.414, 0.390, 0.012])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    ticks = np.arange(math.ceil(-vabs), math.floor(vabs) + 1)
    if len(ticks) >= 2:
        cb.set_ticks(ticks)
    cb.set_label(r"Daytime $\Delta T_x$ (°C)", fontsize=7.2, labelpad=2)
    cb.ax.tick_params(labelsize=6.2, length=2, width=0.6, pad=1)
    cb.outline.set_linewidth(0.6)

    fig.text(0.075, 0.354, "c", fontsize=9.5, fontweight="bold", va="center")
    fig.text(0.527, 0.354, "Illustrative urban–rural diurnal cycles",
             fontsize=7.8, fontweight="bold", ha="center", va="center")
    t, rural, urban = illustration_curves()
    all_y = np.concatenate([rural, *urban])
    cycle_ylim = (float(all_y.min() - 1), float(all_y.max() + 1))
    x_positions = [0.090, 0.4075, 0.725]
    width = 0.239
    cycle_axes = []
    for i, (scenario, city, left) in enumerate(zip(SCENARIOS, urban, x_positions)):
        name, dm, da, color, title, f1, f2 = scenario
        ax = fig.add_axes([left, 0.128, width, 0.178])
        cycle_axes.append(ax)
        ax.fill_between(t, rural, city, where=(city >= rural),
                        color="#e99696", alpha=0.34, interpolate=True,
                        linewidth=0, zorder=1)
        ax.fill_between(t, rural, city, where=(city < rural),
                        color="#8ec1e0", alpha=0.34, interpolate=True,
                        linewidth=0, zorder=1)
        ax.plot(t, rural, color="#666666", linewidth=0.85, ls=(0, (3, 3)), zorder=2)
        ax.plot(t, city, color=color, linewidth=1.1, zorder=3)
        ax.set(xlim=(0, 24), ylim=cycle_ylim, xticks=[0, 12, 24])
        ax.tick_params(top=False, right=False)
        ax.set_yticks([5, 15, 25])
        if i == 0:
            ax.set_ylabel("Illustrative temperature (°C)", fontsize=6.4)
        else:
            ax.tick_params(labelleft=False)
        fig.text(left + width / 2, 0.318, name + "  " + title,
                 ha="center", va="bottom", fontsize=6.4,
                 fontweight="bold", color=color, linespacing=1.05)
        fig.text(left + width / 2, 0.079, f1,
                 ha="center", va="center", fontsize=6.2, color=color)
        fig.text(left + width / 2, 0.056, f2,
                 ha="center", va="center", fontsize=6.2, color=color)

    fig.text(0.527, 0.029, "Local time (h)", ha="center", fontsize=6.4)
    fig.legend(handles=[
        Line2D([], [], color="#333333", linewidth=1.1, label="Urban"),
        Line2D([], [], color="#666666", linewidth=0.85, ls=(0, (3, 3)), label="Rural")
    ], loc="lower center", bbox_to_anchor=(0.527, 0.001), ncol=2,
       frameon=False, borderaxespad=0, handlelength=2)
    if qa:
        fig.text(0.99, 0.99, "SYNTHETIC LAYOUT QA", ha="right", va="top",
                 fontsize=5.5, color="#777777")

    fig.canvas.draw()
    suffix = "_SYNTHETIC_QA" if qa else ""
    target = output / (STEM + suffix)
    fig.savefig(target.with_suffix(".jpg"), format="jpg", dpi=DPI,
                facecolor="white", pil_kwargs={"quality": 95})
    fig.savefig(target.with_suffix(".pdf"), format="pdf", facecolor="white")
    plt.close(fig)
    with Image.open(target.with_suffix(".jpg")) as im:
        expected = (int(WIDTH_MM / 25.4 * DPI), int(HEIGHT_MM / 25.4 * DPI))
        if any(abs(a - b) > 1 for a, b in zip(im.size, expected)):
            raise AssertionError(f"Unexpected canvas dimensions: {im.size}")
        preview = im.convert("RGB")
        preview.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
        preview.save(output / (target.name + "_preview.webp"), "WEBP", quality=90)
    return {"display_xlim": list(xlim), "display_ylim": list(ylim),
            "colour_limit_98th_percentile": float(vabs),
            "fitted_boundaries": fit_audit, "illustration_ylim": cycle_ylim}


CAPTION = """Supplementary Figure. Mean–amplitude states and illustrative diurnal cycles.
a,b, Urban–rural mean and first-harmonic amplitude contrasts under matched
non-heatwave (NHW; a) and heatwave (HW; b) conditions. Colour represents the
period-specific daytime urban–rural temperature contrast, not HW minus NHW.
Dashed lines mark the fitted daytime-zero boundary, obtained separately in each
period by regressing daytime contrast on mean contrast, amplitude contrast and
an intercept. Original percentile-based display limits and colour saturation
are retained; observations outside the displayed range remain in fitting.
c, Prescribed two-harmonic urban and rural temperature cycles illustrate
amplified diurnal UHI (A), damped daytime UHI (B), and daytime UCI with night-time
UHI (C). Mean/amplitude contrasts are (3,1), (3,-2) and (0.9,-2.2) degrees C.
These are schematic curves, not observed city records. Urban curves are solid;
rural curves are dashed. Shading indicates positive/negative urban–rural contrast.
Inequalities summarize the low-order mean–amplitude interpretation; complete
extrema also depend on phase and the semi-diurnal harmonic.
"""


def synthetic_data():
    rng = np.random.default_rng(20260907)
    rows = []
    for i in range(317):
        group = "UHI" if i < 200 else "UCI"
        mean = rng.normal(1.2 if group == "UHI" else 0.4, 1.2)
        amp = rng.normal(0.3 if group == "UHI" else -1.8, 1.3)
        for period in ("annual", "non_heatwave", "heatwave"):
            m, a = mean, amp
            if period == "heatwave":
                m += rng.normal(0.25, 0.15)
                a += rng.normal(-0.12, 0.15)
            rows.append(dict(pair_id=str(i), period=period, group=group,
                             hw_method="percentile", dTmean=m, dAmp1=a,
                             dTx=m + a + rng.normal(0, 0.12), dTn=m - a))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--self-test", action="store_true",
                        help="Synthetic layout QA only; never substitutes for formal data.")
    args = parser.parse_args()
    if not args.self_test and not args.input.is_file():
        raise FileNotFoundError(f"Formal source not found: {args.input}")
    raw = synthetic_data() if args.self_test else pd.read_csv(args.input)
    nhw, hw, paired, audit = prepare_states(raw)
    run = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output = args.output_root / run
    output.mkdir(parents=True, exist_ok=False)
    info = render(nhw, hw, output, qa=args.self_test)
    # Audit exports do not alter or overwrite upstream source data.
    pd.concat([nhw, hw]).to_csv(output / "state_plot_source.csv", index=False)
    audit.to_csv(output / "cohort_audit.csv", index=False)
    (output / "caption.txt").write_text(CAPTION, encoding="utf-8")
    manifest = {
        "script_version": VERSION, "script_sha256": sha256(Path(__file__)),
        "source": "SYNTHETIC_QA" if args.self_test else str(args.input.resolve()),
        "source_sha256": None if args.self_test else sha256(args.input),
        "synthetic_qa": args.self_test, "canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "dpi": DPI, "matched_pairs": int(paired["pair_id"].nunique()),
        "original_plot_function": "plot_figure_combined_mechanism_2x2_fixed",
        "panel_mapping": {"a": "original b", "b": "original c",
                          "c": "original a prescribed A/B/C cycles"},
        "boundary_method": "original per-period np.linalg.lstsq; not fixed y=-x",
        "scenario_parameters": [{"name": s[0], "delta_mean": s[1], "delta_amp": s[2]}
                                for s in SCENARIOS],
        **info,
    }
    (output / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output}")
    print("SYNTHETIC_LAYOUT_QA_ONLY" if args.self_test else "FORMAL_RENDER_COMPLETE")


if __name__ == "__main__":
    main()
