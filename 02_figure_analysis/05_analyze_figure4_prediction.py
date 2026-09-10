#!/usr/bin/env python3
"""analyze figure4 prediction.

Original scientific calculations retained; see README.md for inputs and execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT
DEFAULT_RESPONSE = PROJECT_ROOT / "outputs/analysis/figure4_response_states/tables/block_specific_response_states.csv"
DEFAULT_BASE_SCRIPT = PROJECT_ROOT / "02_figure_analysis/04_analyze_figure4_response_states.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/analysis/figure4_predictive_state"
DEFAULT_REFERENCE_RUN = PROJECT_ROOT / "data/reference_run"
HOURS = [f"R_h{hour:02d}" for hour in range(24)]
VECTOR_MODELS = ("M0", "M1", "M1n", "M2", "M3", "M4")
CURVE_MODELS = ("C0", "C1", "C2", "C3", "C4")
MAIN_VECTOR_MODELS = ("M0", "M1", "M2", "M3", "M4")
MODEL_LABELS = {
    "M0": "Climatology", "M1": "Daytime scalar", "M1n": "Nighttime scalar",
    "M2": "Optimal rank-one", "M3": "Mean-amplitude state", "M4": "Day-night pair",
}
MODEL_COLORS = {
    "M0": "#9e9e9e", "M1": "#3f3f3f", "M1n": "#777777",
    "M2": "#E69F00", "M3": "#7B3294", "M4": "#1F78B4",
}
CURVE_MODEL_LABELS = {
    "C0": "Climatology", "C1": "Daytime scalar input",
    "C2": "Optimal rank-one state input", "C3": "Mean-amplitude 2-D state input",
    "C4": "Full Early 24-h curve input",
}


def progress(message: str) -> None:
    print(f"[Figure 4] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--response-csv", default=str(DEFAULT_RESPONSE))
    p.add_argument("--run-base", action="store_true")
    p.add_argument("--base-script", default=str(DEFAULT_BASE_SCRIPT))
    p.add_argument("--base-extra-arg", action="append", default=[])
    p.add_argument("--uncertainty-csv", default=None, help="Optional script-10 pair-period uncertainty table.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--cut-year", type=int, default=2019)
    p.add_argument("--min-events", type=int, default=2)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-cv-repeats", type=int, default=100)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument(
        "--event-thresholds",
        nargs="*",
        type=int,
        default=[3, 5, 7, 10],
        help="Uniform Early/Late event-count thresholds for secondary vector refits.",
    )
    p.add_argument(
        "--reference-run-dir",
        default=str(DEFAULT_REFERENCE_RUN),
        help="Frozen reference run used to regression-lock the primary baseline when available.",
    )
    p.add_argument("--alternative-cut-years", nargs="*", type=int, default=[2018, 2020])
    p.add_argument("--angle-min-magnitude", type=float, default=0.10)
    p.add_argument("--equivalence-margin-q2", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--skip-loco", action="store_true")
    p.add_argument("--skip-reverse", action="store_true")
    p.add_argument("--skip-alternative-cuts", action="store_true")
    p.add_argument("--skip-pca", action="store_true")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Deprecated compatibility flag; plotting now uses the separate plot script.",
    )
    p.add_argument("--final-analysis", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_overwritable_output(path: Path) -> None:
    """Prepare the dedicated analysis directory for an overwrite-safe rerun."""
    path.mkdir(parents=True, exist_ok=True)
    if path == PROJECT_ROOT or path == PROJECT_ROOT / "outputs":
        raise ValueError(f"Output directory is too broad: {path}")
    complete = path / "ANALYSIS_COMPLETE"
    if complete.exists():
        complete.unlink()


def atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        data.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
        return cls(mean, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self.mean) / self.scale

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, float) * self.scale + self.mean


class CovariateEncoder:
    """Fold-local numeric scaling and one-hot encoding with unknown-level safety."""
    numeric_candidates = ("abs_lat", "pair_distance_km", "elevation_difference_m", "elev_diff_m")
    category_candidates = ("kg_group", "kg_code", "climate_zone_main", "urban_lcz_corrected", "urban_lcz_class")

    def __init__(self) -> None:
        self.numeric: list[str] = []
        self.categories: dict[str, list[str]] = {}
        self.medians: dict[str, float] = {}
        self.scaler: Scaler | None = None

    def fit(self, data: pd.DataFrame) -> "CovariateEncoder":
        self.numeric = [c for c in self.numeric_candidates if c in data and pd.to_numeric(data[c], errors="coerce").notna().sum() >= 5]
        numeric_values = []
        for column in self.numeric:
            values = pd.to_numeric(data[column], errors="coerce")
            self.medians[column] = float(values.median())
            numeric_values.append(values.fillna(self.medians[column]).to_numpy(float))
        if numeric_values:
            self.scaler = Scaler.fit(np.column_stack(numeric_values))
        for column in self.category_candidates:
            if column in data:
                levels = sorted(data[column].fillna("Missing").astype(str).unique())
                self.categories[column] = levels[1:]  # reference level omitted
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric:
            matrix = np.column_stack([pd.to_numeric(data[c], errors="coerce").fillna(self.medians[c]).to_numpy(float) for c in self.numeric])
            parts.append(self.scaler.transform(matrix) if self.scaler else matrix)
        for column, levels in self.categories.items():
            values = data[column].fillna("Missing").astype(str) if column in data else pd.Series("Missing", index=data.index)
            parts.append(np.column_stack([(values == level).to_numpy(float) for level in levels]) if levels else np.empty((len(data), 0)))
        return np.column_stack(parts) if parts else np.empty((len(data), 0))


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 0.0, weights: np.ndarray | None = None) -> np.ndarray:
    design = add_intercept(x)
    if weights is None:
        weights = np.ones(len(design))
    root = np.sqrt(np.asarray(weights, float))
    gram = (design * root[:, None]).T @ (design * root[:, None])
    penalty = np.eye(gram.shape[0]) * lam
    penalty[0, 0] = 0.0
    return np.linalg.pinv(gram + penalty) @ ((design * root[:, None]).T @ (y * root[:, None] if np.ndim(y) == 2 else y * root))


def ridge_predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ beta


def residual_rank_one(
    c_train: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
    c_test: np.ndarray, x_test: np.ndarray, weights: np.ndarray | None = None,
) -> np.ndarray:
    """Covariates remain unrestricted; only the state-to-target block is rank one."""
    bc_y = ridge_fit(c_train, y_train, 1e-8, weights)
    bc_x = ridge_fit(c_train, x_train, 1e-8, weights)
    y_res = y_train - ridge_predict(c_train, bc_y)
    x_res = x_train - ridge_predict(c_train, bc_x)
    root = np.sqrt(weights) if weights is not None else np.ones(len(x_res))
    b_ols = np.linalg.pinv(x_res * root[:, None]) @ (y_res * root[:, None])
    fitted = x_res @ b_ols
    _, _, vt = np.linalg.svd(fitted, full_matrices=False)
    direction = vt[0:1].T
    b_rank1 = b_ols @ direction @ direction.T
    x_test_res = x_test - ridge_predict(c_test, bc_x)
    return ridge_predict(c_test, bc_y) + x_test_res @ b_rank1


def rank_one_latent_score(
    c_train: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
    c_test: np.ndarray, x_test: np.ndarray, weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fold-local predictor score underlying the joint rank-one fit."""
    bc_y = ridge_fit(c_train, y_train, 1e-8, weights)
    bc_x = ridge_fit(c_train, x_train, 1e-8, weights)
    y_res = y_train - ridge_predict(c_train, bc_y)
    x_res = x_train - ridge_predict(c_train, bc_x)
    root = np.sqrt(weights) if weights is not None else np.ones(len(x_res))
    b_ols = np.linalg.pinv(x_res * root[:, None]) @ (y_res * root[:, None])
    fitted = x_res @ b_ols
    _, _, vt = np.linalg.svd(fitted, full_matrices=False)
    b_rank1 = b_ols @ vt[0:1].T @ vt[0:1]
    left, _, _ = np.linalg.svd(b_rank1, full_matrices=False)
    w = left[:, 0]
    test_res = x_test - ridge_predict(c_test, bc_x)
    return x_res @ w.reshape(-1, 1), test_res @ w.reshape(-1, 1)


def choose_ridge_lambda(x: np.ndarray, y: np.ndarray, groups: Sequence[str], seed: int) -> float:
    lambdas = (0.01, 0.1, 1.0, 10.0, 100.0)
    unique = np.asarray(sorted(set(map(str, groups))))
    if len(unique) < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {group: index % 3 for index, group in enumerate(unique)}
    fold = np.asarray([mapping[str(group)] for group in groups])
    scores = []
    for lam in lambdas:
        errors = []
        for k in range(3):
            train, test = fold != k, fold == k
            if train.sum() <= x.shape[1] or not test.any():
                continue
            pred = ridge_predict(x[test], ridge_fit(x[train], y[train], lam))
            errors.append(np.nanmean((y[test] - pred) ** 2))
        scores.append((float(np.mean(errors)) if errors else np.inf, lam))
    return min(scores)[1]


def balanced_block_folds(blocks: pd.Series, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    counts = blocks.astype(str).value_counts().to_dict()
    ordered = list(counts)
    rng.shuffle(ordered)
    ordered.sort(key=lambda value: counts[value], reverse=True)
    loads = np.zeros(n_folds, int)
    assignment: dict[str, int] = {}
    for block in ordered:
        candidates = np.flatnonzero(loads == loads.min())
        fold = int(rng.choice(candidates))
        assignment[block] = fold
        loads[fold] += counts[block]
    return blocks.astype(str).map(assignment).to_numpy(int)


def run_base_if_requested(args: argparse.Namespace, output: Path) -> Path:
    if not args.run_base:
        return Path(args.response_csv).expanduser().resolve()
    base_output = output / "base_run"
    command = [sys.executable, str(Path(args.base_script).expanduser().resolve()), "--output-dir", str(base_output)]
    if args.final_analysis:
        command += ["--final-analysis", "--n-cv-repeats", "100", "--n-bootstrap", "1000"]
    command += list(args.base_extra_arg)
    subprocess.run(command, check=True)
    return base_output / "tables/block_specific_response_states.csv"


def make_cohort(response: pd.DataFrame, cut_year: int, min_events: int) -> pd.DataFrame:
    require_columns(response, ["pair_id", "block", "Rmean", "RAmp", "Rx", "Rn", "n_hw_events", "spatial_block", *HOURS], "response table")
    early = response[response["block"].eq(f"cut{cut_year}_E")].copy()
    late = response[response["block"].eq(f"cut{cut_year}_L")].copy()
    values = ["Rmean", "RAmp", "Rx", "Rn", "dAH", "n_hw_events", *HOURS]
    values = [c for c in values if c in response]
    early = early[["pair_id", *values]].rename(columns={c: f"{c}_E" for c in values})
    late = late[["pair_id", *values]].rename(columns={c: f"{c}_L" for c in values})
    meta_candidates = ["pair_id", "group", "continent", "spatial_block", "abs_lat", "pair_distance_km",
                       "elevation_difference_m", "elev_diff_m", "kg_group", "kg_code", "climate_zone_main",
                       "urban_lcz_corrected", "urban_lcz_class"]
    meta = response[[c for c in meta_candidates if c in response]].drop_duplicates("pair_id")
    cohort = meta.merge(early, on="pair_id", validate="one_to_one").merge(late, on="pair_id", validate="one_to_one")
    needed = ["Rmean_E", "RAmp_E", "Rmean_L", "RAmp_L", "Rx_E", "Rn_E", "Rx_L", "Rn_L", *[f"{c}_E" for c in HOURS], *[f"{c}_L" for c in HOURS]]
    if {"dAH_E", "dAH_L"}.issubset(cohort.columns):
        needed.extend(["dAH_E", "dAH_L"])
    finite = cohort[needed].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    event_ok = (cohort["n_hw_events_E"] >= min_events) & (cohort["n_hw_events_L"] >= min_events)
    cohort = cohort[finite & event_ok].copy()
    # Predictive Section 9: D is the coordinate orthogonal to the
    # conventional daytime projection.
    cohort["D_E"] = cohort["Rmean_E"] - cohort["RAmp_E"]
    cohort["D_L"] = cohort["Rmean_L"] - cohort["RAmp_L"]
    # Preserve the exact day-night redistribution under a distinct name.
    cohort["D_half_E"] = 0.5 * (cohort["Rn_E"] - cohort["Rx_E"])
    cohort["D_half_L"] = 0.5 * (cohort["Rn_L"] - cohort["Rx_L"])
    cohort["target_magnitude"] = np.hypot(cohort["Rx_L"], cohort["Rn_L"])
    return cohort.sort_values("pair_id").reset_index(drop=True)


def add_precision_weights(cohort: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    out = cohort.copy()
    out["precision_weight"] = 1.0
    if path is None:
        return out
    uncertainty = pd.read_csv(path, dtype={"pair_id": str}, low_memory=False)
    late = uncertainty[uncertainty["period"].eq("Late")][["pair_id", "trace_cov_daynight"]].drop_duplicates("pair_id")
    out = out.drop(columns="precision_weight").merge(late, on="pair_id", how="left", validate="one_to_one")
    raw = 1.0 / pd.to_numeric(out["trace_cov_daynight"], errors="coerce").clip(lower=1e-8)
    finite = raw[np.isfinite(raw)]
    if len(finite):
        lo, hi = finite.quantile([0.05, 0.95])
        raw = raw.clip(lo, hi)
        raw = raw / raw.mean()
    out["precision_weight"] = raw.fillna(1.0)
    return out


def fold_predictions(
    train: pd.DataFrame, test: pd.DataFrame, repeat: int, fold: int,
    precision: bool = False, include_covariates: bool = False,
) -> pd.DataFrame:
    if include_covariates:
        encoder = CovariateEncoder().fit(train)
        c_tr, c_te = encoder.transform(train), encoder.transform(test)
    else:
        c_tr, c_te = np.empty((len(train), 0)), np.empty((len(test), 0))
    state_tr, state_te = train[["Rmean_E", "RAmp_E"]].to_numpy(float), test[["Rmean_E", "RAmp_E"]].to_numpy(float)
    dn_tr, dn_te = train[["Rx_E", "Rn_E"]].to_numpy(float), test[["Rx_E", "Rn_E"]].to_numpy(float)
    curve_tr = train[[f"{h}_E" for h in HOURS]].to_numpy(float)
    curve_te = test[[f"{h}_E" for h in HOURS]].to_numpy(float)
    vector_y_tr = train[["Rx_L", "Rn_L"]].to_numpy(float)
    vector_y_te = test[["Rx_L", "Rn_L"]].to_numpy(float)
    curve_y_tr = train[[f"{h}_L" for h in HOURS]].to_numpy(float)
    curve_y_te = test[[f"{h}_L" for h in HOURS]].to_numpy(float)
    weights = train["precision_weight"].to_numpy(float) if precision else None

    state_scale, dn_scale, early_curve_scale = Scaler.fit(state_tr), Scaler.fit(dn_tr), Scaler.fit(curve_tr)
    vector_scale, curve_scale = Scaler.fit(vector_y_tr), Scaler.fit(curve_y_tr)
    s_tr, s_te = state_scale.transform(state_tr), state_scale.transform(state_te)
    d_tr, d_te = dn_scale.transform(dn_tr), dn_scale.transform(dn_te)
    ecurve_tr, ecurve_te = early_curve_scale.transform(curve_tr), early_curve_scale.transform(curve_te)
    vy_tr, cy_tr = vector_scale.transform(vector_y_tr), curve_scale.transform(curve_y_tr)

    def unrestricted(extra_tr: np.ndarray, extra_te: np.ndarray, target: np.ndarray, lam: float = 1e-8) -> np.ndarray:
        return ridge_predict(np.column_stack([c_te, extra_te]), ridge_fit(np.column_stack([c_tr, extra_tr]), target, lam, weights))

    vector_pred_z = {
        "M0": ridge_predict(c_te, ridge_fit(c_tr, vy_tr, 1e-8, weights)),
        "M1": unrestricted(d_tr[:, :1], d_te[:, :1], vy_tr),
        "M1n": unrestricted(d_tr[:, 1:2], d_te[:, 1:2], vy_tr),
        "M2": residual_rank_one(c_tr, s_tr, vy_tr, c_te, s_te, weights),
        "M3": unrestricted(s_tr, s_te, vy_tr),
        "M4": unrestricted(d_tr, d_te, vy_tr),
    }
    inner_groups = train["spatial_block"].astype(str).to_numpy()
    c4_x_tr, c4_x_te = np.column_stack([c_tr, ecurve_tr]), np.column_stack([c_te, ecurve_te])
    c4_lambda = choose_ridge_lambda(c4_x_tr, cy_tr, inner_groups, 9000 + repeat * 17 + fold)
    curve_pred_z = {
        "C0": ridge_predict(c_te, ridge_fit(c_tr, cy_tr, 1e-8, weights)),
        "C1": unrestricted(d_tr[:, :1], d_te[:, :1], cy_tr),
        "C2": residual_rank_one(c_tr, s_tr, cy_tr, c_te, s_te, weights),
        "C3": unrestricted(s_tr, s_te, cy_tr),
        "C4": ridge_predict(c4_x_te, ridge_fit(c4_x_tr, cy_tr, c4_lambda, weights)),
    }
    timing_predictions: dict[str, np.ndarray] = {}
    timing_obs = np.empty(0)
    if {"dAH_E", "dAH_L"}.issubset(train.columns) and {"dAH_E", "dAH_L"}.issubset(test.columns):
        timing_train = train["dAH_L"].to_numpy(float).reshape(-1, 1)
        timing_obs = test["dAH_L"].to_numpy(float)
        timing_scale = Scaler.fit(timing_train)
        timing_z = timing_scale.transform(timing_train)
        latent_tr, latent_te = rank_one_latent_score(c_tr, s_tr, vy_tr, c_te, s_te, weights)

        def timing_fit(extra_tr: np.ndarray, extra_te: np.ndarray) -> np.ndarray:
            fitted = ridge_predict(
                np.column_stack([c_te, extra_te]),
                ridge_fit(np.column_stack([c_tr, extra_tr]), timing_z, 1e-8, weights),
            )
            return timing_scale.inverse(fitted).ravel()

        timing_predictions = {
            "M0": timing_fit(np.empty((len(train), 0)), np.empty((len(test), 0))),
            "M1": timing_fit(d_tr[:, :1], d_te[:, :1]),
            "M1n": timing_fit(d_tr[:, 1:2], d_te[:, 1:2]),
            "M2": timing_fit(latent_tr, latent_te),
            "M3": timing_fit(s_tr, s_te),
            "M4": timing_fit(d_tr, d_te),
        }
    rows = []
    for model, prediction in vector_pred_z.items():
        pred = vector_scale.inverse(prediction)
        for index, (_, rec) in enumerate(test.iterrows()):
            rows.append({"task": "vector", "model": model, "repeat": repeat, "fold": fold,
                         "pair_id": rec["pair_id"], "spatial_block": rec["spatial_block"],
                         "Rx_obs": vector_y_te[index, 0], "Rn_obs": vector_y_te[index, 1],
                         "Rx_pred": pred[index, 0], "Rn_pred": pred[index, 1]})
    for model, prediction in curve_pred_z.items():
        pred = curve_scale.inverse(prediction)
        for index, (_, rec) in enumerate(test.iterrows()):
            row = {"task": "curve", "model": model, "repeat": repeat, "fold": fold,
                   "pair_id": rec["pair_id"], "spatial_block": rec["spatial_block"], "ridge_lambda": c4_lambda if model == "C4" else np.nan}
            for hour in range(24):
                row[f"h{hour:02d}_obs"] = curve_y_te[index, hour]
                row[f"h{hour:02d}_pred"] = pred[index, hour]
            rows.append(row)
    for model, prediction in timing_predictions.items():
        for index, (_, rec) in enumerate(test.iterrows()):
            rows.append({
                "task": "timing", "model": model, "repeat": repeat, "fold": fold,
                "pair_id": rec["pair_id"], "spatial_block": rec["spatial_block"],
                "dAH_obs": timing_obs[index], "dAH_pred": prediction[index],
            })
    return pd.DataFrame(rows)


def repeated_cv(
    cohort: pd.DataFrame, repeats: int, folds: int, seed: int,
    precision: bool = False, include_covariates: bool = False,
) -> pd.DataFrame:
    rows = []
    for repeat in range(repeats):
        assignment = balanced_block_folds(cohort["spatial_block"], folds, np.random.default_rng(seed + repeat))
        for fold in range(folds):
            train, test = cohort[assignment != fold], cohort[assignment == fold]
            if len(test) and len(train) >= 10:
                rows.append(fold_predictions(train, test, repeat, fold, precision, include_covariates))
    if not rows:
        raise RuntimeError("No valid held-out spatial folds were generated.")
    return pd.concat(rows, ignore_index=True)


def average_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vector = oof[oof["task"].eq("vector")]
    vector_mean = vector.groupby(["pair_id", "spatial_block", "model"], as_index=False)[["Rx_obs", "Rn_obs", "Rx_pred", "Rn_pred"]].mean()
    curve = oof[oof["task"].eq("curve")]
    value_columns = [f"h{h:02d}_{kind}" for h in range(24) for kind in ("obs", "pred")]
    curve_mean = curve.groupby(["pair_id", "spatial_block", "model"], as_index=False)[value_columns].mean()
    timing = oof[oof["task"].eq("timing")]
    timing_mean = (
        timing.groupby(["pair_id", "spatial_block", "model"], as_index=False)[["dAH_obs", "dAH_pred"]].mean()
        if not timing.empty else pd.DataFrame(columns=["pair_id", "spatial_block", "model", "dAH_obs", "dAH_pred"])
    )
    return vector_mean, curve_mean, timing_mean


def circular_hour_error(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    observed_hour, predicted_hour = np.argmax(obs, axis=1), np.argmax(pred, axis=1)
    difference = np.abs(observed_hour - predicted_hour)
    return np.minimum(difference, 24 - difference)


def _q2(obs: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.sum((obs - np.nanmean(obs, axis=0)) ** 2))
    return float(1 - np.sum((obs - pred) ** 2) / denominator) if denominator > 0 else np.nan


def _calibration(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(obs) & np.isfinite(pred)
    if ok.sum() < 3 or np.nanstd(pred[ok]) < 1e-12:
        return np.nan, np.nan
    design = np.column_stack([np.ones(ok.sum()), pred[ok]])
    intercept, slope = np.linalg.lstsq(design, obs[ok], rcond=None)[0]
    return float(intercept), float(slope)


def score_vector(data: pd.DataFrame, angle_min_magnitude: float = 0.10) -> pd.DataFrame:
    rows = []
    for model, group in data.groupby("model"):
        obs = group[["Rx_obs", "Rn_obs"]].to_numpy(float)
        pred = group[["Rx_pred", "Rn_pred"]].to_numpy(float)
        d_half_obs = 0.5 * (obs[:, 1] - obs[:, 0])
        d_half_pred = 0.5 * (pred[:, 1] - pred[:, 0])
        magnitude_obs = np.linalg.norm(obs, axis=1)
        magnitude_pred = np.linalg.norm(pred, axis=1)
        valid_angle = (magnitude_obs >= angle_min_magnitude) & (magnitude_pred > 1e-12)
        cosine = np.full(len(obs), np.nan)
        cosine[valid_angle] = np.sum(obs[valid_angle] * pred[valid_angle], axis=1) / (
            magnitude_obs[valid_angle] * magnitude_pred[valid_angle]
        )
        angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        cal_rx = _calibration(obs[:, 0], pred[:, 0]); cal_rn = _calibration(obs[:, 1], pred[:, 1])
        rows += [
            {"task": "vector", "model": model, "metric": "joint_q2", "estimate": _q2(obs, pred)},
            {"task": "vector", "model": model, "metric": "joint_rmse", "estimate": np.sqrt(np.mean((obs - pred) ** 2))},
            {"task": "vector", "model": model, "metric": "Rx_q2", "estimate": _q2(obs[:, 0], pred[:, 0])},
            {"task": "vector", "model": model, "metric": "Rn_q2", "estimate": _q2(obs[:, 1], pred[:, 1])},
            {"task": "vector", "model": model, "metric": "Rx_rmse", "estimate": np.sqrt(np.mean((obs[:, 0] - pred[:, 0]) ** 2))},
            {"task": "vector", "model": model, "metric": "Rn_rmse", "estimate": np.sqrt(np.mean((obs[:, 1] - pred[:, 1]) ** 2))},
            {"task": "vector", "model": model, "metric": "D_half_q2", "estimate": _q2(d_half_obs, d_half_pred)},
            {"task": "vector", "model": model, "metric": "D_half_rmse", "estimate": np.sqrt(np.mean((d_half_obs-d_half_pred)**2))},
            {"task": "vector", "model": model, "metric": "angular_mean_deg", "estimate": np.nanmean(angles)},
            {"task": "vector", "model": model, "metric": "angular_median_deg", "estimate": np.nanmedian(angles)},
            {"task": "vector", "model": model, "metric": "D_half_direction_agreement", "estimate": np.mean(np.sign(d_half_obs) == np.sign(d_half_pred))},
            {"task": "vector", "model": model, "metric": "calibration_intercept_Rx", "estimate": cal_rx[0]},
            {"task": "vector", "model": model, "metric": "calibration_slope_Rx", "estimate": cal_rx[1]},
            {"task": "vector", "model": model, "metric": "calibration_intercept_Rn", "estimate": cal_rn[0]},
            {"task": "vector", "model": model, "metric": "calibration_slope_Rn", "estimate": cal_rn[1]},
        ]
    return pd.DataFrame(rows)


def score_timing(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in data.groupby("model"):
        obs = group["dAH_obs"].to_numpy(float)
        pred = group["dAH_pred"].to_numpy(float)
        rows.extend([
            {"task": "timing", "model": model, "metric": "dAH_q2", "estimate": _q2(obs, pred)},
            {"task": "timing", "model": model, "metric": "dAH_rmse", "estimate": np.sqrt(np.mean((obs-pred)**2))},
        ])
    return pd.DataFrame(rows)


def score_curve(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, hourly = [], []
    for model, group in data.groupby("model"):
        obs = group[[f"h{h:02d}_obs" for h in range(24)]].to_numpy(float)
        pred = group[[f"h{h:02d}_pred" for h in range(24)]].to_numpy(float)
        center = np.mean(obs, axis=0)
        rows += [
            {"task": "curve", "model": model, "metric": "curve_q2", "estimate": 1 - np.sum((obs-pred)**2)/np.sum((obs-center)**2)},
            {"task": "curve", "model": model, "metric": "curve_rmse", "estimate": np.sqrt(np.mean((obs-pred)**2))},
            {"task": "curve", "model": model, "metric": "peak_hour_circular_mae", "estimate": np.mean(circular_hour_error(obs, pred))},
        ]
        for hour in range(24):
            denominator = np.sum((obs[:, hour] - obs[:, hour].mean()) ** 2)
            hourly.append({"model": model, "hour": hour, "q2": 1 - np.sum((obs[:, hour]-pred[:, hour])**2)/denominator if denominator > 0 else np.nan,
                           "rmse": np.sqrt(np.mean((obs[:, hour]-pred[:, hour])**2))})
    return pd.DataFrame(rows), pd.DataFrame(hourly)


def block_bootstrap_scores(
    vector: pd.DataFrame, curve: pd.DataFrame, timing: pd.DataFrame,
    n_boot: int, seed: int, angle_min_magnitude: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = pd.concat([score_vector(vector, angle_min_magnitude), score_curve(curve)[0], score_timing(timing)], ignore_index=True)
    blocks = sorted(set(vector["spatial_block"]).union(curve["spatial_block"]).union(timing.get("spatial_block", pd.Series(dtype=str))))
    rng = np.random.default_rng(seed)
    replicate_rows = []
    for replicate in range(n_boot):
        sampled = rng.choice(blocks, len(blocks), replace=True)
        v_parts, c_parts, t_parts = [], [], []
        for occurrence, block in enumerate(sampled):
            v = vector[vector["spatial_block"].eq(block)].copy(); c = curve[curve["spatial_block"].eq(block)].copy()
            v["pair_id"] = v["pair_id"].astype(str) + f"__{occurrence}"; c["pair_id"] = c["pair_id"].astype(str) + f"__{occurrence}"
            t = timing[timing["spatial_block"].eq(block)].copy()
            if not t.empty:
                t["pair_id"] = t["pair_id"].astype(str) + f"__{occurrence}"
                t_parts.append(t)
            v_parts.append(v); c_parts.append(c)
        t_score = score_timing(pd.concat(t_parts, ignore_index=True)) if t_parts else pd.DataFrame()
        scores = pd.concat([
            score_vector(pd.concat(v_parts), angle_min_magnitude),
            score_curve(pd.concat(c_parts))[0], t_score,
        ], ignore_index=True)
        scores["bootstrap"] = replicate
        replicate_rows.append(scores)
    boot = pd.concat(replicate_rows, ignore_index=True)
    summary = []
    for _, rec in point.iterrows():
        values = boot[(boot["task"].eq(rec["task"])) & (boot["model"].eq(rec["model"])) & (boot["metric"].eq(rec["metric"]))]["estimate"].dropna()
        out = rec.to_dict(); out["ci_low"] = values.quantile(0.025); out["ci_high"] = values.quantile(0.975); out["n_boot_valid"] = len(values)
        summary.append(out)
    return pd.DataFrame(summary), boot


def comparison_table(boot: pd.DataFrame, equivalence_margin_q2: float = 0.05) -> pd.DataFrame:
    comparisons = []
    for metric in ("joint_q2", "Rx_q2", "Rn_q2", "D_half_q2", "joint_rmse", "Rx_rmse", "Rn_rmse", "angular_mean_deg"):
        comparisons.extend([("vector", metric, "M3", "M2"), ("vector", metric, "M3", "M1"), ("vector", metric, "M3", "M4")])
    comparisons.extend([
        ("curve", "curve_q2", "C3", "C2"), ("curve", "curve_q2", "C3", "C4"),
        ("timing", "dAH_q2", "M3", "M2"), ("timing", "dAH_q2", "M3", "M1"),
        ("timing", "dAH_q2", "M3", "M4"),
    ])
    rows = []
    for task, metric, numerator, denominator in comparisons:
        sub = boot[(boot["task"].eq(task)) & (boot["metric"].eq(metric))]
        wide = sub.pivot(index="bootstrap", columns="model", values="estimate")
        if numerator not in wide or denominator not in wide:
            continue
        delta = (wide[numerator] - wide[denominator]).dropna()
        low, high = float(delta.quantile(0.025)), float(delta.quantile(0.975))
        rows.append({"task": task, "metric": metric, "comparison": f"{numerator}_minus_{denominator}",
                     "estimate": float(delta.mean()), "ci_low": float(delta.quantile(0.025)),
                     "ci_high": high, "n_boot_valid": len(delta),
                     "practical_equivalence_margin": equivalence_margin_q2 if metric.endswith("q2") and denominator == "M4" else np.nan,
                     "practically_equivalent": bool(low > -equivalence_margin_q2 and high < equivalence_margin_q2)
                     if metric.endswith("q2") and denominator == "M4" else False})
    return pd.DataFrame(rows)


def pair_identity_permutation(cohort: pd.DataFrame, folds: int, n_perm: int, seed: int) -> pd.DataFrame:
    """One prespecified grouped split per permutation; complete Late vectors move together."""
    rng = np.random.default_rng(seed)
    assignment = balanced_block_folds(cohort["spatial_block"], folds, np.random.default_rng(seed + 991))
    late_columns = ["Rx_L", "Rn_L", *[f"{h}_L" for h in HOURS]]
    if "dAH_L" in cohort:
        late_columns.append("dAH_L")
    strata_columns = [c for c in ("continent", "kg_group") if c in cohort]
    strata = cohort[strata_columns].fillna("Unknown").astype(str).agg("|".join, axis=1) if strata_columns else pd.Series("ALL", index=cohort.index)

    def one_score(data: pd.DataFrame) -> tuple[float, float]:
        rows = []
        for fold in range(folds):
            train, test = data[assignment != fold], data[assignment == fold]
            if len(test) and len(train) >= 10:
                rows.append(fold_predictions(train, test, 0, fold, False))
        oof = pd.concat(rows, ignore_index=True)
        vector, curve, _ = average_oof(oof)
        vs, cs = score_vector(vector), score_curve(curve)[0]
        v = vs.set_index(["model", "metric"])["estimate"]
        c = cs.set_index(["model", "metric"])["estimate"]
        return float(v[("M3", "joint_q2")] - v[("M2", "joint_q2")]), float(c[("C3", "curve_q2")] - c[("C2", "curve_q2")])

    observed_vector, observed_curve = one_score(cohort)
    rows = [{"permutation": -1, "is_observed": True, "delta_vector_M3_minus_M2": observed_vector,
             "delta_curve_C3_minus_C2": observed_curve, "permutation_unit": "complete_late_pair_response"}]
    for replicate in range(n_perm):
        permuted = cohort.copy()
        source = np.arange(len(permuted))
        for _, indices in strata.groupby(strata).groups.items():
            indices = np.asarray(list(indices), int)
            if len(indices) >= 2:
                source[indices] = rng.permutation(indices)
        permuted.loc[:, late_columns] = cohort.loc[source, late_columns].to_numpy()
        dv, dc = one_score(permuted)
        rows.append({"permutation": replicate, "is_observed": False, "delta_vector_M3_minus_M2": dv,
                     "delta_curve_C3_minus_C2": dc, "permutation_unit": "complete_late_pair_response"})
        if (replicate + 1) % max(1, n_perm // 10) == 0 or replicate + 1 == n_perm:
            progress(f"pair-identity permutations: {replicate + 1}/{n_perm}")
    result = pd.DataFrame(rows)
    for column in ("delta_vector_M3_minus_M2", "delta_curve_C3_minus_C2"):
        null = result.loc[~result["is_observed"], column]
        observed = result.loc[result["is_observed"], column].iloc[0]
        result[f"p_one_sided_{column}"] = (1 + int((null >= observed).sum())) / (1 + len(null))
    return result


def block_jackknife(vector: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block in sorted(set(vector["spatial_block"]).union(curve["spatial_block"])):
        vs, cs = score_vector(vector[~vector["spatial_block"].eq(block)]), score_curve(curve[~curve["spatial_block"].eq(block)])[0]
        v, c = vs.set_index(["model", "metric"])["estimate"], cs.set_index(["model", "metric"])["estimate"]
        rows.append({"omitted_spatial_block": block,
                     "delta_vector_M3_minus_M2": v.get(("M3", "joint_q2"), np.nan) - v.get(("M2", "joint_q2"), np.nan),
                     "delta_curve_C3_minus_C2": c.get(("C3", "curve_q2"), np.nan) - c.get(("C2", "curve_q2"), np.nan)})
    return pd.DataFrame(rows)


def reverse_temporal_direction(cohort: pd.DataFrame) -> pd.DataFrame:
    """Swap Early and Late fields without altering any response definition."""
    out = cohort.copy()
    stems = sorted({c[:-2] for c in out.columns if c.endswith("_E") and f"{c[:-2]}_L" in out.columns})
    for stem in stems:
        early = out[f"{stem}_E"].copy()
        out[f"{stem}_E"] = out[f"{stem}_L"].to_numpy()
        out[f"{stem}_L"] = early.to_numpy()
    out["D_E"] = out["Rmean_E"] - out["RAmp_E"]
    out["D_L"] = out["Rmean_L"] - out["RAmp_L"]
    out["D_half_E"] = 0.5 * (out["Rn_E"] - out["Rx_E"])
    out["D_half_L"] = 0.5 * (out["Rn_L"] - out["Rx_L"])
    out["target_magnitude"] = np.hypot(out["Rx_L"], out["Rn_L"])
    return out


def compact_scenario_score(
    cohort: pd.DataFrame, args: argparse.Namespace, scenario: str,
    precision: bool = False, include_covariates: bool = False,
) -> pd.DataFrame:
    progress(f"sensitivity scenario started: {scenario} (n={len(cohort)})")
    repeats = args.n_cv_repeats if args.final_analysis else max(10, min(args.n_cv_repeats, 20))
    oof = repeated_cv(cohort, repeats, args.n_folds, args.seed + sum(map(ord, scenario)), precision, include_covariates)
    vector, curve, timing = average_oof(oof)
    scores = pd.concat([score_vector(vector, args.angle_min_magnitude), score_curve(curve)[0], score_timing(timing)], ignore_index=True)
    scores.insert(0, "scenario", scenario)
    scores.insert(1, "n_pairs", len(cohort))
    progress(f"sensitivity scenario completed: {scenario}")
    return scores


def sensitivity_matrix(response: pd.DataFrame, cohort: pd.DataFrame, args: argparse.Namespace, uncertainty_path: Path | None) -> pd.DataFrame:
    rows = []
    for threshold in (0.05, 0.10, 0.20):
        subset = cohort[cohort["target_magnitude"] >= threshold]
        if len(subset) < 20:
            continue
        rows.extend(compact_scenario_score(subset, args, f"late_magnitude_ge_{threshold:.2f}").to_dict("records"))
    if "precision_weight" in cohort and not np.allclose(cohort["precision_weight"], 1):
        rows.extend(compact_scenario_score(cohort, args, "inverse_variance_training_only", precision=True).to_dict("records"))
    rows.extend(compact_scenario_score(cohort, args, "static_covariates_training_fold_only", include_covariates=True).to_dict("records"))
    if not args.skip_reverse:
        rows.extend(compact_scenario_score(reverse_temporal_direction(cohort), args, "reverse_L_to_E").to_dict("records"))
    if not args.skip_alternative_cuts:
        for cut in args.alternative_cut_years:
            try:
                alternative = add_precision_weights(make_cohort(response, cut, args.min_events), uncertainty_path)
            except (KeyError, ValueError):
                continue
            if len(alternative) >= max(20, args.n_folds * 3):
                rows.extend(compact_scenario_score(alternative, args, f"alternative_cut_{cut}").to_dict("records"))
    rows.append({"scenario": "peak_timing_circular", "task": "timing", "model": "curve_peak", "metric": "status",
                 "estimate": np.nan, "n_pairs": len(cohort), "note": "Peak-hour circular MAE is reported in primary performance; no linear-hour regression."})
    return pd.DataFrame(rows)


def leave_one_continent_out(cohort: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    if "continent" not in cohort:
        return pd.DataFrame(), pd.DataFrame()
    for fold, continent in enumerate(sorted(cohort["continent"].dropna().astype(str).unique())):
        progress(f"LOCO fit: held out {continent}")
        train = cohort[~cohort["continent"].astype(str).eq(continent)]
        test = cohort[cohort["continent"].astype(str).eq(continent)]
        if len(test) and len(train) >= 15:
            pred = fold_predictions(train, test, 0, fold, False, False)
            pred["held_out_continent"] = continent
            rows.append(pred)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat(rows, ignore_index=True)
    vector, curve, timing = average_oof(raw)
    score = pd.concat([score_vector(vector, args.angle_min_magnitude), score_curve(curve)[0], score_timing(timing)], ignore_index=True)
    score["held_out_continent"] = "POOLED"
    continent_rows = [score]
    for continent, data in raw.groupby("held_out_continent"):
        v, c, t = average_oof(data)
        s = pd.concat([score_vector(v, args.angle_min_magnitude), score_curve(c)[0], score_timing(t)], ignore_index=True)
        s["held_out_continent"] = continent
        continent_rows.append(s)
    metrics = pd.concat(continent_rows, ignore_index=True)
    comparisons = []
    rng = np.random.default_rng(args.seed + 61000)
    for continent, data in metrics.groupby("held_out_continent"):
        progress(f"LOCO block bootstrap: {continent}")
        raw_subset = raw if continent == "POOLED" else raw[raw["held_out_continent"].eq(continent)]
        v_base, _, t_base = average_oof(raw_subset)
        for task, metric in (("vector", "joint_q2"), ("vector", "Rn_q2"), ("timing", "dAH_q2")):
            wide = data[(data["task"].eq(task)) & (data["metric"].eq(metric))].set_index("model")["estimate"]
            if {"M2", "M3"}.issubset(wide.index):
                source = v_base if task == "vector" else t_base
                blocks = sorted(source["spatial_block"].dropna().unique())
                boot_delta = []
                for _ in range(args.n_bootstrap):
                    parts = []
                    for occurrence, block in enumerate(rng.choice(blocks, len(blocks), replace=True)):
                        part = source[source["spatial_block"].eq(block)].copy()
                        part["pair_id"] = part["pair_id"].astype(str) + f"__{occurrence}"
                        parts.append(part)
                    draw = pd.concat(parts, ignore_index=True)
                    scored = (score_vector(draw, args.angle_min_magnitude) if task == "vector" else score_timing(draw))
                    values = scored[scored["metric"].eq(metric)].set_index("model")["estimate"]
                    if {"M2", "M3"}.issubset(values.index):
                        boot_delta.append(float(values["M3"] - values["M2"]))
                boot_values = pd.Series(boot_delta, dtype=float).dropna()
                comparisons.append({
                    "held_out_continent": continent, "task": task, "metric": metric,
                    "comparison": "M3_minus_M2", "estimate": float(wide["M3"] - wide["M2"]),
                    "ci_low": boot_values.quantile(.025), "ci_high": boot_values.quantile(.975),
                    "n_boot_valid": len(boot_values),
                })
    return raw, pd.concat([metrics.assign(row_type="performance"), pd.DataFrame(comparisons).assign(row_type="comparison")], ignore_index=True)


def functional_pca_analysis(response: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(response, ["block", "pair_id", "spatial_block", "Rmean", "RAmp", *HOURS], "response table for PCA")
    full = response[response["block"].eq("FULL")].dropna(subset=["spatial_block", "Rmean", "RAmp", *HOURS]).copy()
    if len(full) < 20:
        return pd.DataFrame(), pd.DataFrame()
    matrix = full[HOURS].to_numpy(float)
    centered = matrix - matrix.mean(axis=0)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2
    ratio = variance / variance.sum()
    scores = u * singular
    mean_basis = np.ones(24); mean_basis /= np.linalg.norm(mean_basis)
    # Frozen local-solar-time first-harmonic contrast: maximum near local afternoon.
    amp_basis = np.cos(2 * np.pi * (np.arange(24) - 15) / 24)
    amp_basis -= amp_basis.mean(); amp_basis /= np.linalg.norm(amp_basis)
    summary = []
    for component in range(min(10, len(singular))):
        summary.append({
            "component": component + 1, "variance_explained": ratio[component],
            "cumulative_variance": ratio[:component+1].sum(),
            "alignment_abs_mean_basis": abs(float(vt[component] @ mean_basis)),
            "alignment_abs_amplitude_basis": abs(float(vt[component] @ amp_basis)),
            "corr_score_Rmean": np.corrcoef(scores[:, component], full["Rmean"])[0, 1],
            "corr_score_RAmp": np.corrcoef(scores[:, component], full["RAmp"])[0, 1],
            "n_pairs": len(full),
        })
    assignment = balanced_block_folds(full["spatial_block"], n_folds, np.random.default_rng(seed))
    reconstruction = []
    for fold in range(n_folds):
        train, test = matrix[assignment != fold], matrix[assignment == fold]
        ids = full.loc[assignment == fold, ["pair_id", "spatial_block"]].reset_index(drop=True)
        if len(test) == 0 or len(train) < 10:
            continue
        train_mean = train.mean(axis=0)
        _, _, basis = np.linalg.svd(train - train_mean, full_matrices=False)
        for rank in (1, 2, 3, 4):
            b = basis[:rank]
            predicted = train_mean + ((test - train_mean) @ b.T) @ b
            rmse = np.sqrt(np.mean((test - predicted) ** 2, axis=1))
            for index, value in enumerate(rmse):
                reconstruction.append({
                    "pair_id": ids.loc[index, "pair_id"], "spatial_block": ids.loc[index, "spatial_block"],
                    "fold": fold, "rank": rank, "curve_rmse_degC": value,
                })
    return pd.DataFrame(summary), pd.DataFrame(reconstruction)


def scalar_information_loss(
    cohort: pd.DataFrame, vector: pd.DataFrame, n_boot: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = vector[vector["model"].isin(["M1", "M2", "M3"])].merge(
        cohort[["pair_id", "D_E"]], on="pair_id", how="left", validate="many_to_one"
    )
    points["night_residual"] = points["Rn_obs"] - points["Rn_pred"]

    def slope(data: pd.DataFrame) -> float:
        work = data[["D_E", "night_residual"]].dropna()
        if len(work) < 3 or work["D_E"].std() < 1e-12:
            return np.nan
        return float(np.polyfit(work["D_E"], work["night_residual"], 1)[0])

    rng = np.random.default_rng(seed)
    blocks = sorted(points["spatial_block"].dropna().unique())
    rows = []
    for model, group in points.groupby("model"):
        samples = []
        for _ in range(n_boot):
            draw = pd.concat([group[group["spatial_block"].eq(b)] for b in rng.choice(blocks, len(blocks), replace=True)], ignore_index=True)
            samples.append(slope(draw))
        values = pd.Series(samples, dtype=float).dropna()
        rows.append({
            "model": model, "slope": slope(group), "ci_low": values.quantile(.025),
            "ci_high": values.quantile(.975), "n_pairs": group["pair_id"].nunique(),
        })
    summary = pd.DataFrame(rows)
    if {"M1", "M3"}.issubset(summary["model"].values):
        attenuation = float(summary.set_index("model").loc["M1", "slope"] - summary.set_index("model").loc["M3", "slope"])
        summary["M1_minus_M3_slope_attenuation"] = attenuation
    return points, summary


def curve_semantic_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": model, "label": label, "interpretation": (
            "high-dimensional full Early 24-h curve upper bound" if model == "C4" else
            "two-dimensional mean-amplitude predictor, not two harmonics" if model == "C3" else
            "scalar/rank-one/climatological comparator"
        )}
        for model, label in CURVE_MODEL_LABELS.items()
    ])


def repeated_cv_score_distribution(oof: pd.DataFrame, angle_min_magnitude: float) -> pd.DataFrame:
    rows = []
    for repeat, data in oof.groupby("repeat"):
        vector, curve, timing = average_oof(data)
        score = pd.concat([
            score_vector(vector, angle_min_magnitude), score_curve(curve)[0], score_timing(timing)
        ], ignore_index=True)
        score.insert(0, "repeat", repeat)
        rows.append(score)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fold_assignment_audit(oof: pd.DataFrame, cohort: pd.DataFrame, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = oof[(oof["task"].eq("vector")) & (oof["model"].eq("M0"))][
        ["repeat", "fold", "pair_id", "spatial_block"]
    ].drop_duplicates()
    audit = pd.DataFrame([{
        "n_pairs_expected": cohort["pair_id"].nunique(),
        "n_repeats_expected": repeats,
        "n_assignment_rows": len(assignments),
        "each_pair_once_per_repeat": bool(assignments.groupby(["repeat", "pair_id"]).size().eq(1).all()),
        "each_block_one_fold_per_repeat": bool(assignments.groupby(["repeat", "spatial_block"])["fold"].nunique().eq(1).all()),
        "all_pairs_present_each_repeat": bool(assignments.groupby("repeat")["pair_id"].nunique().eq(cohort["pair_id"].nunique()).all()),
    }])
    return assignments, audit


def sample_retention_summary(cohort: pd.DataFrame, min_events: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "n_pairs": cohort["pair_id"].nunique(), "n_spatial_blocks": cohort["spatial_block"].nunique(),
        "n_continents": cohort["continent"].nunique() if "continent" in cohort else np.nan,
        "minimum_events_per_period": min_events,
        "median_events_E": cohort["n_hw_events_E"].median(), "median_events_L": cohort["n_hw_events_L"].median(),
        "minimum_events_E": cohort["n_hw_events_E"].min(), "minimum_events_L": cohort["n_hw_events_L"].min(),
        "median_precision_weight": cohort["precision_weight"].median() if "precision_weight" in cohort else 1.0,
    }])


def robustness_decision_matrix(
    comparisons: pd.DataFrame, sensitivity: pd.DataFrame, loco: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    primary = comparisons[(comparisons["comparison"].eq("M3_minus_M2")) & (comparisons["metric"].eq("joint_q2"))]
    for _, rec in primary.iterrows():
        rows.append({"scenario": "primary", "estimate": rec["estimate"], "ci_low": rec["ci_low"], "ci_high": rec["ci_high"], "source": "paired_block_bootstrap"})
    if not sensitivity.empty:
        sub = sensitivity[(sensitivity["task"].eq("vector")) & (sensitivity["metric"].eq("joint_q2")) & (sensitivity["model"].isin(["M2", "M3"]))]
        for scenario, data in sub.groupby("scenario"):
            wide = data.set_index("model")["estimate"]
            if {"M2", "M3"}.issubset(wide.index):
                rows.append({"scenario": scenario, "estimate": float(wide["M3"]-wide["M2"]), "ci_low": np.nan, "ci_high": np.nan, "source": "one_factor_sensitivity"})
    if not loco.empty and "row_type" in loco:
        sub = loco[(loco["row_type"].eq("comparison")) & (loco["metric"].eq("joint_q2"))]
        for _, rec in sub.iterrows():
            rows.append({"scenario": f"LOCO_{rec['held_out_continent']}", "estimate": rec["estimate"], "ci_low": rec.get("ci_low",np.nan), "ci_high": rec.get("ci_high",np.nan), "source": "leave_one_continent_out"})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["direction_positive"] = result["estimate"] > 0
        result["ci_excludes_zero_positive"] = result["ci_low"] > 0
    return result


def fold_vector_predictions_primary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    repeat: int,
    fold: int,
) -> pd.DataFrame:
    """Exact state-only vector branch of the frozen primary fold model."""
    c_tr = np.empty((len(train), 0))
    c_te = np.empty((len(test), 0))
    state_tr = train[["Rmean_E", "RAmp_E"]].to_numpy(float)
    state_te = test[["Rmean_E", "RAmp_E"]].to_numpy(float)
    dn_tr = train[["Rx_E", "Rn_E"]].to_numpy(float)
    dn_te = test[["Rx_E", "Rn_E"]].to_numpy(float)
    vector_y_tr = train[["Rx_L", "Rn_L"]].to_numpy(float)
    vector_y_te = test[["Rx_L", "Rn_L"]].to_numpy(float)

    state_scale = Scaler.fit(state_tr)
    dn_scale = Scaler.fit(dn_tr)
    vector_scale = Scaler.fit(vector_y_tr)
    s_tr, s_te = state_scale.transform(state_tr), state_scale.transform(state_te)
    d_tr, d_te = dn_scale.transform(dn_tr), dn_scale.transform(dn_te)
    vy_tr = vector_scale.transform(vector_y_tr)

    def unrestricted(extra_tr: np.ndarray, extra_te: np.ndarray) -> np.ndarray:
        return ridge_predict(extra_te, ridge_fit(extra_tr, vy_tr, 1e-8))

    predictions_z = {
        "M0": ridge_predict(c_te, ridge_fit(c_tr, vy_tr, 1e-8)),
        "M1": unrestricted(d_tr[:, :1], d_te[:, :1]),
        "M1n": unrestricted(d_tr[:, 1:2], d_te[:, 1:2]),
        "M2": residual_rank_one(c_tr, s_tr, vy_tr, c_te, s_te, None),
        "M3": unrestricted(s_tr, s_te),
        "M4": unrestricted(d_tr, d_te),
    }

    rows: list[dict[str, object]] = []
    for model, prediction_z in predictions_z.items():
        prediction = vector_scale.inverse(prediction_z)
        for index, (_, rec) in enumerate(test.iterrows()):
            rows.append(
                {
                    "task": "vector",
                    "model": model,
                    "repeat": repeat,
                    "fold": fold,
                    "pair_id": rec["pair_id"],
                    "spatial_block": rec["spatial_block"],
                    "Rx_obs": vector_y_te[index, 0],
                    "Rn_obs": vector_y_te[index, 1],
                    "Rx_pred": prediction[index, 0],
                    "Rn_pred": prediction[index, 1],
                }
            )
    return pd.DataFrame(rows)


def repeated_vector_cv(
    cohort: pd.DataFrame,
    repeats: int,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for repeat in range(repeats):
        assignment = balanced_block_folds(
            cohort["spatial_block"],
            folds,
            np.random.default_rng(seed + repeat),
        )
        for fold in range(folds):
            train = cohort[assignment != fold]
            test = cohort[assignment == fold]
            if len(test) and len(train) >= 10:
                rows.append(fold_vector_predictions_primary(train, test, repeat, fold))
    if not rows:
        raise RuntimeError("No valid threshold-specific vector folds were generated.")
    return pd.concat(rows, ignore_index=True)


def average_vector_oof(oof: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        oof,
        [
            "pair_id",
            "spatial_block",
            "model",
            "Rx_obs",
            "Rn_obs",
            "Rx_pred",
            "Rn_pred",
        ],
        "vector OOF table",
    )
    return (
        oof.groupby(["pair_id", "spatial_block", "model"], as_index=False)[
            ["Rx_obs", "Rn_obs", "Rx_pred", "Rn_pred"]
        ]
        .mean()
        .sort_values(["pair_id", "model"])
        .reset_index(drop=True)
    )


def residual_line(data: pd.DataFrame) -> tuple[float, float]:
    work = data[["D_E", "night_residual"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(work) < 3 or work["D_E"].std(ddof=1) < 1e-12:
        return np.nan, np.nan
    slope, intercept = np.polyfit(
        work["D_E"].to_numpy(float),
        work["night_residual"].to_numpy(float),
        1,
    )
    return float(slope), float(intercept)


def residual_points_for_models(
    cohort: pd.DataFrame,
    vector: pd.DataFrame,
) -> pd.DataFrame:
    points = vector.loc[vector["model"].isin(["M1", "M2", "M3"])].merge(
        cohort[["pair_id", "D_E", "n_hw_events_E", "n_hw_events_L"]],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )
    points["night_residual"] = points["Rn_obs"] - points["Rn_pred"]
    return points


def threshold_block_bootstrap(
    vector: pd.DataFrame,
    cohort: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    points = residual_points_for_models(cohort, vector)
    blocks = sorted(vector["spatial_block"].dropna().astype(str).unique())
    if len(blocks) < 3:
        raise RuntimeError("Fewer than three spatial blocks in event-threshold cohort.")

    score = score_vector(vector).set_index(["model", "metric"])["estimate"]
    slope_m1, _ = residual_line(points.loc[points["model"].eq("M1")])
    slope_m2, _ = residual_line(points.loc[points["model"].eq("M2")])
    slope_m3, _ = residual_line(points.loc[points["model"].eq("M3")])
    point = {
        "slope_M1": slope_m1,
        "slope_M2": slope_m2,
        "slope_M3": slope_m3,
        "attenuation_estimate": slope_m1 - slope_m3,
        "m3_minus_m2_joint_q2": float(
            score.loc[("M3", "joint_q2")] - score.loc[("M2", "joint_q2")]
        ),
    }

    vector_by_block = {
        block: vector.loc[vector["spatial_block"].astype(str).eq(block)]
        for block in blocks
    }
    points_by_block = {
        block: points.loc[points["spatial_block"].astype(str).eq(block)]
        for block in blocks
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_bootstrap):
        sampled = rng.choice(blocks, len(blocks), replace=True)
        vector_parts = []
        point_parts = []
        for occurrence, block in enumerate(sampled):
            vector_part = vector_by_block[str(block)].copy()
            point_part = points_by_block[str(block)].copy()
            suffix = f"__boot{occurrence}"
            vector_part["pair_id"] = vector_part["pair_id"].astype(str) + suffix
            point_part["pair_id"] = point_part["pair_id"].astype(str) + suffix
            vector_parts.append(vector_part)
            point_parts.append(point_part)
        drawn_vector = pd.concat(vector_parts, ignore_index=True)
        drawn_points = pd.concat(point_parts, ignore_index=True)
        drawn_score = score_vector(drawn_vector).set_index(
            ["model", "metric"]
        )["estimate"]
        boot_m1, _ = residual_line(drawn_points.loc[drawn_points["model"].eq("M1")])
        boot_m2, _ = residual_line(drawn_points.loc[drawn_points["model"].eq("M2")])
        boot_m3, _ = residual_line(drawn_points.loc[drawn_points["model"].eq("M3")])
        rows.append(
            {
                "bootstrap": replicate,
                "slope_M1": boot_m1,
                "slope_M2": boot_m2,
                "slope_M3": boot_m3,
                "attenuation": boot_m1 - boot_m3,
                "m3_minus_m2_joint_q2": float(
                    drawn_score.loc[("M3", "joint_q2")]
                    - drawn_score.loc[("M2", "joint_q2")]
                ),
            }
        )
    return pd.DataFrame(rows), point


def run_event_threshold_analysis(
    response: pd.DataFrame,
    primary_cohort: pd.DataFrame,
    primary_oof: pd.DataFrame,
    primary_vector: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    thresholds = sorted(
        set([int(args.min_events), *[int(value) for value in args.event_thresholds]])
    )
    if any(value < 1 for value in thresholds):
        raise ValueError("All event thresholds must be positive integers.")

    summaries = []
    bootstrap_tables = []
    performances = []
    sample_flow = []
    audits = []
    for threshold in thresholds:
        is_primary = threshold == int(args.min_events)
        if is_primary:
            cohort = primary_cohort.copy()
            oof = primary_oof.copy()
            vector = primary_vector.copy()
        else:
            cohort = make_cohort(response, args.cut_year, threshold)
            cohort["precision_weight"] = 1.0
            if len(cohort) < max(20, args.n_folds * 3):
                raise RuntimeError(
                    f"Threshold n>={threshold} leaves only {len(cohort)} pairs."
                )
            progress(
                f"event threshold n>={threshold}: vector CV started "
                f"({len(cohort)} pairs)"
            )
            oof = repeated_vector_cv(
                cohort,
                args.n_cv_repeats,
                args.n_folds,
                args.seed,
            )
            vector = average_vector_oof(oof)
            progress(f"event threshold n>={threshold}: vector CV completed")

        assignments, audit = fold_assignment_audit(
            oof, cohort, args.n_cv_repeats
        )
        audit.insert(0, "min_events", threshold)
        audit["primary_cohort"] = is_primary
        audits.append(audit)
        gates = audit.iloc[0][
            [
                "each_pair_once_per_repeat",
                "each_block_one_fold_per_repeat",
                "all_pairs_present_each_repeat",
            ]
        ]
        if not bool(gates.all()):
            raise RuntimeError(
                f"Spatial CV audit failed for event threshold n>={threshold}."
            )

        point_performance = score_vector(vector)
        point_performance.insert(0, "min_events", threshold)
        point_performance.insert(1, "primary_cohort", is_primary)
        performances.append(point_performance)

        progress(f"event threshold n>={threshold}: paired block bootstrap started")
        boot, point = threshold_block_bootstrap(
            vector,
            cohort,
            args.n_bootstrap,
            args.seed + 60000 + threshold * 101,
        )
        boot.insert(0, "min_events", threshold)
        boot.insert(1, "primary_cohort", is_primary)
        bootstrap_tables.append(boot)

        attenuation_values = boot["attenuation"].dropna()
        q2_values = boot["m3_minus_m2_joint_q2"].dropna()
        summaries.append(
            {
                "threshold_label": (
                    f"Baseline (>={threshold})" if is_primary else f">={threshold}"
                ),
                "min_events": threshold,
                "primary_cohort": is_primary,
                "n_pairs": cohort["pair_id"].nunique(),
                "n_spatial_blocks": cohort["spatial_block"].nunique(),
                **point,
                "attenuation_ci_low": attenuation_values.quantile(0.025),
                "attenuation_ci_high": attenuation_values.quantile(0.975),
                "attenuation_bootstrap_mean": attenuation_values.mean(),
                "m3_minus_m2_joint_q2_ci_low": q2_values.quantile(0.025),
                "m3_minus_m2_joint_q2_ci_high": q2_values.quantile(0.975),
                "m3_minus_m2_joint_q2_bootstrap_mean": q2_values.mean(),
                "n_boot_valid": len(attenuation_values),
                "bootstrap_unit": "spatial_block",
                "cv_refitted_for_threshold": True,
                "analysis_status": "PRIMARY" if is_primary else "SECONDARY_SENSITIVITY",
            }
        )
        sample_flow.append(
            {
                "min_events": threshold,
                "primary_cohort": is_primary,
                "n_pairs": cohort["pair_id"].nunique(),
                "n_spatial_blocks": cohort["spatial_block"].nunique(),
                "n_continents": (
                    cohort["continent"].nunique() if "continent" in cohort else np.nan
                ),
                "minimum_events_E": cohort["n_hw_events_E"].min(),
                "minimum_events_L": cohort["n_hw_events_L"].min(),
            }
        )
        progress(f"event threshold n>={threshold}: paired block bootstrap completed")

    return {
        "summary": pd.DataFrame(summaries),
        "bootstrap": pd.concat(bootstrap_tables, ignore_index=True),
        "performance": pd.concat(performances, ignore_index=True),
        "sample_flow": pd.DataFrame(sample_flow),
        "cv_audit": pd.concat(audits, ignore_index=True),
    }


def comparison_point_estimates(
    comparisons: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """Separate original score contrasts from bootstrap-distribution means."""
    result = comparisons.copy()
    result = result.rename(columns={"estimate": "bootstrap_mean"})
    lookup = performance.set_index(["task", "model", "metric"])["estimate"]
    point_estimates = []
    for rec in result.itertuples(index=False):
        numerator, denominator = str(rec.comparison).split("_minus_", maxsplit=1)
        try:
            point_estimates.append(
                float(
                    lookup.loc[(rec.task, numerator, rec.metric)]
                    - lookup.loc[(rec.task, denominator, rec.metric)]
                )
            )
        except KeyError:
            point_estimates.append(np.nan)
    result.insert(
        result.columns.get_loc("bootstrap_mean"),
        "point_estimate",
        point_estimates,
    )
    result.insert(
        result.columns.get_loc("point_estimate"),
        "estimate",
        result["point_estimate"],
    )
    return result


def residual_influence_analysis(points: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def record(removal_type: str, removed: str, data: pd.DataFrame) -> None:
        slope_m1, intercept_m1 = residual_line(data.loc[data["model"].eq("M1")])
        slope_m2, intercept_m2 = residual_line(data.loc[data["model"].eq("M2")])
        slope_m3, intercept_m3 = residual_line(data.loc[data["model"].eq("M3")])
        rows.append(
            {
                "removal_type": removal_type,
                "removed": removed,
                "n_pairs": data["pair_id"].nunique(),
                "slope_M1": slope_m1,
                "intercept_M1": intercept_m1,
                "slope_M2": slope_m2,
                "intercept_M2": intercept_m2,
                "slope_M3": slope_m3,
                "intercept_M3": intercept_m3,
                "attenuation": slope_m1 - slope_m3,
            }
        )

    record("none", "NONE", points)
    for pair_id in sorted(points["pair_id"].astype(str).unique()):
        record(
            "leave_one_pair_out",
            pair_id,
            points.loc[~points["pair_id"].astype(str).eq(pair_id)],
        )
    for block in sorted(points["spatial_block"].astype(str).unique()):
        record(
            "leave_one_block_out",
            block,
            points.loc[~points["spatial_block"].astype(str).eq(block)],
        )
    return pd.DataFrame(rows)


def build_figure4_sources(
    performance: pd.DataFrame,
    comparisons: pd.DataFrame,
    vector: pd.DataFrame,
    cohort: pd.DataFrame,
    residual_points: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    panel_a = performance.loc[
        performance["task"].eq("vector")
        & performance["model"].isin(MAIN_VECTOR_MODELS)
        & performance["metric"].eq("joint_q2")
    ].copy()
    panel_a["model_label"] = panel_a["model"].map(MODEL_LABELS)

    outcomes = {
        "joint_q2": "Joint state",
        "Rx_q2": "Daytime response",
        "Rn_q2": "Night-time response",
    }
    panel_b = comparisons.loc[
        comparisons["task"].eq("vector")
        & comparisons["comparison"].eq("M3_minus_M2")
        & comparisons["metric"].isin(outcomes)
    ].copy()
    panel_b["outcome_label"] = panel_b["metric"].map(outcomes)

    panel_c = vector.loc[vector["model"].isin(["M2", "M3"])].merge(
        cohort[
            [
                "pair_id",
                "n_hw_events_E",
                "n_hw_events_L",
                *[c for c in ("group", "continent") if c in cohort.columns],
            ]
        ],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )
    panel_c["night_residual"] = panel_c["Rn_obs"] - panel_c["Rn_pred"]
    m3 = panel_c.loc[panel_c["model"].eq("M3")].copy()
    if m3.empty:
        raise RuntimeError("Figure 4c source has no M3 predictions.")
    highlighted_pair = str(
        m3.loc[m3["night_residual"].abs().idxmax(), "pair_id"]
    )
    panel_c["highlight"] = panel_c["pair_id"].astype(str).eq(highlighted_pair)
    panel_c["highlight_reason"] = np.where(
        panel_c["highlight"], "largest_absolute_M3_night_residual", ""
    )

    panel_d = threshold_summary.copy()
    points = residual_points.copy()
    points["highlight"] = points["pair_id"].astype(str).eq(highlighted_pair)
    influence = residual_influence_analysis(points)
    return {
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": panel_c,
        "panel_d": panel_d,
        "si_points": points,
        "si_influence": influence,
    }


def file_inventory(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            stat = resolved.stat()
            rows.append(
                {
                    "path": str(resolved),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(resolved),
                }
            )
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def assert_inventory_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> None:
    columns = ["path", "bytes", "mtime_ns", "sha256"]
    if not before[columns].equals(after[columns]):
        raise RuntimeError("EXISTING_FILE_MODIFIED")


def run_self_test() -> None:
    rng = np.random.default_rng(4)
    rows = []
    for i in range(60):
        state = rng.normal(size=2); late = np.array([state[0] + state[1], state[0] - state[1]]) + rng.normal(0, .1, 2)
        row = {"pair_id": f"P{i}", "spatial_block": f"B{i//3}", "continent": "X", "group": "UHI",
               "Rmean_E": state[0], "RAmp_E": state[1], "Rx_E": state[0]+state[1], "Rn_E": state[0]-state[1],
               "Rx_L": late[0], "Rn_L": late[1], "dAH_E": state[0]-.5*state[1],
               "dAH_L": .6*state[0]-.9*state[1]+rng.normal(0,.1), "precision_weight": 1.0,
               "n_hw_events_E": 5 + i % 6, "n_hw_events_L": 6 + i % 5}
        for h in range(24):
            row[f"R_h{h:02d}_E"] = state[0] + state[1] * np.cos(2*np.pi*h/24)
            row[f"R_h{h:02d}_L"] = late.mean() + (late[0]-late[1])/2 * np.cos(2*np.pi*h/24) + rng.normal(0,.05)
        rows.append(row)
    cohort = pd.DataFrame(rows)
    cohort["D_E"] = cohort["Rmean_E"] - cohort["RAmp_E"]
    if not np.allclose(cohort["D_E"], cohort["Rmean_E"] - cohort["RAmp_E"]):
        raise AssertionError("Predictive D identity self-test failed.")
    oof = repeated_cv(cohort, 2, 5, 11)
    vector, curve, timing = average_oof(oof)
    score = score_vector(vector).set_index(["model", "metric"])["estimate"]
    if score[("M3", "joint_q2")] <= score[("M1", "joint_q2")]:
        raise AssertionError("Synthetic 2-D information-gain test failed.")
    if timing.empty or "dAH_q2" not in set(score_timing(timing)["metric"]):
        raise AssertionError("Synthetic hysteresis-area prediction test failed.")
    vector_oof = repeated_vector_cv(cohort, 2, 5, 11)
    vector_only = average_vector_oof(vector_oof)
    reference = vector.sort_values(["pair_id", "model"]).reset_index(drop=True)
    if not np.allclose(
        reference[["Rx_pred", "Rn_pred"]].to_numpy(float),
        vector_only[["Rx_pred", "Rn_pred"]].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Vector-only threshold branch differs from primary vector branch.")
    bootstrap, point = threshold_block_bootstrap(vector_only, cohort, 30, 19)
    if len(bootstrap) != 30 or not np.isfinite(point["attenuation_estimate"]):
        raise AssertionError("Paired threshold attenuation bootstrap self-test failed.")
    performance, score_bootstrap = block_bootstrap_scores(
        vector,
        curve,
        timing,
        30,
        23,
        0.10,
    )
    comparisons = comparison_point_estimates(
        comparison_table(score_bootstrap, 0.05),
        performance,
    )
    residual_points, _ = scalar_information_loss(cohort, vector, 30, 29)
    attenuation_values = bootstrap["attenuation"].dropna()
    threshold_summary = pd.DataFrame(
        [
            {
                "threshold_label": "Baseline (>=2)",
                "min_events": 2,
                "primary_cohort": True,
                "n_pairs": cohort["pair_id"].nunique(),
                "n_spatial_blocks": cohort["spatial_block"].nunique(),
                **point,
                "attenuation_ci_low": attenuation_values.quantile(0.025),
                "attenuation_ci_high": attenuation_values.quantile(0.975),
            }
        ]
    )
    sources = build_figure4_sources(
        performance,
        comparisons,
        vector,
        cohort,
        residual_points,
        threshold_summary,
    )
    if set(sources) != {
        "panel_a",
        "panel_b",
        "panel_c",
        "panel_d",
        "si_points",
        "si_influence",
    }:
        raise AssertionError("Figure 4 source-data construction self-test failed.")
    print("SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.plot_only:
        raise SystemExit(
            "--plot-only has moved to 03_plotting/04_plot_figure4_predictive_state.py; "
            "this analysis script never renders figures."
        )
    if args.final_analysis and (
        args.n_cv_repeats < 100
        or args.n_bootstrap < 1000
        or args.n_permutations < 1000
    ):
        raise ValueError(
            "--final-analysis requires >=100 CV repeats, >=1000 bootstraps, "
            "and >=1000 permutations."
        )
    if args.final_analysis and any(
        (
            args.skip_loco,
            args.skip_reverse,
            args.skip_alternative_cuts,
            args.skip_pca,
        )
    ):
        raise ValueError(
            "--final-analysis requires LOCO, reverse direction, alternative cuts "
            "and PCA."
        )

    output = Path(args.output_dir).expanduser().resolve()
    prepare_overwritable_output(output)
    response_path = run_base_if_requested(args, output)
    if not response_path.is_file():
        reference_manifest = (
            Path(args.reference_run_dir).expanduser().resolve() / "run_manifest.json"
        )
        if reference_manifest.is_file():
            reference = json.loads(reference_manifest.read_text(encoding="utf-8"))
            candidate = Path(
                str(reference.get("response_csv_read_only", ""))
            ).expanduser()
            if candidate.is_file():
                response_path = candidate.resolve()
                progress(f"response resolved from frozen reference manifest: {response_path}")
    if not response_path.is_file():
        raise FileNotFoundError(response_path)

    uncertainty_path = (
        Path(args.uncertainty_csv).expanduser().resolve()
        if args.uncertainty_csv
        else None
    )
    input_paths = [Path(__file__).resolve(), response_path]
    if uncertainty_path is not None:
        if not uncertainty_path.is_file():
            raise FileNotFoundError(uncertainty_path)
        input_paths.append(uncertainty_path)
    reference_manifest = (
        Path(args.reference_run_dir).expanduser().resolve() / "run_manifest.json"
    )
    if reference_manifest.is_file():
        input_paths.append(reference_manifest)
    inventory_pre = file_inventory(input_paths)

    response = pd.read_csv(response_path, dtype={"pair_id": str}, low_memory=False)
    cohort = make_cohort(response, args.cut_year, args.min_events)
    if args.final_analysis and not {"dAH_E", "dAH_L"}.issubset(cohort.columns):
        raise KeyError(
            "Final analysis requires dAH_E and dAH_L for the independent "
            "hysteresis-area test."
        )
    cohort = add_precision_weights(cohort, uncertainty_path)
    progress(
        f"cohort ready: {len(cohort)} pairs, "
        f"{cohort['spatial_block'].nunique()} spatial blocks"
    )
    if len(cohort) < max(20, args.n_folds * 3):
        raise RuntimeError(f"Only {len(cohort)} complete pairs; insufficient for grouped CV.")

    # Primary analysis: state-only predictors; no static covariates and no precision weighting.
    progress(f"primary repeated spatial CV started: {args.n_cv_repeats} repeats")
    oof = repeated_cv(cohort, args.n_cv_repeats, args.n_folds, args.seed, False, False)
    progress("primary repeated spatial CV completed")
    vector, curve, timing = average_oof(oof)
    progress(f"spatial-block bootstrap started: {args.n_bootstrap} replicates")
    performance, boot = block_bootstrap_scores(vector, curve, timing, args.n_bootstrap, args.seed + 10000, args.angle_min_magnitude)
    progress("spatial-block bootstrap completed")
    comparisons = comparison_table(boot, args.equivalence_margin_q2)
    comparisons = comparison_point_estimates(comparisons, performance)
    hourly = score_curve(curve)[1]
    progress(f"pair-identity permutation started: {args.n_permutations} permutations")
    permutation = pair_identity_permutation(cohort, args.n_folds, args.n_permutations, args.seed + 20000)
    progress("pair-identity permutation completed")
    jackknife = block_jackknife(vector, curve)
    progress("one-factor sensitivity matrix started")
    sensitivity = sensitivity_matrix(response, cohort, args, uncertainty_path)
    progress("one-factor sensitivity matrix completed")
    progress("leave-one-continent-out validation started")
    loco_oof, loco = leave_one_continent_out(cohort, args) if not args.skip_loco else (pd.DataFrame(), pd.DataFrame())
    progress("leave-one-continent-out validation completed")
    progress("functional PCA/SVD validation started")
    pca_summary, pca_reconstruction = functional_pca_analysis(response, args.n_folds, args.seed + 40000) if not args.skip_pca else (pd.DataFrame(), pd.DataFrame())
    progress("functional PCA/SVD validation completed")
    progress("scalar-information-loss diagnostic started")
    residual_points, residual_summary = scalar_information_loss(cohort, vector, args.n_bootstrap, args.seed + 50000)
    progress("scalar-information-loss diagnostic completed")

    progress("uniform event-support analysis started")
    threshold = run_event_threshold_analysis(
        response,
        cohort,
        oof,
        vector,
        args,
    )
    progress("uniform event-support analysis completed")

    semantic_audit = curve_semantic_audit()
    repeat_scores = repeated_cv_score_distribution(oof, args.angle_min_magnitude)
    fold_assignments, fold_audit = fold_assignment_audit(oof, cohort, args.n_cv_repeats)
    retention = sample_retention_summary(cohort, args.min_events)
    decision_matrix = robustness_decision_matrix(comparisons, sensitivity, loco)
    if args.final_analysis and not bool(
        fold_audit.iloc[0][
            [
                "each_pair_once_per_repeat",
                "each_block_one_fold_per_repeat",
                "all_pairs_present_each_repeat",
            ]
        ].all()
    ):
        raise RuntimeError("Spatial-fold leakage/completeness audit failed.")
    if args.final_analysis:
        required_scenarios = {"reverse_L_to_E", "static_covariates_training_fold_only", "inverse_variance_training_only", *[f"alternative_cut_{year}" for year in args.alternative_cut_years]}
        available_scenarios = set(sensitivity.get("scenario", pd.Series(dtype=str)).dropna())
        missing_scenarios = sorted(required_scenarios.difference(available_scenarios))
        if missing_scenarios:
            raise RuntimeError(f"Required Predictive sensitivities were not generated: {missing_scenarios}")
        if pca_summary.empty or pca_reconstruction.empty:
            raise RuntimeError("Functional PCA/SVD validation was not generated.")
        if loco.empty:
            raise RuntimeError("Leave-one-continent-out validation was not generated.")

    figure_sources = build_figure4_sources(
        performance,
        comparisons,
        vector,
        cohort,
        residual_points,
        threshold["summary"],
    )

    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table_outputs = {
        "primary_model_performance_complete.csv": performance,
        "predictive_primary_performance.csv": performance,
        "primary_model_comparisons_complete.csv": comparisons,
        "coordinate_representation_comparison.csv": comparisons.loc[
            comparisons["comparison"].eq("M3_minus_M4")
        ],
        "timing_hysteresis_performance.csv": pd.concat(
            [
                performance.loc[performance["task"].eq("timing")].assign(
                    row_type="model_performance"
                ),
                comparisons.loc[comparisons["task"].eq("timing")].assign(
                    row_type="paired_comparison"
                ),
            ],
            ignore_index=True,
        ),
        "angular_direction_performance.csv": performance.loc[
            performance["task"].eq("vector")
            & performance["metric"].str.contains("angular|direction", regex=True)
        ],
        "predictive_sensitivity_matrix.csv": sensitivity,
        "full_curve_rank_comparison.csv": comparisons.loc[
            comparisons["task"].eq("curve")
        ],
        "full_curve_hourly_skill.csv": hourly,
        "pair_identity_permutation.csv": permutation,
        "spatial_block_jackknife.csv": jackknife,
        "predictive_cohort.csv": cohort,
        "vector_oof_predictions_mean.csv": vector,
        "curve_oof_predictions_mean.csv": curve,
        "timing_oof_predictions_mean.csv": timing,
        "spatial_block_bootstrap_scores.csv": boot,
        "scalar_information_loss_points.csv": residual_points,
        "scalar_information_loss_summary.csv": residual_summary,
        "curve_model_semantic_audit.csv": semantic_audit,
        "repeated_cv_score_distribution.csv": repeat_scores,
        "spatial_fold_assignments.csv": fold_assignments,
        "spatial_fold_leakage_audit.csv": fold_audit,
        "sample_retention_summary.csv": retention,
        "robustness_decision_matrix.csv": decision_matrix,
        "functional_pca_variance_alignment.csv": pca_summary,
        "functional_pca_spatial_reconstruction.csv": pca_reconstruction,
        "leave_one_continent_out_predictions.csv": loco_oof,
        "leave_one_continent_out_metrics.csv": loco,
        "figure4_oof_night_predictions.csv": vector.loc[
            vector["model"].isin(["M1", "M3"])
        ],
        "figure4_performance_plot_data.csv": performance.loc[
            performance["metric"].isin(["joint_q2", "Rx_q2", "Rn_q2"])
        ],
        "event_threshold_model_performance.csv": threshold["performance"],
        "event_threshold_residual_attenuation.csv": threshold["summary"],
        "event_threshold_residual_attenuation_bootstrap.csv": threshold[
            "bootstrap"
        ],
        "event_threshold_sample_flow.csv": threshold["sample_flow"],
        "event_threshold_spatial_cv_audit.csv": threshold["cv_audit"],
        "residual_influence_complete.csv": figure_sources["si_influence"],
        "figure4a_joint_performance_source.csv": figure_sources["panel_a"],
        "figure4b_incremental_value_source.csv": figure_sources["panel_b"],
        "figure4c_oof_night_predictions_source.csv": figure_sources["panel_c"],
        "figure4d_attenuation_threshold_source.csv": figure_sources["panel_d"],
        "figure4si_residual_points_source.csv": figure_sources["si_points"],
        "figure4si_residual_influence_source.csv": figure_sources["si_influence"],
    }
    for filename, data in table_outputs.items():
        atomic_write_csv(data, tables / filename)

    inventory_post = file_inventory(input_paths)
    assert_inventory_unchanged(inventory_pre, inventory_post)
    atomic_write_csv(inventory_pre, tables / "input_file_inventory_pre.csv")
    atomic_write_csv(inventory_post, tables / "input_file_inventory_post.csv")

    output_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(tables.glob("*.csv"))
    }
    manifest = {
        "status": "COMPLETE",
        "analysis_version": "figure4_predictive_state_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "response_csv_read_only": str(response_path),
        "response_csv_sha256": sha256_file(response_path),
        "cut_year": args.cut_year,
        "n_pairs": len(cohort),
        "n_cv_repeats": args.n_cv_repeats,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "event_thresholds": sorted(
            set([int(args.min_events), *map(int, args.event_thresholds)])
        ),
        "event_threshold_rule": (
            "Both Early and Late periods meet the same minimum event count; "
            "models are refitted within each threshold cohort."
        ),
        "final_analysis": bool(args.final_analysis),
        "primary_static_covariates": False,
        "primary_precision_weighting": False,
        "angle_min_magnitude": args.angle_min_magnitude,
        "primary_claim": "incremental predictive value of 2-D state over optimal rank-one state",
        "secondary_claim": (
            "attenuation of orthogonal-state dependence in night-time residuals, "
            "reported with uniform event-support sensitivities"
        ),
        "comparison_point_estimate_definition": (
            "Difference between full-cohort OOF Q2 point estimates; bootstrap_mean "
            "is stored separately and intervals use paired spatial-block bootstrap."
        ),
        "D_definition": "D = Rmean - RAmp (Predictive orthogonal coordinate)",
        "D_half_definition": "D_half = (Rn - Rx) / 2 (distinct day-night redistribution diagnostic)",
        "C3_role": "two-dimensional mean-amplitude predictor input; not a two-harmonic model label",
        "C4_role": "full Early 24-h curve input upper bound; not a four-harmonic model",
        "bootstrap_unit": "spatial_block",
        "plotting_separated": True,
        "output_sha256": output_hashes,
    }
    atomic_write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        output / "run_manifest.json",
    )
    atomic_write_text(
        "COMPLETE\n",
        output / "ANALYSIS_COMPLETE",
    )
    print(f"PASS: wrote predictive closure analysis to {output}")


if __name__ == "__main__":
    main()
