"""Step 3: do position-cleaning methods restore conflict detection under noise?

For each recording, noise kind, level, seed and cleaning method: degrade the clean
trajectories exactly as in step 2 (same task seeds), clean them, rerun the conflict
screen, and compare with the same reference as step 2 (clean positions, same
kinematics derivation). The extra kind 'clean' applies each method to noise-free
positions, showing how much a method alone distorts the conflicts.

Heading and ID-switch noise are not included: methods re-derive heading from cleaned
motion, which would trivially remove heading noise, and they cannot join split tracks.
"""
from __future__ import annotations

import os
# Windows/conda: PyTorch and MKL each ship an OpenMP runtime; loading both aborts the
# process ("OMP: Error #15"). Allowing the duplicate is the standard workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:                       # load PyTorch's DLLs before NumPy/SciPy (Windows: WinError 127 otherwise)
    import torch  # noqa: F401
except ImportError:
    pass

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import json

from .conflicts import detect_conflicts
from .filters import hybrid_clean, kalman_clean, savgol_clean, tracker_clean
from .io_sind import find_recordings
from .noise import HELDOUT_GRID, UNITS, apply_noise
from .noise_eval import (bootstrap_ci, compare, conflict_table, pair_status,
                         summarise_counts, summarise_errors)
from .noise_pipeline import (CONFLICT_KW, _baseline_reference, _load, _original_reference,
                             _task_seed, save_reference_tables)

METHODS = ("none", "tracker", "savgol", "kalman", "denoiser_jitter", "denoiser_mixed", "hybrid")
STEP3_GRID = {
    "clean": [0.0],
    "jitter": [0.1, 0.3, 0.5, 1.0],
    "distance_jitter": [0.3, 0.5],
    "drift": [0.3, 0.5],
    "offset": [0.3, 0.6, 1.0],
    "dropout": [0.05, 0.15, 0.30],
    **HELDOUT_GRID,
}
DEFAULT_PARAMS = {"savgol": {"window_s": 1.1, "polyorder": 2}, "kalman": {"q_scale": 1.0},
                  "tracker": {"q_scale": 1.0, "r_fixed": 0.3}, "hybrid": {"low": 0.05, "high": 0.25}}


def load_params(models_dir) -> dict:
    """Classical-method parameters: tuned values from <models_dir>/tuning.json if
    present (scripts/05b_tune_classical.py), else the defaults."""
    params = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    f = Path(models_dir or ".") / "tuning.json"
    if f.exists():
        for k, v in json.loads(f.read_text()).get("best", {}).items():
            params.setdefault(k, {}).update(v)
    return params


CHAIN_GRID = {"jitter": [0.3, 1.0], "drift": [0.5], "dropout": [0.15]}
CHAIN_METHODS = ("tracker+savgol", "tracker+kalman", "tracker+denoiser_jitter",
                 "tracker+denoiser_mixed", "tracker+hybrid")


def clean_with(df: pd.DataFrame, method: str, models_dir=None, params=None) -> pd.DataFrame:
    params = params or load_params(models_dir)
    if method.startswith("tracker+"):
        # realistic chain: detections -> causal tracker -> post-hoc cleaning. The tracker
        # output is treated as the observed track (all frames observed, gaps already filled).
        trk = tracker_clean(df, **params["tracker"])
        trk = trk.assign(observed=True) if "observed" in trk.columns else trk
        return clean_with(trk, method.split("+", 1)[1], models_dir, params)
    if method == "none":
        return df
    if method == "tracker":
        return tracker_clean(df, **params["tracker"])
    if method == "savgol":
        return savgol_clean(df, **params["savgol"])
    if method == "kalman":
        return kalman_clean(df, **params["kalman"])
    if method.startswith("denoiser_"):
        from .denoiser import denoise, load_models
        return denoise(df, load_models(models_dir, method.split("_", 1)[1]))
    if method == "hybrid":
        from .denoiser import denoise, load_models
        models = load_models(models_dir, "mixed")
        return hybrid_clean(df, lambda d: denoise(d, models), q_scale=params["kalman"]["q_scale"],
                            **params["hybrid"])
    raise ValueError(f"unknown method '{method}'")


def run_task(rec_path: str, kind: str, level: float, seed: int, method: str,
             parts_dir: str, models_dir: str) -> str:
    rec_name = Path(rec_path).name
    stem = f"{rec_name}__{kind}__{level:g}__s{seed}__{method}"
    out_counts = Path(parts_dir) / f"{stem}__counts.csv"
    if out_counts.exists():
        return stem + " (cached)"
    clean, static_ref = _baseline_reference(rec_path)
    save_reference_tables(rec_path, parts_dir)
    if kind == "clean":
        noisy = apply_noise(_load(rec_path), "baseline", 0.0, np.random.default_rng(0))
    else:   # identical noise realisation to step 2
        noisy = apply_noise(_load(rec_path), kind, level,
                            np.random.default_rng(_task_seed(rec_name, kind, level, seed)))
    cleaned = clean_with(noisy, method, models_dir)
    pt, pp, _ = detect_conflicts(cleaned)
    uid_to_orig = dict(zip(cleaned["uid"], cleaned["orig_uid"]))
    found = conflict_table(pt, pp, uid_to_orig, **CONFLICT_KW)
    counts, matched = compare(clean, found)
    tag = dict(recording=rec_name, kind=kind, level=level, seed=seed, method=method)
    if "route_counts" in cleaned.attrs:
        tag["route_counts"] = json.dumps(cleaned.attrs["route_counts"])
    matched.assign(**tag).to_csv(Path(parts_dir) / f"{stem}__matched.csv", index=False)
    pair_status(clean, found, static_ref).assign(**tag).to_csv(
        Path(parts_dir) / f"{stem}__pairs.csv", index=False)
    found.assign(**tag).to_csv(Path(parts_dir) / f"{stem}__found.csv", index=False)
    _save_pt(pt, uid_to_orig, tag, Path(parts_dir) / f"{stem}__pt.csv")
    counts.assign(**tag).to_csv(out_counts, index=False)
    return stem


def _save_pt(pt: pd.DataFrame, uid_to_orig: dict, tag: dict, path: Path):
    """All TTC-screened pairs of a run (not only conflicts), with original identities,
    for the bias-on-all-reference-pairs and near-threshold analyses."""
    if pt.empty:
        pd.DataFrame().to_csv(path, index=False); return
    cols = ["uid_a", "uid_b", "min_ttc", "min_gap", "frames_below_3.0", "n_frames", "plausible",
            "interaction", "geometry", "rel_speed_at_min", "angle_at_min", "dist_at_min"]
    t = pt[[c for c in cols if c in pt.columns]].copy()
    t["orig_a"] = t["uid_a"].map(uid_to_orig).fillna(t["uid_a"]).astype(str)
    t["orig_b"] = t["uid_b"].map(uid_to_orig).fillna(t["uid_b"]).astype(str)
    a, b = t["orig_a"], t["orig_b"]
    t["orig_a"], t["orig_b"] = np.where(a <= b, a, b), np.where(a <= b, b, a)
    t = t.sort_values("min_ttc").drop_duplicates(["orig_a", "orig_b"])
    t.drop(columns=["uid_a", "uid_b"]).assign(**tag).to_csv(path, index=False)


def run_filter_experiments(data_root, out_dir="results/03_filters", models_dir=None,
                           grid=None, seeds=(0, 1), methods=METHODS, n_jobs: int = 1,
                           recordings=None, make_plots: bool = True, verbose: bool = True) -> dict:
    grid = grid or STEP3_GRID
    out = Path(out_dir)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    models_dir = str(models_dir or out / "models")
    if any(m.startswith("denoiser_") for m in methods):
        missing = [v for v in ("jitter", "mixed") if f"denoiser_{v}" in methods
                   and not (Path(models_dir) / f"{v}_fold0.pt").exists()]
        if missing:
            raise FileNotFoundError(f"Denoiser models {missing} not found in {models_dir}; "
                                    "run scripts/05_train_denoisers.py first.")
    recs = find_recordings(data_root)
    if recordings:
        recs = [r for r in recs if r.name in set(recordings)]
    tasks = [(str(r), kind, float(lv), s, m)
             for r in recs for kind, levels in grid.items() for lv in levels
             for s in ((0,) if kind == "clean" else seeds) for m in methods
             if not (kind == "clean" and m == "none")]
    if verbose:
        print(f"{len(tasks)} tasks; results in {parts.resolve()}")
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(run_task, *t, str(parts), models_dir) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                msg = f.result()
                if verbose:
                    print(f"  [{i}/{len(tasks)}] {msg}")
    else:
        for i, t in enumerate(tasks, 1):
            msg = run_task(*t, str(parts), models_dir)
            if verbose:
                print(f"  [{i}/{len(tasks)}] {msg}")
    return collect_filter_results(out_dir, make_plots=make_plots, verbose=verbose)


def collect_filter_results(out_dir="results/03_filters", make_plots: bool = True,
                           verbose: bool = True) -> dict:
    out = Path(out_dir)
    parts = out / "parts"
    read = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(parts.glob(pat))], ignore_index=True)
    counts, matched = read("*__counts.csv"), read("*__matched.csv")
    by = ("kind", "level", "method", "measure", "group")
    det = summarise_counts(counts, by=by)
    det_all = summarise_counts(counts.assign(group="all"), by=by)
    pairs = read("*__pairs.csv")
    ci = bootstrap_ci(pairs, by=by)
    ci_all = bootstrap_ci(pairs.assign(group="all"), by=by)
    res = {
        "detection": det.join(ci),
        "detection_pooled": det_all.join(ci_all),
        "detection_by_site": summarise_counts(counts, by=("recording",) + by),
        "errors": summarise_errors(matched, by=by),
        "errors_pooled": summarise_errors(matched.assign(group="all"), by=by),
    }
    for name, tab in res.items():
        tab.to_csv(out / f"{name}.csv")
    if make_plots:
        plot_filter_results(det_all, res["errors_pooled"], out)
    if verbose:
        d = det_all.reset_index()
        piv = d.pivot_table(index=["kind", "level"], columns=["measure", "method"],
                            values="precision")
        print("\nPrecision, all road-user groups pooled:")
        print(piv.round(2).to_string())
        print(f"\nOutputs written to {out.resolve()}")
    return res


def plot_filter_results(det_all: pd.DataFrame, err_all: pd.DataFrame, out: Path):
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    det = det_all.reset_index()
    err = err_all.reset_index()
    kinds = [k for k in STEP3_GRID if k != "clean" and k in set(det["kind"])]
    methods = [m for m in METHODS if m in set(det["method"])]
    for measure in sorted(det["measure"].unique()):
        fig, axes = plt.subplots(3, len(kinds), figsize=(3.2 * len(kinds), 8.5),
                                 squeeze=False, sharey="row")
        for j, kind in enumerate(kinds):
            for m in methods:
                d = det[(det["kind"] == kind) & (det["measure"] == measure) & (det["method"] == m)]
                e = err[(err["kind"] == kind) & (err["measure"] == measure) & (err["method"] == m)]
                axes[0, j].plot(d["level"], d["recall"], "o-", label=m)
                axes[1, j].plot(d["level"], d["precision"], "o-", label=m)
                axes[2, j].plot(e["level"], e["median_abs_err_s"], "o-", label=m)
            axes[0, j].set_title(kind.replace("_", " "))
            axes[2, j].set_xlabel(f"level [{UNITS[kind]}]")
        axes[0, 0].set_ylabel("recall")
        axes[1, 0].set_ylabel("precision")
        axes[2, 0].set_ylabel("median |error| [s]")
        for ax in axes[:2].ravel():
            ax.set_ylim(0, 1.05)
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f"{measure}: conflict detection after position cleaning "
                     f"(all road-user groups pooled)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"filters_{measure.replace('<', '_lt_').replace('.', 'p')}.png", dpi=140)
        plt.close(fig)
