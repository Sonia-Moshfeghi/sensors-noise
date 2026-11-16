"""Step 2: how controlled sensor noise changes the conflicts found (no learning yet).

For every recording, noise kind, level and seed: degrade the clean trajectories, rerun
the full conflict screen, and compare with the reference.

Reference: the 'baseline' condition, i.e. clean positions with velocity and heading
re-derived by the same procedure as every noisy condition, so differences reflect the
noise alone. The 'baseline' row itself compares baseline with the original data (which
carries SinD's own velocities and headings) and so measures the effect of kinematics
derivation. Each task's result is
written to `out_dir/parts/` as soon as it finishes, so an interrupted run resumes where
it stopped. Results are pooled into summary tables and plots at the end.
"""
from __future__ import annotations

import functools
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .conflicts import detect_conflicts
from .io_sind import find_recordings, load_recording
from .noise import DEFAULT_GRID, UNITS, apply_noise
from .noise_eval import (bootstrap_ci, compare, conflict_table, pair_status,
                         summarise_counts, summarise_errors)

CONFLICT_KW = dict(ttc_threshold=3.0, pet_threshold=2.0, min_frames=3, gap_threshold=1.0)


@functools.lru_cache(maxsize=4)
def _load(rec_path: str) -> pd.DataFrame:
    return load_recording(rec_path, verbose=False)


@functools.lru_cache(maxsize=4)
def _original_reference(rec_path: str):
    """Conflicts and static agents in the original data (SinD's own kinematics)."""
    pt, pp, static = detect_conflicts(_load(rec_path))
    return conflict_table(pt, pp, **CONFLICT_KW), frozenset(static)


@functools.lru_cache(maxsize=4)
def _baseline_reference(rec_path: str):
    """Conflicts and static agents with clean positions and re-derived kinematics."""
    df = apply_noise(_load(rec_path), "baseline", 0.0, np.random.default_rng(0))
    pt, pp, static = detect_conflicts(df)
    return conflict_table(pt, pp, **CONFLICT_KW), frozenset(static)


def save_reference_tables(rec_path: str, parts_dir: str):
    """Once per recording: the full TTC pair tables (all screened pairs, with min TTC
    and realised gap) of the baseline and original references, and both conflict
    tables, for the near-threshold, bias-on-all-pairs and native-kinematics analyses."""
    rec_name = Path(rec_path).name
    out = Path(parts_dir) / f"{rec_name}__reference.csv"
    if out.exists():
        return
    df = _load(rec_path)
    tabs = []
    for label, data in (("baseline", apply_noise(df, "baseline", 0.0, np.random.default_rng(0))),
                        ("original", df)):
        pt, pp, static = detect_conflicts(data)
        a, b = pt["uid_a"].astype(str), pt["uid_b"].astype(str)
        cols = ["min_ttc", "min_gap", "frames_below_3.0", "n_frames", "plausible", "interaction",
                "geometry", "rel_speed_at_min", "angle_at_min", "dist_at_min"]
        t = pt[[c for c in cols if c in pt.columns]].copy()
        t["orig_a"], t["orig_b"] = np.where(a <= b, a, b), np.where(a <= b, b, a)
        t["reference"], t["recording"] = label, rec_name
        tabs.append(t)
        conflict_table(pt, pp, **CONFLICT_KW).assign(reference=label, recording=rec_name).to_csv(
            Path(parts_dir) / f"{rec_name}__reference_conflicts_{label}.csv", index=False)
    pd.concat(tabs).to_csv(out, index=False)


def _task_seed(rec_name: str, kind: str, level: float, seed: int) -> int:
    """Deterministic, distinct seed per task (independent of run order)."""
    h = zlib.crc32(f"{rec_name}|{kind}|{level:.6g}".encode()) % 10_000_019
    return int(seed * 10_000_019 + h) % (2 ** 32)


def run_task(rec_path: str, kind: str, level: float, seed: int, parts_dir: str) -> str:
    """Run one (recording, kind, level, seed) condition and save its results."""
    rec_name = Path(rec_path).name
    stem = f"{rec_name}__{kind}__{level:g}__s{seed}"
    out_counts = Path(parts_dir) / f"{stem}__counts.csv"
    if out_counts.exists():
        return stem + " (cached)"
    if kind == "baseline":     # derivation effect: baseline vs original kinematics
        clean, static_ref = _original_reference(rec_path)
        noisy, _ = _baseline_reference(rec_path)
    else:
        clean, static_ref = _baseline_reference(rec_path)
        rng = np.random.default_rng(_task_seed(rec_name, kind, level, seed))
        noisy_df = apply_noise(_load(rec_path), kind, level, rng)
        pt, pp, _ = detect_conflicts(noisy_df)
        uid_to_orig = dict(zip(noisy_df["uid"], noisy_df["orig_uid"]))
        noisy = conflict_table(pt, pp, uid_to_orig, **CONFLICT_KW)
    counts, matched = compare(clean, noisy)
    tag = dict(recording=rec_name, kind=kind, level=level, seed=seed)
    matched.assign(**tag).to_csv(Path(parts_dir) / f"{stem}__matched.csv", index=False)
    pair_status(clean, noisy, static_ref).assign(**tag).to_csv(
        Path(parts_dir) / f"{stem}__pairs.csv", index=False)
    noisy.assign(**tag).to_csv(Path(parts_dir) / f"{stem}__found.csv", index=False)
    save_reference_tables(rec_path, parts_dir)
    if kind != "baseline":
        from .filter_pipeline import _save_pt
        _save_pt(pt, uid_to_orig, tag, Path(parts_dir) / f"{stem}__pt.csv")
    counts.assign(**tag).to_csv(out_counts, index=False)   # written last = task complete
    return stem


def run_noise_experiments(data_root, out_dir="results/02_noise", grid=None, seeds=(0, 1, 2),
                          recordings=None, n_jobs: int = 1, make_plots: bool = True,
                          verbose: bool = True) -> dict:
    """Run the noise grid and write pooled summaries.

    grid: {kind: [levels]} (default DEFAULT_GRID). 'baseline' runs with seed 0 only,
    since it is deterministic. recordings: optional list of recording folder names.
    n_jobs > 1 runs tasks in parallel worker processes (use from a script on Windows).
    """
    grid = grid or DEFAULT_GRID
    out = Path(out_dir)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    recs = find_recordings(data_root)
    if recordings:
        recs = [r for r in recs if r.name in set(recordings)]
    tasks = [(str(r), kind, float(lv), s)
             for r in recs for kind, levels in grid.items() for lv in levels
             for s in ((0,) if kind == "baseline" else seeds)]
    if verbose:
        print(f"{len(tasks)} tasks ({len(recs)} recordings); results in {parts.resolve()}")

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(run_task, *t, str(parts)) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                msg = f.result()
                if verbose:
                    print(f"  [{i}/{len(tasks)}] {msg}")
    else:
        for i, t in enumerate(tasks, 1):
            msg = run_task(*t, str(parts))
            if verbose:
                print(f"  [{i}/{len(tasks)}] {msg}")

    return collect_results(out_dir, make_plots=make_plots, verbose=verbose)


def collect_results(out_dir="results/02_noise", make_plots: bool = True, verbose: bool = True) -> dict:
    """Pool all finished task files into summary tables (can be rerun at any time)."""
    out = Path(out_dir)
    parts = out / "parts"
    counts = pd.concat([pd.read_csv(f) for f in sorted(parts.glob("*__counts.csv"))],
                       ignore_index=True)
    mfiles = sorted(parts.glob("*__matched.csv"))
    matched = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True)
    pfiles = sorted(parts.glob("*__pairs.csv"))
    pairs = pd.concat([pd.read_csv(f) for f in pfiles], ignore_index=True) if pfiles else None
    ci = bootstrap_ci(pairs) if pairs is not None else None
    res = {
        "detection": summarise_counts(counts).join(ci) if ci is not None else summarise_counts(counts),
        "detection_by_site": summarise_counts(counts, by=("recording", "kind", "level", "measure", "group")),
        "errors": summarise_errors(matched),
        "errors_by_site": summarise_errors(matched, by=("recording", "kind", "level", "measure", "group")),
    }
    if pairs is not None:
        f = pairs[pairs["status"] == "false"]
        res["false_diagnostics"] = (f.groupby(["kind", "level", "measure", "group"])
                                     .agg(false_per_seed=("status", "size"),
                                          static_share=("static_in_reference", "mean"))
                                     .assign(false_per_seed=lambda x: x["false_per_seed"] /
                                             f.groupby(["kind", "level", "measure", "group"])["seed"].nunique())
                                     .round(3))
    for name, tab in res.items():
        tab.to_csv(out / f"{name}.csv")
    if make_plots:
        plot_results(res["detection"], res["errors"], out)
    if verbose:
        cols = ["n_clean", "found", "missed", "false", "recall", "precision"]
        print("\nDetection, pooled over sites and seeds (counts are per seed):")
        print(res["detection"][cols].to_string())
        print(f"\nOutputs written to {out.resolve()}")
    res["counts_raw"], res["matched_raw"], res["pairs_raw"] = counts, matched, pairs
    return res


def plot_results(detection: pd.DataFrame, errors: pd.DataFrame, out: Path):
    """One figure per measure: recall, precision and median absolute error vs level,
    one panel column per noise kind, one line per road-user group."""
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    det = detection.reset_index()
    err = errors.reset_index()
    kinds = [k for k in DEFAULT_GRID if k != "baseline" and k in set(det["kind"])]
    for measure in sorted(det["measure"].unique()):
        fig, axes = plt.subplots(3, len(kinds), figsize=(3.0 * len(kinds), 8), squeeze=False,
                                 sharey="row")
        for j, kind in enumerate(kinds):
            d = det[(det["kind"] == kind) & (det["measure"] == measure)]
            e = err[(err["kind"] == kind) & (err["measure"] == measure)]
            for grp, gd in d.groupby("group"):
                axes[0, j].plot(gd["level"], gd["recall"], "o-", label=grp)
                axes[1, j].plot(gd["level"], gd["precision"], "o-", label=grp)
            for grp, ge in e.groupby("group"):
                axes[2, j].plot(ge["level"], ge["median_abs_err_s"], "o-", label=grp)
            axes[0, j].set_title(kind.replace("_", " "))
            axes[2, j].set_xlabel(f"level [{UNITS[kind]}]")
        axes[0, 0].set_ylabel("recall")
        axes[1, 0].set_ylabel("precision")
        axes[2, 0].set_ylabel("median |error| [s]")
        for ax in axes[:2].ravel():
            ax.set_ylim(0, 1.05)
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f"{measure}: conflict detection under sensor noise "
                     f"(reference: clean positions, same kinematics derivation)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"noise_{measure.replace('<', '_lt_').replace('.', 'p')}.png", dpi=140)
        plt.close(fig)
