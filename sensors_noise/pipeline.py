"""Step 1: conflict inventory on clean SinD trajectories."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from .conflicts import (detect_conflicts, exclusion_report, gap_sensitivity, summarise,
                        summarise_groups)
from .io_sind import find_recordings, load_recording


def run_conflict_check(data_root, out_dir="results/01_conflict_check",
                       max_recordings=None, frame_stride=1, min_frames=3,
                       ttc_thresholds=(1.5, 3.0), pet_thresholds=(1.0, 2.0),
                       min_rel_speed=1.0, gap_threshold=1.0, gap_horizon_s=3.0,
                       static_extent=3.0, max_overlap_s=0.3,
                       make_plots=True, verbose=True):
    """Scan every recording under `data_root`, save pair tables and summaries.

    A TTC conflict needs: minimum TTC below the threshold, at least `min_frames` frames
    below it, and an actual footprint gap <= `gap_threshold` m within `gap_horizon_s`
    seconds afterwards. Pairs closing slower than `min_rel_speed` m/s are not scanned.
    TTC and PET conflicts must also be plausible: neither agent static (track extent
    < `static_extent` m) and footprints overlapping for at most `max_overlap_s` s.

    Returns a dict of DataFrames: summary, by_recording, groups, gap_sensitivity,
    pair_ttc, pair_pet.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = find_recordings(data_root)
    if not recs:
        raise FileNotFoundError(f"No SinD recordings found under {data_root}")
    if max_recordings:
        recs = recs[:max_recordings]

    all_ttc, all_pet = [], []
    for rec in recs:
        if verbose:
            print(f"Recording {rec.name}")
        df = load_recording(rec, verbose=verbose)
        pt, pp, static = detect_conflicts(df, ttc_thresholds, frame_stride, min_rel_speed,
                                          gap_horizon_s, static_extent, max_overlap_s)
        for tab in (pt, pp):
            if not tab.empty:
                tab.insert(0, "recording", rec.name)
        all_ttc.append(pt)
        all_pet.append(pp)
        if verbose:
            n_static = df[df["uid"].isin(static)].groupby("category")["uid"].nunique().to_dict()
            print(f"  static agents (track extent < {static_extent} m): {n_static or 'none'}")
            print(f"  pairs with TTC <= 10 s: {len(pt)}; crossing pairs with PET: {len(pp)}")

    pair_ttc = pd.concat(all_ttc, ignore_index=True)
    pair_pet = pd.concat(all_pet, ignore_index=True)
    kw = dict(ttc_thresholds=ttc_thresholds, pet_thresholds=pet_thresholds,
              min_frames=min_frames, gap_threshold=gap_threshold)
    res = {
        "summary": summarise(pair_ttc, pair_pet, **kw),
        "by_recording": summarise(pair_ttc, pair_pet, by=("recording", "interaction", "geometry"), **kw),
        "groups": summarise_groups(pair_ttc, pair_pet, **kw),
        "gap_sensitivity": gap_sensitivity(pair_ttc, ttc_thresholds, min_frames=min_frames),
        "exclusions": exclusion_report(pair_ttc, pair_pet, ttc_thresholds[0], pet_thresholds[-1],
                                       min_frames, gap_threshold),
        "pair_ttc": pair_ttc,
        "pair_pet": pair_pet,
    }
    for name in ("summary", "by_recording", "groups", "gap_sensitivity", "exclusions"):
        res[name].to_csv(out / f"{name}.csv")
    pair_ttc.to_csv(out / "pair_ttc.csv", index=False)
    pair_pet.to_csv(out / "pair_pet.csv", index=False)
    if make_plots:
        _plot(pair_ttc, pair_pet, out / "distributions.png")
    if verbose:
        print(f"\nSettings: min relative speed {min_rel_speed} m/s, gap <= {gap_threshold} m "
              f"within {gap_horizon_s} s, >= {min_frames} frames below TTC threshold, "
              f"static extent < {static_extent} m, overlap <= {max_overlap_s} s")
        print("\nConflict candidates removed by the plausibility checks:")
        print(res["exclusions"].to_string())
        print("\nConflict pairs after all checks, by road-user group and recording:")
        print(res["groups"].to_string())
        print("\nTTC conflicts under different gap thresholds (all recordings):")
        print(res["gap_sensitivity"].to_string())
        print(f"\nOutputs written to {out.resolve()}")
    return res


def _plot(pair_ttc, pair_pet, path):
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if not pair_ttc.empty:
        for name, g in pair_ttc.groupby("interaction"):
            axes[0].hist(g["min_ttc"], bins=40, range=(0, 10), histtype="step", label=name)
            axes[1].hist(g["min_gap"].dropna(), bins=40, range=(0, 10), histtype="step", label=name)
    axes[0].set(xlabel="Minimum TTC per pair [s]", ylabel="Pairs", title="TTC (before gap check)")
    axes[1].set(xlabel="Actual minimum footprint gap [m]", title="Realised gap after min-TTC moment")
    if not pair_pet.empty:
        for name, g in pair_pet.groupby("interaction"):
            axes[2].hist(g["pet"], bins=40, range=(0, 10), histtype="step", label=name)
    axes[2].set(xlabel="PET per crossing pair [s]", title="PET")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
