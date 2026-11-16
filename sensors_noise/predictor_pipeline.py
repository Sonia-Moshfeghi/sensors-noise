"""Step 4 runner: proactive conflict detection from predicted trajectories under noise."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except ImportError:
    pass

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .io_sind import find_recordings, load_recording
from .noise import UNITS, apply_noise
from .noise_pipeline import _task_seed
from .predictor import (_agent_tracks, evaluate_predictions, load_ensembles, predict_pairs,
                        reference_pairs)

STEP4_GRID = {"baseline": [0.0], "jitter": [0.1, 0.3, 0.5, 1.0], "distance_jitter": [0.5],
              "drift": [0.5], "offset": [0.6], "dropout": [0.15]}
VARIANTS = ("clean", "noise_aware")
CLEANING = ("none", "denoiser_mixed")


def run_predictor_experiments(data_root, out_dir="results/04_predictor",
                              models_dir="results/04_predictor/models",
                              denoiser_dir="results/03_filters/models", grid=None,
                              seeds=(0, 1), variants=VARIANTS, cleaning=CLEANING,
                              recordings=None, verbose=True) -> dict:
    grid = grid or STEP4_GRID
    out = Path(out_dir); (out / "parts").mkdir(parents=True, exist_ok=True)
    ens = {v: load_ensembles(models_dir, v) for v in variants}
    denoiser = None
    if "denoiser_mixed" in cleaning:
        from .denoiser import denoise, load_models
        dmodels = load_models(denoiser_dir, "mixed")
        denoiser = lambda df: denoise(df, dmodels)
    recs = find_recordings(data_root)
    if recordings:
        recs = [r for r in recs if r.name in set(recordings)]
    for rec in recs:
        clean = load_recording(rec, verbose=False)
        base = apply_noise(clean, "baseline", 0.0, np.random.default_rng(0))
        pairs = reference_pairs(clean)
        ctracks, cattrs = _agent_tracks(base)
        if verbose:
            print(f"Recording {rec.name}: {int((pairs.label == 1).sum())} positive and "
                  f"{int((pairs.label == 0).sum())} negative pairs")
        for kind, levels in grid.items():
            for lv in levels:
                for s in ((0,) if kind == "baseline" else seeds):
                    stem = f"{rec.name}__{kind}__{lv:g}__s{s}"
                    if (out / "parts" / f"{stem}__eval.csv").exists():
                        if verbose:
                            print(f"  {stem} (cached)")
                        continue
                    noisy = (base if kind == "baseline" else
                             apply_noise(clean, kind, lv, np.random.default_rng(_task_seed(rec.name, kind, lv, s))))
                    evals, preds = [], []
                    for cl in cleaning:
                        df = noisy if cl == "none" else denoiser(noisy)
                        for v in variants:
                            pr = predict_pairs(df, pairs, ens[v], ctracks, cattrs)
                            tag = dict(recording=rec.name, kind=kind, level=lv, seed=s, cleaning=cl, variant=v)
                            evals.append(evaluate_predictions(pr).assign(**tag))
                            preds.append(pr.assign(**tag))
                    pd.concat(evals).to_csv(out / "parts" / f"{stem}__eval.csv", index=False)
                    pd.concat(preds).to_csv(out / "parts" / f"{stem}__pred.csv", index=False)
                    if verbose:
                        print(f"  {stem}")
    return collect_predictor_results(out_dir, verbose=verbose)


def collect_predictor_results(out_dir="results/04_predictor", verbose=True) -> dict:
    out = Path(out_dir)
    ev = pd.concat([pd.read_csv(f) for f in sorted((out / "parts").glob("*__eval.csv"))], ignore_index=True)
    pr = pd.concat([pd.read_csv(f) for f in sorted((out / "parts").glob("*__pred.csv"))], ignore_index=True)
    # Pool pairs over sites and seeds, then recompute metrics (counts are small per site).
    by = ["kind", "level", "cleaning", "variant"]
    pooled = (pr.groupby(by).apply(lambda g: evaluate_predictions(g), include_groups=False)
                .reset_index(level=list(range(len(by)))).reset_index(drop=True))
    res = {"pooled": pooled, "by_site_seed": ev, "predictions": pr}
    pooled.to_csv(out / "pooled.csv", index=False)
    ev.to_csv(out / "by_site_seed.csv", index=False)
    plot_predictor_results(pooled, out)
    if verbose:
        cols = ["auc_gap", "precision", "recall", "unc_auc_wrong"]
        print("\nPooled over sites and seeds:")
        print(pooled.pivot_table(index=["source", "kind", "level"], columns=["cleaning", "variant"],
                                 values=cols).round(2).to_string())
        print(f"\nOutputs written to {out.resolve()}")
    return res


def plot_predictor_results(pooled: pd.DataFrame, out: Path):
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    kinds = [k for k in STEP4_GRID if k != "baseline" and k in set(pooled["kind"])]
    for src in sorted(pooled["source"].unique()):
        fig, axes = plt.subplots(2, len(kinds), figsize=(3.2 * len(kinds), 6), squeeze=False, sharey="row")
        for j, kind in enumerate(kinds):
            for (cl, v), g in pooled[(pooled["kind"] == kind) & (pooled["source"] == src)].groupby(["cleaning", "variant"]):
                lab = f"{v} / {cl}"
                axes[0, j].plot(g["level"], g["auc_gap"], "o-", label=lab)
                axes[1, j].plot(g["level"], g["unc_auc_wrong"], "o-", label=lab)
            axes[0, j].set_title(kind.replace("_", " "))
            axes[1, j].set_xlabel(f"level [{UNITS[kind]}]")
            for ax in axes[:, j]:
                ax.axhline(0.5, c="0.6", ls=":", lw=0.8)
        axes[0, 0].set_ylabel("conflict AUC (predicted gap)")
        axes[1, 0].set_ylabel("uncertainty AUC (flags wrong calls)")
        for ax in axes.ravel():
            ax.set_ylim(0.3, 1.02)
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f"{src}-type pairs: proactive conflict detection from predicted trajectories", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"predictor_{src}.png", dpi=140)
        plt.close(fig)
