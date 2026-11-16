"""Tuning of the classical cleaning methods and the hybrid gate.

A small grid search on noisy copies of the recordings generated with *training* seeds
(disjoint from the evaluation seeds 0-2), over the same five error families the
noise-aware denoiser is trained on. The objective is the mean position error over all
copies after cleaning (equal weight per copy). The best settings are written to
<models_dir>/tuning.json and picked up by `filter_pipeline.clean_with`.
"""
from __future__ import annotations

import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except ImportError:
    pass

from pathlib import Path

import numpy as np
import pandas as pd

from .filters import hybrid_clean, kalman_clean, savgol_clean, tracker_clean
from .noise import apply_noise

TUNE_SEED_BASE = 2000
TUNE_CONDS = [("jitter", 0.1), ("jitter", 0.3), ("jitter", 1.0), ("distance_jitter", 0.5),
              ("drift", 0.5), ("offset", 0.6), ("dropout", 0.15), ("jitter", 0.0)]
GRID = {
    "savgol": [{"window_s": w, "polyorder": p} for w in (0.7, 0.9, 1.1, 1.5, 2.1) for p in (2, 3)],
    "kalman": [{"q_scale": q} for q in (0.25, 0.5, 1.0, 2.0, 4.0)],
    "tracker": [{"q_scale": q, "r_fixed": r} for q in (0.5, 1.0, 2.0) for r in (0.1, 0.2, 0.3, 0.5)],
    "hybrid": [{"low": lo, "high": hi} for lo in (0.03, 0.05, 0.08, 0.12)
               for hi in (0.15, 0.25, 0.4, 0.6)],
}


def _subsample(df: pd.DataFrame, n_agents: int, rng) -> pd.DataFrame:
    uids = df["uid"].unique()
    if len(uids) > n_agents:
        uids = rng.choice(uids, n_agents, replace=False)
    return df[df["uid"].isin(uids)]


def _pos_err(clean: pd.DataFrame, cleaned: pd.DataFrame) -> float:
    m = clean[["uid", "tkey", "x", "y"]].merge(
        cleaned[["orig_uid", "tkey", "x", "y"]].rename(columns={"orig_uid": "uid"}),
        on=["uid", "tkey"], suffixes=("", "_c"))
    return float(np.hypot(m["x_c"] - m["x"], m["y_c"] - m["y"]).mean())


def tune_classical(recordings, models_dir, n_agents: int = 250, verbose: bool = True) -> dict:
    """Grid-search savgol, kalman and hybrid settings; returns and saves the results."""
    rng = np.random.default_rng(TUNE_SEED_BASE)
    copies = []
    for ri, rec in enumerate(recordings):
        sub = _subsample(rec, n_agents, rng)
        for ci, (kind, lv) in enumerate(TUNE_CONDS):
            noisy = apply_noise(sub, kind, lv, np.random.default_rng(TUNE_SEED_BASE + 97 * ri + ci + 1))
            copies.append((sub, noisy, kind, lv))
    if verbose:
        print(f"{len(copies)} tuning copies ({n_agents} agents max per recording)")

    results = {}
    for cfg in GRID["savgol"]:
        results[("savgol", json.dumps(cfg))] = np.mean([_pos_err(c, savgol_clean(n, **cfg)) for c, n, *_ in copies])
    for cfg in GRID["kalman"]:
        results[("kalman", json.dumps(cfg))] = np.mean([_pos_err(c, kalman_clean(n, **cfg)) for c, n, *_ in copies])
    for cfg in GRID["tracker"]:
        results[("tracker", json.dumps(cfg))] = np.mean([_pos_err(c, tracker_clean(n, **cfg)) for c, n, *_ in copies])
    best = {}
    for m in ("savgol", "kalman", "tracker"):
        key = min((k for k in results if k[0] == m), key=results.get)
        best[m] = json.loads(key[1])
        if verbose:
            print(f"  best {m}: {best[m]} (mean error {results[key]:.3f} m)")

    try:
        from .denoiser import denoise, load_models
        models = load_models(models_dir, "mixed")
        dn = lambda d: denoise(d, models)
        q = best["kalman"]["q_scale"]
        for cfg in GRID["hybrid"]:
            results[("hybrid", json.dumps(cfg))] = np.mean(
                [_pos_err(c, hybrid_clean(n, dn, q_scale=q, **cfg)) for c, n, *_ in copies])
        key = min((k for k in results if k[0] == "hybrid"), key=results.get)
        best["hybrid"] = json.loads(key[1])
        if verbose:
            print(f"  best hybrid: {best['hybrid']} (mean error {results[key]:.3f} m)")
    except (FileNotFoundError, ImportError):
        if verbose:
            print("  mixed denoiser models (or PyTorch) not available: hybrid gate not tuned (defaults kept)")

    out = {"best": best,
           "grid": [{"method": k[0], "params": json.loads(k[1]), "mean_pos_err_m": round(float(v), 4)}
                    for k, v in results.items()],
           "conditions": TUNE_CONDS, "n_agents_per_recording": n_agents, "seed_base": TUNE_SEED_BASE}
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    (Path(models_dir) / "tuning.json").write_text(json.dumps(out, indent=2))
    return out
