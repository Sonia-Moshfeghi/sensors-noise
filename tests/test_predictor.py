"""Predictor mechanics: window building, the untrained 'stays put' ensemble, and the
pair-level evaluation on a synthetic recording with known conflicts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.filters import fill_gaps  # noqa: E402
from sensors_noise.io_sind import load_recording  # noqa: E402
from sensors_noise.noise import apply_noise  # noqa: E402
from sensors_noise.predictor import (N_ENSEMBLE, T_FUT, T_HIST, _agent_tracks, build_model,  # noqa: E402
                                     evaluate_predictions, predict_pairs, reference_pairs,
                                     training_windows)
from test_ssm import _synthetic_recording  # noqa: E402


def _rec(tmp_path, prepend_s: float = 6.0):
    """Synthetic recording with each track extended backwards at constant velocity, so
    that conflicts have enough history before them for prediction."""
    _synthetic_recording(tmp_path)
    df = load_recording(tmp_path, verbose=False)
    parts = [df]
    for uid, g in df.groupby("uid"):
        g = g.sort_values("tkey")
        first = g.iloc[0]
        k = np.arange(int(prepend_s * 10), 0, -1)
        ext = pd.DataFrame({c: first[c] for c in df.columns}, index=range(len(k)))
        ext["tkey"] = first["tkey"] - 100 * k
        ext["t"] = ext["tkey"] / 1000.0
        ext["x"] = first["x"] - first["vx"] * 0.1 * k
        ext["y"] = first["y"] - first["vy"] * 0.1 * k
        parts.append(ext)
    return pd.concat(parts, ignore_index=True).sort_values(["tkey", "uid"]).reset_index(drop=True)


def test_training_windows_shapes_and_targets(tmp_path):
    df = _rec(tmp_path)
    base = apply_noise(df, "baseline", 0.0, np.random.default_rng(0))
    grid = fill_gaps(base)
    cxy = df.set_index(["uid", "tkey"])[["x", "y"]].reindex(
        pd.MultiIndex.from_arrays([grid["orig_uid"], grid["tkey"]])).to_numpy(float)
    feats, targets, folds = training_windows(grid, cxy)
    assert feats.shape[1:] == (T_HIST, 6) and targets.shape[1:] == (T_FUT, 2)
    assert len(feats) > 0 and np.isfinite(targets).all()
    # last history frame is the origin: relative position zero
    assert np.allclose(feats[:, -1, :2], 0.0)


def test_untrained_ensemble_predicts_stay_put_and_evaluation_runs(tmp_path):
    df = _rec(tmp_path)
    pairs = reference_pairs(df)
    assert (pairs["label"] == 1).sum() >= 1
    ens = [[build_model().eval() for _ in range(N_ENSEMBLE)] for _ in (0, 1)]
    base = apply_noise(df, "baseline", 0.0, np.random.default_rng(0))
    ctracks, cattrs = _agent_tracks(base)
    pr = predict_pairs(base, pairs, ens, ctracks, cattrs)
    assert len(pr) == len(pairs)
    done = pr[~pr["skipped"]]
    assert len(done) >= 1
    # "stays put" predictions: identical across members -> zero spread
    assert np.allclose(done["pred_gap_std"], 0.0)
    assert np.isfinite(done["true_gap"]).all()
    ev = evaluate_predictions(pr)
    assert set(ev.columns) >= {"auc_gap", "precision", "recall", "unc_auc_wrong"}
