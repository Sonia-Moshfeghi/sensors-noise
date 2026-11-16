"""Position-cleaning methods: gap filling, error reduction, and the denoiser's
identity start and inference mechanics."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.filters import fill_gaps, kalman_clean, savgol_clean  # noqa: E402
from sensors_noise.noise import apply_noise  # noqa: E402
from test_noise import _straight_tracks  # noqa: E402


def _pos_err(ref, d):
    m = ref.merge(d, on=["uid", "tkey"], suffixes=("", "_n"))
    return np.hypot(m["x_n"] - m["x"], m["y_n"] - m["y"]).mean(), len(m)


def test_fill_gaps_restores_grid():
    df = _straight_tracks(50)
    d = apply_noise(df, "dropout", 0.3, np.random.default_rng(0))
    g = fill_gaps(d)
    assert g["observed"].sum() == len(d)
    # interpolated positions on straight constant-speed tracks are exact
    e, n = _pos_err(df, g)
    assert e < 1e-9 and n == len(g)


def test_classical_filters_reduce_jitter():
    df = _straight_tracks(100)
    n = apply_noise(df, "jitter", 0.5, np.random.default_rng(1))
    raw, _ = _pos_err(df, n)
    sg, _ = _pos_err(df, savgol_clean(n))
    kf, _ = _pos_err(df, kalman_clean(n))
    assert sg < 0.6 * raw and kf < 0.4 * raw


def test_filters_leave_clean_tracks_nearly_unchanged():
    df = _straight_tracks(50)
    b = apply_noise(df, "baseline", 0.0, np.random.default_rng(0))
    assert _pos_err(df, savgol_clean(b))[0] < 1e-6
    assert _pos_err(df, kalman_clean(b))[0] < 0.02


def test_untrained_denoiser_is_identity_and_fills_gaps():
    from sensors_noise.denoiser import build_model, denoise
    df = _straight_tracks(30)
    d = apply_noise(df, "dropout", 0.2, np.random.default_rng(2))
    models = [build_model().eval(), build_model().eval()]
    out = denoise(d, models)
    assert len(out) == len(fill_gaps(d))            # missed frames imputed
    e, _ = _pos_err(df, out)
    assert e < 1e-5                                 # zero-initialised output = no change


def test_fill_gaps_uses_real_frame_times():
    """Frame times drift (video), and tracks start at different frames: every filled
    track must sit on the recording's actual frame times, so tracks still share frames."""
    rng = np.random.default_rng(3)
    frames = np.cumsum(np.r_[1_000_003, rng.choice([99, 100, 101], 199)])   # irregular
    rows = []
    for uid, (a, b) in {"V_1": (0, 150), "V_2": (37, 200), "P_1": (80, 120)}.items():
        for k in range(a, b):
            rows.append({"uid": uid, "tkey": frames[k], "x": 0.5 * k, "y": float(uid[-1]),
                         "heading": 0.0, "category": "vehicle" if uid[0] == "V" else "pedestrian",
                         "agent_type": "car" if uid[0] == "V" else "pedestrian",
                         "length": 4.5, "width": 1.8})
    import pandas as pd
    df = pd.DataFrame(rows)
    d = apply_noise(df, "dropout", 0.2, np.random.default_rng(0))
    g = fill_gaps(d)
    assert set(g["tkey"]).issubset(set(frames))
    shared = g.groupby("tkey")["uid"].nunique()
    assert (shared.loc[frames[80:120]] == 3).all()       # all three present on those frames
    # A track is filled on every recording frame between its first and last detection.
    v1 = d[d["uid"] == "V_1"]["tkey"]
    noisy_frames = np.unique(d["tkey"])
    expected = ((noisy_frames >= v1.min()) & (noisy_frames <= v1.max())).sum()
    assert len(g[g["uid"] == "V_1"]) == expected


def test_tracker_is_causal_and_reduces_jitter():
    """The tracker-like condition filters forward only: cleaning a track must not
    depend on later samples, and it reduces jitter less than the smoother."""
    from sensors_noise.filters import tracker_clean
    df = _straight_tracks(60)
    n = apply_noise(df, "jitter", 0.5, np.random.default_rng(1))
    raw, _ = _pos_err(df, n)
    tr_err, _ = _pos_err(df, tracker_clean(n))
    kf_err, _ = _pos_err(df, kalman_clean(n))
    assert kf_err < tr_err < raw
    # causality: truncating the future leaves the past output unchanged
    full = tracker_clean(n)
    cut = tracker_clean(n[n["tkey"] <= 15_000])
    m = full.merge(cut, on=["uid", "tkey"], suffixes=("", "_c"))
    assert np.allclose(m["x"], m["x_c"]) and np.allclose(m["y"], m["y_c"])


def test_noise_estimate_and_hybrid_routing():
    from sensors_noise.filters import hybrid_clean, track_noise_estimates
    df = _straight_tracks(60)
    for lv in (0.1, 0.5):
        sig = track_noise_estimates(apply_noise(df, "jitter", lv, np.random.default_rng(2)))
        assert abs(sig.median() - lv / np.sqrt(np.pi / 2)) < 0.03   # per-axis sigma recovered
    n = apply_noise(df, "jitter", 0.5, np.random.default_rng(3))
    out = hybrid_clean(n, denoiser=lambda d: kalman_clean(d), low=0.05, high=0.25)
    assert out.attrs["route_counts"].get("denoiser", 0) == 60       # all tracks are noisy
    assert len(out) == len(fill_gaps(n))
