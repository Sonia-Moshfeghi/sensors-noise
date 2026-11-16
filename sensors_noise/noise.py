"""Controlled, type-specific sensor noise for clean trajectory tables.

Every noisy table is produced the way a roadside camera pipeline would produce it:
positions (and rows) are degraded first, then velocity and heading are re-derived
from the noisy positions with one fixed procedure (`derive_kinematics`). The
'baseline' condition applies that procedure to clean positions, so differences between
'baseline' and the original data come from kinematics derivation alone, and differences
between any noise condition and 'baseline' come from the noise.

Position levels are mean 2-D position errors in metres, the quantity roadside camera
validation studies report; for isotropic Gaussian noise the per-axis standard deviation
is level / sqrt(pi/2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

MEAN_TO_SIGMA = 1.0 / np.sqrt(np.pi / 2.0)   # mean 2-D error -> per-axis sigma
VRU = ("pedestrian", "two_wheeler")

DEFAULT_GRID = {
    "baseline": [0.0],
    "jitter": [0.1, 0.3, 0.5, 1.0],          # mean 2-D error [m], independent per frame
    "distance_jitter": [0.3, 0.5],           # mean 2-D error [m] at 40 m from the camera
    "drift": [0.3, 0.5],                     # mean 2-D error [m], correlated over ~1 s
    "offset": [0.3, 0.6, 1.0],               # constant shift along heading per agent [m]
    "heading": [2.0, 5.0, 10.0],             # per-frame heading error, std [deg]
    "dropout": [0.05, 0.15, 0.30],           # fraction of frames missed, in bursts
    "id_switch": [0.5, 1.0, 3.0],            # expected track splits per VRU track
}

# Held-out variants used only for evaluation (never in denoiser training): same error
# families with parameters the models never saw.
HELDOUT_GRID = {
    "jitter_t": [0.3],        # heavy-tailed (Student-t, 3 d.o.f.) jitter, mean 2-D error 0.3 m
    "drift_slow": [0.5],      # drift with a 3-s instead of 1-s correlation time
    "dropout_long": [0.15],   # missed detections in 1.5-s instead of 0.5-s bursts
}

UNITS = {"baseline": "", "jitter": "m", "distance_jitter": "m at 40 m", "drift": "m",
         "offset": "m", "heading": "deg", "dropout": "fraction", "id_switch": "per VRU track",
         "jitter_t": "m", "drift_slow": "m", "dropout_long": "fraction"}


# --------------------------------------------------------------------------------------
# Kinematics derivation (identical for every condition)
# --------------------------------------------------------------------------------------

def derive_kinematics(df: pd.DataFrame, half_window_s: float = 0.5,
                      min_speed: float = 0.5) -> pd.DataFrame:
    """Velocity by central differences over +/- `half_window_s` (time-aware, so gaps
    from missed detections are handled), heading from the velocity direction where the
    agent moves faster than `min_speed` m/s, carried forward/back otherwise."""
    df = df.sort_values(["uid", "tkey"]).reset_index(drop=True)
    t = df["tkey"].to_numpy(float) / 1000.0
    x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
    vx, vy = np.zeros(len(df)), np.zeros(len(df))
    heading = df["heading"].to_numpy(float).copy()
    dt = np.median(np.diff(np.unique(t))) if len(df) > 1 else 0.1
    k = max(1, int(round(half_window_s / dt)))
    for idx in df.groupby("uid", sort=False).indices.values():
        n = len(idx)
        if n < 2:
            continue
        i = np.arange(n)
        lo, hi = idx[np.maximum(i - k, 0)], idx[np.minimum(i + k, n - 1)]
        span = t[hi] - t[lo]
        span[span == 0] = np.nan
        vx[idx] = np.nan_to_num((x[hi] - x[lo]) / span)
        vy[idx] = np.nan_to_num((y[hi] - y[lo]) / span)
        h = np.where(np.hypot(vx[idx], vy[idx]) > min_speed,
                     np.arctan2(vy[idx], vx[idx]), np.nan)
        h = pd.Series(h).ffill().bfill().to_numpy()
        heading[idx] = np.where(np.isnan(h), heading[idx], h)
    return df.assign(vx=vx, vy=vy, heading=heading)


# --------------------------------------------------------------------------------------
# Noise models
# --------------------------------------------------------------------------------------

def _ar1(n, sigma, phi, rng):
    """Stationary AR(1) series with standard deviation sigma."""
    w = rng.standard_normal(n)
    b = sigma * np.sqrt(1.0 - phi ** 2)
    e0 = rng.normal(0.0, sigma)
    return lfilter([b], [1.0, -phi], w, zi=[phi * e0])[0]


def _burst_mask(n, p_miss, mean_burst, rng):
    """Boolean keep-mask of length n with a fraction ~p_miss of frames missing in
    bursts of mean length `mean_burst` frames (alternating geometric run lengths)."""
    if p_miss <= 0:
        return np.ones(n, bool)
    mean_hit = mean_burst * (1.0 - p_miss) / p_miss
    keep, out, state = 0, [], rng.random() >= p_miss   # True = detected
    while keep < n:
        run = rng.geometric(1.0 / (mean_hit if state else mean_burst))
        out.append(np.full(run, state))
        keep += run
        state = not state
    return np.concatenate(out)[:n]


def apply_noise(df: pd.DataFrame, kind: str, level: float, rng: np.random.Generator,
                camera_xy=None, drift_tau_s: float = 1.0, dropout_burst_s: float = 0.5,
                switch_gap_s: float = 0.3, vehicle_switch_factor: float = 1 / 3,
                min_track_frames: int = 3) -> pd.DataFrame:
    """Return a degraded copy of a clean trajectory table (see module docstring).

    Adds `orig_uid` so conflicts can be matched to the clean reference even after
    identity switches split tracks into new uids.
    """
    if kind not in DEFAULT_GRID and kind not in HELDOUT_GRID:
        raise ValueError(f"unknown noise kind '{kind}'")
    df = df.sort_values(["uid", "tkey"]).reset_index(drop=True).copy()
    if "orig_uid" not in df.columns:
        df["orig_uid"] = df["uid"]
    n = len(df)
    groups = df.groupby("uid", sort=False).indices
    dt = np.median(np.diff(np.unique(df["tkey"]))) / 1000.0
    sigma = level * MEAN_TO_SIGMA

    if kind == "jitter":
        df["x"] += rng.normal(0, sigma, n)
        df["y"] += rng.normal(0, sigma, n)

    elif kind == "jitter_t":
        # Student-t with 3 d.o.f. per axis, scaled so the mean 2-D error equals `level`:
        # the mean radius of a unit t3 pair is 1.789 (Monte Carlo), vs 1.253 for Gaussian.
        s_t = level / 1.789
        df["x"] += s_t * rng.standard_t(3, n)
        df["y"] += s_t * rng.standard_t(3, n)

    elif kind == "distance_jitter":
        cx, cy = camera_xy if camera_xy is not None else (df["x"].min(), df["y"].min())
        d = np.hypot(df["x"] - cx, df["y"] - cy).to_numpy()
        s = sigma * np.maximum(d / 40.0, 0.25)
        df["x"] += rng.normal(0, 1, n) * s
        df["y"] += rng.normal(0, 1, n) * s

    elif kind in ("drift", "drift_slow"):
        phi = np.exp(-dt / (drift_tau_s if kind == "drift" else 3.0 * drift_tau_s))
        ex, ey = np.zeros(n), np.zeros(n)
        for idx in groups.values():
            ex[idx] = _ar1(len(idx), sigma, phi, rng)
            ey[idx] = _ar1(len(idx), sigma, phi, rng)
        df["x"] += ex
        df["y"] += ey

    elif kind == "offset":
        sign = {u: rng.choice([-1.0, 1.0]) for u in groups}
        s = df["uid"].map(sign).to_numpy() * level
        df["x"] += s * np.cos(df["heading"])
        df["y"] += s * np.sin(df["heading"])

    elif kind in ("dropout", "dropout_long"):
        keep = np.ones(n, bool)
        burst = max(1.0, (dropout_burst_s if kind == "dropout" else 3.0 * dropout_burst_s) / dt)
        for idx in groups.values():
            keep[idx] = _burst_mask(len(idx), level, burst, rng)
        df = df[keep]

    elif kind == "id_switch":
        gap = int(round(switch_gap_s / dt))
        uid = df["uid"].to_numpy(object).copy()
        keep = np.ones(n, bool)
        cat = df["category"].to_numpy()
        for u, idx in groups.items():
            rate = level if cat[idx[0]] in VRU else level * vehicle_switch_factor
            k = rng.poisson(rate)
            if k == 0 or len(idx) < 2 * min_track_frames:
                continue
            cuts = np.sort(rng.choice(np.arange(min_track_frames, len(idx) - min_track_frames),
                                      size=min(k, len(idx) - 2 * min_track_frames), replace=False))
            seg = np.searchsorted(cuts, np.arange(len(idx)), side="right")
            uid[idx] = [f"{u}#{s}" if s else u for s in seg]
            for c in cuts:                     # re-identification gap after each split
                keep[idx[c:min(c + gap, len(idx))]] = False
        df = df.assign(uid=uid)[keep]

    # Tracks shorter than a few frames would not survive a real tracker.
    sizes = df.groupby("uid")["uid"].transform("size")
    df = df[sizes >= min_track_frames]

    df = derive_kinematics(df)        # every condition, including 'baseline'
    if kind == "heading":
        df["heading"] = df["heading"] + np.deg2rad(rng.normal(0, level, len(df)))
    return df.sort_values(["tkey", "uid"]).reset_index(drop=True)
