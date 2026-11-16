"""Position-cleaning methods applied to noisy trajectory tables (step 3).

Every method only cleans positions and fills missed frames; velocity and heading are
then re-derived with the same `derive_kinematics` used for the step-2 reference, so
methods are compared on position cleaning alone, not on velocity estimation.

Methods: 'none' (raw differentiation of the noisy positions, no gap filling), 'tracker'
(causal constant-velocity Kalman filter with gap filling by prediction, as carried by
trackers such as BoT-SORT: the condition an agency would actually receive), 'savgol',
'kalman' (forward filter + RTS smoother), the learned denoisers in `denoiser.py`, and
'hybrid' (per-track gate on the estimated noise level, see `hybrid_clean`).
Classical parameters can be tuned on training-seed noise with `tune_classical`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .noise import derive_kinematics

PER_AGENT = ["category", "agent_type", "length", "width", "orig_uid"]

# Kalman process noise: white-acceleration spectral density [m^2/s^3] per category.
# Fixed values, not tuned to the noise levels being tested.
KALMAN_Q = {"vehicle": 2.0, "two_wheeler": 1.0, "pedestrian": 0.5}


def fill_gaps(df: pd.DataFrame, dt_ms: int | None = None) -> pd.DataFrame:
    """Put every track on the recording's frame times from its first to last frame.

    The grid is the set of frame times actually present in the recording (not a
    synthetic constant-step grid: video frame times drift by a millisecond or two, so a
    synthetic grid would not line up between road users). Missed frames get linearly
    interpolated positions and observed = False; heading is carried from the nearest
    observed frame. Per-agent attributes are copied.
    """
    df = df.sort_values(["uid", "tkey"])
    if "orig_uid" not in df.columns:
        df = df.assign(orig_uid=df["uid"])
    frames = np.unique(df["tkey"].to_numpy())
    attrs = df.groupby("uid", sort=False)[PER_AGENT].first()
    out = []
    for uid, g in df.groupby("uid", sort=False):
        tk = g["tkey"].to_numpy()
        i0, i1 = np.searchsorted(frames, tk[0]), np.searchsorted(frames, tk[-1])
        grid = frames[i0:i1 + 1]
        obs = np.isin(grid, tk)
        x = np.interp(grid, tk, g["x"].to_numpy())
        y = np.interp(grid, tk, g["y"].to_numpy())
        h = g["heading"].to_numpy()[np.clip(np.searchsorted(tk, grid), 0, len(tk) - 1)]
        out.append(pd.DataFrame({"uid": uid, "tkey": grid, "x": x, "y": y, "heading": h,
                                 "observed": obs}))
    res = pd.concat(out, ignore_index=True)
    return res.join(attrs, on="uid")


def _finish(grid: pd.DataFrame) -> pd.DataFrame:
    """Common tail of every method: same kinematics derivation as the reference."""
    cols = ["uid", "orig_uid", "category", "agent_type", "tkey", "x", "y", "heading",
            "length", "width"]
    d = derive_kinematics(grid[cols])
    d["t"] = d["tkey"] / 1000.0
    return d.sort_values(["tkey", "uid"]).reset_index(drop=True)


def savgol_clean(df: pd.DataFrame, window_s: float = 1.1, polyorder: int = 2) -> pd.DataFrame:
    """Savitzky-Golay smoothing of positions (after gap filling), per track."""
    g = fill_gaps(df)
    dt = np.median(np.diff(np.unique(g["tkey"]))) / 1000.0
    w = max(polyorder + 2, int(round(window_s / dt)) | 1)
    x, y = g["x"].to_numpy().copy(), g["y"].to_numpy().copy()
    for idx in g.groupby("uid", sort=False).indices.values():
        n = len(idx)
        wn = min(w, n if n % 2 else n - 1)
        if wn > polyorder + 1:
            x[idx] = savgol_filter(x[idx], wn, polyorder)
            y[idx] = savgol_filter(y[idx], wn, polyorder)
    return _finish(g.assign(x=x, y=y))


def _measurement_sigma(z: np.ndarray, obs: np.ndarray) -> float:
    """Robust per-track measurement noise from second differences of consecutive
    observed positions: for white noise sigma on near-constant-velocity motion,
    var(second difference) ~= 6 sigma^2."""
    ok = obs[2:] & obs[1:-1] & obs[:-2]
    if ok.sum() < 5:
        return 0.1
    d2 = (z[2:] - 2 * z[1:-1] + z[:-2])[ok]
    mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0) / 0.6745
    return float(max(np.sqrt(np.mean(mad ** 2) / 6.0), 0.02))


def _rts_track(z: np.ndarray, obs: np.ndarray, dt: float, q: float, r: float) -> np.ndarray:
    """Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother, x and y treated
    as independent axes sharing one covariance. z: (n, 2) measurements."""
    n = len(z)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    xs = np.zeros((n, 2, 2))          # state [pos, vel] for each axis (columns)
    Ps = np.zeros((n, 2, 2))
    xp_all = np.zeros((n, 2, 2))
    Pp_all = np.zeros((n, 2, 2))
    x = np.vstack([z[0], np.zeros(2)])
    P = np.diag([r ** 2, 25.0])
    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xp_all[k], Pp_all[k] = x, P
        if obs[k]:
            S = P[0, 0] + r ** 2
            K = P[:, 0] / S
            x = x + np.outer(K, z[k] - x[0])
            P = P - np.outer(K, P[0, :])
        xs[k], Ps[k] = x, P
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ F.T @ np.linalg.inv(Pp_all[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp_all[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp_all[k + 1]) @ C.T
    return xs[:, 0, :]


def _kf_forward_track(z: np.ndarray, obs: np.ndarray, dt: float, q: float, r: float) -> np.ndarray:
    """Causal constant-velocity Kalman filter (no smoothing); missed frames are
    filled by the prediction step, as a tracker does."""
    n = len(z)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    out = np.zeros((n, 2))
    x = np.vstack([z[0], np.zeros(2)])
    P = np.diag([r ** 2, 25.0])
    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        if obs[k]:
            S = P[0, 0] + r ** 2
            K = P[:, 0] / S
            x = x + np.outer(K, z[k] - x[0])
            P = P - np.outer(K, P[0, :])
        out[k] = x[0]
    return out


def _run_kf(df: pd.DataFrame, q_scale: float, smooth: bool, r_fixed: float | None = None) -> pd.DataFrame:
    g = fill_gaps(df)
    dt = np.median(np.diff(np.unique(g["tkey"]))) / 1000.0
    x, y = g["x"].to_numpy().copy(), g["y"].to_numpy().copy()
    obs = g["observed"].to_numpy()
    cat = g["category"].to_numpy()
    for idx in g.groupby("uid", sort=False).indices.values():
        if len(idx) < 3:
            continue
        z = np.column_stack([x[idx], y[idx]])
        r = r_fixed if r_fixed is not None else _measurement_sigma(z, obs[idx])
        q = q_scale * KALMAN_Q.get(cat[idx[0]], 1.0)
        sm = (_rts_track if smooth else _kf_forward_track)(z, obs[idx], dt, q, r)
        x[idx], y[idx] = sm[:, 0], sm[:, 1]
    return _finish(g.assign(x=x, y=y))


def kalman_clean(df: pd.DataFrame, q_scale: float = 1.0) -> pd.DataFrame:
    """Constant-velocity Kalman smoother (forward filter + RTS) on positions. The
    measurement noise is estimated per track from the data, never from the known noise
    level; process noise is KALMAN_Q per category times `q_scale`."""
    return _run_kf(df, q_scale, smooth=True)


def tracker_clean(df: pd.DataFrame, q_scale: float = 1.0, r_fixed: float = 0.3) -> pd.DataFrame:
    """Tracker-like condition: causal constant-velocity Kalman filtering of the
    positions with gap filling by prediction and no smoothing, i.e. the kind of output a
    detection-and-tracking pipeline (ByteTrack, BoT-SORT) delivers. Fully causal: the
    measurement noise is a fixed value `r_fixed` [m], as in such trackers, not estimated
    from the track."""
    return _run_kf(df, q_scale, smooth=False, r_fixed=r_fixed)


def track_noise_estimates(df: pd.DataFrame) -> pd.Series:
    """Per-track measurement-noise estimate (sigma, metres) from second differences of
    the observed positions, the same estimator the Kalman methods use."""
    g = fill_gaps(df)
    x, y, obs = g["x"].to_numpy(), g["y"].to_numpy(), g["observed"].to_numpy()
    out = {}
    for uid, idx in g.groupby("uid", sort=False).indices.items():
        out[uid] = _measurement_sigma(np.column_stack([x[idx], y[idx]]), obs[idx]) if len(idx) >= 3 else 0.0
    return pd.Series(out)


def hybrid_clean(df: pd.DataFrame, denoiser, low: float = 0.05, high: float = 0.25,
                 q_scale: float = 1.0) -> pd.DataFrame:
    """Gated cleaning: each track is routed by its estimated noise level sigma_hat to
    no cleaning (sigma_hat < low; positions kept, gaps interpolated), the Kalman
    smoother (low <= sigma_hat < high) or the learned denoiser (sigma_hat >= high).
    `denoiser` is a callable df -> cleaned df (e.g. the noise-aware model)."""
    sig = track_noise_estimates(df)
    raw = _finish(fill_gaps(df))
    kal = kalman_clean(df, q_scale)
    dn = denoiser(df)
    route = pd.Series(np.where(sig < low, "none", np.where(sig < high, "kalman", "denoiser")),
                      index=sig.index)
    parts = [raw[raw["uid"].isin(route.index[route == "none"])],
             kal[kal["uid"].isin(route.index[route == "kalman"])],
             dn[dn["uid"].isin(route.index[route == "denoiser"])]]
    out = pd.concat(parts, ignore_index=True).sort_values(["tkey", "uid"]).reset_index(drop=True)
    out.attrs["route_counts"] = route.value_counts().to_dict()
    return out
