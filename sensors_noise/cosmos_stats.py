"""Missed-detection and track-break rates in the public COSMOS release (NYC).

Uses only counts and fractions, so the pixel coordinates of the release need no
conversion to metres. Each .npy file is one 8-s scene at 2.5 fps with columns
frame_id, track_id, label (0 = vehicle, 1 = pedestrian), pos_x, pos_y [pixels].

Measured per class:
  missed-frame fraction   frames missing inside tracks (between a track's first and last
                          observation in its scene); detections missing at the very start
                          or end of a track cannot be seen and are not counted
  interior breaks         tracks that start after the scene's first frame or end before its
                          last frame while away from the image edge, per track-minute. This
                          is an upper bound on fragmentation: real occlusions and people
                          getting into or out of vehicles also end tracks mid-image.
If no gaps are found at all, the release may have been gap-filled by interpolation and
the missed-frame fraction is reported as not measurable rather than as 0.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {0: "vehicle", 1: "pedestrian"}


def load_scenes(root, max_files=None) -> pd.DataFrame:
    """All scenes under `root` (searched recursively) as one table with a scene column."""
    # Scene files carry a numeric horizon suffix (..._l20_f2p5_51054441684.npy); the
    # combined split files (..._l20_f2p5_train.npy / _val / _test) are skipped so scenes
    # are not double-counted.
    files = sorted(f for f in Path(root).rglob("*.npy")
                   if not re.search(r"_(train|val|test)$", f.stem))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No .npy scene files under {root}")
    parts = []
    for i, f in enumerate(files):
        a = np.load(f, allow_pickle=False)
        if a.ndim != 2 or a.shape[1] < 5:
            continue
        parts.append(pd.DataFrame(a[:, :5], columns=["frame", "track", "label", "x", "y"])
                     .assign(scene=i))
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df["label"].round().astype(int).map(LABELS).fillna("other")
    return df


def frame_step(df: pd.DataFrame) -> float:
    """Most common spacing between consecutive frame ids within a scene."""
    d = df.drop_duplicates(["scene", "frame"]).sort_values(["scene", "frame"])
    diffs = d.groupby("scene")["frame"].diff().dropna()
    diffs = diffs[diffs > 0]
    return float(diffs.round(6).mode().iloc[0])


def cosmos_rates(df: pd.DataFrame, fps: float = 2.5, edge_margin: float = 0.08) -> pd.DataFrame:
    """Per-class missed-frame and interior-break statistics (see module docstring).

    edge_margin: fraction of the image extent (inferred from all positions) treated as
    the border zone, where tracks may legitimately start or end.
    """
    step = frame_step(df)
    xlo, xhi = df["x"].quantile([0.001, 0.999])
    ylo, yhi = df["y"].quantile([0.001, 0.999])
    mx, my = edge_margin * (xhi - xlo), edge_margin * (yhi - ylo)
    scene_first = df.groupby("scene")["frame"].transform("min")
    scene_last = df.groupby("scene")["frame"].transform("max")
    df = df.assign(scene_first=scene_first, scene_last=scene_last).sort_values(["scene", "track", "frame"])

    g = df.groupby(["scene", "track"], sort=False)
    t = g.agg(label=("label", "first"), first=("frame", "min"), last=("frame", "max"),
              n_obs=("frame", "size"), scene_first=("scene_first", "first"),
              scene_last=("scene_last", "first"))
    head, tail = g.head(1).set_index(["scene", "track"]), g.tail(1).set_index(["scene", "track"])
    interior = lambda p: (p["x"].between(xlo + mx, xhi - mx) & p["y"].between(ylo + my, yhi - my))
    t["span_frames"] = np.round((t["last"] - t["first"]) / step).astype(int) + 1
    t["missing"] = (t["span_frames"] - t["n_obs"]).clip(lower=0)
    t["interior_start"] = (t["first"] > t["scene_first"]) & interior(head).reindex(t.index).values
    t["interior_end"] = (t["last"] < t["scene_last"]) & interior(tail).reindex(t.index).values

    # Gap lengths (in frames) inside tracks.
    gaps = g["frame"].diff().dropna() / step - 1
    gaps = gaps[gaps > 0.5].round()
    gap_label = df.loc[gaps.index, "label"]

    rows = []
    for lab, s in t.groupby("label"):
        minutes = s["span_frames"].sum() / fps / 60.0
        gl = gaps[gap_label == lab]
        rows.append({
            "class": lab, "scenes": s.index.get_level_values(0).nunique(), "tracks": len(s),
            "track_minutes": round(minutes, 1),
            "missed_frame_fraction": s["missing"].sum() / s["span_frames"].sum(),
            "tracks_with_gap_share": (s["missing"] > 0).mean(),
            "mean_gap_s": gl.mean() / fps if len(gl) else np.nan,
            "interior_start_share": s["interior_start"].mean(),
            "interior_end_share": s["interior_end"].mean(),
            "interior_breaks_per_track_minute": (s["interior_start"].sum() + s["interior_end"].sum()) / 2
                                                / minutes if minutes else np.nan,
        })
    out = pd.DataFrame(rows).set_index("class")
    if (out["missed_frame_fraction"] == 0).all():
        out["missed_frame_fraction"] = np.nan        # gap-free: possibly interpolated
        out["note"] = "no gaps found: release may be gap-filled; missed-frame rate not measurable"
    out.attrs["frame_step"] = step
    return out


def sind_switch_equivalents(sind_root, levels=(0.5, 1.0, 3.0), vehicle_factor=1 / 3) -> pd.DataFrame:
    """Express the fragmentation levels as breaks per track-minute, using SinD's mean
    track durations, so they can be compared with the COSMOS interior-break rate.
    The grid applies the VRU rate to pedestrians and two-wheelers alike, so the VRU
    conversion uses the mean duration over both classes; durations per class are
    reported in the result's attrs."""
    from .io_sind import find_recordings, load_recording
    durs = []
    for rec in find_recordings(sind_root):
        df = load_recording(rec, verbose=False)
        d = df.groupby("uid").agg(cat=("category", "first"), t0=("t", "min"), t1=("t", "max"))
        durs.append(d.assign(minutes=(d["t1"] - d["t0"]) / 60.0))
    d = pd.concat(durs)
    mean_min = {c: float(d.loc[d["cat"] == c, "minutes"].mean()) for c in ("pedestrian", "two_wheeler", "vehicle")}
    mean_min["VRU"] = float(d.loc[d["cat"].isin(["pedestrian", "two_wheeler"]), "minutes"].mean())
    rows = []
    for lv in levels:
        rows.append({"id_switch_level": lv,
                     "VRU_breaks_per_track_minute": lv / mean_min["VRU"],
                     "pedestrian_breaks_per_track_minute": lv / mean_min["pedestrian"],
                     "vehicle_breaks_per_track_minute": lv * vehicle_factor / mean_min["vehicle"]})
    out = pd.DataFrame(rows).set_index("id_switch_level")
    out.attrs["mean_track_minutes"] = mean_min
    out.attrs["n_tracks"] = d["cat"].value_counts().to_dict()
    return out
