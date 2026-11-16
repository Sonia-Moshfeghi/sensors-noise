"""Load SinD recordings into one harmonised trajectory table.

SinD releases differ slightly in column names, so columns are matched through
aliases and the mapping is printed on load. Check that printout on the first run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VEH_FILE = "Veh_smoothed_tracks.csv"
PED_FILE = "Ped_smoothed_tracks.csv"

# Fallback (length, width) in metres, used only where the files carry no size.
DEFAULT_DIMS = {
    "car": (4.5, 1.8),
    "bus": (12.0, 2.5),
    "truck": (8.0, 2.5),
    "motorcycle": (2.0, 0.8),
    "bicycle": (1.8, 0.6),
    "tricycle": (2.5, 1.0),
    "pedestrian": (0.5, 0.5),
}
TWO_WHEELERS = {"motorcycle", "bicycle", "tricycle"}

_ALIASES = {
    "track_id": ["track_id", "trackid", "id"],
    "frame": ["frame_id", "frame", "frameid"],
    "t_ms": ["timestamp_ms", "time_ms", "timestamp"],
    "agent_type": ["agent_type", "class", "type"],
    "x": ["x"],
    "y": ["y"],
    "vx": ["vx"],
    "vy": ["vy"],
    "heading": ["heading_rad", "yaw_rad", "heading", "yaw"],
    "length": ["length"],
    "width": ["width"],
}


def find_recordings(root: str | Path) -> list[Path]:
    """Every folder under `root` holding a SinD vehicle or pedestrian track file."""
    root = Path(root)
    dirs = {p.parent for name in (VEH_FILE, PED_FILE) for p in root.rglob(name)}
    return sorted(dirs)


def _harmonise(df: pd.DataFrame, source: str, verbose: bool) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    mapping = {}
    for target, options in _ALIASES.items():
        for opt in options:
            if opt in df.columns:
                mapping[opt] = target
                break
    df = df.rename(columns=mapping)[list(mapping.values())]
    if verbose:
        print(f"  [{source}] column mapping: {mapping}")
    for col in ("track_id", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"{source} file is missing required column '{col}'")
    if "t_ms" not in df.columns and "frame" not in df.columns:
        raise ValueError(f"{source} file has neither a timestamp nor a frame column")
    return df


def _fill_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity by finite differences and heading from velocity where missing."""
    df = df.sort_values(["uid", "t"]).reset_index(drop=True)
    if "vx" not in df.columns or "vy" not in df.columns:
        vx, vy = np.zeros(len(df)), np.zeros(len(df))
        for _, idx in df.groupby("uid").indices.items():
            if len(idx) > 1:
                t = df["t"].values[idx]
                vx[idx] = np.gradient(df["x"].values[idx], t)
                vy[idx] = np.gradient(df["y"].values[idx], t)
        df["vx"], df["vy"] = vx, vy

    if "heading" in df.columns:
        h = df["heading"].astype(float)
        if np.nanmax(np.abs(h.values)) > 2 * np.pi + 0.1:  # stored in degrees
            h = np.deg2rad(h)
        df["heading"] = h
    else:
        df["heading"] = np.nan
    speed = np.hypot(df["vx"], df["vy"])
    from_vel = np.where(speed > 0.5, np.arctan2(df["vy"], df["vx"]), np.nan)
    df["heading"] = df["heading"].fillna(pd.Series(from_vel, index=df.index))
    df["heading"] = df.groupby("uid")["heading"].transform(lambda s: s.ffill().bfill()).fillna(0.0)
    return df


def load_recording(rec_dir: str | Path, verbose: bool = True) -> pd.DataFrame:
    """One SinD recording as a single table of all agents.

    Columns: uid, source, agent_type, category, tkey, t [s], x, y [m],
    vx, vy [m/s], heading [rad], length, width [m].
    """
    rec_dir = Path(rec_dir)
    parts = []
    for fname, source in ((VEH_FILE, "veh"), (PED_FILE, "ped")):
        path = rec_dir / fname
        if not path.exists():
            continue
        df = _harmonise(pd.read_csv(path), source, verbose)
        df["source"] = source
        df["uid"] = source[0].upper() + "_" + df["track_id"].astype(str)
        if "agent_type" not in df.columns:
            df["agent_type"] = "pedestrian" if source == "ped" else "car"
        df["agent_type"] = df["agent_type"].astype(str).str.strip().str.lower()
        if source == "ped":
            df["agent_type"] = "pedestrian"
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No SinD track files in {rec_dir}")
    df = pd.concat(parts, ignore_index=True)

    # A shared time key across vehicle and pedestrian files.
    if "t_ms" in df.columns and df["t_ms"].notna().all():
        df["tkey"] = df["t_ms"].round().astype(np.int64)
        df["t"] = df["tkey"] / 1000.0
    else:
        raise ValueError(
            "No millisecond timestamps found; set them from the frame index and "
            "the recording frame rate before using this loader."
        )

    df["category"] = np.where(
        df["agent_type"] == "pedestrian",
        "pedestrian",
        np.where(df["agent_type"].isin(TWO_WHEELERS), "two_wheeler", "vehicle"),
    )

    for col, k in (("length", 0), ("width", 1)):
        default = df["agent_type"].map(lambda a: DEFAULT_DIMS.get(a, DEFAULT_DIMS["car"])[k])
        df[col] = df[col].astype(float).fillna(default) if col in df.columns else default
    ped = df["category"] == "pedestrian"
    df.loc[ped, ["length", "width"]] = DEFAULT_DIMS["pedestrian"]

    df = _fill_kinematics(df)
    if verbose:
        dt = np.median(np.diff(np.unique(df["tkey"]))) / 1000.0
        counts = df.groupby("category")["uid"].nunique().to_dict()
        print(f"  sampling interval ~{dt:.3f} s; agents per category: {counts}")
    cols = ["uid", "source", "agent_type", "category", "tkey", "t", "x", "y",
            "vx", "vy", "heading", "length", "width"]
    return df[cols].sort_values(["tkey", "uid"]).reset_index(drop=True)
