"""Visual sanity checks of flagged conflict events.

Each figure shows, for one agent pair:
  left   plan view: both trajectories over +/- `window_s` seconds, footprints at the
         reference moment (solid), the constant-velocity projection to the predicted
         collision (dashed, TTC events), the conflict point (star, PET events), and all
         other trajectories in the recording in light grey, which outlines the lanes.
  right  actual edge-to-edge footprint gap and exact TTC over the same window.
The city map images are not used because their alignment with the track coordinates
is not documented.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .conflicts import road_user_group, ttc_conflicts
from .io_sind import find_recordings, load_recording
from .ssm import rect_corners, rect_gap, ttc_rect


def select_events(pair_ttc: pd.DataFrame, pair_pet: pd.DataFrame, n: int = 3,
                  ttc_threshold: float = 1.5, gap_threshold: float = 1.0,
                  seed: int = 0) -> pd.DataFrame:
    """A small, labelled set of events covering the cases worth checking by eye."""
    rng = np.random.default_rng(seed)
    conf = ttc_conflicts(pair_ttc, ttc_threshold, gap_threshold=gap_threshold)
    pick = []

    def take(frame, label, how="random", by=None):
        if frame.empty:
            return
        if how == "smallest":
            chosen = frame.nsmallest(n, by)
        else:
            chosen = frame.iloc[rng.choice(len(frame), size=min(n, len(frame)), replace=False)]
        pick.append(chosen.assign(category=label))

    vv = conf[conf["interaction"] == "vehicle-vehicle"]
    take(vv[vv["geometry"] == "opposite"], "VV opposite (closest gap)", "smallest", "min_gap")
    take(vv[vv["geometry"] == "same_direction"], "VV same direction (random)")
    take(vv[vv["geometry"] == "crossing"], "VV crossing (random)")
    pv = conf[conf["interaction"] == "pedestrian-vehicle"]
    cq = pv[pv["recording"] == "Chongqing"]
    take(cq if len(cq) else pv, "Pedestrian-vehicle TTC (Chongqing first)", "smallest", "min_gap")
    take(conf[conf["interaction"] == "two_wheeler-vehicle"], "Two-wheeler-vehicle TTC (random)")
    ttc_events = pd.concat(pick, ignore_index=True).assign(kind="ttc") if pick else pd.DataFrame()

    if "plausible" in pair_pet.columns:
        kept_pet = pair_pet[pair_pet["plausible"].astype(bool)]
    else:
        kept_pet = pair_pet
    pet_events = kept_pet.nsmallest(n, "pet").assign(kind="pet", category="Smallest PET")
    parts = [ttc_events, pet_events]
    if "plausible" in pair_ttc.columns:   # a few removed pairs, to check the rules by eye
        removed = ttc_conflicts(pair_ttc, ttc_threshold, gap_threshold=gap_threshold,
                                plausible_only=False)
        removed = removed[~removed["plausible"].astype(bool)]
        if len(removed):
            k = min(2, len(removed))
            parts.append(removed.iloc[rng.choice(len(removed), size=k, replace=False)]
                         .assign(kind="ttc", category="Removed as implausible (random)"))
    events = pd.concat(parts, ignore_index=True)
    events["group"] = road_user_group(events["interaction"])
    return events


def _window(track: pd.DataFrame, t0: float, w: float) -> pd.DataFrame:
    return track[(track["tkey"] >= t0 - w) & (track["tkey"] <= t0 + w)]


def _poly(ax, x, y, h, l, w, **kw):
    c = rect_corners(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(h),
                     np.atleast_1d(l), np.atleast_1d(w))[0]
    ax.fill(c[:, 0], c[:, 1], **kw)


def plot_event(df: pd.DataFrame, ev: pd.Series, path, window_s: float = 4.0):
    """Plot one event (a row from `select_events`) from its recording's trajectory table."""
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    w = window_s * 1000
    t0 = ev["tkey_at_min"] if ev["kind"] == "ttc" else 0.5 * (ev["tkey_first"] + ev["tkey_second"])
    A = _window(df[df["uid"] == ev["uid_a"]], t0, w).sort_values("tkey")
    B = _window(df[df["uid"] == ev["uid_b"]], t0, w).sort_values("tkey")
    if A.empty or B.empty:
        return False

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1)
    ax_gap, ax_ttc = fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 4)

    both = pd.concat([A, B])
    xmin, xmax = both["x"].min() - 12, both["x"].max() + 12
    ymin, ymax = both["y"].min() - 12, both["y"].max() + 12
    bg = df[(df["x"].between(xmin, xmax)) & (df["y"].between(ymin, ymax))].iloc[::5]
    ax.scatter(bg["x"], bg["y"], s=0.2, c="0.85", zorder=0, rasterized=True)

    colours = ("tab:blue", "tab:red")
    for tr, col, uid in ((A, colours[0], ev["uid_a"]), (B, colours[1], ev["uid_b"])):
        kind = tr["agent_type"].iloc[0]
        ax.plot(tr["x"], tr["y"], "-", c=col, lw=1.5, label=f"{uid} ({kind})")
        ax.plot(tr["x"].iloc[0], tr["y"].iloc[0], "o", c=col, ms=4)       # start of window
        ref = tr.iloc[np.argmin(np.abs(tr["tkey"].values - t0))]
        _poly(ax, ref["x"], ref["y"], ref["heading"], ref["length"], ref["width"],
              fc=col, alpha=0.45, ec=col, zorder=3)
        if ev["kind"] == "ttc" and np.isfinite(ev["min_ttc"]):
            tt = ev["min_ttc"]
            px, py = ref["x"] + ref["vx"] * tt, ref["y"] + ref["vy"] * tt
            ax.plot([ref["x"], px], [ref["y"], py], "--", c=col, lw=1)
            _poly(ax, px, py, ref["heading"], ref["length"], ref["width"],
                  fc="none", ec=col, ls="--", lw=1.2, zorder=3)
        if ev["kind"] == "pet":
            tp = ev["tkey_first"] if uid == ev["first"] else ev["tkey_second"]
            at = tr.iloc[np.argmin(np.abs(tr["tkey"].values - tp))]
            _poly(ax, at["x"], at["y"], at["heading"], at["length"], at["width"],
                  fc="none", ec=col, ls=":", lw=1.5, zorder=3)
    if ev["kind"] == "pet":
        ax.plot(ev["x"], ev["y"], "*", c="k", ms=14, zorder=4, label="conflict point")
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")

    # Time series on the frames both agents share.
    _, ia, ib = np.intersect1d(A["tkey"].values, B["tkey"].values, return_indices=True)
    a, b = A.iloc[ia], B.iloc[ib]
    ts = (a["tkey"].values - t0) / 1000.0
    gap = rect_gap(a["x"], a["y"], a["heading"], a["length"], a["width"],
                   b["x"], b["y"], b["heading"], b["length"], b["width"])
    ttc = ttc_rect(a["x"], a["y"], a["heading"], a["length"], a["width"], a["vx"], a["vy"],
                   b["x"], b["y"], b["heading"], b["length"], b["width"], b["vx"], b["vy"])
    ax_gap.plot(ts, gap, c="k")
    ax_gap.axhline(1.0, c="0.6", ls=":", lw=1)
    ax_gap.set(ylabel="Actual gap [m]", ylim=(0, max(3.0, np.nanmin([gap.max(), 15]))))
    ax_ttc.plot(ts, np.where(np.isfinite(ttc), ttc, np.nan), c="tab:purple")
    for thr in (1.5, 3.0):
        ax_ttc.axhline(thr, c="0.6", ls=":", lw=1)
    ax_ttc.set(ylabel="Exact TTC [s]", xlabel="Time relative to reference moment [s]",
               ylim=(0, 10))
    for axx in (ax_gap, ax_ttc):
        axx.axvline(0, c="0.4", lw=0.8)
        axx.set_xlim(-window_s, window_s)

    if ev["kind"] == "ttc":
        metric = (f"min TTC {ev['min_ttc']:.2f} s, actual gap {ev['min_gap']:.2f} m, "
                  f"rel. speed {ev['rel_speed_at_min']:.1f} m/s, angle {ev['angle_at_min']:.0f} deg")
        if "plausible" in ev and not bool(ev["plausible"]):
            why = "static agent" if bool(ev["static_pair"]) else f"overlap {ev['overlap_s']:.1f} s"
            metric += f" | removed: {why}"
    else:
        metric = f"PET {ev['pet']:.2f} s, angle {ev['angle']:.0f} deg"
    fig.suptitle(f"{ev['category']} | {ev['recording']} | {metric}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_selected_events(data_root, results_dir="results/01_conflict_check", n: int = 3,
                         window_s: float = 4.0, seed: int = 0) -> pd.DataFrame:
    """Select events from a finished conflict check and plot them into results_dir/events/."""
    results_dir = Path(results_dir)
    pair_ttc = pd.read_csv(results_dir / "pair_ttc.csv")
    pair_pet = pd.read_csv(results_dir / "pair_pet.csv")
    if "tkey_first" not in pair_pet.columns:
        raise ValueError("pair_pet.csv predates the PET timing columns; rerun the conflict check.")
    events = select_events(pair_ttc, pair_pet, n=n, seed=seed)
    out = results_dir / "events"
    out.mkdir(parents=True, exist_ok=True)
    recs = {p.name: p for p in find_recordings(data_root)}
    files = []
    for rec, evs in events.groupby("recording", sort=False):
        df = load_recording(recs[rec], verbose=False)
        for idx, ev in evs.iterrows():
            slug = ev["category"].split(" (")[0].replace(" ", "_").replace("-", "_").lower()
            path = out / f"{idx:02d}_{slug}_{rec}_{ev['uid_a']}_{ev['uid_b']}.png"
            files.append((idx, str(path) if plot_event(df, ev, path, window_s) else ""))
    events["figure"] = pd.Series(dict(files))
    events.to_csv(out / "index.csv", index=False)
    print(f"{(events['figure'] != '').sum()} event figures written to {out.resolve()}")
    return events
