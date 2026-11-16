"""Publication figures regenerated from the results folders (print style: 300 dpi,
no internal titles, panel letters, larger fonts).

fig_events   2x2 plan views: two kept conflicts and two removed-as-implausible pairs
fig_noise    step-2 propagation: recall / precision / |error| vs level, one column per
             noise kind, one line per road-user group (split into two figures)
fig_filters  step-3 cleaning: the same layout with one line per cleaning method
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .io_sind import find_recordings, load_recording
from .noise import UNITS
from .ssm import rect_corners, rect_gap

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
         "lines.markersize": 3.5, "axes.spines.top": False, "axes.spines.right": False}
GROUP_LABELS = {"vehicle-vehicle": "vehicle–vehicle", "VRU-vehicle": "VRU–vehicle", "VRU-VRU": "VRU–VRU"}
METHOD_LABELS = {"none": "raw", "tracker": "tracker-like", "savgol": "Savitzky–Golay", "kalman": "Kalman–RTS",
                 "denoiser_jitter": "denoiser (jitter-only)", "denoiser_mixed": "denoiser (matched)",
                 "hybrid": "gated hybrid"}
KIND_LABELS = {"jitter": "jitter", "distance_jitter": "distance jitter", "drift": "drift",
               "offset": "offset", "heading": "heading", "dropout": "missed detections",
               "id_switch": "track fragmentation"}


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out / f"{name}.png"


# --------------------------------------------------------------------------------------
# Figure: example events
# --------------------------------------------------------------------------------------

def _poly(ax, row, **kw):
    c = rect_corners(np.array([row["x"]]), np.array([row["y"]]), np.array([row["heading"]]),
                     np.array([row["length"]]), np.array([row["width"]]))[0]
    ax.fill(c[:, 0], c[:, 1], **kw)


def _min_gap_moment(df, ev, window_s=None):
    """Frame at which the two footprints are closest over their whole co-presence."""
    t0 = ev["tkey_at_min"]
    A = df[df["uid"] == ev["uid_a"]].set_index("tkey")
    B = df[df["uid"] == ev["uid_b"]].set_index("tkey")
    common = A.index.intersection(B.index)
    if len(common) == 0:
        return t0
    a, b = A.loc[common], B.loc[common]
    g = rect_gap(a["x"], a["y"], a["heading"], a["length"], a["width"],
                 b["x"], b["y"], b["heading"], b["length"], b["width"])
    return int(common[np.argmin(g)])


def event_panel(ax, df, ev, window_s=4.0, letter="", label="", show_projection=True,
                at_min_gap=False):
    """Plan view of one pair around its reference moment, print style. With
    at_min_gap the footprints are drawn at the moment the two are closest."""
    w = window_s * 1000
    t0 = _min_gap_moment(df, ev, window_s) if at_min_gap else ev["tkey_at_min"]
    A = df[(df["uid"] == ev["uid_a"]) & df["tkey"].between(t0 - w, t0 + w)].sort_values("tkey")
    B = df[(df["uid"] == ev["uid_b"]) & df["tkey"].between(t0 - w, t0 + w)].sort_values("tkey")
    both = pd.concat([A, B])
    cx, cy = both["x"].mean(), both["y"].mean()
    half = max(both["x"].max() - both["x"].min(), both["y"].max() - both["y"].min()) / 2 + 8
    xl, xh, yl, yh = cx - half, cx + half, cy - half, cy + half
    # Other road users as light lines (within the window, inside the box).
    bg = df[df["tkey"].between(t0 - w, t0 + w) & df["x"].between(xl, xh) & df["y"].between(yl, yh)]
    for uid, g in bg.groupby("uid"):
        if uid in (ev["uid_a"], ev["uid_b"]):
            continue
        ax.plot(g["x"], g["y"], "-", c="0.8", lw=0.6, zorder=0)
    for tr, col in ((A, "tab:blue"), (B, "tab:red")):
        ax.plot(tr["x"], tr["y"], "-", c=col, lw=1.4, zorder=2)
        ax.plot(tr["x"].iloc[0], tr["y"].iloc[0], "o", c=col, ms=3, zorder=2)
        ref = tr.iloc[np.argmin(np.abs(tr["tkey"].values - t0))]
        _poly(ax, ref, fc=col, alpha=0.4, ec=col, lw=0.8, zorder=3)
        if show_projection and np.isfinite(ev.get("min_ttc", np.nan)) and ev["min_ttc"] > 0:
            tt = min(ev["min_ttc"], 3.0)
            px, py = ref["x"] + ref["vx"] * tt, ref["y"] + ref["vy"] * tt
            ax.plot([ref["x"], px], [ref["y"], py], "--", c=col, lw=0.8, zorder=3)
            _poly(ax, {**ref, "x": px, "y": py}, fc="none", ec=col, ls="--", lw=0.8, zorder=3)
    ax.set_xlim(xl, xh); ax.set_ylim(yl, yh); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    # 5-m scale bar
    ax.plot([xl + 2, xl + 7], [yl + 2, yl + 2], "k-", lw=1.5)
    ax.text(xl + 4.5, yl + 3, "5 m", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"({letter}) {label}", loc="left", fontsize=8, fontweight="bold", pad=3)


def _first(frames, by, smallest=True):
    """First non-empty candidate frame, picked by the extreme of `by`."""
    for f in frames:
        if len(f):
            return (f.nsmallest(1, by) if smallest else f.nlargest(1, by)).iloc[0]
    return None


def _pick_events(results_dir, overrides=None):
    """Two kept and two removed pairs from the step-1 results (with fallbacks).

    overrides: optional dict {letter: (recording, uid_a, uid_b)} naming the pair for a
    panel explicitly, e.g. {"a": ("Changchun", "V_1224", "V_1253")}."""
    p = pd.read_csv(Path(results_dir) / "pair_ttc.csv")
    conf = (p["min_ttc"] < 1.5) & (p["min_gap"] <= 1.0) & (p["frames_below_1.5"] >= 3)
    kept, removed = p[p["plausible"].astype(bool) & conf], p[~p["plausible"].astype(bool) & conf]
    kept = kept[kept["min_ttc"] > 0.1] if (kept["min_ttc"] > 0.1).any() else kept
    vv = kept[kept["interaction"] == "vehicle-vehicle"]
    opp = vv[vv["geometry"] == "opposite"].sort_values("min_gap")
    sel = [
        (opp if len(opp) else vv.sort_values("min_gap"), "kept: opposing left turn"),
        (_first([kept[kept["interaction"] == "pedestrian-vehicle"],
                 kept[kept["interaction"].str.contains("vehicle") & (kept["interaction"] != "vehicle-vehicle")]],
                "min_gap"), "kept: turning vehicle and pedestrian"),
        (removed[~removed["static_pair"].astype(bool) & (removed["interaction"] == "vehicle-vehicle")]
             .sort_values("overlap_s", ascending=False),
         "removed: sustained footprint overlap"),
        (_first([removed[removed["static_pair"].astype(bool) & (removed["recording"] == "Chongqing")
                         & (removed["interaction"] == "pedestrian-vehicle")],
                 removed[removed["static_pair"].astype(bool)]], "min_gap"),
         "removed: pedestrian at a parked car"),
    ]
    out = []
    for letter, (ev, lab) in zip("abcd", sel):
        if overrides and letter in overrides:
            rec, ua, ub = overrides[letter]
            m = p[(p["recording"] == rec) & (((p["uid_a"] == ua) & (p["uid_b"] == ub))
                                              | ((p["uid_a"] == ub) & (p["uid_b"] == ua)))]
            if len(m):
                ev = m.iloc[0]
        if ev is not None:
            out.append((ev, lab))
    return out


def _speed_at(df, uid, tkey):
    g = df[df["uid"] == uid]
    if g.empty:
        return 0.0
    r = g.iloc[np.argmin(np.abs(g["tkey"].values - tkey))]
    return float(np.hypot(r["vx"], r["vy"]))


def _resolve_candidates(sel, data, min_speed=1.5):
    """Panels (a) and (c) hold ranked candidate tables; take the first pair in which both
    vehicles are moving (own speed >= min_speed at the drawn moment)."""
    out = []
    for ev, lab in sel:
        if isinstance(ev, pd.DataFrame):
            chosen = None
            for _, row in ev.iterrows():
                df = data(row["recording"])
                t = _min_gap_moment(df, row) if lab.startswith("removed") else row["tkey_at_min"]
                if min(_speed_at(df, row["uid_a"], t), _speed_at(df, row["uid_b"], t)) >= min_speed:
                    chosen = row; break
            ev = chosen if chosen is not None else (ev.iloc[0] if len(ev) else None)
        if ev is not None:
            out.append((ev, lab))
    return out


def fig_events(results_dir, data_root, out, name="fig_events", overrides=None):
    sel = _pick_events(results_dir, overrides)
    recs = {r.name: r for r in find_recordings(data_root)}
    cache = {}

    def data(rec):
        if rec not in cache:
            cache[rec] = load_recording(recs[rec], verbose=False)
        return cache[rec]

    sel = _resolve_candidates(sel, data)
    for ev, lab in sel:   # report the gap at the moment actually drawn
        df = data(ev["recording"])
        t_draw = _min_gap_moment(df, ev) if lab.startswith("removed") else ev["tkey_at_min"]
        a = df[(df["uid"] == ev["uid_a"]) & (df["tkey"] == t_draw)]
        b = df[(df["uid"] == ev["uid_b"]) & (df["tkey"] == t_draw)]
        g = (rect_gap(a["x"], a["y"], a["heading"], a["length"], a["width"],
                      b["x"], b["y"], b["heading"], b["length"], b["width"])[0]
             if len(a) and len(b) else np.nan)
        print(f"   {lab}: {ev['recording']} {ev['uid_a']}/{ev['uid_b']} min TTC {ev['min_ttc']:.2f} s, "
              f"realised gap {ev['min_gap']:.2f} m, overlap {ev['overlap_s']:.1f} s, "
              f"gap at drawn moment {g:.2f} m")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))
        for ax in axes.ravel():
            ax.set_visible(False)
        for ax, (ev, label), letter in zip(axes.ravel(), sel, "abcd"):
            ax.set_visible(True)
            rec = ev["recording"]
            event_panel(ax, data(rec), ev, letter=letter, label=f"{label}\n{rec}",
                        show_projection=(letter in "ab"), at_min_gap=(letter in "cd"))
        fig.subplots_adjust(wspace=0.04, hspace=0.18)
        path = _save(fig, out, name)
    return path, sel


# --------------------------------------------------------------------------------------
# Figures: propagation and cleaning
# --------------------------------------------------------------------------------------

def _metric_grid(det, err, kinds, measure, line_col, line_labels, out, name, line_order=None):
    """3 rows (recall, precision, median |error|) x len(kinds) columns."""
    lines = line_order or sorted(det[line_col].unique())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, len(kinds), figsize=(1.75 * len(kinds) + 0.6, 5.2),
                                 squeeze=False, sharey="row")
        for j, kind in enumerate(kinds):
            for ln in lines:
                d = det[(det["kind"] == kind) & (det["measure"] == measure) & (det[line_col] == ln)]
                e = err[(err["kind"] == kind) & (err["measure"] == measure) & (err[line_col] == ln)]
                lab = line_labels.get(ln, ln)
                axes[0, j].plot(d["level"], d["recall"], "o-", label=lab)
                axes[1, j].plot(d["level"], d["precision"], "o-", label=lab)
                axes[2, j].plot(e["level"], e["median_abs_err_s"], "o-", label=lab)
            axes[0, j].set_title(KIND_LABELS.get(kind, kind))
            axes[2, j].set_xlabel(f"level [{UNITS[kind]}]")
            for i in range(3):
                axes[i, j].grid(True, lw=0.3, alpha=0.5)
        axes[0, 0].set_ylabel("recall")
        axes[1, 0].set_ylabel("precision")
        axes[2, 0].set_ylabel("median |error| [s]")
        for ax in axes[:2].ravel():
            ax.set_ylim(0, 1.03)
        for j, kind in enumerate(kinds):
            hi = det[det["kind"] == kind]["level"].max()
            if np.isfinite(hi) and hi > 0:
                for i in range(3):
                    axes[i, j].set_xlim(0, hi * 1.08)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False,
                   bbox_to_anchor=(0.5, -0.03))
        fig.subplots_adjust(wspace=0.12, hspace=0.35)
        return _save(fig, out, name)


def fig_noise(results_dir, out, measure="TTC<3.0s", tag="ttc"):
    """Two figures: position noise (4 kinds) and detection noise (3 kinds)."""
    det = pd.read_csv(Path(results_dir) / "detection.csv")
    err = pd.read_csv(Path(results_dir) / "errors.csv")
    det = det[det["kind"] != "baseline"]; err = err[err["kind"] != "baseline"]
    groups = ["vehicle-vehicle", "VRU-vehicle", "VRU-VRU"]
    a = _metric_grid(det, err, ["jitter", "distance_jitter", "drift", "offset"], measure,
                     "group", GROUP_LABELS, out, f"fig_noise_{tag}_position", groups)
    b = _metric_grid(det, err, ["heading", "dropout", "id_switch"], measure,
                     "group", GROUP_LABELS, out, f"fig_noise_{tag}_detection", groups)
    return a, b


def fig_filters(results_dir, out, measure="TTC<3.0s", tag="ttc"):
    """Two figures: position noise (jitter, distance jitter, drift) and offset + missed."""
    det = pd.read_csv(Path(results_dir) / "detection_pooled.csv")
    err = pd.read_csv(Path(results_dir) / "errors_pooled.csv")
    det = det[det["kind"] != "clean"]; err = err[err["kind"] != "clean"]
    methods = ["none", "tracker", "savgol", "kalman", "denoiser_jitter", "denoiser_mixed", "hybrid"]
    a = _metric_grid(det, err, ["jitter", "distance_jitter", "drift"], measure,
                     "method", METHOD_LABELS, out, f"fig_filters_{tag}_position", methods)
    b = _metric_grid(det, err, ["offset", "dropout"], measure,
                     "method", METHOD_LABELS, out, f"fig_filters_{tag}_other", methods)
    return a, b
