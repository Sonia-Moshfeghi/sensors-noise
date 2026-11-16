"""Find candidate conflicts in a trajectory table and summarise them."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .ssm import (disc_centres, disc_geometry, geometry_class,
                  heading_difference_deg, interaction_class, rect_gap, ttc_multidisc,
                  ttc_rect)


def scan_ttc(df: pd.DataFrame, max_dist: float = 30.0, ttc_cap: float = 10.0,
             frame_stride: int = 1, min_rel_speed: float = 1.0,
             refine: bool = True) -> pd.DataFrame:
    """Frame-level TTC for every closing agent pair (pedestrian pairs skipped).

    Pairs closing slower than `min_rel_speed` [m/s] are skipped, which removes
    near-stationary queue pairs whose TTC is driven by position jitter. Candidates
    are screened with the disc TTC and, if `refine`, recomputed exactly with
    oriented rectangles (see ssm.ttc_rect). Returns one row per (frame, pair) with
    TTC <= ttc_cap; `ttc_disc` keeps the screening value for diagnostics.
    """
    df = df.sort_values(["tkey", "uid"]).reset_index(drop=True)
    is_ped = (df["category"] == "pedestrian").values
    n, r = disc_geometry(df["length"].values, df["width"].values, is_ped)
    centres = disc_centres(df["x"].values, df["y"].values, df["heading"].values,
                           df["length"].values, n)
    xy = df[["x", "y"]].values
    vel = df[["vx", "vy"]].values
    head = df["heading"].values
    L, W = df["length"].values, df["width"].values
    tk = df["tkey"].values
    uid_codes, uid_names = pd.factorize(df["uid"])
    cat = df["category"].values

    keys, starts = np.unique(tk, return_index=True)
    ends = np.append(starts[1:], len(df))
    I, J, T, D, S = [], [], [], [], []
    for f in range(0, len(keys), frame_stride):
        s, e = starts[f], ends[f]
        m = e - s
        if m < 2:
            continue
        iu, ju = np.triu_indices(m, 1)
        iu, ju = iu + s, ju + s
        dp = xy[ju] - xy[iu]
        dv = vel[ju] - vel[iu]
        dist = np.hypot(dp[:, 0], dp[:, 1])
        rel_speed = np.hypot(dv[:, 0], dv[:, 1])
        keep = ((dist < max_dist) & (np.sum(dp * dv, axis=1) < 0)
                & (rel_speed >= min_rel_speed) & ~(is_ped[iu] & is_ped[ju]))
        if not keep.any():
            continue
        iu, ju, dist, rel_speed = iu[keep], ju[keep], dist[keep], rel_speed[keep]
        ttc = ttc_multidisc(centres[iu], centres[ju], vel[iu], vel[ju], r[iu] + r[ju])
        ok = ttc <= ttc_cap
        if ok.any():
            I.append(iu[ok]); J.append(ju[ok]); T.append(ttc[ok])
            D.append(dist[ok]); S.append(rel_speed[ok])
    cols = ["tkey", "uid_a", "uid_b", "ttc", "ttc_disc", "dist", "rel_speed", "angle", "interaction"]
    if not I:
        return pd.DataFrame(columns=cols)
    iu, ju, ttc_disc = np.concatenate(I), np.concatenate(J), np.concatenate(T)
    dist, rel_speed = np.concatenate(D), np.concatenate(S)

    if refine:
        ttc = np.empty_like(ttc_disc)
        for c0 in range(0, len(iu), 100_000):     # chunked to bound memory
            sl = slice(c0, c0 + 100_000)
            a, b = iu[sl], ju[sl]
            ttc[sl] = ttc_rect(xy[a, 0], xy[a, 1], head[a], L[a], W[a], vel[a, 0], vel[a, 1],
                               xy[b, 0], xy[b, 1], head[b], L[b], W[b], vel[b, 0], vel[b, 1],
                               t_start=ttc_disc[sl], t_cap=ttc_cap)
    else:
        ttc = ttc_disc
    ok = ttc <= ttc_cap
    iu, ju = iu[ok], ju[ok]
    a, b = uid_codes[iu], uid_codes[ju]
    return pd.DataFrame({
        "tkey": tk[iu],
        "uid_a": uid_names[np.minimum(a, b)],
        "uid_b": uid_names[np.maximum(a, b)],
        "ttc": ttc[ok],
        "ttc_disc": ttc_disc[ok],
        "dist": dist[ok],
        "rel_speed": rel_speed[ok],
        "angle": heading_difference_deg(head[iu], head[ju]),
        "interaction": interaction_class(cat[iu], cat[ju]),
    })[cols]


def aggregate_ttc(frames: pd.DataFrame, thresholds=(1.5, 3.0)) -> pd.DataFrame:
    """One row per agent pair: minimum TTC, its context, and frames below each threshold."""
    if frames.empty:
        return pd.DataFrame()
    frames = frames.sort_values("ttc")
    pair = frames.groupby(["uid_a", "uid_b"], sort=False)
    res = pair.first().rename(columns={"ttc": "min_ttc", "tkey": "tkey_at_min",
                                       "dist": "dist_at_min", "angle": "angle_at_min",
                                       "rel_speed": "rel_speed_at_min",
                                       "ttc_disc": "ttc_disc_at_min"})
    res["n_frames"] = pair.size().reindex(res.index).astype(int)   # frames the pair was screened in
    for thr in thresholds:
        below = (frames["ttc"] < thr).groupby([frames["uid_a"], frames["uid_b"]]).sum()
        res[f"frames_below_{thr}"] = below.reindex(res.index).astype(int)
    res["geometry"] = geometry_class(res["angle_at_min"].values)
    return res.reset_index()


def motion_heading(df: pd.DataFrame, min_speed: float = 0.5) -> np.ndarray:
    """Direction of motion from velocity where the agent moves faster than
    `min_speed`, else the stored heading. Used for the PET crossing filter so that
    footprint-heading noise does not enter PET."""
    sp = np.hypot(df["vx"].to_numpy(), df["vy"].to_numpy())
    return np.where(sp > min_speed, np.arctan2(df["vy"].to_numpy(), df["vx"].to_numpy()),
                    df["heading"].to_numpy())


def _zone_times(t, d, p, r):
    """Entry and exit times of the disc of radius r around the conflict point for one
    agent, by linear interpolation of its distance series d(t) around passage index p."""
    n = len(t)
    q = p
    while q + 1 < n and d[q + 1] <= r:
        q += 1
    if q + 1 < n:
        t_exit = t[q] + (r - d[q]) / (d[q + 1] - d[q]) * (t[q + 1] - t[q])
    else:
        t_exit = t[n - 1]
    q = p
    while q - 1 >= 0 and d[q - 1] <= r:
        q -= 1
    if q - 1 >= 0:
        t_entry = t[q] - (r - d[q]) / (d[q - 1] - d[q]) * (t[q] - t[q - 1])
    else:
        t_entry = t[0]
    return t_entry, t_exit


def scan_pet(df: pd.DataFrame, pet_max: float = 10.0, sample_hz: float | None = None,
             crossing_range=(30.0, 150.0), extra_radius: float = 0.0) -> pd.DataFrame:
    """Size-aware PET for crossing agent pairs, at the native sampling rate with
    sub-sample interpolation of the conflict-zone entry and exit times.

    For a pair of tracks, the sample points of the two agents that lie within
    r = 0.5*(width_a + width_b) + extra_radius of each other are found with a spatial
    index; only point pairs whose directions of motion differ by an angle in
    `crossing_range` are kept (excludes car following), and the pair with the smallest
    time difference defines the conflict point C (midpoint of the two sample points).
    PET is the time from the first agent's footprint reference point leaving the disc
    of radius r around C to the second agent's entering it, with both times obtained by
    linear interpolation of each agent's distance to C between samples; simultaneous
    occupancy gives PET = 0. `sample_hz` (default: native rate) can subsample for speed.
    """
    df = df.sort_values("t")
    df = df.assign(_mh=motion_heading(df))
    groups = {u: g for u, g in df.groupby("uid")}
    native_dt = np.median(np.diff(np.unique(df["t"].values)))
    stride = 1 if sample_hz is None else max(1, int(round((1.0 / sample_hz) / native_dt)))

    uids = list(groups)
    tr = []
    for u in uids:
        g = groups[u].iloc[::stride]
        tr.append({
            "t": g["t"].values, "xy": g[["x", "y"]].values, "h": g["_mh"].values,
            "w": float(g["width"].iloc[0]), "cat": g["category"].iloc[0],
        })
    t0 = np.array([d["t"][0] for d in tr])
    t1 = np.array([d["t"][-1] for d in tr])
    lo = np.array([d["xy"].min(0) for d in tr])
    hi = np.array([d["xy"].max(0) for d in tr])
    ped = np.array([d["cat"] == "pedestrian" for d in tr])
    trees: dict[int, cKDTree] = {}

    def tree(k):
        if k not in trees:
            trees[k] = cKDTree(tr[k]["xy"])
        return trees[k]

    rows = []
    for i in range(len(tr)):
        j = np.arange(i + 1, len(tr))
        cand = j[(t0[j] <= t1[i] + pet_max) & (t1[j] >= t0[i] - pet_max)
                 & np.all(lo[j] <= hi[i] + 3, axis=1) & np.all(hi[j] >= lo[i] - 3, axis=1)
                 & ~(ped[i] & ped[j])]
        for k in cand:
            radius = 0.5 * (tr[i]["w"] + tr[k]["w"]) + extra_radius
            sdm = tree(i).sparse_distance_matrix(tree(k), radius, output_type="ndarray")
            if len(sdm) == 0:
                continue
            a, b = sdm["i"], sdm["j"]
            ang = heading_difference_deg(tr[i]["h"][a], tr[k]["h"][b])
            ok = (ang >= crossing_range[0]) & (ang <= crossing_range[1])
            if not ok.any():
                continue
            a, b, ang = a[ok], b[ok], ang[ok]
            gap = tr[k]["t"][b] - tr[i]["t"][a]
            best = np.argmin(np.abs(gap))
            if abs(gap[best]) > pet_max:
                continue
            pa, pb = a[best], b[best]
            C = 0.5 * (tr[i]["xy"][pa] + tr[k]["xy"][pb])
            di = np.hypot(*(tr[i]["xy"] - C).T)
            dk = np.hypot(*(tr[k]["xy"] - C).T)
            ei, xi = _zone_times(tr[i]["t"], di, pa, radius)
            ek, xk = _zone_times(tr[k]["t"], dk, pb, radius)
            first_i = tr[i]["t"][pa] <= tr[k]["t"][pb]
            pet = (ek - xi) if first_i else (ei - xk)
            pet = float(max(pet, 0.0))
            if pet > pet_max:
                continue
            ua, ub = sorted((uids[i], uids[k]))
            t_i, t_k = tr[i]["t"][pa], tr[k]["t"][pb]
            rows.append({
                "uid_a": ua, "uid_b": ub, "pet": pet,
                "tkey_first": int(round(1000 * min(t_i, t_k))),
                "tkey_second": int(round(1000 * max(t_i, t_k))),
                "x": float(C[0]), "y": float(C[1]),
                "angle": float(ang[best]),
                "first": uids[i] if first_i else uids[k],
                "interaction": str(interaction_class([tr[i]["cat"]], [tr[k]["cat"]])[0]),
            })
    cols = ["uid_a", "uid_b", "pet", "tkey_first", "tkey_second", "x", "y", "angle",
            "first", "interaction"]
    return pd.DataFrame(rows, columns=cols)


def realized_gap(df: pd.DataFrame, pair_ttc: pd.DataFrame,
                 horizon_s: float = 3.0) -> pd.DataFrame:
    """Add `min_gap`: the smallest edge-to-edge gap [m] between the two agents'
    footprints over the `horizon_s` seconds after their minimum-TTC moment,
    taken from the actual trajectories (no extrapolation)."""
    if pair_ttc.empty:
        return pair_ttc.assign(min_gap=pd.Series(dtype=float))
    cols = ["tkey", "x", "y", "heading", "length", "width"]
    tracks = {u: g[cols].to_numpy(float) for u, g in df.sort_values("tkey").groupby("uid")}
    h = horizon_s * 1000.0
    gaps = np.full(len(pair_ttc), np.nan)
    for k, (ua, ub, t0) in enumerate(zip(pair_ttc["uid_a"], pair_ttc["uid_b"],
                                         pair_ttc["tkey_at_min"])):
        A, B = tracks[ua], tracks[ub]
        A = A[(A[:, 0] >= t0) & (A[:, 0] <= t0 + h)]
        B = B[(B[:, 0] >= t0) & (B[:, 0] <= t0 + h)]
        _, ia, ib = np.intersect1d(A[:, 0], B[:, 0], return_indices=True)
        if len(ia):
            a, b = A[ia], B[ib]
            gaps[k] = rect_gap(*a[:, 1:].T, *b[:, 1:].T).min()
    return pair_ttc.assign(min_gap=gaps)


def static_agents(df: pd.DataFrame, max_extent: float = 3.0, smooth_s: float = 2.0,
                  quantile: float = 0.02) -> set:
    """Agents that barely move over their whole track (parked cars, stopped tricycles,
    people standing still): after a centred rolling median over `smooth_s` seconds, the
    diagonal of the 2nd-98th percentile box of the positions is below `max_extent` m.

    The smoothing and percentiles make the rule robust to position noise: without them,
    jitter alone stretches a parked car's raw extent past the threshold, and it would
    re-enter the conflict screen with noisy velocities. Vehicles that stop in a queue but
    drive in or out of view are not static."""
    d = df.sort_values(["uid", "tkey"]) if "tkey" in df.columns else df
    if "tkey" in d.columns and d["tkey"].nunique() > 1:
        dt = np.median(np.diff(np.unique(d["tkey"].to_numpy()))) / 1000.0
    else:
        dt = 0.1
    w = max(1, int(round(smooth_s / dt))) | 1
    g = d.groupby("uid", sort=False)
    xs = g["x"].transform(lambda v: v.rolling(w, center=True, min_periods=1).median())
    ys = g["y"].transform(lambda v: v.rolling(w, center=True, min_periods=1).median())
    sm = pd.DataFrame({"uid": d["uid"].values, "x": xs.values, "y": ys.values})
    q = sm.groupby("uid")[["x", "y"]].quantile([quantile, 1 - quantile]).unstack()
    extent = np.hypot(q[("x", 1 - quantile)] - q[("x", quantile)],
                      q[("y", 1 - quantile)] - q[("y", quantile)])
    return set(extent[extent < max_extent].index)


def overlap_seconds(df: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Total time [s] each pair's footprints overlap over all frames both agents share.
    Real road users cannot overlap, so sustained overlap marks a tracking or box error
    (or a person boarding or leaving a vehicle)."""
    cols = ["tkey", "x", "y", "heading", "length", "width"]
    tracks = {u: g[cols].to_numpy(float) for u, g in df.sort_values("tkey").groupby("uid")}
    dt = np.median(np.diff(np.unique(df["tkey"].values))) / 1000.0
    out = np.zeros(len(pairs))
    for k, (ua, ub) in enumerate(zip(pairs["uid_a"], pairs["uid_b"])):
        A, B = tracks[ua], tracks[ub]
        _, ia, ib = np.intersect1d(A[:, 0], B[:, 0], return_indices=True)
        if len(ia):
            out[k] = (rect_gap(*A[ia, 1:].T, *B[ib, 1:].T) == 0).sum() * dt
    return out


def flag_plausibility(df: pd.DataFrame, pairs: pd.DataFrame, static: set,
                      max_overlap_s: float = 0.3) -> pd.DataFrame:
    """Add `static_pair`, `overlap_s` and `plausible` to a pair table.

    A pair is implausible if either agent is static or their footprints overlap for
    longer than `max_overlap_s` (brief overlap is tolerated as box-size slack)."""
    if pairs.empty:
        return pairs.assign(static_pair=pd.Series(dtype=bool), overlap_s=pd.Series(dtype=float),
                            plausible=pd.Series(dtype=bool))
    static_pair = pairs["uid_a"].isin(static) | pairs["uid_b"].isin(static)
    overlap = overlap_seconds(df, pairs)
    return pairs.assign(static_pair=static_pair.values, overlap_s=overlap,
                        plausible=(~static_pair.values) & (overlap <= max_overlap_s + 1e-9))


def road_user_group(interaction) -> np.ndarray:
    """Pool interaction classes: vehicle-vehicle, VRU-vehicle, VRU-VRU
    (VRU = pedestrian or two-wheeler)."""
    s = pd.Series(np.asarray(interaction, dtype=str))
    return np.where(s == "vehicle-vehicle", "vehicle-vehicle",
                    np.where(s.str.contains("vehicle"), "VRU-vehicle", "VRU-VRU"))


def ttc_conflicts(pair_ttc: pd.DataFrame, threshold: float, min_frames: int = 3,
                  gap_threshold: float | None = 1.0, plausible_only: bool = True) -> pd.DataFrame:
    """Pairs counted as TTC conflicts: minimum TTC below `threshold`, at least
    `min_frames` frames below it, (unless gap_threshold is None) a realised footprint
    gap of at most `gap_threshold` metres, and (if plausible_only and the column
    exists) a pair that passed the plausibility checks."""
    m = (pair_ttc["min_ttc"] < threshold) & (pair_ttc[f"frames_below_{threshold}"] >= min_frames)
    if plausible_only and "plausible" in pair_ttc.columns:
        m &= pair_ttc["plausible"].astype(bool)
    if gap_threshold is not None:
        m &= pair_ttc["min_gap"] <= gap_threshold
    return pair_ttc[m]


def summarise(pair_ttc: pd.DataFrame, pair_pet: pd.DataFrame,
              ttc_thresholds=(1.5, 3.0), pet_thresholds=(1.0, 2.0),
              min_frames: int = 3, gap_threshold: float | None = 1.0,
              by=("interaction", "geometry"), plausible_only: bool = True) -> pd.DataFrame:
    """Counts of conflict pairs, grouped by the columns in `by`.

    TTC conflicts follow `ttc_conflicts`. PET conflicts are crossing pairs below each
    PET threshold; under a `geometry` grouping they appear as 'crossing (PET)'.
    Valid `by` columns: recording, interaction, group, geometry.
    """
    by = list(by)
    events = []
    ttc = pair_ttc.assign(group=road_user_group(pair_ttc["interaction"])) if len(pair_ttc) else pair_ttc
    pet = pair_pet.assign(group=road_user_group(pair_pet["interaction"]),
                          geometry="crossing (PET)") if len(pair_pet) else pair_pet
    labels = []
    for thr in ttc_thresholds:
        labels.append(f"TTC<{thr}s")
        if len(ttc):
            hit = ttc_conflicts(ttc, thr, min_frames, gap_threshold, plausible_only)
            events.append(hit[by].assign(measure=labels[-1]))
    if len(pet) and plausible_only and "plausible" in pet.columns:
        pet = pet[pet["plausible"].astype(bool)]
    for thr in pet_thresholds:
        labels.append(f"PET<{thr}s")
        if len(pet):
            events.append(pet.loc[pet["pet"] < thr, by].assign(measure=labels[-1]))
    if not events:
        return pd.DataFrame(columns=labels)
    tab = (pd.concat(events, ignore_index=True)
             .groupby(by + ["measure"]).size().unstack("measure", fill_value=0))
    return tab.reindex(columns=labels, fill_value=0).sort_index()


def summarise_groups(pair_ttc, pair_pet, **kw) -> pd.DataFrame:
    """Pooled road-user groups per recording, plus an 'All recordings' block."""
    per_rec = summarise(pair_ttc, pair_pet, by=("recording", "group"), **kw)
    total = summarise(pair_ttc, pair_pet, by=("group",), **kw)
    total.index = pd.MultiIndex.from_product([["All recordings"], total.index],
                                             names=["recording", "group"])
    return pd.concat([per_rec, total])


def gap_sensitivity(pair_ttc: pd.DataFrame, ttc_thresholds=(1.5, 3.0),
                    gap_thresholds=(None, 2.0, 1.0, 0.5), min_frames: int = 3) -> pd.DataFrame:
    """TTC conflict counts by road-user group and geometry under different gap thresholds."""
    ttc = pair_ttc.assign(group=road_user_group(pair_ttc["interaction"]))
    cols = {}
    for thr in ttc_thresholds:
        for g in gap_thresholds:
            hit = ttc_conflicts(ttc, thr, min_frames, g)
            name = "no gap check" if g is None else f"gap<={g} m"
            cols[(f"TTC<{thr}s", name)] = hit.groupby(["group", "geometry"]).size()
    return pd.DataFrame(cols).fillna(0).astype(int).sort_index()


def exclusion_report(pair_ttc: pd.DataFrame, pair_pet: pd.DataFrame, ttc_threshold: float = 1.5,
                     pet_threshold: float = 2.0, min_frames: int = 3,
                     gap_threshold: float | None = 1.0) -> pd.DataFrame:
    """How many conflict candidates each plausibility rule removed, per recording and group.

    'static' = an agent barely moved over its track; 'overlap' = footprints overlapped too
    long (counted only for pairs not already removed as static)."""
    rows = []
    tc = ttc_conflicts(pair_ttc, ttc_threshold, min_frames, gap_threshold, plausible_only=False)
    pc = pair_pet[pair_pet["pet"] < pet_threshold]
    for label, tab in ((f"TTC<{ttc_threshold}s", tc), (f"PET<{pet_threshold}s", pc)):
        if tab.empty:
            continue
        t = tab.assign(group=road_user_group(tab["interaction"]))
        t["reason"] = np.where(t["static_pair"], "static",
                               np.where(t["plausible"], "kept", "overlap"))
        c = t.groupby(["recording", "group", "reason"]).size().unstack("reason", fill_value=0)
        c = c.reindex(columns=["kept", "static", "overlap"], fill_value=0)
        c.columns = pd.MultiIndex.from_product([[label], c.columns])
        rows.append(c)
    rep = pd.concat(rows, axis=1).fillna(0).astype(int)
    total = rep.groupby(level="group").sum()
    total.index = pd.MultiIndex.from_product([["All recordings"], total.index],
                                             names=["recording", "group"])
    return pd.concat([rep, total])


def detect_conflicts(df: pd.DataFrame, ttc_thresholds=(1.5, 3.0), frame_stride: int = 1,
                     min_rel_speed: float = 1.0, gap_horizon_s: float = 3.0,
                     static_extent: float = 3.0, max_overlap_s: float = 0.3):
    """Full conflict screen for one recording: TTC pairs (with actual gap and
    plausibility flags), PET pairs (with plausibility flags) and the static-agent set.
    Shared by the clean check (step 1) and the noise experiments (step 2)."""
    static = static_agents(df, static_extent)
    frames = scan_ttc(df, frame_stride=frame_stride, min_rel_speed=min_rel_speed)
    pt = realized_gap(df, aggregate_ttc(frames, thresholds=ttc_thresholds), gap_horizon_s)
    pt = flag_plausibility(df, pt, static, max_overlap_s)
    pp = flag_plausibility(df, scan_pet(df), static, max_overlap_s)
    return pt, pp, static
