"""Surrogate safety measure primitives.

TTC is computed in two stages, both with constant-velocity extrapolation:
1. a fast multi-disc screen: each footprint is covered by up to MAX_DISCS circles
   along its heading. The circles fully cover the rectangle, so the disc TTC is a
   lower bound on the true TTC;
2. an exact refinement with oriented rectangles, stepping forward from the disc TTC
   and bisecting to the first moment the rectangles overlap (`ttc_rect`).
The refinement removes false TTC = 0 values for side-by-side vehicles, whose
circles overlap even when their rectangles do not.
"""
from __future__ import annotations

import numpy as np

MAX_DISCS = 5
# Pedestrian footprint is a 0.5 m square; this radius covers it, keeping the
# disc TTC a lower bound for the rectangle refinement.
PED_RADIUS = 0.36  # metres


def disc_geometry(length, width, is_ped):
    """Number of discs and disc radius covering each agent's rectangle."""
    length, width = np.asarray(length, float), np.asarray(width, float)
    n = np.clip(np.ceil(length / np.maximum(width, 1e-3)), 1, MAX_DISCS).astype(int)
    r = np.sqrt((length / (2 * n)) ** 2 + (width / 2) ** 2)
    n = np.where(is_ped, 1, n)
    r = np.where(is_ped, PED_RADIUS, r)
    return n, r


def disc_centres(x, y, heading, length, n):
    """Disc centres along the heading axis, shape (N, MAX_DISCS, 2); unused slots are NaN."""
    k = np.arange(MAX_DISCS)[None, :]
    seg = (np.asarray(length, float) / n)[:, None]
    offset = (k - (n[:, None] - 1) / 2.0) * seg
    cx = np.asarray(x)[:, None] + offset * np.cos(heading)[:, None]
    cy = np.asarray(y)[:, None] + offset * np.sin(heading)[:, None]
    centres = np.stack([cx, cy], axis=-1)
    centres[k.repeat(len(n), 0) >= n[:, None]] = np.nan
    return centres


def ttc_multidisc(ci, cj, vi, vj, R):
    """Time to collision for P agent pairs.

    ci, cj: (P, D, 2) disc centres; vi, vj: (P, 2) velocities; R: (P,) sum of radii.
    Returns (P,) TTC in seconds: 0 if already overlapping, inf if no collision course.
    """
    p = cj[:, None, :, :] - ci[:, :, None, :]           # (P, D, D, 2)
    v = (vj - vi)[:, None, None, :]                       # (P, 1, 1, 2)
    a = np.sum(v * v, axis=-1)
    b = 2.0 * np.sum(p * v, axis=-1)
    c = np.sum(p * p, axis=-1) - (R ** 2)[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = b ** 2 - 4 * a * c
        t = (-b - np.sqrt(disc)) / (2 * a)
        ttc = np.where(c <= 0, 0.0,
                       np.where((disc >= 0) & (a > 1e-9) & (t >= 0), t, np.inf))
    return ttc.reshape(len(ttc), -1).min(axis=1)


def heading_difference_deg(h1, h2):
    """Absolute heading difference wrapped to [0, 180] degrees."""
    d = np.abs((np.asarray(h1) - np.asarray(h2) + np.pi) % (2 * np.pi) - np.pi)
    return np.rad2deg(d)


def geometry_class(angle_deg):
    """Interaction geometry from heading difference."""
    angle_deg = np.asarray(angle_deg)
    return np.where(angle_deg < 30, "same_direction",
                    np.where(angle_deg > 150, "opposite", "crossing"))


def interaction_class(cat_i, cat_j):
    """Order-independent label such as 'pedestrian-vehicle'."""
    a, b = np.asarray(cat_i, dtype=object), np.asarray(cat_j, dtype=object)
    lo, hi = np.where(a <= b, a, b), np.where(a <= b, b, a)
    return np.char.add(np.char.add(lo.astype(str), "-"), hi.astype(str))


# --- Realised clearance between oriented rectangles -------------------------------

def rect_corners(x, y, heading, length, width):
    """Corners of oriented rectangles, shape (M, 4, 2), in drawing order."""
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    lx = np.asarray(length, float)[:, None] * (0.5 * np.array([1, 1, -1, -1]))
    wy = np.asarray(width, float)[:, None] * (0.5 * np.array([1, -1, -1, 1]))
    cx = np.asarray(x, float)[:, None] + lx * c - wy * s
    cy = np.asarray(y, float)[:, None] + lx * s + wy * c
    return np.stack([cx, cy], axis=-1)


def _overlapping(A, B, ha, hb):
    """Separating-axis test for M rectangle pairs."""
    axes = np.stack([
        np.stack([np.cos(ha), np.sin(ha)], -1), np.stack([-np.sin(ha), np.cos(ha)], -1),
        np.stack([np.cos(hb), np.sin(hb)], -1), np.stack([-np.sin(hb), np.cos(hb)], -1),
    ], axis=1)                                            # (M, 4 axes, 2)
    pa = np.einsum("mcd,mad->mac", A, axes)               # (M, 4 axes, 4 corners)
    pb = np.einsum("mcd,mad->mac", B, axes)
    separated = (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    return ~separated.any(axis=-1)


def _corner_to_edge(P, Q):
    """Smallest distance from the corners of P to the edges of Q, shape (M,)."""
    s0, s1 = Q, np.roll(Q, -1, axis=1)
    d = s1 - s0                                           # (M, 4 edges, 2)
    v = P[:, :, None, :] - s0[:, None, :, :]              # (M, 4 corners, 4 edges, 2)
    t = np.clip(np.sum(v * d[:, None], -1) / np.sum(d * d, -1)[:, None], 0, 1)
    closest = s0[:, None] + t[..., None] * d[:, None]
    return np.linalg.norm(P[:, :, None, :] - closest, axis=-1).reshape(len(P), -1).min(1)


def rect_gap(xa, ya, ha, la, wa, xb, yb, hb, lb, wb):
    """Edge-to-edge gap between agent footprints in metres (0 when they overlap)."""
    ha, hb = np.asarray(ha, float), np.asarray(hb, float)
    A = rect_corners(xa, ya, ha, la, wa)
    B = rect_corners(xb, yb, hb, lb, wb)
    gap = np.minimum(_corner_to_edge(A, B), _corner_to_edge(B, A))
    gap[_overlapping(A, B, ha, hb)] = 0.0
    return gap


def ttc_rect(xa, ya, ha, la, wa, vxa, vya, xb, yb, hb, lb, wb, vxb, vyb,
             t_start=None, t_cap=10.0, dt=0.1, bisections=6):
    """Exact time to collision between oriented rectangles under constant velocity.

    Steps forward from `t_start` (a lower bound such as the disc TTC; default 0) in
    increments of `dt` until the rectangles overlap, then bisects the last interval
    `bisections` times (resolution dt / 2**bisections). Returns inf when no overlap
    occurs before `t_cap`.
    """
    arr = lambda v: np.asarray(v, float)
    xa, ya, ha, la, wa, vxa, vya = map(arr, (xa, ya, ha, la, wa, vxa, vya))
    xb, yb, hb, lb, wb, vxb, vyb = map(arr, (xb, yb, hb, lb, wb, vxb, vyb))
    P = len(xa)
    t0 = np.zeros(P) if t_start is None else np.clip(arr(t_start), 0, None)

    def overlap(idx, t):
        A = rect_corners(xa[idx] + vxa[idx] * t, ya[idx] + vya[idx] * t, ha[idx], la[idx], wa[idx])
        B = rect_corners(xb[idx] + vxb[idx] * t, yb[idx] + vyb[idx] * t, hb[idx], lb[idx], wb[idx])
        return _overlapping(A, B, ha[idx], hb[idx])

    result = np.full(P, np.inf)
    lo = t0.copy()                  # last time known to be free of overlap (or t0)
    t = t0.copy()
    active = np.flatnonzero(t0 <= t_cap)
    first = np.ones(P, bool)        # overlap already at t0 -> TTC = t0 exactly
    while active.size:
        hit = overlap(active, t[active])
        done = active[hit]
        exact = done[first[done]]
        result[exact] = t[exact]
        refine = done[~first[done]]
        if refine.size:             # bisect between lo (free) and t (overlap)
            a, b = lo[refine].copy(), t[refine].copy()
            for _ in range(bisections):
                mid = 0.5 * (a + b)
                h = overlap(refine, mid)
                b = np.where(h, mid, b)
                a = np.where(h, a, mid)
            result[refine] = b
        rest = active[~hit]
        lo[rest] = t[rest]
        first[rest] = False
        t[rest] += dt
        active = rest[t[rest] <= t_cap + 1e-9]
    return result
