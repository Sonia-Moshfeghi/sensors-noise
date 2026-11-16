"""Compare conflicts found in noisy data against the clean reference.

Conflicts are matched by the pair of *original* agent identities, so a conflict still
counts as found when an identity switch split one of the tracks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .conflicts import road_user_group, ttc_conflicts

# Severity classes used for 'severity shift': TTC < 1.5 s vs 1.5-3 s; PET < 1 s vs 1-2 s.
SEVERITY_EDGES = {"TTC": (1.5,), "PET": (1.0,)}


def conflict_table(pair_ttc: pd.DataFrame, pair_pet: pd.DataFrame, uid_to_orig=None,
                   ttc_threshold: float = 3.0, pet_threshold: float = 2.0,
                   min_frames: int = 3, gap_threshold: float = 1.0) -> pd.DataFrame:
    """One row per (measure, original pair) counted as a conflict, with its value
    (the smallest over any track fragments of the two agents)."""
    rows = []
    if len(pair_ttc):
        c = ttc_conflicts(pair_ttc, ttc_threshold, min_frames, gap_threshold)
        rows.append(c[["uid_a", "uid_b", "interaction"]].assign(
            measure=f"TTC<{ttc_threshold}s", value=c["min_ttc"].values))
    if len(pair_pet):
        p = pair_pet
        if "plausible" in p.columns:
            p = p[p["plausible"].astype(bool)]
        p = p[p["pet"] < pet_threshold]
        rows.append(p[["uid_a", "uid_b", "interaction"]].assign(
            measure=f"PET<{pet_threshold}s", value=p["pet"].values))
    cols = ["measure", "orig_a", "orig_b", "interaction", "group", "value"]
    if not rows:
        return pd.DataFrame(columns=cols)
    t = pd.concat(rows, ignore_index=True)
    if uid_to_orig is not None:
        t["uid_a"] = t["uid_a"].map(uid_to_orig).fillna(t["uid_a"])
        t["uid_b"] = t["uid_b"].map(uid_to_orig).fillna(t["uid_b"])
    a, b = t["uid_a"].astype(str), t["uid_b"].astype(str)
    t["orig_a"], t["orig_b"] = np.where(a <= b, a, b), np.where(a <= b, b, a)
    t = t[t["orig_a"] != t["orig_b"]]                 # fragments of one agent
    t = (t.sort_values("value")
          .groupby(["measure", "orig_a", "orig_b"], as_index=False)
          .agg(interaction=("interaction", "first"), value=("value", "min")))
    t["group"] = road_user_group(t["interaction"])
    return t[cols]


def _severity(measure: pd.Series, value: pd.Series) -> np.ndarray:
    kind = measure.str[:3]
    edge = np.where(kind == "TTC", SEVERITY_EDGES["TTC"][0], SEVERITY_EDGES["PET"][0])
    return (value.to_numpy() >= edge).astype(int)


def compare(clean: pd.DataFrame, noisy: pd.DataFrame):
    """Match noisy conflicts to clean ones.

    Returns (counts, matched): counts per (measure, group) with n_clean, n_noisy,
    found (in both), missed, false, severity_shift; matched holds the clean and noisy
    values of every conflict found in both.
    """
    key = ["measure", "orig_a", "orig_b"]

    def _ordered(t):                       # pair order must not matter
        a, b = t["orig_a"].astype(str), t["orig_b"].astype(str)
        return t.assign(orig_a=np.where(a <= b, a, b), orig_b=np.where(a <= b, b, a))

    clean, noisy = _ordered(clean), _ordered(noisy)
    m = clean.merge(noisy[key + ["value"]], on=key, how="outer",
                    suffixes=("_clean", "_noisy"), indicator=True)
    # Group label for false conflicts comes from the noisy table.
    m["group"] = m["group"].fillna(m[key].merge(noisy, on=key, how="left")["group"])
    both = m[m["_merge"] == "both"].copy()
    both["severity_shift"] = (_severity(both["measure"], both["value_clean"])
                              != _severity(both["measure"], both["value_noisy"]))
    counts = (m.assign(n_clean=m["_merge"] != "right_only",
                       n_noisy=m["_merge"] != "left_only",
                       found=m["_merge"] == "both",
                       missed=m["_merge"] == "left_only",
                       false=m["_merge"] == "right_only")
               .groupby(["measure", "group"])[["n_clean", "n_noisy", "found", "missed", "false"]]
               .sum().astype(int))
    counts["severity_shift"] = both.groupby(["measure", "group"])["severity_shift"].sum()
    counts["severity_shift"] = counts["severity_shift"].fillna(0).astype(int)
    matched = both[["measure", "group", "orig_a", "orig_b", "value_clean", "value_noisy"]]
    return counts.reset_index(), matched.reset_index(drop=True)


def pair_status(clean: pd.DataFrame, noisy: pd.DataFrame, static_ref=frozenset()) -> pd.DataFrame:
    """Every conflict in either table with its status (found / missed / false) and whether
    either agent is static in the reference data, for diagnosing false conflicts."""
    key = ["measure", "orig_a", "orig_b"]
    m = clean.merge(noisy[key + ["group", "value"]], on=key, how="outer",
                    suffixes=("_clean", "_noisy"), indicator=True)
    m["group"] = m["group_clean"].fillna(m["group_noisy"])
    m["status"] = m["_merge"].map({"both": "found", "left_only": "missed", "right_only": "false"})
    m["static_in_reference"] = m["orig_a"].isin(static_ref) | m["orig_b"].isin(static_ref)
    return m[key + ["group", "status", "static_in_reference", "value_clean", "value_noisy"]]


def summarise_counts(counts: pd.DataFrame, by=("kind", "level", "measure", "group")) -> pd.DataFrame:
    """Pool counts over seeds and recordings: recall = found / n_clean, precision =
    found / n_noisy, plus missed and false conflicts per seed."""
    by = list(by)
    g = counts.groupby(by)[["n_clean", "n_noisy", "found", "missed", "false", "severity_shift"]].sum()
    n_seeds = counts.groupby(by)["seed"].nunique()
    out = g.assign(
        recall=g["found"] / g["n_clean"].replace(0, np.nan),
        precision=g["found"] / g["n_noisy"].replace(0, np.nan),
        severity_shift_rate=g["severity_shift"] / g["found"].replace(0, np.nan),
        seeds=n_seeds,
    )
    for col in ("n_clean", "n_noisy", "found", "missed", "false", "severity_shift"):
        out[col] = out[col] / out["seeds"]           # per-seed averages
    return out.round(3)


def summarise_errors(matched: pd.DataFrame, by=("kind", "level", "measure", "group")) -> pd.DataFrame:
    """Value errors on conflicts found in both: median absolute error, median signed
    error (noisy - clean) and 90th-percentile absolute error, in seconds."""
    matched = matched.astype({"value_clean": float, "value_noisy": float})   # empty parts read as object
    d = matched.assign(err=matched["value_noisy"] - matched["value_clean"])
    d["abs_err"] = d["err"].abs()
    g = d.groupby(list(by))
    return pd.DataFrame({
        "n": g.size(),
        "median_abs_err_s": g["abs_err"].median(),
        "median_signed_err_s": g["err"].median(),
        "p90_abs_err_s": g["abs_err"].quantile(0.9),
    }).round(3)


def bootstrap_ci(pairs: pd.DataFrame, by=("kind", "level", "measure", "group"), n_boot: int = 1000,
                 seed: int = 0) -> pd.DataFrame:
    """Cluster bootstrap confidence intervals (2.5-97.5%) for recall and precision.

    The resampling unit is the road-user pair (recording, orig_a, orig_b), which carries
    all of its seed outcomes together: seed replicates of the same reference conflict are
    not independent observations. Recall resamples the reference pairs (status found or
    missed in any seed); precision resamples the reported pairs (found or false in any
    seed). Each cluster contributes its hits and its trials to the ratio.
    """
    rng = np.random.default_rng(seed)
    by = list(by)
    rows = {}
    key_cols = [c for c in ("recording", "orig_a", "orig_b") if c in pairs.columns]
    for key, g in pairs.groupby(by):
        out = {}
        for name, pool_mask in (("recall", g["status"] != "false"), ("precision", g["status"] != "missed")):
            sub = g[pool_mask]
            if sub.empty:
                out[f"{name}_lo"], out[f"{name}_hi"] = np.nan, np.nan
                continue
            clus = sub.groupby(key_cols)["status"].agg(hits=lambda x: (x == "found").sum(), trials="size")
            hits, trials = clus["hits"].to_numpy(float), clus["trials"].to_numpy(float)
            draws = rng.integers(0, len(clus), size=(n_boot, len(clus)))
            vals = hits[draws].sum(axis=1) / trials[draws].sum(axis=1)
            out[f"{name}_lo"], out[f"{name}_hi"] = np.quantile(vals, [0.025, 0.975])
        rows[key] = out
    ci = pd.DataFrame.from_dict(rows, orient="index")
    ci.index = pd.MultiIndex.from_tuples(ci.index, names=by)
    return ci.round(3)
