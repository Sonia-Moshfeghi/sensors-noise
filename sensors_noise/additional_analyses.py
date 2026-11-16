"""Additional analyses computed from the saved parts/ tables (no reruns).

A. severity_shift      rate of found conflicts crossing the 1.5-s TTC / 1-s PET edge
B. near_threshold      share of false TTC conflicts whose reference values lie just outside
                       the conflict definition (TTC < 4 s or gap <= 1.5 m)
C. bias_all_pairs      TTC error on ALL reference conflicts (not only those found), using
                       the noisy/cleaned min TTC of the same original pair
D. native_reference    recall/precision of each run against the SinD-native-kinematics
                       reference instead of the derived-kinematics baseline
E. false_composition   geometry, closing speed and clearance of false TTC conflicts
F. analytical_model    false-collision-course probability for close parallel pairs predicted
                       by the constant-velocity TTC error model vs. the simulated rate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .noise import MEAN_TO_SIGMA, apply_noise

KEY = ["orig_a", "orig_b"]


def _read(parts: Path, pattern: str) -> pd.DataFrame:
    files = sorted(parts.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _tags(df: pd.DataFrame) -> list:
    return [c for c in ("kind", "level", "method", "seed", "recording") if c in df.columns]


def severity_shift(counts: pd.DataFrame) -> pd.DataFrame:
    by = [c for c in ("kind", "level", "method", "measure", "group") if c in counts.columns]
    g = counts.groupby(by)[["found", "severity_shift"]].sum()
    return g.assign(severity_shift_rate=(g["severity_shift"] / g["found"].replace(0, np.nan)).round(3))


def near_threshold(pairs: pd.DataFrame, ref: pd.DataFrame, ttc_slack: float = 1.0,
                   gap_slack: float = 0.5) -> pd.DataFrame:
    """False TTC conflicts joined to the baseline reference pair table."""
    base = ref[ref["reference"] == "baseline"]
    f = pairs[(pairs["status"] == "false") & (pairs["measure"].str.startswith("TTC"))]
    m = f.merge(base[KEY + ["recording", "min_ttc", "min_gap"]], on=KEY + ["recording"], how="left")
    m["in_reference_screen"] = m["min_ttc"].notna()
    m["near_threshold"] = m["in_reference_screen"] & (
        ((m["min_ttc"] < 3.0 + ttc_slack) & (m["min_gap"] <= 1.0 + gap_slack)))
    m["ref_ttc_lt_3"] = m["min_ttc"] < 3.0
    m["ref_gap_le_1"] = m["min_gap"] <= 1.0
    by = [c for c in ("kind", "level", "method", "group") if c in m.columns]
    out = m.groupby(by).agg(n_false=("status", "size"),
                            share_in_reference_screen=("in_reference_screen", "mean"),
                            share_near_threshold=("near_threshold", "mean"),
                            share_ref_ttc_lt_3s=("ref_ttc_lt_3", "mean"),
                            share_ref_gap_le_1m=("ref_gap_le_1", "mean"))
    return out.round(3)


def bias_all_pairs(found_ref: pd.DataFrame, pt: pd.DataFrame) -> pd.DataFrame:
    """TTC error over all baseline-reference TTC conflicts, per run."""
    ref = found_ref[(found_ref["reference"] == "baseline") & found_ref["measure"].str.startswith("TTC")]
    rows = []
    for tags, g in pt.groupby(_tags(pt)):
        tag = dict(zip(_tags(pt), tags))
        r = ref[ref["recording"] == tag["recording"]]
        m = r.merge(g[KEY + ["min_ttc"]], on=KEY, how="left")
        err = (m["min_ttc"] - m["value"]).to_numpy()
        rows.append({**tag, "n_ref": len(m), "n_screened": int(m["min_ttc"].notna().sum()),
                     "median_err_all_s": float(np.nanmedian(err)) if np.isfinite(err).any() else np.nan,
                     "median_abs_err_all_s": float(np.nanmedian(np.abs(err))) if np.isfinite(err).any() else np.nan,
                     "share_not_screened": float(m["min_ttc"].isna().mean())})
    out = pd.DataFrame(rows)
    by = [c for c in ("kind", "level", "method") if c in out.columns]
    return out.groupby(by)[["median_err_all_s", "median_abs_err_all_s", "share_not_screened"]].mean().round(3)


def native_reference(found: pd.DataFrame, found_ref: pd.DataFrame) -> pd.DataFrame:
    """Recall and precision against the SinD-native (original) kinematics reference."""
    ref = found_ref[found_ref["reference"] == "original"]
    rows = []
    for tags, g in found.groupby(_tags(found)):
        tag = dict(zip(_tags(found), tags))
        r = ref[ref["recording"] == tag["recording"]]
        for meas in sorted(set(r["measure"]) | set(g["measure"])):
            rk = r[r["measure"] == meas][KEY].apply(tuple, axis=1)
            gk = g[g["measure"] == meas][KEY].apply(tuple, axis=1)
            both = len(set(rk) & set(gk))
            rows.append({**tag, "measure": meas, "n_ref": len(rk), "n_found": len(gk), "both": both})
    out = pd.DataFrame(rows)
    by = [c for c in ("kind", "level", "method", "measure") if c in out.columns]
    g = out.groupby(by)[["n_ref", "n_found", "both"]].sum()
    return g.assign(recall=(g["both"] / g["n_ref"]).round(3), precision=(g["both"] / g["n_found"]).round(3))


def false_composition(pairs: pd.DataFrame, pt: pd.DataFrame) -> pd.DataFrame:
    """Geometry, closing speed and realised clearance of false TTC conflicts (from the
    noisy/cleaned run's own pair table)."""
    f = pairs[(pairs["status"] == "false") & pairs["measure"].str.startswith("TTC")]
    tags = [c for c in _tags(pt) if c in f.columns]
    m = f.merge(pt, on=KEY + tags, how="left", suffixes=("", "_pt"))
    by = [c for c in ("kind", "level", "method") if c in m.columns]
    geo = m.groupby(by + ["geometry"]).size().unstack("geometry", fill_value=0)
    geo = geo.div(geo.sum(axis=1), axis=0).round(3).add_prefix("share_")
    num = m.groupby(by).agg(n_false=("status", "size"),
                            rel_speed_median=("rel_speed_at_min", "median"),
                            gap_median=("min_gap", "median"),
                            ttc_median=("min_ttc", "median")).round(2)
    return num.join(geo)


def analytical_model(ref: pd.DataFrame, pairs: pd.DataFrame, window_s: float = 0.5,
                     horizon_s: float = 3.0) -> pd.DataFrame:
    """Constant-velocity TTC error model vs. simulation for close parallel pairs.

    Velocity by central differences over +/-W of positions with per-axis noise sigma has
    error variance sigma^2/(2W^2) per agent, so the relative lateral velocity error has
    sigma_rel = sigma / W. A parallel pair with footprint clearance d is projected onto a
    collision course within T when the lateral closing-speed error exceeds d/T:
    p = 1 - Phi(d / (T sigma_rel)). Over n effectively independent evaluations (the
    error is correlated over 2W, so n ~ co-presence / 2W) the pair becomes a false
    conflict with probability 1 - (1-p)^n. Compared with the simulated share of
    close (gap <= 1 m), same-direction, non-conflict reference pairs that became false
    conflicts under jitter.
    """
    base = ref[ref["reference"] == "baseline"]
    close = base[(base["min_gap"] <= 1.0) & (base["geometry"] == "same_direction")
                 & ~((base["min_ttc"] < 3.0) & base["plausible"].astype(bool) & (base["frames_below_3.0"] >= 3))]
    if "n_frames" not in close.columns:
        raise ValueError("reference table lacks n_frames; rerun with the revised code")
    f = pairs[(pairs["status"] == "false") & pairs["measure"].str.startswith("TTC") & (pairs["kind"] == "jitter")]
    rows = []
    for (lv, seed), g in f.groupby(["level", "seed"]):
        sigma = lv * MEAN_TO_SIGMA
        sig_rel = sigma / window_s
        hit = close.merge(g[KEY + ["recording"]].drop_duplicates(), on=KEY + ["recording"], how="left", indicator=True)
        became = (hit["_merge"] == "both").to_numpy()
        n = np.maximum(hit["n_frames"].to_numpy() / (2 * window_s * 10), 1.0)
        d = np.clip(hit["min_gap"].to_numpy(), 0.05, None)
        p = 1 - norm.cdf(d / (horizon_s * sig_rel))
        pred = 1 - (1 - p) ** n
        rows.append({"level": lv, "seed": seed, "n_close_pairs": len(hit),
                     "simulated_share_false": float(became.mean()),
                     "model_predicted_share": float(pred.mean()), "sigma_rel_mps": round(sig_rel, 3)})
    out = pd.DataFrame(rows)
    return out.groupby("level")[["n_close_pairs", "simulated_share_false", "model_predicted_share", "sigma_rel_mps"]].mean().round(3)


def run_all(results_dir: str, out_dir: str | None = None, verbose: bool = True) -> dict:
    R = Path(results_dir)
    out = Path(out_dir or R / "additional_analyses")
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for step in ("02_noise", "03_filters"):
        parts = R / step / "parts"
        if not parts.exists():
            continue
        counts, pairs = _read(parts, "*__counts.csv"), _read(parts, "*__pairs.csv")
        pt, found = _read(parts, "*__pt.csv"), _read(parts, "*__found.csv")
        ref = _read(parts, "*__reference.csv")
        found_ref = _read(parts, "*__reference_conflicts_*.csv")
        if ref.empty:
            if verbose:
                print(f"{step}: no reference tables (rerun with the revised code); skipping")
            continue
        tabs = {
            "severity_shift": severity_shift(counts),
            "near_threshold": near_threshold(pairs, ref),
            "bias_all_pairs": bias_all_pairs(found_ref, pt) if not pt.empty else pd.DataFrame(),
            "native_reference": native_reference(found, found_ref) if not found.empty else pd.DataFrame(),
            "false_composition": false_composition(pairs, pt) if not pt.empty else pd.DataFrame(),
        }
        if step == "02_noise":
            try:
                tabs["analytical_model"] = analytical_model(ref, pairs)
            except ValueError as e:
                if verbose:
                    print("analytical_model:", e)
        for name, tab in tabs.items():
            tab.to_csv(out / f"{step}_{name}.csv")
            res[f"{step}_{name}"] = tab
            if verbose and len(tab):
                print(f"\n== {step}: {name} ==")
                print(tab.head(30).to_string())
    if verbose:
        print(f"\nWritten to {out.resolve()}")
    return res


# ----------------------------------------------------------------------------------
# Additional analyses, part 2 (require the 06b chain results for the routing shares)
# ----------------------------------------------------------------------------------

def close_pairs_removed(ref: pd.DataFrame, pt: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """For the close same-direction non-conflict reference pairs of the analytical-model
    check: per jitter level, the share that became false conflicts, the share removed by
    the plausibility criteria in the noisy data (implausible in the noisy pair table), and
    the share no longer in the screen at all."""
    base = ref[ref["reference"] == "baseline"]
    close = base[(base["min_gap"] <= 1.0) & (base["geometry"] == "same_direction")
                 & ~((base["min_ttc"] < 3.0) & base["plausible"].astype(bool) & (base["frames_below_3.0"] >= 3))]
    f = pairs[(pairs["status"] == "false") & pairs["measure"].str.startswith("TTC") & (pairs["kind"] == "jitter")]
    rows = []
    for (lv, seed), g in pt[pt["kind"] == "jitter"].groupby(["level", "seed"]):
        if "method" in g.columns and (g["method"] != "none").any():
            g = g[g["method"] == "none"]
        m = close.merge(g[KEY + ["recording", "plausible"]].rename(columns={"plausible": "noisy_plausible"}),
                        on=KEY + ["recording"], how="left")
        became = close.merge(f[(f["level"] == lv) & (f["seed"] == seed)][KEY + ["recording"]].drop_duplicates(),
                             on=KEY + ["recording"], how="left", indicator=True)["_merge"].eq("both").to_numpy()
        rows.append({"level": lv, "seed": seed, "n_close": len(m),
                     "share_false_conflict": float(became.mean()),
                     "share_removed_implausible": float((m["noisy_plausible"].notna() & ~m["noisy_plausible"].fillna(True).astype(bool)).mean()),
                     "share_not_in_screen": float(m["noisy_plausible"].isna().mean())})
    out = pd.DataFrame(rows).groupby("level")[["n_close", "share_false_conflict", "share_removed_implausible", "share_not_in_screen"]].mean()
    out["share_false_plus_removed"] = out["share_false_conflict"] + out["share_removed_implausible"]
    return out.round(3)


def severity_direction(matched: pd.DataFrame) -> pd.DataFrame:
    """Severity shifts split by direction: toward the more severe class (clean value above
    the edge, noisy value below) and away from it."""
    m = matched.copy()
    edge = np.where(m["measure"].str.startswith("TTC"), 1.5, 1.0)
    m["toward"] = (m["value_clean"] >= edge) & (m["value_noisy"] < edge)
    m["away"] = (m["value_clean"] < edge) & (m["value_noisy"] >= edge)
    by = [c for c in ("kind", "level", "method", "measure", "group") if c in m.columns]
    g = m.groupby(by).agg(found=("toward", "size"), toward=("toward", "sum"), away=("away", "sum"))
    return g.assign(toward_rate=(g["toward"] / g["found"]).round(3), away_rate=(g["away"] / g["found"]).round(3))


def heading_pet_channel(pairs: pd.DataFrame, pt: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """How heading error changes PET conflict sets: of the PET conflicts missed under
    heading error, the share whose pair is flagged implausible (overlap/static criteria,
    which use oriented footprints) in the noisy pair table; of the false PET conflicts,
    the share whose pair was implausible in the reference."""
    base = ref[ref["reference"] == "baseline"][KEY + ["recording", "plausible"]].rename(columns={"plausible": "ref_plausible"})
    h = pairs[(pairs["kind"] == "heading") & pairs["measure"].str.startswith("PET")]
    rows = []
    for (lv, seed), g in h.groupby(["level", "seed"]):
        npt = pt[(pt["kind"] == "heading") & (pt["level"] == lv) & (pt["seed"] == seed)][KEY + ["recording", "plausible"]]
        miss = g[g["status"] == "missed"].merge(npt, on=KEY + ["recording"], how="left")
        false = g[g["status"] == "false"].merge(base, on=KEY + ["recording"], how="left")
        rows.append({"level": lv, "seed": seed, "n_missed": len(miss),
                     "missed_flagged_implausible_noisy": float((~miss["plausible"].fillna(True).astype(bool)).mean()) if len(miss) else np.nan,
                     "n_false": len(false),
                     "false_implausible_in_reference": float((~false["ref_plausible"].fillna(True).astype(bool)).mean()) if len(false) else np.nan})
    return pd.DataFrame(rows).groupby("level").mean().round(3)


def sigma_hat_tracker(data_root, models_dir, level: float = 0.3) -> pd.DataFrame:
    """Per-track noise estimate on raw detections and on tracker-like output under jitter."""
    from .filters import track_noise_estimates, tracker_clean
    from .filter_pipeline import load_params
    from .io_sind import find_recordings, load_recording
    params = load_params(models_dir)
    rows = []
    for rec in find_recordings(data_root):
        df = load_recording(rec, verbose=False)
        for lv in (0.0, level, 1.0):
            noisy = apply_noise(df, "jitter", lv, np.random.default_rng(123)) if lv > 0 else apply_noise(df, "baseline", 0.0, np.random.default_rng(0))
            raw = track_noise_estimates(noisy)
            trk = track_noise_estimates(tracker_clean(noisy, **params["tracker"]))
            rows.append({"recording": rec.name, "jitter": lv, "sigma_raw_median": raw.median(),
                         "sigma_tracker_median": trk.median(), "tracker_below_s1": float((trk < params["hybrid"]["low"]).mean()),
                         "tracker_below_s2": float((trk < params["hybrid"]["high"]).mean())})
    return pd.DataFrame(rows).round(3)


def run_part2(results_dir: str, out_dir: str | None = None, verbose: bool = True) -> dict:
    R = Path(results_dir)
    out = Path(out_dir or R / "additional_analyses")
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    parts2 = R / "02_noise" / "parts"
    pairs, pt, ref, matched = _read(parts2, "*__pairs.csv"), _read(parts2, "*__pt.csv"), _read(parts2, "*__reference.csv"), _read(parts2, "*__matched.csv")
    res["02_noise_close_pairs_removed"] = close_pairs_removed(ref, pt, pairs)
    res["02_noise_severity_direction"] = severity_direction(matched)
    res["02_noise_heading_pet_channel"] = heading_pet_channel(pairs, pt, ref)
    parts3 = R / "03_filters" / "parts"
    if parts3.exists():
        m3 = _read(parts3, "*__matched.csv")
        res["03_filters_severity_direction"] = severity_direction(m3)
        counts = _read(parts3, "*__counts.csv")
        if "route_counts" in counts.columns:
            rc = counts.dropna(subset=["route_counts"]).drop_duplicates(["recording", "kind", "level", "seed", "method"])
            import json
            rc = rc.assign(**pd.DataFrame([json.loads(x) for x in rc["route_counts"]], index=rc.index).fillna(0))
            cols = [c for c in ("none", "kalman", "denoiser") if c in rc.columns]
            tot = rc[cols].sum(axis=1)
            for c in cols:
                rc[c] = rc[c] / tot
            res["03_filters_hybrid_routing"] = rc.groupby(["kind", "level", "method"])[cols].mean().round(3)
    for name, tab in res.items():
        tab.to_csv(out / f"{name}.csv")
        if verbose:
            print(f"\n== {name} ==")
            print(tab.to_string())
    return res
