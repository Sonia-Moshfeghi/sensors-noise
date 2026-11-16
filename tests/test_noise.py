"""Checks that each noise model produces the error it claims, and that matching works."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.noise import apply_noise, derive_kinematics  # noqa: E402
from sensors_noise.noise_eval import compare, conflict_table  # noqa: E402


def _straight_tracks(n_agents=200, seconds=30, speed=8.0, ped_share=0.5, seed=0):
    """Agents on straight lines at constant speed, 10 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 0.1)
    rows = []
    for a in range(n_agents):
        ped = a < n_agents * ped_share
        h = rng.uniform(-np.pi, np.pi)
        v = 1.4 if ped else speed
        rows.append(pd.DataFrame({
            "uid": f"{'P' if ped else 'V'}_{a}", "category": "pedestrian" if ped else "vehicle",
            "agent_type": "pedestrian" if ped else "car",
            "tkey": np.round(t * 1000).astype(int),
            "x": rng.uniform(-20, 20) + v * np.cos(h) * t, "y": rng.uniform(-20, 20) + v * np.sin(h) * t,
            "vx": v * np.cos(h), "vy": v * np.sin(h), "heading": h,
            "length": 0.5 if ped else 4.5, "width": 0.5 if ped else 1.8,
        }))
    return pd.concat(rows, ignore_index=True)


def _err(clean, noisy):
    m = clean.merge(noisy, on=["uid", "tkey"], suffixes=("", "_n"))
    return m, np.hypot(m["x_n"] - m["x"], m["y_n"] - m["y"])


def test_derive_kinematics_exact_on_constant_velocity():
    df = _straight_tracks(20)
    d = derive_kinematics(df.assign(vx=0.0, vy=0.0, heading=0.0))
    m = df.merge(d, on=["uid", "tkey"], suffixes=("", "_d"))
    assert np.allclose(m["vx_d"], m["vx"], atol=1e-6) and np.allclose(m["vy_d"], m["vy"], atol=1e-6)
    dh = np.angle(np.exp(1j * (m["heading_d"] - m["heading"])))
    assert np.abs(dh).max() < 1e-6


def test_jitter_mean_error_matches_level():
    df = _straight_tracks()
    _, e = _err(df, apply_noise(df, "jitter", 0.5, np.random.default_rng(1)))
    assert abs(e.mean() - 0.5) < 0.01


def test_drift_is_correlated_with_right_size():
    df = _straight_tracks()
    m, e = _err(df, apply_noise(df, "drift", 0.5, np.random.default_rng(2)))
    assert abs(e.mean() - 0.5) < 0.05
    ex = (m["x_n"] - m["x"]).to_numpy()
    same = (m["uid"].to_numpy()[1:] == m["uid"].to_numpy()[:-1])
    lag1 = np.corrcoef(ex[:-1][same], ex[1:][same])[0, 1]
    assert abs(lag1 - np.exp(-0.1 / 1.0)) < 0.02          # AR(1), tau = 1 s at 10 Hz


def test_offset_is_constant_along_heading():
    df = _straight_tracks(50)
    m, e = _err(df, apply_noise(df, "offset", 0.6, np.random.default_rng(3)))
    assert np.allclose(e, 0.6, atol=1e-9)
    along = (m["x_n"] - m["x"]) * np.cos(m["heading"]) + (m["y_n"] - m["y"]) * np.sin(m["heading"])
    assert np.allclose(np.abs(along), 0.6, atol=1e-9)


def test_heading_noise_std():
    df = _straight_tracks()
    n = apply_noise(df, "heading", 5.0, np.random.default_rng(4))
    m = df.merge(n, on=["uid", "tkey"], suffixes=("", "_n"))
    dh = np.rad2deg(np.angle(np.exp(1j * (m["heading_n"] - m["heading"]))))
    assert abs(dh.std() - 5.0) < 0.1


def test_dropout_fraction_and_bursts():
    df = _straight_tracks()
    n = apply_noise(df, "dropout", 0.15, np.random.default_rng(5))
    assert abs(1 - len(n) / len(df) - 0.15) < 0.02
    # Missing frames come in bursts: mean burst length close to 0.5 s = 5 frames.
    k = df.merge(n[["uid", "tkey"]].assign(kept=True), on=["uid", "tkey"], how="left")
    miss = k["kept"].isna().to_numpy()
    starts = miss & ~np.r_[False, miss[:-1]]
    assert 4.0 < miss.sum() / starts.sum() < 6.0


def test_id_switch_rates_and_original_ids():
    df = _straight_tracks(400)
    n = apply_noise(df, "id_switch", 1.0, np.random.default_rng(6))
    splits = n.groupby("orig_uid")["uid"].nunique() - 1
    ped = splits[splits.index.str.startswith("P_")].mean()
    veh = splits[splits.index.str.startswith("V_")].mean()
    assert abs(ped - 1.0) < 0.15 and abs(veh - 1 / 3) < 0.1
    assert set(n["orig_uid"]) == set(df["uid"])


def test_compare_counts_and_fragment_matching():
    clean = pd.DataFrame({"measure": ["TTC<3.0s"] * 3, "orig_a": ["V_1", "V_2", "V_3"],
                          "orig_b": ["P_9", "V_5", "V_7"], "interaction": ["pedestrian-vehicle",
                          "vehicle-vehicle", "vehicle-vehicle"], "group": ["VRU-vehicle",
                          "vehicle-vehicle", "vehicle-vehicle"], "value": [1.0, 2.0, 2.5]})
    # Noisy run: V_1 was split into V_1#1; V_2/V_5 missed; V_8/V_9 is a false conflict.
    pair_ttc = pd.DataFrame({
        "uid_a": ["P_9", "V_3", "V_8"], "uid_b": ["V_1#1", "V_7", "V_9"],
        "interaction": ["pedestrian-vehicle", "vehicle-vehicle", "vehicle-vehicle"],
        "min_ttc": [1.6, 2.4, 1.0], "frames_below_3.0": [5, 5, 5], "min_gap": [0.5, 0.5, 0.5],
        "plausible": [True, True, True]})
    noisy = conflict_table(pair_ttc, pd.DataFrame(), {"V_1#1": "V_1"})
    counts, matched = compare(clean, noisy)
    c = counts.set_index("group")
    assert c.loc["vehicle-vehicle", ["found", "missed", "false"]].tolist() == [1, 1, 1]
    assert c.loc["VRU-vehicle", ["found", "missed", "false"]].tolist() == [1, 0, 0]
    # TTC 1.0 -> 1.6 crosses the 1.5 s severity edge; 2.5 -> 2.4 does not.
    assert c.loc["VRU-vehicle", "severity_shift"] == 1 and c.loc["vehicle-vehicle", "severity_shift"] == 0
    assert len(matched) == 2


def test_heldout_noise_kinds_have_claimed_statistics():
    df = _straight_tracks()
    _, e = _err(df, apply_noise(df, "jitter_t", 0.3, np.random.default_rng(7)))
    assert abs(e.mean() - 0.3) < 0.03                       # heavy-tailed but same mean error
    assert e.quantile(0.99) > 3 * e.mean()                   # heavier tail than Gaussian (~2.5x)
    m, e = _err(df, apply_noise(df, "drift_slow", 0.5, np.random.default_rng(8)))
    ex = (m["x_n"] - m["x"]).to_numpy()
    same = (m["uid"].to_numpy()[1:] == m["uid"].to_numpy()[:-1])
    assert abs(np.corrcoef(ex[:-1][same], ex[1:][same])[0, 1] - np.exp(-0.1 / 3.0)) < 0.02
    d = apply_noise(df, "dropout_long", 0.15, np.random.default_rng(9))
    k = df.merge(d[["uid", "tkey"]].assign(kept=True), on=["uid", "tkey"], how="left")
    miss = k["kept"].isna().to_numpy()
    starts = miss & ~np.r_[False, miss[:-1]]
    assert 12 < miss.sum() / starts.sum() < 18                # ~1.5-s bursts at 10 Hz
