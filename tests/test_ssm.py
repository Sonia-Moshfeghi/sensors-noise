"""Checks of the safety-measure code on scenarios with known answers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.conflicts import (aggregate_ttc, exclusion_report,  # noqa: E402
                                     flag_plausibility, realized_gap, scan_pet, scan_ttc,
                                     static_agents, summarise, summarise_groups)
from sensors_noise.io_sind import load_recording  # noqa: E402
from sensors_noise.ssm import (PED_RADIUS, disc_centres, disc_geometry,  # noqa: E402
                               heading_difference_deg, rect_gap, ttc_multidisc, ttc_rect)


def _single_discs(x, y, h):
    n, r = disc_geometry([0.5], [0.5], [True])
    return disc_centres(np.array([x]), np.array([y]), np.array([h]), [0.5], n), r


def test_head_on_ttc():
    ci, ri = _single_discs(0.0, 0.0, 0.0)
    cj, rj = _single_discs(20.0, 0.0, np.pi)
    ttc = ttc_multidisc(ci, cj, np.array([[5.0, 0]]), np.array([[-5.0, 0]]), ri + rj)
    assert np.isclose(ttc[0], (20 - 2 * PED_RADIUS) / 10)


def test_diverging_is_inf_and_overlap_is_zero():
    ci, ri = _single_discs(0.0, 0.0, 0.0)
    cj, rj = _single_discs(5.0, 0.0, 0.0)
    v_apart = ttc_multidisc(ci, cj, np.array([[-1.0, 0]]), np.array([[1.0, 0]]), ri + rj)
    assert np.isinf(v_apart[0])
    ck, rk = _single_discs(0.2, 0.0, 0.0)
    assert ttc_multidisc(ci, ck, np.zeros((1, 2)), np.zeros((1, 2)), ri + rk)[0] == 0


def test_adjacent_lane_bus_not_flagged():
    """A bus beside a car 3.5 m apart, same speed: no overlap, no collision course."""
    n, r = disc_geometry([12.0, 4.5], [2.5, 1.8], [False, False])
    c = disc_centres(np.array([0.0, 0.0]), np.array([0.0, 3.5]), np.zeros(2),
                     np.array([12.0, 4.5]), n)
    v = np.array([[10.0, 0]])
    ttc = ttc_multidisc(c[[0]], c[[1]], v, v, r[[0]] + r[[1]])
    assert np.isinf(ttc[0])


def test_rect_gap_known_cases():
    one = np.ones(1)
    # Two cars passing in adjacent opposite lanes, 3.5 m apart: gap = 3.5 - 1.8.
    g = rect_gap(0 * one, 0 * one, 0 * one, 4.5 * one, 1.8 * one,
                 0 * one, 3.5 * one, np.pi * one, 4.5 * one, 1.8 * one)
    assert np.isclose(g[0], 1.7)
    # Nose-to-tail, 1 m between bumpers.
    g = rect_gap(0 * one, 0 * one, 0 * one, 4.5 * one, 1.8 * one,
                 5.5 * one, 0 * one, 0 * one, 4.5 * one, 1.8 * one)
    assert np.isclose(g[0], 1.0)
    # Corner of a 45-degree-rotated square pointing at a square's side: gap = 3 - 0.5 - sqrt(0.5).
    g = rect_gap(0 * one, 0 * one, 0 * one, one, one,
                 3 * one, 0 * one, np.pi / 4 * one, one, one)
    assert np.isclose(g[0], 3 - 0.5 - np.sqrt(0.5))
    # Overlapping footprints.
    g = rect_gap(0 * one, 0 * one, 0 * one, 4.5 * one, 1.8 * one,
                 1 * one, 0.5 * one, 0.3 * one, 4.5 * one, 1.8 * one)
    assert g[0] == 0


def test_ttc_rect_cases():
    o, z = np.ones(1), np.zeros(1)
    car = dict(l=4.5 * o, w=1.8 * o)
    # Rear-end: centres 20 m apart, closing at 5 m/s, bumpers touch at 4.5 m -> 3.1 s.
    t = ttc_rect(z, z, z, car["l"], car["w"], 5 * o, z, 20 * o, z, z, car["l"], car["w"], z, z)
    assert np.isclose(t[0], 3.1, atol=0.01)
    # Side by side with 0.5 m clearance, one overtaking: never collides
    # (the disc screen reports overlap here, which the refinement must undo).
    t = ttc_rect(z, z, z, car["l"], car["w"], 12 * o, z, z, 2.3 * o, z, car["l"], car["w"], 10 * o, z)
    assert np.isinf(t[0])
    # Disc TTC is a lower bound on the rectangle TTC.
    n, r = disc_geometry([4.5, 4.5], [1.8, 1.8], [False, False])
    c = disc_centres(np.array([0.0, 0.0]), np.array([0.0, 2.3]), np.zeros(2), np.array([4.5, 4.5]), n)
    assert ttc_multidisc(c[[0]], c[[1]], np.array([[12.0, 0]]), np.array([[10.0, 0]]), r[[0]] + r[[1]])[0] == 0
    # Perpendicular crossing, checked against a brute-force 0.1 ms scan: 1.685 s.
    t = ttc_rect(-20 * o, z, z, car["l"], car["w"], 10 * o, z,
                 z, -20 * o, np.pi / 2 * o, car["l"], car["w"], z, 10 * o)
    assert np.isclose(t[0], 1.685, atol=0.003)


def test_heading_difference_wraps():
    assert np.isclose(heading_difference_deg(np.deg2rad(170), np.deg2rad(-170)), 20)


def _synthetic_recording(folder: Path):
    """Car A eastbound, car B northbound 1.5 s behind A at the crossing point,
    a pedestrian crossing A's path 2.0 s after A, and car C closing on A from behind."""
    t = np.arange(0, 10, 0.1)
    rows = []
    for tid, x, y, vx, vy, h in [
        (1, -30 + 10 * t, 0 * t, 10, 0, 0.0),
        (2, 0 * t, -45 + 10 * t, 0, 10, np.pi / 2),
        (3, -50 + 15 * t, 0 * t, 15, 0, 0.0),  # closes on car 1 from behind
    ]:
        for k in range(len(t)):
            rows.append(dict(track_id=tid, frame_id=k, timestamp_ms=int(round(t[k] * 1000)),
                             agent_type="car", x=x[k], y=y[k], vx=vx, vy=vy,
                             yaw_rad=h, heading_rad=h, length=4.5, width=1.8))
    pd.DataFrame(rows).to_csv(folder / "Veh_smoothed_tracks.csv", index=False)
    # Pedestrian reaches y=0 at x=10 at t=6.0 s; car A passes x=10 at t=4.0 s.
    rows = [dict(track_id=1, frame_id=k, timestamp_ms=int(round(t[k] * 1000)),
                 agent_type="pedestrian", x=10.0, y=-8.4 + 1.4 * t[k], vx=0.0, vy=1.4)
            for k in range(len(t))]
    pd.DataFrame(rows).to_csv(folder / "Ped_smoothed_tracks.csv", index=False)


def test_end_to_end_on_synthetic(tmp_path):
    _synthetic_recording(tmp_path)
    df = load_recording(tmp_path, verbose=False)
    pet = scan_pet(df)
    cars = pet[(pet.uid_a == "V_1") & (pet.uid_b == "V_2")]
    ped = pet[(pet.uid_a == "P_1") & (pet.uid_b == "V_1")]
    # Check both PETs against a dense (0.1 ms) evaluation of the same zone definition:
    # time from the first agent leaving the disc of radius r around the conflict point C
    # to the second agent entering it.
    tt = np.arange(0, 10, 1e-4)

    def dense_pet(row, pa, pb, r):
        C = np.array([row.x, row.y])
        da, db = np.hypot(*(pa(tt) - C).T), np.hypot(*(pb(tt) - C).T)
        ta, tb = tt[da <= r], tt[db <= r]
        first_a = ta[np.argmin(da)] <= tb[np.argmin(db)]
        return max((tb.min() - ta.max()) if first_a else (ta.min() - tb.max()), 0.0)

    car_a = lambda t: np.c_[-30 + 10 * t, 0 * t]
    car_b = lambda t: np.c_[0 * t, -45 + 10 * t]
    ped_p = lambda t: np.c_[10 + 0 * t, -8.4 + 1.4 * t]
    assert abs(cars.pet.iloc[0] - dense_pet(cars.iloc[0], car_a, car_b, 1.8)) < 0.01
    assert abs(ped.pet.iloc[0] - dense_pet(ped.iloc[0], ped_p, car_a, 0.5 * (0.5 + 1.8))) < 0.01
    assert cars.pet.iloc[0] < 1.5 and ped.pet.iloc[0] < 2.0     # still conflicts at the thresholds
    # Rear-end: centres 20 m apart, closing at 5 m/s, bumpers touch at 4.5 m -> 3.1 s.
    t = ttc_rect(z, z, z, car["l"], car["w"], 5 * o, z, 20 * o, z, z, car["l"], car["w"], z, z)
    assert np.isclose(t[0], 3.1, atol=0.01)
    # Side by side with 0.5 m clearance, one overtaking: never collides
    # (the disc screen reports overlap here, which the refinement must undo).
    t = ttc_rect(z, z, z, car["l"], car["w"], 12 * o, z, z, 2.3 * o, z, car["l"], car["w"], 10 * o, z)
    assert np.isinf(t[0])
    # Disc TTC is a lower bound on the rectangle TTC.
    n, r = disc_geometry([4.5, 4.5], [1.8, 1.8], [False, False])
    c = disc_centres(np.array([0.0, 0.0]), np.array([0.0, 2.3]), np.zeros(2), np.array([4.5, 4.5]), n)
    assert ttc_multidisc(c[[0]], c[[1]], np.array([[12.0, 0]]), np.array([[10.0, 0]]), r[[0]] + r[[1]])[0] == 0
    # Perpendicular crossing, checked against a brute-force 0.1 ms scan: 1.685 s.
    t = ttc_rect(-20 * o, z, z, car["l"], car["w"], 10 * o, z,
                 z, -20 * o, np.pi / 2 * o, car["l"], car["w"], z, 10 * o)
    assert np.isclose(t[0], 1.685, atol=0.003)


def test_heading_difference_wraps():
    assert np.isclose(heading_difference_deg(np.deg2rad(170), np.deg2rad(-170)), 20)


def _synthetic_recording(folder: Path):
    """Car A eastbound, car B northbound 1.5 s behind A at the crossing point,
    a pedestrian crossing A's path 2.0 s after A, and car C closing on A from behind."""
    t = np.arange(0, 10, 0.1)
    rows = []
    for tid, x, y, vx, vy, h in [
        (1, -30 + 10 * t, 0 * t, 10, 0, 0.0),
        (2, 0 * t, -45 + 10 * t, 0, 10, np.pi / 2),
        (3, -50 + 15 * t, 0 * t, 15, 0, 0.0),  # closes on car 1 from behind
    ]:
        for k in range(len(t)):
            rows.append(dict(track_id=tid, frame_id=k, timestamp_ms=int(round(t[k] * 1000)),
                             agent_type="car", x=x[k], y=y[k], vx=vx, vy=vy,
                             yaw_rad=h, heading_rad=h, length=4.5, width=1.8))
    pd.DataFrame(rows).to_csv(folder / "Veh_smoothed_tracks.csv", index=False)
    # Pedestrian reaches y=0 at x=10 at t=6.0 s; car A passes x=10 at t=4.0 s.
    rows = [dict(track_id=1, frame_id=k, timestamp_ms=int(round(t[k] * 1000)),
                 agent_type="pedestrian", x=10.0, y=-8.4 + 1.4 * t[k], vx=0.0, vy=1.4)
            for k in range(len(t))]
    pd.DataFrame(rows).to_csv(folder / "Ped_smoothed_tracks.csv", index=False)


def test_end_to_end_on_synthetic(tmp_path):
    _synthetic_recording(tmp_path)
    df = load_recording(tmp_path, verbose=False)
    pet = scan_pet(df)
    cars = pet[(pet.uid_a == "V_1") & (pet.uid_b == "V_2")]
    ped = pet[(pet.uid_a == "P_1") & (pet.uid_b == "V_1")]
    # Centre-crossing gaps are 1.5 s (cars) and 2.0 s (pedestrian). The size-aware PET is
    # shorter: analytically 1.5 - 1.8*sqrt(2)/10 = 1.245 s for the cars and
    # (6 - 1.15/1.4) - (4 + 1.15/10) = 1.064 s for the pedestrian. Sampling at 10 Hz
    # can add up to one 0.1 s sample on each side of the disc (0.2 s in total).
    # Rear-end: centres 20 m apart, closing at 5 m/s. Disc screen: each car is 3 discs
    # of radius sqrt(0.75^2 + 0.9^2), end discs 1.5 m from the centre, so discs touch at
    # 2*1.5 + 2*1.1715 = 5.343 m -> (20 - 5.343)/5. Exact rectangles touch at 4.5 m -> 3.1 s.
    frames = scan_ttc(df)
    first = frames[(frames.tkey == 0) & (frames.uid_a == "V_1") & (frames.uid_b == "V_3")]
    assert np.isclose(first.ttc_disc.iloc[0], (20 - 5.343) / 5, atol=0.01)
    assert np.isclose(first.ttc.iloc[0], 3.1, atol=0.01)
    assert set(frames.interaction) == {"vehicle-vehicle"}  # no other collision courses
    pair_ttc = aggregate_ttc(frames)
    assert pair_ttc.set_index(["uid_a", "uid_b"]).loc[("V_1", "V_3"), "geometry"] == "same_direction"
    # Car C actually runs into car A within 3 s of the minimum-TTC moment.
    pair_ttc = realized_gap(df, pair_ttc)
    assert pair_ttc.set_index(["uid_a", "uid_b"]).loc[("V_1", "V_3"), "min_gap"] == 0
    pair_ttc["recording"], pet["recording"] = "synthetic", "synthetic"
    tab = summarise(pair_ttc, pet)
    assert "PET<2.0s" in tab.columns

    # Plausibility: car C drives straight through car A, so their footprints overlap
    # for seconds; that pair must be removed. Nothing in this scene is static.
    static = static_agents(df)
    assert static == set()
    pair_ttc = flag_plausibility(df, pair_ttc, static)
    pet = flag_plausibility(df, pet, static)
    row = pair_ttc.set_index(["uid_a", "uid_b"]).loc[("V_1", "V_3")]
    assert row["overlap_s"] > 1.0 and not row["plausible"]
    assert pet["plausible"].all()       # crossings 1-2 s apart never overlap
    groups = summarise_groups(pair_ttc, pet)
    assert groups.loc[("All recordings", "vehicle-vehicle"), "TTC<1.5s"] == 0
    raw = summarise_groups(pair_ttc, pet, plausible_only=False)
    assert raw.loc[("All recordings", "vehicle-vehicle"), "TTC<1.5s"] == 1
    # Cars A and C both pass the pedestrian's crossing point at t = 4.0 s.
    assert groups.loc[("All recordings", "VRU-vehicle"), "PET<2.0s"] == 2
    rep = exclusion_report(pair_ttc, pet)
    assert rep.loc[("All recordings", "vehicle-vehicle"), ("TTC<1.5s", "overlap")] == 1


def test_slow_pairs_skipped():
    """Two cars creeping together at 0.3 m/s relative speed are not scanned."""
    t = np.arange(0, 3, 0.1)
    df = pd.DataFrame({
        "uid": np.repeat(["V_1", "V_2"], len(t)), "category": "vehicle",
        "tkey": np.tile(np.round(t * 1000).astype(int), 2),
        "x": np.concatenate([0.15 * t, 6.0 - 0.15 * t]), "y": 0.0,
        "vx": np.repeat([0.15, -0.15], len(t)), "vy": 0.0,
        "heading": np.repeat([0.0, np.pi], len(t)), "length": 4.5, "width": 1.8,
    })
    assert scan_ttc(df, min_rel_speed=1.0).empty
    assert not scan_ttc(df, min_rel_speed=0.0).empty


def test_static_agents():
    """A parked car (jitter only) is static; a car that stops in a queue but drives
    in and out of view is not."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 0.1)
    tk = np.round(t * 1000).astype(int)
    parked = pd.DataFrame({"uid": "V_parked", "tkey": tk, "x": 5 + 0.05 * rng.standard_normal(len(t)),
                           "y": 2 + 0.05 * rng.standard_normal(len(t))})
    queued_x = np.concatenate([np.linspace(-30, 0, 50), np.zeros(100), np.linspace(0, 30, 50)])
    queued = pd.DataFrame({"uid": "V_queued", "tkey": tk, "x": queued_x, "y": 0.0})
    assert static_agents(pd.concat([parked, queued])) == {"V_parked"}
    # Robust to position noise: 1 m mean jitter stretches the parked car's raw extent far
    # past 3 m, but it must still be recognised as static.
    noisy = parked.assign(x=parked["x"] + rng.normal(0, 0.8, len(t)),
                          y=parked["y"] + rng.normal(0, 0.8, len(t)))
    raw = np.hypot(np.ptp(noisy["x"]), np.ptp(noisy["y"]))
    assert raw > 3.0
    assert static_agents(pd.concat([noisy, queued])) == {"V_parked"}
