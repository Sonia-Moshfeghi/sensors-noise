"""COSMOS statistics on fake scenes in the release format, with known gaps and breaks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.cosmos_stats import cosmos_rates, load_scenes  # noqa: E402

STEP = 12.0          # 30 fps frame ids sampled at 2.5 fps
BASE = 5.105444e10


def _scene(path):
    rows = []
    f = lambda k: BASE + k * STEP
    # Pedestrian 1: full 20 frames, frames 5 and 6 missing, walking across the middle.
    for k in range(20):
        if k not in (5, 6):
            rows.append([f(k), 1, 1, 300 + 5 * k, 300])
    # Pedestrian 2: appears at frame 8 in the middle of the image (interior start).
    for k in range(8, 20):
        rows.append([f(k), 2, 1, 400, 250 + 5 * k])
    # Pedestrian 3: ends at frame 10 at the image edge (not an interior break).
    for k in range(0, 11):
        rows.append([f(k), 3, 1, 20 + 0.1 * k, 300])
    # Vehicle 4: full 20 frames, no gaps. Edge anchors define the image extent.
    for k in range(20):
        rows.append([f(k), 4, 0, 100 + 30 * k, 450])
    rows += [[f(0), 9, 0, 0, 0], [f(19), 10, 0, 1000, 600]]   # single-frame corner anchors
    np.save(path, np.array(rows, dtype=float))


def test_rates_on_known_scene(tmp_path):
    _scene(tmp_path / "cosmos_231212_l20_f2p5_1.npy")
    r = cosmos_rates(load_scenes(tmp_path))
    assert r.attrs["frame_step"] == STEP
    ped = r.loc["pedestrian"]
    # Observed spans: 20 + 12 + 11 = 43 frames, 2 missing.
    assert np.isclose(ped["missed_frame_fraction"], 2 / 43)
    assert np.isclose(ped["mean_gap_s"], 2 / 2.5)
    assert np.isclose(ped["interior_start_share"], 1 / 3)
    assert ped["interior_end_share"] == 0
    assert r.loc["vehicle", "missed_frame_fraction"] == 0


def test_gap_free_release_is_flagged(tmp_path):
    rows = [[BASE + k * STEP, 1, 1, 300 + k, 300] for k in range(20)]
    rows += [[BASE, 9, 0, 0, 0], [BASE + 19 * STEP, 10, 0, 1000, 600]]
    np.save(tmp_path / "s.npy", np.array(rows, dtype=float))
    r = cosmos_rates(load_scenes(tmp_path))
    assert r["missed_frame_fraction"].isna().all() and "note" in r.columns


def test_combined_split_files_are_skipped(tmp_path):
    _scene(tmp_path / "cosmos_231212_l20_f2p5_51054441684.npy")
    _scene(tmp_path / "cosmos_231212_l20_f2p5_val.npy")      # combined file: must be ignored
    df = load_scenes(tmp_path)
    assert df["scene"].nunique() == 1
