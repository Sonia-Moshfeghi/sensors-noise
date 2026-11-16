"""Conflict inventory: how many candidate conflicts do the clean SinD trajectories contain?

Usage:
    python scripts/01_conflict_check.py --data-root PATH/TO/SinD --out results/01_conflict_check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.pipeline import run_conflict_check  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="folder containing SinD recordings")
    ap.add_argument("--out", default="results/01_conflict_check")
    ap.add_argument("--max-recordings", type=int, default=None)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="scan every k-th frame for TTC (1 = all frames)")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="frames below a TTC threshold needed to count a conflict")
    ap.add_argument("--min-rel-speed", type=float, default=1.0,
                    help="skip pairs closing slower than this [m/s]")
    ap.add_argument("--gap", type=float, default=1.0,
                    help="max actual footprint gap [m] for a TTC conflict")
    a = ap.parse_args()
    run_conflict_check(a.data_root, a.out, a.max_recordings, a.frame_stride, a.min_frames,
                       min_rel_speed=a.min_rel_speed, gap_threshold=a.gap)
