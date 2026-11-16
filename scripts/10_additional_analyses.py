"""Additional analyses from the saved results (no reruns needed):
severity-shift table, near-threshold share of false conflicts, TTC bias over all reference
pairs, detection against the SinD-native-kinematics reference, composition of false
conflicts, and the analytical constant-velocity TTC error model vs. simulation.

    python scripts/10_additional_analyses.py --results results
Writes results/additional_analyses/*.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.additional_analyses import run_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_all(a.results, a.out)
