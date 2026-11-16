"""Step 3b: conflict detection after position cleaning (none, Savitzky-Golay, Kalman,
two learned denoisers), on the same noise realisations as step 2.

Quick check first (seed 0, one level per noise type):
    python scripts/06_filter_experiments.py --data-root PATH/TO/sind --quick --jobs 4
Full run (2 seeds, all levels; resumable, a few hours):
    python scripts/06_filter_experiments.py --data-root PATH/TO/sind --jobs 4
Rebuild tables and plots from finished tasks:
    python scripts/06_filter_experiments.py --data-root PATH/TO/sind --collect

Train the denoisers first: scripts/05_train_denoisers.py.
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # see sensors_noise/denoiser.py
import torch  # noqa: E402,F401  (must load before NumPy/SciPy on Windows)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.filter_pipeline import collect_filter_results, run_filter_experiments  # noqa: E402

QUICK_GRID = {"clean": [0.0], "jitter": [0.3, 1.0], "distance_jitter": [0.5], "drift": [0.5],
              "offset": [0.6], "dropout": [0.15]}

if __name__ == "__main__":       # required for parallel workers on Windows
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results/03_filters")
    ap.add_argument("--models", default="results/03_filters/models")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    out = a.out + "_quick" if a.quick else a.out
    if a.collect:
        collect_filter_results(out)
    else:
        run_filter_experiments(a.data_root, out, models_dir=a.models,
                               grid=QUICK_GRID if a.quick else None,
                               seeds=(0,) if a.quick else tuple(range(a.seeds)), n_jobs=a.jobs)
