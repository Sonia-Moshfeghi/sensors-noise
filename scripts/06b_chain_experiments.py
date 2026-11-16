"""Cleaning methods applied to tracker-like output (the realistic chain
detections -> causal tracker -> post-hoc cleaning), for jitter 0.3/1.0 m, drift 0.5 m and
15 % missed detections, two seeds. Results land in the same results/03_filters tables
with method names "tracker+<method>". Also reports the per-track noise estimate on
tracker output versus raw detections (the hybrid gate input).

    python scripts/06b_chain_experiments.py --data-root PATH/TO/sind --jobs 4
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: E402,F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.filter_pipeline import CHAIN_GRID, CHAIN_METHODS, run_filter_experiments  # noqa: E402
from sensors_noise.additional_analyses import sigma_hat_tracker  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results/03_filters")
    ap.add_argument("--models", default="results/03_filters/models")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    run_filter_experiments(a.data_root, a.out, models_dir=a.models, grid=CHAIN_GRID, seeds=(0, 1),
                           methods=CHAIN_METHODS, n_jobs=a.jobs, make_plots=False)
    print("\nPer-track noise estimate (sigma_hat, m) on raw detections and on tracker-like output:")
    print(sigma_hat_tracker(a.data_root, a.models).to_string())
