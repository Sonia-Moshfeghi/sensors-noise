"""Step 4b: proactive conflict detection from predicted trajectories under noise.

    python scripts/08_predictor_experiments.py --data-root PATH/TO/sind
Single process (pair-level evaluation is light); resumable per condition.
    --collect  rebuilds tables and plots from finished conditions.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.predictor_pipeline import (collect_predictor_results,  # noqa: E402
                                              run_predictor_experiments)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results/04_predictor")
    ap.add_argument("--models", default="results/04_predictor/models")
    ap.add_argument("--denoiser", default="results/03_filters/models")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect_predictor_results(a.out)
    else:
        run_predictor_experiments(a.data_root, a.out, a.models, a.denoiser, seeds=tuple(range(a.seeds)))
