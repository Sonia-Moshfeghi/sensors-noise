"""Step 3 (before 06): tune the classical cleaning methods and the hybrid gate on
training-seed noise. Run after 05_train_denoisers.py (the hybrid gate needs the mixed
denoiser); without the models only savgol/kalman/tracker are tuned.

    python scripts/05b_tune_classical.py --data-root PATH/TO/sind
Writes results/03_filters/models/tuning.json, used automatically by 06.
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
from sensors_noise.io_sind import find_recordings, load_recording  # noqa: E402
from sensors_noise.tuning import tune_classical  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--models", default="results/03_filters/models")
    ap.add_argument("--agents", type=int, default=250, help="max agents per recording used for tuning")
    a = ap.parse_args()
    recs = [load_recording(r, verbose=False) for r in find_recordings(a.data_root)]
    tune_classical(recs, a.models, n_agents=a.agents)
