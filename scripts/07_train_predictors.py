"""Step 4a: train the trajectory-prediction ensembles (2 variants x 2 folds x 5 members).

    python scripts/07_train_predictors.py --data-root PATH/TO/sind
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402,F401  (must load before NumPy/SciPy on Windows)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.io_sind import find_recordings, load_recording  # noqa: E402
from sensors_noise.predictor import train_variant  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results/04_predictor/models")
    ap.add_argument("--windows", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--variants", nargs="+", default=["clean", "noise_aware"])
    a = ap.parse_args()
    print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    recs = [load_recording(r, verbose=False) for r in find_recordings(a.data_root)]
    for v in a.variants:
        train_variant(recs, v, a.out, total_windows=a.windows, epochs=a.epochs, device=a.device)
    print(f"Models written to {Path(a.out).resolve()}")
