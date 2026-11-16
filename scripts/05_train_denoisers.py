"""Step 3a: train the learned denoisers (two variants x two agent folds = 4 small models).

    python scripts/05_train_denoisers.py --data-root PATH/TO/sind

Uses the GPU if PyTorch sees one. Models are written to results/03_filters/models/.
Variants: 'jitter' (trained on Gaussian jitter only) and 'mixed' (noise-aware: jitter,
distance jitter, drift, offset, missed detections). Training noise uses seeds disjoint
from the evaluation seeds, and each agent is later cleaned by the model of the other fold.
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # see sensors_noise/denoiser.py
import torch  # noqa: E402,F401  (must load before NumPy/SciPy on Windows)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.denoiser import train_variant  # noqa: E402
from sensors_noise.io_sind import find_recordings, load_recording  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results/03_filters/models")
    ap.add_argument("--windows", type=int, default=300_000, help="training windows per variant")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--variants", nargs="+", default=["jitter", "mixed"])
    a = ap.parse_args()

    import torch
    print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    recs = [load_recording(r, verbose=False) for r in find_recordings(a.data_root)]
    for v in a.variants:
        train_variant(recs, v, a.out, total_windows=a.windows, epochs=a.epochs, device=a.device)
    print(f"Models written to {Path(a.out).resolve()}")
