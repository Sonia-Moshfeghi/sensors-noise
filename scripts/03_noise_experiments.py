"""Step 2: controlled sensor-noise experiments on the clean SinD trajectories.

Quick check first (one seed, one level per noise type, ~20-40 min):
    python scripts/03_noise_experiments.py --data-root PATH/TO/sind --quick --jobs 4

Full grid (3 seeds, all levels; a few hours, resumable - safe to run overnight):
    python scripts/03_noise_experiments.py --data-root PATH/TO/sind --jobs 4

Finished tasks are cached in <out>/parts/, so rerunning skips them. --collect only
rebuilds the summary tables and plots from whatever has finished.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.noise_pipeline import collect_results, run_noise_experiments  # noqa: E402

QUICK_GRID = {"baseline": [0.0], "jitter": [0.3, 1.0], "distance_jitter": [0.5], "drift": [0.5],
              "offset": [0.6], "heading": [5.0], "dropout": [0.15], "id_switch": [1.0]}

if __name__ == "__main__":       # required for parallel workers on Windows
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results/02_noise")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="one seed, one level per noise type")
    ap.add_argument("--collect", action="store_true", help="only rebuild summaries from finished tasks")
    a = ap.parse_args()
    if a.collect:
        collect_results(a.out)
    else:
        out = a.out + "_quick" if a.quick else a.out
        run_noise_experiments(a.data_root, out, grid=QUICK_GRID if a.quick else None,
                              seeds=(0,) if a.quick else tuple(range(a.seeds)), n_jobs=a.jobs)
