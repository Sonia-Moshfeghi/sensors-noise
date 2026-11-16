"""Plot a labelled set of flagged conflict events for visual checking.

Run after scripts/01_conflict_check.py:
    python scripts/02_plot_events.py --data-root PATH/TO/sind --results results/01_conflict_check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.events import plot_selected_events  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--results", default="results/01_conflict_check")
    ap.add_argument("--n", type=int, default=3, help="events per category")
    a = ap.parse_args()
    plot_selected_events(a.data_root, a.results, n=a.n)
