"""Additional analyses (part 2) from the saved results: removal of close pairs by the
plausibility criteria (analytical-model check), severity shifts by direction, the channel
by which heading error reaches PET, and the hybrid's routing shares (incl. on tracker
output once 06b has run). Rerun 10_additional_analyses.py afterwards to refresh the
bootstrap intervals (now cluster bootstrap) in the detection tables:

    python scripts/10b_additional_analyses.py --results results
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.additional_analyses import run_part2  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    run_part2(a.results)
