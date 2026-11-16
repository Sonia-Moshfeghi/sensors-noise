"""Regenerate the paper's figures in print form from the results folders.

    python scripts/09_paper_figures.py --data-root sind --out paper_figures

Writes PNG (300 dpi) and PDF versions:
  fig_events                  2x2 example events (step 1 results + SinD data)
  fig_noise_{ttc,pet}_*       step-2 propagation, split into position / detection noise
  fig_filters_{ttc,pet}_*     step-3 cleaning, split into position noise / offset+missed
Figures whose results folder is missing are skipped with a message.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.paper_figures import fig_events, fig_filters, fig_noise  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_figures")
    ap.add_argument("--event", action="append", default=[], metavar="LETTER:RECORDING:UID_A:UID_B",
                    help="override a panel of fig_events, e.g. a:Changchun:V_1224:V_1253 (repeatable)")
    a = ap.parse_args()
    overrides = {}
    for spec in a.event:
        letter, rec, ua, ub = spec.split(":")
        overrides[letter] = (rec, ua, ub)
    R = Path(a.results)
    if (R / "01_conflict_check" / "pair_ttc.csv").exists():
        path, sel = fig_events(R / "01_conflict_check", a.data_root, a.out, overrides=overrides)
        print("fig_events:", path)
    else:
        print("step-1 results not found; skipping fig_events")
    for sub, fn in (("02_noise", fig_noise), ("03_filters", fig_filters)):
        if (R / sub / ("detection.csv" if sub == "02_noise" else "detection_pooled.csv")).exists():
            for measure, tag in (("TTC<3.0s", "ttc"), ("PET<2.0s", "pet")):
                print(sub, measure, fn(R / sub, a.out, measure, tag))
        else:
            print(f"{sub} results not found; skipping")
