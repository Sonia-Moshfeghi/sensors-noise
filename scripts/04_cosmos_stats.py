"""Missed-detection and track-break rates in the public COSMOS release (NYC), for
comparison with the dropout and id_switch levels of the noise grid.

    python scripts/04_cosmos_stats.py --cosmos-root PATH/TO/cosmos-trajectory --sind-root PATH/TO/sind

--cosmos-root: folder of the Hugging Face release (searched recursively for .npy scenes).
--sind-root (optional): converts the id_switch levels into breaks per track-minute.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors_noise.cosmos_stats import cosmos_rates, load_scenes, sind_switch_equivalents  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cosmos-root", required=True)
    ap.add_argument("--sind-root", default=None)
    ap.add_argument("--out", default="results/03_cosmos")
    ap.add_argument("--max-files", type=int, default=None, help="limit scenes (quick test)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_scenes(a.cosmos_root, a.max_files)
    rates = cosmos_rates(df)
    rates.to_csv(out / "cosmos_rates.csv")
    pd.set_option("display.width", 200)
    print(f"Frame-id step between samples: {rates.attrs['frame_step']:g}")
    print("\nCOSMOS (NYC) missed detections and track breaks:")
    print(rates.round(4).T.to_string())
    print("\nNoise grid for comparison: dropout levels 0.05 / 0.15 / 0.30 missed-frame fraction.")
    if a.sind_root:
        eq = sind_switch_equivalents(a.sind_root)
        eq.to_csv(out / "id_switch_equivalents.csv")
        m = eq.attrs["mean_track_minutes"]
        print("SinD mean track duration [min] and track counts by class:")
        for c in ("pedestrian", "two_wheeler", "vehicle", "VRU"):
            print(f"   {c:12s} {m[c]:.2f} min" + (f"  ({eq.attrs['n_tracks'].get(c, '')} tracks)" if c != "VRU" else "  (pedestrians + two-wheelers)"))
        print("Fragmentation levels as breaks per track-minute:")
        print(eq.round(3).to_string())
    print(f"\nOutputs written to {out.resolve()}")
