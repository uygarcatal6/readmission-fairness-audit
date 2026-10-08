"""
scripts/make_figures.py — every figure of the paper and the slides from the saved results.

    python scripts/make_figures.py                 # reads outputs/approved/*.json
    python scripts/make_figures.py --results DIR   # another folder with the five JSONs

Writes paper/figures/<name>.pdf (vector, for the IEEE paper) and paper/figures/<name>.png
(200 dpi, for quick viewing). No model is fitted: the figures only read the five summary JSONs
(readmission/figures.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from readmission import figures  # noqa: E402

PNG_DPI = 200


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", default=str(ROOT), help="repository root or results folder")
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = figures.load_results(args.results)
    for name, fig in figures.all_figures(results).items():
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=PNG_DPI)
        print(f"wrote {out / name}.pdf / .png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
