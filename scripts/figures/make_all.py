"""Regenerate every thesis figure into docs/final_thesis/figures.

Cheap groups run first; the model-driven group (which retrains the V3 flow,
~1 h CPU) runs last and can be skipped with ``--skip-model``.  Figure 32
(SCADA timeline) additionally needs the external Illwerke drive and is
attempted but skipped gracefully when the archive is absent.

Run with:  python -m scripts.figures.make_all [--skip-model]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

GROUPS = [
    "scripts.figures.fig_charts",
    "scripts.figures.fig_schematics",
    "scripts.figures.fig_architecture",
    "scripts.figures.fig_geometry",
    "scripts.figures.fig_signals",
    "scripts.figures.fig_classical",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    modules = list(GROUPS)
    if not args.skip_model:
        modules.append("scripts.figures.fig_model")
    for mod in modules:
        print(f"\n=== {mod} ===")
        subprocess.run([sys.executable, "-m", mod], check=True)

    print("\n=== scripts.figures.fig_scada (needs external drive) ===")
    subprocess.run([sys.executable, "-m", "scripts.figures.fig_scada"], check=False)


if __name__ == "__main__":
    main()
