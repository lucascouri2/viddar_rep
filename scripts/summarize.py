"""Collect every metrics.json under results/ into tables shaped like the paper's Tables 1 and 2.

Usage:
    python scripts/summarize.py [--results results] [--out results/summary.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viddar.reporting import (PAPER_TABLE1, PAPER_TABLE2, collect_manipulation,  # noqa: E402
                              collect_obstruction)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.md")
    args = ap.parse_args()
    root = Path(args.results)

    obs, man = collect_obstruction(root), collect_manipulation(root)
    parts = ["# ViDDAR replication summary", "",
             "All numbers in %. `acc_ci95` = Wilson 95% interval. FPR = FP / (FP + TN).", ""]
    parts += ["## Obstruction detection (Table 1)", "", "### This replication", ""]
    parts += [obs.to_markdown(index=False) if len(obs) else "_no runs yet_", ""]
    parts += ["### Paper", "", PAPER_TABLE1.to_markdown(index=False), ""]
    parts += ["## Information manipulation detection (Table 2)", "", "### This replication", ""]
    parts += [man.to_markdown(index=False) if len(man) else "_no runs yet_", ""]
    parts += ["### Paper", "", PAPER_TABLE2.to_markdown(index=False), ""]
    text = "\n".join(parts)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
