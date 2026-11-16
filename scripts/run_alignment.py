"""Geometric alignment A_geo on a finished manipulation run (path C2 of report/analise_fp_fn_manipulacao.md).

Reuses the VLM answers saved in the run (key object of question 2, factors S and I); only Grounding
DINO + SAM run, so no VLM is loaded.

Examples:
    python scripts/run_alignment.py --run-dir results/manipulation/llava-next-8b-4bit/paper
    python scripts/run_alignment.py --run-dir results/manipulation/qwen3-vl-4b/definition_decomposed

Outputs go to <run-dir>/alignment/{features.csv, predictions.csv, metrics.json, key_masks/}. The same
runner is available from Python as `viddar.experiments.run_alignment`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viddar.alignment import FEATURES  # noqa: E402
from viddar.experiments import AlignmentConfig, run_alignment  # noqa: E402


def main() -> None:
    d = AlignmentConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=d.run_dir, help="folder of a manipulation run")
    ap.add_argument("--manifest", default=d.manifest)
    ap.add_argument("--feature", choices=list(FEATURES), default=d.feature)
    ap.add_argument("--contact-frac", type=float, default=d.contact_frac)
    ap.add_argument("--cv-folds", type=int, default=d.cv_folds)
    ap.add_argument("--cv-repeats", type=int, default=d.cv_repeats)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--detector", default=d.detector)
    ap.add_argument("--sam", default=d.sam)
    ap.add_argument("--box-threshold", type=float, default=d.box_threshold)
    ap.add_argument("--text-threshold", type=float, default=d.text_threshold)
    args = ap.parse_args()

    result = run_alignment(AlignmentConfig(**vars(args)))
    s = result.summary
    print(f"\n{args.run_dir}: objects detected in {s['detected']}/{s['n']} pairs")
    for row in s["feature_auc"]:
        print(f"  AUC {row['feature']:>9}: {row['auc']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]")
    for name in ("base", "hybrid"):
        r = s[name]
        print(f"  {name:>6}: FPR={r['fpr']['value']:.3f} ({r['fpr']['k']}/{r['fpr']['n']})  "
              f"FNR={r['fnr']['value']:.3f} ({r['fnr']['k']}/{r['fnr']['n']})")
    m = s["mcnemar_hybrid_vs_base"]
    print(f"  McNemar hybrid vs base: only hybrid right={m['a_only']}, only base right={m['b_only']}, "
          f"p={m['p_value']:.4f}")
    print(f"  results in {result.run_dir}")


if __name__ == "__main__":
    main()
