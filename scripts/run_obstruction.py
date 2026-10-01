"""Evaluate obstruction attack detection (paper Table 1) on a manifest.

Examples:
    python scripts/run_obstruction.py --methods viddar prior --vlm qwen3-vl-2b
    python scripts/run_obstruction.py --methods all --vlm qwen3-vl-2b --limit 20
    python scripts/run_obstruction.py --methods saliency canny          # no VLM/GPU needed

Outputs go to <out>/<vlm>/<method>/{predictions.csv, metrics.json, metrics_by_group.csv}; VLM
answers are cached in <cache-dir>, so re-running (e.g. with another alpha) does not query the
model again. The same runner is available from Python as `viddar.experiments.run_obstruction`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viddar import obstruction as obs  # noqa: E402
from viddar.experiments import ObstructionConfig, obstruction_table, run_obstruction  # noqa: E402


def main() -> None:
    d = ObstructionConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=d.manifest)
    ap.add_argument("--methods", nargs="+", default=d.methods, help=f"any of {obs.ALL_METHODS} or 'all'")
    ap.add_argument("--vlm", default=d.vlm, help="preset name (see viddar/vlm.py) or HF model id")
    ap.add_argument("--alpha", type=float, default=d.alpha, help="obstruction threshold of Eq. 4 (paper: 0.25)")
    ap.add_argument("--key-prompt", choices=["paper", "notebook"], default=d.key_prompt)
    ap.add_argument("--max-pixels", type=int, default=d.max_pixels, help="images are downscaled to this area")
    ap.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=d.temperature, help="0 = greedy decoding")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--detector", default=d.detector)
    ap.add_argument("--sam", default=d.sam)
    ap.add_argument("--box-threshold", type=float, default=d.box_threshold)
    ap.add_argument("--text-threshold", type=float, default=d.text_threshold)
    ap.add_argument("--all-boxes", action="store_true", help="union of all boxes instead of the top-1 box")
    ap.add_argument("--limit", type=int, default=d.limit, help="evaluate only the first N samples")
    ap.add_argument("--out", default=d.out)
    ap.add_argument("--cache-dir", default=d.cache_dir)
    args = ap.parse_args()

    try:
        results = run_obstruction(ObstructionConfig(**vars(args)))
    except ValueError as e:
        ap.error(str(e))
    print()
    print(obstruction_table(results).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for r in results.values():
        print(f"  {r.name}: {r.run_dir}")


if __name__ == "__main__":
    main()
