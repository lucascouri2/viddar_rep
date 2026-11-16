"""Evaluate information manipulation attack detection (paper Table 2) on a manifest.

Examples:
    python scripts/run_manipulation.py --vlm qwen3-vl-2b
    python scripts/run_manipulation.py --vlm qwen3-vl-2b --prompt notebook --limit 10
    python scripts/run_manipulation.py --vlm qwen3-vl-4b --prompt definition --mode decomposed

Prompt variants (viddar/prompts.py): paper (literal), notebook, neutral (A1), definition (A3).
--mode decomposed asks one question per call and computes M = A and S and I in code (A2).

Outputs go to <out>/<vlm>/<prompt>[_decomposed]/{predictions.csv, metrics.json, metrics_by_group.csv}.
The same runner is available from Python as `viddar.experiments.run_manipulation`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viddar import manipulation, prompts  # noqa: E402
from viddar.experiments import ManipulationConfig, manipulation_table, run_manipulation  # noqa: E402
from viddar.metrics import flatten  # noqa: E402


def main() -> None:
    d = ManipulationConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=d.manifest)
    ap.add_argument("--vlm", default=d.vlm, help="preset name (see viddar/vlm.py) or HF model id")
    ap.add_argument("--prompt", choices=list(prompts.MANIPULATION), default=d.prompt)
    ap.add_argument("--mode", choices=["joint", "decomposed"], default=d.mode,
                    help="joint = the paper's six-question prompt; decomposed = one call per factor")
    ap.add_argument("--max-pixels", type=int, default=d.max_pixels, help="each image is downscaled to this area")
    ap.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=d.temperature, help="0 = greedy decoding")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--limit", type=int, default=d.limit)
    ap.add_argument("--out", default=d.out)
    ap.add_argument("--cache-dir", default=d.cache_dir)
    args = ap.parse_args()

    result = run_manipulation(ManipulationConfig(**vars(args)))
    print(f"\n{args.vlm} [{result.run_dir.name}]")
    print(manipulation_table(result).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for f_ in manipulation.FACTOR_QUESTIONS:
        k = f"factor_{f_}"
        if k in result.summary:
            fm = flatten(result.summary[k])
            print(f"  factor {f_}: acc={fm['accuracy']:.4f}  predicted-positive rate={fm['positive_rate']:.4f}")
    print(f"  results in {result.run_dir}")


if __name__ == "__main__":
    main()
