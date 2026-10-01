"""Experiment runners shared by the command-line scripts and the notebook.

Each runner evaluates a manifest, writes <out>/<vlm>/<variant>/{predictions.csv, metrics.json,
metrics_by_group.csv} and returns the results in memory.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import manipulation
from . import obstruction as obs
from .data import load_manifest
from .metrics import flatten, metrics_by_group

OBSTRUCTION_KEEP_COLS = ["id", "group", "key_object", "label", "pred", "method", "key_phrase", "key_recognized",
                         "n_detections", "det_score", "ratio", "key_iou", "score_obstructed", "score_image",
                         "vlm_response", "e2e_response"]


@dataclass
class ObstructionConfig:
    manifest: str = "data/manifests/viddar_obstruction.csv"
    methods: list[str] = field(default_factory=lambda: ["viddar"])  # or ["all"]
    vlm: str = "qwen3-vl-2b"
    alpha: float = 0.25  # Eq. 4 threshold (paper: 0.25)
    key_prompt: str = "paper"  # "paper" | "notebook"
    max_pixels: int = 1_000_000
    max_new_tokens: int = 256
    temperature: float = 0.0  # 0 = greedy decoding
    seed: int = 0
    detector: str = "IDEA-Research/grounding-dino-base"
    sam: str = "facebook/sam-vit-base"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    all_boxes: bool = False
    limit: int | None = None
    out: str = "results/obstruction"
    cache_dir: str = "results/cache"


@dataclass
class ManipulationConfig:
    manifest: str = "data/manifests/viddar_manipulation.csv"
    vlm: str = "qwen3-vl-2b"
    prompt: str = "paper"  # "paper" | "notebook"
    max_pixels: int = 1_000_000
    max_new_tokens: int = 768
    temperature: float = 0.0
    seed: int = 0
    limit: int | None = None
    out: str = "results/manipulation"
    cache_dir: str = "results/cache"


@dataclass
class RunResult:
    name: str
    run_dir: Path
    predictions: pd.DataFrame  # full frame, including image paths
    summary: dict


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _write(run_dir: Path, predictions: pd.DataFrame, cols: list[str], summary: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions[cols].to_csv(run_dir / "predictions.csv", index=False)
    metrics_by_group(predictions).to_csv(run_dir / "metrics_by_group.csv", index=False)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


def run_obstruction(cfg: ObstructionConfig) -> dict[str, RunResult]:
    methods = obs.ALL_METHODS if cfg.methods == ["all"] else list(cfg.methods)
    unknown = set(methods) - set(obs.ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {obs.ALL_METHODS}")

    df = load_manifest(cfg.manifest, "obstruction")
    if cfg.limit:
        df = df.head(cfg.limit)

    frames: dict[str, pd.DataFrame] = {}
    # Stage 1: every VLM method first, then free the GPU for detection/segmentation.
    vlm_methods = [m for m in methods if m in obs.VLM_METHODS]
    if vlm_methods:
        from .vlm import CachedVLM

        vlm = CachedVLM(cfg.vlm, cfg.cache_dir, cfg.max_pixels, cfg.max_new_tokens, cfg.temperature, cfg.seed)
        try:
            for m in vlm_methods:
                frames[m] = obs.run_vlm_stage(df, m, vlm, cfg.key_prompt)
        finally:
            vlm.close()

    # Stage 2: Grounding DINO + SAM.
    grounding_methods = [m for m in methods if m in obs.GROUNDING_METHODS]
    if grounding_methods:
        from .grounding import GroundedSegmenter

        seg = GroundedSegmenter(cfg.detector, cfg.sam, cfg.box_threshold, cfg.text_threshold)
        try:
            for m in grounding_methods:
                frames[m] = obs.run_grounding_stage(frames.get(m, df), m, seg, top1=not cfg.all_boxes)
        finally:
            seg.close()

    # Stage 3: classical baselines.
    for m in methods:
        if m in obs.CV_METHODS:
            frames[m] = obs.run_cv_stage(df, m)

    results = {}
    for m in methods:
        uses_vlm = m in obs.VLM_METHODS
        res = obs.add_predictions(frames[m], m, cfg.alpha)
        summary = obs.summarize(res, m, cfg.alpha)
        summary["config"] = asdict(cfg) | {"vlm": cfg.vlm if uses_vlm else None}
        variant = m
        if uses_vlm and cfg.key_prompt != "paper" and m in ("viddar", "end_to_end"):
            variant += f"_{cfg.key_prompt}-prompt"
        if uses_vlm and cfg.temperature > 0:
            variant += f"_t{cfg.temperature}_s{cfg.seed}"
        run_dir = Path(cfg.out) / (_slug(cfg.vlm) if uses_vlm else "no_vlm") / variant
        _write(run_dir, res, [c for c in OBSTRUCTION_KEEP_COLS if c in res.columns], summary)
        results[m] = RunResult(m, run_dir, res, summary)
    return results


def run_manipulation(cfg: ManipulationConfig) -> RunResult:
    from .vlm import CachedVLM

    df = load_manifest(cfg.manifest, "manipulation")
    if cfg.limit:
        df = df.head(cfg.limit)

    vlm = CachedVLM(cfg.vlm, cfg.cache_dir, cfg.max_pixels, cfg.max_new_tokens, cfg.temperature, cfg.seed)
    try:
        res = manipulation.run(df, vlm, cfg.prompt)
    finally:
        vlm.close()

    summary = manipulation.summarize(res)
    summary["config"] = asdict(cfg)
    suffix = f"_t{cfg.temperature}_s{cfg.seed}" if cfg.temperature > 0 else ""
    run_dir = Path(cfg.out) / _slug(cfg.vlm) / (cfg.prompt + suffix)
    _write(run_dir, res, [c for c in res.columns if not c.endswith("_path")], summary)
    return RunResult(f"manipulation/{cfg.prompt}", run_dir, res, summary)


# ------------------------------------------------------------------------------ reporting
def obstruction_table(results: dict[str, RunResult]) -> pd.DataFrame:
    rows = []
    for m, r in results.items():
        s, d = r.summary, flatten(r.summary["detection"])
        rows.append({
            "method": m, "vlm": s["config"]["vlm"] or "-",
            "key_obj_acc": s.get("key_object_recognition_accuracy"),
            "key_obj_grounded": s.get("key_object_grounded_accuracy"),
            "mIoU": s.get("segmentation_miou"),
            "accuracy": d["accuracy"], "precision": d["precision"], "recall": d["recall"], "fpr": d["fpr"],
            "tp": d["tp"], "fp": d["fp"], "tn": d["tn"], "fn": d["fn"], "n": d["n"],
        })
    return pd.DataFrame(rows)


def manipulation_table(result: RunResult) -> pd.DataFrame:
    s = result.summary
    rows = []
    for name, key in [("final answer (paper rule)", "detection"), ("A and S and I from Q3-5", "detection_factor_rule")]:
        d = flatten(s[key])
        rows.append({"decision": name, "accuracy": d["accuracy"], "precision": d["precision"],
                     "recall": d["recall"], "fpr": d["fpr"], "tp": d["tp"], "fp": d["fp"], "tn": d["tn"],
                     "fn": d["fn"], "no_answer": d["no_answer"]})
    return pd.DataFrame(rows)


def load_run(run_dir: str | Path, manifest: str | Path | None = None, kind: str | None = None) -> RunResult:
    """Reload a finished run; with `manifest`, image/mask paths are joined back by `id`."""
    run_dir = Path(run_dir)
    preds = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str})
    with open(run_dir / "metrics.json") as f:
        summary = json.load(f)
    if manifest is not None:
        man = load_manifest(manifest, kind or ("manipulation" if "pred_A" in preds.columns else "obstruction"))
        path_cols = ["id"] + [c for c in man.columns if c.endswith("_path")]
        preds = preds.merge(man[path_cols], on="id", how="left")
    return RunResult(run_dir.name, run_dir, preds, summary)
