"""Collect finished runs (metrics.json files) into tables shaped like the paper's Tables 1 and 2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Reference numbers from the paper (GPT-4o / LLaVA-Next-8b, Grounding DINO 1.5, SAM ViT-B).
PAPER_TABLE1 = pd.DataFrame([
    ("viddar", "GPT-4o (paper)", 91.83, 72.15, 92.15), ("viddar", "LLaVA-Next-8b (paper)", 85.95, 71.85, 89.21),
    ("end_to_end", "GPT-4o (paper)", 91.83, None, 81.04), ("end_to_end", "LLaVA-Next-8b (paper)", 85.95, None, 55.23),
    ("underdetailed", "GPT-4o (paper)", 86.60, 70.21, 86.92),
    ("underdetailed", "LLaVA-Next-8b (paper)", 76.47, 61.58, 81.37),
    ("greedy", "GPT-4o (paper)", 93.14, 67.93, 88.89), ("greedy", "LLaVA-Next-8b (paper)", 88.89, 64.84, 85.62),
    ("saliency", "- (paper)", None, None, 51.63), ("canny", "- (paper)", None, None, 51.96),
    ("prior", "- (paper)", 100.0, 78.83, 93.14),
], columns=["method", "vlm", "key_obj_acc", "mIoU", "accuracy"])

PAPER_TABLE2 = pd.DataFrame([
    ("GPT-4o (paper)", "paper", 82.46, 74.00, 84.09),
    ("LLaVA-Next-8b (paper)", "paper", 42.98, 39.80, 95.35),
], columns=["vlm", "prompt", "accuracy", "precision", "recall"])

METHOD_ORDER = ["viddar", "end_to_end", "underdetailed", "greedy", "saliency", "canny", "prior"]


def pct(x):
    return None if x is None or pd.isna(x) else round(100 * x, 2)


def ci(m: dict) -> str:
    lo, hi = m["ci95"]
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]" if m["n"] else "-"


def collect_obstruction(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("obstruction/*/*/metrics.json")):
        m = json.loads(f.read_text())
        d = m["detection"]
        rows.append({
            "method": m["method"], "vlm": f.parts[-3], "variant": f.parts[-2],
            "key_obj_acc": pct(m.get("key_object_recognition_accuracy")),
            "key_obj_grounded": pct(m.get("key_object_grounded_accuracy")),
            "mIoU": pct(m.get("segmentation_miou")),
            "accuracy": pct(d["accuracy"]["value"]), "acc_ci95": ci(d["accuracy"]),
            "precision": pct(d["precision"]["value"]), "recall": pct(d["recall"]["value"]),
            "fpr": pct(d["fpr"]["value"]), "fp": d["fp"], "fn": d["fn"], "no_answer": d["no_answer"],
            "alpha": m["alpha"],
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["order"] = df["method"].map(METHOD_ORDER.index)
        df = df.sort_values(["order", "vlm", "variant"]).drop(columns="order")
    return df


def collect_manipulation(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("manipulation/*/*/metrics.json")):
        m = json.loads(f.read_text())
        d, r = m["detection"], m["detection_factor_rule"]
        row = {
            "vlm": f.parts[-3], "prompt": f.parts[-2],
            "accuracy": pct(d["accuracy"]["value"]), "acc_ci95": ci(d["accuracy"]),
            "precision": pct(d["precision"]["value"]), "recall": pct(d["recall"]["value"]),
            "fpr": pct(d["fpr"]["value"]), "fp": d["fp"], "fn": d["fn"], "no_answer": d["no_answer"],
            "acc_AandSandI": pct(r["accuracy"]["value"]), "fpr_AandSandI": pct(r["fpr"]["value"]),
            "q6_consistent_with_q3-5": pct(m["final_answer_consistent_with_factors"]),
        }
        for fac in "ASI":
            if f"factor_{fac}" in m:
                row[f"acc_{fac}"] = pct(m[f"factor_{fac}"]["accuracy"]["value"])
        rows.append(row)
    return pd.DataFrame(rows)
