"""Pilot: does ViDDAR's key-object step pick the traffic sign in the driving-video frames?

Runs the paper's key-object prompt on a set of frames, grounds the VLM answer with Grounding DINO
(top-1 box, as in the obstruction pipeline) and measures whether the grounded box falls on the
target box (IoU >= 0.3 or box centre inside the target box). Two frame sets are available:

  --source review      frames reviewed in data/annotations/vwise_sign_review.csv
                       (categories traffic_sign and video_overlay; target = annotated box)
  --source detections  for every segment, the frame with the highest-scoring "traffic sign"
                       detection in results/vwise/sign_dets.json with score >= --min-score
                       (target = that detection box; no manual review involved)

Usage:
    python scripts/pilot_vwise_keyobject.py --source detections --min-score 0.45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viddar import prompts  # noqa: E402
from viddar.data import load_rgb  # noqa: E402
from viddar.parsing import clean_object_phrase  # noqa: E402

SIGN_WORDS = ("sign", "stop", "speed", "limit", "warning", "arrow", "placa", "yield", "exit", "parking",
              "road sign", "traffic sign", "signal")


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def detection_frames(dets_path: str, min_score: float) -> pd.DataFrame:
    """Per segment, the frame and box of the highest-scoring sign detection above min_score."""
    import json

    rows = []
    for f in json.load(open(dets_path)):
        for d in f["dets"]:
            if d["score"] >= min_score:
                x0, y0, x1, y1 = d["box"]
                rows.append({"segment": f["frame"].split("/")[1], "frame": f["frame"], "score": d["score"],
                             "x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": x1 - x0})
    df = pd.DataFrame(rows).sort_values("score", ascending=False).groupby("segment").head(1)
    df = df.sort_values("segment").reset_index(drop=True)
    df.insert(0, "idx", range(len(df)))
    df["category"] = "detected_sign"
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["review", "detections"], default="detections")
    ap.add_argument("--review", default="data/annotations/vwise_sign_review.csv")
    ap.add_argument("--dets", default="results/vwise/sign_dets.json")
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--root", default="data/placas/vwise_final")
    ap.add_argument("--vlm", default="qwen3-vl-4b")
    ap.add_argument("--cache-dir", default="results/cache")
    ap.add_argument("--out", default=None, help="default: results/vwise/pilot_keyobject_<source>.csv")
    args = ap.parse_args()
    out_path = Path(args.out or f"results/vwise/pilot_keyobject_{args.source}.csv")

    if args.source == "review":
        review = pd.read_csv(args.review)
        review = review[review["category"].isin(["traffic_sign", "video_overlay"])].reset_index(drop=True)
    else:
        review = detection_frames(args.dets, args.min_score)
    root = Path(args.root)

    from viddar.vlm import CachedVLM

    vlm = CachedVLM(args.vlm, args.cache_dir)
    answers = []
    for r in tqdm(review.itertuples(), total=len(review), desc="key object"):
        answers.append(vlm.ask([str(root / r.frame)], prompts.KEY_OBJECT["paper"]))
    vlm.close()

    from viddar.grounding import GroundedSegmenter

    seg = GroundedSegmenter()
    rows = []
    for r, ans in tqdm(zip(review.itertuples(), answers), total=len(review), desc="grounding"):
        phrase = clean_object_phrase(ans)
        boxes, scores = seg.detect(load_rgb(str(root / r.frame)), phrase)
        target = (r.x0, r.y0, r.x1, r.y1)
        hit_iou = hit_centre = 0.0
        if len(scores):
            b = boxes[int(scores.argmax())]
            hit_iou = box_iou(b, target)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            hit_centre = float(target[0] <= cx <= target[2] and target[1] <= cy <= target[3])
        rows.append({
            "idx": r.idx, "segment": r.segment, "frame": r.frame, "category": r.category, "sign_width": r.w,
            "vlm_response": ans, "key_phrase": phrase,
            "mentions_sign": any(w in phrase for w in SIGN_WORDS),
            "detected": bool(len(scores)), "iou_with_target": round(hit_iou, 3),
            "grounded_on_target": bool(hit_iou >= 0.3 or hit_centre),
        })
    seg.close()
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.groupby("category")[["mentions_sign", "grounded_on_target"]].mean().round(3))
    print(out.groupby("category").size())


if __name__ == "__main__":
    main()
