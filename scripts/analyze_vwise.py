"""Characterise the driving-video dataset provided for the project (data/placas/vwise_final).

Stages (each writes to <out>/ and is skipped when its output already exists, unless --force):
  1. inventory    segments.csv  one row per segment: source video, labels, video properties
  2. quality      frames.csv    one row per frame: brightness and sharpness
  3. signs        sign_dets.json  Grounding DINO "traffic sign." detections on every frame
  4. summary      summary.json  aggregated numbers used in the report

Usage:
    python scripts/analyze_vwise.py [--root data/placas/vwise_final] [--out results/vwise] [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import viddar  # noqa: E402,F401  (environment setup: truststore, USE_TF=0)

SIGN_QUERY = "traffic sign."
BOX_THRESHOLD, TEXT_THRESHOLD = 0.3, 0.25


def source_group(recorte: int) -> str:
    """Source groups identified by inspecting one frame per source video (see REPRODUCAO/report)."""
    if recorte <= 26:
        return "youtube"  # dash/handheld driving recordings, 1280x720
    if recorte <= 43:
        return "stock"  # stock footage with shutterstock watermark
    return "other"  # recortes 44-45: 1080p city drive and Tokyo night drive


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def inventory(root: Path) -> pd.DataFrame:
    rows = []
    for rec_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rec = int(rec_dir.name.split("_")[1])
        for seg in sorted(p for p in rec_dir.iterdir() if p.is_dir()):
            video = next(seg.glob("*.mp4"))
            cap = cv2.VideoCapture(str(video))
            w, h, fps, n = int(cap.get(3)), int(cap.get(4)), cap.get(5), cap.get(7)
            cap.release()
            pred, gt = read_text(seg / "legenda.txt"), read_text(seg / "ground_truth.txt")
            rows.append({
                "segment": seg.name, "recorte": rec, "source": source_group(rec),
                "label_pred": pred, "label_gt": gt, "label": gt or pred,
                "flag_erro": seg.name.endswith("_erro"), "has_gt": gt is not None,
                "width": w, "height": h, "fps": round(fps, 2), "duration_s": round(n / fps, 2) if fps else None,
                "n_frames_png": len(list(seg.glob("*.png"))),
                "naming": "underscored" if re.search(r"_segmento_", seg.name) else "compact",
            })
    return pd.DataFrame(rows)


def quality(root: Path) -> pd.DataFrame:
    rows = []
    for p in tqdm(sorted(root.glob("*/*/*.png")), desc="quality", mininterval=30):
        img = cv2.imread(str(p))
        small = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rows.append({
            "frame": p.relative_to(root).as_posix(), "segment": p.parent.name,
            "brightness": float(gray.mean()),  # 0-255
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),  # variance of Laplacian at 640 px
        })
    return pd.DataFrame(rows)


def detect_signs(root: Path) -> list[dict]:
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").cuda().eval()
    out = []
    with torch.inference_mode():
        for p in tqdm(sorted(root.glob("*/*/*.png")), desc="signs", mininterval=30):
            im = Image.open(p).convert("RGB")
            inp = proc(images=im, text=SIGN_QUERY, return_tensors="pt").to("cuda")
            r = proc.post_process_grounded_object_detection(
                model(**inp), inp.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                target_sizes=[im.size[::-1]])[0]
            dets = [{"box": [round(float(v), 1) for v in b], "score": round(float(s), 3)}
                    for b, s in zip(r["boxes"], r["scores"])]
            out.append({"frame": p.relative_to(root).as_posix(), "w": im.width, "h": im.height, "dets": dets})
    return out


def summarize(seg: pd.DataFrame, frames: pd.DataFrame, dets: list[dict]) -> dict:
    s: dict = {
        "n_source_videos": int(seg["recorte"].nunique()),
        "missing_recortes": sorted(set(range(1, seg["recorte"].max() + 1)) - set(seg["recorte"])),
        "n_segments": len(seg), "n_frames": int(seg["n_frames_png"].sum()),
        "total_minutes": round(float(seg["duration_s"].sum()) / 60, 1),
        "duration_s": seg["duration_s"].describe()[["min", "50%", "max"]].round(1).to_dict(),
        "resolution": seg.groupby(["width", "height"]).size().rename("n").reset_index().to_dict("records"),
        "fps": seg["fps"].value_counts().to_dict(),
        "by_source": seg.groupby("source").agg(segments=("segment", "size"), videos=("recorte", "nunique")).to_dict("index"),
        "label_pred_counts": seg["label_pred"].value_counts().to_dict(),
        "label_counts": seg["label"].value_counts().to_dict(),
        "n_with_gt": int(seg["has_gt"].sum()),
        "gt_equals_pred": int((seg["label_gt"] == seg["label_pred"]).sum()),
        "erro_without_gt": seg.loc[seg["flag_erro"] & ~seg["has_gt"], "segment"].tolist(),
        "gt_without_erro_flag": int((seg["has_gt"] & ~seg["flag_erro"]).sum()),
        "corrections": seg[seg["has_gt"]].groupby(["label_pred", "label_gt"]).size().rename("n").reset_index().to_dict("records"),
        "naming": seg["naming"].value_counts().to_dict(),
    }
    f = frames.merge(seg[["segment", "source", "label"]], on="segment")
    s["quality_by_source"] = f.groupby("source")[["brightness", "sharpness"]].median().round(1).to_dict("index")
    s["dark_frames_share"] = round(float((f["brightness"] < 60).mean()), 3)

    rows = []
    for d in dets:
        segname = d["frame"].split("/")[1]
        best = max(d["dets"], key=lambda x: x["score"], default=None)
        area = width = 0.0
        if best:
            b = best["box"]
            area, width = (b[2] - b[0]) * (b[3] - b[1]) / (d["w"] * d["h"]), b[2] - b[0]
        rows.append({"segment": segname, "score": best["score"] if best else 0.0, "area": area, "box_w": width})
    det = pd.DataFrame(rows).merge(seg[["segment", "source", "label"]], on="segment")
    s["signs"] = {}
    for t in (0.35, 0.45, 0.55):
        per_seg = det.groupby("segment")["score"].max() >= t
        s["signs"][str(t)] = {
            "frames_with_detection": int((det["score"] >= t).sum()),
            "segments_with_detection": int(per_seg.sum()),
        }
    hit = det[det["score"] >= 0.45]
    s["signs"]["area_pct_at_0.45"] = (hit["area"] * 100).describe(percentiles=[.5, .75, .9])[["50%", "75%", "90%"]].round(2).to_dict()
    seg_hit = det.assign(hit=det["score"] >= 0.45).groupby("segment").agg(hit=("hit", "max"), source=("source", "first"), label=("label", "first"))
    s["signs"]["segments_by_source_at_0.45"] = seg_hit.groupby("source")["hit"].agg(["size", "sum"]).rename(columns={"size": "segments", "sum": "with_sign"}).to_dict("index")
    s["signs"]["segments_by_label_at_0.45"] = seg_hit.groupby("label")["hit"].agg(["size", "sum"]).rename(columns={"size": "segments", "sum": "with_sign"}).to_dict("index")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/placas/vwise_final")
    ap.add_argument("--out", default="results/vwise")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.force or not (out / "segments.csv").exists():
        inventory(root).to_csv(out / "segments.csv", index=False)
    seg = pd.read_csv(out / "segments.csv")
    if args.force or not (out / "frames.csv").exists():
        quality(root).to_csv(out / "frames.csv", index=False)
    frames = pd.read_csv(out / "frames.csv")
    if args.force or not (out / "sign_dets.json").exists():
        json.dump(detect_signs(root), open(out / "sign_dets.json", "w"))
    dets = json.load(open(out / "sign_dets.json"))

    summary = summarize(seg, frames, dets)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
