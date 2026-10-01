"""Build manifests for the ViDDAR dataset and derive the missing virtual-content masks.

The released dataset only ships 32 unnamed content masks for 306 obstruction pairs, so a
mask is derived for every pair from |I_r - I_a| (paper Sec. 3.1) and checked against the
32 released ones. The script also measures how well the labels are explained by the
ground-truth key-object masks at several thresholds (an oracle upper bound for Eq. 4).

Usage:
    python scripts/prepare_viddar.py [--dataset "ViDDAR dataset/ViDDAR Dataset"] [--out data]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viddar.data import (derive_content_mask, load_mask, load_rgb, mask_iou,  # noqa: E402
                         obstruction_ratio, read_label_table, save_mask)


def rel(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def build_obstruction(ds: Path, out: Path) -> tuple[pd.DataFrame, dict]:
    labels = pd.read_csv(ds / "obstruction_labels.csv")
    man_dir = out / "manifests"
    mask_dir = out / "content_masks" / "obstruction"
    released = {p.name: load_mask(str(p)) for p in sorted((ds / "obstruction_ar_content_mask").glob("*.png"))}

    rows, best_ious, oracle_ratios = [], [], []
    for _, r in tqdm(labels.iterrows(), total=len(labels), desc="obstruction"):
        name = r["File Name"]
        raw, ar = ds / "obstruction_raw_img" / name, ds / "obstruction_ar_img" / f"AR_{name}"
        key_mask = ds / "obstruction_key_object_mask" / f"mask_{name}"
        content = derive_content_mask(load_rgb(str(raw)), load_rgb(str(ar)))
        cm_path = mask_dir / name
        save_mask(content, cm_path)
        best_ious.append(max(mask_iou(content, m) for m in released.values()))
        oracle_ratios.append(obstruction_ratio(load_mask(str(key_mask)), content))
        rows.append({
            "id": Path(name).stem,
            "group": r["Object Name"],
            "key_object": r["Object Name"],
            "raw_path": rel(raw, man_dir),
            "ar_path": rel(ar, man_dir),
            "key_mask_path": rel(key_mask, man_dir),
            "content_mask_path": rel(cm_path, man_dir),
            "label": int(str(r["obstruction"]).strip().lower() == "yes"),
        })
    df = pd.DataFrame(rows)
    df["oracle_ratio"] = oracle_ratios
    y = df["label"].to_numpy()
    ratios = np.array(oracle_ratios)
    report = {
        "n_pairs": len(df),
        "n_classes": int(df["group"].nunique()),
        "positives": int(y.sum()),
        "derived_vs_released_content_mask_best_iou": {
            "median": float(np.median(best_ious)), "p10": float(np.percentile(best_ious, 10)),
            "share_above_0.8": float(np.mean(np.array(best_ious) > 0.8)),
        },
        "oracle_ratio_max_negative": float(ratios[y == 0].max()),
        "oracle_ratio_min_positive": float(ratios[y == 1].min()),
        "oracle_accuracy_by_alpha": {
            str(a): float(np.mean((ratios >= a).astype(int) == y)) for a in (0.1, 0.15, 0.2, 0.25, 0.3)
        },
    }
    return df, report


def build_manipulation(ds: Path, out: Path) -> tuple[pd.DataFrame, dict]:
    man_dir = out / "manifests"
    # NB: this ".csv" is really an .xlsx workbook; its extra column "result" is undocumented and ignored.
    lab = read_label_table(ds / "confusion_labels.csv").dropna(subset=["file"])
    rows = []
    for _, r in lab.iterrows():
        i = int(r["file"])
        rows.append({
            "id": str(i),
            "group": "all",
            "raw_path": rel(ds / "raw" / f"{i}_raw.png", man_dir),
            "ar_path": rel(ds / "confusion_ar_img" / f"{i}_ar.png", man_dir),
            "A": int(r["alignment precision"]),
            "S": int(r["style similarity"]),
            "I": int(r["function misrepresentation"]),
            "label": int(r["confusion"]),
        })
    df = pd.DataFrame(rows)
    missing = [p for p in df["raw_path"].tolist() + df["ar_path"].tolist() if not (man_dir / p).exists()]
    report = {
        "n_pairs": len(df),
        "positives": int(df["label"].sum()),
        "label_equals_A_and_S_and_I": bool((df["label"] == (df["A"] & df["S"] & df["I"])).all()),
        "factor_positive_rate": {f: float(df[f].mean()) for f in ["A", "S", "I", "label"]},
        "missing_files": missing,
    }
    return df, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ViDDAR dataset/ViDDAR Dataset")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    ds, out = Path(args.dataset).resolve(), Path(args.out).resolve()
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    obs, obs_report = build_obstruction(ds, out)
    obs.drop(columns="oracle_ratio").to_csv(out / "manifests" / "viddar_obstruction.csv", index=False)
    man, man_report = build_manipulation(ds, out)
    man.to_csv(out / "manifests" / "viddar_manipulation.csv", index=False)

    report = {"obstruction": obs_report, "manipulation": man_report}
    with open(out / "manifests" / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
