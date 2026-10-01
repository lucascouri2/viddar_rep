"""Dataset manifests and mask utilities.

Every experiment reads a *manifest* CSV, so a new dataset only needs a new manifest.
Paths inside a manifest are relative to the manifest's own directory (or absolute).

Obstruction manifest columns
    id, raw_path, ar_path, label (1 = obstructed, 0 = not obstructed)
    optional: key_object, key_mask_path, content_mask_path, group

Information manipulation manifest columns
    id, raw_path, ar_path, label (1 = manipulation, 0 = none)
    optional: A, S, I (ground-truth factor labels), group
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

OBSTRUCTION_REQUIRED = ["id", "raw_path", "ar_path", "label"]
MANIPULATION_REQUIRED = ["id", "raw_path", "ar_path", "label"]


def read_label_table(path: str | Path) -> pd.DataFrame:
    """Read a label file that may be a real CSV or an .xlsx saved with a .csv extension.

    The original `information_manipulation_labels.csv` of the ViDDAR repository is an
    Excel workbook, so the file signature is checked instead of trusting the extension.
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        with open(path, "rb") as f:
            return pd.read_excel(f, engine="openpyxl")
    return pd.read_csv(path)


def load_manifest(path: str | Path, kind: str) -> pd.DataFrame:
    """Load a manifest and resolve its relative paths to absolute ones."""
    path = Path(path)
    df = pd.read_csv(path, dtype={"id": str})
    required = OBSTRUCTION_REQUIRED if kind == "obstruction" else MANIPULATION_REQUIRED
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    base = path.parent
    for col in [c for c in df.columns if c.endswith("_path")]:
        df[col] = df[col].map(lambda p: _resolve(base, p))
    df["label"] = df["label"].astype(int)
    if "group" not in df.columns:
        df["group"] = "all"
    df["group"] = df["group"].fillna("all").astype(str)
    return df


def _resolve(base: Path, p) -> str | None:
    if p is None or (isinstance(p, float) and np.isnan(p)) or p == "":
        return None
    p = Path(p)
    return str(p if p.is_absolute() else (base / p).resolve())


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str) -> np.ndarray:
    """Load a binary mask (any non-zero pixel is foreground)."""
    return np.array(Image.open(path).convert("L")) > 0


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def derive_content_mask(
    raw: Image.Image,
    ar: Image.Image,
    diff_threshold: int = 30,
    open_kernel: int = 3,
    close_kernel: int = 7,
) -> np.ndarray:
    """Virtual-content mask m_c obtained by comparing I_r and I_a pixel-wise (Sec. 3.1).

    A pixel belongs to the virtual content when any RGB channel differs by more than
    `diff_threshold`. A morphological opening removes sensor/compression speckle and a
    closing fills small holes inside the rendered object.
    """
    a = np.asarray(raw, dtype=np.int16)
    b = np.asarray(ar, dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError(f"raw/AR size mismatch: {a.shape} vs {b.shape}")
    mask = (np.abs(a - b).max(axis=-1) > diff_threshold).astype(np.uint8)
    if open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    if close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))
    return mask.astype(bool)


def get_content_mask(row: pd.Series) -> np.ndarray:
    """Use the manifest's content mask when present, otherwise derive it from the image pair."""
    p = row.get("content_mask_path")
    if isinstance(p, str) and p:
        return load_mask(p)
    return derive_content_mask(load_rgb(row["raw_path"]), load_rgb(row["ar_path"]))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def obstruction_ratio(key_mask: np.ndarray, content_mask: np.ndarray) -> float:
    """|m_k ∩ m_c| / |m_k| from Eq. (4). Returns 0 for an empty key-object mask."""
    area = key_mask.sum()
    return float(np.logical_and(key_mask, content_mask).sum() / area) if area else 0.0
