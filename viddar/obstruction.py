"""Obstruction attack detection (paper Sec. 3.1, 4.1, 5.1): ViDDAR and baselines.

Pipeline for the VLM-based methods (ViDDAR, Underdetailed, Greedy):
    1. the VLM names the key object(s) in the raw image I_r;
    2. Grounding DINO + SAM produce the key-object mask m_k (highest-confidence box);
    3. the virtual-content mask m_c comes from the manifest or from |I_r - I_a|;
    4. obstruction if |m_k ∩ m_c| / |m_k| >= alpha (Eq. 4).
The per-sample ratio is stored, so metrics for any alpha can be recomputed afterwards.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import prompts
from .data import get_content_mask, load_mask, load_rgb, mask_iou, obstruction_ratio
from .metrics import binary_metrics
from .parsing import clean_object_phrase, key_object_matches, last_yes_no, parse_object_list

VLM_METHODS = {"viddar", "underdetailed", "greedy", "end_to_end"}
GROUNDING_METHODS = {"viddar", "underdetailed", "greedy", "prior"}
CV_METHODS = {"saliency", "canny"}
ALL_METHODS = ["viddar", "end_to_end", "underdetailed", "greedy", "saliency", "canny", "prior"]


# --------------------------------------------------------------------------- stage 1: VLM
def run_vlm_stage(df: pd.DataFrame, method: str, vlm, key_prompt: str = "paper") -> pd.DataFrame:
    df = df.copy()
    first_prompt = {
        "viddar": prompts.KEY_OBJECT[key_prompt],
        "end_to_end": prompts.KEY_OBJECT[key_prompt],
        "underdetailed": prompts.UNDERDETAILED,
        "greedy": prompts.GREEDY,
    }[method]
    responses, phrases, e2e = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{method}: VLM"):
        resp = vlm.ask([row["raw_path"]], first_prompt)
        responses.append(resp)
        if method == "greedy":
            phrases.append(" | ".join(parse_object_list(resp)))
        else:
            phrases.append(clean_object_phrase(resp))
        if method == "end_to_end":
            e2e.append(vlm.ask([row["raw_path"], row["ar_path"]],
                               prompts.END_TO_END_STEP2.format(key_obj=phrases[-1])))
    df["vlm_response"] = responses
    df["key_phrase"] = phrases
    if method == "end_to_end":
        df["e2e_response"] = e2e
    return df


# ------------------------------------------------------------------ stage 2: detection
def run_grounding_stage(df: pd.DataFrame, method: str, segmenter, top1: bool = True) -> pd.DataFrame:
    df = df.copy()
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{method}: detect+segment"):
        raw = load_rgb(row["raw_path"])
        content = get_content_mask(row)
        if method == "prior":
            phrases = [str(row["key_object"])]
        else:
            phrases = [p for p in str(row["key_phrase"]).split(" | ") if p]
        key_mask = np.zeros((raw.height, raw.width), dtype=bool)
        ratios, scores, n_det = [], [], 0
        for phrase in phrases:
            det = segmenter.detect_and_segment(raw, phrase, top1=top1)
            n_det += len(det.scores)
            if len(det.scores):
                m = det.masks.any(axis=0)
                key_mask |= m
                ratios.append(obstruction_ratio(m, content))
                scores.append(float(det.scores.max()))
        rec = {
            "n_detections": n_det,
            "det_score": max(scores) if scores else np.nan,
            # Greedy: obstruction if ANY listed object is obstructed -> use the max ratio.
            "ratio": max(ratios) if ratios else 0.0,
        }
        gt_path = row.get("key_mask_path")
        if isinstance(gt_path, str) and gt_path:
            rec["key_iou"] = mask_iou(key_mask, load_mask(gt_path)) if key_mask.any() else 0.0
        records.append(rec)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)


# --------------------------------------------------------------- classical CV baselines
def spectral_residual_saliency(img: np.ndarray) -> np.ndarray:
    """Hou & Zhang (CVPR 2007) spectral residual saliency, as in OpenCV's implementation."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small)
    log_amp = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (5, 5), 8)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return cv2.resize(sal, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def canny_edges(img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (cv2.Canny(gray, low, high) > 0).astype(np.float32)


def run_cv_stage(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Obstruction if the mean score under m_c exceeds the image mean (S_o > S_r, C_o > C_r)."""
    df = df.copy().reset_index(drop=True)
    score_fn = spectral_residual_saliency if method == "saliency" else canny_edges
    obj, whole = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=method):
        raw = np.asarray(load_rgb(row["raw_path"]))
        content = get_content_mask(row)
        s = score_fn(raw)
        whole.append(float(s.mean()))
        obj.append(float(s[content].mean()) if content.any() else 0.0)
    df["score_obstructed"] = obj
    df["score_image"] = whole
    return df


# ------------------------------------------------------------------------- decisions
def add_predictions(df: pd.DataFrame, method: str, alpha: float) -> pd.DataFrame:
    df = df.copy()
    df["method"] = method
    if method in CV_METHODS:
        df["pred"] = (df["score_obstructed"] > df["score_image"]).astype(int)
    elif method == "end_to_end":
        df["pred"] = df["e2e_response"].map(last_yes_no)
    else:
        df["pred"] = (df["ratio"] >= alpha).astype(int)
    if "key_object" in df.columns:
        if method in VLM_METHODS:
            df["key_recognized"] = [key_object_matches(p, k) for p, k in zip(df["key_phrase"], df["key_object"])]
        elif method == "prior":
            df["key_recognized"] = True
    return df


def summarize(df: pd.DataFrame, method: str, alpha: float,
              alphas=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)) -> dict:
    out = {"method": method, "alpha": alpha, "detection": binary_metrics(df["label"], df["pred"])}
    if "key_recognized" in df.columns:
        out["key_object_recognition_accuracy"] = float(df["key_recognized"].mean())
    if "key_iou" in df.columns:
        out["segmentation_miou"] = float(df["key_iou"].mean())
        # Alternative recognition criterion: the grounded mask overlaps the GT mask (IoU >= 0.5).
        out["key_object_grounded_accuracy"] = float((df["key_iou"] >= 0.5).mean())
    if "n_detections" in df.columns:
        out["no_detection"] = int((df["n_detections"] == 0).sum())
    if "ratio" in df.columns:
        out["alpha_sweep"] = {
            str(a): binary_metrics(df["label"], (df["ratio"] >= a).astype(int))["accuracy"]["value"]
            for a in alphas
        }
    return out
