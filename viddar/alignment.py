"""Geometric alignment factor A_geo (path C2 of report/analise_fp_fn_manipulacao.md).

Factor A of the paper asks whether the virtual content is "accurately aligned to the object,
without a significant gap". Instead of asking the VLM, A_geo measures it in pixels, like the
obstruction pipeline does for Eq. 4: the virtual content mask m_c comes from the pixel difference
between I_r and I_a, and the key-object mask m_k from Grounding DINO + SAM on I_r, prompted with the
key object named by the VLM. The hybrid decision is M = A_geo ∧ S_vlm ∧ I_vlm.

The decision feature (`gap_norm`) is fixed in advance and its threshold is chosen by stratified
cross-validation, never on the whole dataset.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import get_content_mask, load_rgb, save_mask
from .metrics import auc, auc_bootstrap_ci, mcnemar_exact

# Feature -> True when larger values mean "aligned" (used to orient scores for AUC and thresholds).
FEATURES = {"gap_norm": False, "gap_px": False, "contact": True, "overlap": True}


def geometry_features(content: np.ndarray, key: np.ndarray, contact_frac: float = 0.05) -> dict:
    """Distances between the virtual content m_c and the key object m_k.

    gap_px    smallest distance from a content pixel to the object (0 when they touch or overlap)
    gap_norm  gap_px / content size (larger side of its bounding box), so it does not depend on scale
    contact   share of the content's outline within `contact_frac` x content size of the object
    overlap   share of the content's pixels within that same distance of the object
    """
    nan = {"gap_px": np.nan, "gap_norm": np.nan, "contact": np.nan, "overlap": np.nan}
    if not content.any() or not key.any():
        return nan | {"content_px": int(content.sum()), "key_px": int(key.sum())}
    dist = cv2.distanceTransform((~key).astype(np.uint8), cv2.DIST_L2, 5)  # 0 on the object
    ys, xs = np.nonzero(content)
    size = float(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    radius = max(2.0, contact_frac * size)
    outline = content & ~(cv2.erode(content.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    gap = float(dist[content].min())
    return {
        "gap_px": gap,
        "gap_norm": gap / size,
        "contact": float((dist[outline] <= radius).mean()),
        "overlap": float((dist[content] <= radius).mean()),
        "content_px": int(content.sum()),
        "key_px": int(key.sum()),
    }


def measure(df: pd.DataFrame, seg, key_col: str = "key_phrase", contact_frac: float = 0.05,
            mask_dir: str | Path | None = None) -> pd.DataFrame:
    """Detect and segment the key object on I_r and measure its geometry against m_c, pair by pair."""
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="alignment: Grounding DINO + SAM"):
        phrase = r.get(key_col)
        phrase = phrase if isinstance(phrase, str) and phrase.strip() else ""
        content = get_content_mask(r)
        key = np.zeros_like(content)
        score = np.nan
        if phrase:
            det = seg.detect_and_segment(load_rgb(r["raw_path"]), phrase, top1=True)
            if len(det.scores):
                key, score = det.masks[0], float(det.scores[0])
        if mask_dir is not None and key.any():
            save_mask(key, Path(mask_dir) / f"{r['id']}.png")
        rows.append({"id": r["id"], "key_phrase": phrase, "detected": bool(key.any()), "det_score": score,
                     **geometry_features(content, key, contact_frac)})
    return pd.DataFrame(rows)


def _oriented(values, higher_is_aligned: bool) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v if higher_is_aligned else -v


def feature_auc(features: pd.DataFrame, y) -> pd.DataFrame:
    """AUC of each feature against the ground-truth A (pairs without a detection are left out)."""
    rows = []
    y = np.asarray(y, dtype=int)
    for name, higher in FEATURES.items():
        ok = features[name].notna().to_numpy()
        s = _oriented(features[name], higher)[ok]
        lo, hi = auc_bootstrap_ci(s, y[ok])
        rows.append({"feature": name, "auc": auc(s, y[ok]), "ci95_low": lo, "ci95_high": hi, "n": int(ok.sum())})
    return pd.DataFrame(rows)


def _best_threshold(s: np.ndarray, y: np.ndarray) -> float:
    """Threshold on an oriented score (aligned when s >= t) maximising balanced accuracy.

    Candidates are the midpoints between consecutive training values, so a test value slightly
    beyond the training extremes is not decided by an exact tie.
    """
    u = np.unique(s)
    candidates = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]]) if len(u) else [np.inf]
    best_t, best = np.inf, -1.0
    for t in candidates:
        p = s >= t
        tpr = p[y == 1].mean() if (y == 1).any() else 0.0
        tnr = (~p[y == 0]).mean() if (y == 0).any() else 0.0
        if (tpr + tnr) / 2 > best:
            best_t, best = float(t), (tpr + tnr) / 2
    return best_t


def cv_predict(score, y, higher_is_aligned: bool, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold A_geo (1/0, NaN where the score is missing) with stratified k-fold thresholds."""
    s = _oriented(score, higher_is_aligned)
    y = np.asarray(y, dtype=int)
    ok = ~np.isnan(s)
    rng = np.random.default_rng(seed)
    fold = np.full(len(s), -1)
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(ok & (y == cls)))
        fold[idx] = np.arange(len(idx)) % n_splits
    pred = np.full(len(s), np.nan)
    for k in range(n_splits):
        train, test = ok & (fold != k), fold == k
        t = _best_threshold(s[train], y[train])
        pred[test] = (s[test] >= t).astype(float)
    return pred


def hybrid(preds: pd.DataFrame, a_geo: np.ndarray) -> pd.DataFrame:
    """Replace the VLM's A by A_geo (falling back to the VLM where no object was detected)."""
    out = preds.copy()
    a = pd.Series(a_geo, index=out.index)
    out["a_source"] = np.where(a.notna(), "geometric", "vlm")
    out["pred_A_vlm"] = out["pred_A"]
    out["pred_A"] = a.where(a.notna(), out["pred_A"])
    vals = out[["pred_A", "pred_S", "pred_I"]]
    out["pred_hybrid"] = [None if any(pd.isna(v) for v in row) else int(all(v == 1 for v in row))
                          for row in vals.itertuples(index=False)]
    return out


def evaluate(preds: pd.DataFrame, features: pd.DataFrame, feature: str = "gap_norm", n_splits: int = 5,
             n_repeats: int = 20, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """A_geo by cross-validation, hybrid decision and its comparison with the run's own decision.

    Returns the hybrid predictions (repetition `seed`) and a summary with the feature AUCs, the
    error rates of both decisions, McNemar's test and the spread over `n_repeats` CV repetitions.
    """
    from .manipulation import error_rates

    df = preds.merge(features, on="id", how="left", suffixes=("", "_geo"))
    higher = FEATURES[feature]
    reps = []
    for rep in range(n_repeats):
        h = hybrid(df, cv_predict(df[feature], df["A"], higher, n_splits, seed + rep))
        r = error_rates(h, "pred_hybrid")
        reps.append({"fpr": r["fpr"]["value"], "fnr": r["fnr"]["value"], "accuracy": r["accuracy"]["value"]})
    out = hybrid(df, cv_predict(df[feature], df["A"], higher, n_splits, seed))
    geo = out["a_source"] == "geometric"
    a_correct = (out.loc[geo, "pred_A"] == out.loc[geo, "A"]).mean() if geo.any() else np.nan
    a_vlm_correct = (out.loc[geo, "pred_A_vlm"] == out.loc[geo, "A"]).mean() if geo.any() else np.nan
    base_ok = (out["pred"] == out["label"]) & out["pred"].notna()
    hyb_ok = (out["pred_hybrid"] == out["label"]) & out["pred_hybrid"].notna()
    rep_df = pd.DataFrame(reps)
    summary = {
        "feature": feature,
        "detected": int(df["detected"].sum()),
        "n": len(df),
        "feature_auc": feature_auc(df, df["A"]).to_dict(orient="records"),
        "factor_A_accuracy": {"geometric": float(a_correct), "vlm_same_pairs": float(a_vlm_correct),
                              "pairs": int(geo.sum())},
        "base": error_rates(out, "pred"),
        "hybrid": error_rates(out, "pred_hybrid"),
        "mcnemar_hybrid_vs_base": mcnemar_exact(hyb_ok, base_ok),
        "cv_repeats": {k: {"mean": float(rep_df[k].mean()), "std": float(rep_df[k].std())} for k in rep_df},
        "cv": {"n_splits": n_splits, "n_repeats": n_repeats, "seed": seed},
    }
    return out, summary
