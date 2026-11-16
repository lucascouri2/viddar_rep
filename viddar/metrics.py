"""Binary classification metrics with Wilson confidence intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def _ratio(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"value": k / n if n else float("nan"), "k": int(k), "n": int(n), "ci95": [lo, hi]}


def binary_metrics(y_true, y_pred) -> dict:
    """Metrics for binary predictions. `y_pred` may contain None (no answer), counted as wrong.

    FPR (false positive rate) = FP / (FP + TN): share of benign samples flagged as attacks.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred = pd.Series(list(y_pred), dtype=object)
    answered = pred.notna().to_numpy()
    p = pred.fillna(-1).astype(int).to_numpy()
    tp = int(((p == 1) & (y_true == 1)).sum())
    tn = int(((p == 0) & (y_true == 0)).sum())
    fp = int(((p == 1) & (y_true == 0)).sum())
    fn = int(((p == 0) & (y_true == 1)).sum())
    n = len(y_true)
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = (2 * prec["value"] * rec["value"] / (prec["value"] + rec["value"])
          if (tp + fp) and (tp + fn) and (prec["value"] + rec["value"]) else float("nan"))
    return {
        "n": n,
        "no_answer": int((~answered).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _ratio(tp + tn, n),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "positive_rate": _ratio(int((p == 1).sum()), n),
    }


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact two-sided McNemar test for two classifiers evaluated on the same samples.

    Only discordant samples count: b = A right and B wrong, c = A wrong and B right. Under H0 they
    are Binomial(b + c, 0.5). Paired, so more powerful than Fisher's test on the marginal rates.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    b, c = int((a & ~b_).sum()), int((~a & b_).sum())
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p_value": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return {"a_only": b, "b_only": c, "p_value": min(1.0, 2 * tail)}


def auc(scores, y) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic (ties count 1/2). Higher score = positive."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def auc_bootstrap_ci(scores, y, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% interval of the AUC, resampling within each class (stratified bootstrap)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(auc(s[idx], y[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def metrics_by_group(df: pd.DataFrame, pred_col: str = "pred", label_col: str = "label",
                     group_col: str = "group") -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        m = binary_metrics(sub[label_col], sub[pred_col])
        rows.append({
            group_col: g, "n": m["n"], "positives": int(sub[label_col].sum()),
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
            "accuracy": m["accuracy"]["value"], "fpr": m["fpr"]["value"], "fnr": m["fnr"]["value"],
        })
    return pd.DataFrame(rows).sort_values("fp", ascending=False)


def flatten(m: dict) -> dict:
    """Metrics dict -> flat {name: value} for tables."""
    out = {}
    for k, v in m.items():
        out[k] = v["value"] if isinstance(v, dict) and "value" in v else v
    return out
