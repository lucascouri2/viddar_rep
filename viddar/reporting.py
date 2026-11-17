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
            "fpr": pct(d["fpr"]["value"]), "fnr": pct(d["fnr"]["value"]), "fp": d["fp"], "fn": d["fn"],
            "no_answer": d["no_answer"],
            "acc_AandSandI": pct(r["accuracy"]["value"]), "fpr_AandSandI": pct(r["fpr"]["value"]),
            "q6_consistent_with_q3-5": pct(m["final_answer_consistent_with_factors"]),
        }
        for fac in "ASI":
            if f"factor_{fac}" in m:
                row[f"acc_{fac}"] = pct(m[f"factor_{fac}"]["accuracy"]["value"])
        rows.append(row)
    return pd.DataFrame(rows)




# ------------------------------------------------------- false-positive study (paths A and C2)
def load_manipulation_runs(root: Path, models: list[str], variants: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Predictions of every finished (model, variant) run under <root>/manipulation; missing runs are skipped."""
    runs = {}
    for model in models:
        for variant in variants:
            f = Path(root) / "manipulation" / model / variant / "predictions.csv"
            if f.exists():
                df = pd.read_csv(f, dtype={"id": str})
                for c in ["A", "S", "I", "label"]:
                    df[c] = df[c].astype(int)
                runs[(model, variant)] = df
    return runs


def _kn(r: dict) -> str:
    return f"{100 * r['value']:.1f} ({r['k']}/{r['n']})"


def _correct(pred: pd.Series, label: pd.Series) -> pd.Series:
    return (pred == label) & pred.notna()


def _mcnemar(df: pd.DataFrame, col: str, ref: pd.DataFrame, ref_col: str = "pred") -> dict:
    """McNemar of decision `col` of `df` against decision `ref_col` of `ref`, on the same pairs (by id)."""
    from .metrics import mcnemar_exact

    m = df[["id", "label", col]].merge(ref[["id", ref_col]].rename(columns={ref_col: "_ref"}), on="id")
    return mcnemar_exact(_correct(m[col], m["label"]), _correct(m["_ref"], m["label"]))


def verdict(delta_accuracy: float, p_value: float | None, alpha: float = 0.05) -> str:
    """Gain judged by accuracy, like the paper: a change only counts when McNemar's p < alpha."""
    if p_value is None or pd.isna(p_value) or p_value >= alpha:
        return "sem diferença significativa"
    return "melhorou" if delta_accuracy > 0 else "piorou"


def _metric_row(r: dict) -> dict:
    """Accuracy, precision, recall, FPR and FNR (%) of an `error_rates` result."""
    return {"acurácia": 100 * r["accuracy"]["value"], "precisão": 100 * r["precision"]["value"],
            "recall": 100 * r["recall"]["value"], "FPR": 100 * r["fpr"]["value"], "FNR": 100 * r["fnr"]["value"]}


def error_comparison(runs: dict[tuple[str, str], pd.DataFrame], reference: str = "paper") -> pd.DataFrame:
    """Accuracy, precision, recall, FPR and FNR per run, with McNemar's test against the `reference` variant."""
    from .manipulation import error_rates

    rows = []
    for (model, variant), df in runs.items():
        r = error_rates(df)
        row = {"model": model, "variant": variant, "fpr": r["fpr"]["value"], "fnr": r["fnr"]["value"],
               "accuracy": r["accuracy"]["value"], "precision": r["precision"]["value"],
               "recall": r["recall"]["value"], "Acurácia % (k/n)": _kn(r["accuracy"]),
               "Precisão % (k/n)": _kn(r["precision"]), "Recall % (k/n)": _kn(r["recall"]),
               "FPR % (k/n)": _kn(r["fpr"]), "FNR % (k/n)": _kn(r["fnr"]),
               "FPR IC95": ci(r["fpr"]), "FNR IC95": ci(r["fnr"])}
        ref = runs.get((model, reference))
        if ref is not None and variant != reference:
            t = _mcnemar(df, "pred", ref)
            row.update({"só variante acerta": t["a_only"], f"só {reference} acerta": t["b_only"],
                        "McNemar p": t["p_value"]})
        rows.append(row)
    return pd.DataFrame(rows)


def gain_table(runs: dict[tuple[str, str], pd.DataFrame], reference: str = "paper") -> pd.DataFrame:
    """One line per run: the paper's metrics (accuracy, precision, recall), FPR, FNR and a gain verdict.

    The verdict compares accuracy with the `reference` variant of the same model on the same pairs
    (McNemar, p < 0.05). Precision, recall, FPR and FNR show what changed.
    """
    e = error_comparison(runs, reference)
    rows = []
    for _, r in e.iterrows():
        ref = e[(e["model"] == r["model"]) & (e["variant"] == reference)]
        row = {"modelo": r["model"], "variante": r["variant"], "acurácia": 100 * r["accuracy"],
               "precisão": 100 * r["precision"], "recall": 100 * r["recall"], "FPR": 100 * r["fpr"],
               "FNR": 100 * r["fnr"]}
        if r["variant"] == reference or ref.empty:
            row.update({"Δ acurácia (p.p.)": 0.0 if r["variant"] == reference else float("nan"),
                        "McNemar p": float("nan"), "ganho?": "referência" if r["variant"] == reference else "-"})
        else:
            delta = 100 * (r["accuracy"] - ref["accuracy"].iloc[0])
            row.update({"Δ acurácia (p.p.)": delta, "McNemar p": r["McNemar p"],
                        "ganho?": verdict(delta, r["McNemar p"])})
        rows.append(row)
    return _rounded(pd.DataFrame(rows))


def _rounded(df: pd.DataFrame) -> pd.DataFrame:
    """Percentages with one decimal, p-values with three."""
    out = df.copy()
    for c in out.columns:
        if c.startswith("McNemar p"):
            out[c] = out[c].astype(float).round(3)
        elif out[c].dtype.kind == "f":
            out[c] = out[c].round(1)
    return out


def paper_reference() -> pd.DataFrame:
    """Table 2 of the paper (accuracy, precision, recall in %), for direct comparison."""
    return PAPER_TABLE2.rename(columns={"vlm": "modelo", "prompt": "variante", "accuracy": "acurácia",
                                        "precision": "precisão"})


def _run_name(key: tuple[str, str]) -> str:
    return f"{key[0]} / {key[1]}"


def cause_name(cause: str) -> str:
    """Readable name of a cause: the factor(s) the model got wrong, whatever the direction of the error."""
    if cause == "illegible":
        return "ilegível"
    if cause == "conclusion":
        return "conclusão (Q6 contradiz Q3-5)"
    return "errou " + cause.split(":", 1)[1]


def cause_comparison(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Error causes side by side, each cell "k/pairs where it can happen".

    "errou A" on a false positive: the model said A = yes on a benign pair whose only false factor is A;
    on a false negative: it said A = no on an attack (where every factor is true).
    """
    from .manipulation import cause_table

    table = None
    for key, df in runs.items():
        t = cause_table(df)
        t[_run_name(key)] = [f"{k}/{n}" for k, n in zip(t["k"], t["eligible"])]
        t = t.drop(columns=["k"])
        table = t if table is None else table.merge(t, on=["error", "cause", "eligible"], how="outer")
    if table is None:
        return pd.DataFrame()
    for col in table.columns[3:]:
        table[col] = table[col].fillna("0/" + table["eligible"].astype(str))
    table["order"] = table["cause"].map(lambda c: (c != "illegible", c != "conclusion", c))
    table = table.sort_values(["error", "order"], ascending=[False, True]).drop(columns="order")
    table.insert(2, "causa", table["cause"].map(cause_name))
    return table.drop(columns="cause").rename(columns={"error": "erro", "eligible": "pares possíveis"}) \
        .reset_index(drop=True)


def cause_overview(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """All errors (FP + FN) by cause, per run: each cell "total (FP a + FN b)", plus a total line."""
    from .manipulation import cause_table

    counts = {}
    for key, df in runs.items():
        t = cause_table(df)
        t["causa"] = t["cause"].map(cause_name)
        g = t.pivot_table(index="causa", columns="error", values="k", aggfunc="sum", fill_value=0)
        counts[_run_name(key)] = g
    causes = sorted({c for g in counts.values() for c in g.index},
                    key=lambda c: (c != "ilegível", not c.startswith("conclusão"), len(c), c))
    out = pd.DataFrame({"causa": causes + ["total"]})
    for name, g in counts.items():
        fp = [int(g.loc[c, "FP"]) if c in g.index and "FP" in g.columns else 0 for c in causes]
        fn = [int(g.loc[c, "FN"]) if c in g.index and "FN" in g.columns else 0 for c in causes]
        fp.append(sum(fp))
        fn.append(sum(fn))
        out[name] = [f"{a + b} (FP {a} + FN {b})" if a + b else "0" for a, b in zip(fp, fn)]
    return out


def factor_comparison(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Per factor, how often the model got it wrong: when it is false (said yes) and when it is true (said no)."""
    from .manipulation import factor_rates

    rows = []
    for key, df in runs.items():
        row = {"model": key[0], "variant": key[1]}
        for _, r in factor_rates(df).iterrows():
            row[f"{r['factor']}: errou com {r['factor']}=0"] = _kn(r["accepted_when_0"])
            row[f"{r['factor']}: errou com {r['factor']}=1"] = _kn(r["rejected_when_1"])
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_comparison(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """FPR / FNR (%) of A ∧ S ∧ I with one or two factors replaced by the ground truth."""
    from .manipulation import oracle_rates

    rows = []
    for key, df in runs.items():
        o = oracle_rates(df)
        row = {"model": key[0], "variant": key[1]}
        for _, r in o.iterrows():
            row[f"oráculo {r['oracle']}"] = f"{100 * r['fpr']:.1f} / {100 * r['fnr']:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def alignment_summary(root: Path, keys: list[tuple[str, str]]) -> pd.DataFrame:
    """Path C2 of each run that has <run>/alignment/metrics.json: VLM-only decision against the hybrid one.

    The error rates are recomputed from alignment/predictions.csv, so runs saved before precision and
    recall were added get them too.
    """
    from .manipulation import error_rates

    rows = []
    for model, variant in keys:
        f = Path(root) / "manipulation" / model / variant / "alignment" / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        aucs = {a["feature"]: a for a in m["feature_auc"]}
        g = aucs[m["feature"]]
        p = pd.read_csv(f.parent / "predictions.csv", dtype={"id": str})
        for c in ["pred", "pred_hybrid"]:
            p[c] = p[c].astype(object).where(p[c].notna(), None)
        b, h, t = error_rates(p, "pred"), error_rates(p, "pred_hybrid"), m["mcnemar_hybrid_vs_base"]
        delta = 100 * (h["accuracy"]["value"] - b["accuracy"]["value"])
        row = {
            "model": model, "variant": variant, "detected": f"{m['detected']}/{m['n']}",
            f"AUC {m['feature']} [IC95]": f"{g['auc']:.2f} [{g['ci95_low']:.2f}, {g['ci95_high']:.2f}]",
            "acerto em A: geométrico": m["factor_A_accuracy"]["geometric"],
            "acerto em A: VLM (mesmos pares)": m["factor_A_accuracy"]["vlm_same_pairs"],
            "fpr": b["fpr"]["value"], "fnr": b["fnr"]["value"],
            "hybrid_fpr": h["fpr"]["value"], "hybrid_fnr": h["fnr"]["value"],
        }
        for name, key in [("acurácia", "accuracy"), ("precisão", "precision"), ("recall", "recall"),
                          ("FPR", "fpr"), ("FNR", "fnr")]:
            if key in b and key in h:
                row[f"{name}: VLM → híbrido"] = f"{100 * b[key]['value']:.1f} → {100 * h[key]['value']:.1f}"
        row.update({"só híbrido acerta": t["a_only"], "só VLM acerta": t["b_only"], "McNemar p": t["p_value"],
                    "ganho?": verdict(delta, t["p_value"]),
                    "acurácia híbrida (média ± dp, CV)": f"{100 * m['cv_repeats']['accuracy']['mean']:.1f} ± "
                                                         f"{100 * m['cv_repeats']['accuracy']['std']:.1f}"})
        rows.append(row)
    return pd.DataFrame(rows)
