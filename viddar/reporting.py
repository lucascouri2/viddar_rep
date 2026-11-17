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


def _decision(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(object).where(df[col].notna(), None)


def balanced_accuracy(df: pd.DataFrame, pred_col: str = "pred") -> float:
    """(TPR + TNR) / 2. Any constant detector scores 50%, whatever the class balance; undecided pairs are errors."""
    pred = df[pred_col]
    pos, neg = df["label"] == 1, df["label"] == 0
    return ((pos & (pred == 1)).sum() / pos.sum() + (neg & (pred == 0)).sum() / neg.sum()) / 2


def baseline_comparison(runs: dict[tuple[str, str], pd.DataFrame], pred_col: str = "pred") -> pd.DataFrame:
    """Each run against the trivial detector that never fires (always "benign"): does the VLM add anything?

    The trivial detector is right on every benign pair, so on the same pairs a run only wins where it finds
    an attack (TP) and only loses where it raises a false alarm or leaves a benign pair undecided. McNemar
    on those discordant pairs says whether the run is above, below or indistinguishable from it.
    """
    from .manipulation import error_rates

    rows = []
    for (model, variant), df in runs.items():
        df = df.assign(_d=_decision(df, pred_col), _trivial=0)
        r = error_rates(df, "_d")
        t = _mcnemar(df, "_d", df, "_trivial")
        delta = 100 * (r["accuracy"]["value"] - (df["label"] == 0).mean())
        rows.append({"modelo": model, "variante": variant, "acurácia": 100 * r["accuracy"]["value"],
                     "acurácia balanceada": 100 * balanced_accuracy(df, "_d"),
                     "ataques detectados (só a execução acerta)": t["a_only"],
                     "alarmes falsos ou sem decisão (só o trivial acerta)": t["b_only"],
                     "McNemar p": t["p_value"],
                     "vs. detector trivial": verdict(delta, t["p_value"]).replace("melhorou", "acima")
                     .replace("piorou", "abaixo")})
    return _rounded(pd.DataFrame(rows))


def rule_comparison(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Joint-mode runs: the paper's decision (question 6) against A ∧ S ∧ I of answers 3-5.

    A ∧ S ∧ I is the extra column `pred_factor_rule`, never a replacement for `pred` (AGENTS.md). The
    difference measures what the conclusion step adds to, or takes from, the factor answers.
    """
    from .manipulation import error_rates

    rows = []
    for (model, variant), df in runs.items():
        if "resp_A" in df.columns:  # decomposed: the decision already is A ∧ S ∧ I
            continue
        df = df.assign(_q6=_decision(df, "pred"), _rule=_decision(df, "pred_factor_rule"))
        both = df["_q6"].notna() & df["_rule"].notna()
        q, r = error_rates(df, "_q6"), error_rates(df, "_rule")
        t = _mcnemar(df, "_rule", df, "_q6")
        delta = 100 * (r["accuracy"]["value"] - q["accuracy"]["value"])
        rows.append({"modelo": model, "variante": variant,
                     "Q6 coerente com Q3-5 (%)": 100 * (df.loc[both, "_q6"] == df.loc[both, "_rule"]).mean(),
                     "acurácia: Q6 → A∧S∧I": f"{100 * q['accuracy']['value']:.1f} → {100 * r['accuracy']['value']:.1f}",
                     "FPR: Q6 → A∧S∧I": f"{100 * q['fpr']['value']:.1f} → {100 * r['fpr']['value']:.1f}",
                     "FNR: Q6 → A∧S∧I": f"{100 * q['fnr']['value']:.1f} → {100 * r['fnr']['value']:.1f}",
                     "acurácia balanceada A∧S∧I": 100 * balanced_accuracy(df, "_rule"),
                     "só A∧S∧I acerta": t["a_only"], "só Q6 acerta": t["b_only"], "McNemar p": t["p_value"],
                     "A∧S∧I vs Q6": verdict(delta, t["p_value"])})
    return _rounded(pd.DataFrame(rows))


def factor_discrimination(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Youden's J per factor: 1 − P(wrong | F = 0) − P(wrong | F = 1).

    0 means the answer does not depend on the factor (a constant answer, or chance) and 1 is perfect. Unlike
    the factor accuracy it does not reward a bias towards the majority label (I = 1 in most pairs).
    """
    from .manipulation import factor_rates

    rows = []
    for (model, variant), df in runs.items():
        row = {"model": model, "variant": variant}
        for _, r in factor_rates(df).iterrows():
            row[f"J {r['factor']}"] = round(1 - r["accepted_when_0"]["value"] - r["rejected_when_1"]["value"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_comparison(runs: dict[tuple[str, str], pd.DataFrame], keys: list[tuple[str, str]]) -> pd.DataFrame:
    """McNemar between every two runs in `keys` (e.g. the best variant of each model), on the same pairs."""
    from .manipulation import error_rates

    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            da, db = runs[a], runs[b]
            acc_a, acc_b = error_rates(da)["accuracy"]["value"], error_rates(db)["accuracy"]["value"]
            t = _mcnemar(da, "pred", db)
            v = verdict(100 * (acc_a - acc_b), t["p_value"])
            rows.append({"execução 1": _run_name(a), "execução 2": _run_name(b), "acurácia 1": 100 * acc_a,
                         "acurácia 2": 100 * acc_b, "só 1 acerta": t["a_only"], "só 2 acerta": t["b_only"],
                         "McNemar p": t["p_value"],
                         "resultado": {"melhorou": "1 melhor", "piorou": "2 melhor"}.get(v, v)})
    return _rounded(pd.DataFrame(rows))


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
        # In joint mode `pred` is question 6, while the hybrid is A_geo ∧ S ∧ I: part of "VLM → hybrid" can be
        # the change of rule alone. A_vlm ∧ S ∧ I against the hybrid isolates the geometric A.
        p["pred_factor_rule"] = p["pred_factor_rule"].astype(object).where(p["pred_factor_rule"].notna(), None)
        r = error_rates(p, "pred_factor_rule")
        tr = _mcnemar(p, "pred_hybrid", p, "pred_factor_rule")
        delta_r = 100 * (h["accuracy"]["value"] - r["accuracy"]["value"])
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
                    "acurácia: A∧S∧I do VLM → híbrido": f"{100 * r['accuracy']['value']:.1f} → "
                                                        f"{100 * h['accuracy']['value']:.1f}",
                    "só híbrido acerta (vs A∧S∧I)": tr["a_only"], "só A∧S∧I acerta": tr["b_only"],
                    "McNemar p (vs A∧S∧I)": tr["p_value"], "ganho do A geométrico?": verdict(delta_r, tr["p_value"]),
                    "acurácia híbrida (média ± dp, CV)": f"{100 * m['cv_repeats']['accuracy']['mean']:.1f} ± "
                                                         f"{100 * m['cv_repeats']['accuracy']['std']:.1f}"})
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ obstruction benchmark
OBSTRUCTION_VLM_METHODS = ["viddar", "end_to_end", "underdetailed", "greedy"]
OBSTRUCTION_BASELINES = ["prior", "saliency", "canny"]  # no VLM: stored under <root>/obstruction/no_vlm
NO_VLM = "no_vlm"
CAUSE_NAMES_OBSTRUCTION = {
    "illegible": "ilegível (sem sim/não)",
    "no_detection": "nada detectado",
    "wrong_object": "objeto errado (IoU < 0,5)",
    "threshold": "objeto certo, razão do lado errado de α",
    "decision": "decisão (método sem máscara)",
}


def load_obstruction_runs(root: Path, models: list[str], methods: list[str] = OBSTRUCTION_VLM_METHODS,
                          baselines: list[str] = OBSTRUCTION_BASELINES) -> dict[tuple[str, str], pd.DataFrame]:
    """Predictions of every finished (model, method) obstruction run; baselines under (NO_VLM, method)."""
    keys = [(m, meth) for m in models for meth in methods] + [(NO_VLM, b) for b in baselines]
    runs = {}
    for model, method in keys:
        f = Path(root) / "obstruction" / model / method / "predictions.csv"
        if f.exists():
            runs[(model, method)] = pd.read_csv(f, dtype={"id": str})
    return runs


def obstruction_benchmark(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Table 1 of the paper per (model, method), plus precision, recall, FPR, FNR and the accuracy CI.

    Key-object recognition has two criteria: lexical (the VLM's phrase names the class) and grounded
    (the segmented mask overlaps the true one, IoU >= 0.5); mIoU is the mean IoU of the key-object mask.
    """
    from .metrics import binary_metrics
    from .obstruction import GROUNDED_IOU

    rows = []
    for (model, method), df in runs.items():
        m = binary_metrics(df["label"], df["pred"].astype(object).where(df["pred"].notna(), None))
        row = {"modelo": model, "método": method}
        if method in OBSTRUCTION_VLM_METHODS and "key_recognized" in df:
            row["objeto-chave (léxico)"] = 100 * df["key_recognized"].astype(bool).mean()
        if "key_iou" in df:
            row["objeto-chave (IoU ≥ 0,5)"] = 100 * (df["key_iou"] >= GROUNDED_IOU).mean()
            row["mIoU"] = 100 * df["key_iou"].mean()
        row.update({"acurácia": 100 * m["accuracy"]["value"], "acurácia IC95": ci(m["accuracy"]),
                    "precisão": 100 * m["precision"]["value"], "recall": 100 * m["recall"]["value"],
                    "FPR": 100 * m["fpr"]["value"], "FNR": 100 * m["fnr"]["value"],
                    "FP": m["fp"], "FN": m["fn"], "sem resposta": m["no_answer"]})
        rows.append(row)
    return _rounded(pd.DataFrame(rows))


def obstruction_vs(runs: dict[tuple[str, str], pd.DataFrame], models: list[str], method: str = "viddar",
                   other: str = "end_to_end") -> pd.DataFrame:
    """`method` against `other` on the same samples, per model (McNemar): the paper's main comparison."""
    from .metrics import binary_metrics

    rows = []
    for model in models:
        a, b = runs.get((model, method)), runs.get((model, other))
        if a is None or b is None:
            continue
        t = _mcnemar(a, "pred", b)
        acc_a = 100 * binary_metrics(a["label"], a["pred"].astype(object).where(a["pred"].notna(), None))["accuracy"]["value"]
        acc_b = 100 * binary_metrics(b["label"], b["pred"].astype(object).where(b["pred"].notna(), None))["accuracy"]["value"]
        rows.append({"modelo": model, f"acurácia {method}": acc_a, f"acurácia {other}": acc_b,
                     f"só {method} acerta": t["a_only"], f"só {other} acerta": t["b_only"],
                     "McNemar p": t["p_value"],
                     f"{method} vs {other}": verdict(acc_a - acc_b, t["p_value"])})
    return _rounded(pd.DataFrame(rows))


def obstruction_cause_comparison(runs: dict[tuple[str, str], pd.DataFrame], models: list[str],
                                 method: str = "viddar") -> pd.DataFrame:
    """Errors of `method` by pipeline stage, one column per model ("k/n"; in brackets, how many of
    those samples the `prior` baseline, which receives the true object name, also gets wrong)."""
    from .obstruction import cause_table

    prior = runs.get((NO_VLM, "prior"))
    tables = {}
    for model in models:
        df = runs.get((model, method))
        if df is not None:
            tables[model] = cause_table(df, prior).set_index(["error", "cause"])
    if not tables:
        return pd.DataFrame()
    index = sorted(set().union(*[t.index for t in tables.values()]),
                   key=lambda k: (k[0] != "FP", list(CAUSE_NAMES_OBSTRUCTION).index(k[1])))
    rows = []
    for kind, cause in index:
        row = {"erro": kind, "causa": CAUSE_NAMES_OBSTRUCTION[cause]}
        for model, t in tables.items():
            if (kind, cause) in t.index:
                r = t.loc[(kind, cause)]
                cell = f"{int(r['k'])}/{int(r['n'])}"
                if "prior_also_wrong" in r:
                    cell += f" (prior: {int(r['prior_also_wrong'])})"
                row[model] = cell
            else:
                row[model] = "0"
        rows.append(row)
    return pd.DataFrame(rows)


def top_key_phrases(runs: dict[tuple[str, str], pd.DataFrame], models: list[str], method: str = "viddar",
                    n: int = 5) -> pd.DataFrame:
    """Most frequent object phrases per model: repeated answers that do not fit the scenes (e.g. copied
    from the prompt's example) point to a prompt problem rather than to perception."""
    cols = {}
    for model in models:
        df = runs.get((model, method))
        if df is not None and "key_phrase" in df:
            vc = df["key_phrase"].fillna("").value_counts().head(n)
            cols[model] = [f"{p} ({c})" for p, c in vc.items()] + [""] * (n - len(vc))
    return pd.DataFrame(cols, index=range(1, n + 1))


def obstruction_by_group(runs: dict[tuple[str, str], pd.DataFrame], models: list[str], method: str = "viddar",
                         metric: str = "accuracy") -> pd.DataFrame:
    """`metric` (accuracy, fpr or fnr, in %) of `method` per object class (rows) and model (columns)."""
    from .metrics import metrics_by_group

    cols = {}
    for model in models:
        df = runs.get((model, method))
        if df is not None:
            g = metrics_by_group(df.assign(pred=df["pred"].astype(object).where(df["pred"].notna(), None)))
            cols[model] = (100 * g.set_index("group")[metric]).round(1)
    out = pd.DataFrame(cols)
    if len(out):
        counts = next(iter(runs.values())).groupby("group")["label"].agg(n="size", obstruídos="sum")
        out = counts.join(out)
    return out
