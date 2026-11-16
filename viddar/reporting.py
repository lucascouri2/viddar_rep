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


def error_comparison(runs: dict[tuple[str, str], pd.DataFrame], reference: str = "paper") -> pd.DataFrame:
    """FPR, FNR and accuracy per run, with McNemar's test against the `reference` variant of the same model."""
    from .manipulation import error_rates
    from .metrics import mcnemar_exact

    rows = []
    for (model, variant), df in runs.items():
        r = error_rates(df)
        row = {"model": model, "variant": variant, "fpr": r["fpr"]["value"], "fnr": r["fnr"]["value"],
               "accuracy": r["accuracy"]["value"], "FPR % (k/n)": _kn(r["fpr"]), "FNR % (k/n)": _kn(r["fnr"]),
               "FPR IC95": ci(r["fpr"]), "FNR IC95": ci(r["fnr"])}
        ref = runs.get((model, reference))
        if ref is not None and variant != reference:
            m = df.merge(ref[["id", "pred"]], on="id", suffixes=("", "_ref"))
            ok = (m["pred"] == m["label"]) & m["pred"].notna()
            ok_ref = (m["pred_ref"] == m["label"]) & m["pred_ref"].notna()
            t = mcnemar_exact(ok, ok_ref)
            row.update({"só variante acerta": t["a_only"], f"só {reference} acerta": t["b_only"],
                        "McNemar p": t["p_value"]})
        rows.append(row)
    return pd.DataFrame(rows)


def _run_name(key: tuple[str, str]) -> str:
    return f"{key[0]} / {key[1]}"


def cause_comparison(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Error causes (Sec. 1.3 of the report) side by side, each cell "k/pairs where it can happen"."""
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
    return table.sort_values(["error", "order"], ascending=[False, True]).drop(columns="order").reset_index(drop=True)


def factor_comparison(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Wrong acceptance P(answer ≠ no | F = 0) and wrong rejection P(answer ≠ yes | F = 1), per run and factor."""
    from .manipulation import factor_rates

    rows = []
    for key, df in runs.items():
        row = {"model": key[0], "variant": key[1]}
        for _, r in factor_rates(df).iterrows():
            row[f"{r['factor']}: aceitou com 0"] = _kn(r["accepted_when_0"])
            row[f"{r['factor']}: rejeitou com 1"] = _kn(r["rejected_when_1"])
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
    """Path C2 results of each run that has <run>/alignment/metrics.json."""
    rows = []
    for model, variant in keys:
        f = Path(root) / "manipulation" / model / variant / "alignment" / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        aucs = {a["feature"]: a for a in m["feature_auc"]}
        g = aucs[m["feature"]]
        b, h, t = m["base"], m["hybrid"], m["mcnemar_hybrid_vs_base"]
        rows.append({
            "model": model, "variant": variant, "detected": f"{m['detected']}/{m['n']}",
            f"AUC {m['feature']} [IC95]": f"{g['auc']:.2f} [{g['ci95_low']:.2f}, {g['ci95_high']:.2f}]",
            "acc. A geométrico": m["factor_A_accuracy"]["geometric"],
            "acc. A do VLM (mesmos pares)": m["factor_A_accuracy"]["vlm_same_pairs"],
            "fpr": b["fpr"]["value"], "fnr": b["fnr"]["value"],
            "hybrid_fpr": h["fpr"]["value"], "hybrid_fnr": h["fnr"]["value"],
            "FPR base → híbrido": f"{_kn(b['fpr'])} → {_kn(h['fpr'])}",
            "FNR base → híbrido": f"{_kn(b['fnr'])} → {_kn(h['fnr'])}",
            "só híbrido acerta": t["a_only"], "só base acerta": t["b_only"], "McNemar p": t["p_value"],
            "FPR híbrido (média ± dp, CV)": f"{100 * m['cv_repeats']['fpr']['mean']:.1f} ± "
                                           f"{100 * m['cv_repeats']['fpr']['std']:.1f}",
        })
    return pd.DataFrame(rows)
