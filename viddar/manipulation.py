"""Information manipulation attack detection (paper Sec. 3.2, 4.2, 5.2).

The VLM receives (I_r, I_a) and the six-question prompt. The final decision follows the
paper's post-processing: whichever of "yes"/"no" appears last in the answer. In addition,
the answers to questions 3, 4 and 5 are parsed as predictions of the factors
A (alignment), S (style similarity) and I (information misrepresentation), which makes it
possible to see *which* factor drives false positives, and to compare the VLM's own final
answer with the rule M = A ∧ S ∧ I (Eq. 7).

`run_decomposed` is the variant A2 of report/analise_fp_fn_manipulacao.md: one call for the
objects and one per factor, with M computed in code. The error-attribution functions at the end
implement Sec. 1.3 of that report and only read saved predictions.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from . import prompts
from .metrics import binary_metrics, wilson_ci
from .parsing import clean_object_phrase, extract_object_phrase, first_yes_no, last_yes_no, split_numbered_answers

FACTOR_QUESTIONS = {"A": 3, "S": 4, "I": 5}
FACTORS = tuple(FACTOR_QUESTIONS)


def _and(values) -> int | None:
    """M = A ∧ S ∧ I over parsed answers; None when any answer could not be parsed."""
    return None if any(pd.isna(v) for v in values) else int(all(v == 1 for v in values))


def _add_factor_rule(out: pd.DataFrame) -> pd.DataFrame:
    out["pred_factor_rule"] = [_and(vals) for vals in out[[f"pred_{f}" for f in FACTORS]].itertuples(index=False)]
    return out


def run(df: pd.DataFrame, vlm, prompt_variant: str = "paper") -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    prompt = prompts.MANIPULATION[prompt_variant]
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="manipulation: VLM"):
        resp = vlm.ask([row["raw_path"], row["ar_path"]], prompt)
        answers = split_numbered_answers(resp)
        rec = {"vlm_response": resp, "pred": last_yes_no(resp)}
        for factor, q in FACTOR_QUESTIONS.items():
            rec[f"pred_{factor}"] = first_yes_no(answers[q]) if q in answers else None
        rec["virtual_content"] = answers.get(1, "")[:200]
        rec["key_object"] = answers.get(2, "")[:200]
        rec["key_phrase"] = extract_object_phrase(answers.get(2, ""))
        records.append(rec)
    return _add_factor_rule(pd.concat([df, pd.DataFrame(records)], axis=1))


def run_decomposed(df: pd.DataFrame, vlm, prompt_variant: str = "paper") -> pd.DataFrame:
    """One call for the virtual content and key object, then one call per factor (A, S, I).

    There is no question 6: the decision `pred` is M = A ∧ S ∧ I computed from the three answers,
    which removes by construction the "conclusion error" of the joint prompt.
    """
    df = df.copy().reset_index(drop=True)
    templates = prompts.MANIPULATION_DECOMPOSED[prompt_variant]
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="manipulation (decomposed): VLM"):
        images = [row["raw_path"], row["ar_path"]]
        resp_obj = vlm.ask(images, templates["objects"])
        answers = split_numbered_answers(resp_obj)
        content = clean_object_phrase(answers.get(1, "")) or "the added virtual object"
        key = extract_object_phrase(answers.get(2, "")) or "the real object it touches"
        rec = {"resp_objects": resp_obj, "virtual_content": content, "key_object": key, "key_phrase": key}
        parts = [f"[objects] {resp_obj}"]
        for factor in FACTORS:
            resp = vlm.ask(images, templates[factor].format(content=content, key_obj=key))
            rec[f"resp_{factor}"] = resp
            rec[f"pred_{factor}"] = first_yes_no(resp)
            parts.append(f"[{factor}] {resp}")
        rec["vlm_response"] = "\n".join(parts)
        records.append(rec)
    out = _add_factor_rule(pd.concat([df, pd.DataFrame(records)], axis=1))
    out["pred"] = out["pred_factor_rule"]
    return out


def summarize(df: pd.DataFrame) -> dict:
    out = {
        "detection": binary_metrics(df["label"], df["pred"]),
        # Decision recomputed as A ∧ S ∧ I from the parsed answers to questions 3-5.
        "detection_factor_rule": binary_metrics(df["label"], df["pred_factor_rule"]),
        "unparsed_factor_answers": int(df["pred_factor_rule"].isna().sum()),
        "final_answer_consistent_with_factors": float(
            (df["pred"] == df["pred_factor_rule"])[df["pred_factor_rule"].notna()].mean()
        ),
    }
    for factor in FACTOR_QUESTIONS:
        if factor in df.columns:
            out[f"factor_{factor}"] = binary_metrics(df[factor].astype(int), df[f"pred_{factor}"])
    if all(f in df.columns for f in FACTORS):
        out["error_rates"] = error_rates(df)
        out["error_causes"] = {k: int(v) for k, v in error_causes(df).value_counts().items()}
        out["oracle"] = oracle_rates(df).to_dict(orient="records")
    return out


# ------------------------------------------------------------------------- error attribution
def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"value": k / n if n else float("nan"), "k": int(k), "n": int(n), "ci95": [lo, hi]}


def error_rates(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """FPR over all benign pairs and FNR over all attacks; an unparsed decision on an attack is a FN."""
    pred = df[pred_col]
    neg, pos = df["label"] == 0, df["label"] == 1
    fp = int((neg & (pred == 1)).sum())
    fn = int((pos & (pred != 1)).sum())
    correct = int(((pred == df["label"]) & pred.notna()).sum())
    return {"fpr": _rate(fp, int(neg.sum())), "fnr": _rate(fn, int(pos.sum())),
            "accuracy": _rate(correct, len(df))}


def predicted_rule(df: pd.DataFrame, oracle: tuple[str, ...] = ()) -> pd.Series:
    """M = A ∧ S ∧ I from the VLM's factor answers, with the factors in `oracle` replaced by the truth."""
    cols = [f if f in oracle else f"pred_{f}" for f in FACTORS]
    return pd.Series([_and(vals) for vals in df[cols].itertuples(index=False)], index=df.index, dtype=object)


def oracle_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Counterfactual: FPR/FNR of A ∧ S ∧ I when one or two factors come from the ground truth."""
    rows = []
    for oracle in [(), ("A",), ("S",), ("I",), ("A", "S"), ("A", "I"), ("S", "I")]:
        r = error_rates(df.assign(_rule=predicted_rule(df, oracle)), "_rule")
        rows.append({"oracle": "+".join(oracle) or "none", "fpr": r["fpr"]["value"], "fp": r["fpr"]["k"],
                     "fnr": r["fnr"]["value"], "fn": r["fnr"]["k"]})
    return pd.DataFrame(rows)


def factor_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per factor: wrong acceptance P(answer ≠ no | F = 0) and wrong rejection P(answer ≠ yes | F = 1).

    Unparsed answers count as wrong in both directions.
    """
    rows = []
    for f in FACTORS:
        p, truth = df[f"pred_{f}"], df[f]
        k0, n0 = int(((truth == 0) & (p != 0)).sum()), int((truth == 0).sum())
        k1, n1 = int(((truth == 1) & (p != 1)).sum()), int((truth == 1).sum())
        rows.append({"factor": f, "accepted_when_0": _rate(k0, n0), "rejected_when_1": _rate(k1, n1)})
    return pd.DataFrame(rows)


def error_causes(df: pd.DataFrame, pred_col: str = "pred") -> pd.Series:
    """Cause of each error (Sec. 1.3 of the report): illegible, conclusion or perception.

    Values: "correct", "unanswered" (benign pair without decision), "FP:illegible", "FP:conclusion",
    "FP:accepted:<factors>" (accepted every factor whose label is 0), "FN:illegible",
    "FN:conclusion", "FN:rejected:<factors>" (rejected factors whose label is 1).
    """
    causes = []
    for _, r in df.iterrows():
        label, pred = int(r["label"]), r[pred_col]
        if label == 0 and pd.isna(pred):
            causes.append("unanswered")
            continue
        if not ((label == 0 and pred == 1) or (label == 1 and pred != 1)):
            causes.append("correct")
            continue
        kind = "FP" if label == 0 else "FN"
        answers = [r[f"pred_{f}"] for f in FACTORS]
        if pd.isna(pred) or any(pd.isna(a) for a in answers):
            causes.append(f"{kind}:illegible")
            continue
        rule = _and(answers)
        if (kind == "FP" and rule == 0) or (kind == "FN" and rule == 1):
            causes.append(f"{kind}:conclusion")
        elif kind == "FP":
            causes.append("FP:accepted:" + "+".join(f for f in FACTORS if r[f] == 0 and r[f"pred_{f}"] == 1))
        else:
            causes.append("FN:rejected:" + "+".join(f for f in FACTORS if r[f"pred_{f}"] == 0))
    return pd.Series(causes, index=df.index)


def _zero_set(r) -> str:
    return "+".join(f for f in FACTORS if r[f] == 0)


def cause_table(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Error counts per cause, each over the pairs where that error can happen (comparable across runs).

    A perception FP "accepted:A" can only happen on benign pairs whose only factor at 0 is A, so its
    denominator is the number of such pairs; every other cause is counted over all benign pairs (FP)
    or all attacks (FN).
    """
    causes = error_causes(df, pred_col)
    neg, pos = df["label"] == 0, df["label"] == 1
    zero_sets = df[neg].apply(_zero_set, axis=1)
    rows = [("FP", "illegible", int(neg.sum())), ("FP", "conclusion", int(neg.sum()))]
    rows += [("FP", f"accepted:{z}", int(n)) for z, n in zero_sets.value_counts().sort_index().items()]
    fn_causes = sorted({c.split(":", 1)[1] for c in causes if c.startswith("FN:")} | {"illegible", "conclusion"},
                       key=lambda c: (c != "illegible", c != "conclusion", c))
    rows += [("FN", c, int(pos.sum())) for c in fn_causes]
    table = pd.DataFrame(rows, columns=["error", "cause", "eligible"])
    table["k"] = [int((causes == f"{e}:{c}").sum()) for e, c in zip(table["error"], table["cause"])]
    return table
