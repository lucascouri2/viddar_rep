"""Information manipulation attack detection (paper Sec. 3.2, 4.2, 5.2).

The VLM receives (I_r, I_a) and the six-question prompt. The final decision follows the
paper's post-processing: whichever of "yes"/"no" appears last in the answer. In addition,
the answers to questions 3, 4 and 5 are parsed as predictions of the factors
A (alignment), S (style similarity) and I (information misrepresentation), which makes it
possible to see *which* factor drives false positives, and to compare the VLM's own final
answer with the rule M = A ∧ S ∧ I (Eq. 7).
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from . import prompts
from .metrics import binary_metrics
from .parsing import first_yes_no, last_yes_no, split_numbered_answers

FACTOR_QUESTIONS = {"A": 3, "S": 4, "I": 5}


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
        records.append(rec)
    out = pd.concat([df, pd.DataFrame(records)], axis=1)
    factors = out[[f"pred_{f}" for f in FACTOR_QUESTIONS]]
    out["pred_factor_rule"] = [
        None if any(pd.isna(v) for v in vals) else int(all(v == 1 for v in vals))
        for vals in factors.itertuples(index=False)
    ]
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
    return out
