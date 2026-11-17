import numpy as np
import pandas as pd

from viddar import prompts
from viddar.alignment import cv_predict, geometry_features, hybrid
from viddar.manipulation import cause_table, error_causes, error_rates, factor_rates, oracle_rates, run_decomposed
from viddar.reporting import verdict
from viddar.metrics import auc, mcnemar_exact
from viddar.parsing import extract_object_phrase


def _frame(rows):
    """rows: (A, S, I, pred_A, pred_S, pred_I, pred)"""
    df = pd.DataFrame(rows, columns=["A", "S", "I", "pred_A", "pred_S", "pred_I", "pred"])
    df["label"] = (df["A"] & df["S"] & df["I"]).astype(int)
    return df


def test_extract_object_phrase():
    assert extract_object_phrase("The key object interacting with the virtual content is the black fan.") == "black fan"
    assert extract_object_phrase("The 'key object' is the tablet, which is placed on the desk.") == "tablet"
    assert extract_object_phrase("The key object is the wall above the sliding glass door.") == "wall"
    assert extract_object_phrase("Black speaker") == "black speaker"
    assert extract_object_phrase("**Key object:** electric scooter") == "electric scooter"
    assert extract_object_phrase("The computer monitor is the 'key object' interacting with it.") == "computer monitor"
    assert extract_object_phrase("The key object ... is a tablet or a smartphone, which is on the desk.") == "tablet"


def test_prompt_variants_differ_from_paper_only_where_intended():
    paper, neutral, definition = (prompts.MANIPULATION[k] for k in ("paper", "neutral", "definition"))
    assert "Think creatively" in paper and "Think creatively" not in neutral
    assert "may contain virtual content" in neutral
    assert definition.startswith(paper.split("5. ")[0]) and "Information misrepresentation means" in definition
    dec = prompts.MANIPULATION_DECOMPOSED["definition"]
    assert "{key_obj}" in dec["A"] and "Information misrepresentation means" in dec["I"]
    assert "Information misrepresentation means" not in prompts.MANIPULATION_DECOMPOSED["paper"]["I"]


def test_error_causes_and_rates():
    df = _frame([
        (1, 1, 1, 1, 1, 1, 1),  # TP
        (1, 1, 1, 1, 1, 0, 0),  # FN, rejected I
        (1, 1, 1, 1, 1, 1, 0),  # FN, conclusion
        (0, 1, 1, 1, 1, 1, 1),  # FP, accepted A
        (1, 0, 1, 1, 0, 1, 1),  # FP, conclusion (factors say no, final says yes)
        (1, 1, 0, 1, 1, 0, 0),  # TN
        (1, 1, 1, None, 1, 1, None),  # FN, illegible
    ])
    assert list(error_causes(df)) == ["correct", "FN:rejected:I", "FN:conclusion", "FP:accepted:A",
                                      "FP:conclusion", "correct", "FN:illegible"]
    r = error_rates(df)
    assert (r["fpr"]["k"], r["fpr"]["n"], r["fnr"]["k"], r["fnr"]["n"]) == (2, 3, 3, 4)
    t = cause_table(df).set_index(["error", "cause"])
    assert t.loc[("FP", "accepted:A"), "eligible"] == 1 and t.loc[("FP", "accepted:A"), "k"] == 1
    assert t.loc[("FP", "conclusion"), "eligible"] == 3 and t.loc[("FN", "rejected:I"), "eligible"] == 4
    fr = factor_rates(df).set_index("factor")
    assert fr.loc["A", "accepted_when_0"]["k"] == 1 and fr.loc["I", "rejected_when_1"]["k"] == 1


def test_oracle_replaces_one_factor_by_the_truth():
    df = _frame([(0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1)])
    o = oracle_rates(df).set_index("oracle")
    assert o.loc["none", "fpr"] == 1.0 and o.loc["A", "fpr"] == 0.0 and o.loc["A", "fnr"] == 0.0


def test_mcnemar_and_auc():
    assert mcnemar_exact([1, 1, 1], [1, 1, 1])["p_value"] == 1.0
    m = mcnemar_exact([1] * 10, [0] * 10)
    assert m["a_only"] == 10 and m["p_value"] < 0.01
    assert auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5


def test_geometry_features():
    content = np.zeros((100, 100), bool)
    content[10:20, 40:60] = True  # 20 px wide
    touching = np.zeros_like(content)
    touching[20:60, 30:70] = True  # directly below the content
    far = np.zeros_like(content)
    far[50:90, 30:70] = True  # 30 px gap
    f_touch, f_far = geometry_features(content, touching), geometry_features(content, far)
    assert f_touch["gap_px"] <= 1.5 and f_far["gap_px"] >= 29
    assert f_far["gap_norm"] > f_touch["gap_norm"]
    assert f_touch["contact"] > f_far["contact"] == 0.0
    assert np.isnan(geometry_features(content, np.zeros_like(content))["gap_px"])


def test_cv_predict_and_hybrid():
    rng = np.random.default_rng(0)
    y = np.array([1] * 30 + [0] * 20)
    gap = np.where(y == 1, rng.uniform(0, 0.1, 50), rng.uniform(0.3, 1.0, 50))
    gap[0] = np.nan  # no detection
    pred = cv_predict(gap, y, higher_is_aligned=False)
    assert np.isnan(pred[0]) and np.all(pred[1:] == y[1:])
    preds = pd.DataFrame({"pred_A": [0, 1], "pred_S": [1, 1], "pred_I": [1, 1]})
    h = hybrid(preds, np.array([np.nan, 0.0]))
    assert list(h["a_source"]) == ["vlm", "geometric"] and list(h["pred_hybrid"]) == [0, 0]


class _FakeVLM:
    def __init__(self):
        self.prompts = []

    def ask(self, images, prompt):
        self.prompts.append(prompt)
        if "each with no more than 4 words" in prompt:
            return "1. Green plant\n2. The key object is the black speaker."
        return "No, there is a gap." if "aligned" in prompt else "Yes, it is."


def test_run_decomposed_computes_the_decision_in_code():
    df = pd.DataFrame({"id": ["1"], "raw_path": ["r.png"], "ar_path": ["a.png"], "label": [0],
                       "A": [0], "S": [1], "I": [1]})
    vlm = _FakeVLM()
    out = run_decomposed(df, vlm, "paper")
    assert len(vlm.prompts) == 4 and "black speaker" in vlm.prompts[1]
    r = out.iloc[0]
    assert (r["pred_A"], r["pred_S"], r["pred_I"], r["pred"]) == (0, 1, 1, 0)
    assert r["key_phrase"] == "black speaker" and r["virtual_content"] == "green plant"


def test_error_rates_has_the_paper_metrics():
    df = _frame([
        (1, 1, 1, 1, 1, 1, 1),  # TP
        (1, 1, 1, 1, 1, 0, 0),  # FN
        (0, 1, 1, 1, 1, 1, 1),  # FP
        (1, 1, 0, 1, 1, 0, 0),  # TN
    ])
    r = error_rates(df)
    assert (r["precision"]["k"], r["precision"]["n"]) == (1, 2)  # TP over pairs flagged as attacks
    assert (r["recall"]["k"], r["recall"]["n"]) == (1, 2)  # TP over attacks
    assert r["accuracy"]["value"] == 0.5


def test_verdict_needs_significance():
    assert verdict(5.0, 0.01) == "melhorou" and verdict(-5.0, 0.01) == "piorou"
    assert verdict(5.0, 0.2) == "sem diferença significativa"
    assert verdict(5.0, float("nan")) == "sem diferença significativa"
