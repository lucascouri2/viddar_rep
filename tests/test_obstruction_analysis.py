import pandas as pd

from viddar.obstruction import cause_table, error_causes


def _runs():
    # id, label, pred, n_detections, key_iou
    rows = [("1", 1, 1, 1, 0.9),   # TP
            ("2", 0, 1, 1, 0.2),   # FP, another object segmented
            ("3", 0, 1, 1, 0.8),   # FP, right object, ratio above alpha
            ("4", 1, 0, 0, 0.0),   # FN, nothing detected
            ("5", 1, 0, 1, 0.7)]   # FN, right object, ratio below alpha
    df = pd.DataFrame(rows, columns=["id", "label", "pred", "n_detections", "key_iou"])
    prior = pd.DataFrame({"id": ["1", "2", "3", "4", "5"], "label": [1, 0, 0, 1, 1], "pred": [1, 0, 1, 1, 0]})
    return df, prior


def test_obstruction_error_causes():
    df, _ = _runs()
    assert list(error_causes(df)) == ["correct", "FP:wrong_object", "FP:threshold", "FN:no_detection",
                                      "FN:threshold"]
    e2e = pd.DataFrame({"id": ["1", "2"], "label": [1, 0], "pred": [None, 1]})
    assert list(error_causes(e2e)) == ["FN:illegible", "FP:decision"]


def test_obstruction_cause_table_counts_prior_errors():
    df, prior = _runs()
    t = cause_table(df, prior).set_index(["error", "cause"])
    assert t.loc[("FP", "threshold"), "k"] == 1 and t.loc[("FP", "threshold"), "n"] == 2
    assert t.loc[("FP", "threshold"), "prior_also_wrong"] == 1  # the prior also flags sample 3
    assert t.loc[("FP", "wrong_object"), "prior_also_wrong"] == 0  # only the VLM's object was wrong
    assert t.loc[("FN", "threshold"), "prior_also_wrong"] == 1
