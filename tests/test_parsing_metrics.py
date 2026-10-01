import numpy as np

from viddar.data import derive_content_mask, obstruction_ratio
from viddar.metrics import binary_metrics
from viddar.parsing import (clean_object_phrase, first_yes_no, key_object_matches, last_yes_no,
                            parse_object_list, split_numbered_answers)


def test_last_yes_no_follows_paper_rule():
    assert last_yes_no("3. Yes, aligned. 5. No, nothing. 6. **Yes**.") == 1
    assert last_yes_no("Yes it is aligned... so the answer is no.") == 0
    assert last_yes_no("I cannot tell.") is None
    # "not" / "nothing" / "yesterday" must not count as no/yes
    assert last_yes_no("Yes. There is nothing notable yesterday") == 1


def test_numbered_answers_and_factors():
    resp = ("1. A plant.\n2. The speaker.\n3. Yes, it sits on top.\n"
            "4. No, the texture is flat.\n5. Yes, it looks like a pot.\n6. No")
    ans = split_numbered_answers(resp)
    assert [first_yes_no(ans[q]) for q in (3, 4, 5)] == [1, 0, 1]
    assert last_yes_no(resp) == 0


def test_numbered_answers_markdown_headers():
    resp = "**1. Virtual content:** a cup\n**2.** laptop\n**3.** No\n**4.** Yes\n**5.** Yes\n**6.** No"
    ans = split_numbered_answers(resp)
    assert first_yes_no(ans[3]) == 0 and first_yes_no(ans[5]) == 1


def test_object_phrases():
    assert clean_object_phrase("The key object is: Red fire extinguisher.") == "red fire extinguisher"
    assert clean_object_phrase("- Stop sign\n") == "stop sign"
    assert parse_object_list("1. stop sign\n2. trash can, bench and bicycle") == [
        "stop sign", "trash can", "bench", "bicycle"]


def test_key_object_matching():
    assert key_object_matches("red fire extinguisher", "fire extinguisher")
    assert key_object_matches("yellow scissors", "scissors")
    assert key_object_matches("power outlets", "electrical outlet")
    assert not key_object_matches("trash can", "stop sign")


def test_content_mask_and_ratio():
    raw = np.zeros((20, 20, 3), np.uint8)
    ar = raw.copy()
    ar[5:15, 5:15] = 200
    from PIL import Image
    m = derive_content_mask(Image.fromarray(raw), Image.fromarray(ar))
    assert m.sum() == 100
    key = np.zeros((20, 20), bool)
    key[0:10, 0:10] = True  # 25 of its 100 pixels are covered
    assert obstruction_ratio(key, m) == 0.25


def test_binary_metrics_counts_no_answer_as_wrong():
    m = binary_metrics([1, 0, 1, 0], [1, 1, None, 0])
    assert (m["tp"], m["fp"], m["tn"], m["fn"], m["no_answer"]) == (1, 1, 1, 0, 1)
    assert m["accuracy"]["value"] == 0.5
    assert m["fpr"]["value"] == 0.5
