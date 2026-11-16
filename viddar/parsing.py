"""Post-processing of VLM text responses."""

from __future__ import annotations

import re

_NON_ALPHA = re.compile(r"[^a-z]")


def _words(text: str) -> list[str]:
    return [_NON_ALPHA.sub("", w) for w in text.lower().split()]


def last_yes_no(text: str) -> int | None:
    """Paper's post-processing (Sec. 4.2): whichever of "yes"/"no" appears closest to the end.

    Same rule as `check_last_occurrence` in the original notebook, but punctuation and
    markdown around the word are stripped generically instead of via a fixed list.
    Returns 1 (yes), 0 (no) or None (neither found).
    """
    for w in reversed(_words(text)):
        if w == "yes":
            return 1
        if w == "no":
            return 0
    return None


def first_yes_no(text: str) -> int | None:
    for w in _words(text):
        if w == "yes":
            return 1
        if w == "no":
            return 0
    return None


_QUESTION_HEADER = re.compile(r"(?m)^[\s>*#\-]*(?:\*\*)?\s*(?:question\s*)?([1-9])\s*[\.\):]")


def split_numbered_answers(text: str) -> dict[int, str]:
    """Split a response to a numbered questionnaire into {question_number: answer_text}."""
    matches = list(_QUESTION_HEADER.finditer(text))
    out: dict[int, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        q = int(m.group(1))
        out.setdefault(q, text[m.end():end].strip())
    return out


_LEADING_PHRASES = re.compile(
    r"^(the\s+)?(key|most important|important)\s+object\s+(in the image\s+)?(is|would be)\s*:?\s*",
    re.IGNORECASE,
)


def clean_object_phrase(text: str) -> str:
    """Turn a free-form 'key object' answer into a short noun phrase for Grounding DINO."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    phrase = lines[0] if lines else ""
    phrase = re.sub(r"^[\s\-\*\d\.\)#>]+", "", phrase)  # bullets / numbering
    phrase = _LEADING_PHRASES.sub("", phrase)
    phrase = phrase.replace("*", "").strip(" \"'`.:;,")
    return phrase.lower()


_IS_THE = re.compile(r"\b(?:is|are|would be)\s+(?:the|an|a)\s+([^.,;:\n]+)", re.IGNORECASE)
# Inverted form: "The tablet is the 'key object' interacting with ..."
_SUBJECT_IS_KEY = re.compile(r"^(?:the|an|a)?\s*(.+?)\s+(?:is|are)\s+(?:the|an|a)\s+['\"]?key object", re.IGNORECASE)
_TRAILING_CLAUSE = re.compile(
    r"\s+(?:which|that|where|on|in|at|with|next|near|beside|placed|located|positioned|mounted|attached"
    r"|above|below|under|behind|inside|of the|or|displaying|showing)\b.*$", re.IGNORECASE)


def extract_object_phrase(text: str) -> str:
    """Noun phrase for Grounding DINO from a sentence such as "The key object ... is the black fan."

    Short answers ("black fan") fall back to `clean_object_phrase`; trailing location clauses
    ("the wall above the door") are cut so the detector receives the object itself.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).replace("*", "")
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    first = re.sub(r"^.*?key object['\"]?\s*:\s*", "", first, flags=re.IGNORECASE)
    m = _SUBJECT_IS_KEY.search(first) or _IS_THE.search(first)
    phrase = m.group(1) if m else clean_object_phrase(first)
    phrase = _TRAILING_CLAUSE.sub("", phrase)
    return phrase.strip(" \"'`.:;,").lower()


def parse_object_list(text: str, max_objects: int = 10) -> list[str]:
    """Parse the Greedy baseline's answer into individual object phrases."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    parts = re.split(r"\n|,|;|\band\b", text)
    objects: list[str] = []
    for p in parts:
        p = clean_object_phrase(p)
        if p and len(p.split()) <= 6 and p not in objects:
            objects.append(p)
    return objects[:max_objects]


# --- Key object recognition ---------------------------------------------------------------
# The paper does not state how a VLM answer is judged against the key-object label, so a
# transparent lexical rule is used: the answer is correct when it contains at least one of the
# class's keyword phrases (after lower-casing and naive singularisation). Keyword lists were
# written by looking at one sample image per class.
KEY_OBJECT_SYNONYMS: dict[str, list[str]] = {
    "tv": ["tv", "television", "monitor", "screen", "display"],
    "boiling pot": ["pot", "pan", "boiling", "stove", "cooktop", "burner", "saucepan"],
    "caution sign": ["caution", "wet floor", "floor sign"],
    "ceiling fan": ["fan"],
    "coffee maker": ["coffee", "espresso", "brewer", "grinder"],
    "electrical box": ["electrical box", "electric box", "utility box", "transformer", "electrical cabinet",
                       "power box", "junction box", "electrical panel", "switch box", "metal box", "box"],
    "electrical outlet": ["outlet", "socket", "receptacle", "plug"],
    "electrical pot": ["pressure cooker", "instant pot", "rice cooker", "slow cooker", "cooker",
                       "electric pot", "electrical pot", "multicooker", "kettle"],
    "exit sign": ["exit"],
    "fire extinguisher": ["extinguisher"],
    "fire hydrant": ["hydrant"],
    "help station": ["help", "emergency call", "emergency phone", "call box", "call station",
                     "emergency station", "emergency pole", "emergency post"],
    "knife": ["knife", "knive", "cleaver", "blade"],
    "microwave oven": ["microwave"],
    "parking sign": ["parking"],
    "pedestrian sign": ["pedestrian", "crosswalk", "crossing"],
    "rules board": ["rule", "board", "notice", "plaque", "information sign", "info sign", "placard",
                    "instruction"],
    "scissors": ["scissor", "shear"],
    "stop sign": ["stop"],
    "toaster": ["toaster"],
    "traffic light": ["traffic light", "traffic signal", "stoplight", "signal light"],
    "vehicle": ["vehicle", "car", "van", "truck", "suv", "bus"],
    "warning sign": ["warning", "hazard", "danger", "prohibit", "no entry", "restricted", "caution"],
}


def _normalise(text: str) -> str:
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]
    return " " + " ".join(words) + " "


def key_object_matches(prediction: str, label: str) -> bool:
    pred = _normalise(prediction)
    keywords = KEY_OBJECT_SYNONYMS.get(label.lower().strip(), [label])
    return any(_normalise(k) in pred for k in keywords)
