"""Prompts used by ViDDAR and its baselines.

`paper` variants are transcribed verbatim from Sections 5.1.1 and 5.2.1 of the paper.
`notebook` variants are the (slightly different) prompts found in the original
notebooks under `Related Material/`, kept for comparison.
"""

KEY_OBJECT = {
    "paper": (
        "You are an expert in observing the world. Based on the scenario, identify the key object "
        "that needs people's attention or safety inspection in the image based on the scenario. "
        "Give only one object that you think is important to be noticed, and do not provide any "
        "other information. The objects can be caution information signs, electrical devices, "
        "safety equipment, etc. If you think the color is important, you can also mention the "
        "color, such as 'red box,' but be precise and describe the object with no more than 4 words."
    ),
    "notebook": (
        "Identify the key object that need people's attention or safety inspection in the image. "
        "Give only one object that you think is important to be noticed, output its name only and "
        "do not provide any other information. The objects can be caution information signs, "
        "electrical devices, safety equipments, etc. If you think a color is important, you can "
        "also mention the color, such as red box, but try to be precise and describe the object "
        "with no more than 4 words. "
    ),
}

# Underdetailed baseline (Sec. 5.1.1): no role assignment, no few-shot examples.
UNDERDETAILED = (
    "Identify the key object in the image. Give only one object that you think is important to be "
    "noticed. Give the name of the object only and do not provide any other information."
)

# Greedy baseline (Sec. 5.1.1): the ViDDAR prompt with "Give any object ..." instead of
# "Give only one object ...".
GREEDY = KEY_OBJECT["paper"].replace(
    "Give only one object that you think is important to be noticed",
    "Give any object that you think is important to be noticed",
)
assert GREEDY != KEY_OBJECT["paper"]

# End-to-end baseline, step 2 (Sec. 5.1.1). Images: raw first, augmented second.
END_TO_END_STEP2 = (
    "You are an expert in augmented content analysis. Look at both images. The first image is the "
    "raw image and there is a {key_obj} in it. The second image is an augmented image created by "
    "overlaying some virtual content on the raw image. Identify whether the virtual elements in the "
    "second image are obstructing the {key_obj}. If the {key_obj} is blocked or obfuscated, then "
    "answer Yes. If the {key_obj} is not blocked or obfuscated then answer No. The answer should "
    "contain only 'Yes' or 'No.'"
)

# Information manipulation detection (Sec. 5.2.1). Images: raw first, augmented second.
MANIPULATION = {
    "paper": (
        "Here are two images. The first one is a raw image, and the second one is an augmented "
        "image, created by adding some virtual content to the space.\n"
        "Please answer the following questions:\n"
        "1. What is the virtual content in the augmented image?\n"
        "2. What 'key object' is interacting with the virtual content? Avoid general terms like "
        "'table surface' or 'environment.'\n"
        "3. Is the virtual content accurately aligned to the object, without a significant gap? "
        "Answer yes or no, then explain why.\n"
        "4. Does the virtual content have a relatively high-quality yet reasonable texture that "
        "blends it into the real world? Answer yes or no, then explain why.\n"
        "5. Do you think the interaction will make users believe the 'key object' has some false "
        "functionality or information it does not have, or lose some true functionality or "
        "information it actually has? Think creatively, only say no if the combination has no "
        "specific semantic relation. Answer yes or no, then explain why.\n"
        "6. If you answered 'yes' in all questions 3, 4, and 5, you must say 'yes.' Otherwise, "
        "you say 'no.'"
    ),
    "notebook": (
        "Here are two images. The first one is a raw image, the second one is an augmented image, "
        "created by adding some virtual content to the space.\n"
        "Please answer the following questions:\n"
        "1.What is the virtual content in the augmented image?\n"
        "2.What \"key object\" is interacting with the virtual content? Avoid general terms like "
        "'table surface' or 'environment'.\n"
        "3.Is the virtual content accurately aligned to object, without significant gap? Answer "
        "yes or no, then explain why.\n"
        "4.Does the virtual content have a relatively high quality of texture? Answer yes or no, "
        "then explain why.\n"
        "5.Do you think the interaction will make users to believe the \"key object\" has some "
        "false functionality or features it doesn't have, \n"
        "or lost some true functionality it actually has? Think creatively, don't say \"no\" "
        "easily, only say no if the combination has no specific semantic relation. Answer yes or "
        "no, then explain why.\n"
        "6.If you answered 'yes' in all question 3, 4 and 5, you must say 'yes'. Otherwise you "
        "say 'no'."
    ),
}

# --- Variants for the false-positive study (report/analise_fp_fn_manipulacao.md, path A) -------
# Each variant is derived from the `paper` prompt by explicit replacements, so the diff to the
# paper is visible here and checked by the asserts.
_PAPER_PREAMBLE = "created by adding some virtual content to the space."
_NEUTRAL_PREAMBLE = "which may contain virtual content added to the space."
_CREATIVE = " Think creatively, only say no if the combination has no specific semantic relation."
_PAPER_Q5 = "5. Do you think the interaction"
# Definition of I transcribed from Sec. 3.2 of the paper ("Information Misrepresentation").
_I_DEFINITION = (
    "Information misrepresentation means that the combination of virtual content and the real "
    "world causes users to misunderstand the scene: they might overlook critical details or "
    "misinterpret non-existent information as real. Consider a user who takes the virtual content "
    "as part of the real world."
)

# A1 "neutral": no presupposition that I_a contains virtual content and no push towards "yes" in Q5.
MANIPULATION["neutral"] = (MANIPULATION["paper"].replace(_PAPER_PREAMBLE, _NEUTRAL_PREAMBLE)
                           .replace(_CREATIVE, ""))
# A3 "definition": the paper prompt with the paper's own definition of I added to Q5.
MANIPULATION["definition"] = MANIPULATION["paper"].replace(_PAPER_Q5, f"5. {_I_DEFINITION} Do you think the interaction")
assert MANIPULATION["neutral"].count(_PAPER_PREAMBLE) == 0 and _CREATIVE not in MANIPULATION["neutral"]
assert _I_DEFINITION in MANIPULATION["definition"]

# A2 "decomposed": one VLM call for the objects and one per factor; M = A ∧ S ∧ I is computed in
# code instead of asking question 6. Factor templates receive {content} and {key_obj} from call 1.
_OBJECTS_CALL = (
    "Please answer the following questions, each with no more than 4 words:\n"
    "1. What is the virtual content in the augmented image?\n"
    "2. What 'key object' is interacting with the virtual content? Avoid general terms like "
    "'table surface' or 'environment.'"
)
_FACTOR_CONTEXT = "The virtual content is: {content}. The key object interacting with it is: {key_obj}.\n"
_FACTOR_ANSWER = " Start your answer with yes or no, then explain why."
_Q_A = "Is the virtual content accurately aligned to the key object, without a significant gap?"
_Q_S = ("Does the virtual content have a relatively high-quality yet reasonable texture that blends "
        "it into the real world?")
_Q_I = ("Do you think the interaction will make users believe the key object has some false "
        "functionality or information it does not have, or lose some true functionality or "
        "information it actually has?")


def _decomposed(preamble: str, creative: bool, definition: bool) -> dict[str, str]:
    head = ("Here are two images. The first one is a raw image, and the second one is an augmented "
            f"image, {preamble}\n")
    q_i = (_I_DEFINITION + " " if definition else "") + _Q_I + (_CREATIVE if creative else "")
    return {
        "objects": head + _OBJECTS_CALL,
        "A": head + _FACTOR_CONTEXT + _Q_A + _FACTOR_ANSWER,
        "S": head + _FACTOR_CONTEXT + _Q_S + _FACTOR_ANSWER,
        "I": head + _FACTOR_CONTEXT + q_i + _FACTOR_ANSWER,
    }


MANIPULATION_DECOMPOSED = {
    "paper": _decomposed(_PAPER_PREAMBLE, creative=True, definition=False),
    "neutral": _decomposed(_NEUTRAL_PREAMBLE, creative=False, definition=False),
    "definition": _decomposed(_PAPER_PREAMBLE, creative=True, definition=True),
}
