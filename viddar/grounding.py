"""Open-set detection (Grounding DINO) + box-prompted segmentation (SAM).

Mirrors what the original notebook got from `lang_sam.LangSAM(sam_type="vit_b")`:
Grounding DINO with box_threshold=0.3 / text_threshold=0.25, then SAM ViT-B with one
mask per box (multimask_output=False). The paper used Grounding DINO 1.5, which is only
available through a closed API, so the open Grounding DINO checkpoints are used here.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class Detection:
    phrase: str
    boxes: np.ndarray  # (n, 4) xyxy in pixels
    scores: np.ndarray  # (n,)
    masks: np.ndarray  # (n, H, W) bool


class GroundedSegmenter:
    def __init__(
        self,
        detector_id: str = "IDEA-Research/grounding-dino-base",
        sam_id: str = "facebook/sam-vit-base",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.det_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device).eval()
        self.sam_processor = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(device).eval()

    @staticmethod
    def _as_query(phrase: str) -> str:
        # Grounding DINO expects lower-case text terminated by a period.
        phrase = phrase.strip().lower().rstrip(".")
        return phrase + "."

    @torch.inference_mode()
    def detect(self, image: Image.Image, phrase: str) -> tuple[np.ndarray, np.ndarray]:
        inputs = self.det_processor(images=image, text=self._as_query(phrase), return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)
        res = self.det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        return res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy()

    @torch.inference_mode()
    def segment(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, image.height, image.width), dtype=bool)
        inputs = self.sam_processor(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
        outputs = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]  # (n_boxes, 1, H, W)
        return masks[:, 0].numpy().astype(bool)

    def detect_and_segment(self, image: Image.Image, phrase: str, top1: bool = True) -> Detection:
        """Detect `phrase`; with `top1`, keep only the highest-confidence box (notebook default)."""
        boxes, scores = self.detect(image, phrase)
        if top1 and len(scores):
            best = int(np.argmax(scores))
            boxes, scores = boxes[best:best + 1], scores[best:best + 1]
        return Detection(phrase, boxes, scores, self.segment(image, boxes))

    def close(self) -> None:
        del self.detector, self.sam
        gc.collect()
        torch.cuda.empty_cache()
