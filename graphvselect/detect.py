"""Open-vocabulary detection with GroundingDINO.

Produces the initial candidate proposals for a referring expression. The detector is
prompted with a generic object vocabulary (rather than the raw expression) so that
detection stays a proposal step; object-level verification then keeps the
query-relevant proposals and removes false positives (see the Method section).

Frozen and inference-only. Box/text thresholds and the (optional) class-agnostic NMS
IoU follow the values reported in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .geometry import iou_xyxy

# COCO 80 categories — the default detection vocabulary.
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _format_vocab(cats: list[str]) -> str:
    # GroundingDINO expects period-separated, lowercase phrases with a terminating period.
    return ". ".join(c.strip().lower() for c in cats) + "."


@dataclass
class Detection:
    bbox: list[float]   # xyxy
    label: str
    score: float


def _label_overlap(la: str, lb: str) -> bool:
    """True if two phrase-labels share a word, i.e. they likely describe the same
    object under different phrasings (e.g. ``right`` vs ``guy man right``).
    GroundingDINO emits these as separate detections when fed multiple phrases."""
    wa = set(la.lower().split())
    wb = set(lb.lower().split())
    return bool(wa & wb)


def _class_agnostic_nms(dets: list, iou_thr: float = 0.7,
                        same_group_iou_thr: float = 0.5) -> list:
    """Greedy NMS, score-descending. Two boxes are merged (lower score dropped) when
    their IoU exceeds a threshold. The threshold is lower (more aggressive) when the
    labels share a word, since those are duplicate detections of one object; unrelated
    categories use the stricter ``iou_thr`` so genuinely different adjacent objects are
    not merged."""
    order = sorted(dets, key=lambda d: -d.score)
    kept: list = []
    for d in order:
        drop = False
        for k in kept:
            thr = same_group_iou_thr if _label_overlap(d.label, k.label) else iou_thr
            if iou_xyxy(d.bbox, k.bbox) > thr:
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


class GroundingDINODetector:
    def __init__(self, hf_id: str, box_threshold: float = 0.25, text_threshold: float = 0.20,
                 device: str | None = None, nms_iou: float | None = None):
        logger.info("Loading detector {} box_thr={} text_thr={} nms_iou={}",
                    hf_id, box_threshold, text_threshold, nms_iou)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_id).to(device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou   # None = off; else class-agnostic NMS IoU threshold
        logger.info("Detector ready on device={}", self.model.device)

    @torch.inference_mode()
    def __call__(self, image: Image.Image, text: str | None = None) -> list[Detection]:
        """Detect objects matching ``text``. Defaults to the full COCO vocabulary."""
        prompt = text if text is not None else _format_vocab(COCO_CATEGORIES)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],   # (H, W)
        )[0]
        labels = results.get("text_labels", results.get("labels", [""] * len(results["boxes"])))
        dets = [
            Detection(bbox=box.tolist(), label=str(label), score=float(score))
            for box, label, score in zip(results["boxes"], labels, results["scores"])
        ]
        if self.nms_iou is not None:
            dets = _class_agnostic_nms(dets, self.nms_iou)
        return dets
