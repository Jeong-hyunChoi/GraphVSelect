"""Geometric primitives shared across the pipeline (pure functions, no I/O).

All boxes are ``(x1, y1, x2, y2)`` in absolute pixel coordinates.
"""
from __future__ import annotations


def iou_xyxy(a, b):
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def expand_bbox(bbox, image_w, image_h, pad_ratio=0.3):
    """Pad each side of ``bbox`` by ``pad_ratio`` x side length, clamped to the image.

    Used when a crop should include a margin of surrounding context.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = bw * pad_ratio, bh * pad_ratio
    return (
        int(max(0, x1 - pad_w)),
        int(max(0, y1 - pad_h)),
        int(min(image_w, x2 + pad_w)),
        int(min(image_h, y2 + pad_h)),
    )


def pair_overlaps(box_a, box_b, margin=0.2):
    """Whether two boxes are spatially close enough to warrant a relation query.

    Each box is expanded by ``margin`` x side length on every side; the pair is kept
    when the expanded boxes intersect. This gates which candidate pairs are sent to
    the relation model during scene-graph construction.
    """
    def expand(b):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        return (x1 - margin * w, y1 - margin * h,
                x2 + margin * w, y2 + margin * h)
    a = expand(box_a)
    b = expand(box_b)
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
