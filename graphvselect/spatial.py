"""Relation gating used during scene-graph construction."""
from __future__ import annotations


def should_generate_relation(kind_i: str, kind_j: str) -> bool:
    """Only spend a relation query on pairs that involve a main object: main-main and
    main-context pairs are kept, context-context pairs are skipped."""
    if kind_i == "main" and kind_j == "main":
        return True
    if "main" in (kind_i, kind_j):
        return True
    return False


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _direction_from_centers(mc: tuple[float, float], cc: tuple[float, float],
                            image_w: int, image_h: int,
                            threshold_frac: float = 0.05) -> str:
    """Coarse relative direction from a main center ``mc`` to a context center ``cc``.

    Returns one of: 'overlapping', 'left', 'right', 'above', 'below', 'upper-left',
    'upper-right', 'lower-left', 'lower-right'. ``threshold_frac`` controls when an axis
    difference is too small to mention. Used to build the textual nearby-object hints
    that make same-category captions discriminative.
    """
    dx = (cc[0] - mc[0]) / image_w
    dy = (cc[1] - mc[1]) / image_h
    has_x = abs(dx) > threshold_frac
    has_y = abs(dy) > threshold_frac
    if not has_x and not has_y:
        return "overlapping"
    horiz = "right" if dx > 0 else "left"
    vert = "below" if dy > 0 else "above"
    if has_x and has_y:
        return f"{'upper' if vert == 'above' else 'lower'}-{horiz}"
    return horiz if has_x else vert
