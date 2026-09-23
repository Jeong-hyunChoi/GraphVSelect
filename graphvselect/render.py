"""Drawing helpers: annotate an image with candidate boxes and id labels.

These give the frozen VLM a visual cue for which region corresponds to which id, so
its answers and captions refer to the intended box.
"""
from __future__ import annotations

from PIL import ImageDraw, ImageFont

# Distinct outline colours, cycled by candidate index.
BBOX_COLORS = [
    "red", "blue", "lime", "orange", "magenta", "cyan", "yellow", "purple",
    "lightgreen", "deepskyblue", "gold", "deeppink", "chartreuse", "tomato",
    "navy", "darkorange",
]

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_font(size=18):
    try:
        return ImageFont.truetype(_FONT_PATH, size=size)
    except OSError:
        return ImageFont.load_default()


def annotate_image(image, candidates):
    """Draw each candidate's bbox and its ``[i]`` id label on a copy of the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(18)
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = c["bbox"]
        color = BBOX_COLORS[i % len(BBOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"[{i}]"
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((x1, y1), label, fill="white", font=font)
    return annotated


def annotate_image_pair(image, c1, c2):
    """Draw exactly two boxes — ``c1`` in red, ``c2`` in blue — for a pairwise relation
    query, so the model can refer to each object unambiguously in one prompt."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(18)
    for c, color, label in [(c1, "red", f"[{c1['id']}]"),
                            (c2, "blue", f"[{c2['id']}]")]:
        x1, y1, x2, y2 = c["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((x1, y1), label, fill="white", font=font)
    return annotated


def annotate_ids(image, cands):
    """Mark each candidate's box, labelled by its actual candidate id (matching the
    scene-graph object ids)."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = _load_font(18)
    for i, c in enumerate(cands):
        x1, y1, x2, y2 = c["bbox"]
        color = BBOX_COLORS[i % len(BBOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"[{c['id']}]"
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((x1, y1), label, fill="white", font=font)
    return img
