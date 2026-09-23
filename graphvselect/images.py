"""Image resolution for pipeline records (local images only).

A record's image is resolved from local files, in this order:
  1. ``image_path`` : an absolute path, or a path relative to ``image_root``.
  2. ``file_name``  : a bare image filename (e.g. COCO_train2014_000000391895.jpg),
                      resolved under ``image_root``. RefCOCO / RefCOCO+ / RefCOCOg use
                      COCO *train2014* images; Ref-L4 uses COCO + Objects365. Point
                      ``image_root`` at the directory that holds the images.

Records are self-contained (they carry ``file_name`` / ``image_path``); there is no
network or dataset-hub dependency. Pass ``--image-root`` on any script to point at the
prepared image directory.
"""
from __future__ import annotations

import os

from PIL import Image


class ImageResolver:
    def __init__(self, image_root=None):
        self.image_root = image_root

    @staticmethod
    def _rgb(im):
        return im.convert("RGB") if im.mode != "RGB" else im

    def get(self, rec):
        path = rec.get("image_path")
        if path:
            if self.image_root and not os.path.isabs(path):
                path = os.path.join(self.image_root, path)
            return self._rgb(Image.open(path))
        fn = rec.get("file_name")
        if fn:
            if not self.image_root:
                raise ValueError(
                    "pass --image-root to resolve records by file_name "
                    f"(e.g. the COCO train2014 image directory); got file_name={fn!r}")
            return self._rgb(Image.open(os.path.join(self.image_root, fn)))
        raise ValueError(f"record has no image_path or file_name (keys={list(rec.keys())})")
