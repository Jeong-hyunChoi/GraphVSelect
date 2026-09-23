#!/usr/bin/env python3
"""Prepare the official Ref-L4 split for the pipeline (parquet + image folder -> samples.jsonl).

Ref-L4 (Chen et al., *Revisiting REC Evaluation in the Era of Large Multimodal Models*) is
released as a parquet annotation file plus an image folder (COCO *train2014* + Objects365).
This converts one split into a run-ready ``samples.jsonl`` in the pipeline's record schema
(``query``, ``gt_bbox`` as ``[x1,y1,x2,y2]``, ``image_path``).

The parquet boxes are ``[x,y,w,h]`` in each image's native resolution. A few hundred
Objects365 images are very large; because vLLM 0.11 does not honor per-request pixel caps for
some backbones, those images would overflow the model context. This script resizes any image
whose area exceeds ``--max-pixels`` (default 1024x1024) and scales its boxes by the same
factor — exactly the paper's protocol — writing the resized copies under ``--resized-dir``.
Every record embeds an absolute ``image_path``, so the pipeline needs only ``SAMPLES``:

  python scripts/prepare_refl4.py \
      --parquet /path/to/Ref-L4/ref-l4-test.parquet \
      --images  /path/to/Ref-L4 \
      --out     data/refl4_test.jsonl

  SAMPLES=data/refl4_test.jsonl bash run_pipeline.sh        # build SG + select over the full split
  SAMPLES=data/refl4_test.jsonl bash run_selection.sh       # (after the graph is built)
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path

import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="ref-l4-test.parquet or ref-l4-val.parquet")
    ap.add_argument("--images", required=True,
                    help="folder holding the Ref-L4 images (COCO train2014 + Objects365)")
    ap.add_argument("--out", default="data/refl4_test.jsonl", help="output samples.jsonl")
    ap.add_argument("--resized-dir", default="data/refl4_resized",
                    help="where oversized images are rewritten as area-capped copies")
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024,
                    help="area cap; images above it are resized and their boxes scaled")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    rd = Path(args.resized_dir); rd.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    n = n_resized = 0
    seen = set()  # images already resized this run (many records share one image)
    with out.open("w") as fo:
        for _, r in df.iterrows():
            fn = str(r["file_name"])
            src = os.path.join(args.images, fn)
            w, h = int(r["width"]), int(r["height"])
            x, y, bw, bh = (float(v) for v in r["bbox"])
            if w * h > args.max_pixels:
                s = math.sqrt(args.max_pixels / (w * h))
                x, y, bw, bh = x * s, y * s, bw * s, bh * s
                dst = rd / fn
                if fn not in seen:
                    im = Image.open(src).convert("RGB")
                    im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS).save(dst)
                    seen.add(fn); n_resized += 1
                img_path = os.path.abspath(dst)
            else:
                img_path = os.path.abspath(src)
            fo.write(json.dumps({
                "dataset": "ref-l4", "split": str(r["split"]),
                "query": str(r["caption"]),
                "gt_bbox": [round(x, 4), round(y, 4), round(x + bw, 4), round(y + bh, 4)],
                "image_path": img_path,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} records -> {out}  ({n_resized} oversized images resized -> {rd})")


if __name__ == "__main__":
    main()
