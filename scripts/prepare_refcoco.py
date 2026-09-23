#!/usr/bin/env python3
"""Prepare RefCOCO / RefCOCO+ / RefCOCOg for the pipeline (lmms-lab parquet -> samples.jsonl).

We follow the widely used lmms-eval protocol: RefCOCO/+/g are taken from the public lmms-lab
HuggingFace mirror, one record per referring expression (the first entry of each row's
``answer`` list), evaluated against the COCO *train2014* image it points to. This converts one
(dataset, split) into a run-ready ``samples.jsonl`` in the pipeline's record schema
(``query``, ``gt_bbox`` as ``[x1,y1,x2,y2]``, ``file_name``).

The lmms-lab boxes are COCO ``[x,y,w,h]`` and its ``file_name`` carries a per-annotation suffix
(``..._580957_4.jpg``); this restores the plain COCO name (``..._580957.jpg``) so it resolves
against a standard COCO *train2014* directory. Point ``--images`` at that directory.

  # 1) download the split's parquet (public):
  huggingface-cli download lmms-lab/RefCOCO --repo-type dataset --local-dir /data/RefCOCO

  # 2) convert (one call per dataset/split you want):
  python scripts/prepare_refcoco.py --parquet-dir /data/RefCOCO/data --split val \
      --dataset lmms-lab/RefCOCO --images /data/coco2014/train2014 \
      --out data/refcoco_val.jsonl

  # 3) run:
  IMAGES=/data/coco2014/train2014 SAMPLES=data/refcoco_val.jsonl bash run_pipeline.sh

Splits: RefCOCO / RefCOCO+ have val/testA/testB; RefCOCOg has val/test. Concatenate several
``samples.jsonl`` files if you want to score more than one split at once.
"""
from __future__ import annotations
import argparse, glob, json, os, re
from pathlib import Path

import pandas as pd

_SUFFIX = re.compile(r"_\d+(\.jpg)$", re.IGNORECASE)


def coco_name(file_name: str) -> str:
    """lmms-lab '..._580957_4.jpg' -> plain COCO '..._580957.jpg'."""
    return _SUFFIX.sub(r"\1", file_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True,
                    help="folder with the split parquet shards (e.g. /data/RefCOCO/data)")
    ap.add_argument("--split", required=True, help="val | testA | testB | test")
    ap.add_argument("--dataset", default="lmms-lab/RefCOCO",
                    help="tag recorded on each output record (informational)")
    ap.add_argument("--images", default=None,
                    help="COCO train2014 image directory; if given, an absolute image_path is "
                         "embedded per record (otherwise records carry a bare file_name)")
    ap.add_argument("--out", default="data/refcoco_val.jsonl")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.parquet_dir, f"{args.split}-*.parquet")))
    if not shards:
        raise SystemExit(f"no '{args.split}-*.parquet' under {args.parquet_dir}")
    df = pd.concat([pd.read_parquet(s, columns=["question_id", "answer", "bbox", "file_name"])
                    for s in shards], ignore_index=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fo:
        for _, r in df.iterrows():
            ans = r["answer"]
            query = str(ans[0] if hasattr(ans, "__len__") and not isinstance(ans, str) else ans)
            x, y, w, h = (float(v) for v in r["bbox"])
            fn = coco_name(str(r["file_name"]))
            rec = {"dataset": args.dataset, "split": args.split, "query": query,
                   "gt_bbox": [round(x, 4), round(y, 4), round(x + w, 4), round(y + h, 4)],
                   "file_name": fn, "question_id": str(r["question_id"])}
            if args.images:
                rec["image_path"] = os.path.abspath(os.path.join(args.images, fn))
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} records -> {out}  ({args.dataset} {args.split})")


if __name__ == "__main__":
    main()
