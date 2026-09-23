#!/usr/bin/env python3
"""Object-level verification (F, part 1): recompute per-object confidence with an
INDEPENDENT verifier VLM (cross-model check).

The base scene graph's object captions are written by the perception VLM, so a
confidence produced by that same model would be circular. This recomputes a
per-object confidence with a (possibly different) ~7-8B VLM over the SAME candidate
boxes: a Set-of-Mark pass (all regions marked; "Is region [N] the {query}?" Yes/No
first-token log-probabilities) whose P(yes)=sigmoid(yes_lp - no_lp) becomes each
object's ``verify_conf``. ``merge_verify_conf.py`` writes these back onto the dump;
``object_filter.py`` then filters/keeps objects by this confidence.

Output jsonl: {dataset, split, [ds_idx], [rid], verify_conf:{id: P(yes)}}.

  python scripts/gen_verify_conf.py --in-dir data/base_sg \
      --verifier-hf Qwen/Qwen3-VL-8B-Instruct --out outputs/vc.jsonl --image-root <COCO>
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from graphvselect.images import ImageResolver
from graphvselect.vlm import VLM


def cid_of(key):
    try:
        return int(key.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/base_sg")
    ap.add_argument("--verifier-hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm-tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--image-root", default=None,
                    help="directory of prepared dataset images; a record's "
                         "image_path/file_name is resolved under it (see graphvselect/images.py)")
    args = ap.parse_args()

    recs = []
    for f in sorted(Path(args.in_dir).glob("*_w*.jsonl")):
        for line in f.open():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"[gen] {len(recs)} samples from {args.in_dir}", flush=True)

    resolve = ImageResolver(args.image_root).get
    images = {}
    for i, r in enumerate(recs):
        im = resolve(r)
        images[i] = im.convert("RGB") if im.mode != "RGB" else im
    print(f"[gen] loaded images for {len(images)} samples", flush=True)

    vlm = VLM(hf_id=args.verifier_hf, tensor_parallel_size=args.vlm_tp,
              max_model_len=args.max_model_len)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_done = n_empty = 0
    with open(args.out, "w") as fout:
        for i, r in enumerate(recs):
            objs = r["sg"]["objects"]
            cands = [{"id": cid_of(k), "bbox": o["bbox"]} for k, o in objs.items()]
            try:
                scores = vlm.verify_regions_som(images[i], r["query"], cands)
            except Exception as e:  # noqa: BLE001 — keep going on any per-sample failure
                scores = []
                if n_done < 3:
                    print(f"[gen] sample {i} verify error: {e!r}", flush=True)
            conf = {str(s["id"]): round(1.0 / (1.0 + math.exp(-s["score"])), 4)
                    for s in scores}
            if not conf:
                n_empty += 1
            rec_out = {"dataset": r["dataset"], "split": r["split"], "verify_conf": conf}
            if "ds_idx" in r: rec_out["ds_idx"] = r["ds_idx"]
            if "rid" in r: rec_out["rid"] = r["rid"]
            fout.write(json.dumps(rec_out) + "\n")
            fout.flush()
            n_done += 1
            if n_done <= 2 or n_done % 200 == 0:
                print(f"[gen] {n_done}/{len(recs)}  sample conf={list(conf.items())[:3]}",
                      flush=True)
    print(f"[gen] DONE {n_done} written ({n_empty} empty) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
