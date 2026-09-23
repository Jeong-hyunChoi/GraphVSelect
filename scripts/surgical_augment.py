#!/usr/bin/env python3
"""Surgical object augmentation (A): add back missed referents.

Keep the base candidate geometry untouched. Only on records where the label-free
trigger fires -- base-side max relevance < 0.5 AND expanded-pool max relevance
> 0.7 -- add the top-2 relevance candidates from the expanded (noun@0.10) pool.
The trigger fires on a small fraction of records, so this is cheap while recovering
referents the base detection threshold dropped.

New objects get context-aware captions (tight crop + BFAIR_EXPLICIT prompt
with nearest-3 textual hints), generated here in one small VLM batch.

  python scripts/surgical_augment.py --in-dir data/sg_dump \
      --out-dir data/sg_dump_aug --work-dir outputs/aug_work
"""
import argparse, json, sys
from pathlib import Path

from graphvselect.images import ImageResolver
from graphvselect.vlm import VLM
from graphvselect.prompts import CROP_CAPTION_PROMPT_BFAIR_EXPLICIT
from graphvselect.spatial import _direction_from_centers

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1); I = iw * ih
    U = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I
    return I / U if U > 0 else 0

def center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/sg_dump")
    ap.add_argument("--out-dir", default="data/sg_dump_aug")
    ap.add_argument("--work-dir", default="outputs/aug_work")
    ap.add_argument("--cands", default="outputs/cands.jsonl")
    ap.add_argument("--rel", nargs="+", default=["outputs/rel.jsonl"])
    ap.add_argument("--base-thr", type=float, default=0.5)
    ap.add_argument("--new-thr", type=float, default=0.7)
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--image-root", default=None,
                    help="directory of prepared dataset images; a record's "
                         "image_path/file_name is resolved under it (see graphvselect/images.py)")
    args = ap.parse_args()

    files = sorted(Path(args.in_dir).glob("*_w*.jsonl"))
    recs = [json.loads(l) for f in files for l in f.open()]
    fc = {json.loads(l)["rid"]: json.loads(l) for l in open(args.cands)}
    rc = {}
    for f in args.rel:
        for line in open(f):
            d = json.loads(line); rc[d["rid"]] = d.get("rc", {})

    # ---- trigger + pick adds ----
    adds = {}  # rid -> [cand,...]
    for rid, r in enumerate(recs):
        f = fc.get(rid); sc = rc.get(rid, {})
        if not f or not sc: continue
        vb = [o["bbox"] for o in r["sg"]["objects"].values()]
        base_rel, newrel = [], []
        for j, c in enumerate(f["cands"]):
            s = sc.get(str(j), 0.0)
            (base_rel if any(iou(c["bbox"], b) > 0.9 for b in vb) else newrel).append((s, c))
        if max([s for s, _ in base_rel], default=0.0) < args.base_thr and \
           max([s for s, _ in newrel], default=0.0) > args.new_thr:
            adds[rid] = [c for _, c in sorted(newrel, reverse=True, key=lambda x: x[0])[:2]]
    n_add = sum(len(v) for v in adds.values())
    print(f"trigger fired on {len(adds)} records, adding {n_add} candidates", flush=True)

    # ---- caption the new crops (resumable) ----
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    capf = work / "caps.jsonl"
    caps = {}
    if capf.exists():
        for line in capf.open():
            d = json.loads(line); caps[(d["rid"], d["j"])] = d["caption"]
    pend = [(rid, j, c) for rid, cs in adds.items() for j, c in enumerate(cs) if (rid, j) not in caps]
    print(f"captions pending: {len(pend)}", flush=True)
    if pend:
        from PIL import Image
        img_of = ImageResolver(args.image_root).get
        vlm = VLM(hf_id=args.vlm_hf, dtype="float16", tensor_parallel_size=args.tp,
                           gpu_memory_utilization=0.85, max_num_seqs=64, max_model_len=4096)
        sp = vlm._SamplingParams(max_tokens=80, temperature=0.0, top_p=1.0)
        reqs, meta = [], []
        for rid, j, c in pend:
            r = recs[rid]; img = img_of(r)
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            crop = img.crop((max(0,x1), max(0,y1), min(img.width,x2), min(img.height,y2)))
            # nearest-3 textual hints from existing base objects
            cc = center(c["bbox"])
            ranked = sorted(((o, ((center(o["bbox"])[0]-cc[0])**2 + (center(o["bbox"])[1]-cc[1])**2) ** 0.5)
                             for o in recs[rid]["sg"]["objects"].values()), key=lambda x: x[1])
            hints = ", ".join(f"{o.get('category','object')} ({_direction_from_centers(cc, center(o['bbox']), img.width, img.height)})"
                              for o, _ in ranked[:3]) or "none"
            reqs.append({"prompt": vlm._build_prompt(CROP_CAPTION_PROMPT_BFAIR_EXPLICIT.format(
                            category=c["label"], nearby_hints=hints)),
                         "multi_modal_data": {"image": crop}})
            meta.append((rid, j))
        outs = vlm.llm.generate(reqs, sp)
        with capf.open("a") as fo:
            for (rid, j), o in zip(meta, outs):
                cap = o.outputs[0].text.strip()
                fo.write(json.dumps({"rid": rid, "j": j, "caption": cap}, ensure_ascii=False) + "\n")
                caps[(rid, j)] = cap
        print("captioning done", flush=True)

    # ---- assemble augmented dump ----
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    rid = 0; added = 0
    for f in files:
        with (outd / f.name).open("w") as fw:
            for line in f.open():
                r = json.loads(line)
                for j, c in enumerate(adds.get(rid, [])):
                    objs = r["sg"]["objects"]
                    maxid = max([o.get("id", -1) if isinstance(o.get("id"), int) else int(str(o.get("id", -1)))
                                 for o in objs.values()] + [len(objs) - 1])
                    nid = maxid + 1
                    cat = (c["label"] or "object").replace(" ", "_")
                    objs[f"{cat}_{nid}"] = {"bbox": c["bbox"], "category": c["label"], "kind": "main",
                                             "caption": caps.get((rid, j), ""), "id": nid,
                                             "verify_conf": c.get("verify_conf")}
                    added += 1
                fw.write(json.dumps(r, ensure_ascii=False) + "\n")
                rid += 1
    print(f"assembled {args.out_dir}: {rid} recs, {added} objects added", flush=True)

if __name__ == "__main__":
    main()
