#!/usr/bin/env python3
"""Build the surgical-augmentation ingredients (the --cands / --rel that
surgical_augment.py consumes) from a base scene-graph dump:
Phase 1 -- expanded GDINO detection (same query nouns as the base dump, same
NMS, lower box threshold 0.10, cap 20) -> cands file {rid, cands:[...]}.
Phase 2 -- relevance scoring of every expanded candidate (single-box marking,
"is the highlighted object related to this phrase?" yes/no logprob sigmoid)
-> rel file {rid, rc:{j: conf}}.

  python scripts/refl4_expand_rel.py --in-dir data/base_sg \
      --out-cands outputs/cands.jsonl \
      --out-rel outputs/rel.jsonl --work-dir outputs/exp_work --image-root <IMAGES>
"""
import argparse, json, math, sys
from pathlib import Path
from graphvselect.render import annotate_ids
from graphvselect.images import ImageResolver

REL_PROMPT = ("The phrase: \"{q}\". Is the highlighted object [{i}] related to this "
              "phrase? Answer only 'Yes' or 'No'.")

def yes_no_sig(first, tok):
    yes_lp = no_lp = float("-inf")
    top = first.logprobs[0] if first.logprobs else {}
    for tid, lp in top.items():
        t = tok.decode([tid]).strip().lower()
        if t in {"yes", "yes.", "yes,"}: yes_lp = max(yes_lp, lp.logprob)
        elif t in {"no", "no.", "no,"}: no_lp = max(no_lp, lp.logprob)
    txt = first.text.strip().lower()
    if yes_lp == float("-inf") and no_lp == float("-inf"):
        yes_lp, no_lp = (0.0, -10.0) if txt.startswith("yes") else ((-10.0, 0.0) if txt.startswith("no") else (0.0, 0.0))
    elif yes_lp == float("-inf"): yes_lp = no_lp - 10.0
    elif no_lp == float("-inf"): no_lp = yes_lp - 10.0
    return 1.0 / (1.0 + math.exp(-(yes_lp - no_lp)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-cands", required=True)
    ap.add_argument("--out-rel", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--box-thr", type=float, default=0.10)
    ap.add_argument("--text-thr", type=float, default=0.20)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--image-root", default=None,
                    help="directory of prepared dataset images; a record's "
                         "image_path/file_name is resolved under it (see graphvselect/images.py)")
    args = ap.parse_args()

    recs = [json.loads(l) for f in sorted(Path(args.in_dir).glob("*_w*.jsonl")) for l in f.open()]
    from PIL import Image
    
    img_of = ImageResolver(args.image_root).get

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: expanded detection ----
    candf = Path(args.out_cands)
    cands = {}
    if candf.exists():
        for line in candf.open():
            d = json.loads(line); cands[d["rid"]] = d
    pend = [rid for rid in range(len(recs)) if rid not in cands]
    print(f"expand-detect: {len(recs)} recs, pending {len(pend)}", flush=True)
    if pend:
        from graphvselect.detect import GroundingDINODetector, _format_vocab
        from graphvselect.nouns import QueryNounExtractor
        qne = QueryNounExtractor()
        det = GroundingDINODetector("IDEA-Research/grounding-dino-base",
                                    box_threshold=args.box_thr, text_threshold=args.text_thr,
                                    nms_iou=args.nms)
        with candf.open("a") as fo:
            for n, rid in enumerate(pend):
                r = recs[rid]
                nouns = qne.extract(r["query"]) or []
                ds = []
                if nouns:
                    for d in sorted(det(img_of(r), _format_vocab(nouns)), key=lambda x: -x.score)[:args.cap]:
                        ds.append({"bbox": [float(x) for x in d.bbox], "label": d.label, "score": float(d.score)})
                fo.write(json.dumps({"rid": rid, "cands": ds}, ensure_ascii=False) + "\n")
                cands[rid] = {"rid": rid, "cands": ds}
                if (n + 1) % 100 == 0: fo.flush(); print(f"  detect {n+1}/{len(pend)}", flush=True)
        del det
        import torch; torch.cuda.empty_cache()
    print("phase 1 done", flush=True)

    # ---- Phase 2: relevance scoring ----
    relf = Path(args.out_rel)
    rel = {}
    if relf.exists():
        for line in relf.open():
            d = json.loads(line); rel.setdefault(d["rid"], {}).update(d.get("rc", {}))
    jobs = [(rid, j) for rid in range(len(recs)) for j in range(len(cands.get(rid, {}).get("cands", [])))
            if str(j) not in rel.get(rid, {})]
    print(f"rel-score: pending {len(jobs)}", flush=True)
    if jobs:
        from graphvselect.vlm import VLM
        vlm = VLM(hf_id=args.vlm_hf, dtype="float16", tensor_parallel_size=args.tp,
                           gpu_memory_utilization=0.85, max_num_seqs=64, max_model_len=4096)
        sp = vlm._SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, logprobs=20)
        tok = getattr(vlm.processor, "tokenizer", vlm.processor)
        for i in range(0, len(jobs), args.chunk):
            batch = jobs[i:i+args.chunk]; reqs = []
            for rid, j in batch:
                r = recs[rid]; c = cands[rid]["cands"][j]
                img = annotate_ids(img_of(r), [{"id": 0, "bbox": c["bbox"]}])
                reqs.append({"prompt": vlm._build_prompt(REL_PROMPT.format(q=r["query"], i=0)),
                             "multi_modal_data": {"image": img}})
            outs = vlm.llm.generate(reqs, sp)
            buf = {}
            for (rid, j), o in zip(batch, outs):
                conf = round(yes_no_sig(o.outputs[0], tok), 4)
                buf.setdefault(rid, {})[str(j)] = conf
                rel.setdefault(rid, {})[str(j)] = conf
            with relf.open("a") as fo:
                for rid, rc in buf.items():
                    fo.write(json.dumps({"rid": rid, "rc": rc}) + "\n")
            print(f"  rel {min(i+args.chunk,len(jobs))}/{len(jobs)}", flush=True)
    # consolidate rel file (one line per rid)
    with relf.open("w") as fo:
        for rid, rc in sorted(rel.items()):
            fo.write(json.dumps({"rid": rid, "rc": rc}) + "\n")
    print(f"phase 2 done: rel for {len(rel)} rids", flush=True)

if __name__ == "__main__":
    main()
