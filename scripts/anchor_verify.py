#!/usr/bin/env python3
"""Object-level verification (F, part 2): anchor-relevance scores for the VLM rescue.

For each LOW-confidence object (verify_conf < --thr), ask the VLM — on an EXPANDED
crop (object + surroundings) — whether it is PART OF what the expression describes
(an anchor the target is spatially related to / interacts with / that helps locate
it), NOT whether it IS the target. Score = sigmoid(yes_lp - no_lp). These let
``object_filter.py`` keep useful low-confidence context objects instead of dropping
them. Resumable per (rid, key); aggregates to --out as one line per rid:
{"rid", "scores": {key: sigmoid}}.

  python scripts/anchor_verify.py --in-dir data/base_sg \
      --out outputs/anchor.jsonl --work-dir outputs/anchor_work --image-root <COCO>
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

from graphvselect.images import ImageResolver
from graphvselect.vlm import VLM
from graphvselect.geometry import expand_bbox
from graphvselect.render import annotate_image

ANCHOR_PROMPT = (
    "Look at the object highlighted by the box. Given the referring expression "
    "\"{query}\", is this highlighted object PART OF what the expression describes — "
    "for example, an object that the target is spatially related to, interacting with, "
    "sitting on/holding, or that is explicitly mentioned to help locate the target? "
    "Answer with only 'Yes' or 'No'."
)


def yes_no_sigmoid(first, tokenizer):
    yes_lp = no_lp = float("-inf")
    top = first.logprobs[0] if first.logprobs else {}
    for tid, lp in top.items():
        t = tokenizer.decode([tid]).strip().lower()
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
    ap.add_argument("--in-dir", default="data/base_sg")
    ap.add_argument("--out", default="outputs/anchor.jsonl")
    ap.add_argument("--work-dir", default="outputs/anchor_work")
    ap.add_argument("--thr", type=float, default=0.3)   # objects below = LOW = candidates for rescue
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--pad", type=float, default=0.6)    # expanded-crop pad ratio (surroundings)
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--image-root", default=None,
                    help="directory of prepared dataset images; a record's "
                         "image_path/file_name is resolved under it (see graphvselect/images.py)")
    args = ap.parse_args()

    recs = []
    for f in sorted(Path(args.in_dir).glob("*_w*.jsonl")):
        for line in f.open():
            try: recs.append(json.loads(line))
            except json.JSONDecodeError: pass

    def low_objs(r):
        return [(k, o) for k, o in r["sg"]["objects"].items() if (o.get("verify_conf") or 0.0) < args.thr]

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    objf = work / "anchor_obj.jsonl"
    done = set()
    if objf.exists():
        for line in objf.open():
            try: d = json.loads(line); done.add((d["rid"], d["key"]))
            except (json.JSONDecodeError, KeyError): pass

    pending = [(rid, k) for rid, r in enumerate(recs) for k, _ in low_objs(r) if (rid, k) not in done]
    print(f"anchor pending: {len(pending)} objs (done {len(done)})", flush=True)

    if pending:
        vlm = VLM(hf_id=args.vlm_hf, dtype="float16", tensor_parallel_size=args.tp,
                  gpu_memory_utilization=0.85, max_num_seqs=1, max_pixels=args.max_pixels)
        sp = vlm._SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, logprobs=20)
        tok = vlm.processor.tokenizer
        img_of = ImageResolver(args.image_root).get
        fo = objf.open("a")
        for i in range(0, len(pending), args.chunk):
            batch = pending[i:i + args.chunk]; reqs = []
            for rid, k in batch:
                r = recs[rid]; o = r["sg"]["objects"][k]; im = img_of(r)
                ex = expand_bbox(o["bbox"], im.width, im.height, pad_ratio=args.pad)
                crop = im.crop(ex)
                # re-annotate the object box in crop-local coords
                lb = [o["bbox"][0] - ex[0], o["bbox"][1] - ex[1], o["bbox"][2] - ex[0], o["bbox"][3] - ex[1]]
                ann = annotate_image(crop, [{"id": 0, "bbox": lb, "label": o.get("category", "")}])
                reqs.append({"prompt": vlm._build_prompt(ANCHOR_PROMPT.format(query=r["query"])),
                             "multi_modal_data": {"image": ann}})
            outs = vlm.llm.generate(reqs, sp)
            for (rid, k), out in zip(batch, outs):
                fo.write(json.dumps({"rid": rid, "key": k, "score": yes_no_sigmoid(out.outputs[0], tok)}) + "\n")
            fo.flush(); print(f"  {min(i + args.chunk, len(pending))}/{len(pending)}", flush=True)
        fo.close()

    # aggregate -> out (one line per rid)
    done = set(); agg = {}
    for line in objf.open():
        try:
            d = json.loads(line); agg.setdefault(d["rid"], {})[d["key"]] = d["score"]; done.add((d["rid"], d["key"]))
        except (json.JSONDecodeError, KeyError): pass
    all_needed = [(rid, k) for rid, r in enumerate(recs) for k, _ in low_objs(r)]
    if all(x in done for x in all_needed):
        with Path(args.out).open("w") as f:
            for rid in sorted(agg): f.write(json.dumps({"rid": rid, "scores": agg[rid]}) + "\n")
        print(f"ANCHOR_ALL_DONE -> {args.out} ({len(agg)} recs)", flush=True)
    else:
        print(f"PARTIAL anchor ({len(done)}/{len(all_needed)}) — rerun to resume", flush=True)


if __name__ == "__main__":
    main()
