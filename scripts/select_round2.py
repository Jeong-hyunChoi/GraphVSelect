"""Confidence-Aware Structured Selection — round 2 (order-symmetrized re-answer).

Reuses the cached round-1 log-probabilities (r1.jsonl from select_round1.py). Each gated
record is re-asked between its top candidates in BOTH presentation orders (a, b) and
(b, a); the per-candidate log-probabilities are summed across orders before the argmax,
which cancels the answer-slot / position bias. The final prediction is the round-2 choice
on gated records and the round-1 argmax elsewhere; --pred-out writes per-record boxes.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from graphvselect.vlm import VLM
from graphvselect.images import ImageResolver
from graphvselect.render import annotate_ids
from graphvselect.serialize import sg_blocks, cid_of


P_R1 = """\
In this image, candidate regions are marked with coloured boxes labelled [0], [1], [2], ... .

Scene description (produced by an automatic analyzer; use as context, it may contain minor errors):
Objects:
{objects_block}
Relations:
{relations_block}
Scene layout: {global_caption}

Referring expression: "{query}"

Which marked region is the object referred to by the expression? \
Respond with ONLY the region number.
"""

P_R2 = """\
In this image, exactly TWO candidate regions are marked with coloured boxes \
labelled [{a}] and [{b}].

Scene description (produced by an automatic analyzer; use as context, it may contain minor errors):
Objects:
{objects_block}
Relations:
{relations_block}
Scene layout: {global_caption}

Referring expression: "{query}"

Between region [{a}] and region [{b}], which one is the object referred to by \
the expression? Respond with ONLY the number: {a} or {b}.
"""


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0


SPLITS = [("refclef-benchmark", "refclef_berkeley_test"), ("ref-l4", "test"),
          ("RefCOCO", "val"), ("RefCOCO", "testA"), ("RefCOCO", "testB"),
          ("RefCOCOplus", "val"), ("RefCOCOplus", "testA"), ("RefCOCOplus", "testB"),
          ("RefCOCOg", "val"), ("RefCOCOg", "test")]


def id_logprobs(first, tokenizer, valid_ids):
    """Map first-token top-logprobs -> {cid: logprob} over valid candidate ids."""
    out = {}
    top = first.logprobs[0] if first.logprobs else {}
    for tid, lp in top.items():
        t = tokenizer.decode([tid]).strip()
        if t.isascii() and t.isdigit() and int(t) in valid_ids:
            c = int(t)
            if c not in out or lp.logprob > out[c]:
                out[c] = lp.logprob
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/sg_dump")
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--gate-frac", type=float, default=0.30)   # a-priori, no tuning
    ap.add_argument("--calib-n", type=int, default=0,
                    help="split-conformal: draw this many random records (stratified "
                         "by dataset/split), freeze the gate-frac margin quantile tau "
                         "on them, gate every record by margin<tau. 0 = in-sample.")
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--pred-out", default="",
                    help="if set, write per-record final A+B3 predictions "
                         "{rid,dataset,split,pred_bbox,gt_bbox} for post-hoc metrics")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--no-sg-r2", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--r1-cache", default="outputs/select_work/r1.jsonl")
    ap.add_argument("--work-dir", default="outputs/select_work")
    ap.add_argument("--out", default="outputs/select_round2.txt")
    ap.add_argument("--image-root", default=None,
                    help="directory of prepared dataset images; a record's "
                         "image_path/file_name is resolved under it (see graphvselect/images.py)")
    args = ap.parse_args()

    recs = []
    for f in sorted(Path(args.in_dir).glob("*_w*.jsonl")):
        for line in f.open():
            try: recs.append(json.loads(line))
            except json.JSONDecodeError: pass

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    r1f, r2f = Path(args.r1_cache), work / "r2.jsonl"

    vlm = VLM(hf_id=args.vlm_hf, dtype="float16", tensor_parallel_size=args.tp,
                       gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=1,
                       max_model_len=args.max_model_len)
    if "molmo" in args.vlm_hf.lower():
        vlm._build_prompt = lambda t: t   # Molmo: no chat template; vLLM re-wraps plain text
    sp = vlm._SamplingParams(max_tokens=4, temperature=0.0, top_p=1.0, logprobs=20)
    tok = getattr(vlm.processor, "tokenizer", vlm.processor)
    img_of = ImageResolver(args.image_root).get

    # ---------- Round 1 (A) ----------
    done1 = set()
    if r1f.exists():
        for line in r1f.open():
            try: done1.add(json.loads(line)["rid"])
            except (json.JSONDecodeError, KeyError): pass
    pend = [rid for rid in range(len(recs)) if rid not in done1]
    assert not pend, f"R1 cache incomplete: {len(pend)} missing"
    print("R1 cache complete", flush=True)
    fo = r1f.open("a")
    for i in range(0, len(pend), args.chunk):
        batch = pend[i:i + args.chunk]; reqs = []
        for rid in batch:
            r = recs[rid]; sg = r["sg"]
            cands = [{"id": cid_of(k), "bbox": o["bbox"]} for k, o in sg["objects"].items()]
            ob, rb, gc = sg_blocks(sg)
            reqs.append({"prompt": vlm._build_prompt(P_R1.format(
                            objects_block=ob, relations_block=rb, global_caption=gc,
                            query=r["query"])),
                         "multi_modal_data": {"image": annotate_ids(img_of(r), cands)}})
        outs = vlm.llm.generate(reqs, sp)
        for rid, o in zip(batch, outs):
            r = recs[rid]
            valid = {cid_of(k) for k in r["sg"]["objects"]}
            lps = id_logprobs(o.outputs[0], tok, valid)
            fo.write(json.dumps({"rid": rid, "lps": lps,
                                 "raw": o.outputs[0].text.strip()[:16]}) + "\n")
        fo.flush(); print(f"  R1 {min(i+args.chunk,len(pend))}/{len(pend)}", flush=True)
    fo.close()

    # collect R1
    R1 = {}
    for line in r1f.open():
        d = json.loads(line); R1[d["rid"]] = d
    picks1, margins = {}, {}
    for rid, r in enumerate(recs):
        d = R1.get(rid); objs = r["sg"]["objects"]
        valid = sorted(cid_of(k) for k in objs)
        lps = {int(k): v for k, v in (d.get("lps") or {}).items()} if d else {}
        if lps:
            order = sorted(lps, key=lps.get, reverse=True)
            picks1[rid] = order[0]
            margins[rid] = (lps[order[0]] - lps[order[1]]) if len(order) > 1 else 99.0
        else:  # no id token in top-20: fall back to raw text, ungated
            import re
            m = re.search(r"\d+", (d or {}).get("raw", ""))
            picks1[rid] = int(m.group(0)) if m and int(m.group(0)) in valid else (valid[0] if valid else -1)
            margins[rid] = 99.0

    # ---------- Round 2 (B): gate lowest gate-frac margins with >=2 cands ----------
    eligible = [rid for rid in picks1 if len(recs[rid]["sg"]["objects"]) >= 2 and margins[rid] < 99.0]
    if args.calib_n > 0:
        import random
        from collections import defaultdict
        by_split = defaultdict(list)
        for rid in eligible:
            r = recs[rid]; by_split[(r.get("dataset"), r.get("split"))].append(rid)
        rng = random.Random(args.calib_seed); calib = []; total = len(eligible)
        for key, rids in by_split.items():
            take = max(1, round(args.calib_n * len(rids) / total))
            calib += rng.sample(rids, min(take, len(rids)))
        cm = sorted(margins[rid] for rid in calib)
        tau = cm[min(len(cm) - 1, int(args.gate_frac * len(cm)))]
        gated = set(rid for rid in eligible if margins[rid] < tau)  # full-set: calib incl.
        print(f"R2 split-conformal: calib={len(calib)} (n~{args.calib_n}, seed {args.calib_seed}) "
              f"tau={tau:.4f} -> gated {len(gated)}/{total} ({100*len(gated)/max(1,total):.1f}%)", flush=True)
    else:
        eligible.sort(key=lambda rid: margins[rid])
        gated = set(eligible[:int(args.gate_frac * len(recs))])
        print(f"R2 gated {len(gated)} (frac {args.gate_frac})", flush=True)

    done2 = {}
    if r2f.exists():
        for line in r2f.open():
            try:
                d = json.loads(line); done2[d["rid"]] = d
            except (json.JSONDecodeError, KeyError): pass
    pend2 = [rid for rid in sorted(gated) if rid not in done2]
    fo = r2f.open("a")
    for i in range(0, len(pend2), args.chunk):
        batch = pend2[i:i + args.chunk]; reqs = []
        metas = []
        for rid in batch:
            r = recs[rid]; sg = r["sg"]
            lps = {int(k): v for k, v in R1[rid]["lps"].items()}
            order = sorted(lps, key=lps.get, reverse=True)
            top = order[:min(args.topk, len(order))]
            key_by_cid = {cid_of(k): k for k in sg["objects"]}
            cands = [{"id": c, "bbox": sg["objects"][key_by_cid[c]]["bbox"]} for c in top]
            ob, rb, gc = sg_blocks(sg)
            ids_txt = ", ".join(f"[{c}]" for c in top)
            ids_txt_rev = ", ".join(f"[{c}]" for c in reversed(top))
            if args.no_sg_r2:
                p = (f"In this image, {len(top)} candidate regions are marked with coloured boxes "
                     f"labelled {ids_txt}.\n\nReferring expression: \"{r['query']}\"\n\n"
                     f"Among {ids_txt}, which one is the object referred to by the expression? "
                     f"Respond with ONLY the number.")
            else:
                p = (f"In this image, {len(top)} candidate regions are marked with coloured boxes "
                     f"labelled {ids_txt}.\n\nScene description (produced by an automatic analyzer; "
                     f"use as context, it may contain minor errors):\nObjects:\n{ob}\nRelations:\n{rb}\n"
                     f"Scene layout: {gc}\n\nReferring expression: \"{r['query']}\"\n\n"
                     f"Among {ids_txt}, which one is the object referred to by the expression? "
                     f"Respond with ONLY the number.")
            ann = annotate_ids(img_of(r), cands)
            reqs.append({"prompt": vlm._build_prompt(p), "multi_modal_data": {"image": ann}})
            metas.append((rid, top[0], top[1] if len(top) > 1 else top[0], tuple(top), 0))
            p2 = p.replace(f"Among {ids_txt},", f"Among {ids_txt_rev},").replace(
                f"labelled {ids_txt}.", f"labelled {ids_txt_rev}.")
            reqs.append({"prompt": vlm._build_prompt(p2), "multi_modal_data": {"image": ann}})
            metas.append((rid, top[0], top[1] if len(top) > 1 else top[0], tuple(top), 1))
        outs = vlm.llm.generate(reqs, sp)
        acc2 = {}
        for (rid, a, b, top, order), o in zip(metas, outs):
            lps2 = id_logprobs(o.outputs[0], tok, set(top))
            slot = acc2.setdefault(rid, {"a": a, "b": b, "sum": {}})
            for c, lp in lps2.items():
                slot["sum"][c] = slot["sum"].get(c, 0.0) + lp
        for rid, slot in acc2.items():
            pick = max(slot["sum"], key=slot["sum"].get) if slot["sum"] else slot["a"]
            fo.write(json.dumps({"rid": rid, "a": slot["a"], "b": slot["b"], "pick": pick}) + "\n")
        fo.flush(); print(f"  R2 {min(i+args.chunk,len(pend2))}/{len(pend2)}", flush=True)
    fo.close()
    for line in r2f.open():
        d = json.loads(line); done2[d["rid"]] = d

    # ---------- eval ----------
    def acc(picks, subset=None):
        hit = {k: [] for k in SPLITS}
        for rid, r in enumerate(recs):
            key = (r["dataset"].split("/")[-1], r["split"])
            if key not in hit: continue
            pk = picks.get(rid)
            ko = next((o for k, o in r["sg"]["objects"].items() if cid_of(k) == pk), None)
            hit[key].append(ko is not None and iou(ko["bbox"], r["gt_bbox"]) >= 0.5)
        tot = sum(len(v) for v in hit.values()); th = sum(sum(v) for v in hit.values())
        return th, tot, hit
    picksAB = dict(picks1)
    for rid in gated:
        if rid in done2: picksAB[rid] = done2[rid]["pick"]
    if args.pred_out:
        with open(args.pred_out, "w") as pf:
            for rid, r in enumerate(recs):
                pk = picksAB.get(rid)
                ko = next((o for k, o in r["sg"]["objects"].items() if cid_of(k) == pk), None)
                pf.write(json.dumps({"rid": rid, "dataset": r.get("dataset"),
                                     "split": r.get("split"),
                                     "pred_bbox": ko["bbox"] if ko else None,
                                     "gt_bbox": r["gt_bbox"]}) + "\n")
        print(f"wrote per-record preds -> {args.pred_out}", flush=True)
    lines = [f"n={len(recs)}  A=constrained-logprob listwise, B=margin-gated top-2 reselect "
             f"(gate {args.gate_frac:.0%}, {len(gated)} records)", ""]
    for name, picks in [("A only", picks1), ("A + B", picksAB)]:
        th, tot, hit = acc(picks)
        lines.append(f"[{name:6s}] overall: {th}/{tot} ({100*th/max(1,tot):.1f}%)")
        lines.append("         " + "  ".join(f"{100*sum(hit[k])/max(1,len(hit[k])):.1f}" for k in SPLITS))
    # gated-subset detail
    g1 = sum(1 for rid in gated if (lambda pk: (lambda ko: ko is not None and iou(ko["bbox"], recs[rid]["gt_bbox"]) >= 0.5)(next((o for k, o in recs[rid]["sg"]["objects"].items() if cid_of(k) == pk), None)))(picks1.get(rid)))
    g2 = sum(1 for rid in gated if (lambda pk: (lambda ko: ko is not None and iou(ko["bbox"], recs[rid]["gt_bbox"]) >= 0.5)(next((o for k, o in recs[rid]["sg"]["objects"].items() if cid_of(k) == pk), None)))(picksAB.get(rid)))
    lines += ["", f"gated subset ({len(gated)}): R1 correct {g1} -> R1+R2 correct {g2}",
              "splits: " + "  ".join(f"{d.replace('RefCOCOplus','R+').replace('RefCOCO','R')}-{s}" for d, s in SPLITS),]
    text = "\n".join(lines)
    Path(args.out).write_text(text)
    print("\n" + text)
    print("SELECT_AB_DONE")


if __name__ == "__main__":
    main()
