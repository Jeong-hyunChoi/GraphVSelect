"""Confidence-Aware Structured Selection — round 1 (list-wise constrained scoring)
plus the confidence gate.

Renders one image with all candidates marked and asks the VLM a single list-wise
question. The answer is scored by reading the first generated token's log-probabilities
restricted to the valid candidate ids, giving a directly comparable ranking and a
top-1/top-2 margin per record. The lowest-margin records (a calibrated fraction) are
re-asked as a top-2 comparison. Round-1 log-probabilities are cached to
<work-dir>/r1.jsonl for the order-symmetrized round-2 script (select_round2.py).
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


def id_logprobs(first, tokenizer, valid_ids, max_scan=4):
    """Map generated-token top-logprobs -> {cid: logprob} over valid candidate ids.

    Scans the first `max_scan` generated positions and uses the FIRST position
    where any valid-id digit token appears in the top-K. Handles tokenizers whose
    answer format starts with a non-digit token (e.g. Molmo emits "[0]" -> first
    token is "["; the id distribution lives at position 1)."""
    for pos in range(min(max_scan, len(first.logprobs or []))):
        out = {}
        for tid, lp in first.logprobs[pos].items():
            t = tokenizer.decode([tid]).strip()
            if t.isascii() and t.isdigit() and int(t) in valid_ids:
                c = int(t)
                if c not in out or lp.logprob > out[c]:
                    out[c] = lp.logprob
        if out:
            return out
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/sg_dump")
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--gate-frac", type=float, default=0.30)   # a-priori, no tuning
    ap.add_argument("--calib-n", type=int, default=0,
                    help="split-conformal: if >0, draw this many random records "
                         "(stratified by dataset/split) as a calibration set, freeze "
                         "the gate-frac margin quantile tau on them, and gate every "
                         "record independently by margin<tau. 0 = in-sample bottom-frac.")
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--compact-sg", action="store_true")
    ap.add_argument("--sg-parts", choices=["full", "objects", "objrel", "none"], default="full",
                    help="SG component ablation: none / objects-only / objects+relations / full")
    ap.add_argument("--work-dir", default="outputs/select_work")
    ap.add_argument("--out", default="outputs/select_round1.txt")
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
    r1f, r2f = work / "r1.jsonl", work / "r2.jsonl"

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
    print(f"R1 pending {len(pend)}", flush=True)
    fo = r1f.open("a")
    for i in range(0, len(pend), args.chunk):
        batch = pend[i:i + args.chunk]; reqs = []
        for rid in batch:
            r = recs[rid]; sg = r["sg"]
            cands = [{"id": cid_of(k), "bbox": o["bbox"]} for k, o in sg["objects"].items()]
            ob, rb, gc = sg_blocks(sg)
            if args.compact_sg:
                ob = "\n".join(l[:72] for l in ob.splitlines()); gc = "(none)"
            if args.sg_parts == "objects": rb = "  (none)"; gc = "(none)"
            elif args.sg_parts == "objrel": gc = "(none)"
            elif args.sg_parts == "none": ob = "  (none)"; rb = "  (none)"; gc = "(none)"
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
        # split-conformal: freeze tau on a random, split-stratified calibration set,
        # then gate each record independently by margin < tau (no in-sample ranking).
        import random
        from collections import defaultdict
        by_split = defaultdict(list)
        for rid in eligible:
            r = recs[rid]
            by_split[(r.get("dataset"), r.get("split"))].append(rid)
        rng = random.Random(args.calib_seed)
        calib = []
        total = len(eligible)
        for key, rids in by_split.items():
            take = max(1, round(args.calib_n * len(rids) / total))  # proportional strata
            calib += rng.sample(rids, min(take, len(rids)))
        calib_margins = sorted(margins[rid] for rid in calib)
        qi = min(len(calib_margins) - 1, int(args.gate_frac * len(calib_margins)))
        tau = calib_margins[qi]
        # threshold frozen on the 2000-sample; gate EVERY record by margin<tau
        # (calibration records included -> full-set reporting, benchmark-comparable).
        gated = set(rid for rid in eligible if margins[rid] < tau)
        print(f"R2 split-conformal: calib={len(calib)} (n~{args.calib_n}, seed {args.calib_seed}) "
              f"tau={tau:.4f} -> gated {len(gated)}/{total} eligible "
              f"({100*len(gated)/max(1,total):.1f}%)", flush=True)
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
            a, b = order[0], order[1]
            key_by_cid = {cid_of(k): k for k in sg["objects"]}
            cands = [{"id": a, "bbox": sg["objects"][key_by_cid[a]]["bbox"]},
                     {"id": b, "bbox": sg["objects"][key_by_cid[b]]["bbox"]}]
            ob, rb, gc = sg_blocks(sg)
            if args.compact_sg:
                ob = "\n".join(l[:72] for l in ob.splitlines()); gc = "(none)"
            reqs.append({"prompt": vlm._build_prompt(P_R2.format(
                            a=a, b=b, objects_block=ob, relations_block=rb,
                            global_caption=gc, query=r["query"])),
                         "multi_modal_data": {"image": annotate_ids(img_of(r), cands)}})
            metas.append((rid, a, b))
        outs = vlm.llm.generate(reqs, sp)
        for (rid, a, b), o in zip(metas, outs):
            lps2 = id_logprobs(o.outputs[0], tok, {a, b})
            pick = max(lps2, key=lps2.get) if lps2 else a
            fo.write(json.dumps({"rid": rid, "a": a, "b": b, "pick": pick}) + "\n")
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
