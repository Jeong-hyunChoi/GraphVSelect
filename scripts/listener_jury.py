#!/usr/bin/env python3
"""Listener-JURY reranked captioning: candidates are scored by MULTIPLE listener
models and the winner maximizes the WORST-CASE (min) margin across jurors --
"a caption every listener can resolve". Tests the quality-x-readability
decomposition: a competent juror (Qwen) guarantees discriminative quality, a
weaker juror (LLaVA) forces cues weak consumers can actually read.

Phases (run separately, one model per GPU/container):
  --phase gen      (Qwen)  : sample K candidates per cluster member + Qwen margins
  --phase score    (LLaVA) : score the SAME candidates with the second juror
  --phase assemble (host)  : pick argmax_i min(m_qwen_i, m_llava_i); write dump
"""
import argparse, json, sys
from pathlib import Path
from graphvselect.render import annotate_ids
from graphvselect.images import ImageResolver


LISTENER = """\
In this image, {k} regions of the same category are marked with coloured boxes labelled [0] to [{km1}].

Description: "{cap}"

Which marked region does this description refer to? Answer with ONLY the region number.
"""
SPLIT = " Among the "

def center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def digit_lps(first, tok, k):
    for pos in range(min(3, len(first.logprobs or []))):
        lps = {}
        for tid, lp in (first.logprobs[pos] or {}).items():
            t = tok.decode([tid]).strip()
            if t.isdecimal() and t.isascii() and 0 <= int(t) < k:
                lps[int(t)] = max(lps.get(int(t), float("-inf")), lp.logprob)
        if lps: return lps
    return {}

def margin(lps, i):
    own = lps.get(i, -30.0)
    other = max([v for j, v in lps.items() if j != i], default=-20.0)
    return own - other

def load_recs(in_dir):
    files = sorted(Path(in_dir).glob("*_w*.jsonl"))
    return files, [json.loads(l) for f in files for l in f.open()]

def cluster_jobs(recs, kmax):
    head = lambda k: k.rsplit("_", 1)[0]
    jobs = []
    for rid, r in enumerate(recs):
        g = {}
        for k in r["sg"]["objects"]:
            g.setdefault(head(k), []).append(k)
        for h, ks in g.items():
            if 2 <= len(ks) <= kmax:
                ks = sorted(ks)
                for i in range(len(ks)):
                    jobs.append((rid, tuple(ks), i))
    return jobs

def make_img_of(image_root):
    return ImageResolver(image_root).get

def get_vlm(hf, tp, mml):
    from graphvselect.vlm import VLM
    vlm = VLM(hf_id=hf, dtype="float16", tensor_parallel_size=tp,
                       gpu_memory_utilization=0.85, max_num_seqs=64, max_model_len=mml)
    if "molmo" in hf.lower():
        vlm._build_prompt = lambda t: t
    return vlm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--phase", choices=["gen", "score", "assemble"], required=True)
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--nsamp", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gate-margin", type=float, default=4.0,
                    help="margin-gate: members whose ORIGINAL caption already scores "
                         "a listener margin >= this skip alternative gen+scoring "
                         "(nothing to repair). 0 disables the gate (score everything).")
    ap.add_argument("--image-root", default=None,
                    help="directory of prepared dataset images; a record's "
                         "image_path/file_name is resolved under it (see graphvselect/images.py)")
    args = ap.parse_args()

    files, recs = load_recs(args.in_dir)
    jobs = cluster_jobs(recs, args.kmax)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    genf, scoref = work / "jury_gen.jsonl", work / "jury_l2.jsonl"

    if args.phase == "gen":
        from graphvselect.prompts import CROP_CAPTION_PROMPT_BFAIR_EXPLICIT
        from graphvselect.spatial import _direction_from_centers
        done = set()
        if genf.exists():
            for line in genf.open():
                try: d = json.loads(line); done.add((d["rid"], d["key"]))
                except (json.JSONDecodeError, KeyError): pass
        pend = [j for j in jobs if (j[0], j[1][j[2]]) not in done]
        print(f"gen: {len(jobs)} members, pending {len(pend)}", flush=True)
        if not pend: return
        img_of = make_img_of(args.image_root)
        vlm = get_vlm(args.vlm_hf, args.tp, args.max_model_len)
        tok = getattr(vlm.processor, "tokenizer", vlm.processor)
        sp_gen = vlm._SamplingParams(max_tokens=80, temperature=0.85, top_p=0.95, n=args.nsamp)
        sp_lis = vlm._SamplingParams(max_tokens=3, temperature=0.0, top_p=1.0, logprobs=20)
        fo = genf.open("a")
        n_gated = 0
        for b in range(0, len(pend), args.chunk):
            batch = pend[b:b+args.chunk]
            # ---- Pass 1: listener-score the ORIGINAL caption for every member ----
            info = {}; lreqs0 = []; lmeta0 = []
            for rid, ks, i in batch:
                r = recs[rid]; k = ks[i]; o = r["sg"]["objects"][k]
                orig = (o.get("caption") or "").strip()
                base, tail = (orig.split(SPLIT, 1) + [""])[:2]
                base = base.strip(); tail = (SPLIT + tail) if tail else ""
                cimg = annotate_ids(img_of(r), [{"id": j, "bbox": r["sg"]["objects"][kk]["bbox"]}
                                                for j, kk in enumerate(ks)])
                info[(rid, k)] = {"o": o, "base": base, "tail": tail, "ks": ks, "i": i, "cimg": cimg}
                if base:  # only score non-empty originals in pass 1
                    lreqs0.append({"prompt": vlm._build_prompt(LISTENER.format(
                                      k=len(ks), km1=len(ks)-1, cap=(base+tail).strip())),
                                   "multi_modal_data": {"image": cimg}})
                    lmeta0.append((rid, k))
            louts0 = vlm.llm.generate(lreqs0, sp_lis) if lreqs0 else []
            m1 = {}
            for (rid, k), lo in zip(lmeta0, louts0):
                inf = info[(rid, k)]
                m1.setdefault((rid, k), {})[0] = round(
                    margin(digit_lps(lo.outputs[0], tok, len(inf["ks"])), inf["i"]), 3)
            # ---- Gate: skip alternatives if original is already discriminative ----
            # (empty-base members have no m1[0] -> always un-gated -> forced repair)
            ungated = [(rid, ks, i) for (rid, ks, i) in batch
                       if not (info[(rid, ks[i])]["base"]
                               and args.gate_margin > 0
                               and m1.get((rid, ks[i]), {}).get(0, -99) >= args.gate_margin)]
            n_gated += len(batch) - len(ungated)
            # default candidate set = original only (used by gated members)
            cand_by = {(rid, k): ([inf["base"]] if inf["base"] else [], inf["tail"], inf["ks"], inf["i"])
                       for (rid, k), inf in info.items()}
            # ---- Pass 2: generate + score alternatives for un-gated members only ----
            if ungated:
                greqs = []
                for rid, ks, i in ungated:
                    r = recs[rid]; k = ks[i]; o = info[(rid, k)]["o"]; img = img_of(r)
                    x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
                    crop = img.crop((max(0,x1), max(0,y1), min(img.width,x2), min(img.height,y2)))
                    cc = center(o["bbox"])
                    ranked = sorted(((o2, ((center(o2["bbox"])[0]-cc[0])**2+(center(o2["bbox"])[1]-cc[1])**2)**.5)
                                     for k2, o2 in r["sg"]["objects"].items() if k2 != k), key=lambda x: x[1])
                    hints = ", ".join(f"{o2.get('category','object')} ({_direction_from_centers(cc, center(o2['bbox']), img.width, img.height)})"
                                      for o2, _ in ranked[:3]) or "none"
                    greqs.append({"prompt": vlm._build_prompt(CROP_CAPTION_PROMPT_BFAIR_EXPLICIT.format(
                                      category=o.get("category","object"), nearby_hints=hints)),
                                  "multi_modal_data": {"image": crop}})
                gouts = vlm.llm.generate(greqs, sp_gen)
                lreqs, lmeta = [], []
                for (rid, ks, i), go in zip(ungated, gouts):
                    k = ks[i]; inf = info[(rid, k)]
                    cands = [inf["base"]] + [oo.text.strip().rstrip(".") for oo in go.outputs]
                    cands = [c for c in dict.fromkeys(cands) if c]
                    cand_by[(rid, k)] = (cands, inf["tail"], ks, i)
                    skip_first = bool(inf["base"])  # cands[0]==base already scored in pass 1
                    for ci, c in enumerate(cands):
                        if ci == 0 and skip_first:
                            continue  # original already scored in pass 1
                        lreqs.append({"prompt": vlm._build_prompt(LISTENER.format(
                                          k=len(ks), km1=len(ks)-1, cap=(c+inf["tail"]).strip())),
                                      "multi_modal_data": {"image": inf["cimg"]}})
                        lmeta.append((rid, k, ci))
                louts = vlm.llm.generate(lreqs, sp_lis)
                for (rid, k, ci), lo in zip(lmeta, louts):
                    inf = info[(rid, k)]
                    m1.setdefault((rid, k), {})[ci] = round(
                        margin(digit_lps(lo.outputs[0], tok, len(inf["ks"])), inf["i"]), 3)
            # ---- Persist per member (same schema; gated -> single-candidate) ----
            for (rid, k), (cands, tail, ks, i) in cand_by.items():
                fo.write(json.dumps({"rid": rid, "key": k, "ks": list(ks), "i": i, "tail": tail,
                                      "cands": cands, "m1": m1.get((rid, k), {})}, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"  gen {min(b+args.chunk,len(pend))}/{len(pend)} (gated {n_gated})", flush=True)
        return

    entries = {}
    for line in genf.open():
        d = json.loads(line); entries[(d["rid"], d["key"])] = d

    if args.phase == "score":
        done = set()
        if scoref.exists():
            for line in scoref.open():
                try: d = json.loads(line); done.add((d["rid"], d["key"]))
                except (json.JSONDecodeError, KeyError): pass
        pend = [e for k2, e in entries.items() if k2 not in done]
        print(f"score: {len(entries)} members, pending {len(pend)}", flush=True)
        if not pend: return
        img_of = make_img_of(args.image_root)
        vlm = get_vlm(args.vlm_hf, args.tp, args.max_model_len)
        tok = getattr(vlm.processor, "tokenizer", vlm.processor)
        sp_lis = vlm._SamplingParams(max_tokens=3, temperature=0.0, top_p=1.0, logprobs=20)
        fo = scoref.open("a")
        for b in range(0, len(pend), args.chunk):
            batch = pend[b:b+args.chunk]; lreqs, lmeta = [], []
            for e in batch:
                r = recs[e["rid"]]; ks = e["ks"]
                cimg = annotate_ids(img_of(r), [{"id": j, "bbox": r["sg"]["objects"][kk]["bbox"]} for j, kk in enumerate(ks)])
                for ci, c in enumerate(e["cands"]):
                    lreqs.append({"prompt": vlm._build_prompt(LISTENER.format(k=len(ks), km1=len(ks)-1, cap=(c+e["tail"]).strip())),
                                  "multi_modal_data": {"image": cimg}})
                    lmeta.append((e["rid"], e["key"], ci, len(ks), e["i"]))
            louts = vlm.llm.generate(lreqs, sp_lis)
            m2 = {}
            for (rid, key, ci, kk, i), lo in zip(lmeta, louts):
                m2.setdefault((rid, key), {})[ci] = round(margin(digit_lps(lo.outputs[0], tok, kk), i), 3)
            for e in batch:
                fo.write(json.dumps({"rid": e["rid"], "key": e["key"], "m2": m2.get((e["rid"], e["key"]), {})}) + "\n")
            fo.flush(); print(f"  score {min(b+args.chunk,len(pend))}/{len(pend)}", flush=True)
        return

    # assemble: winner = argmax_ci min(m1, m2)
    m2s = {}
    for line in scoref.open():
        d = json.loads(line); m2s[(d["rid"], d["key"])] = d["m2"]
    chosen = {}
    ch = 0
    for (rid, key), e in entries.items():
        m1 = e.get("m1", {}); m2 = m2s.get((rid, key), {})
        best, bv = 0, None
        for ci in range(len(e["cands"])):
            v = min(m1.get(str(ci), m1.get(ci, -99)), m2.get(str(ci), m2.get(ci, -99)))
            if bv is None or v > bv: bv, best = v, ci
        if best != 0: ch += 1
        chosen[(rid, key)] = (e["cands"][best] + e["tail"]).strip()
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    rid = 0
    for f in files:
        with (outd / f.name).open("w") as fw:
            for line in f.open():
                r = json.loads(line)
                for k in r["sg"]["objects"]:
                    if (rid, k) in chosen:
                        r["sg"]["objects"][k]["caption"] = chosen[(rid, k)]
                fw.write(json.dumps(r, ensure_ascii=False) + "\n"); rid += 1
    print(f"assembled {args.out_dir}: {rid} recs, {ch}/{len(chosen)} captions jury-replaced", flush=True)

if __name__ == "__main__":
    main()
