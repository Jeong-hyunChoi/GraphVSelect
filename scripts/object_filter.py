#!/usr/bin/env python3
"""Object-level verification (F, part 4): filter/keep objects by confidence, then
assemble the object-verified dump. RESUMABLE.

Survivors per record are deterministic: main = ``verify_conf >= --thr`` (kind='main');
optionally rescue useful low-confidence context objects (kind='context') by geometric
adjacency or by the VLM anchor score from ``anchor_verify.py``. With ``--filter-only``
(the setting used by the frozen pipeline), NO VLM is loaded: relations + global caption
are left empty and regenerated later by ``sg_relate_global.py`` over the FINAL object set
(i.e. after surgical augmentation A). Without ``--filter-only`` this script also
regenerates relations/global over the survivors (streamed per chunk so a restart resumes).

Per-crop captions and verify_conf are carried over unchanged (a per-crop caption is
independent of which other objects survive).

  python scripts/object_filter.py --in-dir data/vc_dump --out-dir data/obj_sg \
      --work-dir outputs/f_work --thr 0.2 --rescue vlm --rescue-cap 3 \
      --anchor-thr 0.5 --anchor-file outputs/anchor.jsonl --filter-only
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from graphvselect.vlm import VLM
from graphvselect.images import ImageResolver
from graphvselect.render import annotate_image
from graphvselect.parsing import parse_relations_only
from graphvselect.prompts import RELATIONS_ONLY_PROMPT, GLOBAL_CAPTION_PROMPT_QUERYCOND

MAX_SIDE = 100000  # set from args; default = effectively no downscale


def downscale(img):
    w, h = img.size; s = min(1.0, MAX_SIDE / max(w, h))
    if s >= 1.0:
        return img, 1.0
    return img.resize((max(1, int(w * s)), max(1, int(h * s)))), s


def _cxcy_diag(b):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dg = ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
    return cx, cy, dg


def geom_rescue(main, low, cap, max_ndist=1.2):
    """Adjacency rescue: keep low-confidence objects whose nearest main is within
    max_ndist (normalised centre distance). Rank by proximity, take `cap`."""
    scored = []
    for k, o in low:
        cx, cy, dg = _cxcy_diag(o["bbox"]); best = 1e9
        for _, mo in main:
            mcx, mcy, mdg = _cxcy_diag(mo["bbox"])
            d = ((cx - mcx) ** 2 + (cy - mcy) ** 2) ** 0.5 / max(1.0, (dg + mdg) / 2)
            best = min(best, d)
        scored.append((best, k, o))
    scored.sort(key=lambda x: x[0])
    return [(k, o) for d, k, o in scored if d < max_ndist][:cap]


def geomw_rescue(main, low, cap, cutoff=1.5, tau=0.5):
    """Weighted-neighbourhood rescue. score = SUM over mains of k(dist)*verify_conf(main),
    k(d)=max(0, 1 - d/cutoff), dist normalised by the pair's avg object size. An object
    embedded among many close, strong neighbours scores high; one merely near a single
    distant/weak main scores low and is NOT rescued. Retain score>=tau, top `cap`."""
    scored = []
    for k, o in low:
        cx, cy, dg = _cxcy_diag(o["bbox"]); evid = 0.0
        for _, mo in main:
            mcx, mcy, mdg = _cxcy_diag(mo["bbox"])
            d = ((cx - mcx) ** 2 + (cy - mcy) ** 2) ** 0.5 / max(1.0, (dg + mdg) / 2)
            if d < cutoff:
                evid += (1.0 - d / cutoff) * (mo.get("verify_conf") or 0.0)
        if evid >= tau:
            scored.append((evid, k, o))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [(k, o) for s, k, o in scored][:cap]


def survivors_of(objs: dict, thr: float, rescue="none", cap=3, anchor=None, anchor_thr=0.5):
    """Return [(key, obj_with_kind)]. main = verify_conf>=thr (kind='main');
    optionally rescue anchors (kind='context'):
      rescue='geom' -> spatial adjacency to a main; rescue='vlm' -> anchor score
      (from `anchor` dict key->sigmoid) >= anchor_thr. Rescued objects are CONTEXT
      (SG/relation cues, not answer candidates)."""
    items = list(objs.items())
    if not items:                      # record with no detected objects
        return []
    main = [(k, dict(o, kind="main")) for k, o in items if (o.get("verify_conf") or 0.0) >= thr]
    low = [(k, o) for k, o in items if (o.get("verify_conf") or 0.0) < thr]
    if not main:
        bk, bo = max(items, key=lambda kv: kv[1].get("verify_conf") or 0.0)
        main = [(bk, dict(bo, kind="main"))]; low = [(k, o) for k, o in items if k != bk]
    if rescue == "geom":
        resc = geom_rescue(main, low, cap)
    elif rescue == "geomw":
        resc = geomw_rescue(main, low, cap)
    elif rescue == "vlm" and anchor is not None:
        ranked = sorted(((anchor.get(k, 0.0), k, o) for k, o in low), reverse=True, key=lambda x: x[0])
        resc = [(k, o) for s, k, o in ranked if s >= anchor_thr][:cap]
    else:
        resc = []
    resc = [(k, dict(o, kind="context")) for k, o in resc]
    return main + resc


def load_done(path):
    done = {}
    if path.exists():
        for line in path.open():
            try:
                d = json.loads(line); done[d["rid"]] = d
            except json.JSONDecodeError:
                pass
    return done


def assemble(recs, survs, rel_done, glob_done, out_dir, regen_global):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "obj_filter_w0.jsonl").open("w") as fo:
        for rid, r in enumerate(recs):
            keep = survs[rid]
            new = dict(r)
            gcap = (glob_done.get(rid, {}).get("glob", "") if regen_global
                    else r["sg"].get("global_caption", ""))   # keep original when not regenerating
            new["sg"] = {
                "global_caption": gcap,
                "query": r["sg"].get("query", r["query"]),
                "query_nouns": r["sg"].get("query_nouns", []),
                "objects": {k: o for k, o in keep},
                "relationships": rel_done.get(rid, {}).get("rels", []),
            }
            new["sg_objects_count"] = len(keep)
            new["sg_relations_count"] = len(new["sg"]["relationships"])
            fo.write(json.dumps(new, ensure_ascii=False) + "\n")
    print(f"ALL_DONE assembled {len(recs)} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/vc_dump")
    ap.add_argument("--out-dir", default="data/obj_sg")
    ap.add_argument("--work-dir", default="outputs/f_work")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--max-side", type=int, default=100000)   # no downscale by default
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--regen-global", action="store_true")    # off => keep original global
    ap.add_argument("--filter-only", action="store_true",
                    help="filter/rescue ONLY; emit empty relations + empty global. "
                         "Relations and global are regenerated later, over the FINAL "
                         "object set (i.e. after A). No VLM is loaded.")
    ap.add_argument("--rescue", choices=["none", "geom", "geomw", "vlm"], default="none")
    ap.add_argument("--rescue-cap", type=int, default=3)
    ap.add_argument("--anchor-thr", type=float, default=0.5)   # sigmoid threshold for vlm rescue
    ap.add_argument("--anchor-file", default="outputs/anchor.jsonl")
    ap.add_argument("--image-root", default=None,
                    help="directory of prepared dataset images; a record's "
                         "image_path/file_name is resolved under it (see graphvselect/images.py)")
    args = ap.parse_args()
    global MAX_SIDE; MAX_SIDE = args.max_side

    recs = []
    for f in sorted(Path(args.in_dir).glob("*_w*.jsonl")):
        for line in f.open():
            try: recs.append(json.loads(line))
            except json.JSONDecodeError: pass

    anchor = {}   # rid -> {obj_key: sigmoid score}
    if args.rescue == "vlm":
        for line in Path(args.anchor_file).open():
            try:
                d = json.loads(line); anchor[d["rid"]] = d["scores"]
            except (json.JSONDecodeError, KeyError): pass
    survs = [survivors_of(r["sg"]["objects"], args.thr, args.rescue, args.rescue_cap,
                          anchor.get(rid), args.anchor_thr) for rid, r in enumerate(recs)]

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    rel_path, glob_path = work / "rel_done.jsonl", work / "glob_done.jsonl"
    rel_done, glob_done = load_done(rel_path), load_done(glob_path)

    if args.filter_only:
        # Object set is not final yet (A still to come), so relations/global are
        # deliberately left empty here and produced by sg_relate_global.py later.
        assemble(recs, survs, {}, {}, args.out_dir, regen_global=True)
        return

    rel_needed = [rid for rid in range(len(recs)) if len(survs[rid]) >= 2]
    rel_pending = [rid for rid in rel_needed if rid not in rel_done]
    glob_pending = [rid for rid in range(len(recs)) if rid not in glob_done] if args.regen_global else []
    print(f"pending: {len(rel_pending)} rel, {len(glob_pending)} glob  regen_global={args.regen_global} "
          f"max_side={MAX_SIDE} max_pixels={args.max_pixels}", flush=True)

    if not rel_pending and not glob_pending:
        assemble(recs, survs, rel_done, glob_done, args.out_dir, args.regen_global); return

    vlm = VLM(hf_id=args.vlm_hf, dtype="float16", tensor_parallel_size=args.tp,
              gpu_memory_utilization=0.90 if args.tp == 1 else 0.85,
              max_num_seqs=1, max_pixels=args.max_pixels)
    sp_rel = vlm._SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0)
    sp_g = vlm._SamplingParams(max_tokens=128, temperature=0.0, top_p=1.0)

    resolve = ImageResolver(args.image_root).get
    def annotated_of(rid):
        r = recs[rid]
        im = resolve(r)
        if im.mode != "RGB": im = im.convert("RGB")
        img, s = downscale(im)
        cands = [{"id": i, "bbox": [c * s for c in o["bbox"]], "label": o.get("category", "")}
                 for i, (k, o) in enumerate(survs[rid])]
        return annotate_image(img, cands), cands

    def run(pending, kind, fpath, sp):
        fo = fpath.open("a")
        for i in range(0, len(pending), args.chunk):
            batch = pending[i:i + args.chunk]
            reqs = []
            for rid in batch:
                ann, cands = annotated_of(rid)
                if kind == "rel":
                    ids_list = ", ".join(f"[{j}]" for j in range(len(cands)))
                    p = RELATIONS_ONLY_PROMPT.format(query=recs[rid]["query"], ids_list=ids_list)
                else:
                    p = GLOBAL_CAPTION_PROMPT_QUERYCOND.format(query=recs[rid]["query"])
                reqs.append({"prompt": vlm._build_prompt(p), "multi_modal_data": {"image": ann}})
            outs = vlm.llm.generate(reqs, sp)
            for rid, o in zip(batch, outs):
                if kind == "rel":
                    keys = [k for k, _ in survs[rid]]
                    cands = [{"id": j} for j in range(len(keys))]
                    parsed = parse_relations_only(o.outputs[0].text, cands)
                    rels = [{"subject": keys[x["subject"]], "predicate": x["predicate"],
                             "object": keys[x["object"]]} for x in parsed]
                    fo.write(json.dumps({"rid": rid, "rels": rels}) + "\n")
                else:
                    fo.write(json.dumps({"rid": rid, "glob": o.outputs[0].text.strip()}) + "\n")
            fo.flush()
            print(f"  [{kind}] {min(i + args.chunk, len(pending))}/{len(pending)}", flush=True)
        fo.close()

    if rel_pending:
        print("[relations]", flush=True); run(rel_pending, "rel", rel_path, sp_rel)
    if glob_pending:
        print("[global]", flush=True); run(glob_pending, "glob", glob_path, sp_g)

    # reload and assemble if everything now done
    rel_done, glob_done = load_done(rel_path), load_done(glob_path)
    rel_ok = all(rid in rel_done for rid in rel_needed)
    glob_ok = (len(glob_done) >= len(recs)) if args.regen_global else True
    if rel_ok and glob_ok:
        assemble(recs, survs, rel_done, glob_done, args.out_dir, args.regen_global)
    else:
        print("PARTIAL — rerun to resume", flush=True)


if __name__ == "__main__":
    main()
