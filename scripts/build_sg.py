"""Scene-graph construction driver.

For every record (query + image), this builds the verified-graph inputs in two phases:
  Phase A -- extract query nouns, prompt GroundingDINO to obtain candidate proposals.
  Phase B -- for each candidate, generate a per-crop caption (with textual nearby-object
             hints), query-conditional pairwise relations, and a query-conditional global
             caption, all in one batched VLM call, then serialize to the dump schema.

The dumps produced here are the input to run_verify.py (box-wise) and select_round1/2.py
(list-wise selection). Sharded and resumable.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from graphvselect.images import ImageResolver
from graphvselect.spatial import bbox_center, _direction_from_centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/samples.jsonl")
    ap.add_argument("--out-dir", default="data/sg_dump")
    ap.add_argument("--work-dir", default="outputs/build_work")
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.20)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--cap", type=int, default=15)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--hints-topk", type=int, default=3)
    ap.add_argument("--recs-per-batch", type=int, default=16,
                    help="records pooled into one cross-record vLLM batch in Phase B "
                         "(fills max_num_seqs; output identical, ~several x faster).")
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="vLLM concurrent sequences; higher values batch more requests per call.")
    ap.add_argument("--image-root", default=None,
                    help="directory of prepared dataset images; a record's "
                         "image_path/file_name is resolved under it (see graphvselect/images.py)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.sample)]
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    mine = [rid for rid in range(len(recs)) if rid % args.nshards == args.shard]

    img_of = ImageResolver(args.image_root).get

    # ---------- Phase A: nouns + detect (GDINO only) ----------
    detf = work / f"dets_s{args.shard}.jsonl"
    done = set()
    if detf.exists():
        for line in detf.open():
            try: done.add(json.loads(line)["rid"])
            except (json.JSONDecodeError, KeyError): pass
    pend = [rid for rid in mine if rid not in done]
    if pend:
        from graphvselect.detect import GroundingDINODetector, _format_vocab
        from graphvselect.nouns import QueryNounExtractor
        qne = QueryNounExtractor()
        det = GroundingDINODetector("IDEA-Research/grounding-dino-base",
                                    box_threshold=args.box_thr, text_threshold=args.text_thr,
                                    nms_iou=args.nms)
        fo = detf.open("a")
        for n, rid in enumerate(pend):
            r = recs[rid]
            nouns = qne.extract(r["query"]) or []
            dets = []
            if nouns:
                for d in sorted(det(img_of(r), _format_vocab(nouns)),
                                key=lambda x: -x.score)[:args.cap]:
                    dets.append({"bbox": [float(x) for x in d.bbox],
                                 "label": d.label, "score": float(d.score)})
            fo.write(json.dumps({"rid": rid, "nouns": nouns, "dets": dets}) + "\n")
            if (n + 1) % 50 == 0: fo.flush(); print(f"  detect {n+1}/{len(pend)}", flush=True)
        fo.close(); del det
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
    print("phase A done", flush=True)
    DETS = {}
    for line in detf.open():
        d = json.loads(line); DETS[d["rid"]] = d

    # ---------- Phase B: SG generation ----------
    shardf = outd / f"sg_w{args.shard}.jsonl"
    done = set()
    if shardf.exists():
        for line in shardf.open():
            try: done.add(json.loads(line)["rid"])
            except (json.JSONDecodeError, KeyError): pass
    pend = [rid for rid in mine if rid not in done]
    print(f"SG pending {len(pend)}", flush=True)
    if not pend:
        print("SHARD_DONE"); return
    from graphvselect.vlm import VLM
    vlm = VLM(hf_id=args.vlm_hf, dtype="float16", tensor_parallel_size=1,
                       gpu_memory_utilization=0.92, max_num_seqs=args.max_num_seqs, max_model_len=4096)
    def sg_to_dump(sg, dd):
        id_to_key = {}; objects_dump = {}
        for o in sg["objects"]:
            cid = o.get("id"); cat = (o.get("category") or "object").replace(" ", "_")
            key = f"{cat}_{cid}"; id_to_key[cid] = key
            objects_dump[key] = {"bbox": o.get("bbox"), "category": o.get("category"),
                                 "kind": "main", "caption": (o.get("caption") or "").strip(),
                                 "id": cid}
        rels = []
        for rel in sg.get("relations", []):
            sk = id_to_key.get(rel.get("subject")); ok = id_to_key.get(rel.get("object"))
            if sk and ok:
                rels.append({"subject": sk, "predicate": (rel.get("predicate") or "").strip(),
                             "object": ok})
        return {"objects": objects_dump, "relationships": rels,
                "global_caption": sg.get("global_caption", "") or "",
                "query_nouns": dd.get("nouns", [])}

    fo = shardf.open("a")
    RPB = args.recs_per_batch
    for cs in range(0, len(pend), RPB):
        chunk = pend[cs:cs + RPB]
        items = []          # batch inputs for records with candidates
        slots = []          # (out, dd, item_idx or None) preserving chunk order
        for rid in chunk:
            r = recs[rid]; dd = DETS.get(rid, {})
            cands = dd.get("dets", [])
            out = {k: r[k] for k in ("dataset", "split", "query", "gt_bbox") if k in r}
            if "ds_idx" in r: out["ds_idx"] = r["ds_idx"]
            if "image_path" in r: out["image_path"] = r["image_path"]
            if "file_name" in r: out["file_name"] = r["file_name"]
            out["rid"] = rid
            if not cands:
                out["sg"] = {"objects": {}, "relationships": [], "global_caption": "",
                             "query_nouns": dd.get("nouns", [])}
                slots.append((out, dd, None)); continue
            img = img_of(r)
            for i, c in enumerate(cands): c["id"] = i
            hints = {}
            for c in cands:
                mc = bbox_center(c["bbox"])
                ranked = sorted(((o, ((bbox_center(o["bbox"])[0]-mc[0])**2
                                 + (bbox_center(o["bbox"])[1]-mc[1])**2) ** 0.5)
                                 for o in cands if o["id"] != c["id"]), key=lambda x: x[1])
                parts = [f"{o['label']} ({_direction_from_centers(mc, bbox_center(o['bbox']), img.width, img.height)})"
                         for o, _ in ranked[:args.hints_topk]]
                if parts: hints[c["id"]] = ", ".join(parts)
            items.append({"image": img, "query": r["query"], "candidates": cands,
                          "nearby_hints_by_id": hints or None})
            slots.append((out, dd, len(items) - 1))
        results = vlm.generate_sg_per_crop_pairwise_batch(
            items, query_conditional_relations=True, include_global_caption=True,
            query_conditional_global=True, max_caption_tokens=150) if items else []
        for out, dd, idx in slots:
            if idx is not None:
                sg, _ = results[idx]
                out["sg"] = sg_to_dump(sg, dd)
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
        fo.flush(); print(f"  SG {min(cs+RPB, len(pend))}/{len(pend)}", flush=True)
    fo.close()
    print("SHARD_DONE")


if __name__ == "__main__":
    main()
