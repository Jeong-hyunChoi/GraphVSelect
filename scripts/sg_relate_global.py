#!/usr/bin/env python3
"""Regenerate query-relevant relations + a global caption over the FINAL object set. RESUMABLE.

Runs after object augmentation (A) and caption verification (S / jury), so the object
set is final and every object -- including newly added ones -- participates in the
relations. One RELATIONS_ONLY_PROMPT JSON call per record produces pairwise relations
over the marked ids, and one GLOBAL_CAPTION_PROMPT_QUERYCOND call per record produces
the scene-layout line. Neither prompt reads captions: relations are computed pairwise
over the marked ids, and the global caption takes only the annotated image and the query.

  python scripts/sg_relate_global.py --in-dir <dump_after_jury> --out-dir <dump_out> \
      --work-dir outputs/<tag>_rg_work --vlm-hf Qwen/Qwen3-VL-8B-Instruct --tp 1
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
from graphvselect.vlm import VLM
from graphvselect.images import ImageResolver
from graphvselect.render import annotate_image as _annotate_image
from graphvselect.parsing import parse_relations_only as _parse_relations_only
from graphvselect.prompts import RELATIONS_ONLY_PROMPT, GLOBAL_CAPTION_PROMPT_QUERYCOND



def load_done(path):
    done = {}
    if path.exists():
        for line in path.open():
            try:
                d = json.loads(line); done[d["rid"]] = d
            except json.JSONDecodeError:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=24)
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
    keys_of = [list(r["sg"]["objects"].keys()) for r in recs]

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    rel_path, glob_path = work / "rel_done.jsonl", work / "glob_done.jsonl"
    rel_done, glob_done = load_done(rel_path), load_done(glob_path)

    rel_needed = [rid for rid in range(len(recs)) if len(keys_of[rid]) >= 2]
    # A record with no surviving objects has no scene to lay out: GLOBAL_CAPTION_PROMPT_
    # QUERYCOND asks about "the highlighted objects", and on an unmarked image the VLM
    # would just confabulate. Those records are a guaranteed miss anyway (no candidates
    # to select), so leave their global empty.
    glob_needed = [rid for rid in range(len(recs)) if len(keys_of[rid]) >= 1]
    rel_pending = [rid for rid in rel_needed if rid not in rel_done]
    glob_pending = [rid for rid in glob_needed if rid not in glob_done]
    print(f"records={len(recs)}  rel_needed={len(rel_needed)} glob_needed={len(glob_needed)} "
          f"(skipping {len(recs)-len(glob_needed)} objectless)  pending: {len(rel_pending)} rel, "
          f"{len(glob_pending)} glob", flush=True)

    if rel_pending or glob_pending:
        vlm = VLM(hf_id=args.vlm_hf, dtype="float16", tensor_parallel_size=args.tp,
                           gpu_memory_utilization=0.90 if args.tp == 1 else 0.85,
                           max_num_seqs=1, max_pixels=args.max_pixels)
        sp_rel = vlm._SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0)
        sp_g = vlm._SamplingParams(max_tokens=128, temperature=0.0, top_p=1.0)
        resolver = ImageResolver(args.image_root)

        def annotated_of(rid):
            r = recs[rid]
            im = resolver.get(r)
            objs = r["sg"]["objects"]
            cands = [{"id": i, "bbox": list(o["bbox"]), "label": o.get("category", "")}
                     for i, (k, o) in enumerate(objs.items())]
            return _annotate_image(im, cands), cands

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
                        keys = keys_of[rid]
                        parsed = _parse_relations_only(o.outputs[0].text,
                                                       [{"id": j} for j in range(len(keys))])
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
        rel_done, glob_done = load_done(rel_path), load_done(glob_path)

    if not (all(rid in rel_done for rid in rel_needed)
            and all(rid in glob_done for rid in glob_needed)):
        print("PARTIAL — rerun to resume", flush=True); return

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "sgrg_w0.jsonl").open("w") as fo:
        for rid, r in enumerate(recs):
            new = dict(r)
            sg = dict(r["sg"])
            sg["relationships"] = rel_done.get(rid, {}).get("rels", [])
            sg["global_caption"] = glob_done.get(rid, {}).get("glob", "")
            new["sg"] = sg
            new["sg_objects_count"] = len(sg["objects"])
            new["sg_relations_count"] = len(sg["relationships"])
            fo.write(json.dumps(new, ensure_ascii=False) + "\n")
    print(f"ALL_DONE assembled {len(recs)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
