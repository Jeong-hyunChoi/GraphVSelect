"""Box-wise verification (Table 4, "Box-wise Verification").

For each candidate region the VLM is asked an independent Yes/No question, conditioned on
the verified scene graph (per-object captions + relations + global caption) supplied as
textual context. The answer is scored by the Yes-vs-No first-token log-probability
difference and turned into a per-candidate confidence. This is the box-wise interface that
Table 4 compares against list-wise selection.

Output per record: {"rid", "scores": {obj_key: sigmoid(diff)}, "diffs": {obj_key: yes_lp - no_lp}}.
Resumable per record; box-wise accuracy is then computed by eval/box_wise_eval.py.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from graphvselect.vlm import VLM
from graphvselect.images import ImageResolver
from graphvselect.render import annotate_ids
from graphvselect.serialize import sg_blocks, cid_of


PROMPT = """\
In this image, candidate regions are marked with coloured boxes labelled [0], [1], [2], ... .

Scene description (produced by an automatic analyzer; use as context, it may contain minor errors):
Objects:
{objects_block}
Relations:
{relations_block}
Scene layout: {global_caption}

Referring expression: "{query}"

Look at region [{cid}] in the image. Using both the image and the scene description \
(the attributes, positions and relations of the OTHER marked objects) as context, \
decide: is the object in region [{cid}] the one referred to by the expression? \
Answer with only 'Yes' or 'No'.
"""

# single-box + SG WITHOUT the global caption. The query-conditioned global
# ("Scene layout: ...") echoes the query and asserts (often wrongly) the target's
# position — pure interference for a verifier that can see the layout itself.
# Keeps Objects + Relations only.
PROMPT_SINGLE_NOGLOB = """\
In this image, ONE candidate region is highlighted with a coloured box labelled [{cid}].

Scene description of ALL detected objects (produced by an automatic analyzer; \
object [{cid}] is the highlighted one; the others are not drawn but are described below):
Objects:
{objects_block}
Relations:
{relations_block}

Referring expression: "{query}"

Using both the image and the scene description (the attributes and relations of \
the other objects) as context, decide: is the highlighted object [{cid}] the one \
referred to by the expression? Answer with only 'Yes' or 'No'.
"""

# single-box, NO-SG baseline: same perception, no scene-graph context. The
# per-model (with-SG − no-SG) delta isolates SG-conditioning for that verifier.
PROMPT_SINGLE_NOSG = """\
In this image, ONE candidate region is highlighted with a coloured box labelled [{cid}].

Referring expression: "{query}"

Is the highlighted object [{cid}] the one referred to by the expression? \
Answer with only 'Yes' or 'No'.
"""

# single-box variant: only the queried object's box is drawn (literature-style
# "one highlighted box at a time" perception); the SG text still describes ALL
# objects by id, so the prompt explains that correspondence.
PROMPT_SINGLE = """\
In this image, ONE candidate region is highlighted with a coloured box labelled [{cid}].

Scene description of ALL detected objects (produced by an automatic analyzer; \
object [{cid}] is the highlighted one; the others are not drawn but are described below):
Objects:
{objects_block}
Relations:
{relations_block}
Scene layout: {global_caption}

Referring expression: "{query}"

Using both the image and the scene description (the attributes, positions and \
relations of the other objects) as context, decide: is the highlighted object \
[{cid}] the one referred to by the expression? Answer with only 'Yes' or 'No'.
"""


def yes_no(first, tokenizer):
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
    return yes_lp - no_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/sg_dump")
    ap.add_argument("--out", default="outputs/verify_conf.jsonl")
    ap.add_argument("--work-dir", default="outputs/verify_work")
    ap.add_argument("--vlm-hf", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--single-box", action="store_true",
                    help="draw ONLY the queried box per request (literature-style) "
                         "instead of marking all candidates on one shared image")
    ap.add_argument("--no-sg", action="store_true",
                    help="omit the scene-graph context block (baseline for the "
                         "per-model SG-conditioning delta); implies --single-box prompt")
    ap.add_argument("--no-global", action="store_true",
                    help="keep Objects+Relations but drop the (query-conditioned, "
                         "often-wrong) global caption line; implies --single-box")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="raise for models without a pixel cap (LLaVA anyres / InternVL "
                         "tiling: full-image tokens + SG text overflow 4096)")
    ap.add_argument("--compact-sg", action="store_true",
                    help="shrink the SG block for tight-context models (Molmo 4096): "
                         "truncate caption lines, drop the global caption")
    ap.add_argument("--max-num-seqs", type=int, default=64,
                    help="vLLM concurrent sequences; batching the per-candidate requests speeds up "
                         "verification. Greedy decoding, so the outputs are unchanged.")
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
    wf = work / "sg_rec.jsonl"
    done = set()
    if wf.exists():
        for line in wf.open():
            try: done.add(json.loads(line)["rid"])
            except (json.JSONDecodeError, KeyError): pass
    pending = [rid for rid in range(len(recs)) if rid not in done]
    print(f"sg-verify pending: {len(pending)} recs (done {len(done)})", flush=True)

    if pending:
        vlm = VLM(hf_id=args.vlm_hf, dtype="float16", tensor_parallel_size=args.tp,
                           gpu_memory_utilization=0.85, max_num_seqs=args.max_num_seqs,
                           max_model_len=args.max_model_len)
        if "molmo" in args.vlm_hf.lower():
            # Molmo's processor has no chat template; vLLM's molmo path re-wraps
            # plain text through the model's own processor -> pass text as-is.
            vlm._build_prompt = lambda t: t
        elif "internvl" in args.vlm_hf.lower():
            # InternVL's AutoProcessor IS the tokenizer; its chat template needs
            # plain-string content with an explicit <image> placeholder (list-style
            # content crashes its jinja template with str+list TypeError).
            _t = vlm.processor
            vlm._build_prompt = lambda t: _t.apply_chat_template(
                [{"role": "user", "content": "<image>\n" + t}],
                tokenize=False, add_generation_prompt=True)
        sp = vlm._SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, logprobs=20)
        # InternVL's AutoProcessor returns the tokenizer itself (no .tokenizer attr)
        tok = getattr(vlm.processor, "tokenizer", vlm.processor)
        img_of = ImageResolver(args.image_root).get
        fo = wf.open("a")
        for i in range(0, len(pending), args.chunk):
            for rid in pending[i:i + args.chunk]:
                r = recs[rid]; sg = r["sg"]
                cands = [{"id": cid_of(k), "bbox": o["bbox"]} for k, o in sg["objects"].items()]
                key_of = {cid_of(k): k for k in sg["objects"]}
                ob, rb, gc = sg_blocks(sg)
                if args.compact_sg:
                    ob = "\n".join(l[:72] for l in ob.splitlines())
                    gc = "(none)"
                if args.no_sg:
                    base = img_of(r)
                    reqs = [{"prompt": vlm._build_prompt(PROMPT_SINGLE_NOSG.format(
                                query=r["query"], cid=c["id"])),
                             "multi_modal_data": {"image": annotate_ids(base, [c])}} for c in cands]
                elif args.no_global:
                    base = img_of(r)
                    reqs = [{"prompt": vlm._build_prompt(PROMPT_SINGLE_NOGLOB.format(
                                objects_block=ob, relations_block=rb,
                                query=r["query"], cid=c["id"])),
                             "multi_modal_data": {"image": annotate_ids(base, [c])}} for c in cands]
                elif args.single_box:
                    base = img_of(r)
                    reqs = [{"prompt": vlm._build_prompt(PROMPT_SINGLE.format(
                                objects_block=ob, relations_block=rb, global_caption=gc,
                                query=r["query"], cid=c["id"])),
                             "multi_modal_data": {"image": annotate_ids(base, [c])}} for c in cands]
                else:
                    ann = annotate_ids(img_of(r), cands)
                    reqs = [{"prompt": vlm._build_prompt(PROMPT.format(
                                objects_block=ob, relations_block=rb, global_caption=gc,
                                query=r["query"], cid=c["id"])),
                             "multi_modal_data": {"image": ann}} for c in cands]
                try:
                    outs = vlm.llm.generate(reqs, sp)
                    diffs = {key_of[c["id"]]: yes_no(o.outputs[0], tok) for c, o in zip(cands, outs)}
                    scores = {k: 1.0 / (1.0 + math.exp(-d)) for k, d in diffs.items()}
                    fo.write(json.dumps({"rid": rid, "scores": scores,
                                         "diffs": {k: round(d, 4) for k, d in diffs.items()}}) + "\n")
                except Exception as e:   # e.g. prompt-length overflow on one record
                    fo.write(json.dumps({"rid": rid, "scores": {}, "diffs": {},
                                         "err": str(e)[:100]}) + "\n")
            fo.flush()
            print(f"  {min(i + args.chunk, len(pending))}/{len(pending)}", flush=True)
        fo.close()

    agg = {}
    for line in wf.open():
        try:
            d = json.loads(line); agg[d["rid"]] = d
        except (json.JSONDecodeError, KeyError): pass
    if len(agg) >= len(recs):
        with Path(args.out).open("w") as f:
            for rid in sorted(agg):
                f.write(json.dumps(agg[rid]) + "\n")
        print(f"SG_VERIFY_DONE -> {args.out}", flush=True)
    else:
        print(f"PARTIAL ({len(agg)}/{len(recs)}) — rerun to resume", flush=True)


if __name__ == "__main__":
    main()
