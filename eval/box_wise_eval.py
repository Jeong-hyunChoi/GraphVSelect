"""Box-wise verification — evaluation side (Table 4, "Box-wise Verification").

Scores the box-wise interface: given the per-candidate Yes/No confidences produced by
`run_verify.py` over the verified scene-graph dumps, we take, per record, the
argmax-confidence candidate and count it correct when its IoU with the ground-truth
box exceeds 0.5. Reports the pooled RefCOCO/+/g accuracy and the per-split breakdown.

This is the "Box-wise Verification" side of the comparison in Table 4; the list-wise
"Structured Selection" side is produced by `select_round1.py` / `select_round2.py`.

The saturation diagnostic (fraction of candidate confidences below 0.1 or above 0.9)
is the "multiple-True" tie evidence (saturated Yes-confidences).

Host-only (no GPU): it only scores saved confidences.

Inputs:
  --in-dir : scene-graph dump shards (*_w*.jsonl); each record carries
             {sg.objects{id:{bbox,...}}, gt_bbox, dataset, split}
  --conf   : per-record confidence dump {rid, scores{id:p_yes}, diffs{id:score}}
Output: a text summary (pooled + per-split accuracy).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0


SPLITS = [("RefCOCO", "val"), ("RefCOCO", "testA"), ("RefCOCO", "testB"),
          ("RefCOCOplus", "val"), ("RefCOCOplus", "testA"), ("RefCOCOplus", "testB"),
          ("RefCOCOg", "val"), ("RefCOCOg", "test")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/sg_dump")
    ap.add_argument("--conf", default="outputs/verify_conf.jsonl")
    ap.add_argument("--out", default="outputs/box_wise_eval.txt")
    args = ap.parse_args()

    recs = []
    for f in sorted(Path(args.in_dir).glob("*_w*.jsonl")):
        for line in f.open():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    conf = {}
    for line in Path(args.conf).open():
        d = json.loads(line); conf[d["rid"]] = d

    def argmax_hit(r, scores):
        objs = r["sg"]["objects"]
        if not scores:
            return False
        pk = max(scores, key=scores.get)
        return pk in objs and iou(objs[pk]["bbox"], r["gt_bbox"]) >= 0.5

    res = {"argmax_sg": [], "argmax_ctxfree": []}
    keys = []
    n_ans = top1_sg = top1_cf = ret_sg = ret_cf = 0
    ext_sg = ext_cf = tot_obj = 0
    for rid, r in enumerate(recs):
        objs = r["sg"]["objects"]
        sg = conf.get(rid, {}).get("diffs", {})
        cf = {k: (o.get("verify_conf") or 0.0) for k, o in objs.items()}
        res["argmax_sg"].append(argmax_hit(r, sg))
        res["argmax_ctxfree"].append(argmax_hit(r, cf))
        keys.append((r["dataset"].split("/")[-1], r["split"]))
        if not objs:
            continue  # verification can empty a record -> counts as a miss
        ak = max(objs, key=lambda k: iou(objs[k]["bbox"], r["gt_bbox"]))
        ns = conf.get(rid, {}).get("scores", {})
        for k in objs:
            tot_obj += 1
            s = ns.get(k, 0.5); ext_sg += (s < 0.1 or s > 0.9)
            ext_cf += (cf[k] < 0.1 or cf[k] > 0.9)
        if iou(objs[ak]["bbox"], r["gt_bbox"]) >= 0.5:
            n_ans += 1
            top1_sg += ak == max(ns, key=ns.get) if ns else False
            top1_cf += ak == max(cf, key=cf.get)
            ret_sg += ns.get(ak, 0) >= 0.3
            ret_cf += cf[ak] >= 0.3

    def acc(name, flt=None):
        xs = [h for h, k in zip(res[name], keys) if flt is None or k == flt]
        return 100 * sum(xs) / max(1, len(xs))

    lines = [f"n={len(recs)}  answers-in-SG={n_ans}", "",
             "box-wise argmax accuracy:",
             f"  SG-conditioned : {sum(res['argmax_sg'])}/{len(recs)} ({acc('argmax_sg'):.1f}%)",
             f"  context-free   : {sum(res['argmax_ctxfree'])}/{len(recs)} ({acc('argmax_ctxfree'):.1f}%)",
             "",
             f"answer top-1 rate : SG {100*top1_sg/n_ans:.1f}%  vs ctx-free {100*top1_cf/n_ans:.1f}%",
             f"retention @0.3    : SG {100*ret_sg/n_ans:.1f}%  vs ctx-free {100*ret_cf/n_ans:.1f}%",
             f"saturation (<0.1 or >0.9): SG {100*ext_sg/tot_obj:.1f}%  vs ctx-free {100*ext_cf/tot_obj:.1f}%",
             "", "per-split argmax:",
             "config  " + "  ".join(f"{d.replace('RefCOCOplus','R+').replace('RefCOCO','R')} {s}"
                                    for d, s in SPLITS)]
    for name in ("argmax_sg", "argmax_ctxfree"):
        cells = [f"{acc(name, k):.1f}" for k in SPLITS]
        lines.append(f"{name}  " + "  ".join(cells))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
