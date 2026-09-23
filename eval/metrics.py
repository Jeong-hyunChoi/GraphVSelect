"""Score a prediction dump into localization accuracy.

Reads a JSONL of per-record predictions {rid, dataset, split, pred_bbox, gt_bbox} such as
the one written by ``select_round2.py --pred-out`` (list-wise selection). For each split it
reports Acc@0.5 (the RefCOCO/+/g metric) as well as Acc@0.75, Acc@0.9 and mAcc (the Ref-L4
metric, averaged over ten IoU thresholds 0.50:0.05:0.95 with a strict ``>``). Pure recompute,
no inference.

  python eval/metrics.py --preds outputs/preds.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # 0.50 .. 0.95 (Ref-L4 mAcc)
HEAD = [0.5, 0.75, 0.9]                                 # reported individually


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="one or more preds.jsonl files (concatenated)")
    ap.add_argument("--by-split", action="store_true",
                    help="also print a per-(dataset, split) Acc@0.5 breakdown")
    args = ap.parse_args()

    rows = []
    for p in args.preds:
        for line in Path(p).open():
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    ious = [iou(d.get("pred_bbox"), d.get("gt_bbox")) for d in rows]
    n = len(ious)

    def acc(th):
        return 100.0 * sum(v > th for v in ious) / n if n else 0.0

    print(f"n = {n}")
    print("  " + "   ".join(f"Acc@{t}={acc(t):.2f}" for t in HEAD))
    macc = sum(100.0 * sum(v > t for v in ious) / n for t in IOUS) / len(IOUS) if n else 0.0
    print(f"  mAcc (0.50:0.05:0.95) = {macc:.2f}")

    if args.by_split:
        agg = defaultdict(list)
        for d, v in zip(rows, ious):
            agg[(d.get("dataset"), d.get("split"))].append(v)
        print("\nper-(dataset, split) Acc@0.5:")
        tot_h = tot_n = 0
        for k in sorted(agg, key=str):
            vs = agg[k]
            h = sum(v > 0.5 for v in vs)
            tot_h += h; tot_n += len(vs)
            print(f"  {str(k):44s} {100.0*h/len(vs):6.2f}  (n={len(vs)})")
        print(f"  {'POOLED':44s} {100.0*tot_h/max(1,tot_n):6.2f}  (n={tot_n})")


if __name__ == "__main__":
    main()
