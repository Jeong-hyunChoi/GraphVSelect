#!/usr/bin/env python3
"""Step-2 caption repair, layer 1: deterministic spatial-rank injection.

For every same-category cluster (>=2 objects sharing the category head) append a
short discriminative phrase to each member's caption: horizontal (or vertical,
whichever axis has more spread) rank + size extremes. Purely deterministic, no
VLM, no query, no selector feedback -- self-contained SG construction repair.
Targets a common failure mode: same-category siblings with captions that are
true but non-discriminative.

  python scripts/spatial_inject.py <in_dump_dir> <out_dump_dir>
"""
import json, sys, glob, os

def ORDW(i):  # 0-based index -> ordinal word, any size
    n = i + 1
    if 10 <= n % 100 <= 20: suf = "th"
    else: suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def rank_word(i, k, axis):
    if axis == "x":
        if i == 0: return "leftmost"
        if i == k - 1: return "rightmost"
        return f"{ORDW(i)} from the left"
    else:
        if i == 0: return "topmost"
        if i == k - 1: return "bottommost"
        return f"{ORDW(i)} from the top"

def inject(objs):
    heads = {}
    for key, o in objs.items():
        heads.setdefault(key.rsplit("_", 1)[0], []).append(key)
    for h, keys in heads.items():
        if len(keys) < 2:
            continue
        k = len(keys)
        cx = {key: (objs[key]["bbox"][0] + objs[key]["bbox"][2]) / 2 for key in keys}
        cy = {key: (objs[key]["bbox"][1] + objs[key]["bbox"][3]) / 2 for key in keys}
        axis = "x" if (max(cx.values()) - min(cx.values())) >= (max(cy.values()) - min(cy.values())) else "y"
        order = sorted(keys, key=lambda key: cx[key] if axis == "x" else cy[key])
        area = {key: (objs[key]["bbox"][2] - objs[key]["bbox"][0]) * (objs[key]["bbox"][3] - objs[key]["bbox"][1]) for key in keys}
        amax = max(area, key=area.get); amin = min(area, key=area.get)
        cat = (objs[keys[0]].get("category") or h.replace("_", " ")).strip()
        for i, key in enumerate(order):
            note = f" Among the {k} {cat}(s) in the image, this one is the {rank_word(i, k, axis)}"
            if k >= 3 or area[amax] > 1.5 * area[amin]:
                if key == amax: note += " and the largest"
                elif key == amin: note += " and the smallest"
            note += "."
            cap = (objs[key].get("caption") or "").rstrip()
            objs[key]["caption"] = (cap + ("" if cap.endswith(".") or not cap else ".") + note).strip()

def main():
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    n = clusters = touched = 0
    for f in sorted(glob.glob(os.path.join(in_dir, "*_w*.jsonl"))):
        with open(os.path.join(out_dir, os.path.basename(f)), "w") as fo:
            for line in open(f):
                r = json.loads(line); n += 1
                objs = r["sg"]["objects"]
                before = {k: o.get("caption") for k, o in objs.items()}
                inject(objs)
                ch = sum(1 for k in objs if objs[k].get("caption") != before[k])
                if ch: clusters += 1; touched += ch
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{in_dir} -> {out_dir}: {n} recs, {clusters} recs touched, {touched} captions injected")

if __name__ == "__main__":
    main()
