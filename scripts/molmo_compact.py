#!/usr/bin/env python3
"""Molmo compact scene-graph helper: with --compact-sg the selection scripts truncate every
caption line to 72 chars, which silently cuts the spatial-rank tail appended by
spatial_inject.py (it sits at the END of the caption). This transform moves the
tail's information to the FRONT as a compressed tag so it survives truncation:

  "...parked by the road. Among the 5 sedan(s) in the image, this one is the
   2nd from the left and the largest."
     ->  "[2nd-from-L/5,max] ...parked by the road."

  python scripts/molmo_compact.py <in_dump> <out_dump>
"""
import json, re, sys, glob, os

TAIL = re.compile(r"\s*Among the (\d+) (.+?)\(s\) in the image, this one is the ([^.]+)\.")

ABBR = [(" and the largest", ",max"), (" and the smallest", ",min"),
        ("leftmost", "L-most"), ("rightmost", "R-most"),
        ("topmost", "T-most"), ("bottommost", "B-most"),
        (" from the left", "-from-L"), (" from the top", "-from-T")]

def compress(k, rank):
    r = rank
    suf = ""
    for a, b in ABBR:
        if a in (" and the largest", " and the smallest") and a in r:
            suf = b; r = r.replace(a, "")
        else:
            r = r.replace(a, b)
    return f"[{r.strip()}/{k}{suf}] "

def main():
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    n = fixed = 0
    for f in sorted(glob.glob(os.path.join(in_dir, "*_w*.jsonl"))):
        with open(os.path.join(out_dir, os.path.basename(f)), "w") as fo:
            for line in open(f):
                r = json.loads(line); n += 1
                for k, o in r["sg"]["objects"].items():
                    cap = o.get("caption") or ""
                    m = TAIL.search(cap)
                    if m:
                        tag = compress(m.group(1), m.group(3))
                        o["caption"] = (tag + TAIL.sub("", cap).strip()).strip()
                        fixed += 1
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{in_dir} -> {out_dir}: {n} recs, {fixed} captions tail-fronted")

if __name__ == "__main__":
    main()
