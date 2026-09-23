#!/usr/bin/env python3
"""Graft regenerated relations/global from SRC_RG onto CAPSJ, write to OUT.

For Table 3's 'caption' row: CAPSJ has the base object set with S/J-repaired captions;
SRC_RG (the 'none' variant) has the SAME base object set with relations/global
regenerated over it. relations/global depend only on (object set + marked image), not on
captions, so grafting is exact. Keyed by ENUMERATE index (not the rid field), matching
every other stage.

  python scripts/graft_ablation.py <capsj_dir> <none_dir> <out_dir>
"""
import json, glob, os, sys


def enum_lines(d):
    out = []
    for f in sorted(glob.glob(f"{d}/*_w*.jsonl")):
        for l in open(f):
            out.append(json.loads(l))
    return out


def main():
    capsj_d, none_d, out_d = sys.argv[1], sys.argv[2], sys.argv[3]
    CAP, RG = enum_lines(capsj_d), enum_lines(none_d)
    assert len(CAP) == len(RG), (len(CAP), len(RG))
    os.makedirs(out_d, exist_ok=True)
    bad = 0
    with open(f"{out_d}/t3cap_w0.jsonl", "w") as fo:
        for cap, rg in zip(CAP, RG):
            if list(cap["sg"]["objects"]) != list(rg["sg"]["objects"]):
                bad += 1
            r = dict(cap); s = dict(cap["sg"])
            s["relationships"] = rg["sg"].get("relationships", [])
            s["global_caption"] = rg["sg"].get("global_caption", "")
            r["sg"] = s
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    if bad:
        print(f"WARNING: {bad} records have mismatched object sets — graft is invalid there")
    print(f"grafted {len(CAP)} -> {out_d}  (object-set mismatches: {bad})")


if __name__ == "__main__":
    main()
