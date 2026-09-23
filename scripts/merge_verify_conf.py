#!/usr/bin/env python3
"""Object-level verification (F, part 3, CPU): merge cross-model confidences into the dump.

Takes a ``gen_verify_conf.py`` output (rid -> {cid: P(yes)}) and writes it back onto the
base scene-graph dump as a per-object ``verify_conf`` field, so ``anchor_verify.py`` and
``object_filter.py`` (which read ``verify_conf`` from the dump) can consume it. Objects
with no matching score default to 0.5.

  python scripts/merge_verify_conf.py <in_dump_dir> <vc.jsonl> <out_dump_dir>
"""
import json, sys, glob, os

in_dir, vc_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
vc = {}
for line in open(vc_path):
    d = json.loads(line)
    vc[d["rid"]] = d.get("verify_conf", {})

os.makedirs(out_dir, exist_ok=True)
cid_of = lambda k: k.rsplit("_", 1)[-1]  # object key 'red_0' -> '0' (matches gen's str(id))
n = miss = 0
for f in sorted(glob.glob(os.path.join(in_dir, "*_w*.jsonl"))):
    out_f = os.path.join(out_dir, os.path.basename(f))
    with open(out_f, "w") as fo:
        for line in open(f):
            r = json.loads(line); rid = r["rid"]; conf = vc.get(rid, {})
            for k, o in r["sg"]["objects"].items():
                c = conf.get(cid_of(k))
                if c is None: c = conf.get(str(o.get("id")))
                o["verify_conf"] = round(c, 4) if c is not None else 0.5
                if c is None: miss += 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"merged {n} records into {out_dir} ({miss} objects had no verify_conf -> defaulted 0.5)")
