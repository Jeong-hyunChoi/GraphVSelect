"""Scene-graph text serialization and dataset-name mapping.

Turns a stored scene-graph dict into the object / relation / global-caption text blocks
that are placed into the selection and verification prompts.
"""
from __future__ import annotations

def cid_of(k):
    try:
        return int(k.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def sg_blocks(sg):
    objs = sg["objects"]
    lines = [f"  [{cid_of(k)}] {o.get('caption') or o.get('category','')}" for k, o in objs.items()]
    rels = sg.get("relationships") or []
    rl = [f"  [{cid_of(r['subject'])}] {r.get('predicate','')} [{cid_of(r['object'])}]" for r in rels
          if isinstance(r.get("subject"), str)] or ["  (none)"]
    if rels and not isinstance(rels[0].get("subject"), str):   # already ints
        rl = [f"  [{r['subject']}] {r.get('predicate','')} [{r['object']}]" for r in rels]
    return "\n".join(lines), "\n".join(rl), (sg.get("global_caption") or "(none)")
