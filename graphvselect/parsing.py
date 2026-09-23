"""Parsers for the frozen models' text outputs (pure functions; json + regex only).

Vision-language models often wrap answers in prose or emit ids as strings, so each
parser tolerates missing JSON wrappers and coerces ids before bounds-checking.
"""
from __future__ import annotations

import json
import re


def parse_pair_predicate(raw: str) -> str:
    """Extract the predicate phrase from a single per-pair response.

    Expected format: {"predicate": "left of"}. Falls back to the first short line if
    the JSON wrapper is missing.
    """
    match = re.search(r"\{[^{}]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            pred = parsed.get("predicate")
            if isinstance(pred, str):
                return pred.strip()
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: first non-empty line, capped to a short phrase.
    for line in raw.strip().splitlines():
        s = line.strip().strip('"').strip("'")
        if s:
            return s[:64]
    return ""


def parse_relations_only(raw: str, candidates: list[dict]) -> list[dict]:
    """Parse relations from the relation-only call of the per-crop pipeline."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    raw_rels = parsed.get("relations", []) or []
    valid_ids = {c["id"] for c in candidates}
    out = []
    for r in raw_rels:
        if not isinstance(r, dict) or "subject" not in r or "object" not in r:
            continue
        try:
            s = int(r["subject"])
            o = int(r["object"])
        except (ValueError, TypeError):
            continue
        if s in valid_ids and o in valid_ids:
            out.append({"subject": s,
                        "predicate": str(r.get("predicate", "")),
                        "object": o})
    return out
