"""GraphVSelect: training-free zero-shot REC via verified graph-guided selection.

Dependency-free helpers (geometry, parsing, spatial) import directly. Model-backed
modules (detect, render, vlm) require torch / transformers / vllm / PIL and are imported
explicitly, e.g. ``from graphvselect.vlm import VLM``.
"""
from . import geometry, parsing, spatial  # noqa: F401
