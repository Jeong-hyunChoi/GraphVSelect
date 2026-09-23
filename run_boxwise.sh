#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Box-wise verification baseline (the "Box-wise" side of Table 4): each
# candidate is judged independently with a Yes/No question over the same
# verified scene graph.  Compare its score against run_selection.sh.
#
#     bash run_boxwise.sh            # on the graph built by run_pipeline.sh (data/verified_sg)
#     # add --no-sg (below) for the box-wise no-SG ablation
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

IMAGES="${IMAGES:-sample/images}"       # bundled sample images; override with your dataset dir
SG="${SG:-data/verified_sg}"
VLM="${VLM:-Qwen/Qwen3-VL-8B-Instruct}"
WORK="${WORK:-outputs}"
TP="${TP:-1}"

echo "== box-wise verification over the scene graph =="
python scripts/run_verify.py --in-dir "$SG" --vlm-hf "$VLM" --tp "$TP" \
    --out "$WORK/verify_conf.jsonl" --work-dir "$WORK/verify_work" --image-root "$IMAGES"

echo "== box-wise score =="
python eval/box_wise_eval.py --in-dir "$SG" --conf "$WORK/verify_conf.jsonl" \
    --out "$WORK/box_wise.txt"
