#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the list-wise selection result (Table 2 / the "Selection" side of
# Table 4) on a *verified scene-graph* directory.
#
# Quick start (after run_pipeline.sh has built data/verified_sg):
#     IMAGES=/path/to/coco/train2014 bash run_selection.sh
#
# On your own verified graph (built with run_pipeline.sh):
#     IMAGES=/path/to/images SG=data/verified_sg bash run_selection.sh
#
# Backbones: set VLM to the Qwen / LLaVA / Molmo model id (add --compact-sg for
# Molmo inside the script if needed).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

IMAGES="${IMAGES:-sample/images}"       # bundled sample images; override with your dataset dir
SG="${SG:-data/verified_sg}"           # verified-graph dir built by run_pipeline.sh
VLM="${VLM:-Qwen/Qwen3-VL-8B-Instruct}"
WORK="${WORK:-outputs}"
TP="${TP:-1}"

echo "== list-wise selection, round 1 (A: constrained-logprob) =="
python scripts/select_round1.py --in-dir "$SG" --vlm-hf "$VLM" --tp "$TP" \
    --gate-frac 0.30 --calib-n 2000 --calib-seed 0 --max-model-len 8192 \
    --work-dir "$WORK/select" --out "$WORK/select_r1.txt" --image-root "$IMAGES"

echo "== round 2 (B3: order-symmetrized re-answer over gated records) =="
python scripts/select_round2.py --in-dir "$SG" --vlm-hf "$VLM" --tp "$TP" \
    --calib-n 2000 --calib-seed 0 --max-model-len 8192 \
    --r1-cache "$WORK/select/r1.jsonl" --pred-out "$WORK/preds.jsonl" \
    --work-dir "$WORK/select" --out "$WORK/select_r2.txt" --image-root "$IMAGES"

echo "== scores (Acc@0.5/0.75/0.9, mAcc) =="
python eval/metrics.py --preds "$WORK/preds.jsonl"
