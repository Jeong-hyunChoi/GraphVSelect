#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end pipeline: build the verified scene graph from raw images and then
# run list-wise selection.  Input is a `samples.jsonl` where each record has
# `query`, `gt_bbox`, and an image reference (`file_name` or `image_path`).
#
#     IMAGES=/path/to/images SAMPLES=data/samples.jsonl bash run_pipeline.sh
#
# Every stage writes to a resumable work dir, so re-running continues where it
# stopped.  The frozen backbone is set by VLM; the caption-level listener juror
# uses LISTENER_VLM.  Hyper-parameters default to the values reported in the
# paper (see the paper's hyper-parameter table).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

IMAGES="${IMAGES:-sample/images}"       # bundled sample images; override with your dataset dir
SAMPLES="${SAMPLES:-sample/samples.jsonl}"   # bundled 100-record sample; override with your records
VLM="${VLM:-Qwen/Qwen3-VL-8B-Instruct}"
LISTENER_VLM="${LISTENER_VLM:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
D="${D:-data}"                                         # dataset / intermediate dirs
WORK="${WORK:-outputs}"                                # work + output dirs
TP="${TP:-1}"

echo "== [1/6] base scene graph: detect -> captions + relations + global =="
python scripts/build_sg.py --sample "$SAMPLES" --out-dir "$D/base_sg" \
    --shard 0 --nshards 1 --vlm-hf "$VLM" --image-root "$IMAGES"

echo "== [2/6] object-level verification (F): cross-model confidence =="
python scripts/gen_verify_conf.py --in-dir "$D/base_sg" --verifier-hf "$VLM" \
    --out "$WORK/vc.jsonl" --image-root "$IMAGES"
python scripts/merge_verify_conf.py "$D/base_sg" "$WORK/vc.jsonl" "$D/vc_sg"

echo "== [3/6] object-level verification (F): anchor rescue + confidence filter =="
python scripts/anchor_verify.py --in-dir "$D/vc_sg" --out "$WORK/anchor.jsonl" \
    --work-dir "$WORK/anchor_work" --thr 0.3 --vlm-hf "$VLM" --image-root "$IMAGES"
python scripts/object_filter.py --in-dir "$D/vc_sg" --out-dir "$D/obj_sg" \
    --work-dir "$WORK/f_work" --thr 0.2 --rescue vlm --rescue-cap 3 \
    --anchor-thr 0.5 --anchor-file "$WORK/anchor.jsonl" --filter-only

echo "== [4/6] object-level augmentation (A): expanded re-detect + add back =="
python scripts/refl4_expand_rel.py --in-dir "$D/obj_sg" \
    --out-cands "$WORK/cands.jsonl" --out-rel "$WORK/rel.jsonl" \
    --work-dir "$WORK/exp_work" --image-root "$IMAGES"
python scripts/surgical_augment.py --in-dir "$D/obj_sg" --out-dir "$D/fa_sg" \
    --cands "$WORK/cands.jsonl" --rel "$WORK/rel.jsonl" --vlm-hf "$VLM" --image-root "$IMAGES"

echo "== [5/6] caption-level verification: spatial-rank + listener jury =="
python scripts/spatial_inject.py "$D/fa_sg" "$D/fas_sg"
python scripts/listener_jury.py --phase gen   --in-dir "$D/fas_sg" \
    --work-dir "$WORK/jury" --vlm-hf "$VLM" --image-root "$IMAGES"
python scripts/listener_jury.py --phase score --in-dir "$D/fas_sg" \
    --work-dir "$WORK/jury" --vlm-hf "$LISTENER_VLM" --image-root "$IMAGES"
python scripts/listener_jury.py --phase assemble --in-dir "$D/fas_sg" \
    --work-dir "$WORK/jury" --out-dir "$D/cap_sg"

echo "== [6/6] regenerate query-relevant relations + global -> verified SG =="
python scripts/sg_relate_global.py --in-dir "$D/cap_sg" --out-dir "$D/verified_sg" \
    --work-dir "$WORK/rg_work" --vlm-hf "$VLM" --image-root "$IMAGES"

echo "== verified scene graph ready: $D/verified_sg =="
echo "== running list-wise selection on it =="
SG="$D/verified_sg" VLM="$VLM" WORK="$WORK" TP="$TP" IMAGES="$IMAGES" bash ./run_selection.sh
