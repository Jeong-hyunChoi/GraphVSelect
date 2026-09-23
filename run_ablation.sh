#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Table 3 — verification ablation (none / object / caption / both).
#
# Builds four verified-scene-graph variants from ONE base scene graph by toggling
# object-level verification (F + A) and caption-level verification (spatial + jury),
# then scores list-wise selection (A+B3) on each and prints the comparison.
#
#     bash run_ablation.sh                       # on the bundled sample (zero-config)
#     IMAGES=/path SAMPLES=/path/records.jsonl bash run_ablation.sh   # on your data
#
# The `caption' variant keeps the base object set, so its relations/global are grafted
# from the `none' variant (identical object set) rather than regenerated — this isolates
# the caption effect. Every stage is resumable.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

IMAGES="${IMAGES:-sample/images}"
SAMPLES="${SAMPLES:-sample/samples.jsonl}"
VLM="${VLM:-Qwen/Qwen3-VL-8B-Instruct}"
LISTENER_VLM="${LISTENER_VLM:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
D="${D:-data/ablation}"
WORK="${WORK:-outputs/ablation}"
TP="${TP:-1}"

# list-wise selection (A round 1 + B3 round 2) on an SG dir: sel <sg_dir> <tag>
sel () {
  python scripts/select_round1.py --in-dir "$1" --vlm-hf "$VLM" --tp "$TP" \
      --gate-frac 0.30 --calib-n 2000 --calib-seed 0 --max-model-len 8192 \
      --work-dir "$WORK/$2" --out "$WORK/${2}_r1.txt" --image-root "$IMAGES"
  python scripts/select_round2.py --in-dir "$1" --vlm-hf "$VLM" --tp "$TP" \
      --calib-n 2000 --calib-seed 0 --max-model-len 8192 \
      --r1-cache "$WORK/$2/r1.jsonl" --pred-out "$WORK/${2}_preds.jsonl" \
      --work-dir "$WORK/$2" --out "$WORK/${2}_r2.txt" --image-root "$IMAGES"
}

echo "== base scene graph =="
python scripts/build_sg.py --sample "$SAMPLES" --out-dir "$D/base" \
    --shard 0 --nshards 1 --vlm-hf "$VLM" --image-root "$IMAGES"

echo "== [none] no verification: relations/global on base, then select =="
python scripts/sg_relate_global.py --in-dir "$D/base" --out-dir "$D/sg_none" \
    --work-dir "$WORK/none_rg" --vlm-hf "$VLM" --image-root "$IMAGES"
sel "$D/sg_none" none

echo "== [object] object-level verification (F + A), then relations/global, then select =="
python scripts/gen_verify_conf.py --in-dir "$D/base" --verifier-hf "$VLM" \
    --out "$WORK/vc.jsonl" --image-root "$IMAGES"
python scripts/merge_verify_conf.py "$D/base" "$WORK/vc.jsonl" "$D/vc"
python scripts/anchor_verify.py --in-dir "$D/vc" --out "$WORK/anchor.jsonl" \
    --work-dir "$WORK/anchor" --thr 0.3 --vlm-hf "$VLM" --image-root "$IMAGES"
python scripts/object_filter.py --in-dir "$D/vc" --out-dir "$D/obj" \
    --work-dir "$WORK/f" --thr 0.2 --rescue vlm --rescue-cap 3 \
    --anchor-thr 0.5 --anchor-file "$WORK/anchor.jsonl" --filter-only
python scripts/refl4_expand_rel.py --in-dir "$D/obj" --out-cands "$WORK/cands.jsonl" \
    --out-rel "$WORK/rel.jsonl" --work-dir "$WORK/exp" --image-root "$IMAGES"
python scripts/surgical_augment.py --in-dir "$D/obj" --out-dir "$D/fa" \
    --cands "$WORK/cands.jsonl" --rel "$WORK/rel.jsonl" --vlm-hf "$VLM" --image-root "$IMAGES"
python scripts/sg_relate_global.py --in-dir "$D/fa" --out-dir "$D/sg_object" \
    --work-dir "$WORK/obj_rg" --vlm-hf "$VLM" --image-root "$IMAGES"
sel "$D/sg_object" object

echo "== [caption] caption-level verification (spatial + jury), graft rel/global from [none], select =="
python scripts/spatial_inject.py "$D/base" "$D/base_S"
python scripts/listener_jury.py --phase gen   --in-dir "$D/base_S" \
    --work-dir "$WORK/cap_jury" --vlm-hf "$VLM" --image-root "$IMAGES"
python scripts/listener_jury.py --phase score --in-dir "$D/base_S" \
    --work-dir "$WORK/cap_jury" --vlm-hf "$LISTENER_VLM" --image-root "$IMAGES"
python scripts/listener_jury.py --phase assemble --in-dir "$D/base_S" \
    --work-dir "$WORK/cap_jury" --out-dir "$D/base_cap"
python scripts/graft_ablation.py "$D/base_cap" "$D/sg_none" "$D/sg_caption"
sel "$D/sg_caption" caption

echo "== [both] full pipeline (F + A -> spatial + jury -> relations/global), then select =="
python scripts/spatial_inject.py "$D/fa" "$D/fa_S"
python scripts/listener_jury.py --phase gen   --in-dir "$D/fa_S" \
    --work-dir "$WORK/both_jury" --vlm-hf "$VLM" --image-root "$IMAGES"
python scripts/listener_jury.py --phase score --in-dir "$D/fa_S" \
    --work-dir "$WORK/both_jury" --vlm-hf "$LISTENER_VLM" --image-root "$IMAGES"
python scripts/listener_jury.py --phase assemble --in-dir "$D/fa_S" \
    --work-dir "$WORK/both_jury" --out-dir "$D/fa_cap"
python scripts/sg_relate_global.py --in-dir "$D/fa_cap" --out-dir "$D/sg_both" \
    --work-dir "$WORK/both_rg" --vlm-hf "$VLM" --image-root "$IMAGES"
sel "$D/sg_both" both

echo ""
echo "===================== Table 3: verification ablation ====================="
for L in none object caption both; do
  printf "  %-8s : " "$L"
  python eval/metrics.py --preds "$WORK/${L}_preds.jsonl" | grep -E "Acc@0.5|mAcc" | tr '\n' '  '
  echo ""
done
