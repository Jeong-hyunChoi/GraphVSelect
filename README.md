# GraphVSelect

Official code for **"GraphVSelect: Verified Graph-guided Selection for Zero-shot Referring
Expression Comprehension"**.

[Jeonghyun Choi](https://github.com/Jeong-hyunChoi)<sup>1\*</sup>,
[Yao Wei](https://weiyao1996.github.io/)<sup>2\*</sup>,
Hyungcheol Noh<sup>1</sup>,
[Andrea Cavallaro](https://people.epfl.ch/andrea.cavallaro)<sup>3</sup>,
[Changjae Oh](https://cj-oh.github.io/)<sup>2&dagger;</sup>
<sup>1</sup>AI Research Team, Innodep Inc. &middot; <sup>2</sup>Queen Mary University of London &middot; <sup>3</sup>EPFL
<sup>\*</sup>Equal contribution &middot; <sup>&dagger;</sup>Corresponding author

[**Project page**](https://jeong-hyunchoi.github.io/GraphVSelect/) &middot; Paper (coming soon) &middot; arXiv (coming soon)

The method is **training-free**: every component is a frozen, off-the-shelf model used for
inference only — no optimizer, learning rate, or checkpoint. This repository contains the full
method (hierarchically verified scene-graph construction + confidence-aware structured
selection), the box-wise baseline it is compared against, the evaluation scripts, and a
**100-record sample with its images**, so the paper's **Ref-L4** results can be reproduced
end-to-end from raw images.

| You can reproduce | What it shows | One command |
| --- | --- | --- |
| **Table 2** | main Ref-L4 results (Acc@0.5/0.75/0.9, mAcc) across VLM backbones | `run_pipeline.sh` → `run_selection.sh` |
| **Table 3** | verification ablation (none / object / caption / both) | `run_ablation.sh` |
| **Table 4** | list-wise **selection** vs. **box-wise** verification, same candidates | `run_selection.sh` vs. `run_boxwise.sh` |

---

## Environment

Everything runs inside one Docker image.

```bash
docker build -t graphvselect:latest -f docker/Dockerfile .

docker run --gpus all -it \
  -v "$PWD":/workspace/GraphVSelect \
  -e PYTHONPATH=/workspace/GraphVSelect \
  -v /path/to/datasets:/data \                # mount already-downloaded datasets (Ref-L4, COCO, ...)
  -v /path/to/hf_cache:/hf -e HF_HOME=/hf \   # optional: persist model downloads across containers
  graphvselect:latest bash
```

The bundled sample needs no mounts. Add the two `-v` lines only when you go beyond it:

- **`-v /path/to/datasets:/data`** — mount a host directory that already holds Ref-L4 / COCO /
  Objects365 images so the pipeline can read them (the **Full Ref-L4 test** below assumes the
  data is visible at `/data`). Skip it if you have not downloaded any dataset yet.
- **`-v /path/to/hf_cache:/hf -e HF_HOME=/hf`** — **optional**; points HuggingFace's cache at a
  host directory so model weights download only once. Drop it and everything still works; the
  models are just re-downloaded on each fresh container.

Tested on **2× NVIDIA H100 80GB**: Python 3.11, PyTorch 2.8.0 (CUDA 12.8), vLLM 0.11.0,
transformers 4.57.x. A `requirements.txt` is provided for non-Docker installs
(run `python -m spacy download en_core_web_sm` afterwards).

## Models (downloaded automatically)

There is nothing to download by hand — every model is a frozen, off-the-shelf checkpoint that
**vLLM / transformers fetch from HuggingFace on first use** and cache under `HF_HOME`:

| model | HuggingFace id | role |
| --- | --- | --- |
| GroundingDINO | `IDEA-Research/grounding-dino-base` | open-vocabulary detection |
| VLM backbone | `Qwen/Qwen3-VL-8B-Instruct` (default) | scene-graph construction, verification, selection |
| listener juror | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | caption-level verification |

Budget ~10–20 min of download on the very first run; afterwards it is cache-only. Some
backbones are license-gated on HuggingFace — accept the license and `huggingface-cli login`
once if a download 401s. Swap the backbone with `VLM=<hf-id>` (see **Backbones**).

## Quickstart — the bundled sample

The package ships a **100-record Ref-L4 sample with its 100 images** (`sample/`), so the full
pipeline runs from raw images with no download and no configuration:

```bash
bash run_pipeline.sh      # build the verified scene graph from images, then select
```

Six stages print in order (`build_sg → F → A → jury → relate → select`) and the run ends with
`Acc@0.5/0.75/0.9, mAcc`. You can then score the other tables on the graph it built
(`data/verified_sg`):

```bash
bash run_selection.sh     # list-wise selection A+B3      (Table 2 / Table 4 selection side)
bash run_boxwise.sh       # box-wise verification baseline (Table 4 box-wise side)
bash run_ablation.sh      # none/object/caption/both       (Table 3)
```

> The sample is a **100-record subset**: its scores confirm the pipeline runs and reproduces
> bit-for-bit, but they are **not** the paper's numbers. The reported Table 2 results come from
> the **full Ref-L4 test set** below.

## Full Ref-L4 test (reproduce the paper's numbers)

**1. Get the data.** Ref-L4 is public (Chen et al., *Revisiting REC Evaluation in the Era of
Large Multimodal Models*). Download the annotation parquet and its image folder from the
official HuggingFace dataset repo into one directory, e.g.:

```bash
huggingface-cli download JierunChen/Ref-L4 --repo-type dataset --local-dir /data/Ref-L4
# /data/Ref-L4/ref-l4-test.parquet  +  the COCO train2014 / Objects365 images it references
```

If you already have Ref-L4 on the host, skip this step — just mount that directory
(`-v /path/to/Ref-L4:/data/Ref-L4`, see **Environment**) and point the paths below at it.

**2. Convert it to the pipeline's record format.** `scripts/prepare_refl4.py` reads the parquet
and writes a `samples.jsonl` (one record per referring expression: `query`, `gt_bbox`,
`image_path`). A few hundred very large Objects365 images are resized to a 1024×1024 area cap
with their boxes scaled — the paper's protocol, needed because vLLM 0.11 does not honor
per-request pixel caps — so the run never overflows the model context:

```bash
python scripts/prepare_refl4.py \
    --parquet /data/Ref-L4/ref-l4-test.parquet \
    --images  /data/Ref-L4 \
    --out     data/refl4_test.jsonl
```

**3. Run the full pipeline on it.** Records embed an absolute `image_path`, so only `SAMPLES`
changes — you do not even pass an image directory. This builds the scene graph over all ~32k
records and selects; it takes **hours** on 2×H100 and is fully resumable:

```bash
SAMPLES=data/refl4_test.jsonl bash run_pipeline.sh      # -> Table 2 (Qwen Ref-L4: Acc@0.5 ≈ 71.5, mAcc ≈ 60.3)
SAMPLES=data/refl4_test.jsonl bash run_boxwise.sh       # -> Table 4 box-wise side, on the same graph
SAMPLES=data/refl4_test.jsonl bash run_ablation.sh      # -> Table 3, on the same records
```

(Use `ref-l4-val.parquet` for the validation split.)

## RefCOCO / RefCOCO+ / RefCOCOg

The method also runs on the classic RefCOCO-series benchmarks, following the standard
**lmms-eval** protocol: one referring expression per annotation, evaluated against the
COCO *train2014* image. `scripts/prepare_refcoco.py` converts the public lmms-lab mirror into
the same `samples.jsonl` schema.

**1. Get the data.** The annotations come from the lmms-lab HuggingFace mirrors; the images are
standard COCO *train2014* (obtain them from the official COCO site if you do not already have
them):

```bash
huggingface-cli download lmms-lab/RefCOCO     --repo-type dataset --local-dir /data/RefCOCO
huggingface-cli download lmms-lab/RefCOCOplus --repo-type dataset --local-dir /data/RefCOCOplus
huggingface-cli download lmms-lab/RefCOCOg    --repo-type dataset --local-dir /data/RefCOCOg
# images: /data/coco2014/train2014/COCO_train2014_*.jpg
```

**2. Convert the split(s) you want.** RefCOCO / RefCOCO+ have `val` / `testA` / `testB`;
RefCOCOg has `val` / `test`:

```bash
python scripts/prepare_refcoco.py --dataset lmms-lab/RefCOCO --split val \
    --parquet-dir /data/RefCOCO/data --images /data/coco2014/train2014 \
    --out data/refcoco_val.jsonl
```

**3. Run.** These records carry an absolute `image_path` (from `--images`), so `SAMPLES` is all
you need:

```bash
SAMPLES=data/refcoco_val.jsonl bash run_pipeline.sh     # build SG + list-wise selection A+B3
SAMPLES=data/refcoco_val.jsonl bash run_boxwise.sh      # box-wise baseline on the same graph
```

Repeat step 2 with `--dataset lmms-lab/RefCOCOplus` / `lmms-lab/RefCOCOg` and the other splits
to cover the whole series. Each `samples.jsonl` is independent — score them separately, or
concatenate the files to evaluate several splits in one run.

### Backbones (Table 2 generality)

```bash
VLM=llava-hf/llava-onevision-qwen2-7b-ov-hf  bash run_selection.sh   # LLaVA-OneVision
VLM=allenai/Molmo-7B-D-0924                   bash run_selection.sh   # Molmo (see note)
```

Molmo has a tight context window: build a compact graph with
`python scripts/molmo_compact.py <sg_dir> <out_dir>` and add `--compact-sg` to the selection /
verification scripts.

### Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `SAMPLES` | `sample/samples.jsonl` | input records (`query`, `gt_bbox`, `image_path`/`file_name`) for `run_pipeline` |
| `IMAGES` | `sample/images` | image directory for records that carry a bare `file_name` (unused when records embed `image_path`, as `prepare_refl4.py` writes) |
| `SG` | `data/verified_sg` | verified-graph dir built by `run_pipeline`, consumed by `run_selection`/`run_boxwise` |
| `VLM` | `Qwen/Qwen3-VL-8B-Instruct` | frozen backbone |
| `LISTENER_VLM` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | caption-level listener juror |
| `TP` | `1` | tensor-parallel GPU count |

---

## Repository layout → paper

| Path | Role | Paper |
| --- | --- | --- |
| `run_pipeline.sh`, `run_selection.sh`, `run_boxwise.sh`, `run_ablation.sh` | one-command entry points (Quickstart) | — |
| `graphvselect/detect.py`, `nouns.py` | GroundingDINO open-vocab detection; query → detection vocabulary | §Scene-Graph Construction |
| `graphvselect/vlm.py` | frozen VLM wrapper (vLLM); SG-construction + Set-of-Mark verification calls | Method |
| `graphvselect/prompts.py` | all VLM prompt templates | Method |
| `graphvselect/spatial.py`, `geometry.py` | relation gate, IoU, box math | §Scene-Graph Construction |
| `graphvselect/parsing.py`, `serialize.py`, `render.py` | output parsing, SG→text, image marking | Method |
| `scripts/prepare_refl4.py` | official Ref-L4 parquet + image folder → run-ready `samples.jsonl` (resizes oversized images) | §Experiments (data prep) |
| `scripts/prepare_refcoco.py` | RefCOCO/+/g (lmms-lab mirror) → run-ready `samples.jsonl` (lmms-eval protocol) | §Experiments (data prep) |
| `scripts/build_sg.py` | detect → per-crop captions + pairwise relations + global caption → **base** SG dump | §Scene-Graph Construction |
| **Object-level verification (F + A):** | | §Object-level verification |
| `scripts/gen_verify_conf.py` | cross-model per-object confidence (Set-of-Mark Yes/No log-probs) | §Object-level verification |
| `scripts/merge_verify_conf.py` | write those confidences onto the dump (`verify_conf`) | §Object-level verification |
| `scripts/anchor_verify.py` | anchor-relevance scores that rescue useful low-confidence objects | §Object-level verification |
| `scripts/object_filter.py` | keep objects by confidence + anchor rescue (**F** filter) | §Object-level verification |
| `scripts/refl4_expand_rel.py` | expanded re-detection + relevance (the `--cands`/`--rel` producer for A) | §Object-level verification |
| `scripts/surgical_augment.py` | add back missed referents (**A** augmentation) | §Object-level verification |
| **Caption-level verification:** | | §Caption-level verification |
| `scripts/spatial_inject.py` | attach spatial-rank descriptors to same-category objects | §Caption-level verification |
| `scripts/listener_jury.py` | sample M captions, keep the most discriminative by listener agreement | §Caption-level verification |
| `scripts/sg_relate_global.py` | regenerate query-relevant relations + global caption over the final object set | §Scene-Graph Construction |
| **Selection & baselines:** | | |
| `scripts/select_round1.py` | **list-wise** constrained-logprob scoring + confidence gate (A) | §Confidence-Aware Structured Selection |
| `scripts/select_round2.py` | order-symmetrized top-2 re-answer, final prediction (B3) | §Confidence gate and re-answering |
| `scripts/run_verify.py` | **box-wise** Yes/No verification over the SG | Table 4 |
| `scripts/graft_ablation.py` | graft relations/global across variants so an ablation isolates one stage | Table 3 |
| `scripts/molmo_compact.py` | compact captions for Molmo's tight context | Tables 2–4 (Molmo) |
| `eval/metrics.py` | score predictions → Acc@0.5/0.75/0.9, **mAcc** (0.50:0.05:0.95) | Tables 2, 3 |
| `eval/box_wise_eval.py` | score box-wise verification | Table 4 |

The code preserves the exact logic used for the reported numbers; only file organization,
comments, and paths were cleaned for release.

---

## Building the verified scene graph (what `run_pipeline.sh` does)

`run_pipeline.sh` runs the six stages below in order over a `samples.jsonl` where each record
carries `query`, `gt_bbox`, and an image reference (`image_path` or `file_name`; see the **Data & image resolution** section).
To run a stage by hand (all VLM stages share `--vlm-hf`, resolve images with `--image-root`,
and default their hyper-parameters to the paper's values):

```bash
IMG=/path/to/images                 # COCO train2014 + Objects365 (see Data)
Q=Qwen/Qwen3-VL-8B-Instruct

# 1) base SG: detect -> per-crop captions + pairwise relations + global caption
python scripts/build_sg.py --sample data/samples.jsonl --out-dir data/base_sg \
    --shard 0 --nshards 1 --image-root $IMG

# 2) OBJECT-LEVEL VERIFICATION (F): cross-model confidence -> filter + anchor rescue
python scripts/gen_verify_conf.py --in-dir data/base_sg --verifier-hf $Q \
    --out outputs/vc.jsonl --image-root $IMG
python scripts/merge_verify_conf.py data/base_sg outputs/vc.jsonl data/vc_sg
python scripts/anchor_verify.py --in-dir data/vc_sg --out outputs/anchor.jsonl \
    --work-dir outputs/anchor_work --thr 0.3 --image-root $IMG
python scripts/object_filter.py --in-dir data/vc_sg --out-dir data/obj_sg \
    --work-dir outputs/f_work --thr 0.2 --rescue vlm --rescue-cap 3 \
    --anchor-thr 0.5 --anchor-file outputs/anchor.jsonl --filter-only

# 3) OBJECT-LEVEL AUGMENTATION (A): expanded re-detect + relevance -> add back referents
python scripts/refl4_expand_rel.py --in-dir data/obj_sg \
    --out-cands outputs/cands.jsonl --out-rel outputs/rel.jsonl \
    --work-dir outputs/exp_work --image-root $IMG
python scripts/surgical_augment.py --in-dir data/obj_sg --out-dir data/fa_sg \
    --cands outputs/cands.jsonl --rel outputs/rel.jsonl --image-root $IMG

# 4) CAPTION-LEVEL VERIFICATION: spatial-rank prep -> listener jury (gen/score/assemble)
python scripts/spatial_inject.py data/fa_sg data/fas_sg
python scripts/listener_jury.py --phase gen   --in-dir data/fas_sg --work-dir outputs/jury \
    --vlm-hf $Q --image-root $IMG
python scripts/listener_jury.py --phase score --in-dir data/fas_sg --work-dir outputs/jury \
    --vlm-hf llava-hf/llava-onevision-qwen2-7b-ov-hf --image-root $IMG
python scripts/listener_jury.py --phase assemble --in-dir data/fas_sg --work-dir outputs/jury \
    --out-dir data/cap_sg

# 5) regenerate query-relevant relations + global caption over the FINAL object set
python scripts/sg_relate_global.py --in-dir data/cap_sg --out-dir data/verified_sg \
    --work-dir outputs/rg_work --vlm-hf $Q --image-root $IMG
```

`data/verified_sg` is the verified scene graph consumed by selection.

---

## Reproducing the tables by hand

The run scripts wrap these exact commands. All VLM calls use greedy decoding, so each run is
deterministic up to vLLM's floating-point non-determinism; every stage is resumable.

### Table 2 — main results (list-wise selection A+B3) · `run_selection.sh`

```bash
python scripts/select_round1.py --in-dir <SG_DIR> --vlm-hf $Q --tp 1 \
    --gate-frac 0.30 --calib-n 2000 --calib-seed 0 --max-model-len 8192 \
    --work-dir outputs/select --out outputs/select_r1.txt --image-root $IMG
python scripts/select_round2.py --in-dir <SG_DIR> --vlm-hf $Q --tp 1 \
    --calib-n 2000 --calib-seed 0 --max-model-len 8192 \
    --r1-cache outputs/select/r1.jsonl --pred-out outputs/preds.jsonl \
    --work-dir outputs/select --out outputs/select_r2.txt --image-root $IMG
python eval/metrics.py --preds outputs/preds.jsonl        # Acc@0.5/0.75/0.9, mAcc
```

### Table 4 — selection vs. box-wise verification (same candidates) · `run_boxwise.sh`

```bash
python scripts/run_verify.py --in-dir <SG_DIR> --vlm-hf $Q --tp 1 \
    --out outputs/verify_conf.jsonl --work-dir outputs/verify_work --image-root $IMG
python eval/box_wise_eval.py --in-dir <SG_DIR> \
    --conf outputs/verify_conf.jsonl --out outputs/box_wise.txt
```

The list-wise side is `run_selection.sh`. Add `--no-sg` to `run_verify.py` for the box-wise
**no-SG** ablation.

### Table 3 — verification ablation (none / object / caption / both) · `run_ablation.sh`

`run_ablation.sh` builds one base scene graph, derives the four variants by toggling
object-level (F+A) and caption-level (spatial+jury) verification, scores A+B3 selection on
each, and prints the comparison. Each level is the pipeline with a stage switched off:

| level | object set / captions |
| --- | --- |
| none | base SG (skip F, A, and the jury) |
| object | base + **F** (`gen_verify_conf`→`merge_verify_conf`→`anchor_verify`→`object_filter`) + **A** (`surgical_augment`) |
| caption | base + `spatial_inject` + `listener_jury` |
| both | the full pipeline |

Because relation/global regeneration (`sg_relate_global`) depends on the object set,
`scripts/graft_ablation.py` transplants the same relations/global across levels so the
comparison isolates the verification stage rather than confounding it with relation
regeneration.

---

## Data & image resolution

All experiments use **public** datasets — **Ref-L4** (COCO *train2014* + Objects365 images;
Chen et al., *Revisiting REC Evaluation…*) and RefCOCO/RefCOCO+/RefCOCOg (COCO *train2014*).
No new dataset is introduced; only the 100 images for the bundled sample are shipped (under
`sample/images/`). For full runs, obtain the data from its official public source and convert
it with `scripts/prepare_refl4.py` (**Full Ref-L4 test**) or `scripts/prepare_refcoco.py`
(**RefCOCO / RefCOCO+ / RefCOCOg**) above.

A **record** is self-contained — it carries `query`, `gt_bbox` (`[x1,y1,x2,y2]`), and an image
reference: an on-disk `image_path` (what `prepare_refl4.py` writes) or a bare `file_name` (what
the bundled sample uses, resolved under `--image-root`/`IMAGES`). `graphvselect/images.py`
opens local files only — there is **no network or dataset-hub dependency**. To run on your own
data, write records in this schema and point `IMAGES` at your image directory.

> **Objects365 note.** Ref-L4 mixes in large Objects365 images, and vLLM 0.11.0 does not honor
> per-request pixel caps for some backbones, so a very large image can overflow the model
> context. `prepare_refl4.py` handles this for you: any image above a 1024×1024 area cap is
> resized and its boxes scaled (the paper's protocol) before the graph is built.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| First run stalls on "downloading" | Frozen models are being fetched (~10–20 min once). Mount an `HF_HOME` cache so it happens only once. Some backbones are gated on HuggingFace — accept their license and `huggingface-cli login`. |
| CUDA out of memory | Lower `--max-model-len` (default 8192), reduce `gpu_memory_utilization` in the VLM call, or set `TP=2` to shard across two GPUs. |
| A run was interrupted | Just re-run the same command — every stage checkpoints to its `--work-dir` and resumes. Nothing is recomputed. |
| `record has no image_path or file_name` | Your `samples.jsonl` records must carry a `query`, `gt_bbox`, and an image reference; point `--image-root`/`IMAGES` at the directory holding those images. |
| Sample scores don't match the paper | Expected — the bundle is a 100-record subset. Full Table 2 numbers need the full Ref-L4 test set (see **Full Ref-L4 test**). |
| Molmo overflows context | Build a compact graph with `scripts/molmo_compact.py` and pass `--compact-sg` (see Backbones). |

---

## Contents

- [x] One-command run scripts: `run_pipeline.sh` (build+select), `run_selection.sh`, `run_boxwise.sh`, `run_ablation.sh`
- [x] Full pipeline: detection → object-level verification (F filter + A augmentation) →
      caption-level verification (jury) → relation/global regeneration → list-wise selection + gate
- [x] Box-wise verification baseline (Table 4) + verification-ablation plumbing (Table 3)
- [x] Evaluation: Acc@0.5/0.75/0.9, mAcc (0.50:0.05:0.95); box-wise scorer
- [x] Data-prep scripts for the full benchmarks: `prepare_refl4.py` (Ref-L4) and `prepare_refcoco.py` (RefCOCO / RefCOCO+ / RefCOCOg)
- [x] Docker environment + requirements; bundled 100-record Ref-L4 sample (raw records **+ images**) that builds & runs out-of-box
- [x] Apache-2.0 license

---

## Citation

```bibtex
@inproceedings{choi2027graphvselect,
  title     = {GraphVSelect: Verified Graph-guided Selection for Zero-shot Referring Expression Comprehension},
  author    = {Choi, Jeonghyun and Wei, Yao and Noh, Hyungcheol and Cavallaro, Andrea and Oh, Changjae},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2027}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE). The project page
(`webpage` branch) is licensed under CC BY-SA 4.0.
