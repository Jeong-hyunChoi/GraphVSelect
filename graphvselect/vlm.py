"""Frozen vision-language model wrapper (vLLM backend).

Thin interface used by every stage: it loads the frozen VLM once, applies the chat
template, and exposes the raw ``llm.generate`` handle plus two higher-level calls used
during scene-graph construction (query -> relevant categories, and batched per-crop
captioning + pairwise relations). All calls are inference-only (temperature 0).
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from transformers import AutoProcessor

from .geometry import expand_bbox as _expanded_bbox, pair_overlaps
from .render import (annotate_image as _annotate_image,
                     annotate_image_pair as _annotate_image_pair, BBOX_COLORS)
from .parsing import parse_pair_predicate as _parse_pair_predicate
from .prompts import (
    CROP_CAPTION_PROMPT, CROP_CAPTION_PROMPT_BFAIR,
    CROP_CAPTION_PROMPT_BFAIR_EXPLICIT, PAIRWISE_RELATION_PROMPT,
    PAIRWISE_RELATION_PROMPT_QUERYCOND, GLOBAL_CAPTION_PROMPT_AGNOSTIC,
    GLOBAL_CAPTION_PROMPT_QUERYCOND,
)
from .spatial import should_generate_relation


def _pair_should_relate(c1, c2, margin=0.2):
    """Adapter: keep the dict-based call site; delegate to geometry.pair_overlaps."""
    return pair_overlaps(c1["bbox"], c2["bbox"], margin=margin)


class VLM:
    """Frozen VLM served through vLLM (loaded once, used for all query types)."""

    def __init__(self, hf_id: str, dtype: str = "float16",
                 tensor_parallel_size: int = 2,
                 gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 4096,
                 max_num_seqs: int = 16,
                 max_pixels: int = 1024 * 1024,
                 limit_mm_images: int = 1):
        from vllm import LLM, SamplingParams
        logger.info("Loading VLM (vllm) {} dtype={} TP={}", hf_id, dtype, tensor_parallel_size)
        # `max_pixels` is a Qwen-VL-specific processor kwarg; other VLMs (e.g.
        # LLaVA-OneVision) reject it. Only pass it for Qwen-VL models.
        llm_kwargs = dict(
            model=hf_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            limit_mm_per_prompt={"image": limit_mm_images, "video": 0},  # video:0 avoids InternVL profiler crash
            trust_remote_code=True,   # InternVL / Molmo ship custom model code
        )
        if "qwen" in hf_id.lower():
            llm_kwargs["mm_processor_kwargs"] = {"max_pixels": max_pixels}
        self.llm = LLM(**llm_kwargs)
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self._SamplingParams = SamplingParams
        logger.info("VLM (vllm) ready")

    def _build_prompt(self, text: str) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def __call__(self, image: Image.Image, prompt: str,
                 max_new_tokens: int = 256) -> str:
        sp = self._SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
        text_prompt = self._build_prompt(prompt)
        out = self.llm.generate(
            {"prompt": text_prompt, "multi_modal_data": {"image": image}}, sp,
        )[0].outputs[0].text
        return out

    def verify_regions_som(
        self, image: Image.Image, query: str, candidates: list[dict],
        min_crop_pixels: int = 16, max_verify_tokens: int = 4,
        logprobs_topk: int = 20,
    ) -> list[dict]:
        """Set-of-Mark verification (Yang et al. 2023): annotate ALL kept candidates
        on a single full image (numbered, coloured boxes), then ask the VLM Yes/No
        about each region by its id. The VLM sees the full scene context (vs a tight
        per-object crop), which is the leverage for spatial queries ("man on the
        right", "second from left"). Returns per-object Yes-vs-No first-token
        log-probabilities; used to compute an independent object-level confidence."""
        kept = [c for c in candidates
                if (c["bbox"][2] - c["bbox"][0]) >= min_crop_pixels
                and (c["bbox"][3] - c["bbox"][1]) >= min_crop_pixels]
        if not kept:
            return []

        # One annotated image with ALL kept regions marked by their actual id (not
        # enumerate index) so the prompt's region [N] matches the drawn label.
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=20)
        except OSError:
            font = ImageFont.load_default()
        for i, c in enumerate(kept):
            x1, y1, x2, y2 = c["bbox"]
            color = BBOX_COLORS[i % len(BBOX_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"[{c.get('id', i)}]"
            tb = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((x1, y1), label, fill="white", font=font)

        requests = []
        for c in kept:
            cid = c.get("id")
            prompt = (
                f"In this image, several regions are marked with coloured "
                f"bboxes labelled like [0], [1], [2], ... . Look at region "
                f"[{cid}]. Is the object inside region [{cid}] the one "
                f"described by the phrase: \"{query}\"? Answer with only "
                "'Yes' or 'No'."
            )
            text = self._build_prompt(prompt)
            requests.append({"prompt": text, "multi_modal_data": {"image": annotated}})

        sp = self._SamplingParams(
            max_tokens=max_verify_tokens, temperature=0.0, top_p=1.0,
            logprobs=logprobs_topk,
        )
        outputs = self.llm.generate(requests, sp)

        scored: list[dict] = []
        tokenizer = self.processor.tokenizer
        for c, out in zip(kept, outputs):
            first = out.outputs[0]
            text_out = first.text.strip()
            yes_lp = float("-inf"); no_lp = float("-inf")
            top_logprobs = first.logprobs[0] if first.logprobs else {}
            for tok_id, lp_obj in top_logprobs.items():
                tok = tokenizer.decode([tok_id]).strip().lower()
                if tok in {"yes", "yes.", "yes,"}:
                    yes_lp = max(yes_lp, lp_obj.logprob)
                elif tok in {"no", "no.", "no,"}:
                    no_lp = max(no_lp, lp_obj.logprob)
            if yes_lp == float("-inf") and no_lp == float("-inf"):
                t = text_out.lower()
                if t.startswith("yes"): yes_lp, no_lp = 0.0, -10.0
                elif t.startswith("no"): yes_lp, no_lp = -10.0, 0.0
                else: yes_lp, no_lp = 0.0, 0.0
            elif yes_lp == float("-inf"): yes_lp = no_lp - 10.0
            elif no_lp == float("-inf"): no_lp = yes_lp - 10.0
            scored.append({
                "id": c.get("id"),
                "yes_lp": float(yes_lp),
                "no_lp":  float(no_lp),
                "score":  float(yes_lp - no_lp),
                "raw":    text_out,
            })
        return scored

    def _sg_pw_build(self, image, query, candidates, *,
                     max_caption_tokens=80, max_pair_relation_tokens=64,
                     skip_relations=False, min_crop_pixels=16,
                     pair_expand_margin=0.2, max_pairs=30,
                     query_conditional_relations=False, include_global_caption=False,
                     query_conditional_global=True, max_global_caption_tokens=200,
                     caption_context_aware=False, caption_pad_ratio=0.3,
                     nearby_hints_by_id=None):
        crop_requests = []; valid = []; empty_objects = []
        for c in candidates:
            cid = c.get("id", len(empty_objects) + len(valid))
            x1, y1, x2, y2 = c["bbox"]
            if (x2 - x1) < min_crop_pixels or (y2 - y1) < min_crop_pixels:
                empty_objects.append({"id": cid, "bbox": c["bbox"], "category": c["label"],
                                      "caption": "", "kind": c.get("kind", "main"),
                                      **({"spatial_hint": c["spatial_hint"]}
                                         if c.get("spatial_hint") else {})})
                continue
            cid_for_hint = c.get("id")
            if nearby_hints_by_id and cid_for_hint in nearby_hints_by_id:
                crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
                text = self._build_prompt(CROP_CAPTION_PROMPT_BFAIR_EXPLICIT.format(
                    category=c["label"], nearby_hints=nearby_hints_by_id[cid_for_hint]))
            elif caption_context_aware:
                ex1, ey1, ex2, ey2 = _expanded_bbox(
                    c["bbox"], image.width, image.height, caption_pad_ratio)
                crop = image.crop((ex1, ey1, ex2, ey2))
                text = self._build_prompt(CROP_CAPTION_PROMPT_BFAIR.format(category=c["label"]))
            else:
                crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
                text = self._build_prompt(CROP_CAPTION_PROMPT.format(category=c["label"]))
            crop_requests.append({"prompt": text, "multi_modal_data": {"image": crop}})
            valid.append(c)

        pair_requests = []; pair_meta = []
        if not skip_relations and len(candidates) >= 2:
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    c_i, c_j = candidates[i], candidates[j]
                    k_i, k_j = c_i.get("kind"), c_j.get("kind")
                    if k_i is not None and k_j is not None:
                        if not should_generate_relation(k_i, k_j):
                            continue
                    if not _pair_should_relate(c_i, c_j, margin=pair_expand_margin):
                        continue
                    pair_meta.append((c_i["id"], c_j["id"]))
            if max_pairs is not None and len(pair_meta) > max_pairs:
                pair_meta = pair_meta[:max_pairs]
            id_to_cand = {c["id"]: c for c in candidates}
            tpl = (PAIRWISE_RELATION_PROMPT_QUERYCOND if query_conditional_relations
                   else PAIRWISE_RELATION_PROMPT)
            for (id1, id2) in pair_meta:
                ann_pair = _annotate_image_pair(image, id_to_cand[id1], id_to_cand[id2])
                if query_conditional_relations:
                    text = self._build_prompt(tpl.format(id1=id1, id2=id2, query=query))
                else:
                    text = self._build_prompt(tpl.format(id1=id1, id2=id2))
                pair_requests.append({"prompt": text, "multi_modal_data": {"image": ann_pair}})

        global_request = None
        if include_global_caption and candidates:
            annotated = _annotate_image(image, candidates)
            if query_conditional_global:
                gcap_prompt = GLOBAL_CAPTION_PROMPT_QUERYCOND.format(query=query)
            else:
                gcap_prompt = GLOBAL_CAPTION_PROMPT_AGNOSTIC
            text = self._build_prompt(gcap_prompt)
            global_request = {"prompt": text, "multi_modal_data": {"image": annotated}}

        sps = ([self._SamplingParams(max_tokens=max_caption_tokens, temperature=0.0, top_p=1.0)
                for _ in crop_requests]
               + [self._SamplingParams(max_tokens=max_pair_relation_tokens, temperature=0.0, top_p=1.0)
                  for _ in pair_requests])
        if global_request:
            sps.append(self._SamplingParams(max_tokens=max_global_caption_tokens,
                                             temperature=0.0, top_p=1.0))
        reqs = crop_requests + pair_requests + ([global_request] if global_request else [])
        return {"reqs": reqs, "sps": sps, "valid": valid, "empty_objects": empty_objects,
                "pair_meta": pair_meta, "cap_n": len(crop_requests),
                "pair_n": len(pair_requests), "has_global": global_request is not None}

    def _sg_pw_assemble(self, b, raw_outputs):
        cap_n, pair_n = b["cap_n"], b["pair_n"]
        caption_outs = raw_outputs[:cap_n]
        pair_outs = raw_outputs[cap_n:cap_n + pair_n]
        global_caption = raw_outputs[cap_n + pair_n].strip() if b["has_global"] else ""
        objects = list(b["empty_objects"]); raw_chunks = []
        for c, raw_cap in zip(b["valid"], caption_outs):
            caption = raw_cap.strip().split("\n")[0].strip()
            raw_chunks.append(f"[cap {c.get('id', '?')}] {raw_cap}")
            objects.append({"id": c.get("id"), "bbox": c["bbox"], "category": c["label"],
                            "caption": caption, "kind": c.get("kind", "main"),
                            **({"spatial_hint": c["spatial_hint"]}
                               if c.get("spatial_hint") else {})})
        relations = []
        for (id1, id2), raw_rel in zip(b["pair_meta"], pair_outs):
            raw_chunks.append(f"[rel {id1}<->{id2}] {raw_rel}")
            pred = _parse_pair_predicate(raw_rel)
            if pred and pred.lower() != "none":
                relations.append({"subject": id1, "predicate": pred, "object": id2})
        sg = {"objects": objects, "relations": relations}
        if global_caption:
            sg["global_caption"] = global_caption
            raw_chunks.append(f"[global] {global_caption}")
        return sg, "\n\n".join(raw_chunks)

    def generate_sg_per_crop_pairwise_batch(self, items, **kw):
        """Batched over records. `items` = list of dicts with keys image, query,
        candidates, and optional nearby_hints_by_id. All other kwargs (e.g.
        query_conditional_relations, include_global_caption, max_caption_tokens)
        are passed through **kw and apply to every item. Returns a list of
        (sg, raw) tuples aligned with `items`. Equivalent to building each item's
        requests individually, but submitted in one vLLM batch (temperature=0)."""
        builds = [self._sg_pw_build(it["image"], it["query"], it["candidates"],
                                    nearby_hints_by_id=it.get("nearby_hints_by_id"), **kw)
                  for it in items]
        all_reqs = []; all_sps = []; spans = []
        for b in builds:
            spans.append((len(all_reqs), len(b["reqs"])))
            all_reqs += b["reqs"]; all_sps += b["sps"]
        raw = []
        if all_reqs:
            outs = self.llm.generate(all_reqs, all_sps)
            raw = [o.outputs[0].text for o in outs]
        return [self._sg_pw_assemble(b, raw[s:s + n]) for b, (s, n) in zip(builds, spans)]
