"""Prompt templates used across the pipeline.

Each string is a ``str.format`` template; the field names (``{query}``, ``{category}``,
``{objects_block}``, ``{id1}``/``{id2}``, ``{width}``/``{height}``, ``{ids_list}``,
``{nearby_hints}``, ``{vocab}``) are filled by the caller. Prompt wording is kept
verbatim because it is part of what determines the frozen models' outputs.
"""

# --- per-crop object captions ----------------------------------------------------
CROP_CAPTION_PROMPT = """\
Describe the {category} shown in this image in one short sentence (10-20 words). \
Focus on visual appearance: color, posture, what it is wearing or holding, what it is doing, what is near it. \
Do NOT repeat the category name — only describe distinguishing attributes.
"""


# Context-aware caption, paired with an expanded crop (see geometry.expand_bbox) so the
# model sees the object plus its immediate surroundings; asks for spatial position and
# relations with nearby objects.
CROP_CAPTION_PROMPT_BFAIR = """\
Describe the {category} highlighted in this region in 1-2 sentences (30-50 words). Include:
- visual attributes: color, clothing, posture, what it is doing or holding.
- spatial position relative to nearby objects in the scene ("to the left of...", "behind...", "near...").
- any relation with nearby objects ("holding...", "sitting at...", "next to...").
Do NOT repeat the category name — describe distinguishing attributes.
"""


# Caption with the surrounding context injected as text (nearby detections), paired with
# the tight crop instead of an expanded one.
CROP_CAPTION_PROMPT_BFAIR_EXPLICIT = """\
Describe the {category} highlighted in this image in 1-2 sentences (30-50 words). Include:
- visual attributes: color, clothing, posture, what it is doing or holding.
- the object's spatial position relative to the nearby objects listed below.
- any relation with the listed nearby objects.

Nearby objects detected (label, relative direction): {nearby_hints}

Mention these nearby objects' positions (e.g. "to the left of the {{label}}", \
"near the {{label}}") in your description when relevant. \
Do NOT repeat the category name — describe distinguishing attributes.
"""


# --- pairwise / query-level relations --------------------------------------------
RELATIONS_ONLY_PROMPT = """\
You are given an image with detected objects drawn on it. Each object has a colored bbox with its integer id "[N]" at the top-left corner.

For the referring expression below, list pairwise spatial or interaction relations between the labeled objects that would help disambiguate the target object. Keep predicates concise (e.g., "left of", "next to", "holding", "in front of", "behind").

Referring expression: "{query}"

Object ids (each is an integer): {ids_list}

Respond with a SINGLE JSON object, nothing else. The "subject" and "object" fields MUST be integer ids from the list above — never strings or descriptions.

Example (do NOT copy verbatim, this is just a format example):
{{"relations": [{{"subject": 0, "predicate": "left of", "object": 1}}, {{"subject": 2, "predicate": "behind", "object": 0}}]}}
"""


# Per-pair relation, one call per (id1, id2) with only those two boxes highlighted on
# the full image. Query-agnostic.
PAIRWISE_RELATION_PROMPT = """\
The image highlights exactly two objects: object [{id1}] in the RED box and object [{id2}] in the BLUE box.

Describe the spatial relationship or interaction between object [{id1}] and object [{id2}] in one short phrase (e.g., "left of", "right of", "above", "below", "next to", "in front of", "behind", "holding", "sitting on", "near"). If there is no clear relationship, respond "none".

Respond with a SINGLE JSON object, nothing else:
{{"predicate": "<short phrase>"}}
"""


# Query-conditional per-pair relation: the query steers which relation to report, while
# the single-predicate output format prevents the model from naming the target directly.
PAIRWISE_RELATION_PROMPT_QUERYCOND = """\
The image highlights exactly two objects: object [{id1}] in the RED box and object [{id2}] in the BLUE box.

For the referring expression "{query}", describe the spatial relationship or interaction between object [{id1}] and object [{id2}] in one short phrase that is most relevant to disambiguating the expression. Examples: "left of", "right of", "above", "below", "next to", "in front of", "behind", "holding", "sitting on", "near". If no clear relation exists, respond "none". Do NOT identify which object is the target — only describe their relation.

Respond with a SINGLE JSON object, nothing else:
{{"predicate": "<short phrase>"}}
"""


# --- global scene-layout caption -------------------------------------------------
# One call per sample on the annotated full image; asks for spatial structure only
# (arrangement, grouping, rows), not per-object attributes.
GLOBAL_CAPTION_PROMPT_AGNOSTIC = """\
You are given an image with detected objects highlighted by colored bboxes (each with its integer id "[N]").

Describe the SPATIAL LAYOUT of the highlighted objects in 2-3 short sentences. Focus on:
- how objects are arranged (left-to-right, foreground-background, groupings, rows, columns)
- any salient spatial structure (e.g., "five people sitting in a row", "two stacks of plates")
- prominent landmarks if relevant for orientation

Do NOT enumerate per-object attributes like color or clothing — only spatial structure. Respond with plain text, no JSON, no lists.
"""

# Query-conditional variant: the query filters which spatial facts to emphasize, with an
# explicit guard against identifying the target.
GLOBAL_CAPTION_PROMPT_QUERYCOND = """\
You are given an image with detected objects highlighted by colored bboxes (each with its integer id "[N]").

For the referring expression "{query}", describe the SPATIAL LAYOUT of the highlighted objects in 2-3 short sentences. Focus on aspects (arrangement, grouping, ordering, rows, columns, foreground/background) that would help locate the referent. Do NOT identify which object is the target and do NOT enumerate per-object attributes (colors/clothing) — only spatial structure.

Respond with plain text, no JSON, no lists.
"""


