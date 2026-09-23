"""Query noun extraction for query-conditional open-vocabulary detection.

Extracts NOUNS from the referring expression alone (never the full query, which would let
the detector solve REC on its own), applies a light colloquial expansion (e.g. lady ->
[lady, woman, person]), and returns period-joined phrases to prompt GroundingDINO. If no
nouns survive, the caller falls back to the generic object vocabulary.
"""
from __future__ import annotations

from loguru import logger

# Colloquial / RefCOCO-common terms → expansion list.  The idea is to feed
# Grounding DINO multiple synonymous phrases so its semantic recall is preserved
# even when the query uses informal language.
#
# We do NOT map to a closed vocabulary — these are extra phrases Grounding DINO
# can match against alongside the original noun.
COLLOQUIAL_EXPAND: dict[str, list[str]] = {
    # People
    "lady":    ["lady", "woman", "person"],
    "woman":   ["woman", "person"],
    "girl":    ["girl", "woman", "person"],
    "guy":     ["guy", "man", "person"],
    "man":     ["man", "person"],
    "dude":    ["dude", "man", "person"],
    "boy":     ["boy", "child", "person"],
    "kid":     ["kid", "child", "person"],
    "child":   ["child", "person"],
    "baby":    ["baby", "infant", "child", "person"],
    "infant":  ["infant", "baby", "person"],
    "toddler": ["toddler", "child", "person"],
    "teen":    ["teen", "teenager", "person"],
    "teenager":["teenager", "person"],
    "person":  ["person"],
    "people":  ["person"],
    "human":   ["person"],
    "player":  ["player", "person"],
    "athlete": ["athlete", "person"],
    "worker":  ["worker", "person"],
    "chef":    ["chef", "person"],
    "cook":    ["cook", "person"],
    "rider":   ["rider", "person"],
    "skier":   ["skier", "person"],
    "surfer":  ["surfer", "person"],
    # Bags
    "bag":     ["bag", "handbag", "backpack", "suitcase"],
    "purse":   ["purse", "handbag"],
    "luggage": ["luggage", "suitcase"],
    # Vehicles / transport
    "auto":    ["auto", "car"],
    "vehicle": ["vehicle", "car", "truck"],
    # Furniture
    "table":   ["table", "dining table"],
    # Animals
    "puppy":   ["puppy", "dog"],
    "pup":     ["pup", "dog", "puppy"],
    "kitten":  ["kitten", "cat"],
    "kitty":   ["kitty", "cat"],
    # Fruit / food (common ambiguity)
    "fruit":   ["fruit", "banana", "apple", "orange"],
    "veggie":  ["veggie", "broccoli", "carrot"],
    # Sports
    "ball":    ["ball", "sports ball"],
    "bat":     ["bat", "baseball bat"],
    "racquet": ["racquet", "tennis racket"],
    "racket":  ["racket", "tennis racket"],
}


class QueryNounExtractor:
    """SpaCy-backed noun extractor with light colloquial expansion.

    Lazy SpaCy load — pipe state lives in the class instance, so a single
    extractor across all samples reuses the same parser.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        logger.info("Loading SpaCy model {} for query noun extraction", model)
        self.nlp = spacy.load(model)
        # Common stopword nouns that slip through POS tagging — skip these so the
        # detector doesn't waste recall on them.
        self._noun_stop = {
            "thing", "things", "object", "objects", "stuff",
            "side", "part", "piece", "section",
            "image", "photo", "picture", "scene", "view",
            "background", "foreground",
        }
        logger.info("QueryNounExtractor ready")

    def extract(self, query: str) -> list[str]:
        """Return a deduplicated list of phrases to feed Grounding DINO.

        Empty list = nothing useful extracted; caller should fall back.
        """
        if not query or not query.strip():
            return []
        doc = self.nlp(query)
        nouns: list[str] = []
        for tok in doc:
            if tok.pos_ not in ("NOUN", "PROPN"):
                continue
            lemma = tok.lemma_.lower().strip()
            if not lemma or lemma in self._noun_stop:
                continue
            if lemma not in nouns:
                nouns.append(lemma)
        # Apply colloquial expansion — append synonyms after the original.
        expanded: list[str] = []
        seen: set[str] = set()
        for n in nouns:
            for phrase in COLLOQUIAL_EXPAND.get(n, [n]):
                if phrase not in seen:
                    seen.add(phrase)
                    expanded.append(phrase)
        return expanded
