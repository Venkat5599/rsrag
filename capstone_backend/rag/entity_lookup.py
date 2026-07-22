import logging
from typing import Dict, List, Optional, Sequence, Tuple

from .clause_taxonomy import ENTITY_LABEL_ALIASES
from .models import get_ner_pipeline
from .schemas import Entity, RetrievedChunk

logger = logging.getLogger(__name__)

MIN_ENTITY_LENGTH = 3
LABEL_THRESHOLDS: Dict[str, float] = {
    "MONEY": 0.6,
    "DATE": 0.6,
    "ORG": 0.72,
    "PERSON": 0.72,
}
DEFAULT_THRESHOLD = 0.8
STOPWORDS = {"the", "this", "agreement", "party", "section", "clause"}


def _clean_entity_text(text: str) -> str:
    return text.replace("##", "").replace(" ##", "").strip()


def requested_labels(query: str) -> List[str]:
    lowered = (query or "").lower()
    labels = [
        label
        for label, aliases in ENTITY_LABEL_ALIASES.items()
        if any(alias in lowered for alias in aliases)
    ]
    return labels


def extract_entities(
    chunks: Sequence[RetrievedChunk],
    ner_model: str,
    labels: Optional[Sequence[str]] = None,
) -> Tuple[List[Entity], bool]:
    if not chunks:
        return [], False

    wanted = {label.upper() for label in (labels or [])}
    pipeline = get_ner_pipeline(ner_model)
    collected: List[Entity] = []
    seen = set()
    model_used = pipeline is not None

    for chunk in chunks:
        raw_entities = []

        if pipeline is not None:
            try:
                raw_entities = pipeline(chunk.text) or []
            except Exception as error:
                logger.warning("Entity extraction failed on chunk %s: %s", chunk.chunk_id, error)
                raw_entities = []

        if raw_entities:
            for item in raw_entities:
                text = _clean_entity_text(str(item.get("word", "")))
                label = str(item.get("entity_group", "")).upper()
                score = float(item.get("score", 0.0) or 0.0)

                if len(text) < MIN_ENTITY_LENGTH or text.lower() in STOPWORDS:
                    continue
                if wanted and label not in wanted:
                    continue
                if score < LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD):
                    continue

                key = (text.lower(), label)
                if key in seen:
                    continue

                seen.add(key)
                collected.append(Entity(text=text, label=label, score=score))

        for entity in chunk.entities:
            label = entity.label.upper()
            if wanted and label not in wanted:
                continue
            key = (entity.text.lower(), label)
            if key in seen:
                continue
            seen.add(key)
            collected.append(Entity(text=entity.text, label=label, score=entity.score))

    collected.sort(key=lambda item: item.score, reverse=True)
    return collected, model_used


def format_entity_answer(entities: Sequence[Entity]) -> str:
    if not entities:
        return "No matching entities were found in the retrieved clauses."

    grouped: Dict[str, List[str]] = {}
    for entity in entities:
        grouped.setdefault(entity.label or "ENTITY", []).append(entity.text)

    lines = []
    for label in sorted(grouped):
        unique = list(dict.fromkeys(grouped[label]))
        lines.append(f"{label}: {', '.join(unique[:8])}")

    return "\n".join(lines)
