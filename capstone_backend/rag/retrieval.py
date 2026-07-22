import re
import uuid
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from . import semantic
from .clause_taxonomy import clause_importance
from .schemas import Entity, RetrievedChunk


@runtime_checkable
class ClauseRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        clause_filters: Optional[Sequence[str]] = None,
        contract_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        ...


def _coerce_entities(raw_entities: Any) -> List[Entity]:
    entities: List[Entity] = []

    for item in raw_entities or []:
        if isinstance(item, Entity):
            entities.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        entities.append(
            Entity(
                text=text,
                label=str(item.get("label", "")),
                score=float(item.get("score", 0.0) or 0.0),
            )
        )

    return entities


def chunks_from_clause_map(
    clauses: Dict[str, Any],
    contract_id: str,
    filename: str = "",
) -> List[RetrievedChunk]:
    chunks: List[RetrievedChunk] = []

    for index, (clause_type, payload) in enumerate(sorted((clauses or {}).items())):
        if isinstance(payload, dict):
            text = str(payload.get("span", "")).strip()
            score = float(payload.get("score", 0.0) or 0.0)
            entities = _coerce_entities(payload.get("entities"))
            page = int(payload.get("page", 0) or 0)
            section = str(payload.get("section", "") or "")
            risk = str(payload.get("risk", "") or "")
        else:
            text = str(payload).strip()
            score = 0.0
            entities = []
            page = 0
            section = ""
            risk = ""

        if not text:
            continue

        chunks.append(
            RetrievedChunk(
                chunk_id=f"{contract_id}-{index:03d}",
                contract_id=contract_id,
                text=text,
                clause_type=clause_type,
                section=section or clause_type,
                page=page,
                risk=risk,
                entities=entities,
                retrieval_score=score,
                metadata={"filename": filename, "source": "v3_pipeline"},
            )
        )

    return chunks


def chunks_from_text(
    text: str,
    contract_id: Optional[str] = None,
    filename: str = "",
    max_words: int = 180,
) -> List[RetrievedChunk]:
    contract_id = contract_id or str(uuid.uuid4())[:8]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]

    if not blocks:
        blocks = [text.strip()] if text and text.strip() else []

    chunks: List[RetrievedChunk] = []

    for block in blocks:
        words = block.split()
        for start in range(0, len(words), max_words):
            window = " ".join(words[start:start + max_words])
            if len(window.split()) < 5:
                continue
            index = len(chunks)
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{contract_id}-t{index:03d}",
                    contract_id=contract_id,
                    text=window,
                    section=f"Block {index + 1}",
                    metadata={"filename": filename, "source": "raw_text"},
                )
            )

    return chunks


class InMemoryClauseRetriever:
    def __init__(
        self,
        chunks: Optional[Sequence[RetrievedChunk]] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self._chunks: List[RetrievedChunk] = list(chunks or [])
        self._embedding_model = embedding_model

    def add(self, chunks: Sequence[RetrievedChunk]) -> None:
        self._chunks.extend(chunks)

    def clear(self) -> None:
        self._chunks = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        clause_filters: Optional[Sequence[str]] = None,
        contract_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        if not query or not self._chunks:
            return []

        filters = {c.lower() for c in (clause_filters or [])}
        scored: List[RetrievedChunk] = []

        for chunk in self._chunks:
            if contract_id and chunk.contract_id != contract_id:
                continue

            dense = semantic.similarity(query, chunk.text, self._embedding_model)
            sparse = semantic.lexical_similarity(query, chunk.text)
            heading = semantic.lexical_similarity(query, chunk.clause_type or chunk.section)
            entity_hits = sum(
                1 for entity in chunk.entities if entity.text.lower() in query.lower()
            )
            entity_match = min(entity_hits / 3.0, 1.0)
            filter_bonus = 0.15 if chunk.clause_type.lower() in filters else 0.0

            score = (
                0.42 * dense
                + 0.22 * sparse
                + 0.12 * heading
                + 0.10 * entity_match
                + 0.14 * clause_importance(chunk.clause_type)
                + filter_bonus
            )

            candidate = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                contract_id=chunk.contract_id,
                text=chunk.text,
                clause_type=chunk.clause_type,
                section=chunk.section,
                page=chunk.page,
                risk=chunk.risk,
                entities=list(chunk.entities),
                dense_score=dense,
                sparse_score=sparse,
                rerank_score=0.0,
                retrieval_score=min(score, 1.0),
                metadata=dict(chunk.metadata),
            )
            scored.append(candidate)

        scored.sort(key=lambda item: item.retrieval_score, reverse=True)
        return scored[:max(top_k, 1)]
