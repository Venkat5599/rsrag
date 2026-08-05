"""Metadata store and pre-retrieval filtering.

Spec section 7 gives each indexed clause a metadata envelope, and section 8 states
that metadata filtering reduces the search space *before* dense retrieval. This
module holds the inverted indexes that make that filtering cheap: clause type,
contract, page and entity text all map back to chunk ids.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from rag.clause_taxonomy import clause_importance
from rag.schemas import RetrievedChunk


@dataclass
class ChunkMetadata:
    """The section 7 chunk schema, minus the text and the embedding."""

    chunk_id: str
    contract_id: str
    page: int
    section: str
    clause_type: str
    risk: str
    entities: List[str] = field(default_factory=list)
    importance: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "contract_id": self.contract_id,
            "page": self.page,
            "section": self.section,
            "clause_type": self.clause_type,
            "risk": self.risk,
            "entities": list(self.entities),
            "importance": round(self.importance, 4),
            "extra": dict(self.extra),
        }


def describe(chunk: RetrievedChunk) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk.chunk_id,
        contract_id=chunk.contract_id,
        page=chunk.page,
        section=chunk.section,
        clause_type=chunk.clause_type,
        risk=chunk.risk,
        entities=[entity.text for entity in chunk.entities],
        importance=clause_importance(chunk.clause_type),
        extra=dict(chunk.metadata),
    )


class MetadataStore:
    def __init__(self) -> None:
        self._chunks: Dict[str, RetrievedChunk] = {}
        self._metadata: Dict[str, ChunkMetadata] = {}
        self._by_clause_type: Dict[str, Set[str]] = {}
        self._by_contract: Dict[str, Set[str]] = {}
        self._by_page: Dict[int, Set[str]] = {}
        self._by_entity: Dict[str, Set[str]] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunk_ids(self) -> List[str]:
        return list(self._chunks)

    def add(self, chunk: RetrievedChunk) -> ChunkMetadata:
        metadata = describe(chunk)

        self._chunks[chunk.chunk_id] = chunk
        self._metadata[chunk.chunk_id] = metadata

        if metadata.clause_type:
            self._by_clause_type.setdefault(metadata.clause_type.lower(), set()).add(chunk.chunk_id)

        self._by_contract.setdefault(metadata.contract_id, set()).add(chunk.chunk_id)
        self._by_page.setdefault(metadata.page, set()).add(chunk.chunk_id)

        for entity in metadata.entities:
            self._by_entity.setdefault(entity.lower(), set()).add(chunk.chunk_id)

        return metadata

    def add_many(self, chunks: Iterable[RetrievedChunk]) -> int:
        added = 0
        for chunk in chunks:
            self.add(chunk)
            added += 1
        return added

    def clear(self) -> None:
        self._chunks.clear()
        self._metadata.clear()
        self._by_clause_type.clear()
        self._by_contract.clear()
        self._by_page.clear()
        self._by_entity.clear()

    def chunk(self, chunk_id: str) -> Optional[RetrievedChunk]:
        return self._chunks.get(chunk_id)

    def chunks(self, chunk_ids: Optional[Iterable[str]] = None) -> List[RetrievedChunk]:
        if chunk_ids is None:
            return list(self._chunks.values())
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def metadata(self, chunk_id: str) -> Optional[ChunkMetadata]:
        return self._metadata.get(chunk_id)

    def clause_types(self) -> List[str]:
        return sorted(
            {meta.clause_type for meta in self._metadata.values() if meta.clause_type}
        )

    def contracts(self) -> List[str]:
        return sorted(self._by_contract)

    def candidates(
        self,
        clause_filters: Optional[Sequence[str]] = None,
        contract_id: Optional[str] = None,
        entities: Optional[Sequence[str]] = None,
        min_candidates: int = 0,
    ) -> Set[str]:
        """Chunk ids worth scoring, narrowed by metadata.

        ``contract_id`` is a hard boundary: another contract's clauses are never
        valid evidence. Clause filters are not, because they come from keyword
        guessing in ``clause_taxonomy.match_clause_types`` and that guess is often
        wrong. "Can either party walk away from the contract early?" matches the
        keyword "party" and yields the Parties filter, when the answer lives in the
        Termination clause. Obeying that filter literally makes the right clause
        unreachable and the query fails outright.

        So filtering is only allowed to narrow when it still leaves at least
        ``min_candidates`` chunks to rank - normally the candidate pool size. Below
        that threshold there is nothing to gain: scanning the whole contract is
        already cheap, and the ranker can decide. The filter still pays off on the
        large corpora it was meant for, and the clause guess additionally survives
        as a scoring bonus in ``HybridRetriever.retrieve``, which is the right
        strength for a heuristic.
        """

        scope: Set[str] = (
            set(self._by_contract.get(contract_id, set()))
            if contract_id
            else set(self._chunks)
        )

        if not scope:
            return set()

        narrowed: Set[str] = set()

        for clause_type in clause_filters or []:
            narrowed |= self._by_clause_type.get(clause_type.lower(), set())

        for entity in entities or []:
            narrowed |= self._by_entity.get(entity.lower(), set())

        filtered = narrowed & scope

        if not filtered or len(filtered) < min_candidates:
            return scope

        return filtered

    def stats(self) -> Dict[str, object]:
        return {
            "chunks": len(self._chunks),
            "contracts": len(self._by_contract),
            "clause_types": len(self._by_clause_type),
            "entities": len(self._by_entity),
            "pages": len([page for page in self._by_page if page]),
        }
