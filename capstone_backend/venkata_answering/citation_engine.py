import re
from typing import Dict, List, Optional, Sequence

from rag import semantic
from rag.schemas import Citation, RetrievedChunk

CITATION_MARKER = re.compile(r"\[E(\d+)\]")
MIN_SUPPORT_SCORE = 0.2
MAX_QUOTE_CHARS = 320


def _quote(chunk: RetrievedChunk, query: str) -> str:
    sentences = semantic.split_sentences(chunk.text)

    if not sentences:
        return chunk.text[:MAX_QUOTE_CHARS].strip()

    best = max(sentences, key=lambda sentence: semantic.lexical_similarity(query, sentence))
    return best[:MAX_QUOTE_CHARS].strip()


def _build_citation(chunk: RetrievedChunk, focus_text: str, support: float) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        contract_id=chunk.contract_id,
        clause_type=chunk.clause_type or "Clause",
        section=chunk.section or chunk.clause_type or "Unspecified section",
        page=chunk.page,
        quote=_quote(chunk, focus_text),
        support_score=support,
    )


def extract_marker_ids(answer: str) -> List[str]:
    return [f"E{match}" for match in CITATION_MARKER.findall(answer or "")]


def build_citations(
    answer: str,
    evidence: Sequence[RetrievedChunk],
    evidence_map: Optional[Dict[str, RetrievedChunk]] = None,
    embedding_model: Optional[str] = None,
    limit: int = 5,
) -> List[Citation]:
    if not evidence:
        return []

    citations: List[Citation] = []
    seen = set()

    markers = extract_marker_ids(answer)

    if markers and evidence_map:
        for marker in markers:
            chunk = evidence_map.get(marker)
            if chunk is None or chunk.chunk_id in seen:
                continue
            support = semantic.similarity(answer, chunk.text, embedding_model)
            seen.add(chunk.chunk_id)
            citations.append(_build_citation(chunk, answer, support))

    if not citations:
        for chunk in evidence:
            if chunk.chunk_id in seen:
                continue
            support = semantic.similarity(answer, chunk.text, embedding_model) if answer else chunk.retrieval_score
            if support < MIN_SUPPORT_SCORE and citations:
                continue
            seen.add(chunk.chunk_id)
            citations.append(_build_citation(chunk, answer or chunk.clause_type, support))

    citations.sort(key=lambda item: item.support_score, reverse=True)
    return citations[:max(limit, 1)]


def render_citations(citations: Sequence[Citation]) -> str:
    if not citations:
        return "No supporting evidence was retrieved."

    lines = []
    for citation in citations:
        location = citation.section or citation.clause_type
        page = f"page {citation.page}" if citation.page else "page not recorded"
        lines.append(f"{citation.clause_type} | {location} | {page}")

    return "\n".join(lines)
