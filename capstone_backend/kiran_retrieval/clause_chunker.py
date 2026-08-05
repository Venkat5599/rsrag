"""Clause-aware chunking.

Spec section 6: LegalEase stores complete legal clauses instead of arbitrary
fixed-size windows. This module turns cleaned OCR text into whole-clause chunks
that carry their heading, section label, page number and clause type.

Splitting rules, in priority order:
    1. Split on detected clause headings ("8.2 Termination", "ARTICLE V - Notices",
       "Section 12.", "12. Confidentiality").
    2. If no headings exist, group blank-line-separated paragraphs.
    3. Only when a single clause exceeds ``max_words`` is it split further, and
       then at sentence boundaries with an overlap so no sentence is cut in half.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rag.clause_taxonomy import CLAUSE_KEYWORDS
from rag.schemas import Entity, RetrievedChunk
from rag.semantic import split_sentences

PAGE_BREAK = "\f"
PAGE_MARKER = re.compile(r"^\s*(?:-{0,3}\s*)?page\s+(\d+)\s*(?:-{0,3})?\s*$", re.IGNORECASE)

# "8.2 Termination", "12. Confidentiality", "Section 4.1 - Payment"
NUMBERED_HEADING = re.compile(
    r"^\s*(?P<label>(?:article|section|clause)?\s*"
    r"\d+(?:\.\d+)*\s*[.)\-:]?)\s*"
    r"(?P<title>[A-Z][^\n]{0,80})?\s*$",
    re.IGNORECASE,
)

# "ARTICLE V", "Article IV - Notices". Roman numerals are only recognised after an
# explicit article/section/clause keyword. Matching a bare roman numeral would make
# any capitalised word starting with C, D, I, L, M, V or X look like a heading label,
# which turns "CONFIDENTIALITY" into the label "C" plus the title "ONFIDENTIALITY".
ROMAN_HEADING = re.compile(
    r"^\s*(?P<label>(?:article|section|clause)\s+[ivxlcdm]+)\s*[.)\-:]?\s*"
    r"(?P<title>[A-Z][^\n]{0,80})?\s*$",
    re.IGNORECASE,
)
CAPS_HEADING = re.compile(r"^\s*(?P<title>[A-Z][A-Z0-9 ,'&/()\-]{4,70})\s*$")

MIN_CLAUSE_WORDS = 6
DEFAULT_MAX_WORDS = 320
SENTENCE_OVERLAP = 1


@dataclass
class ClauseSegment:
    """A raw clause boundary before it becomes an indexed chunk."""

    heading: str
    body: str
    page: int
    order: int
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.body.split())


def _looks_like_heading(line: str) -> Optional[str]:
    stripped = line.strip()

    if not stripped or len(stripped.split()) > 12:
        return None

    if stripped.endswith((".", ";")) and len(stripped.split()) > 8:
        return None

    for pattern in (NUMBERED_HEADING, ROMAN_HEADING):
        match = pattern.match(stripped)

        if not match or not match.group("label"):
            continue

        label = re.sub(r"\s+", " ", match.group("label")).strip(" .)-:")
        title = (match.group("title") or "").strip(" .-:")
        return f"{label} {title}".strip() if title else label

    caps = CAPS_HEADING.match(stripped)
    if caps:
        return caps.group("title").strip()

    return None


def _split_pages(text: str) -> List[Tuple[int, str]]:
    """Return ``(page_number, page_text)`` pairs.

    Form feeds are honoured first, since OCR emits one per physical page. Failing
    that, standalone "Page 7" markers are used.

    Marker position decides what the marker labels. A marker on the first line is a
    running header, so it names the text *below* it. Anywhere else it is a footer,
    so it names the text *above* it - which is the common case, because a page
    number printed at the foot of a page belongs to that page, not the next one.
    Getting this backwards silently misattributes every citation by one page.

    With no marker at all the document is page 0, which downstream renders as
    "page not recorded" rather than inventing a number.
    """

    if PAGE_BREAK in text:
        return [(index + 1, part) for index, part in enumerate(text.split(PAGE_BREAK))]

    lines = text.splitlines()
    marker_hits = [
        (index, int(PAGE_MARKER.match(line).group(1)))
        for index, line in enumerate(lines)
        if PAGE_MARKER.match(line)
    ]

    if not marker_hits:
        return [(0, text)]

    header_style = marker_hits[0][0] == 0
    pages: List[Tuple[int, str]] = []
    start = 0
    carried_page = marker_hits[0][1] if header_style else 0

    for line_index, page_number in marker_hits:
        body = "\n".join(lines[start:line_index])
        if body.strip():
            pages.append((carried_page if header_style else page_number, body))
        carried_page = page_number
        start = line_index + 1

    tail = "\n".join(lines[start:])

    if tail.strip():
        pages.append((carried_page if header_style else carried_page + 1, tail))

    return pages or [(0, text)]


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-word containment.

    Substring matching is wrong here: "term" is a substring of "termination", so a
    heading reading "8.2 Termination for Convenience" would score the generic Term
    clause exactly as highly as the specific one.
    """

    if not needle:
        return False

    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def infer_clause_type(heading: str, body: str) -> str:
    """Score clause types against the heading (weighted) and the body.

    Ties break towards the longer clause name, because a longer name is the more
    specific one: "Termination for Convenience" says more than "Term".
    """

    heading_lower = (heading or "").lower()
    body_lower = (body or "").lower()
    best_type = ""
    best_score = 0.0

    for clause_type, keywords in CLAUSE_KEYWORDS.items():
        score = 0.0

        if _contains_phrase(heading_lower, clause_type.lower()):
            score += 6.0

        for keyword in keywords:
            if _contains_phrase(heading_lower, keyword):
                score += 3.0
            if _contains_phrase(body_lower, keyword):
                score += 1.0

        if score > best_score or (score == best_score and len(clause_type) > len(best_type)):
            if score > 0.0:
                best_score = score
                best_type = clause_type

    return best_type if best_score >= 2.0 else ""


def _sentence_windows(body: str, max_words: int) -> List[str]:
    sentences = split_sentences(body, min_words=1) or [body]
    windows: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        words = len(sentence.split())

        if current and current_words + words > max_words:
            windows.append(" ".join(current))
            current = current[-SENTENCE_OVERLAP:] if SENTENCE_OVERLAP else []
            current_words = sum(len(part.split()) for part in current)

        current.append(sentence)
        current_words += words

    if current:
        windows.append(" ".join(current))

    return [window.strip() for window in windows if window.strip()]


class ClauseChunker:
    def __init__(self, max_words: int = DEFAULT_MAX_WORDS, min_words: int = MIN_CLAUSE_WORDS) -> None:
        self.max_words = max(max_words, 40)
        self.min_words = max(min_words, 1)

    def segments(self, text: str) -> List[ClauseSegment]:
        segments: List[ClauseSegment] = []
        order = 0

        for page_number, page_text in _split_pages(text or ""):
            heading = ""
            buffer: List[str] = []

            def flush() -> None:
                nonlocal heading, buffer, order
                body = "\n".join(buffer).strip()
                buffer = []
                if not body or len(body.split()) < self.min_words:
                    heading = ""
                    return
                segments.append(
                    ClauseSegment(heading=heading, body=body, page=page_number, order=order)
                )
                order += 1
                heading = ""

            for line in page_text.splitlines():
                if PAGE_MARKER.match(line):
                    continue

                detected = _looks_like_heading(line)

                if detected is not None:
                    flush()
                    heading = detected
                    continue

                if not line.strip():
                    # A blank line only ends a segment when no heading is driving it.
                    if buffer and not heading:
                        flush()
                    continue

                buffer.append(line.strip())

            flush()

        return segments

    def chunk(
        self,
        text: str,
        contract_id: Optional[str] = None,
        filename: str = "",
        entities: Optional[Sequence[Entity]] = None,
    ) -> List[RetrievedChunk]:
        contract_id = contract_id or str(uuid.uuid4())[:8]
        shared_entities = list(entities or [])
        chunks: List[RetrievedChunk] = []

        for segment in self.segments(text):
            windows = (
                [segment.body]
                if segment.word_count <= self.max_words
                else _sentence_windows(segment.body, self.max_words)
            )

            for part, window in enumerate(windows):
                if len(window.split()) < self.min_words:
                    continue

                clause_type = infer_clause_type(segment.heading, window)
                section = segment.heading or (clause_type or f"Block {segment.order + 1}")
                index = len(chunks)

                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{contract_id}-c{index:03d}",
                        contract_id=contract_id,
                        text=window,
                        clause_type=clause_type,
                        section=section,
                        page=segment.page,
                        entities=[
                            entity
                            for entity in shared_entities
                            if entity.text.lower() in window.lower()
                        ],
                        metadata={
                            "filename": filename,
                            "source": "kiran_clause_chunker",
                            "heading": segment.heading,
                            "segment_order": segment.order,
                            "part": part,
                            "part_count": len(windows),
                            "word_count": len(window.split()),
                        },
                    )
                )

        return chunks


def segment_clauses(
    text: str,
    contract_id: Optional[str] = None,
    filename: str = "",
    max_words: int = DEFAULT_MAX_WORDS,
    entities: Optional[Sequence[Entity]] = None,
) -> List[RetrievedChunk]:
    return ClauseChunker(max_words=max_words).chunk(text, contract_id, filename, entities)
