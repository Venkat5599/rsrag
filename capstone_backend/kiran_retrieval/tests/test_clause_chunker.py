from kiran_retrieval.clause_chunker import (
    ClauseChunker,
    _split_pages,
    infer_clause_type,
    segment_clauses,
)

NUMBERED = """8.2 Termination for Convenience
Either party may terminate this Agreement for convenience upon thirty days prior
written notice to the other party.

11.3 Limitation of Liability
The aggregate liability of either party shall not exceed the fees paid in the twelve
months preceding the claim.
"""


def test_splits_on_numbered_headings():
    chunks = segment_clauses(NUMBERED, "c1")

    assert len(chunks) == 2
    assert chunks[0].section == "8.2 Termination for Convenience"
    assert chunks[1].section == "11.3 Limitation of Liability"


def test_clause_text_is_not_split_mid_clause():
    chunks = segment_clauses(NUMBERED, "c1")

    assert "thirty days prior" in chunks[0].text
    assert "written notice" in chunks[0].text
    assert "aggregate liability" not in chunks[0].text


def test_infers_clause_type_from_heading_and_body():
    chunks = segment_clauses(NUMBERED, "c1")

    assert chunks[0].clause_type == "Termination for Convenience"
    assert chunks[1].clause_type == "Limitation of Liability"


def test_infer_clause_type_returns_empty_when_nothing_matches():
    assert infer_clause_type("", "The quick brown fox jumped over something.") == ""


def test_caps_headings_are_detected():
    text = (
        "CONFIDENTIALITY\n"
        "Each party shall keep the other party's proprietary information confidential."
    )
    chunks = segment_clauses(text, "c1")

    assert len(chunks) == 1
    assert chunks[0].section == "CONFIDENTIALITY"
    assert chunks[0].clause_type == "Confidentiality"


def test_form_feed_pages_are_numbered_from_one():
    pages = _split_pages("first page body\fsecond page body")

    assert [page for page, _ in pages] == [1, 2]


def test_footer_markers_label_the_text_above_them():
    text = "1. PARTIES\nThis Agreement is entered into between two named companies.\nPage 7\n"
    chunks = segment_clauses(text, "c1")

    assert chunks[0].page == 7


def test_header_marker_on_first_line_labels_the_text_below():
    text = "Page 3\nARTICLE I\nThis is the first article body text placed here.\n"
    chunks = segment_clauses(text, "c1")

    assert chunks[0].page == 3


def test_page_is_zero_when_no_marker_exists():
    chunks = segment_clauses("SOME HEADING\nA clause body with several words in it.", "c1")

    assert chunks[0].page == 0


def test_paragraphs_are_grouped_when_no_headings_exist():
    text = (
        "First paragraph with enough words to survive.\n\n"
        "Second paragraph with enough words too."
    )
    chunks = segment_clauses(text, "c1")

    assert len(chunks) == 2


def test_oversized_clause_splits_at_sentence_boundaries():
    body = " ".join(
        f"Sentence number {index} carries some contract wording." for index in range(60)
    )
    chunks = ClauseChunker(max_words=60).chunk(f"5. PAYMENT TERMS\n{body}", "c1")

    assert len(chunks) > 1
    assert all(chunk.text.endswith(".") for chunk in chunks)
    assert all(chunk.metadata["part_count"] == len(chunks) for chunk in chunks)


def test_chunk_ids_are_unique_and_carry_the_contract_id():
    chunks = segment_clauses(NUMBERED, "contract-42")
    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids))
    assert all(chunk_id.startswith("contract-42-c") for chunk_id in ids)


def test_short_fragments_are_dropped():
    assert segment_clauses("8.2 Termination\nToo short.", "c1") == []


def test_empty_text_produces_no_chunks():
    assert segment_clauses("", "c1") == []


def test_metadata_records_the_source_and_filename():
    chunk = segment_clauses(NUMBERED, "c1", "msa.pdf")[0]

    assert chunk.metadata["source"] == "kiran_clause_chunker"
    assert chunk.metadata["filename"] == "msa.pdf"
    assert chunk.metadata["word_count"] == len(chunk.text.split())
