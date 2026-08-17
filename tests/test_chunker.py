"""Verify §3.3 chunking: nothing is silently mangled, dropped, or oversized.

The expensive assertions run against the real corpus on disk when it is present; the pure
logic is tested with a stub encoder so the suite stays fast and offline.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.chunker import (
    MAX_CHARS,
    adjacent_distances,
    assemble,
    boundary_indices,
    chunk_semantic,
    has_table,
    normalize,
    split_sentences,
    split_table,
    strip_invisibles,
    strip_provenance_header,
)
from src.ingestion.loader import CHUNK_DIR, document_date, tier_for

CHUNKS_PATH = CHUNK_DIR / "minilm.jsonl"


def stub_encoder(texts):
    """Deterministic unit vectors: identical text embeds identically, unrelated text does not."""
    vectors = np.zeros((len(texts), 8), dtype=np.float32)
    for row, text in enumerate(texts):
        for position, char in enumerate(text[:64]):
            vectors[row, position % 8] += ord(char) % 13
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


# --- normalization -------------------------------------------------------------------


def test_invisible_characters_are_removed():
    """4,467 U+200E marks and 971 U+F0FC bullets survive ObliQA's extraction into the text."""
    dirty = "Rule ‎8.3.1 applies to a Relevant Person"
    clean = strip_invisibles(dirty)
    assert "‎" not in clean and "" not in clean
    assert "Rule 8.3.1 applies to a Relevant Person" == clean
    assert all(unicodedata.category(ch) not in {"Cf", "Co", "Cs"} for ch in clean)


def test_normalize_collapses_whitespace_without_eating_paragraphs():
    assert normalize("a\t\tb   c") == "a b c"
    assert normalize("one\n\n\n\n\ntwo") == "one\n\ntwo"


def test_provenance_header_is_stripped():
    text = "# FINRA Rule 3310\n# Source: https://example\n\nEach member shall develop"
    assert strip_provenance_header(text).startswith("Each member shall")


# --- sentence splitting --------------------------------------------------------------


def test_numbered_rule_references_do_not_split_sentences():
    """`Rule 8.3.1(1)(d)` has a period-space-digit shape that naive splitters cut in half."""
    text = "When undertaking ongoing CDD under Rule 8.3.1(1)(d), a Relevant Person must act."
    assert len(split_sentences(text)) == 1


def test_statutory_citations_do_not_split_sentences():
    text = "requirements of the Bank Secrecy Act (31 U.S.C. 5311, et seq.) and its regulations."
    assert len(split_sentences(text)) == 1


def test_list_items_become_their_own_units():
    text = "A Relevant Person must:\n(a) monitor Transactions;\n(b) pay particular attention."
    units = split_sentences(text)
    assert len(units) == 3
    assert units[1].startswith("(a)") and units[2].startswith("(b)")


def test_ordinary_sentences_still_split():
    text = "Activity that appears unusual is not necessarily suspicious. Many customers differ."
    assert len(split_sentences(text)) == 2


# --- cosine boundaries ---------------------------------------------------------------


def test_adjacent_distances_are_zero_for_identical_vectors():
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1))
    assert np.allclose(adjacent_distances(vectors), 0.0)


def test_boundary_lands_where_similarity_drops():
    distances = np.array([0.01, 0.01, 0.90, 0.01])
    assert boundary_indices(distances, percentile=75.0) == {3}


def test_assemble_never_exceeds_the_budget():
    sentences = [f"sentence number {i} with some filler text. " * 4 for i in range(30)]
    for chunk in assemble(sentences, boundaries=set(), min_chars=100, max_chars=500):
        assert len(chunk) <= 500


# --- tables --------------------------------------------------------------------------


def test_table_rows_split_with_the_header_repeated():
    rows = "\n".join(f"term{i}\tdefinition number {i} of the glossary." for i in range(20))
    table = f"/Table Start\nTerm\tDefinition\n{rows}\n/Table End"
    assert has_table(table)
    chunks = split_table(table, max_chars=120)
    assert len(chunks) > 1, "a table longer than the budget must split"
    assert all(chunk.startswith("Term | Definition") for chunk in chunks)
    assert all(len(chunk) <= 120 for chunk in chunks)
    # No row may be lost between groups.
    joined = " ".join(chunks)
    assert all(f"term{i} |" in joined for i in range(20))


def test_glossary_table_splits_by_row_not_by_prose():
    """GLO 1.2.1.Guidance.4. is a single 152k-character passage of term/definition rows."""
    docs = Path("data/raw/regulations/obliqa/StructuredRegulatoryDocuments")
    if not docs.exists():
        pytest.skip("corpus not downloaded")
    passage = next(
        p
        for f in docs.glob("*.json")
        for p in json.loads(f.read_text())
        if p["DocumentID"] == 8 and p["PassageID"].startswith("1.2.1.Guidance.4")
    )
    chunks = chunk_semantic(passage["Passage"], stub_encoder)
    assert len(chunks) > 50
    assert all(len(chunk) <= MAX_CHARS for chunk in chunks)
    assert sum("Defined Terms | Definitions" in chunk for chunk in chunks) == len(chunks) - 1


# --- dates ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("AML_VER09.211223.txt", "2023-12-21"),
        ("FSMR (Consolidated_December 2023).txt", "2023-12"),
        # "Regulations 2017" matches the month-year shape first; the consolidation date wins.
        ("CRS Regulations 2017 (Consolidated_October 2023) v6.txt", "2023-10"),
        ("UAE_CRS_Guidance_Notes_17 June 2020 (002).txt", "2020-06-17"),
        ("Foreign Tax Account Compliance Regulations 2022.txt", "2022"),
        ("ADGM_Guidance_-_Application_of_English_Laws.txt", None),
    ],
)
def test_document_date_extraction(filename, expected):
    assert document_date(filename) == expected


def test_tiers_partition_the_corpus():
    assert tier_for(1) == 1 and tier_for(17) == 1
    assert tier_for(3) == 2 and tier_for(40) == 2
    assert tier_for(13) == 3 and tier_for(9) == 3


# --- the built corpus ----------------------------------------------------------------

pytestmark_corpus = pytest.mark.skipif(
    not CHUNKS_PATH.exists(),
    reason="chunks not built -- run: uv run python -m src.ingestion.loader",
)


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        pytest.skip("chunks not built")
    return [json.loads(line) for line in CHUNKS_PATH.read_text().splitlines()]


def test_no_chunk_exceeds_the_budget(chunks):
    oversized = [c["chunk_id"] for c in chunks if len(c["text"]) > MAX_CHARS]
    assert not oversized, f"{len(oversized)} chunks over {MAX_CHARS} chars: {oversized[:3]}"


def test_every_chunk_is_citable(chunks):
    """A chunk with no clause reference cannot be shown in §6.4's citations drawer."""
    for chunk in chunks:
        assert chunk["section_clause"], chunk["chunk_id"]
        assert chunk["document_title"], chunk["chunk_id"]
        assert chunk["relevance_tier"] in (1, 2, 3)
        assert chunk["jurisdiction"] in ("ADGM", "US")


def test_chunk_ids_are_unique(chunks):
    """(DocumentID, PassageID) collides for 17 keys, so ids carry the passage UUID."""
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_no_invisible_characters_survive_into_chunks(chunks):
    for chunk in chunks:
        assert all(
            unicodedata.category(ch) not in {"Cf", "Co", "Cs"} for ch in chunk["text"]
        ), chunk["chunk_id"]


def test_bare_headings_were_dropped(chunks):
    """5% empty and 11% heading-only passages would otherwise pollute every retrieval."""
    assert not [c for c in chunks if len(c["text"].strip()) < 40]


def test_finra_red_flag_survives_chunking(chunks):
    """A red flag split across two chunks would be retrievable as neither."""
    needle = "breaks funds transfers into smaller transfers"
    matches = [c for c in chunks if needle in c["text"]]
    assert len(matches) == 1, f"expected exactly one chunk to carry the red flag, got {len(matches)}"
    assert matches[0]["corpus"] == "finra"


def test_page_furniture_is_not_embedded(chunks):
    """FINRA 19-18 prints `May 6, 201919-18` on six pages; verbatim it would skew retrieval."""
    assert not [c for c in chunks if "May 6, 201919-18" in c["text"]]


def test_aml_rulebook_suspicion_guidance_is_present_and_whole(chunks):
    """The clause the June structuring case is audited against must survive intact."""
    matches = [
        c
        for c in chunks
        if c["document_id"] == 1 and c["section_clause"] == "14.2.3.Guidance.1."
    ]
    assert matches, "AML Rulebook 14.2.3.Guidance.1. is missing from the chunk set"
    joined = " ".join(c["text"] for c in matches)
    assert "structured to avoid detection" in joined
    assert all(c["relevance_tier"] == 1 and c["jurisdiction"] == "ADGM" for c in matches)
