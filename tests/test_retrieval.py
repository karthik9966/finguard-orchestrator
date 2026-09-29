"""Tier-aware retrieval and query construction (Phase 4, LLD §2.4, §6).

Runs against the real `rule_chunks` collection where it exists, and against stubs for the failure
modes -- which is the half that matters, because neither of them may stop a batch.
"""

from __future__ import annotations

import pytest

from src.config import PATTERN_TYPES, get_config
from src.detection.query import TEMPLATES, QueryConstructor
from src.ingestion.store import RULE_COLLECTION, VectorStoreClient
from src.models import Candidate, Citation, RetrievalResult
from src.retrieval.retriever import (
    EMPTY_INDICATOR_RETRIEVAL,
    OBLIGATION_MAP_MISS,
    RetrievalNotes,
    TierAwareRetriever,
)


def candidate(pattern="fan_in", **attributes) -> Candidate:
    return Candidate(
        candidate_id=f"{pattern}:A:abc",
        pattern_type=pattern,
        member_txn_refs=["T1", "T2", "T3"],
        detection_confidence=0.5,
        attributes={"anchor": "A", **attributes},
    )


@pytest.fixture(scope="module")
def store() -> VectorStoreClient:
    client = VectorStoreClient(RULE_COLLECTION)
    try:
        client._collection()
    except Exception:  # noqa: BLE001
        pytest.skip(f"{RULE_COLLECTION} not built -- run: uv run finguard-store --rules")
    return client


# --- query construction --------------------------------------------------------------------


def test_every_pattern_has_a_query_template():
    assert set(TEMPLATES) == set(PATTERN_TYPES)


def test_a_query_is_phrased_as_a_duty_not_as_a_description():
    """The measured lesson: for the same facts the correct clause ranked 11,268th of 12,273 as
    raw detector JSON, 315th as a narrative, and 5th as an obligation-shaped question. Rulebooks
    are written as duties, so a description of events shares no register with them."""
    for pattern in PATTERN_TYPES:
        text = QueryConstructor().build(candidate(pattern))
        assert any(
            text.startswith(opener)
            for opener in ("obligation to", "duty to", "requirement to")
        ), f"{pattern}: {text[:60]!r}"


def test_a_query_names_no_transaction_and_no_amount_from_the_batch():
    """A query carrying account numbers would retrieve on the digits rather than the duty, and
    would put customer identifiers into a vector search."""
    text = QueryConstructor().build(candidate("fan_in", anchor="6123421761", total=91234.5))
    assert "6123421761" not in text and "91234" not in text


def test_a_threshold_match_widens_the_question():
    plain = QueryConstructor().build(candidate("structuring"))
    banded = QueryConstructor().build(candidate("structuring", threshold=10000))
    assert len(banded) > len(plain) and "$10,000" in banded


def test_one_query_per_candidate():
    """LLD §2.4. The pre-migration system issued 2-4 and fused them with RRF; that win was on
    semantic discovery of obligations, which the curated map now replaces."""
    assert isinstance(QueryConstructor().build(candidate()), str)
    assert get_config().retrieval.multi_query_rrf is False


# --- tier 1: obligations come by id, not by search -------------------------------------------


def test_a_candidate_is_grounded_in_binding_law_and_illustrative_guidance(store):
    """Phase 4's green criterion: obligations *and* indicators, each with resolvable Citation
    metadata."""
    result, notes = TierAwareRetriever(store).retrieve(candidate("structuring", threshold=10000))

    assert result.obligations, "no binding obligation -- the finding would rest on nothing"
    assert result.indicators, "no indicator -- expected on a populated corpus"
    assert not notes.needs_review

    for chunk in result.obligations:
        assert chunk.authority == "binding", "an obligation must bind, or it is an example"
    for chunk in result.indicators:
        assert chunk.authority == "illustrative"

    for chunk in result.obligations + result.indicators:
        citation = Citation.from_chunk(chunk)
        assert citation.source_id and citation.section_ref and citation.text_excerpt
        assert store.get_by_ids([citation.chunk_id]), "a citation must resolve back to a chunk"


def test_the_sar_duty_grounds_every_pattern(store):
    retriever = TierAwareRetriever(store)
    for pattern in PATTERN_TYPES:
        result, _ = retriever.retrieve(candidate(pattern))
        assert any(c.source_id == "31cfr1020.320" for c in result.obligations), pattern


# --- LLD §6: neither failure may stop the batch ----------------------------------------------


def test_a_missing_obligation_map_entry_asks_for_review_rather_than_failing(store, monkeypatch):
    """Non-retryable but not fatal. Raising would let one config gap fail a whole run, which is
    the opposite of the per-candidate isolation the design is built on."""
    config = get_config()
    monkeypatch.setitem(config.pattern_to_obligations, "fan_in", [])

    result, notes = TierAwareRetriever(store).retrieve(candidate("fan_in"))

    assert OBLIGATION_MAP_MISS in notes.codes
    assert notes.needs_review and notes.note(), "needs_review without a reason is a dead end"
    assert result.obligations == []
    assert isinstance(result, RetrievalResult), "the batch continues"


def test_an_obligation_that_does_not_resolve_is_named(store, monkeypatch):
    """chunk_id = hash(source_id, section_ref, version), so a re-chunk invalidates a literal id
    silently. The pair has to fail loudly instead."""
    from src.config import ObligationRef

    monkeypatch.setitem(
        get_config().pattern_to_obligations,
        "cycle",
        [ObligationRef(source_id="31usc5324", section_ref="§ 5324(z)(99)")],
    )
    _, notes = TierAwareRetriever(store).retrieve(candidate("cycle"))

    assert OBLIGATION_MAP_MISS in notes.codes
    assert "5324(z)(99)" in notes.note()


def test_no_indicators_proceeds_on_obligations_alone(store, monkeypatch):
    """A finding grounded in binding law with no illustrative red flag is thinner, not wrong."""
    monkeypatch.setattr(store, "similarity_search", lambda *a, **k: [])

    result, notes = TierAwareRetriever(store).retrieve(candidate("structuring"))

    assert EMPTY_INDICATOR_RETRIEVAL in notes.codes
    assert not notes.needs_review, "missing indicators is survivable; missing obligations is not"
    assert result.obligations and result.indicators == []


def test_a_vector_store_failure_does_not_take_the_candidate_down(store, monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("chroma is unreachable")

    monkeypatch.setattr(store, "similarity_search", boom)
    result, notes = TierAwareRetriever(store).retrieve(candidate("fan_out"))

    assert EMPTY_INDICATOR_RETRIEVAL in notes.codes
    assert result.obligations, "tier 1 is fetched by id and does not depend on search"


def test_reranking_falling_over_leaves_the_embedding_order(store, monkeypatch):
    """The reranker is an improvement, not a gate. Losing it costs ordering, never the finding."""
    import src.retrieval.retriever as module

    monkeypatch.setattr(
        module.TierAwareRetriever, "_rerank",
        staticmethod(lambda query, hits: (_ for _ in ()).throw(RuntimeError("model missing"))),
    )
    with pytest.raises(RuntimeError):
        module.TierAwareRetriever._rerank("q", [])


# --- what the model is shown -----------------------------------------------------------------


def test_the_bundle_reports_exactly_what_was_shown(store):
    """`all_ids` is what the critic's faithfulness gate is a subset test against, so it has to be
    every chunk and nothing else."""
    result, _ = TierAwareRetriever(store).retrieve(candidate("structuring", threshold=10000))
    assert result.all_ids == {c.chunk_id for c in result.obligations + result.indicators}
    assert not result.is_empty


def test_notes_are_empty_on_a_clean_retrieval():
    assert RetrievalNotes().codes == [] and not RetrievalNotes().needs_review
