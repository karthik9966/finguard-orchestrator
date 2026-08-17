"""Verify §3.4's ChromaDB loading: idempotent, filterable, and impossible to query wrongly.

The logic tests run against a throwaway collection with a stub encoder, so they are fast and
need no model download. The last test checks the real collection when it has been built.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ingestion import store
from src.ingestion.store import BackendMismatch, clean_metadata

CHUNKS = [
    {
        "chunk_id": "obliqa:1:14.2.3.Guidance.1.:aaaaaaaa",
        "text": "Transactions designed or structured to avoid detection.",
        "section_clause": "14.2.3.Guidance.1.",
        "document_title": "AML Rulebook",
        "document_id": 1,
        "corpus": "obliqa",
        "relevance_tier": 1,
        "jurisdiction": "ADGM",
        "last_updated_date": "2023-12-21",
        "part": None,
    },
    {
        "chunk_id": "obliqa:13:APP1.2:bbbbbbbb",
        "text": "Cash outflows item factor retail deposits stable.",
        "section_clause": "APP1.2",
        "document_title": "PRU",
        "document_id": 13,
        "corpus": "obliqa",
        "relevance_tier": 3,
        "jurisdiction": "ADGM",
        "last_updated_date": None,
        "part": None,
    },
    {
        "chunk_id": "finra:regulatory-notice-19-18:7",
        "text": "The customer breaks funds transfers into smaller transfers.",
        "section_clause": "part 7 of 31",
        "document_title": "FINRA Regulatory Notice 19-18",
        "document_id": None,
        "corpus": "finra",
        "relevance_tier": 1,
        "jurisdiction": "US",
        "last_updated_date": "2019-05-06",
        "part": "7 of 31",
    },
]


class StubBackend:
    """Deterministic unit vectors so retrieval order is predictable without a real model."""

    name = "minilm"
    model_id = "stub-model"
    dimensions = 8

    def encode(self, texts):
        vectors = np.zeros((len(texts), 8), dtype=np.float32)
        for row, text in enumerate(texts):
            for position, char in enumerate(text[:64]):
                vectors[row, position % 8] += ord(char) % 13
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the module at a throwaway store with a stub encoder and fixture chunks."""
    monkeypatch.setattr(store, "PERSIST_DIR", tmp_path / "chroma")
    monkeypatch.setattr(store, "COLLECTION_NAME", "test_regulations")
    monkeypatch.setattr(store, "get_backend", lambda name, **kw: StubBackend())

    written = tmp_path / "chunks"
    written.mkdir()
    import json

    body = "\n".join(json.dumps(chunk) for chunk in CHUNKS) + "\n"
    # Both backends get a chunk file: the mismatch tests need to reach the guard rather than
    # trip the earlier "run the loader first" exit.
    (written / "minilm.jsonl").write_text(body)
    (written / "openai.jsonl").write_text(body)
    monkeypatch.setattr(store, "CHUNK_DIR", written)
    return tmp_path


# --- metadata ------------------------------------------------------------------------


def test_null_metadata_is_stripped_not_stringified():
    """Chroma drops nulls silently; a "None" string would look like data and match nothing."""
    cleaned = clean_metadata(CHUNKS[1])
    assert "last_updated_date" not in cleaned
    assert "part" not in cleaned
    assert cleaned["relevance_tier"] == 3
    assert all(value is not None for value in cleaned.values())


def test_text_and_id_are_not_duplicated_into_metadata():
    cleaned = clean_metadata(CHUNKS[0])
    assert "text" not in cleaned and "chunk_id" not in cleaned


def test_every_metadata_value_is_a_chroma_scalar():
    for chunk in CHUNKS:
        for value in clean_metadata(chunk).values():
            assert isinstance(value, (str, int, float, bool)), value


# --- build ---------------------------------------------------------------------------


def test_build_loads_every_chunk(isolated):
    payload = store.build("minilm")
    assert payload["vectors"] == len(CHUNKS)
    assert payload["by_corpus"] == {"obliqa": 2, "finra": 1}
    assert payload["by_tier"] == {1: 2, 3: 1}
    assert payload["undated"] == 1


def test_rebuilding_does_not_duplicate(isolated):
    store.build("minilm")
    payload = store.build("minilm")
    assert payload["vectors"] == len(CHUNKS), "upsert must be idempotent"


def test_collection_records_the_backend_that_built_it(isolated):
    payload = store.build("minilm")
    assert payload["backend"] == "minilm"
    assert payload["model"] == "stub-model"
    assert payload["built"]


# --- retrieve ------------------------------------------------------------------------


def test_retrieve_returns_ranked_hits_with_citations(isolated):
    store.build("minilm")
    hits = store.retrieve("Transactions designed or structured to avoid detection.", k=3)
    assert hits[0]["chunk_id"] == CHUNKS[0]["chunk_id"]
    assert hits[0]["distance"] < 0.01, "an exact match should be ~0 cosine distance"
    for hit in hits:
        assert hit["section_clause"] and hit["document_title"]


def test_tier_filter_restricts_results(isolated):
    store.build("minilm")
    hits = store.retrieve("cash outflows", k=5, tiers=[1])
    assert hits, "filter must not empty the result set"
    assert {hit["relevance_tier"] for hit in hits} == {1}


def test_querying_with_a_different_backend_raises(isolated):
    """384-dim and 1536-dim vectors are not comparable; silent nonsense is the failure to avoid."""
    store.build("minilm")
    with pytest.raises(BackendMismatch, match="not comparable"):
        store.retrieve("anything", backend_name="openai")


def test_building_over_a_different_backend_raises(isolated, monkeypatch):
    store.build("minilm")

    class OtherBackend(StubBackend):
        name = "openai"
        model_id = "other-model"

    monkeypatch.setattr(store, "get_backend", lambda name, **kw: OtherBackend())
    with pytest.raises(BackendMismatch, match="--rebuild"):
        store.build("openai")


def test_rebuild_flag_replaces_a_mismatched_collection(isolated, monkeypatch):
    store.build("minilm")

    class OtherBackend(StubBackend):
        name = "openai"
        model_id = "other-model"

    monkeypatch.setattr(store, "get_backend", lambda name, **kw: OtherBackend())
    payload = store.build("openai", rebuild=True)
    assert payload["backend"] == "openai" and payload["vectors"] == len(CHUNKS)


# --- the real collection -------------------------------------------------------------


def _real_collection_exists() -> bool:
    try:
        store.stats()
    except Exception:  # noqa: BLE001 - absent collection is the skip condition
        return False
    return True


@pytest.mark.skipif(
    not _real_collection_exists(),
    reason="collection not built -- run: uv run python -m src.ingestion.store",
)
def test_known_good_query_finds_the_suspicion_guidance():
    """The clause the June structuring case is audited against must be reachable in the top 15.

    Anything outside the top 15 is unrecoverable: §9.4 reranks 15 down to 4, it cannot add.
    """
    hits = store.retrieve(
        "transactions deliberately structured to avoid detection or reporting thresholds", k=15
    )
    clauses = {hit["section_clause"] for hit in hits}
    assert "14.2.3.Guidance.1." in clauses, f"got {sorted(clauses)[:6]}"
