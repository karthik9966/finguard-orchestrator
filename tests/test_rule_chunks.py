"""The citable US corpus: chunking, metadata and `VectorStoreClient` (Phase 1b).

Runs against the real acquired artifacts and the real collection. A fixture would pin what I
believed eCFR XML and FFIEC PDFs look like; these pin what they are -- which is how the two bugs
these tests now guard against were found in the first place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import get_config
from src.ingestion.loader import chunk_identity, rules_path, section_stem, us_artifacts
from src.ingestion.store import (
    BENCHMARK_COLLECTION,
    RULE_COLLECTION,
    VectorStoreClient,
    VectorStoreUnavailable,
    pack_topics,
    unpack_topics,
)
from src.models import RuleChunk

CHUNKS = rules_path("minilm")
needs_chunks = pytest.mark.skipif(
    not CHUNKS.exists(),
    reason="rule chunks not built -- run: uv run finguard-chunk --rules",
)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text().splitlines()]


# --- identity ---------------------------------------------------------------------------


def test_the_chunk_id_changes_with_the_version():
    """A citation in a filed report must keep resolving to the text it was drafted against, not
    to whatever that section says after the next eCFR edition."""
    first = chunk_identity("31cfr1020.320", "§ 1020.320(a)", "eCFR 2026-09-10")
    assert first == chunk_identity("31cfr1020.320", "§ 1020.320(a)", "eCFR 2026-09-10")
    assert first != chunk_identity("31cfr1020.320", "§ 1020.320(a)", "eCFR 2027-01-01")
    assert first.startswith("31cfr1020.320:"), "the source stays legible in the id"


def test_the_citation_stem_matches_how_each_body_cites_itself():
    assert section_stem("31usc5324") == "§ 5324"
    assert section_stem("finra-3310") == "Rule 3310"


# --- what gets chunked ------------------------------------------------------------------


def test_only_citable_us_law_is_in_scope():
    """ObliQA is ADGM and SAML-D is transactions; neither belongs in a US citation store."""
    ids = {meta["source_id"] for _, meta in us_artifacts()}
    assert "obliqa" not in ids and "saml-d" not in ids
    assert {"31usc5324", "31cfr1010.311", "31cfr1020.320", "ffiec-appendix-f"} <= ids


@needs_chunks
def test_every_chunk_satisfies_the_contract(records):
    """RuleChunk rejects a non-US jurisdiction on the model itself, so validating here means an
    ADGM clause cannot reach a citation even if a metadata filter is later written wrongly."""
    for record in records:
        RuleChunk(**{k: v for k, v in record.items() if k != "source_file"})


@needs_chunks
def test_chunk_ids_are_unique(records):
    ids = [record["chunk_id"] for record in records]
    assert len(set(ids)) == len(ids), "a collision silently overwrites one obligation with another"


@needs_chunks
def test_an_unlettered_section_is_not_dropped(records):
    """31 CFR 1010.311 -- the CTR obligation, and the source of the $10,000 threshold the whole
    structuring detector rests on -- is a single paragraph with no (a). Skipping undesignated
    text for want of a parent removed the entire section from the index."""
    ctr = [r for r in records if r["source_id"] == "31cfr1010.311"]
    assert ctr, "the CTR obligation is missing from the corpus"
    assert "more than $10,000" in " ".join(r["text"] for r in ctr)


@needs_chunks
def test_indicators_are_chunked_one_red_flag_at_a_time(records):
    """"Which indicator matched" is only answerable if an indicator is its own chunk."""
    flags = [r for r in records if r["source_id"] == "ffiec-appendix-f" and " ¶ " in r["section_ref"]]
    assert len(flags) > 100
    assert max(len(r["text"]) for r in flags) < 1200, "a bullet swallowed the prose after it"


@needs_chunks
def test_narrative_guidance_is_kept_as_well_as_its_bullets(records):
    """FFIEC guidance is narrative *with* red-flag lists in it. Taking only the bullets discarded
    the SAR filing requirements: ffiec-manual-sar went in as two chunks of 50k characters."""
    sar = [r for r in records if r["source_id"] == "ffiec-manual-sar"]
    assert len(sar) > 10
    assert any(" ¶ " in r["section_ref"] for r in sar), "indicators"
    assert any("part " in r["section_ref"] for r in sar), "and the prose around them"


@needs_chunks
def test_topic_tags_are_inherited_from_the_curated_map(records):
    tagged = {r["source_id"]: r["topic_tags"] for r in records}
    assert tagged["31usc5324"] == get_config().source_topics["31usc5324"]


# --- the store ---------------------------------------------------------------------------


def test_topic_tags_survive_a_scalar_only_store():
    assert unpack_topics(pack_topics(["ctr", "reporting"])) == ["ctr", "reporting"]
    assert unpack_topics(None) == [] and unpack_topics("") == []


def test_an_unreachable_store_fails_loudly_rather_than_returning_nothing(monkeypatch):
    """LLD §6: VECTOR_STORE_UNAVAILABLE is retryable, and if it stays down the job fails. An
    empty result would look like "no obligation applies", which is a different finding."""
    import src.ingestion.store as store

    monkeypatch.setattr(store, "RETRY_BACKOFF_SECONDS", 0)
    attempts = []

    def refuse():
        attempts.append(1)
        raise ConnectionError("chroma is down")

    client = VectorStoreClient(RULE_COLLECTION)
    with pytest.raises(VectorStoreUnavailable, match="after 3 attempts"):
        client._with_retry("probe", refuse)
    assert len(attempts) == store.RETRY_ATTEMPTS, "it retries before giving up"


@pytest.fixture(scope="module")
def store_client() -> VectorStoreClient:
    client = VectorStoreClient(RULE_COLLECTION)
    try:
        client._collection()
    except Exception:  # noqa: BLE001
        pytest.skip("rule_chunks not built -- run: uv run finguard-store --rules")
    return client


def test_nothing_outside_the_us_is_in_the_collection(store_client):
    metadatas = store_client._collection().get(include=["metadatas"])["metadatas"]
    assert metadatas, "collection is empty"
    assert {meta.get("jurisdiction") for meta in metadatas} == {"US"}


def test_an_obligation_resolves_from_the_pair_phase_1c_will_curate(store_client):
    """The obligation map is curated as (source_id, section_ref) rather than literal ids because
    chunk_id = hash(source_id, section_ref, version). This is the call that makes that work."""
    assert store_client.resolve(("31usc5324", "§ 5324(a)(3)"))
    assert store_client.resolve(("31cfr1010.311", "§ 1010.311"))
    assert store_client.resolve(("31usc5324", "§ 5324(z)(9)")) is None


def test_the_filter_takes_arbitrary_metadata(store_client):
    """The old module-level retrieve could only filter relevance_tier with $in. Tier-2 retrieval
    needs tier, authority and topic at once."""
    binding = store_client._collection().get(where={"authority": "binding"}, include=["metadatas"])
    assert binding["metadatas"]
    assert {m["authority"] for m in binding["metadatas"]} == {"binding"}

    both = store_client._collection().get(
        where={"$and": [{"tier": "statute"}, {"authority": "binding"}]}, include=["metadatas"]
    )
    assert both["metadatas"] and {m["tier"] for m in both["metadatas"]} == {"statute"}


def test_counts_are_reported_per_tier_and_authority(store_client):
    counts = store_client.counts()
    assert counts["total"] > 500
    assert set(counts["tier"]) == {"statute", "regulation", "guidance"}
    assert set(counts["authority"]) == {"binding", "illustrative"}


# --- the benchmark corpus is fenced off (Phase 1c) ---------------------------------------


def test_obliqa_is_not_in_the_citable_corpus(store_client):
    """ADGM law is out of scope for citations, but its 2,786 labelled questions are the only
    retrieval ground truth this project has. So it lives in its own collection: the benchmark
    keeps reproducing and rule_chunks cannot serve an ADGM clause as authority."""
    metadatas = store_client._collection().get(include=["metadatas"])["metadatas"]
    assert not [m for m in metadatas if str(m.get("source_id", "")).startswith("obliqa")]


def test_the_two_collections_do_not_overlap():
    from src.ingestion.store import VectorStoreClient

    benchmark = VectorStoreClient(BENCHMARK_COLLECTION)
    try:
        obliqa = benchmark._collection().get(include=["metadatas"])
    except Exception:  # noqa: BLE001
        pytest.skip("obliqa_benchmark not built -- run: uv run finguard-store --benchmark")

    assert obliqa["ids"], "benchmark collection is empty"
    assert {m.get("corpus") for m in obliqa["metadatas"]} == {"obliqa"}


def test_the_recorded_retrieval_numbers_name_the_model_that_produced_them():
    """hit@1 45.2% -> 55.6% only reproduces for all-MiniLM-L6-v2, and Phase 0 made the model a
    config value. A result recorded without its model is a number nobody can check."""
    import json as _json

    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "retrieval_benchmark.json"
    if not path.exists():
        pytest.skip("no benchmark run recorded -- run: uv run finguard-benchmark")

    runs = _json.loads(path.read_text())
    minilm = [r for r in runs if r["backend"] == "minilm"]
    assert minilm, "no minilm arm recorded"
    assert all(r["model"] for r in minilm), "a recorded result must name its embedding model"

    latest = minilm[-1]
    assert latest["model"] == "all-MiniLM-L6-v2"
    assert latest["hit_at"]["1"] == pytest.approx(0.452, abs=0.005)
    if "hit_at_reranked" in latest:
        assert latest["hit_at_reranked"]["1"] == pytest.approx(0.556, abs=0.005)
        assert latest["hit_at_reranked"]["15"] == latest["hit_at"]["15"], (
            "a reranker reorders and cannot add -- hit@15 must be untouched"
        )
