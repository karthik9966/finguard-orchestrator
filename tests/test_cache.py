"""§9.3's retrieval cache.

Driven against `fakeredis`, an in-process Redis, so the real read/write/TTL paths are exercised
without a server. The suite's promise -- no key, no network -- is unchanged.
"""

from __future__ import annotations

import json

import fakeredis
import numpy as np
import pytest

from src.utils import cache as cache_module
from src.utils.cache import CacheStats, RetrievalCache, fingerprint

HITS = [
    {"chunk_id": "obliqa:1:14.2.3.Guidance.1.:a3f9c210", "text": "A Relevant Person must report.",
     "distance": 0.31, "section_clause": "14.2.3.Guidance.1.", "document_title": "AML Rulebook"},
    {"chunk_id": "finra:19-18:7", "text": "The customer breaks transfers into smaller ones.",
     "distance": 0.44, "section_clause": "part 7 of 31", "document_title": "FINRA 19-18"},
]


@pytest.fixture
def cache():
    return RetrievalCache.connect(client=fakeredis.FakeStrictRedis())


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Deterministic unit vectors, so a similarity threshold can be asserted rather than hoped at."""
    vectors = {}

    def fake(query: str):
        if query not in vectors:
            return None
        v = np.asarray(vectors[query], dtype=np.float32)
        return v / np.linalg.norm(v)

    monkeypatch.setattr(cache_module, "_embed", fake)
    return vectors


# --- the key ----------------------------------------------------------------------------


def test_the_same_lookup_is_the_same_key():
    assert fingerprint("q", [1, 2], 15, "minilm") == fingerprint("q", [2, 1], 15, "minilm")
    assert fingerprint(" Q ", [1], 15, "minilm") == fingerprint("q", [1], 15, "minilm")


@pytest.mark.parametrize(
    "other",
    [
        {"tiers": [1]},          # the tier filter changes which clauses come back
        {"k": 5},                # so does k
        {"backend": "openai"},   # minilm and openai vectors are not comparable at all
        {"query": "different"},
    ],
)
def test_anything_that_changes_the_answer_changes_the_key(other):
    base = {"query": "structuring", "tiers": [1, 2], "k": 15, "backend": "minilm"}
    assert fingerprint(**base) != fingerprint(**{**base, **other})


# --- exact matching ---------------------------------------------------------------------


def test_a_stored_lookup_comes_back_intact(cache):
    sha = fingerprint("structuring", [1], 15, "minilm")
    cache.put(sha, "structuring", HITS)
    assert cache.get(sha, "structuring") == HITS
    assert cache.stats.exact_hits == 1 and cache.stats.misses == 0


def test_a_cold_lookup_is_a_miss(cache):
    assert cache.get(fingerprint("unseen", None, 15, "minilm"), "unseen") is None
    assert cache.stats.misses == 1 and cache.stats.hits == 0


def test_an_empty_result_is_not_cached(cache):
    """Caching "nothing found" would hide a broken vector store for 24 hours."""
    sha = fingerprint("q", None, 15, "minilm")
    cache.put(sha, "q", [])
    assert cache.get(sha, "q") is None


def test_entries_expire(cache, monkeypatch):
    monkeypatch.setattr(cache_module, "TTL_SECONDS", 1)
    sha = fingerprint("q", None, 15, "minilm")
    cache.put(sha, "q", HITS)
    cache.client.expire(cache_module.HITS_KEY.format(sha=sha), 0)  # bring the TTL forward
    assert cache.get(sha, "q") is None


# --- semantic matching ------------------------------------------------------------------


def test_a_paraphrase_reaches_the_cached_clauses(cache, stub_embeddings):
    """The command bar's whole purpose here: a human types it differently and still hits."""
    stub_embeddings["transactions structured to avoid reporting limits"] = [1.0, 0.0, 0.0]
    stub_embeddings["structuring below thresholds"] = [0.99, 0.14, 0.0]  # cosine ~0.99

    original = fingerprint("transactions structured to avoid reporting limits", [1], 15, "minilm")
    cache.put(original, "transactions structured to avoid reporting limits", HITS)

    typed = fingerprint("structuring below thresholds", [1], 15, "minilm")
    assert cache.get(typed, "structuring below thresholds") == HITS
    assert cache.stats.semantic_hits == 1 and cache.stats.exact_hits == 0


def test_a_different_question_does_not_borrow_someone_elses_clauses(cache, stub_embeddings):
    """Below the cutoff is a miss. Serving loosely-related clauses would ground a finding in law
    that was retrieved for a different question entirely."""
    stub_embeddings["structuring"] = [1.0, 0.0, 0.0]
    stub_embeddings["fee disclosure to retail clients"] = [0.0, 1.0, 0.0]  # cosine 0.0

    cache.put(fingerprint("structuring", [1], 15, "minilm"), "structuring", HITS)
    other = fingerprint("fee disclosure to retail clients", [1], 15, "minilm")
    assert cache.get(other, "fee disclosure to retail clients") is None
    assert cache.stats.misses == 1


def test_the_threshold_is_the_line(cache, stub_embeddings, monkeypatch):
    monkeypatch.setattr(cache_module, "THRESHOLD", 0.95)
    stub_embeddings["a"] = [1.0, 0.0]
    stub_embeddings["just below"] = [0.94, np.sqrt(1 - 0.94**2)]
    stub_embeddings["just above"] = [0.96, np.sqrt(1 - 0.96**2)]

    cache.put(fingerprint("a", None, 15, "minilm"), "a", HITS)
    assert cache.get(fingerprint("just below", None, 15, "minilm"), "just below") is None
    assert cache.get(fingerprint("just above", None, 15, "minilm"), "just above") == HITS


def test_semantic_matching_is_skipped_when_the_embedder_is_unavailable(cache, monkeypatch):
    """An embedding failure costs a cache hit, never the audit."""
    monkeypatch.setattr(cache_module, "_embed", lambda query: None)
    cache.put(fingerprint("a", None, 15, "minilm"), "a", HITS)
    assert cache.get(fingerprint("b", None, 15, "minilm"), "b") is None


def test_a_hit_reports_what_the_computation_actually_cost(cache):
    """Timing lives in the entry, not in the run. Derived from the current run it only works
    while something still misses -- a fully warm run has nothing left to measure and would
    report 0.0s saved at the exact moment it saves the most."""
    sha = fingerprint("q", None, 15, "minilm")
    cache.put(sha, "q", HITS, elapsed=1.73)

    assert cache.get(sha, "q") == HITS
    assert cache.stats.seconds_saved == pytest.approx(1.73)
    cache.get(sha, "q")
    assert cache.stats.seconds_saved == pytest.approx(3.46), "each hit credits its own saving"


# --- degrading ----------------------------------------------------------------------------


def test_no_redis_means_no_caching_and_no_error(monkeypatch):
    """The hard requirement. A cache that can break an audit is worse than no cache."""
    monkeypatch.setattr(RetrievalCache, "connect", classmethod(lambda cls, client=None: cls()))
    disabled = RetrievalCache.connect()

    assert not disabled.available
    assert disabled.get("sha", "q") is None
    disabled.put("sha", "q", HITS)      # must not raise
    assert disabled.flush() == 0
    assert disabled.entries() == 0


def test_an_unreachable_server_is_learned_once_not_per_call(monkeypatch):
    """audit_node builds a cache per call. Re-discovering a dead Redis each time cost the full
    connect timeout every time -- 30s of test suite became 96s, and in production every
    refinement pass would stall a second on a socket that is not there."""
    attempts = []

    class Refusing:
        def __init__(self, **kwargs):
            attempts.append(1)

        def ping(self):
            raise ConnectionError("connection refused")

    monkeypatch.setitem(__import__("sys").modules, "redis", type("m", (), {"Redis": Refusing}))
    RetrievalCache.reset()
    try:
        for _ in range(5):
            assert not RetrievalCache.connect().available
        assert len(attempts) == 1, "the verdict is remembered, not re-learned"
    finally:
        RetrievalCache.reset()


def test_a_server_that_dies_mid_run_degrades_rather_than_raising(cache):
    class Dying:
        def get(self, *a, **k):
            raise ConnectionError("connection reset by peer")

        def pipeline(self):
            raise ConnectionError("connection reset by peer")

    cache.client = Dying()
    assert cache.get("sha", "q") is None
    cache.put("sha", "q", HITS)
    assert not cache.available, "one failure disables it for the rest of the run"


def test_a_corrupt_entry_is_a_miss_not_a_crash(cache):
    sha = fingerprint("q", None, 15, "minilm")
    cache.client.set(cache_module.HITS_KEY.format(sha=sha), b"{not json")
    assert cache.get(sha, "q") is None


# --- housekeeping -------------------------------------------------------------------------


def test_flush_removes_everything_it_owns(cache):
    cache.put(fingerprint("a", None, 15, "minilm"), "a", HITS)
    cache.put(fingerprint("b", None, 15, "minilm"), "b", HITS)
    cache.client.set("someone-elses-key", b"untouched")

    assert cache.entries() == 2
    cache.flush()
    assert cache.entries() == 0
    assert cache.client.get("someone-elses-key") == b"untouched", "only our namespace"


def test_a_vector_whose_shape_changed_is_ignored(cache, stub_embeddings):
    """Re-embedding with a different model leaves vectors of another width behind. Comparing them
    would raise inside a cache lookup, which must never happen."""
    stub_embeddings["new"] = [1.0, 0.0, 0.0]
    sha = fingerprint("old", None, 15, "minilm")
    cache.client.set(cache_module.HITS_KEY.format(sha=sha), json.dumps(HITS))
    cache.client.set(cache_module.VECTOR_KEY.format(sha=sha),
                     np.array([1.0, 0.0], dtype=np.float32).tobytes())
    cache.client.sadd(cache_module.INDEX_KEY, sha)

    assert cache.get(fingerprint("new", None, 15, "minilm"), "new") is None


# --- reporting ----------------------------------------------------------------------------


def test_the_summary_says_cold_rather_than_claiming_a_saving():
    cold = CacheStats(misses=8)
    assert "0/8" in cold.summary() and "cold" in cold.summary()


def test_the_summary_reports_what_was_actually_saved():
    warm = CacheStats(exact_hits=7, semantic_hits=1, seconds_saved=13.8)
    text = warm.summary()
    assert "8/8" in text and "100%" in text and "13.8s" in text and "1 semantic" in text


def test_a_run_that_never_consulted_the_cache_says_so():
    assert "not consulted" in CacheStats().summary()
