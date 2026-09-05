"""§9.4's cross-encoder reranking.

Skipped rather than failed when the model is not on disk: FlashRank fetches it on first use, and
the rest of the suite is offline by construction. `uv run python -m src.ingestion.benchmark
--rerank` downloads it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph import rerank as rerank_module

MODEL_CACHE = Path("/tmp") / rerank_module.RERANK_MODEL

pytestmark = pytest.mark.skipif(
    not MODEL_CACHE.exists(),
    reason="FlashRank model not cached -- run: uv run python -m src.ingestion.benchmark --rerank",
)


def hit(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "text": text, "distance": 0.4}


def test_the_clause_that_answers_the_query_is_promoted():
    hits = [
        hit("noise", "Fees for portfolio management are disclosed to the client annually."),
        hit("target", "A Relevant Person must report transactions structured to avoid detection."),
    ]
    order = [h["chunk_id"] for h in rerank_module.rerank(
        "obligation to report transactions structured to avoid reporting thresholds", hits)]
    assert order[0] == "target"


def test_nothing_is_added_or_lost():
    """A reranker reorders. If it could drop a clause it would be a filter, and the 17.2% of
    questions with no correct clause in the top 15 would quietly become worse, not better."""
    hits = [hit(f"c{i}", f"Clause {i} concerning customer due diligence obligations.")
            for i in range(8)]
    out = rerank_module.rerank("duty to verify the source of funds", hits)
    assert {h["chunk_id"] for h in out} == {h["chunk_id"] for h in hits}
    assert len(out) == len(hits)


def test_a_single_hit_needs_no_model():
    one = [hit("only", "text")]
    assert rerank_module.rerank("q", one) == one
    assert rerank_module.rerank("q", []) == []
