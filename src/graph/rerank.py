"""Cross-encoder reranking of retrieved clauses (§9.4).

The embedding search compares a query and a clause *separately*, as vectors computed before
either had seen the other. That is what makes it fast enough for 12,273 chunks and also what
limits it. A cross-encoder reads the query and one clause **together** and scores the pair, which
is far more accurate and far too slow for a whole collection -- exactly right for the 15 a query
already returned.

Measured against ObliQA's 2,786 labelled questions, retrieving 15 and reranking:

    hit@1   45.2% -> 55.6%   (+10.5)
    hit@4   65.2% -> 72.9%   (+7.7)
    hit@8   73.2% -> 77.6%   (+4.4)
    hit@15  79.2% -> 79.2%   (+0.0)

The last row is the point, not a disappointment: a reranker reorders, it cannot add. Phase 1's
ceiling -- 17.2% of questions have no correct clause in the top 15 -- is untouched by this and by
anything else short of better retrieval.

**Each query is reranked against itself, then the reranked lists are fused.** Handing one joined
string to the reranker was measured too and is worse where it matters: it drops clauses the live
reports cite from rank 3 to 7 and rank 4 to 12, because a candidate answering one of seven
questions scores poorly against a paragraph containing all seven. Reranking per query preserves
the same principle RRF exists for -- a list is only ever scored on its own terms.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

# ms-marco-TinyBERT-L-2-v2, FlashRank's default: 3 MB, CPU-only, ~40 ms for 15 passages. Small
# enough that reranking is free next to a single gpt-4o call, which is the whole argument for
# doing this work locally rather than asking a model to choose.
RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"


@lru_cache(maxsize=1)
def _ranker():
    """Built once per process: the constructor downloads and unpacks the model.

    FlashRank calls ``logging.basicConfig`` on import, which turns on INFO logging for every
    library in the process -- httpx then narrates every HTTP request it makes. Confining that is
    not cosmetic: it is the difference between a readable run and 128 KB of scrollback.
    """
    logging.getLogger("flashrank").setLevel(logging.WARNING)
    from flashrank import Ranker

    ranker = Ranker(model_name=RERANK_MODEL)
    for noisy in ("httpx", "sentence_transformers", "flashrank"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return ranker


def rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One query's hits, reordered by how well each clause answers *that* query.

    Returns the hits unchanged if there is nothing to reorder, so callers need no special case.
    """
    if len(hits) < 2:
        return hits

    from flashrank import RerankRequest

    by_id = {hit["chunk_id"]: hit for hit in hits}
    scored = _ranker().rerank(
        RerankRequest(
            query=query,
            passages=[{"id": hit["chunk_id"], "text": hit["text"]} for hit in hits],
        )
    )
    return [by_id[entry["id"]] for entry in scored if entry["id"] in by_id]
