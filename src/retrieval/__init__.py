"""Tier-aware retrieval (LLD §2.4).

Two tiers with different mechanisms, because they answer different questions.

**Tier 1 -- obligations, by curated id.** What *binds* is not a search result. The clause that
obliges a bank to file a SAR is the same clause every time, so it is fetched deterministically
from `pattern_to_obligations` rather than discovered semantically. That also means an empty
obligation list is a config gap, never a retrieval miss.

**Tier 2 -- indicators, by semantic search.** Which red flags an auditor should see genuinely
depends on the candidate, and there are hundreds of them. Search, then rerank.
"""

from src.retrieval.retriever import TierAwareRetriever

__all__ = ["TierAwareRetriever"]
