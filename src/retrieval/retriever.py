"""`TierAwareRetriever` -- the grounding bundle for one candidate (LLD §2.4).

Per-candidate, not per-batch. The pre-migration system built one shared pool for the whole batch
and capped it at 24 clauses, which is why it needed RRF to arbitrate between competing queries.
A candidate now gets its own obligations and its own indicators, so there is nothing to arbitrate.

Neither failure mode is fatal, per LLD §6:

* `OBLIGATION_MAP_MISS` -- non-retryable but **not** fatal. The candidate is marked for review
  with a note; the rest of the batch is unaffected. Raising here would let one config gap fail a
  whole run, which is the opposite of per-candidate isolation.
* `EMPTY_INDICATOR_RETRIEVAL` -- proceed on obligations alone. A finding grounded in binding law
  with no illustrative red flag is thinner, not wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import get_config
from src.detection.query import QueryConstructor
from src.ingestion.embeddings import get_backend
from src.ingestion.store import RULE_COLLECTION, VectorStoreClient
from src.models import Candidate, RetrievalResult, RuleChunk

log = logging.getLogger(__name__)

OBLIGATION_MAP_MISS = "OBLIGATION_MAP_MISS"
EMPTY_INDICATOR_RETRIEVAL = "EMPTY_INDICATOR_RETRIEVAL"


@dataclass
class RetrievalNotes:
    """What went wrong, per candidate, without stopping the batch."""

    codes: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """A missing obligation means the finding has no binding rule behind it."""
        return OBLIGATION_MAP_MISS in self.codes

    def note(self) -> str:
        return "; ".join(self.detail)


class TierAwareRetriever:
    def __init__(self, store: VectorStoreClient | None = None, *, backend_name: str = "minilm"):
        self.store = store or VectorStoreClient(RULE_COLLECTION, backend_name=backend_name)
        self.backend_name = backend_name
        self.queries = QueryConstructor()

    # --- tier 1 ---------------------------------------------------------------------------

    def obligations_for(self, pattern_type: str, notes: RetrievalNotes) -> list[RuleChunk]:
        """Resolve the curated `(source_id, section_ref)` pairs to chunks.

        Pairs rather than literal chunk ids because `chunk_id = hash(source_id, section_ref,
        version)`: a re-chunk or an eCFR version bump invalidates a literal id with no error.
        """
        references = get_config().pattern_to_obligations.get(pattern_type, [])
        if not references:
            notes.codes.append(OBLIGATION_MAP_MISS)
            notes.detail.append(f"no obligations are curated for {pattern_type}")
            log.warning("%s: no obligations curated for %s", OBLIGATION_MAP_MISS, pattern_type)
            return []

        resolved: dict[str, tuple[str, str]] = {}
        missing: list[str] = []
        for reference in references:
            pair = (reference.source_id, reference.section_ref)
            chunk_id = self.store.resolve(pair)
            if chunk_id is None:
                missing.append(f"{pair[0]} {pair[1]}")
                continue
            resolved[chunk_id] = pair

        if missing:
            notes.detail.append(
                f"{len(missing)} curated obligation(s) did not resolve: {', '.join(missing)}"
            )
            log.warning("%s: unresolved %s", OBLIGATION_MAP_MISS, missing)

        chunks = self._chunks(self.store.get_by_ids(list(resolved)))
        if not chunks:
            notes.codes.append(OBLIGATION_MAP_MISS)
            if not missing:
                notes.detail.append(f"no obligation resolved for {pattern_type}")
        return chunks

    # --- tier 2 ---------------------------------------------------------------------------

    def indicators_for(self, query_text: str, notes: RetrievalNotes) -> list[RuleChunk]:
        """Search the illustrative pool, then rerank and keep the top slice.

        The filter is `authority: illustrative`, not `tier: guidance`: what matters is whether a
        clause *binds*, and conflating the two is how a report ends up citing an example as
        though it were law.
        """
        retrieval = get_config().retrieval
        try:
            with get_backend(self.backend_name) as backend:
                vector = backend.encode([query_text])[0]
            hits = self.store.similarity_search(
                vector, where={"authority": "illustrative"}, k=retrieval.k_indicators
            )
        except Exception as error:  # noqa: BLE001 - an empty indicator set is survivable
            notes.codes.append(EMPTY_INDICATOR_RETRIEVAL)
            notes.detail.append(f"indicator search failed: {type(error).__name__}")
            log.warning("%s: %s", EMPTY_INDICATOR_RETRIEVAL, error)
            return []

        if not hits:
            notes.codes.append(EMPTY_INDICATOR_RETRIEVAL)
            notes.detail.append("no indicators matched; grounding proceeds on obligations alone")
            return []

        ranked = self._rerank(query_text, hits)
        return self._chunks({hit["chunk_id"]: hit for hit in ranked[: retrieval.rerank_top_n]})

    @staticmethod
    def _rerank(query_text: str, hits: list[dict]) -> list[dict]:
        """Cross-encoder pass. Measured on ObliQA: hit@1 45.2% -> 55.6%."""
        try:
            from src.retrieval.rerank import rerank

            return rerank(query_text, hits)
        except Exception as error:  # noqa: BLE001 - reranking is an improvement, not a gate
            log.warning("rerank unavailable, using embedding order: %s", error)
            return hits

    # --- the bundle -----------------------------------------------------------------------

    def retrieve(
        self, candidate: Candidate, *, hint: str | None = None
    ) -> tuple[RetrievalResult, RetrievalNotes]:
        """One candidate's bundle. `hint` is the critic's reformulated question on a loop.

        The hint replaces the *indicator* query only. Obligations come from the curated map keyed
        on the pattern type, so a second pass down a different route could only return the same
        chunks -- what a refinement can legitimately change is which illustrative clauses the model
        is looking at, which is where a thin finding's missing support actually lives.
        """
        notes = RetrievalNotes()
        obligations = self.obligations_for(self.queries.key(candidate), notes)
        query_text = hint.strip() if hint and hint.strip() else self.queries.build(candidate)
        indicators = self.indicators_for(query_text, notes)
        return RetrievalResult(obligations=obligations, indicators=indicators), notes

    @staticmethod
    def _chunks(found: dict[str, dict]) -> list[RuleChunk]:
        """Validate every hit as a RuleChunk.

        The model rejects a non-US jurisdiction on itself, so an ADGM clause cannot reach a
        citation even if a metadata filter is written wrongly later.
        """
        chunks = []
        for chunk_id, payload in found.items():
            data = {key: value for key, value in payload.items() if key != "distance"}
            data.setdefault("chunk_id", chunk_id)
            chunks.append(RuleChunk(**data))
        return chunks
