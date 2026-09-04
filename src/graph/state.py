"""The graph's shared memory (§4.1) and the report it must produce (§5.1).

``AgentState`` is the object every node reads and writes. LangGraph merges what a node returns
into it, so a node is just ``state -> dict of updates``.

``ComplianceReport`` is pulled forward from the blueprint's Phase 3. The milestone list defers
schema enforcement to Phase 4, but a generation node without its output schema has to be written
twice -- binding it now costs nothing and removes the rewrite.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

from src.utils.detectors import Candidate
from src.utils.swift_parser import Wire

RiskRating = Literal["Low", "Medium", "High"]

# §4.2's critic thresholds. 0.75 is a draft that is supported but may be thin; below it, the
# retrieval question is reformulated. Two refinements is the give-up point -- Phase 1 measured
# that 17.2% of questions have no correct clause in the top 15 at all, so a third attempt is
# usually spending money on a clause that is not in the collection.
CONFIDENCE_THRESHOLD = 0.75
MAX_REFINEMENTS = 2

# The bar a finding must clear before it may be filed as High risk. CRITIC_SYSTEM's own scale
# reserves 1.0 for "every regulatory claim cites a retrieved clause that genuinely says what is
# claimed" and puts 0.75 at "supported, but thin -- a claim leans on a clause that is only
# loosely on point". High means *file a SAR*, so it needs the top band, not merely enough score
# to stop the refinement loop.
HIGH_RISK_CONFIDENCE = 0.9


class AgentState(TypedDict, total=False):
    """§4.1's schema, plus the batch fields Phase 2's deterministic nodes need.

    The first seven keys are the blueprint's, unchanged. The rest exist because §4.1 was written
    for a single query and our unit of work is a 220-wire batch: the parsed ledger and the
    detector's candidates have to live somewhere between the PARSE node and the AUDIT node.
    """

    # --- §4.1 ---
    raw_query: str
    extracted_entities: dict[str, Any]
    retrieved_context: list[Document]
    compliance_draft: str
    loop_count: int
    confidence_score: float
    is_audit_complete: bool

    # --- batch handling ---
    batch_path: str
    wires: list[Wire]
    candidates: list[Candidate]
    extraction_failures: list[dict[str, Any]]

    # --- audit / critic bookkeeping ---
    queries: list[str]
    critique: str
    reservations: list[str]
    report: "ComplianceReport | None"


class ComplianceReport(BaseModel):
    """§5.1 verbatim. This is what the generation node is bound to, so the model cannot
    return prose where the filing system expects fields."""

    risk_rating: RiskRating = Field(
        description="Low, Medium, or High Risk Assessment of transaction run"
    )
    flagged_wires: list[str] = Field(
        description="List of suspicious wire reference IDs matching illegal AML patterns"
    )
    applicable_regulations: list[str] = Field(
        description="References to audited compliance sections and regulatory clauses"
    )
    audit_summary: str = Field(
        description="Markdown formatted detailed explanation of the analytical findings"
    )
    source_document_hashes: list[str] = Field(
        description="Cryptographic or metadata IDs of cited source compliance records"
    )

    @field_validator("flagged_wires", "applicable_regulations", "source_document_hashes")
    @classmethod
    def drop_blanks_and_duplicates(cls, values: list[str]) -> list[str]:
        """Models pad lists. An empty citation is worse than a missing one -- §6.4's drawer
        would render a row that resolves to nothing."""
        seen: dict[str, None] = {}
        for value in values:
            cleaned = value.strip()
            if cleaned:
                seen.setdefault(cleaned, None)
        return list(seen)


def initial_state(batch_path: str) -> AgentState:
    """A run starts with nothing but the document. Everything else is derived."""
    return AgentState(
        raw_query=f"Audit {batch_path} for money laundering indicators and report findings.",
        batch_path=batch_path,
        extracted_entities={},
        retrieved_context=[],
        compliance_draft="",
        loop_count=0,
        confidence_score=0.0,
        is_audit_complete=False,
        wires=[],
        candidates=[],
        extraction_failures=[],
        queries=[],
        critique="",
        reservations=[],
        report=None,
    )
