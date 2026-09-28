"""The data contracts every component passes across a boundary -- LLD §3.1.

One module, because these models are the migration's actual interface: detectors emit
``Candidate``, the retriever emits ``RetrievalResult``, the grounding model is bound to
``DraftFinding``, the critic to ``Critique``, and the run ends as one ``ComplianceReport``. When
they live in the module that happens to produce them, a change to a field reads as a change to
that module rather than as a change to the contract, which is how the pre-migration system ended
up with ``Candidate`` defined inside the detector file and imported by the graph, the prompts and
the state.

Three deliberate departures from LLD §3.1, each recorded where it happens:

* ``TransactionRecord.memo`` -- the LLD's field list omits it, but the Evaluation Design's
  ``Injected_Memo`` corpus tests that a free-text instruction in a memo line is treated as inert
  data. Without the field the test is vacuous, because the string never reaches the model.
* ``Critique`` -- LLD §4.1-B specifies the critic's output (a score plus a brief reason) and
  §4.2 requires it be enforced through structured output, but §3.1 does not list the model.
* ``AgentState`` stays a ``TypedDict`` rather than becoming a ``BaseModel``. LangGraph merges a
  node's returned dict into the state, and the objects *inside* the state are the ones the
  Evaluation Design's Tier-1 schema gate validates. Making the container itself a model buys
  re-validation of unchanged fields on every node return and nothing else.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import PatternType

# Bumped when a stored report's shape changes. It travels inside the report so a reader pulled
# out of the results store years later can tell what it is looking at -- which is the whole
# point of Journey 3 (audit-defence lookup).
SCHEMA_VERSION = "2.0"

RiskLevel = Literal["high", "medium", "low"]
# A report may additionally be rated "none": a clean month is a real, valid answer, not the
# absence of one. Silence would be indistinguishable from a crash.
RiskRating = Literal["high", "medium", "low", "none"]
Tier = Literal["statute", "regulation", "guidance"]
Authority = Literal["binding", "illustrative"]
ExtractionMethod = Literal["deterministic", "llm_fallback"]
FindingStatus = Literal["pending_review", "cleared", "escalated", "approved", "needs_review"]


# =============================================================================================
# Transactions
# =============================================================================================
class TransactionRecord(BaseModel):
    """One standardised payment. The output of ingestion and the only view of the batch that
    detection or the model ever sees."""

    txn_ref: str = Field(min_length=1, description="The batch's own reference; the primary key")
    sender_account: str = Field(min_length=1)
    receiver_account: str = Field(min_length=1)
    # Decimal, never float. `float("5810,46".replace(",", ""))` is 581046.0 -- a 100x error
    # inside a regulatory filing. The parser's guard against that is the reason this is a Decimal
    # all the way through to the report.
    amount: Decimal = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    timestamp: datetime
    sender_country: str = Field(min_length=2, max_length=2)
    receiver_country: str = Field(min_length=2, max_length=2)
    instrument: str = Field(min_length=1, description="ACH, cheque, cross-border wire, card, ...")
    txn_type: str | None = None
    # Free text, attacker-controlled. Redacted before any external call and treated as data by
    # the grounding prompt -- see Evaluation Design §5's injection row.
    memo: str = ""
    extraction_method: ExtractionMethod = "deterministic"

    @field_validator("currency", "sender_country", "receiver_country")
    @classmethod
    def upper(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("timestamp")
    @classmethod
    def must_be_utc_aware(cls, value: datetime) -> datetime:
        """A naive timestamp cannot be compared across a batch that spans a DST boundary, and
        every detector works on a time window. Rejecting naive input here is cheaper than
        debugging a window that is an hour wrong once a year."""
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @property
    def corridor(self) -> str:
        return f"{self.sender_country}->{self.receiver_country}"

    @property
    def is_cross_border(self) -> bool:
        return self.sender_country != self.receiver_country


class QuarantinedMessage(BaseModel):
    """A message that neither the parser nor the fallback could read.

    Kept with its raw text so a human can see what was lost. A quarantined message is a message
    the audit did not see, which is a different thing from a message it saw and cleared -- and
    the difference has to be visible, or a batch that half-parsed reports as a clean batch.
    """

    ordinal: int = Field(ge=1, description="Position in the batch, for a message with no ref")
    reference: str | None = None
    reason: str = Field(min_length=1)
    raw: str = ""
    fallback_attempted: bool = False


class ValidationReport(BaseModel):
    """What ingestion accepted, rescued and refused."""

    batch: str = ""
    declared: int | None = Field(default=None, description="The count the statement claims")
    parsed: int = 0
    rescued: int = Field(default=0, description="Read by the fallback after the parser refused")
    quarantined: list[QuarantinedMessage] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Every message the statement declared came back as a record."""
        return not self.quarantined and (self.declared is None or self.parsed == self.declared)

    def summary(self) -> str:
        parts = [f"{self.parsed} parsed"]
        if self.declared is not None and self.declared != self.parsed:
            parts[0] = f"{self.parsed} of {self.declared} parsed"
        if self.rescued:
            parts.append(f"{self.rescued} rescued by the fallback")
        if self.quarantined:
            parts.append(f"{len(self.quarantined)} quarantined")
        return " · ".join(parts)


# =============================================================================================
# Detection
# =============================================================================================
class Candidate(BaseModel):
    """A pattern the deterministic pass found. Carries geometry and measurements only -- naming
    the offence is the model's job, after it has been shown the rule."""

    candidate_id: str = Field(min_length=1)
    pattern_type: PatternType
    member_txn_refs: list[str] = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    detection_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("member_txn_refs")
    @classmethod
    def dedupe_preserving_order(cls, refs: list[str]) -> list[str]:
        return list(dict.fromkeys(ref for ref in refs if ref.strip()))

    @staticmethod
    def make_id(pattern_type: str, anchor: str, refs: list[str]) -> str:
        """Stable across runs on the same input, so a re-run of a batch produces the same
        candidate ids and a report can be diffed against its predecessor."""
        digest = hashlib.sha256("|".join(sorted(refs)).encode("utf-8")).hexdigest()[:10]
        return f"{pattern_type}:{anchor}:{digest}"


# =============================================================================================
# The knowledge base
# =============================================================================================
class RuleChunk(BaseModel):
    """One retrievable unit of regulation, with the metadata a citation needs.

    ``tier`` and ``authority`` are separate on purpose. Tier says what kind of document the text
    came from; authority says whether it *binds*. A finding must rest on binding obligations, and
    may be illustrated by red-flag guidance -- conflating the two is how a report ends up citing
    an example as though it were law.
    """

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    tier: Tier
    authority: Authority
    source_id: str = Field(min_length=1)
    section_ref: str = Field(min_length=1)
    jurisdiction: str = "US"
    topic_tags: list[str] = Field(default_factory=list)
    effective_date: date | None = None
    version: str = ""
    # Present only on a retrieved chunk, absent on one fetched by id.
    score: float | None = None

    @field_validator("jurisdiction")
    @classmethod
    def us_only(cls, value: str) -> str:
        """The PRD puts non-US rulebooks out of scope for this version. Enforcing it on the model
        rather than on the query means an ADGM clause cannot reach a citation even if a metadata
        filter is later written wrongly."""
        cleaned = value.strip().upper()
        if cleaned != "US":
            raise ValueError(
                f"jurisdiction {cleaned!r}: only US rules are citable in this version "
                "(PRD §2); non-US corpora belong in the benchmark collection"
            )
        return cleaned


class RetrievalResult(BaseModel):
    """The grounding bundle for one candidate: what binds, and what illustrates.

    ``obligations`` are fetched deterministically by id from the curated map, so an empty list
    means a config gap (LLD §6 OBLIGATION_MAP_MISS), never a retrieval miss. ``indicators`` come
    from semantic search and may legitimately be empty, in which case grounding proceeds on the
    obligations alone and says the evidence is limited.
    """

    obligations: list[RuleChunk] = Field(default_factory=list)
    indicators: list[RuleChunk] = Field(default_factory=list)

    @property
    def all_ids(self) -> set[str]:
        """Everything the model was shown. The critic's faithfulness gate is a subset test
        against exactly this set."""
        return {chunk.chunk_id for chunk in self.obligations + self.indicators}

    @property
    def is_empty(self) -> bool:
        return not self.obligations and not self.indicators


class Citation(BaseModel):
    """A resolved reference, carrying enough text to be checked without a second lookup. A
    citation that cannot be resolved back to a stored clause is worse than no citation, because
    it looks like authority."""

    source_id: str
    section_ref: str
    title: str = ""
    text_excerpt: str = ""
    effective_date: date | None = None
    chunk_id: str = ""

    @classmethod
    def from_chunk(cls, chunk: RuleChunk, *, excerpt_chars: int = 400) -> Citation:
        excerpt = chunk.text.strip()
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[: excerpt_chars - 1].rstrip() + "…"
        return cls(
            source_id=chunk.source_id,
            section_ref=chunk.section_ref,
            title=chunk.source_id,
            text_excerpt=excerpt,
            effective_date=chunk.effective_date,
            chunk_id=chunk.chunk_id,
        )


# =============================================================================================
# What the models return
# =============================================================================================
class DraftFinding(BaseModel):
    """GroundingNode's structured output (LLD §4.1-A). Every id in it is checked against the
    retrieval bundle before it is allowed to become a Finding."""

    candidate_id: str
    risk_level: RiskLevel
    narrative: str = Field(min_length=1)
    matched_indicator_ids: list[str] = Field(default_factory=list)
    cited_obligation_ids: list[str] = Field(default_factory=list)
    self_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    # The model's own escape hatch. It is asked to say so when the context does not support a
    # finding, rather than to produce one anyway.
    insufficient_evidence: bool = False

    @property
    def cited_ids(self) -> set[str]:
        return set(self.matched_indicator_ids) | set(self.cited_obligation_ids)


class Critique(BaseModel):
    """CriticNode's structured output (LLD §4.1-B).

    The critic judges grounding and nothing else: its prompt forbids adding facts, and it has no
    authority over the risk level. That is why there is no risk field here -- the High-risk
    confidence bar is applied later, in report generation, as a named policy step.
    """

    score: float = Field(ge=0.0, le=1.0, description="Faithfulness / support, 0.0-1.0")
    reason: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)
    # Where AgentState.refinement_hint comes from: a thin finding is usually missing law rather
    # than bad prose, so the hint steers the next retrieval instead of the next draft.
    refinement_hint: str = ""


# =============================================================================================
# Output
# =============================================================================================
class Finding(BaseModel):
    """One finalised item in a report: the pattern, the judgement, and the law behind it."""

    finding_id: str = Field(min_length=1)
    candidate: Candidate
    risk_level: RiskLevel
    narrative: str
    applicable_regulations: list[Citation] = Field(default_factory=list)
    red_flag_indicators: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: FindingStatus = "pending_review"
    # Why a human is being asked to look, when status is needs_review. Empty otherwise.
    review_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def needs_review_must_say_why(self) -> Finding:
        """An unexplained needs_review is an analyst's dead end: they cannot tell whether the
        model timed out, the map missed, or the evidence was genuinely thin."""
        if self.status == "needs_review" and not self.review_notes:
            raise ValueError("a needs_review finding must carry at least one review note")
        return self


class ComplianceReport(BaseModel):
    """The deliverable, in one shape that serves both readers -- the machine-readable schema for
    the downstream persona and the rendered summary for the analyst (PRD §5.3)."""

    report_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    period: str = Field(min_length=1, description="The audited month, YYYY-MM")
    generated_at: datetime
    risk_rating: RiskRating
    findings: list[Finding] = Field(default_factory=list)
    flagged_transactions: list[str] = Field(default_factory=list)
    summary: str = ""
    clean: bool = False
    source_document_refs: list[Citation] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    # Rows that could not be parsed even after the light-model fallback. Surfaced rather than
    # buried: a month whose report is clean because a third of it failed to parse is not clean.
    quarantined_count: int = 0

    @model_validator(mode="after")
    def clean_means_clean(self) -> ComplianceReport:
        """A clean report is a specific claim -- no qualifying pattern was found -- and it must
        not be able to coexist with findings or with a risk rating."""
        if self.clean:
            if self.findings:
                raise ValueError("a clean report cannot carry findings")
            if self.risk_rating != "none":
                raise ValueError("a clean report must be rated 'none'")
        elif self.risk_rating == "none" and self.findings:
            raise ValueError("a report with findings cannot be rated 'none'")
        return self

    @model_validator(mode="after")
    def flagged_transactions_are_evidenced(self) -> ComplianceReport:
        """Every flagged reference must trace to a finding's candidate. The pre-migration system
        learned this the hard way: the model returned account numbers where wire references
        belong, so the field was recomputed in Python. Here it is simply not allowed to drift."""
        evidenced = {ref for finding in self.findings for ref in finding.candidate.member_txn_refs}
        invented = [ref for ref in self.flagged_transactions if ref not in evidenced]
        if invented:
            raise ValueError(
                f"flagged_transactions not traceable to any finding: {sorted(invented)[:5]}"
            )
        return self

    @property
    def needs_review_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status == "needs_review")


# =============================================================================================
# Graph state
# =============================================================================================
class AgentState(TypedDict, total=False):
    """LLD §3.1's AgentState. LangGraph merges each node's returned dict into this, last write
    winning per key, so any field that must accumulate is rebuilt and returned whole by the node
    that owns it rather than appended to in place.

    ``current_index`` is what makes the self-check loop per-candidate: retrieval, grounding and
    critique all read the candidate at that index, and only the critic advances it. The
    pre-migration system looped over the whole batch against a single scalar confidence, so one
    thin finding sent every candidate back through retrieval.
    """

    batch_id: str
    run_id: str
    period: str
    records: list[TransactionRecord]
    candidates: list[Candidate]
    current_index: int
    retrieval: RetrievalResult | None
    draft_finding: DraftFinding | None
    findings: list[Finding]
    loop_count: int
    confidence_score: float
    clean_flag: bool
    refinement_hint: str | None
    is_complete: bool
    quarantined_count: int
    report: ComplianceReport | None


def initial_state(
    *, batch_id: str, run_id: str, period: str, records: list[TransactionRecord],
    quarantined_count: int = 0,
) -> AgentState:
    """A run starts with the parsed batch and nothing else. Note that parsing happens *outside*
    the graph (LLD §5.1 steps 2-3), so records arrive already standardised -- a malformed file
    is a client error before a run id is ever minted."""
    return AgentState(
        batch_id=batch_id,
        run_id=run_id,
        period=period,
        records=records,
        candidates=[],
        current_index=0,
        retrieval=None,
        draft_finding=None,
        findings=[],
        loop_count=0,
        confidence_score=0.0,
        clean_flag=False,
        refinement_hint=None,
        is_complete=False,
        quarantined_count=quarantined_count,
        report=None,
    )
