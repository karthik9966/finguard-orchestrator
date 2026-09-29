"""The reasoning core's five nodes -- LLD §5.1 steps 4-7.

Two of the five reach a model. `DetectionNode`, `RetrievalNode` and `ReportGenerationNode` are
deterministic, which is not an economy measure so much as a statement about where judgement
belongs: finding a shape is arithmetic, resolving a curated obligation is a lookup, and rendering
a report is a template. Only *grounding a shape in law* and *judging whether that grounding holds*
are judgements, and those are the two that cost money.

Three properties are load-bearing, and each replaces a measured failure of the pre-migration
system.

**Per-candidate isolation.** Retrieval, grounding and critique all read `candidates[
current_index]`, and only the critic advances the index. A candidate that cannot be grounded
finalises as `needs_review` while its neighbours keep their findings. The old graph gated the
whole batch on one scalar confidence, so one thin finding sent every candidate back through
retrieval and one fabricated citation vetoed the run.

**The faithfulness gate is deterministic and runs first.** Before the critic model is constructed,
Python checks that every id the draft cites was actually in the bundle and that the narrative
invents no transaction reference. On a live June run that check caught two genuinely fabricated
citations -- the failure a self-assessed confidence score reliably misses, because from inside the
draft a fabricated citation looks perfectly well formed. A failed gate is not a penalty on the
score; it is a veto, and `FAITHFULNESS_CHECK_FAILED` can never become an accepted finding.

**The critic cannot change a fact.** It scores grounding. The High-risk bar lives in
`ReportGenerationNode` as a named policy step, because "High" means *file a SAR* and that is a
filing decision, not a review comment.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from src.config import get_config, get_settings
from src.detection.base import detect_all
from src.graph import prompts
from src.models import (
    AgentState,
    Candidate,
    Citation,
    ComplianceReport,
    Critique,
    DraftFinding,
    Finding,
    RetrievalResult,
)
from src.retrieval.retriever import TierAwareRetriever

log = logging.getLogger(__name__)

# LLD §6's taxonomy, as the strings that reach a review note. Named rather than inlined so a note
# an analyst reads and a log line an operator greps carry the same token.
FAITHFULNESS_CHECK_FAILED = "FAITHFULNESS_CHECK_FAILED"
SCHEMA_PARSE_FAILURE = "SCHEMA_PARSE_FAILURE"
LLM_CALL_FAILED = "LLM_CALL_FAILED"
RETRIEVAL_FAILED = "RETRIEVAL_FAILED"

# A transaction reference as this system mints them: FGO + YYMMDD + a counter. Used only to catch
# a *fabricated* one, so it is deliberately loose -- any bank-reference-shaped token in a narrative
# that is not among the candidate's own references is an invention.
TXN_REF = re.compile(r"\b[A-Z]{2,5}\d{8,}\b")

# A chunk id as the store mints it: `source_id:hash16`. The models cite these *in the prose* as well
# as in the structured fields -- a live June run produced "aligns with red-flag indicator
# [ffiec-appendix-f:91b4dc940b784a98]" inside the narrative. Checking only the structured lists
# would leave the readable half of the finding ungated, which is the half an analyst reads.
CHUNK_ID = re.compile(r"\b[a-z0-9][a-z0-9._-]*:[0-9a-f]{16}\b")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _model(*, structured, node: str):
    """The reasoning model, bound late so the suite needs no key and reaches no network.

    `request_timeout` and `max_retries` are both explicit (LLD §6): the default OpenAI client
    retries twice with no ceiling on how long it will wait, and a node that hangs is
    indistinguishable from a node that is working.
    """
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    reasoning = get_config().reasoning
    model = ChatOpenAI(
        model=settings.reasoning_model,
        temperature=settings.llm_temperature,
        timeout=reasoning.llm_timeout_seconds,
        max_retries=reasoning.llm_max_attempts - 1,
    )
    return model.with_structured_output(structured)


def trace_config(state: AgentState, node: str) -> dict[str, Any]:
    """Per-call tracing metadata (HLD §6).

    The run-level config cannot carry any of this: the candidate under review and the loop number
    change underneath the run, and *which candidate on which pass* is the only question worth
    asking of a trace here. Inert when tracing is off.
    """
    candidates = state.get("candidates") or []
    index = state.get("current_index", 0)
    candidate = candidates[index] if index < len(candidates) else None
    return {
        "tags": [f"node:{node}", f"loop:{state.get('loop_count', 0)}"],
        "metadata": {
            "node": node,
            "run_id": state.get("run_id", ""),
            "batch_id": state.get("batch_id", ""),
            "candidate_index": index,
            "candidate_count": len(candidates),
            "candidate_id": candidate.candidate_id if candidate else "",
            "pattern_type": candidate.pattern_type if candidate else "",
            "loop_count": state.get("loop_count", 0),
        },
    }


def current(state: AgentState) -> Candidate:
    return (state.get("candidates") or [])[state.get("current_index", 0)]


def finding_id(state: AgentState, candidate: Candidate) -> str:
    """Unique per *run*, not per candidate.

    `candidate_id` is deliberately stable across runs -- it is a hash of the transaction references,
    so re-auditing a batch produces the same candidate ids and two reports can be diffed. A finding
    is not: it is one run's judgement about that candidate, reached with whatever retrieval and
    review that run had, and re-auditing the same batch after the corpus changed must produce a
    second finding that is separately reviewable rather than colliding with the first.

    Found in Phase 6b, where `findings.finding_id` is a primary key and the collision showed up as
    an integrity error on the second audit of the same month.
    """
    return f"f-{state['run_id']}:{candidate.candidate_id}"


# --- 1. detection -------------------------------------------------------------------------


class DetectionNode:
    """`detect_all` over the parsed batch. No model, ever.

    An empty candidate list sets `clean_flag` and the graph routes straight to report generation,
    so a clean month costs exactly $0.0000. That is the whole of LLD §9's cost routing, applied at
    the level that decides it.
    """

    name = "detection"

    def __init__(self, detector=detect_all) -> None:
        self._detect = detector

    def __call__(self, state: AgentState) -> dict[str, Any]:
        candidates = self._detect(state["records"])
        log.info(
            "detection: %d candidate(s) over %d record(s)", len(candidates), len(state["records"])
        )
        return {
            "candidates": candidates,
            "clean_flag": not candidates,
            "current_index": 0,
            "loop_count": 0,
            "review_notes": [],
        }


def route_after_detection(state: AgentState) -> str:
    """The only router that saves money: nothing to audit means no model is constructed."""
    return "report" if state.get("clean_flag") else "retrieval"


# --- 2. retrieval -------------------------------------------------------------------------


class RetrievalNode:
    """The grounding bundle for the candidate at `current_index`.

    On a loop the `refinement_hint` is used as the indicator query instead of the template. The
    obligations do not change on a loop and are not re-fetched by a different route -- they come
    from the curated map, so a second pass could only return the same chunks. What a refinement
    can legitimately change is which *illustrative* clauses are in front of the model.
    """

    name = "retrieval"

    def __init__(self, retriever: TierAwareRetriever | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> TierAwareRetriever:
        # Built on first use: constructing it opens the vector store, which a clean batch must
        # never pay for.
        if self._retriever is None:
            self._retriever = TierAwareRetriever()
        return self._retriever

    def __call__(self, state: AgentState) -> dict[str, Any]:
        candidate = current(state)
        notes = list(state.get("review_notes") or [])
        hint = state.get("refinement_hint")

        try:
            result, retrieval_notes = self.retriever.retrieve(candidate, hint=hint)
        except Exception as error:  # noqa: BLE001 - one candidate's retrieval, not the batch's
            log.warning("%s for %s: %s", RETRIEVAL_FAILED, candidate.candidate_id, error)
            return {
                "retrieval": RetrievalResult(),
                "review_notes": notes + [f"{RETRIEVAL_FAILED}: {type(error).__name__}: {error}"],
            }

        if retrieval_notes.detail:
            notes.append("; ".join(retrieval_notes.detail))
        return {"retrieval": result, "review_notes": notes}


# --- 3. grounding -------------------------------------------------------------------------


class GroundingNode:
    """Prompt A: one candidate, its obligations and its indicators, in the DraftFinding schema.

    Off-schema output is re-prompted with the validation error (`SCHEMA_PARSE_FAILURE`, up to
    `schema_retries`), because a model that returned prose where a schema was asked for usually
    returns the schema when shown what was wrong with the prose. Anything still failing leaves
    `draft_finding` as None -- the critic finalises that as `needs_review`, so there is exactly
    one place in the graph where a candidate is written off.
    """

    name = "grounding"

    def __init__(self, model_factory=None) -> None:
        self._factory = model_factory or (lambda: _model(structured=DraftFinding, node="grounding"))

    def __call__(self, state: AgentState) -> dict[str, Any]:
        candidate = current(state)
        retrieval = state.get("retrieval") or RetrievalResult()
        notes = list(state.get("review_notes") or [])

        if retrieval.is_empty:
            # Grounding against nothing would produce a SAR that cites nothing and looks
            # confident doing it. Refused here rather than caught by the gate downstream, and
            # refused without spending anything.
            log.warning("grounding skipped for %s: empty bundle", candidate.candidate_id)
            return {
                "draft_finding": None,
                "review_notes": notes + ["no obligations or indicators were retrieved"],
            }

        messages = prompts.grounding_messages(candidate, retrieval, state.get("refinement_hint"))
        config = trace_config(state, "grounding")
        retries = get_config().reasoning.schema_retries

        for attempt in range(1, retries + 1):
            try:
                draft = self._factory().invoke(messages, config=config)
            except ValidationError as error:
                if attempt == retries:
                    log.warning("%s for %s after %d attempts", SCHEMA_PARSE_FAILURE,
                                candidate.candidate_id, attempt)
                    return {
                        "draft_finding": None,
                        "review_notes": notes + [f"{SCHEMA_PARSE_FAILURE}: {error}"],
                    }
                messages = messages + [
                    ("user", prompts.SCHEMA_REPAIR.format(error=str(error)[:800]))
                ]
                continue
            except Exception as error:  # noqa: BLE001 - per-candidate, never the batch
                log.warning("%s for %s: %s", LLM_CALL_FAILED, candidate.candidate_id, error)
                return {
                    "draft_finding": None,
                    "review_notes": notes + [f"{LLM_CALL_FAILED}: {type(error).__name__}: {error}"],
                }

            # The model is given the candidate's id in context but is not trusted to echo it:
            # a draft filed against the wrong candidate would attach a narrative to the wrong
            # transactions, which no downstream check would notice.
            draft.candidate_id = candidate.candidate_id
            return {"draft_finding": draft, "review_notes": notes}

        return {"draft_finding": None, "review_notes": notes + [SCHEMA_PARSE_FAILURE]}


# --- 4. critic ----------------------------------------------------------------------------


def fabricated_ids(draft: DraftFinding, retrieval: RetrievalResult) -> list[str]:
    """Ids the draft cites that were not in the bundle. The hard half of the critic.

    A draft citing a chunk that was never retrieved has invented the authority for its own
    finding. `RetrievalResult.all_ids` is exactly the set the model was shown, so this is a subset
    test rather than a judgement -- which is why it can run before the model and override it.

    Both halves of the draft are checked: the structured citation lists **and** any chunk id named
    in the narrative. A fabricated id in the prose is the more dangerous of the two, because the
    prose is what an analyst reads and a structured field they never see cannot mislead them.
    """
    cited = draft.cited_ids | set(CHUNK_ID.findall(draft.narrative))
    return sorted(cited - retrieval.all_ids)


def fabricated_references(narrative: str, candidate: Candidate) -> list[str]:
    """Transaction references in the narrative that are not this candidate's.

    The pre-migration system put account numbers in a field specified as wire references, because
    a model asked for identifiers reaches for the numbers most present in its context. Here the
    narrative is prose, so nothing can be repaired -- a reference that is not the candidate's is
    a claim about a transaction this finding does not cover, and it fails the gate.
    """
    own = set(candidate.member_txn_refs)
    return sorted({ref for ref in TXN_REF.findall(narrative) if ref not in own})


class CriticNode:
    """Prompt B behind a deterministic gate, and the only node that finalises a candidate.

    Order matters. The gate runs first and, on failure, the model is never called: there is no
    score a judge could return that would make a fabricated citation acceptable, so paying for one
    would be paying to be told something already known.
    """

    name = "critic"

    def __init__(self, model_factory=None) -> None:
        self._factory = model_factory or (lambda: _model(structured=Critique, node="critic"))

    def __call__(self, state: AgentState) -> dict[str, Any]:
        candidate = current(state)
        retrieval = state.get("retrieval") or RetrievalResult()
        draft = state.get("draft_finding")
        notes = list(state.get("review_notes") or [])
        reasoning = get_config().reasoning
        loops = state.get("loop_count", 0)

        # (a) nothing to review -- grounding gave up, or refused an empty bundle.
        if draft is None:
            return self._finalise(state, candidate, None, 0.0, notes)

        # (b) the model said so itself. Taken at its word rather than argued with, and for free.
        if draft.insufficient_evidence:
            notes.append("the model reported the retrieved context insufficient for a finding")
            return self._advance_or_loop(state, candidate, draft, 0.0, notes,
                                         hint=self._evidence_hint(candidate))

        # (c) the deterministic gate.
        invented_ids = fabricated_ids(draft, retrieval)
        invented_refs = fabricated_references(draft.narrative, candidate)
        if invented_ids or invented_refs:
            for chunk_id in invented_ids:
                notes.append(f"{FAITHFULNESS_CHECK_FAILED}: cites {chunk_id!r}, which was not "
                             "among the retrieved clauses")
            for reference in invented_refs:
                notes.append(f"{FAITHFULNESS_CHECK_FAILED}: names transaction {reference!r}, "
                             "which is not part of this candidate")
            log.warning("%s for %s: %d invented id(s), %d invented ref(s)",
                        FAITHFULNESS_CHECK_FAILED, candidate.candidate_id,
                        len(invented_ids), len(invented_refs))
            # Score 0.0 is not a penalty -- it is the veto, expressed on the scale the router
            # reads, so a vetoed draft can never clear the acceptance bar however it loops.
            return self._advance_or_loop(state, candidate, draft, 0.0, notes,
                                         hint=self._evidence_hint(candidate))

        # (d) the model judge.
        try:
            critique = self._factory().invoke(
                prompts.critic_messages(draft, retrieval), config=trace_config(state, "critic")
            )
        except Exception as error:  # noqa: BLE001 - per-candidate
            log.warning("%s in critic for %s: %s", LLM_CALL_FAILED, candidate.candidate_id, error)
            notes.append(f"{LLM_CALL_FAILED} during review: {type(error).__name__}")
            return self._finalise(state, candidate, draft, 0.0, notes)

        score = critique.score
        if critique.unsupported_claims:
            notes.extend(f"unsupported: {claim}" for claim in critique.unsupported_claims)

        if score >= reasoning.confidence_threshold:
            return self._accept(state, candidate, draft, score, notes)

        hint = critique.refinement_hint or critique.reason or self._evidence_hint(candidate)
        log.info("critic scored %s at %.2f (loop %d) -- %s",
                 candidate.candidate_id, score, loops, hint[:120])
        return self._advance_or_loop(state, candidate, draft, score, notes, hint=hint)

    # --- the three exits ------------------------------------------------------------------

    def _advance_or_loop(self, state, candidate, draft, score, notes, *, hint):
        """Loop while there is a refinement left, then finalise as needs_review."""
        if state.get("loop_count", 0) < get_config().reasoning.max_loops:
            return {
                "loop_count": state.get("loop_count", 0) + 1,
                "confidence_score": score,
                "refinement_hint": hint,
                "review_notes": notes,
            }
        notes.append(
            f"review score {score:.2f} stayed below the acceptance bar after "
            f"{state.get('loop_count', 0)} refinement(s)"
        )
        return self._finalise(state, candidate, draft, score, notes)

    def _accept(self, state, candidate, draft, score, notes) -> dict[str, Any]:
        finding = Finding(
            finding_id=finding_id(state, candidate),
            candidate=candidate,
            risk_level=draft.risk_level,
            narrative=draft.narrative,
            applicable_regulations=self._citations(state, draft.cited_obligation_ids, "obligations"),
            red_flag_indicators=self._citations(state, draft.matched_indicator_ids, "indicators"),
            confidence=score,
            status="pending_review",
            review_notes=notes,
        )
        return self._next(state, finding, score)

    def _finalise(self, state, candidate, draft, score, notes) -> dict[str, Any]:
        """A candidate a human has to look at. Never silently dropped: a shape was found, and
        saying nothing about it is indistinguishable from not having looked."""
        if not notes:
            notes = ["the candidate could not be grounded in the retrieved context"]
        finding = Finding(
            finding_id=finding_id(state, candidate),
            candidate=candidate,
            # A finding nobody could ground is not thereby low risk. `medium` says "unresolved",
            # which is the honest reading, and the High bar in report generation is unreachable
            # from here anyway.
            risk_level=draft.risk_level if draft is not None else "medium",
            narrative=(
                draft.narrative if draft is not None
                else f"No grounded narrative was produced for this {candidate.pattern_type} "
                     f"pattern across {len(candidate.member_txn_refs)} transactions."
            ),
            applicable_regulations=(
                self._citations(state, draft.cited_obligation_ids, "obligations")
                if draft is not None else []
            ),
            red_flag_indicators=(
                self._citations(state, draft.matched_indicator_ids, "indicators")
                if draft is not None else []
            ),
            confidence=score,
            status="needs_review",
            review_notes=notes,
        )
        return self._next(state, finding, score)

    @staticmethod
    def _next(state: AgentState, finding: Finding, score: float) -> dict[str, Any]:
        """Record the finding and move to the next candidate, clearing everything candidate-scoped.

        `findings` is rebuilt and returned whole rather than appended to in place: LangGraph merges
        a node's return dict into the state with last-write-wins per key, so a list mutated in
        place would be merged over by the next node's copy.
        """
        return {
            "findings": [*(state.get("findings") or []), finding],
            "current_index": state.get("current_index", 0) + 1,
            "loop_count": 0,
            "confidence_score": score,
            "refinement_hint": None,
            "draft_finding": None,
            "retrieval": None,
            "review_notes": [],
        }

    @staticmethod
    def _citations(state: AgentState, ids: list[str], half: str) -> list[Citation]:
        """Resolve cited ids back to the chunks the model was actually shown.

        Not re-searched. The bundle is in hand, so a citation in a report is the same text that
        was in the prompt -- which is the only version of a citation an auditor can check.
        """
        retrieval = state.get("retrieval") or RetrievalResult()
        chunks = {chunk.chunk_id: chunk for chunk in getattr(retrieval, half)}
        return [Citation.from_chunk(chunks[cited]) for cited in ids if cited in chunks]

    @staticmethod
    def _evidence_hint(candidate: Candidate) -> str:
        from src.detection.query import QueryConstructor

        return QueryConstructor().build(candidate)


def route_after_critic(state: AgentState) -> str:
    """Three ways out, in the order they are checked.

    A refinement hint means this candidate is going round again; otherwise the index has advanced,
    and there is either another candidate or a report to write.
    """
    if state.get("refinement_hint"):
        return "retrieval"
    if state.get("current_index", 0) < len(state.get("candidates") or []):
        return "retrieval"
    return "report"


# --- 5. report generation -----------------------------------------------------------------


class ReportGenerationNode:
    """The filing, assembled in Python. No model call.

    The pre-migration system had a model write this and then repaired three of its fields from the
    state afterwards -- the ratings were *anti-correlated* with the truth (clean May came back High
    recommending a SAR; July, with 23 laundering patterns, came back Low), the reference list came
    back holding account numbers, and the citation list came back empty while the draft plainly
    cited a clause. Every field here is derived from findings that have already been reviewed, so
    there is nothing for a model to add and three things it measurably got wrong.
    """

    name = "report"

    def __call__(self, state: AgentState) -> dict[str, Any]:
        findings = list(state.get("findings") or [])
        reasoning = get_config().reasoning
        report = ComplianceReport(
            report_id=f"rep-{state['run_id']}",
            run_id=state["run_id"],
            period=state["period"],
            generated_at=_now(),
            risk_rating=self.rating(findings, reasoning.high_risk_min_confidence),
            findings=findings,
            flagged_transactions=[
                ref for finding in findings for ref in finding.candidate.member_txn_refs
            ],
            summary=self.summary(state, findings),
            clean=not findings,
            source_document_refs=self.sources(findings),
            quarantined_count=state.get("quarantined_count", 0),
        )
        return {"report": report, "is_complete": True, "confidence_score":
                state.get("confidence_score", 0.0)}

    @staticmethod
    def rating(findings: list[Finding], high_risk_min_confidence: float) -> str:
        """The batch's rating is its worst grounded finding, with one named policy step.

        **High requires confidence.** "High" means *file a SAR*, so it needs a finding the review
        actually stood behind -- not merely one that scored enough to stop the loop. A high-risk
        finding below the bar is reported as medium, which is a statement about the evidence rather
        than about the transactions.
        """
        if not findings:
            return "none"
        high = [f for f in findings if f.risk_level == "high"]
        if any(f.confidence >= high_risk_min_confidence and f.status != "needs_review"
               for f in high):
            return "high"
        if high or any(f.risk_level == "medium" for f in findings):
            return "medium"
        return "low"

    @staticmethod
    def sources(findings: list[Finding]) -> list[Citation]:
        """Every clause the report rests on, deduplicated by chunk id and ordered as encountered.

        Populated from the findings' own citations, so a reference in this list is by construction
        one some finding actually used -- there is no path by which the report can cite something
        no finding cited.
        """
        seen: dict[str, Citation] = {}
        for finding in findings:
            for citation in finding.applicable_regulations + finding.red_flag_indicators:
                seen.setdefault(citation.chunk_id or f"{citation.source_id} {citation.section_ref}",
                                citation)
        return list(seen.values())

    @staticmethod
    def summary(state: AgentState, findings: list[Finding]) -> str:
        """The analyst-facing half of the deliverable (PRD §5.3), rendered from the findings."""
        records = len(state.get("records") or [])
        quarantined = state.get("quarantined_count", 0)
        lines = [f"# Compliance review -- {state.get('period', '?')}", ""]

        if not findings:
            lines += [
                f"{records} transaction(s) were screened for the five monitored typologies "
                "(structuring, fan-in, fan-out, cycle, scatter-gather). No qualifying pattern was "
                "found, so no obligation was engaged and no model was consulted.",
            ]
            if quarantined:
                lines += [
                    "",
                    f"**{quarantined} message(s) could not be parsed** and were excluded from "
                    "screening. This review does not cover them.",
                ]
            return "\n".join(lines)

        needs_review = [f for f in findings if f.status == "needs_review"]
        by_pattern: dict[str, int] = {}
        for finding in findings:
            by_pattern[finding.candidate.pattern_type] = (
                by_pattern.get(finding.candidate.pattern_type, 0) + 1
            )

        lines += [
            f"{records} transaction(s) screened. **{len(findings)} pattern(s)** were detected and "
            f"assessed against the US obligations curated for their typology: "
            + ", ".join(f"{count}× {pattern}" for pattern, count in sorted(by_pattern.items()))
            + ".",
            "",
        ]
        if needs_review:
            lines += [
                f"**{len(needs_review)} of {len(findings)} could not be grounded** to the "
                "acceptance bar and are recorded for analyst review rather than presented as "
                "findings.",
                "",
            ]
        if quarantined:
            lines += [
                f"**{quarantined} message(s) could not be parsed** and were excluded from "
                "screening.",
                "",
            ]

        lines.append("## Findings")
        for finding in findings:
            candidate = finding.candidate
            lines += [
                "",
                f"### {candidate.pattern_type} -- {finding.risk_level} risk "
                f"({finding.status}, confidence {finding.confidence:.2f})",
                "",
                f"{len(candidate.member_txn_refs)} transaction(s); "
                f"detection confidence {candidate.detection_confidence:.2f}.",
                "",
                finding.narrative.strip(),
            ]
            if finding.applicable_regulations:
                lines += ["", "**Obligations cited**"]
                lines += [
                    f"- {c.source_id} {c.section_ref}" for c in finding.applicable_regulations
                ]
            if finding.red_flag_indicators:
                lines += ["", "**Indicators matched**"]
                lines += [f"- {c.source_id} {c.section_ref}" for c in finding.red_flag_indicators]
            if finding.status == "needs_review" and finding.review_notes:
                lines += ["", "**Why this needs review**"]
                lines += [f"- {note}" for note in finding.review_notes]
        return "\n".join(lines)
