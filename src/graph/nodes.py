"""The graph's nodes (§4.2). Each takes the state and returns the keys it changes.

Three nodes are free and deterministic -- PARSE, DETECT and the clean-batch exit. The model is
reached only after Python has established there is something to audit, which is §9.2's cost
routing applied at the level that matters: a batch with no candidate patterns costs $0.00.

The critic is a **hybrid**. A Python gate runs first and checks that every clause the draft
cites actually appears in what retrieval returned; a citation that is not there is a fabrication
and fails outright, whatever the model thinks of the draft. The model then scores how well the
surviving claims are supported. A model grading its own work skews high, and the failure this
gate catches -- a confident citation of a rule that was never retrieved -- is exactly the one
that would survive a self-assessment.
"""

from __future__ import annotations

import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.graph import prompts
from src.graph.rerank import rerank
from src.graph.state import (
    CONFIDENCE_THRESHOLD,
    HIGH_RISK_CONFIDENCE,
    MAX_REFINEMENTS,
    AgentState,
    ComplianceReport,
)
from src.ingestion.store import retrieve
from src.utils.detectors import Candidate, detect
from src.utils.swift_parser import MalformedMessage, Wire, parse_batch

RETRIEVE_K = 15

# Seven queries at k=15 dedupe to ~93 clauses on a real batch, and handing all of them to the
# drafter measurably made the report worse: with 93 clauses in context the model cited nothing
# at all and the critic scored the draft 0.00, while the same batch with a pruned context cited
# the clauses it should. Retrieval was never the problem -- AML Rulebook 8.5.1, 10.3.2, 8.2.1
# and FINRA 19-18 rank 1, 2, 6 and 4 -- the noise underneath them was.
#
# 24 rather than 15 because 14.2.3.Guidance.1., the clause the June structuring cluster turns
# on, only reaches rank 10 of 93 once the lists are fused. §9.4's reranker is the experiment that
# decides whether this crude cut can come down toward the blueprint's 4.
MAX_CONTEXT_CLAUSES = 24

# Reciprocal rank fusion. Each query's distances are measured against its own vector, so they are
# not comparable across queries: on the June batch the seven queries' best hits span 0.3433 to
# 0.4827, and pooling every hit into one distance sort therefore ranks *how easy the question
# was* above *how good the answer is* -- the hardest query's champion lands below easier queries'
# also-rans. RRF scores a clause only by its position within the list it came from, which is
# immune to that. Measured on June, it moves 14.2.3.Guidance.1. from rank 20 to rank 10; the two
# intuitive alternatives are worse (round-robin 23, min-max normalised distance 42). 60 is the
# constant from the original paper -- it damps the top of each list so a single query's rank-1
# cannot dominate a clause that several queries agree on.
RRF_K = 60

# §9.4. Reranking is on by default because it is free -- a 3 MB CPU model beside a gpt-4o call --
# and measurably better: on ObliQA's 2,786 labelled questions it lifts hit@1 from 45.2% to 55.6%
# and hit@4 from 65.2% to 72.9%. See src/graph/rerank.py for the full table and the reason each
# query is reranked against itself rather than against a joined string.
USE_RERANKER = True

# Seats held for the critic's reformulated query on a loop-back. Without them the refinement is
# very nearly inert: after seven queries the 24th incumbent already holds an RRF score of
# 0.01562, while a brand-new clause at rank 1 of a single list scores 1/(60+1) = 0.01639 -- a
# margin so thin that only the refinement's *first* hit can clear it and ranks 2-15 are all cut.
# Measured on June: 1 new clause out of 15 retrieved. The loop exists to fill a gap the first
# pass missed, so the clauses that answer the critic's question cannot be made to out-argue
# seven queries' worth of accumulated agreement -- they have to be let in.
REFINEMENT_RESERVE = 5

# Tier 1 is the AML-bearing corpus. A cross-border leg opens tier 2 -- the wider ADGM rulebook
# carries the cross-border and correspondent-banking obligations. This is a deterministic rule
# rather than a model choice: letting the model pick would mean putting the 46-document
# inventory into every prompt to choose from.
BASE_TIERS = [1]
CROSS_BORDER_TIERS = [1, 2]

# A bracketed span in the draft is treated as a citation. Markdown links are excluded by the
# lookahead -- "[text](url)" is a link, not a claim about the rulebook.
CITATION = re.compile(r"\[([^\[\]\n]{3,160})\](?!\()")


def trace_config(state: AgentState, node: str) -> dict[str, Any]:
    """§7.2's custom metadata and run-tagging, attached to each model call.

    The run-level config set at ``invoke()`` cannot carry any of this: wire and candidate counts
    do not exist until PARSE and DETECT have run, and ``loop_count`` changes underneath the run.
    Attaching it per call is what makes the question §7.2 actually poses -- *which context block
    caused the critic to trigger a loop revision* -- answerable from a trace instead of from a
    print statement, because the second draft's span carries a different clause count and loop
    number than the first.

    Costs nothing when tracing is off: LangChain ignores the config unless LANGCHAIN_TRACING_V2
    is set, so this is inert on a normal run.
    """
    candidates = state.get("candidates", []) or []
    return {
        "tags": [f"node:{node}", f"loop:{state.get('loop_count', 0)}"],
        "metadata": {
            "node": node,
            "batch": Path(str(state.get("batch_path", ""))).name,
            "wire_count": len(state.get("wires", []) or []),
            "candidate_count": len(candidates),
            "shapes": sorted({c.shape for c in candidates}),
            "query_count": len(state.get("queries", []) or []),
            "clause_count": len(state.get("retrieved_context", []) or []),
            "loop_count": state.get("loop_count", 0),
        },
    }


def _model(env_var: str, default: str, **kwargs):
    """Bound late so the test suite never needs a key and never reaches the network."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.environ.get(env_var, default), temperature=0, **kwargs)


# --- 1. PARSE ---------------------------------------------------------------------------


def escalate(failure: MalformedMessage) -> Wire | None:
    """§4.2's Extraction Node, demoted to a fallback for the messages regex refused.

    Returns None rather than raising: one unreadable message must not cost the other 219.
    """
    model = _model("EXTRACTION_MODEL", "gpt-4o-mini").with_structured_output(
        prompts.ExtractedWire
    )
    try:
        extracted = model.invoke(
            [
                ("system", prompts.EXTRACTION_SYSTEM),
                ("user", prompts.EXTRACTION_USER.format(reason=failure.reason, raw=failure.raw)),
            ],
            # No state to draw on -- a rescue is per message, not per batch. The reference and
            # the reason are what make a fallback span searchable at all.
            config={
                "tags": ["node:extraction_fallback"],
                "metadata": {"node": "extraction_fallback", "reference": failure.reference,
                             "reason": failure.reason},
            },
        )
        from datetime import date

        year, month, day = (int(part) for part in extracted.value_date.split("-")) # type: ignore
        return Wire(
            reference=extracted.reference, # type: ignore
            value_date=date(year, month, day),
            currency=extracted.currency, # type: ignore
            amount=Decimal(extracted.amount), # type: ignore
            sender_account=extracted.sender_account, # type: ignore
            sender_name=extracted.sender_name, # type: ignore
            sender_address="",
            sender_bic=extracted.sender_bic, # type: ignore
            sender_country=extracted.sender_bic[4:6], # type: ignore
            receiver_account=extracted.receiver_account, # type: ignore
            receiver_name=extracted.receiver_name, # type: ignore
            receiver_address="",
            receiver_bic=extracted.receiver_bic, # type: ignore
            receiver_country=extracted.receiver_bic[4:6], # type: ignore
            bank_operation_code="CRED",
        )
    except Exception:  # noqa: BLE001 - a failed rescue is reported, never guessed at
        return None


def parse_node(state: AgentState) -> dict[str, Any]:
    batch = parse_batch(state["batch_path"]) # type: ignore
    wires = list(batch.wires)
    failures: list[dict[str, Any]] = []

    for failure in batch.failures:
        rescued = escalate(failure)
        failures.append(
            {
                "reference": failure.reference,
                "reason": failure.reason,
                "rescued_by_model": rescued is not None,
            }
        )
        if rescued is not None:
            wires.append(rescued)

    if not wires:
        raise ValueError(
            f"{state['batch_path']} yielded no readable wires -- " # type: ignore
            f"{len(batch.failures)} messages refused, none rescued"
        )

    return {
        "wires": wires,
        "extraction_failures": failures,
        "extracted_entities": {
            "batch": batch.source.name,
            "statement_reference": batch.statement_reference,
            "declared_messages": batch.declared_messages,
            "parsed_messages": len(wires),
            "unreadable_messages": len([f for f in failures if not f["rescued_by_model"]]),
        },
    }


# --- 2. DETECT --------------------------------------------------------------------------


def detect_node(state: AgentState) -> dict[str, Any]:
    candidates = detect(state["wires"]) # type: ignore
    entities = dict(state.get("extracted_entities", {}))
    entities["candidates"] = [c.as_row() for c in candidates]
    entities["wires_under_review"] = len({r for c in candidates for r in c.references})
    return {"candidates": candidates, "extracted_entities": entities}


# --- 3. ROUTE ---------------------------------------------------------------------------


def route_after_detect(state: AgentState) -> str:
    """The blueprint's router predicate ("contains a cross-border wire") can never fire --
    cross-border is 9.77% of SAML-D, so a 220-wire batch is domestic with probability 1.5e-10.
    Routing on an empty candidate list is the decision that actually saves money."""
    return "audit" if state["candidates"] else "no_findings" # type: ignore


def no_findings_node(state: AgentState) -> dict[str, Any]:
    """A clean batch still gets a report. Silence would be indistinguishable from a crash."""
    return {
        "is_audit_complete": True,
        "confidence_score": 1.0,
        "report": ComplianceReport(
            risk_rating="Low",
            flagged_wires=[],
            applicable_regulations=[],
            audit_summary=prompts.NO_FINDINGS_SUMMARY.format(
                wire_count=len(state["wires"]), batch=state["batch_path"] # type: ignore
            ),
            source_document_hashes=[],
        ),
    }


# --- 4. AUDIT ---------------------------------------------------------------------------


def tiers_for(candidates: list[Candidate]) -> list[int]:
    return CROSS_BORDER_TIERS if any(c.is_cross_border for c in candidates) else BASE_TIERS


def audit_node(state: AgentState) -> dict[str, Any]:
    """Translate each candidate's geometry into obligation-shaped questions and retrieve.

    On a refinement pass the critic's reformulated query is used instead, and its results are
    *added* to what is already held -- the loop is there to fill a gap, not to trade one
    incomplete context for another. Its hits fuse into the running RRF score the same way, so a
    clause the first pass already found is reinforced rather than re-scored from scratch.
    """
    queries = list(state.get("queries", []))
    if state.get("loop_count", 0) and state.get("critique"):
        new_queries = [state["critique"]] # type: ignore
    else:
        new_queries = []
        for candidate in state["candidates"]: # type: ignore
            for query in prompts.obligation_queries(candidate):
                if query not in new_queries:
                    new_queries.append(query)

    tiers = tiers_for(state["candidates"]) # type: ignore
    documents = {d.metadata["chunk_id"]: d for d in state.get("retrieved_context", [])}
    refining = bool(state.get("loop_count", 0)) and bool(state.get("critique"))
    reserved: list[str] = []
    for query in new_queries:
        hits = retrieve(query, k=RETRIEVE_K, tiers=tiers)
        if USE_RERANKER:
            hits = rerank(query, hits)
        for rank, hit in enumerate(hits, start=1):
            if refining and rank <= REFINEMENT_RESERVE:
                reserved.append(hit["chunk_id"])
            document = documents.get(hit["chunk_id"])
            if document is None:
                document = Document(
                    page_content=hit["text"],
                    metadata={k: v for k, v in hit.items() if k != "text"},
                )
                documents[hit["chunk_id"]] = document
            document.metadata["rrf"] = document.metadata.get("rrf", 0.0) + 1 / (RRF_K + rank)
            # A clause several queries return keeps the best distance it earned, not whichever
            # query happened to reach it first. The distance no longer decides the order, but it
            # is reported, and reporting the worse of two scores for no reason is a lie.
            document.metadata["distance"] = min(
                document.metadata.get("distance", hit["distance"]), hit["distance"]
            )

    if not documents:
        # §4.2's third fallback. Drafting against an empty regulations block would produce a SAR
        # that cites nothing and looks confident doing it; with 12,273 chunks indexed this state
        # means the store is unreachable or the tier filter excluded everything, which is a
        # fault, not a finding. `no_findings_node` already covers "nothing to report".
        raise ValueError(
            f"{len(new_queries)} queries against tiers {tiers} retrieved no clauses at all -- "
            f"the vector store is empty or unreachable"
        )

    ranked = sorted(documents.values(), key=lambda d: d.metadata.get("rrf", 0.0), reverse=True)
    if reserved:
        held = [documents[chunk_id] for chunk_id in dict.fromkeys(reserved)]
        ranked = held + [d for d in ranked if d.metadata["chunk_id"] not in set(reserved)]
    return {
        "queries": queries + new_queries,
        "retrieved_context": ranked[:MAX_CONTEXT_CLAUSES],
    }


# --- 5. DRAFT + CRITIC -------------------------------------------------------------------


def draft_node(state: AgentState) -> dict[str, Any]:
    feedback = ""
    if state.get("critique") and state.get("compliance_draft"):
        feedback = prompts.REDRAFT_FEEDBACK.format(
            critique=state["critique"], # type: ignore
            unsupported="\n".join(f"- {claim}" for claim in state.get("reservations", [])) or "-",
        )

    response = _model("AUDIT_MODEL", "gpt-4o").invoke(
        [
            ("system", prompts.DRAFT_SYSTEM),
            (
                "user",
                prompts.DRAFT_USER.format(
                    batch=state["batch_path"], # type: ignore
                    wire_count=len(state["wires"]), # type: ignore
                    candidate_count=len(state["candidates"]), # type: ignore
                    candidates=prompts.render_candidates(state["candidates"]), # type: ignore
                    context=prompts.render_context(state["retrieved_context"]), # type: ignore
                    feedback=feedback,
                ),
            ),
        ],
        config=trace_config(state, "draft"), # type: ignore
    )
    return {"compliance_draft": response.content}


def cited_chunk_ids(draft: str, documents: list[Document]) -> list[str]:
    """The chunk_id of every retrieved clause the draft actually cites.

    Derived rather than requested. Asked for it directly, the model returned an empty list while
    the draft plainly cited 14.2.3.Guidance.1. -- and §6.4's citations drawer resolves each hash
    back to a stored chunk, so an empty list means an auditor cannot check a single claim. The
    draft text and the retrieved set are both in hand here; this is a lookup, not a judgement.
    """
    return [
        document.metadata["chunk_id"]
        for document in documents
        if (clause := str(document.metadata.get("section_clause", "")).strip())
        and clause.rstrip(".") in draft
    ]


def fabricated_citations(draft: str, documents: list[Document]) -> list[str]:
    """Citations in the draft that match no retrieved clause. The hard half of the critic.

    A model that cites "AML Rulebook 12.4.1" when nothing of the sort was retrieved has invented
    the authority for its own finding. That is the one failure a self-assessed confidence score
    reliably misses, because from inside the draft the citation looks perfectly well-formed.
    """
    clauses = {
        str(document.metadata.get("section_clause", "")).strip()
        for document in documents
    }
    clauses.discard("")
    return [
        cited
        for cited in CITATION.findall(draft)
        if not any(clause in cited for clause in clauses)
    ]


def critic_node(state: AgentState) -> dict[str, Any]:
    draft = state["compliance_draft"] # type: ignore
    fabricated = fabricated_citations(draft, state["retrieved_context"]) # type: ignore

    verdict = _model("AUDIT_MODEL", "gpt-4o").with_structured_output(prompts.Critique).invoke(
        [
            ("system", prompts.CRITIC_SYSTEM),
            (
                "user",
                prompts.CRITIC_USER.format(
                    candidates=prompts.render_candidates(state["candidates"]), # type: ignore
                    context=prompts.render_context(state["retrieved_context"]), # type: ignore
                    draft=draft,
                ),
            ),
        ],
        config=trace_config(state, "critic"), # type: ignore
    )

    score = verdict.confidence_score # type: ignore
    reservations = list(verdict.unsupported_claims) # type: ignore
    if fabricated:
        # The gate overrides the model. Not a penalty applied to its score -- a veto.
        score = 0.0
        reservations = [
            f"cites {cited!r}, which is not among the retrieved clauses" for cited in fabricated
        ] + reservations

    return {
        "loop_count": state.get("loop_count", 0) + 1,
        "confidence_score": score,
        "critique": verdict.refined_query or verdict.reasoning, # type: ignore
        "reservations": reservations,
        "is_audit_complete": score >= CONFIDENCE_THRESHOLD,
    }


def route_after_critic(state: AgentState) -> str:
    """Loop back to retrieval while the draft is thin and we have refinements left."""
    if state["confidence_score"] >= CONFIDENCE_THRESHOLD: # type: ignore
        return "generate"
    if state["loop_count"] >= MAX_REFINEMENTS: # type: ignore
        return "generate"  # graceful give-up; the reservations are carried into the report
    return "audit"


# --- 6. GENERATE --------------------------------------------------------------------------

# gpt-4o will emit up to 16,384 output tokens, and a formatting step asked to restate a draft has
# no business approaching that. Bounding it turns a runaway into a fast failure the fallback can
# absorb, rather than a minute of billed generation that ends in an exception.
GENERATE_MAX_TOKENS = 4096


def fallback_report(state: AgentState) -> ComplianceReport:
    """The filing, assembled in Python when the formatting call fails.

    Deliberately conservative on the one field that is a judgement: risk_rating is read off the
    critic's confidence rather than guessed at, and never reaches High -- a report produced by a
    fallback path is not the place to assert that a SAR must be filed.
    """
    confidence = state.get("confidence_score", 0.0) or 0.0
    summary = state.get("compliance_draft") or "No draft was produced."
    if state.get("reservations"):
        summary += "\n\n### Reservations\n" + "\n".join(
            f"- {claim}" for claim in state["reservations"] # type: ignore
        )
    summary += (
        "\n\n_This report was assembled from the approved draft after the formatting step "
        "failed. The analysis is unchanged; only its presentation was not model-written._"
    )
    return ComplianceReport(
        risk_rating="Medium" if confidence >= CONFIDENCE_THRESHOLD else "Low",
        flagged_wires=[],
        applicable_regulations=[
            f"{d.metadata.get('document_title')} {d.metadata.get('section_clause')}"
            for d in state.get("retrieved_context", []) # type: ignore
            if str(d.metadata.get("section_clause", "")).rstrip(".") in summary
        ],
        audit_summary=summary,
        source_document_hashes=[],
    )



def generate_node(state: AgentState) -> dict[str, Any]:
    citations = "\n".join(
        f"{d.metadata['chunk_id']} -> {d.metadata.get('document_title')} "
        f"{d.metadata.get('section_clause')}"
        for d in state["retrieved_context"] # type: ignore
    )
    reservations = ""
    if not state.get("is_audit_complete") and state.get("reservations"):
        reservations = (
            "\n\nUNRESOLVED AFTER REVIEW -- record these in audit_summary as reservations:\n"
            + "\n".join(f"- {claim}" for claim in state["reservations"]) # type: ignore
        )

    try:
        report = (
            _model("AUDIT_MODEL", "gpt-4o", max_tokens=GENERATE_MAX_TOKENS)
            .with_structured_output(ComplianceReport)
            .invoke(
                [
                    ("system", prompts.GENERATE_SYSTEM),
                    (
                        "user",
                        prompts.GENERATE_USER.format(
                            batch=state["batch_path"], # type: ignore
                            draft=state["compliance_draft"], # type: ignore
                            citations=citations,
                            reservations=reservations,
                        ),
                    ),
                ],
                config=trace_config(state, "generate"), # type: ignore
            )
        )
    except Exception:  # noqa: BLE001 -- see below; the specific class is provider-internal
        # A live June run died here: the model ran to the 16,384-token output ceiling emitting
        # JSON that never closed, and the whole run was lost *after* four paid calls had already
        # succeeded. GENERATE_MAX_TOKENS makes that failure cheap; this makes it non-fatal.
        #
        # Falling back costs nothing in fidelity because this node was never the analysis. The
        # draft is the finding, and every evidence field is recomputed below from the ledger and
        # the retrieved set anyway -- so a Python-built report differs from the model's only in
        # prose it was told to copy across verbatim.
        report = fallback_report(state)

    # Both evidence fields are repaired from the ledger and the retrieved set rather than trusted
    # as written. A model asked for identifiers reaches for the numbers most present in its
    # context, which is how account 6123421761 ended up in a field specified as wire references.
    report.source_document_hashes = cited_chunk_ids(state["compliance_draft"], state["retrieved_context"]) # type: ignore

    # Third repair, and the one the live runs forced. The rating is the model's reading of the
    # draft alone: GENERATE_SYSTEM says High when a cited clause imposes a reporting obligation,
    # and the model applies that faithfully to a draft the critic has already called thin. Across
    # the four batches the result was anti-correlated with the truth -- clean May (0 laundering)
    # came back High recommending a SAR, July (23 laundering) came back Low. High means *file*,
    # so it has to clear the critic's top band, not merely score enough to stop the loop.
    if report.risk_rating == "High" and state.get("confidence_score", 0.0) < HIGH_RISK_CONFIDENCE: # type: ignore
        report.risk_rating = "Medium" # type: ignore

    known = {wire.reference for wire in state["wires"]} # type: ignore
    flagged = [reference for reference in report.flagged_wires if reference in known] # type: ignore
    if not flagged:
        # Nothing usable came back. Fall back to the wires of every candidate the draft named,
        # so the report points at transactions rather than at nothing.
        flagged = [
            reference
            for candidate in state["candidates"] # type: ignore
            if candidate.anchor in state["compliance_draft"] # type: ignore
            for reference in candidate.references
        ]
    report.flagged_wires = list(dict.fromkeys(flagged)) # type: ignore

    return {"report": report, "is_audit_complete": True}
