"""The reasoning core -- LLD §5.1's five nodes, the faithfulness gate and the loop.

No key and no network: `GroundingNode` and `CriticNode` take a model factory, and `RetrievalNode`
takes a retriever, so every property here is asserted against the real node with a stub in the one
place that would otherwise cost money.

What these tests are actually defending is the set of failures the pre-migration system shipped
with: a fabricated citation that a self-assessed score called faithful, a batch-wide veto where one
thin finding sent every candidate back through retrieval, and a model-written risk rating that came
back anti-correlated with the truth.
"""

from __future__ import annotations

import pytest

from src.config import get_config
from src.graph import nodes
from src.graph.graph import build_graph, recursion_limit
from src.retrieval.retriever import RetrievalNotes
from src.models import (
    Candidate,
    Critique,
    DraftFinding,
    RetrievalResult,
    RuleChunk,
    initial_state,
)

REASONING = get_config().reasoning


# --- fixtures ---------------------------------------------------------------------------------


def chunk(chunk_id: str, *, authority="binding", tier="regulation", section="§ 1020.320(a)"):
    return RuleChunk(
        chunk_id=chunk_id,
        text=f"The text of {section}, long enough to be a clause rather than a heading.",
        tier=tier,
        authority=authority,
        source_id="31cfr1020.320",
        section_ref=section,
    )


OBLIGATION = chunk("oblig-1")
INDICATOR = chunk("indic-1", authority="illustrative", tier="guidance", section="Appendix F 3")
BUNDLE = RetrievalResult(obligations=[OBLIGATION], indicators=[INDICATOR])


def candidate(suffix: str = "a", *, refs: list[str] | None = None, pattern="structuring"):
    return Candidate(
        candidate_id=f"{pattern}:acct-{suffix}:0000000000",
        pattern_type=pattern,
        member_txn_refs=refs or [f"FGO2306010000{suffix}1", f"FGO2306010000{suffix}2",
                                 f"FGO2306010000{suffix}3"],
        attributes={"threshold": 10000, "anchor": "6123421761"},
        detection_confidence=0.6,
    )


def draft_for(target: Candidate, **overrides):
    payload = dict(
        candidate_id=target.candidate_id,
        risk_level="medium",
        narrative="Three transfers of $9,200 each within eleven days sit just below the $10,000 "
                  "reporting threshold.",
        matched_indicator_ids=[INDICATOR.chunk_id],
        cited_obligation_ids=[OBLIGATION.chunk_id],
        self_confidence=0.8,
    )
    payload.update(overrides)
    return DraftFinding(**payload)


class StubModel:
    """A `with_structured_output` model. Records what it was asked, returns what it was told to."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


class StubRetriever:
    def __init__(self, result=BUNDLE, error=None):
        self.result, self.error = result, error
        self.queries: list[str | None] = []

    def retrieve(self, target, *, hint=None):
        self.queries.append(hint)
        if self.error is not None:
            raise self.error
        return self.result, RetrievalNotes()


def state_with(*candidates, **overrides):
    state = initial_state(
        batch_id="2023-06.pdf", run_id="run-test", period="2023-06", records=[]
    )
    state.update(candidates=list(candidates), **overrides)
    return state


# --- detection --------------------------------------------------------------------------------


def test_no_candidates_sets_the_clean_flag_and_routes_past_every_model():
    update = nodes.DetectionNode(detector=lambda records: [])(state_with(records=[]))
    assert update["clean_flag"] is True
    assert nodes.route_after_detection(update) == "report"


def test_candidates_route_into_retrieval():
    found = [candidate("a")]
    update = nodes.DetectionNode(detector=lambda records: found)(state_with(records=[]))
    assert update["clean_flag"] is False and update["current_index"] == 0
    assert nodes.route_after_detection(update) == "retrieval"


# --- retrieval --------------------------------------------------------------------------------


def test_the_refinement_hint_reaches_the_retriever():
    """The loop is only worth its cost if the reformulated question is actually asked."""
    retriever = StubRetriever()
    target = candidate("a")
    node = nodes.RetrievalNode(retriever=retriever)
    node(state_with(target))
    node(state_with(target, refinement_hint="obligation to report structured transfers"))
    assert retriever.queries == [None, "obligation to report structured transfers"]


def test_a_retrieval_failure_is_a_note_not_an_exception():
    """Per-candidate isolation starts here: one candidate's store error must not end the batch."""
    node = nodes.RetrievalNode(retriever=StubRetriever(error=RuntimeError("chroma is gone")))
    update = node(state_with(candidate("a")))
    assert update["retrieval"].is_empty
    assert any(nodes.RETRIEVAL_FAILED in note for note in update["review_notes"])


# --- grounding --------------------------------------------------------------------------------


def test_an_empty_bundle_is_never_grounded_and_never_billed():
    """A SAR citing nothing, written confidently, is the worst output this system could produce."""
    model = StubModel(draft_for(candidate("a")))
    node = nodes.GroundingNode(model_factory=lambda: model)
    update = node(state_with(candidate("a"), retrieval=RetrievalResult()))
    assert update["draft_finding"] is None
    assert model.calls == [], "the model must not be constructed for an empty bundle"


def test_the_drafts_candidate_id_is_overwritten_not_trusted():
    """A draft filed against the wrong candidate attaches a narrative to the wrong transactions,
    and nothing downstream would notice."""
    target = candidate("a")
    wrong = draft_for(target, candidate_id="structuring:someone-else:9999999999")
    node = nodes.GroundingNode(model_factory=lambda: StubModel(wrong))
    update = node(state_with(target, retrieval=BUNDLE))
    assert update["draft_finding"].candidate_id == target.candidate_id


def test_off_schema_output_is_re_prompted_with_the_validation_error():
    from pydantic import ValidationError

    try:
        DraftFinding(candidate_id="x", risk_level="medium", narrative="")
    except ValidationError as error:
        failure = error

    target = candidate("a")
    model = StubModel(failure, draft_for(target))
    node = nodes.GroundingNode(model_factory=lambda: model)
    update = node(state_with(target, retrieval=BUNDLE))

    assert update["draft_finding"] is not None
    assert len(model.calls) == 2
    repair = model.calls[1][-1]
    assert repair[0] == "user" and "did not validate" in repair[1]


def test_persistent_schema_failure_gives_up_with_a_note_rather_than_raising():
    from pydantic import ValidationError

    try:
        DraftFinding(candidate_id="x", risk_level="medium", narrative="")
    except ValidationError as error:
        failure = error

    model = StubModel(failure)
    update = nodes.GroundingNode(model_factory=lambda: model)(
        state_with(candidate("a"), retrieval=BUNDLE)
    )
    assert update["draft_finding"] is None
    assert len(model.calls) == REASONING.schema_retries
    assert any(nodes.SCHEMA_PARSE_FAILURE in note for note in update["review_notes"])


def test_a_transport_failure_is_one_candidates_problem():
    model = StubModel(TimeoutError("the model did not answer in 60s"))
    update = nodes.GroundingNode(model_factory=lambda: model)(
        state_with(candidate("a"), retrieval=BUNDLE)
    )
    assert update["draft_finding"] is None
    assert len(model.calls) == 1, "a timeout is not a schema error; it is not re-prompted"
    assert any(nodes.LLM_CALL_FAILED in note for note in update["review_notes"])


# --- the faithfulness gate --------------------------------------------------------------------


def test_a_citation_that_was_never_retrieved_is_a_fabrication():
    target = candidate("a")
    invented = draft_for(target, cited_obligation_ids=["31cfr9999.999:invented"])
    assert nodes.fabricated_ids(invented, BUNDLE) == ["31cfr9999.999:invented"]
    assert nodes.fabricated_ids(draft_for(target), BUNDLE) == []


def test_a_transaction_the_candidate_does_not_cover_is_a_fabrication():
    target = candidate("a")
    assert nodes.fabricated_references(
        f"Transfers {target.member_txn_refs[0]} and FGO23069999999 were structured.", target
    ) == ["FGO23069999999"]
    assert nodes.fabricated_references(
        f"Transfer {target.member_txn_refs[0]} was structured.", target
    ) == []


def test_the_gate_vetoes_before_the_critic_model_is_reached():
    """Not a penalty on the score -- a veto. There is no number a judge could return that would
    make a fabricated citation acceptable, so paying for one buys nothing."""
    target = candidate("a")
    model = StubModel(Critique(score=1.0, reason="looks great to me"))
    node = nodes.CriticNode(model_factory=lambda: model)
    update = node(state_with(
        target, retrieval=BUNDLE,
        draft_finding=draft_for(target, cited_obligation_ids=["invented-clause"]),
    ))
    assert model.calls == [], "the gate must run before the model"
    assert update["confidence_score"] == 0.0
    assert update["refinement_hint"], "a vetoed draft loops rather than being accepted"


def test_an_unfaithful_draft_can_never_be_accepted_however_it_loops():
    """The phase's hardest criterion: the gate blocks it, the candidate loops, and it finalises
    needs_review. There is no path from a fabricated citation to a filed finding."""
    target = candidate("a")
    model = StubModel(Critique(score=1.0, reason="looks great to me"))
    node = nodes.CriticNode(model_factory=lambda: model)

    state = state_with(target, retrieval=BUNDLE,
                       draft_finding=draft_for(target, cited_obligation_ids=["invented"]))
    for expected_loop in range(1, REASONING.max_loops + 1):
        update = node(state)
        assert update["loop_count"] == expected_loop
        assert not update.get("findings"), "nothing may be filed while the gate is failing"
        state.update(update)

    final = node(state)
    assert model.calls == []
    (finding,) = final["findings"]
    assert finding.status == "needs_review"
    assert any(nodes.FAITHFULNESS_CHECK_FAILED in note for note in finding.review_notes)
    assert final["current_index"] == 1, "the candidate is finished with, not retried forever"


# --- the critic's exits -----------------------------------------------------------------------


def test_a_well_grounded_draft_is_accepted_with_its_citations_resolved():
    target = candidate("a")
    model = StubModel(Critique(score=0.9, reason="every claim rests on a provided excerpt"))
    update = nodes.CriticNode(model_factory=lambda: model)(
        state_with(target, retrieval=BUNDLE, draft_finding=draft_for(target))
    )
    (finding,) = update["findings"]
    assert finding.status == "pending_review" and finding.confidence == 0.9
    # Resolved from the bundle, not re-searched: this is the text the model actually saw.
    assert [c.chunk_id for c in finding.applicable_regulations] == [OBLIGATION.chunk_id]
    assert finding.applicable_regulations[0].text_excerpt in OBLIGATION.text
    assert [c.chunk_id for c in finding.red_flag_indicators] == [INDICATOR.chunk_id]
    assert update["refinement_hint"] is None and update["current_index"] == 1


def test_insufficient_evidence_is_taken_at_its_word_and_costs_nothing():
    target = candidate("a")
    model = StubModel(Critique(score=1.0))
    update = nodes.CriticNode(model_factory=lambda: model)(state_with(
        target, retrieval=BUNDLE, draft_finding=draft_for(target, insufficient_evidence=True),
    ))
    assert model.calls == []
    assert update["refinement_hint"], "one more retrieval is worth trying before giving up"


def test_a_thin_draft_loops_with_the_critics_own_question():
    target = candidate("a")
    model = StubModel(Critique(
        score=0.4, reason="the threshold claim cites nothing",
        unsupported_claims=["the $10,000 figure"],
        refinement_hint="obligation to report currency transactions exceeding $10,000",
    ))
    update = nodes.CriticNode(model_factory=lambda: model)(
        state_with(target, retrieval=BUNDLE, draft_finding=draft_for(target))
    )
    assert update["loop_count"] == 1
    assert update["refinement_hint"].startswith("obligation to report currency")
    assert any("the $10,000 figure" in note for note in update["review_notes"])
    assert "findings" not in update


def test_a_draft_that_never_clears_the_bar_finalises_as_needs_review_with_its_reasons():
    target = candidate("a")
    model = StubModel(Critique(score=0.4, reason="still thin", refinement_hint="try again"))
    node = nodes.CriticNode(model_factory=lambda: model)
    update = node(state_with(target, retrieval=BUNDLE, draft_finding=draft_for(target),
                             loop_count=REASONING.max_loops))
    (finding,) = update["findings"]
    assert finding.status == "needs_review" and finding.review_notes


def test_a_candidate_grounding_could_not_draft_still_produces_a_finding():
    """Silence about a detected shape is indistinguishable from not having looked."""
    target = candidate("a")
    update = nodes.CriticNode(model_factory=lambda: StubModel(Critique(score=1.0)))(
        state_with(target, retrieval=BUNDLE, draft_finding=None,
                   review_notes=["no obligations or indicators were retrieved"])
    )
    (finding,) = update["findings"]
    assert finding.status == "needs_review"
    assert finding.risk_level == "medium", "ungroundable is unresolved, not low risk"
    assert finding.candidate.candidate_id == target.candidate_id


# --- report generation ------------------------------------------------------------------------


def report_from(*findings, records=500, quarantined=0, period="2023-06"):
    state = state_with(period=period, records=[None] * records, findings=list(findings),
                       quarantined_count=quarantined)
    return nodes.ReportGenerationNode()(state)["report"]


def accepted(target, *, risk="medium", confidence=0.9, status="pending_review"):
    from src.models import Citation, Finding

    return Finding(
        finding_id=f"f-{target.candidate_id}",
        candidate=target,
        risk_level=risk,
        narrative="A grounded narrative.",
        applicable_regulations=[Citation.from_chunk(OBLIGATION)],
        red_flag_indicators=[Citation.from_chunk(INDICATOR)],
        confidence=confidence,
        status=status,
        review_notes=["thin"] if status == "needs_review" else [],
    )


def test_a_clean_batch_gets_a_report_that_says_so():
    report = report_from()
    assert report.clean is True and report.risk_rating == "none"
    assert "No qualifying pattern was found" in report.summary
    assert not report.flagged_transactions

def test_high_risk_requires_the_confidence_bar_because_high_means_file():
    """The pre-migration system let the model rate the batch, and clean May came back High
    recommending a SAR while July's 23 planted patterns came back Low."""
    target = candidate("a")
    below = REASONING.high_risk_min_confidence - 0.05
    assert report_from(accepted(target, risk="high", confidence=below)).risk_rating == "medium"
    assert report_from(
        accepted(target, risk="high", confidence=REASONING.high_risk_min_confidence)
    ).risk_rating == "high"


def test_a_needs_review_finding_cannot_carry_the_batch_to_high():
    target = candidate("a")
    report = report_from(accepted(target, risk="high", confidence=1.0, status="needs_review"))
    assert report.risk_rating == "medium"


def test_every_flagged_transaction_traces_to_a_finding():
    """Enforced by the contract, asserted here because the old system put account numbers in
    this field."""
    first, second = candidate("a"), candidate("b")
    report = report_from(accepted(first), accepted(second))
    assert set(report.flagged_transactions) == set(
        first.member_txn_refs + second.member_txn_refs
    )


def test_the_source_list_is_deduplicated_across_findings():
    report = report_from(accepted(candidate("a")), accepted(candidate("b")))
    ids = [c.chunk_id for c in report.source_document_refs]
    assert sorted(ids) == [INDICATOR.chunk_id, OBLIGATION.chunk_id]


def test_quarantined_messages_are_stated_not_buried():
    """A month whose report is clean because a third of it failed to parse is not clean."""
    assert "12 message(s) could not be parsed" in report_from(quarantined=12).summary
    assert report_from(accepted(candidate("a")), quarantined=12).quarantined_count == 12


def test_the_summary_names_why_a_finding_needs_review():
    report = report_from(accepted(candidate("a"), status="needs_review"))
    assert "Why this needs review" in report.summary
    assert report.needs_review_count == 1


# --- the graph ---------------------------------------------------------------------------------


def run_graph(*candidates, critiques, retriever=None):
    """The real graph, real nodes, stub models."""
    graph = build_graph(
        detection=nodes.DetectionNode(detector=lambda records: list(candidates)),
        retrieval=nodes.RetrievalNode(retriever=retriever or StubRetriever()),
        grounding=nodes.GroundingNode(
            model_factory=lambda: StubModel(*[draft_for(c) for c in candidates] or [None])
        ),
        critic=nodes.CriticNode(model_factory=lambda: StubModel(*critiques)),
    )
    state = initial_state(batch_id="2023-06.pdf", run_id="run-graph", period="2023-06", records=[])
    return graph.invoke(state, config={"recursion_limit": recursion_limit(len(candidates), 2)})


def test_the_graph_walks_every_candidate_and_reports_once():
    first, second = candidate("a"), candidate("b")
    final = run_graph(first, second, critiques=[Critique(score=0.9)])
    assert final["current_index"] == 2
    assert [f.candidate.candidate_id for f in final["findings"]] == [
        first.candidate_id, second.candidate_id
    ]
    assert final["report"].risk_rating == "medium" and final["is_complete"] is True


def test_a_clean_batch_reaches_the_report_without_entering_retrieval():
    retriever = StubRetriever()
    graph = build_graph(
        detection=nodes.DetectionNode(detector=lambda records: []),
        retrieval=nodes.RetrievalNode(retriever=retriever),
        grounding=nodes.GroundingNode(model_factory=lambda: StubModel()),
        critic=nodes.CriticNode(model_factory=lambda: StubModel()),
    )
    state = initial_state(batch_id="2023-05.pdf", run_id="r", period="2023-05", records=[])
    final = graph.invoke(state)
    assert retriever.queries == []
    assert final["report"].clean is True


def test_one_candidate_failing_leaves_its_neighbours_findings_intact():
    """Per-candidate isolation, which is the point of the whole phase. The middle candidate's
    draft cites a clause that was never retrieved; the other two must still be filed."""
    first, bad, last = candidate("a"), candidate("b"), candidate("c")
    drafts = {
        first.candidate_id: draft_for(first),
        bad.candidate_id: draft_for(bad, cited_obligation_ids=["invented-clause"]),
        last.candidate_id: draft_for(last),
    }

    class ByCandidate:
        """Answers for whichever candidate is under review, which the trace metadata names."""

        def invoke(self, messages, config=None):
            index = config["metadata"]["candidate_index"] if config else 0
            return list(drafts.values())[index]

    graph = build_graph(
        detection=nodes.DetectionNode(detector=lambda records: [first, bad, last]),
        retrieval=nodes.RetrievalNode(retriever=StubRetriever()),
        grounding=nodes.GroundingNode(model_factory=ByCandidate),
        critic=nodes.CriticNode(model_factory=lambda: StubModel(Critique(score=0.95))),
    )
    state = initial_state(batch_id="2023-06.pdf", run_id="r", period="2023-06", records=[])
    final = graph.invoke(state, config={"recursion_limit": recursion_limit(3, 2)})

    by_id = {f.candidate.candidate_id: f for f in final["findings"]}
    assert len(by_id) == 3, "every candidate is accounted for, whatever happened to it"
    assert by_id[first.candidate_id].status == "pending_review"
    assert by_id[last.candidate_id].status == "pending_review"
    assert by_id[bad.candidate_id].status == "needs_review"
    assert any(nodes.FAITHFULNESS_CHECK_FAILED in note
               for note in by_id[bad.candidate_id].review_notes)
    # And the neighbours' narratives survive untouched.
    assert by_id[first.candidate_id].applicable_regulations


def test_the_step_budget_is_sized_to_the_batch_not_left_at_the_default():
    """LangGraph's default of 25 is exceeded by the fourth candidate, which on the 10k batch would
    be a GraphRecursionError reported as a fault when nothing had gone wrong."""
    assert recursion_limit(0, 2) == 25
    assert recursion_limit(304, 2) > 25 * 100


# --- tracing (HLD §6) --------------------------------------------------------------------------


def test_every_run_carries_one_searchable_id():
    """One id shared by every span, so a run can be found again afterwards."""
    from src.graph.graph import new_run_id, run_config

    config = run_config(run_id="run-abc", batch_id="2023-06.pdf", tags=["NIGHTLY"])
    assert "AML_AUDIT_RUN" in config["tags"] and "NIGHTLY" in config["tags"]
    assert config["metadata"] == {"run_id": "run-abc", "batch_id": "2023-06.pdf"}
    assert new_run_id() != new_run_id()


def test_the_run_config_cannot_carry_what_it_does_not_know_yet():
    """Candidate counts do not exist at invoke() -- detection has not run. Anything read from
    state before the call would be zero for every run, which is worse than absent."""
    from src.graph.graph import run_config

    metadata = run_config(run_id="r", batch_id="b")["metadata"]
    assert "candidate_count" not in metadata and "pattern_type" not in metadata


def test_a_model_call_names_the_candidate_it_was_made_for():
    """The question a trace exists to answer here is *which candidate on which pass*, so the
    per-call metadata has to carry both."""
    target = candidate("a")
    config = nodes.trace_config(
        state_with(target, candidate("b"), current_index=1, loop_count=1, run_id="run-x"), "critic"
    )
    assert config["tags"] == ["node:critic", "loop:1"]
    assert config["metadata"]["candidate_index"] == 1
    assert config["metadata"]["candidate_id"] == candidate("b").candidate_id
    assert config["metadata"]["candidate_count"] == 2
    assert config["metadata"]["run_id"] == "run-x"


def test_the_two_passes_of_a_looping_candidate_are_distinguishable_in_the_trace():
    """A constant config would make the two attempts indistinguishable, and *which context caused
    the loop* is then unanswerable from the trace."""
    target = candidate("a")
    first = nodes.trace_config(state_with(target, loop_count=0), "grounding")
    second = nodes.trace_config(state_with(target, loop_count=1), "grounding")
    assert [first["tags"][1], second["tags"][1]] == ["loop:0", "loop:1"]


def test_trace_config_survives_an_index_past_the_last_candidate():
    """Report generation runs with the index already advanced past the end."""
    config = nodes.trace_config(state_with(candidate("a"), current_index=1), "report")
    assert config["metadata"]["candidate_id"] == ""


def test_tracing_reports_itself_as_off_without_a_key(monkeypatch):
    """A trace that is silently not being written is worse than none: you go looking for it after
    the run instead of before."""
    from src.graph.graph import tracing_project

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert tracing_project() is None

    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-fake")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "finguard-orchestrator")
    assert tracing_project() == "finguard-orchestrator"

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    assert tracing_project() is None


def test_a_chunk_id_invented_in_the_prose_is_caught_too():
    """A live June run cited ids inside the narrative as well as in the structured lists. The prose
    is the half an analyst reads, so leaving it ungated would gate the half that cannot mislead."""
    target = candidate("a")
    invented = draft_for(
        target,
        narrative="This aligns with red-flag indicator [ffiec-appendix-f:0123456789abcdef], "
                  f"alongside [{INDICATOR.chunk_id}].",
    )
    assert nodes.fabricated_ids(invented, BUNDLE) == ["ffiec-appendix-f:0123456789abcdef"]


def test_a_narrative_citing_only_what_it_was_shown_passes():
    target = candidate("a")
    honest = draft_for(
        target,
        narrative=f"The obligation [{OBLIGATION.chunk_id}] applies, illustrated by "
                  f"[{INDICATOR.chunk_id}].",
    )
    assert nodes.fabricated_ids(honest, BUNDLE) == []
