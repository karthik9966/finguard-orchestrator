"""Node contracts, the citation gate and routing (§4.1, §4.2).

Every model call and every retrieval is stubbed. The suite needs no API key, touches no network
and costs nothing -- which is the only way an assertion about the critic's *veto* can be run on
every commit rather than admired once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.graph import nodes, prompts
from src.graph.graph import build_graph
from src.graph.state import (
    CONFIDENCE_THRESHOLD,
    HIGH_RISK_CONFIDENCE,
    MAX_REFINEMENTS,
    ComplianceReport,
    initial_state,
)
from src.utils.detectors import CONCENTRATION, MAGNITUDE, PATH, detect
from tests.test_detectors import chain, wire

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
BATCH = LEDGER / "2023-06_private_banking_log.pdf"
needs_ledger = pytest.mark.skipif(
    not BATCH.exists(), reason="run: uv run python -m src.utils.pdf_generator"
)

CLAUSE = "14.2.3.Guidance.1."
CHUNK_ID = "obliqa:1:14.2.3.Guidance.1.:a3f9c210"


def clause_doc(chunk_id: str = CHUNK_ID, clause: str = CLAUSE, distance: float = 0.31) -> Document:
    return Document(
        page_content="A Relevant Person must report transactions structured to avoid detection.",
        metadata={
            "chunk_id": chunk_id,
            "section_clause": clause,
            "document_title": "AML Rulebook",
            "relevance_tier": 1,
            "distance": distance,
        },
    )


class StubResponse:
    def __init__(self, content: str):
        self.content = content


class StubModel:
    """Stands in for ChatOpenAI. Returns whatever the test queued, and records the prompts."""

    def __init__(self, text: str = "draft", structured=None, calls: list | None = None,
                 configs: list | None = None):
        self.text = text
        self.structured = structured
        self.calls = calls if calls is not None else []
        self.configs = configs if configs is not None else []
        self._schema = None

    def with_structured_output(self, schema):
        clone = StubModel(self.text, self.structured, self.calls, self.configs)
        clone._schema = schema
        return clone

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        self.configs.append(config)
        if self._schema is None:
            return StubResponse(self.text(len(self.calls)) if callable(self.text) else self.text)
        value = self.structured
        return value(len(self.calls)) if callable(value) else value


@pytest.fixture
def stub_retrieval(monkeypatch):
    """One clause for every query, so the audit node can be exercised without the vector store."""
    seen: list[dict] = []

    def fake(query, *, k=15, tiers=None, backend_name=None):
        seen.append({"query": query, "k": k, "tiers": tiers})
        return [
            {
                "chunk_id": CHUNK_ID,
                "text": "A Relevant Person must report transactions structured to avoid detection.",
                "distance": 0.31,
                "section_clause": CLAUSE,
                "document_title": "AML Rulebook",
                "relevance_tier": 1,
            },
            {
                "chunk_id": "finra:regulatory-notice-19-18:7",
                "text": "The customer breaks funds transfers into smaller transfers.",
                "distance": 0.44,
                "section_clause": "part 7 of 31",
                "document_title": "FINRA Regulatory Notice 19-18",
                "relevance_tier": 1,
            },
        ]

    monkeypatch.setattr(nodes, "retrieve", fake)
    return seen


def sample_candidates():
    wires = [wire(f"R{i}", f"S{i}", "COLLECTOR", "5000", i + 1) for i in range(4)]
    return detect(wires)


# --- 1. PARSE ----------------------------------------------------------------------------


@needs_ledger
def test_parse_node_reads_the_batch_without_a_model(monkeypatch):
    monkeypatch.setattr(nodes, "_model", lambda *a, **k: pytest.fail("PARSE must not call a model"))
    update = nodes.parse_node(initial_state(str(BATCH)))
    assert len(update["wires"]) == 220
    assert update["extraction_failures"] == []
    assert update["extracted_entities"]["parsed_messages"] == 220
    assert update["extracted_entities"]["statement_reference"] == "NPB-LOG-2023-06"


@needs_ledger
def test_a_refused_message_is_escalated_to_the_model(monkeypatch, tmp_path):
    """The fallback route: one message goes to the model, the other 219 stay free."""
    source = BATCH.with_suffix(".txt")
    corrupted = tmp_path / source.name
    corrupted.write_text(source.read_text().replace(":32A:", ":32A:XX", 1))

    escalations: list = []

    def fake_escalate(failure):
        escalations.append(failure)
        return None

    monkeypatch.setattr(nodes, "escalate", fake_escalate)
    update = nodes.parse_node(initial_state(str(corrupted)))

    assert len(escalations) == 1, "exactly the refused message is escalated"
    assert len(update["wires"]) == 219
    assert update["extraction_failures"][0]["rescued_by_model"] is False
    assert update["extracted_entities"]["unreadable_messages"] == 1


def test_a_batch_with_no_readable_wires_fails_loudly(tmp_path):
    """An empty SAR is worse than an error: it reads as "we looked and found nothing"."""
    empty = tmp_path / "empty.txt"
    empty.write_text("NORTHGATE PRIVATE BANK\nMessages in batch   : 0\n")
    with pytest.raises(ValueError, match="no readable wires"):
        nodes.parse_node(initial_state(str(empty)))


# --- 2/3. DETECT and ROUTE ----------------------------------------------------------------


def test_detect_node_records_candidates_for_the_prompt():
    state = initial_state("x")
    state["wires"] = [wire(f"R{i}", f"S{i}", "COLLECTOR", "5000", i + 1) for i in range(4)]
    update = nodes.detect_node(state)
    assert update["candidates"]
    assert update["extracted_entities"]["candidates"][0]["shape"] == CONCENTRATION
    assert update["extracted_entities"]["wires_under_review"] == 4


def test_a_batch_with_no_candidates_skips_the_model_entirely():
    state = initial_state("x")
    state["candidates"] = []
    assert nodes.route_after_detect(state) == "no_findings"


def test_a_batch_with_candidates_goes_to_audit():
    state = initial_state("x")
    state["candidates"] = sample_candidates()
    assert nodes.route_after_detect(state) == "audit"


def test_the_clean_batch_report_is_a_negative_result_not_a_blank():
    state = initial_state("2023-05_clean.pdf")
    state["wires"] = [wire("A", "X", "Y", "5000")]
    report = nodes.no_findings_node(state)["report"]
    assert report.risk_rating == "Low"
    assert report.flagged_wires == [] and report.source_document_hashes == []
    assert "1 wires" in report.audit_summary and "screened" in report.audit_summary


# --- 4. AUDIT -----------------------------------------------------------------------------


def test_queries_are_obligations_not_descriptions():
    """Measured: the same facts as a narrative ranked the target clause 315th, as an
    obligation 5th. The templates must keep the rulebook's grammatical mood."""
    for candidate in sample_candidates():
        for query in prompts.obligation_queries(candidate):
            assert not any(char.isdigit() for char in query), f"no batch figures: {query}"
            assert candidate.anchor not in query, f"no account numbers: {query}"


def test_a_tight_run_asks_the_measured_best_question():
    tight = [wire(f"R{i}", f"S{i}", "COLLECTOR", "5673", i + 1) for i in range(4)]
    candidate = next(c for c in detect(tight) if c.shape == CONCENTRATION)
    assert candidate.coefficient_of_variation < prompts.TIGHT_AMOUNTS
    assert prompts.TIGHT_AMOUNTS_QUERY in prompts.obligation_queries(candidate)


def test_a_cross_border_candidate_widens_the_tiers():
    domestic = sample_candidates()
    assert nodes.tiers_for(domestic) == nodes.BASE_TIERS

    crossed = detect(
        [wire(f"R{i}", f"S{i}", "COLLECTOR", "5000", i + 1, receiver_country="AE")
         for i in range(4)]
    )
    assert nodes.tiers_for(crossed) == nodes.CROSS_BORDER_TIERS
    assert prompts.CROSS_BORDER_QUERY in prompts.obligation_queries(crossed[0])


def test_audit_node_deduplicates_clauses_and_ranks_them(stub_retrieval):
    state = initial_state("x")
    state["candidates"] = sample_candidates()
    update = nodes.audit_node(state)

    assert len(update["queries"]) >= 2
    assert len(update["queries"]) == len(set(update["queries"])), "no query is asked twice"
    ids = [d.metadata["chunk_id"] for d in update["retrieved_context"]]
    assert len(ids) == len(set(ids)) == 2, "the same clause from two queries is held once"
    scores = [d.metadata["rrf"] for d in update["retrieved_context"]]
    assert scores == sorted(scores, reverse=True), "fused rank decides the order, not distance"


def test_a_refinement_pass_asks_the_critics_question_and_keeps_what_it_had(stub_retrieval):
    state = initial_state("x")
    state["candidates"] = sample_candidates()
    state["loop_count"] = 1
    state["critique"] = "obligation to report onward transfer of unexplained funds"
    state["queries"] = ["the original question"]
    state["retrieved_context"] = [clause_doc("already:held:1", "9.9.9", 0.20)]

    update = nodes.audit_node(state)
    assert [q["query"] for q in stub_retrieval] == [state["critique"]]
    held = {d.metadata["chunk_id"] for d in update["retrieved_context"]}
    assert "already:held:1" in held, "refinement adds context, it does not replace it"
    assert CHUNK_ID in held


def test_a_clause_two_queries_return_keeps_the_best_distance_it_earned(monkeypatch):
    """`setdefault` used to keep whichever query reached a clause first. On the June batch that
    stored 0.4442 for the clause the report cites when it had also earned 0.4365, purely by loop
    order -- 7 of 93 clauses held a worse score than they had."""
    scores = iter([0.48, 0.31])

    def fake(query, *, k=15, tiers=None, backend_name=None):
        return [dict(clause_doc(distance=next(scores)).metadata,
                     text="A Relevant Person must report structured transactions.")]

    monkeypatch.setattr(nodes, "retrieve", fake)
    state = initial_state("x")
    state["candidates"] = sample_candidates()

    held = nodes.audit_node(state)["retrieved_context"]
    assert len(held) == 1
    assert held[0].metadata["distance"] == 0.31, "the worse score must not win on loop order"


def test_a_hard_querys_best_answer_is_not_buried_by_an_easy_querys_tail(monkeypatch):
    """Distances are measured against each query's own vector, so they do not compare across
    queries. On June the seven queries' best hits spanned 0.3433 to 0.4827 -- pooling them into
    one distance sort ranked *how easy the question was* above *how good the answer is*."""
    easy = [{"chunk_id": f"easy:{i}", "text": "t", "distance": 0.10 + i / 100,
             "section_clause": f"1.{i}", "document_title": "COBS", "relevance_tier": 1}
            for i in range(3)]
    hard = [{"chunk_id": "hard:top", "text": "t", "distance": 0.40,
             "section_clause": "14.2.3.Guidance.1.", "document_title": "AML Rulebook",
             "relevance_tier": 1}]
    answers = iter([easy, hard])
    monkeypatch.setattr(nodes, "retrieve", lambda query, **kw: next(answers))

    state = initial_state("x")
    state["candidates"] = sample_candidates()
    update = nodes.audit_node(state)
    # The premise: one easy query and one hard one. Stated so this fails loudly rather than
    # quietly inverting if the fixture ever asks a different number of questions.
    assert len(update["queries"]) == 2

    order = [d.metadata["chunk_id"] for d in update["retrieved_context"]]
    # Under the old raw-distance sort this clause came last of four, at 0.40 against 0.10-0.12.
    assert order.index("hard:top") < order.index("easy:1")
    assert order.index("hard:top") < order.index("easy:2")


def test_retrieving_nothing_at_all_stops_before_a_model_is_reached(monkeypatch):
    """§4.2's third fallback. An empty regulations block would yield a SAR that cites nothing and
    looks confident doing it -- and with 12,273 chunks indexed, this state is a broken store."""
    monkeypatch.setattr(nodes, "retrieve", lambda query, **kw: [])
    state = initial_state("x")
    state["candidates"] = sample_candidates()

    with pytest.raises(ValueError, match="retrieved no clauses at all"):
        nodes.audit_node(state)


def test_the_refinement_query_gets_seats_it_cannot_win_on_score(monkeypatch):
    """RRF accumulates, so after seven queries an incumbent holds far more score than any hit
    from the single refinement list can earn. Left to compete, the loop brings in one clause of
    fifteen and is effectively inert -- so the critic's answer is seated, not ranked."""
    incumbents = [clause_doc(f"held:{i}", f"1.{i}", 0.20) for i in range(nodes.MAX_CONTEXT_CLAUSES)]
    for doc in incumbents:
        doc.metadata["rrf"] = 0.9  # seven queries' worth of agreement

    fresh = [{"chunk_id": f"fresh:{i}", "text": "t", "distance": 0.45,
              "section_clause": f"9.{i}", "document_title": "AML Rulebook", "relevance_tier": 1}
             for i in range(nodes.RETRIEVE_K)]
    monkeypatch.setattr(nodes, "retrieve", lambda query, **kw: fresh)

    state = initial_state("x")
    state["candidates"] = sample_candidates()
    state["loop_count"] = 1
    state["critique"] = "obligation to report linked transfers above a threshold"
    state["retrieved_context"] = incumbents

    held = [d.metadata["chunk_id"] for d in nodes.audit_node(state)["retrieved_context"]]
    arrived = [c for c in held if c.startswith("fresh:")]
    assert len(arrived) == nodes.REFINEMENT_RESERVE, "the reserved seats are actually filled"
    assert held[: nodes.REFINEMENT_RESERVE] == arrived, "and they are read first, not last"


# --- 5. the citation gate -------------------------------------------------------------------


def test_a_citation_of_a_retrieved_clause_passes():
    draft = f"Wires FGO1 and FGO2 fall under [AML Rulebook {CLAUSE}]."
    assert nodes.fabricated_citations(draft, [clause_doc()]) == []


def test_a_citation_of_a_clause_that_was_never_retrieved_is_caught():
    """The failure a self-assessed score misses: from inside the draft it looks well-formed."""
    draft = "This engages [AML Rulebook 12.4.1] and requires a report."
    assert nodes.fabricated_citations(draft, [clause_doc()]) == ["AML Rulebook 12.4.1"]


def test_markdown_links_are_not_mistaken_for_citations():
    draft = "See [the ledger](data/processed/ledger.csv) for detail."
    assert nodes.fabricated_citations(draft, [clause_doc()]) == []


def test_the_gate_vetoes_a_confident_model(monkeypatch):
    """A fabricated citation fails the draft outright, whatever the critic scored it."""
    verdict = prompts.Critique(
        confidence_score=0.95, unsupported_claims=[], refined_query="", reasoning="looks great"
    )
    monkeypatch.setattr(nodes, "_model", lambda *a, **k: StubModel(structured=verdict))

    state = initial_state("x")
    state["candidates"] = sample_candidates()
    state["retrieved_context"] = [clause_doc()]
    state["compliance_draft"] = "This engages [AML Rulebook 12.4.1]."

    update = nodes.critic_node(state)
    assert update["confidence_score"] == 0.0
    assert update["is_audit_complete"] is False
    assert "not among the retrieved clauses" in update["reservations"][0]


def test_a_grounded_draft_keeps_the_models_score(monkeypatch):
    verdict = prompts.Critique(confidence_score=0.9, unsupported_claims=[], reasoning="grounded")
    monkeypatch.setattr(nodes, "_model", lambda *a, **k: StubModel(structured=verdict))

    state = initial_state("x")
    state["candidates"] = sample_candidates()
    state["retrieved_context"] = [clause_doc()]
    state["compliance_draft"] = f"Reported under [AML Rulebook {CLAUSE}]."

    update = nodes.critic_node(state)
    assert update["confidence_score"] == 0.9
    assert update["is_audit_complete"] is True
    assert update["loop_count"] == 1


# --- routing after the critic ----------------------------------------------------------------


def test_a_supported_draft_goes_straight_to_the_filing_schema():
    state = initial_state("x")
    state["confidence_score"] = CONFIDENCE_THRESHOLD
    state["loop_count"] = 1
    assert nodes.route_after_critic(state) == "generate"


def test_a_thin_draft_goes_back_to_retrieval():
    state = initial_state("x")
    state["confidence_score"] = 0.4
    state["loop_count"] = 1
    assert nodes.route_after_critic(state) == "audit"


def test_the_loop_gives_up_rather_than_spinning():
    """Phase 1 measured that 17.2% of questions have no correct clause in the top 15 at all.
    A third refinement is usually money spent looking for a clause that is not there."""
    state = initial_state("x")
    state["confidence_score"] = 0.1
    state["loop_count"] = MAX_REFINEMENTS
    assert nodes.route_after_critic(state) == "generate"


# --- 6. GENERATE -------------------------------------------------------------------------------


def generate_with(report, draft, monkeypatch, **state_extra):
    monkeypatch.setattr(nodes, "_model", lambda *a, **k: StubModel(structured=report))
    state = initial_state("x")
    state["retrieved_context"] = [clause_doc()]
    state["compliance_draft"] = draft
    state["is_audit_complete"] = True
    state.update(state_extra)
    return nodes.generate_node(state)["report"]


def test_a_thin_finding_cannot_be_filed_as_high_risk(monkeypatch):
    """The live runs made this necessary. On clean May -- 0 laundering in the answer key -- the
    model rated its one false-positive candidate High and recommended filing a SAR, off a draft
    the critic scored 0.75 ("supported, but thin"). Meanwhile July, with 23 laundering wires,
    came back Low. High means *file*, so it must clear the critic's top band."""
    def high(): return ComplianceReport(
        risk_rating="High", flagged_wires=["FGO1"],
        applicable_regulations=[f"AML Rulebook {CLAUSE}"],
        audit_summary="x", source_document_hashes=[],
    )
    draft = f"Reported under [AML Rulebook {CLAUSE}]."

    thin = generate_with(high(), draft, monkeypatch, confidence_score=CONFIDENCE_THRESHOLD)
    assert thin.risk_rating == "Medium", "0.75 stops the loop; it does not justify a filing"

    solid = generate_with(high(), draft, monkeypatch, confidence_score=1.0)
    assert solid.risk_rating == "High", "a fully grounded finding keeps the rating it earned"


def test_the_cap_only_ever_lowers_a_rating(monkeypatch):
    """A weak Medium is not promoted, and Low is never touched -- under-rating a real finding is
    the dangerous direction of error in an AML filing."""
    for rating in ("Low", "Medium"):
        report = ComplianceReport(
            risk_rating=rating, flagged_wires=["FGO1"], applicable_regulations=[],
            audit_summary="x", source_document_hashes=[],
        )
        result = generate_with(report, "No clause was on point.", monkeypatch, confidence_score=0.0)
        assert result.risk_rating == rating


def test_hashes_are_derived_from_the_draft_not_taken_on_trust(monkeypatch):
    """§6.4's drawer resolves every hash back to a chunk. Asked for these directly, the model
    returned an empty list while the draft plainly cited the clause -- so they are looked up."""
    report = ComplianceReport(
        risk_rating="High", flagged_wires=["FGO1"],
        applicable_regulations=[f"AML Rulebook {CLAUSE}"],
        audit_summary="## Findings", source_document_hashes=[],
    )
    result = generate_with(report, f"Reported under [AML Rulebook {CLAUSE}].", monkeypatch)
    assert result.source_document_hashes == [CHUNK_ID]


def test_a_hash_for_an_uncited_clause_is_not_carried(monkeypatch):
    report = ComplianceReport(
        risk_rating="High", flagged_wires=["FGO1"], applicable_regulations=[],
        audit_summary="x", source_document_hashes=[CHUNK_ID, "obliqa:1:invented:00000000"],
    )
    result = generate_with(report, "No clause was on point.", monkeypatch)
    assert result.source_document_hashes == []


def test_account_numbers_are_rejected_from_the_wire_reference_field(monkeypatch):
    """Observed on a live run: the model filled `flagged_wires` with anchor account numbers.

    A filing that names accounts where the schema specifies wire references points the
    investigation at the wrong object, and every number in it looks plausible.
    """
    candidates = detect([wire(f"FGO{i}", f"S{i}", "6123421761", "5000", i + 1) for i in range(4)])
    report = ComplianceReport(
        risk_rating="High", flagged_wires=["6123421761"], applicable_regulations=[],
        audit_summary="x", source_document_hashes=[],
    )
    result = generate_with(
        report, "Account 6123421761 collected a run of wires.", monkeypatch,
        candidates=candidates,
        wires=[wire(f"FGO{i}", f"S{i}", "6123421761", "5000", i + 1) for i in range(4)],
    )
    assert "6123421761" not in result.flagged_wires
    assert set(result.flagged_wires) == {"FGO0", "FGO1", "FGO2", "FGO3"}


def test_valid_wire_references_are_kept_as_the_model_chose_them(monkeypatch):
    wires = [wire(f"FGO{i}", f"S{i}", "COLLECTOR", "5000", i + 1) for i in range(4)]
    report = ComplianceReport(
        risk_rating="High", flagged_wires=["FGO1", "FGO2", "NOT-A-WIRE"],
        applicable_regulations=[], audit_summary="x", source_document_hashes=[],
    )
    result = generate_with(report, "draft", monkeypatch, wires=wires, candidates=detect(wires))
    assert result.flagged_wires == ["FGO1", "FGO2"]


def test_reservations_are_carried_into_the_prompt_when_the_loop_gave_up(monkeypatch):
    report = ComplianceReport(
        risk_rating="Medium", flagged_wires=[], applicable_regulations=[],
        audit_summary="x", source_document_hashes=[],
    )
    model = StubModel(structured=report)
    monkeypatch.setattr(nodes, "_model", lambda *a, **k: model)

    state = initial_state("x")
    state["retrieved_context"] = [clause_doc()]
    state["compliance_draft"] = "draft"
    state["is_audit_complete"] = False
    state["reservations"] = ["no clause covers the onward transfer"]

    nodes.generate_node(state)
    assert "UNRESOLVED AFTER REVIEW" in model.calls[0][1][1]
    assert "no clause covers the onward transfer" in model.calls[0][1][1]


# --- the report schema ---------------------------------------------------------------------------


def test_blank_and_duplicate_citations_are_dropped():
    report = ComplianceReport(
        risk_rating="High",
        flagged_wires=["FGO1", "FGO1", "  ", "FGO2"],
        applicable_regulations=["AML Rulebook 14.2.3", "AML Rulebook 14.2.3"],
        audit_summary="x",
        source_document_hashes=[""],
    )
    assert report.flagged_wires == ["FGO1", "FGO2"]
    assert report.applicable_regulations == ["AML Rulebook 14.2.3"]
    assert report.source_document_hashes == []


def test_risk_rating_is_constrained():
    with pytest.raises(ValueError):
        ComplianceReport(
            risk_rating="Catastrophic", flagged_wires=[], applicable_regulations=[],
            audit_summary="x", source_document_hashes=[],
        )


# --- the whole graph -----------------------------------------------------------------------------


def test_the_graph_has_the_cycle_that_justifies_using_a_graph():
    mermaid = build_graph().get_graph().draw_mermaid()
    assert "critic -.-> audit;" in mermaid, "without this edge a chain would do"
    assert "detect -.-> no_findings;" in mermaid


@needs_ledger
def test_a_full_run_produces_a_report_and_the_critic_loop_fires(monkeypatch, stub_retrieval):
    """End to end with a stub model: thin first draft, refinement, then a grounded report."""
    final = ComplianceReport(
        risk_rating="High",
        flagged_wires=["FGO23060500038"],
        applicable_regulations=[f"AML Rulebook {CLAUSE}"],
        audit_summary="## Findings\n\nA run of wires into one account.",
        source_document_hashes=[CHUNK_ID],
    )
    thin = prompts.Critique(confidence_score=0.4, unsupported_claims=["unsupported"],
                            refined_query="obligation to report a linked series of transactions")
    grounded = prompts.Critique(confidence_score=0.9, unsupported_claims=[], reasoning="ok")
    critiques = [thin, grounded]

    def structured(call_index):
        return critiques.pop(0) if critiques else final

    monkeypatch.setattr(
        nodes, "_model",
        lambda *a, **k: StubModel(text=f"Findings under [AML Rulebook {CLAUSE}].",
                                 structured=structured),
    )

    state = build_graph().invoke(initial_state(str(BATCH)))

    assert state["loop_count"] == 2, "the critic sent the draft back exactly once"
    assert len(stub_retrieval) > 2, "the refinement query reached retrieval"
    assert stub_retrieval[-1]["query"] == thin.refined_query
    assert state["report"].risk_rating == "High"
    assert state["is_audit_complete"] is True


@needs_ledger
def test_a_clean_batch_never_reaches_a_model(monkeypatch):
    """§9.2's cost routing, at the level that matters: $0.00 when there is nothing to audit."""
    monkeypatch.setattr(nodes, "detect", lambda wires: [])
    monkeypatch.setattr(nodes, "_model", lambda *a, **k: pytest.fail("no model on a clean batch"))
    monkeypatch.setattr(nodes, "retrieve", lambda *a, **k: pytest.fail("no retrieval either"))

    state = build_graph().invoke(initial_state(str(BATCH)))
    assert state["report"].risk_rating == "Low"
    assert state["report"].flagged_wires == []
    assert state["loop_count"] == 0

# --- 7. tracing (§7) ------------------------------------------------------------------------


def test_every_run_carries_one_searchable_id():
    """§7.2's point: one id shared by every span, so a run can be found again afterwards."""
    from src.graph.graph import run_config

    config = run_config("data/processed/ledger/2023-06_private_banking_log.pdf", tags=["NIGHTLY"])
    assert "AML_AUDIT_RUN" in config["tags"] and "NIGHTLY" in config["tags"]
    assert config["metadata"]["batch"] == "2023-06_private_banking_log.pdf"
    assert config["metadata"]["audit_id"].startswith("aud-")
    assert run_config("x")["metadata"]["audit_id"] != run_config("x")["metadata"]["audit_id"]


def test_the_run_config_cannot_carry_what_it_does_not_know_yet():
    """Wire and candidate counts do not exist at invoke() -- PARSE has not run. The blueprint's
    §7.2 example reads them from state before the call, which would always be zero."""
    from src.graph.graph import run_config

    assert "wire_count" not in run_config("x")["metadata"]


def test_a_model_call_is_tagged_with_the_state_it_was_made_in():
    state = initial_state("data/processed/ledger/2023-06_private_banking_log.pdf")
    state["candidates"] = sample_candidates()
    state["retrieved_context"] = [clause_doc()]
    state["loop_count"] = 1

    config = nodes.trace_config(state, "critic")
    assert config["tags"] == ["node:critic", "loop:1"]
    assert config["metadata"]["batch"] == "2023-06_private_banking_log.pdf"
    assert config["metadata"]["clause_count"] == 1
    assert config["metadata"]["shapes"] == [CONCENTRATION]


def test_the_two_drafts_of_a_looping_run_are_distinguishable_in_the_trace(monkeypatch, stub_retrieval):
    """The question §7.2 exists to answer -- which context caused the loop -- needs the two
    attempts to differ in the trace. A constant config would make them indistinguishable."""
    model = StubModel(text=f"Under [AML Rulebook {CLAUSE}].")
    monkeypatch.setattr(nodes, "_model", lambda *a, **k: model)

    state = initial_state("x")
    state["candidates"] = sample_candidates()
    state.update(nodes.audit_node(state))
    nodes.draft_node(state)

    state["loop_count"] = 1
    state["critique"] = "obligation to report linked transfers"
    state.update(nodes.audit_node(state))
    nodes.draft_node(state)

    loops = [c["tags"][1] for c in model.configs]
    assert loops == ["loop:0", "loop:1"]


def test_tracing_reports_itself_as_off_without_a_key(monkeypatch):
    """A trace that is silently not being written is worse than none: you go looking for it
    after the run instead of before."""
    from src.graph.graph import tracing_project

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert tracing_project() is None

    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-fake")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "finguard-orchestrator")
    assert tracing_project() == "finguard-orchestrator"

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    assert tracing_project() is None
