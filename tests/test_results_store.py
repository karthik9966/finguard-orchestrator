"""The results store -- LLD §3.2's three tables and §5.1's step 8.

The tests that matter here are about one decision: **`reports.report_json` is immutable and
`findings.status` is not.** Everything else is plumbing around that.

Each test gets its own SQLite file, so nothing shares state and nothing writes into the repo.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models import Candidate, Citation, ComplianceReport, Finding, RuleChunk
from src.store.results import (
    FINDINGS,
    REPORTS,
    REVIEWS,
    InMemoryResultsStore,
    InvalidTransition,
    ResultsStore,
    ResultsStoreUnavailable,
    SqlResultsStore,
)


@pytest.fixture
def store(tmp_path) -> SqlResultsStore:
    return SqlResultsStore(f"sqlite:///{tmp_path / 'results.db'}")


CLAUSE = RuleChunk(
    chunk_id="31cfr1020.320:d000653bf144c90b",
    text="A bank shall file a report of any suspicious transaction relevant to a possible "
         "violation of law or regulation.",
    tier="regulation",
    authority="binding",
    source_id="31cfr1020.320",
    section_ref="§ 1020.320(a)",
)


def candidate(suffix: str = "a", pattern: str = "structuring") -> Candidate:
    return Candidate(
        candidate_id=f"{pattern}:acct-{suffix}:0000000000",
        pattern_type=pattern,
        member_txn_refs=[f"FGO2306010000{suffix}1", f"FGO2306010000{suffix}2"],
        attributes={"threshold": 10000},
        detection_confidence=0.6,
    )


def finding(target: Candidate, *, status="pending_review", risk="medium", confidence=0.9,
            run="run-abc") -> Finding:
    return Finding(
        # Run-scoped, as `nodes.finding_id` mints it: a candidate id is stable across runs by
        # design, so a report-scoped judgement about it cannot be keyed on the candidate alone.
        finding_id=f"f-{run}:{target.candidate_id}",
        candidate=target,
        risk_level=risk,
        narrative="Two transfers just below the reporting threshold within eleven days.",
        applicable_regulations=[Citation.from_chunk(CLAUSE)],
        confidence=confidence,
        status=status,
        review_notes=["thin"] if status == "needs_review" else [],
    )


def report(*findings: Finding, period="2023-06", run="run-abc", rating=None) -> ComplianceReport:
    return ComplianceReport(
        report_id=f"rep-{run}",
        run_id=run,
        period=period,
        generated_at=datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc),
        risk_rating=rating or ("none" if not findings else "medium"),
        findings=list(findings),
        flagged_transactions=[r for f in findings for r in f.candidate.member_txn_refs],
        summary="## Compliance review",
        clean=not findings,
        source_document_refs=[Citation.from_chunk(CLAUSE)] if findings else [],
    )


# --- the seam ---------------------------------------------------------------------------------


def test_both_stores_satisfy_the_protocol():
    """The orchestrator depends on this and nothing else, which is what let Phase 5 be written
    before the database existed."""
    assert isinstance(InMemoryResultsStore(), ResultsStore)
    assert isinstance(SqlResultsStore("sqlite:///:memory:"), ResultsStore)


def test_the_schema_is_the_one_the_lld_specifies():
    """Column-for-column, because a store whose shape drifts from the design document is a store
    nobody can reason about from the design document."""
    assert set(REPORTS.c.keys()) == {
        "report_id", "run_id", "period", "generated_at", "risk_rating", "clean",
        "report_json", "schema_version",
    }
    assert set(FINDINGS.c.keys()) == {
        "finding_id", "report_id", "candidate_id", "pattern_type", "risk_level", "confidence",
        "status", "ordinal",
    }
    assert set(REVIEWS.c.keys()) == {
        "review_id", "finding_id", "action", "reviewer", "timestamp", "note",
    }


def test_the_lld_indexes_exist():
    """reports(period), findings(report_id), reviews(finding_id) -- the three queries this store
    actually serves."""
    assert {i.name for i in REPORTS.indexes} == {"ix_reports_period"}
    assert {i.name for i in FINDINGS.indexes} == {"ix_findings_report_id"}
    assert {i.name for i in REVIEWS.indexes} == {"ix_reviews_finding_id"}


# --- saving -----------------------------------------------------------------------------------


def test_a_report_and_its_findings_are_saved_together(store):
    first, second = candidate("a"), candidate("b", "fan_in")
    store.save(report(finding(first), finding(second)))
    assert store.counts() == {"reports": 1, "findings": 2, "reviews": 0, "jobs": 0}


def test_a_clean_report_saves_with_no_findings(store):
    store.save(report())
    assert store.counts()["findings"] == 0
    restored = store.get("rep-run-abc")
    assert restored.clean is True and restored.risk_rating == "none"


def test_a_saved_report_round_trips_through_the_schema(store):
    original = report(finding(candidate("a")))
    store.save(original)
    restored = store.get(original.report_id)
    # Compared as JSON rather than by identity: what matters is that nothing was lost crossing the
    # database, including the Decimal and datetime fields a naive serialiser mangles.
    assert restored.model_dump(mode="json") == original.model_dump(mode="json")


def test_a_report_survives_a_new_process(tmp_path):
    """The point of the phase. A second store object on the same file is what a uvicorn restart
    looks like from the data's side."""
    url = f"sqlite:///{tmp_path / 'results.db'}"
    SqlResultsStore(url).save(report(finding(candidate("a"))))

    reopened = SqlResultsStore(url)
    restored = reopened.get("rep-run-abc")
    assert restored is not None and len(restored.findings) == 1


def test_saving_the_same_report_twice_is_idempotent(store):
    """The retry path must not leave two half-written copies."""
    store.save(report(finding(candidate("a"))))
    store.save(report(finding(candidate("a"))))
    assert store.counts() == {"reports": 1, "findings": 1, "reviews": 0, "jobs": 0}


def test_findings_keep_the_order_they_were_filed_in(store):
    order = [candidate("a"), candidate("b", "fan_in"), candidate("c", "cycle")]
    store.save(report(*[finding(c) for c in order]))
    restored = store.get("rep-run-abc")
    assert [f.candidate.pattern_type for f in restored.findings] == [
        "structuring", "fan_in", "cycle"
    ]


def test_two_audits_of_the_same_batch_are_separately_reviewable(store):
    """`candidate_id` is stable across runs on purpose, so a finding keyed on it alone collides on
    the second audit of a month -- which is what `?force=true` does. Found in Phase 6b as an
    integrity error; the finding id is run-scoped so both judgements stand and each can be reviewed
    on its own."""
    target = candidate("a")
    store.save(report(finding(target, run="run-first"), run="run-first"))
    store.save(report(finding(target, run="run-second"), run="run-second"))

    assert store.counts()["findings"] == 2
    first = store.get("rep-run-first").findings[0]
    second = store.get("rep-run-second").findings[0]
    assert first.finding_id != second.finding_id
    assert first.candidate.candidate_id == second.candidate.candidate_id

    store.review(second.finding_id, "escalate", reviewer="analyst@bank")
    assert store.get("rep-run-first").findings[0].status == "pending_review", (
        "reviewing one audit's finding must not move the other's"
    )
    assert store.get("rep-run-second").findings[0].status == "escalated"


# --- the write failure ------------------------------------------------------------------------


def test_a_write_failure_holds_the_report_for_re_save(store, monkeypatch):
    """LLD §6 RESULTS_STORE_WRITE_FAILURE. By step 8 the run is already paid for -- five
    candidates of grounding and review -- so a transient disk fault must not cost the result."""
    attempts = {"n": 0}

    def explode(*args, **kwargs):
        attempts["n"] += 1
        raise OSError("database is locked")

    monkeypatch.setattr(store, "_save", explode)
    monkeypatch.setattr("src.store.results.time.sleep", lambda seconds: None)

    filed = report(finding(candidate("a")))
    with pytest.raises(ResultsStoreUnavailable, match="RESULTS_STORE_WRITE_FAILURE"):
        store.save(filed)

    from src.config import get_config

    assert attempts["n"] == get_config().persistence.write_attempts, "it backs off and retries"
    assert store.unsaved == {filed.report_id: filed}, "held, not lost"

    # And the hold is useful: once the fault clears, the report goes in without re-running anything.
    monkeypatch.undo()
    assert store.flush() == [filed.report_id]
    assert store.get(filed.report_id) is not None and store.unsaved == {}


def test_a_successful_save_holds_nothing(store):
    store.save(report(finding(candidate("a"))))
    assert store.unsaved == {}


# --- the review loop --------------------------------------------------------------------------


def test_reviewing_a_finding_moves_its_status_without_touching_the_filed_report(store):
    """The decision this whole module is arranged around. `report_json` is the record of what the
    engine concluded; review changes where the work stands, not what was concluded."""
    filed = report(finding(candidate("a")))
    store.save(filed)
    target = filed.findings[0].finding_id

    assert store.review(target, "escalate", reviewer="analyst@bank", note="to the officer") == (
        "escalated"
    )

    assert store.get(filed.report_id).findings[0].status == "escalated"
    assert store.stored(filed.report_id).findings[0].status == "pending_review"


def test_the_review_history_is_append_only(store):
    filed = report(finding(candidate("a")))
    store.save(filed)
    target = filed.findings[0].finding_id

    store.review(target, "escalate", reviewer="analyst@bank", note="unusual counterparties")
    store.review(target, "approve", reviewer="officer@bank", note="filing a SAR")

    history = store.reviews_for(target)
    assert [entry.action for entry in history] == ["escalate", "approve"]
    assert [entry.reviewer for entry in history] == ["analyst@bank", "officer@bank"]
    assert history[1].note == "filing a SAR"
    assert store.status_of(target) == "approved"


def test_the_joined_report_carries_the_review_trail(store):
    """An analyst reading a finding needs to see who moved it and why, not just where it landed."""
    filed = report(finding(candidate("a")))
    store.save(filed)
    target = filed.findings[0].finding_id
    store.review(target, "clear", reviewer="analyst@bank", note="salary payments, verified")

    notes = store.get(filed.report_id).findings[0].review_notes
    assert any("clear by analyst@bank" in note for note in notes)
    assert any("salary payments, verified" in note for note in notes)


def test_an_officer_cannot_approve_something_nobody_escalated(store):
    """Approving is signing off on a filing. Allowing it from `pending_review` would make
    'approved' mean two different things in one column."""
    filed = report(finding(candidate("a")))
    store.save(filed)
    target = filed.findings[0].finding_id

    with pytest.raises(InvalidTransition, match="cannot approve"):
        store.review(target, "approve", reviewer="officer@bank")


def test_a_cleared_finding_is_final(store):
    filed = report(finding(candidate("a")))
    store.save(filed)
    target = filed.findings[0].finding_id
    store.review(target, "clear", reviewer="analyst@bank")

    with pytest.raises(InvalidTransition, match="it is final"):
        store.review(target, "escalate", reviewer="analyst@bank")


def test_a_needs_review_finding_is_exactly_what_an_analyst_can_act_on(store):
    """It is the state the engine puts a candidate in when it could not ground one, so it has to be
    reviewable -- otherwise the loop's give-up path is a dead end."""
    filed = report(finding(candidate("a"), status="needs_review"))
    store.save(filed)
    target = filed.findings[0].finding_id
    assert store.review(target, "escalate", reviewer="analyst@bank") == "escalated"


def test_reviewing_a_finding_that_does_not_exist_says_so(store):
    with pytest.raises(KeyError):
        store.review("f-nothing", "clear", reviewer="analyst@bank")


def test_an_escalation_can_be_cleared_on_second_thoughts(store):
    """The officer's other option. Without it an escalation is a one-way door and a mistaken one
    can only be fixed in the database."""
    filed = report(finding(candidate("a")))
    store.save(filed)
    target = filed.findings[0].finding_id
    store.review(target, "escalate", reviewer="analyst@bank")
    assert store.review(target, "clear", reviewer="officer@bank", note="no case") == "cleared"


# --- listing ----------------------------------------------------------------------------------


def test_the_listing_omits_the_report_bodies(store):
    """A listing of ten audits should not carry ten full narratives."""
    store.save(report(finding(candidate("a")), run="run-1"))
    listed = store.list_reports()
    assert len(listed) == 1
    assert "report_json" not in listed[0] and "summary" not in listed[0]
    assert listed[0]["findings"] == 1 and listed[0]["needs_review"] == 0


def test_the_listing_can_be_narrowed_to_one_period(store):
    """`reports(period)` is indexed because 'show me June' is the question that gets asked."""
    store.save(report(finding(candidate("a")), run="run-june", period="2023-06"))
    store.save(report(finding(candidate("b")), run="run-july", period="2023-07"))
    assert {r["period"] for r in store.list_reports(period="2023-07")} == {"2023-07"}
    assert len(store.list_reports()) == 2


def test_the_listing_counts_what_still_needs_a_human(store):
    store.save(report(
        finding(candidate("a")), finding(candidate("b"), status="needs_review"), run="run-1",
    ))
    assert store.list_reports()[0]["needs_review"] == 1


def test_an_unknown_report_reads_as_absent_rather_than_raising(store):
    assert store.get("rep-nothing") is None
    assert store.stored("rep-nothing") is None
    assert store.findings_for("rep-nothing") == []


def test_a_file_backed_url_creates_its_directory(tmp_path):
    """`RESULTS_DB_URL=sqlite:///./data/results.db` is an obvious thing to set, and SQLite's own
    error for a missing directory says nothing about the actual problem."""
    nested = tmp_path / "does" / "not" / "exist" / "results.db"
    SqlResultsStore(f"sqlite:///{nested}").save(report())
    assert nested.exists()
