"""The cockpit's API client, driven against the real FastAPI app.

Not a mocked client. Every test here goes through FastAPI's own TestClient, so what is exercised is
the actual routing, the actual auth, the actual lifespan and the actual single worker -- with no
server and no network. A client tested against a stub of the API is a client tested against my
beliefs about the API.

Journeys 1 and 3 are end-to-end here: upload a batch, poll it to completion, read the report, review
a finding, and see the status move while the filed report does not.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.graph.cost import UsageLedger
from src.graph.run import RunResult
from src.models import (
    Candidate,
    Citation,
    ComplianceReport,
    Finding,
    QuarantinedMessage,
    RuleChunk,
    ValidationReport,
)
from src.store import SqlResultsStore
from src.ui.client import ApiError, FinGuardClient

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
BATCH = LEDGER / "2023-06_private_banking_log.txt"
needs_ledger = pytest.mark.skipif(not BATCH.exists(), reason="run: uv run finguard-ledger")

TOKEN = "test-token-not-a-real-secret"

CLAUSE = RuleChunk(
    chunk_id="31cfr1020.320:d000653bf144c90b",
    text="A bank shall file a report of any suspicious transaction relevant to a possible "
         "violation of law or regulation.",
    tier="regulation", authority="binding",
    source_id="31cfr1020.320", section_ref="§ 1020.320(a)",
)
INDICATOR = RuleChunk(
    chunk_id="ffiec-appendix-f:91b4dc940b784a98",
    text="Multiple and frequent deposits to accounts that appear unrelated.",
    tier="guidance", authority="illustrative",
    source_id="ffiec-appendix-f", section_ref="Other Unusual Activity ¶ 18",
)


def a_finding(run="run-stub", *, status="pending_review") -> Finding:
    target = Candidate(
        candidate_id="structuring:acct-a:0000000000",
        pattern_type="structuring",
        member_txn_refs=["FGO23060100001", "FGO23060100002"],
        detection_confidence=0.6,
    )
    return Finding(
        finding_id=f"f-{run}:{target.candidate_id}",
        candidate=target,
        risk_level="medium",
        narrative="Two transfers just below the reporting threshold.",
        applicable_regulations=[Citation.from_chunk(CLAUSE)],
        red_flag_indicators=[Citation.from_chunk(INDICATOR)],
        confidence=0.9,
        status=status,
        review_notes=["the evidence was thin"] if status == "needs_review" else [],
    )


def a_report(run="run-stub", *, findings=None, quarantined=0) -> ComplianceReport:
    findings = [a_finding(run)] if findings is None else list(findings)
    return ComplianceReport(
        report_id=f"rep-{run}", run_id=run, period="2023-06",
        generated_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        risk_rating="medium" if findings else "none",
        findings=findings,
        flagged_transactions=[r for f in findings for r in f.candidate.member_txn_refs],
        summary="## Compliance review -- 2023-06",
        clean=not findings,
        source_document_refs=[Citation.from_chunk(CLAUSE), Citation.from_chunk(INDICATOR)],
        quarantined_count=quarantined,
    )


VALIDATION = ValidationReport(
    batch="2023-06_private_banking_log.txt", declared=501, parsed=500, rescued=2,
    quarantined=[QuarantinedMessage(
        ordinal=317, reference="FGO23060100317", reason="missing required tag :32A:",
        raw=":20:FGO23060100317\n:23B:CRED\n:50K:/6123421761", fallback_attempted=True,
    )],
)


@pytest.fixture
def store(monkeypatch, tmp_path) -> SqlResultsStore:
    fresh = SqlResultsStore(f"sqlite:///{tmp_path / 'results.db'}")
    monkeypatch.setattr(main, "_STORE", fresh)
    monkeypatch.setenv("API_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("API_BASE_URL", "http://testserver")
    from src.config import reset_caches

    reset_caches()
    yield fresh
    reset_caches()


@pytest.fixture
def api(monkeypatch, store) -> FinGuardClient:
    """The client, pointed at the running app. Entered as a context manager so the lifespan runs and
    the single worker actually drains the queue."""
    def stub_audit(path, *, run_id=None, store=None, tags=None, **kwargs):
        report = a_report(run_id or "run-stub", quarantined=len(VALIDATION.quarantined))
        if store is not None:
            store.save(report, validation=VALIDATION)
        return RunResult(
            report=report, validation=VALIDATION, usage=UsageLedger(),
            run_id=run_id or "run-stub", candidates=1, records=500,
        )

    monkeypatch.setattr(main, "audit_batch", stub_audit)
    monkeypatch.setattr(main, "counts", lambda: {
        "total": 731, "tier": {"statute": 12, "regulation": 219, "guidance": 500},
        "authority": {"binding": 231, "illustrative": 500},
    })
    with TestClient(main.app) as session:
        yield FinGuardClient(base_url="http://testserver", token=TOKEN, session=session)


def finish(api: FinGuardClient, job_id: str, *, tries: int = 100) -> dict:
    for _ in range(tries):
        job = api.audit(job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never left running")


# --- the client's own behaviour -----------------------------------------------------------


def test_an_unreachable_api_is_a_sentence_not_a_traceback():
    """The common case in development is that the service is simply not running, and an analyst
    needs to be told that rather than shown a transport error."""
    offline = FinGuardClient(base_url="http://127.0.0.1:9", token="x")
    with pytest.raises(ApiError) as raised:
        offline.health()
    assert raised.value.status == 0 and "cannot reach the API" in raised.value.detail


def test_a_rejected_token_keeps_its_status(api, store):
    """401, 503 and 409 each mean something different for an analyst to do, so the status survives
    rather than collapsing into one error."""
    wrong = FinGuardClient(base_url="http://testserver", token="wrong", session=api.session)
    with pytest.raises(ApiError) as raised:
        wrong.reports()
    assert raised.value.status == 401


def test_the_client_reads_its_settings_when_not_told(store):
    """The cockpit constructs it with no arguments."""
    default = FinGuardClient()
    assert default.base_url == "http://testserver" and default.token == TOKEN


def test_an_injected_session_is_not_closed_by_the_client(api):
    """It belongs to the caller. Closing it would break the next request in the same page render."""
    api.health()
    api.health()
    assert not api.session.is_closed


# --- Journey 1: upload, poll, read --------------------------------------------------------


@needs_ledger
def test_a_batch_can_be_submitted_and_polled_to_a_report(api):
    accepted = api.submit(BATCH.name, BATCH.read_bytes())
    assert accepted["status"] == "running" and accepted["deduplicated"] is False

    job = finish(api, accepted["job_id"])
    assert job["status"] == "complete"

    report = api.report(job["report"]["report_id"])
    assert isinstance(report, ComplianceReport)
    assert report.period == "2023-06" and len(report.findings) == 1


@needs_ledger
def test_a_re_submitted_batch_is_reported_as_deduplicated(api):
    """The cockpit says so rather than silently showing an old run as if it were new."""
    first = api.submit(BATCH.name, BATCH.read_bytes())
    finish(api, first["job_id"])
    again = api.submit(BATCH.name, BATCH.read_bytes())
    assert again["deduplicated"] is True and again["job_id"] == first["job_id"]


@needs_ledger
def test_force_runs_it_again(api):
    first = api.submit(BATCH.name, BATCH.read_bytes())
    finish(api, first["job_id"])
    again = api.submit(BATCH.name, BATCH.read_bytes(), force=True)
    assert again["deduplicated"] is False and again["job_id"] != first["job_id"]


@needs_ledger
def test_a_bad_upload_is_refused_with_its_reason(api):
    with pytest.raises(ApiError) as raised:
        api.submit("empty.txt", b"not a swift message")
    assert raised.value.status == 400 and "no transactions" in raised.value.detail


@needs_ledger
def test_wait_returns_the_finished_report_in_one_call(api):
    """Journey 2's shape, available to the UI as well -- useful for a short batch."""
    body = api.submit(BATCH.name, BATCH.read_bytes(), wait=True)
    assert body["status"] == "complete" and body["report"]["findings"]


# --- Journey 3: the review loop through the client ----------------------------------------


def test_reviewing_moves_the_status_and_leaves_the_filed_report_alone(api, store):
    """The whole of §6.4's review loop, as the cockpit's buttons drive it."""
    filed = a_report("run-review")
    store.save(filed, validation=VALIDATION)
    finding_id = filed.findings[0].finding_id

    body = api.review(finding_id, "escalate", reviewer="analyst@bank", note="12 unrelated senders")
    assert body["status"] == "escalated"

    assert api.report(filed.report_id).findings[0].status == "escalated"
    assert api.filed(filed.report_id).findings[0].status == "pending_review"


def test_the_review_trail_reaches_the_page(api, store):
    """An analyst opening a reviewed finding has to see who moved it and why."""
    filed = a_report("run-review")
    store.save(filed)
    finding_id = filed.findings[0].finding_id
    api.review(finding_id, "clear", reviewer="analyst@bank", note="payroll, verified")

    notes = api.report(filed.report_id).findings[0].review_notes
    assert any("clear by analyst@bank" in note for note in notes)
    assert any("payroll, verified" in note for note in notes)


def test_an_impossible_action_is_a_409_the_page_can_explain(api, store):
    store.save(a_report("run-review"))
    with pytest.raises(ApiError) as raised:
        api.review(a_finding("run-review").finding_id, "approve", reviewer="officer@bank")
    assert raised.value.status == 409 and "cannot approve" in raised.value.detail


def test_past_reports_are_listable_by_month(api, store):
    """Journey 3's ingress: "an officer requests a past report (by month or case)"."""
    store.save(a_report("run-june"))
    assert len(api.reports(period="2023-06")) == 1
    assert api.reports(period="2023-07") == []


# --- the quarantine panel -----------------------------------------------------------------


def test_the_ingestion_record_names_the_messages_that_were_lost(api, store):
    """A count is not actionable: an analyst told a message was lost needs to see which one, with
    its raw text, to go and fix the source."""
    filed = a_report("run-quarantine", quarantined=1)
    store.save(filed, validation=VALIDATION)

    validation = api.validation(filed.report_id)
    assert validation is not None
    assert (validation.parsed, validation.declared, validation.rescued) == (500, 501, 2)
    assert len(validation.quarantined) == 1

    lost = validation.quarantined[0]
    assert lost.ordinal == 317 and "missing required tag" in lost.reason
    assert ":20:FGO23060100317" in lost.raw, "the raw text is what makes it fixable"
    assert validation.complete is False


def test_a_report_with_no_ingestion_record_says_so_rather_than_showing_zeroes(api, store):
    """"Nothing was kept" and "nothing was quarantined" are different answers, and conflating them
    would be reassuring rather than true."""
    store.save(a_report("run-bare"))
    assert api.validation("rep-run-bare") is None


def test_the_ingestion_record_of_an_unknown_report_is_absent(api, store):
    assert api.validation("rep-nothing") is None


@needs_ledger
def test_a_real_run_records_what_it_ingested(api, monkeypatch, store):
    """Not a fixture: the validation that reaches the store is the one ingestion actually built.

    `audit_batch` is restored by name rather than with `monkeypatch.undo()`, which would also revert
    this test's store and environment -- and did, on the first run of it, writing a results.db into
    the repository from the settings cache that survived the revert.
    """
    from src.graph.run import audit_batch as real_audit_batch

    monkeypatch.setattr(main, "audit_batch", real_audit_batch)

    # Still no model and no vector store: the real ingestion, then a graph that finds nothing.
    def clean_graph():
        class Graph:
            def invoke(self, state, config=None):
                from src.graph.nodes import ReportGenerationNode

                state = {**state, "candidates": [], "clean_flag": True, "findings": []}
                return {**state, **ReportGenerationNode()(state)}

        return Graph()

    monkeypatch.setattr("src.graph.run.build_graph", clean_graph)
    accepted = api.submit(BATCH.name, BATCH.read_bytes())
    job = finish(api, accepted["job_id"])
    assert job["status"] == "complete", job.get("error")

    validation = api.validation(job["report"]["report_id"])
    declared = BATCH.read_text().count("{1:F01")
    assert validation is not None and validation.parsed == declared > 0
    assert isinstance(validation.declared, int)
