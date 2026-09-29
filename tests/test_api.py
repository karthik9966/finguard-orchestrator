"""The HTTP surface (LLD §10).

The run itself is stubbed: these tests are about the contract -- what a caller gets back, when, and
what happens when the upload or the corpus is wrong. No API key, no network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.graph.run import BatchUnreadable, RunResult
from src.graph.cost import UsageLedger
from src.models import Candidate, Citation, ComplianceReport, Finding, RuleChunk, ValidationReport
from src.store import SqlResultsStore

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
BATCH = LEDGER / "2023-06_private_banking_log.txt"
needs_ledger = pytest.mark.skipif(not BATCH.exists(), reason="run: uv run finguard-ledger")

REPORT = ComplianceReport(
    report_id="rep-run-stub",
    run_id="run-stub",
    period="2023-06",
    generated_at=datetime(2026, 9, 28, tzinfo=timezone.utc),
    risk_rating="none",
    clean=True,
    summary="## Compliance review -- 2023-06",
)


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch, tmp_path):
    """A fresh registry and a fresh database per test. The store is a real SqlResultsStore on a temp
    file rather than a stub: Journey 3's endpoints are mostly *store* behaviour, and stubbing it
    would leave the join between a frozen report and a live status untested where it is served."""
    main.AUDITS.clear()
    monkeypatch.setattr(main, "_STORE", SqlResultsStore(f"sqlite:///{tmp_path / 'results.db'}"))
    yield
    main.AUDITS.clear()


def stub_result(report=REPORT, *, candidates=2, run_id="run-stub") -> RunResult:
    return RunResult(
        report=report, validation=ValidationReport(parsed=500), usage=UsageLedger(),
        run_id=run_id, candidates=candidates, records=500,
    )


@pytest.fixture
def client(monkeypatch):
    """A run that returns instantly, so the contract is tested rather than the model."""
    def stub_audit(path, *, run_id=None, store=None, tags=None, **kwargs):
        report = REPORT.model_copy(update={"run_id": run_id, "report_id": f"rep-{run_id}"})
        if store is not None:
            store.save(report)
        return stub_result(report, run_id=run_id or "run-stub")

    monkeypatch.setattr(main, "audit_batch", stub_audit)
    monkeypatch.setattr(main, "counts", lambda: {
        "total": 731, "tier": {"statute": 12, "regulation": 219, "guidance": 500},
        "authority": {"binding": 231, "illustrative": 500},
    })
    # TestClient runs background tasks synchronously on response, so a poll right after the POST
    # already sees the finished audit.
    return TestClient(main.app)


# --- health -----------------------------------------------------------------------------


def test_health_reports_the_corpus_it_would_actually_query(client):
    """Probed against `rule_chunks` -- the collection the retriever really reads, not whichever
    collection an environment variable happens to name."""
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["vectors"] == 731
    assert body["collection"] == "rule_chunks"
    assert body["by_authority"]["binding"] == 231


def test_health_fails_when_the_collection_is_empty(client, monkeypatch):
    """A 200 from a service with no vectors sends every candidate to needs_review one paid call at
    a time, which is the expensive way to discover an empty collection."""
    monkeypatch.setattr(main, "counts", lambda: {"total": 0, "tier": {}, "authority": {}})
    assert client.get("/health").status_code == 503


def test_health_fails_when_the_store_is_unreachable(client, monkeypatch):
    def boom():
        raise RuntimeError("no such collection")

    monkeypatch.setattr(main, "counts", boom)
    assert client.get("/health").status_code == 503


# --- submitting -------------------------------------------------------------------------


@needs_ledger
def test_a_batch_is_accepted_and_audited_in_the_background(client):
    """202 with an id, not a 60-second held connection."""
    response = client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())})
    assert response.status_code == 202

    body = response.json()
    assert body["status"] == "running"
    # Derived, not pinned -- the batch was 220 messages before Phase 2 regenerated the ledgers.
    declared = int(
        next(l for l in BATCH.read_text().splitlines() if l.startswith("Messages in batch")).split(":")[1]
    )
    assert body["wires"] == declared, "validated during upload, before the audit ran"
    assert body["poll"] == f"/audit/{body['audit_id']}"

    result = client.get(body["poll"]).json()
    assert result["status"] == "complete"
    assert result["report"]["risk_rating"] == "none"
    assert result["candidates"] == 2 and result["findings"] == 0


@needs_ledger
def test_the_trace_id_the_resource_id_and_the_report_id_are_one_run(client):
    """One id, so a LangSmith trace, an API result and a stored report join without a lookup
    table."""
    body = client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())}).json()
    audit_id = body["audit_id"]
    assert audit_id.startswith("run-")

    result = client.get(body["poll"]).json()
    assert result["report"]["run_id"] == audit_id
    assert main.reports().get(f"rep-{audit_id}") is not None, "step 8 persisted it"


def test_a_file_that_is_not_a_batch_is_refused_immediately(client):
    """A 400 in a second, not a background task that fails a minute later."""
    response = client.post("/audit", files={"batch": ("empty.txt", b"not a swift message")})
    assert response.status_code == 400
    assert "no wires" in response.json()["detail"]


def test_an_unsupported_file_type_is_refused(client):
    response = client.post("/audit", files={"batch": ("ledger.csv", b"a,b,c")})
    assert response.status_code == 415


@needs_ledger
def test_a_failing_audit_is_reported_not_swallowed(client, monkeypatch):
    def explode(path, **kwargs):
        raise RuntimeError("the vector store went away")

    monkeypatch.setattr(main, "audit_batch", explode)
    body = client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())}).json()

    result = client.get(body["poll"]).json()
    assert result["status"] == "failed"
    assert "vector store went away" in result["error"]
    assert result["report"] is None


@needs_ledger
def test_an_unreadable_batch_is_reported_as_the_clients_file(client, monkeypatch):
    """LLD §6's one loud failure. Distinguished from a server fault in the error text, because
    the two need different actions from whoever reads it."""
    def unreadable(path, **kwargs):
        raise BatchUnreadable("INGEST_FILE_UNREADABLE: yielded no readable transactions")

    monkeypatch.setattr(main, "audit_batch", unreadable)
    body = client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())}).json()
    result = client.get(body["poll"]).json()
    assert result["status"] == "failed"
    assert result["error"].startswith("INGEST_FILE_UNREADABLE")


@needs_ledger
def test_the_uploaded_file_does_not_outlive_the_audit(client):
    """Every submission writes a temp file. Left behind, they accumulate silently."""
    written: list[Path] = []
    original = main.parse_batch

    def spy(path, **kwargs):
        written.append(Path(path))
        return original(path, **kwargs)

    main.parse_batch = spy
    try:
        client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())})
    finally:
        main.parse_batch = original

    assert written and not written[0].exists()


# --- reading ----------------------------------------------------------------------------


def test_an_unknown_audit_is_a_404(client):
    assert client.get("/audit/aud-does-not-exist").status_code == 404


@needs_ledger
def test_the_listing_omits_the_report_bodies(client):
    """A listing of ten audits should not carry ten full narratives."""
    client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())})
    listed = client.get("/audits").json()
    assert len(listed) == 1
    assert "report" not in listed[0]
    assert listed[0]["status"] == "complete"


# --- Journey 3: stored reports and the review loop ---------------------------------------


CLAUSE = RuleChunk(
    chunk_id="31cfr1020.320:d000653bf144c90b",
    text="A bank shall file a report of any suspicious transaction relevant to a possible "
         "violation of law or regulation.",
    tier="regulation",
    authority="binding",
    source_id="31cfr1020.320",
    section_ref="§ 1020.320(a)",
)


def stored_report(client) -> ComplianceReport:
    """One report in the store, with one reviewable finding."""
    target = Candidate(
        candidate_id="structuring:acct-a:0000000000",
        pattern_type="structuring",
        member_txn_refs=["FGO23060100001", "FGO23060100002"],
        detection_confidence=0.6,
    )
    filed = ComplianceReport(
        report_id="rep-run-journey3",
        run_id="run-journey3",
        period="2023-06",
        generated_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        risk_rating="medium",
        findings=[Finding(
            finding_id="f-structuring-a",
            candidate=target,
            risk_level="medium",
            narrative="Two transfers just below the reporting threshold.",
            applicable_regulations=[Citation.from_chunk(CLAUSE)],
            confidence=0.9,
        )],
        flagged_transactions=list(target.member_txn_refs),
        summary="## Compliance review",
    )
    main.reports().save(filed)
    return filed


def test_a_stored_report_is_readable_after_the_audit_record_is_gone(client):
    """The AUDITS dict is in-process; the store is not. This is what surviving a restart means."""
    filed = stored_report(client)
    main.AUDITS.clear()
    body = client.get(f"/reports/{filed.report_id}").json()
    assert body["report_id"] == filed.report_id and len(body["findings"]) == 1


def test_the_report_listing_omits_the_bodies(client):
    stored_report(client)
    listed = client.get("/reports").json()
    assert len(listed) == 1 and "summary" not in listed[0]
    assert listed[0]["findings"] == 1


def test_the_listing_can_be_narrowed_to_a_period(client):
    stored_report(client)
    assert len(client.get("/reports", params={"period": "2023-06"}).json()) == 1
    assert client.get("/reports", params={"period": "2023-07"}).json() == []


def test_reviewing_a_finding_changes_its_status_but_not_the_filed_report(client):
    """The phase's own criterion, over HTTP: escalating a finding changes its status without
    mutating report_json."""
    filed = stored_report(client)
    response = client.post(
        "/findings/f-structuring-a/review",
        json={"action": "escalate", "reviewer": "analyst@bank", "note": "unusual counterparties"},
    )
    assert response.status_code == 200 and response.json()["status"] == "escalated"

    live = client.get(f"/reports/{filed.report_id}").json()
    assert live["findings"][0]["status"] == "escalated"

    as_filed = client.get(f"/reports/{filed.report_id}/filed").json()
    assert as_filed["findings"][0]["status"] == "pending_review"


def test_the_review_history_comes_back_with_the_decision(client):
    stored_report(client)
    client.post("/findings/f-structuring-a/review",
                json={"action": "escalate", "reviewer": "analyst@bank"})
    body = client.post("/findings/f-structuring-a/review",
                       json={"action": "approve", "reviewer": "officer@bank",
                             "note": "filing"}).json()
    assert [entry["action"] for entry in body["history"]] == ["escalate", "approve"]


def test_an_impossible_transition_is_a_409_not_a_500(client):
    """Approving something nobody escalated is a conflict with the current state, which is a thing
    the caller can understand and act on."""
    stored_report(client)
    response = client.post("/findings/f-structuring-a/review",
                           json={"action": "approve", "reviewer": "officer@bank"})
    assert response.status_code == 409 and "cannot approve" in response.json()["detail"]


def test_reviewing_an_unknown_finding_is_a_404(client):
    stored_report(client)
    response = client.post("/findings/f-nothing/review",
                           json={"action": "clear", "reviewer": "analyst@bank"})
    assert response.status_code == 404


def test_an_unsupported_review_action_is_rejected_by_the_schema(client):
    stored_report(client)
    response = client.post("/findings/f-structuring-a/review",
                           json={"action": "delete", "reviewer": "analyst@bank"})
    assert response.status_code == 422


def test_an_unknown_report_is_a_404(client):
    assert client.get("/reports/rep-nothing").status_code == 404
    assert client.get("/reports/rep-nothing/filed").status_code == 404


def test_health_reports_how_many_reports_are_held(client):
    stored_report(client)
    assert client.get("/health").json()["reports_stored"] == 1
