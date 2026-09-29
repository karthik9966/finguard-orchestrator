"""The HTTP surface (LLD §10) and HLD §2.2's three journeys.

The run itself is stubbed: these tests are about the contract -- what a caller gets back, when, what
happens when the upload or the corpus is wrong, and who is allowed to ask. No API key, no network.

The client is entered as a context manager on purpose. The single worker is started by the app's
lifespan, and a TestClient used without `with` never runs it -- every job would sit at `running`
forever, and every test here would be testing a queue that nobody drains.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.graph.cost import UsageLedger
from src.graph.run import BatchUnreadable, RunResult
from src.models import (
    Candidate,
    Citation,
    ComplianceReport,
    Finding,
    RuleChunk,
    ValidationReport,
)
from src.store import SqlResultsStore

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
BATCH = LEDGER / "2023-06_private_banking_log.txt"
needs_ledger = pytest.mark.skipif(not BATCH.exists(), reason="run: uv run finguard-ledger")

TOKEN = "test-token-not-a-real-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

CLAUSE = RuleChunk(
    chunk_id="31cfr1020.320:d000653bf144c90b",
    text="A bank shall file a report of any suspicious transaction relevant to a possible "
         "violation of law or regulation.",
    tier="regulation",
    authority="binding",
    source_id="31cfr1020.320",
    section_ref="§ 1020.320(a)",
)


def a_finding() -> Finding:
    target = Candidate(
        candidate_id="structuring:acct-a:0000000000",
        pattern_type="structuring",
        member_txn_refs=["FGO23060100001", "FGO23060100002"],
        detection_confidence=0.6,
    )
    return Finding(
        finding_id="f-structuring-a",
        candidate=target,
        risk_level="medium",
        narrative="Two transfers just below the reporting threshold.",
        applicable_regulations=[Citation.from_chunk(CLAUSE)],
        confidence=0.9,
    )


def a_report(run_id="run-stub", *, findings=()) -> ComplianceReport:
    findings = list(findings)
    return ComplianceReport(
        report_id=f"rep-{run_id}",
        run_id=run_id,
        period="2023-06",
        generated_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        risk_rating="medium" if findings else "none",
        findings=findings,
        flagged_transactions=[r for f in findings for r in f.candidate.member_txn_refs],
        summary="## Compliance review -- 2023-06",
        clean=not findings,
    )


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path) -> SqlResultsStore:
    """A real store on a temp file per test.

    Not a stub: most of what these endpoints do *is* store behaviour -- job state, dedup, the join
    between a frozen report and a live status -- and stubbing it would leave all of that untested at
    the point where it is actually served.
    """
    fresh = SqlResultsStore(f"sqlite:///{tmp_path / 'results.db'}")
    monkeypatch.setattr(main, "_STORE", fresh)
    monkeypatch.setenv("API_AUTH_TOKEN", TOKEN)
    from src.config import reset_caches

    reset_caches()
    yield fresh
    reset_caches()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """A run that returns instantly, so the contract is tested rather than the model."""
    def stub_audit(path, *, run_id=None, store=None, tags=None, **kwargs):
        report = a_report(run_id or "run-stub", findings=[a_finding()])
        if store is not None:
            store.save(report)
        return RunResult(
            report=report, validation=ValidationReport(parsed=500), usage=UsageLedger(),
            run_id=run_id or "run-stub", candidates=1, records=500,
        )

    monkeypatch.setattr(main, "audit_batch", stub_audit)
    monkeypatch.setattr(main, "counts", lambda: {
        "total": 731, "tier": {"statute": 12, "regulation": 219, "guidance": 500},
        "authority": {"binding": 231, "illustrative": 500},
    })
    with TestClient(main.app) as entered:
        yield entered


def post_batch(client, *, name=None, body=None, **params):
    return client.post(
        "/audits",
        files={"batch": (name or BATCH.name, body if body is not None else BATCH.read_bytes())},
        params=params or None,
        headers=AUTH,
    )


def await_job(client, job_id: str, *, tries: int = 100) -> dict:
    """Poll until the single worker has finished the job. This is Journey 1's own loop."""
    for _ in range(tries):
        body = client.get(f"/audits/{job_id}", headers=AUTH).json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never left running")


# --- authentication -----------------------------------------------------------------------


def test_every_endpoint_but_health_needs_a_token(client):
    """LLD §6 AUTH_FAILURE. This service audits financial records."""
    for method, path in (
        ("get", "/audits"), ("get", "/audits/anything"), ("get", "/reports"),
        ("get", "/reports/anything"), ("get", "/reports/anything/filed"),
    ):
        response = getattr(client, method)(path)
        assert response.status_code == 401, path
        assert response.headers["www-authenticate"] == "Bearer"

    assert client.post("/findings/f-x/review",
                       json={"action": "clear", "reviewer": "a"}).status_code == 401
    assert client.post("/audits", files={"batch": ("b.txt", b"x")}).status_code == 401


def test_health_is_deliberately_open(client):
    """A load balancer cannot carry a bearer token, and this returns counts, not report content."""
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["authenticated"] is True


def test_the_wrong_token_is_a_401(client):
    assert client.get("/reports", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/reports", headers={"Authorization": TOKEN}).status_code == 401, (
        "a bare token with no Bearer scheme is not authentication"
    )


def test_an_unconfigured_service_is_closed_not_open(client, monkeypatch):
    """The safe default. A missing token means the service cannot authenticate anyone, which is a
    server state -- hence 503 -- and never a reason to serve the endpoints unprotected."""
    monkeypatch.setenv("API_AUTH_TOKEN", "")
    from src.config import reset_caches

    reset_caches()
    response = client.get("/reports", headers=AUTH)
    assert response.status_code == 503 and "API_AUTH_TOKEN" in response.json()["detail"]


# --- health -------------------------------------------------------------------------------


def test_health_reports_the_corpus_it_would_actually_query(client):
    body = client.get("/health").json()
    assert body["collection"] == "rule_chunks" and body["vectors"] == 731
    assert body["by_authority"]["binding"] == 231


def test_health_fails_when_the_collection_is_empty(client, monkeypatch):
    """A 200 here would send every candidate to needs_review one paid call at a time."""
    monkeypatch.setattr(main, "counts", lambda: {"total": 0, "tier": {}, "authority": {}})
    assert client.get("/health").status_code == 503


def test_health_fails_when_the_store_is_unreachable(client, monkeypatch):
    def boom():
        raise RuntimeError("no such collection")

    monkeypatch.setattr(main, "counts", boom)
    assert client.get("/health").status_code == 503


# --- Journey 1: submit and poll -----------------------------------------------------------


@needs_ledger
def test_a_batch_is_accepted_and_audited_on_the_worker(client):
    """202 with an id, not a held connection."""
    response = post_batch(client)
    assert response.status_code == 202

    body = response.json()
    assert body["status"] == "running" and body["deduplicated"] is False
    # Derived, not pinned -- the batch was 220 messages before Phase 2 regenerated the ledgers.
    declared = int(
        next(l for l in BATCH.read_text().splitlines() if l.startswith("Messages in batch"))
        .split(":")[1]
    )
    assert body["transactions"] == declared, "validated during upload, before the audit ran"
    assert body["poll"] == f"/audits/{body['job_id']}"

    finished = await_job(client, body["job_id"])
    assert finished["status"] == "complete"
    assert finished["report"]["risk_rating"] == "medium"
    assert finished["finished_at"] is not None


@needs_ledger
def test_the_job_id_the_run_id_and_the_report_id_are_one_run(client, store):
    """One id, so a trace, a job row and a stored report join without a lookup table."""
    job_id = post_batch(client).json()["job_id"]
    assert job_id.startswith("run-")
    finished = await_job(client, job_id)
    assert finished["report"]["run_id"] == job_id
    assert store.get(f"rep-{job_id}") is not None, "step 8 persisted it"


@needs_ledger
def test_a_job_outlives_the_process_that_ran_it(client, store, tmp_path):
    """The AUDITS dict this replaced could not answer GET /audits/{id} after a restart, which is
    exactly what §5.1 step 9 asks for."""
    job_id = post_batch(client).json()["job_id"]
    await_job(client, job_id)

    reopened = SqlResultsStore(f"sqlite:///{tmp_path / 'results.db'}")
    job = reopened.get_job(job_id)
    assert job is not None and job.status == "complete"
    assert reopened.get(job.report_id) is not None


def test_a_file_that_is_not_a_batch_is_refused_immediately(client):
    """A 400 in a second, not a queued job that fails a minute later."""
    response = post_batch(client, name="empty.txt", body=b"not a swift message")
    assert response.status_code == 400
    assert "no transactions" in response.json()["detail"]


def test_an_unsupported_file_type_is_refused(client):
    assert post_batch(client, name="ledger.csv", body=b"a,b,c").status_code == 415


@needs_ledger
def test_a_failing_audit_is_reported_not_swallowed(client, monkeypatch):
    def explode(path, **kwargs):
        raise RuntimeError("the vector store went away")

    monkeypatch.setattr(main, "audit_batch", explode)
    finished = await_job(client, post_batch(client).json()["job_id"])
    assert finished["status"] == "failed"
    assert "vector store went away" in finished["error"]
    assert finished["report"] is None


@needs_ledger
def test_an_unreadable_batch_is_reported_as_the_clients_file(client, monkeypatch):
    """LLD §6's one loud failure, distinguished from a server fault in the error text -- the two
    need different actions from whoever reads it."""
    def unreadable(path, **kwargs):
        raise BatchUnreadable("INGEST_FILE_UNREADABLE: yielded no readable transactions")

    monkeypatch.setattr(main, "audit_batch", unreadable)
    finished = await_job(client, post_batch(client).json()["job_id"])
    assert finished["status"] == "failed"
    assert finished["error"].startswith("INGEST_FILE_UNREADABLE")


@needs_ledger
def test_the_uploaded_file_does_not_outlive_the_audit(client, monkeypatch):
    """Every submission writes a temp file. Left behind, they accumulate silently."""
    seen: list[Path] = []
    original = main.audit_batch

    def spy(path, **kwargs):
        seen.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(main, "audit_batch", spy)
    await_job(client, post_batch(client).json()["job_id"])
    assert seen and not seen[0].exists()


def test_an_unknown_audit_is_a_404(client):
    assert client.get("/audits/run-does-not-exist", headers=AUTH).status_code == 404


@needs_ledger
def test_the_job_listing_omits_the_report_bodies(client):
    await_job(client, post_batch(client).json()["job_id"])
    listed = client.get("/audits", headers=AUTH).json()
    assert len(listed) == 1 and "report" not in listed[0]
    assert listed[0]["status"] == "complete" and listed[0]["report_id"]


# --- the dedup ----------------------------------------------------------------------------


@needs_ledger
def test_the_same_batch_posted_twice_returns_the_same_report(client):
    """The phase's own criterion. The retry worth protecting against is a client re-posting because
    the first response was slow, and the audit is the expensive thing here."""
    first = post_batch(client).json()
    await_job(client, first["job_id"])

    second = post_batch(client).json()
    assert second["job_id"] == first["job_id"]
    assert second["deduplicated"] is True

    assert len(client.get("/audits", headers=AUTH).json()) == 1, "nothing ran a second time"


@needs_ledger
def test_a_re_post_while_the_first_is_still_running_is_also_deduplicated(client, monkeypatch):
    """The case that actually costs money: an impatient client posting again mid-run."""
    release = {"go": False}

    def slow(path, **kwargs):
        while not release["go"]:
            time.sleep(0.01)
        return RunResult(
            report=a_report("run-slow"), validation=ValidationReport(parsed=1),
            usage=UsageLedger(), run_id="run-slow", candidates=0, records=1,
        )

    monkeypatch.setattr(main, "audit_batch", slow)
    first = post_batch(client).json()
    second = post_batch(client).json()
    release["go"] = True

    assert second["job_id"] == first["job_id"] and second["deduplicated"] is True


@needs_ledger
def test_a_failed_batch_can_be_retried(client, monkeypatch):
    """A failure may have been the vector store being briefly down. Refusing the retry would make a
    transient fault permanent."""
    def explode(path, **kwargs):
        raise RuntimeError("transient")

    monkeypatch.setattr(main, "audit_batch", explode)
    first = post_batch(client).json()
    await_job(client, first["job_id"])

    second = post_batch(client).json()
    assert second["job_id"] != first["job_id"] and second["deduplicated"] is False


@needs_ledger
def test_force_re_audits_the_same_bytes(client):
    """A real thing to want: the corpus was rebuilt and the same batch should be re-examined."""
    first = post_batch(client).json()
    await_job(client, first["job_id"])
    second = post_batch(client, force="true").json()
    assert second["job_id"] != first["job_id"] and second["deduplicated"] is False


@needs_ledger
def test_a_different_batch_is_not_deduplicated(client):
    """Dedup is on the bytes, so two genuinely different batches must both run."""
    first = post_batch(client).json()
    await_job(client, first["job_id"])
    other = post_batch(client, name=BATCH.name, body=BATCH.read_bytes() + b"\n").json()
    assert other["job_id"] != first["job_id"]


# --- Journey 2: one round trip ------------------------------------------------------------


@needs_ledger
def test_wait_returns_the_report_in_the_response(client):
    """HLD Journey 2 wants the report in the response; the LLD wants 202 + poll. This is the agreed
    resolution -- the same queue and the same worker, held open for one caller."""
    body = post_batch(client, wait="true").json()
    assert body["status"] == "complete"
    assert body["report"]["report_id"].startswith("rep-run-")


@needs_ledger
def test_wait_degrades_to_the_job_id_rather_than_hanging(client, monkeypatch):
    """A 10,000-message batch will not finish inside a proxy's patience, and a gateway timeout is a
    worse answer than an id to poll."""
    release = {"go": False}

    def slow(path, **kwargs):
        while not release["go"]:
            time.sleep(0.01)
        return RunResult(
            report=a_report("run-slow"), validation=ValidationReport(parsed=1),
            usage=UsageLedger(), run_id="run-slow", candidates=0, records=1,
        )

    monkeypatch.setattr(main, "audit_batch", slow)
    body = post_batch(client, wait="true", timeout="0.05").json()
    release["go"] = True
    assert body["status"] == "running" and body["poll"].endswith(body["job_id"])


@needs_ledger
def test_wait_on_an_already_finished_batch_answers_immediately(client):
    """A dedup hit on a complete job: the report is right there, so hand it over."""
    await_job(client, post_batch(client).json()["job_id"])
    body = post_batch(client, wait="true").json()
    assert body["status"] == "complete" and body["report"] is not None


# --- Journey 3: audit-defence lookup ------------------------------------------------------


def stored_report(store) -> ComplianceReport:
    filed = a_report("run-journey3", findings=[a_finding()])
    store.save(filed)
    return filed


def test_a_stored_report_is_readable_without_its_job(client, store):
    filed = stored_report(store)
    body = client.get(f"/reports/{filed.report_id}", headers=AUTH).json()
    assert body["report_id"] == filed.report_id and len(body["findings"]) == 1


def test_the_report_listing_omits_the_bodies(client, store):
    stored_report(store)
    listed = client.get("/reports", headers=AUTH).json()
    assert len(listed) == 1 and "summary" not in listed[0]
    assert listed[0]["findings"] == 1


def test_the_listing_can_be_narrowed_to_a_period(client, store):
    stored_report(store)
    assert len(client.get("/reports", params={"period": "2023-06"}, headers=AUTH).json()) == 1
    assert client.get("/reports", params={"period": "2023-07"}, headers=AUTH).json() == []


def test_reviewing_a_finding_changes_its_status_but_not_the_filed_report(client, store):
    """The criterion, over HTTP: escalating changes the status without mutating report_json."""
    filed = stored_report(store)
    response = client.post(
        "/findings/f-structuring-a/review",
        json={"action": "escalate", "reviewer": "analyst@bank", "note": "unusual counterparties"},
        headers=AUTH,
    )
    assert response.status_code == 200 and response.json()["status"] == "escalated"

    live = client.get(f"/reports/{filed.report_id}", headers=AUTH).json()
    assert live["findings"][0]["status"] == "escalated"

    as_filed = client.get(f"/reports/{filed.report_id}/filed", headers=AUTH).json()
    assert as_filed["findings"][0]["status"] == "pending_review"


def test_the_review_history_comes_back_with_the_decision(client, store):
    stored_report(store)
    client.post("/findings/f-structuring-a/review",
                json={"action": "escalate", "reviewer": "analyst@bank"}, headers=AUTH)
    body = client.post("/findings/f-structuring-a/review",
                       json={"action": "approve", "reviewer": "officer@bank", "note": "filing"},
                       headers=AUTH).json()
    assert [entry["action"] for entry in body["history"]] == ["escalate", "approve"]


def test_an_impossible_transition_is_a_409_not_a_500(client, store):
    """Approving something nobody escalated is a conflict with the current state, which the caller
    can understand and act on."""
    stored_report(store)
    response = client.post("/findings/f-structuring-a/review",
                           json={"action": "approve", "reviewer": "officer@bank"}, headers=AUTH)
    assert response.status_code == 409 and "cannot approve" in response.json()["detail"]


def test_reviewing_an_unknown_finding_is_a_404(client, store):
    stored_report(store)
    response = client.post("/findings/f-nothing/review",
                           json={"action": "clear", "reviewer": "analyst@bank"}, headers=AUTH)
    assert response.status_code == 404


def test_an_unsupported_review_action_is_rejected_by_the_schema(client, store):
    stored_report(store)
    response = client.post("/findings/f-structuring-a/review",
                           json={"action": "delete", "reviewer": "analyst@bank"}, headers=AUTH)
    assert response.status_code == 422


def test_an_unknown_report_is_a_404(client):
    assert client.get("/reports/rep-nothing", headers=AUTH).status_code == 404
    assert client.get("/reports/rep-nothing/filed", headers=AUTH).status_code == 404
