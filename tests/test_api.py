"""The HTTP surface (§10).

The graph itself is stubbed: these tests are about the contract -- what a caller gets back, when,
and what happens when the upload or the corpus is wrong. No API key, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.graph.state import ComplianceReport

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
BATCH = LEDGER / "2023-06_private_banking_log.txt"
needs_ledger = pytest.mark.skipif(not BATCH.exists(), reason="run: uv run finguard-ledger")

REPORT = ComplianceReport(
    risk_rating="Medium",
    flagged_wires=["FGO23060100001"],
    applicable_regulations=["AML Rulebook 14.2.3.Guidance.1."],
    audit_summary="## Findings",
    source_document_hashes=["obliqa:1:14.2.3.Guidance.1.:a3f9c210"],
)


@pytest.fixture(autouse=True)
def clean_registry():
    main.AUDITS.clear()
    yield
    main.AUDITS.clear()


@pytest.fixture
def client(monkeypatch):
    """A graph that returns instantly, so the contract is tested rather than the model."""
    class StubGraph:
        def invoke(self, state, config=None):
            return {"report": REPORT, "confidence_score": 0.9, "loop_count": 1}

    monkeypatch.setattr(main, "build_graph", lambda: StubGraph())
    monkeypatch.setattr(main, "stats", lambda: {
        "vectors": 12273, "collection": "regulations", "backend": "minilm",
    })
    # TestClient runs background tasks synchronously on response, so a poll right after the POST
    # already sees the finished audit.
    return TestClient(main.app)


# --- health -----------------------------------------------------------------------------


def test_health_reports_the_corpus_it_would_actually_query(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["vectors"] == 12273


def test_health_fails_when_the_collection_is_empty(client, monkeypatch):
    """A 200 from a service with no vectors sends every audit into a retrieval that returns
    nothing -- which audit_node refuses outright. Better to fail at the probe."""
    monkeypatch.setattr(main, "stats", lambda: {
        "vectors": 0, "collection": "regulations", "backend": "minilm",
    })
    assert client.get("/health").status_code == 503


def test_health_fails_when_the_store_is_unreachable(client, monkeypatch):
    def boom():
        raise RuntimeError("no such collection")

    monkeypatch.setattr(main, "stats", boom)
    assert client.get("/health").status_code == 503


# --- submitting -------------------------------------------------------------------------


@needs_ledger
def test_a_batch_is_accepted_and_audited_in_the_background(client):
    """202 with an id, not a 60-second held connection."""
    response = client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())})
    assert response.status_code == 202

    body = response.json()
    assert body["status"] == "running"
    assert body["wires"] == 220, "validated during upload, before the audit ran"
    assert body["poll"] == f"/audit/{body['audit_id']}"

    result = client.get(body["poll"]).json()
    assert result["status"] == "complete"
    assert result["report"]["risk_rating"] == "Medium"
    assert result["confidence_score"] == 0.9


@needs_ledger
def test_the_trace_id_and_the_resource_id_are_the_same_run(client):
    """One id, so a LangSmith trace and an API result join without a lookup table."""
    body = client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())}).json()
    assert body["audit_id"].startswith("aud-")


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
    class Exploding:
        def invoke(self, state, config=None):
            raise RuntimeError("the vector store went away")

    monkeypatch.setattr(main, "build_graph", lambda: Exploding())
    body = client.post("/audit", files={"batch": (BATCH.name, BATCH.read_bytes())}).json()

    result = client.get(body["poll"]).json()
    assert result["status"] == "failed"
    assert "vector store went away" in result["error"]
    assert result["report"] is None


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
