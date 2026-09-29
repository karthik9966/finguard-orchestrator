"""The run orchestrator -- LLD §5.1 steps 1-3 and 8.

What sits outside the graph, and why: a file that yields nothing readable is a client error before
a run id exists, and where a report is stored is not a decision the reasoning core should be able
to see. These tests hold both seams.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.graph import run
from src.graph.run import BatchUnreadable, audit_batch
from src.store import InMemoryResultsStore, ResultsStore
from src.models import ComplianceReport, TransactionRecord, ValidationReport

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
CONTROL = LEDGER / "2023-05_private_banking_log.pdf"
needs_ledger = pytest.mark.skipif(not CONTROL.exists(), reason="run: uv run finguard-ledger")


class StubIngestor:
    def __init__(self, records, report=None):
        self.records = records
        self.report = report or ValidationReport(parsed=len(records))
        self.paths: list[list[Path]] = []

    def ingest(self, paths):
        self.paths.append(list(paths))
        return self.records, self.report


class StubGraph:
    """Returns a finished state, so these tests are about the orchestration and not the nodes."""

    def __init__(self, report=None, candidates=0):
        self.report = report
        self.invocations: list[dict] = []
        self.candidates = candidates

    def invoke(self, state, config=None):
        self.invocations.append({"state": state, "config": config})
        return {**state, "report": self.report, "candidates": [None] * self.candidates}


def record(ref: str, when: str) -> TransactionRecord:
    from datetime import datetime, timezone
    from decimal import Decimal

    return TransactionRecord(
        txn_ref=ref,
        sender_account="6123421761",
        receiver_account="8734512098",
        amount=Decimal("9200.00"),
        currency="USD",
        timestamp=datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
        sender_country="US",
        receiver_country="US",
        instrument="WIRE",
    )


def report_for(run_id="run-test", period="2023-06") -> ComplianceReport:
    from datetime import datetime, timezone

    return ComplianceReport(
        report_id=f"rep-{run_id}", run_id=run_id, period=period,
        generated_at=datetime.now(timezone.utc), risk_rating="none", clean=True,
    )


# --- step 1: the one loud failure -------------------------------------------------------------


def test_a_missing_file_fails_before_a_run_id_exists(tmp_path):
    with pytest.raises(BatchUnreadable, match=run.INGEST_FILE_UNREADABLE):
        audit_batch(tmp_path / "nothing.pdf")


def test_a_file_that_yields_no_records_is_a_client_error_not_a_clean_report(tmp_path):
    """The failure this prevents: a batch that parsed nothing reporting as a clean month."""
    empty = tmp_path / "empty.txt"
    empty.write_text("not a swift message")
    graph = StubGraph(report_for())
    with pytest.raises(BatchUnreadable, match="no readable transactions"):
        audit_batch(empty, graph=graph, ingestor=StubIngestor([]))
    assert graph.invocations == [], "the graph is never entered"


# --- steps 2-3: what reaches the graph --------------------------------------------------------


@pytest.fixture(autouse=True)
def never_the_repo_database(monkeypatch, tmp_path):
    """`audit_batch` now persists by default, which is right for a run and wrong for a test that
    did not ask for a database. Pointed at a temp file rather than stubbed, so the default path is
    the one being exercised."""
    monkeypatch.setenv("RESULTS_DB_URL", f"sqlite:///{tmp_path / 'results.db'}")
    from src.config import reset_caches

    reset_caches()
    yield
    reset_caches()


def test_the_period_comes_from_the_records_not_the_filename(tmp_path):
    """A filename is a label a human chose; the records are what is being audited."""
    mislabelled = tmp_path / "2023-01_whatever.txt"
    mislabelled.write_text("x")
    graph = StubGraph(report_for(period="2023-06"))
    audit_batch(
        mislabelled, graph=graph,
        ingestor=StubIngestor([record("FGO23060100001", "2023-06-14T09:00:00")]),
    )
    assert graph.invocations[0]["state"]["period"] == "2023-06"


def test_the_quarantine_count_travels_into_the_state(tmp_path):
    batch = tmp_path / "2023-06.txt"
    batch.write_text("x")
    from src.models import QuarantinedMessage

    validation = ValidationReport(
        parsed=1,
        quarantined=[QuarantinedMessage(ordinal=1, reason="missing :32A:")],
    )
    graph = StubGraph(report_for())
    audit_batch(batch, graph=graph, ingestor=StubIngestor(
        [record("FGO23060100001", "2023-06-14T09:00:00")], validation
    ))
    assert graph.invocations[0]["state"]["quarantined_count"] == 1


def test_the_run_id_is_one_id_shared_by_the_state_and_the_trace(tmp_path):
    """One id, so a Langfuse trace and a stored report join without a lookup table."""
    batch = tmp_path / "2023-06.txt"
    batch.write_text("x")
    graph = StubGraph(report_for())
    result = audit_batch(batch, graph=graph, ingestor=StubIngestor(
        [record("FGO23060100001", "2023-06-14T09:00:00")]
    ))
    invocation = graph.invocations[0]
    assert result.run_id == invocation["state"]["run_id"]
    assert invocation["config"]["metadata"]["run_id"] == result.run_id


def test_the_step_budget_travels_with_the_run(tmp_path):
    batch = tmp_path / "2023-06.txt"
    batch.write_text("x")
    graph = StubGraph(report_for())
    audit_batch(batch, graph=graph, ingestor=StubIngestor(
        [record(f"FGO2306010000{n}", "2023-06-14T09:00:00") for n in range(9)]
    ))
    assert graph.invocations[0]["config"]["recursion_limit"] > 25


# --- step 8: the store seam -------------------------------------------------------------------


def test_the_in_memory_store_satisfies_the_protocol():
    """Phase 6a swaps SQLite in behind this with no caller change, so the contract is asserted
    now rather than designed blind later."""
    assert isinstance(InMemoryResultsStore(), ResultsStore)


def test_a_finished_report_is_saved_under_its_own_id(tmp_path):
    batch = tmp_path / "2023-06.txt"
    batch.write_text("x")
    store = InMemoryResultsStore()
    report = report_for()
    audit_batch(batch, graph=StubGraph(report), store=store,
                ingestor=StubIngestor([record("FGO23060100001", "2023-06-14T09:00:00")]))
    assert store.get(report.report_id) is report


def test_a_run_with_no_store_named_still_persists(tmp_path):
    """Phase 6a changed this default. A run that silently discards its report is not what anyone
    wants from an audit engine, so it now takes an explicit InMemoryResultsStore to get one."""
    from src.store import SqlResultsStore

    batch = tmp_path / "2023-06.txt"
    batch.write_text("x")
    report = report_for()
    audit_batch(batch, graph=StubGraph(report),
                ingestor=StubIngestor([record("FGO23060100001", "2023-06-14T09:00:00")]))

    # Read back through a *separate* store object on the same URL -- the run's own store is gone.
    reopened = SqlResultsStore(os.environ["RESULTS_DB_URL"])
    assert reopened.get(report.report_id) is not None


def test_a_run_that_produces_no_report_is_an_error_not_a_none(tmp_path):
    """A caller receiving None here could not tell a clean batch from a crash."""
    batch = tmp_path / "2023-06.txt"
    batch.write_text("x")
    with pytest.raises(RuntimeError, match="without producing a report"):
        audit_batch(batch, graph=StubGraph(None),
                    ingestor=StubIngestor([record("FGO23060100001", "2023-06-14T09:00:00")]))


# --- what the caller gets back ----------------------------------------------------------------


def test_per_candidate_cost_is_reported_for_phase_8(tmp_path):
    """Phase 8 sets any candidate cap against this number; a cap chosen without it is a guess."""
    batch = tmp_path / "2023-06.txt"
    batch.write_text("x")
    result = audit_batch(batch, graph=StubGraph(report_for(), candidates=4),
                         ingestor=StubIngestor([record("FGO23060100001", "2023-06-14T09:00:00")]))
    assert result.candidates == 4
    # No model ran, so there is no cost to divide -- reported as absent, not as zero.
    assert result.cost_per_candidate is None


# --- the real thing ---------------------------------------------------------------------------


@needs_ledger
def test_the_clean_control_batch_costs_nothing_end_to_end(monkeypatch):
    """The phase's own criterion, on the real graph over the real control ledger: zero model
    calls, $0.0000, and a report that says clean rather than nothing at all."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = audit_batch(CONTROL)

    assert result.candidates == 0
    assert result.usage.calls == 0
    assert result.usage.total_cost is None or float(result.usage.total_cost) == 0.0
    assert result.report.clean is True and result.report.risk_rating == "none"
    assert result.report.period == "2023-05"
    assert result.report.model_dump(mode="json")["schema_version"]
