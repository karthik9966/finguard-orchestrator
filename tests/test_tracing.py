"""Langfuse tracing -- HLD §6, and the privacy defect it exists to close.

Langfuse v4 is OpenTelemetry underneath, which means the payload can be captured instead of guessed
at: an `InMemorySpanExporter` is handed to the client and every test here asserts on the *exact*
attributes that would have gone over the wire. That is a better check than opening the UI, because it
runs in CI and it can assert an absence.

The absence is the point. The pre-migration system traced to a hosted project and uploaded ~137 KB
per run including counterparty names and account numbers. So these tests assert, against a real run
over a real batch, that no account number and no memo text from that batch reaches the payload.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.observability import tracing
from src.store import InMemoryResultsStore

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
CONTROL = LEDGER / "2023-05_private_banking_log.pdf"
needs_ledger = pytest.mark.skipif(not CONTROL.exists(), reason="run: uv run finguard-ledger")


@pytest.fixture
def spans(monkeypatch) -> InMemorySpanExporter:
    """A real Langfuse client whose spans land in memory instead of over the network.

    Two details are load-bearing and were both found the hard way. The SDK's own `span_exporter`
    hook is used rather than a custom `TracerProvider`, so the client never builds its OTLP exporter
    and no test touches the network. And the public key is unique per test, because Langfuse caches
    one client per key: with a fixed key, a client built by an earlier test is handed back here --
    ignoring the exporter -- and every span goes to a socket nobody is listening on.
    """
    from uuid import uuid4

    from langfuse import Langfuse

    exporter = InMemorySpanExporter()
    key = f"pk-lf-{uuid4().hex[:12]}"

    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("CLIENT_TIER", "tier-1-private-banking")
    from src.config import reset_caches

    reset_caches()
    tracing.reset()
    # Built here rather than through `langfuse_client()` so the exporter can be injected; the mask is
    # the real one, which is the thing under test. `CallbackHandler()` resolves the client by the
    # public key in the environment, which is why that has to be the same unique key.
    client = Langfuse(
        public_key=key, secret_key="sk-lf-test", host="http://localhost:3000",
        mask=tracing.mask, span_exporter=exporter,
    )
    monkeypatch.setattr(tracing, "_CLIENT", client)
    monkeypatch.setattr(tracing, "_CHECKED", True)
    yield exporter
    client.shutdown()
    tracing.reset()
    reset_caches()


def finished(exporter: InMemorySpanExporter):
    """The spans, after a flush. The SDK batches, so reading the exporter without flushing first
    reports an empty trace and every assertion below would pass vacuously."""
    tracing.langfuse_client().flush()
    return exporter.get_finished_spans()


def payload(exporter: InMemorySpanExporter) -> str:
    """Everything that would have left the process, as one string to search."""
    return " ".join(
        f"{key}={value}"
        for span in finished(exporter)
        for key, value in span.attributes.items()
    )


# --- the mask, on its own ------------------------------------------------------------------


def test_the_mask_pseudonymises_accounts_and_scrubs_memos():
    masked = tracing.mask(data={
        "sender_account": "6123421761",
        "memo": "/RFB/ salary for a.person@bank.com ref 8734512098",
        "txn_ref": "FGO23060100001",
    })
    assert masked["sender_account"].startswith("ACCT-")
    assert "6123421761" not in json.dumps(masked)
    assert "8734512098" not in masked["memo"] and "[EMAIL]" in masked["memo"]
    assert masked["txn_ref"] == "FGO23060100001", "references are kept -- HLD §6 asks for them"


def test_the_mask_summarises_the_ledger_rather_than_shipping_it():
    """Redaction alone does not fix the recorded defect: pseudonymised bulk is still bulk, and no
    model ever saw the ledger in this form."""
    masked = tracing.mask(data={"records": [{"txn_ref": f"T{n}"} for n in range(500)]})
    assert masked["records"] == "[500 record(s) -- omitted from the trace]"


def test_the_mask_caps_an_unexpectedly_long_list():
    masked = tracing.mask(data={"whatever": list(range(100))})
    assert len(masked["whatever"]) == tracing.TRACE_LIST_LIMIT + 1
    assert masked["whatever"][-1].endswith("more omitted")


def test_a_failing_mask_drops_the_payload_rather_than_passing_it_through(monkeypatch):
    """The only safe failure. A mask that raises would drop the span or -- worse, in some SDK
    versions -- send the unmasked original."""
    monkeypatch.setattr(tracing, "redact", lambda data: 1 / 0)
    assert tracing.mask(data={"sender_account": "6123421761"}).startswith("[REDACTION FAILED")


def test_amounts_survive_the_mask_as_numbers():
    """A trace about sub-threshold structuring that cannot state the amounts is not much use."""
    from decimal import Decimal

    masked = tracing.mask(data={"attributes": {"total": Decimal("9200.00")}})
    assert masked["attributes"]["total"] == "9200.00"


# --- tracing is optional ------------------------------------------------------------------


def test_with_no_credentials_everything_is_a_no_op(monkeypatch):
    """An audit must not fail, or change, because an observability stack is down."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_TRACING", "false")
    from src.config import reset_caches

    reset_caches()
    tracing.reset()

    assert tracing.langfuse_client() is None
    assert tracing.handler() is None
    assert tracing.tracing_target() is None
    with tracing.audit_trace(run_id="r", batch_id="b", period="2023-06", records=1) as trace_id:
        assert trace_id is None
    tracing.score_findings(None, [object()], run_id="r")  # must not raise

    tracing.reset()
    reset_caches()


def test_the_target_is_reported_rather_than_assumed(spans):
    """A trace that is silently not being written is worse than none: you go looking for it after
    the run instead of before."""
    assert tracing.tracing_target() == "http://localhost:3000"


# --- run-level tagging (HLD §6) -----------------------------------------------------------


def test_run_metadata_carries_everything_the_hld_asks_for():
    from src.config import reset_caches

    metadata = tracing.run_metadata(
        run_id="run-abc", batch_id="2023-06.pdf", period="2023-06", records=500
    )
    assert metadata["langfuse_session_id"] == "run-abc"
    assert metadata["run_id"] == "run-abc" and metadata["batch_id"] == "2023-06.pdf"
    assert metadata["period"] == "2023-06" and metadata["record_count"] == 500
    assert "client_tier" in metadata
    assert tracing.TRACE_TAG in metadata["langfuse_tags"]
    assert "period:2023-06" in metadata["langfuse_tags"]


# --- a real run ---------------------------------------------------------------------------


@needs_ledger
def test_a_run_is_traced_node_by_node(spans, monkeypatch, tmp_path):
    """HLD §6: the reasoning core traced node-by-node, so an engineer can see which step did what."""
    monkeypatch.setenv("RESULTS_DB_URL", f"sqlite:///{tmp_path / 'results.db'}")
    from src.graph.run import audit_batch

    audit_batch(CONTROL, store=InMemoryResultsStore(), tags=["TEST"])

    names = [span.name for span in finished(spans)]
    assert "audit 2023-05" in names, "the run itself is one trace"
    assert "detection" in names, "and each node is a span beneath it"
    assert "report" in names


@needs_ledger
def test_the_trace_is_tagged_and_sessioned_by_the_run(spans, monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DB_URL", f"sqlite:///{tmp_path / 'results.db'}")
    from src.graph.run import audit_batch

    result = audit_batch(CONTROL, store=InMemoryResultsStore(), tags=["TEST"])

    root = next(s for s in finished(spans) if s.name == "audit 2023-05")
    assert root.attributes["session.id"] == result.run_id, (
        "one id for the trace, the job row and the report"
    )
    tags = root.attributes["langfuse.trace.tags"]
    assert tracing.TRACE_TAG in tags and "period:2023-05" in tags
    assert "tier:tier-1-private-banking" in tags
    # OpenTelemetry attribute values are strings on the wire, so the count arrives as one.
    assert str(root.attributes["langfuse.trace.metadata.record_count"]) == "500"


@needs_ledger
def test_no_account_number_or_memo_text_reaches_the_payload(spans, monkeypatch, tmp_path):
    """The phase's own criterion, asserted against the batch's real contents rather than a fixture.

    Accounts and memos are read out of the ledger itself, so this fails if either ever starts leaking
    -- it cannot pass by testing values the batch does not contain.
    """
    monkeypatch.setenv("RESULTS_DB_URL", f"sqlite:///{tmp_path / 'results.db'}")
    from src.graph.run import audit_batch
    from src.utils.swift_parser import parse_batch

    parsed = parse_batch(CONTROL, strict=False)
    audit_batch(CONTROL, store=InMemoryResultsStore())
    emitted = payload(spans)

    accounts = {w.sender_account for w in parsed.wires} | {
        w.receiver_account for w in parsed.wires
    }
    leaked = sorted(account for account in accounts if account and account in emitted)
    assert not leaked, f"{len(leaked)} account number(s) in the trace payload: {leaked[:3]}"

    names = {w.sender_name for w in parsed.wires} | {w.receiver_name for w in parsed.wires}
    leaked_names = sorted(name for name in names if name and len(name) > 6 and name in emitted)
    assert not leaked_names, f"counterparty name(s) in the payload: {leaked_names[:3]}"

    memos = {w.memo for w in parsed.wires if w.memo and len(w.memo) > 12}
    leaked_memos = sorted(memo for memo in memos if memo in emitted)
    assert not leaked_memos, f"memo text in the payload: {leaked_memos[:2]}"


@needs_ledger
def test_the_ledger_is_not_uploaded_with_the_run(spans, monkeypatch, tmp_path):
    """The volume half of the defect. ~137 KB per run was the measurement that started this."""
    monkeypatch.setenv("RESULTS_DB_URL", f"sqlite:///{tmp_path / 'results.db'}")
    from src.graph.run import audit_batch

    audit_batch(CONTROL, store=InMemoryResultsStore())
    emitted = payload(spans)

    assert "record(s) -- omitted from the trace" in emitted
    # A clean 500-record batch: the whole payload is smaller than the old per-run figure by an order
    # of magnitude. Asserted as a ceiling rather than a pinned number, so it fails on a regression
    # rather than on a wording change.
    assert len(emitted) < 40_000, f"{len(emitted):,} bytes of trace for one clean batch"
    assert len(re.findall(r'"txn_ref"', emitted)) < 25, "the parsed ledger is not in the trace"
