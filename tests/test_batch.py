"""Transaction batch ingestion (Phase 2, LLD §2.2).

The fallback is injected, so nothing here needs a key or a network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.batch import TransactionBatchIngestor, slice_month, to_wires
from src.models import TransactionRecord
from src.utils.swift_parser import MalformedMessage, parse_batch, to_record, to_wire

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
BATCH = LEDGER / "2023-06_private_banking_log.txt"
needs_ledger = pytest.mark.skipif(not BATCH.exists(), reason="run: uv run finguard-ledger --profile dev")


def a_record(**overrides) -> TransactionRecord:
    base = dict(
        txn_ref="FGO23060100001", sender_account="111", receiver_account="222",
        amount=Decimal("5669.49"), currency="USD",
        timestamp=datetime(2023, 6, 1, 4, 22, 5, tzinfo=timezone.utc),
        sender_country="US", receiver_country="GB", instrument="WIRE",
    )
    return TransactionRecord(**{**base, **overrides})


# --- the batch boundary -------------------------------------------------------------------


def test_a_month_is_the_batch_boundary():
    """LLD §2.2: the ingestor receives one month, and no cross-month state is kept. Every
    detector's window is then bounded by construction rather than by a check."""
    frame = pd.DataFrame({"Date": ["2023-05-31", "2023-06-01", "2023-06-30", "2023-07-01"]})
    assert list(slice_month(frame, "2023-06").Date) == ["2023-06-01", "2023-06-30"]


def test_an_unparseable_date_is_excluded_rather_than_assumed():
    frame = pd.DataFrame({"Date": ["2023-06-15", "not a date"]})
    assert list(slice_month(frame, "2023-06").Date) == ["2023-06-15"]


# --- deterministic parse → fallback → quarantine -------------------------------------------


@needs_ledger
def test_a_clean_batch_needs_no_fallback():
    records, report = TransactionBatchIngestor(fallback=_never_called).ingest([BATCH])
    assert report.complete and not report.quarantined
    assert report.parsed == len(records) == report.declared
    assert all(r.extraction_method == "deterministic" for r in records)


@needs_ledger
def test_a_refused_message_is_rescued_and_marked(tmp_path):
    """The rescue is recorded as llm_fallback, not laundered into looking deterministic: a
    reviewer has to be able to see which records a model read rather than a parser."""
    corrupted = tmp_path / BATCH.name
    corrupted.write_text(BATCH.read_text().replace(":32A:", ":32A:XX", 1))

    rescued = a_record(txn_ref="RESCUED")
    records, report = TransactionBatchIngestor(fallback=lambda failure: rescued).ingest([corrupted])

    assert report.rescued == 1 and not report.quarantined
    assert [r for r in records if r.extraction_method == "llm_fallback"]


@needs_ledger
def test_a_message_neither_path_can_read_is_quarantined_not_dropped(tmp_path):
    """A quarantined message is a message the audit did not see. Dropping it silently would let
    a batch that half-parsed report as a clean batch."""
    corrupted = tmp_path / BATCH.name
    corrupted.write_text(BATCH.read_text().replace(":32A:", ":32A:XX", 1))

    records, report = TransactionBatchIngestor(fallback=lambda failure: None).ingest([corrupted])

    assert len(report.quarantined) == 1
    assert not report.complete, "an incomplete batch must not look complete"
    quarantined = report.quarantined[0]
    assert quarantined.reason and quarantined.raw.startswith("{1:")
    assert quarantined.fallback_attempted
    assert len(records) == report.declared - 1


@needs_ledger
def test_the_fallback_is_attempted_exactly_once(tmp_path):
    """LLD §6 gives INGEST_PARSE_FAILURE one retry. A model that could not read a message the
    first time produces something *plausible* on the second, and a plausible account number in a
    filing is worse than a refusal."""
    corrupted = tmp_path / BATCH.name
    corrupted.write_text(BATCH.read_text().replace(":32A:", ":32A:XX", 1))

    attempts: list[MalformedMessage] = []

    def counting(failure):
        attempts.append(failure)
        return None

    TransactionBatchIngestor(fallback=counting).ingest([corrupted])
    assert len(attempts) == 1


@needs_ledger
def test_a_fallback_that_raises_quarantines_rather_than_failing_the_batch(tmp_path):
    corrupted = tmp_path / BATCH.name
    corrupted.write_text(BATCH.read_text().replace(":32A:", ":32A:XX", 1))

    def explode(failure):
        raise RuntimeError("the model is down")

    records, report = TransactionBatchIngestor(fallback=explode).ingest([corrupted])
    assert len(report.quarantined) == 1 and records


# --- the adapters -------------------------------------------------------------------------


@needs_ledger
def test_what_a_detector_reads_survives_the_round_trip():
    """`detectors.py` runs through `to_wire` until Phase 5 replaces it. The adapter is lossy by
    design -- no BICs, no names -- but nothing a detector reads may be lost."""
    wire = parse_batch(BATCH, strict=True).wires[0]
    back = to_wire(to_record(wire))
    for field in (
        "reference", "value_date", "currency", "amount", "sender_account",
        "receiver_account", "sender_country", "receiver_country", "memo",
    ):
        assert getattr(back, field) == getattr(wire, field), field


@needs_ledger
def test_the_timestamp_carries_the_time_not_just_the_date():
    """:32A: gives a date; :72: carries the wall-clock time. Every detector is a time window, so
    a batch of midnights would collapse them all into one instant."""
    records = [to_record(w) for w in parse_batch(BATCH, strict=True).wires]
    assert all(r.timestamp.tzinfo is not None for r in records)
    assert sum(1 for r in records if r.timestamp.hour) > len(records) // 2


@needs_ledger
def test_the_memo_survives_ingestion():
    """It is attacker-controlled free text and it has to reach the candidate, or Evaluation
    Design §5's injection fixture tests nothing. Redaction covers it before any external call."""
    records, _ = TransactionBatchIngestor(fallback=_never_called).ingest([BATCH])
    assert any(r.memo for r in records)


@needs_ledger
def test_records_convert_back_for_the_old_detectors():
    records, _ = TransactionBatchIngestor(fallback=_never_called).ingest([BATCH])
    wires = to_wires(records)
    assert len(wires) == len(records)
    assert {w.reference for w in wires} == {r.txn_ref for r in records}


def _never_called(failure):
    raise AssertionError("the fallback must not run on a clean batch")
