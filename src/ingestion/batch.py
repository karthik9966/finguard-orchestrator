"""Transaction batch ingestion (LLD §2.2).

Turns a month's statements into `TransactionRecord`s and a `ValidationReport`, in the LLD's order:

    rec = deterministic_parse(line)
    if not valid(rec):
        rec = llm_extract(line)          # light_model, prompt C -- extract, never fabricate
        rec.extraction_method = 'llm_fallback'
    if not valid(rec) or missing_required(rec):
        quarantine(line); log(); continue
    records.append(rec)

Two things about that order are load-bearing.

**The fallback runs once.** LLD §6 gives `INGEST_PARSE_FAILURE` exactly one retry. A model that
could not read a malformed message the first time will usually produce something *plausible* on
the second, and a plausible account number in a filing is worse than a refusal.

**Quarantine is a result, not an error.** A message neither path could read is recorded with its
raw text and excluded from the records. Silently dropping it would let a batch that half-parsed
report as a clean batch -- the audit would be describing transactions it never saw.

This lives here rather than in `graph/nodes.py` because ingestion is not a graph concern, and
because LLD §5.1 puts parsing outside the graph entirely -- `graph/run.py` calls it before a run id
is minted, so an unreadable file is a client error rather than a failed run.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from src.models import QuarantinedMessage, TransactionRecord, ValidationReport
from src.utils.swift_parser import MalformedMessage, parse_batch, to_record


def slice_month(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    """One month of a continuous ledger, by `Date` in [month_start, month_end].

    LLD §2.2's batch boundary. No cross-month state is retained, which is why every detector's
    window is bounded by construction rather than by a check.
    """
    period = pd.Period(month, freq="M")
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    inside = (dates >= period.start_time) & (dates <= period.end_time)
    return frame.loc[inside].reset_index(drop=True)


class TransactionBatchIngestor:
    """Parse statements into records; rescue what the parser refuses; quarantine the rest."""

    def __init__(self, *, fallback=None) -> None:
        # Injected so the suite never needs a key. Defaults to the real light-model extractor.
        self._fallback = fallback if fallback is not None else llm_extract

    def ingest(self, paths: Iterable[Path | str]) -> tuple[list[TransactionRecord], ValidationReport]:
        records: list[TransactionRecord] = []
        report = ValidationReport()
        names: list[str] = []
        declared = 0
        saw_declaration = False

        for path in paths:
            path = Path(path)
            names.append(path.name)
            batch = parse_batch(path, strict=False)
            if batch.declared_messages is not None:
                declared += batch.declared_messages
                saw_declaration = True

            for wire in batch.wires:
                records.append(to_record(wire))

            for ordinal, failure in enumerate(batch.failures, start=1):
                rescued = self._rescue(failure)
                if rescued is not None:
                    records.append(rescued)
                    report.rescued += 1
                    continue
                report.quarantined.append(
                    QuarantinedMessage(
                        ordinal=failure.ordinal or ordinal,
                        reference=failure.reference,
                        reason=failure.reason,
                        raw=failure.raw,
                        fallback_attempted=True,
                    )
                )

        report.batch = ", ".join(names)
        report.declared = declared if saw_declaration else None
        report.parsed = len(records)
        return records, report

    def _rescue(self, failure: MalformedMessage) -> TransactionRecord | None:
        """One attempt, and only one. A second pass invents rather than reads."""
        try:
            rescued = self._fallback(failure)
        except Exception:  # noqa: BLE001 - a failed rescue is quarantined, never guessed at
            return None
        if rescued is None:
            return None
        rescued.extraction_method = "llm_fallback"
        return rescued


def llm_extract(failure: MalformedMessage) -> TransactionRecord | None:
    """LLD §4.1 prompt C: read the fields that are there, never invent one that is not.

    Returns None rather than raising, so one unreadable message never costs the other 499.
    """
    from src.config import get_settings
    from src.graph import prompts

    settings = get_settings()
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.light_model, temperature=0
    ).with_structured_output(prompts.ExtractedWire)

    extracted = model.invoke(
        [
            ("system", prompts.EXTRACTION_SYSTEM),
            ("user", prompts.EXTRACTION_USER.format(reason=failure.reason, raw=failure.raw)),
        ]
    )
    return _record_from_extraction(extracted)


def _record_from_extraction(extracted) -> TransactionRecord | None:
    """Validate what the model returned. A field it could not read stays unread.

    The amount is the one that matters: MT103 writes 5669,49 with a comma decimal, and
    `float("5810,46".replace(",", ""))` is 581046.00 -- a hundredfold error inside a filing. The
    prompt says so and this checks it anyway.
    """
    try:
        year, month, day = (int(part) for part in extracted.value_date.split("-"))
        amount = Decimal(extracted.amount)
    except (AttributeError, ValueError, InvalidOperation):
        return None
    if amount <= 0:
        return None

    return TransactionRecord(
        txn_ref=extracted.reference,
        sender_account=extracted.sender_account,
        receiver_account=extracted.receiver_account,
        amount=amount,
        currency=extracted.currency,
        timestamp=datetime.combine(date(year, month, day), datetime.min.time(), tzinfo=timezone.utc),
        sender_country=extracted.sender_bic[4:6],
        receiver_country=extracted.receiver_bic[4:6],
        instrument="UNKNOWN",
        extraction_method="llm_fallback",
    )
