"""One redaction pass, shared by the two places data leaves the process -- HLD §5, §6.

The pre-migration system had none, and the consequence was recorded rather than hypothetical:
tracing uploaded full node inputs and outputs, ~137 KB per run, including every parsed wire with
its counterparty names and account numbers -- most of which no model ever saw. Fine for a
synthetic ledger; a real decision before pointing the system at live payment data.

There are exactly two exits, and they get the same function so they cannot drift:

1. the grounding context, before it reaches the hosted model (LLD §2 Group 4); and
2. the observability payload, before it reaches Langfuse (HLD §6).

**What this is, precisely.** Accounts are *pseudonymised*, not anonymised: the same account maps
to the same token every time, because a fan-in narrative that cannot say "these eleven senders
all paid the same account" is useless, and an analyst has to be able to map a finding back to the
ledger. Determinism is bought with reversibility -- an adversary holding both the token and a list
of candidate account numbers can confirm a match. Set ``REDACTION_PEPPER`` to break that
correlation at the cost of cross-run comparability; on synthetic data the default is the right
trade, and stating it is better than implying an anonymity guarantee that is not there.

**Memos are scrubbed, not dropped.** A memo line is the one realistic prompt-injection vector the
system has (Evaluation Design §5), so removing it would make the injection fixture vacuous. The
text is kept as inert data with identifier-shaped substrings masked out.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

# Optional. Absent by default so a token is stable across runs and machines, which is what makes
# a trace comparable to the one before it.
_PEPPER = os.environ.get("REDACTION_PEPPER", "")

ACCOUNT_TOKEN_CHARS = 8

# Field names whose values are account identifiers. Matched case-insensitively against the last
# path segment, so `attributes.anchor_account` is covered without enumerating every nesting.
ACCOUNT_FIELDS = frozenset(
    {
        "account",
        "sender_account",
        "receiver_account",
        "anchor",
        "anchor_account",
        "originator",
        "beneficiary",
        "counterparty",
        "counterparties",
        "collector_account",
    }
)

# Free text that may carry identifiers a regex can find, but whose content is otherwise wanted.
FREE_TEXT_FIELDS = frozenset(
    {"memo", "narrative", "summary", "reason", "note", "notes", "review_notes", "text_excerpt"}
)

# Fields that are *not* PII and must survive, because redacting them would defeat the purpose of
# the trace or the citation. Listed explicitly so a future reader can see the line being drawn.
PRESERVED_FIELDS = frozenset(
    {
        "txn_ref",
        "member_txn_refs",
        "flagged_transactions",
        "candidate_id",
        "finding_id",
        "report_id",
        "run_id",
        "batch_id",
        "chunk_id",
        "source_id",
        "section_ref",
        "pattern_type",
        "risk_level",
        "risk_rating",
        "status",
        "tier",
        "authority",
        "jurisdiction",
        "currency",
        "instrument",
        "sender_country",
        "receiver_country",
    }
)

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# IBAN-shaped: two letters, two check digits, then 10+ alphanumerics.
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
# Any run of 7 or more digits, optionally grouped by spaces or hyphens and optionally carrying a
# leading "+". Accounts, cards and phone numbers all have this shape.
#
# One label for all three, deliberately. An earlier version tried to tell a phone number from an
# account number with separate patterns, which was false precision in two directions: the account
# pattern matched first and mislabelled "+44 7700 900123" as an account, and an 8-digit account
# fell through the 9-digit floor entirely. A regex cannot distinguish these reliably, so it should
# not claim to -- it only has to guarantee that neither reaches the trace.
#
# "." is NOT a separator here, which is what keeps decimal amounts intact: "95000.00" reads as a
# 5-digit and a 2-digit run, not a 7-digit identifier. The residual cost is that an amount above
# a million written without separators ("10000000") is masked inside free text. Worth it -- the
# amount a finding relies on comes from the structured field, never from the memo line.
# `(?!\d{4}-\d{2}-\d{2})` exempts an ISO date. Without it "2023-06-14" reads as an 8-digit run with
# hyphen separators and is masked -- which matters because `narrative` is scrubbed free text, and a
# finding whose explanation says "three transfers on [REDACTED]" has lost the thing that makes it
# checkable. A hyphenated phone number is 3-3-4 and still matches; only the 4-2-2 shape is exempt.
_DIGIT_RUN = re.compile(r"(?<![\d.])(?!\d{4}-\d{2}-\d{2})\+?(?:\d[ -]?){6,}\d(?![\d.])")


def pseudonymise_account(value: str) -> str:
    """A stable, non-obvious token for one account identifier.

    Short by design: eight hex characters is 4 billion buckets, which for a single institution's
    ledger makes a collision a curiosity rather than a risk, and keeps a prompt listing twelve
    counterparties readable.
    """
    if not value or not str(value).strip():
        return ""
    digest = hashlib.sha256(f"{_PEPPER}|{str(value).strip()}".encode("utf-8")).hexdigest()
    return f"ACCT-{digest[:ACCOUNT_TOKEN_CHARS]}"


def scrub(text: str) -> str:
    """Mask identifier-shaped substrings in free text, leaving everything else intact."""
    if not text:
        return text
    scrubbed = _EMAIL.sub("[EMAIL]", text)
    scrubbed = _IBAN.sub("[REDACTED]", scrubbed)
    scrubbed = _DIGIT_RUN.sub("[REDACTED]", scrubbed)
    return scrubbed


def redact(value: Any, *, field: str | None = None) -> Any:
    """Recursively redact a value for export.

    Dispatch is by *field name*, not by inspecting the value, because "is this string an account
    number" is unanswerable in general -- a five-digit account and a five-digit amount look the
    same. The caller's schema is the only reliable signal, so a field this module does not know
    about is left alone and a new identifier field must be added to ``ACCOUNT_FIELDS`` to be
    covered. That is a deliberate fail-visible choice: an unredacted new field shows up in a
    trace, where the alternative (redacting anything that looks identifier-ish) silently
    destroys amounts.
    """
    key = (field or "").lower()

    if isinstance(value, BaseModel):
        return {name: redact(item, field=name) for name, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {name: redact(item, field=str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact(item, field=field) for item in value]

    if key in ACCOUNT_FIELDS:
        return pseudonymise_account(value) if isinstance(value, str) else value
    if key in FREE_TEXT_FIELDS and isinstance(value, str):
        return scrub(value)
    if key in PRESERVED_FIELDS:
        return value
    if isinstance(value, Decimal):
        # Amounts are evidence, not identity: a report that cannot state an amount cannot justify a
        # threshold finding. Rendered as a string because that is what survives a JSON serialiser --
        # an exported Decimal came out of Langfuse as the literal "<Decimal>", which is worse than
        # useless in a trace about sub-threshold structuring.
        return str(value)
    if isinstance(value, (datetime, date)):
        return value
    if isinstance(value, str) and field is None:
        # A bare string with no field context: scrub it rather than trust it.
        return scrub(value)
    return value


def redact_record(record: Any) -> dict[str, Any]:
    """One TransactionRecord, ready for a prompt or a trace."""
    return redact(record)


def contains_identifier(text: str) -> bool:
    """True if raw text still carries something identifier-shaped. Used by the tests that assert
    a payload is clean, so the assertion and the scrubber share one definition."""
    return bool(_EMAIL.search(text) or _IBAN.search(text) or _DIGIT_RUN.search(text))
