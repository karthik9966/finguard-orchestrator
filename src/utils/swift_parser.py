"""Read a batch log of SWIFT MT103 messages into structured wires (§4.2, Extraction).

This is the deterministic half of the blueprint's Extraction Node. §4.2 nominates
``gpt-4o-mini`` for the job; a regex state machine does it for $0.00 in ~100 ms with a result
that is byte-identical on every run, so the model is demoted to a per-message *fallback* for
input this parser refuses (§9.2's "deterministic parser script costing $0.00").

Refusing is the point. Two details of the MT103 grammar are silent 100x bugs if guessed at:

*The comma is the decimal separator.* ``:32A:230601GBP5669,49`` is GBP 5,669.49. SWIFT forbids
a thousands separator entirely, so ``float(value.replace(",", ""))`` yields 566949.0 -- a
hundredfold overstatement inside a regulatory filing. We parse to ``Decimal`` and reject any
amount carrying a ``.``, because at that point the intent is genuinely ambiguous.

*A field is not a line.* ``:50K:`` is the account number followed by three continuation lines
carrying the ordering customer's name and address. A reader that treats every line as its own
tag drops the customer from the wire, and a SAR that cannot name who sent the money is worthless.

Usage::

    uv run python -m src.utils.swift_parser data/processed/ledger/2023-06_private_banking_log.pdf
    uv run python -m src.utils.swift_parser data/processed/ledger/2023-06_private_banking_log.txt --json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

# A tag is two digits and an optional letter, anchored at the start of a line. Anything else
# inside a message is a continuation of whichever tag is currently open -- which is how
# ":72:/INS/CHEQUE 04:22:05" keeps its colons without being mistaken for a new field.
TAG = re.compile(r"^:(\d{2}[A-Z]?):(.*)$")

BLOCK_1 = re.compile(r"\{1:F01(?P<terminal>[A-Z0-9]+?)\d{10}\}")
BLOCK_2 = re.compile(r"\{2:I103(?P<receiver>[A-Z0-9]{8,11})[NUS]\}")
BLOCK_3 = re.compile(r"\{3:\{121:(?P<uetr>[0-9a-fA-F-]{36})\}\}")

MESSAGE_OPEN = "{1:"
MESSAGE_CLOSE = "-}"

# :32A: is fixed-width by construction: YYMMDD, then a 3-letter ISO currency, then the amount.
VALUE_DATE_AMOUNT = re.compile(r"^(?P<value_date>\d{6})(?P<currency>[A-Z]{3})(?P<amount>[\d.,]+)$")

DECLARED_COUNT = re.compile(r"^Messages in batch\s*:\s*(\d+)", re.MULTILINE)
STATEMENT_REF = re.compile(r"^Statement reference\s*:\s*(\S+)", re.MULTILINE)

# Present in every message pdf_generator.py emits. A message missing any of these cannot be
# turned into a ledger row, so it goes to the fallback rather than being half-parsed.
REQUIRED_TAGS = ("20", "23B", "32A", "50K", "52A", "57A", "59")


class MalformedMessage(ValueError):
    """One message this parser will not guess at. Carries enough context for the LLM fallback."""

    def __init__(self, reason: str, *, reference: str | None, ordinal: int, lines: list[str]):
        self.reason = reason
        self.reference = reference
        self.ordinal = ordinal
        self.lines = lines
        where = reference or f"message #{ordinal}"
        super().__init__(f"{where}: {reason}")

    @property
    def raw(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class Wire:
    """One customer credit transfer, normalized.

    ``amount`` is a ``Decimal`` deliberately: this value ends up in a filed report, and binary
    floats cannot represent 5669.49 exactly. Callers building a DataFrame convert explicitly.
    """

    reference: str
    value_date: date
    currency: str
    amount: Decimal
    sender_account: str
    sender_name: str
    sender_address: str
    sender_bic: str
    sender_country: str
    receiver_account: str
    receiver_name: str
    receiver_address: str
    receiver_bic: str
    receiver_country: str
    bank_operation_code: str
    memo: str = ""
    charge_code: str = ""
    instruction: str = ""
    uetr: str = ""

    @property
    def is_cross_border(self) -> bool:
        """Drives the tier-2 retrieval widen: a cross-border leg pulls in more of the rulebook."""
        return self.sender_country != self.receiver_country

    @property
    def corridor(self) -> str:
        return f"{self.sender_country}->{self.receiver_country}"

    def as_row(self) -> dict:
        """Flat, JSON-safe, pandas-friendly. Money becomes float only at this boundary."""
        row = asdict(self)
        row["value_date"] = self.value_date.isoformat()
        row["amount"] = float(self.amount)
        row["corridor"] = self.corridor
        row["is_cross_border"] = self.is_cross_border
        return row


@dataclass
class Batch:
    """Everything one log file yielded, including what it refused."""

    source: Path
    statement_reference: str | None
    declared_messages: int | None
    wires: list[Wire] = field(default_factory=list)
    failures: list[MalformedMessage] = field(default_factory=list)

    @property
    def parsed(self) -> int:
        return len(self.wires)

    @property
    def complete(self) -> bool:
        """Did we recover every message the statement header claims it contains?"""
        return self.declared_messages is None or self.parsed == self.declared_messages


# --- text loading ---------------------------------------------------------------------


LEDGER_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "ledger"


def existing_log(value: str) -> Path:
    """argparse ``type`` for a batch log: it must exist before any work starts.

    Without this the failure surfaces as a ``FileNotFoundError`` raised inside a LangGraph node,
    thirty frames deep, long after the run has begun. Validating at the CLI boundary turns that
    into argparse's own one-line error -- and names the batches actually on disk, since pasting
    the usage placeholder (``--batch BATCH``) is the way this goes wrong in practice.
    """
    path = Path(value)
    if path.is_file():
        return path
    available = sorted(p.name for p in LEDGER_DIR.glob("*.pdf")) if LEDGER_DIR.is_dir() else []
    hint = f"\navailable: {', '.join(available)}" if available else ""
    raise argparse.ArgumentTypeError(f"no such batch log: {value}{hint}")


def read_text(path: Path) -> str:
    """A batch arrives as either the rendered PDF or the source text; both parse identically.

    pdf_generator.py writes one PDF cell per line, so extraction round-trips the lines. Blank
    lines are dropped along the way, which is why messages are delimited by ``{1:``/``-}``
    rather than by blank-line separation.
    """
    if path.suffix.lower() == ".pdf":
        import pypdf

        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def split_messages(text: str) -> list[list[str]]:
    """Slice the log into message blocks, ignoring statement headers and footers."""
    messages: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith(MESSAGE_OPEN):
            current = [stripped]
        elif current is not None:
            current.append(stripped)
            if stripped == MESSAGE_CLOSE:
                messages.append(current)
                current = None
    if current is not None:
        messages.append(current)  # unterminated; parse_message reports it properly
    return messages


# --- field parsing --------------------------------------------------------------------


def read_fields(lines: list[str]) -> dict[str, list[str]]:
    """Collapse a message body into tag -> [value, continuation, ...].

    The state machine *is* the fix for the four-line ``:50K:`` field: a line that does not
    open a tag belongs to whichever tag is still open. The block delimiters are the exception --
    ``-}`` closes the message, and absorbing it would append the terminator to whichever field
    happens to come last (``:72:`` in every message we render).
    """
    fields: dict[str, list[str]] = {}
    open_tag: str | None = None
    for line in lines:
        if line == MESSAGE_CLOSE or line.startswith(MESSAGE_OPEN):
            open_tag = None
            continue
        match = TAG.match(line)
        if match:
            open_tag = match.group(1)
            fields[open_tag] = [match.group(2).strip()] # type: ignore
        elif open_tag is not None and line.strip():
            fields[open_tag].append(line.strip())
    return fields


def parse_amount(raw: str) -> Decimal:
    """SWIFT writes 5669,49 -- comma decimal, no thousands separator, mandatory decimal comma.

    A dot here means the value has been reformatted by something upstream and we can no longer
    tell 1.234 (one point two three four) from 1.234 (one thousand two hundred thirty four),
    so we refuse instead of picking one.
    """
    if "." in raw:
        raise ValueError(f"amount {raw!r} contains '.', which MT103 does not permit")
    if raw.count(",") != 1:
        raise ValueError(f"amount {raw!r} must carry exactly one decimal comma")
    try:
        value = Decimal(raw.replace(",", "."))
    except InvalidOperation:
        raise ValueError(f"amount {raw!r} is not a number") from None
    if value <= 0:
        raise ValueError(f"amount {raw!r} is not positive")
    return value


def parse_value_date(raw: str) -> date:
    year, month, day = int(raw[0:2]), int(raw[2:4]), int(raw[4:6])
    return date(2000 + year, month, day)


def parse_party(values: list[str]) -> tuple[str, str, str]:
    """``:50K:``/``:59:`` -> (account, name, address). Account carries a leading slash."""
    account = values[0].lstrip("/").strip()
    if not account.isdigit():
        raise ValueError(f"party account {values[0]!r} is not an account number")
    name = values[1] if len(values) > 1 else ""
    address = ", ".join(values[2:])
    return account, name, address


def country_of(bic: str) -> str:
    """Characters 5-6 of a BIC are its ISO country. No message carries a country field."""
    if len(bic) < 6 or not bic[4:6].isalpha():
        raise ValueError(f"{bic!r} is not a BIC, so no country can be derived")
    return bic[4:6].upper()


def parse_message(lines: list[str], ordinal: int) -> Wire:
    """One message block -> one Wire. Raises rather than filling a field in with a guess."""
    fields = read_fields(lines)
    reference = fields.get("20", [None])[0] or None

    def fail(reason: str) -> MalformedMessage:
        return MalformedMessage(reason, reference=reference, ordinal=ordinal, lines=lines)

    if lines[-1] != MESSAGE_CLOSE:
        raise fail("message block is not terminated with '-}'")

    missing = [tag for tag in REQUIRED_TAGS if tag not in fields]
    if missing:
        raise fail(f"missing required tag(s) {', '.join(':' + t + ':' for t in missing)}")

    header = BLOCK_3.search(lines[0])
    money = VALUE_DATE_AMOUNT.match(fields["32A"][0])
    if not money:
        raise fail(f":32A: {fields['32A'][0]!r} is not YYMMDD + currency + amount")

    try:
        sender_bic = fields["52A"][0]
        receiver_bic = fields["57A"][0]
        sender_account, sender_name, sender_address = parse_party(fields["50K"])
        receiver_account, receiver_name, receiver_address = parse_party(fields["59"])
        return Wire(
            reference=reference, # type: ignore
            value_date=parse_value_date(money.group("value_date")),
            currency=money.group("currency"),
            amount=parse_amount(money.group("amount")),
            sender_account=sender_account,
            sender_name=sender_name,
            sender_address=sender_address,
            sender_bic=sender_bic,
            sender_country=country_of(sender_bic),
            receiver_account=receiver_account,
            receiver_name=receiver_name,
            receiver_address=receiver_address,
            receiver_bic=receiver_bic,
            receiver_country=country_of(receiver_bic),
            bank_operation_code=fields["23B"][0],
            memo=" ".join(fields.get("70", [])),
            charge_code=fields.get("71A", [""])[0],
            instruction=" ".join(fields.get("72", [])),
            uetr=header.group("uetr") if header else "",
        )
    except ValueError as error:
        raise fail(str(error)) from None


def parse_batch(path: Path | str, *, strict: bool = False) -> Batch:
    """Parse a whole log.

    By default a bad message is *collected*, not fatal: §4.2's fallback route re-parses exactly
    those with ``EXTRACTION_MODEL`` while the rest stay free. ``strict=True`` re-raises, which
    is what the parser's own CLI and tests use.
    """
    path = Path(path)
    text = read_text(path)
    declared = DECLARED_COUNT.search(text)
    reference = STATEMENT_REF.search(text)

    batch = Batch(
        source=path,
        statement_reference=reference.group(1) if reference else None,
        declared_messages=int(declared.group(1)) if declared else None,
    )
    for ordinal, lines in enumerate(split_messages(text), start=1):
        try:
            batch.wires.append(parse_message(lines, ordinal))
        except MalformedMessage as error:
            if strict:
                raise
            batch.failures.append(error)
    return batch


# --- CLI ------------------------------------------------------------------------------


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch", type=existing_log, help="a .pdf or .txt batch log")
    parser.add_argument("--json", action="store_true", help="dump parsed wires as JSON lines")
    parser.add_argument("--strict", action="store_true", help="raise on the first bad message")
    args = parser.parse_args()

    batch = parse_batch(args.batch, strict=args.strict)

    if args.json:
        for wire in batch.wires:
            print(json.dumps(wire.as_row()))
        return 0

    print(f"source     : {batch.source.name}")
    print(f"statement  : {batch.statement_reference}")
    print(f"parsed     : {batch.parsed} of {batch.declared_messages} declared")
    if batch.wires:
        total = sum(w.amount for w in batch.wires)
        cross = sum(1 for w in batch.wires if w.is_cross_border)
        dates = [w.value_date for w in batch.wires]
        print(f"value      : {total:,.2f} across {len({w.currency for w in batch.wires})} currencies")
        print(f"period     : {min(dates)} to {max(dates)}")
        print(f"cross-border: {cross} wires ({cross / len(batch.wires):.1%})")
    for failure in batch.failures:
        print(f"REFUSED    : {failure}")
    return 0 if batch.complete and not batch.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
