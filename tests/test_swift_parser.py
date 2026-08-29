"""Verify §4.2's deterministic extraction: exact, or it refuses.

The unit tests run on inline fixtures. The batch tests run against the real logs when they have
been generated, and check every parsed field against `ledger_labels.csv` -- the answer key
pdf_generator.py wrote at the same time as the documents.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from src.utils.swift_parser import (
    MalformedMessage,
    existing_log,
    country_of,
    parse_amount,
    parse_batch,
    parse_message,
    parse_party,
    read_fields,
    split_messages,
)

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
LABELS = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger_labels.csv"

MESSAGE = """\
{1:F01NRGTGB2L1BR0000000000}{2:I103VRDLGB3A5DPN}{3:{121:5e44476b-41c4-4c82-8efc-ff2b56db21aa}}{4:
:20:FGO23060100001
:23B:CRED
:32A:230601GBP5669,49
:50K:/7482144095
ELENA ADEYEMI
66 KINGSWAY
LONDON GB
:52A:NRGTGB2L1BR
:57A:VRDLGB3A5DP
:59:/15749055
CLARA WHITFIELD
171 KINGSWAY
LONDON GB
:70:/RFB/LOAN REPAYMENT
:71A:SHA
:72:/INS/CHEQUE 04:22:05
-}"""


def message_lines(text: str = MESSAGE) -> list[str]:
    return text.splitlines()


def without(tag: str) -> list[str]:
    """The fixture with one tag and its continuation lines removed."""
    lines, dropping = [], False
    for line in message_lines():
        if line.startswith(f":{tag}:"):
            dropping = True
            continue
        if line.startswith(":") or line == "-}":
            dropping = False
        if not dropping:
            lines.append(line)
    return lines


# --- the comma is the decimal separator ------------------------------------------------


def test_comma_is_the_decimal_point():
    assert parse_amount("5669,49") == Decimal("5669.49")


def test_the_naive_reading_would_be_a_hundredfold_error():
    """Guard the specific bug this parser exists to prevent, not just the happy path."""
    raw = "5810,46"
    assert parse_amount(raw) == Decimal("5810.46")
    assert float(raw.replace(",", "")) == 581046.0, "the trap is still a trap"


def test_a_dot_is_refused_rather_than_interpreted():
    """1.234 could be one-point-two-three-four or one thousand; we cannot tell, so we stop."""
    with pytest.raises(ValueError, match=r"does not permit"):
        parse_amount("5,669.49")


@pytest.mark.parametrize("raw", ["566949", "5.669,49", "1,2,3", "abc,de", "0,00", "-5,00"])
def test_amounts_that_are_not_swift_are_refused(raw):
    with pytest.raises(ValueError):
        parse_amount(raw)


def test_amount_keeps_full_precision():
    """Decimal, not float: this number is filed with a regulator."""
    assert str(parse_amount("12345678,91")) == "12345678.91"


# --- a field is not a line -------------------------------------------------------------


def test_ordering_customer_survives_its_continuation_lines():
    wire = parse_message(message_lines(), 1)
    assert wire.sender_account == "7482144095"
    assert wire.sender_name == "ELENA ADEYEMI"
    assert wire.sender_address == "66 KINGSWAY, LONDON GB"


def test_beneficiary_survives_its_continuation_lines():
    wire = parse_message(message_lines(), 1)
    assert wire.receiver_account == "15749055"
    assert wire.receiver_name == "CLARA WHITFIELD"
    assert wire.receiver_address == "171 KINGSWAY, LONDON GB"


def test_a_value_containing_colons_is_not_read_as_a_new_tag():
    """':72:/INS/CHEQUE 04:22:05' -- the time's colons must not open fields 22 and 05."""
    fields = read_fields(message_lines())
    assert fields["72"] == ["/INS/CHEQUE 04:22:05"]
    assert "22" not in fields and "05" not in fields


def test_every_line_of_a_field_is_kept():
    assert read_fields(message_lines())["50K"] == [
        "/7482144095",
        "ELENA ADEYEMI",
        "66 KINGSWAY",
        "LONDON GB",
    ]


# --- whole message ---------------------------------------------------------------------


def test_message_parses_to_the_expected_wire():
    wire = parse_message(message_lines(), 1)
    assert wire.reference == "FGO23060100001"
    assert wire.value_date == date(2023, 6, 1)
    assert wire.currency == "GBP"
    assert wire.amount == Decimal("5669.49")
    assert wire.bank_operation_code == "CRED"
    assert wire.charge_code == "SHA"
    assert wire.memo == "/RFB/LOAN REPAYMENT"
    assert wire.uetr == "5e44476b-41c4-4c82-8efc-ff2b56db21aa"


def test_country_comes_from_the_bic_since_no_field_carries_it():
    assert country_of("NRGTGB2L1BR") == "GB"
    assert country_of("VRDLAE3A5DP") == "AE"
    with pytest.raises(ValueError, match="not a BIC"):
        country_of("GB12")


def test_corridor_and_cross_border_flag():
    wire = parse_message(message_lines(), 1)
    assert wire.corridor == "GB->GB" and not wire.is_cross_border

    crossed = parse_message(message_lines(MESSAGE.replace(":57A:VRDLGB", ":57A:VRDLAE")), 1)
    assert crossed.corridor == "GB->AE" and crossed.is_cross_border


def test_as_row_is_json_safe_and_pandas_friendly():
    row = parse_message(message_lines(), 1).as_row()
    assert row["amount"] == 5669.49 and isinstance(row["amount"], float)
    assert row["value_date"] == "2023-06-01"
    assert row["corridor"] == "GB->GB"


# --- refusing --------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ["20", "23B", "32A", "50K", "52A", "57A", "59"])
def test_a_missing_required_tag_is_refused(tag):
    with pytest.raises(MalformedMessage, match="missing required tag"):
        parse_message(without(tag), 1)


def test_the_refusal_names_the_reference_so_the_fallback_knows_what_to_retry():
    error = pytest.raises(MalformedMessage, parse_message, without("32A"), 1).value
    assert error.reference == "FGO23060100001"
    assert ":32A:" in str(error)
    assert error.raw.startswith("{1:F01"), "the fallback needs the original text"


def test_a_message_with_no_reference_is_identified_by_ordinal():
    error = pytest.raises(MalformedMessage, parse_message, without("20"), 7).value
    assert error.reference is None and "message #7" in str(error)


def test_an_unterminated_message_is_refused():
    with pytest.raises(MalformedMessage, match="not terminated"):
        parse_message(message_lines()[:-1], 1)


def test_a_malformed_value_date_line_is_refused():
    broken = MESSAGE.replace(":32A:230601GBP5669,49", ":32A:2023-06-01 GBP 5669.49")
    with pytest.raises(MalformedMessage, match="YYMMDD"):
        parse_message(message_lines(broken), 1)


def test_a_party_that_is_not_an_account_number_is_refused():
    with pytest.raises(ValueError, match="not an account number"):
        parse_party(["/NOT-AN-ACCOUNT", "ELENA ADEYEMI"])


def test_split_ignores_statement_headers_and_footers():
    log = f"NORTHGATE PRIVATE BANK\nMessages in batch   : 1\n\n{MESSAGE}\n\nEND OF STATEMENT - 1 MESSAGES\n"
    blocks = split_messages(log)
    assert len(blocks) == 1 and blocks[0][0].startswith("{1:F01")


# --- the CLI boundary ------------------------------------------------------------------


def test_a_missing_batch_is_rejected_before_any_work_starts():
    """Pasting the usage placeholder is the common mistake; it must not reach a graph node."""
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="no such batch log: BATCH"):
        existing_log("BATCH")


def test_a_real_batch_passes_through_as_a_path(tmp_path):
    log = tmp_path / "batch.txt"
    log.write_text(MESSAGE)
    assert existing_log(str(log)) == log


def test_a_directory_is_not_a_batch(tmp_path):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        existing_log(str(tmp_path))


# --- the real batches ------------------------------------------------------------------

pytestmark_reason = "batch logs not generated -- run: uv run python -m src.utils.pdf_generator"
needs_ledger = pytest.mark.skipif(not LABELS.exists(), reason=pytestmark_reason)


@needs_ledger
@pytest.mark.parametrize("suffix", [".txt", ".pdf"])
def test_every_declared_message_is_recovered(suffix):
    for log in sorted(LEDGER.glob(f"*{suffix}")):
        batch = parse_batch(log, strict=True)
        assert batch.complete, f"{log.name}: {batch.parsed} of {batch.declared_messages}"
        assert not batch.failures


@needs_ledger
def test_pdf_and_text_render_of_the_same_batch_parse_identically():
    """The auditor uploads a PDF; the parser must not see a different ledger than the source."""
    for text_log in sorted(LEDGER.glob("*.txt")):
        from_text = parse_batch(text_log, strict=True)
        from_pdf = parse_batch(text_log.with_suffix(".pdf"), strict=True)
        assert from_text.wires == from_pdf.wires, text_log.name


@needs_ledger
def test_every_field_matches_the_ground_truth_ledger():
    """The documents were rendered from `ledger_labels.csv`; parsing must invert that exactly."""
    import pandas as pd

    from src.utils.pdf_generator import COUNTRY_ISO, CURRENCY_ISO

    labels = pd.read_csv(LABELS, dtype={"Sender_account": str, "Receiver_account": str})
    checked = 0
    for log, group in labels.groupby("Log_file"):
        wires = {w.reference: w for w in parse_batch(LEDGER / log, strict=True).wires}
        assert set(wires) == set(group.Reference), f"{log}: reference set differs"
        for row in group.itertuples():
            wire = wires[row.Reference]
            assert wire.amount == Decimal(f"{row.Amount:.2f}"), row.Reference
            assert wire.currency == CURRENCY_ISO[row.Payment_currency], row.Reference
            assert wire.value_date.isoformat() == row.Date, row.Reference
            assert wire.sender_account == row.Sender_account, row.Reference
            assert wire.receiver_account == row.Receiver_account, row.Reference
            assert wire.sender_country == COUNTRY_ISO[row.Sender_bank_location], row.Reference
            assert wire.receiver_country == COUNTRY_ISO[row.Receiver_bank_location], row.Reference
            checked += 1
    # Every labelled row, however many batches have been generated -- three planted logs of 220
    # plus any control batch added with --append.
    assert checked == len(labels)
    assert checked >= 660


@needs_ledger
def test_a_corrupted_message_is_collected_not_fatal(tmp_path):
    """§4.2's fallback needs the bad message isolated while the other 219 stay free."""
    source = next(iter(sorted(LEDGER.glob("*.txt"))))
    text = source.read_text()
    corrupted = tmp_path / source.name
    corrupted.write_text(text.replace(":32A:", ":32A:XX", 1))

    batch = parse_batch(corrupted)
    assert len(batch.failures) == 1
    assert batch.parsed == batch.declared_messages - 1
    assert batch.failures[0].reference

    with pytest.raises(MalformedMessage):
        parse_batch(corrupted, strict=True)
