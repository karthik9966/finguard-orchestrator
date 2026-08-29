"""Verify the generated SWIFT logs survive PDF extraction and never leak their labels.

The round-trip test is the one that decides whether §3.4 can work at all: if the MT103
field tags do not come back out of the PDF, the extraction node has nothing to parse.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
from pypdf import PdfReader

from src.utils.pdf_generator import (
    CHAINED,
    LABELS_PATH,
    LEDGER_DIR,
    MAX_FLAGGED_SHARE,
    SINGLE_WIRE,
    bic_for,
    iso_country,
    iso_currency,
    mt103,
    party_for,
    select_chain,
    shape_of,
)

pytestmark = pytest.mark.skipif(
    not LABELS_PATH.exists(),
    reason="ledger not generated -- run: uv run python -m src.utils.pdf_generator",
)

REFERENCE = re.compile(r"^:20:(\S+)$", re.MULTILINE)


@pytest.fixture(scope="module")
def labels() -> pd.DataFrame:
    return pd.read_csv(LABELS_PATH)


@pytest.fixture(scope="module")
def log_text() -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(LEDGER_DIR.glob("*.txt"))}


# --- synthesised identities ----------------------------------------------------------


def test_bic_is_well_formed_and_stable():
    first = bic_for(8724731955, "UK")
    assert first == bic_for(8724731955, "UK"), "identities must be reproducible across runs"
    assert len(first) == 11
    assert first[4:6] == "GB", "positions 5-6 of a BIC are the ISO country code"
    assert bic_for(8724731955, "UAE")[4:6] == "AE"


def test_party_details_are_stable_per_account():
    assert party_for(4601790850, "UK") == party_for(4601790850, "UK")
    name, street, city = party_for(4601790850, "UK")
    assert name and street and city.endswith("GB")


def test_unmapped_currency_or_country_fails_loudly():
    """A silent default would put the wrong ISO code on a compliance document."""
    with pytest.raises(KeyError, match="unmapped SAML-D currency"):
        iso_currency("Dogecoin")
    with pytest.raises(KeyError, match="unmapped SAML-D bank location"):
        iso_country("Atlantis")


def test_select_chain_walks_a_ring_that_anchoring_cannot_see():
    """The case that motivated the shape table: a ring where no account ever repeats."""
    ring = pd.DataFrame(
        {
            "Sender_account": [10, 20, 30, 40],
            "Receiver_account": [20, 30, 40, 10],
            "Amount": [1000.0, 900.0, 810.0, 729.0],
        }
    )
    # Anchoring is blind here -- every account appears exactly once as a sender.
    assert ring.Sender_account.value_counts().max() == 1

    chain = select_chain(ring)
    assert len(chain) == 4, "the whole ring should be recovered"

    walked = ring.loc[chain]
    assert list(walked.Receiver_account)[:-1] == list(walked.Sender_account)[1:]


def test_select_chain_respects_the_hop_limit():
    line = pd.DataFrame({"Sender_account": range(0, 12), "Receiver_account": range(1, 13)})
    assert len(select_chain(line, max_hops=4)) == 4


def test_mt103_carries_the_blocks_the_blueprint_requires():
    row = pd.Series(
        {
            "Time": "10:35:19",
            "Date": "2023-06-01",
            "Sender_account": 8724731955,
            "Receiver_account": 2769355426,
            "Amount": 1459.15,
            "Payment_currency": "UK pounds",
            "Sender_bank_location": "UK",
            "Receiver_bank_location": "UAE",
            "Payment_type": "Cross-border",
        }
    )
    message = "\n".join(mt103(row, "FGO23060100001"))

    assert message.startswith("{1:F01")          # block 1: sender BIC
    assert "{2:I103" in message                   # block 2: receiver BIC, MT103
    assert "{121:" in message                     # block 3: UETR
    assert ":20:FGO23060100001" in message        # transaction reference
    assert ":32A:230601GBP1459,15" in message     # value date, currency, amount
    assert ":50K:/8724731955" in message          # ordering customer


# --- generated corpus ----------------------------------------------------------------


def test_every_reference_in_the_logs_joins_to_exactly_one_label(labels, log_text):
    in_logs = [ref for text in log_text.values() for ref in REFERENCE.findall(text)]
    assert len(in_logs) == len(set(in_logs)), "duplicate :20: references"
    assert sorted(in_logs) == sorted(labels.Reference), "labels and logs disagree"


def test_logs_never_disclose_the_ground_truth(labels, log_text):
    """Leaked labels would let the agent 'detect' laundering by reading the label."""
    # A few typology names are also legitimate SAML-D payment types, which belong on the
    # wire ( :72:/INS/CASH DEPOSIT ). Those are not leaks, so exclude them by name.
    payment_types = {value.lower() for value in labels.Payment_type.unique()}

    for name, text in log_text.items():
        lowered = text.lower()
        assert "laundering" not in lowered, f"{name} leaks the label column"
        assert "is_laundering" not in lowered
        for typology in labels.Laundering_type.unique():
            token = typology.lower().replace("_", " ")
            if token in payment_types:
                continue
            assert token not in lowered, f"{name} leaks {typology}"


def test_pdf_round_trip_preserves_the_swift_fields():
    pdfs = sorted(LEDGER_DIR.glob("*.pdf"))
    assert pdfs, "no PDFs generated"

    for path in pdfs:
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        assert "MONTHLY PRIVATE BANKING INSTITUTIONAL TRANSACTION LOG" in text
        for tag in (":20:", ":32A:", ":50K:", ":59:", "{1:F01", "{2:I103"):
            assert tag in text, f"{path.name} lost {tag} during PDF extraction"

        # The references must survive intact, not just the tags.
        extracted = set(REFERENCE.findall(text))
        expected = set(REFERENCE.findall(path.with_suffix(".txt").read_text()))
        assert extracted == expected, f"{path.name} lost references during extraction"


def test_every_flagged_typology_forms_a_detectable_pattern(labels):
    """A flagged run only teaches the agent something if the pattern is actually present.

    Asserted per typology, not per log: a single strong cluster used to mask typologies
    that had been reduced to one isolated wire.
    """
    for (log_file, typology), group in labels[labels.Is_laundering == 1].groupby(
        ["Log_file", "Laundering_type"]
    ):
        shape = shape_of(typology)
        if shape == SINGLE_WIRE:
            continue  # the amount is the signal; one wire is the whole pattern

        if shape == CHAINED:
            assert len(group) >= 3, f"{log_file}: {typology} is not a walkable chain"
            continue

        run = max(
            group.groupby("Receiver_account").size().max(),
            group.groupby("Sender_account").size().max(),
        )
        assert run >= 3, f"{log_file}: {typology} has no multi-wire run on a shared account"


def test_chained_typologies_actually_chain(labels):
    """Each hop's beneficiary must be the next hop's ordering customer, or it is not a ring."""
    chained = labels[
        (labels.Is_laundering == 1) & (labels.Laundering_type.map(shape_of) == CHAINED)
    ]
    if chained.empty:
        pytest.skip("no chained typology in the current corpus")

    for (log_file, typology), group in chained.groupby(["Log_file", "Laundering_type"]):
        senders = set(group.Sender_account)
        receivers = set(group.Receiver_account)
        # In a path A->B->C->D every account except the two endpoints is both.
        overlap = len(senders & receivers)
        assert overlap >= len(group) - 2, (
            f"{log_file}: {typology} wires do not connect end to end "
            f"({overlap} linking accounts across {len(group)} wires)"
        )


def test_logs_respect_the_message_budget(labels):
    """Context traffic for a long chain must be trimmed, not allowed to overflow the batch."""
    for log_file, group in labels.groupby("Log_file"):
        assert len(group) <= 220, f"{log_file} has {len(group)} messages, over the batch size"


def test_flagged_share_is_visible_but_not_implausible(labels):
    """A control batch (``--cases-per-month 0``) is legitimately 0%; anything else must carry
    a share that is findable without being one an auditor would disbelieve."""
    for log_file, group in labels.groupby("Log_file"):
        share = group.Is_laundering.mean()
        if share == 0:
            continue
        assert 0.01 <= share <= MAX_FLAGGED_SHARE, f"{log_file} flagged share {share:.1%} is unrealistic"


def test_a_control_batch_carries_no_planted_pattern(labels):
    """The router's "no candidates -> no model" path needs a document with nothing in it."""
    control = labels[labels.Log_file.str.startswith("2023-05")]
    if control.empty:
        pytest.skip("no control batch generated")
    assert control.Is_laundering.sum() == 0
    assert len(control) == 220
