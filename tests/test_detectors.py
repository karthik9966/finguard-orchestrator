"""Verify the four shape primitives (§4.2).

Recall is the contract: a missed cluster is a regulatory failure, an extra one costs tokens.
The ground-truth tests assert 100% recall against `ledger_labels.csv` and deliberately assert
nothing about precision beyond a loose ceiling -- tightening it is a later pass, and a test that
pins today's precision would just have to be edited when that work happens.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.utils.detectors import (
    CONCENTRATION,
    DISPERSION,
    MAGNITUDE,
    PATH,
    Candidate,
    coefficient_of_variation,
    covered_references,
    detect,
    find_clusters,
    find_magnitude,
    find_paths,
)
from src.utils.swift_parser import Wire, parse_batch

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
LABELS = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger_labels.csv"
needs_ledger = pytest.mark.skipif(
    not LABELS.exists(), reason="run: uv run python -m src.utils.pdf_generator"
)

# The nine typologies pdf_generator.py plants, and the primitive each must fall to.
EXPECTED_SHAPES = {
    "Structuring": CONCENTRATION,
    "Gather-Scatter": CONCENTRATION,
    "Layered_Fan_In": CONCENTRATION,
    "Smurfing": CONCENTRATION,
    "Deposit-Send": DISPERSION,
    "Scatter-Gather": DISPERSION,
    "Layered_Fan_Out": DISPERSION,
    "Cycle": PATH,
    "Over-Invoicing": MAGNITUDE,
}


def wire(ref: str, sender: str, receiver: str, amount: str, day: int = 1, currency: str = "GBP",
         sender_country: str = "GB", receiver_country: str = "GB") -> Wire:
    return Wire(
        reference=ref,
        value_date=date(2023, 6, 1) + timedelta(days=day - 1),
        currency=currency,
        amount=Decimal(amount),
        sender_account=sender,
        sender_name="ORDERING PARTY",
        sender_address="1 STREET, LONDON GB",
        sender_bic=f"ADVN{sender_country}2L1BR",
        sender_country=sender_country,
        receiver_account=receiver,
        receiver_name="BENEFICIARY",
        receiver_address="2 STREET, LONDON GB",
        receiver_bic=f"BRGT{receiver_country}3A5DP",
        receiver_country=receiver_country,
        bank_operation_code="CRED",
    )


def chain(hops: int, *, start_day: int = 1, gap: int = 1, decay: float = 0.9) -> list[Wire]:
    """A -> B -> C -> ..., losing `decay` of the amount at each hop."""
    wires, amount = [], 10_000.0
    for i in range(hops):
        wires.append(wire(f"C{i}", f"ACC{i}", f"ACC{i + 1}", f"{amount:.2f}", start_day + i * gap))
        amount *= decay
    return wires


# --- coefficient of variation ----------------------------------------------------------


def test_cv_of_near_identical_amounts_is_near_zero():
    """Ten wires that are effectively the same payment made ten times."""
    tight = [Decimal(a) for a in ("5526", "5600", "5680", "5710", "5750", "5800", "5850", "5984")]
    assert coefficient_of_variation(tight) < 0.05


def test_one_outlier_destroys_the_measure():
    """Why CV over a whole account is unreliable: a tight run plus one ordinary wire.

    This is the known weakness carried into the deferred precision work -- the ten laundering
    wires are just as tight with the outlier present, but the group's CV no longer says so.
    """
    tight = [Decimal("5673")] * 10
    assert coefficient_of_variation(tight) == 0.0
    assert coefficient_of_variation([*tight, Decimal("34121")]) > 0.8


def test_cv_is_defined_for_degenerate_inputs():
    assert coefficient_of_variation([]) == 0.0
    assert coefficient_of_variation([Decimal("100")]) == 0.0


# --- concentration and dispersion -------------------------------------------------------


def test_concentration_finds_the_collecting_account():
    wires = [wire(f"R{i}", f"SENDER{i}", "COLLECTOR", "5000", i + 1) for i in range(4)]
    found = [c for c in find_clusters(wires) if c.shape == CONCENTRATION]
    assert len(found) == 1
    assert found[0].anchor == "COLLECTOR"
    assert found[0].wire_count == 4 and found[0].distinct_counterparties == 4


def test_dispersion_finds_the_distributing_account():
    wires = [wire(f"S{i}", "SPRAYER", f"TARGET{i}", "5000", i + 1) for i in range(4)]
    found = [c for c in find_clusters(wires) if c.shape == DISPERSION]
    assert len(found) == 1 and found[0].anchor == "SPRAYER"


def test_a_repeated_pair_is_still_a_run():
    """Smurfing is one sender paying one receiver eight times.

    Gating on distinct counterparties instead of wire count would miss it entirely -- the
    counterparty count is 1. The measured cluster is a run of wires, not a fan.
    """
    wires = [wire(f"P{i}", "SMURF", "COLLECTOR", "2000", i + 1) for i in range(8)]
    shapes = {c.shape: c for c in find_clusters(wires)}
    assert set(shapes) == {CONCENTRATION, DISPERSION}
    assert shapes[CONCENTRATION].distinct_counterparties == 1
    assert shapes[CONCENTRATION].wire_count == 8


def test_two_wires_are_not_a_run():
    wires = [wire("A", "X", "COLLECTOR", "5000"), wire("B", "Y", "COLLECTOR", "5000", 2)]
    assert find_clusters(wires) == []


def test_an_account_can_be_both_collector_and_distributor():
    wires = [wire(f"I{i}", f"IN{i}", "HUB", "5000", i + 1) for i in range(3)]
    wires += [wire(f"O{i}", "HUB", f"OUT{i}", "4800", i + 4) for i in range(3)]
    anchors = {(c.shape, c.anchor) for c in find_clusters(wires)}
    assert (CONCENTRATION, "HUB") in anchors and (DISPERSION, "HUB") in anchors


def test_cluster_records_the_span_and_the_amount_band():
    wires = [wire(f"R{i}", f"S{i}", "COLLECTOR", a, d)
             for i, (a, d) in enumerate([("5526", 5), ("5984", 9), ("5700", 14)])]
    found = next(c for c in find_clusters(wires) if c.shape == CONCENTRATION)
    assert found.first_date == date(2023, 6, 5) and found.last_date == date(2023, 6, 14)
    assert found.span_days == 10
    assert found.min_amount == Decimal("5526") and found.max_amount == Decimal("5984")
    assert found.total_amount == Decimal("17210")


# --- path -------------------------------------------------------------------------------


def test_a_chain_is_found_where_counting_finds_nothing():
    """Every account appears once as sender and once as receiver, so no cluster exists."""
    wires = chain(5)
    assert find_clusters(wires) == []
    paths = find_paths(wires)
    assert len(paths) == 1
    assert paths[0].wire_count == 5 and paths[0].anchor == "ACC0"


def test_path_records_how_much_survives_the_route():
    """A ring that skims a slice per hop is the point; 0.9^4 of the opening amount is left."""
    found = find_paths(chain(5, decay=0.9))[0]
    assert found.retained_fraction == pytest.approx(0.9**4, rel=1e-3)


def test_two_hops_are_not_a_path():
    assert find_paths(chain(2)) == []


def test_a_chain_broken_by_a_long_gap_is_not_followed():
    assert find_paths(chain(5, gap=30)) == []


def test_branches_off_one_ring_are_not_reported_as_separate_findings():
    """A ring with side-payments hanging off it must not surface a dozen times."""
    wires = chain(8)
    for i in range(4):  # branches leaving the ring at hop 4
        wires.append(wire(f"B{i}", "ACC4", f"OFF{i}", "3000", 5 + i))
    found = find_paths(wires)
    assert len(found) <= 2, [c.summary() for c in found]
    assert found[0].wire_count == 8


def test_a_walk_cannot_revisit_an_account():
    """Without this the walker loops forever around a two-account ping-pong."""
    wires = [
        wire("P1", "A", "B", "5000", 1),
        wire("P2", "B", "A", "4500", 2),
        wire("P3", "A", "B", "4000", 3),
        wire("P4", "B", "A", "3500", 4),
    ]
    for found in find_paths(wires):
        accounts = [found.references]
        assert found.wire_count <= len(wires), accounts


def test_a_path_must_move_forward_in_time():
    backwards = [
        wire("P1", "A", "B", "5000", 10),
        wire("P2", "B", "C", "4500", 5),
        wire("P3", "C", "D", "4000", 1),
    ]
    assert find_paths(backwards) == []


# --- magnitude ---------------------------------------------------------------------------


def test_a_single_implausible_wire_is_its_own_finding():
    wires = [wire(f"N{i}", f"S{i}", f"R{i}", "5000", i + 1) for i in range(20)]
    wires.append(wire("BIG", "EXPORTER", "IMPORTER", "9216360.00", 21))
    found = find_magnitude(wires)
    assert [c.references for c in found] == [("BIG",)]
    assert found[0].wire_count == 1 and found[0].anchor == "EXPORTER"


def test_an_outlier_is_judged_against_its_own_currency():
    """A large figure in a currency the batch barely uses is not evidence of anything."""
    wires = [wire(f"G{i}", f"S{i}", f"R{i}", "5000", i + 1) for i in range(20)]
    wires += [wire(f"J{i}", f"S{i}", f"R{i}", "600000", i + 1, currency="JPY") for i in range(6)]
    assert find_magnitude(wires) == []


def test_ordinary_variation_is_not_magnitude():
    wires = [wire(f"N{i}", f"S{i}", f"R{i}", str(4000 + i * 400), i + 1) for i in range(20)]
    assert find_magnitude(wires) == []


# --- what a candidate is allowed to say ---------------------------------------------------


def test_a_candidate_never_names_a_typology():
    """The legal conclusion has to come from a retrieved rule, not from Python.

    If this module emitted "Structuring", the retrieval query would be built from a label the
    code invented, and the SAR would cite whichever clause best matched our own guess.
    """
    forbidden = {t.lower() for t in EXPECTED_SHAPES} | {"laundering", "suspicious", "smurf"}
    wires = [wire(f"R{i}", f"S{i}", "COLLECTOR", "5000", i + 1) for i in range(4)]
    wires += chain(4, start_day=10)
    for candidate in detect(wires):
        spoken = candidate.summary().lower()
        assert not [word for word in forbidden if word in spoken], candidate.summary()
        assert candidate.shape in {CONCENTRATION, DISPERSION, PATH, MAGNITUDE}


def test_as_row_is_json_safe():
    import json

    wires = [wire(f"R{i}", f"S{i}", "COLLECTOR", "5000", i + 1) for i in range(4)]
    row = detect(wires)[0].as_row()
    assert json.loads(json.dumps(row))["shape"] == CONCENTRATION


def test_cross_border_is_reported_so_retrieval_can_widen_its_tiers():
    domestic = [wire(f"D{i}", f"S{i}", "COLLECTOR", "5000", i + 1) for i in range(3)]
    assert not detect(domestic)[0].is_cross_border

    crossed = domestic + [wire("X", "SX", "COLLECTOR", "5000", 4, receiver_country="AE")]
    assert detect(crossed)[0].is_cross_border


# --- against the answer key ----------------------------------------------------------------


@needs_ledger
def test_no_planted_laundering_wire_is_missed():
    """The contract. 52 flagged wires across three batches; every one must reach a candidate."""
    import pandas as pd

    labels = pd.read_csv(LABELS)
    total_flagged = total_found = total_swept = total_wires = 0
    for log, group in labels.groupby("Log_file"):
        batch = parse_batch(LEDGER / log, strict=True)
        swept = covered_references(detect(batch.wires))
        flagged = set(group[group.Is_laundering == 1].Reference)

        missed = flagged - swept
        assert not missed, (
            f"{log}: missed {len(missed)} wires "
            f"({sorted(set(group[group.Reference.isin(missed)].Laundering_type))})"
        )
        total_flagged += len(flagged)
        total_found += len(flagged & swept)
        total_swept += len(swept)
        total_wires += batch.parsed

    assert total_found == total_flagged == 52
    # A loose ceiling, not a precision target: if a change starts sweeping half the batch the
    # candidate list has stopped being a shortlist, whatever its recall.
    assert total_swept / total_wires < 0.40, f"swept {total_swept}/{total_wires}"


@needs_ledger
def test_each_typology_is_caught_by_the_primitive_it_should_be():
    """Recall alone could pass by accident -- a rule sweeping everything catches everything."""
    import pandas as pd

    labels = pd.read_csv(LABELS)
    seen: dict[str, set[str]] = {}
    for log, group in labels.groupby("Log_file"):
        candidates = detect(parse_batch(LEDGER / log, strict=True).wires)
        for typology, rows in group[group.Is_laundering == 1].groupby("Laundering_type"):
            references = set(rows.Reference)
            shapes = {c.shape for c in candidates if references & set(c.references)}
            seen.setdefault(typology, set()).update(shapes)

    assert set(seen) == set(EXPECTED_SHAPES), "the planted typologies changed"
    for typology, expected in EXPECTED_SHAPES.items():
        assert expected in seen[typology], f"{typology} was not found as {expected}: {seen[typology]}"


@needs_ledger
def test_the_candidate_list_is_short_enough_to_prompt_with():
    """Everything downstream costs tokens per candidate; a batch must yield a shortlist."""
    for log in sorted(LEDGER.glob("*.pdf")):
        candidates = detect(parse_batch(log, strict=True).wires)
        assert 1 <= len(candidates) <= 20, f"{log.name}: {len(candidates)} candidates"
        assert all(isinstance(c, Candidate) for c in candidates)
