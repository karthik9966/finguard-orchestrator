"""The detection package (Phase 3, LLD §2.4).

Unit tests build records directly; the recall harness runs the real ledgers. This replaced
`utils/detectors.py` and its 26 tests, which Phase 5 deleted along with the module.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from src.config import PATTERN_TYPES, get_config
from src.detection import detect_all
from src.detection.confidence import coefficient_of_variation, score
from src.detection.graph_engine import build_graph
from src.detection.reconciler import CandidateReconciler
from src.ingestion.batch import TransactionBatchIngestor
from src.models import Candidate, TransactionRecord

LEDGER = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"
LABELS = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger_labels.csv"
needs_ledger = pytest.mark.skipif(
    not LABELS.exists(), reason="run: uv run finguard-ledger --profile dev"
)

T0 = datetime(2023, 6, 1, 9, 0, tzinfo=timezone.utc)


def record(ref, sender, receiver, amount, day=0, hour=0, instrument="WIRE") -> TransactionRecord:
    return TransactionRecord(
        txn_ref=ref, sender_account=sender, receiver_account=receiver,
        amount=Decimal(str(amount)), currency="USD",
        timestamp=T0 + timedelta(days=day, hours=hour),
        sender_country="US", receiver_country="US", instrument=instrument,
    )


def only(pattern, records):
    return [c for c in detect_all(records) if c.pattern_type == pattern]


# --- the shared graph --------------------------------------------------------------------


def test_repeated_payments_between_two_accounts_are_separate_edges():
    """A MultiDiGraph, not a DiGraph: collapsing parallel edges would hide exactly the
    repetition that structuring is."""
    batch = build_graph([record(f"R{i}", "A", "B", 9000, day=i) for i in range(3)])
    assert batch.graph.number_of_nodes() == 2
    assert batch.graph.number_of_edges() == 3


# --- structuring --------------------------------------------------------------------------


def test_one_originator_below_the_ctr_threshold_is_structuring():
    records = [record(f"S{i}", "LAUNDERER", "MULE", 9200, day=i) for i in range(4)]
    found = [c for c in detect_all(records) if c.pattern_type == "structuring"]

    assert found, "four sub-threshold transfers from one account is the shape §5324 prohibits"
    assert found[0].attributes["threshold"] == 10000
    assert found[0].attributes["total"] >= 10000, "the group must reach the threshold it evades"


def test_a_single_transfer_over_the_threshold_is_not_structuring():
    """Paying $40,000 in one go is the opposite of structuring: it files a CTR."""
    records = [record("BIG", "A", "B", 40000)] + [record(f"N{i}", f"X{i}", "B", 500, day=i) for i in range(3)]
    assert not [c for c in detect_all(records) if c.pattern_type == "structuring"]


def test_transfers_outside_the_window_are_not_one_pattern():
    window = get_config().detection.window_days
    records = [record(f"S{i}", "A", "B", 9200, day=i * (window + 5)) for i in range(4)]
    assert not [c for c in detect_all(records) if c.pattern_type == "structuring"]


# --- fan in / fan out ---------------------------------------------------------------------


def test_many_senders_into_one_account_is_fan_in():
    minimum = get_config().detection.fan_in.min_sources
    records = [record(f"F{i}", f"SRC{i}", "COLLECTOR", 4000, day=i) for i in range(minimum)]
    found = [c for c in detect_all(records) if c.pattern_type == "fan_in"]
    assert found and found[0].attributes["distinct_senders"] == minimum


def test_the_same_sender_repeating_is_not_a_fan():
    """Fan-in is about *distinct* counterparties. One account paying five times is a different
    shape, and calling it fan-in would report every regular payer."""
    minimum = get_config().detection.fan_in.min_sources
    records = [record(f"F{i}", "ONE", "COLLECTOR", 4000, day=i) for i in range(minimum + 2)]
    assert not [c for c in detect_all(records) if c.pattern_type == "fan_in"]


@pytest.mark.parametrize("pattern", ["fan_in", "fan_out"])
def test_a_fan_spread_beyond_the_window_does_not_fire(pattern):
    """The behaviour change from the old primitives, asserted. They applied no window at all, so
    fifteen counterparties across a month scored like fifteen in an afternoon."""
    minimum = get_config().detection.fan_in.min_sources
    window = get_config().detection.window_days
    if pattern == "fan_in":
        records = [record(f"F{i}", f"SRC{i}", "HUB", 4000, day=i * (window + 3)) for i in range(minimum)]
    else:
        records = [record(f"F{i}", "HUB", f"DST{i}", 4000, day=i * (window + 3)) for i in range(minimum)]
    assert not [c for c in detect_all(records) if c.pattern_type == pattern]


# --- cycle ---------------------------------------------------------------------------------


def test_money_returning_through_intermediaries_is_a_cycle():
    records = [
        record("C1", "A", "B", 10000, day=0),
        record("C2", "B", "C", 9000, day=1),
        record("C3", "C", "A", 8200, day=2),
    ]
    found = [c for c in detect_all(records) if c.pattern_type == "cycle"]
    assert found and found[0].attributes["hops"] >= 3
    assert found[0].attributes["retained_fraction"] > 0.5


def test_a_chain_that_loses_most_of_its_value_is_not_a_ring():
    """Funds must come back roughly intact. A chain that decays to a tenth is a sequence of
    ordinary payments that happen to touch, not a ring."""
    records = [
        record("C1", "A", "B", 10000, day=0),
        record("C2", "B", "C", 5000, day=1),
        record("C3", "C", "A", 300, day=2),
    ]
    assert not [c for c in detect_all(records) if c.pattern_type == "cycle"]


# --- scatter-gather --------------------------------------------------------------------------


def test_one_to_many_to_one_is_scatter_gather():
    minimum = get_config().detection.scatter_gather.min_fan
    records = []
    for i in range(minimum):
        records.append(record(f"OUT{i}", "SOURCE", f"MULE{i}", 5000, day=0, hour=i))
        records.append(record(f"IN{i}", f"MULE{i}", "SINK", 4800, day=1, hour=i))
    found = [c for c in detect_all(records) if c.pattern_type == "scatter_gather"]
    assert found and found[0].attributes["fan"] >= minimum
    assert found[0].attributes["sink"] == "SINK"


# --- the reconciler ---------------------------------------------------------------------------


def test_scatter_gather_outranks_the_fan_out_inside_it():
    """Load-bearing, not hygiene: fan_out fires on the first leg of every scatter_gather, so
    without precedence the same transactions are reported twice under different names."""
    refs = [f"T{i}" for i in range(6)]
    pair = [
        Candidate(candidate_id="fan_out:S:x", pattern_type="fan_out",
                  member_txn_refs=refs, detection_confidence=0.9),
        Candidate(candidate_id="scatter_gather:S:y", pattern_type="scatter_gather",
                  member_txn_refs=refs, detection_confidence=0.4),
    ]
    kept = CandidateReconciler().reconcile(pair)
    assert [c.pattern_type for c in kept] == ["scatter_gather"], (
        "precedence must beat confidence -- half a pattern looks tighter than the whole of it"
    )


def test_patterns_that_merely_touch_are_both_kept():
    """A busy account can be the sink of one pattern and the source of another, and both are
    real. Only a substantial overlap means they describe the same event."""
    pair = [
        Candidate(candidate_id="a", pattern_type="fan_in",
                  member_txn_refs=["T1", "T2", "T3", "T4"], detection_confidence=0.5),
        Candidate(candidate_id="b", pattern_type="fan_out",
                  member_txn_refs=["T4", "T5", "T6", "T7"], detection_confidence=0.5),
    ]
    assert len(CandidateReconciler().reconcile(pair)) == 2


def test_precedence_covers_every_pattern_type():
    order = get_config().detection.precedence_order
    assert set(order) == set(PATTERN_TYPES), "a pattern missing here is reconciled last by accident"


def test_the_reconciler_records_what_it_absorbed():
    refs = [f"T{i}" for i in range(6)]
    pair = [
        Candidate(candidate_id="fan_out:S:x", pattern_type="fan_out",
                  member_txn_refs=refs, detection_confidence=0.9),
        Candidate(candidate_id="cycle:S:y", pattern_type="cycle",
                  member_txn_refs=refs, detection_confidence=0.4),
    ]
    result = CandidateReconciler().explain(pair)
    assert result.absorbed == {"cycle:S:y": ["fan_out:S:x"]}


# --- confidence -------------------------------------------------------------------------------


def test_tightness_is_measured_on_the_band_not_the_whole_group():
    """The recorded weakness of the pre-migration scoring: one legitimate $34,121 wire moved a
    cluster's coefficient of variation from 0.024 to 1.039 -- 43x -- so a score keyed on the
    whole group went blind to the tight subset inside it."""
    band = [Decimal("9500"), Decimal("9400"), Decimal("9600")]
    contaminated = band + [Decimal("34121")]
    assert coefficient_of_variation(contaminated) > 10 * coefficient_of_variation(band)

    whole = score(amounts=contaminated, timestamps=[T0] * 4, minimum_members=3)
    banded = score(amounts=contaminated, timestamps=[T0] * 4, minimum_members=3, band_amounts=band)
    assert banded > whole


def test_a_burst_scores_above_the_same_count_spread_across_the_window():
    amounts = [Decimal("9000")] * 4
    window = get_config().detection.window_days
    burst = score(amounts=amounts, timestamps=[T0] * 4, minimum_members=3)
    spread = score(
        amounts=amounts,
        timestamps=[T0 + timedelta(days=i * window / 3) for i in range(4)],
        minimum_members=3,
    )
    assert burst > spread


def test_confidence_stays_in_range():
    assert 0.0 <= score(amounts=[], timestamps=[], minimum_members=3) <= 1.0
    huge = [Decimal("1000000")] * 50
    assert 0.0 <= score(amounts=huge, timestamps=[T0] * 50, minimum_members=3) <= 1.0


# --- against the real ledgers -----------------------------------------------------------------


@needs_ledger
def test_detector_level_recall_clears_the_bar():
    """Phase 3's green criterion: >= 0.90 on the regenerated in-scope labels.

    An upper bound on the PRD KPI, not the KPI -- the KPI is whether the system *reports* the
    finding, which depends on everything downstream of here.
    """
    labels = pd.read_csv(LABELS)
    ingestor = TransactionBatchIngestor(fallback=lambda failure: None)
    found = planted = 0
    for log, group in labels.groupby("Log_file"):
        if len(group) > 5_000:
            continue  # the 10k batch is timed separately; recall is measured on the dev set
        records, _ = ingestor.ingest([LEDGER / log.replace(".pdf", ".txt")])
        swept = {ref for c in detect_all(records) for ref in c.member_txn_refs}
        refs = set(group[group.Is_laundering == 1].Reference)
        found += len(refs & swept)
        planted += len(refs)

    assert planted, "the answer key plants nothing -- the test proves nothing"
    assert found / planted >= 0.90, f"detector recall {found / planted:.0%} ({found}/{planted})"


@needs_ledger
def test_the_clean_control_produces_nothing():
    """The router's "no candidates -> no model, $0.00" path needs a batch with nothing in it."""
    records, _ = TransactionBatchIngestor(fallback=lambda f: None).ingest(
        [LEDGER / "2023-05_private_banking_log.txt"]
    )
    assert records and detect_all(records) == []


@needs_ledger
def test_the_large_batch_detects_in_seconds():
    """`window_days` and `cycle.max_length` are what bound the DFS. Without them the walk on
    10,000 messages does not finish."""
    import time

    large = LEDGER / "2023-04_private_banking_log.txt"
    if not large.exists():
        pytest.skip("no large batch -- run: uv run finguard-ledger --profile large --append")

    records, _ = TransactionBatchIngestor(fallback=lambda f: None).ingest([large])
    started = time.perf_counter()
    candidates = detect_all(records)
    elapsed = time.perf_counter() - started

    assert len(records) > 5_000
    assert elapsed < 30, f"detection took {elapsed:.1f}s on {len(records):,} records"
    assert candidates, "a 10,000-message batch with planted patterns found nothing"


@needs_ledger
def test_candidate_ids_are_stable_across_runs():
    """A report has to be diffable against its predecessor, so the same input must produce the
    same ids."""
    records, _ = TransactionBatchIngestor(fallback=lambda f: None).ingest(
        [LEDGER / "2023-06_private_banking_log.txt"]
    )
    first = {c.candidate_id for c in detect_all(records)}
    assert first == {c.candidate_id for c in detect_all(records)}


# --- v2: gather-scatter ------------------------------------------------------------------------


def gather_scatter_hub(out_amount=4000, spread_days=1):
    rules = get_config().detection.gather_scatter
    records = [record(f"GI{i}", f"SRC{i}", "HUB", 4000, day=i * spread_days) for i in range(rules.min_in)]
    records += [
        record(f"GO{i}", "HUB", f"DST{i}", out_amount, day=rules.min_in * spread_days + i)
        for i in range(rules.min_out)
    ]
    return records


def test_a_hub_that_fills_then_empties_is_gather_scatter_not_two_fans():
    (found,) = only("gather_scatter", gather_scatter_hub())
    assert found.attributes["hub"] == "HUB" and found.attributes["conservation"] == 1.0
    # The reconciler must not also report the halves.
    assert not only("fan_in", gather_scatter_hub()) and not only("fan_out", gather_scatter_hub())


def test_a_hub_that_keeps_the_money_is_not_a_pass_through():
    assert not only("gather_scatter", gather_scatter_hub(out_amount=500))


def test_gather_scatter_spread_beyond_its_window_does_not_fire():
    window = get_config().detection.gather_scatter.window_days
    assert not only("gather_scatter", gather_scatter_hub(spread_days=window))


# --- v2: deposit-send --------------------------------------------------------------------------


def test_cash_in_then_the_same_amount_wired_out_is_deposit_send():
    records = [
        record("D1", "CUST", "MULE", 9500, instrument="CASH DEPOSIT"),
        record("S1", "MULE", "OFFSHORE", 9540, hour=30, instrument="CROSS-BORDER"),
    ]
    (found,) = only("deposit_send", records)
    assert found.member_txn_refs == ["D1", "S1"]
    assert found.attributes["cross_border"] is True


@pytest.mark.parametrize(
    ("instrument", "amount", "hours"),
    [
        ("CASH WITHDRAWAL", 9540, 30),   # a withdrawal is not a deposit -- the P2 parser fix
        ("CASH DEPOSIT", 4000, 30),      # timing alone: the amounts do not match
        ("CASH DEPOSIT", 9540, 24 * 10), # the match, but far outside the window
    ],
)
def test_deposit_send_needs_a_deposit_a_matching_amount_and_the_window(instrument, amount, hours):
    records = [
        record("D1", "CUST", "MULE", 9500, instrument=instrument),
        record("S1", "MULE", "OFFSHORE", amount, hour=hours, instrument="CROSS-BORDER"),
    ]
    assert not only("deposit_send", records)


def test_one_send_cannot_be_claimed_by_two_deposits():
    records = [
        record("D1", "C1", "MULE", 9500, instrument="CASH DEPOSIT"),
        record("D2", "C2", "MULE", 9500, hour=1, instrument="CASH DEPOSIT"),
        record("S1", "MULE", "X", 9500, hour=5, instrument="ACH"),
    ]
    (found,) = only("deposit_send", records)
    assert found.attributes["pairs"] == 1


# --- v2: layered fan ---------------------------------------------------------------------------


def layered(direction="in"):
    records, n = [], 0
    for c in range(3):
        for leaf in range(3):
            n += 1
            leg = (f"L{c}{leaf}", f"C{c}") if direction == "in" else (f"C{c}", f"L{c}{leaf}")
            records.append(record(f"LF{n}", *leg, 3000, day=leaf))
        top = (f"C{c}", "ROOT") if direction == "in" else ("ROOT", f"C{c}")
        records.append(record(f"LT{c}", *top, 9000, day=5))
    return records


@pytest.mark.parametrize("direction", ["in", "out"])
def test_funnels_feeding_one_root_are_a_layered_fan(direction):
    (found,) = only("layered_fan", layered(direction))
    assert found.attributes["direction"] == direction
    assert found.attributes["collectors"] == ["C0", "C1", "C2"]
    assert len(found.member_txn_refs) == 12
    # The collectors' own fans are absorbed rather than reported beside it.
    assert not only("fan_in", layered(direction)) and not only("fan_out", layered(direction))


def test_a_single_level_fan_is_not_layered():
    records = [record(f"F{i}", f"S{i}", "ROOT", 3000, day=i) for i in range(8)]
    assert not only("layered_fan", records)


# --- v2: bipartite -----------------------------------------------------------------------------


def block(prefix, senders, receivers, day=0):
    return [
        record(f"{prefix}{s}{r}", s, r, 5000, day=day)
        for s in senders for r in receivers
    ]


def test_senders_sharing_their_receivers_are_bipartite():
    (found,) = only("bipartite", block("B", ["S1", "S2"], [f"R{i}" for i in range(6)]))
    assert found.attributes["stacked"] is False and found.attributes["density"] == 1.0


def test_a_block_handed_on_to_a_second_block_is_one_stacked_candidate():
    first = block("A", ["S1", "S2"], ["M1", "M2", "M3", "M4"])
    second = block("Z", ["M1", "M2"], ["E1", "E2", "E3", "E4"], day=3)
    (found,) = only("bipartite", first + second)
    assert found.attributes["stacked"] is True and found.attributes["layers"] == 2


# --- v2: evidence ------------------------------------------------------------------------------


def test_every_surviving_candidate_carries_its_subgraph():
    for candidate in detect_all(layered("in") + gather_scatter_hub()):
        edges = candidate.subgraph["edges"]
        assert [e["ref"] for e in edges] == candidate.member_txn_refs
