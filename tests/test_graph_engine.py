"""The GraphEngine's structural queries (LLD v2 §2.5, §7 "Graph-engine correctness").

Evaluation Design v2 makes this a hard gate at 100%: on graphs whose structure is known by
construction, every query must recover exactly that structure, and stay inside its bounds on a
graph built to blow them up. These are the crafted fixtures; the detectors' own tests build on them.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.config import get_config
from src.detection.graph_engine import build_graph
from src.models import TransactionRecord
from src.observability.tracing import trim
from src.utils.redaction import redact

T0 = datetime(2023, 6, 1, 9, 0, tzinfo=timezone.utc)
_counter = iter(range(1_000_000))


def txn(sender, receiver, amount=5000, day=0, instrument="WIRE") -> TransactionRecord:
    return TransactionRecord(
        txn_ref=f"T{next(_counter):06d}", sender_account=sender, receiver_account=receiver,
        amount=Decimal(str(amount)), currency="USD", timestamp=T0 + timedelta(days=day),
        sender_country="US", receiver_country="US", instrument=instrument,
    )


def chained_funnel() -> list[TransactionRecord]:
    """Ten feeders -> three collectors -> one sink: SAML-D's Layered_Fan_In, two hops deep."""
    records = []
    for c, collector in enumerate(("C1", "C2", "C3")):
        for f in range(c * 4, c * 4 + 4 if c < 2 else 10):
            records.append(txn(f"F{f}", collector, day=f % 5))
        records.append(txn(collector, "SINK", day=6))
    return records


def complete_block(senders, receivers, day=0) -> list[TransactionRecord]:
    return [txn(s, r, day=day) for s in senders for r in receivers]


# --- construction -------------------------------------------------------------------------


def test_edges_carry_the_payment_kind():
    batch = build_graph([txn("A", "B", instrument="CASH DEPOSIT")])
    (_, _, data), = batch.graph.edges(data=True)
    assert data["payment_kind"] == "cash_deposit"
    assert list(batch.frame.payment_kind) == ["cash_deposit"]


def test_counterparties_are_distinct_and_exclude_self():
    batch = build_graph([txn("A", "B"), txn("A", "B"), txn("A", "C"), txn("A", "A")])
    assert batch.counterparties("A", "out") == {"B", "C"}
    assert batch.counterparties("B", "in") == {"A"}
    assert batch.counterparties("nobody", "in") == set()


def test_hub_nodes_need_both_sides():
    records = [txn(f"S{i}", "HUB") for i in range(4)] + [txn("HUB", f"R{i}") for i in range(4)]
    records += [txn(f"S{i}", "SINK_ONLY") for i in range(4)]
    batch = build_graph(records)
    assert batch.hub_nodes(min_in=4, min_out=4) == ["HUB"]


# --- multi-hop traversal ------------------------------------------------------------------


def test_a_chained_funnel_is_recovered_layer_by_layer():
    batch = build_graph(chained_funnel())
    walk = batch.multi_hop_layers("SINK", "in", depth=2)
    assert walk.levels == [frozenset({"C1", "C2", "C3"}), frozenset(f"F{i}" for i in range(10))]
    assert not walk.truncated


def test_the_mirror_direction_walks_downstream():
    records = [txn("SRC", c) for c in ("C1", "C2")] + [
        txn(c, f"{c}-R{i}") for c in ("C1", "C2") for i in range(3)
    ]
    walk = build_graph(records).multi_hop_layers("SRC", "out", depth=2)
    assert walk.levels[0] == {"C1", "C2"}
    assert len(walk.levels[1]) == 6


def test_depth_is_clamped_to_the_configured_maximum():
    """Asking for more depth than the guardrail allows gets the guardrail, not the request."""
    chain = [txn(f"N{i}", f"N{i + 1}") for i in range(20)]
    walk = build_graph(chain).multi_hop_layers("N0", "out", depth=99)
    assert len(walk.levels) == get_config().graph.max_traversal_depth


def test_a_level_wider_than_the_frontier_bound_stops_the_walk_and_says_so():
    limit = get_config().graph.max_frontier
    records = [txn(f"S{i}", "HUB") for i in range(limit + 1)]
    walk = build_graph(records).multi_hop_layers("HUB", "in")
    assert walk.truncated and walk.levels == []


def test_an_account_is_reported_at_its_nearest_level_only():
    records = [txn("A", "B"), txn("B", "C"), txn("A", "C")]
    walk = build_graph(records).multi_hop_layers("A", "out", depth=3)
    assert walk.levels == [frozenset({"B", "C"})]


# --- bipartite blocks ---------------------------------------------------------------------


def test_a_complete_block_is_recovered_exactly():
    """SAML-D's Bipartite: two senders each paying the same seven receivers."""
    senders, receivers = ["S1", "S2"], [f"R{i}" for i in range(7)]
    noise = [txn("X", "Y"), txn("Y", "Z")]
    batch = build_graph(complete_block(senders, receivers) + noise)
    (block,) = batch.bipartite_blocks(min_src=2, min_dst=5)
    assert block.senders == set(senders) and block.receivers == set(receivers)
    assert block.density == 1.0 and len(block.refs) == 14


def test_stacked_blocks_are_found_as_separate_layers():
    """Stacked bipartite: the receivers of one block are the senders of the next."""
    first = complete_block(["S1", "S2"], ["M1", "M2", "M3", "M4", "M5"])
    second = complete_block(["M1", "M2", "M3"], ["E1", "E2", "E3", "E4", "E5"], day=3)
    blocks = build_graph(first + second).bipartite_blocks(min_src=2, min_dst=5)
    assert {block.senders for block in blocks} == {
        frozenset({"S1", "S2"}), frozenset({"M1", "M2", "M3"})
    }


def test_senders_that_share_too_few_receivers_are_not_a_block():
    records = [txn("S1", f"R{i}") for i in range(5)] + [txn("S2", f"Q{i}") for i in range(5)]
    records += [txn("S2", "R0")]
    assert build_graph(records).bipartite_blocks(min_src=2, min_dst=5) == []


def test_a_hub_receiver_does_not_join_every_sender_into_one_block():
    side = get_config().graph.max_block_side
    records = [txn(f"S{i}", "HUB") for i in range(side + 1)]
    records += [txn(f"S{i}", f"R{i}-{j}") for i in range(side + 1) for j in range(4)]
    assert build_graph(records).bipartite_blocks(min_src=2, min_dst=1) == []


def test_a_disconnected_graph_has_no_structure_to_find():
    records = [txn(f"A{i}", f"B{i}") for i in range(200)]
    batch = build_graph(records)
    assert batch.bipartite_blocks(min_src=2, min_dst=2) == []
    assert batch.hub_nodes(min_in=2, min_out=2) == []


# --- bounded on a dense graph (Tier 3's degenerate case) ------------------------------------


def test_a_dense_graph_stays_inside_the_bounds_and_finishes():
    """A near-complete graph over 120 accounts: ~14,000 edges, every account a hub. Every query
    must return -- truncated if need be -- well inside the batch budget, never hang."""
    accounts = [f"N{i}" for i in range(120)]
    records = [txn(a, b, day=(i + j) % 20) for i, a in enumerate(accounts)
               for j, b in enumerate(accounts) if a != b]
    batch = build_graph(records)

    started = time.perf_counter()
    walk = batch.multi_hop_layers("N0", "out")
    blocks = batch.bipartite_blocks(min_src=2, min_dst=5)
    hubs = batch.hub_nodes(min_in=5, min_out=5)
    elapsed = time.perf_counter() - started

    assert len(walk.levels) <= get_config().graph.max_traversal_depth
    assert len(hubs) == 120
    # Every receiver has 119 senders, over max_block_side, so no receiver is block material.
    assert blocks == []
    assert elapsed < 30, f"dense-graph queries took {elapsed:.1f}s"


# --- evidence -----------------------------------------------------------------------------


def test_subgraph_evidence_is_plain_data_with_one_edge_per_transaction():
    records = chained_funnel()
    batch = build_graph(records)
    refs = [r.txn_ref for r in records if r.receiver_account == "SINK"]
    evidence = batch.subgraph(refs)
    assert set(evidence["nodes"]) == {"C1", "C2", "C3", "SINK"}
    assert [e["ref"] for e in evidence["edges"]] == refs
    assert all(isinstance(e["amount"], float) for e in evidence["edges"])


@pytest.mark.parametrize(
    "attributes",
    [
        {"route": ["40510055", "40510066"]},
        {"sink": "40510055", "intermediaries": ["40510066"]},
        {"nodes": ["40510055"], "edges": [{"source": "40510055", "target": "40510066"}]},
    ],
)
def test_detector_geometry_is_redacted_before_it_leaves(attributes):
    """`route`, `sink` and `intermediaries` reached the grounding prompt as raw account numbers
    until v2; the subgraph evidence would have added `source`, `target` and `nodes`."""
    out = str(redact(attributes))
    assert "40510055" not in out and "40510066" not in out


def test_the_batch_graph_never_reaches_a_trace():
    batch = build_graph([txn("A", "B")])
    assert trim({"batch_graph": batch}) == {"batch_graph": "[batch graph -- omitted from the trace]"}
