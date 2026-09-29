"""The three renderings of a finding's structure (PRD v2 §5.3): summary line, prompt, cockpit."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.config import get_config
from src.detection import detect_all, evidence
from src.graph.prompts import render_candidate
from src.models import TransactionRecord

T0 = datetime(2023, 6, 1, 9, 0, tzinfo=timezone.utc)


def txn(ref, sender, receiver, amount=3000, day=0):
    return TransactionRecord(
        txn_ref=ref, sender_account=sender, receiver_account=receiver,
        amount=Decimal(str(amount)), currency="USD", timestamp=T0 + timedelta(days=day),
        sender_country="US", receiver_country="US", instrument="ACH",
    )


def layered_fan_in():
    records = []
    for c in range(3):
        records += [txn(f"L{c}{i}", f"4000000{c}{i}0", f"5000000{c}00", day=i) for i in range(3)]
        records.append(txn(f"T{c}", f"5000000{c}00", "9000000000", 9000, day=5))
    (found,) = [c for c in detect_all(records) if c.pattern_type == "layered_fan"]
    return found


def test_the_summary_line_is_templated_from_the_detector_measurements():
    line = evidence.describe(layered_fan_in())
    assert line.startswith("Funds collected into one account through 3 intermediary")
    assert "9 outer account(s)" in line


def test_the_prompt_carries_the_structure_redacted():
    rendered = render_candidate(layered_fan_in())
    assert "structure (one line per transaction):" in rendered
    assert "T0: ACCT-" in rendered, "edges are listed by reference with pseudonymised accounts"
    assert "9000000000" not in rendered and "5000000000" not in rendered


def test_the_prompt_caps_the_edge_list_and_says_so():
    candidate = layered_fan_in()
    limit = 4
    lines = evidence.edge_lines(candidate.subgraph, limit=limit)
    assert len(lines) == limit + 1 and lines[-1] == "... 8 more edge(s) omitted"
    assert get_config().reasoning.evidence_edges_in_prompt >= len(candidate.member_txn_refs)


def test_the_cockpit_graph_draws_every_transaction_as_its_own_edge():
    dot = evidence.to_dot(layered_fan_in().subgraph)
    assert dot.startswith("digraph {") and dot.count("->") == 12
    assert evidence.to_dot(None) is None
