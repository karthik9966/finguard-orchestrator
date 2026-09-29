"""The matched structure, in the three forms a finding needs it (PRD v2 §5.3).

PRD v2 asks that a multi-hop finding show the money-flow structure, not a flat transaction list.
The GraphEngine attaches that structure to every candidate as `Candidate.subgraph`; this module
renders it -- a sentence for the report summary, an edge list for the grounding prompt, a DOT graph
for the cockpit. Pure functions over plain data, so the report generator, the prompt and the UI
client draw the same structure the same way, and none of them needs networkx or the batch.
"""

from __future__ import annotations

from typing import Any

from src.models import Candidate
from src.utils.redaction import redact

_KIND_LABEL = {
    "cash_deposit": "cash deposit",
    "cash_withdrawal": "cash withdrawal",
    "cross_border": "cross-border",
}


def describe(candidate: Candidate) -> str:
    """One templated sentence of the shape, from the detector's own measurements. No model."""
    a = candidate.attributes or {}
    n = len(candidate.member_txn_refs)
    pattern = candidate.pattern_type
    if pattern == "gather_scatter":
        return (
            f"One account received from {a.get('distinct_senders', '?')} payers and paid "
            f"{a.get('distinct_recipients', '?')} payees within {a.get('window_days', '?')} days, "
            f"passing on {a.get('conservation', 0):.0%} of what came in."
        )
    if pattern == "deposit_send":
        return (
            f"{a.get('pairs', '?')} cash deposit(s) each followed within "
            f"{a.get('max_gap_hours', '?')} hours by a transfer out of about the same amount"
            + (", cross-border." if a.get("cross_border") else ".")
        )
    if pattern == "layered_fan":
        verb = "collected into" if a.get("direction") == "in" else "dispersed from"
        return (
            f"Funds {verb} one account through {a.get('branch_count', '?')} intermediary "
            f"account(s) from {a.get('leaf_count', '?')} outer account(s): two layers, "
            f"{n} transaction(s)."
        )
    if pattern == "bipartite":
        shape = f"{a.get('layers', 1)} stacked layers" if a.get("stacked") else "one block"
        return (
            f"{len(a.get('senders', []))} sender(s) paying the same {len(a.get('receivers', []))} "
            f"receiver(s) at density {a.get('density', 0):.0%} ({shape}), {n} transaction(s)."
        )
    if pattern == "cycle":
        return f"Funds left and returned to one account over {a.get('hops', '?')} hops."
    if pattern == "scatter_gather":
        return f"One account paid {a.get('fan', '?')} intermediaries, which paid one account."
    return f"{n} transaction(s) matching {pattern.replace('_', '-')}."


def edge_lines(subgraph: dict[str, Any] | None, *, limit: int) -> list[str]:
    """The structure as redacted edges for the grounding prompt, capped at `limit`.

    Redacted with the same pseudonyms as the candidate's attributes, so an account named in
    `collectors` is recognisably the same account in an edge.
    """
    if not subgraph:
        return []
    edges = redact(subgraph).get("edges", [])
    lines = [
        f"{e['ref']}: {e['source']} -> {e['target']} "
        f"{e['amount']:,.2f} {_KIND_LABEL.get(e['payment_kind'], e['payment_kind'])} "
        f"{str(e['timestamp'])[:10]}"
        for e in edges[:limit]
    ]
    if len(edges) > limit:
        lines.append(f"... {len(edges) - limit} more edge(s) omitted")
    return lines


def to_dot(subgraph: dict[str, Any] | None) -> str | None:
    """A Graphviz DOT string of the structure, for `st.graphviz_chart`. None if there is none.

    Accounts are shortened to their last six digits for legibility; the reviewer sees the full
    numbers in the transaction table. Parallel payments stay separate edges, because repetition
    between the same two accounts is itself evidence.
    """
    if not subgraph or not subgraph.get("edges"):
        return None

    def node(account: str) -> str:
        return f'"{account}" [label="…{account[-6:]}"]'

    lines = ["digraph {", "  rankdir=LR;", '  node [shape=box, style=rounded, fontsize=10];']
    lines += [f"  {node(str(account))};" for account in subgraph.get("nodes", [])]
    for e in subgraph["edges"]:
        kind = _KIND_LABEL.get(e["payment_kind"], e["payment_kind"])
        lines.append(
            f'  "{e["source"]}" -> "{e["target"]}" '
            f'[label="{e["amount"]:,.0f} {kind}", fontsize=9];'
        )
    lines.append("}")
    return "\n".join(lines)
