"""The GraphEngine: one graph per batch, built once and shared (LLD v2 §2.5).

Cycle detection needs a directed graph; the fan and structuring detectors need grouped frames.
Building both once and handing them to every detector keeps a 10,000-message batch from being
walked nine times, and keeps every detector's view of the batch identical -- a detector that
built its own would be one refactor away from disagreeing about what the batch contains.

v2 adds the structural queries the multi-hop detectors need: distinct counterparties, hubs,
bounded layer-by-layer traversal and bipartite blocks. **Structure only.** No query here knows a
pattern's thresholds or window -- those are the detectors' -- and every query is bounded by
`config.graph`, so a dense batch truncates (and says so in the log) instead of blowing up.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations
from typing import Literal

import networkx as nx
import pandas as pd

from src.config import get_config
from src.models import TransactionRecord

log = logging.getLogger(__name__)

Direction = Literal["in", "out"]


@dataclass(frozen=True)
class Traversal:
    """Accounts by hop distance from a root: `levels[0]` is the root's direct counterparties.

    `truncated` is True when a bound stopped the walk. A detector may still report what was
    found, but it must not claim the structure ends where the walk did.
    """

    root: str
    direction: Direction
    levels: list[frozenset[str]]
    truncated: bool = False


@dataclass(frozen=True)
class Block:
    """A dense set-of-senders -> set-of-receivers block."""

    senders: frozenset[str]
    receivers: frozenset[str]
    refs: tuple[str, ...]

    @property
    def density(self) -> float:
        """Share of the |S| x |R| sender-receiver pairs that actually transacted."""
        pairs = len(self.senders) * len(self.receivers)
        return len(self.pair_set) / pairs if pairs else 0.0

    @property
    def pair_set(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._pairs)

    _pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BatchGraph:
    """The shared view of one batch."""

    records: list[TransactionRecord]
    frame: pd.DataFrame
    graph: nx.MultiDiGraph
    by_ref: dict[str, TransactionRecord] = field(default_factory=dict)

    def refs(self, rows: pd.DataFrame) -> list[str]:
        return [str(ref) for ref in rows["txn_ref"]]

    def amounts(self, refs: list[str]) -> list[Decimal]:
        return [self.by_ref[ref].amount for ref in refs if ref in self.by_ref]

    # --- structural queries (v2) ----------------------------------------------------------

    def counterparties(self, account: str, direction: Direction) -> set[str]:
        """Distinct accounts that paid (`in`) or were paid by (`out`) this one, self excluded."""
        if account not in self.graph:
            return set()
        nodes = (
            self.graph.predecessors(account) if direction == "in" else self.graph.successors(account)
        )
        return {node for node in nodes if node != account}

    def edge_refs(self, sender: str, receiver: str) -> list[str]:
        """Every transaction from `sender` to `receiver`, in time order."""
        if not self.graph.has_edge(sender, receiver):
            return []
        edges = self.graph.get_edge_data(sender, receiver)
        return [ref for ref, _ in sorted(edges.items(), key=lambda item: item[1]["timestamp"])]

    def hub_nodes(self, *, min_in: int, min_out: int) -> list[str]:
        """Accounts with at least `min_in` distinct payers and `min_out` distinct payees."""
        return sorted(
            node
            for node in self.graph.nodes
            if len(self.counterparties(node, "in")) >= min_in
            and len(self.counterparties(node, "out")) >= min_out
        )

    def multi_hop_layers(
        self, root: str, direction: Direction, depth: int | None = None
    ) -> Traversal:
        """Breadth-first layers out from `root`, each account appearing at its nearest level.

        Bounded twice: by `max_traversal_depth` (a request for more is clamped, not honoured) and
        by `max_frontier`, the widest a level may grow before the walk stops.
        """
        bounds = get_config().graph
        limit = min(depth or bounds.max_traversal_depth, bounds.max_traversal_depth)
        seen = {root}
        frontier = {root}
        levels: list[frozenset[str]] = []
        truncated = False
        for _ in range(limit):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= self.counterparties(node, direction) - seen
            if not nxt:
                break
            if len(nxt) > bounds.max_frontier:
                log.info("traversal from %s truncated at %d accounts", root, len(nxt))
                truncated = True
                break
            levels.append(frozenset(nxt))
            seen |= nxt
            frontier = nxt
        return Traversal(root=root, direction=direction, levels=levels, truncated=truncated)

    def bipartite_blocks(self, *, min_src: int, min_dst: int) -> list[Block]:
        """Maximal groups of >= `min_src` senders that share >= `min_dst` common receivers.

        Senders are linked when a pair shares `min_dst` receivers, and each connected group of
        linked senders is one block over the receivers at least two of them share. Exact
        biclique search is NP-hard; this is the bounded approximation the LLD asks for, and
        `Block.density` reports how close to complete the result is so the detector can judge.

        Receivers with more than `max_block_side` senders are left out: a hub links everyone.
        """
        bounds = get_config().graph
        payees = {
            node: self.counterparties(node, "out") for node in self.graph.nodes
        }
        eligible_receivers = {
            node
            for node in self.graph.nodes
            if len(self.counterparties(node, "in")) <= bounds.max_block_side
        }
        senders = sorted(
            (
                node
                for node, out in payees.items()
                if len(out & eligible_receivers) >= min_dst
            ),
            key=lambda node: (-len(payees[node]), node),
        )
        if len(senders) > bounds.max_block_senders:
            log.info(
                "bipartite search truncated to %d of %d senders",
                bounds.max_block_senders,
                len(senders),
            )
            senders = senders[: bounds.max_block_senders]

        # Union-find over senders linked by >= min_dst shared receivers.
        parent = {node: node for node in senders}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for first, second in combinations(senders, 2):
            shared = payees[first] & payees[second] & eligible_receivers
            if len(shared) >= min_dst:
                parent[find(first)] = find(second)

        groups: dict[str, set[str]] = defaultdict(set)
        for node in senders:
            groups[find(node)].add(node)

        blocks: list[Block] = []
        for group in groups.values():
            if len(group) < min_src:
                continue
            counts: dict[str, int] = defaultdict(int)
            for sender in group:
                for receiver in payees[sender] & eligible_receivers:
                    counts[receiver] += 1
            receivers = {receiver for receiver, n in counts.items() if n >= 2} - group
            if len(receivers) < min_dst:
                continue
            pairs = tuple(
                sorted(
                    (sender, receiver)
                    for sender in group
                    for receiver in receivers
                    if self.graph.has_edge(sender, receiver)
                )
            )
            refs = tuple(ref for sender, receiver in pairs for ref in self.edge_refs(sender, receiver))
            blocks.append(
                Block(
                    senders=frozenset(group),
                    receivers=frozenset(receivers),
                    refs=refs,
                    _pairs=pairs,
                )
            )
        return sorted(blocks, key=lambda block: (-len(block.refs), sorted(block.senders)))

    def subgraph(self, refs: list[str]) -> dict:
        """The evidence form of a set of transactions: nodes, and one edge per transaction.

        Plain data, so it validates into `Candidate.subgraph`, serialises into the immutable
        report, and renders in the cockpit without the reader needing networkx.
        """
        edges = []
        nodes: dict[str, None] = {}
        for ref in refs:
            record = self.by_ref.get(ref)
            if record is None:
                continue
            nodes.setdefault(record.sender_account)
            nodes.setdefault(record.receiver_account)
            edges.append(
                {
                    "ref": ref,
                    "source": record.sender_account,
                    "target": record.receiver_account,
                    "amount": float(record.amount),
                    "timestamp": record.timestamp.isoformat(),
                    "payment_kind": record.payment_kind,
                }
            )
        return {"nodes": list(nodes), "edges": edges}


def build_graph(records: list[TransactionRecord]) -> BatchGraph:
    """Build the frame and the directed multigraph for a batch.

    A MultiDiGraph rather than a DiGraph because two accounts can transact repeatedly and each
    payment is its own edge: collapsing them would hide exactly the repetition structuring is.
    """
    frame = pd.DataFrame(
        [
            {
                "txn_ref": record.txn_ref,
                "sender_account": record.sender_account,
                "receiver_account": record.receiver_account,
                # float for grouping arithmetic only; every reported figure is read back from
                # the record's Decimal, never from this column.
                "amount": float(record.amount),
                "timestamp": record.timestamp,
                "currency": record.currency,
                "payment_kind": record.payment_kind,
            }
            for record in records
        ]
    )
    if not frame.empty:
        frame = frame.sort_values("timestamp").reset_index(drop=True)

    graph = nx.MultiDiGraph()
    for record in records:
        graph.add_edge(
            record.sender_account,
            record.receiver_account,
            key=record.txn_ref,
            amount=float(record.amount),
            timestamp=record.timestamp,
            payment_kind=record.payment_kind,
        )

    return BatchGraph(
        records=list(records),
        frame=frame,
        graph=graph,
        by_ref={record.txn_ref: record for record in records},
    )
