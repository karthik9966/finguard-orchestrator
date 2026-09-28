"""One graph per batch, built once and shared (LLD §2.4).

Cycle detection needs a directed graph; the fan and structuring detectors need grouped frames.
Building both once and handing them to every detector keeps a 10,000-message batch from being
walked five times, and keeps every detector's view of the batch identical -- a detector that
built its own would be one refactor away from disagreeing about what the batch contains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import networkx as nx
import pandas as pd

from src.models import TransactionRecord


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
        )

    return BatchGraph(
        records=list(records),
        frame=frame,
        graph=graph,
        by_ref={record.txn_ref: record for record in records},
    )
