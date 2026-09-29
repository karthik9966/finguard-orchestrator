"""Cycle: money leaving an account and returning through intermediaries.

Ported from the pre-migration `find_paths` (removed in Phase 5), keeping the two things that
made it work.

**Counting cannot see this shape.** A ring's edges span as many distinct senders as receivers, so
no account stands out -- it only exists along the direction of travel.

**`path_overlap` suppression.** One ring is reachable from every edge along it, and every branch
walks the same opening hops before diverging, so a single ten-hop chain surfaces as a dozen
variants sharing most of their transactions. Containment alone does not remove them, because a
branch is not a subset of the trunk. Without suppression one ring produced 11 near-duplicate
chains; it produced 2 after.

`max_length` and the window bound the DFS: both are what keep a 10,000-message batch from
becoming a graph walk that does not finish.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from decimal import Decimal

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.graph_builder import BatchGraph
from src.models import Candidate, TransactionRecord


class CycleDetector(BaseDetector):
    pattern_type = "cycle"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        rules = get_config().detection.cycle
        gap = timedelta(days=self.window_days)

        outgoing: dict[str, list[TransactionRecord]] = defaultdict(list)
        for record in batch.records:
            outgoing[record.sender_account].append(record)

        chains: list[list[TransactionRecord]] = []

        def walk(chain: list[TransactionRecord], visited: set[str]) -> None:
            last = chain[-1]
            origin = chain[0].sender_account
            extended = False
            if len(chain) < rules.max_length:
                for nxt in outgoing.get(last.receiver_account, ()):
                    delta = nxt.timestamp - last.timestamp
                    if not (timedelta(0) <= delta <= gap):
                        continue
                    # Closing back to the origin is the whole point: a *directed cycle* is money
                    # that comes home. The visited set exists to stop a busy hub generating
                    # spurious routes through itself, and it was rejecting the one edge that
                    # makes a ring a ring.
                    if nxt.receiver_account == origin:
                        closed = chain + [nxt]
                        if len(closed) >= rules.min_hops:
                            chains.append(closed)
                        continue
                    if nxt.receiver_account not in visited:
                        extended = True
                        walk(chain + [nxt], visited | {nxt.receiver_account})
            if not extended and len(chain) >= rules.min_hops:
                chains.append(chain)

        for record in batch.records:
            walk([record], {record.sender_account, record.receiver_account})

        found: list[Candidate] = []
        kept: list[set[str]] = []
        for chain in sorted(chains, key=len, reverse=True):
            members = {record.txn_ref for record in chain}
            if any(
                len(members & seen) / len(members) > rules.path_overlap for seen in kept
            ):
                continue

            opening, closing = chain[0].amount, chain[-1].amount
            retained = float(closing / opening) if opening else 0.0
            # Funds must come back roughly intact for a ring to read as laundering rather than
            # as a coincidence of ordinary payments. SAML-D rings decay 10-20% a hop.
            if retained < rules.min_retained_fraction:
                continue

            kept.append(members)
            refs = [record.txn_ref for record in chain]
            found.append(
                self.candidate(
                    anchor=chain[0].sender_account,
                    refs=refs,
                    confidence=confidence.score(
                        amounts=[record.amount for record in chain],
                        timestamps=[record.timestamp for record in chain],
                        minimum_members=rules.min_hops,
                    ),
                    hops=len(chain),
                    retained_fraction=round(retained, 4),
                    route=[chain[0].sender_account] + [r.receiver_account for r in chain],
                    total=float(sum((r.amount for r in chain), Decimal(0))),
                    window_days=self.window_days,
                )
            )
        return found


register(CycleDetector())
