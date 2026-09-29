"""Layered fan-in / fan-out: funnels feeding funnels, found by multi-hop traversal (LLD v2 §2.6).

One hop cannot see this. Each intermediate account is an ordinary fan-in (or fan-out); the
laundering structure is that several of them feed the *same* root. So the detector walks two
levels from every candidate root with `BatchGraph.multi_hop_layers` and asks whether enough of the
first level are themselves fans.

Both directions are one pattern type, `layered_fan`, with `direction` in the attributes -- the LLD
v2 Literal names one value, and the obligations and indicators do not differ by direction.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.graph_engine import BatchGraph, Direction
from src.models import Candidate


class LayeredFanDetector(BaseDetector):
    pattern_type = "layered_fan"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        found: list[Candidate] = []
        for direction in ("in", "out"):
            found.extend(self._funnels(batch, direction))
        return found

    def _funnels(self, batch: BatchGraph, direction: Direction) -> list[Candidate]:
        rules = get_config().detection.layered_fan
        window = timedelta(days=rules.window_days)
        found: list[Candidate] = []

        for root in sorted(batch.graph.nodes):
            if len(batch.counterparties(root, direction)) < rules.min_branches:
                continue
            walk = batch.multi_hop_layers(root, direction, depth=2)
            if not walk.levels:
                continue

            # A leaf counts only if its leg is within the window of that branch's own hand-off to
            # the root. Measuring the window over the whole structure instead let one busy branch's
            # month of unrelated traffic push a real funnel outside it.
            branches: dict[str, list[str]] = {}
            for branch in walk.levels[0]:
                top = self._legs(batch, root, branch, direction, upper=True)
                handoffs = [batch.by_ref[ref].timestamp for ref in top]
                leaf_refs = [
                    ref
                    for leaf in sorted(batch.counterparties(branch, direction) - {root} - walk.levels[0])
                    for ref in self._legs(batch, branch, leaf, direction, upper=False)
                    if any(abs(batch.by_ref[ref].timestamp - t) <= window for t in handoffs)
                ]
                leaves = {self._far_end(batch.by_ref[ref], direction) for ref in leaf_refs}
                if len(leaves) >= rules.min_leaves_per_branch:
                    branches[branch] = top + leaf_refs
            leaves_all = {
                self._far_end(batch.by_ref[ref], direction)
                for refs in branches.values() for ref in refs
            } - set(branches) - {root}
            if len(branches) < rules.min_branches or len(leaves_all) < rules.min_total_leaves:
                continue

            refs = [ref for branch in sorted(branches) for ref in branches[branch]]
            members = [batch.by_ref[ref] for ref in refs]
            times = [record.timestamp for record in members]

            amounts = [record.amount for record in members]
            found.append(
                self.candidate(
                    anchor=str(root),
                    refs=refs,
                    confidence=confidence.score(
                        amounts=amounts,
                        timestamps=times,
                        minimum_members=rules.min_total_leaves + rules.min_branches,
                    ),
                    direction=direction,
                    layers=2,
                    collectors=sorted(branches),
                    branch_count=len(branches),
                    leaf_count=len(leaves_all),
                    traversal_truncated=walk.truncated,
                    total=float(sum(amounts, Decimal(0))),
                    window_days=rules.window_days,
                )
            )
        return found


    @staticmethod
    def _legs(batch: BatchGraph, near: str, far: str, direction: Direction, *, upper: bool):
        """Transactions on one hop of the funnel, oriented by the funnel's direction.

        For a fan-in, money flows far -> near (leaf -> branch, branch -> root); for a fan-out,
        near -> far. `upper` is the root <-> branch hop, where `near` is the root.
        """
        if direction == "in":
            return batch.edge_refs(far, near)
        return batch.edge_refs(near, far)

    @staticmethod
    def _far_end(record, direction: Direction) -> str:
        """The account on the outer side of a leg: its sender for a fan-in, receiver for a fan-out."""
        return record.sender_account if direction == "in" else record.receiver_account


register(LayeredFanDetector())
