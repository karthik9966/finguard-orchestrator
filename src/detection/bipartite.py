"""Bipartite / stacked bipartite: a set of senders all paying the same set of receivers
(LLD v2 §2.6).

The subgraph-structure detector, and the one with no one-hop reading at all: every sender is a
modest fan-out and every receiver a modest fan-in, and only the *shared* receivers make it a
shape. `BatchGraph.bipartite_blocks` finds the blocks; this module judges them -- density,
window -- and chains blocks whose receivers are the next block's senders into one stacked
candidate, since a layer handed on is one scheme, not two.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.graph_engine import BatchGraph, Block
from src.models import Candidate


class BipartiteDetector(BaseDetector):
    pattern_type = "bipartite"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        rules = get_config().detection.bipartite
        window = timedelta(days=rules.window_days)
        blocks = [
            block
            for block in batch.bipartite_blocks(min_src=rules.min_src, min_dst=rules.min_dst)
            if block.density >= rules.min_density
        ]

        found: list[Candidate] = []
        for stack in _stacks(blocks):
            refs = list(dict.fromkeys(ref for block in stack for ref in block.refs))
            members = [batch.by_ref[ref] for ref in refs]
            times = [record.timestamp for record in members]
            if max(times) - min(times) > window:
                continue
            senders = sorted(set().union(*(block.senders for block in stack)))
            receivers = sorted(set().union(*(block.receivers for block in stack)))
            amounts = [record.amount for record in members]
            found.append(
                self.candidate(
                    anchor=senders[0],
                    refs=refs,
                    confidence=confidence.score(
                        amounts=amounts,
                        timestamps=times,
                        minimum_members=rules.min_src * rules.min_dst,
                    ),
                    stacked=len(stack) > 1,
                    layers=len(stack),
                    senders=senders,
                    receivers=receivers,
                    density=round(min(block.density for block in stack), 4),
                    total=float(sum(amounts, Decimal(0))),
                    window_days=rules.window_days,
                )
            )
        return found


def _stacks(blocks: list[Block]) -> list[list[Block]]:
    """Group blocks into stacks: two blocks join when one's receivers are the other's senders."""
    groups: list[list[Block]] = []
    for block in blocks:
        joined = [
            group
            for group in groups
            if any(block.senders & other.receivers or block.receivers & other.senders
                   for other in group)
        ]
        merged = [block] + [b for group in joined for b in group]
        groups = [group for group in groups if group not in joined] + [merged]
    # Layer order, upstream first: a block whose senders nobody else pays leads the stack.
    return [
        sorted(group, key=lambda b: (any(b.senders & o.receivers for o in group), sorted(b.senders)))
        for group in groups
    ]


register(BipartiteDetector())
