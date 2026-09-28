"""Scatter-gather: one account pays many, and the many pay one, inside the window.

Distinct from fan-out precisely because of the second leg. `fan_out` fires on the first leg
alone, which is why `precedence_order` puts this detector ahead of it -- otherwise the same
transactions are reported twice, once under the shape that explains them and once under the shape
that only sees half.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from decimal import Decimal

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.graph_builder import BatchGraph
from src.models import Candidate


class ScatterGatherDetector(BaseDetector):
    pattern_type = "scatter_gather"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        minimum = get_config().detection.scatter_gather.min_fan
        gap = timedelta(days=self.window_days)

        outgoing = defaultdict(list)
        incoming_by_sender = defaultdict(list)
        for record in batch.records:
            outgoing[record.sender_account].append(record)
            incoming_by_sender[record.sender_account].append(record)

        found: list[Candidate] = []
        for source, first_leg in outgoing.items():
            if len({r.receiver_account for r in first_leg}) < minimum:
                continue

            # Where do the intermediaries send it next?
            collectors: dict[str, list] = defaultdict(list)
            for scatter in first_leg:
                for gather in incoming_by_sender.get(scatter.receiver_account, ()):
                    delta = gather.timestamp - scatter.timestamp
                    if timedelta(0) <= delta <= gap and gather.receiver_account != source:
                        collectors[gather.receiver_account].append((scatter, gather))

            for sink, pairs in collectors.items():
                intermediaries = {scatter.receiver_account for scatter, _ in pairs}
                if len(intermediaries) < minimum:
                    continue
                records = [record for pair in pairs for record in pair]
                refs = list(dict.fromkeys(record.txn_ref for record in records))
                amounts = [batch.by_ref[ref].amount for ref in refs]
                found.append(
                    self.candidate(
                        anchor=str(source),
                        refs=refs,
                        confidence=confidence.score(
                            amounts=amounts,
                            timestamps=[batch.by_ref[ref].timestamp for ref in refs],
                            minimum_members=minimum * 2,
                        ),
                        sink=str(sink),
                        intermediaries=sorted(intermediaries),
                        fan=len(intermediaries),
                        total=float(sum(amounts, Decimal(0))),
                        window_days=self.window_days,
                    )
                )
        return found


register(ScatterGatherDetector())
