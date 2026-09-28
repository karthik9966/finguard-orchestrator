"""Fan-out: one account paying many distinct recipients inside the window.

The mirror of fan-in, and the detector that most needs reconciliation after it. It fires on the
**first leg of every scatter-gather**, and on several typologies PRD §2 puts out of scope --
gather-scatter, layered fan-out, bipartite. `precedence_order` therefore lets the more specific
shape claim the transactions first.
"""

from __future__ import annotations

from decimal import Decimal

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.fan_in import windowed_groups
from src.detection.graph_builder import BatchGraph
from src.models import Candidate


class FanOutDetector(BaseDetector):
    pattern_type = "fan_out"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        minimum = get_config().detection.fan_out.min_targets
        found: list[Candidate] = []
        for account, rows in windowed_groups(
            batch.frame,
            key="sender_account",
            counterparty="receiver_account",
            minimum=minimum,
            window_days=self.window_days,
        ):
            refs = batch.refs(rows)
            amounts = batch.amounts(refs)
            found.append(
                self.candidate(
                    anchor=str(account),
                    refs=refs,
                    confidence=confidence.score(
                        amounts=amounts,
                        timestamps=list(rows.timestamp),
                        minimum_members=minimum,
                    ),
                    distinct_recipients=int(rows.receiver_account.nunique()),
                    total=float(sum(amounts, Decimal(0))),
                    count=len(refs),
                    window_days=self.window_days,
                )
            )
        return found


register(FanOutDetector())
