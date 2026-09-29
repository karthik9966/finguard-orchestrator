"""Fan-in: many distinct senders paying one account inside the window.

Ported from the pre-migration `find_clusters`, with one behaviour change: **a time
window**. The old primitive applied none, so an account with fifteen counterparties spread over a
month scored exactly like fifteen in an afternoon -- and on the 10,000-message batch the
unwindowed primitives returned 932 candidates.

Counting via pandas `groupby` rather than walking the graph: the shape is an in-degree, and a
graph walk would cost more and say the same thing.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.graph_engine import BatchGraph
from src.models import Candidate


def windowed_groups(
    frame: pd.DataFrame, *, key: str, counterparty: str, minimum: int, window_days: int
):
    """Yield `(account, rows)` where `rows` are inside one window and come from `minimum`
    distinct counterparties."""
    if frame.empty:
        return
    window = pd.Timedelta(days=window_days)
    for account, rows in frame.groupby(key):
        rows = rows.sort_values("timestamp")
        times = list(rows.timestamp)
        start = 0
        best: tuple[int, int] | None = None
        for end in range(len(times)):
            while times[end] - times[start] > window:
                start += 1
            span = rows.iloc[start : end + 1]
            if span[counterparty].nunique() >= minimum:
                # Keep the widest qualifying window per account: an auditor wants the whole run,
                # not the first moment it became reportable.
                if best is None or (end - start) > (best[1] - best[0]):
                    best = (start, end)
        if best is not None:
            yield account, rows.iloc[best[0] : best[1] + 1]


class FanInDetector(BaseDetector):
    pattern_type = "fan_in"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        minimum = get_config().detection.fan_in.min_sources
        found: list[Candidate] = []
        for account, rows in windowed_groups(
            batch.frame,
            key="receiver_account",
            counterparty="sender_account",
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
                    distinct_senders=int(rows.sender_account.nunique()),
                    total=float(sum(amounts, Decimal(0))),
                    count=len(refs),
                    window_days=self.window_days,
                )
            )
        return found


register(FanInDetector())
