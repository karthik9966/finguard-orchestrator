"""Structuring: deliberately kept below a reporting threshold (31 USC §5324, 31 CFR 1010.311).

Group by originator; inside window W, at least `min_count` transactions each sitting in
`[T - band, T)` for T in the monitored thresholds, with the group totalling at least T.

**And by beneficiary (v2).** SAML-D's Structuring clusters are all receiver-anchored: ten parties
each paying one account once (measured: every one of 1,870 rows has a distinct sender per month,
and the 224 clusters hang off receivers). Grouping by originator alone could never see one, so
they were counted as "found" only because fan-in happened to cover them. An account taking several
just-under-threshold deposits from different people is the FFIEC red flag for structuring through
others, and §5324 reaches whoever *causes* the splitting -- so both sides are grouped, the
originator first, and `side` records which one a candidate hangs off.

The band is a **fraction** of T, not an absolute. LLD §8 names one absolute band, which cannot
serve both thresholds: an absolute 2000 makes the $3,000 band [1000, 3000) and catches most
ordinary payments. 0.2 gives [8000, 10000) and [2400, 3000). Deviation recorded in config.yaml.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.graph_engine import BatchGraph
from src.models import Candidate


# Originator first: a run one party split is the plainer reading, and `claimed` stops the same
# transactions being reported again from the receiving side.
SIDES = (("originator", "sender_account"), ("beneficiary", "receiver_account"))


class StructuringDetector(BaseDetector):
    pattern_type = "structuring"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        if batch.frame.empty:
            return []

        rules = get_config().detection.structuring
        window = pd.Timedelta(days=self.window_days)
        found: list[Candidate] = []
        claimed: set[str] = set()

        for threshold in sorted(rules.thresholds, reverse=True):
            floor = threshold * (1 - rules.band_fraction)
            in_band = batch.frame[
                (batch.frame.amount >= floor) & (batch.frame.amount < threshold)
            ]
            for side, column in SIDES:
                for account, rows in in_band.groupby(column):
                    if len(rows) < rules.min_count:
                        continue
                    for window_rows in self._windows(rows, window, rules.min_count):
                        refs = batch.refs(window_rows)
                        # The larger threshold is checked first; a group already reported under
                        # $10,000 is not reported again under $3,000.
                        if claimed.intersection(refs):
                            continue
                        total = sum(batch.amounts(refs), Decimal(0))
                        if total < Decimal(threshold) * Decimal(str(rules.min_aggregate_multiple)):
                            continue

                        amounts = batch.amounts(refs)
                        found.append(
                            self.candidate(
                                anchor=str(account),
                                refs=refs,
                                confidence=confidence.score(
                                    amounts=amounts,
                                    timestamps=list(window_rows.timestamp),
                                    minimum_members=rules.min_count,
                                    band_amounts=amounts,  # every member is in-band by construction
                                    threshold=Decimal(threshold),
                                ),
                                threshold=threshold,
                                side=side,
                                band=[float(floor), float(threshold)],
                                total=float(total),
                                count=len(refs),
                                window_days=self.window_days,
                            )
                        )
                        claimed.update(refs)
        return found

    @staticmethod
    def _windows(rows: pd.DataFrame, window: pd.Timedelta, minimum: int):
        """Maximal runs of in-band transfers that fit inside the window.

        A sliding window rather than a calendar month: structuring is defined by the *pace* of
        the deposits, and a month boundary would split a run that straddles it while joining two
        that merely share a month.
        """
        rows = rows.sort_values("timestamp")
        times = list(rows.timestamp)
        start = 0
        emitted: set[tuple[int, int]] = set()
        for end in range(len(times)):
            while times[end] - times[start] > window:
                start += 1
            if end - start + 1 >= minimum and (start, end) not in emitted:
                emitted.add((start, end))
        # Keep only the longest run from each start, so one cluster is not reported at every
        # length it passes through.
        best: dict[int, int] = {}
        for start, end in emitted:
            best[start] = max(best.get(start, end), end)
        seen_spans: list[set[int]] = []
        for start in sorted(best):
            span = set(range(start, best[start] + 1))
            if any(span <= previous for previous in seen_spans):
                continue
            seen_spans.append(span)
            yield rows.iloc[sorted(span)]


register(StructuringDetector())
