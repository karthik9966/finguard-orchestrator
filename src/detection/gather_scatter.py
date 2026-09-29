"""Gather-scatter: many pay one hub, and the hub pays many, inside the window (LLD v2 §2.6).

The inverse of scatter-gather, and a fan-in and a fan-out at the same account. Without its own
detector those two fired on the halves and each reported half the event, which is why
`precedence_order` puts this ahead of both.

**Conservation** is what separates a pass-through from a busy account: SAML-D hubs empty what
they fill. A hub that keeps the money, or pays out far more than came in, is two unrelated flows
that happen to share an account.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.graph_engine import BatchGraph
from src.models import Candidate


class GatherScatterDetector(BaseDetector):
    pattern_type = "gather_scatter"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        rules = get_config().detection.gather_scatter
        window = timedelta(days=rules.window_days)
        found: list[Candidate] = []

        for hub in batch.hub_nodes(min_in=rules.min_in, min_out=rules.min_out):
            inbound = sorted(
                (r for r in batch.records if r.receiver_account == hub and r.sender_account != hub),
                key=lambda r: r.timestamp,
            )
            outbound = sorted(
                (r for r in batch.records if r.sender_account == hub and r.receiver_account != hub),
                key=lambda r: r.timestamp,
            )
            # The window opens at the first inflow: scattering before any gathering is a payer
            # that later got paid, not a pass-through.
            best = None
            for start in inbound:
                end = start.timestamp + window
                gathered = [r for r in inbound if start.timestamp <= r.timestamp <= end]
                scattered = [r for r in outbound if start.timestamp <= r.timestamp <= end]
                senders = {r.sender_account for r in gathered}
                receivers = {r.receiver_account for r in scattered}
                if len(senders) < rules.min_in or len(receivers) < rules.min_out:
                    continue
                if best is None or len(gathered) + len(scattered) > len(best[0]) + len(best[1]):
                    best = (gathered, scattered)
            if best is None:
                continue

            gathered, scattered = best
            total_in = sum((r.amount for r in gathered), Decimal(0))
            total_out = sum((r.amount for r in scattered), Decimal(0))
            conservation = float(total_out / total_in) if total_in else 0.0
            if not rules.min_conservation <= conservation <= rules.max_conservation:
                continue

            members = gathered + scattered
            refs = [r.txn_ref for r in members]
            found.append(
                self.candidate(
                    anchor=str(hub),
                    refs=refs,
                    confidence=confidence.score(
                        amounts=[r.amount for r in members],
                        timestamps=[r.timestamp for r in members],
                        minimum_members=rules.min_in + rules.min_out,
                    ),
                    hub=str(hub),
                    distinct_senders=len({r.sender_account for r in gathered}),
                    distinct_recipients=len({r.receiver_account for r in scattered}),
                    total_in=float(total_in),
                    total_out=float(total_out),
                    conservation=round(conservation, 4),
                    window_days=rules.window_days,
                )
            )
        return found


register(GatherScatterDetector())
