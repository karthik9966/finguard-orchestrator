"""Deposit-send: a cash deposit, then a transfer of about the same amount out (LLD v2 §2.6).

Detected from payment kind, edge direction and timing, as the LLD specifies -- but keyed on an
actual `cash_deposit`, which SAML-D does carry. The LLD's note that the data has only a flat
"cash" value described our parser, which used to keep the first word of the payment type.

**The amount match is the discriminator, not the timing.** Measured over one clean SAML-D month,
66% of depositors send *something* within three days; requiring the send to match the deposit
within tolerance brings that to about 4% while keeping most of the planted cases (config.yaml
has the curve). Each deposit is paired with the nearest matching send, and a send pairs at most
once, so one large wire cannot be claimed by every deposit before it.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from decimal import Decimal

from src.config import get_config
from src.detection import confidence
from src.detection.base import BaseDetector, register
from src.detection.graph_engine import BatchGraph
from src.models import Candidate, TransactionRecord


class DepositSendDetector(BaseDetector):
    pattern_type = "deposit_send"

    def detect(self, batch: BatchGraph) -> list[Candidate]:
        rules = get_config().detection.deposit_send
        window = timedelta(hours=rules.window_hours)
        send_kinds = set(rules.send_kinds)

        deposits: dict[str, list[TransactionRecord]] = defaultdict(list)
        sends: dict[str, list[TransactionRecord]] = defaultdict(list)
        for record in batch.records:
            if record.payment_kind == "cash_deposit":
                deposits[record.receiver_account].append(record)
            elif record.payment_kind in send_kinds:
                sends[record.sender_account].append(record)

        found: list[Candidate] = []
        for account in sorted(deposits):
            if account not in sends:
                continue
            used: set[str] = set()
            pairs: list[tuple[TransactionRecord, TransactionRecord]] = []
            for deposit in sorted(deposits[account], key=lambda r: r.timestamp):
                match = next(
                    (
                        send
                        for send in sorted(sends[account], key=lambda r: r.timestamp)
                        if send.txn_ref not in used
                        and timedelta(0) <= send.timestamp - deposit.timestamp <= window
                        and abs(float(send.amount / deposit.amount) - 1.0) <= rules.amount_tolerance
                    ),
                    None,
                )
                if match is not None:
                    used.add(match.txn_ref)
                    pairs.append((deposit, match))
            if len(pairs) < rules.min_pairs:
                continue

            members = [record for pair in pairs for record in pair]
            refs = [record.txn_ref for record in members]
            deposited = sum((d.amount for d, _ in pairs), Decimal(0))
            sent = sum((s.amount for _, s in pairs), Decimal(0))
            found.append(
                self.candidate(
                    anchor=str(account),
                    refs=refs,
                    confidence=confidence.score(
                        amounts=[record.amount for record in members],
                        timestamps=[record.timestamp for record in members],
                        minimum_members=2 * rules.min_pairs,
                    ),
                    pairs=len(pairs),
                    total_deposited=float(deposited),
                    total_sent=float(sent),
                    max_gap_hours=round(
                        max((s.timestamp - d.timestamp).total_seconds() for d, s in pairs) / 3600,
                        2,
                    ),
                    cross_border=any(s.is_cross_border or s.payment_kind == "cross_border"
                                     for _, s in pairs),
                    cash_intensive=True,
                    window_hours=rules.window_hours,
                )
            )
        return found


register(DepositSendDetector())
