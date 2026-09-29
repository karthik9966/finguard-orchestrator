"""The rules-only baseline detector -- HLD §1.1, in scope and explicitly "not a runtime component".

This exists to make the recall number mean something. "99% recall" on its own is unfalsifiable: a
detector that flags every transaction also scores 100%, and the interesting claim is never recall
alone but **recall at a given alert volume**. So the comparator is the thing a bank's legacy
transaction-monitoring rules actually do -- flat thresholds, no graph, no time window -- and the
comparison reported is both numbers side by side.

The rules below are deliberately the obvious ones, drawn from the same US thresholds the real
detectors cite, because a strawman baseline would flatter the system as much as no baseline at all:

* **R1 CTR threshold** -- any single transfer at or above $10,000 (31 CFR 1010.311's filing trigger).
* **R2 sub-threshold amount** -- any transfer in [$8,000, $10,000), the band a structuring rule
  watches. This is the rule most likely to fire on real structuring, and also on a great deal of
  ordinary business.
* **R3 daily aggregate** -- any account whose same-day total across transfers reaches $10,000, which
  is how a legacy engine approximates structuring without a graph.
* **R4 counterparty count** -- any account with 8 or more distinct counterparties in the batch. No
  window, which is exactly the limitation the windowed detectors were built to fix.
* **R5 recordkeeping threshold** -- any transfer at or above $3,000 (31 CFR 1010.410(e)). Included
  because it is genuinely in the regulations and genuinely useless as an alert: it fires on most of
  the batch, which is the point being demonstrated.

Every rule is per-transaction or per-account arithmetic. Nothing here walks a graph, and nothing here
knows what a typology is -- that is the whole difference being measured.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

from src.models import TransactionRecord

CTR_THRESHOLD = Decimal("10000")
BAND_FLOOR = Decimal("8000")
RECORDKEEPING_THRESHOLD = Decimal("3000")
COUNTERPARTY_LIMIT = 8


@dataclass
class BaselineAlert:
    """One alert. A legacy engine alerts on a *transaction*, not on a shape."""

    rule: str
    txn_ref: str
    account: str
    why: str


@dataclass
class BaselineResult:
    alerts: list[BaselineAlert] = field(default_factory=list)

    @property
    def flagged_refs(self) -> set[str]:
        return {alert.txn_ref for alert in self.alerts}

    def by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for alert in self.alerts:
            counts[alert.rule] += 1
        return dict(sorted(counts.items()))


def run_baseline(records: list[TransactionRecord]) -> BaselineResult:
    """Every rule against every record. Order is irrelevant -- there is no state to carry."""
    result = BaselineResult()

    for record in records:
        if record.amount >= CTR_THRESHOLD:
            result.alerts.append(BaselineAlert(
                "R1_ctr_threshold", record.txn_ref, record.sender_account,
                f"{record.amount} at or above the ${CTR_THRESHOLD:,.0f} CTR threshold",
            ))
        elif BAND_FLOOR <= record.amount < CTR_THRESHOLD:
            result.alerts.append(BaselineAlert(
                "R2_sub_threshold_amount", record.txn_ref, record.sender_account,
                f"{record.amount} inside the ${BAND_FLOOR:,.0f}-${CTR_THRESHOLD:,.0f} band",
            ))
        if record.amount >= RECORDKEEPING_THRESHOLD:
            result.alerts.append(BaselineAlert(
                "R5_recordkeeping_threshold", record.txn_ref, record.sender_account,
                f"{record.amount} at or above the ${RECORDKEEPING_THRESHOLD:,.0f} "
                "funds-transfer recordkeeping threshold",
            ))

    # R3: same-day totals per account, on either side of the transfer.
    daily: dict[tuple[str, str], list[TransactionRecord]] = defaultdict(list)
    for record in records:
        day = record.timestamp.date().isoformat()
        daily[(record.sender_account, day)].append(record)
        daily[(record.receiver_account, day)].append(record)
    for (account, day), group in sorted(daily.items()):
        total = sum((r.amount for r in group), Decimal(0))
        if total >= CTR_THRESHOLD and len(group) > 1:
            for record in group:
                result.alerts.append(BaselineAlert(
                    "R3_daily_aggregate", record.txn_ref, account,
                    f"{account} totalled {total} across {len(group)} transfers on {day}",
                ))

    # R4: distinct counterparties per account across the whole batch, with no window at all.
    counterparties: dict[str, set[str]] = defaultdict(set)
    touching: dict[str, list[TransactionRecord]] = defaultdict(list)
    for record in records:
        counterparties[record.receiver_account].add(record.sender_account)
        counterparties[record.sender_account].add(record.receiver_account)
        touching[record.receiver_account].append(record)
        touching[record.sender_account].append(record)
    for account, others in sorted(counterparties.items()):
        if len(others) >= COUNTERPARTY_LIMIT:
            for record in touching[account]:
                result.alerts.append(BaselineAlert(
                    "R4_counterparty_count", record.txn_ref, account,
                    f"{account} dealt with {len(others)} distinct counterparties in the batch",
                ))

    return result
