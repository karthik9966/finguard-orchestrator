"""Find the wires worth auditing, using four shape primitives (§4.2, AML Audit input).

All 17 of SAML-D's suspicious typologies collapse onto four geometries, so this module codes
four rules rather than seventeen:

======================  ====================================================================
``concentration``       an account *receives* a run of wires -- fan-in. Structuring,
                        Gather-Scatter, Layered_Fan_In, and the receiving half of Smurfing.
``dispersion``          an account *sends* a run of wires -- fan-out. Deposit-Send,
                        Scatter-Gather, Layered_Fan_Out.
``path``                money walks A -> B -> C. No account concentrates, so counting finds
                        nothing; the pattern is only visible by following the edges.
``magnitude``           the pattern *is* one wire. An implausible amount, not a repetition.
======================  ====================================================================

**A candidate names a shape, never a typology.** "Ten wires into one account over seven days,
all between GBP 5,526 and 5,984" is an observation this code can defend. "Structuring" is a
legal conclusion, and it has to come from a retrieved rule that says so -- otherwise the SAR
cites a regulation the agent picked to match a label Python had already decided on.

Recall is what these rules are tuned for, and precision is deliberately not. A missed cluster is
a regulatory failure; an extra clean cluster costs tokens and a few minutes of an analyst's
attention. Measured on all three batches, ``MIN_CLUSTER_WIRES = 3`` finds every planted cluster
at 23-39% precision.

Usage::

    uv run python -m src.utils.detectors data/processed/ledger/2023-06_private_banking_log.pdf
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from src.utils.swift_parser import Wire, existing_log, parse_batch

CONCENTRATION = "concentration"
DISPERSION = "dispersion"
PATH = "path"
MAGNITUDE = "magnitude"

# Three is the floor at which a run is a run rather than a coincidence, and it is also the
# smallest cluster pdf_generator.py plants. Raising it to 4 loses Layered_Fan_Out entirely.
MIN_CLUSTER_WIRES = 3

# A chain of two wires is just a payment being passed on. Three hops is where "money is moving
# through accounts" starts to describe it.
MIN_PATH_HOPS = 3
MAX_PATH_GAP_DAYS = 7
MAX_PATH_LENGTH = 25  # a guard against pathological graphs, never reached on a 220-wire batch
PATH_OVERLAP = 0.6  # share of wires above which a chain is a retelling of one already reported

# An outlier is measured against its own currency where there is enough of it to have a norm,
# because SAML-D quotes each wire in its payment currency and a JPY figure is not a GBP figure.
MAGNITUDE_MULTIPLE = 20
MIN_CURRENCY_SAMPLE = 5


@dataclass(frozen=True)
class Candidate:
    """A set of wires that share a geometry, plus the measurements that justify saying so."""

    shape: str
    anchor: str
    references: tuple[str, ...]
    first_date: date
    last_date: date
    currencies: tuple[str, ...]
    corridors: tuple[str, ...]
    total_amount: Decimal
    min_amount: Decimal
    max_amount: Decimal
    coefficient_of_variation: float
    distinct_counterparties: int
    counterparties: tuple[str, ...]
    # path only: what fraction of the opening amount survives to the closing wire. A ring that
    # loses a slice per hop looks very different from a chain of unrelated payments.
    retained_fraction: float | None = None

    @property
    def wire_count(self) -> int:
        return len(self.references)

    @property
    def span_days(self) -> int:
        return (self.last_date - self.first_date).days + 1

    @property
    def is_cross_border(self) -> bool:
        return any(corridor[:2] != corridor[-2:] for corridor in self.corridors)

    def summary(self) -> str:
        """Neutral description of the geometry. No typology, no legal conclusion."""
        money = (
            f"{self.min_amount:,.2f}"
            if self.min_amount == self.max_amount
            else f"{self.min_amount:,.2f} to {self.max_amount:,.2f}"
        )
        currencies = "/".join(self.currencies)
        window = (
            f"on {self.first_date}"
            if self.span_days == 1
            else f"over {self.span_days} days ({self.first_date} to {self.last_date})"
        )
        if self.shape == MAGNITUDE:
            return f"A single wire of {currencies} {money} {window}, from account {self.anchor}."
        if self.shape == PATH:
            retained = f", {self.retained_fraction:.0%} of the opening amount arriving at the end"
            return (
                f"{self.wire_count} wires forming a chain through {self.distinct_counterparties} "
                f"accounts {window}, {currencies} {money}{retained}."
            )
        direction = "into" if self.shape == CONCENTRATION else "out of"
        return (
            f"{self.wire_count} wires {direction} account {self.anchor} {window}, involving "
            f"{self.distinct_counterparties} counterparties, {currencies} {money}, "
            f"coefficient of variation {self.coefficient_of_variation:.3f}."
        )

    def as_row(self) -> dict:
        return {
            "shape": self.shape,
            "anchor": self.anchor,
            "wire_count": self.wire_count,
            "span_days": self.span_days,
            "first_date": self.first_date.isoformat(),
            "last_date": self.last_date.isoformat(),
            "total_amount": float(self.total_amount),
            "min_amount": float(self.min_amount),
            "max_amount": float(self.max_amount),
            "coefficient_of_variation": round(self.coefficient_of_variation, 4),
            "distinct_counterparties": self.distinct_counterparties,
            "currencies": list(self.currencies),
            "corridors": list(self.corridors),
            "is_cross_border": self.is_cross_border,
            "retained_fraction": self.retained_fraction,
            "references": list(self.references),
        }


# --- measurements ---------------------------------------------------------------------


def coefficient_of_variation(amounts: Sequence[Decimal]) -> float:
    """Standard deviation as a fraction of the mean -- spread on a scale you can compare.

    Ten wires averaging 5,673 with a deviation of 139 give 0.024: they are effectively the same
    payment made ten times. The measure is deliberately unitless so a GBP 5,000 cluster and a
    GBP 500,000 cluster are judged on shape rather than size.
    """
    if len(amounts) < 2:
        return 0.0
    values = [float(a) for a in amounts]
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _build(shape: str, anchor: str, wires: Sequence[Wire], counterparties: Sequence[str],
           retained_fraction: float | None = None) -> Candidate:
    amounts = [w.amount for w in wires]
    ordered = sorted(wires, key=lambda w: (w.value_date, w.reference))
    return Candidate(
        shape=shape,
        anchor=anchor,
        references=tuple(w.reference for w in ordered),
        first_date=ordered[0].value_date,
        last_date=ordered[-1].value_date,
        currencies=tuple(sorted({w.currency for w in wires})),
        corridors=tuple(sorted({w.corridor for w in wires})),
        total_amount=sum(amounts, Decimal(0)),
        min_amount=min(amounts),
        max_amount=max(amounts),
        coefficient_of_variation=coefficient_of_variation(amounts),
        distinct_counterparties=len(set(counterparties)),
        counterparties=tuple(sorted(set(counterparties))),
        retained_fraction=retained_fraction,
    )


# --- the four primitives --------------------------------------------------------------


def find_clusters(wires: Sequence[Wire], *, min_wires: int = MIN_CLUSTER_WIRES) -> list[Candidate]:
    """Accounts that collect or disperse a run of wires.

    Both directions are tested independently, because an account can be a collector and a
    distributor in the same month -- deposit money in, push it out -- and those are two
    different observations about it.
    """
    found: list[Candidate] = []
    sides = (
        (CONCENTRATION, lambda w: w.receiver_account, lambda w: w.sender_account),
        (DISPERSION, lambda w: w.sender_account, lambda w: w.receiver_account),
    )
    for shape, anchor_of, counterparty_of in sides:
        grouped: dict[str, list[Wire]] = defaultdict(list)
        for wire in wires:
            grouped[anchor_of(wire)].append(wire)
        for anchor, members in sorted(grouped.items()):
            if len(members) >= min_wires:
                found.append(
                    _build(shape, anchor, members, [counterparty_of(w) for w in members])
                )
    return found


def find_paths(
    wires: Sequence[Wire],
    *,
    min_hops: int = MIN_PATH_HOPS,
    max_gap_days: int = MAX_PATH_GAP_DAYS,
) -> list[Candidate]:
    """Follow the money: wires where each receiver is the next wire's sender.

    Counting cannot see this. A ring's edges span as many distinct senders as receivers, so no
    account stands out -- the shape only exists along the direction of travel. An account is
    never revisited as a sender, which both terminates the walk and stops a busy hub from
    generating a spurious path through itself.
    """
    outgoing: dict[str, list[Wire]] = defaultdict(list)
    for wire in wires:
        outgoing[wire.sender_account].append(wire)

    longest: list[list[Wire]] = []

    def walk(chain: list[Wire], visited: set[str]) -> None:
        last = chain[-1]
        extended = False
        if len(chain) < MAX_PATH_LENGTH:
            for nxt in outgoing.get(last.receiver_account, ()):
                gap = (nxt.value_date - last.value_date).days
                if 0 <= gap <= max_gap_days and nxt.receiver_account not in visited:
                    extended = True
                    walk(chain + [nxt], visited | {nxt.receiver_account})
        if not extended and len(chain) >= min_hops:
            longest.append(chain)

    for wire in wires:
        walk([wire], {wire.sender_account, wire.receiver_account})

    # One ring is reached from every edge along it, and every branch off it walks the same
    # opening hops before diverging -- so a single ten-hop chain surfaces as a dozen variants
    # sharing most of their wires. Containment alone does not remove them, because a branch is
    # not a subset. Keep the longest chain, then drop anything that mostly repeats one already
    # kept: an auditor needs the route, not every way of tracing part of it.
    kept: list[list[Wire]] = []
    for chain in sorted(longest, key=len, reverse=True):
        members = {w.reference for w in chain}
        if not any(
            len(members & {w.reference for w in seen}) / len(members) > PATH_OVERLAP
            for seen in kept
        ):
            kept.append(chain)

    found = []
    for chain in kept:
        opening, closing = float(chain[0].amount), float(chain[-1].amount)
        accounts = [chain[0].sender_account] + [w.receiver_account for w in chain]
        found.append(
            _build(
                PATH,
                chain[0].sender_account,
                chain,
                accounts,
                retained_fraction=closing / opening if opening else None,
            )
        )
    return found


def find_magnitude(
    wires: Sequence[Wire], *, multiple: float = MAGNITUDE_MULTIPLE
) -> list[Candidate]:
    """Wires far outside their currency's norm for the batch.

    The comparison is per-currency where the batch holds enough of that currency to establish
    one; below that it falls back to the batch median, which is safe here because SAML-D quotes
    every currency in a similar numeric band.
    """
    if not wires:
        return []
    by_currency: dict[str, list[float]] = defaultdict(list)
    for wire in wires:
        by_currency[wire.currency].append(float(wire.amount))
    overall = statistics.median(float(w.amount) for w in wires)
    norms = {
        currency: statistics.median(values) if len(values) >= MIN_CURRENCY_SAMPLE else overall
        for currency, values in by_currency.items()
    }
    return [
        _build(MAGNITUDE, wire.sender_account, [wire], [wire.receiver_account])
        for wire in sorted(wires, key=lambda w: -w.amount)
        if float(wire.amount) > multiple * norms[wire.currency]
    ]


def detect(wires: Sequence[Wire]) -> list[Candidate]:
    """Every candidate in a batch, most wires first."""
    candidates = find_clusters(wires) + find_paths(wires) + find_magnitude(wires)
    return sorted(candidates, key=lambda c: (-c.wire_count, c.shape, c.anchor))


def covered_references(candidates: Iterable[Candidate]) -> set[str]:
    return {reference for candidate in candidates for reference in candidate.references}


# --- CLI ------------------------------------------------------------------------------


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch", type=existing_log)
    parser.add_argument("--shape", choices=[CONCENTRATION, DISPERSION, PATH, MAGNITUDE])
    args = parser.parse_args()

    batch = parse_batch(args.batch, strict=True)
    candidates = [c for c in detect(batch.wires) if not args.shape or c.shape == args.shape]
    swept = covered_references(candidates)

    print(f"{batch.source.name}: {batch.parsed} wires -> {len(candidates)} candidates "
          f"covering {len(swept)} wires ({len(swept) / batch.parsed:.0%})\n")
    for candidate in candidates:
        print(f"[{candidate.shape}] {candidate.summary()}")
        print(f"    {', '.join(candidate.references[:8])}"
              f"{' ...' if candidate.wire_count > 8 else ''}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
