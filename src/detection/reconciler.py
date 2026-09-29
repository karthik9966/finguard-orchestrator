"""`CandidateReconciler` -- one shape per set of transactions (LLD §2.4).

**Load-bearing, not hygiene.** `fan_out` fires on the first leg of every `scatter_gather`, and the
two fans fire inside every v2 structure -- a layered funnel's branches, a gather-scatter's halves, a
bipartite block's senders. Left alone, the same transactions are reported twice: once under the shape that explains them and once
under a shape that only sees half of it. The model is then asked to ground both, and a reviewer
reads two findings about one event.

Precedence comes from `config.yaml`, strongest claim first, so the ordering is reviewable in a
diff rather than buried in an `if`. It is deliberately *not* confidence-ordered: a fan-out reading
half a scatter-gather can score higher than the scatter-gather, because half a pattern looks
tighter than the whole of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import get_config
from src.models import Candidate


@dataclass
class Reconciliation:
    """What survived, and what each survivor absorbed -- kept so the choice is auditable."""

    kept: list[Candidate] = field(default_factory=list)
    absorbed: dict[str, list[str]] = field(default_factory=dict)


class CandidateReconciler:
    """Drop a candidate whose transactions a stronger-claim pattern already explains."""

    # Above this share of overlap, two candidates are describing the same event. Below it they
    # genuinely share a few transactions -- a busy account can be the sink of one pattern and the
    # source of another, and both are real.
    OVERLAP = 0.6

    def __init__(self, precedence: list[str] | None = None) -> None:
        self.precedence = precedence or list(get_config().detection.precedence_order)

    def rank(self, candidate: Candidate) -> int:
        try:
            return self.precedence.index(candidate.pattern_type)
        except ValueError:  # pragma: no cover - config validation rejects this first
            return len(self.precedence)

    def reconcile(self, candidates: list[Candidate]) -> list[Candidate]:
        return self.explain(candidates).kept

    def explain(self, candidates: list[Candidate]) -> Reconciliation:
        """Reconcile, and record which candidate absorbed which."""
        result = Reconciliation()
        # Strongest claim first; within a rank, the larger pattern is the better explanation.
        ordered = sorted(
            candidates, key=lambda c: (self.rank(c), -len(c.member_txn_refs), c.candidate_id)
        )

        for candidate in ordered:
            members = set(candidate.member_txn_refs)
            absorber = next(
                (
                    kept
                    for kept in result.kept
                    if len(members & set(kept.member_txn_refs)) / len(members) > self.OVERLAP
                ),
                None,
            )
            if absorber is None:
                result.kept.append(candidate)
            else:
                result.absorbed.setdefault(absorber.candidate_id, []).append(
                    candidate.candidate_id
                )
        return result
