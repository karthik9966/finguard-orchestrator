"""The detector interface and its registry (LLD §2.4).

A registry rather than a hardcoded list so `precedence_order` in config.yaml is the single place
that knows about all five typologies. A detector added without an entry there fails at startup
rather than being silently reconciled last.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import PatternType, get_config
from src.detection.graph_builder import BatchGraph, build_graph
from src.models import Candidate, TransactionRecord

DETECTORS: dict[str, "BaseDetector"] = {}


class BaseDetector(ABC):
    """One typology. Emits geometry and measurements; never names an offence."""

    pattern_type: PatternType

    @abstractmethod
    def detect(self, batch: BatchGraph) -> list[Candidate]:
        """Candidates of this detector's pattern type, or an empty list."""

    # --- helpers shared by the implementations ------------------------------------------

    @property
    def window_days(self) -> int:
        return get_config().detection.window_days

    def candidate(
        self,
        *,
        anchor: str,
        refs: list[str],
        confidence: float,
        **attributes,
    ) -> Candidate:
        """Build a Candidate with a stable id.

        `Candidate.make_id` hashes the sorted refs, so a re-run of the same batch produces the
        same ids and two reports can be diffed.
        """
        return Candidate(
            candidate_id=Candidate.make_id(self.pattern_type, anchor, refs),
            pattern_type=self.pattern_type,
            member_txn_refs=refs,
            detection_confidence=confidence,
            attributes={"anchor": anchor, **attributes},
        )


def register(detector: BaseDetector) -> BaseDetector:
    DETECTORS[detector.pattern_type] = detector
    return detector


def detect_all(records: list[TransactionRecord], *, reconcile: bool = True) -> list[Candidate]:
    """Run every registered detector over one batch, then reconcile.

    Reconciliation is on by default because it is not hygiene: `fan_out` fires on the first leg
    of every `scatter_gather`, so without it the same transactions are reported twice under
    different names and the model is asked to ground both.
    """
    from src.detection import structuring as _structuring  # noqa: F401  (registers)
    from src.detection import fan_in as _fan_in  # noqa: F401
    from src.detection import fan_out as _fan_out  # noqa: F401
    from src.detection import cycle as _cycle  # noqa: F401
    from src.detection import scatter_gather as _scatter_gather  # noqa: F401
    from src.detection.reconciler import CandidateReconciler

    if not records:
        return []

    batch = build_graph(records)
    found: list[Candidate] = []
    for pattern in get_config().detection.precedence_order:
        detector = DETECTORS.get(pattern)
        if detector is None:
            raise RuntimeError(f"precedence_order names {pattern!r}, which no detector registers")
        found.extend(detector.detect(batch))

    return CandidateReconciler().reconcile(found) if reconcile else found
