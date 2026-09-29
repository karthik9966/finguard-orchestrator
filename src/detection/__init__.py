"""Pattern detection (LLD v2 §2.5-2.7, PRD v2 §1).

Nine named typologies over one shared batch graph, each with a time window. This replaced the four geometric primitives of
`utils/detectors.py`, which Phase 5 deleted once the reasoning core stopped calling it.

Three things differ from those primitives, and each is a recorded defect of theirs:

* **A window.** `find_clusters` applies none, so an account with fifteen counterparties spread
  over a month scores exactly like fifteen in an afternoon. On the 10,000-message batch the old
  primitives return 932 candidates.
* **Named patterns, not geometry.** The old module emitted `concentration`/`dispersion` and left
  naming to retrieval. The nine PRD typologies are what `pattern_to_obligations` is keyed on, so
  a candidate has to carry one to be grounded at all.
* **Reconciliation.** `fan_out` fires on the first leg of every `scatter_gather`, and the fans
  fire inside every layered, bipartite and gather-scatter shape. Without an explicit precedence the
  same transactions are reported twice under different names.
"""

from src.detection.base import BaseDetector, DETECTORS, detect_all, register
from src.detection.graph_engine import BatchGraph, build_graph
from src.detection.reconciler import CandidateReconciler

__all__ = [
    "BaseDetector",
    "BatchGraph",
    "CandidateReconciler",
    "DETECTORS",
    "build_graph",
    "detect_all",
    "register",
]
