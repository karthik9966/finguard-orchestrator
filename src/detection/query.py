"""`QueryConstructor` -- candidate geometry to one retrieval question (LLD §2.4).

**Deterministic templates, not model output.** Phase 1 measured that phrasing decides retrieval:
for the same facts, the correct clause ranked 11,268th of 12,273 as raw detector JSON, 315th as a
narrative of what happened, and **5th** as an obligation-shaped question. Rulebooks are written as
duties -- *"a Relevant Person must..."* -- so a description of events shares no register with
them, and a model asked to describe a candidate produces a description.

Templates also make the phrasing reproducible, which a generated query cannot be: a retrieval
regression is then attributable to the corpus rather than to variance in a question nobody logged.

**One query per candidate**, per LLD §2.4. The pre-migration system issued 2-4 and fused them with
RRF, which beat every alternative merge -- but that win was on *semantic discovery of
obligations*, and obligations now come from a curated map. What is left for search is a small
`authority: illustrative` pool that a cross-encoder reranks anyway.
"""

from __future__ import annotations

from src.config import PatternType
from src.models import Candidate

# Each template names the *duty* a pattern implicates, not the pattern. "Fan-in" is our word; the
# FFIEC's is "multiple unrelated parties funding a single account", and the corpus is written in
# the FFIEC's.
TEMPLATES: dict[PatternType, str] = {
    "structuring": (
        "obligation to report transactions deliberately structured to stay below the currency "
        "transaction reporting threshold"
    ),
    "fan_in": (
        "duty to monitor an account receiving repeated deposits from multiple unrelated sources "
        "in a short period"
    ),
    "fan_out": (
        "duty to scrutinise funds dispersed from one account to many beneficiaries shortly after "
        "being received"
    ),
    "cycle": (
        "obligation to identify funds moved through a chain of accounts and returned to their "
        "origin to obscure their source"
    ),
    "scatter_gather": (
        "requirement to report funds split across intermediaries and recombined into a single "
        "account"
    ),
}

# Appended when a candidate's own measurements say the clause set should widen. Each is a duty in
# its own right, not a description of the transactions.
CROSS_BORDER = "enhanced due diligence obligations for cross-border wire transfers"
CASH_INTENSIVE = "red flags for cash-intensive businesses and currency transaction reporting"


class QueryConstructor:
    """One obligation-shaped question per candidate."""

    def build(self, candidate: Candidate) -> str:
        template = TEMPLATES.get(candidate.pattern_type)
        if template is None:  # pragma: no cover - PatternType is a closed Literal
            raise KeyError(f"no query template for {candidate.pattern_type!r}")

        parts = [template]
        attributes = candidate.attributes or {}

        # A threshold in the attributes means the detector matched an amount band, so the
        # reporting rule for that band is worth asking about by name.
        threshold = attributes.get("threshold")
        if threshold:
            parts.append(
                f"currency transaction report filing requirement for amounts near ${threshold:,}"
            )
        if attributes.get("cross_border"):
            parts.append(CROSS_BORDER)
        if attributes.get("cash_intensive"):
            parts.append(CASH_INTENSIVE)

        return "; ".join(parts)

    def key(self, candidate: Candidate) -> PatternType:
        """The `pattern_to_obligations` key -- what Tier 1 is fetched by."""
        return candidate.pattern_type
