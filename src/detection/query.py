"""`QueryConstructor` -- candidate geometry to one retrieval question (LLD §2.4).

**Deterministic templates, not model output.** Phase 1 measured that phrasing decides retrieval: for
the same facts, the correct clause ranked 11,268th of 12,273 as raw detector JSON, 315th as a
narrative of what happened, and **5th** as an obligation-shaped question. Templates also make the
phrasing reproducible, which a generated query cannot be -- a retrieval regression is then
attributable to the corpus rather than to variance in a question nobody logged.

**Two registers, because there are two kinds of target.** Phase 8 measured the indicator search at
hit@1 of 0.22 with the correct clause absent from the top 5 entirely, and the suspect is that the
obligation-shaped template was being used for it. Asking *"obligation to report transactions
deliberately structured to stay below the reporting threshold"* returns prose describing what a CTR
*is*, and never reaches *"Currency is deposited or withdrawn in amounts just below identification or
reporting thresholds"* -- the actual red flag.

The reason is that the two corpora are written differently:

* **Obligations** are duties. *"A bank shall file a report of any suspicious transaction..."* An
  obligation-shaped question shares that register, which is what Phase 1 measured.
* **Indicators** are descriptions of behaviour. FFIEC Appendix F reads *"Customer makes multiple and
  frequent currency deposits to various accounts that are purportedly unrelated."* Nothing in it is
  phrased as a duty, so a duty-shaped question is the wrong register for every one of them.

Phase 1's finding did not transfer because the architecture changed underneath it: obligations were
discovered by search then and come from a curated map now, so the only thing still *searched* is the
illustrative pool -- which is the half the duty shape does not fit. `indicator_query` is what the
retriever uses; `obligation_query` is kept because it is still the right shape for its own target,
and because the measurement that produced it is worth not losing.

**One query per candidate**, per LLD §2.4. The pre-migration system issued 2-4 and fused them with
RRF, which beat every alternative merge -- but that win was on semantic discovery of obligations,
which the curated map replaced.
"""

from __future__ import annotations

from src.config import PatternType
from src.models import Candidate

# The duty a pattern implicates. Not used by the retriever -- obligations come from the curated map --
# and kept because it is the right shape for searching obligations, which a later phase may need.
OBLIGATION_TEMPLATES: dict[PatternType, str] = {
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

# The *behaviour* a pattern looks like, written in the register the red flags are written in. Each one
# is deliberately close to how FFIEC Appendix F or FINRA 19-18 phrases the same observation --
# "customer deposits", "funds are transferred", "beneficiaries receive" -- because that is the text
# these have to match against.
INDICATOR_TEMPLATES: dict[PatternType, str] = {
    "structuring": (
        "customer deposits or withdraws currency in amounts just below the identification or "
        "reporting threshold, or structures deposits through multiple branches or several people"
    ),
    "fan_in": (
        "customer makes multiple and frequent deposits from apparently unrelated parties into one "
        "account, funds collected and funnelled from many accounts to a small number of "
        "beneficiaries"
    ),
    "fan_out": (
        "an unusually large number and variety of beneficiaries receive funds transfers from one "
        "company or account shortly after funds arrive"
    ),
    "cycle": (
        "funds are transferred through a series of accounts and returned to the originating account "
        "with no apparent business purpose"
    ),
    "scatter_gather": (
        "customer deposits funds into several accounts in small amounts which are subsequently "
        "consolidated into one account and transferred out"
    ),
}

# Appended when a candidate's own measurements say the clause set should widen. Each register gets its
# own wording for the same widening, for the same reason the templates do.
CROSS_BORDER_OBLIGATION = "enhanced due diligence obligations for cross-border wire transfers"
CROSS_BORDER_INDICATOR = (
    "funds are transferred to or from a higher-risk jurisdiction with no apparent business reason"
)
CASH_INTENSIVE_OBLIGATION = (
    "red flags for cash-intensive businesses and currency transaction reporting"
)
CASH_INTENSIVE_INDICATOR = (
    "cash-intensive business making currency deposits inconsistent with its stated trade"
)


class QueryConstructor:
    """One question per candidate, in the register of whatever it is being asked of."""

    def indicator_query(self, candidate: Candidate) -> str:
        """The Tier-2 search. Behaviour-shaped, because red flags describe behaviour."""
        template = INDICATOR_TEMPLATES.get(candidate.pattern_type)
        if template is None:  # pragma: no cover - PatternType is a closed Literal
            raise KeyError(f"no indicator template for {candidate.pattern_type!r}")

        parts = [template]
        attributes = candidate.attributes or {}
        threshold = attributes.get("threshold")
        if threshold:
            # Named as an amount rather than as a filing rule: the indicators talk about amounts.
            parts.append(f"amounts of just under ${threshold:,} in currency or transfers")
        if attributes.get("cross_border"):
            parts.append(CROSS_BORDER_INDICATOR)
        if attributes.get("cash_intensive"):
            parts.append(CASH_INTENSIVE_INDICATOR)
        return "; ".join(parts)

    def obligation_query(self, candidate: Candidate) -> str:
        """The duty-shaped question. Nothing searches obligations today -- they come from the curated
        map -- so this is unused at runtime and kept for the register it documents."""
        template = OBLIGATION_TEMPLATES.get(candidate.pattern_type)
        if template is None:  # pragma: no cover - PatternType is a closed Literal
            raise KeyError(f"no obligation template for {candidate.pattern_type!r}")

        parts = [template]
        attributes = candidate.attributes or {}
        threshold = attributes.get("threshold")
        if threshold:
            parts.append(
                f"currency transaction report filing requirement for amounts near ${threshold:,}"
            )
        if attributes.get("cross_border"):
            parts.append(CROSS_BORDER_OBLIGATION)
        if attributes.get("cash_intensive"):
            parts.append(CASH_INTENSIVE_OBLIGATION)
        return "; ".join(parts)

    def key(self, candidate: Candidate) -> PatternType:
        """The `pattern_to_obligations` key -- what Tier 1 is fetched by."""
        return candidate.pattern_type
