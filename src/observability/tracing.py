"""Langfuse tracing for the reasoning core -- HLD §6.

What tracing is for here is narrow and worth stating: an ordinary log says a finding was produced,
and a trace says *why* -- which clauses were in front of the model, what the critic scored the draft,
whether the loop re-ran and whether it hit its cap. That is the question an engineer actually arrives
with, and it is the one thing logs cannot answer after the fact.

**Redaction is the load-bearing part of this module, not an afterthought.** The pre-migration system
traced to a hosted LangSmith project and uploaded ~137 KB per run, including every parsed wire with
its counterparty names and account numbers -- most of which no model ever saw. That was measured,
not hypothetical. Two things fix it, and both are here:

* Langfuse is **self-hosted**, so traces stay in-environment (HLD §5).
* Every payload passes through `mask`, which is `redaction.redact`, *before* it leaves the process.
  The hook is the Langfuse client's own `mask` parameter, so it applies to inputs, outputs and
  metadata on every span the SDK emits -- including the ones the LangChain integration creates
  without being asked. Redacting at each call site instead would mean the one span somebody forgets
  is the one carrying an account number.

Tags follow HLD §6: run and batch id, the period audited, the record count and the client tier at run
level; pattern type, risk level, loop count and the final critique score per finding. References and
pattern metadata are kept deliberately -- a trace that cannot say *which* transactions a finding
covers is not much use -- while names, accounts and memo text are not.

Tracing is entirely optional. With no keys configured every function here is a no-op that costs an
attribute lookup, because an audit must not fail because an observability stack is down.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from src.config import get_settings
from src.utils.redaction import redact

log = logging.getLogger(__name__)

# Every trace carries it, so a Langfuse view can be scoped to audit runs and nothing else.
TRACE_TAG = "AML_AUDIT_RUN"

# Keys whose value is bulk rather than reasoning. The graph state carries the whole parsed ledger, so
# the LangChain integration would put all 500 records in the root span's input *and* its output --
# measured at ~370 KB for one clean batch. Redaction alone does not fix that: pseudonymised bulk is
# still bulk, no model ever saw the ledger in that form, and HLD §6 asks for references and pattern
# metadata rather than the transactions themselves. Replaced by a count, which is the only thing
# about a 500-record list a trace reader wants.
TRACE_SUMMARISE = frozenset({"records", "wires", "transactions"})

# Keys whose value is never sent in any form. The batch graph holds every record *and* a frame of
# them, so it is the ledger twice over; a trace learns nothing from it that `graph_build`'s span
# count does not already say.
TRACE_OMIT = frozenset({"batch_graph"})

# Any other list longer than this is truncated with a note. A cap rather than a drop, because the
# first few entries of an unexpected list are usually what makes a trace readable, and the point is
# to bound the payload rather than to hide it.
TRACE_LIST_LIMIT = 25

_CLIENT: Any = None
_CHECKED = False


def trim(value: Any) -> Any:
    """Bound the payload: summarise bulk, cap long lists, leave reasoning alone.

    Applied after redaction, because the two are different problems. Redaction decides what may
    leave the process; this decides how much of it is worth sending. A trace exists to answer *why a
    finding was reached*, and the parsed ledger is not part of that answer.
    """
    if isinstance(value, dict):
        trimmed: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in TRACE_OMIT:
                trimmed[str(key)] = "[batch graph -- omitted from the trace]"
            elif str(key).lower() in TRACE_SUMMARISE and isinstance(item, (list, tuple)):
                trimmed[str(key)] = f"[{len(item)} record(s) -- omitted from the trace]"
            else:
                trimmed[str(key)] = trim(item)
        return trimmed
    if isinstance(value, (list, tuple)):
        kept = [trim(item) for item in value[:TRACE_LIST_LIMIT]]
        if len(value) > TRACE_LIST_LIMIT:
            kept.append(f"... {len(value) - TRACE_LIST_LIMIT} more omitted")
        return kept
    return value


def mask(*, data: Any) -> Any:
    """Langfuse's hook: every payload the SDK is about to send passes through here.

    Two passes, in this order. `redact` decides what may leave the process at all -- accounts
    pseudonymised, memos scrubbed, names gone. `trim` then decides how much is worth sending, because
    pseudonymised bulk is still bulk and the pre-migration defect was as much about volume as about
    identity.

    Keyword-only `data` because that is how Langfuse calls it. Failure is swallowed deliberately: a
    mask that raises would either drop the span or, worse in some SDK versions, send the unmasked
    original. Returning a placeholder is the only safe failure.
    """
    try:
        return trim(redact(data))
    except Exception as error:  # noqa: BLE001 - never let a masking failure emit raw data
        log.warning("redaction failed for a trace payload; dropping it: %s", error)
        return "[REDACTION FAILED -- PAYLOAD DROPPED]"


def langfuse_client() -> Any:
    """The configured client, or None when tracing is off.

    Built once and cached. `auth_check` is deliberately *not* called: it is a network round trip, and
    a failed one at import time would make every CLI invocation wait on an observability stack.
    """
    global _CLIENT, _CHECKED
    if _CHECKED:
        return _CLIENT
    _CHECKED = True

    settings = get_settings()
    if not settings.tracing_enabled:
        log.debug("Langfuse tracing is off (no host/keys configured)")
        return None

    from langfuse import Langfuse

    _CLIENT = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        # The privacy hook. Set on the client, so it covers every span the SDK emits rather than the
        # ones a caller remembered to sanitise.
        mask=mask,
        environment=settings.langfuse_environment or None,
    )
    log.info("Langfuse tracing on: %s", settings.langfuse_host)
    return _CLIENT


def reset() -> None:
    """Drop the cached client. For tests, which reconfigure between cases."""
    global _CLIENT, _CHECKED
    _CLIENT = None
    _CHECKED = False


def tracing_target() -> str | None:
    """The host traces are going to, or None when tracing is off.

    Reported rather than assumed. A trace that is silently not being written is worse than none,
    because you go looking for it after the run instead of before.
    """
    return get_settings().langfuse_host if langfuse_client() is not None else None


def handler() -> Any:
    """The LangChain callback handler, or None. Registered run-level, so every nested node, retrieval
    and model call becomes a child span without being registered individually.

    The public key is passed explicitly. Without it the handler resolves its client implicitly, and
    when more than one client exists in the process the SDK refuses -- "skipping tracing for this
    function to avoid cross-project leakage" -- so tracing silently produces nothing. Being explicit
    costs a line and removes a failure whose symptom is an empty Langfuse project.
    """
    client = langfuse_client()
    if client is None:
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler(public_key=get_settings().langfuse_public_key)


def run_metadata(
    *, run_id: str, batch_id: str, period: str, records: int, candidates: int | None = None
) -> dict[str, Any]:
    """HLD §6's run-level tags, in the shape the LangChain integration reads them.

    `langfuse_session_id` is the run id, which is what makes a run's spans one thing in the UI and
    joins them to the job row and the stored report -- all three are the same id by construction.
    """
    settings = get_settings()
    metadata: dict[str, Any] = {
        "langfuse_session_id": run_id,
        "langfuse_trace_name": f"audit {period}",
        "langfuse_tags": [TRACE_TAG, f"period:{period}", f"tier:{settings.client_tier}"],
        "run_id": run_id,
        "batch_id": batch_id,
        "period": period,
        "record_count": records,
        "client_tier": settings.client_tier,
    }
    if candidates is not None:
        metadata["candidate_count"] = candidates
    return metadata


@contextmanager
def audit_trace(*, run_id: str, batch_id: str, period: str, records: int):
    """One run, as one trace. Yields the trace id, or None when tracing is off.

    A root span is opened explicitly rather than letting the callback handler create the trace on
    its own, for one reason: the per-finding scores are emitted *after* the graph returns, and they
    need a trace id to attach to. Reading it from inside a node would depend on OpenTelemetry context
    surviving however LangGraph happens to schedule that node, which is not a thing to bet a
    privacy-sensitive audit trail on.
    """
    client = langfuse_client()
    if client is None:
        yield None
        return

    try:
        with client.start_as_current_observation(
            name=f"audit {period}",
            as_type="agent",
            input={"batch_id": batch_id, "period": period, "records": records},
            metadata=run_metadata(
                run_id=run_id, batch_id=batch_id, period=period, records=records
            ),
        ) as span:
            yield client.get_current_trace_id()
    except Exception as error:  # noqa: BLE001 - an audit never fails because tracing did
        log.warning("Langfuse tracing failed for run %s: %s", run_id, error)
        yield None


def score_findings(trace_id: str | None, findings: list[Any], *, run_id: str) -> None:
    """HLD §6's per-finding tags: pattern type, risk level, loop count, final critique score.

    One score per finding rather than one per run, so a run with four confident findings and one the
    review could not stand behind does not average into a single reassuring number. The pattern and
    risk travel as score metadata, which is what makes "show me every low-confidence structuring
    finding" a filter rather than a grep.
    """
    client = langfuse_client()
    if client is None or trace_id is None or not findings:
        return

    try:
        for finding in findings:
            client.create_score(
                name="critique",
                value=float(finding.confidence),
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=(
                    f"{finding.candidate.pattern_type} · {finding.risk_level} risk · "
                    f"{finding.status}"
                ),
                metadata={
                    "finding_id": finding.finding_id,
                    "pattern_type": finding.candidate.pattern_type,
                    "risk_level": finding.risk_level,
                    "status": finding.status,
                    # The loop count a finding cost is not on the Finding -- it is per candidate and
                    # the critic resets it -- so it is derived from what the notes recorded.
                    "refinements": sum(
                        1 for note in finding.review_notes if "refinement" in note.lower()
                    ),
                    "transactions": len(finding.candidate.member_txn_refs),
                    "run_id": run_id,
                },
            )
        client.flush()
    except Exception as error:  # noqa: BLE001
        log.warning("could not record finding scores for run %s: %s", run_id, error)
