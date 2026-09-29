"""Cross-cutting observability (HLD §6). Langfuse, self-hosted, with redaction on the way out."""

from src.observability.tracing import (
    TRACE_TAG,
    audit_trace,
    handler,
    langfuse_client,
    mask,
    score_findings,
    tracing_target,
)

__all__ = [
    "TRACE_TAG",
    "audit_trace",
    "handler",
    "langfuse_client",
    "mask",
    "score_findings",
    "tracing_target",
]
