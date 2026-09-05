"""The audit engine as an HTTP service (§10).

An audit takes 30-60 seconds and costs $0.06-$0.18. Both facts shape the interface: ``POST
/audit`` accepts the batch, returns an id immediately and works in the background, and the
caller polls ``GET /audit/{id}``. A synchronous endpoint that holds a connection open for a
minute is not a design -- it is a timeout waiting for a proxy to find it.

§10 specifies ``graph.ainvoke``, and this uses it. The honest caveat: **the graph's nodes are
synchronous**, so ``ainvoke`` hands them to a threadpool rather than yielding on I/O. That is
correct and it does not block the event loop, but it is not the same thing as async nodes, and
describing it as such would be false. Making the nodes genuinely async would mean async ChromaDB
and async model calls throughout, which buys nothing at this concurrency.

Run with::

    uv run uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.graph.cost import UsageLedger
from src.graph.graph import build_graph, run_config
from src.graph.state import ComplianceReport, initial_state
from src.ingestion.store import COLLECTION_NAME, stats
from src.utils.swift_parser import parse_batch

app = FastAPI(
    title="FinGuard Orchestrator",
    description="Agentic AML compliance audit engine (ADGM / FINRA corpus).",
    version="0.1.0",
)

# In-process, because the alternative is a database this project does not otherwise need. It is
# the right size for one service instance and the wrong size for two -- a second worker would not
# see the first one's audits. Redis or Postgres is the fix if this is ever scaled out, and the
# limitation is stated rather than hidden behind an interface that pretends otherwise.
AUDITS: dict[str, dict[str, Any]] = {}

Status = Literal["running", "complete", "failed"]


class AuditAccepted(BaseModel):
    audit_id: str
    status: Status
    batch: str
    wires: int = Field(description="Wires parsed during upload validation, before the audit ran")
    poll: str


class AuditResult(BaseModel):
    audit_id: str
    status: Status
    batch: str
    submitted_at: str
    report: ComplianceReport | None = None
    error: str | None = None
    confidence_score: float | None = None
    loop_count: int | None = None
    cost_usd: float | None = None
    model_calls: int | None = None


def _run(audit_id: str, batch_path: Path, auditor_query: str) -> None:
    """Execute one audit and record the outcome. Never raises: a failed audit is a result."""
    record = AUDITS[audit_id]
    try:
        config = run_config(str(batch_path), tags=["API"])
        # The audit_id is minted by run_config and reused as the resource id, so a trace in
        # LangSmith and a row in this dict are the same run rather than two id schemes to join.
        config["metadata"]["audit_id"] = audit_id
        ledger = UsageLedger()
        config["callbacks"] = [ledger]

        state: dict[str, Any] = dict(initial_state(str(batch_path)))
        if auditor_query.strip():
            state["auditor_query"] = auditor_query.strip()

        result = build_graph().invoke(state, config=config)
        total = ledger.total_cost
        record.update(
            status="complete",
            report=result.get("report"),
            confidence_score=result.get("confidence_score"),
            loop_count=result.get("loop_count"),
            cost_usd=float(total) if total is not None else None,
            model_calls=ledger.calls,
        )
    except Exception as error:  # noqa: BLE001 - reported to the caller, not swallowed
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        batch_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, Any]:
    """Ready means the corpus is actually queryable, not merely that the process is up.

    A 200 from a service whose vector store is empty would send every audit into a retrieval that
    returns nothing -- which `audit_node` now refuses outright. Better to fail here.
    """
    try:
        payload = stats()
    except Exception as error:  # noqa: BLE001
        raise HTTPException(503, f"vector store unavailable: {type(error).__name__}") from error

    if not payload["vectors"]:
        raise HTTPException(503, f"collection {COLLECTION_NAME!r} is empty")
    return {
        "status": "ok",
        "collection": payload["collection"],
        "vectors": payload["vectors"],
        "backend": payload["backend"],
        "audits_held": len(AUDITS),
    }


@app.post("/audit", response_model=AuditAccepted, status_code=202)
async def submit_audit(
    background: BackgroundTasks, batch: UploadFile, auditor_query: str = ""
) -> AuditAccepted:
    """Accept a batch, validate it synchronously, then audit in the background."""
    suffix = Path(batch.filename or "batch.txt").suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(415, f"expected a .pdf or .txt MT103 log, got {suffix or 'no suffix'}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(await batch.read())
        path = Path(handle.name)

    # Parsed before accepting, so a bad upload is a 400 in a second rather than a background task
    # that fails a minute later for a reason the caller has to poll to discover.
    try:
        parsed = await asyncio.to_thread(parse_batch, path, strict=False)
    except Exception as error:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"unreadable MT103 batch: {error}") from error

    if not parsed.wires:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "no wires could be parsed from this file")

    config = run_config(str(path), tags=["API"])
    audit_id = config["metadata"]["audit_id"]
    AUDITS[audit_id] = {
        "audit_id": audit_id,
        "status": "running",
        "batch": batch.filename or path.name,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    background.add_task(_run, audit_id, path, auditor_query)

    return AuditAccepted(
        audit_id=audit_id,
        status="running",
        batch=AUDITS[audit_id]["batch"],
        wires=parsed.parsed,
        poll=f"/audit/{audit_id}",
    )


@app.get("/audit/{audit_id}", response_model=AuditResult)
def read_audit(audit_id: str) -> AuditResult:
    record = AUDITS.get(audit_id)
    if record is None:
        raise HTTPException(404, f"no audit {audit_id!r}")
    return AuditResult(**record)


@app.get("/audits")
def list_audits() -> list[dict[str, Any]]:
    """Everything this process has run, newest first."""
    return sorted(
        ({k: v for k, v in record.items() if k != "report"} for record in AUDITS.values()),
        key=lambda record: record["submitted_at"],
        reverse=True,
    )
