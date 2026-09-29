"""The audit engine as an HTTP service (LLD §10).

An audit runs per candidate and costs accordingly, so the interface is asynchronous by design:
``POST /audit`` accepts the batch, validates it synchronously, returns an id immediately and works
in the background; the caller polls ``GET /audit/{id}``. An endpoint that holds a connection open
for the length of a run is not a design -- it is a timeout waiting for a proxy to find it.

Every run enters through ``graph.run.audit_batch``, the same path the CLI takes, so there is one
execution path rather than two that drift.

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

from src.graph.graph import new_run_id
from src.graph.run import BatchUnreadable, InMemoryResultsStore, audit_batch
from src.ingestion.store import RULE_COLLECTION, VectorStoreClient
from src.models import ComplianceReport
from src.utils.swift_parser import parse_batch

app = FastAPI(
    title="FinGuard Orchestrator",
    description="Agentic AML compliance audit engine (US BSA/AML corpus).",
    version="0.1.0",
)

# In-process, because the alternative is a database this project does not otherwise need. It is
# the right size for one service instance and the wrong size for two -- a second worker would not
# see the first one's audits. Redis or Postgres is the fix if this is ever scaled out, and the
# limitation is stated rather than hidden behind an interface that pretends otherwise.
AUDITS: dict[str, dict[str, Any]] = {}

# LLD §5.1 step 8. In-memory for now, and swapped for SQLite in Phase 6a with no change here --
# which is the whole reason `ResultsStore` is a protocol rather than a concrete class.
REPORTS = InMemoryResultsStore()

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
    candidates: int | None = None
    findings: int | None = None
    needs_review: int | None = None
    cost_usd: float | None = None
    cost_per_candidate_usd: float | None = None
    model_calls: int | None = None


def _run(audit_id: str, batch_path: Path) -> None:
    """Execute one audit and record the outcome. Never raises: a failed audit is a result.

    The audit_id is passed down as the run_id, so a LangSmith trace, a row in this registry and a
    stored report are the same run rather than three id schemes to join.
    """
    record = AUDITS[audit_id]
    try:
        result = audit_batch(batch_path, run_id=audit_id, store=REPORTS, tags=["API"])
        total = result.usage.total_cost
        record.update(
            status="complete",
            report=result.report,
            candidates=result.candidates,
            findings=len(result.report.findings),
            needs_review=result.report.needs_review_count,
            cost_usd=float(total) if total is not None else None,
            cost_per_candidate_usd=result.cost_per_candidate,
            model_calls=result.usage.calls,
        )
    except BatchUnreadable as error:
        # LLD §6's one loud failure, reported as what it is: the client's file, not our fault.
        record.update(status="failed", error=str(error))
    except Exception as error:  # noqa: BLE001 - reported to the caller, not swallowed
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        batch_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, Any]:
    """Ready means the corpus is actually queryable, not merely that the process is up.

    Probed against `rule_chunks`, the collection the retriever really reads. A 200 from a service
    whose obligations do not resolve would send every candidate to `needs_review` one paid call at
    a time, which is the expensive way to discover an empty collection.
    """
    try:
        payload = counts()
    except Exception as error:  # noqa: BLE001
        raise HTTPException(503, f"vector store unavailable: {type(error).__name__}") from error

    if not payload["total"]:
        raise HTTPException(503, f"collection {RULE_COLLECTION!r} is empty")
    return {
        "status": "ok",
        "collection": RULE_COLLECTION,
        "vectors": payload["total"],
        "by_tier": payload["tier"],
        "by_authority": payload["authority"],
        "audits_held": len(AUDITS),
    }


def counts() -> dict[str, Any]:
    """Indirected so the health probe can be stubbed without a vector store."""
    return VectorStoreClient(RULE_COLLECTION).counts()


@app.post("/audit", response_model=AuditAccepted, status_code=202)
async def submit_audit(background: BackgroundTasks, batch: UploadFile) -> AuditAccepted:
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

    audit_id = new_run_id()
    AUDITS[audit_id] = {
        "audit_id": audit_id,
        "status": "running",
        "batch": batch.filename or path.name,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    background.add_task(_run, audit_id, path)

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
