"""The audit engine as an HTTP service (LLD §10, HLD §2.2's three journeys).

An audit runs per candidate and costs accordingly, so the interface is asynchronous by design:
`POST /audits` validates the batch synchronously, returns a `job_id` immediately, and the caller
polls `GET /audits/{job_id}`. An endpoint that holds a connection open for the length of a run is
not a design; it is a timeout waiting for a proxy to find it.

**HLD §2.2's Journey 2 wants the report returned in the response and LLD §5.1 wants 202 + poll.**
The LLD wins -- the poll is what survives a proxy and a 10,000-message batch -- and `?wait=true` is
Journey 2's synchronous variant, with a bounded timeout that degrades to the job_id rather than
hanging. A system-to-system caller gets one round trip when the run is short enough and a poll when
it is not.

Three things here are deliberately not the obvious thing:

**Runs are serialised through a single worker.** Not for correctness -- the graph holds no shared
state -- but because two concurrent audits contend for one vector store and one rate limit, and the
failure mode is both runs getting slower and one of them hitting a 429. One queue, one worker, and
the queue depth is visible on `/health`.

**Job state lives in the database, not in a dict.** The dict this replaced could not answer
`GET /audits/{id}` after a restart, which is precisely what §5.1 step 9 asks for.

**The same batch posted twice does not run twice.** The retry worth protecting against is a client
re-posting because the first response was slow, and an audit is the expensive thing in this system.
Dedup is on the sha256 of the uploaded bytes; `?force=true` is the escape hatch for re-auditing the
same file deliberately, which is a real thing to want after the corpus is rebuilt.

Run with::

    uv run uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graph.graph import new_run_id
from src.graph.run import BatchUnreadable, audit_batch
from src.ingestion.store import RULE_COLLECTION, VectorStoreClient
from src.models import ComplianceReport
from src.store import InvalidTransition, ResultsStoreUnavailable, SqlResultsStore
from src.utils.swift_parser import parse_batch

log = logging.getLogger(__name__)

# How long `?wait=true` will hold a connection before giving up and handing back the job_id. A dev
# batch finishes inside this; the 10,000-message batch does not, and pretending otherwise would
# produce a gateway timeout instead of a usable answer.
WAIT_TIMEOUT_SECONDS = 300.0


@dataclass
class Queued:
    """One unit of work for the worker: the job, its temp file, and who is waiting on it."""

    job_id: str
    path: Path
    done: asyncio.Event


# --- the store, built on first use --------------------------------------------------------
#
# Constructing it creates the database file, and a module that writes to disk merely by being
# imported leaves a results.db beside every test run and every `--help`.
_STORE: SqlResultsStore | None = None


def reports() -> SqlResultsStore:
    global _STORE
    if _STORE is None:
        _STORE = SqlResultsStore()
    return _STORE


# --- the single worker --------------------------------------------------------------------
#
# The queue and the waiter events live on `app.state`, created by the lifespan, because an
# asyncio primitive binds to the first event loop that touches it. At module scope they would
# outlive the loop they were bound to -- which is not hypothetical: it made every job in the suite
# sit at `running` forever, because the worker awaiting the queue was on a loop that had closed.


def _execute(job_id: str, path: Path) -> None:
    """One audit, start to finish, on a worker thread. Never raises: a failed audit is a result."""
    store = reports()
    try:
        result = audit_batch(path, run_id=job_id, store=store, tags=["API"])
        store.finish_job(job_id, report_id=result.report.report_id)
        log.info("job %s complete: %s", job_id, result.report.report_id)
    except BatchUnreadable as error:
        # LLD §6's one loud failure, reported as what it is: the client's file, not our fault.
        store.fail_job(job_id, error=str(error))
    except ResultsStoreUnavailable as error:
        # The run succeeded and the write did not. The report is held by the store for re-save, so
        # say so -- this is recoverable without re-billing the audit.
        store.fail_job(job_id, error=f"{error} (the report is held for re-save)")
    except Exception as error:  # noqa: BLE001 - reported to the caller, not swallowed
        store.fail_job(job_id, error=f"{type(error).__name__}: {error}")
    finally:
        path.unlink(missing_ok=True)


async def _worker(queue: asyncio.Queue[Queued], waiters: dict[str, asyncio.Event]) -> None:
    """Drain the queue one job at a time, for as long as the app is up."""
    while True:
        item = await queue.get()
        try:
            await asyncio.to_thread(_execute, item.job_id, item.path)
        finally:
            item.done.set()
            waiters.pop(item.job_id, None)
            queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.queue = asyncio.Queue()
    app.state.waiters = {}
    if not get_settings().api_auth_token:
        # Said once, loudly, at startup. The alternative -- discovering it as a 503 per request --
        # sends an operator looking for a credential problem when the problem is their config.
        log.error(
            "API_AUTH_TOKEN is not set: every endpoint except /health will refuse with 503. "
            "This service audits financial records; it does not serve them unauthenticated."
        )
    worker = asyncio.create_task(
        _worker(app.state.queue, app.state.waiters), name="finguard-audit-worker"
    )
    try:
        yield
    finally:
        worker.cancel()


app = FastAPI(
    title="FinGuard Orchestrator",
    description="Agentic AML compliance audit engine (US BSA/AML corpus).",
    version="0.2.0",
    lifespan=lifespan,
)


# --- authentication -----------------------------------------------------------------------


def authenticate(request: Request) -> None:
    """Bearer token on everything but `/health` (LLD §6 AUTH_FAILURE).

    **Unconfigured is closed, not open.** A missing token means the service cannot authenticate
    anyone, which is a server state rather than a client mistake -- hence 503 -- and it is the only
    safe default for a service that reads financial records.

    Compared with `compare_digest` rather than `==`: an early-exit comparison leaks how much of a
    guessed token was right, one byte at a time.
    """
    expected = get_settings().api_auth_token
    if not expected:
        raise HTTPException(503, "the service has no API_AUTH_TOKEN configured")

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(presented.strip(), expected):
        raise HTTPException(
            401, "a valid bearer token is required", headers={"WWW-Authenticate": "Bearer"}
        )


PROTECTED = [Depends(authenticate)]


# --- schemas ------------------------------------------------------------------------------

Status = Literal["running", "complete", "failed"]


class AuditAccepted(BaseModel):
    job_id: str
    status: Status
    batch: str
    transactions: int = Field(description="Parsed during upload validation, before the audit ran")
    poll: str
    # True when these exact bytes were already submitted, so nothing was re-run and nothing
    # re-billed. The caller is looking at the first submission's job.
    deduplicated: bool = False


class AuditResult(BaseModel):
    job_id: str
    status: Status
    batch: str
    submitted_at: str
    finished_at: str | None = None
    report: ComplianceReport | None = None
    error: str | None = None


class ReviewRequest(BaseModel):
    action: Literal["clear", "escalate", "approve"]
    reviewer: str = Field(min_length=1)
    note: str = ""


# --- health -------------------------------------------------------------------------------


def counts() -> dict[str, Any]:
    """Indirected so the health probe can be stubbed without a vector store."""
    return VectorStoreClient(RULE_COLLECTION).counts()


def _queue_depth() -> int:
    queue = getattr(app.state, "queue", None)
    return queue.qsize() if queue is not None else 0


@app.get("/health")
def health() -> dict[str, Any]:
    """Ready means the corpus is genuinely queryable, not merely that the process is up.

    Deliberately unauthenticated: a load balancer cannot carry a bearer token, and this returns
    counts rather than any report content. A 200 from a service whose obligations do not resolve
    would send every candidate to needs_review one paid call at a time, which is the expensive way
    to discover an empty collection.
    """
    try:
        corpus = counts()
    except Exception as error:  # noqa: BLE001
        raise HTTPException(503, f"vector store unavailable: {type(error).__name__}") from error

    if not corpus["total"]:
        raise HTTPException(503, f"collection {RULE_COLLECTION!r} is empty")
    return {
        "status": "ok",
        "collection": RULE_COLLECTION,
        "vectors": corpus["total"],
        "by_tier": corpus["tier"],
        "by_authority": corpus["authority"],
        "reports_stored": reports().counts()["reports"],
        "queue_depth": _queue_depth(),
        "authenticated": bool(get_settings().api_auth_token),
    }


# --- Journeys 1 and 2: submitting a batch -------------------------------------------------


@app.post(
    "/audits",
    # Two shapes, because there are two journeys. Journey 1 gets the 202 and polls; Journey 2 asks
    # for `wait=true` and gets the finished result in the same response.
    response_model=AuditAccepted | AuditResult,
    status_code=202,
    dependencies=PROTECTED,
)
async def submit_audit(
    request: Request, response: Response, batch: UploadFile, wait: bool = False,
    force: bool = False, timeout: float = WAIT_TIMEOUT_SECONDS,
) -> AuditAccepted | AuditResult:
    """Accept a batch, validate it synchronously, then audit it on the single worker.

    `wait=true` is Journey 2's synchronous variant: it holds the connection until the run finishes
    and returns the report inline, or gives up at `timeout` and hands back the job_id to poll. It
    does not get its own execution path -- the same queue, the same worker -- because two ways to
    run an audit is two ways for an audit to behave.
    """
    suffix = Path(batch.filename or "batch.txt").suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(415, f"expected a .pdf or .txt MT103 log, got {suffix or 'no suffix'}")

    payload = await batch.read()
    digest = hashlib.sha256(payload).hexdigest()
    name = batch.filename or f"batch{suffix}"

    if not force:
        existing = reports().job_for_batch(digest)
        if existing is not None:
            # Nothing runs and nothing is billed. The caller is pointed at the first submission.
            log.info("job %s reused for a re-post of %s", existing.job_id, name)
            accepted = AuditAccepted(
                job_id=existing.job_id, status=existing.status, batch=existing.batch_name,
                transactions=0, poll=f"/audits/{existing.job_id}", deduplicated=True,
            )
            return await _maybe_wait(request, response, accepted, wait=wait, timeout=timeout)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        path = Path(handle.name)

    # Parsed before accepting, so a bad upload is a 400 in a second rather than a queued job that
    # fails a minute later for a reason the caller has to poll to discover.
    try:
        parsed = await asyncio.to_thread(parse_batch, path, strict=False)
    except Exception as error:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"unreadable MT103 batch: {error}") from error

    if not parsed.wires:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "no transactions could be parsed from this file")

    # The job_id *is* the run_id, so a trace, a job row and a report id are one run rather than
    # three id schemes to join.
    job_id = new_run_id()
    reports().create_job(job_id, batch_name=name, batch_sha256=digest)
    done = asyncio.Event()
    request.app.state.waiters[job_id] = done
    await request.app.state.queue.put(Queued(job_id=job_id, path=path, done=done))

    accepted = AuditAccepted(
        job_id=job_id, status="running", batch=name, transactions=parsed.parsed,
        poll=f"/audits/{job_id}",
    )
    return await _maybe_wait(request, response, accepted, wait=wait, timeout=timeout)


async def _maybe_wait(
    request: Request, response: Response, accepted: AuditAccepted, *, wait: bool, timeout: float
) -> AuditAccepted | AuditResult:
    """Journey 2's one round trip, or the 202 that Journey 1 polls.

    The status code follows the answer rather than the endpoint: a 202 means "accepted, come back",
    and handing over a finished report under a 202 would tell a correct client to keep polling
    something that is already done.
    """
    if not wait:
        return accepted

    done = request.app.state.waiters.get(accepted.job_id)
    if done is not None and accepted.status == "running":
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Not an error: the run is still going and the client has an id to poll. Saying so beats
            # holding the connection until a proxy kills it.
            log.info("job %s still running after %.0fs; returning the id to poll",
                     accepted.job_id, timeout)
            return accepted

    finished = read_audit(accepted.job_id)
    if finished.status == "running":
        # A dedup hit on a job this process is not the waiter for -- another worker has it. There is
        # nothing to await, so hand back the id.
        return accepted
    response.status_code = 200
    return finished


# --- reading a run ------------------------------------------------------------------------


@app.get("/audits", dependencies=PROTECTED)
def list_audits(limit: int = 50) -> list[dict[str, Any]]:
    """Every job this service has run, newest first. Survives a restart."""
    return [
        {
            "job_id": job.job_id,
            "status": job.status,
            "batch": job.batch_name,
            "submitted_at": job.submitted_at.isoformat(timespec="seconds"),
            "finished_at": job.finished_at.isoformat(timespec="seconds") if job.finished_at else None,
            "report_id": job.report_id,
            "error": job.error,
        }
        for job in reports().list_jobs(limit=limit)
    ]


@app.get("/audits/{job_id}", response_model=AuditResult, dependencies=PROTECTED)
def read_audit(job_id: str) -> AuditResult:
    """A job's status, and its report once there is one (LLD §5.1 step 9)."""
    job = reports().get_job(job_id)
    if job is None:
        raise HTTPException(404, f"no audit {job_id!r}")
    return AuditResult(
        job_id=job.job_id,
        status=job.status,
        batch=job.batch_name,
        submitted_at=job.submitted_at.isoformat(timespec="seconds"),
        finished_at=job.finished_at.isoformat(timespec="seconds") if job.finished_at else None,
        report=reports().get(job.report_id) if job.report_id else None,
        error=job.error,
    )


# --- Journey 3: audit-defence lookup ------------------------------------------------------


@app.get("/reports", dependencies=PROTECTED)
def list_reports(period: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Stored reports, newest first, optionally one period ("by month", HLD Journey 3)."""
    return reports().list_reports(period=period, limit=limit)


@app.get("/reports/{report_id}", response_model=ComplianceReport, dependencies=PROTECTED)
def read_report(report_id: str) -> ComplianceReport:
    """The frozen report joined to its findings' current review statuses.

    Not `report_json` as stored: that would show every finding as pending_review forever, however
    much review had happened. `/reports/{id}/filed` is the verbatim original.
    """
    report = reports().get(report_id)
    if report is None:
        raise HTTPException(404, f"no report {report_id!r}")
    return report


@app.get("/reports/{report_id}/filed", response_model=ComplianceReport, dependencies=PROTECTED)
def read_filed_report(report_id: str) -> ComplianceReport:
    """What the engine concluded, with no human review applied -- "the rule as cited at the time"."""
    report = reports().stored(report_id)
    if report is None:
        raise HTTPException(404, f"no report {report_id!r}")
    return report


@app.post("/findings/{finding_id}/review", dependencies=PROTECTED)
def review_finding(finding_id: str, request: ReviewRequest) -> dict[str, Any]:
    """Record one review decision (LLD §5.1 step 9).

    Append-only: the review is added to the finding's history and its status moves. `report_json` is
    not touched, which is what makes the filed report still be the filed report afterwards.
    """
    try:
        status = reports().review(
            finding_id, request.action, reviewer=request.reviewer, note=request.note
        )
    except KeyError as error:
        raise HTTPException(404, f"no finding {finding_id!r}") from error
    except InvalidTransition as error:
        raise HTTPException(409, str(error)) from error

    return {
        "finding_id": finding_id,
        "status": status,
        "history": [
            {
                "action": entry.action,
                "reviewer": entry.reviewer,
                "timestamp": entry.timestamp.isoformat(timespec="seconds"),
                "note": entry.note,
            }
            for entry in reports().reviews_for(finding_id)
        ],
    }
