"""The results store -- LLD §3.2's three tables, §5.1's step 8.

    reports(report_id PK, run_id, period, generated_at, risk_rating, clean, report_json, schema_version)
    findings(finding_id PK, report_id FK, candidate_id, pattern_type, risk_level, confidence, status)
    reviews(review_id PK, finding_id FK, action, reviewer, timestamp, note)   -- append-only
    indexes: reports(period), findings(report_id), reviews(finding_id)

**The load-bearing decision here is what "immutable" means when review changes something.**

`reports.report_json` is written once and never updated. That is not tidiness: it is the record of
what the engine actually produced on a given day from a given corpus, and a regulator asking "what
did your system conclude in June" must get an answer that later human review cannot have edited.

But `findings.status` *does* change -- that is the entire point of the review loop -- and a status
denormalised inside `report_json` would immediately disagree with the `findings` row beside it.
So the two are kept apart and **`get()` returns a join, not the stored JSON**: the frozen report
re-hydrated with each finding's current status and review notes. `stored()` returns the original
verbatim for anyone who needs to see what was filed rather than where it stands now.

Getting that wrong is discovered in the UI as findings that will not change status, or as an audit
trail that silently rewrites history, so it is settled here.

The reviews table is append-only. A status is a *derived* value -- the action of the latest review --
and the history of how a finding got there is the part an auditor cares about.

There is a fourth table the LLD does not list: **`jobs`**. §3.2 specifies the *results* tables and
§5.1 step 1 says only "enqueue background graph run" -- but step 9 then has the client come back for
`GET /audits/{job_id}`, and a queue that lives in a process dict cannot answer that after a restart.
It is also where batch-hash dedup has to look, because the retry worth protecting against is a
client re-posting while the first run is *still going*. Recorded as a deliberate deviation.

SQLAlchemy Core rather than the ORM: there are four tables, no relationships worth mapping, and the
URL is the point -- `RESULTS_DB_URL` is `sqlite:///./results.db` in v1 and a Postgres URL in
production (LLD §8), which is a URL change and nothing else.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from src.config import get_config, get_settings
from src.models import ComplianceReport, Finding, FindingStatus

log = logging.getLogger(__name__)

RESULTS_STORE_WRITE_FAILURE = "RESULTS_STORE_WRITE_FAILURE"

# What a human can do to a finding. The engine sets `pending_review` or `needs_review`; these three
# are the only transitions a person makes, and they are LLD §5.1 step 9's own words: "analyst
# clear/escalate -> officer approve".
ReviewAction = Literal["clear", "escalate", "approve"]

RESULT_OF: dict[str, FindingStatus] = {
    "clear": "cleared",
    "escalate": "escalated",
    "approve": "approved",
}

# Who may do what, and from where. Approval is reachable only from `escalated` because approving is
# signing off on a filing -- there is nothing to approve about a finding an analyst already cleared,
# and allowing it would make "approved" mean two different things in the same column.
PERMITTED: dict[str, set[str]] = {
    "pending_review": {"clear", "escalate"},
    "needs_review": {"clear", "escalate"},
    "escalated": {"approve", "clear"},
    "cleared": set(),
    "approved": set(),
}


class ResultsStoreUnavailable(RuntimeError):
    """The store could not be written after retrying (LLD §6 RESULTS_STORE_WRITE_FAILURE).

    Raised *with the report still held* by the store, so the caller can re-save rather than re-run:
    by step 8 the expensive part is already paid for, and a run thrown away because the disk was
    busy for 300ms is the most costly possible response to a transient fault.
    """


class InvalidTransition(ValueError):
    """A review action that the finding's current status does not permit."""


@runtime_checkable
class ResultsStore(Protocol):
    """Where a finished report goes, and where the review loop reads it back from."""

    def save(self, report: ComplianceReport) -> None: ...

    def get(self, report_id: str) -> ComplianceReport | None: ...


# --- the in-memory default ------------------------------------------------------------------


@dataclass
class InMemoryResultsStore:
    """Sufficient for a test and honest about being nothing more: a process restart loses it.

    Kept after Phase 6a rather than replaced, because a suite that needs a database to assert
    orchestration is a suite that stops being run.
    """

    reports: dict[str, ComplianceReport] = field(default_factory=dict)

    def save(self, report: ComplianceReport) -> None:
        self.reports[report.report_id] = report

    def get(self, report_id: str) -> ComplianceReport | None:
        return self.reports.get(report_id)

    def stored(self, report_id: str) -> ComplianceReport | None:
        return self.reports.get(report_id)

    def latest(self) -> ComplianceReport | None:
        return next(reversed(self.reports.values()), None) if self.reports else None


# --- schema ---------------------------------------------------------------------------------

METADATA = MetaData()

REPORTS = Table(
    "reports", METADATA,
    Column("report_id", String(128), primary_key=True),
    Column("run_id", String(128), nullable=False),
    Column("period", String(16), nullable=False),
    Column("generated_at", String(64), nullable=False),
    Column("risk_rating", String(16), nullable=False),
    Column("clean", Boolean, nullable=False),
    # The frozen deliverable. Written once, never updated -- see the module docstring.
    Column("report_json", Text, nullable=False),
    Column("schema_version", String(16), nullable=False),
    Index("ix_reports_period", "period"),
)

FINDINGS = Table(
    "findings", METADATA,
    Column("finding_id", String(128), primary_key=True),
    Column("report_id", String(128), nullable=False),
    Column("candidate_id", String(256), nullable=False),
    Column("pattern_type", String(32), nullable=False),
    Column("risk_level", String(16), nullable=False),
    Column("confidence", Float, nullable=False),
    # The one mutable column in the schema, and the reason `get()` is a join.
    Column("status", String(32), nullable=False),
    # Position in the report, so a join can rebuild the findings list in the order it was filed.
    Column("ordinal", Integer, nullable=False),
    Index("ix_findings_report_id", "report_id"),
)

REVIEWS = Table(
    "reviews", METADATA,
    Column("review_id", String(128), primary_key=True),
    Column("finding_id", String(128), nullable=False),
    Column("action", String(32), nullable=False),
    Column("reviewer", String(128), nullable=False),
    Column("timestamp", String(64), nullable=False),
    Column("note", Text, nullable=False, default=""),
    Index("ix_reviews_finding_id", "finding_id"),
)


# Not in LLD §3.2 -- see the module docstring. A job is the *request*; a report is the result, and
# the two have different lifetimes: a job can be running or failed with no report at all.
JOBS = Table(
    "jobs", METADATA,
    Column("job_id", String(128), primary_key=True),
    Column("batch_name", String(512), nullable=False),
    # sha256 of the uploaded bytes. Indexed because every POST asks "have we already run this?"
    Column("batch_sha256", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    Column("submitted_at", String(64), nullable=False),
    Column("finished_at", String(64), nullable=True),
    Column("report_id", String(128), nullable=True),
    Column("error", Text, nullable=True),
    Index("ix_jobs_batch_sha256", "batch_sha256"),
    Index("ix_jobs_status", "status"),
)

JobStatus = Literal["running", "complete", "failed"]

# A retry re-runs a batch that failed -- the failure may well have been the vector store being
# briefly down -- but never one that is running or already done. That asymmetry is the whole point
# of the dedup: it exists to stop a client's impatient retry from paying for a second audit.
DEDUPLICATED: set[str] = {"running", "complete"}


@dataclass
class Job:
    """One submitted batch and what became of it."""

    job_id: str
    batch_name: str
    batch_sha256: str
    status: JobStatus
    submitted_at: datetime
    finished_at: datetime | None = None
    report_id: str | None = None
    error: str | None = None


@dataclass
class Review:
    """One entry in a finding's history."""

    review_id: str
    finding_id: str
    action: str
    reviewer: str
    timestamp: datetime
    note: str = ""


# --- the SQL store --------------------------------------------------------------------------


class SqlResultsStore:
    """SQLite in v1, Postgres in production -- the URL decides and nothing else changes."""

    def __init__(self, url: str | None = None, *, engine: Engine | None = None) -> None:
        self.url = url or get_settings().results_db_url
        self._engine = engine or self._build_engine(self.url)
        METADATA.create_all(self._engine)
        # LLD §6: the report is *held* when a write fails, so a caller can re-save. This is the
        # holding place, and `flush()` is the retry.
        self.unsaved: dict[str, ComplianceReport] = {}

    @staticmethod
    def _build_engine(url: str) -> Engine:
        """Create the engine, making the parent directory for a file-backed SQLite URL.

        SQLite will not create a missing directory and fails with `unable to open database file`,
        which says nothing about the actual problem. `RESULTS_DB_URL=sqlite:///./data/results.db`
        is an obvious thing to set, so it should just work.
        """
        if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
            target = Path(url[len("sqlite:///"):])
            target.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread is SQLite-only; FastAPI runs the background task on a threadpool thread
        # while the request handler that reads it back is on another.
        arguments = {"check_same_thread": False} if url.startswith("sqlite") else {}
        return create_engine(url, future=True, connect_args=arguments)

    # --- plumbing -------------------------------------------------------------------------

    def _write(self, what: str, call):
        """One write, retried with exponential backoff, then surfaced as a typed failure."""
        persistence = get_config().persistence
        last: Exception | None = None
        for attempt in range(persistence.write_attempts):
            try:
                return call()
            except Exception as error:  # noqa: BLE001 - re-raised below as a typed failure
                last = error
                log.warning("%s: %s attempt %d failed: %s",
                            RESULTS_STORE_WRITE_FAILURE, what, attempt + 1, error)
                if attempt + 1 < persistence.write_attempts:
                    time.sleep(persistence.write_backoff_seconds * (2**attempt))
        raise ResultsStoreUnavailable(
            f"{RESULTS_STORE_WRITE_FAILURE}: {what} failed after {persistence.write_attempts} "
            f"attempts against {self.url}: {last}"
        ) from last

    # --- writing --------------------------------------------------------------------------

    def save(self, report: ComplianceReport) -> None:
        """Persist one report and its findings in a single transaction.

        Atomic by necessity, not by preference: a `reports` row without its `findings` rows would
        read back as a report whose findings had all been reviewed away, which is a different claim
        from the one the engine made.

        Re-saving the same report_id replaces its findings rows but leaves `report_json` as first
        written. That makes `save` idempotent for the retry path -- the caller that re-saves after a
        `ResultsStoreUnavailable` must not end up with two half-written copies -- while keeping the
        frozen deliverable frozen.
        """
        payload = report.model_dump(mode="json")
        try:
            self._write("save", lambda: self._save(report, payload))
        except ResultsStoreUnavailable:
            # Held, not lost. By step 8 the run is already paid for.
            self.unsaved[report.report_id] = report
            raise
        self.unsaved.pop(report.report_id, None)

    def _save(self, report: ComplianceReport, payload: dict[str, Any]) -> None:
        with self._engine.begin() as connection:
            existing = connection.execute(
                select(REPORTS.c.report_id).where(REPORTS.c.report_id == report.report_id)
            ).first()
            if existing is None:
                connection.execute(insert(REPORTS).values(
                    report_id=report.report_id,
                    run_id=report.run_id,
                    period=report.period,
                    generated_at=report.generated_at.isoformat(),
                    risk_rating=report.risk_rating,
                    clean=report.clean,
                    report_json=json.dumps(payload),
                    schema_version=report.schema_version,
                ))
            # Findings are rewritten rather than merged: on the retry path they have never been
            # reviewed (the report was never readable), so there is no status to preserve.
            connection.execute(delete(FINDINGS).where(FINDINGS.c.report_id == report.report_id))
            if report.findings:
                connection.execute(insert(FINDINGS), [
                    {
                        "finding_id": finding.finding_id,
                        "report_id": report.report_id,
                        "candidate_id": finding.candidate.candidate_id,
                        "pattern_type": finding.candidate.pattern_type,
                        "risk_level": finding.risk_level,
                        "confidence": finding.confidence,
                        "status": finding.status,
                        "ordinal": ordinal,
                    }
                    for ordinal, finding in enumerate(report.findings)
                ])

    def flush(self) -> list[str]:
        """Re-save everything held from a failed write. Returns the report ids that went through."""
        saved: list[str] = []
        for report in list(self.unsaved.values()):
            self.save(report)
            saved.append(report.report_id)
        return saved

    # --- the review loop ------------------------------------------------------------------

    def review(
        self, finding_id: str, action: ReviewAction, *, reviewer: str, note: str = ""
    ) -> FindingStatus:
        """Append a review and move the finding's status. Returns the new status.

        The append is what matters. A status column alone answers "where is this now" and nothing
        about how it got there -- and "who cleared this, when, and what did they say" is the question
        an examiner actually asks.
        """
        current = self.status_of(finding_id)
        if current is None:
            raise KeyError(f"no finding {finding_id!r}")
        if action not in PERMITTED.get(current, set()):
            raise InvalidTransition(
                f"{finding_id}: cannot {action} a finding that is {current!r} "
                f"(permitted: {sorted(PERMITTED.get(current, set())) or 'nothing -- it is final'})"
            )

        status = RESULT_OF[action]
        entry = Review(
            review_id=f"rev-{uuid.uuid4().hex[:12]}",
            finding_id=finding_id,
            action=action,
            reviewer=reviewer,
            timestamp=datetime.now(timezone.utc),
            note=note,
        )

        def apply() -> None:
            with self._engine.begin() as connection:
                connection.execute(insert(REVIEWS).values(
                    review_id=entry.review_id,
                    finding_id=entry.finding_id,
                    action=entry.action,
                    reviewer=entry.reviewer,
                    timestamp=entry.timestamp.isoformat(),
                    note=entry.note,
                ))
                connection.execute(
                    FINDINGS.update().where(FINDINGS.c.finding_id == finding_id).values(
                        status=status
                    )
                )

        self._write(f"review {action}", apply)
        log.info("%s %s by %s -> %s", finding_id, action, reviewer, status)
        return status

    def reviews_for(self, finding_id: str) -> list[Review]:
        """Oldest first: this is a history, and a history read backwards is a different story.

        Ordered by the ISO timestamp, which is safe precisely because it is ISO: the format is
        fixed-width down to the microsecond, so a lexical sort of the column is chronological. Two
        reviews of the same finding in the same microsecond fall back to the review_id, which is
        arbitrary -- and unreachable, since a human is on the other end of each one.
        """
        with self._engine.connect() as connection:
            rows = connection.execute(
                select(REVIEWS).where(REVIEWS.c.finding_id == finding_id)
                .order_by(REVIEWS.c.timestamp, REVIEWS.c.review_id)
            ).mappings().all()
        return [
            Review(
                review_id=row["review_id"],
                finding_id=row["finding_id"],
                action=row["action"],
                reviewer=row["reviewer"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                note=row["note"] or "",
            )
            for row in rows
        ]

    def status_of(self, finding_id: str) -> FindingStatus | None:
        with self._engine.connect() as connection:
            row = connection.execute(
                select(FINDINGS.c.status).where(FINDINGS.c.finding_id == finding_id)
            ).first()
        return row[0] if row else None

    def statuses_for(self, report_id: str) -> dict[str, FindingStatus]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                select(FINDINGS.c.finding_id, FINDINGS.c.status)
                .where(FINDINGS.c.report_id == report_id)
            ).all()
        return {finding_id: status for finding_id, status in rows}

    # --- jobs -----------------------------------------------------------------------------

    def create_job(self, job_id: str, *, batch_name: str, batch_sha256: str) -> Job:
        job = Job(
            job_id=job_id,
            batch_name=batch_name,
            batch_sha256=batch_sha256,
            status="running",
            submitted_at=datetime.now(timezone.utc),
        )
        self._write("create_job", lambda: self._insert_job(job))
        return job

    def _insert_job(self, job: Job) -> None:
        with self._engine.begin() as connection:
            connection.execute(insert(JOBS).values(
                job_id=job.job_id,
                batch_name=job.batch_name,
                batch_sha256=job.batch_sha256,
                status=job.status,
                submitted_at=job.submitted_at.isoformat(),
            ))

    def finish_job(self, job_id: str, *, report_id: str) -> None:
        self._set_job(job_id, status="complete", report_id=report_id)

    def fail_job(self, job_id: str, *, error: str) -> None:
        self._set_job(job_id, status="failed", error=error)

    def _set_job(self, job_id: str, **values: Any) -> None:
        values["finished_at"] = datetime.now(timezone.utc).isoformat()

        def apply() -> None:
            with self._engine.begin() as connection:
                connection.execute(
                    JOBS.update().where(JOBS.c.job_id == job_id).values(**values)
                )

        self._write("update_job", apply)

    def get_job(self, job_id: str) -> Job | None:
        with self._engine.connect() as connection:
            row = connection.execute(select(JOBS).where(JOBS.c.job_id == job_id)).mappings().first()
        return self._job(row) if row else None

    def job_for_batch(self, batch_sha256: str) -> Job | None:
        """The most recent job for these exact bytes that a second POST should reuse.

        Only `running` or `complete`: a failed batch is worth re-running, and one that is mid-flight
        or finished is exactly what the dedup exists to protect.
        """
        with self._engine.connect() as connection:
            row = connection.execute(
                select(JOBS)
                .where(JOBS.c.batch_sha256 == batch_sha256)
                .where(JOBS.c.status.in_(sorted(DEDUPLICATED)))
                .order_by(JOBS.c.submitted_at.desc())
            ).mappings().first()
        return self._job(row) if row else None

    def list_jobs(self, *, limit: int = 50) -> list[Job]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                select(JOBS).order_by(JOBS.c.submitted_at.desc()).limit(limit)
            ).mappings().all()
        return [self._job(row) for row in rows]

    @staticmethod
    def _job(row) -> Job:
        return Job(
            job_id=row["job_id"],
            batch_name=row["batch_name"],
            batch_sha256=row["batch_sha256"],
            status=row["status"],
            submitted_at=datetime.fromisoformat(row["submitted_at"]),
            finished_at=(
                datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
            ),
            report_id=row["report_id"],
            error=row["error"],
        )

    # --- reading --------------------------------------------------------------------------

    def stored(self, report_id: str) -> ComplianceReport | None:
        """Exactly what was filed, with no review applied.

        The answer to "what did the engine conclude", as distinct from "where does this stand".
        """
        payload = self._payload(report_id)
        return None if payload is None else ComplianceReport(**payload)

    def get(self, report_id: str) -> ComplianceReport | None:
        """The frozen report joined to its findings' current statuses.

        This is what a UI and an API render. Reading `report_json` alone would show every finding
        as `pending_review` forever, however much review had happened -- which is the bug this join
        exists to prevent.
        """
        payload = self._payload(report_id)
        if payload is None:
            return None
        statuses = self.statuses_for(report_id)
        for finding in payload.get("findings", []):
            status = statuses.get(finding["finding_id"])
            if status is None or status == finding["status"]:
                continue
            finding["status"] = status
            history = self.reviews_for(finding["finding_id"])
            notes = list(finding.get("review_notes") or [])
            notes += [
                f"{entry.action} by {entry.reviewer} at "
                f"{entry.timestamp.isoformat(timespec='seconds')}"
                + (f": {entry.note}" if entry.note else "")
                for entry in history
            ]
            # `needs_review` requires a note by contract; a reviewed finding keeps its engine notes
            # plus the human trail, so that holds either way.
            finding["review_notes"] = notes
        return ComplianceReport(**payload)

    def _payload(self, report_id: str) -> dict[str, Any] | None:
        with self._engine.connect() as connection:
            row = connection.execute(
                select(REPORTS.c.report_json).where(REPORTS.c.report_id == report_id)
            ).first()
        return json.loads(row[0]) if row else None

    def list_reports(self, *, period: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """The listing an API and a UI page from: header rows, never report bodies.

        A listing of ten audits should not carry ten full narratives, and `period` is indexed
        precisely because "show me June" is the question that gets asked.
        """
        query = select(
            REPORTS.c.report_id, REPORTS.c.run_id, REPORTS.c.period, REPORTS.c.generated_at,
            REPORTS.c.risk_rating, REPORTS.c.clean, REPORTS.c.schema_version,
        ).order_by(REPORTS.c.generated_at.desc()).limit(limit)
        if period is not None:
            query = query.where(REPORTS.c.period == period)

        with self._engine.connect() as connection:
            rows = connection.execute(query).mappings().all()

        listing = []
        for row in rows:
            statuses = self.statuses_for(row["report_id"])
            listing.append({
                **dict(row),
                "findings": len(statuses),
                "needs_review": sum(1 for s in statuses.values() if s == "needs_review"),
            })
        return listing

    def findings_for(self, report_id: str) -> list[Finding]:
        """The findings as they now stand, in the order they were filed."""
        report = self.get(report_id)
        return list(report.findings) if report else []

    def counts(self) -> dict[str, int]:
        """Row counts per table -- what `/health` reports and what a smoke test asserts."""
        with self._engine.connect() as connection:
            return {
                table.name: connection.execute(
                    select(func.count()).select_from(table)
                ).scalar_one()
                for table in (REPORTS, FINDINGS, REVIEWS, JOBS)
            }


def default_store() -> ResultsStore:
    """The store the CLI and the API use: SQLite at `RESULTS_DB_URL` (LLD §8)."""
    return SqlResultsStore()
