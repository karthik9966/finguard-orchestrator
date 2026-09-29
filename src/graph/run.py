"""The run orchestrator -- LLD §5.1 steps 1-3 and 8, the parts that sit outside the graph.

    paths → ingest → initial_state → graph.invoke → store.save → ComplianceReport
            (step 2)   (step 3)      (steps 4-7)     (step 8)

Parsing is outside the graph on purpose. A file that yields no readable transaction is a client
error, and it should be reported as one before a run id is minted, a vector store is opened or a
node is entered. Persistence is outside for the mirror reason: a report is the run's product, and
where it is stored is not a decision the reasoning core should be able to see.

`ResultsStore` is therefore a Protocol with an in-memory default. Phase 6a puts SQLite behind it
with no change to any caller here, and the seam gets exercised by tests now rather than designed
blind later.

Both the CLI and the API enter through `audit_batch`, so there is one execution path and not two
that drift.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.config import get_config
from src.graph.cost import UsageLedger
from src.graph.graph import build_graph, new_run_id, run_config, tracing_project
from src.ingestion.batch import TransactionBatchIngestor
from src.models import ComplianceReport, TransactionRecord, ValidationReport, initial_state

log = logging.getLogger(__name__)

INGEST_FILE_UNREADABLE = "INGEST_FILE_UNREADABLE"


class BatchUnreadable(ValueError):
    """LLD §6's one loud failure. Every other error in a run is per-candidate and survivable;
    a batch nothing could be read from has no partial result worth reporting, and pretending
    otherwise would produce a clean report for a file that never parsed."""


@runtime_checkable
class ResultsStore(Protocol):
    """Where a finished report goes (LLD §5.1 step 8).

    A Protocol rather than a base class: the API's registry, Phase 6a's SQLite table and the
    in-memory default below have nothing in common to inherit, and a `save`/`get` pair is the whole
    contract the orchestrator depends on.
    """

    def save(self, report: ComplianceReport) -> None: ...

    def get(self, report_id: str) -> ComplianceReport | None: ...


@dataclass
class InMemoryResultsStore:
    """The default. Sufficient for a CLI run, and honest about being nothing more: a process
    restart loses it, which is exactly what Phase 6a's SQLite store exists to fix."""

    reports: dict[str, ComplianceReport] = field(default_factory=dict)

    def save(self, report: ComplianceReport) -> None:
        self.reports[report.report_id] = report

    def get(self, report_id: str) -> ComplianceReport | None:
        return self.reports.get(report_id)

    def latest(self) -> ComplianceReport | None:
        return next(reversed(self.reports.values()), None) if self.reports else None


@dataclass
class RunResult:
    """One run, and everything a caller needs to report on it without reaching into graph state."""

    report: ComplianceReport
    validation: ValidationReport
    usage: UsageLedger
    run_id: str
    candidates: int
    records: int

    @property
    def cost_per_candidate(self) -> float | None:
        """Recorded because Phase 8 sets any candidate cap against real numbers, and a cap chosen
        without them would be a guess dressed as a threshold."""
        total = self.usage.total_cost
        if total is None or not self.candidates:
            return None
        return float(total) / self.candidates


def period_of(paths: list[Path], records: list[TransactionRecord]) -> str:
    """The audited month, YYYY-MM.

    Read from the records rather than from the filename: the filename is a label a human chose and
    the records are what is actually being audited, and a report whose period disagrees with its
    own transactions is worse than one with no period at all.
    """
    if records:
        return min(record.timestamp for record in records).strftime("%Y-%m")
    stem = paths[0].stem if paths else ""
    return stem[:7] if len(stem) >= 7 and stem[4] == "-" else "unknown"


@dataclass
class _Prepared:
    """Steps 1-3, done. Shared by the blocking and streaming entry points so there is one
    definition of what a run starts with."""

    state: dict[str, Any]
    config: dict[str, Any]
    ledger: UsageLedger
    validation: ValidationReport
    run_id: str
    records: int


def _prepare(
    paths: str | Path | list[str | Path],
    *,
    ingestor: TransactionBatchIngestor | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> _Prepared:
    # --- step 1: the file itself ---------------------------------------------------------
    files = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise BatchUnreadable(f"{INGEST_FILE_UNREADABLE}: {', '.join(missing)}")

    # --- step 2: ingest ------------------------------------------------------------------
    records, validation = (ingestor or TransactionBatchIngestor()).ingest(files)
    if not records:
        raise BatchUnreadable(
            f"{INGEST_FILE_UNREADABLE}: {', '.join(p.name for p in files)} yielded no readable "
            f"transactions ({len(validation.quarantined)} quarantined)"
        )

    # --- step 3: state -------------------------------------------------------------------
    run_id = run_id or new_run_id()
    state = dict(initial_state(
        batch_id=", ".join(p.name for p in files),
        run_id=run_id,
        period=period_of(files, records),
        records=records,
        quarantined_count=len(validation.quarantined),
    ))

    # The ledger is registered as a run-level callback. LangChain propagates those into every
    # nested call, so a node added later is accounted for without being registered anywhere.
    config = run_config(
        run_id=run_id, batch_id=state["batch_id"], tags=tags, metadata=metadata,
        candidate_count=_upper_bound_candidates(records),
    )
    ledger = UsageLedger()
    config["callbacks"] = [ledger]
    return _Prepared(state, config, ledger, validation, run_id, len(records))


def _finish(prepared: _Prepared, final: dict[str, Any], store: ResultsStore | None) -> RunResult:
    report = final.get("report")
    if report is None:
        # Unreachable while the graph ends at `report`; asserted rather than assumed, because a
        # caller that receives None here would have no way to tell a clean batch from a crash.
        raise RuntimeError(f"run {prepared.run_id} completed without producing a report")

    # --- step 8: persist -----------------------------------------------------------------
    (store or InMemoryResultsStore()).save(report)

    result = RunResult(
        report=report,
        validation=prepared.validation,
        usage=prepared.ledger,
        run_id=prepared.run_id,
        candidates=len(final.get("candidates") or []),
        records=prepared.records,
    )
    log.info(
        "run %s: %d record(s), %d candidate(s), %d finding(s), rating %s",
        result.run_id, result.records, result.candidates,
        len(report.findings), report.risk_rating,
    )
    return result


def audit_batch(
    paths: str | Path | list[str | Path],
    *,
    store: ResultsStore | None = None,
    graph=None,
    ingestor: TransactionBatchIngestor | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> RunResult:
    """One batch, end to end. The CLI and the API both come through here."""
    prepared = _prepare(paths, ingestor=ingestor, tags=tags, metadata=metadata, run_id=run_id)
    final = (graph or build_graph()).invoke(prepared.state, prepared.config)  # type: ignore[arg-type]
    return _finish(prepared, dict(final), store)


def stream_audit(
    paths: str | Path | list[str | Path],
    *,
    store: ResultsStore | None = None,
    graph=None,
    ingestor: TransactionBatchIngestor | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
):
    """Yield `(node_name, accumulated_state)` as each node finishes, then `("__final__", RunResult)`.

    `audit_batch` returns only when the run is done, which is right for a CLI and wrong for a
    cockpit: a spinner for a minute and then everything at once tells an auditor nothing about
    *which candidate* the time and money went on. LangGraph's update stream yields each node's
    delta, so the accumulated state is maintained here -- a consumer wanting the current candidate
    should not have to reassemble it.
    """
    prepared = _prepare(paths, ingestor=ingestor, tags=tags, metadata=metadata, run_id=run_id)
    state = dict(prepared.state)
    for step in (graph or build_graph()).stream(
        prepared.state, prepared.config, stream_mode="updates"  # type: ignore[arg-type]
    ):
        for node, update in step.items():
            state.update(update or {})
            yield node, state
    yield "__final__", _finish(prepared, state, store)


def _upper_bound_candidates(records: list[TransactionRecord]) -> int:
    """A ceiling for the step budget, not a detection pass.

    Detection runs inside the graph, so the real count is not knowable here, and running detection
    twice to size a limit would double the one expensive deterministic step. One candidate per
    transaction cannot be exceeded -- every candidate holds at least `min_count` of them.
    """
    return len(records)


# --- CLI ------------------------------------------------------------------------------------


def print_run(result: RunResult) -> None:
    report = result.report
    project = tracing_project()
    print(f"\nrun_id     : {result.run_id}")
    print(f"tracing    : {f'LangSmith project {project!r}' if project else 'off'}")
    print(f"batch      : {report.period} · {result.records} record(s)")
    print(f"ingested   : {result.validation.summary()}")
    print(f"candidates : {result.candidates}")
    print(f"findings   : {len(report.findings)} "
          f"({report.needs_review_count} needing review)")
    print(f"risk       : {report.risk_rating}")
    print(result.usage.summary())
    per_candidate = result.cost_per_candidate
    if per_candidate is not None:
        print(f"per cand.  : ${per_candidate:.4f}")
    print(f"\n{report.summary}")


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, action="append", default=[],
                        help="a .pdf or .txt batch log; repeatable")
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    parser.add_argument("--tag", action="append", metavar="TAG", default=[],
                        help="extra LangSmith run tag; repeatable")
    parser.add_argument("--mermaid", action="store_true", help="print the graph and exit")
    parser.add_argument("--ascii", action="store_true", help="draw the graph in the terminal")
    parser.add_argument("--png", type=Path, metavar="FILE",
                        help="render the graph to a PNG (uploads node names to mermaid.ink)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.mermaid or args.ascii or args.png:
        drawable = build_graph().get_graph()
        if args.mermaid:
            print(drawable.draw_mermaid())
        if args.ascii:
            print(drawable.draw_ascii())
        if args.png:
            # Nothing about a batch is sent: the graph is built before any log is read, so this is
            # node names only.
            args.png.write_bytes(drawable.draw_mermaid_png())
            print(f"graph -> {args.png}")
        return 0

    if not args.batch:
        parser.error("--batch is required (or use --mermaid/--ascii/--png to draw the graph)")

    try:
        result = audit_batch(args.batch, tags=args.tag)
    except BatchUnreadable as error:
        print(f"error: {error}")
        return 2

    print_run(result)

    if args.json:
        payload = result.report.model_dump(mode="json")
        total = result.usage.total_cost
        payload["usage"] = {
            "calls": result.usage.calls,
            "total_tokens": result.usage.total_tokens,
            "total_cost_usd": float(total) if total is not None else None,
            "cost_per_candidate_usd": result.cost_per_candidate,
            "by_node": result.usage.rows(),
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nreport -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
