"""Tier 1 metrics -- everything measurable without a model call.

Four of Evaluation Design §4's five KPIs have a deterministic component, and three can be measured
with no model at all. That is not a cost dodge; it is the design. Faithfulness is a subset check
rather than a judged score, retrieval quality is a ranking against a curated answer, and the
clean-batch guarantee is an assertion that nothing was spent. A suite that needs an API key to tell
you whether retrieval regressed is a suite that stops being run.

    uv run python -m eval.run --tier deterministic
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

from eval import corpora
from eval.baseline import run_baseline
from src.detection.base import detect_all
from src.ingestion.batch import TransactionBatchIngestor
from src.models import Candidate

# No fallback: the golden batches parse cleanly by construction, and a silent model call inside a
# "deterministic, free" runner would make the tier a lie.
NO_FALLBACK = TransactionBatchIngestor(fallback=lambda failure: None)


@dataclass
class Metric:
    name: str
    value: float
    target: float | None
    unit: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    advisory: bool = False

    @property
    def passed(self) -> bool | None:
        if self.target is None or self.advisory:
            return None
        return self.value >= self.target

    def line(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "----"}[self.passed]
        target = f" (target >= {self.target:g})" if self.target is not None else ""
        return f"  [{mark}] {self.name:38} {self.value:.4g}{self.unit}{target}"


def _records(path):
    records, report = NO_FALLBACK.ingest([path])
    return records, report


# --- recall, and the comparator that makes it mean something -------------------------------


def recall_against_baseline() -> list[Metric]:
    """Pattern-level recall for both detectors, with the alert volume each paid for it.

    Recall alone is unfalsifiable -- flag everything and score 1.00 -- so the two numbers are
    reported together and the baseline runs over exactly the same batches.
    """
    instances = corpora.labeled_patterns()
    by_batch: dict[str, list] = {}
    for instance in instances:
        by_batch.setdefault(instance.batch, []).append(instance)

    found = named = baseline_found = 0
    swept_total = baseline_total = records_total = 0
    per_pattern: dict[str, dict[str, int]] = {}
    missed: list[str] = []

    for batch, group in sorted(by_batch.items()):
        records, _ = _records(corpora.EVAL_LEDGER / batch)
        records_total += len(records)

        candidates = detect_all(records)
        swept = {ref for c in candidates for ref in c.member_txn_refs}
        swept_total += len(swept)
        alerted = run_baseline(records).flagged_refs
        baseline_total += len(alerted)

        for instance in group:
            tally = per_pattern.setdefault(
                instance.pattern_type, {"found": 0, "named": 0, "baseline": 0, "total": 0}
            )
            tally["total"] += 1
            # Named: found by a candidate *of this pattern*. The pattern-level figure counts any
            # candidate that covers the instance, which is what the reviewer sees; this one is
            # whether the right detector saw it, which is what PRD v2's per-pattern KPI asks.
            named_refs = {
                ref for c in candidates if c.pattern_type == instance.pattern_type
                for ref in c.member_txn_refs
            }
            if instance.found_by(named_refs):
                named += 1
                tally["named"] += 1
            if instance.found_by(swept):
                found += 1
                tally["found"] += 1
            else:
                missed.append(instance.id)
            if instance.found_by(alerted):
                baseline_found += 1
                tally["baseline"] += 1

    total = len(instances)
    weakest = min(per_pattern, key=lambda p: per_pattern[p]["found"] / per_pattern[p]["total"])
    return [
        Metric(
            "recall (pattern level)", found / total, 0.90,
            detail={
                "found": found, "instances": total, "missed": missed,
                "per_pattern": per_pattern,
            },
        ),
        # Eval Design v2 §4: ">= 0.90 each", so a weak detector cannot hide behind a strong one.
        Metric(
            "recall (weakest pattern)",
            per_pattern[weakest]["found"] / per_pattern[weakest]["total"], 0.90,
            detail={"pattern": weakest, **per_pattern[weakest]},
        ),
        Metric(
            "recall (named pattern)", named / total, None,
            detail={
                "named": named, "instances": total,
                "per_pattern": {p: f"{t['named']}/{t['total']}" for p, t in per_pattern.items()},
            },
        ),
        Metric(
            "alert volume (share of batch)", swept_total / records_total, None, unit=" of records",
            detail={"transactions_swept": swept_total, "records": records_total},
        ),
        Metric(
            "baseline recall (rules only)", baseline_found / total, None,
            detail={"found": baseline_found, "instances": total},
        ),
        Metric(
            "baseline alert volume", baseline_total / records_total, None, unit=" of records",
            detail={"transactions_alerted": baseline_total, "records": records_total},
        ),
    ]


# --- context precision ---------------------------------------------------------------------


def context_precision() -> list[Metric]:
    """Whether the top reranked indicator is the curated correct one (Eval Design §4).

    Free, because the whole indicator path is local: MiniLM for the query vector, Chroma for the
    search, FlashRank for the rerank. No hosted model is involved in retrieval at all.
    """
    from src.retrieval.retriever import TierAwareRetriever

    queries = corpora.complex_queries(resolve=True)
    retriever = TierAwareRetriever()
    scored = [q for q in queries if q.scored]

    top1 = top3 = 0
    outcomes: list[dict[str, Any]] = []
    for query in scored:
        candidate = _candidate_for(query)
        notes = _notes()
        ranked = retriever.indicators_for(
            retriever.queries.indicator_query(candidate), notes
        )
        ids = [chunk.chunk_id for chunk in ranked]
        correct = corpora.resolve_pair(query.correct) # type: ignore

        hit_at_1 = bool(ids) and ids[0] == correct
        hit_at_3 = correct in ids[:3]
        top1 += hit_at_1
        top3 += hit_at_3
        outcomes.append({
            "id": query.id,
            "hit@1": hit_at_1,
            "hit@3": hit_at_3,
            "rank": ids.index(correct) + 1 if correct in ids else None,
            "returned": len(ids),
        })

    unscored = [q.id for q in queries if not q.scored]
    return [
        Metric(
            "context precision (hit@1)", top1 / len(scored), 0.90,
            detail={"scored": len(scored), "excluded": unscored, "outcomes": outcomes},
        ),
        Metric(
            "context precision (hit@3)", top3 / len(scored), None,
            detail={"note": "reported because rerank_top_n is 5: hit@3 is what a model actually sees"},
        ),
    ]


def _candidate_for(query) -> Candidate:
    """A Candidate from a dataset spec. The references are placeholders -- retrieval never reads them,
    only the pattern type and the measured attributes."""
    return Candidate(
        candidate_id=f"{query.pattern_type}:{query.id}:0000000000",
        pattern_type=query.pattern_type,
        member_txn_refs=[f"{query.id}-{n}" for n in range(1, 4)],
        attributes=dict(query.candidate.get("attributes", {})),
        detection_confidence=0.5,
    )


def _notes():
    from src.retrieval.retriever import RetrievalNotes

    return RetrievalNotes()


# --- the guarantees ------------------------------------------------------------------------


def clean_batch() -> list[Metric]:
    """Zero candidates on the clean month. The premise of every "$0.0000" claim in this repo."""
    record = corpora.clean_batch()
    records, _ = _records(record["path"])
    candidates = detect_all(records)
    return [
        Metric(
            "clean batch candidates", 0.0 if candidates else 1.0, 1.0,
            detail={"candidates": len(candidates), "records": len(records)},
        )
    ]


def malformed_inputs() -> list[Metric]:
    """Every malformed case handled as its record says, with no exception escaping.

    The expectations live in the dataset rather than here, so a reader can see what each file is
    supposed to prove without reading the runner.
    """
    outcomes: list[dict[str, Any]] = []
    honoured = 0
    for case in corpora.malformed_inputs():
        outcome: dict[str, Any] = {"id": case["id"], "expect": case["expect"]}
        try:
            records, report = NO_FALLBACK.ingest([case["path"]])
            outcome["parsed"] = len(records)
            outcome["quarantined"] = len(report.quarantined)
            outcome["raised"] = None
        except Exception as error:  # noqa: BLE001 - a clean batch-level error is an allowed outcome
            records, report = [], None
            outcome["parsed"] = 0
            outcome["quarantined"] = 0
            outcome["raised"] = f"{type(error).__name__}: {error}"

        outcome["ok"] = _honours(case, records, report, outcome)
        honoured += outcome["ok"]
        outcomes.append(outcome)

    return [
        Metric(
            "malformed inputs handled", honoured / len(outcomes), 1.0,
            detail={"outcomes": outcomes},
        )
    ]


def _honours(case, records, report, outcome) -> bool:
    """Check one malformed case against its stated expectation."""
    expect = case["expect"]

    if expect.get("no_unhandled_exception") or expect.get("batch_unreadable"):
        # Both allow a clean refusal; neither allows a crash inside a parser.
        if outcome["raised"] and "UnicodeDecodeError" in outcome["raised"]:
            return False

    if "parsed" in expect and outcome["parsed"] != expect["parsed"]:
        return False
    if expect.get("batch_unreadable") and outcome["parsed"]:
        return False
    if "quarantined" in expect and outcome["quarantined"] != expect["quarantined"]:
        return False
    if "quarantined_or_rescued" in expect:
        # Either path is correct: what must not happen is the message vanishing.
        accounted = outcome["quarantined"] + (report.rescued if report else 0)
        if accounted < expect["quarantined_or_rescued"]:
            return False
    if expect.get("quarantine_has_ordinal") and report is not None:
        if not all(message.ordinal >= 1 for message in report.quarantined):
            return False
    if "amount_must_not_be" in expect:
        forbidden = Decimal(expect["amount_must_not_be"])
        if any(record.amount == forbidden for record in records):
            return False
    if "distinct_refs" in expect:
        if len({record.txn_ref for record in records}) != expect["distinct_refs"]:
            return False
    return True


def schema_conformance() -> list[Metric]:
    """Every object the detectors produce validates against its model (Eval Design §4, 100% strict).

    Pydantic enforces this at construction, so what is actually being measured is that the golden
    corpus exercises the path -- a validation error anywhere in 75 instances across eleven batches
    would surface here rather than in production.
    """
    checked = failures = 0
    for batch in sorted({instance.batch for instance in corpora.labeled_patterns()}):
        records, report = _records(corpora.EVAL_LEDGER / batch)
        checked += len(records)
        for candidate in detect_all(records):
            try:
                Candidate(**candidate.model_dump())
                checked += 1
            except Exception:  # noqa: BLE001
                failures += 1
        try:
            report.model_validate(report.model_dump())
            checked += 1
        except Exception:  # noqa: BLE001
            failures += 1

    return [
        Metric(
            "schema conformance", (checked - failures) / checked if checked else 0.0, 1.0,
            detail={"objects": checked, "failures": failures},
        )
    ]


def run() -> dict[str, Any]:
    """Every deterministic metric, in the order a reader wants them."""
    started = time.perf_counter()
    metrics: list[Metric] = []
    for name, step in (
        ("recall", recall_against_baseline),
        ("retrieval", context_precision),
        ("clean batch", clean_batch),
        ("malformed inputs", malformed_inputs),
        ("schema", schema_conformance),
    ):
        print(f"  running {name} ...", flush=True)
        metrics.extend(step())

    elapsed = time.perf_counter() - started
    failed = [m.name for m in metrics if m.passed is False]
    return {
        "tier": "deterministic",
        "seconds": round(elapsed, 1),
        "passed": not failed,
        "failed": failed,
        "metrics": [asdict(m) for m in metrics],
        "_metrics": metrics,
    }
