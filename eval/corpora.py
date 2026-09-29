"""Loading the golden datasets -- Eval Design §3.

One place that reads them, so a runner cannot invent its own idea of what a record means, and so the
`(source_id, section_ref)` pairs in `complex_queries.json` are resolved the same way the obligation
map's are: **at load time, never stored as chunk ids.** `chunk_id = hash(source_id, section_ref,
version)`, so a re-chunk or a corpus version bump would silently invalidate a literal id -- and a
golden dataset that silently stops pointing at anything measures nothing while still reporting a
number. Phase 1c learned that on the obligation map; there is no reason to learn it twice.

The `Labeled_Patterns` records name transactions in `data/processed/eval_ledger/`, which is generated
rather than committed:

    uv run finguard-ledger --profile eval

That corpus is deterministic -- same seed, same SAML-D, same references -- which is what makes it
safe to commit a dataset that points into it. `uv run python -m eval.build_datasets --check` is the
assertion that the two still agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
DATASETS = EVAL_ROOT / "datasets"
PROJECT_ROOT = EVAL_ROOT.parent
EVAL_LEDGER = PROJECT_ROOT / "data" / "processed" / "eval_ledger"


class CorpusMissing(FileNotFoundError):
    """The dataset is committed but the ledger it points into has not been generated."""


def _load(name: str) -> Any:
    path = DATASETS / f"{name}.json"
    if not path.exists():
        raise CorpusMissing(f"{path.relative_to(PROJECT_ROOT)} is missing")
    return json.loads(path.read_text())


@dataclass
class LabeledPattern:
    """One planted instance the system is expected to report."""

    id: str
    pattern_type: str
    batch: str
    cluster: str
    txn_refs: list[str]
    saml_d_typology: str = ""
    # Half the cluster is enough. Requiring every transaction fails an instance because one leg fell
    # outside the window; requiring one passes a candidate that merely clipped its edge.
    coverage_required: float = 0.5

    @property
    def path(self) -> Path:
        return EVAL_LEDGER / self.batch

    def found_by(self, swept: set[str]) -> bool:
        covered = len(set(self.txn_refs) & swept)
        return covered >= self.coverage_required * len(self.txn_refs)


@dataclass
class ComplexQuery:
    """A candidate whose correct indicator competes with lexically similar wrong ones."""

    id: str
    pattern_type: str
    candidate: dict[str, Any]
    why_hard: str
    correct: tuple[str, str] | None
    distractors: list[tuple[str, str]] = field(default_factory=list)
    expect_no_indicator: bool = False
    excluded_from_precision: str = ""

    @property
    def scored(self) -> bool:
        """Whether this record counts toward context precision.

        CQ-009 does not: the corpus contains no round-trip red flag at all, so scoring it would
        measure the corpus rather than the retriever. Stated in the record and honoured here instead
        of quietly dropping it.
        """
        return self.correct is not None and not self.excluded_from_precision


def labeled_patterns(*, require_ledger: bool = True) -> list[LabeledPattern]:
    records = [
        LabeledPattern(
            id=row["id"],
            pattern_type=row["pattern_type"],
            batch=row["batch"],
            cluster=row["cluster"],
            txn_refs=list(row["txn_refs"]),
            saml_d_typology=row.get("saml_d_typology", ""),
        )
        for row in _load("labeled_patterns")
    ]
    if require_ledger:
        missing = sorted({r.batch for r in records if not r.path.exists()})
        if missing:
            raise CorpusMissing(
                f"{len(missing)} golden batch(es) are not on disk ({missing[0]} ...) -- run: "
                "uv run finguard-ledger --profile eval"
            )
    return records


def complex_queries(*, resolve: bool = True) -> list[ComplexQuery]:
    """The retrieval set. With `resolve`, every curated pair is checked against the live corpus.

    Checked rather than trusted: a pair that no longer resolves would make context precision
    unmeasurable in the one direction that still produces a number -- the correct answer simply never
    appears, and the retriever takes the blame for a curation error.
    """
    queries = []
    for row in _load("complex_queries"):
        correct = row.get("correct_indicator")
        queries.append(ComplexQuery(
            id=row["id"],
            pattern_type=row["pattern_type"],
            candidate=row["candidate"],
            why_hard=row["why_hard"],
            correct=(correct["source_id"], correct["section_ref"]) if correct else None,
            distractors=[(d["source_id"], d["section_ref"]) for d in row.get("distractors", [])],
            expect_no_indicator=bool(row.get("expect_no_indicator")),
            excluded_from_precision=row.get("excluded_from_precision", ""),
        ))

    if resolve:
        unresolved = [
            f"{query.id}: {pair[0]} | {pair[1]}"
            for query in queries
            for pair in ([query.correct] if query.correct else []) + query.distractors
            if resolve_pair(pair) is None
        ]
        if unresolved:
            raise CorpusMissing(
                "curated indicator pairs no longer resolve against rule_chunks:\n  "
                + "\n  ".join(unresolved)
            )
    return queries


def resolve_pair(pair: tuple[str, str]) -> str | None:
    """`(source_id, section_ref)` -> chunk_id, through the same client the retriever uses."""
    from src.ingestion.store import RULE_COLLECTION, VectorStoreClient

    return _store(VectorStoreClient, RULE_COLLECTION).resolve(pair)


_CLIENT: Any = None


def _store(factory, collection):
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = factory(collection)
    return _CLIENT


def benign_lookalikes() -> list[dict[str, Any]]:
    return _load("benign_lookalikes")


def injected_memos() -> list[dict[str, Any]]:
    return _load("injected_memos")


def malformed_inputs() -> list[dict[str, Any]]:
    records = _load("malformed_inputs")
    for record in records:
        record["path"] = PROJECT_ROOT / record["file"]
    return records


def clean_batch() -> dict[str, Any]:
    record = _load("clean_batch")
    record["path"] = PROJECT_ROOT / record["batch"]
    return record


def summary() -> dict[str, int]:
    """What is on disk, for a runner to print before it starts spending money."""
    return {
        "labeled_patterns": len(_load("labeled_patterns")),
        "benign_lookalikes": len(benign_lookalikes()),
        "complex_queries": len(_load("complex_queries")),
        "malformed_inputs": len(_load("malformed_inputs")),
        "injected_memos": len(injected_memos()),
        "clean_batch": 1,
    }
