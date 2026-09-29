"""The golden datasets themselves -- Eval Design §3.

A dataset is not fixtures, it is the measuring instrument, so it gets tested like one. The failures
these catch are all quiet: a curated indicator pair that stops resolving makes context precision
unmeasurable while still producing a number; a `labeled_patterns` record whose references are not in
the batch it names makes recall look worse than it is; a lookalike that expects `high` makes triage
precision meaningless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import corpora
from src.config import PATTERN_TYPES

EVAL_LEDGER = corpora.EVAL_LEDGER
needs_corpus = pytest.mark.skipif(
    not (EVAL_LEDGER / "2022-10_private_banking_log.txt").exists(),
    reason="run: uv run finguard-ledger --profile eval",
)


# --- shape ---------------------------------------------------------------------------------


def test_all_six_corpora_are_present():
    """Eval Design §3's set. A missing one is a metric that silently is not measured."""
    assert corpora.summary() == {
        "labeled_patterns": 75,
        "benign_lookalikes": 20,
        "complex_queries": 10,
        "malformed_inputs": 10,
        "injected_memos": 5,
        "clean_batch": 1,
    }


def test_fifteen_labeled_instances_of_every_pattern():
    """15 rather than Eval Design's "~10 / pattern", per the migration plan's conflict table: a
    denominator where one miss is 6.7% rather than 10%."""
    counts: dict[str, int] = {}
    for record in corpora.labeled_patterns(require_ledger=False):
        counts[record.pattern_type] = counts.get(record.pattern_type, 0) + 1
    assert counts == {pattern: 15 for pattern in PATTERN_TYPES}


def test_every_id_is_unique_across_every_corpus():
    ids = [r.id for r in corpora.labeled_patterns(require_ledger=False)]
    ids += [r["id"] for r in corpora.benign_lookalikes()]
    ids += [r["id"] for r in corpora.malformed_inputs()]
    ids += [r["id"] for r in corpora.injected_memos()]
    ids += [q.id for q in corpora.complex_queries(resolve=False)]
    assert len(ids) == len(set(ids))


def test_no_instance_is_used_twice():
    """A pattern counted twice inflates recall's denominator and its numerator together, which hides
    a miss rather than reporting one."""
    seen = [(r.batch, r.cluster) for r in corpora.labeled_patterns(require_ledger=False)]
    assert len(seen) == len(set(seen))
    refs = [ref for r in corpora.labeled_patterns(require_ledger=False) for ref in r.txn_refs]
    assert len(refs) == len(set(refs)), "a transaction belongs to at most one planted instance"


def test_the_instances_are_spread_across_the_months_that_have_them():
    """15 clusters taken in batch order would come from the first two months, and a recall number
    measured on two months of one year is a narrower claim than it looks."""
    for pattern in PATTERN_TYPES:
        batches = {
            r.batch for r in corpora.labeled_patterns(require_ledger=False)
            if r.pattern_type == pattern
        }
        assert len(batches) >= 5, f"{pattern} draws from only {len(batches)} batch(es)"


# --- the labels point at something real ----------------------------------------------------


@needs_corpus
def test_every_labeled_reference_is_in_the_batch_it_names():
    """The check that makes the dataset trustworthy: a reference that is not in its batch would count
    as a miss for ever, and look exactly like a detector fault."""
    by_batch: dict[str, set[str]] = {}
    for record in corpora.labeled_patterns():
        if record.batch not in by_batch:
            text = record.path.read_text()
            by_batch[record.batch] = {
                line[len(":20:"):].strip()
                for line in text.splitlines() if line.startswith(":20:")
            }
        missing = sorted(set(record.txn_refs) - by_batch[record.batch])
        assert not missing, f"{record.id}: {len(missing)} reference(s) not in {record.batch}"


@needs_corpus
def test_the_derived_datasets_still_match_the_corpus():
    """`build_datasets --check`, as a test. The committed dataset points into a *generated* ledger, so
    the guarantee that makes that safe -- same seed, same references -- has to be asserted rather
    than assumed."""
    from eval.build_datasets import clean_batch, labeled_patterns

    committed = json.loads((corpora.DATASETS / "labeled_patterns.json").read_text())
    assert committed == labeled_patterns(), "run: uv run python -m eval.build_datasets"
    assert json.loads((corpora.DATASETS / "clean_batch.json").read_text()) == clean_batch()


@needs_corpus
def test_the_clean_batch_really_is_clean():
    """It is the one corpus shared with the dev set, and the whole "$0.0000 on a clean month" claim
    rests on it having nothing in it."""
    import pandas as pd

    record = corpora.clean_batch()
    assert record["path"].exists()
    labels = pd.read_csv(
        corpora.PROJECT_ROOT / "data" / "processed" / "ledger_labels.csv", dtype=str
    )
    rows = labels[labels.Log_file == "2023-05_private_banking_log.pdf"]
    assert len(rows) == 500
    assert set(rows.Is_laundering) == {"0"}, "the clean control has a suspicious row in it"


# --- the authored labels -------------------------------------------------------------------


def test_no_benign_lookalike_expects_high():
    """If a lookalike should be filed High, it is not a lookalike -- the dataset is wrong rather than
    the system, and triage precision measured against it would be meaningless."""
    for record in corpora.benign_lookalikes():
        assert record["expected_risk_band"] in {"low", "medium"}, record["id"]


def test_every_authored_record_says_why():
    """The reasoning *is* the label. A reviewer who disagrees with it is disagreeing with the label,
    which is only possible if it is written down."""
    for record in corpora.benign_lookalikes():
        assert len(record["why_benign"]) > 80, record["id"]
    for record in corpora.injected_memos():
        assert len(record["why"]) > 80, record["id"]
    for record in corpora.malformed_inputs():
        assert len(record["why"]) > 80, record["id"]
    for query in corpora.complex_queries(resolve=False):
        assert len(query.why_hard) > 80, query.id


def test_the_lookalikes_cover_every_pattern():
    """A shape with no benign counterpart is a shape whose triage is untested."""
    shapes = {record["shape"] for record in corpora.benign_lookalikes()}
    assert shapes == set(PATTERN_TYPES)


def test_every_injection_has_a_clean_control():
    """The same candidate with a clean memo, so a difference in outcome is attributable to the
    injection and to nothing else."""
    for record in corpora.injected_memos():
        assert record["control_memo"] and record["control_memo"] != record["memo"]
        assert record["spec"]["memo"] == record["memo"]


def test_the_malformed_files_exist_and_include_a_genuinely_non_utf8_one():
    """It is a file rather than a JSON string precisely because it cannot be a JSON string."""
    records = corpora.malformed_inputs()
    for record in records:
        assert record["path"].exists(), record["id"]

    latin1 = next(r for r in records if r["id"] == "MI-004")
    raw = latin1["path"].read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")


# --- the curated pairs ---------------------------------------------------------------------


@pytest.mark.skipif(
    not (corpora.PROJECT_ROOT / "chroma_db").exists(), reason="run: uv run finguard-store --rules"
)
def test_every_curated_indicator_pair_resolves():
    """Pairs, not chunk ids: `chunk_id = hash(source_id, section_ref, version)`, so a literal id goes
    stale on a re-chunk with no error. Phase 1c learned this on the obligation map."""
    queries = corpora.complex_queries(resolve=True)
    assert len(queries) == 10
    assert sum(1 for q in queries if q.scored) == 9


def test_the_unscored_query_says_why_it_is_unscored():
    """CQ-009: the corpus has no round-trip red flag, so scoring it would measure the corpus rather
    than the retriever. Excluded explicitly rather than quietly dropped."""
    unscored = [q for q in corpora.complex_queries(resolve=False) if not q.scored]
    assert len(unscored) == 1
    assert unscored[0].id == "CQ-009"
    assert unscored[0].expect_no_indicator is True
    assert "no correct answer exists" in unscored[0].excluded_from_precision


def test_a_distractor_is_never_the_correct_answer():
    for query in corpora.complex_queries(resolve=False):
        assert query.correct not in query.distractors, query.id
