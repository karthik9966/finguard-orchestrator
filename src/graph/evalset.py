"""Capture real pipeline runs as scoreable evaluation cases (§8).

The blueprint's §8.2 example is a hand-written paragraph about a fictional ``TXN-093`` and a FINRA
rule. We can do better and should: there are four batches on disk, 880 parsed wires, and
``ledger_labels.csv`` recording which wires are laundering and under which typology. A case built
from what the pipeline actually produced, judged against what was actually planted, is worth more
than fifteen invented scenarios -- and the blueprint's own example cannot be grounded here at all,
since Phase 1 established FINRA publishes no rule PDFs and the ADGM AML Rulebook is what we
indexed.

Capture is separated from scoring because they fail for different reasons and cost different
money. Running the pipeline is ~$0.13 a batch; judging its output is a further gpt-4o call per
metric per case. Freezing the runs to disk means a prompt change is scored against the same
evidence twice -- before and after -- rather than against two different runs that also happened to
retrieve different clauses.

Usage::

    uv run python -m src.graph.evalset              # capture all four batches
    uv run python -m src.graph.evalset --batch <p>  # just one
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEDGER_DIR = PROJECT_ROOT / "data" / "processed" / "ledger"
LABELS = PROJECT_ROOT / "data" / "processed" / "ledger_labels.csv"
CASES_PATH = PROJECT_ROOT / "data" / "processed" / "eval_cases.json"


def ground_truth(batch: str) -> dict[str, Any]:
    """What was actually planted in this batch, from the answer key.

    This is the half of an evaluation that cannot be inferred from the run itself. Without it
    Faithfulness would only ever ask "is the report consistent with what it retrieved" -- a
    question a confidently wrong report passes.
    """
    import pandas as pd

    labels = pd.read_csv(LABELS, dtype={"Sender_account": str, "Receiver_account": str})
    rows = labels[labels.Log_file.str.startswith(batch[:7])]
    laundering = rows[rows.Is_laundering == 1]
    return {
        "wires": len(rows),
        "laundering_wires": len(laundering),
        "typologies": laundering.Laundering_type.value_counts().to_dict(),
        "references": sorted(laundering.Reference),
    }


def expected_output(truth: dict[str, Any]) -> str:
    """The reference answer, phrased as a finding rather than as a label dump.

    Deliberately does *not* name the wire references. A report that listed all 21 by luck would
    score well on a string comparison while having reasoned about none of them; what we want to
    reward is a report that describes the right *kind* of activity.
    """
    if not truth["laundering_wires"]:
        return (
            "This batch contains no money laundering. Any pattern the detectors surfaced is a "
            "false positive arising from legitimate activity, and the report should reach no "
            "grounded finding of suspicious conduct."
        )
    named = ", ".join(
        # SAML-D writes these as Layered_Fan_In. A judge reads the reference answer as prose, so
        # it gets prose.
        f"{count} wires exhibiting {typology.replace('_', ' ').replace('-', ' ').lower()}"
        for typology, count in sorted(truth["typologies"].items(), key=lambda kv: -kv[1])
    )
    return (
        f"This batch contains {truth['laundering_wires']} laundering wires out of "
        f"{truth['wires']}: {named}. The report should identify these patterns and cite the "
        f"monitoring, due-diligence or reporting obligations that govern them."
    )


def audit_question(batch: str, state: dict[str, Any]) -> str:
    """The 'input' the pipeline was given.

    Our pipeline takes a batch, not a free-text question -- Decision 3 replaced the auditor's
    query with obligation-shaped templates. So the input is reconstructed as the question the run
    answers, which is what Answer Relevancy needs to judge against.
    """
    candidates = state.get("candidates") or []
    shapes = sorted({c.shape for c in candidates})
    return (
        f"Audit the transaction batch {batch} for AML compliance. "
        f"{len(state.get('wires') or [])} wires were parsed and {len(candidates)} candidate "
        f"patterns were detected ({', '.join(shapes) or 'none'}). "
        f"Which regulatory obligations apply, and which wires warrant review?"
    )


def build_case(batch_path: Path) -> dict[str, Any]:
    """Run the pipeline once and freeze everything a judge needs."""
    from src.graph.graph import audit_batch

    batch = batch_path.name
    state = audit_batch(str(batch_path), tags=["EVAL_CAPTURE"])
    report = state.get("report")
    if report is None:
        raise RuntimeError(f"{batch}: the run produced no report")

    truth = ground_truth(batch)
    usage = state.get("usage")
    return {
        "name": batch,
        "input": audit_question(batch, state), # type: ignore
        "actual_output": report.audit_summary,
        # The clauses the drafter actually saw, in the order it saw them -- Contextual Precision
        # scores that ordering, so a shuffled copy would measure a different pipeline.
        "retrieval_context": [
            f"[{d.metadata.get('document_title')} {d.metadata.get('section_clause')}] "
            f"{d.page_content}"
            for d in state.get("retrieved_context", []) # type: ignore
        ],
        "expected_output": expected_output(truth),
        "ground_truth": truth,
        "run": {
            "audit_id": state.get("audit_id"),
            "risk_rating": report.risk_rating,
            "flagged_wires": report.flagged_wires,
            "applicable_regulations": report.applicable_regulations,
            "confidence_score": state.get("confidence_score"),
            "loop_count": state.get("loop_count"),
            "cost_usd": float(usage.total_cost) if usage and usage.total_cost else None,
        },
    }


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, help="capture one batch; default is all of them")
    parser.add_argument("--out", type=Path, default=CASES_PATH)
    args = parser.parse_args()

    batches = [args.batch] if args.batch else sorted(LEDGER_DIR.glob("*.pdf"))
    if not batches:
        raise SystemExit(f"no batches in {LEDGER_DIR} -- run: uv run finguard-ledger")

    existing = {}
    if args.out.exists():
        existing = {case["name"]: case for case in json.loads(args.out.read_text())}

    for path in batches:
        print(f"capturing {path.name} ...")
        case = build_case(path)
        existing[case["name"]] = case
        print(f"  risk {case['run']['risk_rating']}, "
              f"{len(case['retrieval_context'])} clauses, "
              f"truth: {case['ground_truth']['laundering_wires']} laundering wires")

    cases = [existing[name] for name in sorted(existing)]
    args.out.write_text(json.dumps(cases, indent=2))
    print(f"\n{len(cases)} case(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
