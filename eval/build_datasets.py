"""Build the derived half of the golden datasets -- Eval Design §3.

Two of the six corpora are *derived* and this script writes them; four are *authored* and live in
version control as hand-written JSON. The distinction matters more than it looks:

* **Derived** means the labels are SAML-D's own. `Labeled_Patterns` is a selection from a corpus
  whose suspicious rows arrive already labelled with their typology, and `Clean_Batch` is one month
  with none. Nobody's judgement is in the label, so re-running this script reproduces them exactly.
* **Authored** means somebody decided. A payroll fan-in that *should* rank below a real launderer, or
  which of four similar-sounding red flags is the right one for a candidate, is a judgement --
  and a judgement is worth reviewing, so it is written down rather than computed.

The golden corpus is `data/processed/eval_ledger/`, which is **not** the dev corpus. Every number in
`config.yaml` cites "measured across the four dev batches" as its evidence; evaluating on that same
data would be marking my own homework.

    uv run python -m eval.build_datasets            # rebuild the derived datasets
    uv run python -m eval.build_datasets --check    # verify they match what is committed
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path(__file__).resolve().parent / "datasets"
EVAL_LEDGER = PROJECT_ROOT / "data" / "processed" / "eval_ledger"
EVAL_LABELS = PROJECT_ROOT / "data" / "processed" / "eval_labels.csv"

# The dev corpus's clean control. Reused rather than re-cut: it is already the batch every "$0.0000
# on a clean month" claim was measured on, and a second clean batch would only be a second chance
# for the same assertion to pass.
CLEAN_BATCH = PROJECT_ROOT / "data" / "processed" / "ledger" / "2023-05_private_banking_log.txt"

# SAML-D's typology labels mapped onto PRD v2's nine patterns. Two families collapse: layered fan-in
# and fan-out are one `layered_fan` (direction is an attribute), plain and stacked bipartite are one
# `bipartite`. Smurfing is no longer here -- PRD v2 §2 defers it, and the generator stopped planting
# it -- so a Smurfing cluster reaching this map is a generator bug and fails as unmapped.
PATTERN_OF = {
    "Structuring": "structuring",
    "Fan_In": "fan_in",
    "Fan_Out": "fan_out",
    "Cycle": "cycle",
    "Scatter-Gather": "scatter_gather",
    "Gather-Scatter": "gather_scatter",
    "Deposit-Send": "deposit_send",
    "Layered_Fan_In": "layered_fan",
    "Layered_Fan_Out": "layered_fan",
    "Bipartite": "bipartite",
    "Stacked Bipartite": "bipartite",
}

# Eval Design v2 §3 says "~10 / pattern"; 15 was kept (confirmed 2026-09-29), for a
# recall figure whose denominator is large enough that one miss is 6.7% rather than 10%.
PER_PATTERN = 15

# Patterns SAML-D cannot supply 15 of once the golden set is both held out and whole (confirmed
# 2026-09-29: take what exists and say so, rather than plant month-truncated halves or tune on the
# golden clusters). Measured over all of SAML-D: 15 Scatter-Gather clusters lie wholly inside one
# month, and the dev/eval partition leaves 6 of them to eval; Gather-Scatter has 21, 12 in eval.
# Each takes every instance the corpus has, above this floor, and the note travels with the record
# so a recall figure over 6 is never read as one over 15.
SUPPLY_LIMITED: dict[str, tuple[int, str]] = {
    "scatter_gather": (
        5,
        "SAML-D has 15 Scatter-Gather clusters wholly inside one month; the dev/eval partition "
        "leaves 6 to the golden set",
    ),
    "gather_scatter": (
        10,
        "SAML-D has 21 Gather-Scatter clusters wholly inside one month; the dev/eval partition "
        "leaves 12 to the golden set",
    ),
}


def instances() -> pd.DataFrame:
    """Every planted instance in the golden corpus, from the answer key the generator wrote.

    `Cluster` is emitted by the planter itself, so an instance is what was planted rather than what a
    consumer could infer: a cycle is a walked chain with no anchor account, and guessing its
    membership from the labels afterwards is not reliably possible.
    """
    if not EVAL_LABELS.exists():
        raise SystemExit(
            f"{EVAL_LABELS.relative_to(PROJECT_ROOT)} is missing -- run: "
            "uv run finguard-ledger --profile eval"
        )
    labels = pd.read_csv(EVAL_LABELS, dtype={"Reference": str, "Cluster": str})
    planted = labels[(labels.Is_laundering == 1) & labels.Cluster.notna()].copy()
    planted["typology"] = planted.Cluster.str.rsplit("-", n=1).str[0]
    planted["pattern"] = planted.typology.map(PATTERN_OF)
    unmapped = sorted(set(planted.loc[planted.pattern.isna(), "typology"]))
    if unmapped:
        raise SystemExit(f"typologies with no PRD §2 pattern: {unmapped}")
    return planted


def labeled_patterns() -> list[dict[str, Any]]:
    """`PER_PATTERN` instances of each pattern, spread across the months that have them.

    Spread deliberately. Taking 15 structuring clusters in batch order would take most of them from
    the first two months, and a recall number measured on two months of one year's traffic is a
    narrower claim than it looks. Round-robin over batches instead, and the selection is still
    deterministic -- so the committed dataset is reproducible from the corpus.
    """
    planted = instances()
    grouped = planted.groupby(["pattern", "Log_file", "Cluster"], sort=True)

    by_pattern: dict[str, dict[str, list[tuple[str, list[str]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (pattern, log, cluster), rows in grouped:
        refs = sorted(rows.Reference)
        by_pattern[pattern][log].append((cluster, refs))

    records: list[dict[str, Any]] = []
    for pattern in sorted(by_pattern):
        batches = by_pattern[pattern]
        # One from each batch, then a second from each, and so on.
        ordered: list[tuple[str, str, list[str]]] = []
        for depth in range(max(len(v) for v in batches.values())):
            for log in sorted(batches):
                if depth < len(batches[log]):
                    cluster, refs = batches[log][depth]
                    ordered.append((log, cluster, refs))

        chosen = ordered[:PER_PATTERN]
        floor, note = SUPPLY_LIMITED.get(pattern, (PER_PATTERN, ""))
        if len(chosen) < floor:
            raise SystemExit(
                f"only {len(chosen)} {pattern} instances in the golden corpus, need {floor} "
                "-- raise `clusters_per_typology` or the month count in the eval profile"
            )
        for ordinal, (log, cluster, refs) in enumerate(chosen, start=1):
            records.append({
                "id": f"LP-{pattern}-{ordinal:03d}",
                "pattern_type": pattern,
                "saml_d_typology": cluster.rsplit("-", 1)[0],
                "batch": log.replace(".pdf", ".txt"),
                "cluster": cluster,
                "transactions": len(refs),
                "txn_refs": refs,
                # Recall is "did the system report this instance", so what counts as reported has to
                # be stated rather than left to a runner: a candidate that covers the majority of a
                # planted cluster's transactions has found it. Requiring every transaction would fail
                # an instance for one leg falling outside the window; requiring one would pass a
                # candidate that clipped the edge of it.
                "detected_when": "a candidate covers >= 0.5 of txn_refs",
                "label_source": "SAML-D Is_laundering + Laundering_type (derived, not authored)",
                **({"supply_limited": note} if note and len(chosen) < PER_PATTERN else {}),
            })
    return records


def clean_batch() -> dict[str, Any]:
    if not CLEAN_BATCH.exists():
        raise SystemExit(f"{CLEAN_BATCH} is missing -- run: uv run finguard-ledger --profile dev")
    return {
        "id": "CB-001",
        "batch": str(CLEAN_BATCH.relative_to(PROJECT_ROOT)),
        "transactions": sum(
            1 for line in CLEAN_BATCH.read_text().splitlines() if line.startswith(":20:")
        ),
        "expect": {
            "candidates": 0,
            "llm_calls": 0,
            "cost_usd": 0.0,
            "report": {"clean": True, "risk_rating": "none", "findings": 0},
        },
        "why": (
            "Verifies the detectors produce zero candidates, the graph makes zero LLM calls, and the "
            "system emits an empty-but-valid report rather than inventing a finding. The one corpus "
            "shared with the dev set: it is the batch every '$0.0000 on a clean month' claim was "
            "already measured on."
        ),
        "label_source": "SAML-D Is_laundering == 0 for every row (derived, not authored)",
    }


def write(name: str, payload: Any) -> Path:
    path = DATASETS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed datasets differ from a fresh build")
    args = parser.parse_args()

    built = {"labeled_patterns": labeled_patterns(), "clean_batch": clean_batch()}

    if args.check:
        drifted = []
        for name, payload in built.items():
            path = DATASETS / f"{name}.json"
            if not path.exists() or json.loads(path.read_text()) != payload:
                drifted.append(name)
        if drifted:
            print(f"derived datasets differ from the corpus: {', '.join(drifted)}")
            print("run: uv run python -m eval.build_datasets")
            return 1
        print("derived datasets match the corpus")
        return 0

    DATASETS.mkdir(parents=True, exist_ok=True)
    for name, payload in built.items():
        path = write(name, payload)
        count = len(payload) if isinstance(payload, list) else 1
        print(f"{path.relative_to(PROJECT_ROOT)}: {count} record(s)")

    counts: dict[str, int] = defaultdict(int)
    for record in built["labeled_patterns"]:
        counts[record["pattern_type"]] += 1
    print("  " + " · ".join(f"{p} {n}" for p, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
