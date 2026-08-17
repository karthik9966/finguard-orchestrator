"""Score a chunking strategy against ObliQA's labelled questions (§3.3).

"Is this chunked well?" is unanswerable by inspection. ObliQA ships 2,786 test questions, each
labelled with the passages that answer it -- 3,666 gold pairs, all of which resolve against the
corpus on disk. That turns chunking from taste into measurement.

A retrieved chunk counts as a hit when its ``(document_id, section_clause)`` is one of the
question's gold passages. Chunks split from an oversized passage all share that key, so a
question is satisfied by retrieving any part of the right clause.

Reported per backend:

* **hit@k**    -- share of questions with at least one gold passage in the top *k*
* **recall@k** -- share of *each question's* gold passages found in the top *k*, averaged
* **MRR**      -- mean reciprocal rank of the first gold hit (0 when none is found)

Usage::

    uv run python -m src.ingestion.benchmark
    uv run python -m src.ingestion.benchmark --backend both --tier 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ingestion.embeddings import BACKENDS, MissingCredentials, get_backend

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHUNK_DIR = DATA_DIR / "processed" / "chunks"
GOLD_PATH = DATA_DIR / "raw" / "regulations" / "obliqa" / "ObliQA_test.json"
RESULTS_PATH = DATA_DIR / "processed" / "retrieval_benchmark.json"

CUTOFFS = (1, 5, 15)
QUERY_BATCH = 256


def load_chunks(backend_name: str, tiers: set[int] | None) -> list[dict]:
    path = CHUNK_DIR / f"{backend_name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run: uv run python -m src.ingestion.loader")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if tiers is not None:
        records = [r for r in records if r["relevance_tier"] in tiers]
    return records


def load_gold() -> list[dict]:
    if not GOLD_PATH.exists():
        raise SystemExit(f"{GOLD_PATH} missing -- run: uv run python -m src.ingestion.download")
    return json.loads(GOLD_PATH.read_text())


def evaluate(backend_name: str, tiers: set[int] | None) -> dict:
    chunks = load_chunks(backend_name, tiers)
    questions = load_gold()

    # A chunk's identity for scoring is the clause it came from, not the chunk itself.
    chunk_keys = [
        (record["document_id"], record["section_clause"]) if record["corpus"] == "obliqa" else None
        for record in chunks
    ]

    with get_backend(backend_name) as backend:
        print(f"  embedding {len(chunks):,} chunks with {backend.model_id} ...")
        matrix = backend.encode([record["text"] for record in chunks])
        print(f"  embedding {len(questions):,} questions ...")
        queries = backend.encode([q["Question"] for q in questions])

    max_k = max(CUTOFFS)
    hits = {k: 0 for k in CUTOFFS}
    recall = {k: 0.0 for k in CUTOFFS}
    reciprocal = 0.0

    for start in range(0, len(queries), QUERY_BATCH):
        block = queries[start : start + QUERY_BATCH]
        scores = block @ matrix.T
        # argpartition finds the top-k without sorting all 12k scores, then we sort just those.
        top = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        for row, indices in enumerate(top):
            ordered = indices[np.argsort(-scores[row, indices])]
            gold = {
                (passage["DocumentID"], passage["PassageID"])
                for passage in questions[start + row]["Passages"]
            }
            ranked = [chunk_keys[i] for i in ordered]

            first = next((rank for rank, key in enumerate(ranked, 1) if key in gold), None)
            if first is not None:
                reciprocal += 1.0 / first
            for k in CUTOFFS:
                found = {key for key in ranked[:k] if key in gold}
                if found:
                    hits[k] += 1
                recall[k] += len(found) / len(gold)

    total = len(questions)
    return {
        "backend": backend_name,
        "model": get_backend(backend_name).model_id,
        "chunks": len(chunks),
        "questions": total,
        "tiers": sorted(tiers) if tiers else "all",
        "hit_at": {str(k): hits[k] / total for k in CUTOFFS},
        "recall_at": {str(k): recall[k] / total for k in CUTOFFS},
        "mrr": reciprocal / total,
    }


def print_table(results: list[dict]) -> None:
    header = f"{'backend':<10}{'chunks':>9}{'hit@1':>9}{'hit@5':>9}{'hit@15':>9}"
    header += f"{'rec@5':>9}{'rec@15':>9}{'MRR':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['backend']:<10}{r['chunks']:>9,}"
            f"{r['hit_at']['1']:>9.1%}{r['hit_at']['5']:>9.1%}{r['hit_at']['15']:>9.1%}"
            f"{r['recall_at']['5']:>9.1%}{r['recall_at']['15']:>9.1%}{r['mrr']:>9.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="both", choices=[*BACKENDS, "both"])
    parser.add_argument(
        "--tier", type=int, nargs="*", default=None, help="restrict retrieval to these tiers"
    )
    args = parser.parse_args()

    tiers = set(args.tier) if args.tier else None
    names = list(BACKENDS) if args.backend == "both" else [args.backend]

    results = []
    for name in names:
        print(f"\n{name}:")
        try:
            results.append(evaluate(name, tiers))
        except MissingCredentials as error:
            print(f"  SKIPPED -- {error}")
        except SystemExit as error:
            print(f"  SKIPPED -- {error}")

    if not results:
        return 1

    print_table(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    RESULTS_PATH.write_text(json.dumps([*existing, *results], indent=2) + "\n")
    print(f"\nResults -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
