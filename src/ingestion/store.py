"""Load §3.3's chunks into ChromaDB and serve retrieval to the agent (§3.4).

One collection, ``regulations``, holding all 46 regulatory documents. Transactions are
deliberately *not* embedded -- measured on 220 MT103s, laundering and clean wires separate by
+0.029 cosine, i.e. noise. Parsed wires belong in a table queried with SQL.

The collection records which embedding backend built it. Querying with a different one is a
silent-nonsense bug -- 384-dim MiniLM vectors and 1536-dim OpenAI vectors describe the same text
in incompatible coordinate spaces -- so ``retrieve`` raises instead. That guard is also what
makes swapping backends later a one-command rebuild rather than a refactor.

Usage::

    uv run python -m src.ingestion.store                     # build from minilm chunks
    uv run python -m src.ingestion.store --backend openai --rebuild
    uv run python -m src.ingestion.store --stats
    uv run python -m src.ingestion.store --query "..." --tier 1 2
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.ingestion.embeddings import BACKENDS, MissingCredentials, get_backend

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"

PERSIST_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "regulations")

UPSERT_BATCH = 1000
DEFAULT_K = 15

# Chroma stores the text in `documents` and everything else in `metadatas`; these two are not
# metadata. `passage_uuid` stays -- it is ObliQA's real primary key and worth keeping for tracing.
NOT_METADATA = frozenset({"chunk_id", "text"})


class BackendMismatch(RuntimeError):
    """Raised when a collection is queried with a different model than built it."""


def _client() -> chromadb.ClientAPI: # type: ignore
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(PERSIST_DIR), settings=Settings(anonymized_telemetry=False)
    )


def clean_metadata(record: dict) -> dict:
    """Chroma metadata takes only str/int/float/bool.

    Nulls are *silently dropped* rather than rejected, so strip them here instead: an absent key
    behaves correctly in a ``where`` filter, whereas the string "None" would match nothing and
    look like data. Our records carry nulls in `last_updated_date` (185 chunks from the four
    genuinely undated documents), `document_id` (all FINRA/FinCEN chunks) and `part`.
    """
    return {
        key: value
        for key, value in record.items()
        if key not in NOT_METADATA and value is not None
    }


def chunk_path(backend_name: str) -> Path:
    path = CHUNK_DIR / f"{backend_name}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} missing -- run: uv run python -m src.ingestion.loader --backend {backend_name}"
        )
    return path


def build(backend_name: str = "minilm", *, rebuild: bool = False) -> dict:
    records = [json.loads(line) for line in chunk_path(backend_name).read_text().splitlines()]
    client = _client()

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  dropped existing collection {COLLECTION_NAME!r}")
        except Exception:  # noqa: BLE001 - absent collection is the normal case
            pass

    with get_backend(backend_name) as backend:
        existing = next(
            (c for c in client.list_collections() if c.name == COLLECTION_NAME), None
        )
        if existing is not None and existing.metadata.get("backend") not in (None, backend_name):
            raise BackendMismatch(
                f"collection {COLLECTION_NAME!r} was built with "
                f"{existing.metadata['backend']!r}; re-run with --rebuild to replace it"
            )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                # Unit-normalized vectors make L2 and cosine rank identically, but asking for
                # cosine keeps the reported distances interpretable (0 = same, 1 = unrelated).
                "hnsw:space": "cosine",
                "backend": backend_name,
                "model": backend.model_id,
                "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "chunks": len(records),
            },
        )

        print(f"  embedding {len(records):,} chunks with {backend.model_id} ...")
        vectors = backend.encode([r["text"] for r in records])

        for start in range(0, len(records), UPSERT_BATCH):
            block = records[start : start + UPSERT_BATCH]
            collection.upsert(
                ids=[r["chunk_id"] for r in block],
                embeddings=vectors[start : start + UPSERT_BATCH].tolist(),
                documents=[r["text"] for r in block],
                metadatas=[clean_metadata(r) for r in block],
            )
            print(f"    upserted {min(start + UPSERT_BATCH, len(records)):>6,}/{len(records):,}")

    return stats()


def retrieve(
    query: str,
    *,
    k: int = DEFAULT_K,
    tiers: list[int] | None = None,
    backend_name: str | None = None,
) -> list[dict]:
    """Top-``k`` regulatory chunks for ``query``. This is what §4's AML Audit node calls.

    ``query`` should be an *obligation-shaped question* ("transactions structured to avoid
    reporting thresholds"), not a description of what the transactions did. Measured on this
    corpus, the obligation phrasing ranked the target clause 5th where a narrative of the same
    facts ranked it 315th -- rulebooks are written as duties, so descriptions of events share
    no register with them.
    """
    client = _client()
    collection = client.get_collection(COLLECTION_NAME)
    built_with = collection.metadata.get("backend")
    backend_name = backend_name or built_with

    if backend_name != built_with:
        raise BackendMismatch(
            f"collection was built with {built_with!r} but queried with {backend_name!r}; "
            "their vector spaces are not comparable"
        )

    with get_backend(backend_name) as backend: # type: ignore
        vector = backend.encode([query])[0].tolist()

    where = {"relevance_tier": {"$in": list(tiers)}} if tiers else None
    result = collection.query(query_embeddings=[vector], n_results=k, where=where) # type: ignore

    return [
        {"chunk_id": cid, "text": doc, "distance": dist, **meta}
        for cid, doc, dist, meta in zip(
            result["ids"][0],
            result["documents"][0], # type: ignore
            result["distances"][0], # type: ignore
            result["metadatas"][0], # type: ignore
        )
    ]


def stats() -> dict:
    client = _client()
    collection = client.get_collection(COLLECTION_NAME)
    everything = collection.get(include=["metadatas"])
    metadatas = everything["metadatas"]
    return {
        "collection": COLLECTION_NAME,
        "vectors": collection.count(),
        "backend": collection.metadata.get("backend"),
        "model": collection.metadata.get("model"),
        "built": collection.metadata.get("built"),
        "by_corpus": dict(Counter(m["corpus"] for m in metadatas)), # type: ignore
        "by_tier": dict(sorted(Counter(m["relevance_tier"] for m in metadatas).items())), # type: ignore
        "undated": sum(1 for m in metadatas if "last_updated_date" not in m), # type: ignore
    }


def inventory() -> list[dict]:
    """One row per loaded regulation, for §6.1's "so the auditor knows which rules are active"."""
    client = _client()
    collection = client.get_collection(COLLECTION_NAME)
    rows: dict[str, dict] = {}
    for meta in collection.get(include=["metadatas"])["metadatas"]: # type: ignore
        row = rows.setdefault(
            meta["document_title"], # type: ignore
            {
                "document": meta["document_title"],
                "corpus": meta["corpus"],
                "tier": meta["relevance_tier"],
                "jurisdiction": meta["jurisdiction"],
                "updated": meta.get("last_updated_date"),
                "chunks": 0,
            },
        )
        row["chunks"] += 1
    return sorted(rows.values(), key=lambda r: (r["tier"], -r["chunks"]))


def print_stats(payload: dict) -> None:
    print(f"\ncollection : {payload['collection']}  ({payload['vectors']:,} vectors)")
    print(f"model      : {payload['model']}  [{payload['backend']}]")
    print(f"built      : {payload['built']}")
    print(f"by corpus  : {payload['by_corpus']}")
    print(f"by tier    : {payload['by_tier']}")
    print(f"undated    : {payload['undated']:,} chunks carry no last_updated_date")


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="minilm", choices=list(BACKENDS))
    parser.add_argument("--rebuild", action="store_true", help="drop the collection first")
    parser.add_argument("--stats", action="store_true", help="report on the existing collection")
    parser.add_argument("--query", help="run a retrieval and print the hits")
    parser.add_argument("--tier", type=int, nargs="*", default=None)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    try:
        if args.stats:
            print_stats(stats())
            return 0

        if args.query:
            for rank, hit in enumerate(
                retrieve(args.query, k=args.k, tiers=args.tier), start=1
            ):
                print(
                    f"{rank:>3}. [{hit['distance']:.3f}] {hit['document_title']} "
                    f"- {hit['section_clause']}  (tier {hit['relevance_tier']})"
                )
                print(f"     {hit['text'][:140].replace(chr(10), ' ')}...")
            return 0

        print_stats(build(args.backend, rebuild=args.rebuild))
        return 0
    except MissingCredentials as error:
        print(f"SKIPPED -- {error}")
        return 1
    except BackendMismatch as error:
        print(f"ERROR -- {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
