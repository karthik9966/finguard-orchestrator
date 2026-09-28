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
import time
from functools import lru_cache
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import get_config
from src.ingestion.embeddings import BACKENDS, MissingCredentials, get_backend

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"

PERSIST_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "regulations")

# Value now lives in config.yaml (LLD §8). Shim for the migration -- Phase 5 deletes it.
UPSERT_BATCH = get_config().ingestion.upsert_batch
DEFAULT_K = get_config().retrieval.k_indicators

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


# --- VectorStoreClient (LLD §2.1, §3.2, §6) -------------------------------------------
#
# A class rather than more module functions because the new design needs two collections at once
# -- `rule_chunks` for citable US law and `obliqa_benchmark` for the reproducible hit@k arm --
# and because `resolve()` has to exist before Phase 1c can curate an obligation map against it.
#
# The module functions below stay as they are and delegate here: they have live callers in the
# cockpit, the benchmark and 300-odd tests, and the migration's promise is that the suite is green
# at the end of every phase. Phase 5 removes the wrappers when the last caller goes.

RULE_COLLECTION = "rule_chunks"
TOPIC_SEPARATOR = "|"

# LLD §6: VECTOR_STORE_UNAVAILABLE is retryable -- back off, and if it stays down fail the job
# with a clear error. The one thing not to do is proceed: a run that cannot retrieve cannot
# ground, and an ungrounded finding is the failure this system exists to prevent.
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.5


class VectorStoreUnavailable(RuntimeError):
    """The store could not be reached after retrying. Fail the job; never fabricate a citation."""


def pack_topics(tags: list[str] | str | None) -> str:
    """Chroma metadata is scalar-only, so ``topic_tags`` travels as a delimited string."""
    if isinstance(tags, str):
        return tags
    return TOPIC_SEPARATOR.join(tags or [])


def unpack_topics(packed: object) -> list[str]:
    if not packed or not isinstance(packed, str):
        return []
    return [tag for tag in packed.split(TOPIC_SEPARATOR) if tag]


class VectorStoreClient:
    """Reads and writes one Chroma collection.

    Every call goes through :meth:`_with_retry`, so a store that is briefly unreachable costs a
    pause rather than the run.
    """

    def __init__(self, collection: str = RULE_COLLECTION, *, backend_name: str = "minilm") -> None:
        self.collection_name = collection
        self.backend_name = backend_name

    # --- plumbing ----------------------------------------------------------------------

    def _with_retry(self, what: str, call):
        last: Exception | None = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return call()
            except Exception as error:  # noqa: BLE001 - re-raised below as a typed failure
                last = error
                if attempt + 1 < RETRY_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
        raise VectorStoreUnavailable(
            f"{what} failed after {RETRY_ATTEMPTS} attempts against "
            f"{self.collection_name!r}: {last}"
        ) from last

    def _collection(self, *, create: bool = False, records: int = 0):
        client = _client()
        if not create:
            return client.get_collection(self.collection_name)
        with get_backend(self.backend_name) as backend:
            existing = next(
                (c for c in client.list_collections() if c.name == self.collection_name), None
            )
            if existing is not None and existing.metadata.get("backend") not in (
                None, self.backend_name
            ):
                raise BackendMismatch(
                    f"collection {self.collection_name!r} was built with "
                    f"{existing.metadata['backend']!r}; rebuild to replace it"
                )
            return client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "backend": self.backend_name,
                    "model": backend.model_id,
                    "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "chunks": records,
                },
            )

    # --- writing -----------------------------------------------------------------------

    def upsert(self, records: list[dict], vectors) -> int:
        """Write chunks in batches. ``records`` are RuleChunk-shaped dicts."""
        collection = self._collection(create=True, records=len(records))
        for start in range(0, len(records), UPSERT_BATCH):
            block = records[start : start + UPSERT_BATCH]
            metadatas = []
            for record in block:
                meta = clean_metadata({**record, "topic_tags": pack_topics(record.get("topic_tags"))})
                metadatas.append(meta)
            self._with_retry(
                "upsert",
                lambda block=block, metadatas=metadatas, start=start: collection.upsert(
                    ids=[r["chunk_id"] for r in block],
                    embeddings=vectors[start : start + UPSERT_BATCH].tolist(),
                    documents=[r["text"] for r in block],
                    metadatas=metadatas,
                ),
            )
        return len(records)

    # --- reading -----------------------------------------------------------------------

    def similarity_search(self, embedding, *, where: dict | None = None, k: int = 15) -> list[dict]:
        """Nearest chunks, filtered by **arbitrary** metadata.

        The old module-level ``retrieve`` could only filter ``relevance_tier`` with ``$in``. The
        new retrieval model filters on tier, authority and topic at once, so the filter is the
        caller's to compose.
        """
        collection = self._collection()
        result = self._with_retry(
            "similarity_search",
            # float() per element: a numpy row arrives as np.float32 values, which Chroma
            # rejects with a message about lists of floats -- true, and unhelpfully so.
            lambda: collection.query(
                query_embeddings=[[float(value) for value in embedding]],
                n_results=k,
                where=where or None,
            ),
        )
        return [
            {
                "chunk_id": cid,
                "text": document,
                "score": 1.0 - distance,
                "distance": distance,
                **{**meta, "topic_tags": unpack_topics(meta.get("topic_tags"))},
            }
            for cid, document, distance, meta in zip(
                result["ids"][0],
                result["documents"][0], # type: ignore
                result["distances"][0], # type: ignore
                result["metadatas"][0], # type: ignore
            )
        ]

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Fetch by id, for the citations drawer and for resolved obligations."""
        if not chunk_ids:
            return {}
        collection = self._collection()
        found = self._with_retry(
            "get_by_ids",
            lambda: collection.get(
                ids=list(dict.fromkeys(chunk_ids)), include=["documents", "metadatas"]
            ),
        )
        return {
            cid: {
                "chunk_id": cid,
                "text": document,
                **{**(meta or {}), "topic_tags": unpack_topics((meta or {}).get("topic_tags"))},
            }
            for cid, document, meta in zip(
                found["ids"], found["documents"] or [], found["metadatas"] or [] # type: ignore
            )
        }

    def resolve(self, reference: tuple[str, str]) -> str | None:
        """``(source_id, section_ref)`` → ``chunk_id``, or None if it is not in the store.

        Phase 1c curates the obligation map as these pairs rather than as literal ids precisely
        because ``chunk_id = hash(source_id, section_ref, version)``: a re-chunk or a version bump
        invalidates a literal id with no error, and per LLD §6 an OBLIGATION_MAP_MISS disables
        grounding for a whole typology. Resolving at load turns that into a startup failure.
        """
        source_id, section_ref = reference
        collection = self._collection()
        found = self._with_retry(
            "resolve",
            lambda: collection.get(
                where={"$and": [{"source_id": source_id}, {"section_ref": section_ref}]},
                include=["metadatas"],
                limit=1,
            ),
        )
        ids = found.get("ids") or []
        return ids[0] if ids else None

    def counts(self) -> dict[str, dict[str, int]]:
        """Chunk counts per tier and per authority -- Phase 1b's green criterion."""
        collection = self._collection()
        everything = self._with_retry(
            "counts", lambda: collection.get(include=["metadatas"])
        )
        tiers: Counter[str] = Counter()
        authorities: Counter[str] = Counter()
        for meta in everything["metadatas"] or []:
            tiers[str(meta.get("tier", "-"))] += 1
            authorities[str(meta.get("authority", "-"))] += 1
        return {"tier": dict(tiers), "authority": dict(authorities), "total": collection.count()}


BENCHMARK_COLLECTION = "obliqa_benchmark"


def build_benchmark(backend_name: str = "minilm", *, rebuild: bool = False) -> dict:
    """Index ObliQA into its own collection, away from the citable corpus.

    ObliQA is ADGM law and the new design puts non-US rulebooks out of scope, but the 2,786
    labelled questions are the only ground truth this project has for retrieval quality -- the
    reranker was adopted on them, and hit@1 45.2% -> 55.6% is a regression floor. So it stays,
    fenced off: `rule_chunks` cannot serve an ADGM clause as a citation, and the benchmark keeps
    reproducing.
    """
    source = chunk_path(backend_name)
    if not source.exists():
        raise SystemExit(f"{source.name} missing -- run: uv run finguard-chunk --backend {backend_name}")
    records = [
        record
        for line in source.read_text().splitlines()
        if (record := json.loads(line))["corpus"] == "obliqa"
    ]
    if not records:
        raise SystemExit(f"{source.name} holds no ObliQA chunks")

    store = VectorStoreClient(BENCHMARK_COLLECTION, backend_name=backend_name)
    if rebuild:
        try:
            _client().delete_collection(BENCHMARK_COLLECTION)
            print(f"  dropped existing collection {BENCHMARK_COLLECTION!r}")
        except Exception:  # noqa: BLE001 - absent collection is the normal case
            pass

    with get_backend(backend_name) as backend:
        print(f"  embedding {len(records):,} ObliQA chunks with {backend.model_id} ...")
        vectors = backend.encode([r["text"] for r in records])

    store.upsert(records, vectors)
    total = store.counts()["total"]
    print(f"  {BENCHMARK_COLLECTION}: {total:,} chunks")
    return {"collection": BENCHMARK_COLLECTION, "chunks": total}


def build_rules(backend_name: str = "minilm", *, rebuild: bool = False) -> dict:
    """Index the citable US corpus into `rule_chunks`.

    Every record is validated as a ``RuleChunk`` before it is written. The model rejects a
    non-US jurisdiction on the model itself, so an ADGM clause cannot reach a citation even if a
    metadata filter is later written wrongly -- which is the failure this collection exists to
    make impossible.
    """
    from src.ingestion.loader import rules_path
    from src.models import RuleChunk

    source = rules_path(backend_name)
    if not source.exists():
        raise SystemExit(
            f"{source.name} missing -- run: uv run finguard-chunk --rules --backend {backend_name}"
        )
    records = [json.loads(line) for line in source.read_text().splitlines()]
    for record in records:
        RuleChunk(**{k: v for k, v in record.items() if k != "source_file"})

    store = VectorStoreClient(RULE_COLLECTION, backend_name=backend_name)
    if rebuild:
        try:
            _client().delete_collection(RULE_COLLECTION)
            print(f"  dropped existing collection {RULE_COLLECTION!r}")
        except Exception:  # noqa: BLE001 - absent collection is the normal case
            pass

    with get_backend(backend_name) as backend:
        print(f"  embedding {len(records):,} rule chunks with {backend.model_id} ...")
        vectors = backend.encode([r["text"] for r in records])

    store.upsert(records, vectors)
    counts = store.counts()
    print(f"  {RULE_COLLECTION}: {counts['total']:,} chunks")
    print(f"    by tier      {counts['tier']}")
    print(f"    by authority {counts['authority']}")
    return counts


@lru_cache(maxsize=1)
def collection_backend() -> str:
    """Which embedding model built the collection.

    Cached: §9.3's cache key needs it on every lookup, and re-reading collection metadata per
    query would cost more than the cache saves. A rebuild changes the process, so a process-
    lifetime cache is the right lifetime.
    """
    try:
        return str(_client().get_collection(COLLECTION_NAME).metadata.get("backend") or "unknown")
    except Exception:  # noqa: BLE001 - an absent collection is handled by the callers that care
        return "unknown"


def by_id(chunk_ids: list[str]) -> dict[str, dict]:
    """Fetch stored chunks by id -- the lookup §6.4's citations drawer needs.

    ``retrieve`` searches by vector and is the wrong tool here: the drawer already knows exactly
    which clauses to show, because ``generate_node`` derived ``source_document_hashes`` from the
    retrieved set in Python rather than trusting the model to report them. Re-searching would
    risk returning a *different* clause than the one the report was actually grounded in, which
    defeats the entire purpose of an audit trail.

    Returned as a dict keyed by chunk_id so a caller can render citations in the report's own
    order. Ids that are not in the collection are simply absent -- the caller decides whether a
    missing citation is worth shouting about, and the drawer says so plainly rather than
    rendering an empty card.
    """
    if not chunk_ids:
        return {}

    collection = _client().get_collection(COLLECTION_NAME)
    found = collection.get(ids=list(dict.fromkeys(chunk_ids)), include=["documents", "metadatas"])
    return {
        cid: {"chunk_id": cid, "text": doc, **(meta or {})}
        for cid, doc, meta in zip(
            found["ids"], found["documents"] or [], found["metadatas"] or [] # type: ignore
        )
    }


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
    parser.add_argument(
        "--rules", action="store_true", help=f"build {RULE_COLLECTION!r} from the US corpus"
    )
    parser.add_argument(
        "--rule-stats", action="store_true", help=f"chunk counts per tier in {RULE_COLLECTION!r}"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help=f"build {BENCHMARK_COLLECTION!r} -- ObliQA, fenced off from the citable corpus",
    )
    parser.add_argument("--query", help="run a retrieval and print the hits")
    parser.add_argument("--tier", type=int, nargs="*", default=None)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    try:
        if args.rules:
            build_rules(args.backend, rebuild=args.rebuild)
            return 0

        if args.benchmark:
            build_benchmark(args.backend, rebuild=args.rebuild)
            return 0

        if args.rule_stats:
            counts = VectorStoreClient(RULE_COLLECTION, backend_name=args.backend).counts()
            print(f"{RULE_COLLECTION}: {counts['total']:,} chunks")
            for label in ("tier", "authority"):
                for key, value in sorted(counts[label].items()):
                    print(f"  {label:<10} {key:<14} {value:>5,}")
            return 0

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
