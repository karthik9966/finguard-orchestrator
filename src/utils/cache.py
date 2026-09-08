"""Semantic cache for regulatory retrieval (§9.3).

Caches one thing: the reranked clause list a query returns. Measured on this pipeline, that is
where the audit node's time goes -- ``retrieve()`` is 1.73s per query against ``rerank()``'s
0.04s, and a batch asks eight of them, so roughly 14 of the run's 45 seconds are spent embedding
sentences and scanning 12,273 vectors for answers we have already computed.

It works here for a reason particular to this design: the queries are **ten fixed templates**
selected by a dict lookup on the candidate's shape, not written by a model. Across the four
sample batches, 23 query executions resolve to 9 distinct strings -- 61% repeats before the cache
exists at all, rising toward 100% as more batches run. A pipeline that had a model rewrite its
queries per batch would cache almost nothing.

**Retrieval only. Reports are deliberately not cached.** A report narrative names real account
numbers and real amounts -- the June one carries 3 accounts and 11 figures -- so serving a
"similar enough" cached report would put another batch's identifiers into a regulatory filing.
Retrieved clauses carry no such risk: what the rulebook requires about structuring is the same
answer regardless of which batch asked. That asymmetry, not squeamishness, is why the blueprint's
single similarity threshold is split here into "semantic for clauses, never for findings".

So this saves **time, not money**. Retrieval was already free; the model calls it protects are
untouched.

**Invalidation is TTL alone.** Consequence, stated rather than buried: re-index the corpus with
``finguard-store`` and cached entries can serve clauses from the *old* index until they expire.
Run ``--flush`` after a rebuild.

Usage::

    docker run -d -p 6379:6379 --name finguard-redis redis:7-alpine
    uv run python -m src.utils.cache --stats
    uv run python -m src.utils.cache --flush
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

log = logging.getLogger(__name__)

NAMESPACE = "finguard"
HITS_KEY = f"{NAMESPACE}:hits:{{sha}}"
VECTOR_KEY = f"{NAMESPACE}:vec:{{sha}}"
INDEX_KEY = f"{NAMESPACE}:queries"

TTL_SECONDS = int(os.environ.get("REDIS_CACHE_TTL", 24 * 60 * 60))
THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", 0.95))
CONNECT_TIMEOUT = 1.0  # a cache that stalls an audit is worse than no cache


def fingerprint(query: str, tiers: list[int] | None, k: int, backend: str) -> str:
    """Identity of a lookup.

    ``backend`` is in the key because minilm and openai vectors are not comparable -- the same
    mistake ``store.retrieve`` guards with BackendMismatch. ``tiers`` is in it because the tier
    filter changes which clauses come back for an identical question.
    """
    tier_part = ",".join(str(t) for t in sorted(tiers)) if tiers else "all"
    raw = f"{backend}|{k}|{tier_part}|{query.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class CacheStats:
    """Per-run tally, reported in the cockpit's §6.5 panel."""

    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    seconds_saved: float = 0.0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def summary(self) -> str:
        if not self.lookups:
            return "cache      : not consulted"
        if not self.hits:
            return f"cache      : 0/{self.lookups} hits (cold)"
        return (
            f"cache      : {self.hits}/{self.lookups} hits ({self.hit_rate:.0%}), "
            f"{self.semantic_hits} semantic, ~{self.seconds_saved:.1f}s saved"
        )


@dataclass
class RetrievalCache:
    """Redis-backed, and inert when Redis is not there.

    Enabled by presence rather than by a flag: no reachable server means no caching, silently and
    correctly. Every call is wrapped -- an audit must never fail because a cache did. That is not
    defensive habit, it is the whole risk profile of adding one.
    """

    client: Any = None
    stats: CacheStats = field(default_factory=CacheStats)
    _warned: bool = False

    # Class-level, so one failed connection spares every later run the same timeout. `reset()`
    # exists for tests and for the case where Redis is started after the process was.
    _unreachable: ClassVar[bool] = False

    @classmethod
    def reset(cls) -> None:
        cls._unreachable = False

    @classmethod
    def connect(cls, client: Any = None) -> RetrievalCache:
        """A cache handle for one run. Never raises; an unreachable server yields an inert one.

        The failed-connection verdict is remembered for the process, because it is not free to
        re-learn: `audit_node` builds a cache per call, and with Redis down each attempt burned
        the full CONNECT_TIMEOUT. That took the test suite from 30s to 96s -- and in production
        it would add a second of dead waiting to every refinement pass. Learn it once.
        """
        if client is not None:
            return cls(client=client)
        if cls._unreachable:
            return cls(client=None)
        try:
            import redis

            handle = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                db=int(os.environ.get("REDIS_DB", 0)),
                socket_connect_timeout=CONNECT_TIMEOUT,
                socket_timeout=CONNECT_TIMEOUT,
            )
            handle.ping()
            return cls(client=handle)
        except Exception as error:  # noqa: BLE001 - absence is the ordinary case, not a fault
            log.debug("retrieval cache disabled: %s", error)
            cls._unreachable = True
            return cls(client=None)

    @property
    def available(self) -> bool:
        return self.client is not None

    def _degrade(self, error: Exception) -> None:
        """Report the first failure, then stay quiet: one warning is a signal, eight is noise."""
        if not self._warned:
            log.warning("retrieval cache unavailable, continuing uncached: %s", error)
            self._warned = True
        self.client = None

    # --- reading ------------------------------------------------------------------------

    def get(self, sha: str, query: str) -> list[dict] | None:
        """Exact match first, then semantic. Returns None on a miss or any failure."""
        if not self.available:
            return None
        try:
            raw = self.client.get(HITS_KEY.format(sha=sha))
            if raw is not None:
                self.stats.exact_hits += 1
                return self._unwrap(raw)

            found = self._semantic(query)
            if found is not None:
                self.stats.semantic_hits += 1
                return found
        except Exception as error:  # noqa: BLE001
            self._degrade(error)
            return None

        self.stats.misses += 1
        return None

    def _semantic(self, query: str) -> list[dict] | None:
        """Cosine-compare against every cached query vector.

        A warm cache is ~20 entries, so a full scan in Python is microseconds -- faster than the
        extra round trip a Redis vector index would cost. Redis Stack is the blueprint-literal
        choice and is not worth a heavier image at this key count.

        Only the command bar reaches here. The ten templates are fixed strings and always match
        exactly, which is why the embedding is computed lazily, on a miss, rather than up front.
        """
        shas = self.client.smembers(INDEX_KEY)
        if not shas:
            return None

        vector = _embed(query)
        if vector is None:
            return None

        best_sha, best_score = None, 0.0
        for member in shas:
            stored = self.client.get(VECTOR_KEY.format(sha=member.decode()))
            if stored is None:
                continue  # expired; the index entry is tidied on the next write
            other = np.frombuffer(stored, dtype=np.float32)
            if other.shape != vector.shape:
                continue
            score = float(vector @ other)  # both are L2-normalised, so this is cosine
            if score > best_score:
                best_sha, best_score = member.decode(), score

        if best_sha is None or best_score < THRESHOLD:
            return None
        raw = self.client.get(HITS_KEY.format(sha=best_sha))
        return self._unwrap(raw) if raw else None

    def _unwrap(self, raw: bytes) -> list[dict]:
        """Read an entry and credit its stored computation time to this run's saving."""
        payload = json.loads(raw)
        self.stats.seconds_saved += float(payload.get("elapsed", 0.0))
        return payload["hits"]

    # --- writing ------------------------------------------------------------------------

    def put(self, sha: str, query: str, hits: list[dict], elapsed: float = 0.0) -> None:
        """Store the hits, and what computing them cost.

        The elapsed time is stored with the entry so a later hit can report what it actually
        saved. Deriving it from the current run instead only works while something still misses:
        a fully warm run has no uncached lookup left to measure, and would report 0.0s saved
        while saving the most.
        """
        if not self.available or not hits:
            return
        try:
            payload = {"hits": hits, "elapsed": round(elapsed, 4)}
            pipe = self.client.pipeline()
            # set(ex=) rather than setex(): redis-py 8 deprecated the latter.
            pipe.set(HITS_KEY.format(sha=sha), json.dumps(payload), ex=TTL_SECONDS)
            vector = _embed(query)
            if vector is not None:
                pipe.set(VECTOR_KEY.format(sha=sha), vector.tobytes(), ex=TTL_SECONDS)
                pipe.sadd(INDEX_KEY, sha)
                pipe.expire(INDEX_KEY, TTL_SECONDS)
            pipe.execute()
        except Exception as error:  # noqa: BLE001
            self._degrade(error)

    def flush(self) -> int:
        """Drop everything. Run this after `finguard-store` re-indexes the corpus."""
        if not self.available:
            return 0
        keys = list(self.client.scan_iter(match=f"{NAMESPACE}:*"))
        if keys:
            self.client.delete(*keys)
        return len(keys)

    def entries(self) -> int:
        if not self.available:
            return 0
        return len(list(self.client.scan_iter(match=HITS_KEY.format(sha="*"))))


def _embed(query: str) -> np.ndarray | None:
    """L2-normalised query vector, from the same model the collection was built with."""
    try:
        from src.ingestion.embeddings import get_backend

        with get_backend("minilm") as backend:
            vector = np.asarray(backend.encode([query])[0], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else None
    except Exception as error:  # noqa: BLE001 - semantic matching is optional, exact is not
        log.debug("cache embedding failed: %s", error)
        return None


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flush", action="store_true", help="drop every cached entry")
    parser.add_argument("--stats", action="store_true", help="show what is cached")
    args = parser.parse_args()

    cache = RetrievalCache.connect()
    if not cache.available:
        print("redis   : unreachable -- the pipeline runs uncached")
        print("start it: docker run -d -p 6379:6379 --name finguard-redis redis:7-alpine")
        return 1

    if args.flush:
        print(f"flushed : {cache.flush()} key(s)")
        return 0

    print(f"redis   : {os.environ.get('REDIS_HOST', 'localhost')}:"
          f"{os.environ.get('REDIS_PORT', 6379)} db {os.environ.get('REDIS_DB', 0)}")
    print(f"entries : {cache.entries()} cached retrievals")
    print(f"ttl     : {TTL_SECONDS}s ({TTL_SECONDS / 3600:.0f}h)")
    print(f"cutoff  : {THRESHOLD} cosine for a semantic hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
