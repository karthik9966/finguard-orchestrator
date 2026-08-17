"""Embedding backends for §3.3/§3.4, behind one interface so they can be compared.

Two candidates, deliberately kept swappable until the ObliQA gold set says which is better:

* ``minilm``  -- ``all-MiniLM-L6-v2`` via sentence-transformers. 384-dim, local, free, offline.
* ``openai``  -- ``text-embedding-3-small``. 1536-dim, stronger on dense legal prose, needs a key.

Every backend returns **L2-normalized** vectors, so a dot product *is* cosine similarity and
``chunker.adjacent_distances`` needs no per-backend special casing.

Embeddings are cached on disk by ``(backend, sha256(text))``. Re-running a percentile sweep or
a benchmark then costs nothing, which matters more for the paid backend than the local one.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The only module reading the environment so far, so .env is loaded here. When §4 adds graph
# nodes that need LANGCHAIN_* and REDIS_*, lift this into a shared src/config.py.
# override=False: a real exported variable always beats the file.
load_dotenv(PROJECT_ROOT / ".env", override=False)

CACHE_DIR = PROJECT_ROOT / "data" / "processed" / ".embedding_cache"

MINILM_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")


class MissingCredentials(RuntimeError):
    """Raised when a backend needs a key that is not configured."""


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class MiniLMBackend:
    name = "minilm"
    model_id = MINILM_MODEL
    dimensions = 384
    cost_per_million_tokens = 0.0

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MINILM_MODEL)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAIBackend:
    name = "openai"
    model_id = OPENAI_MODEL
    dimensions = 1536
    cost_per_million_tokens = 0.02

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingCredentials(
                "OPENAI_API_KEY is not set. Add it to .env or export it, then re-run. "
                "The minilm backend needs no credentials and works offline."
            )
        from langchain_openai import OpenAIEmbeddings

        self._model = OpenAIEmbeddings(model=OPENAI_MODEL)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # embed_documents batches internally; it returns unit-length vectors already, but
        # normalizing again is cheap and makes the contract independent of that promise.
        vectors = np.asarray(self._model.embed_documents(list(texts)), dtype=np.float32)
        return _normalize_rows(vectors)


BACKENDS = {"minilm": MiniLMBackend, "openai": OpenAIBackend}


class CachedBackend:
    """Wraps a backend with an on-disk vector cache keyed by content hash."""

    def __init__(self, backend, cache_dir: Path = CACHE_DIR) -> None:
        self._backend = backend
        self.name = backend.name
        self.model_id = backend.model_id
        self.dimensions = backend.dimensions
        self._path = cache_dir / f"{backend.name}.npz"
        self._store: dict[str, np.ndarray] = {}
        self._dirty = False
        if self._path.exists():
            with np.load(self._path, allow_pickle=False) as data:
                self._store = {key: vector for key, vector in zip(data["keys"], data["vectors"])}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        keys = [self._key(text) for text in texts]
        missing = [text for text, key in zip(texts, keys) if key not in self._store]

        if missing:
            unique = list(dict.fromkeys(missing))
            fresh = self._backend.encode(unique)
            for text, vector in zip(unique, fresh):
                self._store[self._key(text)] = vector
            self._dirty = True

        return np.vstack([self._store[key] for key in keys]).astype(np.float32)

    def flush(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._store), dtype=object)
        vectors = np.vstack([self._store[key] for key in keys])
        np.savez(self._path, keys=keys.astype(str), vectors=vectors)
        self._dirty = False

    def __enter__(self) -> CachedBackend:
        return self

    def __exit__(self, *exc) -> None:
        self.flush()


def get_backend(name: str, *, cached: bool = True) -> CachedBackend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    backend = BACKENDS[name]()
    return CachedBackend(backend) if cached else backend


def available_backends() -> dict[str, str | None]:
    """Map backend name -> reason it cannot run, or ``None`` when it can."""
    status: dict[str, str | None] = {}
    for name, factory in BACKENDS.items():
        try:
            factory()
        except MissingCredentials as error:
            status[name] = str(error).split(".")[0]
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            status[name] = f"{type(error).__name__}: {error}"
        else:
            status[name] = None
    return status
