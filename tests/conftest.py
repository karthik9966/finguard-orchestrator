"""Shared test plumbing.

One job so far: parse each ledger batch once per session rather than once per test.

Phase 2's ~10,000-message batch takes about 10 seconds to parse, and six tests across four files
were each parsing it independently -- 40 seconds of suite became 120. The suite is the thing that
has to stay cheap enough to run on every change, so it is cached here rather than made optional.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from src.utils.swift_parser import Batch, parse_batch

LEDGER = Path(__file__).resolve().parents[1] / "data" / "processed" / "ledger"


@lru_cache(maxsize=None)
def cached_batch(path: str, *, strict: bool = True) -> Batch:
    """A parsed batch, shared across the session.

    Safe to share because `Batch` is read-only in practice: every consumer reads `.wires`,
    `.failures` and the counts. A test that needs to mutate should parse its own copy.
    """
    return parse_batch(Path(path), strict=strict)


@pytest.fixture(scope="session")
def parsed():
    """`parsed(path)` -> Batch, parsed at most once per session."""
    return lambda path, strict=True: cached_batch(str(path), strict=strict)
