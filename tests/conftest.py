"""Shared test plumbing.

Two jobs: parse each ledger batch once per session rather than once per test, and make sure no test
writes a results database into the working tree.

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


REPO_DB = Path(__file__).resolve().parents[1] / "results.db"


@pytest.fixture(autouse=True, scope="session")
def no_database_in_the_working_tree():
    """Fail the suite if a test writes `results.db` into the repository.

    `audit_batch` persists by default and `RESULTS_DB_URL` defaults to `sqlite:///./results.db`, so
    any test that forgets to point the store somewhere temporary leaves a database in the working
    tree. It is gitignored, which is exactly why this is worth asserting: the mistake is otherwise
    invisible, and it happened -- a `monkeypatch.undo()` reverted a store fixture while the settings
    cache kept the old URL alive, and the run wrote a 57 KB database into the checkout.
    """
    existed = REPO_DB.exists()
    yield
    if REPO_DB.exists() and not existed:
        REPO_DB.unlink()
        raise AssertionError(
            "a test wrote results.db into the repository -- point RESULTS_DB_URL at a tmp_path, "
            "or pass an explicit store"
        )
