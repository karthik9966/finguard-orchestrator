"""Persistence for finished work (LLD §3.2, §5.1 step 8).

`ResultsStore` is the seam; `SqlResultsStore` is the implementation the CLI and the API use, and
`InMemoryResultsStore` is what the suite runs against.
"""

from src.store.results import (
    RESULTS_STORE_WRITE_FAILURE,
    InMemoryResultsStore,
    InvalidTransition,
    ResultsStore,
    ResultsStoreUnavailable,
    ReviewAction,
    SqlResultsStore,
    default_store,
)

__all__ = [
    "RESULTS_STORE_WRITE_FAILURE",
    "InMemoryResultsStore",
    "InvalidTransition",
    "ResultsStore",
    "ResultsStoreUnavailable",
    "ReviewAction",
    "SqlResultsStore",
    "default_store",
]
