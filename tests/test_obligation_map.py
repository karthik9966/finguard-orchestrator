"""`pattern_to_obligations` must resolve, entry for entry (Phase 1c).

Written before the map was filled, and deliberately so. Per LLD §6 an `OBLIGATION_MAP_MISS`
disables grounding for a whole typology: every finding of that pattern loses the binding rule it
should rest on, and nothing raises -- the report simply comes back citing indicators alone. A
curated map with no resolve test is a config file that fails silently.

The map holds `(source_id, section_ref)` pairs rather than chunk ids because
`chunk_id = hash(source_id, section_ref, version)`. A re-chunk or an eCFR version bump changes
every id, so literal ids would rot with no error at all. Pairs rot loudly, here.
"""

from __future__ import annotations

import pytest

from src.config import PATTERN_TYPES, get_config
from src.ingestion.store import RULE_COLLECTION, VectorStoreClient


@pytest.fixture(scope="module")
def store() -> VectorStoreClient:
    client = VectorStoreClient(RULE_COLLECTION)
    try:
        client._collection()
    except Exception:  # noqa: BLE001
        pytest.skip(f"{RULE_COLLECTION} not built -- run: uv run finguard-store --rules")
    return client


@pytest.fixture(scope="module")
def mapping() -> dict[str, list]:
    return get_config().pattern_to_obligations


def test_every_typology_has_at_least_one_obligation(mapping):
    """An empty list was a valid state only before the corpus existed. A pattern with no binding
    rule behind it produces a finding that can cite red flags and nothing that obliges anyone to
    act on them."""
    bare = [pattern for pattern in PATTERN_TYPES if not mapping.get(pattern)]
    assert not bare, f"no obligations curated for {bare}"


def test_every_entry_resolves_to_a_chunk_that_exists(store, mapping):
    """The test the whole map exists for. A typo in a section_ref is indistinguishable from a
    correct entry until a finding quietly has no law behind it."""
    unresolved = []
    for pattern, references in mapping.items():
        for reference in references:
            pair = (reference.source_id, reference.section_ref)
            if store.resolve(pair) is None:
                unresolved.append(f"{pattern}: {pair}")
    assert not unresolved, "unresolvable obligations:\n  " + "\n  ".join(unresolved)


def test_every_obligation_is_binding_law_not_guidance(store, mapping):
    """Guidance illustrates; it does not oblige. Citing an FFIEC red flag as the rule that
    requires action is the exact conflation `authority` exists to prevent."""
    wrong = []
    for pattern, references in mapping.items():
        for reference in references:
            chunk_id = store.resolve((reference.source_id, reference.section_ref))
            if chunk_id is None:
                continue
            chunk = store.get_by_ids([chunk_id])[chunk_id]
            if chunk.get("authority") != "binding":
                wrong.append(f"{pattern}: {reference.source_id} is {chunk.get('authority')}")
    assert not wrong, "non-binding obligations:\n  " + "\n  ".join(wrong)


def test_the_sar_duty_backs_every_typology(mapping):
    """31 CFR 1020.320 is the duty to report a suspicious transaction. Whatever shape the
    activity took, that is the obligation the filing rests on."""
    for pattern in PATTERN_TYPES:
        sources = {reference.source_id for reference in mapping[pattern]}
        assert "31cfr1020.320" in sources, f"{pattern} has no SAR duty behind it"


def test_structuring_cites_the_statute_that_prohibits_it(mapping):
    """§ 5324 is the prohibition itself and 1010.311 is the $10,000 CTR duty it evades. A
    structuring finding that cites neither has described a shape and named no law."""
    sources = {reference.source_id for reference in mapping["structuring"]}
    assert {"31usc5324", "31cfr1010.311"} <= sources
