"""The config spine -- LLD §8.

What these tests protect is the split itself: a number reaching its consumer from the file, a
secret *not* reaching anyone from the file, and the pairs of values that must not be allowed to
contradict each other.
"""

from __future__ import annotations

import re

import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src import config as config_module
from src.config import (
    KNOWN_EMBEDDING_DIMENSIONS,
    PATTERN_TYPES,
    Config,
    Settings,
    get_config,
    load_config,
    reset_caches,
)

REPO_CONFIG = config_module.PROJECT_ROOT / "config.yaml"


@pytest.fixture(autouse=True)
def clean_caches():
    reset_caches()
    yield
    reset_caches()


def raw_config() -> dict:
    with REPO_CONFIG.open() as handle:
        return yaml.safe_load(handle)


def write_config(tmp_path, mutate) -> object:
    data = raw_config()
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# --- 1. the file the repository actually ships ----------------------------------------------
def test_the_shipped_config_validates():
    """Every other test in the suite is worthless if the real file does not load."""
    config = load_config(REPO_CONFIG)
    assert config.schema_version == 1
    assert set(config.pattern_to_obligations) == set(PATTERN_TYPES)


def test_a_number_changed_in_the_file_reaches_its_consumer(tmp_path):
    """The whole point of the split. If this fails, a threshold is still hardcoded somewhere."""
    path = write_config(tmp_path, lambda d: d["detection"].__setitem__("window_days", 21))
    assert load_config(path).detection.window_days == 21


def test_no_secret_is_readable_from_the_config_file():
    """LLD §8's other half: *secrets never in the file*. A key committed to config.yaml would be
    a key in git history, so the schema must have nowhere to put one."""
    fields = set(Config.model_fields)
    for forbidden in ("llm_api_key", "api_auth_token", "langfuse_secret_key", "results_db_url"):
        assert forbidden not in fields
    # Scan the parsed data, not the file text: the file's own comments explain the rule, and a
    # substring scan over prose flags the explanation rather than a violation.
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield str(key)
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)
        elif isinstance(node, str):
            yield node

    data = " ".join(walk(raw_config())).lower()
    for smell in ("api_key", "apikey", "secret", "password", "bearer"):
        assert smell not in data, f"config.yaml carries {smell!r} in its data"

    # A key, not the letters. `sk-` as a bare substring matches `ffiec-risk-ach`, which is a
    # source_id in source_topics -- the scanner flagged it as a committed credential. An OpenAI
    # key is the prefix followed by a long opaque run, and that is what to look for.
    assert not re.search(r"\bsk-[A-Za-z0-9_-]{16,}", data), "config.yaml carries an API key"


# --- 2. pairs that must not contradict each other ------------------------------------------
def test_rerank_top_n_cannot_exceed_the_candidate_pool(tmp_path):
    """Asking for 20 reranked indicators out of 15 retrieved is not an error the reranker
    reports -- it returns fewer, and the prompt quietly gets a thinner context than configured."""
    path = write_config(tmp_path, lambda d: d["retrieval"].__setitem__("rerank_top_n", 99))
    with pytest.raises(ValueError, match="exceeds k_indicators"):
        load_config(path)


def test_a_high_risk_bar_below_the_acceptance_threshold_is_rejected(tmp_path):
    """Such a bar can never bind, so configuring one is a mistake rather than a policy."""
    path = write_config(tmp_path, lambda d: d["reasoning"].__setitem__("high_risk_min_confidence", 0.5))
    with pytest.raises(ValueError, match="never bind"):
        load_config(path)


def test_confidence_weights_must_sum_to_one(tmp_path):
    path = write_config(
        tmp_path, lambda d: d["detection"]["confidence_weights"].__setitem__("tightness", 0.9)
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_config(path)


def test_chunk_size_floor_must_sit_below_its_ceiling(tmp_path):
    path = write_config(tmp_path, lambda d: d["chunking"].__setitem__("min_chars", 9000))
    with pytest.raises(ValueError, match="below max_chars"):
        load_config(path)


# --- 3. the five patterns, in one place ----------------------------------------------------
def test_precedence_order_must_cover_every_pattern(tmp_path):
    """A pattern missing from the order has undefined behaviour in the reconciler: neither
    suppressed nor deterministically allowed to win."""
    path = write_config(
        tmp_path, lambda d: d["detection"].__setitem__("precedence_order", ["cycle", "fan_in"])
    )
    with pytest.raises(ValueError, match="precedence_order omits"):
        load_config(path)


def test_precedence_order_rejects_duplicates(tmp_path):
    order = ["structuring", "structuring", "cycle", "scatter_gather", "fan_in", "fan_out"]
    path = write_config(tmp_path, lambda d: d["detection"].__setitem__("precedence_order", order))
    with pytest.raises(ValueError, match="duplicates"):
        load_config(path)


def test_the_obligation_map_must_have_a_key_for_every_pattern(tmp_path):
    """An absent key is a config bug; an empty list is "not curated yet". Only the first is
    rejected here -- tests/test_obligation_map.py rejects the second, once there is a corpus to
    resolve against."""
    path = write_config(tmp_path, lambda d: d["pattern_to_obligations"].pop("cycle"))
    with pytest.raises(ValueError, match="pattern_to_obligations omits"):
        load_config(path)


def test_fan_out_is_in_scope():
    """The PRD §2 in-scope list names fan-out; the LLD's pattern_type Literal has four values and
    omits it. This suite follows the PRD. If the documents are ever reconciled the other way,
    this test is where that decision has to be made explicitly rather than by deletion."""
    assert "fan_out" in PATTERN_TYPES
    assert len(PATTERN_TYPES) == 5


# --- 4. the environment half ---------------------------------------------------------------
def test_the_embedding_dimension_cannot_contradict_the_model(monkeypatch):
    """The trap this replaced: the pre-migration .env names a 1536-dimension hosted model, and a
    separately-configured dimension defaulted to MiniLM's 384. A collection built that way
    mis-measures every distance and raises nothing."""
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
    assert Settings().embedding_dimension == 1536

    monkeypatch.setenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    assert Settings().embedding_dimension == 384


def test_an_unknown_embedding_model_demands_an_explicit_dimension(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "a-model-nobody-has-heard-of")
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
    with pytest.raises(ValueError, match="unknown embedding model"):
        Settings().embedding_dimension

    monkeypatch.setenv("EMBEDDING_DIMENSION", "512")
    assert Settings().embedding_dimension == 512


def test_every_known_dimension_is_plausible():
    """Guards a typo in the table, which would otherwise surface as a silent mis-measure."""
    assert all(64 <= dim <= 8192 for dim in KNOWN_EMBEDDING_DIMENSIONS.values())


def test_the_pre_migration_env_names_are_still_accepted(monkeypatch):
    """An existing .env must keep working while the migration runs, or every phase boundary
    needs a coordinated environment change."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old-name")
    monkeypatch.setenv("AUDIT_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("EXTRACTION_MODEL", "gpt-4o-mini-2024-07-18")
    settings = Settings()
    assert settings.llm_api_key == "sk-old-name"
    assert settings.reasoning_model == "gpt-4o-2024-08-06"
    assert settings.light_model == "gpt-4o-mini-2024-07-18"


def test_the_rule_collection_does_not_inherit_the_old_collection_name(monkeypatch):
    """CHROMA_COLLECTION=regulations in an existing .env must NOT redirect the new collection.
    Honouring it would write the new tier-aware metadata into the old ADGM-bearing collection --
    the exact mixing the two-collection split exists to prevent."""
    monkeypatch.setenv("CHROMA_COLLECTION", "regulations")
    settings = Settings()
    assert settings.rule_collection == "rule_chunks"
    assert settings.benchmark_collection == "obliqa_benchmark"


def test_tracing_needs_both_the_flag_and_credentials(monkeypatch):
    """A flag alone was a silent no-op in the pre-migration LangSmith wiring: runs reported
    tracing on and emitted nothing."""
    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    assert Settings().tracing_enabled is False

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert Settings().tracing_enabled is True

    monkeypatch.setenv("LANGFUSE_TRACING", "false")
    assert Settings().tracing_enabled is False


def test_a_missing_key_is_not_an_import_time_failure(monkeypatch):
    """The deterministic suite runs with no key at all. A required field here would make every
    test in the repository depend on a secret."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings(_env_file=None).llm_api_key == ""


# --- 5. structuring bands ------------------------------------------------------------------
def test_the_structuring_band_scales_with_its_threshold():
    """A single absolute band cannot serve both monitored thresholds: 2000 below $10,000 is the
    intended [8000, 10000), but the same 2000 below $3,000 is [1000, 3000) -- wide enough to
    catch most ordinary payments. A fraction of T is the recorded deviation from LLD §8."""
    structuring = load_config(REPO_CONFIG).detection.structuring
    assert structuring.band(10_000) == (8_000.0, 10_000)
    assert structuring.band(3_000) == (2_400.0, 3_000)
    assert structuring.thresholds == [10_000, 3_000]


def test_caches_can_be_dropped():
    """The caches exist so reading a threshold in a loop is free, not to freeze a process."""
    first = get_config()
    assert get_config() is first
    reset_caches()
    assert get_config() is not first


# --- 6. config.yaml is live, not decorative -------------------------------------------------
# Phase 0 of the migration repointed the pre-migration module constants at config.yaml rather than
# deleting them, so the file was live during Phases 1-4 instead of being dead config that only took
# effect after the Phase 5 rewrite. Phase 5 removed the last of those shims: every tunable is now
# read from `get_config()` at the point of use. This test is what keeps that true -- a re-introduced
# module constant would be frozen at import, and the file would look live while being dead.
def test_no_module_reads_a_tunable_at_import_time():
    """The shims are gone. A `NAME = get_config().x.y` at module level is the pattern that made
    config.yaml editable-but-inert, so it is banned rather than merely discouraged."""
    import re

    offenders: list[str] = []
    pattern = re.compile(r"^[A-Z_][A-Z0-9_]*\s*(?::[^=]+)?=\s*get_config\(\)", re.MULTILINE)
    for path in sorted((Path(__file__).resolve().parents[1] / "src").rglob("*.py")):
        for match in pattern.finditer(path.read_text()):
            offenders.append(f"{path.name}: {match.group(0).strip()}")
    assert not offenders, "module-level tunables read config at import and freeze: " + "; ".join(
        offenders
    )


def test_editing_the_file_changes_the_detector(tmp_path):
    """End to end: edit the window in config.yaml and the detector that reads it sees the new
    value. This is the assertion that makes the config split more than decorative.

    Run in a subprocess rather than by reloading the module in-process. `importlib.reload` rebinds
    every class the module defines, so a model class becomes a *new* class object while other test
    modules still hold the old one -- and `isinstance` starts failing in an unrelated test file.
    That is not a hypothetical: it is what the first version of this test did. A subprocess cannot
    pollute the session at all, and it exercises the real import path instead of a reloaded
    approximation of it.
    """
    path = write_config(tmp_path, lambda d: d["detection"].__setitem__("window_days", 30))
    program = (
        "from src.detection.base import BaseDetector;"
        "from src.config import get_config;"
        "print(BaseDetector.window_days.fget(None), get_config().chunking.max_chars)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=config_module.PROJECT_ROOT,
        env={**os.environ, "CONFIG_PATH": str(path)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    window, chars = result.stdout.split()
    assert window == "30", "config.yaml does not reach the detector"
    assert chars == "2000", "an unrelated section changed with it"
