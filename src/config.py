"""Typed configuration -- LLD §2 Group 6, §8.

Two mechanisms, one rule each:

* ``Settings`` reads **secrets and infrastructure endpoints** from the environment. Nothing here
  has a default that would be wrong to commit, and nothing here is a tunable number.
* ``Config`` reads **every tunable number** from ``config.yaml``. Nothing here is a secret.

That split is LLD §8's, quoted: *numbers never in code, secrets never in the file*. Before this
module the repository had ~30 tunables as module-level constants across six files and read the
environment ad hoc in six more, which meant no single place answered "what is this run
configured to do".

Canonical field names follow the LLD. The pre-migration environment variable names are accepted
as aliases (``OPENAI_API_KEY`` for ``LLM_API_KEY``, ``CHROMA_PERSIST_DIR`` for
``CHROMA_DB_PATH``, and so on) so an existing ``.env`` keeps working while the migration runs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `Settings` reads .env by itself through `env_file` below, but that populates the *model*, not
# `os.environ` -- and several modules (the Chroma paths, the embedding backends) read the environment
# directly at import. Loading it here, in the module every one of them already imports, is what makes
# that deterministic instead of depending on which import happened to run first. It used to live in
# `graph/graph.py` for the LangSmith variables; those are gone, the need is not.
load_dotenv(PROJECT_ROOT / ".env", override=False)

PatternType = Literal["structuring", "fan_in", "fan_out", "cycle", "scatter_gather"]

# The five in-scope typologies, in the order the PRD lists them. Exported so detectors, the
# obligation map and the eval runners cannot drift apart on spelling.
PATTERN_TYPES: tuple[PatternType, ...] = (
    "structuring",
    "fan_in",
    "fan_out",
    "cycle",
    "scatter_gather",
)


# Embedding dimensions, by model. Used to derive `Settings.embedding_dimension` so the number
# cannot contradict the model it describes. A model absent from here needs
# EMBEDDING_DIMENSION set explicitly -- which fails loudly rather than guessing.
KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


# =============================================================================================
# Secrets and infrastructure
# =============================================================================================
class Settings(BaseSettings):
    """Environment-sourced configuration. Secrets, endpoints, model identifiers."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- the reasoning provider -------------------------------------------------------------
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
        description="Hosted frontier model key. Empty is valid: the deterministic suite runs "
        "without one, and a missing key must fail at the call site, not at import.",
    )
    # Model tiering (HLD §4): the strong model is reserved for judgement and grounding, the
    # light one does parsing rescue. Naming them separately is what makes that reservation real.
    reasoning_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("REASONING_MODEL", "AUDIT_MODEL"),
    )
    light_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LIGHT_MODEL", "EXTRACTION_MODEL"),
    )
    llm_temperature: float = 0.0

    # --- the retrieval pipeline, all in-environment ------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    # LLD §3.2 puts the vector dimension in config, and it is not cosmetic: a collection built at
    # one dimension cannot answer a query embedded at another, and the failure is a silent
    # distance rather than an exception.
    #
    # So it is *derived* from the model, with this as an override for a model not in the table
    # below. A plain configured number can disagree with the configured model -- which is not
    # hypothetical: the pre-migration .env sets EMBEDDING_MODEL=text-embedding-3-small (1536),
    # and a dimension defaulted to MiniLM's 384 alongside it would have built a collection that
    # silently mis-measures every distance. A value that cannot contradict its source is better
    # than a validator that catches the contradiction.
    embedding_dimension_override: int | None = Field(
        default=None, validation_alias=AliasChoices("EMBEDDING_DIMENSION")
    )
    cross_encoder_model: str = Field(
        default="ms-marco-TinyBERT-L-2-v2",
        validation_alias=AliasChoices("CROSS_ENCODER_MODEL", "RERANK_MODEL"),
    )
    embedding_cache_dir: Path = PROJECT_ROOT / "data" / "processed" / ".embedding_cache"

    chroma_db_path: Path = Field(
        default=PROJECT_ROOT / "chroma_db",
        validation_alias=AliasChoices("CHROMA_DB_PATH", "CHROMA_PERSIST_DIR"),
    )
    # Two collections, deliberately. `rule_chunks` is the citable US knowledge base; nothing
    # outside US jurisdiction may enter it. `obliqa_benchmark` holds the 40 ADGM documents purely
    # so the 2,786-question retrieval benchmark stays reproducible -- it is never retrieved from
    # at runtime, which is what guarantees no out-of-jurisdiction clause can be cited.
    # Deliberately NOT aliased to the pre-migration CHROMA_COLLECTION. An existing .env sets
    # that to "regulations", and honouring it here would write the new tier-aware metadata into
    # the old ADGM-bearing collection -- the exact mixing this split exists to prevent.
    rule_collection: str = "rule_chunks"
    benchmark_collection: str = "obliqa_benchmark"

    # --- persistence ------------------------------------------------------------------------
    results_db_url: str = "sqlite:///./results.db"

    # --- observability ----------------------------------------------------------------------
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_tracing: bool = False
    # Separates one deployment's traces from another's in the same Langfuse project.
    langfuse_environment: str = "development"
    # HLD §6 asks for a client-tier tag so runs can be compared across tiers. The engine has no
    # notion of a client yet -- one deployment audits one institution -- so it is a deployment
    # label rather than a lookup, and it is here rather than in config.yaml because it identifies
    # *this install* rather than tuning behaviour.
    client_tier: str = "unspecified"

    # --- serving ----------------------------------------------------------------------------
    api_auth_token: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Where the cockpit dials the API. Separate from api_host because that is a *bind* address:
    # 0.0.0.0 means "every interface" to a server and is not a thing a client can connect to.
    api_base_url: str = "http://127.0.0.1:8000"

    # --- meta -------------------------------------------------------------------------------
    config_path: Path = PROJECT_ROOT / "config.yaml"
    env: Literal["dev", "test", "prod"] = "dev"

    @property
    def embedding_dimension(self) -> int:
        """The active model's vector width."""
        if self.embedding_dimension_override is not None:
            return self.embedding_dimension_override
        # Match on the bare model name: a hosted id may arrive namespaced ("openai/..."), and a
        # sentence-transformers id may arrive with its org prefix.
        name = self.embedding_model.strip()
        for known, dimension in KNOWN_EMBEDDING_DIMENSIONS.items():
            if name == known or name.endswith("/" + known.split("/")[-1]):
                return dimension
        raise ValueError(
            f"unknown embedding model {name!r}: set EMBEDDING_DIMENSION explicitly, or add it to "
            "KNOWN_EMBEDDING_DIMENSIONS in src/config.py"
        )

    @property
    def tracing_enabled(self) -> bool:
        """Tracing needs the flag *and* credentials. A flag alone produced silent no-ops in the
        pre-migration LangSmith wiring, which is why both are checked in one place."""
        return bool(self.langfuse_tracing and self.langfuse_public_key and self.langfuse_secret_key)


# =============================================================================================
# Tunable numbers
# =============================================================================================
class RetrievalConfig(BaseModel):
    k_indicators: int = Field(gt=0)
    rerank_top_n: int = Field(gt=0)
    multi_query_rrf: bool
    rrf_k: int = Field(gt=0)

    @model_validator(mode="after")
    def rerank_cannot_exceed_candidates(self) -> RetrievalConfig:
        """Asking for more reranked indicators than were retrieved is not an error the reranker
        reports -- it just returns fewer, and the grounding prompt quietly gets a thinner
        context than the config claims."""
        if self.rerank_top_n > self.k_indicators:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) exceeds k_indicators ({self.k_indicators})"
            )
        return self


class StructuringConfig(BaseModel):
    thresholds: list[int] = Field(min_length=1)
    band_fraction: float = Field(gt=0.0, lt=1.0)
    min_count: int = Field(gt=1)
    min_aggregate_multiple: float = Field(gt=0.0)

    def band(self, threshold: int) -> tuple[float, int]:
        """The half-open interval [T - band, T) a transaction must sit in to count."""
        return threshold * (1.0 - self.band_fraction), threshold


class FanInConfig(BaseModel):
    min_sources: int = Field(gt=1)


class FanOutConfig(BaseModel):
    min_targets: int = Field(gt=1)


class CycleConfig(BaseModel):
    min_hops: int = Field(ge=3)
    max_length: int = Field(gt=3)
    path_overlap: float = Field(gt=0.0, le=1.0)
    min_retained_fraction: float = Field(gt=0.0, le=1.0)


class ScatterGatherConfig(BaseModel):
    min_fan: int = Field(gt=1)


class ConfidenceWeights(BaseModel):
    tightness: float
    member_count: float
    window_compactness: float
    aggregate_ratio: float

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> ConfidenceWeights:
        total = self.tightness + self.member_count + self.window_compactness + self.aggregate_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"confidence weights must sum to 1.0, got {total}")
        return self

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class DetectionConfig(BaseModel):
    window_days: int = Field(gt=0)
    structuring: StructuringConfig
    fan_in: FanInConfig
    fan_out: FanOutConfig
    cycle: CycleConfig
    scatter_gather: ScatterGatherConfig
    confidence_weights: ConfidenceWeights
    precedence_order: list[PatternType]

    @field_validator("precedence_order")
    @classmethod
    def precedence_covers_every_pattern(cls, order: list[str]) -> list[str]:
        """A pattern missing from the precedence order has undefined behaviour in the
        reconciler -- it would be neither suppressed nor allowed to win deterministically."""
        missing = set(PATTERN_TYPES) - set(order)
        if missing:
            raise ValueError(f"precedence_order omits {sorted(missing)}")
        if len(order) != len(set(order)):
            raise ValueError("precedence_order contains duplicates")
        return order


class ReasoningConfig(BaseModel):
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    max_loops: int = Field(ge=0)
    high_risk_min_confidence: float = Field(ge=0.0, le=1.0)
    schema_retries: int = Field(ge=1)
    llm_max_attempts: int = Field(ge=1)
    llm_timeout_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def high_risk_bar_is_not_below_acceptance(self) -> ReasoningConfig:
        """A High-risk bar below the acceptance threshold can never bind, so setting it there is
        almost certainly a mistake rather than a policy."""
        if self.high_risk_min_confidence < self.confidence_threshold:
            raise ValueError(
                "high_risk_min_confidence is below confidence_threshold, so it can never bind"
            )
        return self


class PersistenceConfig(BaseModel):
    write_attempts: int = Field(ge=1)
    write_backoff_seconds: float = Field(ge=0.0)


class ChunkingConfig(BaseModel):
    min_chars: int = Field(gt=0)
    max_chars: int = Field(gt=0)
    percentile: float = Field(gt=0.0, le=100.0)
    min_sentences_for_percentile: int = Field(gt=0)
    min_passage_chars: int = Field(ge=0)
    context_prefix_below: int = Field(ge=0)

    @model_validator(mode="after")
    def min_below_max(self) -> ChunkingConfig:
        if self.min_chars >= self.max_chars:
            raise ValueError("min_chars must be below max_chars")
        return self


class IngestionConfig(BaseModel):
    upsert_batch: int = Field(gt=0)
    llm_fallback_attempts: int = Field(ge=0)


class ObligationRef(BaseModel):
    """One Tier-1 obligation, addressed the way it is curated: by source and section, never by
    chunk id. See config.yaml's pattern_to_obligations comment for why."""

    source_id: str
    section_ref: str

    def __str__(self) -> str:  # what an error message should show
        return f"{self.source_id} {self.section_ref}"


class Config(BaseModel):
    """The whole of config.yaml, validated."""

    schema_version: int = Field(validation_alias=AliasChoices("schema", "schema_version"))
    retrieval: RetrievalConfig
    detection: DetectionConfig
    reasoning: ReasoningConfig
    persistence: PersistenceConfig
    chunking: ChunkingConfig
    ingestion: IngestionConfig
    pattern_to_obligations: dict[PatternType, list[ObligationRef]]
    source_topics: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("source_topics")
    @classmethod
    def typology_tags_are_spelled_correctly(cls, mapping: dict[str, list[str]]) -> dict[str, list[str]]:
        """A tag that looks like a typology must be one.

        `fan_in` and `faninn` both filter to nothing at retrieval time, and only one of them is
        a mistake -- but neither raises, so a typo would surface as a finding that quietly lost
        its indicators. Tags that are not typology names (`sar`, `cash`, `red_flags`) are free
        text by design; only the near-misses are worth catching.
        """
        known = set(PATTERN_TYPES)
        for source_id, tags in mapping.items():
            for tag in tags:
                squashed = tag.replace("-", "_").replace(" ", "_").lower()
                if squashed not in known and any(
                    squashed.replace("_", "") == pattern.replace("_", "") for pattern in known
                ):
                    raise ValueError(
                        f"{source_id}: topic tag {tag!r} looks like a typology but is not one of "
                        f"{sorted(known)}"
                    )
        return mapping

    @field_validator("pattern_to_obligations")
    @classmethod
    def every_pattern_has_an_entry(
        cls, mapping: dict[str, list[ObligationRef]]
    ) -> dict[str, list[ObligationRef]]:
        """An absent key and an empty list mean different things: absent is a config bug, empty
        is "not curated yet". Only the first is rejected here; tests/test_obligation_map.py
        rejects the second, once the corpus exists to resolve against."""
        missing = set(PATTERN_TYPES) - set(mapping)
        if missing:
            raise ValueError(f"pattern_to_obligations omits {sorted(missing)}")
        return mapping

    def obligations_for(self, pattern_type: str) -> list[ObligationRef]:
        return self.pattern_to_obligations.get(pattern_type, [])


# =============================================================================================
# Accessors
# =============================================================================================
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_config(path: Path | str | None = None) -> Config:
    """Read and validate config.yaml. Separate from the cached accessor so a test can load an
    alternative file without poisoning the process-wide cache."""
    resolved = Path(path) if path is not None else get_settings().config_path
    if not resolved.is_file():
        raise FileNotFoundError(f"config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


def reset_caches() -> None:
    """Drop both caches. For tests that monkeypatch the environment or the config file; the
    caches exist so that reading a threshold in a loop is free, not to freeze a process."""
    get_settings.cache_clear()
    get_config.cache_clear()
