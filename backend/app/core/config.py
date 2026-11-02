"""Application settings, loaded from environment variables / `.env` files.

Everything the application needs to know about its environment lives here.
Modules must never read ``os.environ`` directly - they receive a
:class:`~app.core.config.Settings` instance (usually via
:func:`app.core.config.get_settings`).

Configuration is read from ``backend/.env`` (the committed configuration source)
and from the repository root ``.env`` when one exists, so an operator can drop a
single override file at the top of the repo if they prefer.

List-typed values (``CORS_ORIGINS``, ``ALLOWED_EXTENSIONS``) are parsed as JSON
so a single ``.env`` file works for every deployment target.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: ``backend/`` - where the committed ``.env`` lives.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
#: Repository root - an optional extra override location.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# Generation budget ceilings for the default model (Qwen2.5-0.5B-Instruct).
#
# Two different limits apply, and conflating them is what truncates answers:
#
# * ``LLM_CONTEXT_WINDOW_TOKENS`` (32768) - the hard total budget. The model
#   uses absolute position embeddings up to 32768, so
#   ``prompt_tokens + generated_tokens`` may never exceed it. The 0.5B variant
#   has no YaRN rope scaling (unlike the 7B+ siblings, which reach 131072).
# * ``LLM_MAX_GENERATION_TOKENS`` (8192) - the longest answer Qwen2.5 was
#   trained to produce. ``transformers`` would allow more, but generation past
#   this point is outside the training distribution and degrades into
#   repetition rather than a longer coherent answer.
#
# Revisit both if ``LLM_MODEL_ID`` is pointed at a different model.
# --------------------------------------------------------------------------- #
LLM_CONTEXT_WINDOW_TOKENS = 32_768
LLM_MAX_GENERATION_TOKENS = 8_192


class Settings(BaseSettings):
    """Strongly typed view over the process environment."""

    model_config = SettingsConfigDict(
        env_file=(str(BACKEND_ROOT / ".env"), str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Enterprise RAG Knowledge Assistant"
    app_env: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # ------------------------------------------------------------- security
    secret_key: str = "change-me-in-production-with-a-64-char-random-string"
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "rag-knowledge-assistant"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # --------------------------------------------------- manual rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    rate_limit_auth_requests: int = 10
    rate_limit_chat_requests: int = 20
    rate_limit_upload_requests: int = 10

    # -------------------------------------------------------------- storage
    # PostgreSQL is the only supported database.
    database_url: str = "postgresql+psycopg2://rag:rag@localhost:5432/rag"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    # ------------------------------------------------------------------- redis
    # Redis backs the cache, the rate-limit counters and the Celery broker.
    redis_url: str | None = "redis://localhost:6379/0"
    redis_port: int = 6379

    # ----------------------------------------------------------- vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "knowledge_base"

    # --------------------------------------------------------------- uploads
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 25
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".txt",
            ".md",
            ".csv",
            ".docx",
            ".json",
            ".html",
        ]
    )

    # ------------------------------------------------------------ embeddings
    embedding_provider: Literal["huggingface", "hash"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    embedding_normalize: bool = True

    # ---------------------------------------------------- chunking/retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    #: Number of chunks returned as context (the standard RAG default, matching
    #: LangChain's ``k=4``). Raise it for broader coverage, lower it for focus.
    retrieval_top_k: int = 4
    #: Candidates fetched before thresholding. The retriever asks Chroma for
    #: ``max(retrieval_top_k, retrieval_fetch_k)`` neighbours, applies the score
    #: threshold, then returns the top ``retrieval_top_k`` - so a wider pool can
    #: surface chunks from documents that would otherwise be crowded out.
    retrieval_fetch_k: int = 20
    #: Minimum similarity a chunk must reach to be used as context. The offline
    #: ``hash`` embedder is lexical and produces weak scores, so the default keeps
    #: everything retrieved (``0.0``) rather than silently dropping whole
    #: documents. Raise it once a semantic embedding model is in use.
    retrieval_score_threshold: float = 0.0

    # -------------------------------------------------------------- LLM layer
    llm_provider: Literal["transformers", "stub"] = "transformers"
    llm_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    #: Longest answer the model may generate, in tokens. Defaults to the full
    #: 8192-token generation length Qwen2.5 was trained for, so replies are as
    #: detailed as the model is capable of. The structured schema emits prose +
    #: bullet points + citations in one object, so a small budget truncates the
    #: JSON mid-way and the answer loses its citations.
    #:
    #: Raising this is bounded by the context window: ``prompt_tokens`` (system
    #: prompt + retrieved context + history) plus this value must stay within
    #: ``LLM_CONTEXT_WINDOW_TOKENS`` (32768). See ``_check_generation_budget``.
    llm_max_new_tokens: int = LLM_MAX_GENERATION_TOKENS
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_device: str = "cpu"
    llm_torch_dtype: str = "auto"
    llm_trust_remote_code: bool = False
    #: Constrain decoding to a JSON schema (outlines + transformers) so answers
    #: arrive as a validated object with explicit citations. Falls back to plain
    #: text automatically when ``outlines`` is not installed.
    llm_structured_output: bool = True
    chat_use_celery: bool = False
    hf_home: str | None = None

    # ---------------------------------------------------------------- celery
    # Both fall back to ``redis_url`` when left empty (see the properties below).
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    #: ``true`` runs tasks inline in the calling process. Defaults to ``True``
    #: because the embedded Chroma client keeps its ANN index *in memory*: a
    #: separate worker process writing to the same persist directory would be
    #: invisible to the API process's index, so retrieval would return nothing
    #: while the rows (and ``count()``) say otherwise. Running ingestion in the
    #: API process keeps one owner of the collection. Set ``false`` only when the
    #: API and worker run as separate containers *and* ingestion throughput
    #: matters more than read-after-write visibility.
    celery_task_always_eager: bool = True
    celery_worker_concurrency: int = 2

    # ------------------------------------------------------------ validators
    @field_validator("cors_origins", "allowed_extensions", mode="before")
    @classmethod
    def _parse_json_list(cls, value: Any) -> Any:
        """Accept JSON arrays, comma separated strings or real lists."""
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                return json.loads(raw)
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value

    @field_validator("allowed_extensions", mode="after")
    @classmethod
    def _normalise_extensions(cls, value: list[str]) -> list[str]:
        normalised: list[str] = []
        for extension in value:
            extension = extension.strip().lower()
            if not extension:
                continue
            normalised.append(extension if extension.startswith(".") else f".{extension}")
        return normalised

    @field_validator("log_level", mode="after")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("llm_temperature")
    @classmethod
    def _check_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("llm_temperature must be between 0.0 and 2.0")
        return value

    @field_validator("llm_max_new_tokens")
    @classmethod
    def _check_generation_budget(cls, value: int) -> int:
        """Keep the generation budget inside what the model can actually do.

        Exceeding the trained 8192-token generation length produces degenerate,
        repetitive output rather than a longer answer, and exceeding the 32768
        context window makes the position embeddings run out entirely - so both
        are rejected at configuration time rather than silently misbehaving at
        request time.
        """
        if value < 1:
            raise ValueError("llm_max_new_tokens must be at least 1")
        if value > LLM_MAX_GENERATION_TOKENS:
            raise ValueError(
                f"llm_max_new_tokens must not exceed {LLM_MAX_GENERATION_TOKENS} "
                "(the longest generation length Qwen2.5 was trained for)"
            )
        if value > LLM_CONTEXT_WINDOW_TOKENS:
            raise ValueError(
                f"llm_max_new_tokens must not exceed the {LLM_CONTEXT_WINDOW_TOKENS} "
                "token context window"
            )
        return value

    # ------------------------------------------------------------- helpers
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        return self.app_env == "test"

    @property
    def broker_url(self) -> str:
        """Celery broker, falling back to the cache Redis instance."""
        return self.celery_broker_url or self.redis_url or "memory://"

    @property
    def result_backend_url(self) -> str:
        """Celery result backend, falling back to the cache Redis instance."""
        return self.celery_result_backend or self.redis_url or "cache+memory://"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def llm_context_window_tokens(self) -> int:
        """Hard total token budget (prompt + completion) for the model."""
        return LLM_CONTEXT_WINDOW_TOKENS

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir).expanduser()

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).expanduser()

    def scoped_rate_limit(self, scope: str) -> tuple[int, int]:
        """Return ``(limit, window_seconds)`` for a named rate-limit scope."""
        limits = {
            "default": self.rate_limit_requests,
            "auth": self.rate_limit_auth_requests,
            "chat": self.rate_limit_chat_requests,
            "upload": self.rate_limit_upload_requests,
        }
        return limits.get(scope, self.rate_limit_requests), self.rate_limit_window_seconds


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()


# Convenient module level alias for import-time configuration.
settings = get_settings()
