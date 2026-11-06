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
# * ``LLM_MAX_GENERATION_TOKENS`` (32768) - the highest number of tokens the
#   model can emit for the answer. It is set equal to the context window so the
#   completion is allowed the model's full capacity; the provider still trims it
#   to ``context_window - prompt_tokens`` at request time so ``prompt`` and
#   ``completion`` together never overflow the window.
#
# Revisit both if ``LLM_MODEL_ID`` is pointed at a different model.
# --------------------------------------------------------------------------- #
LLM_CONTEXT_WINDOW_TOKENS = 32_768
LLM_MAX_GENERATION_TOKENS = 32_768


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
    #: Highest number of tokens the model may generate for the answer. This is a
    #: hard latency ceiling: generation is CPU-bound, so the value also caps how
    #: long a single synchronous reply can take. The provider trims it at request
    #: time so ``prompt_tokens`` (system prompt + retrieved context + history)
    #: plus the completion stay within ``LLM_CONTEXT_WINDOW_TOKENS``. The shipped
    #: ``backend/.env`` sets a snappy 512; the ceiling here is the model's maximum
    #: (``LLM_MAX_GENERATION_TOKENS``). See ``_check_generation_budget``.
    llm_max_new_tokens: int = LLM_MAX_GENERATION_TOKENS
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    #: Repetition penalty applied during generation. Values > 1.0 discourage the
    #: model from repeating tokens it has already produced. Small models left at a
    #: weak penalty (<=1.05) tend to loop and duplicate whole list items; 1.1 is a
    #: safe default that removes the duplication without hurting fluency.
    llm_repetition_penalty: float = 1.1
    #: Hard block on repeating any n-gram of this many tokens (0 disables it).
    #: The repetition penalty alone did not stop the 0.5B model looping the same
    #: phrase, which burned the whole token budget before finishing; banning
    #: repeated 4-grams forces it to move on so the answer completes.
    llm_no_repeat_ngram_size: int = 4
    llm_device: str = "cpu"
    #: CPU inference threads for the CPU-only deployment. ``0`` (the default)
    #: sizes the PyTorch thread pool to every logical core
    #: (``torch.set_num_threads(os.cpu_count())``) - the single biggest
    #: throughput lever for synchronous CPU generation. A positive value caps it
    #: (leaving some cores for the web server under concurrency).
    llm_num_threads: int = 0
    #: Attention backend passed to ``from_pretrained``. "sdpa" (the default) uses
    #: PyTorch's fused scaled-dot-product attention, which is the fastest backend
    #: available on both CPU and GPU without extra dependencies. It is safe here
    #: because the loader clears the model's ``sliding_window`` before
    #: construction (so sdpa never sees a window it cannot implement and emits no
    #: warning). "eager" is the slow reference implementation; "flash_attention_2"
    #: needs a compatible GPU.
    llm_attn_implementation: str = "sdpa"
    #: Weight precision for the model. The default loads plain **float32**
    #: (full-precision, no quantization): the highest-fidelity option, and the
    #: dtype with the most complete set of CPU kernels, so it is also the most
    #: robust choice on a CPU-only host. It uses ~2x the memory of a 16-bit load;
    #: set ``float16``/``bfloat16`` to halve memory at a small quality cost, or
    #: ``auto`` to pick float16 on CUDA and bfloat16 elsewhere.
    #: Allowed: "float32" | "float16" | "bfloat16" | "auto".
    llm_torch_dtype: str = "float32"
    llm_trust_remote_code: bool = False
    #: Load the weights with ``low_cpu_mem_usage=True`` (streams the checkpoint
    #: instead of building a full fp32 copy first). Cuts peak RAM and cold-load
    #: time; requires ``accelerate`` (already a dependency).
    llm_low_cpu_mem_usage: bool = True
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

        The upper bound is the model's maximum generation length, which equals
        its context window: ``prompt_tokens + max_new_tokens`` must stay within
        the window or the position embeddings run out, so a value above either
        ceiling is rejected at configuration time rather than silently
        misbehaving at request time.
        """
        if value < 1:
            raise ValueError("llm_max_new_tokens must be at least 1")
        if value > LLM_MAX_GENERATION_TOKENS:
            raise ValueError(
                f"llm_max_new_tokens must not exceed {LLM_MAX_GENERATION_TOKENS} "
                "(the model's maximum generation length)"
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
