"""Provider factory: single place that decides which LLM backend to use."""

from __future__ import annotations

import threading

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.llm.stub import StubLLMProvider
from app.llm.transformers_llm import TransformersLLMProvider

logger = get_logger(__name__)

_provider_lock = threading.Lock()
_provider: LLMProvider | None = None


def build_llm_provider(config: Settings | None = None) -> LLMProvider:
    """Construct a provider from settings (no caching)."""
    config = config or get_settings()
    if config.llm_provider == "stub":
        logger.info("LLM provider: stub (deterministic, offline)")
        return StubLLMProvider(config=config)
    logger.info("LLM provider: transformers (%s)", config.llm_model_id)
    return TransformersLLMProvider(config=config)


def get_llm_provider() -> LLMProvider:
    """Return the lazily created process-wide provider singleton."""
    global _provider  # noqa: PLW0603
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = build_llm_provider()
    return _provider


def reset_llm_provider(provider: LLMProvider | None = None) -> None:
    """Replace (or clear) the singleton - used by tests."""
    global _provider  # noqa: PLW0603
    _provider = provider
