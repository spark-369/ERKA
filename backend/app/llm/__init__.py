"""LLM inference layer: prompts, providers and the Qwen2.5 pipeline."""

from app.llm.base import GenerationResult, LLMProvider, Message
from app.llm.factory import get_llm_provider, reset_llm_provider
from app.llm.prompt import PromptBuilder, build_rag_prompt
from app.llm.stub import StubLLMProvider
from app.llm.transformers_llm import TransformersLLMProvider

__all__ = [
    "GenerationResult",
    "LLMProvider",
    "Message",
    "PromptBuilder",
    "StubLLMProvider",
    "TransformersLLMProvider",
    "build_rag_prompt",
    "get_llm_provider",
    "reset_llm_provider",
]
