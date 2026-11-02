"""Local inference with Hugging Face ``transformers`` + Qwen2.5-0.5B.

The model is loaded lazily on first use and cached for the lifetime of the
process (a Celery worker, a Gunicorn worker or the dev server). Generation uses
``TextIteratorStreamer`` so the API can push tokens over SSE as they are
produced instead of waiting for the full completion.

When ``outlines`` is installed, :meth:`TransformersLLMProvider.generate_structured`
constrains decoding to a Pydantic schema (masked logits), which turns the same
Qwen2.5-0.5B weights into a reliable structured-output model for RAG answers.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm.base import GenerationResult, LLMProvider, Message

logger = get_logger(__name__)

# Module-level cache so multiple provider instances share one loaded model.
_model_cache: dict[str, tuple[Any, Any, str]] = {}
_cache_lock = threading.Lock()


def clear_model_cache() -> None:
    """Release loaded weights (tests / shutdown)."""
    with _cache_lock:
        _model_cache.clear()


class TransformersLLMProvider(LLMProvider):
    """Qwen2.5-0.5B-Instruct served through local transformers weights."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or get_settings()
        self.model_name = self._config.llm_model_id
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str = "cpu"
        # Cached outlines model wrapper + one compiled Generator per schema.
        self._outlines_model: Any | None = None
        self._outlines_generators: dict[Any, Any] = {}

    # ------------------------------------------------------------ model load
    def _resolve_device(self) -> str:
        device = (self._config.llm_device or "cpu").lower()
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except ImportError:  # pragma: no cover - torch present in prod image
            pass
        return "cpu"

    def _resolve_dtype(self, device: str) -> Any:  # noqa: ANN401 - torch dtype
        import torch

        requested = (self._config.llm_torch_dtype or "auto").lower()
        if requested == "auto":
            return torch.float16 if device.startswith("cuda") else torch.float32
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(requested, torch.float32)

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Load tokenizer + model once per process (thread-safe)."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        with _cache_lock:
            cached = _model_cache.get(self.model_name)
            if cached is not None:
                self._tokenizer, self._model, self._device = cached
                return self._tokenizer, self._model

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - dependency missing
                raise LLMError(
                    "transformers is not installed. Set LLM_PROVIDER=stub or install it.",
                    code="transformers_missing",
                ) from exc

            device = self._resolve_device()
            dtype = self._resolve_dtype(device)
            logger.info("Loading LLM '%s' on %s (dtype=%s)", self.model_name, device, dtype)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self._config.llm_trust_remote_code,
                    cache_dir=self._config.hf_home or None,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    trust_remote_code=self._config.llm_trust_remote_code,
                    cache_dir=self._config.hf_home or None,
                )
                model.to(device)
                model.eval()
            except Exception as exc:
                raise LLMError(
                    f"Failed to load model '{self.model_name}': {exc}",
                    code="model_load_failed",
                ) from exc

            _model_cache[self.model_name] = (tokenizer, model, device)
            self._tokenizer, self._model, self._device = tokenizer, model, device
            logger.info("LLM '%s' ready.", self.model_name)
            return tokenizer, model

    # ------------------------------------------------------------- prompting
    def _render_prompt(self, messages: list[Message]) -> str:
        """Render ``messages`` to the raw chat-template string."""
        tokenizer, _ = self._ensure_loaded()
        payload = [message.to_dict() for message in messages]

        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
            )

        # Fallback for tokenizers without a chat template.
        rendered = "\n".join(
            f"<|{message.role}|>\n{message.content}" for message in messages
        )
        return rendered + "\n<|assistant|>\n"

    def _prepare_inputs(self, messages: list[Message]) -> dict[str, Any]:
        tokenizer, _ = self._ensure_loaded()
        return dict(tokenizer(self._render_prompt(messages), return_tensors="pt"))

    def _resolve_generation_budget(
        self, max_new_tokens: int | None, prompt_tokens: int | None
    ) -> int:
        """Clamp the generation budget to what the context window can hold.

        The context window is a hard total budget, not just an input limit:
        ``prompt_tokens + max_new_tokens`` must stay within it or the position
        embeddings run out. A long retrieved context (or a long history) can
        therefore legitimately reduce the room left for the answer, so the
        budget is trimmed here rather than letting generation fail or silently
        truncate the prompt.
        """
        budget = max_new_tokens or self._config.llm_max_new_tokens
        if not prompt_tokens:
            return budget

        room = self._config.llm_context_window_tokens - prompt_tokens
        if room < budget:
            logger.warning(
                "Trimming max_new_tokens from %s to %s: the %s-token prompt "
                "leaves only %s of the %s-token context window.",
                budget,
                max(room, 1),
                prompt_tokens,
                max(room, 0),
                self._config.llm_context_window_tokens,
            )
            budget = max(room, 1)
        return budget

    def _count_prompt_tokens(self, messages: list[Message]) -> int:
        """Token count of the rendered prompt (0 when it cannot be computed)."""
        try:
            tokenizer, _ = self._ensure_loaded()
            encoded = tokenizer(self._render_prompt(messages))
            return int(len(encoded["input_ids"]))
        except Exception:  # noqa: BLE001 - budgeting must never break generation
            return 0

    def _generation_kwargs(
        self,
        *,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        prompt_tokens: int | None = None,
    ) -> dict[str, Any]:
        resolved_temperature = (
            self._config.llm_temperature if temperature is None else temperature
        )
        tokenizer, _ = self._ensure_loaded()
        budget = self._resolve_generation_budget(max_new_tokens, prompt_tokens)

        kwargs: dict[str, Any] = {
            "max_new_tokens": budget,
            "top_p": top_p if top_p is not None else self._config.llm_top_p,
            "do_sample": resolved_temperature > 0,
            "repetition_penalty": 1.05,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if resolved_temperature > 0:
            kwargs["temperature"] = max(resolved_temperature, 1e-2)
        return kwargs

    def _move_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        _, model = self._ensure_loaded()
        device = next(model.parameters()).device
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    # ---------------------------------------------------------------- public
    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        import torch

        tokenizer, model = self._ensure_loaded()
        inputs = self._move_to_device(self._prepare_inputs(messages))
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        gen_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_tokens=prompt_tokens,
        )

        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **gen_kwargs)
        except Exception as exc:
            raise LLMError(f"Generation failed: {exc}", code="generation_failed") from exc

        # Strip the prompt prefix so only the completion is returned.
        generated = output_ids[0][prompt_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        finish_reason = (
            "length" if generated.shape[-1] >= gen_kwargs["max_new_tokens"] else "stop"
        )
        return GenerationResult(
            text=text,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated.shape[-1]),
            finish_reason=finish_reason,
            metadata={"provider": "transformers", "device": self._device},
        )

    def stream(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        tokenizer, model = self._ensure_loaded()
        inputs = self._move_to_device(self._prepare_inputs(messages))
        gen_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
        )

        # The streamer timeout is the maximum gap between two yielded tokens,
        # not the total budget: a large answer legitimately streams for minutes
        # on CPU, so this only needs to be long enough to cover the slowest
        # single forward pass.
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600.0
        )
        gen_kwargs.update(streamer=streamer, **inputs)

        error: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
                error.append(exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=_run, name="qwen-stream", daemon=True)
        thread.start()

        for delta in streamer:
            if delta:
                yield delta
        thread.join(timeout=5)

        if error:
            raise LLMError(f"Streaming failed: {error[0]}", code="generation_failed")

    # ------------------------------------------------------ structured output
    def supports_structured_output(self) -> bool:
        """True when ``outlines`` is importable (enables constrained decoding)."""
        return self._outlines_available()

    @staticmethod
    def _outlines_available() -> bool:
        try:
            import outlines  # noqa: F401
        except ImportError:  # pragma: no cover - optional dependency
            return False
        return True

    def _build_outlines_model(self) -> Any | None:  # noqa: ANN401 - outlines type
        """Wrap the loaded transformers model/tokenizer in an outlines model.

        ``outlines.from_transformers`` returns a *model* wrapper (not a
        generator); the output type is applied later via ``outlines.Generator``.
        It reuses the already-loaded weights, so no second copy is created.
        """
        if self._outlines_model is not None:
            return self._outlines_model

        try:
            import outlines
        except ImportError:  # pragma: no cover - optional dependency
            return None

        tokenizer, model = self._ensure_loaded()
        try:
            self._outlines_model = outlines.from_transformers(model, tokenizer)
        except Exception as exc:  # noqa: BLE001 - outlines API/version drift
            logger.warning("outlines is installed but unusable: %s", exc)
            self._outlines_model = None
        return self._outlines_model

    def generate_structured(
        self,
        messages: list[Message],
        schema: type[Any],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """Constrained generation: only tokens valid for ``schema`` are emitted.

        ``outlines`` compiles the Pydantic model into a token-level state machine
        and masks the logits at every step, so even a 0.5B model returns valid,
        parseable JSON. The messages are passed as an ``outlines.inputs.Chat`` so
        outlines renders them through the model's own chat template exactly once
        (a bare string would be re-wrapped as a single user message, dropping the
        system prompt and history).

        Returns a validated ``schema`` instance, or ``None`` (so the caller falls
        back to plain text) when outlines is unavailable or generation fails.
        """
        import torch

        outlines_model = self._build_outlines_model()
        if outlines_model is None:
            return None

        try:
            import outlines
            from outlines.inputs import Chat

            # A Generator caches the compiled logits processor, so build one per
            # schema and reuse it across calls.
            generator = self._outlines_generators.get(schema)
            if generator is None:
                generator = outlines.Generator(outlines_model, schema)
                self._outlines_generators[schema] = generator

            chat = Chat([message.to_dict() for message in messages])
            resolved_temperature = (
                self._config.llm_temperature if temperature is None else temperature
            )
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self._resolve_generation_budget(
                    max_new_tokens, self._count_prompt_tokens(messages)
                ),
                "do_sample": resolved_temperature > 0,
            }
            if resolved_temperature > 0:
                gen_kwargs["temperature"] = max(resolved_temperature, 1e-2)
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

            with torch.inference_mode():
                raw = generator(chat, **gen_kwargs)
        except Exception as exc:  # noqa: BLE001 - degrade to plain text
            logger.warning("Structured generation failed, falling back: %s", exc)
            return None

        return self._coerce_schema(schema, raw)

    @staticmethod
    def _coerce_schema(schema: type[Any], raw: Any) -> Any | None:
        """Parse outlines' JSON string output into ``schema`` (or ``None``)."""
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            return schema.model_validate_json(raw)
        except Exception:  # noqa: BLE001 - last-resort parse before giving up
            try:
                import json

                return schema(**json.loads(raw))
            except Exception:  # noqa: BLE001 - caller falls back to plain text
                return None

    def health(self) -> dict[str, Any]:
        loaded = self._tokenizer is not None and self._model is not None
        return {
            "provider": "transformers",
            "model": self.model_name,
            "status": "ok",
            "loaded": loaded,
            "device": self._device,
            "structured_output": self._outlines_available(),
        }

    def warmup(self) -> None:
        """Force weight loading (called from the Celery worker bootstrap)."""
        self._ensure_loaded()

    def count_tokens(self, text: str) -> int:
        """Exact token count when the tokenizer is available."""
        if not text:
            return 0
        try:
            tokenizer, _ = self._ensure_loaded()
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # pragma: no cover - fall back to heuristic
            return super().count_tokens(text)
