"""Local inference with Hugging Face ``transformers`` + Qwen2.5-0.5B.

Everything here is synchronous: the model is loaded lazily on first use and
cached for the lifetime of the process (a Celery worker, a Gunicorn worker or
the dev server), and ``generate`` runs a single blocking ``model.generate`` call
that returns the whole completion.
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
# Each entry is ``(tokenizer, model, device, quantized)``.
_model_cache: dict[str, tuple[Any, Any, str, bool]] = {}
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
        self._quantized: bool = False

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

    def _configure_torch_threads(self, device: str) -> None:
        """Size the CPU thread pool for fast, synchronous generation.

        PyTorch's intra-op parallelism is the primary throughput lever for a
        CPU-only deployment: each generated token is a sequence of matmuls, and
        the default pool can under-use a multi-core box. ``LLM_NUM_THREADS=0``
        (the default) uses every logical core; a positive value caps it. This is
        a no-op for GPU devices, which parallelise on the accelerator instead.
        """
        if device.startswith("cuda") or device == "mps":
            return
        try:
            import os

            import torch

            configured = self._config.llm_num_threads
            threads = configured if configured and configured > 0 else (os.cpu_count() or 1)
            torch.set_num_threads(threads)
            logger.info("PyTorch CPU threads set to %d for inference.", threads)
        except Exception as exc:  # noqa: BLE001 - tuning must never block load
            logger.debug("Could not set PyTorch thread count (%s).", exc)

    def _resolve_attn_implementation(self) -> str:
        """Attention backend for ``from_pretrained``.

        Defaults to ``eager``, whose reference implementation applies the
        model's sliding window and emits no warnings. ``sdpa`` is faster where
        supported but ignores sliding window, so transformers warns that
        "unexpected results may be encountered" for models that declare one
        (Qwen2.5 sets ``sliding_window`` in its config).
        """
        return (self._config.llm_attn_implementation or "eager").lower()

    def _resolve_dtype(self, device: str) -> Any:  # noqa: ANN401 - torch dtype
        """Resolve the weight dtype, defaulting to float32.

        ``float32`` (the default request) is full precision and has the most
        complete CPU kernel coverage, so it is the most robust choice on a
        CPU-only host; it costs ~2x the memory of a 16-bit load. ``float16`` is
        16-bit and fastest where the kernels implement it, but on CPU/MPS it is
        coerced to ``bfloat16`` because half-precision CPU ops are largely
        unimplemented, so the load stays 16-bit without failing. ``auto`` picks
        float16 on CUDA and bfloat16 elsewhere.
        """
        import torch

        requested = (self._config.llm_torch_dtype or "float32").lower()
        if requested == "auto":
            return torch.float16 if device.startswith("cuda") else torch.bfloat16

        resolved = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(requested, torch.float32)

        if resolved is torch.float16 and not device.startswith("cuda"):
            logger.info(
                "float16 requested on %s; using bfloat16 (16-bit) instead "
                "(half-precision CPU kernels are unimplemented).",
                device,
            )
            return torch.bfloat16
        return resolved

    @staticmethod
    def _sanitise_model_config(model: Any) -> None:  # noqa: ANN401 - HF model
        """Quiet two harmless-but-noisy defaults on the loaded model.

        * ``sliding_window`` - Qwen2.5 declares a sliding window, which
          transformers flags ("not implemented for 'eager'") on every load for
          every backend short of flash-attention. Setting it to ``None`` stops
          the warning; for this model the window equals the full context length,
          so attention is effectively global anyway and behaviour is unchanged.
        * the checkpoint's ``generation_config`` sampling defaults
          (``temperature``/``top_p``/``top_k``) - these are only used when
          ``do_sample=True``; leaving them on the config makes transformers warn
          on every greedy call. Clearing them lets our explicit per-call kwargs
          (or greedy decoding) stand alone.
        """
        config = getattr(model, "config", None)
        if config is not None:
            if getattr(config, "sliding_window", None) is not None:
                config.sliding_window = None
            if getattr(config, "use_sliding_window", None):
                config.use_sliding_window = False

        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            for attribute in ("temperature", "top_p", "top_k"):
                if getattr(generation_config, attribute, None) is not None:
                    setattr(generation_config, attribute, None)

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Load tokenizer + model once per process (thread-safe)."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        with _cache_lock:
            cached = _model_cache.get(self.model_name)
            if cached is not None:
                self._tokenizer, self._model, self._device, self._quantized = cached
                return self._tokenizer, self._model

            try:
                from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - dependency missing
                raise LLMError(
                    "transformers is not installed. Set LLM_PROVIDER=stub or install it.",
                    code="transformers_missing",
                ) from exc

            device = self._resolve_device()
            self._configure_torch_threads(device)
            dtype = self._resolve_dtype(device)
            logger.info(
                "Loading LLM '%s' on %s (dtype=%s, attn=%s).",
                self.model_name,
                device,
                dtype,
                self._resolve_attn_implementation(),
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self._config.llm_trust_remote_code,
                    cache_dir=self._config.hf_home or None,
                )
                load_kwargs: dict[str, Any] = {
                    "attn_implementation": self._resolve_attn_implementation(),
                    "trust_remote_code": self._config.llm_trust_remote_code,
                    "cache_dir": self._config.hf_home or None,
                    "torch_dtype": dtype,
                    # Streams the checkpoint into the target dtype instead of
                    # building a full-precision copy first: lower peak RAM and a
                    # faster cold load. Needs ``accelerate`` (a dependency).
                    "low_cpu_mem_usage": self._config.llm_low_cpu_mem_usage,
                }
                # Load the config first and clear its sliding window so the
                # "Sliding Window Attention is enabled but not implemented for
                # 'sdpa'/'eager'" warning is never emitted (it fires during model
                # construction, so patching the model afterwards is too late).
                # The window equals the full context length here, so attention is
                # effectively global and behaviour is unchanged.
                try:
                    model_config = AutoConfig.from_pretrained(
                        self.model_name,
                        trust_remote_code=self._config.llm_trust_remote_code,
                        cache_dir=self._config.hf_home or None,
                    )
                    if getattr(model_config, "sliding_window", None) is not None:
                        model_config.sliding_window = None
                    if getattr(model_config, "use_sliding_window", None):
                        model_config.use_sliding_window = False
                    load_kwargs["config"] = model_config
                except Exception as exc:  # noqa: BLE001 - config is an optimisation
                    logger.debug("Could not pre-load model config (%s); using default.", exc)

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **load_kwargs
                )
                # Unquantized model: place it explicitly on the resolved device
                # before inference.
                model.to(device)
                model.eval()
                self._sanitise_model_config(model)
            except LLMError:
                raise
            except Exception as exc:
                raise LLMError(
                    f"Failed to load model '{self.model_name}': {exc}",
                    code="model_load_failed",
                ) from exc

            _model_cache[self.model_name] = (tokenizer, model, device, False)
            self._tokenizer, self._model, self._device = tokenizer, model, device
            self._quantized = False
            logger.info("LLM '%s' ready (%s).", self.model_name, dtype)
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

        The default budget is the model's full output capacity (the context
        window itself), so this clamp is a normal, expected event and is logged
        at INFO. When a caller *explicitly* asked for a budget that had to be
        cut, that is a genuine limitation and is logged at WARNING.
        """
        explicit = max_new_tokens is not None
        budget = max_new_tokens or self._config.llm_max_new_tokens
        if not prompt_tokens:
            return budget

        room = self._config.llm_context_window_tokens - prompt_tokens
        if room < budget:
            log = logger.warning if explicit else logger.info
            log(
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
        do_sample = resolved_temperature > 0

        kwargs: dict[str, Any] = {
            "max_new_tokens": budget,
            "do_sample": do_sample,
            "repetition_penalty": self._config.llm_repetition_penalty,
            "pad_token_id": tokenizer.eos_token_id,
            # Greedy decoding (a single beam) and the KV cache are the fastest
            # settings for synchronous generation: no beam search, and previously
            # computed keys/values are reused for every new token.
            "num_beams": 1,
            "use_cache": True,
        }
        # Banning repeated n-grams is what actually stops the small model from
        # looping the same phrase until it runs out of tokens.
        if self._config.llm_no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self._config.llm_no_repeat_ngram_size
        # temperature/top_p/top_k only participate in sample-based generation;
        # passing them with do_sample=False makes transformers warn on every
        # call, so they are attached only when we actually sample.
        if do_sample:
            kwargs["temperature"] = max(resolved_temperature, 1e-2)
            kwargs["top_p"] = top_p if top_p is not None else self._config.llm_top_p
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

    def generate_stream(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield decoded text as the model produces it (real token streaming).

        ``model.generate`` is blocking, so it runs on a worker thread while a
        Hugging Face :class:`TextIteratorStreamer` hands each newly decoded piece
        back to this (still synchronous) generator. The caller therefore receives
        the first words within a fraction of the full generation time instead of
        waiting for the whole answer.

        Prompt-token accounting is trivial here: the streamer emits only the
        completion, so the prompt prefix never has to be stripped.
        """
        import torch
        from transformers import TextIteratorStreamer

        tokenizer, model = self._ensure_loaded()
        inputs = self._move_to_device(self._prepare_inputs(messages))
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        gen_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_tokens=prompt_tokens,
        )

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        error: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**inputs, streamer=streamer, **gen_kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
                error.append(exc)
            finally:
                # Always close the streamer; otherwise a failure on the worker
                # thread would leave this generator blocked on ``for text in ...``.
                streamer.end()

        worker = threading.Thread(target=_run, name="llm-stream", daemon=True)
        worker.start()
        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            worker.join(timeout=1.0)

        if error:
            raise LLMError(
                f"Generation failed: {error[0]}", code="generation_failed"
            ) from error[0]

    def health(self) -> dict[str, Any]:
        loaded = self._tokenizer is not None and self._model is not None
        return {
            "provider": "transformers",
            "model": self.model_name,
            "status": "ok",
            "loaded": loaded,
            "device": self._device,
            "quantized_4bit": self._quantized,
            "dtype": (self._config.llm_torch_dtype or "float32"),
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
