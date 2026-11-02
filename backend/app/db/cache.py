"""Cache layer: Redis, with an in-process safety net.

Responsibilities
----------------
* Cache hot reads (document listings, retrieval results, embeddings).
* Store ephemeral session/task metadata.
* Provide a shared counter primitive for the manual rate limiter.

Redis is the configured backend (``REDIS_URL``). A thread-safe in-memory store is
used only when Redis is unconfigured or unreachable, which keeps the test suite
infrastructure-free without changing the public surface.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "rag:cache"


class InMemoryCache:
    """Thread-safe TTL cache used only when Redis is unavailable."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and expires_at <= time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        with self._lock:
            if len(self._store) >= self._max_entries:
                self._purge_expired_locked()
            expires_at = time.time() + ttl if ttl else None
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in [k for k in self._store if k.startswith(prefix)]:
                self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [
            key
            for key, (_, expires_at) in self._store.items()
            if expires_at is not None and expires_at <= now
        ]
        for key in expired:
            self._store.pop(key, None)
        # Still full of live entries: drop the oldest half.
        if len(self._store) >= self._max_entries:
            for key in list(self._store)[: self._max_entries // 2]:
                self._store.pop(key, None)


class CacheClient:
    """Facade over Redis, falling back to an in-process store."""

    def __init__(
        self,
        config: Settings | None = None,
        *,
        redis_client: Any | None = None,
        use_memory: bool = False,
    ) -> None:
        self._config = config or get_settings()
        self._memory = InMemoryCache()
        self._redis: Any | None = None

        if redis_client is not None:
            self._redis = redis_client
        elif not use_memory:
            self._redis = self._build_redis_client()

    # ------------------------------------------------------------- internals
    def _build_redis_client(self) -> Any | None:  # noqa: ANN401 - optional dep
        if not self._config.redis_url:
            logger.info("REDIS_URL not set: cache using in-memory backend.")
            return None
        try:
            import redis  # lazy import keeps redis optional

            client = redis.Redis.from_url(
                self._config.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            logger.info("Cache using Redis backend at %s", self._config.redis_url)
            return client
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Redis unavailable (%s): cache using in-memory backend.", exc)
            return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    @staticmethod
    def build_key(*parts: Any) -> str:  # noqa: ANN401 - heterogeneous parts
        joined = ":".join(str(part) for part in parts if part is not None)
        return f"{_KEY_PREFIX}:{joined}"

    # ---------------------------------------------------------------- public
    def get(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                return self._redis.get(key)
            except Exception as exc:  # pragma: no cover
                logger.warning("Cache get failed for %s: %s", key, exc)
        return self._memory.get(key)

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        if self._redis is not None:
            try:
                if ttl:
                    self._redis.setex(key, ttl, value)
                else:
                    self._redis.set(key, value)
                return
            except Exception as exc:  # pragma: no cover
                logger.warning("Cache set failed for %s: %s", key, exc)
        self._memory.set(key, value, ttl)

    def delete(self, key: str) -> None:
        if self._redis is not None:
            try:
                self._redis.delete(key)
                return
            except Exception as exc:  # pragma: no cover
                logger.warning("Cache delete failed for %s: %s", key, exc)
        self._memory.delete(key)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every key starting with ``prefix``; returns keys removed."""
        removed = 0
        if self._redis is not None:
            try:
                for found in self._redis.scan_iter(match=f"{prefix}*"):
                    self._redis.delete(found)
                    removed += 1
                return removed
            except Exception as exc:  # pragma: no cover
                logger.warning("Cache prefix delete failed for %s: %s", prefix, exc)
        self._memory.delete_prefix(prefix)
        return 0

    def clear(self) -> None:
        if self._redis is not None:
            try:
                for found in self._redis.scan_iter(match=f"{_KEY_PREFIX}:*"):
                    self._redis.delete(found)
            except Exception as exc:  # pragma: no cover
                logger.warning("Cache clear failed: %s", exc)
        self._memory.clear()

    # --------------------------------------------------------- JSON helpers
    def get_json(self, key: str) -> Any | None:  # noqa: ANN401 - decoded JSON
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: ANN401
        self.set(key, json.dumps(value, default=str), ttl)

    def incr(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        """Increment a counter, applying ``ttl`` on first write."""
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline(transaction=True)
                pipe.incrby(key, amount)
                if ttl:
                    pipe.expire(key, ttl, nx=True)
                new_value, _ = pipe.execute()
                return int(new_value)
            except Exception as exc:  # pragma: no cover
                logger.warning("Cache incr failed for %s: %s", key, exc)
        current = self.get(key)
        new_value = int(current or 0) + amount
        self.set(key, str(new_value), ttl)
        return new_value


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_cache_lock = threading.Lock()
_cache: CacheClient | None = None


def get_cache() -> CacheClient:
    """Return the lazily created process-wide cache client."""
    global _cache  # noqa: PLW0603
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = CacheClient()
    return _cache


def reset_cache() -> None:
    """Drop the singleton (tests / shutdown)."""
    global _cache  # noqa: PLW0603
    _cache = None
