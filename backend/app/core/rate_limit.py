"""Manual (custom) rate limiting.

This module intentionally implements rate limiting from scratch instead of
pulling in ``slowapi``/``limits``:

* **Algorithm** - sliding window log. Each request is recorded as a member of a
  Redis sorted set keyed by ``ratelimit:{scope}:{identity}`` with the score being
  the request timestamp in microseconds. Counting the members inside the current
  window yields an exact sliding window (no fixed-window bursts).
* **Atomicity** - the read/prune/count/write sequence runs inside a single
  ``MULTI/EXEC`` pipeline, so concurrent workers cannot over-admit a caller.
* **Fallback** - when Redis is unconfigured or unreachable, an in-process
  thread-safe store is used instead. Redis is the configured backend
  (``REDIS_URL``); the fallback exists so the test suite needs no infrastructure
  and so a Redis blip degrades rate limiting rather than failing requests.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "ratelimit"


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    """Outcome of a single rate-limit evaluation."""

    allowed: bool
    limit: int
    remaining: int
    reset_after: int
    total_hits: int

    @property
    def retry_after(self) -> int:
        return max(self.reset_after, 1)

    def as_headers(self) -> dict[str, str]:
        """Standard ``X-RateLimit-*`` headers for the response."""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(self.remaining, 0)),
            "X-RateLimit-Reset": str(self.reset_after),
        }


# --------------------------------------------------------------------------- #
# Storage backends
# --------------------------------------------------------------------------- #
class InMemoryRateLimitStore:
    """Thread-safe sliding-window log kept in process memory."""

    def __init__(self, max_entries: int = 100_000) -> None:
        self._hits: dict[str, deque[int]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def hit(self, key: str, *, window_seconds: int, limit: int) -> RateLimitResult:
        now_us = int(time.time() * 1_000_000)
        window_us = window_seconds * 1_000_000
        cutoff = now_us - window_us

        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            allowed = len(bucket) < limit
            if allowed:
                bucket.append(now_us)

            total_hits = len(bucket)
            oldest = bucket[0] if bucket else now_us
            reset_after = max(int((oldest + window_us - now_us) / 1_000_000), 1)

            # Opportunistic eviction so long-running processes cannot leak.
            if len(self._hits) > self._max_entries:
                self._evict_locked(cutoff)

        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=max(limit - total_hits, 0),
            reset_after=reset_after,
            total_hits=total_hits,
        )

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)

    def _evict_locked(self, cutoff: int) -> None:
        stale = [
            key for key, bucket in self._hits.items() if not bucket or bucket[-1] <= cutoff
        ]
        for key in stale:
            self._hits.pop(key, None)


class RedisRateLimitStore:
    """Sliding-window log backed by a Redis sorted set."""

    def __init__(self, client: Any) -> None:  # noqa: ANN401 - redis client duck type
        self._client = client

    def hit(self, key: str, *, window_seconds: int, limit: int) -> RateLimitResult:
        now_us = int(time.time() * 1_000_000)
        window_us = window_seconds * 1_000_000
        cutoff = now_us - window_us
        member = f"{now_us}-{time.monotonic_ns()}"

        # Atomic prune + count + read-oldest inside one MULTI/EXEC block.
        # ``execute()`` yields one result per queued command, in order:
        # [removed, cardinality, oldest]. Unpacking that into two names raised
        # ``ValueError`` on every call, which ``RateLimiter.check`` swallowed by
        # falling back to the in-process store - so Redis-backed limiting never
        # actually ran. ``total_hits`` must be the post-prune *cardinality*
        # (members still inside the window), not the number of members removed.
        pipe = self._client.pipeline(transaction=True)
        pipe.zremrangebyscore(key, 0, cutoff)
        pipe.zcard(key)
        pipe.zrange(key, 0, 0, withscores=True)
        _removed, cardinality, oldest_pair = pipe.execute()

        total_hits = int(cardinality or 0)
        allowed = total_hits < limit

        pipe = self._client.pipeline(transaction=True)
        if allowed:
            pipe.zadd(key, {member: now_us})
            total_hits += 1
        pipe.expire(key, window_seconds + 1)
        pipe.execute()

        oldest_score = now_us
        if oldest_pair:
            oldest_score = int(oldest_pair[0][1])
        reset_after = max(int((oldest_score + window_us - now_us) / 1_000_000), 1)

        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=max(limit - total_hits, 0),
            reset_after=reset_after,
            total_hits=total_hits,
        )

    def reset(self, key: str | None = None) -> None:
        if key is None:
            for found in self._client.scan_iter(match=f"{_KEY_PREFIX}:*"):
                self._client.delete(found)
            return
        self._client.delete(key)


# --------------------------------------------------------------------------- #
# Limiter
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Scope-aware manual rate limiter.

    Example
    -------
    >>> limiter = RateLimiter()
    >>> result = limiter.check("chat", "user:42")
    >>> result.remaining
    19
    """

    def __init__(
        self,
        config: Settings | None = None,
        *,
        store: Any | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self._config = config or get_settings()
        self._memory_store = InMemoryRateLimitStore()
        self._redis_store: RedisRateLimitStore | None = None

        if store is not None:
            self._store = store
        else:
            client = redis_client if redis_client is not None else self._build_redis_client()
            if client is not None:
                self._redis_store = RedisRateLimitStore(client)
                self._store = self._redis_store
            else:
                self._store = self._memory_store

    # ------------------------------------------------------------- internals
    def _build_redis_client(self) -> Any | None:  # noqa: ANN401 - optional dep
        if not self._config.redis_url:
            return None
        try:
            import redis  # imported lazily: optional at runtime

            client = redis.Redis.from_url(
                self._config.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            logger.info("Rate limiter using Redis backend at %s", self._config.redis_url)
            return client
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning(
                "Redis unavailable for rate limiting (%s); using in-memory store.", exc
            )
            return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis_store is not None else "memory"

    @staticmethod
    def build_key(scope: str, identity: str) -> str:
        return f"{_KEY_PREFIX}:{scope}:{identity}"

    # ---------------------------------------------------------------- public
    def check(
        self,
        scope: str = "default",
        identity: str = "anonymous",
        *,
        limit: int | None = None,
        window_seconds: int | None = None,
        raise_on_limit: bool = False,
    ) -> RateLimitResult:
        """Record a hit for ``identity`` in ``scope`` and report the outcome."""
        default_limit, default_window = self._config.scoped_rate_limit(scope)
        effective_limit = limit or default_limit
        effective_window = window_seconds or default_window

        if not self._config.rate_limit_enabled:
            return RateLimitResult(
                allowed=True,
                limit=effective_limit,
                remaining=effective_limit,
                reset_after=effective_window,
                total_hits=0,
            )

        key = self.build_key(scope, identity)
        try:
            result = self._store.hit(
                key, window_seconds=effective_window, limit=effective_limit
            )
        except Exception as exc:  # pragma: no cover - redis outage mid-flight
            logger.warning("Rate limit store failed (%s); falling back to memory.", exc)
            result = self._memory_store.hit(
                key, window_seconds=effective_window, limit=effective_limit
            )

        if not result.allowed:
            logger.warning(
                "Rate limit exceeded scope=%s identity=%s hits=%s limit=%s",
                scope,
                identity,
                result.total_hits,
                result.limit,
            )
            if raise_on_limit:
                raise RateLimitError(
                    f"Rate limit exceeded for '{scope}'. "
                    f"Try again in {result.retry_after}s.",
                    retry_after=result.retry_after,
                    limit=result.limit,
                    remaining=0,
                )
        return result

    def peek(self, scope: str = "default", identity: str = "anonymous") -> RateLimitResult:
        """Report the current state without consuming a hit."""
        limit, window = self._config.scoped_rate_limit(scope)
        key = self.build_key(scope, identity)
        now_us = int(time.time() * 1_000_000)
        cutoff = now_us - window * 1_000_000

        if isinstance(self._store, RedisRateLimitStore):
            client = self._store._client  # noqa: SLF001 - internal store detail
            client.zremrangebyscore(key, 0, cutoff)
            total_hits = int(client.zcard(key) or 0)
            oldest = client.zrange(key, 0, 0, withscores=True)
            oldest_score = int(oldest[0][1]) if oldest else now_us
        else:
            bucket = self._memory_store._hits.get(key, deque())  # noqa: SLF001
            hits = [hit for hit in bucket if hit > cutoff]
            total_hits = len(hits)
            oldest_score = hits[0] if hits else now_us

        reset_after = max(int((oldest_score + window * 1_000_000 - now_us) / 1_000_000), 1)
        return RateLimitResult(
            allowed=total_hits < limit,
            limit=limit,
            remaining=max(limit - total_hits, 0),
            reset_after=reset_after,
            total_hits=total_hits,
        )

    def reset(self, scope: str | None = None, identity: str | None = None) -> None:
        """Clear counters (all, per scope, or per scope+identity)."""
        if scope is None:
            self._store.reset(None)
            return
        if identity is None:
            prefix = f"{_KEY_PREFIX}:{scope}:"
            if isinstance(self._store, RedisRateLimitStore):
                for found in self._store._client.scan_iter(  # noqa: SLF001
                    match=f"{prefix}*"
                ):
                    self._store.reset(found)
            else:
                keys = [k for k in self._memory_store._hits if k.startswith(prefix)]  # noqa: SLF001
                for key in keys:
                    self._store.reset(key)
            return
        self._store.reset(self.build_key(scope, identity))


# --------------------------------------------------------------------------- #
# Process-wide singleton (used by the FastAPI dependency)
# --------------------------------------------------------------------------- #
_limiter_lock = threading.Lock()
_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Return the lazily created process-wide limiter."""
    global _limiter  # noqa: PLW0603
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter()
    return _limiter


def reset_rate_limiter() -> None:
    """Drop the singleton (used by tests and on application shutdown)."""
    global _limiter  # noqa: PLW0603
    _limiter = None
