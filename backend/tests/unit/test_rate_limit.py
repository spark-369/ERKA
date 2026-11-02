"""Unit tests for the manual (custom) rate limiter."""

from __future__ import annotations

import threading

import pytest

from app.core.config import Settings
from app.core.exceptions import RateLimitError
from app.core.rate_limit import (
    InMemoryRateLimitStore,
    RateLimiter,
    RateLimitResult,
)

pytestmark = pytest.mark.unit


def make_settings(**overrides) -> Settings:
    """Settings with rate limiting enabled and tight limits."""
    defaults = {
        "rate_limit_enabled": True,
        "rate_limit_requests": 3,
        "rate_limit_window_seconds": 60,
        "rate_limit_auth_requests": 2,
        "rate_limit_chat_requests": 1,
        "rate_limit_upload_requests": 2,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestInMemoryStore:
    def test_allows_up_to_the_limit(self) -> None:
        store = InMemoryRateLimitStore()
        results = [store.hit("k", window_seconds=60, limit=3) for _ in range(3)]
        assert all(result.allowed for result in results)
        assert [result.remaining for result in results] == [2, 1, 0]

    def test_blocks_beyond_the_limit(self) -> None:
        store = InMemoryRateLimitStore()
        for _ in range(3):
            store.hit("k", window_seconds=60, limit=3)
        blocked = store.hit("k", window_seconds=60, limit=3)
        assert blocked.allowed is False
        assert blocked.remaining == 0
        assert blocked.reset_after >= 1

    def test_window_expiry_frees_quota(self) -> None:
        store = InMemoryRateLimitStore()
        # A zero-length window means every previous hit is immediately stale.
        store.hit("k", window_seconds=0, limit=1)
        assert store.hit("k", window_seconds=0, limit=1).allowed is True

    def test_keys_are_isolated(self) -> None:
        store = InMemoryRateLimitStore()
        assert store.hit("a", window_seconds=60, limit=1).allowed is True
        assert store.hit("a", window_seconds=60, limit=1).allowed is False
        assert store.hit("b", window_seconds=60, limit=1).allowed is True

    def test_reset_clears_state(self) -> None:
        store = InMemoryRateLimitStore()
        store.hit("k", window_seconds=60, limit=1)
        store.reset("k")
        assert store.hit("k", window_seconds=60, limit=1).allowed is True

    def test_concurrent_hits_never_exceed_limit(self) -> None:
        """The in-memory store must be safe under concurrent access."""
        store = InMemoryRateLimitStore()
        limit = 25
        allowed: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            outcome = store.hit("shared", window_seconds=60, limit=limit)
            with lock:
                allowed.append(outcome.allowed)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sum(allowed) == limit


class TestRateLimiter:
    def test_uses_in_memory_backend_without_redis(self) -> None:
        limiter = RateLimiter(make_settings(redis_url=None))
        assert limiter.backend == "memory"

    def test_scoped_limits_are_independent(self) -> None:
        limiter = RateLimiter(make_settings())
        assert limiter.check("chat", "user:1").allowed is True
        assert limiter.check("chat", "user:1").allowed is False  # chat limit is 1
        assert limiter.check("default", "user:1").allowed is True

    def test_identity_isolation(self) -> None:
        limiter = RateLimiter(make_settings())
        assert limiter.check("auth", "ip:1.1.1.1").allowed is True
        assert limiter.check("auth", "ip:1.1.1.1").allowed is True
        assert limiter.check("auth", "ip:1.1.1.1").allowed is False
        assert limiter.check("auth", "ip:2.2.2.2").allowed is True

    def test_raise_on_limit_raises_rate_limit_error(self) -> None:
        limiter = RateLimiter(make_settings())
        limiter.check("chat", "user:1")
        with pytest.raises(RateLimitError) as excinfo:
            limiter.check("chat", "user:1", raise_on_limit=True)
        assert excinfo.value.retry_after >= 1
        assert excinfo.value.status_code == 429

    def test_disabled_limiter_allows_everything(self) -> None:
        limiter = RateLimiter(make_settings(rate_limit_enabled=False))
        for _ in range(50):
            assert limiter.check("chat", "user:1").allowed is True

    def test_explicit_limit_overrides_config(self) -> None:
        limiter = RateLimiter(make_settings())
        assert limiter.check("chat", "u", limit=2).allowed is True
        assert limiter.check("chat", "u", limit=2).allowed is True
        assert limiter.check("chat", "u", limit=2).allowed is False

    def test_peek_does_not_consume_quota(self) -> None:
        limiter = RateLimiter(make_settings())
        limiter.check("default", "u")
        assert limiter.peek("default", "u").total_hits == 1
        assert limiter.peek("default", "u").total_hits == 1

    def test_reset_scope_clears_quota(self) -> None:
        limiter = RateLimiter(make_settings())
        limiter.check("chat", "u")
        assert limiter.check("chat", "u").allowed is False
        limiter.reset("chat")
        assert limiter.check("chat", "u").allowed is True

    def test_build_key_format(self) -> None:
        assert RateLimiter.build_key("chat", "user:9") == "ratelimit:chat:user:9"


class TestRateLimitResult:
    def test_headers_are_well_formed(self) -> None:
        result = RateLimitResult(
            allowed=True, limit=10, remaining=7, reset_after=42, total_hits=3
        )
        assert result.as_headers() == {
            "X-RateLimit-Limit": "10",
            "X-RateLimit-Remaining": "7",
            "X-RateLimit-Reset": "42",
        }

    def test_retry_after_is_never_zero(self) -> None:
        result = RateLimitResult(
            allowed=False, limit=1, remaining=0, reset_after=0, total_hits=1
        )
        assert result.retry_after == 1
