"""SQLAlchemy engine / session factory and the FastAPI session dependency.

PostgreSQL is the only supported database. Connection pooling is tuned from
settings so a single configuration works for the API process, Celery workers and
scripts.

:func:`configure_engine` exists as an explicit seam so an embedding process (the
test suite, or a script) can install a specific engine instead of the one derived
from settings.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def build_engine(config: Settings | None = None, *, database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for ``database_url`` (or settings)."""
    config = config or get_settings()
    url = database_url or config.database_url

    engine = create_engine(
        url,
        echo=config.db_echo,
        future=True,
        pool_pre_ping=config.db_pool_pre_ping,
        pool_size=config.db_pool_size,
        max_overflow=config.db_max_overflow,
    )

    logger.info("Database engine created for %s", _safe_url(url))
    return engine


def _safe_url(url: str) -> str:
    """Redact credentials before logging a database URL."""
    if "@" not in url:
        return url
    scheme, _, remainder = url.partition("://")
    _, _, host = remainder.rpartition("@")
    return f"{scheme}://***@{host}"


# Module level engine/session factory (built lazily on first import use).
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
#: True when this module created the engine itself (and may therefore dispose it).
_engine_owned: bool = False


def get_engine() -> Engine:
    """Return (and memoise) the process-wide engine."""
    global _engine, _engine_owned  # noqa: PLW0603
    if _engine is None:
        _engine = build_engine()
        _engine_owned = True
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return (and memoise) the process-wide session factory."""
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False
        )
    return _session_factory


def reset_engine() -> None:
    """Unconditionally dispose and forget the engine (explicit teardown)."""
    global _engine, _session_factory, _engine_owned  # noqa: PLW0603
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _engine_owned = False


def shutdown_engine() -> None:
    """Release the engine at application shutdown.

    Only an engine this module created is disposed *and cleared*. An engine
    installed through :func:`configure_engine` belongs to its installer and is
    left completely untouched - disposing it, or even just forgetting it, would
    make the next accessor rebuild an engine from settings. For the test suite
    that would mean losing the in-memory database and then trying to reach a
    real PostgreSQL server.
    """
    global _engine, _session_factory, _engine_owned  # noqa: PLW0603
    if not _engine_owned:
        return
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _engine_owned = False


def configure_engine(
    engine: Engine, *, session_factory: sessionmaker[Session] | None = None
) -> None:
    """Install ``engine`` as the process-wide engine.

    Every accessor (:func:`get_engine`, :func:`get_session_factory`,
    :func:`get_db`, :func:`session_scope` and therefore Celery tasks) uses this
    engine afterwards, and :func:`shutdown_engine` will leave it alone.

    The test suite uses it to run against an in-memory database so no
    PostgreSQL server is required. Call :func:`reset_engine` to undo.
    """
    global _engine, _session_factory, _engine_owned  # noqa: PLW0603
    _engine = engine
    _engine_owned = False
    _session_factory = session_factory or sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    logger.info("Database engine overridden with %s", _safe_url(str(engine.url)))


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a session and always closing it."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for non-request contexts (Celery workers, scripts)."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
