"""Pytest configuration and shared fixtures.

The suite runs **without any external service** - no PostgreSQL, no Redis, no
model downloads:

* **SQLite** (temporary file, WAL mode) for the database. The application itself
  is PostgreSQL-only; SQLite exists purely as a test fixture (see
  :func:`test_engine`) so `pytest` works on a clean machine with no server.
* **In-memory cache / eager Celery** instead of Redis.
* **Stub LLM + hash embeddings** instead of Qwen2.5 / sentence-transformers.
* **Embedded Chroma** against a temp directory, which is exactly what
  production does.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Environment must be configured BEFORE importing application modules, because
# ``Settings`` is cached and singletons are built at import time.
# --------------------------------------------------------------------------- #
_TEST_ROOT = Path(__file__).resolve().parent
#: Root for throwaway test artefacts (embedded Chroma, uploads, HF cache).
#: The database itself is created in a pytest ``tmp_path`` so no run can inherit
#: state from a previous one.
_TMP_ROOT = _TEST_ROOT / ".tmp"
_TMP_ROOT.mkdir(exist_ok=True)

os.environ.update(
    {
        "APP_ENV": "test",
        "DEBUG": "false",
        "LOG_LEVEL": "WARNING",
        "SECRET_KEY": "test-secret-key-for-unit-and-integration-tests-0123456789",
        # Never contacted: the test engine is installed via configure_engine().
        "DATABASE_URL": "sqlite+pysqlite:///:memory:",
        "REDIS_URL": "",
        "CELERY_BROKER_URL": "memory://",
        "CELERY_RESULT_BACKEND": "cache+memory://",
        "CELERY_TASK_ALWAYS_EAGER": "true",
        "CHROMA_PERSIST_DIR": str(_TMP_ROOT / "chroma"),
        "UPLOAD_DIR": str(_TMP_ROOT / "uploads"),
        "HF_HOME": str(_TMP_ROOT / "hf"),
        "LLM_PROVIDER": "stub",
        "EMBEDDING_PROVIDER": "hash",
        "EMBEDDING_DIMENSION": "64",
        "CHUNK_SIZE": "400",
        "CHUNK_OVERLAP": "50",
        "RETRIEVAL_TOP_K": "4",
        "RETRIEVAL_SCORE_THRESHOLD": "0.0",
        "RATE_LIMIT_ENABLED": "false",
        "MAX_UPLOAD_SIZE_MB": "5",
    }
)

from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.rate_limit import reset_rate_limiter  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.cache import reset_cache  # noqa: E402
from app.db.session import configure_engine, reset_engine  # noqa: E402
from app.db.vector_store import reset_vector_store  # noqa: E402
from app.llm.factory import reset_llm_provider  # noqa: E402
from app.llm.stub import StubLLMProvider  # noqa: E402
from app.rag.embeddings import reset_embedding_provider  # noqa: E402
from app.rag.pipeline import reset_ingestion_pipeline  # noqa: E402
from app.rag.retriever import reset_retriever  # noqa: E402
from app.workers.celery_app import celery_app  # noqa: E402

# --------------------------------------------------------------------------- #
# Force inline task execution for the whole suite.
#
# ``CELERY_TASK_ALWAYS_EAGER`` is set in the environment above, but that has to
# win over ``backend/.env`` (which sets it to false so real workers consume the
# Redis queue). Rather than depend on that precedence - and on ``Settings`` not
# having been cached already - the Celery app is configured directly. Without
# this, ingestion is published to the broker and never consumed during tests, so
# documents stay ``pending`` and every citation assertion fails.
# --------------------------------------------------------------------------- #
celery_app.conf.task_always_eager = True
# Failures are recorded on the task result and by the task's own handler, rather
# than propagating into the request and turning a 202 into a 500.
celery_app.conf.task_eager_propagates = False


def build_test_engine(db_path: Path):
    """Create the SQLite engine used by the whole suite.

    ``NullPool`` gives every Session a brand-new connection. That matters because
    the suite interleaves Sessions - a request Session plus one per eager Celery
    task - and a *reused* pooled connection can carry a stale read snapshot, so a
    task could not see the document the request had just committed (observed as
    the first seed succeeding and every later one reporting ``pending``).
    Connections are cheap for a local file, so the clarity is worth it.

    The database is a file rather than ``:memory:`` because the suite genuinely
    uses concurrent connections, and ``check_same_thread=False`` is required
    because TestClient and the eager task run on different threads.
    """
    engine = create_engine(
        f"sqlite+pysqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
        """Enable foreign keys and lock waiting on every connection.

        The journal mode is deliberately left at SQLite's default (``delete``).
        WAL was tried and caused the eager Celery task to miss documents the
        request had just committed: WAL readers pin a snapshot, so a second
        connection can fail to observe a committed row.
        """
        cursor = dbapi_connection.cursor()
        try:
            # Off by default in SQLite; without it ON DELETE CASCADE is ignored.
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()

    return engine


def pytest_configure(config) -> None:  # noqa: ANN001 - pytest hook
    """Register custom markers.

    Done here (rather than in a pytest.ini / pyproject.toml) so the project
    keeps no configuration files of those types.
    """
    config.addinivalue_line("markers", "unit: fast, isolated tests with no external services")
    config.addinivalue_line(
        "markers", "integration: end-to-end tests through the FastAPI application"
    )


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings instance bound to the test environment."""
    return Settings()


@pytest.fixture(scope="session")
def test_engine(tmp_path_factory):
    """Session-scoped engine; the schema is created once for the whole run.

    Installed through :func:`configure_engine` so that every session source -
    request handlers, the streaming endpoint and the eager Celery tasks - shares
    this one engine.

    The database lives in a fresh temporary directory so a previous run can never
    leave stale data or journal files behind, which is the kind of state that
    produces "first test passes, the rest fail" results.
    """
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    engine = build_test_engine(db_path)
    # Import models so every table is registered before create_all().
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    configure_engine(engine)
    yield engine
    reset_engine()


@pytest.fixture(autouse=True)
def _clean_database(test_engine) -> Iterator[None]:
    """Empty every table before each test.

    Autouse (rather than attached to ``session``) so isolation also applies to
    tests that only use the HTTP client and never request a Session directly.
    Rows are deleted instead of recreating the schema, which keeps the reset
    cheap; the reverse dependency order avoids violating foreign keys.
    """
    import app.models  # noqa: F401

    with test_engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(table.delete())
    yield


@pytest.fixture()
def session(test_engine) -> Iterator[Session]:
    """A Session bound to the test engine, for tests that need the ORM directly."""
    factory = sessionmaker(bind=test_engine, expire_on_commit=False, autoflush=False)
    db = factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """Ensure process-wide singletons never leak between tests.

    ``reset_ingestion_pipeline`` matters as much as the others: the pipeline
    singleton caches a ``VectorStore`` reference, so without it a pipeline built
    during an earlier test would keep writing to that test's Chroma directory.

    The database engine is deliberately **not** reset here: it is owned by the
    session-scoped ``test_engine`` fixture. Disposing it per test would close
    every connection and discard the test database.
    """
    reset_llm_provider()
    reset_embedding_provider()
    reset_retriever()
    reset_vector_store()
    reset_ingestion_pipeline()
    reset_cache()
    reset_rate_limiter()
    yield
    reset_llm_provider()
    reset_embedding_provider()
    reset_retriever()
    reset_vector_store()
    reset_ingestion_pipeline()
    reset_cache()
    reset_rate_limiter()


@pytest.fixture()
def llm() -> StubLLMProvider:
    """Deterministic LLM provider."""
    return StubLLMProvider()


@pytest.fixture(scope="session")
def _shared_vector_store():
    """One embedded Chroma collection reused by the whole session.

    Creating a ``PersistentClient`` per test would spawn dozens of file handles
    and Chroma clients, so the collection is created once and simply emptied
    before each test instead.
    """
    from app.db.vector_store import VectorStore

    return VectorStore(
        persist_directory=_TMP_ROOT / "chroma",
        collection_name="test_knowledge_base",
    )


@pytest.fixture(autouse=True)
def _isolate_vector_store(_shared_vector_store, _reset_singletons) -> Iterator[None]:
    """Install a clean vector store as the process-wide singleton per test.

    Chroma persists to disk, so without clearing it a collection written by one
    test would be visible to the next (breaking "no context" assertions).
    """
    from app.db import vector_store as vector_store_module

    _shared_vector_store.reset()
    vector_store_module._store = _shared_vector_store  # noqa: SLF001 - singleton injection
    yield
    vector_store_module.reset_vector_store()


@pytest.fixture()
def vector_store(_shared_vector_store):
    """The shared, freshly cleared embedded collection."""
    _shared_vector_store.reset()
    return _shared_vector_store


# --------------------------------------------------------------------------- #
# HTTP client fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def app(test_engine):
    """FastAPI app whose ``get_db`` mirrors production: a **fresh session per request**.

    The naive override ``lambda: session`` reuses one Session for the whole test.
    SQLAlchemy caches loaded objects in that Session's identity map, so a
    ``Document`` read before an eager Celery task updated it would be returned in
    its stale state - the status endpoint kept reporting ``pending`` even though
    ingestion had committed ``ready``. Production opens a session per request, so
    the fixture does the same.
    """
    from app.db.session import get_db
    from app.main import create_app

    factory = sessionmaker(bind=test_engine, expire_on_commit=False, autoflush=False)

    def _override() -> Iterator[Session]:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    application = create_app()
    application.dependency_overrides[get_db] = _override
    yield application
    application.dependency_overrides.clear()


@pytest.fixture()
def client(app):
    """Synchronous ``TestClient`` (auth helper fixtures build on this)."""
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def user_factory(session: Session):
    """Create persisted users (bypasses HTTP for speed)."""
    from app.core.security import hash_password
    from app.models.user import User

    def _create(
        email: str = "owner@example.com",
        password: str = "Password123",
        *,
        role: str = "user",
        is_active: bool = True,
    ) -> User:
        user = User(
            email=email,
            hashed_password=hash_password(password),
            full_name=email.split("@")[0].title(),
            role=role,
            is_active=is_active,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    return _create


@pytest.fixture()
def auth_headers(current_user) -> dict[str, str]:
    """Bearer headers for the :func:`current_user` fixture."""
    from app.core.security import create_access_token

    token = create_access_token(current_user.id, scopes=["user"])
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def current_user(user_factory):
    """A persisted, active user account."""
    return user_factory()
