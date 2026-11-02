"""Bootstrap helpers: create tables and seed development data."""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.base import Base
from app.db.session import get_engine, session_scope

# Importing the models package registers every table on ``Base.metadata``.
from app.models import (  # noqa: F401  (import for side effects)
    BackgroundTask,
    Conversation,
    Document,
    DocumentChunk,
    Message,
    User,
)

logger = get_logger(__name__)


def create_all() -> None:
    """Create every table that does not exist yet.

    Alembic owns schema migrations in production; this helper exists so local
    development, Docker first-boot and the test suite work with one call.
    """
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured (%s tables).", len(Base.metadata.tables))


def seed_default_admin() -> None:
    """Create the bootstrap admin account when one is configured."""
    import os

    email = os.getenv("DEFAULT_ADMIN_EMAIL")
    password = os.getenv("DEFAULT_ADMIN_PASSWORD")
    if not email or not password:
        return

    from app.services.auth_service import AuthService

    with session_scope() as session:
        AuthService(session).ensure_default_admin(email, password)


def init_database() -> None:
    """Full initialisation used by the app lifespan and CLI entrypoints."""
    settings = get_settings()
    create_all()
    if settings.app_env == "development":
        seed_default_admin()
