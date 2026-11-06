"""Health and readiness endpoints (used by Docker/compose health checks).

Rate limiting applies here too: health endpoints are cheap but a flooded
liveness probe is still an outage vector, and it guarantees the
``X-RateLimit-*`` headers are present on every response the API returns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__
from app.api.rate_limit_dep import rate_limit_default
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.common import HealthResponse

router = APIRouter(dependencies=[Depends(rate_limit_default)])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Cheap liveness check: the process is up and serving requests."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc),
        dependencies={"api": "ok"},
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
def readiness(
    session: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Deep readiness check: verify storage and inference dependencies."""
    dependencies: dict[str, str] = {}

    # --- PostgreSQL ----------------------------------------------------------
    try:
        session.execute(text("SELECT 1"))
        dependencies["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 - report, never raise from health
        dependencies["database"] = f"error: {exc.__class__.__name__}"

    # --- Redis cache ---------------------------------------------------------
    try:
        from app.db.cache import get_cache

        dependencies["cache"] = f"ok ({get_cache().backend})"
    except Exception as exc:  # noqa: BLE001
        dependencies["cache"] = f"error: {exc.__class__.__name__}"

    # --- Celery broker -------------------------------------------------------
    try:
        from app.workers.celery_app import celery_app

        broker = celery_app.connection_for_read()
        broker.ensure_connection(max_retries=0)
        broker.release()
        dependencies["broker"] = "ok"
    except Exception as exc:  # noqa: BLE001
        dependencies["broker"] = f"degraded: {exc.__class__.__name__}"

    # --- Embeddings ----------------------------------------------------------
    try:
        from app.rag.embeddings import get_embedding_provider

        dependencies["embeddings"] = get_embedding_provider().health()["status"]
    except Exception as exc:  # noqa: BLE001
        dependencies["embeddings"] = f"error: {exc.__class__.__name__}"

    # --- LLM -----------------------------------------------------------------
    try:
        from app.llm.factory import get_llm_provider

        provider = get_llm_provider()
        dependencies["llm"] = provider.health()["status"]
    except Exception as exc:  # noqa: BLE001
        dependencies["llm"] = f"error: {exc.__class__.__name__}"

    healthy = all(
        not value.startswith("error") for value in dependencies.values()
    )
    return HealthResponse(
        status="ok" if healthy else "degraded",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc),
        dependencies=dependencies,
    )
