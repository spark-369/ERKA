"""FastAPI application factory.

Wires together configuration, logging, middleware, exception handlers, the v1
router, CORS for the Next.js frontend and lifespan startup/shutdown hooks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.middleware import (
    RateLimitHeadersMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Enterprise **Domain-Specific RAG Knowledge Assistant** API.

* **Auth** - JWT access/refresh tokens with manual, scope-aware rate limiting.
* **Documents** - asynchronous ingestion (Celery + Redis) into an embedded,
  persistent **ChromaDB** collection using LangChain for loading/chunking.
* **Chat** - grounded answers from a locally hosted **Qwen2.5-0.5B-Instruct**
  model, delivered as a full response or streamed over **SSE**.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown: ensure schema, warm singletons, release resources."""
    settings = get_settings()
    configure_logging(settings.log_level, force=True)
    logger.info("Starting %s v%s (%s)", settings.app_name, __version__, settings.app_env)

    # Ensure the schema exists before serving traffic (idempotent).
    try:
        from app.db.init_db import init_database

        init_database()
    except Exception as exc:  # noqa: BLE001 - do not block boot on optional deps
        logger.warning("Database initialisation skipped: %s", exc)

    # Make sure upload/chroma directories exist.
    for path in (settings.upload_path, settings.chroma_path):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover
            logger.warning("Could not create directory %s: %s", path, exc)

    yield

    logger.info("Shutting down %s", settings.app_name)
    try:
        from app.db.session import shutdown_engine

        # Disposes the engine only if this process created it, so an engine
        # installed by an embedding process (the test suite) is left intact.
        shutdown_engine()
    except Exception:  # pragma: no cover
        pass


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the configured FastAPI application."""
    config = settings or get_settings()
    configure_logging(config.log_level, force=True)

    app = FastAPI(
        title=config.app_name,
        description=DESCRIPTION,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=config.debug,
    )

    # ---------------------------------------------------------------- middleware
    # Order matters: the outermost middleware runs first on the way in.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(RateLimitHeadersMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time-Ms",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
        ],
    )

    # ------------------------------------------------------------- error handlers
    register_exception_handlers(app)

    # -------------------------------------------------------------------- routes
    app.include_router(api_router, prefix=config.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": config.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": f"{config.api_v1_prefix}/health",
        }

    def custom_openapi() -> dict:
        """Attach a bearer security scheme so Swagger UI can authenticate."""
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema.setdefault("components", {}).setdefault("securitySchemes", {})[
            "BearerAuth"
        ] = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    return app


# Module-level ASGI app for ``uvicorn app.main:app``.
app = create_app()
