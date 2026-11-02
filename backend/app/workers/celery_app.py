"""Celery application: broker/backend wiring, queues and task discovery.

The broker and result backend are Redis by default (``REDIS_URL``), so ingestion
runs in a separate worker process:

    celery -A app.workers.worker:celery_app worker --loglevel=info

Setting ``CELERY_TASK_ALWAYS_EAGER=true`` instead executes every task inline in
the calling process, which is what the test suite uses so it needs no broker.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)

celery_app = Celery(
    "rag_worker",
    broker=settings.broker_url,
    backend=settings.result_backend_url,
    include=[
        "app.workers.tasks.ingestion",
        "app.workers.tasks.maintenance",
    ],
)

celery_app.conf.update(
    # ------------------------------------------------------------- serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # ------------------------------------------------------------------ routing
    task_default_queue="default",
    task_default_exchange="rag",
    task_default_routing_key="rag.default",
    task_routes={
        "app.workers.tasks.ingestion.*": {"queue": "ingestion"},
        "app.workers.tasks.maintenance.*": {"queue": "maintenance"},
    },
    task_queues=None,
    # -------------------------------------------------------------- execution
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=settings.celery_task_always_eager,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    worker_concurrency=settings.celery_worker_concurrency,
    # --------------------------------------------------------------- results
    result_expires=60 * 60 * 24,
    result_extended=True,
    # ------------------------------------------------------------------- time
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    task_soft_time_limit=60 * 20,
    task_time_limit=60 * 25,
)

# --------------------------------------------------------------------------- #
# Register this app as Celery's *global default* app.
#
# ``shared_task`` does not bind to this module's ``celery_app``: it returns a
# proxy that resolves the task through ``celery.current_app`` at call time. That
# value is thread-local and only populated in the thread that constructed the
# app (the ``celery`` CLI does the equivalent for a worker), so a dispatch from
# any *other* thread falls back to an implicit ``Celery('default')`` app. That
# implicit app knows nothing about ``task_always_eager`` or our broker, so tasks
# are published to the broker and never executed - documents stay ``pending``.
# This bites the test suite (Starlette's TestClient runs the ASGI app on a worker
# thread) and any threaded ASGI/WSGI server, so make the app the process-wide
# default for every thread.
# --------------------------------------------------------------------------- #
celery_app.set_default()

# Configure logging once for the worker process.
configure_logging(force=True)
logger.info(
    "Celery configured (broker=%s, eager=%s)",
    settings.broker_url,
    settings.celery_task_always_eager,
)
