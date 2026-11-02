"""Celery entrypoint module (``celery -A app.workers.worker:celery_app``)."""

from app.workers.celery_app import celery_app

__all__ = ["celery_app"]
