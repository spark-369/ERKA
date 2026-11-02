"""Celery application factory and worker configuration."""

from app.workers.celery_app import celery_app

__all__ = ["celery_app"]
