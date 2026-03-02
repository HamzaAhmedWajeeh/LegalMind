"""
core/tasks/celery_app.py
========================
Celery application factory and configuration.

Why Celery for ingestion?
  Document parsing + embedding can take 30–120 seconds for large PDFs.
  Doing this synchronously in a FastAPI route would time out the HTTP
  client. Celery offloads this to a background worker, so the API
  immediately returns a task_id the client can poll.

Architecture:
  - Broker  : Redis (same instance as semantic cache, different DB index)
  - Backend : Redis (stores task results / status)
  - Worker  : Separate Docker container running the same image as the API
"""

from celery import Celery
from core.config import get_settings

settings = get_settings()

# Use Redis DB 1 for Celery (DB 0 is used for semantic cache)
_BROKER_URL  = settings.redis_url.replace("/0", "/1")
_BACKEND_URL = settings.redis_url.replace("/0", "/2")

celery_app = Celery(
    "legalmind",
    broker=_BROKER_URL,
    backend=_BACKEND_URL,
    include=["core.tasks.ingest_task"],   # Auto-discover tasks
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task behaviour
    task_acks_late=True,          # Acknowledge only after task completes (safer)
    task_reject_on_worker_lost=True,  # Re-queue if worker dies mid-task
    worker_prefetch_multiplier=1, # One task at a time per worker (memory safety)

    # Result expiry — keep task results for 24 hours
    result_expires=86400,

    # Retry defaults
    task_max_retries=3,
    task_default_retry_delay=10,  # seconds between retries
)
