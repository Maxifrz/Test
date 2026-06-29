from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "legal_automation",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.tasks_email",
        "app.workers.tasks_transcription",
        "app.workers.tasks_maintenance",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Berlin",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "app.workers.tasks_transcription.*": {"queue": "transcription"},
        "app.workers.tasks_email.*": {"queue": "email"},
        "app.workers.tasks_maintenance.*": {"queue": "default"},
    },
    beat_schedule={
        "email-sync-every-2-minutes": {
            "task": "app.workers.tasks_email.sync_all_accounts",
            "schedule": 120.0,
        },
        "sla-check-hourly": {
            "task": "app.workers.tasks_maintenance.check_sla_breaches",
            "schedule": 3600.0,
        },
        "retention-cleanup-nightly": {
            "task": "app.workers.tasks_maintenance.run_retention_cleanup",
            "schedule": 86400.0,
        },
    },
)
