from celery import Celery
from celery.schedules import crontab
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
        "app.workers.tasks_ki",
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
    # Ohne Zeitlimits blockiert eine haengende IMAP- oder SMTP-Verbindung den
    # Worker-Slot dauerhaft. soft_time_limit wirft erst eine Exception (der
    # Task kann aufraeumen), time_limit killt hart.
    task_soft_time_limit=25 * 60,
    task_time_limit=30 * 60,
    # Celery 5.4 warnt sonst beim Start; ohne das bricht ein Worker ab, wenn
    # Redis beim Hochfahren noch nicht bereit ist.
    broker_connection_retry_on_startup=True,
    # Transkription laeuft im eigenen Worker mit eigenen, laengeren Limits
    task_annotations={
        "app.workers.tasks_transcription.*": {
            "soft_time_limit": 4 * 60 * 60,
            "time_limit": 5 * 60 * 60,
        },
    },
    task_routes={
        "app.workers.tasks_transcription.*": {"queue": "transcription"},
        "app.workers.tasks_email.*": {"queue": "email"},
        "app.workers.tasks_maintenance.*": {"queue": "default"},
        "app.workers.tasks_ki.*": {"queue": "default"},
    },
    # Feste Uhrzeiten statt Intervallen: "schedule: 86400.0" bedeutet
    # "24 h nach Beat-Start", nicht "nachts". Der Retention-Report lief damit zu
    # der Uhrzeit, zu der der Container zuletzt neu startete, und driftete bei
    # jedem Deployment.
    beat_schedule={
        "email-sync-every-2-minutes": {
            "task": "app.workers.tasks_email.sync_all_accounts",
            "schedule": 120.0,
        },
        "outbox-retry-every-5-minutes": {
            "task": "app.workers.tasks_email.retry_pending_outbox",
            "schedule": 300.0,
        },
        "sla-check-hourly": {
            "task": "app.workers.tasks_maintenance.check_sla_breaches",
            "schedule": crontab(minute=5),
        },
        "retention-report-nightly": {
            "task": "app.workers.tasks_maintenance.run_retention_cleanup",
            "schedule": crontab(hour=3, minute=0),
        },
    },
)
