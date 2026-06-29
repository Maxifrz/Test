"""Email sync Celery tasks. Actual IMAP logic lives in app.services.email_service."""
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks_email.sync_all_accounts")
def sync_all_accounts():
    """Trigger IMAP sync for all configured email accounts."""
    import asyncio
    asyncio.run(_async_sync_all())


async def _async_sync_all():
    # Placeholder: will be wired to EmailService in Phase 2
    pass
