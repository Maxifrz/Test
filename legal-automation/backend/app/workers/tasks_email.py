"""Email sync Celery tasks. IMAP fetch → parse → ingest with dedup + routing."""
import asyncio

from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks_email.sync_all_accounts")
def sync_all_accounts():
    """Poll the configured IMAP account and ingest new messages."""
    return asyncio.run(_async_sync())


async def _async_sync() -> dict:
    from app.core.config import get_settings
    from app.core.deps import AsyncSessionLocal
    from app.services import email_service

    settings = get_settings()
    if not settings.IMAP_HOST:
        return {"status": "skipped", "reason": "IMAP not configured"}

    import aioimaplib

    ingested = 0
    duplicates = 0

    client = aioimaplib.IMAP4_SSL(host=settings.IMAP_HOST, port=settings.IMAP_PORT)
    await client.wait_hello_from_server()
    await client.login(settings.IMAP_USERNAME, settings.IMAP_PASSWORD)
    await client.select("INBOX")

    # Only fetch unseen messages to keep each poll cheap
    _, data = await client.search("UNSEEN")
    msg_nums = data[0].split() if data and data[0] else []

    async with AsyncSessionLocal() as db:
        for num in msg_nums:
            _, fetch_data = await client.fetch(num.decode(), "(RFC822)")
            raw = _extract_rfc822(fetch_data)
            if raw is None:
                continue
            parsed = email_service.parse_raw_email(raw)
            result = await email_service.ingest_email(db, parsed)
            if result is None:
                duplicates += 1
            else:
                ingested += 1

    await client.logout()
    return {"status": "ok", "ingested": ingested, "duplicates": duplicates}


def _extract_rfc822(fetch_data) -> bytes | None:
    """aioimaplib returns a list; the raw message is the bytes payload entry."""
    for item in fetch_data:
        if isinstance(item, (bytes, bytearray)) and len(item) > 50:
            return bytes(item)
    return None
