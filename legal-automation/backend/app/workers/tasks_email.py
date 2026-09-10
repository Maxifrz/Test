"""Email sync Celery tasks. IMAP fetch → parse → ingest with dedup + routing."""
import asyncio
import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger("app.email.sync")


@celery_app.task(name="app.workers.tasks_email.sync_all_accounts")
def sync_all_accounts():
    """Poll the configured IMAP account and ingest new messages."""
    return asyncio.run(_async_sync())


async def _async_sync() -> dict:
    """
    Ruft ALLE aktiven Postfaecher ab.

    Vorher lag genau ein Postfach global in der .env -- bei 19 Standorten
    unbrauchbar. Faellt ein Postfach aus, laufen die uebrigen weiter und der
    Fehler wird am Konto vermerkt, statt den gesamten Sync abzubrechen.
    """
    from app.core.deps import AsyncSessionLocal
    from app.services import email_account_service

    async with AsyncSessionLocal() as db:
        mailboxes = await email_account_service.resolve_accounts(db)

    if not mailboxes:
        return {"status": "skipped", "reason": "kein Postfach konfiguriert"}

    per_account = []
    total_ingested = 0
    total_duplicates = 0

    for mailbox in mailboxes:
        if not mailbox.imap_host:
            continue
        try:
            ingested, duplicates = await _sync_one(mailbox)
            total_ingested += ingested
            total_duplicates += duplicates
            per_account.append(
                {"account": mailbox.name, "ingested": ingested, "duplicates": duplicates}
            )
            async with AsyncSessionLocal() as db:
                await email_account_service.record_sync_result(db, mailbox.account_id)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:1000]
            logger.error("IMAP-Sync fuer %s fehlgeschlagen: %s", mailbox.name, message)
            per_account.append({"account": mailbox.name, "error": message})
            async with AsyncSessionLocal() as db:
                await email_account_service.record_sync_result(
                    db, mailbox.account_id, error=message
                )

    return {
        "status": "ok",
        "ingested": total_ingested,
        "duplicates": total_duplicates,
        "accounts": per_account,
    }


async def _sync_one(mailbox) -> tuple[int, int]:
    """Ruft ein einzelnes Postfach ab. Gibt (neu, Duplikate) zurueck."""
    import aioimaplib

    from app.core.deps import AsyncSessionLocal
    from app.services import email_service

    ingested = 0
    duplicates = 0

    if mailbox.imap_ssl:
        client = aioimaplib.IMAP4_SSL(host=mailbox.imap_host, port=mailbox.imap_port)
    else:
        client = aioimaplib.IMAP4(host=mailbox.imap_host, port=mailbox.imap_port)

    await client.wait_hello_from_server()
    await client.login(mailbox.imap_username, mailbox.imap_password)
    try:
        await client.select(mailbox.imap_folder)

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
                parsed["account_id"] = mailbox.account_id
                result = await email_service.ingest_email(db, parsed)
                if result is None:
                    duplicates += 1
                else:
                    ingested += 1
    finally:
        # Auch bei einem Fehler die Verbindung sauber schliessen -- sonst
        # haelt der Server die Sitzung offen und das naechste Login scheitert.
        await client.logout()

    return ingested, duplicates


def _extract_rfc822(fetch_data) -> bytes | None:
    """aioimaplib returns a list; the raw message is the bytes payload entry."""
    for item in fetch_data:
        if isinstance(item, (bytes, bytearray)) and len(item) > 50:
            return bytes(item)
    return None


@celery_app.task(name="app.workers.tasks_email.retry_pending_outbox")
def retry_pending_outbox():
    """
    Erneuter Zustellversuch fuer haengengebliebene Outbox-Eintraege.

    Ohne diesen Task waere die Outbox nur eine Fehleranzeige: eine Mail, die
    beim ersten Versuch am Mailserver scheitert (Netz weg, Greylisting,
    Wartungsfenster), bliebe fuer immer liegen.
    """
    return asyncio.run(_async_retry_outbox())


async def _async_retry_outbox() -> dict:
    from app.core.deps import AsyncSessionLocal
    from app.services import email_service

    sent = 0
    failed = 0
    async with AsyncSessionLocal() as db:
        for record in await email_service.pending_outbox(db):
            if await email_service.attempt_delivery(db, record):
                sent += 1
            else:
                failed += 1
    return {"status": "ok", "sent": sent, "still_pending": failed}
