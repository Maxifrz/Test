"""
Verwaltung der Postfaecher (Admin).

Bisher lag genau ein Postfach global in der .env -- eine Aenderung erforderte
einen Neustart des Stacks, und mehrere Standorte waren gar nicht abbildbar.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import DB, require_permission
from app.models.email_account import EmailAccount
from app.schemas.email_account import (
    EmailAccountCreate,
    EmailAccountResponse,
    EmailAccountTestResult,
    EmailAccountUpdate,
)
from app.services import email_account_service

router = APIRouter(prefix="/email-accounts", tags=["email-accounts"])


async def _load(db: DB, account_id: int) -> EmailAccount:
    acc = await db.get(EmailAccount, account_id)
    if acc is None or acc.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Postfach nicht gefunden")
    return acc


@router.get("", response_model=list[EmailAccountResponse])
async def list_accounts(db: DB, current_user=Depends(require_permission("email.rule.create"))):
    return await email_account_service.list_accounts(db)


@router.post("", response_model=EmailAccountResponse, status_code=status.HTTP_201_CREATED)
async def create_account(
    data: EmailAccountCreate,
    db: DB,
    current_user=Depends(require_permission("email.rule.create")),
):
    payload = data.model_dump()
    payload["from_email"] = str(payload["from_email"])
    acc = EmailAccount(**payload)

    # Das erste Konto wird automatisch Standard -- sonst haette der Versand
    # nach dem Anlegen kein Ziel.
    if not await email_account_service.list_accounts(db, only_active=True):
        acc.is_default = True

    db.add(acc)
    await db.commit()
    await db.refresh(acc)
    return acc


@router.patch("/{account_id}", response_model=EmailAccountResponse)
async def update_account(
    account_id: int,
    data: EmailAccountUpdate,
    db: DB,
    current_user=Depends(require_permission("email.rule.update")),
):
    acc = await _load(db, account_id)
    for field, value in data.model_dump(exclude_unset=True).items():
        # Leeres Passwortfeld bedeutet "unveraendert", nicht "loeschen".
        if field.endswith("_password") and not value:
            continue
        if value is not None:
            setattr(acc, field, str(value) if field == "from_email" else value)
    await db.commit()
    await db.refresh(acc)
    return acc


@router.post("/{account_id}/default", response_model=EmailAccountResponse)
async def make_default(
    account_id: int,
    db: DB,
    current_user=Depends(require_permission("email.rule.update")),
):
    """Setzt das Standard-Postfach fuer den Versand ohne explizite Auswahl."""
    acc = await _load(db, account_id)
    if not acc.is_active:
        raise HTTPException(status_code=409, detail="Ein inaktives Postfach kann nicht Standard sein")
    await email_account_service.set_default(db, account_id)
    await db.refresh(acc)
    return acc


@router.post("/{account_id}/test", response_model=EmailAccountTestResult)
async def test_account(
    account_id: int,
    db: DB,
    current_user=Depends(require_permission("email.rule.update")),
):
    """
    Prueft IMAP- und SMTP-Anmeldung. Ohne diesen Endpunkt faellt ein falsches
    Passwort erst beim naechsten Sync-Lauf auf -- also womoeglich Stunden
    spaeter und nur im Log.
    """
    import aioimaplib
    import aiosmtplib

    acc = await _load(db, account_id)
    mailbox = await email_account_service.default_account(db, account_id)
    result = EmailAccountTestResult(imap_ok=False, smtp_ok=False)

    try:
        client = (
            aioimaplib.IMAP4_SSL(host=mailbox.imap_host, port=mailbox.imap_port)
            if mailbox.imap_ssl
            else aioimaplib.IMAP4(host=mailbox.imap_host, port=mailbox.imap_port)
        )
        await client.wait_hello_from_server()
        await client.login(mailbox.imap_username, mailbox.imap_password)
        await client.select(mailbox.imap_folder)
        await client.logout()
        result.imap_ok = True
    except Exception as exc:
        result.imap_error = f"{type(exc).__name__}: {exc}"[:500]

    try:
        smtp = aiosmtplib.SMTP(
            hostname=mailbox.smtp_host, port=mailbox.smtp_port, start_tls=mailbox.smtp_tls
        )
        await smtp.connect()
        if mailbox.smtp_username:
            await smtp.login(mailbox.smtp_username, mailbox.smtp_password)
        await smtp.quit()
        result.smtp_ok = True
    except Exception as exc:
        result.smtp_error = f"{type(exc).__name__}: {exc}"[:500]

    # Das Ergebnis am Konto vermerken, damit es im Dashboard sichtbar ist
    acc.last_sync_error = result.imap_error or result.smtp_error
    await db.commit()
    return result


@router.delete("/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    account_id: int,
    db: DB,
    current_user=Depends(require_permission("email.rule.delete")),
):
    """
    Soft-Delete. Bereits eingegangene Nachrichten verweisen ueber account_id
    weiter auf das Konto -- die Zeile darf deshalb nicht verschwinden.
    """
    from datetime import UTC, datetime

    acc = await _load(db, account_id)
    acc.deleted_at = datetime.now(UTC)
    acc.deleted_by_id = current_user.id
    acc.is_active = False
    acc.is_default = False
    await db.commit()
