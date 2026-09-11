"""
Verwaltung der Postfächer.

Die alte globale .env-Konfiguration bleibt als Rückfall erhalten: eine
bestehende Installation soll nach dem Update ohne Umkonfiguration
weiterlaufen. `resolve_accounts` liefert deshalb entweder die konfigurierten
Konten aus der Datenbank oder — wenn keine angelegt sind — das eine Konto aus
den Settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.email_account import EmailAccount


@dataclass
class MailboxConfig:
    """Einheitliche Sicht auf ein Postfach, egal ob aus DB oder .env."""

    account_id: int | None
    name: str
    imap_host: str
    imap_port: int
    imap_ssl: bool
    imap_username: str
    imap_password: str
    imap_folder: str
    smtp_host: str
    smtp_port: int
    smtp_tls: bool
    smtp_username: str
    smtp_password: str
    from_name: str
    from_email: str


def _from_model(a: EmailAccount) -> MailboxConfig:
    return MailboxConfig(
        account_id=a.id,
        name=a.name,
        imap_host=a.imap_host,
        imap_port=a.imap_port,
        imap_ssl=a.imap_ssl,
        imap_username=a.imap_username,
        imap_password=a.imap_password or "",
        imap_folder=a.imap_folder,
        smtp_host=a.smtp_host,
        smtp_port=a.smtp_port,
        smtp_tls=a.smtp_tls,
        smtp_username=a.smtp_username or "",
        smtp_password=a.smtp_password or "",
        from_name=a.from_name,
        from_email=a.from_email,
    )


def _from_settings() -> MailboxConfig | None:
    """Das alte Einzelpostfach aus der .env — Rückfall für Bestandsinstallationen."""
    from app.core.config import get_settings

    s = get_settings()
    if not s.IMAP_HOST and not s.SMTP_HOST:
        return None
    return MailboxConfig(
        account_id=None,
        name="Standardpostfach (.env)",
        imap_host=s.IMAP_HOST,
        imap_port=s.IMAP_PORT,
        imap_ssl=s.IMAP_SSL,
        imap_username=s.IMAP_USERNAME,
        imap_password=s.IMAP_PASSWORD,
        imap_folder="INBOX",
        smtp_host=s.SMTP_HOST,
        smtp_port=s.SMTP_PORT,
        smtp_tls=s.SMTP_TLS,
        smtp_username=s.SMTP_USERNAME,
        smtp_password=s.SMTP_PASSWORD,
        from_name=s.SMTP_FROM_NAME,
        from_email=s.SMTP_FROM_EMAIL,
    )


async def list_accounts(db: AsyncSession, *, only_active: bool = False) -> list[EmailAccount]:
    query = select(EmailAccount).where(EmailAccount.deleted_at.is_(None))
    if only_active:
        query = query.where(EmailAccount.is_active.is_(True))
    result = await db.execute(query.order_by(EmailAccount.name.asc()))
    return list(result.scalars().all())


async def resolve_accounts(db: AsyncSession) -> list[MailboxConfig]:
    """Alle aktiven Postfächer; leer nur, wenn auch die .env keins definiert."""
    accounts = await list_accounts(db, only_active=True)
    if accounts:
        return [_from_model(a) for a in accounts]
    fallback = _from_settings()
    return [fallback] if fallback else []


async def default_account(db: AsyncSession, account_id: int | None = None) -> MailboxConfig | None:
    """
    Postfach für den Versand: explizit gewähltes, sonst das als Standard
    markierte, sonst das erste aktive, sonst die .env.
    """
    if account_id is not None:
        acc = (await db.execute(
            select(EmailAccount).where(
                EmailAccount.id == account_id,
                EmailAccount.deleted_at.is_(None),
                EmailAccount.is_active.is_(True),
            )
        )).scalar_one_or_none()
        if acc is None:
            raise ValueError(f"Postfach {account_id} nicht gefunden oder inaktiv")
        return _from_model(acc)

    accounts = await list_accounts(db, only_active=True)
    for a in accounts:
        if a.is_default:
            return _from_model(a)
    if accounts:
        return _from_model(accounts[0])
    return _from_settings()


async def set_default(db: AsyncSession, account_id: int) -> None:
    """Markiert genau ein Konto als Standard (die übrigen werden zurückgesetzt)."""
    await db.execute(
        update(EmailAccount)
        .where(EmailAccount.id != account_id)
        .values(is_default=False)
    )
    await db.execute(
        update(EmailAccount).where(EmailAccount.id == account_id).values(is_default=True)
    )
    await db.commit()


async def record_sync_result(
    db: AsyncSession, account_id: int | None, *, error: str | None = None
) -> None:
    """Hält Zeitpunkt und Ergebnis des letzten Abrufs fest."""
    if account_id is None:
        return  # .env-Rückfall hat keine Zeile
    await db.execute(
        update(EmailAccount)
        .where(EmailAccount.id == account_id)
        .values(last_sync_at=datetime.now(UTC), last_sync_error=error)
    )
    await db.commit()
