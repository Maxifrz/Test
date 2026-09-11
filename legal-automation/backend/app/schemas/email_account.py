"""Schemas der Postfach-Verwaltung."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class EmailAccountResponse(BaseModel):
    """Antwort OHNE Passwoerter — sie verlassen die Datenbank nie."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    standort: str | None
    imap_host: str
    imap_port: int
    imap_ssl: bool
    imap_username: str
    imap_folder: str
    smtp_host: str
    smtp_port: int
    smtp_tls: bool
    smtp_username: str | None
    from_name: str
    from_email: str
    is_active: bool
    is_default: bool
    last_sync_at: datetime | None
    last_sync_error: str | None


class EmailAccountCreate(BaseModel):
    name: str
    standort: str | None = None
    imap_host: str
    imap_port: int = 993
    imap_ssl: bool = True
    imap_username: str
    imap_password: str
    imap_folder: str = "INBOX"
    smtp_host: str
    smtp_port: int = 587
    smtp_tls: bool = True
    smtp_username: str | None = None
    smtp_password: str | None = None
    from_name: str = "Kanzlei"
    from_email: EmailStr
    is_active: bool = True


class EmailAccountUpdate(BaseModel):
    name: str | None = None
    standort: str | None = None
    imap_host: str | None = None
    imap_port: int | None = None
    imap_ssl: bool | None = None
    imap_username: str | None = None
    # Leer lassen = Passwort unveraendert. So laesst sich ein Konto bearbeiten,
    # ohne das Passwort erneut eingeben zu muessen.
    imap_password: str | None = None
    imap_folder: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_tls: bool | None = None
    smtp_username: str | None = None
    smtp_password: str | None = None
    from_name: str | None = None
    from_email: EmailStr | None = None
    is_active: bool | None = None


class EmailAccountTestResult(BaseModel):
    imap_ok: bool
    smtp_ok: bool
    imap_error: str | None = None
    smtp_error: str | None = None
