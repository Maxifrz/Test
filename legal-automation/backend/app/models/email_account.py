"""
Postfach-Konfiguration je Standort.

Bisher lag genau EIN Postfach global in der .env. Bei 19 Standorten ist das
keine Feinheit, sondern eine Grundvoraussetzung: jeder Standort führt eigene
Korrespondenz, und ein gemeinsames Sammelpostfach würde die Aktentrennung
untergraben.

Zugangsdaten sind feldverschlüsselt (Fernet) — sie stehen sonst im Klartext in
der Datenbank und damit in jedem Backup.
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.encryption import EncryptedText
from app.models.base import Base, SoftDeleteMixin


class EmailAccount(Base, SoftDeleteMixin):
    __tablename__ = "email_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Anzeigename, z. B. "München — Insolvenzverwaltung"
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    standort: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # --- IMAP ---
    imap_host: Mapped[str] = mapped_column(String(255), nullable=False)
    imap_port: Mapped[int] = mapped_column(Integer, nullable=False, default=993)
    imap_ssl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    imap_username: Mapped[str] = mapped_column(String(255), nullable=False)
    imap_password: Mapped[str | None] = mapped_column(EncryptedText(255), nullable=True)
    imap_folder: Mapped[str] = mapped_column(String(255), nullable=False, default="INBOX")

    # --- SMTP ---
    smtp_host: Mapped[str] = mapped_column(String(255), nullable=False)
    smtp_port: Mapped[int] = mapped_column(Integer, nullable=False, default=587)
    smtp_tls: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    smtp_username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    smtp_password: Mapped[str | None] = mapped_column(EncryptedText(255), nullable=True)

    from_name: Mapped[str] = mapped_column(String(200), nullable=False, default="Kanzlei")
    from_email: Mapped[str] = mapped_column(String(320), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    # Genau ein Konto ist Standard für den Versand ohne explizite Auswahl.
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- Betriebszustand ---
    # Ohne diese Felder faellt ein stillgelegtes Postfach erst auf, wenn jemand
    # eine fehlende Mail vermisst.
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_seen_uid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<EmailAccount id={self.id} name={self.name!r} from={self.from_email}>"
