"""Kontaktanfragen von der öffentlichen Kanzlei-Website (Formular-Endpunkt)."""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ContactRequest(Base, TimestampMixin):
    """
    Eine über POST /api/public/contact eingegangene Anfrage. Inhalte sind
    Untrusted (öffentliches Formular) — Bearbeitung nur durch Kanzlei-Personal.
    Datensparsamkeit: keine IP-/UA-Speicherung; Missbrauchsabwehr läuft über
    nginx-Rate-Limiting und Honeypot (verworfene Anfragen werden NICHT gespeichert).
    """

    __tablename__ = "contact_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    standort: Mapped[str | None] = mapped_column(String(100), nullable=True)
    rolle: Mapped[str | None] = mapped_column(String(100), nullable=True)  # z. B. Unternehmen/Gläubiger
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # DSGVO-Einwilligung (Pflicht-Checkbox im Formular) — Zeitpunkt der Abgabe
    consent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="neu")  # neu | erledigt
    processed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    def __repr__(self) -> str:
        return f"<ContactRequest id={self.id} email={self.email!r} status={self.status}>"
