from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ProcessingRecord(Base, TimestampMixin):
    """Verarbeitungstätigkeit im Verarbeitungsverzeichnis (Art. 30 DSGVO)."""

    __tablename__ = "processing_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    legal_basis: Mapped[str] = mapped_column(String(255), nullable=False)  # z.B. "Art. 6 Abs. 1 lit. b"
    data_categories: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    data_subjects: Mapped[str | None] = mapped_column(Text, nullable=True)
    recipients: Mapped[str | None] = mapped_column(Text, nullable=True)
    retention: Mapped[str | None] = mapped_column(Text, nullable=True)
    tom: Mapped[str | None] = mapped_column(Text, nullable=True)  # technisch-organisatorische Maßnahmen
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class DataRetentionPolicy(Base, TimestampMixin):
    """Aufbewahrungsregel je Aktentyp (nightly Cleanup prüft dagegen)."""

    __tablename__ = "data_retention_policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    matter_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # null = alle
    retention_years: Mapped[int] = mapped_column(Integer, nullable=False)
    legal_basis: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class ErasureRequest(Base, TimestampMixin):
    """Antrag auf Löschung (Art. 17 DSGVO) — geprüfter Workflow, keine 1-Klick-Löschung."""

    __tablename__ = "erasure_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), nullable=False, index=True)
    requested_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open", index=True)
    # open | blocked | approved | executed | rejected
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    blocking_reasons: Mapped[list] = mapped_column(JSONB, nullable=True, default=list)

    decided_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    certificate_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)


class DataExport(Base, TimestampMixin):
    """Datenportabilität (Art. 20 DSGVO) — ZIP-Export, Link 48h gültig, single-use."""

    __tablename__ = "data_exports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), nullable=False, index=True)
    requested_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # pending | ready | downloaded | expired | failed
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    file_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
