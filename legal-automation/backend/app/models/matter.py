from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin

if TYPE_CHECKING:
    from app.models.client import Client
    from app.models.matter_access import MatterAccess
    from app.models.user import User


class Matter(Base, SoftDeleteMixin):
    """Akte — the central entity for all work in the platform."""

    __tablename__ = "matters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    matter_number: Mapped[str] = mapped_column(String(30), unique=True, nullable=False, index=True)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    matter_type: Mapped[str] = mapped_column(String(50), nullable=False)  # civil, criminal, family, labor, admin

    status: Mapped[str] = mapped_column(String(30), nullable=False, default="open")
    # open → active → pending_closing → closed | archived

    # Foreign keys
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), nullable=False, index=True)
    lead_anwalt_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    # Court-related fields
    court_file_ref: Mapped[str | None] = mapped_column(String(100), nullable=True)
    court_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    opposing_party: Mapped[str | None] = mapped_column(String(255), nullable=True)
    opposing_counsel: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Dates
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    statute_of_limitations: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Retention: how long to keep after closing (BRAO §50 → 6 years min).
    # Die Frist laeuft ab Schluss des Kalenderjahres (§ 50 Abs. 1 S. 2 BRAO),
    # nicht ab dem Tagesdatum — siehe services/dsgvo_retention.py.
    retention_years: Mapped[int] = mapped_column(Integer, default=6, nullable=False)
    # Steuerrelevante Akten unterliegen § 147 Abs. 3 AO (10 Jahre). Ohne dieses
    # Flag haette der Loeschworkflow solche Akten nach 6 Jahren freigegeben.
    tax_relevant: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Token für das öffentliche Gläubiger-Portal (Online-Forderungsanmeldung).
    # Unguessbar; nur gesetzt, wenn das Portal für dieses Verfahren aktiviert wurde.
    creditor_portal_token: Mapped[str | None] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )

    # Relationships
    client: Mapped["Client"] = relationship("Client", back_populates="matters")
    lead_anwalt: Mapped["User"] = relationship("User", foreign_keys=[lead_anwalt_id])
    creator: Mapped["User"] = relationship("User", foreign_keys=[created_by_id])
    access_grants: Mapped[list["MatterAccess"]] = relationship(
        "MatterAccess", back_populates="matter", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Matter id={self.id} number={self.matter_number!r} status={self.status!r}>"
