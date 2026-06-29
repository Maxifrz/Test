from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin

# Ränge nach InsO
RANK_38 = "insolvenz_38"
RANK_39 = "nachrangig_39"
RANK_ABSONDERUNG = "absonderung"
RANK_MASSE = "masseverbindlichkeit"

# Status der Forderungsprüfung
STATUS_ANGEMELDET = "angemeldet"
STATUS_GEPRUEFT = "geprueft"
STATUS_FESTGESTELLT = "festgestellt"
STATUS_BESTRITTEN = "bestritten"


class InsolvencyClaim(Base, SoftDeleteMixin):
    """Angemeldete Insolvenzforderung (Position der Insolvenztabelle)."""

    __tablename__ = "insolvency_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    matter_id: Mapped[int] = mapped_column(ForeignKey("matters.id"), nullable=False, index=True)

    # Laufende Nummer in der Tabelle (pro Verfahren)
    claim_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    creditor_name: Mapped[str] = mapped_column(String(255), nullable=False)
    creditor_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    creditor_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    creditor_reference: Mapped[str | None] = mapped_column(String(140), nullable=True)

    claim_amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)        # angemeldet
    established_amount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)  # festgestellt
    claim_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    rank: Mapped[str] = mapped_column(String(30), nullable=False, default=RANK_38, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=STATUS_ANGEMELDET, index=True)
    dispute_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    source: Mapped[str] = mapped_column(String(20), nullable=False, default="intern")  # intern | glaeubiger_portal
    filed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    def __repr__(self) -> str:
        return f"<InsolvencyClaim id={self.id} matter={self.matter_id} {self.creditor_name} {self.claim_amount}€ {self.status}>"


class Distribution(Base, TimestampMixin):
    """Eine Verteilung (Abschlags- oder Schlussverteilung) eines Verfahrens."""

    __tablename__ = "distributions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    matter_id: Mapped[int] = mapped_column(ForeignKey("matters.id"), nullable=False, index=True)
    distribution_type: Mapped[str] = mapped_column(String(20), nullable=False, default="abschlag")  # abschlag | schluss
    distributable_amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    quote_38_pct: Mapped[float] = mapped_column(Numeric(7, 4), nullable=False, default=0)
    distributed_sum: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False, default=0)
    remainder: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False, default=0)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    items: Mapped[list["DistributionItem"]] = relationship(
        "DistributionItem", back_populates="distribution", cascade="all, delete-orphan"
    )


class DistributionItem(Base):
    """Auszahlungsposition einer Verteilung je Forderung (Verteilungsverzeichnis)."""

    __tablename__ = "distribution_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    distribution_id: Mapped[int] = mapped_column(ForeignKey("distributions.id"), nullable=False, index=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("insolvency_claims.id"), nullable=False)
    established_amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    quote_pct: Mapped[float] = mapped_column(Numeric(7, 4), nullable=False, default=0)

    distribution: Mapped["Distribution"] = relationship("Distribution", back_populates="items")
