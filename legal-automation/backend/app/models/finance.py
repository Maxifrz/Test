from datetime import date, datetime

from sqlalchemy import (
    Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin


class MassAccount(Base, SoftDeleteMixin):
    """Massekonto/Sonderkonto eines Insolvenzverfahrens (eine Akte → ≥1 Konto)."""

    __tablename__ = "mass_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    matter_id: Mapped[int] = mapped_column(ForeignKey("matters.id"), nullable=False, index=True)

    iban: Mapped[str] = mapped_column(String(34), nullable=False, unique=True, index=True)
    bic: Mapped[str | None] = mapped_column(String(11), nullable=True)
    bank_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    account_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    account_type: Mapped[str] = mapped_column(String(20), nullable=False, default="sonderkonto")
    # sonderkonto | anderkonto | treuhand

    opening_balance: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="EUR")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    transactions: Mapped[list["MassTransaction"]] = relationship(
        "MassTransaction", back_populates="account"
    )

    def __repr__(self) -> str:
        return f"<MassAccount id={self.id} matter={self.matter_id} iban={self.iban}>"


class MassTransaction(Base, TimestampMixin):
    """Eine Massebuchung (aus CAMT/MT940 importiert oder manuell)."""

    __tablename__ = "mass_transactions"
    __table_args__ = (
        UniqueConstraint("dedup_key", name="uq_mass_transactions_dedup"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mass_account_id: Mapped[int] = mapped_column(ForeignKey("mass_accounts.id"), nullable=False, index=True)
    matter_id: Mapped[int] = mapped_column(ForeignKey("matters.id"), nullable=False, index=True)
    import_batch_id: Mapped[int | None] = mapped_column(ForeignKey("import_batches.id"), nullable=True, index=True)

    booking_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    value_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)  # positiv
    direction: Mapped[str] = mapped_column(String(3), nullable=False)      # in | out
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="EUR")

    purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    counterparty_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    counterparty_iban: Mapped[str | None] = mapped_column(String(34), nullable=True)
    category: Mapped[str] = mapped_column(String(30), nullable=False, default="unassigned", index=True)

    end_to_end_id: Mapped[str | None] = mapped_column(String(140), nullable=True)
    bank_reference: Mapped[str | None] = mapped_column(String(140), nullable=True)
    dedup_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    is_reconciled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    account: Mapped["MassAccount"] = relationship("MassAccount", back_populates="transactions")


class ImportBatch(Base, TimestampMixin):
    """Provenienz eines Bankauszug-Imports (Audit-Nachweis)."""

    __tablename__ = "import_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mass_account_id: Mapped[int | None] = mapped_column(ForeignKey("mass_accounts.id"), nullable=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)  # camt053 | mt940
    imported_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    num_transactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_assigned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_unassigned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_duplicates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    statement_opening_balance: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    statement_closing_balance: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    computed_closing_balance: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    reconciled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="completed")
    storage_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)  # verschlüsselte Rohdatei


class MassAssignmentRule(Base, TimestampMixin):
    """Kategorisierungsregel (conditions/actions als JSONB) — Muster wie email_rules."""

    __tablename__ = "mass_assignment_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100, index=True)
    conditions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
