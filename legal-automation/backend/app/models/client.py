from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.encryption import EncryptedText
from app.models.base import Base, SoftDeleteMixin


class Client(Base, SoftDeleteMixin):
    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)

    # Basic identifiers (not encrypted — needed for search/display)
    first_name: Mapped[str] = mapped_column(String(255), nullable=False)
    last_name: Mapped[str] = mapped_column(String(255), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Contact (not PII-encrypted — needed for routing and display)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    address_line1: Mapped[str | None] = mapped_column(String(255), nullable=True)
    address_line2: Mapped[str | None] = mapped_column(String(255), nullable=True)
    postal_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    country: Mapped[str] = mapped_column(String(2), default="DE", nullable=False)

    # PII fields — encrypted at rest via EncryptedText TypeDecorator (Fernet AES-128-CBC)
    date_of_birth: Mapped[str | None] = mapped_column(EncryptedText(255), nullable=True)
    tax_id: Mapped[str | None] = mapped_column(EncryptedText(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(EncryptedText(4096), nullable=True)

    # DSGVO Art. 6 legal basis tracking
    dsgvo_consent_given_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dsgvo_legal_basis: Mapped[str | None] = mapped_column(String(50), nullable=True)  # e.g. "contract", "consent"

    # Metadata
    created_by_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_company: Mapped[bool] = mapped_column(default=False, nullable=False)

    # Relationships
    matters: Mapped[list["Matter"]] = relationship("Matter", back_populates="client", lazy="select")

    @property
    def display_name(self) -> str:
        if self.is_company and self.company_name:
            return self.company_name
        return f"{self.last_name}, {self.first_name}"

    def __repr__(self) -> str:
        return f"<Client id={self.id} number={self.client_number} name={self.display_name!r}>"
