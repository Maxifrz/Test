from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class MatterAccess(Base):
    """Per-matter access grant: which user can access which matter and in what role."""

    __tablename__ = "matter_access"
    __table_args__ = (
        UniqueConstraint("user_id", "matter_id", name="uq_matter_access_user_matter"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    matter_id: Mapped[int] = mapped_column(ForeignKey("matters.id"), nullable=False, index=True)
    matter_role: Mapped[str] = mapped_column(String(20), nullable=False)  # MatterRole enum: lead/support/readonly
    granted_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    matter: Mapped["Matter"] = relationship("Matter", back_populates="access_grants")
    granted_by: Mapped["User"] = relationship("User", foreign_keys=[granted_by_id])

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None

    def __repr__(self) -> str:
        return f"<MatterAccess user={self.user_id} matter={self.matter_id} role={self.matter_role!r}>"
