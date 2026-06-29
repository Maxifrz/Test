from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin


class EmailMessage(Base, SoftDeleteMixin):
    """A single email (inbound or outbound), filed against a matter where possible."""

    __tablename__ = "email_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(String(998), unique=True, nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # inbound | outbound

    from_address: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    to_addresses: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    cc_addresses: Mapped[list] = mapped_column(JSONB, nullable=True, default=list)
    subject: Mapped[str | None] = mapped_column(String(998), nullable=True)
    body_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    body_html: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Threading
    in_reply_to: Mapped[str | None] = mapped_column(String(998), nullable=True, index=True)
    references: Mapped[str | None] = mapped_column(Text, nullable=True)
    thread_key: Mapped[str | None] = mapped_column(String(998), nullable=True, index=True)

    # Filing
    matter_id: Mapped[int | None] = mapped_column(ForeignKey("matters.id"), nullable=True, index=True)
    client_id: Mapped[int | None] = mapped_column(ForeignKey("clients.id"), nullable=True, index=True)
    matched_rule_id: Mapped[int | None] = mapped_column(ForeignKey("email_rules.id"), nullable=True)

    # State flags
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_confidential: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    unknown_sender: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Outbound delivery
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivery_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    sent_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    email_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    attachments: Mapped[list["EmailAttachment"]] = relationship(
        "EmailAttachment", back_populates="email", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<EmailMessage id={self.id} dir={self.direction} subject={self.subject!r}>"


class EmailAttachment(Base, TimestampMixin):
    __tablename__ = "email_attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("email_messages.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)  # encrypted file on disk

    email: Mapped["EmailMessage"] = relationship("EmailMessage", back_populates="attachments")


class EmailRule(Base, TimestampMixin):
    """Routing rule. Conditions and actions stored as JSONB. First matching rule (by priority) wins."""

    __tablename__ = "email_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False, index=True)

    # conditions: {"from_domain": "...", "from_address": "...", "subject_contains": [...],
    #              "body_contains": [...], "match": "all"|"any"}
    conditions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # actions: {"file_to_matter": <id>, "set_confidential": true, "flag_review": true,
    #           "create_ticket": {...}}
    actions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    def __repr__(self) -> str:
        return f"<EmailRule id={self.id} name={self.name!r} prio={self.priority}>"


class EmailTemplate(Base, SoftDeleteMixin):
    """Outbound Jinja2 template (e.g. Mandatsannahme, Fristverlängerung)."""

    __tablename__ = "email_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    subject_template: Mapped[str] = mapped_column(Text, nullable=False)
    body_template: Mapped[str] = mapped_column(Text, nullable=False)
    # Documents available variables, e.g. {"mandant_name": "Voller Name des Mandanten"}
    variables_doc: Mapped[dict] = mapped_column(JSONB, nullable=True, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    def __repr__(self) -> str:
        return f"<EmailTemplate id={self.id} name={self.name!r}>"
