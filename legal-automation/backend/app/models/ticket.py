from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin


class Ticket(Base, SoftDeleteMixin):
    """Task / Frist / court date / meeting. Self-referential for subtasks."""

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    ticket_type: Mapped[str] = mapped_column(String(30), nullable=False, default="task")
    # task | frist | court_date | client_meeting | follow_up
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="open", index=True)
    # open | in_progress | pending_review | closed | blocked
    priority: Mapped[str] = mapped_column(String(20), nullable=False, default="normal")
    # urgent | high | normal | low

    # Deadline
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    # For Fristen: the statutory basis and how the date was computed (audit trail)
    frist_basis: Mapped[str | None] = mapped_column(String(100), nullable=True)  # e.g. "ZPO §339"
    frist_trigger_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    frist_calculation_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relations
    matter_id: Mapped[int | None] = mapped_column(ForeignKey("matters.id"), nullable=True, index=True)
    assignee_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("tickets.id"), nullable=True, index=True)

    # SLA
    sla_policy_id: Mapped[int | None] = mapped_column(ForeignKey("sla_policies.id"), nullable=True)
    sla_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sla_breached: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Recurrence (RRULE string, e.g. "FREQ=MONTHLY;BYMONTHDAY=1")
    recurrence_rule: Mapped[str | None] = mapped_column(String(255), nullable=True)

    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    comments: Mapped[list["TicketComment"]] = relationship(
        "TicketComment", back_populates="ticket", cascade="all, delete-orphan"
    )
    time_entries: Mapped[list["TicketTimeEntry"]] = relationship(
        "TicketTimeEntry", back_populates="ticket", cascade="all, delete-orphan"
    )
    subtasks: Mapped[list["Ticket"]] = relationship("Ticket", remote_side=[parent_id])

    def __repr__(self) -> str:
        return f"<Ticket id={self.id} type={self.ticket_type} status={self.status} due={self.due_date}>"


class TicketComment(Base, TimestampMixin):
    __tablename__ = "ticket_comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False, index=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # internal comments are never client-visible
    is_internal: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="comments")


class TicketTimeEntry(Base, TimestampMixin):
    """Billable / non-billable time tracking. Rate snapshot is stored to survive rate changes."""

    __tablename__ = "ticket_time_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    matter_id: Mapped[int | None] = mapped_column(ForeignKey("matters.id"), nullable=True, index=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    billable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    rate_eur: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)  # snapshot
    invoiced: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="time_entries")


class SLAPolicy(Base, TimestampMixin):
    """SLA target per matter type. Hourly Celery check flags breaches."""

    __tablename__ = "sla_policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    matter_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # null = applies to all
    ticket_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    # Response/resolution target in hours from ticket creation
    target_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:
        return f"<SLAPolicy id={self.id} name={self.name!r} target={self.target_hours}h>"
