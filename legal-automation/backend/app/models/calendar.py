from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin


class CalendarEvent(Base, SoftDeleteMixin):
    """A calendar entry: court hearing, meeting, reminder, vacation, etc."""

    __tablename__ = "calendar_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    event_type: Mapped[str] = mapped_column(String(30), nullable=False, default="internal_meeting")
    # court_hearing | client_meeting | internal_meeting | frist_reminder | vacation | other

    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    all_day: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    location: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Travel buffer (minutes) reserved before/after court hearings
    travel_buffer_minutes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relations
    matter_id: Mapped[int | None] = mapped_column(ForeignKey("matters.id"), nullable=True, index=True)
    organizer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    # Link back to the Frist ticket this event reminds about (frist_reminder type)
    ticket_id: Mapped[int | None] = mapped_column(ForeignKey("tickets.id"), nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="confirmed")
    # confirmed | tentative | cancelled

    recurrence_rule: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Provenance: how the event entered the system
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="manual")
    # manual | ics_import | email_ladung
    external_uid: Mapped[str | None] = mapped_column(String(998), nullable=True, index=True)

    attendees: Mapped[list["CalendarAttendee"]] = relationship(
        "CalendarAttendee", back_populates="event", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<CalendarEvent id={self.id} type={self.event_type} start={self.start_at}>"


class CalendarAttendee(Base, TimestampMixin):
    """An internal user or an external (name+email) participant of an event."""

    __tablename__ = "calendar_attendees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("calendar_events.id"), nullable=False, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    external_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    external_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    response_status: Mapped[str] = mapped_column(String(20), nullable=False, default="needs_action")
    # needs_action | accepted | declined | tentative

    event: Mapped["CalendarEvent"] = relationship("CalendarEvent", back_populates="attendees")
