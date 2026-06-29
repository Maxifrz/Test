"""
Kalender-Service: CRUD, Konflikterkennung (Doppelbuchung + Urlaub + Feiertag),
Gerichtstermin-Automatik (Vorbereitungskette, Anfahrtspuffer) und .ics I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.calendar import CalendarAttendee, CalendarEvent


@dataclass
class Conflict:
    kind: str  # "overlap" | "vacation" | "holiday"
    detail: str
    event_id: int | None = None


@dataclass
class ConflictReport:
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def has_blocking(self) -> bool:
        # overlaps and vacation block; holiday is a soft warning
        return any(c.kind in ("overlap", "vacation") for c in self.conflicts)


def _is_holiday(d: date, bundesland: str | None = None) -> bool:
    """Reuse the deadline calculator's holiday awareness."""
    from app.services.deadline_calculator import is_business_day

    return not is_business_day(d, bundesland)


async def detect_conflicts(
    db: AsyncSession,
    *,
    organizer_id: int,
    start_at: datetime,
    end_at: datetime,
    exclude_event_id: int | None = None,
    bundesland: str | None = None,
) -> ConflictReport:
    """
    Find scheduling conflicts for the organizer:
      - overlap with another confirmed event (incl. travel buffer)
      - the organizer is on vacation during the slot
      - the slot falls on a weekend/holiday (soft warning)
    """
    report = ConflictReport()

    query = select(CalendarEvent).where(
        CalendarEvent.deleted_at.is_(None),
        CalendarEvent.organizer_id == organizer_id,
        CalendarEvent.status != "cancelled",
        # time overlap: existing.start < new.end AND existing.end > new.start
        CalendarEvent.start_at < end_at,
        CalendarEvent.end_at > start_at,
    )
    if exclude_event_id is not None:
        query = query.where(CalendarEvent.id != exclude_event_id)

    result = await db.execute(query)
    for ev in result.scalars().all():
        if ev.event_type == "vacation":
            report.conflicts.append(
                Conflict(kind="vacation", detail=f"Urlaub: {ev.title}", event_id=ev.id)
            )
        else:
            report.conflicts.append(
                Conflict(
                    kind="overlap",
                    detail=f"Überschneidung mit '{ev.title}' ({ev.start_at:%d.%m.%Y %H:%M})",
                    event_id=ev.id,
                )
            )

    if _is_holiday(start_at.date(), bundesland):
        report.conflicts.append(
            Conflict(kind="holiday", detail=f"{start_at:%d.%m.%Y} ist ein Wochenende/Feiertag")
        )

    return report


async def create_event(
    db: AsyncSession,
    *,
    title: str,
    event_type: str,
    start_at: datetime,
    end_at: datetime,
    organizer_id: int,
    created_by_id: int,
    description: str | None = None,
    all_day: bool = False,
    location: str | None = None,
    matter_id: int | None = None,
    travel_buffer_minutes: int = 0,
    recurrence_rule: str | None = None,
    source: str = "manual",
    external_uid: str | None = None,
    status: str = "confirmed",
    internal_attendee_ids: list[int] | None = None,
    external_attendees: list[dict] | None = None,
    ticket_id: int | None = None,
) -> CalendarEvent:
    event = CalendarEvent(
        title=title,
        description=description,
        event_type=event_type,
        start_at=start_at,
        end_at=end_at,
        all_day=all_day,
        location=location,
        matter_id=matter_id,
        organizer_id=organizer_id,
        created_by_id=created_by_id,
        travel_buffer_minutes=travel_buffer_minutes,
        recurrence_rule=recurrence_rule,
        source=source,
        external_uid=external_uid,
        status=status,
        ticket_id=ticket_id,
    )
    db.add(event)
    await db.flush()

    for uid in internal_attendee_ids or []:
        db.add(CalendarAttendee(event_id=event.id, user_id=uid))
    for ext in external_attendees or []:
        db.add(
            CalendarAttendee(
                event_id=event.id,
                external_name=ext.get("name"),
                external_email=ext.get("email"),
            )
        )

    await db.commit()
    await db.refresh(event)
    return event


async def get_event(db: AsyncSession, event_id: int) -> CalendarEvent | None:
    result = await db.execute(
        select(CalendarEvent).where(
            CalendarEvent.id == event_id, CalendarEvent.deleted_at.is_(None)
        )
    )
    return result.scalar_one_or_none()


async def list_events(
    db: AsyncSession,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    organizer_id: int | None = None,
    matter_id: int | None = None,
    event_type: str | None = None,
) -> list[CalendarEvent]:
    query = select(CalendarEvent).where(CalendarEvent.deleted_at.is_(None))
    if start:
        query = query.where(CalendarEvent.end_at >= start)
    if end:
        query = query.where(CalendarEvent.start_at <= end)
    if organizer_id:
        query = query.where(CalendarEvent.organizer_id == organizer_id)
    if matter_id:
        query = query.where(CalendarEvent.matter_id == matter_id)
    if event_type:
        query = query.where(CalendarEvent.event_type == event_type)
    query = query.order_by(CalendarEvent.start_at.asc())
    result = await db.execute(query)
    return result.scalars().all()


async def update_event(db: AsyncSession, event: CalendarEvent, updates: dict) -> CalendarEvent:
    for field_name, value in updates.items():
        setattr(event, field_name, value)
    event.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(event)
    return event


async def soft_delete_event(db: AsyncSession, event: CalendarEvent, deleted_by_id: int) -> None:
    event.deleted_at = datetime.now(UTC)
    event.deleted_by_id = deleted_by_id
    await db.commit()


# --- Gerichtstermin-Automatik (Phase-4-Kombination mit Phase 3) ---

# Standard-Vorbereitungskette: (Tage vor Termin, Titel, Tickettyp)
COURT_PREP_CHAIN = [
    (7, "Aktenstudium vorbereiten", "task"),
    (3, "Mandantenbesprechung vor Gerichtstermin", "client_meeting"),
    (1, "Termin final vorbereiten (Unterlagen, Anreise)", "task"),
]


async def generate_court_preparation(
    db: AsyncSession,
    *,
    event: CalendarEvent,
    created_by_id: int,
    assignee_id: int | None = None,
) -> list[int]:
    """
    Erzeugt für einen Gerichtstermin die Vorbereitungskette als Tickets.
    Gibt die IDs der erzeugten Tickets zurück. Kombiniert Kalender (Phase 4)
    mit dem Aufgaben-/Ticketsystem (Phase 3).
    """
    from app.services import ticket_service

    created_ids: list[int] = []
    hearing_day = event.start_at.date()
    for days_before, title, ttype in COURT_PREP_CHAIN:
        due = hearing_day - timedelta(days=days_before)
        ticket = await ticket_service.create_ticket(
            db,
            title=f"{title} — {event.title}",
            description=f"Automatisch erzeugt für Gerichtstermin am {hearing_day:%d.%m.%Y} (Event #{event.id}).",
            created_by_id=created_by_id,
            ticket_type=ttype if ttype in {"task", "client_meeting"} else "task",
            due_date=due,
            matter_id=event.matter_id,
            assignee_id=assignee_id or event.organizer_id,
        )
        created_ids.append(ticket.id)
    return created_ids


# --- .ics Import / Export ---

def event_to_ics(event: CalendarEvent) -> bytes:
    """Serialise a single event to an iCalendar (.ics) byte string."""
    from icalendar import Calendar as ICalCalendar, Event as ICalEvent

    cal = ICalCalendar()
    cal.add("prodid", "-//Legal Automation Platform//DE")
    cal.add("version", "2.0")

    ie = ICalEvent()
    ie.add("uid", event.external_uid or f"event-{event.id}@legal-automation")
    ie.add("summary", event.title)
    if event.description:
        ie.add("description", event.description)
    if event.location:
        ie.add("location", event.location)
    ie.add("dtstart", event.start_at)
    ie.add("dtend", event.end_at)
    ie.add("dtstamp", datetime.now(UTC))
    cal.add_component(ie)
    return cal.to_ical()


@dataclass
class ParsedIcsEvent:
    title: str
    start_at: datetime
    end_at: datetime
    location: str | None
    description: str | None
    external_uid: str | None


def parse_ics(ics_bytes: bytes) -> list[ParsedIcsEvent]:
    """Parse an uploaded .ics file into event candidates (e.g. court Ladungen)."""
    from icalendar import Calendar as ICalCalendar

    parsed: list[ParsedIcsEvent] = []
    cal = ICalCalendar.from_ical(ics_bytes)
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        dtstart = component.get("dtstart")
        dtend = component.get("dtend")
        if dtstart is None:
            continue
        start = dtstart.dt
        end = dtend.dt if dtend is not None else start
        # Normalize date-only values to datetimes (all-day)
        if isinstance(start, date) and not isinstance(start, datetime):
            start = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        if isinstance(end, date) and not isinstance(end, datetime):
            end = datetime.combine(end, datetime.min.time(), tzinfo=UTC)
        parsed.append(
            ParsedIcsEvent(
                title=str(component.get("summary") or "Importierter Termin"),
                start_at=start,
                end_at=end,
                location=str(component.get("location")) if component.get("location") else None,
                description=str(component.get("description")) if component.get("description") else None,
                external_uid=str(component.get("uid")) if component.get("uid") else None,
            )
        )
    return parsed
