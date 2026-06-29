from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Response, status
from sqlalchemy import select

from app.core.deps import DB, require_permission
from app.models.calendar import CalendarEvent
from app.models.email import EmailMessage
from app.schemas.calendar import (
    ConflictCheckRequest,
    ConflictCheckResponse,
    ConflictItem,
    EventCreate,
    EventCreateResponse,
    EventResponse,
    EventUpdate,
    IcsImportResult,
    LadungParseRequest,
    LadungParseResponse,
)
from app.services import calendar_service
from app.services.ladung_parser import parse_ladung

router = APIRouter(prefix="/calendar", tags=["calendar"])


def _conflict_items(report) -> list[ConflictItem]:
    return [ConflictItem(kind=c.kind, detail=c.detail, event_id=c.event_id) for c in report.conflicts]


@router.post("/conflict-check", response_model=ConflictCheckResponse)
async def conflict_check(
    data: ConflictCheckRequest,
    db: DB,
    current_user=Depends(require_permission("calendar.read")),
):
    report = await calendar_service.detect_conflicts(
        db,
        organizer_id=data.organizer_id,
        start_at=data.start_at,
        end_at=data.end_at,
        exclude_event_id=data.exclude_event_id,
    )
    return ConflictCheckResponse(conflicts=_conflict_items(report), has_blocking=report.has_blocking)


@router.post("", response_model=EventCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_event(
    data: EventCreate,
    db: DB,
    current_user=Depends(require_permission("calendar.create")),
):
    organizer_id = data.organizer_id or current_user.id

    report = await calendar_service.detect_conflicts(
        db, organizer_id=organizer_id, start_at=data.start_at, end_at=data.end_at
    )
    # Blocking conflicts (overlap/vacation) require explicit force; holiday is a soft warning.
    if report.has_blocking and not data.force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Terminkonflikt erkannt. Mit force=true bestätigen.",
                "conflicts": [c.__dict__ for c in report.conflicts],
            },
        )

    event = await calendar_service.create_event(
        db,
        title=data.title,
        event_type=data.event_type,
        start_at=data.start_at,
        end_at=data.end_at,
        organizer_id=organizer_id,
        created_by_id=current_user.id,
        description=data.description,
        all_day=data.all_day,
        location=data.location,
        matter_id=data.matter_id,
        travel_buffer_minutes=data.travel_buffer_minutes,
        recurrence_rule=data.recurrence_rule,
        status=data.status,
        internal_attendee_ids=data.internal_attendee_ids,
        external_attendees=[a.model_dump() for a in data.external_attendees],
    )

    prep_ids: list[int] = []
    if data.generate_preparation and data.event_type == "court_hearing":
        prep_ids = await calendar_service.generate_court_preparation(
            db, event=event, created_by_id=current_user.id
        )

    await db.refresh(event)
    return EventCreateResponse(
        event=EventResponse.model_validate(event),
        conflicts=_conflict_items(report),
        created_preparation_ticket_ids=prep_ids,
    )


@router.get("", response_model=list[EventResponse])
async def list_events(
    db: DB,
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    organizer_id: int | None = Query(None),
    matter_id: int | None = Query(None),
    event_type: str | None = Query(None),
    current_user=Depends(require_permission("calendar.read")),
):
    events = await calendar_service.list_events(
        db, start=start, end=end, organizer_id=organizer_id, matter_id=matter_id, event_type=event_type
    )
    return events


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: int,
    db: DB,
    current_user=Depends(require_permission("calendar.read")),
):
    event = await calendar_service.get_event(db, event_id)
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return event


@router.patch("/{event_id}", response_model=EventResponse)
async def update_event(
    event_id: int,
    data: EventUpdate,
    db: DB,
    current_user=Depends(require_permission("calendar.update")),
):
    event = await calendar_service.get_event(db, event_id)
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return await calendar_service.update_event(db, event, data.model_dump(exclude_unset=True))


@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event(
    event_id: int,
    db: DB,
    current_user=Depends(require_permission("calendar.delete")),
):
    event = await calendar_service.get_event(db, event_id)
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    await calendar_service.soft_delete_event(db, event, deleted_by_id=current_user.id)


@router.get("/{event_id}/export.ics")
async def export_ics(
    event_id: int,
    db: DB,
    current_user=Depends(require_permission("calendar.read")),
):
    event = await calendar_service.get_event(db, event_id)
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    ics = calendar_service.event_to_ics(event)
    return Response(
        content=ics,
        media_type="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="event-{event_id}.ics"'},
    )


@router.post("/import-ics", response_model=IcsImportResult)
async def import_ics(
    db: DB,
    file: UploadFile = File(...),
    current_user=Depends(require_permission("calendar.create")),
):
    raw = await file.read()
    try:
        parsed = calendar_service.parse_ics(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Ungültige .ics-Datei: {exc}"
        )

    event_ids: list[int] = []
    for p in parsed:
        # Dedup by external UID
        if p.external_uid:
            existing = await db.execute(
                select(CalendarEvent.id).where(CalendarEvent.external_uid == p.external_uid)
            )
            if existing.scalar_one_or_none():
                continue
        event = await calendar_service.create_event(
            db,
            title=p.title,
            event_type="court_hearing" if "termin" in p.title.lower() else "other",
            start_at=p.start_at,
            end_at=p.end_at,
            organizer_id=current_user.id,
            created_by_id=current_user.id,
            location=p.location,
            description=p.description,
            source="ics_import",
            external_uid=p.external_uid,
        )
        event_ids.append(event.id)

    return IcsImportResult(imported=len(event_ids), event_ids=event_ids)


@router.post("/parse-ladung", response_model=LadungParseResponse)
async def parse_ladung_from_email(
    data: LadungParseRequest,
    db: DB,
    current_user=Depends(require_permission("calendar.read")),
):
    """
    Lese eine bereits eingegangene E-Mail und schlage einen Gerichtstermin vor.
    Verbindet das E-Mail-Modul (Phase 2) mit dem Kalender (Phase 4). Das
    Ergebnis ist ein Vorschlag — der Anwender bestätigt den Termin manuell.
    """
    result = await db.execute(
        select(EmailMessage).where(EmailMessage.id == data.email_id, EmailMessage.deleted_at.is_(None))
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Email not found")

    parsed = parse_ladung(
        subject=msg.subject or "",
        body=msg.body_text or "",
        sender=msg.from_address,
    )

    suggested_title = None
    if parsed.found:
        az = f" ({parsed.aktenzeichen})" if parsed.aktenzeichen else ""
        suggested_title = f"Gerichtstermin{az}"

    return LadungParseResponse(
        found=parsed.found,
        hearing_date=parsed.hearing_date.isoformat() if parsed.hearing_date else None,
        hearing_time=parsed.hearing_time.strftime("%H:%M") if parsed.hearing_time else None,
        aktenzeichen=parsed.aktenzeichen,
        room=parsed.room,
        confidence=parsed.confidence,
        note=parsed.note,
        suggested_title=suggested_title,
    )
