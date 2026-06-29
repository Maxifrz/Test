from datetime import datetime

from pydantic import BaseModel, field_validator

EVENT_TYPES = {
    "court_hearing", "client_meeting", "internal_meeting",
    "frist_reminder", "vacation", "other",
}
EVENT_STATUSES = {"confirmed", "tentative", "cancelled"}


class ExternalAttendee(BaseModel):
    name: str | None = None
    email: str | None = None


class EventCreate(BaseModel):
    title: str
    event_type: str = "internal_meeting"
    start_at: datetime
    end_at: datetime
    description: str | None = None
    all_day: bool = False
    location: str | None = None
    matter_id: int | None = None
    organizer_id: int | None = None  # defaults to current user
    travel_buffer_minutes: int = 0
    recurrence_rule: str | None = None
    status: str = "confirmed"
    internal_attendee_ids: list[int] = []
    external_attendees: list[ExternalAttendee] = []
    # If true and event_type == court_hearing, auto-generate the prep ticket chain
    generate_preparation: bool = False
    # Override conflict block (organizer acknowledges the warning)
    force: bool = False

    @field_validator("event_type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        if v not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {sorted(EVENT_TYPES)}")
        return v

    @field_validator("end_at")
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        start = info.data.get("start_at")
        if start and v < start:
            raise ValueError("end_at must be on or after start_at")
        return v


class EventUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    location: str | None = None
    status: str | None = None
    travel_buffer_minutes: int | None = None

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str | None) -> str | None:
        if v is not None and v not in EVENT_STATUSES:
            raise ValueError(f"status must be one of {sorted(EVENT_STATUSES)}")
        return v


class AttendeeResponse(BaseModel):
    id: int
    user_id: int | None
    external_name: str | None
    external_email: str | None
    response_status: str

    model_config = {"from_attributes": True}


class EventResponse(BaseModel):
    id: int
    title: str
    description: str | None
    event_type: str
    start_at: datetime
    end_at: datetime
    all_day: bool
    location: str | None
    travel_buffer_minutes: int
    matter_id: int | None
    organizer_id: int
    ticket_id: int | None
    status: str
    recurrence_rule: str | None
    source: str
    external_uid: str | None
    created_at: datetime
    attendees: list[AttendeeResponse] = []

    model_config = {"from_attributes": True}


class ConflictItem(BaseModel):
    kind: str
    detail: str
    event_id: int | None = None


class ConflictCheckRequest(BaseModel):
    organizer_id: int
    start_at: datetime
    end_at: datetime
    exclude_event_id: int | None = None


class ConflictCheckResponse(BaseModel):
    conflicts: list[ConflictItem]
    has_blocking: bool


class EventCreateResponse(BaseModel):
    event: EventResponse
    conflicts: list[ConflictItem] = []
    created_preparation_ticket_ids: list[int] = []


# --- Ladung parsing ---

class LadungParseRequest(BaseModel):
    email_id: int


class LadungParseResponse(BaseModel):
    found: bool
    hearing_date: str | None = None
    hearing_time: str | None = None
    aktenzeichen: str | None = None
    room: str | None = None
    confidence: float
    note: str
    suggested_title: str | None = None


# --- ICS ---

class IcsImportResult(BaseModel):
    imported: int
    event_ids: list[int]
