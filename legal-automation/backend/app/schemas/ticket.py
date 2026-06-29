from datetime import date, datetime

from pydantic import BaseModel, field_validator

TICKET_TYPES = {"task", "frist", "court_date", "client_meeting", "follow_up"}
TICKET_STATUSES = {"open", "in_progress", "pending_review", "closed", "blocked"}
PRIORITIES = {"urgent", "high", "normal", "low"}


class TicketCreate(BaseModel):
    title: str
    description: str | None = None
    ticket_type: str = "task"
    priority: str = "normal"
    due_date: date | None = None
    matter_id: int | None = None
    assignee_id: int | None = None
    parent_id: int | None = None
    recurrence_rule: str | None = None

    @field_validator("ticket_type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        if v not in TICKET_TYPES:
            raise ValueError(f"ticket_type must be one of {sorted(TICKET_TYPES)}")
        return v

    @field_validator("priority")
    @classmethod
    def valid_priority(cls, v: str) -> str:
        if v not in PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(PRIORITIES)}")
        return v


class FristTicketCreate(BaseModel):
    """Create a ticket whose due date is computed by the statutory deadline calculator."""
    frist_type: str  # key in deadline_calculator.FRIST_CALCULATORS
    trigger_date: date
    matter_id: int | None = None
    assignee_id: int | None = None


class TicketUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    due_date: date | None = None
    assignee_id: int | None = None

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str | None) -> str | None:
        if v is not None and v not in TICKET_STATUSES:
            raise ValueError(f"status must be one of {sorted(TICKET_STATUSES)}")
        return v

    @field_validator("priority")
    @classmethod
    def valid_priority(cls, v: str | None) -> str | None:
        if v is not None and v not in PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(PRIORITIES)}")
        return v


class TicketResponse(BaseModel):
    id: int
    title: str
    description: str | None
    ticket_type: str
    status: str
    priority: str
    due_date: date | None
    frist_basis: str | None
    frist_trigger_date: date | None
    frist_calculation_note: str | None
    matter_id: int | None
    assignee_id: int | None
    created_by_id: int
    parent_id: int | None
    sla_due_at: datetime | None
    sla_breached: bool
    recurrence_rule: str | None
    closed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TicketListItem(BaseModel):
    id: int
    title: str
    ticket_type: str
    status: str
    priority: str
    due_date: date | None
    matter_id: int | None
    assignee_id: int | None
    sla_breached: bool

    model_config = {"from_attributes": True}


class TicketListResponse(BaseModel):
    items: list[TicketListItem]
    total: int
    page: int
    page_size: int


class CommentCreate(BaseModel):
    body: str
    is_internal: bool = True


class CommentResponse(BaseModel):
    id: int
    ticket_id: int
    author_id: int
    body: str
    is_internal: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TimeEntryCreate(BaseModel):
    started_at: datetime
    ended_at: datetime | None = None
    description: str | None = None
    billable: bool = True
    rate_eur: float | None = None
    matter_id: int | None = None


class TimeEntryResponse(BaseModel):
    id: int
    ticket_id: int
    user_id: int
    started_at: datetime
    ended_at: datetime | None
    duration_minutes: int | None
    description: str | None
    billable: bool
    rate_eur: float | None
    invoiced: bool

    model_config = {"from_attributes": True}


# --- SLA policies ---

class SLAPolicyCreate(BaseModel):
    name: str
    matter_type: str | None = None
    ticket_type: str | None = None
    target_hours: int
    is_active: bool = True


class SLAPolicyResponse(BaseModel):
    id: int
    name: str
    matter_type: str | None
    ticket_type: str | None
    target_hours: int
    is_active: bool

    model_config = {"from_attributes": True}
